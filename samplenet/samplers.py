import torch
import torch.nn as nn

import samplenet.ops as ops


class RandomSampler(nn.Module):
    def __init__(self, num_out_points):
        # type: (RandomSampler, int, str, str) -> None
        super().__init__()
        self.num_out_points = num_out_points

    def forward(self, x):
        # type: (RandomSampler, torch.Tensor) -> torch.Tensor
        B, _, N = x.shape

        idx = torch.zeros(B, self.num_out_points, dtype=torch.int32, device=x.device)
        for i in range(B):
            rand_perm = torch.randperm(N, dtype=torch.int32, device=x.device)
            idx[i] = rand_perm[: self.num_out_points]

        y = ops.gather(x, idx)
        return y


class FPSSampler(nn.Module):
    def __init__(self, num_out_points, permute):
        # type: (FPSSampler, int, int, str, str) -> None
        super().__init__()
        self.num_out_points = num_out_points
        self.permute = permute

    def forward(self, x):
        # type: (FPSSampler, torch.Tensor) -> torch.Tensor
        if self.permute:
            N = x.shape[2]
            y = x[:, :, torch.randperm(N)]
        else:
            y = x
        idx = ops.fps(y, self.num_out_points)
        y = ops.gather(y, idx)
        return y


class NullUpsampler(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, p, patches):
        return p


class VanillaUpsampler(nn.Module):
    """Non learned up-sampling of a point cloud.

    Args:
        mode (string): 'uniform', 'weighted' - mode of averaging operation. Defaults to 'uniform'.
        m (int): Patch size for upsampling (might be different than the input patch size k). Defaults to 2.

    Inputs:
        p (Tensor): Point cloud of shape [batch_size, 3, num_point].
        patches (Tensor): Patches tensor of shape [batch_size, 3, num_patch, k].

    Returns:
        Upsampled point cloud of shape [batch_size, 3, 2 * num_point].
    """

    def __init__(self, mode="uniform", k=2):
        # type: (VanillaUpsampler, str, int) -> None
        super().__init__()
        self._mode, self._k = mode, k

    def _get_distances(self, grouped_points, query_cloud):
        # type: (VanillaUpsampler, torch.Tensor, torch.Tensor) -> torch.Tensor
        deltas = grouped_points - query_cloud.unsqueeze(-1)
        dist = torch.sum(deltas ** 2, dim=1, keepdim=True)
        return dist

    def forward(self, p, patches):
        # type: (VanillaUpsampler, torch.Tensor, torch.Tensor) -> torch.Tensor
        grouped_points = patches[:, :, :, : self._k]

        if self._mode == "uniform":
            new_points = torch.mean(grouped_points, dim=-1)
        elif self._mode == "weighted":
            dist = self._get_distances(grouped_points, grouped_points[:, :, :, 0])
            weights = torch.softmax(-dist, dim=3)
            new_points = torch.sum(grouped_points * weights, dim=3)
        else:
            raise ValueError(f"unknown upsampling mode: {self._mode}")

        p_upsamp = torch.cat((p, new_points), dim=2)

        return p_upsamp


class Patcher(nn.Module):
    """Gathers local patches of the input point cloud.

    Args:
        mode: 'knn', 'ball' - mode of gathering operation. Defaults to 'knn'.
        k (optional): Patch size. Defaults to 8.
        r (optional): Radius in the query ball mode. defaults to 0.1.

    Inputs:
        p: Reference point cloud of shape [batch_size, dim, num_point].
        q (optional): Query Point cloud of shape [batch_size, dim, num_query].
           Defaults to None. It that case, it is set to p.
        use_last (optional): Use last computed patche indices. Defaults to False.
        normalize_shift (optional): Normalize patches offset. Defaults to True.
        normalize_scale (optional): Normalize patches scale. Defaults to True.
        normalize_method (optional): Patches offset normalization method.
            see ops.normalize_group().

    Returns:
        Grouped point cloud tensor of shape [batch_size, dim, num_query, k]
    """

    def __init__(
        self,
        k: int,
        mode: str = "knn",
        r: float = 0.1,
        normalize_shift: bool = True,
        normalize_scale: bool = True,
        normalize_method: str = "mean",
    ):
        super().__init__()
        if mode not in ["ball", "knn"]:
            raise ValueError(f"unknown query mode: {self._mode}")

        self._mode, self._k, self._r = mode, k, r
        self._normalize_shift, self._normalize_scale, self._normalize_method = (
            normalize_shift,
            normalize_scale,
            normalize_method,
        )

        # Memory
        self._last_idx = None
        self._last_shift = None
        self._last_scale = None

    @staticmethod
    def group(p, idx):
        return ops.group(p, idx)

    def query(self, p, q=None):
        if q is None:
            q = p

        # idx for group [batch_size, num_query, k]
        with torch.no_grad():
            if self._mode == "knn":
                idx, _ = ops.knn_query(p, q, self._k)
            elif self._mode == "ball":
                idx = ops.ball_query(p, q, self._r, self._k)
        return idx

    def normalize(self, patches):
        """Normalize given patches of shape [batch_size, dim, num_query, k]."""
        normalized_patches, shift, scale = ops.normalize_group(
            patches,
            self._normalize_shift,
            self._normalize_scale,
            self._normalize_method,
        )
        self._last_shift = shift
        self._last_scale = scale

        return normalized_patches

    def denormalize(self, normalized_patches):
        """Inverse the last transform done by self.normalize()."""
        res = (normalized_patches * self._last_scale) + self._last_shift
        self._last_scale = None
        self._last_shift = None
        return res

    def forward(self, p, q=None, use_last=False):
        """Query and group points. Pass use_last=True to reuse the last calculated indices."""
        if q is None:
            q = p

        if use_last:
            grouped = self.group(p, self._last_idx)
        else:
            idx = self.query(p, q)  # [batch_size, num_query, k]
            self._last_idx = idx
            grouped = self.group(p, idx)  # [batch_size, dim, num_query, k]

        return grouped


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from data.toy_examples import Cube
    from utils.plotting import plot_3d_point_cloud

    show_plots = True
    seed = 100
    np.random.seed(seed)
    num_samp = 60
    cube = Cube().sample(num_samp)

    query = np.array(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    ).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cube_tensor = (
        torch.from_numpy(np.expand_dims(cube, axis=0))
        .transpose(1, 2)
        .contiguous()
        .to(device)
    )
    query_tensor = (
        torch.from_numpy(np.expand_dims(query, axis=0))
        .transpose(1, 2)
        .contiguous()
        .to(device)
    )

    ################
    # test Patcher #
    ################
    k = 9
    patcher_knn = Patcher(mode="knn", k=k)
    grouped_knn = patcher_knn(cube_tensor, query_tensor)

    _ = patcher_knn.normalize(grouped_knn)

    grouped_knn_np = np.transpose(grouped_knn.cpu().numpy()[0], (1, 2, 0))

    patcher_ball = Patcher(mode="ball", k=k, r=1.0)
    grouped_ball = patcher_ball(cube_tensor, query_tensor)

    grouped_ball_np = np.transpose(grouped_ball.cpu().numpy()[0], (1, 2, 0))

    if show_plots:
        plot_3d_point_cloud(cube, in_u_sphere=False, miv=-1.5, mav=1.5, title="cube")
        plot_3d_point_cloud(
            grouped_knn_np[0], in_u_sphere=False, miv=-1.5, mav=1.5, title="knn patch",
        )
        plot_3d_point_cloud(
            grouped_ball_np[0],
            in_u_sphere=False,
            miv=-1.5,
            mav=1.5,
            title="ball patch",
        )

        plt.show()

    # compare patches
    patch_knn = grouped_knn_np[0]
    r_knn = np.sqrt(np.sum((patch_knn - np.expand_dims(query[0], axis=0)) ** 2, axis=1))

    patch_ball = grouped_ball_np[0]
    r_ball = np.sqrt(
        np.sum((patch_ball - np.expand_dims(query[0], axis=0)) ** 2, axis=1)
    )
    sort_idx = np.argsort(r_ball)
    patch_ball_sorted = patch_ball[sort_idx]

    print(
        "Patcher test result:", np.array_equal(patch_knn, patch_ball_sorted)
    )  # this test should be True

    ##################
    # test Upsampler #
    ##################
    upsampler_uniform = VanillaUpsampler(mode="uniform", m=k)
    grouped_upsamped = upsampler_uniform(cube_tensor, grouped_knn)

    grouped_upsamped_np = np.transpose(grouped_upsamped.cpu().numpy()[0], (1, 0))
    new_points_np = grouped_upsamped_np[num_samp:]

    grouped_knn_mean_np = np.mean(grouped_knn_np, axis=1)
    diff = np.abs(new_points_np - grouped_knn_mean_np)
    print("Upsampler test result:", np.all(diff < 1e-7))  # this test should be True

    upsampler_weighted = VanillaUpsampler(mode="weighted", m=4)
    grouped_upsamped = upsampler_weighted(cube_tensor, grouped_knn)
    print("Upsampler weighted result size:", grouped_upsamped.size())

    # ###################
    #  test FPSSampler  #
    # ###################
    # from utils.plotting import tensor_2d_scatter, tensor_3d_scatter

    # NUM_POINTS = 512
    # BATCH_SIZE = 1

    # cuda = True if torch.cuda.is_available() else False
    # device = torch.device('cuda') if cuda else torch.device('cpu')

    # # get FPS (uniform) cube
    # gt_batch_np = np.zeros((BATCH_SIZE, NUM_POINTS * 4, 3), dtype=np.float32)
    # for i in range(BATCH_SIZE):
    #     gt_batch_np[i] = sample_cube(NUM_POINTS * 4)
    # gt_batch = torch.from_numpy(gt_batch_np).transpose(1, 2).contiguous().to(device)
    # gt_batch = FPSSampler(NUM_POINTS, permute=False)(gt_batch)
    # gt_batch_max_norm = torch.max(torch.norm(gt_batch, dim=1))
    # gt_batch = gt_batch / gt_batch_max_norm  # maps gt_batch to the unit sphere
