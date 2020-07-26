import torch
import torch.nn as nn

import pointnet2.utils.pointnet2_utils as _PU


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

        y = gather(x, idx)
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
        idx = fps(y, self.num_out_points)
        y = gather(y, idx)
        return y


def fps(p: torch.Tensor, k: int) -> torch.Tensor:
    """Point cloud FPS sampling.
    Args:
        p: Reference point cloud of shape [batch_size, 3, num_point].
        k (int): Number of sampled points.
    Returns:
        Indices tensor of shape [batch_size, k].
    """

    p_t = p.transpose(1, 2).contiguous()
    return _PU.furthest_point_sample(p_t, k)


def gather(p: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Point cloud gathering by indices.
    Args:
        p: Reference point cloud of shape [batch_size, dim, num_point].
        idx: Indices tensor of shape [batch_size, num_query].
    Returns:
        Point cloud tensor of shape [batch_size, dim, num_query].
    """

    p = p.contiguous()
    return _PU.gather_operation(p, idx)