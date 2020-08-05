"""A collection of point cloud operations.

All ops are operating on point clouds represented
as torch.Tensor() of shape [batch_size, dim, num_point].

To operate on a point clouds `pc` of shape [batch_size, num_point, dim],
pass `pc.transpose(1, 2)` to the operation.
There is no need to explicitly apply Tensor.contiguous() before passing the tensor.
"""

from typing import Callable, List

import etw_pytorch_utils as _ET
import torch
import torch.nn as nn
from knn_cuda import KNN as _KNN

import pointnet2.utils.pointnet2_utils as _PU
from samplenet.chamfer_distance import ChamferDistance as _CD


def knn_query(p: torch.Tensor, q: torch.Tensor, k: int) -> (torch.Tensor, torch.Tensor):
    """Point cloud K-Nearest Neighbors query.

    Args:
        p: Reference point cloud of shape [batch_size, dim, num_point].
        q: Query Point cloud of shape [batch_size, dim, num_query].
        k: Group size.

    Returns:
        A tuple of tensors:
            idx: Indices tensor of shape [batch_size, num_query, k].
            dist: Distance tensor of shape [batch_size, num_query, k].
    """

    p = p.contiguous()
    q = q.contiguous()
    dist, idx = _KNN(k, transpose_mode=False)(p, q)
    dist = dist.transpose(1, 2).contiguous()
    idx = idx.transpose(1, 2).contiguous().type(torch.int32)
    return idx, dist


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


def group(p: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Group point cloud indices.

    Args:
        p: Reference point cloud of shape [batch_size, dim, num_point].
        idx: Indices tensor of shape [batch_size, num_query, k].

    Returns:
        A tensor of shape [batch_size, dim, num_query, k].
    """

    p = p.contiguous()
    return _PU.grouping_operation(p, idx)


def ball_query(p: torch.Tensor, q: torch.Tensor, r: float, k: int) -> torch.Tensor:
    """Point cloud ball (sphere) query.

    Args:
        p: Reference point cloud of shape [batch_size, dim, num_point].
        q: Query Point cloud of shape [batch_size, dim, num_query].
        r (float): Ball radius.
        k (int): Maximum group size.

    Returns:
        Indices tensor of shape [batch_size, num_query, k].
    """

    p_t = p.transpose(1, 2).contiguous()
    q_t = q.transpose(1, 2).contiguous()
    idx = _PU.ball_query(r, k, p_t, q_t)
    return idx


def chamfer_distance(p: torch.Tensor, q: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Calculates chamfer distance between two point clouds.

    Args:
        p: Reference point cloud of shape [batch_size, dim, num_point].
        q: Query Point cloud of shape [batch_size, dim, num_query].

    Returns:
        A tuple of two tensor scalars representing
        the two sided Chamfer distance between the point clouds.
    """

    p_t = p.transpose(1, 2).contiguous()
    q_t = q.transpose(1, 2).contiguous()
    cost_p_q, cost_q_p, _, _ = _CD()(p_t, q_t)
    return cost_p_q, cost_q_p


def adaptive_add_fusion(z: torch.Tensor, p: torch.Tensor):
    """Adds a point cloud to a noise tensor that can have a smaller dimension.

    The noise tensor `z` with `dim_z` channels will be added to the first `dim_z`
    channels of the point cloud `p`.

    Args:
        z: Reference point cloud of shape [batch_size, dim_z, num_point].
        p: Reference point cloud of shape [batch_size, dim_p, num_point].

    Returns:
        Point cloud tensor of shape [batch_size, dim_p, num_point].
    """
    dim_z = z.shape[1]
    dim_p = p.shape[1]
    if dim_z <= dim_p:
        return torch.cat([z + p[:, :dim_z, :], p[:, dim_z:, :]], dim=1)
    else:
        raise RuntimeError(
            "dim_z must be less or equal to dim_p. "
            + f"got `dim_z` = {dim_z} > {dim_p} = `dim_p`",
        )


def normalize_group(
    grouped: torch.Tensor,
    shift: bool = True,
    scale: bool = True,
    method: str = "minmax",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize grouped into the half unit sphere with equal absolute value of max and min in each axis.

    Args:
        grouped: Grouped point cloud tensor of shape [batch_size, dim, num_query, k]
        shift (optional): Normalize offset. Defaults to True.
        scale (optional): Normalize scale. Defaults to True.
        method (optional): Offset normalization method.
            Passing 'mean' will reduce mean of every group.
            Passing 'center' will remove the center value of the patch
                (point with index 0 in the K channel)
            Passing 'minmax' will shift each group such that
                its minimum and maximum values are equal in every axis.
        eps (optional): Scale normalization regularizer.

    Returns:
        A tuple containing:
            - The normalized grouped point cloud tensor of shape [batch_size, dim, num_query, k]
            - The shift applied per group
            - The scale applied per group
    """

    grouped_normalized = grouped

    if shift:
        if method == "minmax":
            shift_per_group = (
                torch.min(grouped, dim=-1, keepdim=True)[0]
                + torch.max(grouped, dim=-1, keepdim=True)[0]
            ) / 2.0
            grouped_normalized = grouped_normalized - shift_per_group
        elif method == "mean":
            shift_per_group = torch.mean(grouped, dim=-1, keepdim=True)
            grouped_normalized = grouped - shift_per_group
        elif method == "center":
            shift_per_group = grouped[:, :, :, 0:1]
            grouped_normalized = grouped - shift_per_group
        else:
            raise ValueError(f"unknown normalization method. got {method}")
    else:
        shift_per_group = torch.Tensor([0.0]).to(grouped)

    if scale:
        norms = torch.norm(grouped_normalized, dim=1, keepdim=True)  # [B, 1, N, K]
        max_norm_per_group = torch.max(norms, dim=-1, keepdim=True)[0]
        scale_per_group = 2 * max_norm_per_group + eps
        grouped_normalized = grouped_normalized / scale_per_group + eps
    else:
        scale_per_group = torch.Tensor([1.0]).to(grouped)

    return grouped_normalized, shift_per_group, scale_per_group


# The SharedMLP class from _ET module adds activation after each convolution.
# This class gives an option to disable the activation and batch normalization
# on the last layer.
class SharedMLP(nn.Sequential):
    def __init__(
        self,
        args: List[int],
        bn: bool = False,
        activation: Callable = nn.ReLU(inplace=True),
        add_last_bn: bool = False,
        add_last_activation: bool = False,
    ):
        super().__init__()
        for i in range(len(args) - 1):
            if i == len(args) - 2:
                _act = activation if add_last_activation else None
                _bn = bn if add_last_bn else False
            else:
                _act = activation
                _bn = bn
            self.add_module(
                f"layer_{i}",
                _ET.Conv2d(
                    args[i], args[i + 1], bias=not _bn, bn=_bn, activation=_act,
                ),
            )


class DenseMLP(nn.Module):
    def __init__(
        self,
        args: List[int],
        bn: bool = False,
        activation: Callable = nn.ReLU(inplace=True),
        add_last_bn: bool = False,
        add_last_activation: bool = False,
    ):
        """
            #TODO asaf: write full docstring.
            Dense network encoder of patches.
        """
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(len(args) - 1):
            if i == len(args) - 2:
                _act = activation if add_last_activation else None
                _bn = bn if add_last_bn else False
            else:
                _act = activation
                _bn = bn
            input_dim = sum(args[: (i + 1)])  # Sum of all input dimensions until here
            output_dim = args[i + 1]
            self.convs.append(
                _ET.Conv2d(input_dim, output_dim, bias=not _bn, bn=_bn, activation=_act)
            )

    def forward(self, patches):
        inputs = patches
        for i, conv in enumerate(self.convs):
            outputs = conv(inputs)
            inputs = torch.cat([inputs, outputs], dim=1)
        return outputs
