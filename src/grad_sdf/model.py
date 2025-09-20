from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

from grad_sdf.residual_net import ResidualNet, ResidualNetConfig
from grad_sdf.utils.config_abc import ConfigABC
from octree_config import OctreeConfig
from semi_sparse_octree_v1 import SemiSparseOctree


@dataclass
class SdfNetworkConfig(ConfigABC):
    octree_cfg: OctreeConfig = OctreeConfig()
    residual_net_cfg: ResidualNetConfig = ResidualNetConfig()


class SdfNetwork(nn.Module):
    def __init__(self, cfg: SdfNetworkConfig):
        super().__init__()
        self.cfg = cfg
        self.octree: SemiSparseOctree = SemiSparseOctree(cfg.octree_cfg)
        self.residual: ResidualNet = ResidualNet(cfg.residual_net_cfg)

    def forward(self, points: torch.Tensor, voxel_indices: torch.Tensor = None):
        """
        Computes the SDF values for the given points.
        Args:
            points: (..., 3) points in world coordinates
            voxel_indices: (...,) optional voxel indices for the points

        Returns:
            (..., ) voxel indices for the points
            (..., ) SDF prior from the octree
            (..., ) SDF residual from the residual network
            (..., ) final SDF values (prior + residual)
        """
        shape = points.shape
        points = points.view(-1, 3)
        if voxel_indices is not None:
            voxel_indices = voxel_indices.view(-1)
        sdf_prior, voxel_indices = self.octree(points, voxel_indices)
        sdf_residual = self.residual(points)

        sdf_prior = sdf_prior.view(shape[:-1])
        sdf_residual = sdf_residual.view(shape[:-1])
        voxel_indices = voxel_indices.view(shape[:-1])

        return voxel_indices, sdf_prior, sdf_residual, sdf_prior + sdf_residual

    def voxel_filter_by_size(
        self,
        voxel_coords: torch.Tensor,
        grid_voxel_indices: torch.Tensor,
        min_size: int = 1,
        max_size: int = 2,
    ) -> list:
        """
        Filter voxels based on their voxel sizes.
        Args:
            voxel_coords: (N, 3) tensor of voxel coordinates
            grid_voxel_indices: (X, Y, Z) tensor of voxel indices for each grid point
            min_size: minimum voxel size to keep
            max_size: maximum voxel size to keep
        Returns:
            indices of valid voxels
        """
        voxel_vertex_offsets = torch.tensor(  # (8, 3)
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=torch.long,
            device=voxel_coords.device,
        )

        vertex_coords = voxel_coords.view(-1, 1, 3) + voxel_vertex_offsets.view(1, 8, 3)  # (n_valid, 8, 3)
        vertex_indices = grid_voxel_indices[
            vertex_coords[..., 0].flatten(), vertex_coords[..., 1].flatten(), vertex_coords[..., 2].flatten()
        ].reshape(-1, 8)
        voxel_sizes = self.octree.get_voxel_discrete_size(vertex_indices)
        valid_mask = torch.any((voxel_sizes >= min_size) & (voxel_sizes <= max_size), dim=1)  # (n_valid, )
        return torch.nonzero(valid_mask, as_tuple=False).cpu().flatten().tolist()


def forward_with_batches(
    model: SdfNetwork,
    batch_size: int,
    points: torch.Tensor,
    voxel_indices: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward the model with batches to avoid OOM.
    Args:
        model: SdfNetwork model
        batch_size: batch size
        points: (..., 3) points in world coordinates
        voxel_indices: (...,) optional voxel indices for the points
    Returns:
        (..., ) voxel indices for the points
        (..., ) SDF prior from the octree
        (..., ) SDF residual from the residual network
        (..., ) final SDF values (prior + residual)
    """
    if voxel_indices is None:
        with torch.no_grad():
            # this should not cause OOM
            voxel_indices = model.octree.find_voxel_indices(points=points.view(-1, 3))
            voxel_indices = voxel_indices.view(points.shape[:-1])

    if batch_size <= 0:
        bs = points.shape[0]
    else:
        bs = int(batch_size * 3 * points.shape[0] / points.numel()) + 1

    sdf_prior = []
    sdf_residual = []
    sdf_pred = []
    for i in tqdm(range(0, points.shape[0], bs), desc="Batches", ncols=120):
        j = min(i + bs, points.shape[0])
        _, sdf_prior_batch, sdf_residual_batch, sdf_pred_batch = model(points[i:j], voxel_indices[i:j])
        sdf_prior.append(sdf_prior_batch)
        sdf_residual.append(sdf_residual_batch)
        sdf_pred.append(sdf_pred_batch)

    if len(sdf_prior) == 1:
        return voxel_indices, sdf_prior[0], sdf_residual[0], sdf_pred[0]

    sdf_prior = torch.cat(sdf_prior, dim=0)
    sdf_residual = torch.cat(sdf_residual, dim=0)
    sdf_pred = torch.cat(sdf_pred, dim=0)
    return voxel_indices, sdf_prior, sdf_residual, sdf_pred
