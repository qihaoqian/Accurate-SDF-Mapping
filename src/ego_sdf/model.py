from dataclasses import dataclass

import torch
import torch.nn as nn
from octree_config import OctreeConfig
from semi_sparse_octree_v1 import SemiSparseOctree

from ego_sdf.residual_net import ResidualNet, ResidualNetConfig
from ego_sdf.utils.config_abc import ConfigABC


@dataclass
class SdfNetworkConfig(ConfigABC):
    octree_cfg: OctreeConfig = OctreeConfig()
    residual_net_cfg: ResidualNetConfig = ResidualNetConfig()


class SdfNetwork(nn.Module):
    def __init__(self, cfg):
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
