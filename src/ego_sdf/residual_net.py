from dataclasses import dataclass

import tinycudann as tcnn
import torch
import torch.nn as nn

from ego_sdf.utils.config_abc import ConfigABC


@dataclass
class ResidualNetConfig(ConfigABC):
    bound: list = None  # (3, 2) bounding box of the scene
    resolution: float = 0.05  # the size of the smallest voxel
    num_levels: int = 4  # number of levels in the hash encoding
    feature_dims: int = 2  # number of features per level
    log2_T: int = 19  # log2 of the hash table size
    per_level_scale: float = 2.0  # per level scale factor
    mlp_activation: str = "LeakyReLU"  # activation function for the MLP
    hidden_dims: int = 64  # number of hidden dimensions
    n_hidden_layers: int = 2  # number of hidden layers
    scale_with_bound: bool = True  # whether to scale the input with the bounding box
    output_sdf_scale: float = 0.1  # scale the output SDF


class ResidualNet(nn.Module):
    def __init__(self, cfg: ResidualNetConfig):
        """
        Args:
            cfg: configuration of the network
        """
        super().__init__()
        self.cfg = cfg

        self.bound = torch.FloatTensor(cfg.bound)
        self.bound_dis = self.bound[:, 1] - self.bound[:, 0]
        self.max_dis = torch.ceil(torch.max(self.bound_dis))
        N_min = int(self.max_dis / cfg.resolution)
        self.hash_sdf_out = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1,
            encoding_config={
                "otype": "Grid",
                "type": "Hash",
                "n_levels": cfg.num_levels,  # number of levels
                "n_features_per_level": cfg.feature_dims,  # number of features per level
                "log2_hashmap_size": cfg.log2_T,  # each level has a hash table of size 2^log2_T
                "base_resolution": N_min,  # coarsest grid size
                "per_level_scale": cfg.per_level_scale,  # each level's grid size = base_resolution * per_level_scale^level
                "interpolation": "Linear",
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": cfg.mlp_activation,
                "output_activation": "None",
                "n_neurons": cfg.hidden_dims,
                "n_hidden_layers": cfg.n_hidden_layers,
            },
        )

    def get_sdf(self, xyz: torch.Tensor):
        if self.cfg.scale_with_bound:
            xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.bound_dis.to(xyz.device)
        sdf = self.hash_sdf_out(xyz)
        return sdf

    def forward(self, xyz: torch.Tensor):
        sdf = self.get_sdf(xyz).view(-1) * self.cfg.output_sdf_scale
        return sdf


if __name__ == "__main__":
    cfg = ResidualNetConfig(
        bound=[[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
        resolution=0.05,
        num_levels=16,
        feature_dims=2,
        log2_T=19,
        per_level_scale=1.3819,
        mlp_activation="ReLU",
        hidden_dims=64,
        n_hidden_layers=2,
        scale_with_bound=True,
        output_sdf_scale=0.1,
    )
    network = ResidualNet(cfg)
    print(network)
