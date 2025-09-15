import tinycudann as tcnn
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,
                 bound=None,
                 voxel_size=None,
                 L=None,
                 F_entry=None,
                 log2_T=None,
                 b=None,
                 mlp_layer=None,
                 mlp_hidden_dim=None,
                 activation="ReLU",
                 **kwargs):
        super().__init__()

        self.bound = torch.FloatTensor(bound)
        self.bound_dis = self.bound[:, 1] - self.bound[:, 0]
        self.max_dis = torch.ceil(torch.max(self.bound_dis))
        N_min = int(self.max_dis / voxel_size)
        self.hash_sdf_out = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=1,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F_entry,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,  # 1/base_resolution is the grid_size
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": activation,
                    "output_activation": "None",
                    "n_neurons": mlp_hidden_dim,
                    "n_hidden_layers": mlp_layer,
                }
            )

    def get_sdf(self, xyz):
        xyz = (xyz - self.bound[:, 0].to(xyz.device)) / self.bound_dis.to(xyz.device)
        sdf = self.hash_sdf_out(xyz) * 0.1
        return sdf

    def forward(self, xyz):
        sdf = self.get_sdf(xyz) 

        return {
            'sdf': sdf[:, 0],
        }

if __name__ == "__main__":
    network = Decoder(1, 128, 16, skips=[], embedder='none', multires=0)
    print(network)
