from dataclasses import dataclass

from ego_sdf.utils.config_abc import ConfigABC


@dataclass
class OctreeConfig(ConfigABC):
    resolution: float = 0.1
    tree_depth: int = 8
    full_depth: int = 5
    init_voxel_num: int = 200000
