import torch

from .utils.sampling import generate_sample_mask

rays_dir = None


class RGBDFrame:
    def __init__(
        self,
        fid: int,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        K: torch.Tensor,
        offset: torch.Tensor,
        ref_pose: torch.Tensor,
    ) -> None:
        """
        Args:
            fid: int, frame idx
            rgb: (H, W, 3) in [0, 1]
            depth: (H, W) in meter
            K: (3, 3) intrinsic matrix
            offset: (3,) offset to be added to the translation of ref_pose
            ref_pose: (4, 4) reference pose in world coordinates
        """
        super().__init__()
        self.stamp = fid
        self.h, self.w = depth.shape
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.FloatTensor(rgb)
        if not isinstance(depth, torch.Tensor):
            depth = torch.FloatTensor(depth)  # / 2
        self.rgb = rgb
        self.depth = depth
        self.K = K

        if ref_pose.ndim != 2:
            ref_pose = ref_pose.reshape(4, 4)
        if not isinstance(ref_pose, torch.Tensor):  # from gt data
            self.ref_pose = torch.tensor(ref_pose, requires_grad=False, dtype=torch.float32)
        else:  # from tracked data
            self.ref_pose = ref_pose.clone().requires_grad_(False)
        self.ref_pose[:3, 3] += offset  # Offset ensures voxel coordinates > 0

        self.rays_d: torch.Tensor = None  # (H, W, 3) in camera coordinates
        self.points: torch.Tensor = None  # (H, W, 3) in world coordinates
        self.valid_mask: torch.Tensor = None  # (H, W) depth > 0
        self.sample_mask: torch.Tensor = None  # (H, W) sampled rays

        self.precompute()

    def get_ref_pose(self):
        return self.ref_pose

    def get_ref_translation(self):
        return self.ref_pose[:3, 3]

    def get_ref_rotation(self):
        return self.ref_pose[:3, :3]

    @torch.no_grad()
    def get_rays(self, w=None, h=None, K=None):
        w = self.w if w == None else w
        h = self.h if h == None else h
        if K is None:
            K = torch.eye(3)
            K[0, 0] = self.K[0, 0] * w / self.w
            K[1, 1] = self.K[1, 1] * h / self.h
            K[0, 2] = self.K[0, 2] * w / self.w
            K[1, 2] = self.K[1, 2] * h / self.h
        ix, iy = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        rays_d = torch.stack(
            [(ix - K[0, 2]) / K[0, 0], (iy - K[1, 2]) / K[1, 1], torch.ones_like(ix)], -1
        ).float()  # camera coordinate
        return rays_d

    @torch.no_grad()
    def precompute(self):
        global rays_dir
        if rays_dir is None:
            rays_dir = self.get_rays(K=self.K)
        self.rays_d = rays_dir
        self.points = self.rays_d * self.depth[..., None]
        self.valid_mask = self.depth > 0

    @torch.no_grad()
    def get_points(self):
        return self.points[self.valid_mask].reshape(-1, 3)  # [N,3]

    @torch.no_grad()
    def sample_rays(self, n_rays):
        self.sample_mask = generate_sample_mask(self.depth.shape, n_rays)
