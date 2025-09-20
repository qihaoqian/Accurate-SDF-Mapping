import sparse_octree
import torch

from grad_sdf.ga_trilinear import ga_trilinear
from grad_sdf.octree_config import OctreeConfig


class SemiSparseOctree(torch.nn.Module):
    def __init__(self, cfg: OctreeConfig):
        super(SemiSparseOctree, self).__init__()
        self.cfg = cfg
        self.svo = sparse_octree.Octree()
        self.svo.init(
            1 << self.cfg.tree_depth,
            self.cfg.init_voxel_num,
            self.cfg.resolution,
            self.cfg.full_depth,
        )
        # Initialize learnable parameters for SDF and gradient priors of each vertex
        self.sdf_priors = torch.nn.Parameter(
            torch.zeros((self.cfg.init_voxel_num,), dtype=torch.float32),
            requires_grad=True,
        )
        self.grad_priors = torch.nn.Parameter(
            torch.zeros((self.cfg.init_voxel_num, 3), dtype=torch.float32),
            requires_grad=True,
        )

        n = self.cfg.init_voxel_num
        self.register_buffer("voxels", torch.zeros((n, 4), dtype=torch.float32))
        self.register_buffer("voxel_centers", torch.zeros((n, 3), dtype=torch.float32))
        self.register_buffer("vertex_indices", torch.zeros((n, 8), dtype=torch.int32))
        self.register_buffer("structure", torch.zeros((n, 9), dtype=torch.int32))

        self.voxels: torch.Tensor  # (N, 4) [x, y, z, voxel_size]
        self.voxel_centers: torch.Tensor  # (N, 3) in meter
        self.vertex_indices: torch.Tensor  # (N, 8) index of vertices, -1 if not exists
        self.structure: torch.Tensor  # (N, 9) [children(8), voxel_size]

    @torch.no_grad()
    def insert_points(self, points: torch.Tensor):
        """
        Inserts points into the octree.
        Args:
            points: (n_points, 3) point cloud in world coordinates
        Returns:
            voxels_unique: (n_unique, 3) unique voxel coordinates inserted
            svo_idx: (n_unique,) index of the voxel for each voxel
        """
        voxels = torch.div(points, self.cfg.resolution, rounding_mode="floor").long()  # Divides each element
        voxels_raw, inverse_indices, counts = torch.unique(voxels, dim=0, return_inverse=True, return_counts=True)
        voxels_valid = voxels_raw[counts > 3]  # (n_valid, 3) of grid coordinates
        voxels_unique = torch.unique(voxels_valid, dim=0)  # (n_unique, 3) of grid coordinates
        voxels_svo, children_svo, vertices_svo, svo_mask, svo_idx = self.svo.insert(voxels_unique.cpu().int())

        # voxels_svo: (N, 4) [x, y, z, voxel_size]
        # children_svo: (N, 8) index of children, -1 if not exists
        # vertices_svo: (N, 8) index of vertices, -1 if not exists
        # svo_mask: (N, 1) 1 if the voxel is valid, 0 if not
        # svo_idx: (n_unique, 1) voxel index for each input point, -1 if not inserted

        # svo_mask = svo_mask[:, 0] > 0
        # voxels_svo = voxels_svo[svo_mask]
        # children_svo = children_svo[svo_mask]
        # vertices_svo = vertices_svo[svo_mask]
        # voxels_unique = voxels_unique[svo_idx.view(-1) >= 0]

        # update grid state
        self.voxels = voxels_svo.to(self.sdf_priors.device)
        self.voxel_centers = (self.voxels[:, :3] + self.voxels[:, [-1]] * 0.5) * self.cfg.resolution
        self.vertex_indices = vertices_svo.to(self.sdf_priors.device)
        self.structure = torch.cat([children_svo, voxels_svo[:, [-1]]], dim=1).int().to(self.sdf_priors.device)

        return voxels_unique, svo_idx

    @torch.no_grad()
    def get_voxel_discrete_size(self, voxel_indices: torch.Tensor) -> torch.Tensor:
        """
        Get the voxel sizes for the given voxel indices.
        Args:
            voxel_indices: (...) index of the voxels

        Returns:
            (..., ) voxel discrete sizes
        """
        assert self.voxels is not None, "Octree is empty. Please insert points first."
        assert voxel_indices.dtype == torch.long, "voxel_indices must be of type torch.long"

        voxel_sizes = self.structure[voxel_indices.view(-1), -1]  # (..., ) discrete sizes
        voxel_sizes = voxel_sizes.view(voxel_indices.shape)
        voxel_sizes[voxel_indices < 0] = -1  # set invalid voxel sizes to -1
        return voxel_sizes

    @torch.no_grad()
    def find_voxel_indices(self, points: torch.Tensor):
        """
        Finds the voxel indices for the given points.
        Args:
            points: (n_points, 3) point cloud in world coordinates
        Returns:
            voxel_indices: (n_points,) index of the voxel for each point, -1 if not exists
        """
        assert self.voxels is not None, "Octree is empty. Please insert points first."
        assert points.device == self.voxels.device, "Points and octree must be on the same device."

        device = points.device
        n_points = points.shape[0]
        root_idx = 0

        # Initialize result to -1
        voxel_indices = torch.full((n_points,), -1, dtype=torch.long, device=device)

        # Point indices still being traversed and their current node row numbers
        active_pts = torch.arange(n_points, device=device)  # [A]
        cur_nodes = torch.full_like(active_pts, root_idx)  # Initially all at root node

        for i in range(self.cfg.tree_depth + 1):
            if active_pts.numel() == 0:
                break

            # Calculate child numbers
            c = self.voxel_centers[cur_nodes]  # [A,3]
            ge_mask = (points[active_pts] >= c).long()  # [A,3]
            child_id = ge_mask[:, 0] + (ge_mask[:, 1] << 1) + (ge_mask[:, 2] << 2)
            child_idx = self.structure[cur_nodes, child_id].long()  # [A]

            # Hit condition: reach a leaf node or no expected child
            hit_mask = child_idx == -1
            if hit_mask.any():
                voxel_indices[active_pts[hit_mask]] = cur_nodes[hit_mask]
            # Continue only with those that didn't hit
            keep_mask = ~hit_mask
            if not keep_mask.any():
                break

            active_pts = active_pts[keep_mask]
            cur_nodes = child_idx[keep_mask]

        return voxel_indices

    def forward(self, points: torch.Tensor, voxel_indices: torch.Tensor = None, batch_size: int = -1) -> torch.Tensor:
        """
        Forward pass of the octree.
        Args:
            points: (n_points, 3) point cloud in world coordinates
            voxel_indices: (n_points,) index of the voxel for each point, -1 if not exists
            batch_size: int, number of points to process in a batch. If -1, process all points at once.
        Returns:
            (n_points, ) of sdf predictions.
            (n_points,) of voxel indices for each point. If voxel_indices is provided, it will be returned as is.
        """
        if voxel_indices is None:
            voxel_indices = self.find_voxel_indices(points)

        if batch_size > 0:
            n_points = points.shape[0]
            sdf_preds = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                sdf_preds[start:end] = self.forward(points[start:end], voxel_indices[start:end])
            return sdf_preds, voxel_indices

        # Implement the forward pass logic here
        assert voxel_indices.dtype == torch.long

        # find the voxel centers for each point
        voxel_centers = self.voxel_centers[voxel_indices]  # (n_points, 3)
        # find the vertex indices for each point
        vertex_indices = self.vertex_indices[voxel_indices]  # (n_points, 8)
        # find the voxel sizes for each point
        voxel_sizes = self.structure[voxel_indices, -1:]  # (n_points, 1)
        # get the sdf priors and gradient priors for each vertex
        vertex_sdf_priors = self.sdf_priors[vertex_indices]  # (n_points, 8)
        vertex_grad_priors = self.grad_priors[vertex_indices]  # (n_points, 8, 3)

        sdf_preds = ga_trilinear(
            points=points,
            voxel_centers=voxel_centers,
            voxel_sizes=voxel_sizes,
            vertex_values=vertex_sdf_priors,
            vertex_grad=vertex_grad_priors,
            resolution=self.cfg.resolution,
            gradient_augmentation=self.cfg.gradient_augmentation,
        )
        return sdf_preds, voxel_indices
