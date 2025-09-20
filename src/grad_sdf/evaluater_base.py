import os
from functools import reduce
from typing import Callable, Dict, List

import trimesh
from scipy.spatial import cKDTree

from grad_sdf import MarchingCubes, torch, o3d, np


class EvaluaterBase:

    def __init__(
        self,
        model_forward_func: Callable[[torch.nn.Module, torch.Tensor], Dict[str, torch.Tensor]],
        model: torch.nn.Module | None = None,
        model_path: str | None = None,
        model_create_func: Callable[[str], torch.nn.Module] | None = None,
        device: str = "cuda",
    ):
        """
        Base class for evaluators.
        Args:
            model_forward_func: function to forward the model, takes model and input tensor, returns a dict of output tensors
            model: optional, if provided, use this model
            model_path: optional, if model is not provided, load the model from this path
            model_create_func: optional, function to create the model, takes model_path as input, returns the model
            device: device to run the model on
        """
        assert model_forward_func is not None
        self.model_forward_func = model_forward_func
        self.device = device

        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        else:
            assert model_path is not None
            assert model_create_func is not None

            self.model: torch.nn.Module = model_create_func(model_path).to(self.device)
            self.model.eval()

    @staticmethod
    def _sdf_metrics(sdf_pred: torch.Tensor, sdf_gt: torch.Tensor):
        diff = sdf_pred - sdf_gt
        return dict(mae=diff.abs().mean().item(), rmse=(diff**2).mean().sqrt().item())

    @staticmethod
    def _grad_metrics(grad_pred: torch.Tensor, grad_gt: torch.Tensor):
        grad_pred /= grad_pred.norm(dim=-1, keepdim=True) + 1e-8
        grad_gt /= grad_gt.norm(dim=-1, keepdim=True) + 1e-8
        cos_sim = (grad_pred * grad_gt).sum(dim=-1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        angle_diff = torch.acos(cos_sim)  # in radians
        return angle_diff.abs().mean().item()

    def sdf_and_grad_metrics(
        self,
        test_set_dir: str,
        sdf_fields: list[str],
        grad_method: str = "autograd",
        eps: float = 0.001,
    ):
        grid_points = np.load(os.path.join(test_set_dir, "grid_points.npy"))
        gt_sdf_values = np.load(os.path.join(test_set_dir, "gt_sdf_values.npy"))
        gt_sdf_grad = np.load(os.path.join(test_set_dir, "gt_sdf_grad.npy"))

        grid_points = torch.from_numpy(grid_points).float().to(self.device)
        gt_sdf_values = torch.from_numpy(gt_sdf_values).float().to(self.device)
        gt_sdf_grad = torch.from_numpy(gt_sdf_grad).float().to(self.device)

        autograd = grad_method == "autograd"
        grid_points.requires_grad_(autograd)
        self.model.eval()
        with torch.set_grad_enabled(autograd):
            sdf_pred = self.model_forward_func(self.model, grid_points)
        if autograd:
            sdf_grad = torch.autograd.grad(
                outputs=[sdf_pred[k] for k in sdf_fields],
                inputs=[grid_points],
                grad_outputs=[torch.ones_like(sdf_pred[k]) for k in sdf_fields],
                create_graph=True,
                allow_unused=True,
            )
            sdf_grad = {k: g for k, g in zip(sdf_fields, sdf_grad)}
        else:
            sdf_grad = {k: torch.zeros_like(grid_points) for k in sdf_fields}
            for i in range(3):
                offset = torch.zeros((3,), device=grid_points.device)
                offset[i] = eps
                offset = offset.view(*[1] * (grid_points.ndim - 1), 3)
                sdf_plus = self.model_forward_func(self.model, grid_points + offset)
                sdf_minus = self.model_forward_func(self.model, grid_points - offset)
                for k in sdf_fields:
                    sdf_grad[k][..., i] = (sdf_plus[k] - sdf_minus[k]) / (2 * eps)

        near_surface_mask = np.load(os.path.join(test_set_dir, "near_surface_mask.npy"))
        far_away_mask = np.load(os.path.join(test_set_dir, "far_away_mask.npy"))
        all_mask = near_surface_mask | far_away_mask

        sdf_metrics = dict(
            near_surface={
                k: self._sdf_metrics(sdf_pred[k][near_surface_mask], gt_sdf_values[near_surface_mask])
                for k in sdf_fields
            },
            far_away={
                k: self._sdf_metrics(sdf_pred[k][far_away_mask], gt_sdf_values[far_away_mask]) for k in sdf_fields
            },
            all={k: self._sdf_metrics(sdf_pred[k][all_mask], gt_sdf_values[all_mask]) for k in sdf_fields},
        )

        grad_metrics = dict(
            near_surface={
                k: self._grad_metrics(sdf_grad[k][near_surface_mask], gt_sdf_grad[near_surface_mask])
                for k in sdf_fields
            },
            far_away={
                k: self._grad_metrics(sdf_grad[k][far_away_mask], gt_sdf_grad[far_away_mask]) for k in sdf_fields
            },
            all={k: self._grad_metrics(sdf_grad[k][all_mask], gt_sdf_grad[all_mask]) for k in sdf_fields},
        )

        return dict(sdf_metrics=sdf_metrics, grad_metrics=grad_metrics)

    @staticmethod
    def mesh_metrics(
        pred_mesh_path: str,
        gt_mesh_path: str,
        threshold: float = 0.05,
        num_samples: int = 200_000,
        seed: int = 0,
    ):
        pred_mesh = trimesh.load_mesh(pred_mesh_path)
        gt_mesh = trimesh.load_mesh(gt_mesh_path)

        pred_pts = trimesh.sample.sample_surface(pred_mesh, num_samples, seed=seed)
        gt_pts = trimesh.sample.sample_surface(gt_mesh, num_samples, seed=seed)

        pred_tree = cKDTree(pred_pts)
        gt_tree = cKDTree(gt_pts)

        dist_pred_to_gt, _ = gt_tree.query(pred_pts, k=1, workers=-1)
        dist_gt_to_pred, _ = pred_tree.query(gt_pts, k=1, workers=-1)

        completion_ratio = np.mean(dist_gt_to_pred < threshold).item()
        completion = np.mean(dist_gt_to_pred)

        accuracy = np.mean(dist_pred_to_gt)
        chamfer = (completion + accuracy) / 2.0

        tp = np.sum(dist_pred_to_gt < threshold).item()
        fp = num_samples - tp
        fn = np.sum(dist_gt_to_pred >= threshold).item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return dict(
            completion_ratio=completion_ratio,
            completion=completion,
            accuracy=accuracy,
            chamfer=chamfer,
            precision=precision,
            recall=recall,
            f1=f1,
            threshold=threshold,
            num_samples=num_samples,
            seed=seed,
        )

    def compute_metrics(
        self,
        test_set_dir: str,
        sdf_fields: List[str],
        grad_method: str,
        eps: float,
        pred_mesh_path: str,
        gt_mesh_path: str,
        threshold: float = 0.05,
        num_samples: int = 200_000,
        seed: int = 0,
    ):
        metrics = self.sdf_and_grad_metrics(test_set_dir, sdf_fields, grad_method, eps)
        metrics["mesh_metrics"] = self.mesh_metrics(pred_mesh_path, gt_mesh_path, threshold, num_samples, seed)
        return metrics

    @torch.no_grad()
    def extract_mesh(
        self,
        bound_min: List[float],
        bound_max: List[float],
        grid_resolution: float,
        fields: List[str],
        iso_value: float = 0.0,
        voxel_filter: Callable[[torch.Tensor, torch.Tensor], List[int]] | None = None,
    ):
        """
        Extract mesh from the model using marching cubes.
        Args:
            bound_min: Minimum bound of the 3D grid (list of 3 floats)
            bound_max: Maximum bound of the 3D grid (list of 3 floats)
            grid_resolution: Resolution of the grid (float)
            fields: List of fields to extract
            iso_value: Iso value for marching cubes (float)
            voxel_filter: Optional callable to filter valid voxels before mesh extraction.
                          It takes a (N, 3) tensor of voxel coordinates and a (X, Y, Z) tensor of voxel indices for
                          each grid point, and returns a filtered list of valid voxel indices.
        Returns:
            list of open3d.geometry.TriangleMesh: Extracted mesh
        """

        self.model.eval()

        x_size = int((bound_max[0] - bound_min[0]) / grid_resolution) + 1
        y_size = int((bound_max[1] - bound_min[1]) / grid_resolution) + 1
        z_size = int((bound_max[2] - bound_min[2]) / grid_resolution) + 1
        grid_shape = [x_size, y_size, z_size]

        x = torch.linspace(bound_min[0], bound_max[0], x_size)
        y = torch.linspace(bound_min[1], bound_max[1], y_size)
        z = torch.linspace(bound_min[2], bound_max[2], z_size)
        grid_points = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)

        results = self.model_forward_func(self.model, grid_points.to(self.device))

        meshes: List[o3d.geometry.TriangleMesh] = []
        for field in fields:
            assert field in results, f"Field {field} not found in model output"
            values = results[field].cpu().numpy().astype(np.float64)
            mc = MarchingCubes()
            valid_voxels = mc.collect_valid_cubes(
                grid_shape=grid_shape,
                grid_values=values.flatten(),
                iso_value=iso_value,
                row_major=True,
                parallel=True,
            )
            # convert list of list to a single list
            valid_voxels = reduce(lambda x, y: x + y, valid_voxels)

            if voxel_filter is not None and len(valid_voxels) > 0:
                voxel_coords = torch.tensor(np.stack([voxel.coords for voxel in valid_voxels], axis=0))
                voxel_coords = voxel_coords.long().to(self.device)
                valid_voxels = [valid_voxels[i] for i in voxel_filter(voxel_coords, results["voxel_indices"])]

            vertices, triangles, face_normals = mc.process_valid_cubes(
                valid_cubes=[valid_voxels],
                coords_min=[bound_min[0], bound_min[1], bound_min[2]],
                grid_res=[
                    (bound_max[0] - bound_min[0]) / x_size,
                    (bound_max[1] - bound_min[1]) / y_size,
                    (bound_max[2] - bound_min[2]) / z_size,
                ],
                grid_shape=grid_shape,
                grid_values=values.flatten(),
                iso_value=iso_value,
                row_major=True,
                parallel=True,
            )

            # save mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.triangle_normals = o3d.utility.Vector3dVector(face_normals)

            meshes.append(mesh)

        return meshes

    @torch.no_grad()
    def extract_slice(self, axis: int, pos: float, resolution: float, bound_min: List[float], bound_max: List[float]):
        """
        Extract a 2D slice of the prediction values along the given axis at the given position.
        Args:
            axis: int, 0 for x, 1 for y, 2 for z
            pos: float, position along the axis to extract the slice
            resolution: float, resolution of the slice
            bound_min: Minimum bound of the 3D grid (list of 3 floats)
            bound_max: Maximum bound of the 3D grid (list of 3 floats)

        Returns:
            dict with keys:
                slice_bound: (2, ) list of boundary min and max for the slice
                sdf_prior: (n, m) torch tensor of SDF prior values on the slice
                sdf_residual: (n, m) torch tensor of SDF residual values on the slice
                sdf: (n, m) torch tensor of final SDF values on the slice
        """
        assert axis in [0, 1, 2], "axis must be 0, 1, or 2"
        self.model.eval()

        if axis == 0:
            y = torch.arange(bound_min[1], bound_max[1], resolution)  # (ny,)
            z = torch.arange(bound_min[2], bound_max[2], resolution)  # (nz,)
            yy, zz = torch.meshgrid(y, z, indexing="xy")
            grid_points = torch.stack((torch.full_like(yy, pos), yy, zz), dim=-1)
            slice_bound = [bound_min[1], bound_max[1]], [bound_min[2], bound_max[2]]
        elif axis == 1:
            x = torch.arange(bound_min[0], bound_max[0], resolution)  # (nx,)
            z = torch.arange(bound_min[2], bound_max[2], resolution)  # (nz,)
            xx, zz = torch.meshgrid(x, z, indexing="xy")
            grid_points = torch.stack((xx, torch.full_like(xx, pos), zz), dim=-1)
            slice_bound = [bound_min[0], bound_max[0]], [bound_min[2], bound_max[2]]
        else:  # axis == 2
            x = torch.arange(bound_min[0], bound_max[0], resolution)  # (nx,)
            y = torch.arange(bound_min[1], bound_max[1], resolution)  # (ny,)
            xx, yy = torch.meshgrid(x, y, indexing="xy")
            grid_points = torch.stack((xx, yy, torch.full_like(xx, pos)), dim=-1)
            slice_bound = [bound_min[0], bound_max[0]], [bound_min[1], bound_max[1]]

        results = self.model_forward_func(self.model, grid_points.to(self.device))
        results["slice_bound"] = torch.tensor(slice_bound)
        return results
