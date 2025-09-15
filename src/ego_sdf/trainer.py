# import this first to avoid runtime issues with open3d, torch, etc.
from erl_geometry import MarchingCubes

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
import random
import open3d as o3d

# import ego_sdf after other packages

from ego_sdf.criterion import Criterion, CriterionConfig
from ego_sdf.frame import RGBDFrame
from ego_sdf.key_frame_set import KeyFrameSet, KeyFrameSetConfig
from ego_sdf.loggers import BasicLogger
from ego_sdf.model import SdfNetwork, SdfNetworkConfig
from ego_sdf.utils.config_abc import ConfigABC
from ego_sdf.utils.import_util import get_dataset
from ego_sdf.utils.sampling import SampleRaysConfig, generate_sdf_samples
from ego_sdf.utils.profiling import GpuTimer


@dataclass
class DataConfig(ConfigABC):
    dataset_name: str = "replica"
    dataset_args: dict = field(
        default_factory=lambda: {
            "data_path": "",
            "use_gt": True,
            "max_depth": -1.0,
        }
    )
    start_frame: int = 0
    end_frame: int = -1
    offset: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class TrainerConfig(ConfigABC):
    seed: int = 12345
    log_dir: str = "logs"
    exp_name: str = "ego_sdf"
    device: str = "cuda"
    data: DataConfig = DataConfig()
    key_frame_set: KeyFrameSetConfig = KeyFrameSetConfig()
    model: SdfNetworkConfig = SdfNetworkConfig()
    criterion: CriterionConfig = CriterionConfig()
    num_iterations_per_frame: int = 10
    first_frame_iterations: int = 10
    num_rays_per_iteration: int = 2048
    early_stopping: bool = False  # whether to stop early if loss is above average
    early_stopping_patience: int = 2  # number of iterations that loss is above average
    sample_rays: SampleRaysConfig = SampleRaysConfig()
    batch_size: int = 2048000
    lr: float = 0.01
    grad_method: str = "finite_difference"  # autodiff | finite_difference
    finite_difference_eps: float = 0.03
    final_iterations: int = 0  # number of iterations after all frames are processed, 0 means no extra iterations
    save_mesh: bool = True  # whether to save the final mesh
    mesh_resolution: float = 0.02
    mesh_iso_value: float = 0.0
    clean_mesh: bool = True
    save_slice: bool = True
    slice_center: Optional[list] = None  # if None, use the center of the scene bounding box


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg

        self.setup_seed(self.cfg.seed)

        self.data_stream = get_dataset(cfg.data.dataset_name, cfg.data.dataset_args)
        if self.cfg.data.end_frame < 0:
            self.cfg.data.end_frame = len(self.data_stream)
        self.cfg.data.start_frame = min(self.cfg.data.start_frame, len(self.data_stream) - 1)
        self.cfg.data.end_frame = min(self.cfg.data.end_frame, len(self.data_stream))
        self.current_frame_idx = self.cfg.data.start_frame

        self.key_frame_set = KeyFrameSet(
            cfg=self.cfg.key_frame_set,
            max_num_voxels=self.cfg.model.octree_cfg.init_voxel_num,
            device=self.cfg.device,
        )
        self.model = SdfNetwork(cfg.model)
        self.model.to(self.cfg.device)

        self.logger = BasicLogger(cfg.log_dir, cfg.exp_name, cfg.as_dict())

        self.scene_offset = torch.tensor(self.cfg.data.offset)
        self.epoch = 0
        self.global_step = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.criterion = Criterion(
            cfg=self.cfg.criterion,
            n_stratified=self.cfg.sample_rays.n_stratified,
            n_perturbed=self.cfg.sample_rays.n_perturbed,
        )

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self):
        for frame_id in tqdm(
            range(self.cfg.data.start_frame, self.cfg.data.end_frame),
            desc="Mapping",
            ncols=100,
            leave=False,
        ):
            frame = self.fetch_one_frame()
            if frame is None:
                self.logger.info("No more valid frames, finish mapping.")
                break  # no more valid frames

            points = self.frame_to_points(frame)
            voxels, seen_voxels = self.insert_points_to_octree(points)

            is_key_frame = self.update_key_frame_set(frame, seen_voxels)
            if is_key_frame:
                self.logger.info(f"Frame {frame_id} is selected as a key frame.")

            with GpuTimer(f"train with frame {frame_id}"):
                self.train_with_frame(
                    frame=frame,
                    num_iterations=(
                        self.cfg.first_frame_iterations if self.epoch == 0 else self.cfg.num_iterations_per_frame
                    ),
                )
            self.epoch += 1

        for _ in range(self.cfg.final_iterations):
            self.train_with_frame(None, self.cfg.num_iterations_per_frame)

        tqdm.write("Training completed.")

        self.evaluate()

    def fetch_one_frame(self) -> RGBDFrame:
        frame = None
        while self.current_frame_idx < self.cfg.data.end_frame:
            frame_idx, rgb, depth, K, pose = self.data_stream[self.current_frame_idx]
            self.current_frame_idx += 1
            pose: torch.Tensor
            if not torch.all(pose.isfinite()):  # bad pose
                continue
            frame = RGBDFrame(frame_idx, rgb, depth, K, self.scene_offset, pose)
            break
        return frame

    def frame_to_points(self, frame: RGBDFrame):
        points = frame.get_points().to(self.cfg.device)
        pose = frame.get_ref_pose().to(self.cfg.device)
        points = points @ pose[:3, :3].T + pose[:3, 3]
        return points

    @torch.no_grad()
    def insert_points_to_octree(self, points: torch.Tensor):
        voxels, seen_voxels = self.model.octree.insert_points(points)
        return voxels, seen_voxels

    @torch.no_grad()
    def find_voxel_indices(self, points: torch.Tensor):
        """
        Find the voxel indices for the given points.
        Args:
            points: (..., 3) points to find the voxel indices for

        Returns:
            (..., ) voxel indices for the given points, -1 if not exists
        """
        shape = points.shape
        voxel_indices = self.model.octree.find_voxel_indices(points.view(-1, 3))
        voxel_indices = voxel_indices.view(shape[:-1])
        return voxel_indices

    def update_key_frame_set(self, frame: RGBDFrame, seen_voxels: torch.Tensor) -> bool:
        return self.key_frame_set.add_key_frame(frame, seen_voxels)

    def select_key_frames(self) -> list[int]:
        return self.key_frame_set.select_key_frames()

    def train_with_frame(self, frame: RGBDFrame | None, num_iterations: int):
        self.model.train()
        with GpuTimer("select key frames"):
            key_frame_indices = self.key_frame_set.select_key_frames()
        with GpuTimer("sample rays"):
            rays_o_all, rays_d_all, rgb_samples_all, depth_samples_all = self.key_frame_set.sample_rays(
                num_samples=self.cfg.num_iterations_per_frame * self.cfg.num_rays_per_iteration,
                key_frame_indices=key_frame_indices,
                current_frame=frame,
                get_rgb=False,
                get_depth=True,
            )
            rays_o_all = rays_o_all.to(self.cfg.device)
            rays_d_all = rays_d_all.to(self.cfg.device)
            depth_samples_all = depth_samples_all.to(self.cfg.device)
        with GpuTimer("generate sdf samples"):
            samples = generate_sdf_samples(
                rays_d_all=rays_d_all,
                rays_o_all=rays_o_all,
                depth_samples_all=depth_samples_all,
                cfg=self.cfg.sample_rays,
                device=self.cfg.device,
            )

        # samples.sampled_xyz: (n, m, 3)
        num_rays = samples.sampled_xyz.shape[0]
        offset_points_plus = None
        offset_points_minus = None
        voxel_indices_plus = None
        voxel_indices_minus = None
        if self.cfg.grad_method == "autodiff":
            samples.sampled_xyz.requires_grad_(True)
        else:
            with GpuTimer("compute offset points for finite difference"):
                offset_points_plus, offset_points_minus = self.compute_offset_points_for_finite_diff(
                    samples.sampled_xyz
                )
            with GpuTimer("find voxel indices for offset points"):
                voxel_indices_plus = self.find_voxel_indices(offset_points_plus)  # (n, m, 3)
                voxel_indices_minus = self.find_voxel_indices(offset_points_minus)  # (n, m, 3)
        with GpuTimer("find voxel indices for sampled_xyz"):
            voxel_indices = self.find_voxel_indices(samples.sampled_xyz)  # (n, m)

        early_stopping_counter = 0
        loss_all = 0.0
        bs = int(self.cfg.batch_size / samples.sampled_xyz.shape[1])
        with GpuTimer("training iteration"):
            for itr_idx in range(self.cfg.num_iterations_per_frame):
                self.optimizer.zero_grad()
                # sdf_prior_all = []
                sdf_pred_all = []
                sdf_grad_all = []
                for i in range(0, num_rays, bs):
                    j = min(i + bs, num_rays)
                    points = samples.sampled_xyz[i:j]  # (b, m, 3)
                    voxel_indices_batch = voxel_indices[i:j]
                    _, sdf_prior, sdf_residual, sdf_pred = self.model(points, voxel_indices_batch)
                    if self.cfg.grad_method == "autodiff":
                        sdf_grad = self.compute_sdf_grad_autodiff(points, sdf_pred)
                    else:
                        sdf_grad = self.compute_sdf_grad_finite_difference(
                            points=points,
                            offset_points_plus=offset_points_plus[i:j],
                            offset_points_minus=offset_points_minus[i:j],
                            voxel_indices_plus=voxel_indices_plus[i:j],
                            voxel_indices_minus=voxel_indices_minus[i:j],
                        )[0]

                    # sdf_prior_all.append(sdf_prior)
                    sdf_pred_all.append(sdf_pred)  # (b, m)
                    sdf_grad_all.append(sdf_grad)  # (b, m, 3)

                if len(sdf_pred_all) == 1:
                    # sdf_prior_all = sdf_prior_all[0]
                    sdf_pred_all = sdf_pred_all[0]
                    sdf_grad_all = sdf_grad_all[0]
                else:
                    # sdf_prior_all = torch.cat(sdf_prior_all, dim=0)
                    sdf_pred_all = torch.cat(sdf_pred_all, dim=0)
                    sdf_grad_all = torch.cat(sdf_grad_all, dim=0)

                loss, loss_dict = self.criterion(
                    pred_sdf=sdf_pred_all,
                    pred_grad=sdf_grad_all,
                    gt_sdf_perturb=samples.perturbation_sdf,
                    gt_sdf_stratified=samples.stratified_sdf,
                )
                loss.backward()
                self.optimizer.step()
                self.global_step += 1

                tqdm.write(f"loss_dict: {loss_dict}")
                for k, v in loss_dict.items():
                    self.logger.tb.add_scalar(f"loss/{k}", v, self.global_step)

                loss_all += loss_dict["total_loss"]
                loss_mean = loss_all / (itr_idx + 1)
                if self.cfg.early_stopping and self.global_step > self.cfg.num_iterations_per_frame:
                    if loss_dict["total_loss"] > loss_mean:  # loss is above average
                        early_stopping_counter += 1
                    if early_stopping_counter >= self.cfg.early_stopping_patience:
                        break

    def compute_sdf_grad_autodiff(self, points: torch.Tensor, pred_sdf: torch.Tensor):
        sdf_grad = torch.autograd.grad(
            outputs=pred_sdf,
            inputs=[points],
            grad_outputs=torch.ones_like(pred_sdf),
            create_graph=True,
            # retain_graph=True,
        )[0]
        return sdf_grad

    @torch.no_grad()
    def compute_offset_points_for_finite_diff(self, points: torch.Tensor):
        """
        Compute the offset points for finite difference gradient estimation.
        Args:
            points: (..., 3) points to compute the offset points for

        Returns:
            (..., 3, 3) tensor of points + offset
            (..., 3, 3) tensor of points - offset
        """
        eps = self.cfg.finite_difference_eps
        offset_points_plus = []
        offset_points_minus = []
        for i in range(3):
            points_plus = points.clone()
            points_plus[..., i] += eps  # (..., 3)
            offset_points_plus.append(points_plus)

            points_minus = points.clone()
            points_minus[..., i] -= eps  # (..., 3)
            offset_points_minus.append(points_minus)

        offset_points_plus = torch.stack(offset_points_plus, dim=-2)  # (..., 3, 3)
        offset_points_minus = torch.stack(offset_points_minus, dim=-2)  # (..., 3, 3)

        return offset_points_plus, offset_points_minus

    def compute_sdf_grad_finite_difference(
        self,
        points: torch.Tensor,
        offset_points_plus: torch.Tensor = None,
        offset_points_minus: torch.Tensor = None,
        voxel_indices_plus: torch.Tensor = None,
        voxel_indices_minus: torch.Tensor = None,
    ):
        """
        Compute the gradient of the SDF at the given points using finite difference.
        Args:
            points: (..., 3) points to compute the gradient for
            offset_points_plus: (..., 3, 3) tensor of points + offset, if None, will be computed
            offset_points_minus: (..., 3, 3) tensor of points - offset, if None, will be computed
            voxel_indices_plus: (..., 3) voxel indices for offset_points_plus, if None, will be computed
            voxel_indices_minus: (..., 3) voxel indices for offset_points_minus, if None, will be computed

        Returns:
            (..., 3) gradient of the SDF at the given points
            (..., 3, 3) offset_points_plus
            (..., 3, 3) offset_points_minus
            (..., 3) voxel_indices_plus
            (..., 3) voxel_indices_minus
        """
        n = points.shape[0]
        eps = self.cfg.finite_difference_eps
        if offset_points_plus is None or offset_points_minus is None:
            offset_points_plus, offset_points_minus = self.compute_offset_points_for_finite_diff(points)
        voxel_indices_plus, _, _, sdf_plus = self.model(offset_points_plus, voxel_indices_plus)
        voxel_indices_minus, _, _, sdf_minus = self.model(offset_points_minus, voxel_indices_minus)

        grad = (sdf_plus - sdf_minus) / (2 * eps)

        return grad, offset_points_plus, offset_points_minus, voxel_indices_plus, voxel_indices_minus

    @torch.no_grad()
    def extract_mesh_impl(
        self,
        bound: list,
        grid_point_voxel_indices: torch.Tensor,
        sdf_np: np.ndarray,
    ):
        """

        Args:
            bound: list of boundary min and max, [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            grid_point_voxel_indices: (nx, ny, nz)
            sdf_np: (nx, ny, nz)

        Returns:
            mesh: open3d.geometry.TriangleMesh
        """

        grid_shape = grid_point_voxel_indices.shape

        # marching cubes step 1: collect valid voxels
        mc = MarchingCubes()
        valid_voxels = mc.collect_valid_cubes(
            grid_shape=grid_shape,
            grid_values=sdf_np.flatten(),  # (nx*ny*nz, )
            iso_value=self.cfg.mesh_iso_value,
            row_major=True,
            parallel=True,
        )

        if self.cfg.clean_mesh:

            voxel_coords = torch.stack([torch.from_numpy(voxel.coords) for voxel in valid_voxels], dim=0)
            voxel_coords = voxel_coords.long().to(self.cfg.device)

            # voxel_coords: (n_valid, 3)

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
                device=self.cfg.device,
            )

            vertex_coords = voxel_coords.view(-1, 1, 3) + voxel_vertex_offsets.view(1, 8, 3)  # (n_valid, 8, 3)
            vertex_indices = grid_point_voxel_indices[
                vertex_coords[..., 0].flatten(), vertex_coords[..., 1].flatten(), vertex_coords[..., 2].flatten()
            ].reshape(-1, 8)
            voxel_sizes = self.model.octree.get_voxel_discrete_size(vertex_indices)
            valid_mask = torch.any((voxel_sizes >= 1) & (voxel_sizes <= 2), dim=1)  # (n_valid, )
            valid_mask = torch.nonzero(valid_mask, as_tuple=False).cpu().flatten().tolist()
            valid_voxels = [valid_voxels[i] for i in valid_mask]

        # marching cubes step 2: extract mesh
        vertices, triangles, face_normals = mc.process_valid_cubes(
            valid_cubes=valid_voxels,
            coords_min=[bound[0][0], bound[1][0], bound[2][0]],
            grid_res=[
                (bound[0][1] - bound[0][0]) / grid_shape[0],
                (bound[1][1] - bound[1][0]) / grid_shape[1],
                (bound[2][1] - bound[2][0]) / grid_shape[2],
            ],
            grid_shape=grid_shape,
            grid_values=sdf_np.flatten(),
            iso_value=self.cfg.mesh_iso_value,
            row_major=True,
            parallel=True,
        )

        # save mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.triangle_normals = o3d.utility.Vector3dVector(face_normals)

        return mesh

    @torch.no_grad()
    def extract_mesh(self, bound: list):
        """
        Extract the mesh with the given resolution and bound using marching cubes.
        Args:
            bound: list of boundary min and max, [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        Returns:
            mesh_prior: open3d.geometry.TriangleMesh, the mesh extracted from the prior SDF
            mesh: open3d.geometry.TriangleMesh, the mesh extracted from the final SDF
        """

        self.model.eval()

        # create grid
        res = self.cfg.mesh_resolution
        x = torch.arange(bound[0][0], bound[0][1], res)  # (nx,)
        y = torch.arange(bound[1][0], bound[1][1], res)  # (ny,)
        z = torch.arange(bound[2][0], bound[2][1], res)  # (nz,)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="xy")  # (nx, ny, nz)
        grid_points = torch.stack((xx, yy, zz), dim=-1)  # (nx, ny, nz, 3)

        # evaluate SDF on the grid
        grid_shape = grid_points.shape[:-1]
        grid_points = grid_points.view(-1, grid_points.shape[-1]).to(self.cfg.device)
        voxel_indices_all = []
        sdf_prior_all = []
        sdf_all = []
        for i in range(0, grid_points.shape[0], self.cfg.batch_size):
            j = min(i + self.cfg.batch_size, grid_points.shape[0])
            points = grid_points[i:j]
            voxel_indices, sdf_prior, _, sdf = self.model(points)
            voxel_indices_all.append(voxel_indices)
            sdf_prior_all.append(sdf_prior)
            sdf_all.append(sdf)
        if len(voxel_indices_all) == 1:
            voxel_indices_all = voxel_indices_all[0]
            sdf_prior_all = sdf_prior_all[0]
            sdf_all = sdf_all[0]
        else:
            voxel_indices_all = torch.cat(voxel_indices_all, dim=0)
            sdf_prior_all = torch.cat(sdf_prior_all, dim=0)
            sdf_all = torch.cat(sdf_all, dim=0)

        voxel_indices_all = voxel_indices_all.view(grid_shape)
        sdf_prior_np = sdf_prior_all.view(grid_shape).cpu().double().numpy()  # (nx, ny, nz)  double for marching cubes
        sdf_np = sdf_all.view(grid_shape).cpu().double().numpy()  # (nx, ny, nz)  double for marching cubes

        mesh_prior = self.extract_mesh_impl(bound, voxel_indices_all, sdf_prior_np)
        mesh = self.extract_mesh_impl(bound, voxel_indices_all, sdf_np)

        return mesh_prior, mesh

    @torch.no_grad()
    def extract_slice(self, axis: int, pos: float, resolution: float):
        """
        Extract a 2D slice of the prediction values along the given axis at the given position.
        Args:
            axis: int, 0 for x, 1 for y, 2 for z
            pos: float, position along the axis to extract the slice
            resolution: float, resolution of the slice

        Returns:
            bound_slice: (2, ) list of boundary min and max for the slice
            sdf_prior: (n, m) torch tensor of SDF prior values on the slice
            sdf_residual: (n, m) torch tensor of SDF residual values on the slice
            sdf: (n, m) torch tensor of final SDF values on the slice
        """
        assert axis in [0, 1, 2], "axis must be 0, 1, or 2"
        self.model.eval()

        # create grid
        bound = self.cfg.model.residual_net_cfg.bound
        if axis == 0:
            y = torch.arange(bound[1][0], bound[1][1], resolution)  # (ny,)
            z = torch.arange(bound[2][0], bound[2][1], resolution)  # (nz,)
            yy, zz = torch.meshgrid(y, z, indexing="xy")
            grid_points = torch.stack((torch.full_like(yy, pos), yy, zz), dim=-1)
            bound_slice = [bound[1], bound[2]]
        elif axis == 1:
            x = torch.arange(bound[0][0], bound[0][1], resolution)  # (nx,)
            z = torch.arange(bound[2][0], bound[2][1], resolution)  # (nz,)
            xx, zz = torch.meshgrid(x, z, indexing="xy")
            grid_points = torch.stack((xx, torch.full_like(xx, pos), zz), dim=-1)
            bound_slice = [bound[0], bound[2]]
        else:  # axis == 2
            x = torch.arange(bound[0][0], bound[0][1], resolution)  # (nx,)
            y = torch.arange(bound[1][0], bound[1][1], resolution)  # (ny,)
            xx, yy = torch.meshgrid(x, y, indexing="xy")
            grid_points = torch.stack((xx, yy, torch.full_like(xx, pos)), dim=-1)
            bound_slice = [bound[0], bound[1]]

        # evaluate SDF on the grid
        bs = int(self.cfg.batch_size / grid_points.shape[1])
        grid_points = grid_points.to(self.cfg.device)
        sdf_prior_all = []
        sdf_residual_all = []
        sdf_all = []
        for i in range(0, grid_points.shape[0], bs):
            j = min(i + bs, grid_points.shape[0])
            points = grid_points[i:j]  # (b, m, 3)
            _, sdf_prior, sdf_residual, sdf_pred = self.model(points)
            sdf_prior_all.append(sdf_prior)
            sdf_residual_all.append(sdf_residual)
            sdf_all.append(sdf_pred)
        if len(sdf_prior_all) == 1:
            sdf_prior_all = sdf_prior_all[0]
            sdf_residual_all = sdf_residual_all[0]
            sdf_all = sdf_all[0]
        else:
            sdf_prior_all = torch.cat(sdf_prior_all, dim=0)
            sdf_residual_all = torch.cat(sdf_residual_all, dim=0)
            sdf_all = torch.cat(sdf_all, dim=0)

        return bound_slice, sdf_prior_all, sdf_residual_all, sdf_all

    def evaluate(self):
        bound = self.cfg.model.residual_net_cfg.bound
        if self.cfg.save_mesh:
            mesh_prior, mesh = self.extract_mesh(bound=bound)
            mesh_filename_prior = os.path.join(self.logger.mesh_dir, "final_mesh_prior.ply")
            mesh_filename = os.path.join(self.logger.mesh_dir, "final_mesh.ply")
            o3d.io.write_triangle_mesh(mesh_filename_prior, mesh_prior)
            o3d.io.write_triangle_mesh(mesh_filename, mesh)

        if self.cfg.save_slice:
            import matplotlib.pyplot as plt

            slice_configs = [
                {
                    "axis_name": "x",
                    "xlabel": "y (m)",
                    "ylabel": "z (m)",
                },
                {
                    "axis_name": "y",
                    "xlabel": "x (m)",
                    "ylabel": "z (m)",
                },
                {
                    "axis_name": "z",
                    "xlabel": "x (m)",
                    "ylabel": "y (m)",
                },
            ]
            fontsize = 12
            for axis in range(3):
                if self.cfg.slice_center is None:
                    pos = 0.5 * (bound[axis][0] + bound[axis][1])
                else:
                    pos = self.cfg.slice_center[axis]
                bound_slice, sdf_prior, sdf_residual, sdf = self.extract_slice(
                    axis=axis, pos=pos, resolution=self.cfg.mesh_resolution
                )
                sdf_prior = sdf_prior.cpu().numpy()
                sdf_residual = sdf_residual.cpu().numpy()
                sdf = sdf.cpu().numpy()

                slice_config = slice_configs[axis]
                axis_name = slice_config["axis_name"]

                for slice_name, slice_values in zip(
                    ["sdf_prior", "sdf_residual", "sdf"],
                    [sdf_prior, sdf_residual, sdf],
                ):
                    plt.figure()
                    im = plt.imshow(
                        slice_values,
                        extent=[bound_slice[0][0], bound_slice[0][1], bound_slice[1][0], bound_slice[1][1]],
                        origin="lower",
                        cmap="jet",
                    )
                    plt.colorbar(im, shrink=0.8)
                    plt.xlabel(slice_config["xlabel"], fontsize=fontsize)
                    plt.ylabel(slice_config["ylabel"], fontsize=fontsize)
                    plt.title(f"At {axis_name} = {pos:.2f} m", fontsize=fontsize)
                    plt.tight_layout()
                    img_path = os.path.join(self.logger.misc_dir, f"slice_{axis_name}_{slice_name}.png")
                    plt.savefig(img_path, dpi=300)
                    plt.close()

        tqdm.write("Evaluation completed.")


def main():
    original_stdout = sys.stdout

    class TqdmStdout:

        def write(self, s):
            if s.strip() != "":
                tqdm.write(s, file=original_stdout)  # preserve original spacing/newlines

        def flush(self):
            original_stdout.flush()

    sys.stdout = TqdmStdout()

    parser = TrainerConfig.get_argparser()
    cfg: TrainerConfig = parser.parse_args()
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
