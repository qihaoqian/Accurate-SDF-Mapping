import random

import numpy as np
import torch
from tqdm import tqdm
import os
from copy import deepcopy
from criterion import Criterion
from frame import DepthFrame
from functions.render_helpers import bundle_adjust_frames
from loggers import BasicLogger
from utils.import_util import get_decoder, get_property
from utils.keyframe_util import multiple_max_set_coverage
from functions.render_helpers import get_features
from torch.utils.tensorboard import SummaryWriter

torch.classes.load_library(
    "third_party/sparse_octree/build/lib.linux-x86_64-cpython-310/svo.cpython-310-x86_64-linux-gnu.so")


class Mapping:
    def __init__(self, args, logger: BasicLogger, data_stream=None, **kwargs):
        super().__init__()
        self.args = args
        self.logger = logger
        mapper_specs = args.mapper_specs
        debug_args = args.debug_args
        data_specs = args.data_specs
        self.sample_specs = args.sample_specs
        self.h1 = args.criteria["h1"]
        self.writer = SummaryWriter(log_dir=self.logger.misc_dir)

        # get data stream
        if data_stream != None:
            self.data_stream = data_stream
            self.start_frame = mapper_specs["start_frame"]
            self.end_frame = mapper_specs["end_frame"]
            if self.end_frame == -1:
                self.end_frame = len(self.data_stream)
            self.start_frame = min(self.start_frame, len(self.data_stream))
            self.end_frame = min(self.end_frame, len(self.data_stream))

        self.decoder = get_decoder(args).cuda()
        self.loss_criteria = Criterion(args)
        # keyframes set
        self.kf_graph = []
        # used for Coverage-maximizing keyframe selection
        self.kf_seen_voxel = []
        self.kf_seen_voxel_num = []
        self.kf_svo_idx = []

        # optional args
        self.ckpt_freq = get_property(args, "ckpt_freq", -1)
        self.final_iter = get_property(mapper_specs, "final_iter", 0)
        self.mesh_res = get_property(mapper_specs, "mesh_res", 8)
        self.save_data_freq = get_property(debug_args, "save_data_freq", 0)

        # required args
        self.use_adaptive_ending = mapper_specs["use_adaptive_ending"]
        self.voxel_size = mapper_specs["voxel_size"]
        self.kf_window_size = mapper_specs["kf_window_size"]
        self.num_iterations = mapper_specs["num_iterations"]
        self.n_rays = mapper_specs["N_rays_each"]
        self.max_voxel_hit = mapper_specs["max_voxel_hit"]
        self.step_size = mapper_specs["step_size"] * self.voxel_size
        self.inflate_margin_ratio = mapper_specs["inflate_margin_ratio"]
        self.kf_selection_random_radio = mapper_specs["kf_selection_random_radio"]
        self.offset = mapper_specs["offset"]
        self.kf_selection_method = mapper_specs["kf_selection_method"]
        self.insert_method = mapper_specs["insert_method"]
        self.insert_ratio = mapper_specs["insert_ratio"]
        self.num_vertexes = mapper_specs["num_vertexes"]
        
        self.max_distance = data_specs["max_depth"]

        self.sdf_priors = torch.zeros(
            (self.num_vertexes, 1),
            requires_grad=True, dtype=torch.float32,
            device=torch.device("cuda"))
        self.vector_features = torch.zeros(
            (self.num_vertexes, 3),
            requires_grad=True, dtype=torch.float32,
            device=torch.device("cuda"))

        self.svo = torch.classes.svo.Octree()
        self.svo.init(256, int(self.num_vertexes), self.voxel_size, 0)  # Must be a multiple of 2
        self.optimize_params = [{'params': self.decoder.parameters(), 'lr': 1e-2},
                                {'params': self.sdf_priors, 'lr': 1e-2},
                                {'params': self.vector_features, 'lr': 1e-2}]

        self.optim = torch.optim.Adam(self.optimize_params)
        self.scaler = torch.amp.GradScaler('cuda')

        self.frame_poses = []
        self.bound = args.decoder_specs["bound"]


    def mapping_step(self, frame_id, tracked_frame, epoch):
        ######################
        self.idx = tracked_frame.stamp
        self.create_voxels(tracked_frame)

        self.sdf_priors = self.map_states["sdf_priors"]
        self.vector_features = self.map_states["vector_features"]
        if self.idx == 0:
            self.insert_kf(tracked_frame)
        self.do_mapping(tracked_frame=tracked_frame, epoch=epoch)
        # Fixed 50 frames to insert pictures(naive)
        if (tracked_frame.stamp - self.current_kf.stamp) > 50 and self.insert_method == "naive":
            self.insert_kf(tracked_frame)
        # The keyframe strategy we designed
        if self.insert_method == "intersection":
            insert_bool = self.voxel_field_insert_kf(self.insert_ratio)
            if insert_bool \
                    or (tracked_frame.stamp - self.current_kf.stamp) > 100:
                self.insert_kf(tracked_frame)

        # if self.save_ckpt_freq > 0 and (tracked_frame.stamp + 1) % self.save_ckpt_freq == 0:
        #     self.logger.log_ckpt(self, name=f"{tracked_frame.stamp:06d}.pth")

    def run(self, first_frame):
        self.idx = 0
        self.voxel_initialized = torch.zeros(self.num_vertexes).cuda().bool()
        self.vertex_initialized = torch.zeros(self.num_vertexes).cuda().bool()
        self.kf_unoptimized_voxels = None
        self.kf_optimized_voxels = None
        self.kf_all_voxels = None

        self.create_voxels(first_frame)
        self.sdf_priors = self.map_states["sdf_priors"]
        self.vector_features = self.map_states["vector_features"]
        self.insert_kf(first_frame)
        self.do_mapping(tracked_frame=first_frame)

        print("mapping started!")

        progress_bar = tqdm(range(self.start_frame, self.end_frame), position=0)
        progress_bar.set_description("mapping frame")
        epoch = 0

        for frame_id in progress_bar:
            data_in = self.data_stream[frame_id]
            tracked_frame = DepthFrame(*data_in[:-1], offset=self.offset, ref_pose=data_in[-1])

            if tracked_frame.ref_pose.isinf().any():
                continue
            self.mapping_step(frame_id, tracked_frame, epoch)
            epoch += 1

        print("******* mapping process died *******")
        print(f"********** post-processing {self.final_iter} steps **********")
        self.num_iterations = 1
        for iter in range(self.final_iter):
            self.do_mapping(tracked_frame=None)

        print("******* extracting final mesh *******")
        self.kf_graph = None
        self.logger.log_ckpt(self, name="final_ckpt.pth")
        mesh, mesh_priors, u, u_priors, u_hash_features = self.extract_mesh(res=self.mesh_res, map_states=self.map_states)
        
        self.logger.log_numpy_data(self.extract_voxels(map_states=self.map_states), "final_voxels")
        
        # Save SDF slice images for debugging
        save_dir = self.logger.misc_dir  # Save to misc directory

        self.save_sdf_slice(u, save_dir=save_dir, title="SDF")
        self.save_sdf_slice(u_priors, save_dir=save_dir, title="SDF Priors")
        self.save_sdf_slice(u_hash_features, save_dir=save_dir, title="Hash Features")
        # # Save u to file for subsequent analysis
        # np.save(os.path.join(save_dir, "sdf_u.npy"), u.cpu().numpy() if hasattr(u, "cpu") else u)
        # np.save(os.path.join(save_dir, "sdf_priors.npy"), u_priors.cpu().numpy() if hasattr(u_priors, "cpu") else u_priors)
        # np.save(os.path.join(save_dir, "hash_features.npy"), u_hash_features.cpu().numpy() if hasattr(u_hash_features, "cpu") else u_hash_features)
        # print(f"SDF voxel grid u saved to: {os.path.join(save_dir, 'sdf_u.npy')}")

        if self.args.evaluate:
            # 动态导入以避免循环导入问题
            from evaluate_mesh import evaluate_mesh
            from evaluate_sdf import evaluate_sdf
            evaluate_mesh(self.args)
            evaluate_sdf(self.args)
        
        print("******* mapping process died *******")

    def initfirst_onlymap(self):
        init_pose = self.data_stream.get_init_pose(self.start_frame)
        fid, depth, K, _ = self.data_stream[self.start_frame]
        first_frame = DepthFrame(fid, depth, K, offset=self.offset, ref_pose=init_pose)

        print("******* initializing first_frame: %d********" % first_frame.stamp)
        self.last_frame = first_frame
        self.start_frame += 1
        return first_frame

    def do_mapping(self, tracked_frame=None, epoch=0):
        self.decoder.train()
        optimize_targets = self.select_optimize_targets(tracked_frame)
        bundle_adjust_frames(
            optimize_targets,
            self.map_states,
            self.decoder,
            self.loss_criteria,
            self.voxel_size,
            self.n_rays,
            self.num_iterations,
            self.bound,
            optim=self.optim,
            scaler=self.scaler,
            frame_id=tracked_frame.stamp,
            use_adaptive_ending=self.use_adaptive_ending,
            writer=self.writer,
            h1=self.h1,
            epoch=epoch,
            sample_specs=self.sample_specs
        )

    def select_optimize_targets(self, tracked_frame=None):
        targets = []
        selection_method = self.kf_selection_method
        if len(self.kf_graph) <= self.kf_window_size:
            targets = self.kf_graph[:]
        elif selection_method == 'random':
            targets = random.sample(self.kf_graph, self.kf_window_size)
        elif selection_method == 'multiple_max_set_coverage':
            targets, self.kf_unoptimized_voxels, self.kf_optimized_voxels, self.kf_all_voxels = multiple_max_set_coverage(
                self.kf_graph,
                self.kf_seen_voxel_num,
                self.kf_unoptimized_voxels,
                self.kf_optimized_voxels,
                self.kf_window_size,
                self.kf_svo_idx,
                self.kf_all_voxels,
                self.num_vertexes)

        if tracked_frame is not None and (tracked_frame != self.current_kf):
            targets += [tracked_frame]
        return targets

    def insert_kf(self, frame):
        self.last_kf_observed = self.current_seen_voxel
        self.current_kf = frame
        self.last_kf_seen_voxel = self.seen_voxel
        self.kf_graph += [frame]
        self.kf_seen_voxel += [self.seen_voxel]
        self.kf_seen_voxel_num += [self.last_kf_observed]
        self.kf_svo_idx += [self.svo_idx]
        # If a new keyframe is inserted,
        # add the voxel in the newly inserted keyframe to the unoptimized voxel (remove the overlapping voxel)
        if self.kf_selection_method == 'multiple_max_set_coverage' and self.kf_unoptimized_voxels != None:
            self.kf_unoptimized_voxels[self.svo_idx.long() + 1] += True
            self.kf_unoptimized_voxels[0] = False

    def voxel_field_insert_kf(self, insert_ratio):
        # compute intersection
        voxel_no_repeat, cout = torch.unique(torch.cat([self.last_kf_seen_voxel,
                                                        self.seen_voxel], dim=0), return_counts=True, sorted=False,
                                             dim=0)
        N_i = voxel_no_repeat[cout > 1].shape[0]
        N_a = voxel_no_repeat.shape[0]
        ratio = N_i / N_a
        if ratio < insert_ratio:
            return True
        return False

    def get_margin_vox(self, inverse_indices, margin_mask, unique_vox_counts, unique_vox):
        unique_inv_id, counts_2 = torch.unique(inverse_indices[margin_mask], dim=0, return_counts=True)
        temp = torch.zeros(unique_vox.shape[0]).to(unique_vox.device)
        temp[unique_inv_id.long()] = counts_2.float()
        margin_vox = unique_vox[(temp == unique_vox_counts) * (unique_vox_counts > 10)]
        return margin_vox

    def updownsampling_voxel(self, points, indices, counts):
        summed_elements = torch.zeros(counts.shape[0], points.shape[-1]).cuda()
        summed_elements = torch.scatter_add(summed_elements, dim=0,
                                            index=indices.unsqueeze(1).repeat(1, points.shape[-1]), src=points)
        updownsample_points = summed_elements / counts.unsqueeze(-1).repeat(1, points.shape[-1])
        return updownsample_points

    def create_voxels(self, frame):
        points_raw = frame.get_points().cuda()

        pose = frame.get_ref_pose().cuda()

        points = points_raw @ pose[:3, :3].transpose(-1, -2) + pose[:3, 3]  # change to world frame (Rx)^T = x^T R^T

        voxels = torch.div(points, self.voxel_size, rounding_mode='floor')  # Divides each element

        voxels_raw, inverse_indices, counts = torch.unique(voxels, dim=0, return_inverse=True, return_counts=True)
        voxels_vaild = voxels_raw[counts > 3]
        self.voxels_vaild = voxels_vaild

        voxels_unique = torch.unique(voxels_vaild, dim=0)
        self.seen_voxel = voxels_unique
        self.current_seen_voxel = voxels_unique.shape[0]
        voxels_svo, children_svo, vertexes_svo, svo_mask, svo_idx = self.svo.insert(voxels_unique.cpu().int())
        svo_mask = svo_mask[:, 0].bool()
        voxels_svo = voxels_svo[svo_mask]
        children_svo = children_svo[svo_mask]
        vertexes_svo = vertexes_svo[svo_mask]
        self.octant_idx = svo_idx.nonzero().cuda()
        self.svo_idx = svo_idx
        self.update_grid(voxels_svo, children_svo, vertexes_svo, svo_idx)

    @torch.enable_grad()
    def update_grid(self, voxels, children, vertexes, svo_idx):

        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size
        children = torch.cat([children, voxels[:, -1:]], -1)

        centres = centres.cuda().float()
        children = children.cuda().int()

        map_states = {}
        map_states["voxels"] = voxels.cuda()
        map_states["voxel_vertex_idx"] = vertexes.cuda()
        map_states["voxel_center_xyz"] = centres.cuda()
        map_states["voxel_structure"] = children.cuda()
        map_states["sdf_priors"] = self.sdf_priors
        map_states["vector_features"] = self.vector_features
        map_states["svo_idx"] = svo_idx.cuda()

        self.map_states = map_states

    @torch.no_grad()
    def extract_mesh(self, res=80, map_states=None):
        from functions.render_helpers import find_voxel_idx
        from torch.nn import functional as F
        sdf_network = self.decoder
        sdf_network.eval()

        bound = self.bound
        x_min, x_max = bound[0]
        y_min, y_max = bound[1]
        z_min, z_max = bound[2]
        
        # Calculate range for each dimension
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        x_res = int(torch.round(torch.tensor(x_range * res)).item()) + 1
        y_res = int(torch.round(torch.tensor(y_range * res)).item()) + 1
        z_res = int(torch.round(torch.tensor(z_range * res)).item()) + 1

        x = torch.linspace(x_min, x_max, x_res)
        y = torch.linspace(y_min, y_max, y_res)
        z = torch.linspace(z_min, z_max, z_res)
        x, y, z = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)
        points = points.cuda()
        
        # Process point cloud in batches to avoid GPU memory insufficiency
        batch_size = 100000  # Adjust this value based on GPU memory
        total_points = points.shape[0]
        sdf_results = []
        sdf_priors = []
        hash_features = []
        hash_surface_sdf = []
        print(f"Processing {total_points} points in batches of {batch_size}...")
        
        with torch.no_grad():
            for i in range(0, total_points, batch_size):
                end_idx = min(i + batch_size, total_points)
                batch_points = points[i:end_idx]
                
                # Find voxel indices for current batch
                batch_voxel_idx = find_voxel_idx(batch_points, map_states)
                
                # Get features and predict SDF
                batch_sdf_priors = get_features(batch_points, batch_voxel_idx, map_states, self.voxel_size)
                batch_hash_features = sdf_network(batch_points)
                sdf_priors_features = batch_sdf_priors['sdf_priors'].squeeze(1)
                batch_sdf_pred = sdf_priors_features + batch_hash_features['sdf']
                
                # Immediately transfer to CPU to free GPU memory
                sdf_results.append(batch_sdf_pred.cpu())
                sdf_priors.append(sdf_priors_features.cpu())
                hash_features.append(batch_hash_features['sdf'].cpu())
                
                # Manually clean up temporary variables on GPU
                del batch_sdf_priors, batch_hash_features, batch_sdf_pred
                
                # Clear GPU cache every 10 batches
                if (i//batch_size + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    print(f"  Processed batch {i//batch_size + 1}/{(total_points + batch_size - 1)//batch_size}")
            
            # Concatenate results from all batches
            sdf_pred = torch.cat(sdf_results, dim=0)
            sdf_priors = torch.cat(sdf_priors, dim=0)
            hash_features = torch.cat(hash_features, dim=0)
        u = sdf_pred.reshape(x_res, y_res, z_res).cpu().numpy()
        u_priors = sdf_priors.reshape(x_res, y_res, z_res).cpu().numpy()
        u_hash_features = hash_features.reshape(x_res, y_res, z_res).cpu().numpy()

        import mcubes, trimesh
        vertices, triangles = mcubes.marching_cubes(u, 0)
        # Correct scaling logic: convert grid indices to actual coordinates
        vertices = vertices * [x_range/(x_res-1), y_range/(y_res-1), z_range/(z_res-1)] + [x_min, y_min, z_min] -self.offset
        vertices_priors, triangles_priors = mcubes.marching_cubes(u_priors, 0)
        vertices_priors = vertices_priors * [x_range/(x_res-1), y_range/(y_res-1), z_range/(z_res-1)] + [x_min, y_min, z_min] -self.offset
        print(f"==> vertices: {vertices.shape}, triangles: {triangles.shape}")
        print(f"==> vertices_priors: {vertices_priors.shape}, triangles_priors: {triangles_priors.shape}")
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh_priors = trimesh.Trimesh(vertices_priors, triangles_priors)
        mesh.invert()
        mesh_priors.invert()
        if self.args.save_mesh:
            out_path = os.path.join(self.logger.mesh_dir, f"mesh_{res}.obj")
            mesh.export(out_path)
            out_path_priors = os.path.join(self.logger.mesh_dir, f"mesh_{res}_priors.obj")
            mesh_priors.export(out_path_priors)
            print(f"==> mesh saved to {out_path}")
            print(f"==> mesh_priors saved to {out_path_priors}")
        return mesh, mesh_priors, u, u_priors, u_hash_features

    @torch.no_grad()
    def extract_voxels(self, map_states=None):
        vertexes = map_states["voxel_vertex_idx"]
        voxels = map_states["voxels"]

        index = vertexes.eq(-1).any(-1)
        voxels = voxels[~index.cpu(), :]
        voxels = (voxels[:, :3] + voxels[:, -1:] / 2) * \
                 self.voxel_size - self.offset
        return voxels

    @torch.no_grad()
    def save_sdf_slice(self, u, save_dir=None, title=None):
        import matplotlib.pyplot as plt
        import numpy as np

        if save_dir is None:
            save_dir = "./debug_slices"
        os.makedirs(save_dir, exist_ok=True)

        # Get shape and bounds of u
        x_res, y_res, z_res = u.shape
        bound = self.bound
        x_min, x_max = bound[0]
        y_min, y_max = bound[1]
        z_min, z_max = bound[2]

        # Convert to numpy
        if hasattr(u, 'cpu'):
            u_np = u.cpu().numpy()
        else:
            u_np = np.array(u)

        # Create coordinate grid
        x_coords = np.linspace(x_min, x_max, x_res)
        y_coords = np.linspace(y_min, y_max, y_res)
        z_coords = np.linspace(z_min, z_max, z_res)

        # Define slice configurations
        slice_configs = [
            {
                'name': 'x_slice',
                'axis': 0,  # x-axis direction
                'slice_idx': x_res // 2,  # middle slice
                'coord_names': ('Y', 'Z'),
                'title': f'{title} X-Slice (X={x_coords[x_res//2]:.2f})',
                'filename': f'{title}_slice_x.png',
                'coords': (y_coords, z_coords)
            },
            {
                'name': 'y_slice',
                'axis': 1,  # y-axis direction
                'slice_idx': y_res // 2,  # middle slice
                'coord_names': ('X', 'Z'),
                'title': f'{title} Y-Slice (Y={y_coords[y_res//2]:.2f})',
                'filename': f'{title}_slice_y.png',
                'coords': (x_coords, z_coords)
            },
            {
                'name': 'z_slice',
                'axis': 2,  # z-axis direction
                'slice_idx': z_res // 2,  # middle slice
                'coord_names': ('X', 'Y'),
                'title': f'{title} Z-Slice (Z={z_coords[z_res//2]:.2f})',
                'filename': f'{title}_slice_z.png',
                'coords': (x_coords, y_coords)
            }
        ]

        # Generate slices for each direction
        for config in slice_configs:
            axis = config['axis']
            slice_idx = config['slice_idx']
            coord_names = config['coord_names']
            title = config['title']
            filename = config['filename']
            coord1_vals, coord2_vals = config['coords']
            
            # Extract slice data
            if axis == 0:  # x slice, fixed x, varying y,z
                sdf_slice = u_np[slice_idx, :, :]
            elif axis == 1:  # y slice, fixed y, varying x,z
                sdf_slice = u_np[:, slice_idx, :]
            else:  # z slice, fixed z, varying x,y
                sdf_slice = u_np[:, :, slice_idx]
            
            # Create coordinate grid
            coord1_grid, coord2_grid = np.meshgrid(coord1_vals, coord2_vals, indexing='ij')
            
            # Draw image
            plt.figure(figsize=(12, 10))
            
            # Use imshow to display SDF slice
            im = plt.imshow(
                sdf_slice.T,  # transpose to display direction correctly
                extent=[coord1_vals[0], coord1_vals[-1], coord2_vals[0], coord2_vals[-1]],
                origin='lower',
                cmap='RdBu_r',  # red for positive values, blue for negative values
                aspect='equal'
            )
            
            # Add contour lines
            levels = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
            cs = plt.contour(coord1_grid, coord2_grid, sdf_slice, 
                           levels=levels, colors='black', alpha=0.3, linewidths=0.5)
            
            # Highlight zero contour line (surface)
            zero_contour = plt.contour(coord1_grid, coord2_grid, sdf_slice, 
                                     levels=[0], colors='red', linewidths=2)
            
            plt.colorbar(im, label='SDF Value', shrink=0.8)
            plt.xlabel(f'{coord_names[0]} (m)', fontsize=12)
            plt.ylabel(f'{coord_names[1]} (m)', fontsize=12)
            plt.title(title, fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add data range information to subtitle
            plt.suptitle(
                f'SDF Range: [{sdf_slice.min():.3f}, {sdf_slice.max():.3f}] | '
                f'Grid: {sdf_slice.shape[0]}×{sdf_slice.shape[1]}',
                fontsize=10, y=0.95
            )

            # Save image
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"SDF {config['name']} saved to: {save_path}")
            print(f"  Slice index: {slice_idx} (axis {axis})")
            print(f"  SDF range: [{sdf_slice.min():.3f}, {sdf_slice.max():.3f}]")
            print(f"  Grid size: {sdf_slice.shape[0]}*{sdf_slice.shape[1]}")
        
    


        