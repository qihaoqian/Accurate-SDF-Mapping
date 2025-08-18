import random

# import message_filters
import numpy as np
# import rospy
import torch
# from cv_bridge import CvBridge
# from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
# from sensor_msgs.msg import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from criterion import Criterion
from frame import RGBDFrame
from functions.initialize_sdf import initemb_sdf
from functions.render_helpers import bundle_adjust_frames
from functions.render_helpers import fill_in, render_rays
from loggers import BasicLogger
from utils.import_util import get_decoder, get_property
from utils.keyframe_util import multiple_max_set_coverage
from utils.mesh_util import MeshExtractor

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
        self.run_ros = args.run_ros

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
        if self.run_ros:
            ros_args = args.ros_args
            self.intrinsic = np.eye(3)
            self.intrinsic[0, 0] = ros_args["intrinsic"][0]
            self.intrinsic[1, 1] = ros_args["intrinsic"][1]
            self.intrinsic[0, 2] = ros_args["intrinsic"][2]
            self.intrinsic[1, 2] = ros_args["intrinsic"][3]
            print("intrinsic: ", self.intrinsic)
            self.color_topic = ros_args["color_topic"]
            self.depth_topic = ros_args["depth_topic"]
            self.pose_topic = ros_args["pose_topic"]

        self.use_gt = data_specs["use_gt"]
        self.max_distance = data_specs["max_depth"]

        self.render_freq = debug_args["render_freq"]
        self.render_res = debug_args["render_res"]
        self.mesh_freq = debug_args["mesh_freq"]
        self.save_ckpt_freq = debug_args["save_ckpt_freq"]

        self.sdf_truncation = args.criteria["sdf_truncation"]

        self.mesher = MeshExtractor(args)

        self.sdf_priors = torch.zeros(
            (self.num_vertexes, 1),
            requires_grad=True, dtype=torch.float32,
            device=torch.device("cuda"))

        self.svo = torch.classes.svo.Octree()
        self.svo.init(256, int(self.num_vertexes), self.voxel_size)  # Must be a multiple of 2
        self.optimize_params = [{'params': self.decoder.parameters(), 'lr': 1e-2},
                                {'params': self.sdf_priors, 'lr': 1e-2}]

        self.optim = torch.optim.Adam(self.optimize_params)
        self.scaler = torch.cuda.amp.GradScaler()

        self.frame_poses = []

    def callback(self, color, depth, pose_vins):
        update_pose = False
        bridge = CvBridge()
        color_image = bridge.imgmsg_to_cv2(color, desired_encoding="passthrough")
        depth_image = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        q = pose_vins.pose.pose.orientation
        dcm = Rotation.from_quat(np.array([q.x, q.y, q.z, q.w])).as_matrix()
        trans = pose_vins.pose.pose.position
        trans = np.array([trans.x, trans.y, trans.z])
        pose = np.eye(4)
        pose[:3, :3] = dcm
        pose[:3, 3] = trans
        depth_image = depth_image * 0.001
        if self.max_distance > 0:
            depth_image[(depth_image > self.max_distance)] = 0
        color_image = color_image / 256

        tracked_frame = RGBDFrame(self.idx, color_image, depth_image, K=self.intrinsic, offset=self.offset,
                                  ref_pose=pose)
        self.mapping_step(self.idx, tracked_frame, update_pose)
        self.idx += 1
        print("idx: ", self.idx)

    def mapping_step(self, frame_id, tracked_frame, update_pose):
        ######################
        self.idx = tracked_frame.stamp
        self.create_voxels(tracked_frame)

        self.map_states, self.voxel_initialized, self.vertex_initialized = initemb_sdf(tracked_frame,
                                                                                       self.map_states,
                                                                                       self.sdf_truncation,
                                                                                       voxel_size=self.voxel_size,
                                                                                       voxel_initialized=self.voxel_initialized,
                                                                                       octant_idx=self.octant_idx,
                                                                                       vertex_initialized=self.vertex_initialized,
                                                                                       use_gt=self.use_gt)
        self.sdf_priors = self.map_states["sdf_priors"]
        if self.idx == 0:
            self.insert_kf(tracked_frame)
        self.do_mapping(tracked_frame=tracked_frame, update_pose=update_pose)
        # Fixed 50 frames to insert pictures(naive)
        if (tracked_frame.stamp - self.current_kf.stamp) > 50 and self.insert_method == "naive":
            self.insert_kf(tracked_frame)
        # The keyframe strategy we designed
        if self.insert_method == "intersection":
            insert_bool = self.voxel_field_insert_kf(self.insert_ratio)
            if insert_bool \
                    or (tracked_frame.stamp - self.current_kf.stamp) > 100:
                self.insert_kf(tracked_frame)

        self.tracked_pose = tracked_frame.get_ref_pose().detach() @ tracked_frame.get_d_pose().detach()
        ref_pose = self.current_kf.get_ref_pose().detach() @ self.current_kf.get_d_pose().detach()
        rel_pose = torch.linalg.inv(ref_pose) @ self.tracked_pose
        self.frame_poses += [(len(self.kf_graph) - 1, rel_pose.cpu())]

        if self.mesh_freq > 0 and (tracked_frame.stamp + 1) % self.mesh_freq == 0:
            self.logger.log_mesh(self.extract_mesh(
                res=self.mesh_res, clean_mesh=False, map_states=self.map_states),
                name=f"mesh_{tracked_frame.stamp:06d}.ply")

        if self.save_data_freq > 0 and (tracked_frame.stamp + 1) % self.save_data_freq == 0:
            self.save_debug_data(tracked_frame)

        if self.render_freq > 0 and (frame_id + 1) % self.render_freq == 0:
            self.render_debug_images(tracked_frame)

        if self.save_ckpt_freq > 0 and (tracked_frame.stamp + 1) % self.save_ckpt_freq == 0:
            self.logger.log_ckpt(self, name=f"{tracked_frame.stamp:06d}.pth")

    def run(self, first_frame, update_pose):
        self.idx = 0
        self.voxel_initialized = torch.zeros(self.num_vertexes).cuda().bool()
        self.vertex_initialized = torch.zeros(self.num_vertexes).cuda().bool()
        self.kf_unoptimized_voxels = None
        self.kf_optimized_voxels = None
        self.kf_all_voxels = None
        if self.run_ros:
            rospy.init_node('listener', anonymous=True)
            # realsense
            color_sub = message_filters.Subscriber(self.color_topic, Image)
            depth_sub = message_filters.Subscriber(self.depth_topic, Image)
            pose_sub = message_filters.Subscriber(self.pose_topic, Odometry)

            ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, pose_sub], 2, 1 / 10,
                                                             allow_headerless=False)
            print(" ========== MAPPING START ===========")
            ts.registerCallback(self.callback)
            rospy.spin()
        else:
            if self.mesher is not None:
                self.mesher.rays_d = first_frame.get_rays()
            self.create_voxels(first_frame)
            self.map_states, self.voxel_initialized, self.vertex_initialized = initemb_sdf(first_frame,
                                                                                           self.map_states,
                                                                                           self.sdf_truncation,
                                                                                           voxel_size=self.voxel_size,
                                                                                           voxel_initialized=self.voxel_initialized,
                                                                                           octant_idx=self.octant_idx,
                                                                                           vertex_initialized=self.vertex_initialized,
                                                                                           use_gt=self.use_gt)
            self.sdf_priors = self.map_states["sdf_priors"]
            self.insert_kf(first_frame)
            self.do_mapping(tracked_frame=first_frame, update_pose=update_pose)

            self.tracked_pose = first_frame.get_ref_pose().detach() @ first_frame.get_d_pose().detach()
            ref_pose = self.current_kf.get_ref_pose().detach() @ self.current_kf.get_d_pose().detach()
            rel_pose = torch.linalg.inv(ref_pose) @ self.tracked_pose
            self.frame_poses += [(len(self.kf_graph) - 1, rel_pose.cpu())]
            self.render_debug_images(first_frame)

            print("mapping started!")

            progress_bar = tqdm(range(self.start_frame, self.end_frame), position=0)
            progress_bar.set_description("mapping frame")
            for frame_id in progress_bar:
                data_in = self.data_stream[frame_id]
                if self.use_gt:
                    tracked_frame = RGBDFrame(*data_in[:-1], offset=self.offset, ref_pose=data_in[-1])
                else:
                    tracked_frame = RGBDFrame(*data_in[:-1], offset=self.offset, ref_pose=self.tracked_pose.clone())
                if update_pose is False:
                    tracked_frame.d_pose.requires_grad_(False)
                if tracked_frame.ref_pose.isinf().any():
                    continue
                self.mapping_step(frame_id, tracked_frame, update_pose)

        print("******* mapping process died *******")
        print(f"********** post-processing {self.final_iter} steps **********")
        self.num_iterations = 1
        for iter in range(self.final_iter):
            self.do_mapping(tracked_frame=None, update_pose=False)

        print("******* extracting final mesh *******")
        pose = self.get_updated_poses()
        self.kf_graph = None
        mesh = self.extract_mesh(res=self.mesh_res, clean_mesh=False, map_states=self.map_states, return_sdf_priors=False)
        mesh_sdf_priors = self.extract_mesh(res=self.mesh_res, clean_mesh=False, map_states=self.map_states, return_sdf_priors=True)
        self.logger.log_ckpt(self, name="final_ckpt.pth")
        pose = np.asarray(pose)
        pose[:, 0:3, 3] -= self.offset
        self.logger.log_numpy_data(pose, "frame_poses")
        self.logger.log_mesh(mesh)
        self.logger.log_mesh(mesh_sdf_priors, "mesh_sdf_priors.ply")
        self.logger.log_numpy_data(self.extract_voxels(map_states=self.map_states), "final_voxels")
        
        # 保存SDF切片图用于debug
        save_dir = self.logger.misc_dir  # 保存到misc目录下
        self.save_sdf_slices(self.map_states, self.decoder, self.voxel_size, self.offset, save_dir=save_dir)
        
        # 保存三维体素分布图用于debug
        self.save_voxel_distribution_3d(self.map_states, save_dir=save_dir)
        
        print("******* mapping process died *******")

    def initfirst_onlymap(self):
        init_pose = self.data_stream.get_init_pose(self.start_frame)
        fid, rgb, depth, K, _ = self.data_stream[self.start_frame]
        first_frame = RGBDFrame(fid, rgb, depth, K, offset=self.offset, ref_pose=init_pose)
        first_frame.d_pose.requires_grad_(False)

        print("******* initializing first_frame: %d********" % first_frame.stamp)
        self.last_frame = first_frame
        self.start_frame += 1
        return first_frame

    def do_mapping(self, tracked_frame=None, update_pose=True):
        self.decoder.train()
        optimize_targets = self.select_optimize_targets(tracked_frame)
        bundle_adjust_frames(
            optimize_targets,
            self.map_states,
            self.decoder,
            self.loss_criteria,
            self.voxel_size,
            self.step_size,
            self.n_rays,
            self.num_iterations,
            self.sdf_truncation,
            self.max_voxel_hit,
            self.max_distance,
            update_pose=update_pose,
            optim=self.optim,
            scaler=self.scaler,
            frame_id=tracked_frame.stamp,
            use_adaptive_ending=self.use_adaptive_ending
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
        if self.use_gt:
            pose = frame.get_ref_pose().cuda()
        else:
            pose = frame.get_ref_pose().cuda() @ frame.get_d_pose().cuda()
        points = points_raw @ pose[:3, :3].transpose(-1, -2) + pose[:3, 3]  # change to world frame (Rx)^T = x^T R^T

        voxels = torch.div(points, self.voxel_size, rounding_mode='floor')  # Divides each element

        inflate_margin_ratio = self.inflate_margin_ratio

        voxels_raw, inverse_indices, counts = torch.unique(voxels, dim=0, return_inverse=True, return_counts=True)

        voxels_vaild = voxels_raw[counts > 10]
        self.voxels_vaild = voxels_vaild
        offsets = torch.LongTensor([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]).to(
            voxels.device)

        updownsampling_points = self.updownsampling_voxel(points, inverse_indices, counts)
        for offset in offsets:
            offset_axis = offset.nonzero().item()
            if offset[offset_axis] > 0:
                margin_mask = updownsampling_points[:, offset_axis] % self.voxel_size > (
                        1 - inflate_margin_ratio) * self.voxel_size
            else:
                margin_mask = updownsampling_points[:,
                              offset_axis] % self.voxel_size < inflate_margin_ratio * self.voxel_size
            margin_vox = voxels_raw[margin_mask * (counts > 10)]
            voxels_vaild = torch.cat((voxels_vaild, torch.clip(margin_vox + offset, min=0)), dim=0)

        voxels_unique = torch.unique(voxels_vaild, dim=0)
        self.seen_voxel = voxels_unique
        self.current_seen_voxel = voxels_unique.shape[0]
        voxels_svo, children_svo, vertexes_svo, svo_mask, svo_idx = self.svo.insert(voxels_unique.cpu().int())
        svo_mask = svo_mask[:, 0].bool()
        voxels_svo = voxels_svo[svo_mask]
        children_svo = children_svo[svo_mask]
        vertexes_svo = vertexes_svo[svo_mask]
        import os
        save_dir = self.logger.misc_dir
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(os.path.join(save_dir, "vertexes_svo.txt"), vertexes_svo.cpu().numpy(), fmt="%.6f", header="x y z")
        # 将体素信息保存为txt文件，方便查看
        np.savetxt(os.path.join(save_dir, "voxels.txt"), voxels_svo.cpu().numpy(), fmt="%.6f", header="x y z size") 
        self.octant_idx = svo_mask.nonzero().cuda()
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
        map_states["svo_idx"] = svo_idx.cuda()

        self.map_states = map_states

    @torch.no_grad()
    def get_updated_poses(self):
        frame_poses = []
        for i in range(len(self.frame_poses)):
            ref_frame_ind, rel_pose = self.frame_poses[i]
            ref_frame = self.kf_graph[ref_frame_ind]
            ref_pose = ref_frame.get_ref_pose().detach().cpu() @ ref_frame.get_d_pose().detach().cpu()
            pose = ref_pose @ rel_pose
            frame_poses += [pose.detach().cpu().numpy()]
        return frame_poses

    """
    Get the mesh at the position of the voxel
    Args:
        res: The number of points collected in each dimension in each voxel.
        clean_mesh: Whether to keep only the mesh of the current frame.
        map_states: state parameters of the map.
    Returns:
        mesh
    """

    @torch.no_grad()
    def extract_mesh(self, res=8, clean_mesh=False, map_states=None, return_sdf_priors=True):
        sdf_network = self.decoder
        sdf_network.eval()
        vertexes = map_states["voxel_vertex_idx"]
        voxels = map_states["voxels"]

        index = vertexes.eq(-1).any(-1)  # remove no smallest voxel
        voxels = voxels[~index.cpu(), :]
        vertexes = vertexes[~index.cpu(), :]
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size

        encoder_states = {}
        encoder_states["voxel_vertex_idx"] = vertexes.cuda()
        encoder_states["voxel_center_xyz"] = centres.cuda()
        encoder_states["sdf_priors"] = self.sdf_priors

        mesh = self.mesher.create_mesh(
            self.decoder, encoder_states, self.voxel_size, voxels,
            frame_poses=None, depth_maps=None,
            clean_mseh=clean_mesh, require_color=True, offset=-self.offset, res=res, return_sdf_priors=return_sdf_priors)
        
        return mesh

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
    def save_debug_data(self, tracked_frame):
        """
        save per-frame voxel, mesh and pose
        """
        if self.use_gt:
            pose = tracked_frame.get_ref_pose().detach().cpu().numpy()
        else:
            pose = tracked_frame.get_ref_pose().detach().cpu().numpy() @ tracked_frame.get_d_pose().detach().cpu().numpy()
        pose[:3, 3] -= self.offset
        frame_poses = self.get_updated_poses()
        mesh = self.extract_mesh(res=8, clean_mesh=True)
        voxels = self.extract_voxels(map_states=self.map_states).detach().cpu().numpy()
        if self.use_gt:
            kf_poses = [p.get_ref_pose().detach().cpu().numpy()
                        for p in self.kf_graph]
        else:
            kf_poses = [p.get_ref_pose().detach().cpu().numpy() @ p.get_d_pose().detach().cpu().numpy()
                        for p in self.kf_graph]

        for f in frame_poses:
            f[:3, 3] -= self.offset
        for kf in kf_poses:
            kf[:3, 3] -= self.offset

        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        color = np.asarray(mesh.vertex_colors)

        self.logger.log_debug_data({
            "pose": pose,
            "updated_poses": frame_poses,
            "mesh": {"verts": verts, "faces": faces, "color": color},
            "voxels": voxels,
            "voxel_size": self.voxel_size,
            "keyframes": kf_poses,
            "is_kf": (tracked_frame == self.current_kf)
        }, tracked_frame.stamp)

    """"
    Used to render a complete picture
    """

    @torch.no_grad()
    def render_debug_images(self, current_frame, batch_size=200000):
        rgb = current_frame.rgb
        depth = current_frame.depth
        rotation = (current_frame.get_ref_pose().cuda() @ current_frame.get_d_pose().cuda())[:3, :3]
        ind = current_frame.stamp
        w, h = self.render_res
        final_outputs = dict()

        decoder = self.decoder.cuda()
        map_states = {}
        for k, v in self.map_states.items():
            map_states[k] = v.cuda()

        rays_d = current_frame.get_rays(w, h).cuda()
        rays_d = rays_d @ rotation.transpose(-1, -2)

        rays_o = (current_frame.get_ref_pose().cuda() @ current_frame.get_d_pose().cuda())[:3, 3]
        rays_o = rays_o.unsqueeze(0).expand_as(rays_d)

        rays_o = rays_o.reshape(1, -1, 3).contiguous()
        rays_d = rays_d.reshape(1, -1, 3)
        torch.cuda.empty_cache()

        batch_size = batch_size
        ray_mask_list = []
        color_list = []
        depth_list = []
        # To prevent memory overflow, batch_size can be given according to the video memory
        for batch_iter in range(0, rays_o.shape[1], batch_size):
            final_outputs = render_rays(
                rays_o[:, batch_iter:batch_iter + batch_size, :].clone(),
                rays_d[:, batch_iter:batch_iter + batch_size, :].clone(),
                map_states,
                decoder,
                self.step_size,
                self.voxel_size,
                self.sdf_truncation,
                self.max_voxel_hit,
                self.max_distance,
                chunk_size=500000000,
                return_raw=True,
                eval=True
            )
            if final_outputs["color"] == None:
                ray_mask_list.append(final_outputs["ray_mask"])
                continue
            ray_mask_list.append(final_outputs["ray_mask"])
            depth_list.append(final_outputs["depth"])
            color_list.append(final_outputs["color"])

        ray_mask_input = torch.cat(ray_mask_list, dim=1)

        if len(depth_list) == 0:
            return None, None, None

        depth_input = torch.cat(depth_list)
        color_input = torch.cat(color_list, dim=0)

        rdepth = fill_in((h, w, 1),
                         ray_mask_input.view(h, w),
                         depth_input, 0)
        rcolor = fill_in((h, w, 3),
                         ray_mask_input.view(h, w),
                         color_input, 0)
        if self.logger.for_eva:
            ssim, psnr, depth_L1 = self.logger.log_images(ind, rgb, depth, rcolor, rdepth)
            return ssim, psnr, depth_L1
        else:
            self.logger.log_images(ind, rgb, depth, rcolor, rdepth)

    @torch.no_grad()
    def save_sdf_slices(self, map_states, decoder, voxel_size, offset, grid_size=256, save_dir=None):
        """
        保存SDF在三个方向的切片图用于debug
        Args:
            map_states: 地图状态信息
            decoder: 训练好的SDF解码器
            voxel_size: 体素大小
            offset: 坐标偏移
            grid_size: 切片网格大小，默认256x256
            save_dir: 保存目录
        """
        print("******* Saving SDF slices for debug *******")
        
        # 如果没有指定保存目录，使用默认目录
        if save_dir is None:
            save_dir = "./debug_slices"
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        device = torch.device("cuda")
        decoder.eval()
        
        # 获取有效体素信息
        voxels = map_states["voxels"].cpu().numpy()  # [N, 4] 前三维是坐标，最后一维是size
        vertexes = map_states["voxel_vertex_idx"].cpu().numpy()
        
        # 过滤掉无效体素（与extract_mesh中的逻辑保持一致）
        index = (vertexes == -1).any(-1)  # remove no smallest voxel
        valid_voxels = voxels[~index, :]
        voxels_xyz = valid_voxels[:, :3]  # 只取坐标部分
        
        if len(voxels_xyz) == 0:
            print("No valid voxels found!")
            return
        
        # 计算体素边界
        min_xyz = voxels_xyz.min(axis=0)
        max_xyz = voxels_xyz.max(axis=0) + 1
        
        print(f"Voxel bounds: min={min_xyz}, max={max_xyz}")
        print(f"Total valid voxels: {len(voxels_xyz)}")
        print(f"Sample voxel coordinates: {voxels_xyz[:5] if len(voxels_xyz) > 0 else 'None'}")
        
        # 创建体素集合用于快速查询（使用体素的整数坐标）
        voxel_set = set(map(tuple, voxels_xyz.astype(int)))
        
        # 三个方向的切片
        for axis, name in zip([0, 1, 2], ["x", "y", "z"]):
            print(f"Processing {name} axis slice...")
            
            # 取中间切片
            slice_idx = int((min_xyz[axis] + max_xyz[axis]) // 2)
            
            # 创建网格点
            if axis == 0:  # x方向切片，在yz平面
                y_coords = np.linspace(min_xyz[1], max_xyz[1] - 1, grid_size)
                z_coords = np.linspace(min_xyz[2], max_xyz[2] - 1, grid_size)
                Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')
                grid_coords = np.stack([np.full_like(Y, slice_idx), Y, Z], axis=-1)
            elif axis == 1:  # y方向切片，在xz平面
                x_coords = np.linspace(min_xyz[0], max_xyz[0] - 1, grid_size)
                z_coords = np.linspace(min_xyz[2], max_xyz[2] - 1, grid_size)
                X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')
                grid_coords = np.stack([X, np.full_like(X, slice_idx), Z], axis=-1)
            else:  # z方向切片，在xy平面
                x_coords = np.linspace(min_xyz[0], max_xyz[0] - 1, grid_size)
                y_coords = np.linspace(min_xyz[1], max_xyz[1] - 1, grid_size)
                X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
                grid_coords = np.stack([X, Y, np.full_like(X, slice_idx)], axis=-1)
            
            # 初始化SDF图
            sdf_map = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
            valid_mask = np.zeros((grid_size, grid_size), dtype=bool)
            
            # 批量处理以提高效率
            batch_size = 1000
            valid_points = []
            valid_indices = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # 网格坐标（连续值）
                    grid_point = grid_coords[i, j]
                    # 对应的体素坐标（整数）
                    voxel_coord = np.floor(grid_point).astype(int)
                    
                    # 检查是否在有效体素内
                    if tuple(voxel_coord) in voxel_set:
                        # 转为世界坐标：体素坐标转世界坐标
                        world_coord = grid_point * voxel_size - offset
                        valid_points.append(world_coord)
                        valid_indices.append((i, j))
                        valid_mask[i, j] = True
            
            if len(valid_points) == 0:
                print(f"No valid points found for {name} axis slice")
                continue
                
            print(f"Found {len(valid_points)} valid points for {name} axis")
            
            # 批量推理SDF
            valid_points = np.array(valid_points)
            for batch_start in range(0, len(valid_points), batch_size):
                batch_end = min(batch_start + batch_size, len(valid_points))
                batch_points = torch.tensor(valid_points[batch_start:batch_end], 
                                          dtype=torch.float32, device=device)
                
                try:
                    # 直接调用解码器的get_sdf方法获取SDF值
                    sdf_values = decoder.get_sdf(batch_points)
                    
                    # 提取SDF值
                    if isinstance(sdf_values, torch.Tensor):
                        if sdf_values.dim() > 1:
                            sdf_values = sdf_values[:, -1]  # 取最后一列
                        sdf_values = sdf_values.detach().cpu().numpy()
                    else:
                        sdf_values = np.array(sdf_values)
                        
                    # 填充SDF图
                    for k, sdf_val in enumerate(sdf_values):
                        idx = batch_start + k
                        if idx < len(valid_indices):
                            i, j = valid_indices[idx]
                            sdf_map[i, j] = float(sdf_val)
                            
                except Exception as e:
                    print(f"Error during SDF inference: {e}")
                    # 如果网络推理失败，使用默认值
                    for k in range(batch_end - batch_start):
                        idx = batch_start + k
                        if idx < len(valid_indices):
                            i, j = valid_indices[idx]
                            sdf_map[i, j] = 0.0  # 默认值
            
            # 可视化并保存
            plt.figure(figsize=(8, 8))
            
            # 对不在voxel的点保留空白（透明显示）
            masked_sdf = np.ma.masked_where(~valid_mask, sdf_map)
            
            # 设置背景为白色，无效区域为透明
            plt.gca().set_facecolor('white')
            
            im = plt.imshow(masked_sdf, cmap='RdBu_r', origin='lower', 
                           vmin=-0.1, vmax=0.1, alpha=1.0)  # SDF通常在[-0.1, 0.1]范围
            plt.colorbar(im, label='SDF Value')
            plt.title(f'SDF Slice - {name.upper()} axis (slice={slice_idx})')
            plt.xlabel(f'{"Y" if axis == 0 else "X" if axis == 1 else "X"} coordinate')
            plt.ylabel(f'{"Z" if axis != 2 else "Y"} coordinate')
            
            # 保存图像
            save_path = os.path.join(save_dir, f'sdf_slice_{name}_axis.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved SDF slice for {name} axis to {save_path}")
            
            # 同时保存数值数据
            np.save(os.path.join(save_dir, f'sdf_slice_{name}_axis.npy'), sdf_map)
        
        print("******* SDF slices saved successfully *******")

    @torch.no_grad()
    def save_voxel_distribution_3d(self, map_states, save_dir=None):
        """
        保存三维体素分布图用于debug
        Args:
            map_states: 地图状态信息
            save_dir: 保存目录
        """
        print("******* Saving 3D voxel distribution *******")
        
        # 如果没有指定保存目录，使用默认目录
        if save_dir is None:
            save_dir = "./debug_slices"
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取有效体素信息
        voxels = map_states["voxels"].cpu().numpy()  # [N, 4] 前三维是坐标，最后一维是size
        vertexes = map_states["voxel_vertex_idx"].cpu().numpy()
        
        # 过滤掉无效体素
        index = (vertexes == -1).any(-1)  # remove no smallest voxel
        valid_voxels = voxels[~index, :]
        voxels_xyz = valid_voxels[:, :3]  # 只取坐标部分
        voxel_sizes = valid_voxels[:, 3]  # 体素大小
        
        if len(voxels_xyz) == 0:
            print("No valid voxels found for 3D distribution!")
            return
        
        print(f"Plotting {len(voxels_xyz)} valid voxels in 3D")
        
        # 计算体素中心坐标（世界坐标系）
        voxel_centers = (voxels_xyz + 0.5) * self.voxel_size - self.offset
        
        # 创建3D散点图
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 根据体素大小设置颜色和点大小
        colors = voxel_sizes
        sizes = np.clip(voxel_sizes * 20, 10, 100)  # 调整点的大小范围
        
        scatter = ax.scatter(voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2], 
                           c=colors, s=sizes, alpha=0.6, cmap='viridis')
        
        # 设置标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Voxel Distribution ({len(voxels_xyz)} voxels)')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='Voxel Size', shrink=0.5)
        
        # 设置相等的坐标轴比例
        max_range = np.array([voxel_centers[:, 0].max() - voxel_centers[:, 0].min(),
                             voxel_centers[:, 1].max() - voxel_centers[:, 1].min(),
                             voxel_centers[:, 2].max() - voxel_centers[:, 2].min()]).max() / 2.0
        
        mid_x = (voxel_centers[:, 0].max() + voxel_centers[:, 0].min()) * 0.5
        mid_y = (voxel_centers[:, 1].max() + voxel_centers[:, 1].min()) * 0.5
        mid_z = (voxel_centers[:, 2].max() + voxel_centers[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 保存3D视图
        save_path = os.path.join(save_dir, 'voxel_distribution_3d.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"3D voxel distribution saved to {save_path}")
        
        # 同时创建投影图（从三个方向看的2D投影）
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY平面投影（俯视图）
        axes[0].scatter(voxel_centers[:, 0], voxel_centers[:, 1], 
                       c=voxel_centers[:, 2], s=20, alpha=0.7, cmap='plasma')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title('XY Projection (Top View)')
        axes[0].set_aspect('equal')
        
        # XZ平面投影（侧视图）
        axes[1].scatter(voxel_centers[:, 0], voxel_centers[:, 2], 
                       c=voxel_centers[:, 1], s=20, alpha=0.7, cmap='plasma')
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Z (m)')
        axes[1].set_title('XZ Projection (Side View)')
        axes[1].set_aspect('equal')
        
        # YZ平面投影（正视图）
        axes[2].scatter(voxel_centers[:, 1], voxel_centers[:, 2], 
                       c=voxel_centers[:, 0], s=20, alpha=0.7, cmap='plasma')
        axes[2].set_xlabel('Y (m)')
        axes[2].set_ylabel('Z (m)')
        axes[2].set_title('YZ Projection (Front View)')
        axes[2].set_aspect('equal')
        
        plt.tight_layout()
        
        # 保存投影图
        projection_path = os.path.join(save_dir, 'voxel_projections_2d.png')
        plt.savefig(projection_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"2D voxel projections saved to {projection_path}")
        
        # 保存体素数据为numpy文件
        voxel_data = {
            'voxel_centers': voxel_centers,
            'voxel_sizes': voxel_sizes,
            'voxel_coords': voxels_xyz,
            'total_voxels': len(voxels_xyz)
        }
        np.save(os.path.join(save_dir, 'voxel_distribution_data.npy'), voxel_data)
        
        print("******* 3D voxel distribution saved successfully *******")
