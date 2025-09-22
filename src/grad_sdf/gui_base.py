import asyncio
import os
import threading
import time
from dataclasses import dataclass
from multiprocessing import Queue
from typing import Optional

import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

from grad_sdf import MarchingCubes, np, o3d, o3d_gui, o3d_rendering
from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class GuiBaseConfig(ConfigABC):
    panel_split_ratio: float = 0.7

    scan_point_size: int = 2
    scan_point_color: list = [0.9, 0.9, 0.9]

    sdf_point_size: int = 2
    sdf_color_map: str = "jet"  # jet, bwr, viridis

    traj_line_width: int = 2
    traj_color: list = [1.0, 0.0, 0.0]

    camera_line_width: int = 3
    camera_size: float = 1.0
    camera_color_current: list = [0.0, 1.0, 0.0]
    camera_color_key_frame: list = [0.0, 0.0, 1.0]
    camera_color_selected_key_frame: list = [1.0, 0.5, 0.0]

    octree_line_width: int = 1
    octree_color: list = [0.8, 0.8, 0.8]

    mesh_update_freq: int = 10
    mesh_resolution: int = 100
    mesh_height_color_map: str = "jet"

    sdf_slice_update_freq: int = 10
    sdf_slice_resolution: int = 100

    experiment_name: str = "grad_sdf"
    scene_bound_min: list = [-5.0, -5.0, -5.0]
    scene_bound_max: list = [5.0, 5.0, 5.0]
    gt_mesh_path: Optional[str] = None


@dataclass
class GuiControlPacket:
    """
    This class is used to send control signals from GUI to the mapping process.
    """

    flag_mapping_run: bool = True
    flag_gui_closed: bool = False

    sdf_slice_frequency: int = -1
    sdf_slice_axis: int = 2
    sdf_slice_position: float = 0.0
    sdf_slice_resolution: int = -1

    sdf_grid_frequency: int = -1
    sdf_grid_resolution: int = -1

    save_model_to_path: Optional[str] = None  # if not None, save the model to this path


@dataclass
class GuiDataPacket:
    """
    This class is used to receive data from the mapping process to GUI.
    """

    flag_exit: bool = False  # whether the mapping process is going to exit
    mapping_end: bool = False  # whether the mapping has ended

    frame_idx: int = -1  # current frame index
    frame_pose: Optional[np.ndarray] = None  # used to build trajectory
    scan_points: Optional[np.ndarray] = None  # (N, 3) point cloud of the scan

    key_frame_indices: Optional[list] = None  # indices of key frames
    selected_key_frame_indices: Optional[list] = None  # indices of selected key frames

    octree_voxel_centers: Optional[np.ndarray] = None  # (M, 3) array of voxel centers (x, y, z)
    octree_voxel_sizes: Optional[np.ndarray] = None  # (M, 1) array of voxel sizes
    octree_vertices: Optional[np.ndarray] = None  # (N, 3) array of voxel corner indices
    octree_little_endian_vertex_order: bool = False

    sdf_slice_bounds: Optional[list] = None  # (bound_min, bound_max)
    sdf_slice_axis: Optional[int] = None  # 0, 1, or 2
    sdf_slice_position: Optional[float] = None  # position along the axis
    sdf_slice_resolution: Optional[float] = None  # cell size in meters
    sdf_slice: Optional[dict[str, np.ndarray]] = None  # str -> (H, W) array of SDF values

    sdf_grid_bounds: Optional[list] = None  # (bound_min, bound_max)
    sdf_grid_resolution: Optional[float] = None  # voxel size in meters
    sdf_grid: Optional[dict[str, np.ndarray]] = None  # str -> (X, Y, Z) array of SDF values in a grid

    model_saved_path: Optional[str] = None  # path where the model is saved

    time_stats: Optional[dict] = None  # time statistics

    loss_stats: Optional[dict] = None  # loss statistics


class GuiBase:

    def __init__(self, cfg: GuiBaseConfig, queue_in: Queue = None, queue_out: Queue = None):
        o3d_gui.Application.instance.initialize()

        self.cfg = cfg
        self.queue_in = queue_in
        self.queue_out = queue_out

        self.last_control_packet_timestamp = time.time()
        self.data_packet = GuiDataPacket()  # buffer to hold the latest of each field received
        self.sleep_interval = 0.001  # 1 ms
        self.frame_poses = []
        self.traj_length = 0

        cut = np.array([-0.5, 0.5], dtype=np.float32)
        xx, yy, zz = np.meshgrid(cut, cut, cut, indexing="ij")  # big-endian
        octree_vertex_offsets = np.stack([xx, yy, zz], axis=-1).reshape(1, 8, 3)  # (1,8,3)
        self.octree_vertex_offsets = [
            octree_vertex_offsets,  # big-endian
            octree_vertex_offsets[:, :, ::-1].copy(),  # little-endian
        ]
        self.octree_voxel_lines = np.array(
            [
                [0, 1],
                [1, 3],
                [3, 2],
                [2, 0],
                [0, 4],
                [4, 5],
                [5, 7],
                [7, 6],
                [6, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ],
            dtype=np.int32,
        )

        self.camera_init = False
        self.sdf_slice_to_save = None

        self._init_widgets()

        threading.Thread(target=self.communicate_thread).start()

    def _init_widgets(self):
        self.window_width = 2560
        self.window_height = 1440

        self.window = o3d_gui.Application.instance.create_window("grad_sdf", self.window_width, self.window_height)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        # left: 3d widget, right: panel

        # I. initialize 3d widget
        # 1. create scene widget
        self.widget3d = o3d_gui.SceneWidget()
        self.widget3d.scene = o3d_rendering.Open3DScene(self.window.renderer)
        cg_setting = o3d_rendering.ColorGrading(
            o3d_rendering.ColorGrading.Quality.ULTRA,
            o3d_rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_setting)
        self.window.add_child(self.widget3d)
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())  # type: ignore
        # 2. set renders
        # scan
        self.scan_name = "scan"
        self.scan_render = o3d_rendering.MaterialRecord()
        self.scan_render.shader = "defaultLit"  # defaultUnlit, defaultLit, normals, depth
        self.scan_render.point_size = self.cfg.scan_point_size * self.window.scaling
        self.scan_render.base_color = [0.9, 0.9, 0.9, 0.8]  # type: ignore

        # trajectory
        self.traj_name = "traj"
        self.traj_render = o3d_rendering.MaterialRecord()
        self.traj_render.shader = "unlitLine"
        self.traj_render.line_width = self.cfg.traj_line_width * self.window.scaling

        # key frame cameras
        self.kf_cams_name = "key_frame_cams"
        self.kf_cams_render = o3d_rendering.MaterialRecord()
        self.kf_cams_render.shader = "unlitLine"
        self.kf_cams_render.line_width = self.cfg.camera_line_width * self.window.scaling

        # current camera
        self.curr_cam_name = "current_cam"
        self.curr_cam_render = o3d_rendering.MaterialRecord()
        self.curr_cam_render.shader = "unlitLine"
        self.curr_cam_render.line_width = self.cfg.camera_line_width * self.window.scaling

        # octree
        self.octree_name = "octree"
        self.octree_render = o3d_rendering.MaterialRecord()
        self.octree_render.shader = "unlitLine"
        self.octree_render.line_width = self.cfg.octree_line_width * self.window.scaling

        # sdf slice
        self.sdf_name = "sdf_slice"
        self.sdf_prior_name = "sdf_slice_prior"
        self.sdf_residual_name = "sdf_slice_residual"
        self.sdf_render = o3d_rendering.MaterialRecord()
        self.sdf_render.shader = "defaultLit"
        self.sdf_render.point_size = self.cfg.sdf_point_size * self.window.scaling
        self.sdf_render.base_color = [1.0, 1.0, 1.0, 1.0]  # type: ignore

        # mesh
        self.mesh_name = "mesh"
        self.mesh_prior_name = "mesh_prior"
        self.mesh_render = o3d_rendering.MaterialRecord()
        self.mesh_render.shader = "normals"

        # gt mesh
        self.gt_mesh_name = "gt_mesh"
        self.gt_mesh_render = o3d_rendering.MaterialRecord()
        self.gt_mesh_render.shader = "defaultLit"
        self.gt_mesh_render.base_color = [0.9, 0.9, 0.9, 1.0]  # type: ignore

        # 3. create geometry instances
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])  # type: ignore
        self.scan = o3d.geometry.PointCloud()
        self.traj = o3d.geometry.LineSet()
        self.kf_cams = o3d.geometry.LineSet()
        self.curr_cam = o3d.geometry.LineSet()
        self.octree = o3d.geometry.LineSet()
        self.sdf_slice = o3d.geometry.PointCloud()
        self.sdf_slice_prior = o3d.geometry.PointCloud()
        self.sdf_slice_res = o3d.geometry.PointCloud()
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh_prior = o3d.geometry.TriangleMesh()
        self.gt_mesh = None
        if self.cfg.gt_mesh_path is not None and os.path.exists(self.cfg.gt_mesh_path):
            self.gt_mesh = o3d.io.read_triangle_mesh(self.cfg.gt_mesh_path)
        self.org_cam = o3d.geometry.LineSet()  # camera of identity pose
        s = f = self.cfg.camera_size * self.window.scaling
        self.org_cam.points = o3d.utility.Vector3dVector(
            np.array(
                [
                    [-s, -s, f],
                    [s, -s, f],
                    [s, s, f],
                    [-s, s, f],
                    [0, 0, 0],
                ],
                dtype=np.float64,
            )
        )
        self.org_cam.lines = o3d.utility.Vector2iVector(
            np.array(
                [
                    [4, 0],
                    [4, 1],
                    [4, 2],
                    [4, 3],
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                ],
                dtype=np.int32,
            )
        )

        # II. initialize panel
        # 1. create panel
        em = self.window.theme.font_size
        self.panel = o3d_gui.Vert(0.5 * em, o3d_gui.Margins(left=0.5 * em))

        # 2. add widgets to panel
        # a. switches to pause / resume mapping and visualization
        switch_line = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))
        self.switch_run = o3d_gui.ToggleSwitch("Pause / Resume Mapping")
        self.switch_run.is_on = True
        self.switch_run.set_on_clicked(self._on_switch_run)
        switch_line.add_child(self.switch_run)

        self.switch_vis = o3d_gui.ToggleSwitch("Pause / Resume Visualization")
        self.switch_vis.is_on = True
        self.switch_vis.set_on_clicked(self._on_switch_vis)
        switch_line.add_child(self.switch_vis)

        self.panel.add_child(switch_line)

        # b. options to control view point
        self.panel.add_child(o3d_gui.Label("View Options:"))
        view_options_line = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))

        self.checkbox_view_follow = o3d_gui.Checkbox("Follow")
        self.checkbox_view_follow.checked = True
        self.checkbox_view_follow.set_on_checked(self._on_checkbox_view_follow)
        view_options_line.add_child(self.checkbox_view_follow)

        self.checkbox_view_keyboard = o3d_gui.Checkbox("Keyboard Control")
        self.checkbox_view_keyboard.checked = False
        self.checkbox_view_keyboard.set_on_checked(self._on_checkbox_view_keyboard)
        view_options_line.add_child(self.checkbox_view_keyboard)

        self.panel.add_child(view_options_line)

        # c. options to show / hide different objects
        self.panel.add_child(o3d_gui.Label("3D Objects:"))
        object_options_line1 = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))

        self.checkbox_show_scan = o3d_gui.Checkbox("Scan")
        self.checkbox_show_scan.checked = True
        self.checkbox_show_scan.set_on_checked(self._on_checkbox_show_scan)
        object_options_line1.add_child(self.checkbox_show_scan)

        self.checkbox_show_traj = o3d_gui.Checkbox("Trajectory")
        self.checkbox_show_traj.checked = True
        self.checkbox_show_traj.set_on_checked(self._on_checkbox_show_traj)
        object_options_line1.add_child(self.checkbox_show_traj)

        self.checkbox_show_kf_cams = o3d_gui.Checkbox("Key Frame Cameras")
        self.checkbox_show_kf_cams.checked = False
        self.checkbox_show_kf_cams.set_on_checked(self._on_checkbox_show_kf_cams)
        object_options_line1.add_child(self.checkbox_show_kf_cams)

        self.checkbox_show_curr_cam = o3d_gui.Checkbox("Current Camera")
        self.checkbox_show_curr_cam.checked = False
        self.checkbox_show_curr_cam.set_on_checked(self._on_checkbox_show_curr_cam)
        object_options_line1.add_child(self.checkbox_show_curr_cam)

        object_options_line2 = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))

        self.checkbox_show_octree = o3d_gui.Checkbox("Octree")
        self.checkbox_show_octree.checked = False
        self.checkbox_show_octree.set_on_checked(self._on_checkbox_show_octree)
        object_options_line2.add_child(self.checkbox_show_octree)

        self.checkbox_show_sdf = o3d_gui.Checkbox("SDF Slice")
        self.checkbox_show_sdf.checked = True
        self.checkbox_show_sdf.set_on_checked(self._on_checkbox_show_sdf)
        object_options_line2.add_child(self.checkbox_show_sdf)

        self.checkbox_show_sdf_by_prior = o3d_gui.Checkbox("SDF Slice (Prior)")
        self.checkbox_show_sdf_by_prior.checked = False
        self.checkbox_show_sdf_by_prior.set_on_checked(self._on_checkbox_show_sdf_by_prior)
        object_options_line2.add_child(self.checkbox_show_sdf_by_prior)

        self.checkbox_show_sdf_res = o3d_gui.Checkbox("SDF Slice (Residual)")
        self.checkbox_show_sdf_res.checked = False
        self.checkbox_show_sdf_res.set_on_checked(self._on_checkbox_show_sdf_residual)
        object_options_line2.add_child(self.checkbox_show_sdf_res)

        self.checkbox_show_mesh = o3d_gui.Checkbox("Mesh")
        self.checkbox_show_mesh.checked = True
        self.checkbox_show_mesh.set_on_checked(self._on_checkbox_show_mesh)
        object_options_line2.add_child(self.checkbox_show_mesh)

        self.checkbox_show_mesh_by_prior = o3d_gui.Checkbox("Mesh (Prior)")
        self.checkbox_show_mesh_by_prior.checked = False
        self.checkbox_show_mesh_by_prior.set_on_checked(self._on_checkbox_show_mesh_by_prior)
        object_options_line2.add_child(self.checkbox_show_mesh_by_prior)

        self.checkbox_show_gt_mesh = o3d_gui.Checkbox("GT Mesh")
        self.checkbox_show_gt_mesh.checked = False
        self.checkbox_show_gt_mesh.set_on_checked(self._on_checkbox_show_gt_mesh)
        object_options_line2.add_child(self.checkbox_show_gt_mesh)

        self.panel.add_child(object_options_line1)
        self.panel.add_child(object_options_line2)

        # d. options to control mesh coloring
        mesh_color_options_line = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))
        mesh_color_options_line.add_child(o3d_gui.Label("Mesh Color Options:"))

        self.checkbox_mesh_color_by_normal = o3d_gui.Checkbox("Normal")
        self.checkbox_mesh_color_by_normal.checked = True
        self.checkbox_mesh_color_by_normal.set_on_checked(self._on_checkbox_mesh_color_by_normal)
        mesh_color_options_line.add_child(self.checkbox_mesh_color_by_normal)

        self.checkbox_mesh_color_by_height = o3d_gui.Checkbox("Height")
        self.checkbox_mesh_color_by_height.checked = False
        self.checkbox_mesh_color_by_height.set_on_checked(self._on_checkbox_mesh_color_by_height)
        mesh_color_options_line.add_child(self.checkbox_mesh_color_by_height)

        self.panel.add_child(mesh_color_options_line)

        # e. slider to control mesh update frequency
        mesh_update_freq_line = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))
        mesh_update_freq_line.add_child(o3d_gui.Label("Mesh Update Frequency:"))
        self.slider_mesh_update_freq = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_mesh_update_freq.set_limits(1, 100)
        self.slider_mesh_update_freq.int_value = self.cfg.mesh_update_freq
        mesh_update_freq_line.add_child(self.slider_mesh_update_freq)

        self.panel.add_child(mesh_update_freq_line)

        # f. slider to control mesh resolution
        mesh_resolution_line = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))
        mesh_resolution_line.add_child(o3d_gui.Label("Mesh Resolution (#voxels per meter):"))
        self.slider_mesh_resolution = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_mesh_resolution.set_limits(10, 200)
        self.slider_mesh_resolution.int_value = self.cfg.mesh_resolution
        mesh_resolution_line.add_child(self.slider_mesh_resolution)

        self.panel.add_child(mesh_resolution_line)

        # g. slider to control sdf slice update frequency
        sdf_update_freq_line = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))
        sdf_update_freq_line.add_child(o3d_gui.Label("SDF Slice Update Frequency:"))
        self.slider_sdf_update_freq = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_sdf_update_freq.set_limits(1, 100)
        self.slider_sdf_update_freq.int_value = self.cfg.sdf_slice_update_freq
        sdf_update_freq_line.add_child(self.slider_sdf_update_freq)

        self.panel.add_child(sdf_update_freq_line)

        # h. slider to control sdf point size
        sdf_point_size_line = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))
        sdf_point_size_line.add_child(o3d_gui.Label("SDF Point Size:"))
        self.slider_sdf_point_size = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_sdf_point_size.set_limits(1, 50)
        self.slider_sdf_point_size.int_value = self.cfg.sdf_point_size
        self.slider_sdf_point_size.set_on_value_changed(self._on_slider_sdf_point_size)
        sdf_point_size_line.add_child(self.slider_sdf_point_size)

        self.panel.add_child(sdf_point_size_line)

        # i. options to control sdf slice axis and position
        sdf_axis_position_line = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))
        sdf_axis_position_line.add_child(o3d_gui.Label("SDF Slice Axis:"))
        self.combobox_sdf_axis = o3d_gui.Combobox()
        self.combobox_sdf_axis.add_item("X")
        self.combobox_sdf_axis.add_item("Y")
        self.combobox_sdf_axis.add_item("Z")
        self.combobox_sdf_axis.set_on_selection_changed(self._on_combobox_sdf_axis)
        self.combobox_sdf_axis.selected_index = 2
        sdf_axis_position_line.add_child(self.combobox_sdf_axis)
        sdf_axis_position_line.add_child(o3d_gui.Label("Position:"))
        self.slider_sdf_position = o3d_gui.Slider(o3d_gui.Slider.DOUBLE)
        self.slider_sdf_position.set_limits(-10.0, 10.0)
        self.slider_sdf_position.double_value = 0.0
        self.slider_sdf_position.set_on_value_changed(self._on_slider_sdf_position)
        sdf_axis_position_line.add_child(self.slider_sdf_position)

        self.panel.add_child(sdf_axis_position_line)

        # j. slider to control sdf slice resolution
        sdf_resolution_line = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))
        sdf_resolution_line.add_child(o3d_gui.Label("SDF Slice Resolution (#points per meter):"))
        self.slider_sdf_resolution = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_sdf_resolution.set_limits(10, 200)
        self.slider_sdf_resolution.int_value = self.cfg.sdf_slice_resolution
        sdf_resolution_line.add_child(self.slider_sdf_resolution)

        self.panel.add_child(sdf_resolution_line)

        # k. buttons to save mesh and sdf
        save_buttons_line = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))
        self.button_save_mesh = o3d_gui.Button("Save Mesh")
        self.button_save_mesh.set_on_clicked(self._on_button_save_mesh)
        save_buttons_line.add_child(self.button_save_mesh)

        self.button_save_sdf = o3d_gui.Button("Save SDF")
        self.button_save_sdf.set_on_clicked(self._on_button_save_sdf)
        save_buttons_line.add_child(self.button_save_sdf)

        self.button_save_model = o3d_gui.Button("Save Model")
        self.button_save_model.set_on_clicked(self._on_button_save_model)
        save_buttons_line.add_child(self.button_save_model)

        self.button_save_screenshot = o3d_gui.Button("Save Screenshot")
        self.button_save_screenshot.set_on_clicked(self._on_button_save_screenshot)
        save_buttons_line.add_child(self.button_save_screenshot)

        self.panel.add_child(save_buttons_line)

        # l. info tab
        tabs = o3d_gui.TabControl()
        tab_info = o3d_gui.Vert(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))
        self.label_info_experiment_name = o3d_gui.Label(f"Experiment: {self.cfg.experiment_name}")
        tab_info.add_child(self.label_info_experiment_name)
        self.label_info_frame_idx = o3d_gui.Label("Frame:")
        tab_info.add_child(self.label_info_frame_idx)
        self.label_info_traj_length = o3d_gui.Label("Travel Distance:")
        tab_info.add_child(self.label_info_traj_length)
        self.label_info_gpu_mem = o3d_gui.Label("GPU Memory Usage:")
        tab_info.add_child(self.label_info_gpu_mem)
        self.label_info_fps = o3d_gui.Label("FPS:")
        tab_info.add_child(self.label_info_fps)
        self.label_info_timing = o3d_gui.Label("Timing:")
        tab_info.add_child(self.label_info_timing)
        self.label_info_loss = o3d_gui.Label("Loss:")
        tab_info.add_child(self.label_info_loss)
        tabs.add_tab("Info", tab_info)
        self.panel.add_child(tabs)

        # 3. add panel to window
        self.window.add_child(self.panel)

    def _on_layout(self, layout_context):
        content_rect = self.window.content_rect
        self.widget3d_width = int(self.window.size.width * self.cfg.panel_split_ratio)
        self.widget3d.frame = o3d_gui.Rect(content_rect.x, content_rect.y, self.widget3d_width, content_rect.height)
        self.panel.frame = o3d_gui.Rect(
            self.widget3d.frame.get_right(),
            content_rect.y,
            self.window_width - self.widget3d_width,
            content_rect.height,
        )

    def _on_close(self):
        self.data_packet.flag_exit = True
        tqdm.write("[GUI] Window is being closed.")
        return True

    def _on_switch_run(self, is_on: bool) -> None:
        if self.switch_run.is_on:
            tqdm.write("[GUI] Resume mapping.")
        else:
            tqdm.write("[GUI] Pause mapping.")

    def _on_switch_vis(self, is_on: bool) -> None:
        if is_on:
            tqdm.write("[GUI] Resume visualization.")
        else:
            tqdm.write("[GUI] Pause visualization.")

    def _on_checkbox_view_follow(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] View follow mode on.")
        else:
            tqdm.write("[GUI] View follow mode off.")
            # if self.data_packet.frame_pose is not None:
            #     self.widget3d.scene.camera.look_at(
            #         self.data_packet.frame_pose[:3, 3],
            #         self.data_packet.frame_pose[:3, 3] + self.data_packet.frame_pose[:3, 2],
            #         self.data_packet.frame_pose[:3, 1],
            #     )

    def _on_checkbox_view_keyboard(self, is_checked: bool) -> None:
        if is_checked:
            self.widget3d.set_view_controls(o3d_gui.SceneWidget.Controls.FLY)
        else:
            self.widget3d.set_view_controls(o3d_gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)

    def _on_checkbox_show_scan(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show scan.")
        else:
            tqdm.write("[GUI] Hide scan.")
        self.visualize_scan()

    def _on_checkbox_show_traj(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show trajectory.")
        else:
            tqdm.write("[GUI] Hide trajectory.")
        self.visualize_trajectory()

    def _on_checkbox_show_kf_cams(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show key frame cameras.")
        else:
            tqdm.write("[GUI] Hide key frame cameras.")
        self.visualize_kf_cams()

    def _on_checkbox_show_curr_cam(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show current camera.")
        else:
            tqdm.write("[GUI] Hide current camera.")
        self.visualize_curr_cam()

    def _on_checkbox_show_octree(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show octree.")
        else:
            tqdm.write("[GUI] Hide octree.")
        self.visualize_octree()

    def _on_checkbox_show_sdf(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show SDF slice.")
        else:
            tqdm.write("[GUI] Hide SDF slice.")
        self.visualize_sdf_slice(self.sdf_name, self.sdf_slice, is_checked)

    def _on_checkbox_show_sdf_by_prior(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show SDF slice by prior.")
        else:
            tqdm.write("[GUI] Hide SDF slice by prior.")
        self.visualize_sdf_slice(self.sdf_prior_name, self.sdf_slice_prior, is_checked)

    def _on_checkbox_show_sdf_residual(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show SDF slice residual.")
        else:
            tqdm.write("[GUI] Hide SDF slice residual.")
        self.visualize_sdf_slice(self.sdf_residual_name, self.sdf_slice_res, is_checked)

    def _on_checkbox_show_mesh(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show mesh.")
        else:
            tqdm.write("[GUI] Hide mesh.")
        self.visualize_mesh(self.mesh_name, self.mesh, is_checked)

    def _on_checkbox_show_mesh_by_prior(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show mesh by prior.")
        else:
            tqdm.write("[GUI] Hide mesh by prior.")
        self.visualize_mesh(self.mesh_prior_name, self.mesh_prior, is_checked)

    def _on_checkbox_show_gt_mesh(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show ground truth mesh.")
        else:
            tqdm.write("[GUI] Hide ground truth mesh.")
        self.visualize_gt_mesh()

    def _on_checkbox_mesh_color_by_normal(self, is_checked: bool) -> None:
        if is_checked:
            self.mesh_render.shader = "normals"
            self.checkbox_mesh_color_by_height.checked = False
        self.visualize_mesh(self.mesh_name, self.mesh, self.checkbox_show_mesh.checked)
        self.visualize_mesh(self.mesh_prior_name, self.mesh_prior, self.checkbox_show_mesh_by_prior.checked)

    def _on_checkbox_mesh_color_by_height(self, is_checked: bool) -> None:
        if is_checked:
            self.mesh_render.shader = "defaultLit"
            self.checkbox_mesh_color_by_normal.checked = False
        self.visualize_mesh(self.mesh_name, self.mesh, self.checkbox_show_mesh.checked)
        self.visualize_mesh(self.mesh_prior_name, self.mesh_prior, self.checkbox_show_mesh_by_prior.checked)

    def _on_slider_sdf_point_size(self, point_size: float) -> None:
        tqdm.write(f"[GUI] Set SDF point size to {point_size}.")
        self.sdf_render.point_size = point_size * self.window.scaling
        self.visualize_sdf_slice(self.sdf_name, self.sdf_slice, self.checkbox_show_sdf.checked)
        self.visualize_sdf_slice(self.sdf_prior_name, self.sdf_slice_prior, self.checkbox_show_sdf_by_prior.checked)
        self.visualize_sdf_slice(self.sdf_residual_name, self.sdf_slice_res, self.checkbox_show_sdf_res.checked)

    def _on_combobox_sdf_axis(self, axis_name: str, axis: int) -> None:
        tqdm.write(f"[GUI] Set SDF slice axis to {axis_name}.")
        position = self.slider_sdf_position.double_value
        self.slider_sdf_position.set_limits(self.cfg.scene_bound_min[axis], self.cfg.scene_bound_max[axis])
        if position < self.cfg.scene_bound_min[axis]:
            position = self.cfg.scene_bound_min[axis]
        if position > self.cfg.scene_bound_max[axis]:
            position = self.cfg.scene_bound_max[axis]
        self.slider_sdf_position.double_value = position
        self._on_slider_sdf_position(position)

    def _on_slider_sdf_position(self, position: float) -> None:
        tqdm.write(f"[GUI] Set SDF slice position to {position:.3f} m.")

    def _on_button_save_mesh(self) -> None:
        if self.mesh.is_empty():
            tqdm.write("[GUI] Warning: mesh is empty, not saving.")
            return
        dialog = o3d_gui.FileDialog(o3d_gui.FileDialog.SAVE, "Save Mesh As", self.window.theme)
        dialog.set_path(os.path.join(os.curdir, "mesh.ply"))

        def on_done(path: str) -> None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            o3d.io.write_triangle_mesh(path, self.mesh)
            tqdm.write(f"[GUI] Mesh saved to {path}.")

        dialog.set_on_done(on_done)
        self.window.show_dialog(dialog)

    def _on_button_save_sdf(self) -> None:
        if self.sdf_slice_to_save is None:
            tqdm.write("[GUI] Warning: no SDF slice to save.")
            return
        dialog = o3d_gui.FileDialog(o3d_gui.FileDialog.SAVE, "Save SDF Slice At", self.window.theme)
        dialog.set_path(os.curdir)

        def on_done(folder_path: str) -> None:
            if self.sdf_slice_to_save is None:
                return

            os.makedirs(folder_path, exist_ok=True)

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

            slice_bound: list = self.sdf_slice_to_save["slice_bound"]  # type: ignore
            sdf_slice = self.sdf_slice_to_save["sdf_slice"]
            axis: int = self.sdf_slice_to_save["axis"]  # type: ignore
            axis_name: str = slice_configs[axis]["axis_name"]
            pos: float = self.sdf_slice_to_save["pos"]  # type: ignore
            for slice_name in ["sdf_prior", "sdf_residual", "sdf"]:
                slice_values = sdf_slice[slice_name]  # type: ignore
                plt.figure()
                im = plt.imshow(
                    slice_values,
                    extent=(slice_bound[0][0], slice_bound[1][0], slice_bound[0][1], slice_bound[1][1]),
                    origin="lower",
                    cmap="jet",
                )
                plt.colorbar(im, shrink=0.8)
                plt.xlabel(slice_configs[axis]["xlabel"], fontsize=fontsize)
                plt.ylabel(slice_configs[axis]["ylabel"], fontsize=fontsize)
                plt.title(f"At {axis_name} = {pos:.2f} m", fontsize=fontsize)
                plt.tight_layout()
                img_path = os.path.join(folder_path, f"slice_{axis_name}_{slice_name}.png")
                plt.savefig(img_path, dpi=300)
                plt.close()

            tqdm.write(f"[GUI] SDF slice saved to {folder_path}.")

        dialog.set_on_done(on_done)
        self.window.show_dialog(dialog)

    def _on_button_save_model(self) -> None:
        dialog = o3d_gui.FileDialog(o3d_gui.FileDialog.SAVE, "Save Model As", self.window.theme)
        dialog.set_on_done(lambda path: self.queue_out.put_nowait(GuiControlPacket(save_model_to_path=path)))
        dialog.add_filter(".pth", "PyTorch Model (.pth)")
        self.window.show_dialog(dialog)

    def _on_button_save_screenshot(self) -> None:
        diaglog = o3d_gui.FileDialog(o3d_gui.FileDialog.SAVE, "Save Screenshot As", self.window.theme)
        diaglog.set_path(os.path.join(os.curdir, "screenshot.png"))

        def on_done(path: str) -> None:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            height = self.window.size.height
            width = self.widget3d_width
            app = o3d.visualization.gui.Application.instance
            img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, img)
            tqdm.write(f"[GUI] Screenshot saved to {path}.")

        diaglog.set_on_done(on_done)
        self.window.show_dialog(diaglog)

    def visualize_scan(self, points: np.ndarray = None):
        if points is not None:
            self.scan.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            self.scan.paint_uniform_color(np.array(self.cfg.scan_point_color, dtype=np.float64))

        if not self.switch_vis.is_on:
            return

        if self.checkbox_show_scan.checked:
            self.widget3d.scene.remove_geometry(self.scan_name)
            self.widget3d.scene.add_geometry(self.scan_name, self.scan, self.scan_render)
        self.widget3d.scene.show_geometry(self.scan_name, self.checkbox_show_scan.checked)

    def visualize_trajectory(self):
        if len(self.traj.points) < len(self.frame_poses):
            for i in range(len(self.traj.points), len(self.frame_poses)):
                self.traj.points.append(self.frame_poses[i][:3, 3])
                n = len(self.traj.points)
                if n > 1:
                    self.traj.lines.append(np.array([n - 2, n - 1], dtype=np.int32))
                    self.traj.colors.append(np.array(self.cfg.traj_color, dtype=np.float64))
                    self.traj_length += np.linalg.norm(self.traj.points[-1] - self.traj.points[-2])

        if not self.switch_vis.is_on:
            return

        if self.checkbox_show_traj.checked:
            self.widget3d.scene.remove_geometry(self.traj_name)
            self.widget3d.scene.add_geometry(self.traj_name, self.traj, self.traj_render)

        self.widget3d.scene.show_geometry(self.traj_name, self.checkbox_show_traj.checked and len(self.traj.lines) > 0)

    def visualize_kf_cams(self, key_frame_indices: list = None, selected_key_frame_indices: list = None):
        if key_frame_indices is not None and selected_key_frame_indices is not None:
            n = len(key_frame_indices) * len(self.org_cam.points)
            if len(self.kf_cams.points) < n:  # need to add more cameras
                existing_kf_count = len(self.kf_cams.points) // len(self.org_cam.points)
                for idx in key_frame_indices[existing_kf_count:]:
                    pose = self.frame_poses[idx]
                    points = np.asarray(self.org_cam.points)
                    points = points @ pose[:3, :3].T + pose[:3, [3]].T
                    start_idx = len(self.kf_cams.points)
                    self.kf_cams.points.extend(o3d.utility.Vector3dVector(points.astype(np.float64)))
                    self.kf_cams.lines.extend(np.asarray(self.org_cam.lines) + start_idx)
            self.kf_cams.paint_uniform_color(np.array(self.cfg.camera_color_key_frame, dtype=np.float64))
            for idx in selected_key_frame_indices:
                m = len(self.org_cam.lines)
                for i in range(m * idx, m * (idx + 1)):
                    self.kf_cams.colors[i] = np.array(self.cfg.camera_color_selected_key_frame, dtype=np.float64)

        if not self.switch_vis.is_on:
            return

        if self.checkbox_show_kf_cams.checked:
            self.widget3d.scene.remove_geometry(self.kf_cams_name)
            self.widget3d.scene.add_geometry(self.kf_cams_name, self.kf_cams, self.kf_cams_render)

        self.widget3d.scene.show_geometry(self.kf_cams_name, self.checkbox_show_kf_cams.checked)

    def visualize_curr_cam(self, pose: np.ndarray = None):
        if pose is not None:  # update visualization only when new pose is given
            points = np.asarray(self.org_cam.points)
            points = points @ pose[:3, :3].T + pose[:3, [3]].T
            self.curr_cam.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            if len(self.curr_cam.lines) == 0:
                self.curr_cam.lines = self.org_cam.lines
                self.curr_cam.paint_uniform_color(np.array(self.cfg.camera_color_current, dtype=np.float64))

        if not self.switch_vis.is_on:
            return

        if self.checkbox_show_curr_cam.checked:
            self.widget3d.scene.remove_geometry(self.curr_cam_name)
            self.widget3d.scene.add_geometry(self.curr_cam_name, self.curr_cam, self.curr_cam_render)

        self.widget3d.scene.show_geometry(self.curr_cam_name, self.checkbox_show_curr_cam.checked)

    def visualize_octree(
        self,
        voxel_centers: Optional[np.ndarray] = None,  # (N, 3)
        voxel_sizes: Optional[np.ndarray] = None,  # (N, 1)
        voxel_vertices: Optional[np.ndarray] = None,  # (N, 8)
        little_endian_vertex_order: bool = False,
    ):
        if voxel_centers is not None and voxel_sizes is not None and voxel_vertices is not None:
            vertex_offsets = self.octree_vertex_offsets[1 if little_endian_vertex_order else 0]  # (1, 8, 3)
            n_vertices = np.max(voxel_vertices) + 1
            vertices = voxel_centers.reshape(-1, 1, 3) + voxel_sizes.reshape(-1, 1, 1) * vertex_offsets
            vertices_unique = np.zeros((n_vertices, 3), dtype=np.float64)
            vertices_unique[voxel_vertices.flatten()] = vertices.reshape(-1, 3)
            lines = voxel_vertices[:, self.octree_voxel_lines].reshape(-1, 2)  # (N, 12, 2) -> (N*12, 2)
            self.octree.points = o3d.utility.Vector3dVector(vertices_unique)
            self.octree.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
            self.octree.paint_uniform_color(np.array(self.cfg.octree_color, dtype=np.float64))

        if not self.switch_vis.is_on:
            return

        if self.checkbox_show_octree.checked:
            self.widget3d.scene.remove_geometry(self.octree_name)
            self.widget3d.scene.add_geometry(self.octree_name, self.octree, self.octree_render)

        self.widget3d.scene.show_geometry(self.octree_name, self.checkbox_show_octree.checked)

    def visualize_sdf_slice(
        self,
        sdf_name: str,
        sdf_slice: o3d.geometry.PointCloud,
        show: bool,
        bounds: list = None,
        axis: int = None,
        pos: float = None,
        resolution: float = None,
        sdf_values: np.ndarray = None,
    ):
        if (
            bounds is not None
            and axis is not None
            and pos is not None
            and resolution is not None
            and sdf_values is not None
        ):
            if axis == 0:
                # bounds = (bounds_min, bounds_max)
                y, z = np.meshgrid(
                    [
                        np.arange(bounds[0][0], bounds[1][0], resolution),
                        np.arange(bounds[0][1], bounds[1][1], resolution),
                    ],
                    indexing="ij",
                )
                points = np.stack((np.full_like(y, pos), y, z), axis=-1).astype(np.float64)
            elif axis == 1:
                x, z = np.meshgrid(
                    [
                        np.arange(bounds[0][0], bounds[1][0], resolution),
                        np.arange(bounds[0][2], bounds[1][2], resolution),
                    ],
                    indexing="ij",
                )
                points = np.stack((x, np.full_like(x, pos), z), axis=-1).astype(np.float64)
            else:  # axis == 2
                x, y = np.meshgrid(
                    [
                        np.arange(bounds[0][0], bounds[1][0], resolution),
                        np.arange(bounds[0][1], bounds[1][1], resolution),
                    ],
                    indexing="ij",
                )
                points = np.stack((x, y, np.full_like(x, pos)), axis=-1).astype(np.float64)

            color_map = cm.get_cmap(self.cfg.sdf_color_map)
            sdf_min = np.min(sdf_values)
            sdf_max = np.max(sdf_values)
            sdf_values = np.clip((sdf_values - sdf_min) / (sdf_max - sdf_min + 1e-6), 0, 1)
            colors = color_map(sdf_values.flatten())[:, :3]
            self.sdf_slice.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
            self.sdf_slice.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        if not self.switch_vis.is_on:
            return

        if show:
            self.widget3d.scene.remove_geometry(sdf_name)
            self.widget3d.scene.add_geometry(sdf_name, sdf_slice, self.sdf_render)

        self.widget3d.scene.show_geometry(sdf_name, show)

    def visualize_mesh(
        self,
        mesh_name: str,
        mesh: o3d.geometry.TriangleMesh,
        show: bool,
        bounds: list = None,
        resolution: float = None,
        sdf_grid: np.ndarray = None,
    ):
        if bounds is not None and resolution is not None and sdf_grid is not None:
            mc = MarchingCubes()
            vertices, faces, face_normals = mc.run(
                coords_min=bounds[0],
                grid_res=[resolution] * 3,
                grid_shape=sdf_grid.shape,
                grid_values=sdf_grid.flatten(),
                iso_value=0.0,
                row_major=True,
                parallel=True,
            )
            mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
            mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
            mesh.triangle_normals = o3d.utility.Vector3dVector(face_normals.astype(np.float64))
            mesh.compute_vertex_normals()
            mesh.vertex_colors.clear()

        if not self.switch_vis.is_on:
            return

        if show:
            if self.checkbox_mesh_color_by_height.checked:
                z_values = np.asarray(mesh.vertices)[:, 2]
                z_min, z_max = np.min(z_values), np.max(z_values)
                z_normalized = np.clip((z_values - z_min) / (z_max - z_min + 1e-6), 0.0, 1.0)
                color_map = cm.get_cmap(self.cfg.mesh_height_color_map)
                colors = color_map(z_normalized)[:, :3].astype(np.float64)
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

            self.widget3d.scene.remove_geometry(mesh_name)
            self.widget3d.scene.add_geometry(mesh_name, mesh, self.mesh_render)

        self.widget3d.scene.show_geometry(mesh_name, show)

    def visualize_gt_mesh(self, data=None):
        if self.gt_mesh is None:
            return

        if not self.switch_vis.is_on:
            return

        if self.checkbox_show_gt_mesh.checked:
            self.widget3d.scene.remove_geometry(self.gt_mesh_name)
            self.widget3d.scene.add_geometry(self.gt_mesh_name, self.gt_mesh, self.gt_mesh_render)
        self.widget3d.scene.show_geometry(self.gt_mesh_name, self.checkbox_show_gt_mesh.checked)

    def center_bev(self):
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(90, bounds, bounds.get_center())

    @classmethod
    def run(cls, *args, **kwargs):
        app = o3d_gui.Application.instance
        app.initialize()
        window = cls(*args, **kwargs)
        app.run()

    def send_control_packet(self):
        if self.queue_out is None:
            return
        packet = GuiControlPacket()
        packet.flag_mapping_run = self.switch_run.is_on
        if self.checkbox_show_sdf.checked:
            packet.sdf_slice_frequency = self.slider_sdf_update_freq.int_value
            packet.sdf_slice_axis = self.combobox_sdf_axis.selected_index
            packet.sdf_slice_position = self.slider_sdf_position.double_value
            packet.sdf_slice_resolution = self.slider_sdf_resolution.int_value
        if self.checkbox_show_mesh.checked:
            packet.sdf_grid_frequency = self.slider_mesh_update_freq.int_value
            packet.sdf_grid_resolution = self.slider_mesh_resolution.int_value
        self.queue_out.put(packet)

    def receive_data_packet(self, get_latest: bool = True):
        if self.queue_in is None:
            return None
        packet: Optional[GuiDataPacket] = None
        while True and not self.queue_in.empty():
            try:
                packet_new = self.queue_in.get_nowait()
                if packet is not None:
                    del packet
                packet = packet_new
                if packet.flag_exit:
                    tqdm.write("[GUI] Received exit signal. Closing GUI...")
                    self.data_packet.flag_exit = True
                    break
                if packet.mapping_end:
                    self.data_packet.mapping_end = True

                if packet.frame_idx >= 0:
                    self.data_packet.frame_idx = packet.frame_idx
                if packet.frame_pose is not None:
                    self.data_packet.frame_pose = packet.frame_pose
                if packet.scan_points is not None:
                    self.data_packet.scan_points = packet.scan_points

                if packet.key_frame_indices is not None:
                    self.data_packet.key_frame_indices = packet.key_frame_indices
                if packet.selected_key_frame_indices is not None:
                    self.data_packet.selected_key_frame_indices = packet.selected_key_frame_indices

                if packet.octree_voxel_centers is not None:
                    self.data_packet.octree_voxel_centers = packet.octree_voxel_centers
                if packet.octree_voxel_sizes is not None:
                    self.data_packet.octree_voxel_sizes = packet.octree_voxel_sizes
                if packet.octree_vertices is not None:
                    self.data_packet.octree_vertices = packet.octree_vertices
                if packet.octree_little_endian_vertex_order is not None:
                    self.data_packet.octree_little_endian_vertex_order = packet.octree_little_endian_vertex_order

                if packet.sdf_slice_bounds is not None:
                    self.data_packet.sdf_slice_bounds = packet.sdf_slice_bounds
                if packet.sdf_slice_axis is not None:
                    self.data_packet.sdf_slice_axis = packet.sdf_slice_axis
                if packet.sdf_slice_position is not None:
                    self.data_packet.sdf_slice_position = packet.sdf_slice_position
                if packet.sdf_slice_resolution is not None:
                    self.data_packet.sdf_slice_resolution = packet.sdf_slice_resolution
                if packet.sdf_slice is not None:
                    self.data_packet.sdf_slice = packet.sdf_slice

                if packet.sdf_grid_bounds is not None:
                    self.data_packet.sdf_grid_bounds = packet.sdf_grid_bounds
                if packet.sdf_grid_resolution is not None:
                    self.data_packet.sdf_grid_resolution = packet.sdf_grid_resolution
                if packet.sdf_grid is not None:
                    self.data_packet.sdf_grid = packet.sdf_grid

                if packet.model_saved_path is not None:
                    self.data_packet.model_saved_path = packet.model_saved_path
                if packet.time_stats is not None:
                    self.data_packet.time_stats = packet.time_stats
                if packet.loss_stats is not None:
                    self.data_packet.loss_stats = packet.loss_stats

                if not get_latest:
                    break
            except asyncio.QueueEmpty:
                break
        return packet

    def update(self):
        data_packet = self.receive_data_packet(get_latest=True)
        if self.data_packet.flag_exit:
            tqdm.write("[GUI] Received exit signal. Closing GUI...")
            return

        if data_packet is not None:
            # we received some new data
            # multiple data packets may be merged into one and stored in self.data_packet
            if self.data_packet.mapping_end:
                tqdm.write("[GUI] Mapping has ended.")
                self.sleep_interval = 0.1  # slow down

            if self.data_packet.frame_pose is not None:
                self.frame_poses.append(self.data_packet.frame_pose)
                self.visualize_trajectory()  # update trajectory
                self.visualize_curr_cam(self.data_packet.frame_pose)  # update current camera
                self.label_info_traj_length.text = f"Travel Distance: {self.traj_length:.3f} m"
                self.data_packet.frame_pose = None  # reset

            if self.data_packet.scan_points is not None:
                self.visualize_scan(self.data_packet.scan_points)  # update scan
                self.data_packet.scan_points = None  # reset

            if (
                self.data_packet.key_frame_indices is not None
                or self.data_packet.selected_key_frame_indices is not None
            ):
                assert self.data_packet.key_frame_indices is not None
                assert self.data_packet.selected_key_frame_indices is not None
                self.visualize_kf_cams(
                    self.data_packet.key_frame_indices,
                    self.data_packet.selected_key_frame_indices,
                )
                self.data_packet.key_frame_indices = None  # reset
                self.data_packet.selected_key_frame_indices = None  # reset

            if (
                self.data_packet.octree_voxel_centers is not None
                or self.data_packet.octree_voxel_sizes is not None
                or self.data_packet.octree_vertices is not None
            ):
                assert self.data_packet.octree_voxel_centers is not None
                assert self.data_packet.octree_voxel_sizes is not None
                assert self.data_packet.octree_vertices is not None
                self.visualize_octree(
                    self.data_packet.octree_voxel_centers,
                    self.data_packet.octree_voxel_sizes,
                    self.data_packet.octree_vertices,
                    self.data_packet.octree_little_endian_vertex_order,
                )
                self.data_packet.octree_voxel_centers = None  # reset
                self.data_packet.octree_voxel_sizes = None  # reset
                self.data_packet.octree_vertices = None  # reset

            if self.data_packet.frame_idx >= 0:
                self.label_info_frame_idx.text = f"Frame: {self.data_packet.frame_idx}"
                self.data_packet.frame_idx = -1  # reset

            # update sdf slice
            if (
                self.data_packet.sdf_slice_bounds is not None
                or self.data_packet.sdf_slice_axis is not None
                or self.data_packet.sdf_slice_position is not None
                or self.data_packet.sdf_slice_resolution is not None
                or self.data_packet.sdf_slice is not None
            ):
                assert self.data_packet.sdf_slice_bounds is not None
                assert self.data_packet.sdf_slice_axis is not None
                assert self.data_packet.sdf_slice_position is not None
                assert self.data_packet.sdf_slice_resolution is not None
                assert self.data_packet.sdf_slice is not None

                self.sdf_slice_to_save = dict(
                    bounds=self.data_packet.sdf_slice_bounds,
                    axis=self.data_packet.sdf_slice_axis,
                    position=self.data_packet.sdf_slice_position,
                    resolution=self.data_packet.sdf_slice_resolution,
                    sdf_slice=self.data_packet.sdf_slice,
                )

                self.visualize_sdf_slice(
                    self.sdf_name,
                    self.sdf_slice,
                    self.checkbox_show_sdf.checked,
                    self.data_packet.sdf_slice_bounds,
                    self.data_packet.sdf_slice_axis,
                    self.data_packet.sdf_slice_position,
                    self.data_packet.sdf_slice_resolution,
                    self.data_packet.sdf_slice["sdf"],
                )
                self.visualize_sdf_slice(
                    self.sdf_prior_name,
                    self.sdf_slice_prior,
                    self.checkbox_show_sdf_by_prior.checked,
                    self.data_packet.sdf_slice_bounds,
                    self.data_packet.sdf_slice_axis,
                    self.data_packet.sdf_slice_position,
                    self.data_packet.sdf_slice_resolution,
                    self.data_packet.sdf_slice["sdf_by_prior"],
                )
                self.data_packet.sdf_slice_bounds = None  # reset
                self.data_packet.sdf_slice_axis = None  # reset
                self.data_packet.sdf_slice_position = None  # reset
                self.data_packet.sdf_slice_resolution = None  # reset
                self.data_packet.sdf_slice = None  # reset

            # update mesh
            if (
                self.data_packet.sdf_grid_bounds is not None
                or self.data_packet.sdf_grid_resolution is not None
                or self.data_packet.sdf_grid is not None
            ):
                assert self.data_packet.sdf_grid_bounds is not None
                assert self.data_packet.sdf_grid_resolution is not None
                assert self.data_packet.sdf_grid is not None

                self.visualize_mesh(
                    self.mesh_name,
                    self.mesh,
                    self.checkbox_show_mesh.checked,
                    self.data_packet.sdf_grid_bounds,
                    self.data_packet.sdf_grid_resolution,
                    self.data_packet.sdf_grid["sdf"],
                )
                self.visualize_mesh(
                    self.mesh_prior_name,
                    self.mesh_prior,
                    self.checkbox_show_mesh_by_prior.checked,
                    self.data_packet.sdf_grid_bounds,
                    self.data_packet.sdf_grid_resolution,
                    self.data_packet.sdf_grid["sdf_by_prior"],
                )
                self.data_packet.sdf_grid_bounds = None  # reset
                self.data_packet.sdf_grid_resolution = None  # reset
                self.data_packet.sdf_grid = None  # reset

            if self.data_packet.model_saved_path is not None:
                tqdm.write(f"[GUI] Model saved to {self.data_packet.model_saved_path}.")
                self.data_packet.model_saved_path = None  # reset

            if self.data_packet.time_stats is not None:
                self.label_info_fps.text = f"FPS: {1.0 / (self.data_packet.time_stats['train_frame'] + 1e-6):.3f}"
                self.label_info_timing.text = f"Timing:\n" + "\n".join(
                    [f"  {k}: {v:.6f} s" for k, v in self.data_packet.time_stats.items()]
                )
                self.data_packet.time_stats = None  # reset

            if self.data_packet.loss_stats is not None:
                self.label_info_loss.text = f"Loss:\n" + "\n".join(
                    [f"  {k}: {v:.6f}" for k, v in self.data_packet.loss_stats.items()]
                )
                self.data_packet.loss_stats = None  # reset

        if not self.camera_init:
            self.center_bev()
            self.camera_init = True

        current_time = time.time()
        if current_time - self.last_control_packet_timestamp > 0.005:  # 200 Hz
            self.send_control_packet()
            self.last_control_packet_timestamp = current_time

    def communicate_thread(self):
        while True:
            time.sleep(self.sleep_interval)
            if self.data_packet.flag_exit:
                tqdm.write("[GUI] Exiting communicate thread...")
                while not self.queue_in.empty():
                    self.queue_in.get_nowait()
                self.queue_in = None
                self.queue_out = None
                o3d_gui.Application.instance.quit()
                break  # exit thread
            o3d_gui.Application.instance.post_to_main_thread(self.window, self.update)
