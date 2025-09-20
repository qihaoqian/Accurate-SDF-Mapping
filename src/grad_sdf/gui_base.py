from dataclasses import dataclass

from tqdm import tqdm

from grad_sdf import o3d, o3d_gui, o3d_rendering
from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class GuiBaseConfig(ConfigABC):
    panel_split_ratio: float = 0.7
    scan_point_size: int = 2
    sdf_point_size: int = 2
    traj_line_width: int = 2
    mesh_update_freq: int = 10
    mesh_resolution: int = 100
    sdf_slice_update_freq: int = 10
    sdf_slice_resolution: int = 100
    experiment_name: str = "default_experiment"


class GuiBase:

    def __init__(self, cfg: GuiBaseConfig):
        self.cfg = cfg
        self.is_done = False

        self._init_widgets()

    def _init_widgets(self):
        self.window_width = 2560
        self.window_height = 1440

        self.window = o3d_gui.Application.instance.create_window(
            "grad_sdf", self.window_width, self.window_height
        )
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
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center()) # type: ignore
        # 2. set renders
        # scan
        self.scan_name = "scan"
        self.scan_render = o3d_rendering.MaterialRecord()
        self.scan_render.shader = "defaultLit"  # defaultUnlit, defaultLit, normals, depth
        self.scan_render.point_size = self.cfg.scan_point_size * self.window.scaling
        self.scan_render.base_color = [0.9, 0.9, 0.9, 0.8] # type: ignore

        # sdf slice
        self.sdf_name = "sdf_slice"
        self.sdf_render = o3d_rendering.MaterialRecord()
        self.sdf_render.shader = "defaultLit"
        self.sdf_render.point_size = self.cfg.sdf_point_size * self.window.scaling
        self.sdf_render.base_color = [1.0, 1.0, 1.0, 1.0] # type: ignore

        # mesh
        self.mesh_name = "mesh"
        self.mesh_render = o3d_rendering.MaterialRecord()
        self.mesh_render.shader = "normals"

        # octree
        self.octree_name = "octree"
        self.octree_render = o3d_rendering.MaterialRecord()
        self.octree_render.shader = "unlitLine"
        self.octree_render.base_color = [1.0, 1.0, 1.0, 1.0] # type: ignore

        # gt mesh
        self.gt_mesh_name = "gt_mesh"
        self.gt_mesh_render = o3d_rendering.MaterialRecord()
        self.gt_mesh_render.shader = "defaultLit"
        self.gt_mesh_render.base_color = [0.9, 0.9, 0.9, 1.0] # type: ignore

        # trajectory
        self.traj_name = "traj"
        self.traj_render = o3d_rendering.MaterialRecord()
        self.traj_render.shader = "unlitLine"
        self.traj_render.line_width = self.cfg.traj_line_width * self.window.scaling

        # 3. create geometry instances
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]) # type: ignore
        self.scan = o3d.geometry.PointCloud()
        self.sdf_slice = o3d.geometry.PointCloud()
        self.mesh = o3d.geometry.TriangleMesh()
        self.gt_mesh = o3d.geometry.TriangleMesh()
        self.traj = o3d.geometry.LineSet()

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

        object_options_line2 = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))

        self.checkbox_show_sdf = o3d_gui.Checkbox("SDF Slice")
        self.checkbox_show_sdf.checked = True
        self.checkbox_show_sdf.set_on_checked(self._on_checkbox_show_sdf)
        object_options_line2.add_child(self.checkbox_show_sdf)

        self.checkbox_show_mesh = o3d_gui.Checkbox("Mesh")
        self.checkbox_show_mesh.checked = True
        self.checkbox_show_mesh.set_on_checked(self._on_checkbox_show_mesh)
        object_options_line2.add_child(self.checkbox_show_mesh)

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

        self.checkbox_mesh_by_height = o3d_gui.Checkbox("Height")
        self.checkbox_mesh_by_height.checked = False
        self.checkbox_mesh_by_height.set_on_checked(self._on_checkbox_mesh_by_height)
        mesh_color_options_line.add_child(self.checkbox_mesh_by_height)

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

        # h. slider to control sdf slice resolution
        sdf_resolution_line = o3d_gui.Horiz(spacing=0.5 * em, margins=o3d_gui.Margins(left=0.5 * em, top=0.5 * em))
        sdf_resolution_line.add_child(o3d_gui.Label("SDF Slice Resolution (#points per meter):"))
        self.slider_sdf_resolution = o3d_gui.Slider(o3d_gui.Slider.INT)
        self.slider_sdf_resolution.set_limits(10, 200)
        self.slider_sdf_resolution.int_value = self.cfg.sdf_slice_resolution
        sdf_resolution_line.add_child(self.slider_sdf_resolution)

        self.panel.add_child(sdf_resolution_line)

        # i. buttons to save mesh and sdf
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

        # j. info tab
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
        self.is_done = True
        tqdm.write("[GUI] Window is being closed.")

        # TODO: clean up resources, e.g. pipes

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
        raise NotImplementedError

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

    def _on_checkbox_show_sdf(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show SDF slice.")
        else:
            tqdm.write("[GUI] Hide SDF slice.")
        self.visualize_sdf_slice()

    def _on_checkbox_show_mesh(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show mesh.")
        else:
            tqdm.write("[GUI] Hide mesh.")
        self.visualize_mesh()

    def _on_checkbox_show_gt_mesh(self, is_checked: bool) -> None:
        if is_checked:
            tqdm.write("[GUI] Show ground truth mesh.")
        else:
            tqdm.write("[GUI] Hide ground truth mesh.")
        self.visualize_gt_mesh()

    def _on_checkbox_mesh_color_by_normal(self, is_checked: bool) -> None:
        if is_checked:
            self.mesh_render.shader = "normals"
            self.checkbox_mesh_by_height.checked = False
        self.visualize_mesh()

    def _on_checkbox_mesh_by_height(self, is_checked: bool) -> None:
        if is_checked:
            self.mesh_render.shader = "defaultLit"
            self.checkbox_mesh_color_by_normal.checked = False
        self.visualize_mesh()

    def _on_button_save_mesh(self) -> None:
        raise NotImplementedError

    def _on_button_save_sdf(self) -> None:
        raise NotImplementedError

    def _on_button_save_model(self) -> None:
        raise NotImplementedError

    def _on_button_save_screenshot(self) -> None:
        raise NotImplementedError

    def visualize_scan(self, data=None):
        if data is None:
            # TODO: get data from somewhere
            pass
        if data is None:
            return
        if self.checkbox_show_scan.checked:
            self.render_scan(data)

            self.widget3d.scene.remove_geometry(self.scan_name)
            self.widget3d.scene.add_geometry(self.scan_name, self.scan, self.scan_render)

        self.widget3d.scene.show_geometry(self.scan_name, self.checkbox_show_scan.checked)

    def render_scan(self, data=None):
        raise NotImplementedError

    def visualize_trajectory(self, data=None):
        if data is None:
            # TODO: get data from somewhere
            pass
        if data is None:
            return
        if self.checkbox_show_traj.checked:
            self.render_trajectory(data)

            self.widget3d.scene.remove_geometry(self.traj_name)
            self.widget3d.scene.add_geometry(self.traj_name, self.traj, self.traj_render)

        self.widget3d.scene.show_geometry(self.traj_name, self.checkbox_show_traj.checked)

    def render_trajectory(self, data=None):
        raise NotImplementedError

    def visualize_sdf_slice(self, data=None):
        if data is None:
            # TODO: get data from somewhere
            pass
        if data is None:
            return
        if self.checkbox_show_sdf.checked:
            self.render_sdf_slice(data)

            self.widget3d.scene.remove_geometry(self.sdf_name)
            self.widget3d.scene.add_geometry(self.sdf_name, self.sdf_slice, self.sdf_render)
        self.widget3d.scene.show_geometry(self.sdf_name, self.checkbox_show_sdf.checked)

    def render_sdf_slice(self, data=None):
        raise NotImplementedError

    def visualize_mesh(self, data=None):
        if data is None:
            # TODO: get data from somewhere
            pass
        if data is None:
            return
        if self.checkbox_show_mesh.checked:
            self.render_mesh(data)

            self.widget3d.scene.remove_geometry(self.mesh_name)
            self.widget3d.scene.add_geometry(self.mesh_name, self.mesh, self.mesh_render)
        self.widget3d.scene.show_geometry(self.mesh_name, self.checkbox_show_mesh.checked)

    def render_mesh(self, data=None):
        raise NotImplementedError

    def visualize_gt_mesh(self, data=None):
        if data is None:
            # TODO: get data from somewhere
            pass
        if data is None:
            return
        if self.checkbox_show_gt_mesh.checked:
            self.render_gt_mesh(data)

            self.widget3d.scene.remove_geometry(self.gt_mesh_name)
            self.widget3d.scene.add_geometry(self.gt_mesh_name, self.gt_mesh, self.gt_mesh_render)
        self.widget3d.scene.show_geometry(self.gt_mesh_name, self.checkbox_show_gt_mesh.checked)

    def render_gt_mesh(self, data=None):
        raise NotImplementedError



    @classmethod
    def run(cls, *args, **kwargs):
        app = o3d_gui.Application.instance
        app.initialize()
        window = cls(*args, **kwargs)
        app.run()
