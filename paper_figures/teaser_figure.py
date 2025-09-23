import numpy as np
import vedo
import cv2
import argparse

from grad_sdf.evaluator_grad_sdf import GradSdfEvaluator
from grad_sdf.trainer_config import TrainerConfig


def get_prediction():
    evaluator = GradSdfEvaluator(
        batch_size=20480,
    )


def visualize_sdf_slice_with_mesh(sdf_grid, mesh, traj_path, img_path):
    mesh.backface_culling(True).color("gray")
    mesh.scale(80)  # set scale according to sdf grid

    bounds = mesh.bounds()
    bounds[0] += 30
    mesh.crop(bounds=bounds)

    sdf_slice = sdf_grid[:, :, 210]
    sdf_slice[sdf_slice < -0.1] = -0.1  # clip for better visualization
    sdf_colored = (sdf_slice - sdf_slice.min()) / (sdf_slice.max() - sdf_slice.min()) * 255
    print(sdf_slice.min(), sdf_slice.max())
    sdf_colored = sdf_colored.astype(np.uint8)
    sdf_colored = cv2.applyColorMap(sdf_colored, cv2.COLORMAP_JET)
    sdf_colored = cv2.cvtColor(sdf_colored, cv2.COLOR_BGR2RGB)

    img = vedo.Image(sdf_colored)
    pos = mesh.bounds()[::2]
    print("mesh bounds:", mesh.bounds())
    pos[0] = 85
    pos[1] -= 15
    img.rotate_y(-90).pos(*pos)

    center = mesh.center_of_mass().tolist()
    center[0] -= sdf_colored.shape[0] // 2
    center[1] -= sdf_colored.shape[1] // 2
    print(center)
    light1 = vedo.Light(pos=(50, 14.9236, -4.83036), focal_point=(55, -30, 0), c="white", intensity=0.5)
    light2 = vedo.Light(pos=(50, 14.9236, -4.83036), focal_point=(55, 30, 0), c="white", intensity=0.8)
    light3 = vedo.Light(pos=(50, 14.9236, -4.83036), focal_point=(55, 0, 30), c="white", intensity=0.8)
    light4 = vedo.Light(pos=(50, 14.9236, -4.83036), focal_point=(55, 0, -30), c="white", intensity=0.8)
    light5 = vedo.Light(pos=(50, 14.9236, -4.83036), focal_point=(0, 0, 0), c="white", intensity=0.5)

    traj_pose = np.loadtxt(traj_path).reshape(-1, 4, 4)
    traj_pose[:, :3, 3] = traj_pose[:, :3, 3] * 80
    traj = traj_pose[:, :3, 3]
    traj_lines: vedo.Lines = vedo.Lines(start_pts=traj[:-1], end_pts=traj[1:])
    traj_lines.color("red")
    traj_lines.linewidth(5)

    # s = 8
    # f = 8
    # cam_frames = np.array([
    #     [-s, -s, f],
    #     [ s, -s, f],
    #     [ s,  s, f],
    #     [-s,  s, f],
    #     [ 0,  0, 0],
    #     [-s, -s, f],
    #     [ 0,  0, 0],
    #     [ s, -s, f],
    #     [ 0,  0, 0],
    #     [ s,  s, f],
    #     [ 0,  0, 0],
    # ])

    # # 扩展为齐次坐标
    # cam_frames_h = np.hstack([cam_frames, np.ones((cam_frames.shape[0], 1))])  # (11,4)

    # # 3. 每隔200帧画一个相机框
    # cam_frame_actors = []
    # for i in range(0, len(traj_pose), 200):
    #     pose = traj_pose[i]   # (4,4)
    #     world_pts = (pose @ cam_frames_h.T).T[:, :3]  # 变换到世界坐标
    #     lines = vedo.Lines(world_pts[:-1], world_pts[1:])
    #     lines.color("blue").linewidth(2)
    #     cam_frame_actors.append(lines)

    # cam_frames_vedo: vedo.Lines = vedo.Lines(start_pts=cam_frames[:-1], end_pts=cam_frames[1:])
    # cam_frames_vedo.color("blue").linewidth(2)

    s, f = 10, 10
    corners = np.array(
        [
            [-s, -s, f],
            [s, -s, f],
            [s, s, f],
            [-s, s, f],
        ],
        dtype=float,
    )
    center = np.array([[0.0, 0.0, 0.0]], dtype=float)

    rect_start = corners
    rect_end = np.roll(corners, -1, axis=0)

    rays_start = np.repeat(center, 4, axis=0)
    rays_end = corners

    start_local = np.vstack([rect_start, rays_start])  # (8,3)
    end_local = np.vstack([rect_end, rays_end])  # (8,3)

    start_h = np.hstack([start_local, np.ones((start_local.shape[0], 1))])  # (8,4)
    end_h = np.hstack([end_local, np.ones((end_local.shape[0], 1))])  # (8,4)

    cam_frame_actors = []
    step = 400

    for i in range(0, len(traj_pose), step):
        pose = traj_pose[i]  # (4,4)

        start_world = (pose @ start_h.T).T[:, :3]
        end_world = (pose @ end_h.T).T[:, :3]

        lines = vedo.Lines(start_pts=start_world, end_pts=end_world)
        lines.color("blue").linewidth(2)
        cam_frame_actors.append(lines)

    traj_xyz = traj_pose[:, :3, 3]
    traj_lines = vedo.Lines(traj_xyz[:-1], traj_xyz[1:]).c("red").lw(5)

    vedo.show(
        mesh,
        img,
        light1,
        light2,
        light3,
        light4,
        light5,
        traj_lines,
        # cam_frames_vedo,
        *cam_frame_actors,
        size=(1920, 1080),
        axes=1,
        # camera=dict(
        #     pos=(-785.997, 16.8091, 19.4312),
        #     focal_point=(2.29340, -0.0386084, 0.829294),
        #     viewup=(0.0214245, 0.999768, 2.41345e-3),
        #     roll=0.167201,
        #     distance=788.690,
        #     clipping_range=(481.157, 1158.23),
        # ),
        # screenshot=img_path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--batch-size", type=int, default=20480)
    parser.add_argument("--gt-mesh-path", type=str, reverse=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    trainer_cfg = TrainerConfig.from_yaml(args.config)

    evaluator = GradSdfEvaluator(
        batch_size=args.batch_size,
        model_cfg=trainer_cfg.model,
        model_path=args.model_path,
        # add offset to the model input if specified in the dataset args
        # the test data is not offset, so we need to offset the model input
        model_input_offset=trainer_cfg.data.dataset_args.get("offset", None),
        device=args.device,
    )
    results = evaluator.extract_slice(
        axis=2,
        pos=0.0,
        resolution=0.0125,
        bound_min=trainer_cfg.model.residual_net_cfg.bound_min,
        bound_max=trainer_cfg.model.residual_net_cfg.bound_max,
    )

    gt_mesh: vedo.Mesh = vedo.load(args.gt_mesh_path)
    offset = trainer_cfg.data.dataset_args.get("offset", None)
    if offset is not None:
        offset = np.array(offset)
        gt_mesh.pos(offset - np.array(gt_mesh.pos()))

    mesh: vedo.Mesh = vedo.load(
        "/home/daizhirui/D/GoogleDrive/Documents/UCSD/Research/ERL/SDF/Neural-SDF/reconstructed_mesh_result/room0-our.obj"
    )
    mesh.backface_culling(True).color("gray")
    mesh.scale(20)  # set scale according to sdf grid

    sdf_grid = np.load(
        "/home/daizhirui/D/GoogleDrive/Documents/UCSD/Research/ERL/SDF/Neural-SDF/sdf_grid_result/room0-voxblox.npy"
    )

    sdf_slice = sdf_grid[48]
    sdf_colored = (sdf_slice - sdf_slice.min()) / (sdf_slice.max() - sdf_slice.min()) * 255
    sdf_colored = sdf_colored.astype(np.uint8)
    sdf_colored = cv2.applyColorMap(sdf_colored, cv2.COLORMAP_JET)
    sdf_colored = cv2.cvtColor(sdf_colored, cv2.COLOR_BGR2RGB)

    img = vedo.Image(sdf_colored)
    pos = mesh.bounds()[::2]
    pos[0] = 20
    img.rotate_y(-90).pos(*pos)

    center = mesh.center_of_mass().tolist()
    center[0] -= sdf_colored.shape[0] // 2
    center[1] -= sdf_colored.shape[1] // 2
    print(center)

    traj = np.loadtxt("").reshape(-1, 4, 4)
    traj = traj[:, :3, 3]  # (N, 3)
    traj_lines: vedo.Lines = vedo.Lines(start_pts=traj[:-1], end_pts=traj[1:])
    traj_lines.color("red")
    traj_lines.linewidth(5)

    s = 2
    f = 2
    cam_frames = np.array(
        [
            [-s, -s, f],
            [s, -s, f],
            [s, s, f],
            [-s, s, f],
            [0, 0, 0],
            [-s, -s, f],
            [0, 0, 0],
            [s, -s, f],
            [0, 0, 0],
            [s, s, f],
            [0, 0, 0],
        ]
    )
    cam_frames_vedo: vedo.Lines = vedo.Lines(start_pts=cam_frames[:-1], end_pts=cam_frames[1:])
    cam_frames_vedo.color("blue").linewidth(2)

    vedo.show(
        mesh,
        img,
        size=(1440, 1440),
        axes=1,
        camera=dict(
            pos=(-0.856456, 1.59392, -3.71767),
            focal_point=(0.379982, 0.178761, -0.204810),
            viewup=(-0.950620, -0.116014, 0.287858),
            roll=97.1292,
            distance=3.98393,
            clipping_range=(0.0158820, 15.8820),
        ),
    )


if __name__ == "__main__":
    main()
