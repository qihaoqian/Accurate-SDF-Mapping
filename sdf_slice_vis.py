import os
import sys
import torch
import numpy as np
import yaml
import trimesh
import pysdf
from scipy.spatial import cKDTree
import open3d as o3d
from tqdm import tqdm
import random
import pandas as pd
import matplotlib.pyplot as plt
import vedo
import cv2

# Add paths to import project modules
sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath('src'))

# 添加项目根目录到Python路径，确保可以导入demo模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 现在可以正确导入demo模块
from demo.parser import get_parser

from src.utils.import_util import get_dataset, get_decoder
from src.frame import DepthFrame
from src.loggers import BasicLogger
from src.mapping import Mapping
from src.functions.render_helpers import find_voxel_idx, get_features


def load_checkpoint(ckpt_path, args=None):
    """
    Load trained checkpoint file
    
    Args:
        ckpt_path (str): checkpoint file path, e.g.:
                        "mapping/logs/replica/room0/2025-08-06-19-44-27/ckpt/final_ckpt.pth"
        args: training parameters, if None then need to load from config file
    
    Returns:
        mapper: Mapping object with loaded state
        decoder: decoder with loaded state
    """
    
    # Load sparse octree library
    torch.classes.load_library(
        "third_party/sparse_octree/build/lib.linux-x86_64-cpython-310/svo.cpython-310-x86_64-linux-gnu.so")
    
    # 1. Load checkpoint file
    print(f"Loading checkpoint: {ckpt_path}")
    training_result = torch.load(ckpt_path)
    
    # Check checkpoint content
    print("Checkpoint keys:", list(training_result.keys()))
    
    # 3. Create data stream (for initialization)
    data_stream = get_dataset(args)
    # data_in = data_stream[0]
    # first_frame = DepthFrame(*data_in[:-1], offset=args.mapper_specs['offset'], 
    #                        ref_pose=data_in[-1]).cuda()
    # W, H = first_frame.rgb.shape[1], first_frame.rgb.shape[0]
    
    # 4. Create logger and mapper
    logger = BasicLogger(args, for_eva=True)
    mapper = Mapping(args, logger, data_stream=data_stream)
    
    # 5. Restore state from checkpoint
    print("Restoring model state...")
    
    # Restore decoder state
    mapper.decoder.load_state_dict(training_result['decoder_state'])
    
    # Restore map state
    mapper.map_states = training_result['map_state']
    
    # Set to evaluation mode
    mapper.decoder = mapper.decoder.cuda()
    mapper.decoder.eval()
    
    print("Checkpoint loading completed!")
    print(f"Decoder parameters: {sum(p.numel() for p in mapper.decoder.parameters())}")
    print(f"Map state keys: {list(mapper.map_states.keys())}")
    
    return mapper


def inference(mapper, points, batch_size=100000):
    points = points
    points = torch.tensor(points, dtype=torch.float32, device='cuda', requires_grad=False)
    sdf_pred = []
    sdf_priors = []
    hash_features = []
    # print(points.min(axis=0), points.max(axis=0))
    with torch.no_grad():
        for i in range(0, points.shape[0], batch_size):
            batch_points = points[i:i+batch_size]
            batch_points_voxel_idx = find_voxel_idx(batch_points, mapper.map_states)
            batch_sdf_priors = get_features(batch_points, batch_points_voxel_idx, mapper.map_states, mapper.voxel_size)
            batch_hash_features = mapper.decoder(batch_points)
            sdf_priors_features = batch_sdf_priors['sdf_priors'].squeeze(1)
            batch_sdf_pred = sdf_priors_features + batch_hash_features['sdf']
            sdf_pred.append(batch_sdf_pred.cpu())
            sdf_priors.append(sdf_priors_features.cpu())
            hash_features.append(batch_hash_features['sdf'].cpu())
            # print(batch_hash_features['sdf'].min(), batch_hash_features['sdf'].max())
            del batch_sdf_priors, batch_hash_features, batch_sdf_pred
            torch.cuda.empty_cache()
        sdf_pred = torch.cat(sdf_pred, dim=0)
        sdf_priors = torch.cat(sdf_priors, dim=0)
        hash_features = torch.cat(hash_features, dim=0)
    return sdf_pred, sdf_priors, hash_features


def get_points(bound, res_per_meter, offset=0):
    """
    :param bound: 体素边界，形如[[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    :param res_per_meter: 每米采样多少个点（分辨率）
    :param offset: 偏移量
    :return: points, (x_res, y_res, z_res)
    """
    bound = np.array(bound) - offset
    x_min, x_max = bound[0]
    y_min, y_max = bound[1]
    z_min, z_max = bound[2]

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_res = int(np.round(x_range * res_per_meter)) + 1
    y_res = int(np.round(y_range * res_per_meter)) + 1
    z_res = int(np.round(z_range * res_per_meter)) + 1

    x = np.linspace(x_min, x_max, x_res)
    y = np.linspace(y_min, y_max, y_res)
    z = np.linspace(z_min, z_max, z_res)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)

    return points, (x_res, y_res, z_res)


def visualize_sdf_slice(
    sdf_grid,
    slice_idx,
    slice_axis,
    *,
    x_vals=None,
    y_vals=None,
    z_vals=None,
    save_dir=".",
    filename=None,
    title=None,
    cmap="jet",
    vlim=None,                  # 手动设定[-vlim, vlim]，不设则自动对称
    show=False,                  # 需要屏幕显示就设 True
    smin=None,
    smax=None
):
    """
    可视化 SDF 体数据在某一轴向上的二维切片。

    参数
    ----
    sdf_grid : (Nx, Ny, Nz) numpy array
    slice_idx : int，切片索引
    slice_axis : {"x","y","z"}，切片轴
    x_vals, y_vals, z_vals : 1D array，可选；对应各轴坐标（米）。不传则用索引 0..N-1
    save_dir : 保存目录
    filename : 保存文件名；不传则自动生成
    title : 图标题；不传则自动生成
    cmap : 颜色图
    vlim : 颜色轴范围的正半幅（对称到 [-vlim, vlim]）
    show : 是否调用 plt.show()
    """
    # 1) 基础检查与坐标向量准备
    assert slice_axis in {"x","y","z"}, "slice_axis 必须是 'x'/'y'/'z'"
    Nx, Ny, Nz = sdf_grid.shape

    if x_vals is None: x_vals = np.arange(Nx)
    if y_vals is None: y_vals = np.arange(Ny)
    if z_vals is None: z_vals = np.arange(Nz)
    
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color='white')
    
    # 2) 取切片 & 确定绘图的两个坐标轴（横轴=第二维，纵轴=第一维）
    if slice_axis == "x":
        assert 0 <= slice_idx < Nx, "slice_idx 越界"
        sdf_slice = sdf_grid[slice_idx, :, :]          # (Ny, Nz)
        coord1_vals, coord2_vals = y_vals, z_vals      # 行对应 y，列对应 z
        coord_names = ("y", "z")
    elif slice_axis == "y":
        assert 0 <= slice_idx < Ny, "slice_idx 越界"
        sdf_slice = sdf_grid[:, slice_idx, :]          # (Nx, Nz)
        coord1_vals, coord2_vals = x_vals, z_vals      # 行对应 x，列对应 z
        coord_names = ("x", "z")
    else:  # "z"
        assert 0 <= slice_idx < Nz, "slice_idx 越界"
        sdf_slice = sdf_grid[:, :, slice_idx]          # (Nx, Ny)
        coord1_vals, coord2_vals = x_vals, y_vals      # 行对应 x，列对应 y
        coord_names = ("x", "y")

    # 3) 生成网格（注意：contour 默认 'xy' 语义 => X=横轴(列)，Y=纵轴(行)）
    X, Y = np.meshgrid(coord2_vals, coord1_vals, indexing="xy")  # X 对应列，Y 对应行

    # 4) 颜色范围：对称到 0
    if smin is None: smin = float(np.min(sdf_slice))
    if smax is None: smax = float(np.max(sdf_slice))
    print(f"SDF range: [{smin:.4f}, {smax:.4f}]")
    if vlim is None:
        vlim = max(abs(smin), abs(smax)) if (smin != 0 or smax != 0) else 1.0

    # 5) 绘图
    plt.figure(figsize=(12, 10))

    # imshow: 数据 shape=(n行, n列) => (len(coord1), len(coord2))
    # extent: [xmin, xmax, ymin, ymax] = [横轴min,max, 纵轴min,max]
    im = plt.imshow(
        sdf_slice,
        extent=[coord2_vals[0], coord2_vals[-1], coord1_vals[0], coord1_vals[-1]],
        origin="lower",
        cmap=cmap,
        vmin=-vlim, vmax=vlim,
        aspect="equal"
    )

    # 等高线：用与 imshow 一致的 X(横) / Y(纵)
    # 自动给出一组等高线（含 0 等高），也可按需改 levels
    # 这里采用对称 levels，保证包含零等高线
    n_levels = 10
    levels = np.linspace(-vlim, vlim, n_levels)
    cs = plt.contour(X, Y, sdf_slice, levels=levels, colors="k", alpha=0.25, linewidths=0.6)

    # 0 等高线高亮（表面）
    zero_cs = plt.contour(X, Y, sdf_slice, levels=[0.0], colors="red", linewidths=1.8)

    cbar = plt.colorbar(im, shrink=0.6)
    cbar.ax.tick_params(labelsize=21)
    plt.gca().tick_params(axis='both', which='major', labelsize=21)
    plt.tight_layout()
    # cbar.set_label("SDF Value")

    # 轴标签与标题
    # plt.xlabel(f"{coord_names[1]} (m)", fontsize=12)  # 横轴是第二个坐标
    # plt.ylabel(f"{coord_names[0]} (m)", fontsize=12)  # 纵轴是第一个坐标
    # if title is None:
    #     title = f"SDF slice @ {slice_axis}={slice_idx}"
    # plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)

    # # 辅助信息
    # plt.suptitle(
    #     f"Range: [{smin:.4f}, {smax:.4f}] | Grid: {sdf_slice.shape[0]}×{sdf_slice.shape[1]} | vlim=±{vlim:.4f}",
    #     fontsize=10, y=0.95
    # )

    # 6) 保存/显示
    if filename is None:
        filename = f"sdf_slice_{slice_axis}{slice_idx}.png"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close()

    # 7) 日志
    print(f"SDF slice saved to: {save_path}")
    print(f"  Slice index: {slice_idx} (axis {slice_axis})")
    print(f"  SDF range: [{smin:.4f}, {smax:.4f}]")
    print(f"  Grid size: {sdf_slice.shape[0]}×{sdf_slice.shape[1]}")


def visualize_sdf_slice_with_mesh(
        sdf_grid,
        mesh_path,
        traj_path):
        
    mesh: vedo.Mesh = vedo.load(
        mesh_path
    )
    mesh.backface_culling(True).color("gray")
    mesh.scale(80)  # set scale according to sdf grid

    bounds = mesh.bounds()
    bounds[0] += 30
    mesh.crop(bounds=bounds)

    sdf_slice = sdf_grid[210]
    # 将sdf_slice小于-0.1的都变为-0.1
    sdf_slice[sdf_slice < -0.1] = -0.1
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
    light1 = vedo.Light(pos=(50, 14.9236, -4.83036), focal_point=(55,-30,0), c='white', intensity=0.5)
    light2 = vedo.Light(pos=(50, 14.9236, -4.83036), focal_point=(55,30,0), c='white', intensity=0.8)
    light3 = vedo.Light(pos=(50, 14.9236, -4.83036), focal_point=(55,0,30), c='white', intensity=0.8)
    light4 = vedo.Light(pos=(50, 14.9236, -4.83036), focal_point=(55,0,-30), c='white', intensity=0.8)
    light5 = vedo.Light(pos=(50, 14.9236, -4.83036), focal_point=(0,0,0), c='white', intensity=0.5)

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
    corners = np.array([
        [-s, -s, f],
        [ s, -s, f],
        [ s,  s, f],
        [-s,  s, f],
    ], dtype=float)
    center = np.array([[0.0, 0.0, 0.0]], dtype=float)

    rect_start = corners
    rect_end   = np.roll(corners, -1, axis=0)

    rays_start = np.repeat(center, 4, axis=0)
    rays_end   = corners

    start_local = np.vstack([rect_start, rays_start])  # (8,3)
    end_local   = np.vstack([rect_end,   rays_end])    # (8,3)

    start_h = np.hstack([start_local, np.ones((start_local.shape[0], 1))])  # (8,4)
    end_h   = np.hstack([end_local,   np.ones((end_local.shape[0],   1))])  # (8,4)

    cam_frame_actors = []
    step = 400

    for i in range(0, len(traj_pose), step):
        pose = traj_pose[i]  # (4,4)

        start_world = (pose @ start_h.T).T[:, :3]
        end_world   = (pose @ end_h.T).T[:, :3]

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
        camera = dict(
            pos=(-785.997, 16.8091, 19.4312),
            focal_point=(2.29340, -0.0386084, 0.829294),
            viewup=(0.0214245, 0.999768, 2.41345e-3),
            roll=0.167201,
            distance=788.690,
            clipping_range=(481.157, 1158.23),
        ),
        screenshot=os.path.join("sdf_slice_with_mesh_with_light_traj.png")
    )


def crop_mesh_roof(mesh_path, aabb_min, aabb_max):
    mesh = trimesh.load(mesh_path)
    V = mesh.vertices
    F = mesh.faces
    vmask = np.all((V >= aabb_min) & (V <= aabb_max), axis=1)  # Whether vertices are inside the box
    fmask = vmask[F].all(axis=1)                               # All three vertices of faces are inside the box
    cropped = trimesh.Trimesh(vertices=V, faces=F[fmask], process=False)
    cropped.remove_unreferenced_vertices()
    cropped_mesh_path = os.path.join(os.path.dirname(mesh_path), "roof_cropped_mesh.obj")
    cropped.export(cropped_mesh_path)
    return cropped_mesh_path

def main():
    parser = get_parser()
    args = parser.parse_args()
    ckpt_path = os.path.join(args.log_dir, args.exp_name, "ckpt", "final_ckpt.pth")
    save_dir = os.path.join(args.log_dir, args.exp_name, "misc")
    mesh_path = os.path.join(args.log_dir, args.exp_name, "mesh", "mesh_80.obj")
    traj_path = os.path.join(args.data_specs['data_path'], "traj.txt")
    mapper = load_checkpoint(ckpt_path, args)
    points, (x_res, y_res, z_res) = get_points([[2.47, 5.57], [1.52, 6.52], [0.0, 8.04]], 80, 0)
    print(x_res, y_res, z_res)
    print(f"Points shape: {points.shape}",points.min(axis=0),points.max(axis=0))
    points = torch.tensor(points, dtype=torch.float32, device='cuda', requires_grad=False)
    sdf_pred, sdf_prior, hash_features = inference(mapper, points)
    sdf_pred_grid = sdf_pred.cpu().numpy().reshape(x_res, y_res, z_res)
    print(sdf_pred_grid.min(), sdf_pred_grid.max())
    # crop_aabb_min = np.array([2.47, 1.52, 1.0]) - args.mapper_specs['offset']
    # crop_aabb_max = np.array([5.57, 6.52, 8.04]) - args.mapper_specs['offset']
    # cropped_mesh_path = crop_mesh_roof(mesh_path, crop_aabb_min, crop_aabb_max)
    # sdf_prior_grid = sdf_prior.cpu().numpy().reshape(x_res, y_res, z_res)
    # np.save(os.path.join(save_dir, "sdf_pred_grid.npy"), sdf_pred_grid)
    # np.save(os.path.join(save_dir, "sdf_prior_grid.npy"), sdf_prior_grid)
    # visualize_sdf_slice(
    #     sdf_pred_grid,
    #     200,
    #     "x",
    #     x_vals=np.linspace(2.47, 5.57, x_res),
    #     y_vals=np.linspace(1.52, 6.52, y_res),
    #     z_vals=np.linspace(0.0, 8.04, z_res),
    #     save_dir=save_dir,
    #     filename=f"h2mapping_sdf_pred_200_x_ego_colorbar.png",
    #     title="SDF Pred",
    #     # smin=-0.9948,
    #     # smax=0.5117
    # )
    visualize_sdf_slice_with_mesh(
        sdf_pred_grid,
        mesh_path,
        traj_path
    )

if __name__ == "__main__":
    main()