import os
import sys
import torch
import numpy as np
import yaml
import trimesh
import pysdf
from scipy.spatial import cKDTree
# Add paths to import project modules
sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath('src'))

# 添加项目根目录到Python路径，确保可以导入demo模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 现在可以正确导入demo模块
from demo.parser import get_parser

from src.utils.import_util import get_dataset, get_decoder
from src.frame import RGBDFrame
from src.loggers import BasicLogger
from src.mapping import Mapping

torch.set_grad_enabled(False)


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
    training_result = torch.load(ckpt_path, map_location='cuda:0')
    
    # Check checkpoint content
    print("Checkpoint keys:", list(training_result.keys()))
    
    # 2. Create decoder
    decoder = get_decoder(args).cuda()
    print("Decoder created")
    
    # 3. Create data stream (for initialization)
    data_stream = get_dataset(args)
    data_in = data_stream[0]
    first_frame = RGBDFrame(*data_in[:-1], offset=args.mapper_specs['offset'], 
                           ref_pose=data_in[-1]).cuda()
    W, H = first_frame.rgb.shape[1], first_frame.rgb.shape[0]
    
    # 4. Create logger and mapper
    logger = BasicLogger(args, for_eva=True)
    mapper = Mapping(args, logger, data_stream=data_stream)
    
    # 5. Restore state from checkpoint
    print("Restoring model state...")
    
    # Restore decoder state
    mapper.decoder.load_state_dict(training_result['decoder_state'])
    
    # Restore SDF priors and map state
    mapper.sdf_priors = training_result['sdf_priors'].cuda()
    mapper.map_states = training_result['map_state']
    
    # Set to evaluation mode
    mapper.decoder = mapper.decoder.cuda()
    mapper.decoder.eval()
    
    print("Checkpoint loading completed!")
    print(f"Decoder parameters: {sum(p.numel() for p in mapper.decoder.parameters())}")
    print(f"SDF priors shape: {mapper.sdf_priors.shape}")
    print(f"Map state keys: {list(mapper.map_states.keys())}")
    
    return mapper, decoder


def load_and_extract_mesh(ckpt_path, args, mesh_res=256, output_dir=None):
    """
    Load checkpoint and extract mesh
    
    Args:
        ckpt_path (str): checkpoint file path
        args: configuration parameters
        mesh_res (int): mesh resolution, default 256
        output_dir (str): output directory, default to mesh directory at same level as checkpoint
    
    Returns:
        mesh: extracted mesh object
        output_path: mesh save path
    """
    
    # 1. Load checkpoint
    print("=" * 50)
    print("Start loading checkpoint and reconstructing mesh")
    print("=" * 50)
    
    mapper, decoder = load_checkpoint(ckpt_path, args)
    
    # 2. Set output directory
    if output_dir is None:
        # Default save to mesh directory at same level as checkpoint
        ckpt_dir = os.path.dirname(ckpt_path)
        result_dir = os.path.dirname(ckpt_dir)  # parent directory
        output_dir = os.path.join(result_dir, "mesh")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # 3. Update mapper's mesh directory
    mapper.logger.mesh_dir = output_dir
    
    # 4. Extract mesh
    print(f"\nStart extracting mesh, resolution: {mesh_res}")
    print("This may take several minutes...")
    
    try:
        mesh, sdf_u, prior_u, decoder_u = mapper.extract_mesh(
            res=mesh_res, 
            map_states=mapper.map_states
        )
        
        # 5. Save mesh
        if args.save_mesh:
            mesh_name = f"reconstructed_mesh_res{mesh_res}.ply"
            output_path = os.path.join(output_dir, mesh_name)
            mesh.export(output_path)
            print(f"🔍 Mesh saved to: {output_path}")
        else:
            output_path = None

        
        print(f"\n✅ Mesh reconstruction completed!")
        print(f"📁 Output path: {output_path}")
        print(f"📊 Vertex count: {len(mesh.vertices)}")
        print(f"📊 Face count: {len(mesh.faces)}")
        
        # 6. Save additional debug information
        if args.save_mesh:
            debug_dir = os.path.join(output_dir, "debug")
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            np.save(os.path.join(debug_dir, f"sdf_u_res{mesh_res}.npy"), sdf_u)
            np.save(os.path.join(debug_dir, f"prior_u_res{mesh_res}.npy"), prior_u)
            np.save(os.path.join(debug_dir, f"decoder_u_res{mesh_res}.npy"), decoder_u)
            
            print(f"🔍 Debug data saved to: {debug_dir}")
        
        return mesh, sdf_u, prior_u
        
    except Exception as e:
        print(f"❌ Mesh extraction failed: {str(e)}")
        raise e
    
def get_points(bound, res, offset=0):
    # Convert bound to numpy array and apply offset
    bound = np.array(bound) - offset
    x_min, x_max = bound[0]
    y_min, y_max = bound[1]
    z_min, z_max = bound[2]
    
    # Calculate the range for each dimension
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    # Find the minimum range as baseline to ensure consistent point spacing
    min_range = min(x_range, y_range, z_range)
    spacing = min_range / (res - 1)  # Baseline point spacing
    
    # Calculate resolution for each dimension based on range and baseline spacing
    x_res = int(x_range / spacing) + 1
    y_res = int(y_range / spacing) + 1
    z_res = int(z_range / spacing) + 1
    
    x = np.linspace(x_min, x_max, x_res)
    y = np.linspace(y_min, y_max, y_res)
    z = np.linspace(z_min, z_max, z_res) 
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    
    return points, (x_res, y_res, z_res)

def rgba_visualize(points, values, alpha=1.0, save_path=None):
    """
    向量化版：values 为 ndarray，返回 shape=(N,4) 的 RGBA
    """
    v_min, v_max = np.min(values), np.max(values)
    v = ((values - v_min) / (v_max - v_min)).astype(np.float32)
    h = v * 5.0 + 1.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    f = np.where((i % 2) == 0, 1.0 - f, f)  # 偶数翻转
    n = 1.0 - f

    # 分段赋值（与上面标量逻辑一致）
    r = np.where(i <= 1, n,
        np.where(i == 2, 0.0,
        np.where(i == 3, 0.0,
        np.where(i == 4, n, 1.0))))
    g = np.where(i <= 1, 0.0,
        np.where(i == 2, n,
        np.where(i == 3, 1.0,
        np.where(i == 4, 1.0, n))))
    b = np.where(i <= 1, 1.0,
        np.where(i == 2, 1.0,
        np.where(i == 3, n, 0.0)))

    a = np.full_like(v, float(alpha))

    rgba = np.stack([r, g, b, a], axis=-1)
    
    import open3d as o3d
    
    
    # 展平颜色数组 (只取RGB，忽略alpha通道)
    colors = rgba.reshape(-1, 4)[:, :3]  # 只要RGB通道
    points = points.reshape(-1, 3)
    
    # 可选：过滤掉值为0的点（如果不想显示空白区域）
    # non_zero_mask = values_flat > 1e-6
    # points = points[non_zero_mask]
    # colors = colors[non_zero_mask]
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存点云
    success = o3d.io.write_point_cloud(save_path, pcd)
    
    if success:
        print(f"点云已保存到: {save_path}")
        print(f"点云包含 {len(points)} 个点")
        print(f"坐标范围: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] "
              f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] "
              f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    else:
        print(f"保存点云失败: {save_path}")
    
    return pcd


def get_sdf_loss(args, sdf_u, prior_u, gt_mesh):
    bound = args.decoder_specs['bound']
    offset = args.mapper_specs['offset']
    res = args.mapper_specs['mesh_res']
    
    points, (x_res, y_res, z_res) = get_points(bound, res, offset)
    
    vertices = gt_mesh.vertices.astype(np.float32)
    faces = gt_mesh.faces.astype(np.int32)
    f = pysdf.SDF(vertices, faces)
    gt_sdf_values = f(points)
    gt_sdf_values = gt_sdf_values.reshape(x_res, y_res, z_res)
    threshold = 2.0
    outliers_mask = np.abs(gt_sdf_values) > threshold
    num_outliers = np.sum(outliers_mask)
    if num_outliers > 0:
        print(f"Found {num_outliers} outlier SDF values (|sdf| > {threshold})")
        print(f"Outliers range: [{gt_sdf_values[outliers_mask].min():.4f}, {gt_sdf_values[outliers_mask].max():.4f}]")
        gt_sdf_values = np.clip(gt_sdf_values, -threshold, threshold)
    near_surface_mask = (gt_sdf_values >= -1e-3) & (gt_sdf_values < 0.1)
    num_true = np.count_nonzero(near_surface_mask)
    total = near_surface_mask.size
    print(f"Number of near surface points: {num_true} / {total} ({num_true / total * 100:.2f}%)")

    gt_sdf_positive_mask = (gt_sdf_values >= -1e-3)
    gt_sdf_negative_mask = (gt_sdf_values < -1e-3)
    
    # Convert to PyTorch tensors
    import torch
    
    sdf_u_tensor = torch.from_numpy(sdf_u).float()
    prior_u_tensor = torch.from_numpy(prior_u).float()
    gt_sdf_values_tensor = torch.from_numpy(gt_sdf_values).float()
    gt_sdf_positive_mask_tensor = torch.from_numpy(gt_sdf_positive_mask).bool()
    near_surface_mask_tensor = torch.from_numpy(near_surface_mask).bool()
    
    print(f"sdf_u.shape: {sdf_u.shape}, prior_u.shape: {prior_u.shape}")
    print(f"gt_sdf_values.shape: {gt_sdf_values.shape}")
    print(f"gt_sdf_positive_mask.shape: {gt_sdf_positive_mask.shape}")
    print(f"gt_sdf_negative_mask.shape: {gt_sdf_negative_mask.shape}")
    print(f"near_surface_mask.shape: {near_surface_mask.shape}")
    
    # Calculate MAE (Mean Absolute Error)
    sdf_mae = torch.mean(torch.abs(sdf_u_tensor[gt_sdf_positive_mask_tensor] - gt_sdf_values_tensor[gt_sdf_positive_mask_tensor]))
    prior_sdf_mae = torch.mean(torch.abs(prior_u_tensor[gt_sdf_positive_mask_tensor] - gt_sdf_values_tensor[gt_sdf_positive_mask_tensor]))
    near_surface_sdf_mae = torch.mean(torch.abs(sdf_u_tensor[near_surface_mask_tensor] - gt_sdf_values_tensor[near_surface_mask_tensor]))
    near_surface_prior_sdf_mae = torch.mean(torch.abs(prior_u_tensor[near_surface_mask_tensor] - gt_sdf_values_tensor[near_surface_mask_tensor]))
    
    # Calculate MSE (Mean Squared Error)
    sdf_mse = torch.mean((sdf_u_tensor[gt_sdf_positive_mask_tensor] - gt_sdf_values_tensor[gt_sdf_positive_mask_tensor]) ** 2)
    prior_sdf_mse = torch.mean((prior_u_tensor[gt_sdf_positive_mask_tensor] - gt_sdf_values_tensor[gt_sdf_positive_mask_tensor]) ** 2)
    near_surface_sdf_mse = torch.mean((sdf_u_tensor[near_surface_mask_tensor] - gt_sdf_values_tensor[near_surface_mask_tensor]) ** 2)
    near_surface_prior_sdf_mse = torch.mean((prior_u_tensor[near_surface_mask_tensor] - gt_sdf_values_tensor[near_surface_mask_tensor]) ** 2)
    

    # Create dictionary containing MAE and MSE metrics
    loss_dict = {
        'sdf_mae': sdf_mae.item(),
        'prior_sdf_mae': prior_sdf_mae.item(),
        'near_surface_sdf_mae': near_surface_sdf_mae.item(),
        'near_surface_prior_sdf_mae': near_surface_prior_sdf_mae.item(),
        'sdf_mse': sdf_mse.item(),
        'prior_sdf_mse': prior_sdf_mse.item(),
        'near_surface_sdf_mse': near_surface_sdf_mse.item(),
        'near_surface_prior_sdf_mse': near_surface_prior_sdf_mse.item()
    }

    return loss_dict, gt_sdf_positive_mask

def crop_mesh_to_aabb(mesh: trimesh.Trimesh, aabb_min, aabb_max):
    """Keep faces with all three vertices inside AABB and remove unreferenced vertices"""
    V = mesh.vertices
    F = mesh.faces
    vmask = np.all((V >= aabb_min) & (V <= aabb_max), axis=1)  # Whether vertices are inside the box
    fmask = vmask[F].all(axis=1)                               # All three vertices of faces are inside the box
    cropped = trimesh.Trimesh(vertices=V, faces=F[fmask], process=False)
    cropped.remove_unreferenced_vertices()
    return cropped

def sample_points_on_mesh(mesh: trimesh.Trimesh, n_points=200_000):
    """Area-weighted uniform sampling of surface points; n_points can be adjusted up/down based on scene"""
    if mesh.is_empty or len(mesh.faces) == 0:
        return np.empty((0, 3), dtype=np.float32)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)

def mesh_metrics(pts_a: np.ndarray, pts_b: np.ndarray, threshold: float):
    """
    基于点云的 Chamfer 距离和 F1 score
    
    Args:
        pts_a: GT 点云，shape (N, 3)
        pts_b: 预测点云，shape (M, 3)  
        threshold: 匹配的最大距离阈值（例如 0.05 表示 5 cm）
        
    Returns:
        dict: 包含所有评估指标的字典
    """
    tree_a = cKDTree(pts_a) #gt
    tree_b = cKDTree(pts_b) #pred

    d_b2a, _ = tree_a.query(pts_b, k=1, workers=-1)
    d_a2b, _ = tree_b.query(pts_a, k=1, workers=-1)
    comp_ratio = np.mean(d_a2b < threshold).astype(np.float32)
    comp_ratio_std = np.std(d_a2b < threshold).astype(np.float32)
    comp = np.mean(d_a2b)
    comp_std = np.std(d_a2b)
    acc = np.mean(d_b2a)
    acc_std = np.std(d_b2a)

    # Chamfer（非平方版）
    chamfer = float(d_a2b.mean() + d_b2a.mean())
    rgba_visualize(pts_a, d_a2b, save_path=os.path.join(args.log_dir, args.exp_name, "misc", "d_a2b.ply"))
    rgba_visualize(pts_b, d_b2a, save_path=os.path.join(args.log_dir, args.exp_name, "misc", "d_b2a.ply"))

    # F1 计算
    tp = np.sum(d_b2a <= threshold)      # Predicted points correctly matched to GT
    fp = np.sum(d_b2a > threshold)       # Predicted points too far from GT
    fn = np.sum(d_a2b > threshold)       # GT points not matched by predicted

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        'chamfer_distance': float(chamfer),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'complete_ratio': float(comp_ratio),  # GT点被覆盖的比例
        'complete_ratio_std': float(comp_ratio_std),
        'complete': float(comp),  # 平均完整性距离
        'complete_std': float(comp_std),
        'accuracy': float(acc),  # 平均准确性距离  
        'accuracy_std': float(acc_std),
        'threshold': float(threshold),
        'num_gt_points': int(pts_a.shape[0]),
        'num_pred_points': int(pts_b.shape[0])
    }

def get_distance_metrics(gt_mesh: trimesh.Trimesh, reconstructed_mesh: trimesh.Trimesh,
                         n_samples=200_000, threshold=0.2):
    """
    Returns (chamfer_distance, hausdorff_distance)
    - By default, crop reconstructed mesh using GT's AABB; can also read custom bound from args.
    - n_samples: number of sampling points per mesh (larger = more stable, slower)
    """
    # 1) Calculate bounds
    aabb_min = np.min(gt_mesh.vertices, axis=0)-0.01
    aabb_max = np.max(gt_mesh.vertices, axis=0)+0.01

    # 2) Crop reconstructed mesh
    recon_cropped = crop_mesh_to_aabb(reconstructed_mesh, aabb_min, aabb_max)
    if args.save_mesh:
        recon_cropped_path = os.path.join(args.log_dir, args.exp_name, "mesh", "recon_cropped.ply")
        recon_cropped.export(recon_cropped_path)
        print(f"Reconstructed mesh cropped and saved to: {recon_cropped_path}")
    # 3) Sample points
    pts_gt = sample_points_on_mesh(gt_mesh, n_points=n_samples)
    pts_rec = sample_points_on_mesh(recon_cropped, n_points=n_samples)

    # 4) Calculate metrics
    metrics = mesh_metrics(pts_gt, pts_rec, threshold=threshold)
    return metrics

def calculate_gt_gradients(points, gt_mesh, h1=0.04):
    grad = np.zeros_like(points, dtype=np.float32) 
    vertices = gt_mesh.vertices.astype(np.float32) 
    faces = gt_mesh.faces.astype(np.int32)
    f = pysdf.SDF(vertices, faces)
    for i in range(3):
        offset = np.zeros_like(points, dtype=np.float32)
        offset[:, i] = h1
        a = f(points+offset)
        b = f(points-offset)
        # clip to avoid numerical instability
        a = np.clip(a, -2.0, 2.0)
        b = np.clip(b, -2.0, 2.0)
        grad[:, i] = (a-b)/(2*h1)
    return grad
            
def get_sdf_gradient_metrics(args, gt_mesh, gt_sdf_positive_mask, ckpt_path):
    bound = args.decoder_specs['bound']
    offset = args.mapper_specs['offset']
    res = args.mapper_specs['mesh_res']
    
    points, (x_res, y_res, z_res) = get_points(bound, res, offset)
    points = points.reshape(x_res, y_res, z_res, 3)
    points = points[gt_sdf_positive_mask]
    points = points[::100]
    mapper, decoder = load_checkpoint(ckpt_path, args)
    from src.functions.render_helpers import finite_diff_grad_combined_safe
    batch_size = 50000
    grad_pred = []
    for i in range(0, points.shape[0], batch_size):
        batch_points = points[i:i+batch_size]
        batch_points= batch_points + args.mapper_specs['offset']
        batch_grad = finite_diff_grad_combined_safe(torch.from_numpy(batch_points).cuda(), mapper.map_states, mapper.voxel_size, mapper.decoder, h1=args.criteria['h1'])
        grad_pred.append(batch_grad.cpu().numpy())
    grad_pred = np.concatenate(grad_pred, axis=0)
    grad_gt = calculate_gt_gradients(points, gt_mesh, h1=args.criteria['h1'])
    grad_pred_normalized = grad_pred / (np.linalg.norm(grad_pred, axis=1, keepdims=True)+1e-8)
    grad_gt_normalized = grad_gt / (np.linalg.norm(grad_gt, axis=1, keepdims=True)+1e-8)
    cosine = np.sum(grad_pred_normalized * grad_gt_normalized, axis=1)
    cosine = np.clip(cosine, -1, 1)
    angle = np.arccos(cosine)
    grad_angle_diff = np.abs(angle).mean()
    return grad_angle_diff


def evaluate(args):
    ckpt_path = os.path.join(args.log_dir, args.exp_name, "ckpt", "final_ckpt.pth")
    gt_mesh_path = args.data_specs['data_path']+'_mesh.ply'  # './Datasets/Replica/room0'
    gt_mesh = trimesh.load(gt_mesh_path)
    
    if args.calculate_sdf_loss:
        # Load and extract mesh
        reconstructed_mesh, sdf_u, prior_u = load_and_extract_mesh(ckpt_path, args, mesh_res=256, output_dir=None)

        # Calculate SDF loss
        loss_dict, gt_sdf_positive_mask = get_sdf_loss(args, sdf_u, prior_u, gt_mesh)
        print("SDF loss:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value:.6f}")
    else:
        reconstructed_mesh = trimesh.load(os.path.join(args.log_dir, args.exp_name, "mesh", "final_mesh.ply"))

    # Calculate mesh metrics
    mesh_metrics_dict = get_distance_metrics(gt_mesh, reconstructed_mesh, threshold=0.05)
    # print in cm and percentage
    print(f"Chamfer distance: {mesh_metrics_dict['chamfer_distance']*100:.2f} cm, F1 score: {mesh_metrics_dict['f1_score']*100:.2f}%")
    print(f"Precision: {mesh_metrics_dict['precision']*100:.2f}%, Recall: {mesh_metrics_dict['recall']*100:.2f}%")
    print(f"Complete ratio: {mesh_metrics_dict['complete_ratio']*100:.2f} ± {mesh_metrics_dict['complete_ratio_std']*100:.2f}")
    print(f"Complete: {mesh_metrics_dict['complete']*100:.2f} ± {mesh_metrics_dict['complete_std']*100:.2f}")
    print(f"Accuracy: {mesh_metrics_dict['accuracy']*100:.2f} ± {mesh_metrics_dict['accuracy_std']*100:.2f}")
    print(f"GT points: {mesh_metrics_dict['num_gt_points']}, Pred points: {mesh_metrics_dict['num_pred_points']}")


    grad_angle_diff = get_sdf_gradient_metrics(args, gt_mesh, gt_sdf_positive_mask, ckpt_path)
    print(f"SDF gradient angle diff: {grad_angle_diff}")

# Usage examples
if __name__ == "__main__":
    args = get_parser().parse_args()
    print(args)
    
    evaluate(args)