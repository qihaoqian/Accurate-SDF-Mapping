import numpy as np
import pysdf
import trimesh
import matplotlib.pyplot as plt
import os

def save_sdf_slice(u, bound, save_dir=None, title=None):
    """
    保存SDF的切片图像用于可视化调试
    """
    if save_dir is None:
        save_dir = "./debug_slices"
    os.makedirs(save_dir, exist_ok=True)

    # 获取u的形状和边界
    x_res, y_res, z_res = u.shape
    x_min, x_max = bound[0]
    y_min, y_max = bound[1]
    z_min, z_max = bound[2]

    # 转换为numpy
    if hasattr(u, 'cpu'):
        u_np = u.cpu().numpy()
    else:
        u_np = np.array(u)

    # 创建坐标网格
    x_coords = np.linspace(x_min, x_max, x_res)
    y_coords = np.linspace(y_min, y_max, y_res)
    z_coords = np.linspace(z_min, z_max, z_res)

    # 定义切片配置
    slice_configs = [
        {
            'name': 'x_slice',
            'axis': 0,  # x轴方向
            'slice_idx': x_res // 2,  # 中间切片
            'coord_names': ('Y', 'Z'),
            'title': f'{title} X-Slice (X={x_coords[x_res//2]:.2f})',
            'filename': f'{title}_slice_x.png',
            'coords': (y_coords, z_coords)
        },
        {
            'name': 'y_slice',
            'axis': 1,  # y轴方向
            'slice_idx': y_res // 2,  # 中间切片
            'coord_names': ('X', 'Z'),
            'title': f'{title} Y-Slice (Y={y_coords[y_res//2]:.2f})',
            'filename': f'{title}_slice_y.png',
            'coords': (x_coords, z_coords)
        },
        {
            'name': 'z_slice',
            'axis': 2,  # z轴方向
            'slice_idx': z_res // 2,  # 中间切片
            'coord_names': ('X', 'Y'),
            'title': f'{title} Z-Slice (Z={z_coords[z_res//2]:.2f})',
            'filename': f'{title}_slice_z.png',
            'coords': (x_coords, y_coords)
        }
    ]

    # 为每个方向生成切片
    for config in slice_configs:
        axis = config['axis']
        slice_idx = config['slice_idx']
        coord_names = config['coord_names']
        title = config['title']
        filename = config['filename']
        coord1_vals, coord2_vals = config['coords']
        
        # 提取切片数据
        if axis == 0:  # x切片，固定x，变化y,z
            sdf_slice = u_np[slice_idx, :, :]
        elif axis == 1:  # y切片，固定y，变化x,z
            sdf_slice = u_np[:, slice_idx, :]
        else:  # z切片，固定z，变化x,y
            sdf_slice = u_np[:, :, slice_idx]
        
        # 创建坐标网格
        coord1_grid, coord2_grid = np.meshgrid(coord1_vals, coord2_vals, indexing='ij')
        
        # 绘制图像
        plt.figure(figsize=(12, 10))
        
        # 使用imshow显示SDF切片
        im = plt.imshow(
            sdf_slice.T,  # 转置以正确显示方向
            extent=[coord1_vals[0], coord1_vals[-1], coord2_vals[0], coord2_vals[-1]],
            origin='lower',
            cmap='RdBu_r',  # 红色为正值，蓝色为负值
            aspect='equal'
        )
        
        # 添加等高线
        levels = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        cs = plt.contour(coord1_grid, coord2_grid, sdf_slice, 
                       levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        # 特别标出零等高线（表面）
        zero_contour = plt.contour(coord1_grid, coord2_grid, sdf_slice, 
                                 levels=[0], colors='red', linewidths=2)
        
        plt.colorbar(im, label='SDF Value', shrink=0.8)
        plt.xlabel(f'{coord_names[0]} (m)', fontsize=12)
        plt.ylabel(f'{coord_names[1]} (m)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加数据范围信息到副标题
        plt.suptitle(
            f'SDF Range: [{sdf_slice.min():.3f}, {sdf_slice.max():.3f}] | '
            f'Grid: {sdf_slice.shape[0]}×{sdf_slice.shape[1]}',
            fontsize=10, y=0.95
        )

        # 保存图像
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"SDF {config['name']} saved to: {save_path}")
        print(f"  Slice index: {slice_idx} (axis {axis})")
        print(f"  SDF range: [{sdf_slice.min():.3f}, {sdf_slice.max():.3f}]")
        print(f"  Grid size: {sdf_slice.shape[0]}×{sdf_slice.shape[1]}")

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

def get_sdf(points, mesh_path, bound, res):
    """
    Calculate Signed Distance Field (SDF) values from points to mesh using pysdf
    
    Args:
        points: numpy array with shape (N, 3), representing N 3D points
        mesh_path: mesh file path (.ply or .obj format)
    
    Returns:
        sdf_values: numpy array with shape (N,), SDF value for each point
                   negative values indicate points inside mesh, positive values outside
    """
    # Load mesh file using trimesh
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path)
    
    # Ensure correct data types
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)
    
    print(f"Mesh loaded: {len(vertices)} vertices, {len(faces)} faces")
    
    # Create pysdf SDF function
    f = pysdf.SDF(vertices, faces)
    
    # Calculate SDF values
    print(f"Computing SDF for {len(points)} points using pysdf...")
    # pysdf expects points as float32 type
    points_f32 = points.astype(np.float32)
    
    # Batch compute SDF values
    sdf_values = f(points_f32)
    
    print(f"SDF computation completed. Range: [{sdf_values.min():.4f}, {sdf_values.max():.4f}]")
    
    # Filter outlier SDF values
    threshold = 2.0
    outliers_mask = np.abs(sdf_values) > threshold
    num_outliers = np.sum(outliers_mask)
    
    if num_outliers > 0:
        print(f"Found {num_outliers} outlier SDF values (|sdf| > {threshold})")
        print(f"Outliers range: [{sdf_values[outliers_mask].min():.4f}, {sdf_values[outliers_mask].max():.4f}]")
        
        # Clip outlier values to threshold range
        sdf_values = np.clip(sdf_values, -threshold, threshold)
        print(f"Clipped SDF values to range: [{sdf_values.min():.4f}, {sdf_values.max():.4f}]")
    else:
        print(f"No outlier SDF values found (all |sdf| <= {threshold})")
    
    # find near surface points
    near_surface_mask = (sdf_values >= 0) & (sdf_values < 0.1)
    print(f"Number of near surface points: {np.sum(near_surface_mask)} / {len(near_surface_mask)} ({np.sum(near_surface_mask)/len(near_surface_mask)*100:.2f}%)")
    # save_sdf_slice(sdf_values, bound, save_dir="../Datasets/Replica/room0", title="SDF")
    
    return sdf_values, outliers_mask, near_surface_mask

def get_loss_sdf(pred_sdf_path, gt_sdf_path, invalid_mask_path, positive_mask_path, near_surface_mask_path):
    invalid_mask = np.load(invalid_mask_path)
    positive_mask = np.load(positive_mask_path)
    near_surface_mask = np.load(near_surface_mask_path)
    loss_mask = ~invalid_mask & positive_mask
    
    print(f"Number of True values in loss_mask: {np.sum(loss_mask)} / {loss_mask.size} ({np.sum(loss_mask)/loss_mask.size*100:.2f}%)")
    pred_sdf = np.load(pred_sdf_path)
    gt_sdf = np.load(gt_sdf_path)
    # 定义边界，这里需要根据实际情况设置
    bound = np.array([[ 7.1,18.9 ],[ 6.8,15.5 ],[ 6.5,13.3 ]])
    save_sdf_slice(gt_sdf, bound, save_dir="../Datasets/Replica/room0", title="GT_SDF")

    near_surface_diff = np.abs(pred_sdf[near_surface_mask] - gt_sdf[near_surface_mask])
    space_diff_mask = ~near_surface_mask & ~invalid_mask & positive_mask
    space_diff = np.abs(pred_sdf[space_diff_mask] - gt_sdf[space_diff_mask])

    diff = pred_sdf[loss_mask] - gt_sdf[loss_mask]
    # 构建一个(resx, resy, resz)的diff矩阵，只有被loss_mask过滤后的值，其余为0
    resx, resy, resz = gt_sdf.shape
    diff_matrix = np.zeros((resx, resy, resz), dtype=gt_sdf.dtype)
    # loss_mask的形状应与gt_sdf一致
    diff_matrix[loss_mask] = diff
    save_sdf_slice(diff_matrix, bound, save_dir="../Datasets/Replica/room0", title="Diff SDF")
    
    show_sdf_distribution(np.abs(diff).reshape(-1), title="Diff SDF Distribution")
    show_sdf_distribution(pred_sdf.reshape(-1), title="Pred SDF Distribution")
    show_sdf_distribution(gt_sdf.reshape(-1), title="GT SDF Distribution")
    show_sdf_distribution(near_surface_diff.reshape(-1), title="Near Surface Diff SDF Distribution")
    show_sdf_distribution(space_diff.reshape(-1), title="Space Diff SDF Distribution")
    loss = np.mean(np.abs(pred_sdf - gt_sdf))
    near_surface_loss = np.mean(np.abs(near_surface_diff))
    space_loss = np.mean(np.abs(space_diff))
    return loss, near_surface_loss, space_loss

def show_sdf_distribution(sdf_values, title="SDF Distribution", save_path=None, show_stats=True):
    """
    Display the distribution of SDF values
    
    Args:
        sdf_values: SDF value array
        title: Chart title
        save_path: Save path, if None then don't save
        show_stats: Whether to display statistics
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(12, 8))
    
    # Calculate statistics
    mean_val = np.mean(sdf_values)
    std_val = np.std(sdf_values)
    median_val = np.median(sdf_values)
    min_val = np.min(sdf_values)
    max_val = np.max(sdf_values)
    
    # Create subplots
    plt.subplot(2, 2, 1)
    # Draw histogram
    n, bins, patches = plt.hist(sdf_values, bins=100, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    plt.xlabel('SDF Value')
    plt.ylabel('Density')
    plt.title(f'{title} - Histogram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Draw box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(sdf_values, vert=True)
    plt.ylabel('SDF Value')
    plt.title(f'{title} - Box Plot')
    plt.grid(True, alpha=0.3)
    
    # Draw cumulative distribution function
    plt.subplot(2, 2, 3)
    sorted_data = np.sort(sdf_values)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cumulative, linewidth=2, color='purple')
    plt.xlabel('SDF Value')
    plt.ylabel('Cumulative Probability')
    plt.title(f'{title} - Cumulative Distribution Function')
    plt.grid(True, alpha=0.3)
    
    # Draw positive/negative value distribution
    plt.subplot(2, 2, 4)
    positive_count = np.sum(sdf_values > 0)
    negative_count = np.sum(sdf_values < 0)
    zero_count = np.sum(sdf_values == 0)
    
    labels = ['Positive', 'Negative', 'Zero']
    sizes = [positive_count, negative_count, zero_count]
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    # Only display non-zero parts
    non_zero_labels = []
    non_zero_sizes = []
    non_zero_colors = []
    for i, size in enumerate(sizes):
        if size > 0:
            non_zero_labels.append(f'{labels[i]}\n({size} pts)')
            non_zero_sizes.append(size)
            non_zero_colors.append(colors[i])
    
    if non_zero_sizes:
        plt.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'{title} - Positive/Negative Distribution')
    
    plt.tight_layout()
    
    # Display statistics
    if show_stats:
        stats_text = f"""
Statistics:
Total count: {len(sdf_values)}
Mean: {mean_val:.6f}
Standard deviation: {std_val:.6f}
Median: {median_val:.6f}
Minimum: {min_val:.6f}
Maximum: {max_val:.6f}
Positive count: {positive_count} ({positive_count/len(sdf_values)*100:.1f}%)
Negative count: {negative_count} ({negative_count/len(sdf_values)*100:.1f}%)
Zero count: {zero_count} ({zero_count/len(sdf_values)*100:.1f}%)
        """
        print(stats_text)
    
    # Save image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # # Set parameters
    # bound = np.array([[ 7.1,18.9 ],[ 6.8,15.5 ],[ 6.5,13.3 ]])
    # resolution = 256
    # offset = 10
    # mesh_path = "/home/qihao/Downloads/room.ply"
    
    # # Generate grid points
    # print("Generating grid points...")
    # points, res = get_points(bound, resolution, offset)
    # print(f"Generated {points.shape[0]} points with resolution {res}")
    
    # # Calculate SDF values
    # sdf_values, invalid_mask, near_surface_mask = get_sdf(points, mesh_path)
    # # sdf_values = -sdf_values
    
    # # Reshape back to 3D grid
    # x_res, y_res, z_res = res
    # sdf_grid = sdf_values.reshape(x_res, y_res, z_res)
    # invalid_mask_grid = invalid_mask.reshape(x_res, y_res, z_res)
    # positive_mask_grid = sdf_grid >= 0
    # near_surface_mask_grid = near_surface_mask.reshape(x_res, y_res, z_res)
    # # Save results
    # output_file = "../Datasets/Replica/room0/gt_sdf.npy"
    # invalid_mask_file = "../Datasets/Replica/room0/gt_sdf_invalid_mask.npy"
    # positive_mask_file = "../Datasets/Replica/room0/gt_sdf_positive_mask.npy"
    # near_surface_mask_file = "../Datasets/Replica/room0/gt_sdf_near_surface_mask.npy"
    # np.save(output_file, sdf_grid)
    # np.save(invalid_mask_file, invalid_mask_grid)
    # np.save(positive_mask_file, positive_mask_grid)
    # np.save(near_surface_mask_file, near_surface_mask_grid)
    # print(f"SDF grid saved to {output_file}")
    # print(f"Invalid mask saved to {invalid_mask_file}")
    # print(f"Invalid mask statistics:")
    # print(f"  Total invalid points: {np.sum(invalid_mask)} / {len(invalid_mask)} ({np.sum(invalid_mask)/len(invalid_mask)*100:.2f}%)")
    # print(f"  Invalid points in grid: {np.sum(invalid_mask_grid)} / {invalid_mask_grid.size}")
    
    # # Print statistics
    # print(f"SDF statistics:")
    # print(f"  Shape: {sdf_grid.shape}")
    # print(f"  Min: {sdf_values.min():.4f}")
    # print(f"  Max: {sdf_values.max():.4f}")
    # print(f"  Mean: {sdf_values.mean():.4f}")
    # print(f"  Std: {sdf_values.std():.4f}")
    # print(f"  Points inside mesh (sdf < 0): {np.sum(sdf_values < 0)}")
    # print(f"  Points outside mesh (sdf > 0): {np.sum(sdf_values > 0)}")
    # print(f"  Points on surface (|sdf| < 1e-6): {np.sum(np.abs(sdf_values) < 1e-6)}")
    # print(f"  Points at clipping boundary (|sdf| = 30): {np.sum(np.abs(sdf_values) >= 29.99)}")
    
    # # Display SDF value distribution
    # print(f"SDF value distribution:")
    # percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    # sdf_percentiles = np.percentile(sdf_values, percentiles)
    # for p, val in zip(percentiles, sdf_percentiles):
    #     print(f"  {p:2d}th percentile: {val:.4f}")

    gt_sdf_path = "../Datasets/Replica/room0/gt_sdf.npy"
    invalid_mask_path = "../Datasets/Replica/room0/gt_sdf_invalid_mask.npy"
    positive_mask_path = "../Datasets/Replica/room0/gt_sdf_positive_mask.npy"
    pred_sdf_path = "../mapping/logs/replica/room0/split_hash/misc/sdf_priors.npy"
    near_surface_mask_path = "../Datasets/Replica/room0/gt_sdf_near_surface_mask.npy"
    sdf_loss, near_surface_loss, space_loss = get_loss_sdf(pred_sdf_path, gt_sdf_path, invalid_mask_path, positive_mask_path, near_surface_mask_path)
    print(f"SDF Loss: {sdf_loss}")
    print(f"Near Surface Loss: {near_surface_loss}")
    print(f"Space Loss: {space_loss}")