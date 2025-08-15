import numpy as np
import matplotlib.pyplot as plt

# 定义圆心和半径
center = (1, 1)
radius = 1

# 定义正方形的四个顶点
square_points = np.array([
    [3, 2],
    [6, 2],
    [3, 5],
    [6, 5]
])

vertex_features = []
# 计算每个顶点到圆的距离
for vertex in square_points:
    dist = np.linalg.norm(vertex - np.array(center)) - radius
    vertex_features.append(dist)
vertex_directions = []
for vertex in square_points:
    direction = (vertex - np.array(center)) / np.linalg.norm(vertex - np.array(center))
    vertex_directions.append(direction)

def sample_points(square_points):
    min_x, min_y = np.min(square_points, axis=0)
    max_x, max_y = np.max(square_points, axis=0)
    xs = np.linspace(min_x, max_x, 64)
    ys = np.linspace(min_y, max_y, 64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x, grid_y], axis=-1)

def extra_interpolation(points, square_points, vertex_features, vertex_directions):
    extra_interpolated_sdf = []
    for i in range(4):
        # 计算每个采样点与顶点的距离
        offsets = points - square_points[i]
        # 使用内积计算在方向上的投影（保留符号）
        lengths = np.sum(offsets * vertex_directions[i], axis=-1)
        # 利用广播，把 scalar + (64,64) 变成 (64,64)，再累加并均分
        extra_interpolated_sdf.append(vertex_features[i] + lengths)
    return extra_interpolated_sdf

def linear_interpolation(points, extra_interpolated_sdf, square_points):
    # 动态计算矩形尺寸
    min_corner = np.min(square_points, axis=0)
    max_corner = np.max(square_points, axis=0)
    size = max_corner - min_corner
    offsets = (points - min_corner) / size
    u = offsets[:, :, 0]
    v = offsets[:, :, 1]
    f00 = extra_interpolated_sdf[0]
    f10 = extra_interpolated_sdf[1]
    f01 = extra_interpolated_sdf[2]
    f11 = extra_interpolated_sdf[3]
    # 沿 x 方向插值
    f0 = (1 - u) * f00 + u * f10
    f1 = (1 - u) * f01 + u * f11
    
    # 沿 y 方向插值
    return (1 - v) * f0 + v * f1

def calculate_sdf(points, center, radius):
    return np.linalg.norm(points - center, axis=-1) - radius

if __name__ == "__main__":
    points = sample_points(square_points)
    gt_sdf = calculate_sdf(points, center, radius)
    # 测试修复后的函数
    extra_interpolated_sdf = extra_interpolation(points, square_points, vertex_features, vertex_directions)
    extra_interpolated_sdf_result = np.mean(np.stack(extra_interpolated_sdf, axis=-1), axis=-1)
    linear_interpolated_sdf_result = linear_interpolation(points, extra_interpolated_sdf, square_points)
    error_extra_interpolated = extra_interpolated_sdf_result - gt_sdf
    error_linear_interpolated = linear_interpolated_sdf_result - gt_sdf
    loss_extra_interpolated = np.mean(np.abs(error_extra_interpolated))
    loss_linear_interpolated = np.mean(np.abs(error_linear_interpolated))
    print(f"loss_extra_interpolated: {loss_extra_interpolated}")
    print(f"loss_linear_interpolated: {loss_linear_interpolated}")
    
    # 可视化结果
    plt.figure(figsize=(20, 8))
    
    # 第一个子图：Extra Interpolation 结果
    plt.subplot(1, 2, 1)
    
    # 动态计算边界
    min_x, min_y = np.min(square_points, axis=0)
    max_x, max_y = np.max(square_points, axis=0)
    
    # 显示插值结果（只在正方形区域内）
    plt.imshow(extra_interpolated_sdf_result, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap='viridis', alpha=0.7)
    plt.colorbar(label='Extra Interpolated SDF')
    
    # 绘制圆
    circle1 = plt.Circle(center, radius, fill=False, color='blue', linewidth=2, label='Circle')
    plt.gca().add_patch(circle1)
    
    # 绘制正方形边界
    square_x = [min_x, max_x, max_x, min_x, min_x]  # 闭合正方形
    square_y = [min_y, min_y, max_y, max_y, min_y]
    plt.plot(square_x, square_y, 'r-', linewidth=2, label='Square Boundary')
    
    # Plot square vertices
    plt.scatter(square_points[:, 0], square_points[:, 1], c='red', s=100, marker='o', 
                edgecolors='black', linewidth=1, label='Square Vertices', zorder=5)
    
    # Plot circle center
    plt.scatter(center[0], center[1], c='blue', s=100, marker='x', 
                linewidth=3, label='Circle Center', zorder=5)
    
    # Set axis limits to show all elements
    plt.xlim(0, 6)
    plt.ylim(0, 5)
    
    # Set grid and axis labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(f'Extra Interpolation (Loss: {loss_extra_interpolated:.4f})', fontsize=14)
    
    # 确保坐标轴比例相等
    plt.axis('equal')
    
    # 添加图例
    plt.legend(loc='upper left')
    
    # 添加坐标轴标注
    plt.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    # 第二个子图：Linear Interpolation 结果
    plt.subplot(1, 2, 2)
    
    # 显示插值结果（只在正方形区域内）
    plt.imshow(linear_interpolated_sdf_result, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap='viridis', alpha=0.7)
    plt.colorbar(label='Linear Interpolated SDF')
    
    # 绘制圆
    circle2 = plt.Circle(center, radius, fill=False, color='blue', linewidth=2, label='Circle')
    plt.gca().add_patch(circle2)
    
    # 绘制正方形边界
    plt.plot(square_x, square_y, 'r-', linewidth=2, label='Square Boundary')
    
    # Plot square vertices
    plt.scatter(square_points[:, 0], square_points[:, 1], c='red', s=100, marker='o', 
                edgecolors='black', linewidth=1, label='Square Vertices', zorder=5)
    
    # Plot circle center
    plt.scatter(center[0], center[1], c='blue', s=100, marker='x', 
                linewidth=3, label='Circle Center', zorder=5)
    
    # Set axis limits to show all elements
    plt.xlim(0, 6)
    plt.ylim(0, 5)
    
    # Set grid and axis labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(f'Linear Interpolation (Loss: {loss_linear_interpolated:.4f})', fontsize=14)
    
    # 确保坐标轴比例相等
    plt.axis('equal')
    
    # 添加图例
    plt.legend(loc='upper left')
    
    # 添加坐标轴标注
    plt.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.show()  # 显示插值结果
    
    # 创建误差可视化图表
    plt.figure(figsize=(20, 8))
    
    # 第一个子图：Extra Interpolation 误差
    plt.subplot(1, 2, 1)
    
    im1 = plt.imshow(error_extra_interpolated, extent=[min_x, max_x, min_y, max_y], 
                     origin='lower', cmap='RdBu_r', alpha=0.8)
    plt.colorbar(im1, label='Error (Predicted - Ground Truth)')
    
    # 绘制圆
    circle_err1 = plt.Circle(center, radius, fill=False, color='black', linewidth=2, label='Circle')
    plt.gca().add_patch(circle_err1)
    
    # 绘制正方形边界
    plt.plot(square_x, square_y, 'k-', linewidth=2, label='Square Boundary')
    
    # Plot square vertices
    plt.scatter(square_points[:, 0], square_points[:, 1], c='black', s=100, marker='o', 
                edgecolors='white', linewidth=1, label='Square Vertices', zorder=5)
    
    # Plot circle center
    plt.scatter(center[0], center[1], c='black', s=100, marker='x', 
                linewidth=3, label='Circle Center', zorder=5)
    
    # Set axis limits to show all elements
    plt.xlim(0, 7)
    plt.ylim(0, 6)
    
    # Set grid and axis labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(f'Extra Interpolation Error (MAE: {loss_extra_interpolated:.4f})', fontsize=14)
    
    # 确保坐标轴比例相等
    plt.axis('equal')
    
    # 添加图例
    plt.legend(loc='upper left')
    
    # 第二个子图：Linear Interpolation 误差
    plt.subplot(1, 2, 2)
    
    im2 = plt.imshow(error_linear_interpolated, extent=[min_x, max_x, min_y, max_y], 
                     origin='lower', cmap='RdBu_r', alpha=0.8)
    plt.colorbar(im2, label='Error (Predicted - Ground Truth)')
    
    # 绘制圆
    circle_err2 = plt.Circle(center, radius, fill=False, color='black', linewidth=2, label='Circle')
    plt.gca().add_patch(circle_err2)
    
    # 绘制正方形边界
    plt.plot(square_x, square_y, 'k-', linewidth=2, label='Square Boundary')
    
    # Plot square vertices
    plt.scatter(square_points[:, 0], square_points[:, 1], c='black', s=100, marker='o', 
                edgecolors='white', linewidth=1, label='Square Vertices', zorder=5)
    
    # Plot circle center
    plt.scatter(center[0], center[1], c='black', s=100, marker='x', 
                linewidth=3, label='Circle Center', zorder=5)
    
    # Set axis limits to show all elements
    plt.xlim(0, 7)
    plt.ylim(0, 6)
    
    # Set grid and axis labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(f'Linear Interpolation Error (MAE: {loss_linear_interpolated:.4f})', fontsize=14)
    
    # 确保坐标轴比例相等
    plt.axis('equal')
    
    # 添加图例
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    
    print("误差分析完成！")
    print("插值结果和误差图表即将显示，关闭图表窗口以继续程序...")
    plt.show()  # 阻塞显示，直到手动关闭窗口
    print(f"结果矩阵形状: {extra_interpolated_sdf_result.shape}")
    print(f"数值范围: [{extra_interpolated_sdf_result.min():.3f}, {extra_interpolated_sdf_result.max():.3f}]")
    print(f"圆心位置: {center}")
    print(f"圆的半径: {radius}")
    print(f"正方形顶点:")
    for i, vertex in enumerate(square_points):
        print(f"  顶点{i+1}: ({vertex[0]}, {vertex[1]})")


