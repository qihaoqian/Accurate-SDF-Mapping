import open3d as o3d
import numpy as np
import os

# 加载 mesh
mesh_path_list = [
    # "Datasets/Replica/office0_mesh.ply",
    "Datasets/Replica/office1_mesh.ply",
    "Datasets/Replica/office2_mesh.ply",
    "Datasets/Replica/office3_mesh.ply",
    "Datasets/Replica/office4_mesh.ply",
    "Datasets/Replica/room0_mesh.ply",
    "Datasets/Replica/room1_mesh.ply",
    "Datasets/Replica/room2_mesh.ply",
]

for mesh_path in mesh_path_list:
    print(mesh_path)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # 计算 OBB
    obb = mesh.get_minimal_oriented_bounding_box()

    pt8 = np.asarray(obb.get_box_points())

    print("中心:", obb.center)
    print("范围(长宽高):", obb.extent)
    print("旋转矩阵:\n", obb.R)
    print("8个顶点:\n", pt8)

    # 将mesh转换到OBB坐标系
    mesh_rotated = o3d.geometry.TriangleMesh(mesh)
    
    # 步骤1: 将mesh平移到OBB中心的坐标系原点
    mesh_rotated.translate(-obb.center)
    
    # 步骤2: 应用OBB旋转矩阵的逆矩阵来对齐坐标轴
    # OBB的旋转矩阵R将OBB的局部坐标转换为世界坐标
    # 所以我们需要R的逆矩阵(即转置，因为旋转矩阵是正交矩阵)来进行反向转换
    R_inv = obb.R.T
    mesh_rotated.rotate(R_inv, center=(0, 0, 0))
    
    print("变换后的mesh边界框:")
    aabb_transformed = mesh_rotated.get_axis_aligned_bounding_box()
    print("变换后AABB最小值:", aabb_transformed.min_bound)
    print("变换后AABB最大值:", aabb_transformed.max_bound)
    print("变换后AABB范围:", aabb_transformed.max_bound - aabb_transformed.min_bound)
    print("OBB原始范围:", obb.extent)
    offset = np.abs(aabb_transformed.min_bound.min()) + 0.15
    print("offset:", offset)
    bound_min = aabb_transformed.min_bound + offset - 0.15
    bound_max = aabb_transformed.max_bound + offset + 0.15
    bound = [[round(float(mn), 2), round(float(mx), 2)]
         for mn, mx in zip(bound_min, bound_max)]
    print("bound:", bound)
    
    # 先将原始mesh重命名为_original
    original_mesh_path = mesh_path.replace(".ply", "_original.ply")
    os.rename(mesh_path, original_mesh_path)
    print(f"原始mesh已重命名为: {original_mesh_path}")
    
    # 保存旋转后的mesh为原始文件名
    success = o3d.io.write_triangle_mesh(mesh_path, mesh_rotated)
    if success:
        print(f"旋转后的mesh已保存到: {mesh_path}")
    else:
        print(f"保存失败: {mesh_path}")
    
    # 处理相应的相机外参文件
    traj_path = mesh_path.replace("_mesh.ply", "/traj.txt")
    if os.path.exists(traj_path):
        print(f"处理相机外参文件: {traj_path}")
        
        # 读取相机外参矩阵
        camera_poses = []
        with open(traj_path, 'r') as f:
            lines = f.readlines()
            
        # 解析每行16个值的4x4矩阵
        for line in lines:
            if line.strip():  # 跳过空行
                values = list(map(float, line.strip().split()))
                if len(values) == 16:
                    # 将16个值重塑为4x4矩阵
                    matrix = np.array(values).reshape(4, 4)
                    camera_poses.append(matrix)
        
        print(f"读取到 {len(camera_poses)} 个相机位姿")
        
        # 对每个相机外参应用相同的变换
        transformed_poses = []
        for pose in camera_poses:
            # 提取旋转矩阵和平移向量
            R_cam = pose[:3, :3]
            t_cam = pose[:3, 3]
            
            # 应用变换：先平移再旋转
            # 1. 平移到OBB中心
            t_cam_translated = t_cam - obb.center
            
            # 2. 应用旋转变换
            R_cam_rotated = R_inv @ R_cam
            t_cam_rotated = R_inv @ t_cam_translated
            
            # 构建新的4x4矩阵
            new_pose = np.eye(4)
            new_pose[:3, :3] = R_cam_rotated
            new_pose[:3, 3] = t_cam_rotated
            
            transformed_poses.append(new_pose)
        
        # 先将原始traj文件重命名为_original
        original_traj_path = traj_path.replace(".txt", "_original.txt")
        os.rename(traj_path, original_traj_path)
        print(f"原始相机外参已重命名为: {original_traj_path}")
        
        # 保存变换后的相机外参为原始文件名
        with open(traj_path, 'w') as f:
            for pose in transformed_poses:
                # 将4x4矩阵展平为16个值的一行
                flattened = pose.flatten()
                f.write(' '.join(map(str, flattened)) + '\n')
        
        print(f"变换后的相机外参已保存到: {traj_path}")
    else:
        print(f"未找到相机外参文件: {traj_path}")
    
    # 创建对应的变换后的OBB用于可视化对比
    obb_transformed = o3d.geometry.OrientedBoundingBox(obb)
    obb_transformed.translate(-obb.center)
    obb_transformed.rotate(R_inv, center=(0, 0, 0))
    
    obb.color = [1, 0, 0]  # 原始OBB为红色
    obb_transformed.color = [0, 1, 0]  # 变换后OBB为绿色
    mesh_rotated.paint_uniform_color([0, 0, 1])  # 变换后mesh为蓝色

    # 创建坐标轴
    # 世界坐标系坐标轴（较大）
    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    # OBB中心的坐标轴
    obb_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    obb_axis.translate(obb.center)
    obb_axis.rotate(obb.R, center=obb.center)
    
    # 变换后坐标系的坐标轴（在原点）
    transformed_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8, origin=[0, 0, 0])

    # 可视化原始mesh和OBB
    print("显示原始mesh(灰色)、OBB(红色)、世界坐标轴(RGB)、OBB坐标轴...")
    o3d.visualization.draw_geometries([mesh, obb, world_axis, obb_axis])
    
    # 可视化变换后的mesh和OBB
    print("显示变换后的mesh(蓝色)、OBB(绿色)、变换后坐标轴(RGB)...")
    o3d.visualization.draw_geometries([mesh_rotated, obb_transformed, transformed_axis])
