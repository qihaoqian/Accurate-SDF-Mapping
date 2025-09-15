import json
import os
import shutil

import cv2
import numpy as np
import open3d as o3d


def preprocess_replica(data_path):
    # load ply file
    mesh = o3d.io.read_triangle_mesh(data_path)
    # save as ply file
    o3d.io.write_triangle_mesh(data_path, mesh)


def cam_params(data_path):
    """
    读取相机参数文件
    Args:
        data_path: 相机参数JSON文件的路径
    Returns:
        dict: 包含相机参数的字典
    """
    with open(data_path, "r") as f:
        params = json.load(f)
    return params["camera"]


def render_images(mesh, camera_pose, camera_params, img_width=1200, img_height=680):
    """
    使用Open3D同时渲染深度图像和RGB图像，参考C++实现
    Args:
        mesh: Open3D三角网格
        camera_pose: 4x4相机姿态矩阵
        camera_params: 相机参数字典
        img_width: 图像宽度
        img_height: 图像高度
    Returns:
        tuple: (depth_array, rgb_array) 深度图像和RGB图像
    """
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, visible=False)

    # 添加网格
    vis.add_geometry(mesh)

    # 创建相机参数对象，参考C++代码结构
    camera_parameters = o3d.camera.PinholeCameraParameters()

    # 设置外参 - 参考C++中的ComputeExtrinsic逻辑
    # camera_pose已经是相机到世界的变换矩阵，需要转换为世界到相机
    camera_parameters.extrinsic = np.linalg.inv(camera_pose).astype(np.float64)

    # 设置内参矩阵 - 使用numpy数组而不是列表，参考C++中的intrinsic_matrix_
    intrinsic_matrix = np.array(
        [[camera_params["fx"], 0, camera_params["cx"]], [0, camera_params["fy"], camera_params["cy"]], [0, 0, 1]],
        dtype=np.float64,
    )

    camera_parameters.intrinsic.intrinsic_matrix = intrinsic_matrix
    camera_parameters.intrinsic.height = img_height
    camera_parameters.intrinsic.width = img_width

    # 应用相机参数 - 参考C++中的ConvertFromPinholeCameraParameters
    view_control = vis.get_view_control()
    if not view_control.convert_from_pinhole_camera_parameters(camera_parameters):
        print(f"警告: Open3D无法设置相机参数，窗口尺寸: {img_width}x{img_height}")

    # 渲染
    vis.poll_events()
    vis.update_renderer()

    # 获取图像 - 参考C++中的CaptureScreenFloatBuffer和CaptureDepthFloatBuffer
    # 使用do_render=True确保渲染完成
    rgb_buffer = vis.capture_screen_float_buffer(do_render=True)
    depth_buffer = vis.capture_depth_float_buffer(do_render=True)

    # 转换为numpy数组并执行深拷贝，参考C++中的clone()操作
    rgb_array = np.asarray(rgb_buffer).copy()
    depth_array = np.asarray(depth_buffer).copy()
    print(
        f"depth_array.min: {depth_array.min()}, depth_array.max: {depth_array.max()}, depth_array.mean:"
        f" {depth_array.mean()}"
    )

    # 关闭可视化器
    vis.destroy_window()

    return depth_array, rgb_array


def insert_upward_frames(data_dir, mesh_path, insert_interval=10):
    """
    每隔n帧插入一个向上看的frame，深度图像编号从0开始
    Args:
        data_dir: 数据集目录路径
        mesh_path: 网格文件路径
        insert_interval: 插入间隔，默认每10帧插入一个向上看的frame
    """
    # 加载网格
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # 读取相机参数
    cam_params_path = os.path.join(os.path.dirname(data_dir), "cam_params.json")
    camera_params = cam_params(cam_params_path)

    # 读取原始轨迹
    traj_path = os.path.join(data_dir, "traj.txt")
    original_poses = np.loadtxt(traj_path)

    # 创建向上看的旋转矩阵
    upward_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # 按顺序处理，每10帧插入一个向上看的frame
    new_poses = []
    new_depth_data = []  # [(frame_idx, depth_type, depth_data)]
    new_rgb_data = []  # [(frame_idx, rgb_type, rgb_data)]

    for i, pose_vec in enumerate(original_poses):
        current_new_idx = len(new_poses)

        # 添加原始frame
        new_poses.append(pose_vec)

        # 记录原始深度图像的映射（编号从0开始）
        original_depth_path = os.path.join(data_dir, "results", f"depth{i:06d}.png")
        if not os.path.exists(original_depth_path):
            # 尝试从1开始的编号
            original_depth_path = os.path.join(data_dir, "results", f"depth{i+1:06d}.png")

        if os.path.exists(original_depth_path):
            new_depth_data.append((current_new_idx, "existing", original_depth_path))
        else:
            print(f"警告: 找不到原始深度图像 depth{i:06d}.png 或 depth{i+1:06d}.png")

        # 记录原始RGB图像的映射
        original_rgb_path = os.path.join(data_dir, "results", f"frame{i:06d}.jpg")
        if not os.path.exists(original_rgb_path):
            # 尝试从1开始的编号
            original_rgb_path = os.path.join(data_dir, "results", f"frame{i+1:06d}.jpg")

        if os.path.exists(original_rgb_path):
            new_rgb_data.append((current_new_idx, "existing", original_rgb_path))
        else:
            print(f"警告: 找不到原始RGB图像 frame{i:06d}.jpg 或 frame{i+1:06d}.jpg")

        # 每隔n帧插入向上看的frame
        if (i + 1) % insert_interval == 0:
            # 将pose向量重塑为4x4矩阵
            original_pose = pose_vec.reshape(4, 4)

            # 创建新的向上看pose
            upward_pose = original_pose.copy()
            upward_pose[:3, :3] = upward_rotation  # 替换旋转部分，保持平移不变

            # 添加向上看的pose
            new_poses.append(upward_pose.flatten())

            print(f"正在生成第{i+1}帧后的向上看图像 (每{insert_interval}帧插入)...")
            print(f"向上看pose矩阵:\n{upward_pose}")

            # 同时生成深度图像和RGB图像
            depth_image, rgb_image = render_images(mesh, upward_pose, camera_params)

            # 转换深度值到正确的范围（乘以scale factor）
            depth_image_scaled = (depth_image * camera_params["scale"]).astype(np.uint16)

            # 转换RGB图像格式 (0-1 float -> 0-255 uint8, RGB -> BGR for OpenCV)
            rgb_image_scaled = (rgb_image * 255).astype(np.uint8)
            rgb_image_bgr = cv2.cvtColor(rgb_image_scaled, cv2.COLOR_RGB2BGR)

            # 记录新生成的图像
            upward_frame_idx = len(new_poses) - 1
            new_depth_data.append((upward_frame_idx, "new", depth_image_scaled))
            new_rgb_data.append((upward_frame_idx, "new", rgb_image_bgr))

    # 保存新的轨迹文件
    new_traj_path = os.path.join(data_dir, "traj_with_upward.txt")
    np.savetxt(new_traj_path, np.array(new_poses), fmt="%.16e")
    print(f"已保存新轨迹文件，包含{len(new_poses)}个poses")

    # 备份并重新整理所有图像
    results_dir = os.path.join(data_dir, "results")
    backup_dir = os.path.join(data_dir, "results_backup")

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        # 移动现有图像到备份目录
        for file in os.listdir(results_dir):
            if (file.startswith("depth") and file.endswith(".png")) or (
                file.startswith("frame") and file.endswith(".jpg")
            ):
                src = os.path.join(results_dir, file)
                dst = os.path.join(backup_dir, file)
                os.rename(src, dst)
        print(f"已备份原始图像到: {backup_dir}")

    # 按新的顺序保存所有图像（从0开始编号）
    depth_mapping = {frame_idx: (depth_type, depth_data) for frame_idx, depth_type, depth_data in new_depth_data}
    rgb_mapping = {frame_idx: (rgb_type, rgb_data) for frame_idx, rgb_type, rgb_data in new_rgb_data}

    for frame_idx in range(len(new_poses)):
        # 处理深度图像
        new_depth_filename = f"depth{frame_idx:06d}.png"  # 从0开始编号
        new_depth_path = os.path.join(results_dir, new_depth_filename)

        if frame_idx in depth_mapping:
            depth_type, depth_data = depth_mapping[frame_idx]

            if depth_type == "existing":
                # 复制现有的深度图像
                original_depth_path = depth_data
                filename = os.path.basename(original_depth_path)
                backup_path = os.path.join(backup_dir, filename)

                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, new_depth_path)
                    print(f"已复制现有深度图像: {new_depth_filename}")
                else:
                    print(f"警告: 找不到备份文件 {backup_path}")

            elif depth_type == "new":
                # 保存新生成的深度图像
                cv2.imwrite(new_depth_path, depth_data)
                print(f"已保存新生成深度图像: {new_depth_filename} (向上看frame)")
        else:
            print(f"警告: 第{frame_idx}帧没有对应的深度图像")

        # 处理RGB图像
        new_rgb_filename = f"frame{frame_idx:06d}.jpg"  # 从0开始编号
        new_rgb_path = os.path.join(results_dir, new_rgb_filename)

        if frame_idx in rgb_mapping:
            rgb_type, rgb_data = rgb_mapping[frame_idx]

            if rgb_type == "existing":
                # 复制现有的RGB图像
                original_rgb_path = rgb_data
                filename = os.path.basename(original_rgb_path)
                backup_path = os.path.join(backup_dir, filename)

                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, new_rgb_path)
                    print(f"已复制现有RGB图像: {new_rgb_filename}")
                else:
                    print(f"警告: 找不到备份文件 {backup_path}")

            elif rgb_type == "new":
                # 保存新生成的RGB图像
                cv2.imwrite(new_rgb_path, rgb_data)
                print(f"已保存新生成RGB图像: {new_rgb_filename} (向上看frame)")
        else:
            print(f"警告: 第{frame_idx}帧没有对应的RGB图像")

    # 用新轨迹替换原轨迹
    os.replace(new_traj_path, traj_path)

    print(f"\n=== 处理完成！===")
    print(f"插入间隔: 每{insert_interval}帧插入一个向上看的frame")
    print(f"原始帧数: {len(original_poses)}")
    print(f"新增向上看帧数: {len(new_poses) - len(original_poses)}")
    print(f"总帧数: {len(new_poses)}")
    print(f"深度图像编号: depth000000.png 到 depth{len(new_poses)-1:06d}.png")
    print(f"RGB图像编号: frame000000.jpg 到 frame{len(new_poses)-1:06d}.jpg")
    print(f"备份目录: {backup_dir}")

    # 验证traj.txt中的矩阵
    print(f"\n=== 验证轨迹文件 ===")
    new_traj_data = np.loadtxt(traj_path)
    print(f"轨迹文件中的pose数量: {len(new_traj_data)}")

    # 显示几个向上看的pose示例
    for i, (frame_idx, depth_type, _) in enumerate(new_depth_data):
        if depth_type == "new":
            pose_matrix = new_traj_data[frame_idx].reshape(4, 4)
            print(f"\n向上看frame {frame_idx} 的4x4矩阵:")
            print(pose_matrix)
            if i >= 2:  # 只显示前3个例子
                break


def reorder_depth_images(data_dir, total_frames):
    """
    重新编号深度图像文件以保持顺序
    Args:
        data_dir: 数据集目录路径
        total_frames: 总帧数
    """
    results_dir = os.path.join(data_dir, "results")

    # 获取所有现有的深度图像文件
    depth_files = []
    for file in os.listdir(results_dir):
        if file.startswith("depth") and file.endswith(".png"):
            # 提取文件号
            file_num = int(file[5:11])  # depth后的6位数字
            depth_files.append((file_num, file))

    # 按文件号排序
    depth_files.sort(key=lambda x: x[0])

    # 临时重命名，避免冲突
    temp_names = []
    for i, (old_num, filename) in enumerate(depth_files):
        old_path = os.path.join(results_dir, filename)
        temp_path = os.path.join(results_dir, f"temp_depth_{i:06d}.png")
        os.rename(old_path, temp_path)
        temp_names.append(temp_path)

    # 重新编号为连续的序号
    for i, temp_path in enumerate(temp_names):
        new_filename = f"depth{i+1:06d}.png"
        new_path = os.path.join(results_dir, new_filename)
        os.rename(temp_path, new_path)

    print(f"深度图像文件重新编号完成，共{len(temp_names)}个文件")


def process_all_replica_scenes(base_dir="./Datasets/Replica", interval=10):
    """
    批量处理所有Replica场景
    Args:
        base_dir: Replica数据集根目录
        interval: 插入间隔，每n帧插入一个向上看的frame
    """
    # 定义所有场景名称
    scenes = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]

    # 相机参数文件路径
    cam_params_path = os.path.join(base_dir, "cam_params.json")

    # 检查相机参数文件是否存在
    if not os.path.exists(cam_params_path):
        print(f"错误: 找不到相机参数文件 {cam_params_path}")
        return

    print(f"开始批量处理{len(scenes)}个Replica场景...")
    print(f"插入间隔: 每{interval}帧插入一个向上看的frame")
    print(f"数据集目录: {base_dir}")
    print(f"场景列表: {scenes}")
    print("=" * 60)

    success_count = 0
    failed_scenes = []

    for i, scene in enumerate(scenes, 1):
        print(f"\n[{i}/{len(scenes)}] 正在处理场景: {scene}")
        print("-" * 40)

        # 构建场景相关路径
        data_dir = os.path.join(base_dir, scene)
        mesh_path = os.path.join(base_dir, f"{scene}_mesh.ply")

        # 检查必要文件是否存在
        if not os.path.exists(data_dir):
            print(f"错误: 数据目录不存在 - {data_dir}")
            failed_scenes.append(scene)
            continue

        if not os.path.exists(mesh_path):
            print(f"错误: 网格文件不存在 - {mesh_path}")
            failed_scenes.append(scene)
            continue

        traj_path = os.path.join(data_dir, "traj.txt")
        if not os.path.exists(traj_path):
            print(f"错误: 轨迹文件不存在 - {traj_path}")
            failed_scenes.append(scene)
            continue

        results_dir = os.path.join(data_dir, "results")
        if not os.path.exists(results_dir):
            print(f"错误: results目录不存在 - {results_dir}")
            failed_scenes.append(scene)
            continue

        print(f"数据目录: {data_dir}")
        print(f"网格文件: {mesh_path}")

        try:
            # 处理当前场景
            insert_upward_frames(data_dir=data_dir, mesh_path=mesh_path, insert_interval=interval)
            success_count += 1
            print(f"✓ 场景 {scene} 处理完成")
        except Exception as e:
            print(f"✗ 场景 {scene} 处理失败: {str(e)}")
            failed_scenes.append(scene)
            continue

    # 输出总结
    print("\n" + "=" * 60)
    print(f"批量处理完成！")
    print(f"成功处理: {success_count}/{len(scenes)} 个场景")
    print(f"失败场景: {failed_scenes}")
    if success_count == len(scenes):
        print("🎉 所有场景都处理成功！")
    elif success_count > 0:
        print(f"⚠️  部分场景处理成功，请检查失败的场景")
    else:
        print("❌ 所有场景都处理失败，请检查配置和文件路径")


if __name__ == "__main__":
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Replica数据集预处理：插入向上看的frames")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch"],
        default="single",
        help="处理模式: single(单个场景) 或 batch(批量处理所有场景)",
    )
    parser.add_argument("--data_dir", type=str, default="./Datasets/Replica/room0", help="数据集目录路径 (single模式)")
    parser.add_argument(
        "--mesh_path", type=str, default="./Datasets/Replica/room0_mesh.ply", help="网格文件路径 (single模式)"
    )
    parser.add_argument("--base_dir", type=str, default="./Datasets/Replica", help="Replica数据集根目录 (batch模式)")
    parser.add_argument("--interval", type=int, default=10, help="插入间隔，每n帧插入一个向上看的frame (默认: 10)")
    parser.add_argument("--cam_params", type=str, default="./Datasets/Replica/cam_params.json", help="相机参数文件路径")

    args = parser.parse_args()

    if args.mode == "batch":
        # 批量处理模式
        print("=== 批量处理模式 ===")
        process_all_replica_scenes(base_dir=args.base_dir, interval=args.interval)
    else:
        # 单个场景处理模式
        print("=== 单个场景处理模式 ===")

        # 读取相机参数
        camera_params = cam_params(args.cam_params)
        print("相机参数:")
        for key, value in camera_params.items():
            print(f"  {key}: {value}")

        # 插入向上看的frames
        print(f"\n开始处理Replica数据集，每{args.interval}帧插入一个向上看的frame...")
        print(f"数据目录: {args.data_dir}")
        print(f"网格文件: {args.mesh_path}")

        insert_upward_frames(data_dir=args.data_dir, mesh_path=args.mesh_path, insert_interval=args.interval)
