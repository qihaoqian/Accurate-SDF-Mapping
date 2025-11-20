#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NewerColleage To Replica
"""
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

def reorganize_depth_images():
    """
    重新组织深度图像：
    1. 从cam0, cam1, cam2, cam3文件夹中依次取图片
    2. 重新编号为0000.png, 0001.png, 0002.png...
    3. 保存到新的文件夹中
    """
    
    # 定义路径
    base_path = Path("Datasets/NewerCollege4Cam/depth")
    output_path = Path("Datasets/NewerCollege/quat/results")
    
    # 创建输出文件夹
    output_path.mkdir(exist_ok=True)
    print(f"创建输出文件夹: {output_path}")
    
    # 定义相机文件夹
    cam_folders = ["cam0", "cam1", "cam2", "cam3"]
    
    # 检查所有相机文件夹是否存在
    for cam_folder in cam_folders:
        cam_path = base_path / cam_folder
        if not cam_path.exists():
            print(f"错误: 文件夹 {cam_path} 不存在")
            return
    
    # 获取图片数量（假设所有文件夹都有相同数量的图片）
    first_cam_path = base_path / cam_folders[0]
    image_files = sorted([f for f in first_cam_path.glob("*.png")])
    total_images_per_cam = len(image_files)
    
    print(f"每个相机文件夹包含 {total_images_per_cam} 张图片")
    print(f"总共将处理 {total_images_per_cam * len(cam_folders)} 张图片")
    
    # 重新编号计数器
    new_index = 0
    
    # 按图片编号遍历
    for img_num in range(total_images_per_cam):
        # 按相机顺序遍历
        for cam_folder in cam_folders:
            # 构建源文件路径
            source_file = base_path / cam_folder / f"{img_num:04d}.png"
            
            # 检查源文件是否存在
            if not source_file.exists():
                print(f"警告: 文件 {source_file} 不存在，跳过")
                continue
            
            # 构建目标文件路径
            target_file = output_path / f"depth{new_index:06d}.png"
            
            # 复制文件
            try:
                shutil.copy2(source_file, target_file)
                print(f"复制: {source_file} -> {target_file}")
                new_index += 1
            except Exception as e:
                print(f"错误: 复制文件 {source_file} 失败: {e}")
    
    print(f"\n完成! 总共处理了 {new_index} 张图片")
    print(f"输出文件夹: {output_path}")


def reorganize_pose():
    """
    重新组织相机姿态数据：
    1. 从camera_poses_0.csv到camera_poses_3.csv中依次取数据
    2. 重新编号并保存到新的CSV文件中
    """
    # 定义路径
    base_path = Path("Datasets/NewerCollege4Cam")
    output_path = Path("Datasets/NewerCollege/traj.csv")
    
    # 定义姿态文件
    pose_files = ["camera_poses_0.csv", "camera_poses_1.csv", "camera_poses_2.csv", "camera_poses_3.csv"]
    
    # 检查所有姿态文件是否存在
    for pose_file in pose_files:
        pose_path = base_path / pose_file
        if not pose_path.exists():
            print(f"错误: 文件 {pose_path} 不存在")
            return
    
    # 读取第一个文件来获取数据行数
    first_pose_path = base_path / pose_files[0]
    first_df = pd.read_csv(first_pose_path, header=None)
    total_poses_per_cam = len(first_df)
    
    print(f"每个姿态文件包含 {total_poses_per_cam} 行数据")
    print(f"总共将处理 {total_poses_per_cam * len(pose_files)} 行数据")
    
    # 存储所有重新组织的数据
    reorganized_data = []
    
    # 按数据行编号遍历
    for row_num in range(total_poses_per_cam):
        # 按相机顺序遍历
        for pose_file in pose_files:
            pose_path = base_path / pose_file
            
            try:
                # 读取CSV文件
                df = pd.read_csv(pose_path, header=None)
                
                # 检查行是否存在
                if row_num < len(df):
                    # 获取该行数据
                    row_data = df.iloc[row_num].values
                    reorganized_data.append(row_data)
                    print(f"添加: {pose_file} 第{row_num}行")
                else:
                    print(f"警告: {pose_file} 第{row_num}行不存在，跳过")
                    
            except Exception as e:
                print(f"错误: 读取文件 {pose_path} 失败: {e}")
    
    # 将重新组织的数据保存到新文件
    if reorganized_data:
        reorganized_df = pd.DataFrame(reorganized_data)
        reorganized_df.to_csv(output_path, header=False, index=False)
        print(f"\n完成! 总共处理了 {len(reorganized_data)} 行数据")
        print(f"输出文件: {output_path}")
    else:
        print("错误: 没有数据被处理")

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    将四元数转换为3x3旋转矩阵
    四元数格式: (qx, qy, qz, qw)
    """
    # 归一化四元数
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # 转换为旋转矩阵
    R = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [  2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz - qx*qw)],
        [  2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=float)
    
    return R

def create_transformation_matrix(x, y, z, qx, qy, qz, qw):
    """
    创建4x4变换矩阵
    """
    # 获取旋转矩阵
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    
    # 创建4x4变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T

def transform_traj():
    """
    将traj.csv转换为traj.txt
    输入格式: x, y, z, qx, qy, qz, qw (位置 + 四元数)
    输出格式: 4x4变换矩阵的16个元素 (按行展开)
    """
    # 定义路径
    input_file = Path("Datasets/NewerCollege/traj.csv")
    output_file = Path("Datasets/NewerCollege/traj.txt")
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"错误: 输入文件 {input_file} 不存在")
        return
    
    print(f"正在读取文件: {input_file}")
    
    # 读取原始数据
    data = []
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # 解析CSV格式
            parts = line.split(',')
            if len(parts) != 7:
                print(f"警告: 第{line_num}行数据格式不正确，跳过")
                continue
            
            try:
                x, y, z, qx, qy, qz, qw = [float(p) for p in parts]
                data.append((x, y, z, qx, qy, qz, qw))
            except ValueError:
                print(f"警告: 第{line_num}行数据解析失败，跳过")
                continue
    
    print(f"成功读取 {len(data)} 行数据")
    
    # 转换格式
    print("正在转换格式...")
    transformed_data = []
    
    for i, (x, y, z, qx, qy, qz, qw) in enumerate(data):
        # 创建4x4变换矩阵
        T = create_transformation_matrix(x, y, z, qx, qy, qz, qw)
        
        # 按行展开为16个数字
        flattened = T.flatten()
        transformed_data.append(flattened)
        
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1} 行数据")
    
    # 保存转换后的数据
    print(f"正在保存到文件: {output_file}")
    with open(output_file, 'w') as f:
        for row in transformed_data:
            # 将16个数字用空格分隔写入
            f.write(' '.join([f'{val:.16e}' for val in row]) + '\n')
    
    print(f"转换完成! 共处理 {len(transformed_data)} 个姿态")
    print(f"输出文件: {output_file}")
    
    # 显示前几行作为验证
    print("\n转换结果预览（前3行）:")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"第{i+1}行: {line.strip()}")
    

def convert_camera_intrinsic():
    # 定义路径
    input_file = Path('Datasets/NewerCollege/camera_intrinsic.csv')
    output_file = Path('Datasets/NewerCollege/cam_params.json')
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f'错误: 输入文件 {input_file} 不存在')
        return
    
    print(f'正在读取相机内参文件: {input_file}')
    
    # 读取相机内参矩阵
    K = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                row = [float(x) for x in line.split(',')]
                K.append(row)
    
    K = np.array(K)
    print(f'相机内参矩阵:')
    print(K)
    
    # 提取参数
    fx = K[0, 0]  # 焦距x
    fy = K[1, 1]  # 焦距y
    cx = K[0, 2]  # 主点x
    cy = K[1, 2]  # 主点y
    
    # 假设图像尺寸（需要根据实际情况调整）
    # 从NewerCollege数据集文档来看，图像尺寸可能是256x256或其他
    w = 256  # 图像宽度
    h = 256  # 图像高度
    
    # 创建相机参数字典
    camera_params = {
        'camera': {
            'w': int(w),
            'h': int(h),
            'fx': float(fx),
            'fy': float(fy),
            'cx': float(cx),
            'cy': float(cy),
            'scale': 6553.5  # 深度缩放因子，与Replica保持一致
        }
    }
    
    # 保存为JSON文件
    print(f'正在保存相机参数到: {output_file}')
    with open(output_file, 'w') as f:
        json.dump(camera_params, f, indent=4)
    
    print('相机内参转换完成!')
    print(f'输出文件: {output_file}')
    print('相机参数:')
    print(json.dumps(camera_params, indent=2))


if __name__ == "__main__":
    reorganize_depth_images()
    # reorganize_pose()
    # transform_traj()
    # convert_camera_intrinsic()