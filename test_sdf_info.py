import numpy as np
import os

sdf_info_path = "/home/qihao/workplace/H2-Mapping-cursor-gradient/Datasets/Replica/room0/sdf_info"

# 读取并打印每个npy文件的shape
file_list = [
    "grad_gt_all.npy",
    "grad_gt_far_surface.npy",
    "grad_gt_near_surface.npy",
    "gt_sdf_values_far_surface.npy",
    "gt_sdf_values_for_eval.npy",
    "gt_sdf_values_near_surface.npy",
    "points_far_surface.npy",
    "points_for_eval.npy",
    "points_near_surface.npy"
]

for fname in file_list:
    fpath = os.path.join(sdf_info_path, fname)
    if os.path.exists(fpath):
        arr = np.load(fpath)
        print(f"{fname}: shape = {arr.shape}")
    else:
        print(f"{fname}: 文件不存在")
