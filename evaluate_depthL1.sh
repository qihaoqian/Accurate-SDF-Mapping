#!/bin/bash
# evaluate_all.sh
# 循环运行 evaluate_sdf.py，遍历 room0~2 和 office0~4

scenes=(office0 office1 office2 office3 office4)  #room0 room1 room2 
for scene in "${scenes[@]}"; do
    echo ">>> 正在评估 $scene ..."
    python src/evaluate_depthL1.py --gt_mesh_path Datasets/Replica/${scene}_mesh.ply --rec_mesh_path logs/replica/baseline_latest/${scene}/mesh/mesh_80.obj --n_imgs 1000
done

echo "所有场景的深度L1评估已完成！"
