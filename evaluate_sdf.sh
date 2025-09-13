#!/bin/bash
# evaluate_all.sh
# 循环运行 evaluate_sdf.py，遍历 room0~2 和 office0~4

scenes=(room0 room1 room2 office0 office1 office2 office3 office4)
for scene in "${scenes[@]}"; do
    echo ">>> 正在评估 $scene ..."
    python src/evaluate_sdf.py logs/replica/$scene/baseline/bak/config.yaml
done

echo "所有场景的SDF评估已完成！"
