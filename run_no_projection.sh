#!/bin/bash

# 脚本用于按顺序执行所有baseline配置文件的mapping
# 执行命令: python demo/run_mapping.py configs/baseline/*.yaml --save_mesh --evaluate --calculate_sdf_loss

echo "开始执行所有baseline配置文件..."
echo "======================================"

# 获取所有yaml配置文件并排序
config_files=(configs/ablation-exp/no_projection/*.yaml)
IFS=$'\n' sorted_files=($(sort <<<"${config_files[*]}"))
unset IFS

# 记录开始时间
start_time=$(date)
echo "开始时间: $start_time"
echo "======================================"

# 初始化计数器
total_files=${#sorted_files[@]}
current_file=0

# 遍历所有配置文件
for config_file in "${sorted_files[@]}"; do
    current_file=$((current_file + 1))
    echo ""
    echo "[$current_file/$total_files] 正在处理: $config_file"
    echo "命令: python demo/run_mapping.py $config_file --save_mesh --evaluate"
    echo "--------------------------------------"
    
    # 记录单个任务开始时间
    task_start=$(date)
    echo "任务开始时间: $task_start"
    
    # 执行命令
    python demo/run_mapping.py "$config_file" --save_mesh --evaluate
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        task_end=$(date)
        echo "✅ 任务完成: $config_file"
        echo "任务结束时间: $task_end"
    else
        echo "❌ 任务失败: $config_file"
        echo "错误代码: $?"
        # 可以选择在此处退出或继续执行下一个配置文件
        # exit 1  # 取消注释这行以在失败时退出
        echo "继续执行下一个配置文件..."
    fi
    
    echo "======================================"
done

# 记录结束时间
end_time=$(date)
echo ""
echo "所有任务执行完成!"
echo "开始时间: $start_time"
echo "结束时间: $end_time"
echo "总共处理了 $total_files 个配置文件" 