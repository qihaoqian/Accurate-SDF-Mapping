#!/usr/bin/env python3
import pandas as pd
import os
import glob

def merge_csv_files():
    """合并logs/metric目录中的CSV文件"""
    
    # 定义目录路径
    metric_dir = "logs/metric/baseline2.0"
    
    # 定义三种类型的文件模式
    file_patterns = {
        "mesh_metrics": "mesh_metrics_*.csv",
        "mesh_metrics_priors": "mesh_metrics_priors_*.csv", 
        "evaluation_sdf_metrics": "evaluation_sdf_metrics_*.csv"
    }
    
    # 为每种类型创建合并的DataFrame
    merged_data = {}
    
    for pattern_name, pattern in file_patterns.items():
        print(f"正在处理 {pattern_name} 文件...")
        
        # 获取匹配的文件列表
        files = glob.glob(os.path.join(metric_dir, pattern))
        files.sort()  # 按文件名排序
        
        if not files:
            print(f"未找到匹配 {pattern} 的文件")
            continue
            
        print(f"找到 {len(files)} 个文件: {[os.path.basename(f) for f in files]}")
        
        # 读取并合并所有文件
        dataframes = []
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                # 从文件名提取场景信息
                filename = os.path.basename(file_path)
                scene_id = None
                if "room" in filename:
                    idx = filename.find("room")
                    # 提取 room 后面的数字
                    rest = filename[idx:]
                    num = ""
                    for c in rest[4:]:
                        if c.isdigit():
                            num += c
                        else:
                            break
                    scene_id = f"room{num}" if num else "room"
                elif "office" in filename:
                    idx = filename.find("office")
                    # 提取 office 后面的数字
                    rest = filename[idx:]
                    num = ""
                    for c in rest[6:]:
                        if c.isdigit():
                            num += c
                        else:
                            break
                    scene_id = f"office{num}" if num else "office"
                else:
                    scene_id = "unknown"
                
                # 添加场景信息列（合并为 room0/office1 格式）
                df['scene_id'] = scene_id
                df['filename'] = filename
                
                dataframes.append(df)
                print(f"  已读取: {filename}")
                
            except Exception as e:
                print(f"  读取文件 {file_path} 时出错: {e}")
        
        if dataframes:
            # 合并所有DataFrame
            merged_df = pd.concat(dataframes, ignore_index=True)
            merged_data[pattern_name] = merged_df
            
            # 保存合并后的文件
            output_file = f"logs/metric/merged_{pattern_name}.csv"
            merged_df.to_csv(output_file, index=False)
            print(f"已保存合并文件: {output_file}")
            print(f"合并后数据形状: {merged_df.shape}")
            print()
    
    # 创建所有数据的总体合并文件
    if merged_data:
        print("创建总体合并文件...")
        all_dataframes = []
        for name, df in merged_data.items():
            # 添加数据类型标识
            df_copy = df.copy()
            df_copy['data_type'] = name
            all_dataframes.append(df_copy)
        
        if all_dataframes:
            final_merged = pd.concat(all_dataframes, ignore_index=True)
            final_output = "logs/metric/baseline2.0/merged_all_metrics.csv"
            final_merged.to_csv(final_output, index=False)
            print(f"已保存总体合并文件: {final_output}")
            print(f"总体数据形状: {final_merged.shape}")
    
    print("CSV文件合并完成！")

if __name__ == "__main__":
    merge_csv_files()
