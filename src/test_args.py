#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试命令行参数的简单脚本
"""

import argparse
import yaml
import sys
import os
from pathlib import Path

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='测试命令行参数解析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python test_args.py                                    # 使用默认参数
  python test_args.py --file test.ply                   # 指定文件
  python test_args.py --config config.yaml              # 使用配置文件
  python test_args.py --list-available                  # 列出可用文件
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='配置文件路径 (YAML格式)'
    )
    
    parser.add_argument(
        '--file', 
        type=str,
        default="Datasets/Replica/room0_mesh.ply",
        help='要分析的文件路径'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细信息'
    )
    
    parser.add_argument(
        '--list-available',
        action='store_true',
        help='列出Datasets/Replica目录下所有可用的PLY文件'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果指定了配置文件，先加载配置
    if args.config:
        try:
            with open(args.config, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            print(f"加载配置文件: {args.config}")
            print(f"配置内容: {cfg}")
            
            # 将配置文件中的参数合并到args中
            for key, value in cfg.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
                    print(f"从配置文件设置 {key} = {value}")
        except Exception as e:
            print(f"加载配置文件时发生错误: {e}")
            return
    
    # 如果指定了--list-available，列出可用文件
    if args.list_available:
        list_available_ply_files()
        return
    
    print(f"最终参数:")
    print(f"  file: {args.file}")
    print(f"  verbose: {args.verbose}")
    print(f"  config: {args.config}")

def list_available_ply_files():
    """列出Datasets/Replica目录下所有可用的PLY文件"""
    replica_dir = Path("Datasets/Replica")
    
    if not replica_dir.exists():
        print(f"错误：目录 {replica_dir} 不存在")
        return
    
    ply_files = list(replica_dir.glob("*.ply"))
    
    if not ply_files:
        print(f"在 {replica_dir} 目录下没有找到PLY文件")
        return
    
    print(f"\n在 {replica_dir} 目录下找到以下PLY文件:")
    print("-" * 50)
    
    for i, ply_file in enumerate(ply_files, 1):
        file_size = ply_file.stat().st_size / (1024 * 1024)  # 转换为MB
        print(f"{i:2d}. {ply_file.name} ({file_size:.1f} MB)")
    
    print("-" * 50)
    print(f"总共找到 {len(ply_files)} 个PLY文件")

if __name__ == "__main__":
    main() 