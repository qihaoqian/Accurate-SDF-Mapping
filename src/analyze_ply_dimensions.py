#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLY文件尺寸分析脚本
分析3D网格文件的长宽高信息
"""

import numpy as np
import open3d as o3d
import sys
import os
import argparse
import yaml
from pathlib import Path

def analyze_ply_dimensions(ply_file_path, verbose=False):
    """
    分析PLY文件的长宽高
    
    Args:
        ply_file_path (str): PLY文件路径
        verbose (bool): 是否显示详细信息
    
    Returns:
        dict: 包含长宽高信息的字典
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(ply_file_path):
            print(f"错误：文件 {ply_file_path} 不存在")
            return None
        
        if verbose:
            print(f"正在加载PLY文件: {ply_file_path}")
        
        # 加载PLY文件
        mesh = o3d.io.read_triangle_mesh(ply_file_path)
        
        if not mesh.has_vertices():
            print("错误：PLY文件没有顶点数据")
            return None
        
        # 获取顶点坐标
        vertices = np.asarray(mesh.vertices)
        
        if len(vertices) == 0:
            print("错误：PLY文件没有有效的顶点")
            return None
        
        # 计算边界框
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        
        # 计算长宽高
        dimensions = max_coords - min_coords
        
        # 计算中心点
        center = (min_coords + max_coords) / 2
        
        # 计算体积（如果网格是封闭的）
        volume = 0
        if mesh.is_watertight():
            volume = mesh.get_volume()
        
        # 计算表面积
        surface_area = mesh.get_surface_area()
        
        # 统计信息
        stats = {
            'file_path': ply_file_path,
            'vertex_count': len(vertices),
            'triangle_count': len(mesh.triangles) if mesh.has_triangles() else 0,
            'bounding_box': {
                'min': min_coords.tolist(),
                'max': max_coords.tolist(),
                'center': center.tolist()
            },
            'dimensions': {
                'length_x': dimensions[0],
                'width_y': dimensions[1], 
                'height_z': dimensions[2]
            },
            'surface_area': surface_area,
            'volume': volume if volume > 0 else None,
            'is_watertight': mesh.is_watertight()
        }
        
        return stats
        
    except Exception as e:
        print(f"分析PLY文件时发生错误: {str(e)}")
        return None

def print_analysis_results(stats, verbose=False):
    """
    打印分析结果
    
    Args:
        stats (dict): 分析结果字典
        verbose (bool): 是否显示详细信息
    """
    if not stats:
        return
    
    print("\n" + "="*60)
    print("PLY文件尺寸分析结果")
    print("="*60)
    
    print(f"文件路径: {stats['file_path']}")
    print(f"顶点数量: {stats['vertex_count']:,}")
    print(f"面片数量: {stats['triangle_count']:,}")
    
    print("\n边界框信息:")
    print(f"  最小值 (X, Y, Z): {stats['bounding_box']['min']}")
    print(f"  最大值 (X, Y, Z): {stats['bounding_box']['max']}")
    print(f"  中心点 (X, Y, Z): {stats['bounding_box']['center']}")
    
    print("\n尺寸信息:")
    print(f"  长度 (X轴): {stats['dimensions']['length_x']:.4f} 单位")
    print(f"  宽度 (Y轴): {stats['dimensions']['width_y']:.4f} 单位")
    print(f"  高度 (Z轴): {stats['dimensions']['height_z']:.4f} 单位")
    
    print(f"\n表面积: {stats['surface_area']:.4f} 平方单位")
    
    if stats['volume'] is not None:
        print(f"体积: {stats['volume']:.4f} 立方单位")
    else:
        print("体积: 无法计算（网格可能不封闭）")
    
    print(f"网格是否封闭: {'是' if stats['is_watertight'] else '否'}")
    
    # 计算最大尺寸
    max_dim = max(stats['dimensions'].values())
    print(f"\n最大尺寸: {max_dim:.4f} 单位")
    
    if verbose:
        print(f"\n详细信息:")
        print(f"  文件大小: {os.path.getsize(stats['file_path']) / (1024*1024):.2f} MB")
        print(f"  顶点密度: {stats['vertex_count'] / stats['surface_area']:.2f} 顶点/平方单位")
    
    print("="*60)

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
    print("\n使用方法:")
    print(f"  python {sys.argv[0]} <文件名>")
    print(f"  例如: python {sys.argv[0]} {ply_files[0].name}")

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='分析PLY文件的长宽高信息',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python analyze_ply_dimensions.py                                    # 分析默认文件
  python analyze_ply_dimensions.py Datasets/Replica/room0_mesh.ply    # 分析指定文件
  python analyze_ply_dimensions.py --config config.yaml              # 使用配置文件
  python analyze_ply_dimensions.py --list-available                  # 列出可用文件
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
        help='要分析的PLY文件路径'
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
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出结果到JSON文件'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果指定了配置文件，先加载配置
    if args.config:
        try:
            with open(args.config, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            # 将配置文件中的参数合并到args中
            for key, value in cfg.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
        except Exception as e:
            print(f"加载配置文件时发生错误: {e}")
            return
    
    # 如果指定了--list-available，列出可用文件
    if args.list_available:
        list_available_ply_files()
        return
    
    if args.verbose:
        print(f"命令行参数: {sys.argv}")
        print(f"分析PLY文件: {args.file}")
    
    # 分析PLY文件
    stats = analyze_ply_dimensions(args.file, args.verbose)
    
    if stats:
        print_analysis_results(stats, args.verbose)
        
        # 如果指定了输出文件，保存结果
        if args.output:
            try:
                import json
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                print(f"\n结果已保存到: {args.output}")
            except Exception as e:
                print(f"保存结果时发生错误: {e}")
    else:
        print("分析失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 