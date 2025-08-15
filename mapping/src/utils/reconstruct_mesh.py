#!/usr/bin/env python3
"""
重建Mesh脚本
从训练好的checkpoint加载模型并重建3D mesh

使用方法:
python reconstruct_mesh.py configs/replica/room_0.yaml --ckpt_path logs/replica/room0/2025-08-06-19-44-27/ckpt/final_ckpt.pth

或者使用默认路径:
python reconstruct_mesh.py configs/replica/room_0.yaml
"""

import os
import sys
import torch

# 添加路径
sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath('src'))

from demo.parser import get_parser
from src.utils.inference import load_and_extract_mesh


def main():
    # 使用项目原有的参数解析器
    parser = get_parser()
    
    # 添加额外的mesh重建参数
    parser.add_argument('--ckpt_path', type=str, 
                       default='logs/replica/room0/2025-08-06-19-44-27/ckpt/final_ckpt.pth',
                       help='checkpoint文件路径')
    parser.add_argument('--mesh_res', type=int, default=256,
                       help='mesh分辨率 (默认: 256)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认: checkpoint同级的mesh目录)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU设备ID (默认: 0)')
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # 检查文件是否存在
    if not os.path.exists(args.ckpt_path):
        print(f"❌ Checkpoint文件不存在: {args.ckpt_path}")
        return
    
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return
    
    print(f"🔧 使用GPU: {args.gpu}")
    print(f"📄 Checkpoint: {args.ckpt_path}")
    print(f"⚙️  配置文件: {args.config}")
    print(f"🎯 Mesh分辨率: {args.mesh_res}")
    
    try:
        # 重建mesh，如果内存不足则自动降低分辨率
        current_res = args.mesh_res
        success = False
        
        while current_res >= 64 and not success:  # 最低分辨率64
            try:
                print(f"\n🎯 尝试分辨率: {current_res}")
                mesh, output_path = load_and_extract_mesh(
                    ckpt_path=args.ckpt_path,
                    args=args,
                    mesh_res=current_res,
                    output_dir=args.output_dir
                )
                success = True
                
                print(f"\n🎉 重建完成！Mesh已保存到: {output_path}")
                print(f"📊 最终使用分辨率: {current_res}")
                print(f"📈 你可以使用MeshLab、Blender或其他3D软件查看mesh")
                
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️  分辨率 {current_res} 内存不足，尝试降低分辨率...")
                current_res = current_res // 2
                torch.cuda.empty_cache()
                
                if current_res < 64:
                    print("❌ 已达到最低分辨率64，仍然内存不足")
                    print("💡 建议:")
                    print("   1. 关闭其他GPU程序")
                    print("   2. 使用更大GPU内存的设备")
                    print("   3. 手动设置更低的mesh_res (如32)")
                    break
                else:
                    print(f"🔄 降低到分辨率: {current_res}")
        
        if not success and current_res >= 64:
            raise Exception("无法在任何分辨率下完成重建")
            
    except Exception as e:
        if "OutOfMemoryError" not in str(e):
            print(f"❌ 重建失败: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 