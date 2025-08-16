#!/usr/bin/env python3
"""
Mesh Reconstruction Script
Load model from trained checkpoint and reconstruct 3D mesh

Usage:
python reconstruct_mesh.py configs/replica/room_0.yaml --ckpt_path logs/replica/room0/2025-08-06-19-44-27/ckpt/final_ckpt.pth

Or use default path:
python reconstruct_mesh.py configs/replica/room_0.yaml
"""

import os
import sys
import torch

# Add paths
sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath('src'))

from demo.parser import get_parser
from src.utils.inference import load_and_extract_mesh


def main():
    # Use project's original argument parser
    parser = get_parser()
    
    # Add additional mesh reconstruction parameters
    parser.add_argument('--ckpt_path', type=str, 
                       default='logs/replica/room0/2025-08-06-19-44-27/ckpt/final_ckpt.pth',
                       help='checkpoint file path')
    parser.add_argument('--mesh_res', type=int, default=256,
                       help='mesh resolution (default: 256)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='output directory (default: mesh directory at same level as checkpoint)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Check if files exist
    if not os.path.exists(args.ckpt_path):
        print(f"❌ Checkpoint file does not exist: {args.ckpt_path}")
        return
    
    if not os.path.exists(args.config):
        print(f"❌ Config file does not exist: {args.config}")
        return
    
    print(f"🔧 Using GPU: {args.gpu}")
    print(f"📄 Checkpoint: {args.ckpt_path}")
    print(f"⚙️  Config file: {args.config}")
    print(f"🎯 Mesh resolution: {args.mesh_res}")
    
    try:
        # Reconstruct mesh, automatically reduce resolution if out of memory
        current_res = args.mesh_res
        success = False
        
        while current_res >= 64 and not success:  # minimum resolution 64
            try:
                print(f"\n🎯 Trying resolution: {current_res}")
                mesh, output_path = load_and_extract_mesh(
                    ckpt_path=args.ckpt_path,
                    args=args,
                    mesh_res=current_res,
                    output_dir=args.output_dir
                )
                success = True
                
                print(f"\n🎉 Reconstruction completed! Mesh saved to: {output_path}")
                print(f"📊 Final resolution used: {current_res}")
                print(f"📈 You can view the mesh using MeshLab, Blender or other 3D software")
                
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️  Resolution {current_res} out of memory, trying to reduce resolution...")
                current_res = current_res // 2
                torch.cuda.empty_cache()
                
                if current_res < 64:
                    print("❌ Reached minimum resolution 64, still out of memory")
                    print("💡 Suggestions:")
                    print("   1. Close other GPU programs")
                    print("   2. Use device with larger GPU memory")
                    print("   3. Manually set lower mesh_res (e.g. 32)")
                    break
                else:
                    print(f"🔄 Reducing to resolution: {current_res}")
        
        if not success and current_res >= 64:
            raise Exception("Cannot complete reconstruction at any resolution")
            
    except Exception as e:
        if "OutOfMemoryError" not in str(e):
            print(f"❌ Reconstruction failed: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 