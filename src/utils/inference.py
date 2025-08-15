import os
import sys
import torch
import numpy as np

# 添加路径以便导入项目模块
sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath('src'))

from demo.parser import get_parser
from src.utils.import_util import get_dataset, get_decoder
from src.frame import RGBDFrame
from src.loggers import BasicLogger
from src.mapping import Mapping


def load_checkpoint(ckpt_path, args=None):
    """
    加载训练好的checkpoint文件
    
    Args:
        ckpt_path (str): checkpoint文件路径，例如：
                        "mapping/logs/replica/room0/2025-08-06-19-44-27/ckpt/final_ckpt.pth"
        args: 训练参数，如果为None则需要从配置文件加载
    
    Returns:
        mapper: 已加载状态的Mapping对象
        decoder: 已加载状态的解码器
    """
    
    # 加载sparse octree库
    torch.classes.load_library(
        "third_party/sparse_octree/build/lib.linux-x86_64-cpython-310/svo.cpython-310-x86_64-linux-gnu.so")
    
    # 1. 加载checkpoint文件
    print(f"正在加载checkpoint: {ckpt_path}")
    training_result = torch.load(ckpt_path, map_location='cuda:0')
    
    # 检查checkpoint内容
    print("Checkpoint包含的键:", list(training_result.keys()))
    
    # 2. 创建解码器
    decoder = get_decoder(args).cuda()
    print("解码器已创建")
    
    # 3. 创建数据流（用于初始化）
    data_stream = get_dataset(args)
    data_in = data_stream[0]
    first_frame = RGBDFrame(*data_in[:-1], offset=args.mapper_specs['offset'], 
                           ref_pose=data_in[-1]).cuda()
    W, H = first_frame.rgb.shape[1], first_frame.rgb.shape[0]
    
    # 4. 创建logger和mapper
    logger = BasicLogger(args, for_eva=True)
    mapper = Mapping(args, logger, data_stream=data_stream)
    
    # 5. 从checkpoint恢复状态
    print("正在恢复模型状态...")
    
    # 恢复解码器状态
    mapper.decoder.load_state_dict(training_result['decoder_state'])
    
    # 恢复SDF先验和地图状态
    mapper.sdf_priors = training_result['sdf_priors'].cuda()
    mapper.map_states = training_result['map_state']
    
    # 设置为评估模式
    mapper.decoder = mapper.decoder.cuda()
    mapper.decoder.eval()
    
    print("Checkpoint加载完成！")
    print(f"解码器参数数量: {sum(p.numel() for p in mapper.decoder.parameters())}")
    print(f"SDF先验形状: {mapper.sdf_priors.shape}")
    print(f"地图状态键: {list(mapper.map_states.keys())}")
    
    return mapper, decoder


def load_checkpoint_simple(ckpt_path):
    """
    简单的checkpoint加载（仅加载权重，不创建完整的mapper）
    
    Args:
        ckpt_path (str): checkpoint文件路径
    
    Returns:
        dict: 包含所有保存状态的字典
    """
    print(f"正在加载checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    print("Checkpoint内容:")
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        elif isinstance(value, dict):
            print(f"  {key}: 字典包含 {len(value)} 个键")
        else:
            print(f"  {key}: {type(value)}")
    
    return checkpoint


def load_and_extract_mesh(ckpt_path, args, mesh_res=256, output_dir=None):
    """
    加载checkpoint并提取mesh
    
    Args:
        ckpt_path (str): checkpoint文件路径
        args: 配置参数
        mesh_res (int): mesh分辨率，默认256
        output_dir (str): 输出目录，默认为checkpoint同级的mesh目录
    
    Returns:
        mesh: 提取的mesh对象
        output_path: mesh保存路径
    """
    
    # 1. 加载checkpoint
    print("=" * 50)
    print("开始加载checkpoint并重建mesh")
    print("=" * 50)
    
    mapper, decoder = load_checkpoint(ckpt_path, args)
    
    # 2. 设置输出目录
    if output_dir is None:
        # 默认保存到checkpoint同级的mesh目录
        ckpt_dir = os.path.dirname(ckpt_path)
        result_dir = os.path.dirname(ckpt_dir)  # 上一级目录
        output_dir = os.path.join(result_dir, "mesh")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 3. 更新mapper的mesh目录
    mapper.logger.mesh_dir = output_dir
    
    # 4. 提取mesh
    print(f"\n开始提取mesh，分辨率: {mesh_res}")
    print("这可能需要几分钟时间...")
    
    try:
        mesh, sdf_field, sdf_priors, hash_features = mapper.extract_mesh(
            res=mesh_res, 
            map_states=mapper.map_states
        )
        
        # 5. 保存mesh
        mesh_name = f"reconstructed_mesh_res{mesh_res}.ply"
        output_path = os.path.join(output_dir, mesh_name)
        mesh.export(output_path)
        
        print(f"\n✅ Mesh重建完成！")
        print(f"📁 输出路径: {output_path}")
        print(f"📊 顶点数量: {len(mesh.vertices)}")
        print(f"📊 面片数量: {len(mesh.faces)}")
        
        # 6. 保存额外的调试信息
        debug_dir = os.path.join(output_dir, "debug")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        np.save(os.path.join(debug_dir, f"sdf_field_res{mesh_res}.npy"), sdf_field)
        np.save(os.path.join(debug_dir, f"sdf_priors_res{mesh_res}.npy"), sdf_priors)
        np.save(os.path.join(debug_dir, f"hash_features_res{mesh_res}.npy"), hash_features)
        
        print(f"🔍 调试数据保存到: {debug_dir}")
        
        return mesh, output_path
        
    except Exception as e:
        print(f"❌ Mesh提取失败: {str(e)}")
        raise e


# 使用示例
if __name__ == "__main__":
    # 方法1: 完整加载（需要配置参数）
    # parser = get_parser()
    # args = parser.parse_args()
    # ckpt_path = "mapping/logs/replica/room0/2025-08-06-19-44-27/ckpt/final_ckpt.pth"
    # mapper, decoder = load_checkpoint(ckpt_path, args)
    
    # 方法2: 简单加载（仅查看checkpoint内容）
    ckpt_path = "mapping/logs/replica/room0/2025-08-06-19-44-27/ckpt/final_ckpt.pth"
    checkpoint = load_checkpoint_simple(ckpt_path)
    
    # 方法3: 加载并重建mesh
    # parser = get_parser()
    # args = parser.parse_args(['--config', 'configs/replica/room0.yaml'])  # 替换为正确的配置文件
    # mesh, output_path = load_and_extract_mesh(ckpt_path, args, mesh_res=256)
