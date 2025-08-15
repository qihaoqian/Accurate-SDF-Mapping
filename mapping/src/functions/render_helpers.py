import torch
import torch.nn.functional as F

from .voxel_helpers import ray_intersect, ray_sample, ray_sample_isdf, edge_favored_samples


def ray(ray_start, ray_dir, depths):
    """
    Calculate the coordinates of the sampling point in the world coordinate system according to the light origin, 
    direction and depth value of the sampling point
    
    Args:
        ray_start (tensor, N_rays*1*3): ray's origin coordinates in world coordinate.
        ray_dir (tensor, N_rays*1*3): ray's dir in world coordinate.
        depths (tensor, N_rays*N_points*1): depths of sampling points along the ray.

    Returns:
        ray_start+ray_dir*depth (tensor, N_rays*N_points*3): sampling points in world
    """
    return ray_start + ray_dir * depths


def fill_in(shape, mask, input, initial=1.0):
    if isinstance(initial, torch.Tensor):
        output = initial.expand(*shape)
    else:
        output = input.new_ones(*shape) * initial
    return output.masked_scatter(mask.unsqueeze(-1).expand(*shape), input)


def masked_scatter(mask, x):
    """
    The sampling points that did not hit the voxel were masked in the previous program, 
    this function restores the previous dimension, and the masked element is set to 0
    
    Args:
        mask (tensor, N_rays*N_points_every_ray):
        x (tensor, samples_points):
    
    Returns:
        (tensor, N_rays*n_points_every_ray):Restore the dimension before the mask, the position of the unimpacted voxel is 0
    """
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_zeros(B, K).masked_scatter(mask, x)
    return x.new_zeros(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x
    )


def masked_scatter_ones(mask, x):
    """
    The sampling points that did not hit the voxel were masked in the previous program, 
    this function restores the previous dimension, and the masked element is set to 1
    
    Args:
        mask (tensor, N_rays*N_points_every_ray):
        x (tensor, samples_points):
    
    Returns:
        (tensor, N_rays*n_points_every_ray):Restore the dimension before the mask, the position of the unimpacted voxel is 1
    """
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_ones(B, K).masked_scatter(mask, x)
    return x.new_ones(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x
    )


@torch.enable_grad()
def trilinear_interp(p, q, point_feats):
    """
    For the feature vector stored in a voxel, 
    perform cubic linear interpolation to obtain the feature of each point.
    """
    weights = (p * q + (1 - p) * (1 - q)).prod(dim=-1, keepdim=True)
    if point_feats.dim() == 2:
        point_feats = point_feats.view(point_feats.size(0), 8, -1)

    point_feats = (weights * point_feats).sum(1)
    return point_feats

@torch.enable_grad()
def extra_interp(sampled_xyz, point_xyz, point_feats, point_vector_features, voxel_size, point_voxel_size):
    """
    For the feature vector stored in a voxel, 
    perform cubic linear interpolation to obtain the feature of each point.
    
    Args:
        p: 归一化的采样点位置 (N_points, 1, 3)
        q: 体素8个顶点的偏移坐标 (1, 8, 3) 
        point_feats: 8个顶点的特征 (N_points, 8*embed_dim)
        point_vector_features: 8个顶点的方向特征 (N_points, 8, 3)
    
    Returns:
        interpolated_feats: 插值后的特征 (N_points, embed_dim)
    """
    # === 1) 计算所有 (N*8) 顶点世界坐标 ===
    cut = torch.linspace(-1, 1, 2, device=sampled_xyz.device)
    xx, yy, zz = torch.meshgrid(cut, cut, cut, indexing='ij')
    offsets = torch.stack([xx, yy, zz], dim=-1).reshape(1, 8, 3)  # (1,8,3)

    half  = (point_voxel_size / 2).view(-1, 1, 1)                             # (N,1,1)
    scale = half * voxel_size
    verts = point_xyz.unsqueeze(1) + offsets * scale                # (N,8,3)
    verts = verts.reshape(-1, 3).contiguous()
    
    # 计算每个采样点与每个顶点的偏移向量
    # 使用广播：(N_points, 1, 3) - (1, 8, 3) = (N_points, 8, 3)
    verts = verts.reshape(-1, 8, 3)
    point_vector_features = point_vector_features.reshape(-1, 8, 3)
    # 在最后一个维度上normalize
    # point_vector_features = torch.nn.functional.normalize(point_vector_features, dim=-1)
    
    point_feats = point_feats.reshape(-1, 8)
    offsets = sampled_xyz.unsqueeze(1) - verts
    projection_lengths = torch.sum(offsets * point_vector_features, dim=-1, keepdim=True)  # (N_points, 8, 1)
    extra_interpolated_features = point_feats + projection_lengths.squeeze(-1)

    return extra_interpolated_features


@torch.enable_grad()  
def combined_interpolation(sampled_xyz, point_xyz, point_feats, point_vector_features, voxel_size, point_voxel_size):
    """
    结合extra_interp和linear_interp_3d的综合插值方法
    
    Args:
        p: 归一化的采样点位置 (N_points, 1, 3)
        q: 体素8个顶点的偏移坐标 (1, 8, 3)
        point_feats: 8个顶点的特征 (N_points, 8*embed_dim)
        point_vector_features: 8个顶点的方向特征 (N_points, 8, 3)
    
    Returns:
        final_result: 最终插值结果 (N_points, embed_dim)
    """
    # 方法1：使用额外插值（类似于test_interpolation.py中的extra_interpolation方法）
    extra_interpolated_features = extra_interp(sampled_xyz, point_xyz, point_feats, point_vector_features, voxel_size, point_voxel_size)
    
    # tri-linear interpolation
    p = ((sampled_xyz - point_xyz) / (point_voxel_size * voxel_size) + 0.5).unsqueeze(1)  # add 0.5 value are clamped to [0,1]
    q = offset_points(p, 0.5, offset_only=True).unsqueeze(0) + 0.5  # range[-0.5,0.5] + 0.5 shape:(1,8,3)
    feats = trilinear_interp(p, q, extra_interpolated_features).float()  # (N_points,32)
    
    return feats

def offset_points(point_xyz, quarter_voxel=1, offset_only=False, bits=2):
    c = torch.arange(1, 2 * bits, 2, device=point_xyz.device)
    ox, oy, oz = torch.meshgrid([c, c, c], indexing='ij')
    offset = (torch.cat([
        ox.reshape(-1, 1),
        oy.reshape(-1, 1),
        oz.reshape(-1, 1)], 1).type_as(point_xyz) - bits) / float(bits - 1)
    if not offset_only:
        return (
                point_xyz.unsqueeze(1) + offset.unsqueeze(0).type_as(point_xyz) * quarter_voxel)
    return offset.type_as(point_xyz) * quarter_voxel  # (8,3)


@torch.enable_grad()
def get_embeddings(sampled_xyz, point_xyz, point_feats, point_vector_features, point_voxel_size, voxel_size):
    """
    run cubic linear interrpolation and get features corresponding to sampling points.
    
    Args:
        sampled_xyz (tensor, N_points*3): points x,y,z which belong to it voxel
        point_xyz (tensor, N_points*3): voxel center x,y,z
        point_feats (tensor, N_points, 8*embed_dim): features of sample point vertices 
        voxel_size (int): voxel size
    
    Returns:
        feats (tensor, N_points*embed_dim): features after cubic linear interpolation.
    """
    # tri-linear interpolation
    p = ((sampled_xyz - point_xyz) / (point_voxel_size * voxel_size) + 0.5).unsqueeze(1)  # add 0.5 value are clamped to [0,1]
    q = offset_points(p, 0.5, offset_only=True).unsqueeze(0) + 0.5  # range[-0.5,0.5] + 0.5 shape:(1,8,3)
    feats = trilinear_interp(p, q, point_feats).float()  # (N_points,32)
    return feats


@torch.enable_grad()
def get_features(sampled_xyz, sampled_idx, map_states, voxel_size):
    """
    Retrieve the voxel corresponding to the sampling point and the surrounding vertices, 
    and obtain the input features of each sampling point through cubic linear interpolation
    
    Args:
        samples (dict): sampling points information.
        map_states (dict): voxel information according to the octrees.
        voxel_size (int): voxel size.
    
    Returns:
        inputs (dict): sampled distance(N_points) and embedding feature vectors(N_points,emb_dim).
    """
    # encoder states
    point_feats = map_states["voxel_vertex_idx"].cuda()
    point_xyz = map_states["voxel_center_xyz"].cuda()  # (voxel_num,3)
    sdf_priors_all = map_states["sdf_priors"].cuda()
    point_voxel_size = map_states["voxels"][:,-1].unsqueeze(1).cuda()
    vector_features = map_states["vector_features"].cuda()
    # ray point samples
    sampled_idx = sampled_idx.long()
    sampled_xyz = sampled_xyz.requires_grad_(True)
    # sampled_idx stores the index of each sampling point corresponding to voxel_center_xyz, 
    # and after F.embedding, the voxel center corresponding to each point is obtained
    point_xyz = F.embedding(sampled_idx, point_xyz).squeeze(1)  # (chunk_size,3)
    # sampled_idx is the voxel id corresponding to the sampling point, 
    # and find the vert id contained in feats according to the voxel id
    point_emd_idx = F.embedding(sampled_idx, point_feats).squeeze(1)  # (chunk_size,8)
    point_voxel_size = F.embedding(sampled_idx, point_voxel_size)

    point_sdf_priors = F.embedding(point_emd_idx, sdf_priors_all).view(point_xyz.size(0),
                                                                       -1)  # (chunk_size,8,emd_dim) -> (chunk_size,8*emd_dim)
    point_vector_features = F.embedding(point_emd_idx, vector_features).view(point_xyz.size(0), 8, 3) # (chunk_size,8,3)
    # sdf_priors = get_embeddings(sampled_xyz, point_xyz, point_sdf_priors, point_vector_features, point_voxel_size, voxel_size)
    sdf_priors = combined_interpolation(sampled_xyz, point_xyz, point_sdf_priors, point_vector_features, voxel_size, point_voxel_size)
    feats = None

    inputs = {"emb": feats, "sdf_priors": sdf_priors}
    return inputs


@torch.no_grad()
def get_scores(sdf_network, map_states, voxel_size, bits=8, model="parallel_hash_net"):
    """
    This function is used in the get_mesh process to obtain the sdf value of each sampling point.
    """
    feats = map_states["voxel_vertex_idx"]
    points = map_states["voxel_center_xyz"]
    sdf_priors = map_states["sdf_priors"]

    chunk_size = 32
    res = bits

    @torch.no_grad()
    def get_scores_once(feats, points):
        # sample points inside voxels
        start = -.5
        end = .5  # - 1./bits

        x = y = z = torch.linspace(start, end, res)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        sampled_xyz = torch.stack([xx, yy, zz], dim=-1).float().cuda()

        sampled_xyz *= voxel_size
        sampled_xyz = sampled_xyz.reshape(1, -1, 3) + points.unsqueeze(1)

        sampled_idx = torch.arange(points.size(0), device=points.device)
        sampled_idx = sampled_idx[:, None].expand(*sampled_xyz.size()[:2])
        sampled_idx = sampled_idx.reshape(-1)
        sampled_xyz = sampled_xyz.reshape(-1, 3)

        if sampled_xyz.shape[0] == 0:
            return

        field_inputs = get_features(
            sampled_xyz,
            sampled_idx,
            {
                "voxel_vertex_idx": feats,
                "voxel_center_xyz": points,
                "sdf_priors": sdf_priors,
            },
            voxel_size
        )

        sdf_values = sdf_network.get_sdf(sampled_xyz)
        sdf_values = sdf_values[:, -1] + field_inputs['sdf_priors'][:, -1].float().cuda()

        return sdf_values.reshape(-1, res ** 3, 1).detach().cpu()

    return torch.cat([
        get_scores_once(feats[i: i + chunk_size],
                        points[i: i + chunk_size])
        for i in range(0, points.size(0), chunk_size)], 0).view(-1, res, res, res, 1)


@torch.no_grad()
def eval_points(sdf_network, sampled_xyz):
    def get_scores_once(sampled_xyz):
        sampled_xyz = sampled_xyz.reshape(-1, 3)

        if sampled_xyz.shape[0] == 0:
            return
        color = sdf_network.get_color(sampled_xyz)
        return color.detach().cpu()

    chunk_size = 3200

    results = []
    for i in range(0, sampled_xyz.size(0), chunk_size):
        score_once = get_scores_once(sampled_xyz[i: i + chunk_size].cuda())
        results.append(score_once)
    results = torch.cat(results, dim=0)
    return results


# convert sdf to weight
def sdf2weights(sdf_in, trunc, z_vals, sample_mask_per):
    weights = torch.sigmoid(sdf_in / trunc) * \
              torch.sigmoid(-sdf_in / trunc)
    # use the change of sign to find the surface, sdf's sign changes as it cross the surface
    signs = sdf_in[:, 1:] * sdf_in[:, :-1]
    mask = torch.where(
        signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs)
    )
    # return the index of the closest point outside the surface
    inds = torch.argmax(mask, axis=1)
    inds = inds[..., None]
    z_min = torch.gather(z_vals, 1, inds)
    # calculate truncation mask, delete the point behind the surface and exceed trunc, z_min is here approximate the surface
    mask = torch.where(
        z_vals < z_min + trunc,
        torch.ones_like(z_vals),
        torch.zeros_like(z_vals),
    )
    # mask truncation and mask not hit voxel
    weights = weights * mask * sample_mask_per
    return weights / (torch.sum(weights, dim=-1, keepdims=True) + 1e-8), z_min


def render_rays(
        rays_o,
        rays_d,
        map_states,
        sdf_network,
        step_size,
        voxel_size,
        truncation,
        max_voxel_hit,
        max_distance,
        chunk_size=-1,
        profiler=None,
        return_raw=False,
        eval=False
):
    centres = map_states["voxel_center_xyz"]
    childrens = map_states["voxel_structure"]

    if profiler is not None:
        profiler.tick("ray_intersect")
    """
    intersections (dict):min_depth:(1,N_rays,N_hit_voxels) depth in camera coordinate if ray intersect voxel
            max_depth: (1,N_rays,N_hit_voxels)
            intersected_voxel_id: (1,N_rays,N_hit_voxels)
    hits:(1,N_rays),Whether each ray hits a voxel
    """
    intersections, hits = ray_intersect(
        rays_o, rays_d, centres,
        childrens, voxel_size, max_voxel_hit, max_distance)
    if profiler is not None:
        profiler.tok("ray_intersect")
    if hits.sum() == 0 and eval == True:
        ray_mask = torch.zeros_like(hits).bool().cuda()
        rgb = torch.zeros_like(rays_o).squeeze(0).cuda()
        depth = torch.zeros((rays_o.shape[1],)).cuda()
        return {
            "weights": None,
            "color": None,
            "depth": None,
            "z_vals": None,
            "sdf": None,
            "ray_mask": ray_mask,
            "raw": None if return_raw else None
        }

    else:
        assert (hits.sum() > 0)

    ray_mask = hits.view(1, -1)  # Whether each ray hits a voxel
    intersections = {
        name: outs[ray_mask].reshape(-1, outs.size(-1))
        for name, outs in intersections.items()
    }  # min_depth max_depth intersected_voxel_id: (N_rays,N_hit_voxels)

    rays_o = rays_o[ray_mask].reshape(-1, 3)  # remove rays which don't hit voxel
    rays_d = rays_d[ray_mask].reshape(-1, 3)

    """
    samples = {
        "sampled_point_depth": sampled_depth:(N_rays, N_points)
        "sampled_point_distance": sampled_dists:(N_rays, N_points)
        "sampled_point_voxel_idx": sampled_idx:(N_rays, N_points)
    }
    """
    samples = ray_sample(intersections, step_size=step_size)

    sampled_depth = samples['sampled_point_depth']
    sampled_idx = samples['sampled_point_voxel_idx'].long()

    # only compute when the ray hits, if don't hit setting False
    sample_mask = sampled_idx.ne(-1)
    if sample_mask.sum() == 0:  # miss everything skip
        return None, 0

    sampled_xyz = ray(rays_o.unsqueeze(
        1), rays_d.unsqueeze(1), sampled_depth.unsqueeze(2))
    samples['sampled_point_xyz'] = sampled_xyz

    # apply mask(remove don't hit the voxel)
    samples_valid = {name: s[sample_mask] for name, s in samples.items()}  # flatten to points

    num_points = samples_valid['sampled_point_depth'].shape[0]
    field_outputs = []
    if chunk_size < 0:
        chunk_size = num_points

    for i in range(0, num_points, chunk_size):
        chunk_samples = {name: s[i:i + chunk_size]
                         for name, s in samples_valid.items()}

        # get encoder features as inputs
        chunk_inputs = get_features(chunk_samples, map_states, voxel_size)

        # forward implicit fields
        if profiler is not None:
            profiler.tick("render_core")
        chunk_outputs = sdf_network(chunk_samples['sampled_point_xyz'])
        chunk_outputs['sdf'] = chunk_outputs['sdf'] + chunk_inputs['sdf_priors'][:, -1]

        field_outputs.append(chunk_outputs)

    field_outputs = {name: torch.cat(
        [r[name] for r in field_outputs], dim=0) for name in field_outputs[0]}

    outputs = {'sample_mask': sample_mask}

    sdf = masked_scatter_ones(sample_mask, field_outputs['sdf']).squeeze(-1)
    color = masked_scatter(sample_mask, field_outputs['color'])
    sample_mask = outputs['sample_mask']

    z_vals = samples["sampled_point_depth"]  # the depth from cam

    weights, z_min = sdf2weights(sdf, truncation, z_vals, sample_mask)

    rgb = torch.sum(weights[..., None] * color, dim=-2)

    depth = torch.sum(weights * z_vals, dim=-1)

    return {
        "weights": weights,
        "color": rgb,
        "depth": depth,
        "z_vals": z_vals,
        "sdf": sdf,
        "ray_mask": ray_mask,
        "raw": z_min if return_raw else None
    }


def finite_diff_grad_combined_safe(chunk_samples, map_states, voxel_size, sdf_network, h1=1e-4):
    """
    安全的有限差分梯度计算，确保xyz三个方向的扰动都不越界才计算梯度
    
    Args:
        chunk_samples (dict): 包含采样点信息的字典
        map_states (dict): 体素地图状态信息
        voxel_size (int): 体素大小
        sdf_network: SDF神经网络
        h1 (float): 有限差分步长
    
    Returns:
        grad (tensor): 梯度张量 [N_points, 3]
        grad_magnitude (tensor): 梯度大小 [N_points, 1]
        valid_mask (tensor): 有效梯度的掩码 [N_points]
    """
    X = chunk_samples['sampled_point_xyz']  # [N_points, 3]
    
    def safe_combined_forward_subset(xyz_points, voxel_idx_subset):
        """只对有效点子集进行前向传播"""
        if xyz_points.shape[0] == 0:
            return torch.zeros(0, 1, device=xyz_points.device)
        

        chunk_inputs = get_features(xyz_points, voxel_idx_subset, map_states, voxel_size)
        
        # 通过SDF网络
        chunk_outputs = sdf_network(xyz_points)
        
        # 组合最终的SDF值
        final_sdf = chunk_outputs['sdf'] + chunk_inputs['sdf_priors'][:, -1]
            
        return final_sdf
            
    # 自适应步长：确保偏移后的点仍在体素内
    adaptive_h1 = min(h1, voxel_size * 0.1)  # 限制步长不超过体素大小的10%
    
    # 预先检查所有方向的扰动是否都在体素内
    # 为xyz三个方向创建所有可能的扰动
    all_offsets = []
    for i in range(3):  # x, y, z三个维度
        for direction in [1, -1]:  # 正向和负向
            offset = torch.zeros_like(X)
            offset[:, i] = direction * adaptive_h1
            all_offsets.append(X + offset)
    
    # 检查所有扰动是否都在体素内
    voxel_idx_list = []
    for offset_points in all_offsets:
        # 使用优化版本的体素索引查找
        voxel_idx, _ = find_voxel_idx(offset_points, map_states)
        voxel_idx_list.append(voxel_idx)
    
    # 只有当所有6个方向的扰动的voxel_idx都不为-1时，该点才能计算梯度
    voxel_idx_stack = torch.stack(voxel_idx_list, dim=0)  # [6, N_points]
    
    
    # 初始化梯度张量
    grad = torch.zeros_like(X)
    
    # 只对有效的点计算梯度
        
    for i in range(3):  # 对于x, y, z三个维度
        # 创建偏移向量
        offset = torch.zeros_like(X)
        offset[:, i] = adaptive_h1
        
        # 计算正向和负向扰动的SDF值（只对有效点）
        sdf_plus = safe_combined_forward_subset(X + offset, voxel_idx_stack[2*i,:])
        sdf_minus = safe_combined_forward_subset(X - offset, voxel_idx_stack[2*i+1,:])
        
        # 使用中心差分公式计算梯度
        grad_i = (sdf_plus - sdf_minus) / (2 * adaptive_h1)
        
        grad[:, i] = grad_i.squeeze(-1)
        
    return grad

def bundle_adjust_frames(
        keyframe_graph,
        map_states,
        sdf_network,
        loss_criteria,
        voxel_size,
        N_rays=512,
        num_iterations=10,
        bound=None,
        update_pose=True,
        optim=None,
        scaler=None,
        frame_id=None,
        use_adaptive_ending=False,
        writer=None,
        epoch=0,
        h1=0.05,
        sample_specs=None,
):
    # sample rays from keyframes
    rays_o_all, rays_d_all, rgb_samples_all, depth_samples_all = [], [], [], []
    num_keyframe = len(keyframe_graph)
    for i, frame in enumerate(keyframe_graph):
        if update_pose:
            d_pose = frame.get_d_pose().cuda()
            ref_pose = frame.get_ref_pose().cuda()
            pose = d_pose @ ref_pose
        else:
            pose = frame.get_ref_pose().cuda()
        valid_idx = torch.nonzero(frame.valid_mask.reshape(-1))
        sample_idx = valid_idx[torch.randint(low=0, high=int(valid_idx.shape[0]),
                                             size=(int(num_iterations * (N_rays / num_keyframe)),))][:, 0]
        sampled_rays_d = frame.rays_d.cuda().reshape(-1, 3)[sample_idx]
        R = pose[: 3, : 3].transpose(-1, -2)
        sampled_rays_d = sampled_rays_d @ R
        sampled_rays_o = pose[: 3, 3].reshape(
            1, -1).expand_as(sampled_rays_d)
        rays_d_all += [sampled_rays_d]
        rays_o_all += [sampled_rays_o]
        rgb_samples_all += [frame.rgb.cuda().reshape(-1, 3)[sample_idx]]
        depth_samples_all += [frame.depth.cuda().reshape(-1)[sample_idx]]

    rays_d_all = torch.cat(rays_d_all, dim=0)
    rays_o_all = torch.cat(rays_o_all, dim=0)
    rgb_samples_all = torch.cat(rgb_samples_all, dim=0)
    depth_samples_all = torch.cat(depth_samples_all, dim=0)

    loss_all = 0
    exceed_cnt = 0

    sampled_xyz, positive_sdf_mask, negative_sdf_mask, gaussian_positive_mask, surface_mask, perturbation_mask, ray_sample_mask, valid_indices = ray_sample_isdf(
        rays_d_all, rays_o_all, depth_samples_all, N=sample_specs['N'], M=sample_specs['M'], 
        d_min=sample_specs['d_min'], delta=sample_specs['delta'], 
        sigma_s=sample_specs['sigma_s'], max_depth=sample_specs['max_depth']
    )
    sampled_xyz = sampled_xyz.reshape(-1, 3)
    positive_sdf_mask = positive_sdf_mask.reshape(-1, 1)
    negative_sdf_mask = negative_sdf_mask.reshape(-1, 1)
    gaussian_positive_mask = gaussian_positive_mask.reshape(-1, 1)
    surface_mask = surface_mask.reshape(-1, 1)
    perturbation_mask = perturbation_mask.reshape(-1, 1)
    ray_sample_mask = ray_sample_mask.reshape(-1, 1)
    
    surface_xyz = sampled_xyz[surface_mask.squeeze(-1)]
    perturbation_xyz = sampled_xyz[perturbation_mask.squeeze(-1)]
    
    with torch.no_grad():
        batch_size = 10000  # 按显存情况调节
        nearest_distances_list = []

        for start in range(0, perturbation_xyz.shape[0], batch_size):
            end = start + batch_size
            perturb_batch = perturbation_xyz[start:end]  # (B, 3)

            # 计算当前批次与所有 surface 点的距离
            distances = torch.cdist(perturb_batch, surface_xyz)  # (B, M)

            # 找每个点最近的 surface
            batch_min_dists, _ = torch.min(distances, dim=1)

            nearest_distances_list.append(batch_min_dists)

        # 拼接回完整结果
        nearest_distances = torch.cat(nearest_distances_list, dim=0)
    
    # 在positive_mask的地方乘以-1
    nearest_distances[gaussian_positive_mask.squeeze(-1)] *= -1
    
    gt_sdf = nearest_distances.reshape(-1, 1)

    point_voxel_idx = find_voxel_idx(sampled_xyz, map_states)
    # 将sampled_xyz, sampled_depth, negative_sdf_mask, point_voxel_idx 打包成一个字典 samples_valid
    samples = {
        'sampled_point_xyz': sampled_xyz,
        'sampled_point_voxel_idx': point_voxel_idx,
    }

    chunk_size = 100000000
    idx = 0
    for _ in range(num_iterations):
        optim.zero_grad()
        # apply mask(remove don't hit the voxel)
        num_points = sampled_xyz.shape[0]
        field_outputs = []
        
        # Separate mixed precision training: gradient computation outside autocast
        # First perform normal forward pass within autocast
        with torch.cuda.amp.autocast():
            for i in range(0, num_points, chunk_size):
                chunk_samples = {name: s[i:i + chunk_size]
                                 for name, s in samples.items()}
                chunk_inputs = get_features(chunk_samples['sampled_point_xyz'],chunk_samples['sampled_point_voxel_idx'], map_states, voxel_size)
                chunk_outputs = sdf_network(chunk_samples['sampled_point_xyz'])
                chunk_outputs['sdf'] = chunk_outputs['sdf'] + chunk_inputs['sdf_priors'][:, -1]
                # print(f"chunk_outputs['sdf']: {chunk_outputs['sdf'].mean().item()}, {chunk_outputs['sdf'].min().item()}, {chunk_outputs['sdf'].max().item()}")
                field_outputs.append(chunk_outputs)
        
        # Perform gradient computation outside autocast for high precision
        for i in range(0, num_points, chunk_size):
            chunk_samples = {name: s[i:i + chunk_size]
                             for name, s in samples.items()}
            
            # Use higher precision for gradient computation
            grad = finite_diff_grad_combined_safe(
                chunk_samples, map_states, voxel_size, sdf_network, h1=h1
            )
            
        # Combine all outputs
        field_outputs = {name: torch.cat(
            [r[name] for r in field_outputs], dim=0) for name in field_outputs[0]}
        
        # Final loss computation within autocast
        with torch.cuda.amp.autocast():
            pred_sdf = field_outputs['sdf']

            # Loss computation
            loss , loss_dict = loss_criteria(
                pred_sdf, 
                points_xyz=sampled_xyz,
                grad=grad,
                positive_sdf_mask=positive_sdf_mask,
                negative_sdf_mask=negative_sdf_mask,
                surface_mask=surface_mask,
                perturbation_mask=perturbation_mask,
                ray_sample_mask=ray_sample_mask,
                gt_sdf=gt_sdf,
            )
            
        global_step = epoch * num_iterations + idx
        print(f"global_step: {global_step}, loss_dict: {loss_dict}")
        if writer is not None:
            writer.add_scalar('loss/total_loss', loss.item(), global_step)
            for key, value in loss_dict.items():
                writer.add_scalar(f'loss/{key}', value, global_step)
            
        if use_adaptive_ending:
            loss_all += loss.item()
            loss_mean = loss_all / (epoch + 1)
            if loss_mean < loss:
                exceed_cnt += 1
            else:
                exceed_cnt = 0
            if exceed_cnt >= 2 and frame_id != 0:
                break
        idx += 1
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

def find_voxel_idx(points, map_states):
    """
    points: [N, 3] 世界坐标
    返回:  [N] 每个点对应的全局体素索引，-1 表示不在树覆盖范围内 / 未命中叶子
    """
    device = points.device
    tree   = map_states["voxel_structure"].to(device)      # [M, 9]
    centers= map_states["voxel_center_xyz"].to(device)     # [M, 3]
    size = map_states["voxels"][:, -1].to(device)
    
    
    N = points.shape[0]
    max_steps = 9
    root_idx = 0

    # 结果初始化为 -1
    voxel_idx = torch.full((N,), -1, dtype=torch.long, device=device)

    # 当前仍在遍历的点索引、它们所在节点行号
    active_pts   = torch.arange(N, device=device)        # [A]
    cur_nodes    = torch.full_like(active_pts, root_idx) # 初始都在根节点

    for _ in range(max_steps):
        if active_pts.numel() == 0:
            break

        # 计算子编号
        c        = centers[cur_nodes]                # [A,3]
        ge_mask  = (points[active_pts] >= c).long()  # [A,3]
        child_id = ge_mask[:, 0] + (ge_mask[:, 1] << 1) + (ge_mask[:, 2] << 2)
        child_idx = tree[cur_nodes, child_id].long() # [A]

        # 命中条件：size==1 就直接确定
        hit_mask = (child_idx == -1)
        if hit_mask.any():
            voxel_idx[active_pts[hit_mask]] = cur_nodes[hit_mask]
        # 仅继续没命中的
        keep_mask = ~hit_mask
        if not keep_mask.any():
            break

        active_pts   = active_pts[keep_mask]
        cur_nodes = child_idx[keep_mask]

    return voxel_idx




