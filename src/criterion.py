import torch
import torch.nn as nn


class Criterion(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        # loss weights
        self.boundary_loss_weight = args.criteria["boundary_loss_weight"]
        self.perturbation_loss_weight = args.criteria["perturbation_loss_weight"]
        self.eikonal_surface_weight = args.criteria["eikonal_surface_weight"]
        self.eikonal_space_weight = args.criteria["eikonal_space_weight"]
        self.heat_weight = args.criteria["heat_weight"]
        self.sign_weight_free = args.criteria["sign_weight_free"]
        self.sign_weight_occ = args.criteria["sign_weight_occ"]
        self.projection_weight = args.criteria["projection_weight"]
        self.grad_dir_weight = args.criteria["grad_dir_weight"]
        self.heat_lambda = args.criteria["heat_lambda"]

    def forward(self, pred_sdf,
                points_xyz=None,
                grad=None,
                positive_sdf_mask=None,
                negative_sdf_mask=None,
                surface_mask=None, perturbation_mask=None,
                ray_sample_mask=None,
                gt_sdf=None):

        loss = 0
        loss_dict = {}

        boundary_loss, perturbation_loss = self.get_sdf_loss(
                perturbation_mask, surface_mask, pred_sdf, gt_sdf,
            )
        loss += self.boundary_loss_weight * boundary_loss + self.perturbation_loss_weight * perturbation_loss
        loss_dict["boundary_loss"] = boundary_loss.item()
        loss_dict["perturbation_loss"] = perturbation_loss.item()
        if self.sign_weight_free > 0 or self.sign_weight_occ > 0:
            sign_loss_free, sign_loss_occ = self.compute_sign_loss(positive_sdf_mask, negative_sdf_mask, pred_sdf)
            loss += self.sign_weight_free * sign_loss_free + self.sign_weight_occ * sign_loss_occ
            loss_dict["sign_loss_free"] = sign_loss_free.item()
            loss_dict["sign_loss_occ"] = sign_loss_occ.item()
        # 计算eikonal loss和heat loss
        if self.eikonal_surface_weight > 0 or self.eikonal_space_weight > 0:
            eikonal_loss_surface, eikonal_loss_space = self.compute_eikonal_loss(grad, surface_mask)
            loss += self.eikonal_surface_weight * eikonal_loss_surface + self.eikonal_space_weight * eikonal_loss_space
            loss_dict["eikonal_loss_surface"] = eikonal_loss_surface.item()
            loss_dict["eikonal_loss_space"] = eikonal_loss_space.item()
        if self.heat_weight > 0:
            heat_loss = self.compute_heat_loss(ray_sample_mask, pred_sdf, grad)
            loss += self.heat_weight * heat_loss
            loss_dict["heat_loss"] = heat_loss.item()
        if self.projection_weight > 0 or self.grad_dir_weight > 0:
            dir_gt_space = self.find_nearest_neighbor(points_xyz, surface_mask, perturbation_mask)
            if self.projection_weight > 0:
                projection_loss = self.compute_projection_loss(dir_gt_space, pred_sdf, surface_mask, perturbation_mask)
                loss += self.projection_weight * projection_loss
                loss_dict["projection_loss"] = projection_loss.item()
            if self.grad_dir_weight > 0:
                grad_dir_loss = self.compute_grad_dir_loss(grad, dir_gt_space, surface_mask)
                loss += self.grad_dir_weight * grad_dir_loss
                loss_dict["grad_dir_loss"] = grad_dir_loss.item()
        loss_dict["total_loss"] = loss.item()
        # print(f"loss_dict: {loss_dict}")
        return loss, loss_dict

    def get_sdf_loss(self, perturbation_mask, surface_mask, pred_sdf, gt_sdf, loss_type="l2"):
        """
        compute sdf loss for perturbation and surface points using L2 loss

        Args:
            perturbation_mask (tensor): mask indicating perturbation samples (num_rays, N+M+1)
            surface_mask (tensor): mask indicating surface samples (num_rays, N+M+1)
            pred_sdf (tensor): predicted sdf values (num_rays, N+M+1)
            gt_sdf_from_depth (tensor): ground truth sdf values from depth (num_rays, N+M+1)
            loss_type (str, Default="l2"): loss style
        
        Returns:
            sdf_loss (tensor): SDF loss for perturbation and surface points
        """
        pred_sdf_perturb = pred_sdf[perturbation_mask.squeeze()]
        pred_sdf_surface = pred_sdf[surface_mask.squeeze()]
        
        boundary_loss = pred_sdf_surface.abs().mean()
        perturbation_loss = (pred_sdf_perturb - gt_sdf.squeeze()).abs().mean()
        
        return boundary_loss, perturbation_loss
    
    def compute_sign_loss(self, positive_sdf_mask, negative_sdf_mask, pred_sdf):
        """
        计算sign loss - 只在free space上计算
        """
        free_pred = pred_sdf[positive_sdf_mask.squeeze()]
        occ_pred = pred_sdf[negative_sdf_mask.squeeze()]
        sign_loss_free = (torch.tanh(100 * free_pred) - 1).abs().mean()
        sign_loss_occ = (torch.tanh(100 * occ_pred) + 1).abs().mean()
        return sign_loss_free, sign_loss_occ
        
    def compute_eikonal_loss(self, grad, surface_mask):
        """
        计算eikonal loss - 梯度模长应该接近1
        
        Args:
            gradient (tensor): 梯度 (N_rays, N_samples, 3)
            gradient_valid_mask (tensor): 梯度有效性mask (N_rays, N_samples)
            
        Returns:
            eikonal_loss (tensor): eikonal loss
        """
        grad_norm = torch.norm(grad, dim=-1, keepdim=True)
        eikonal_loss_surface = (grad_norm[surface_mask.squeeze()] - 1).abs().mean()
        eikonal_loss_space = (grad_norm[~surface_mask.squeeze()] - 1).abs().mean()
        return eikonal_loss_surface, eikonal_loss_space
    
    def compute_heat_loss(self, ray_sample_mask, pred_sdf, grad, heat_lambda=4):
        """
        计算heat loss - 只在free space上计算
        
        Args:
            z_vals (tensor): 采样点深度 (N_rays, N_samples)
            gt_depth (tensor): GT深度 (N_rays)
            pred_sdf (tensor): 预测SDF (N_rays, N_samples)
            grad (tensor): 梯度 (N_rays, N_samples, 3)
            grad_valid_mask (tensor): 梯度有效性mask (N_rays, N_samples)
            heat_lambda (float): heat kernel参数
            
        Returns:
            heat_loss (tensor): heat loss
        """
        
        # 使用组合mask选择有效的梯度和SDF值
        valid_grad = grad[ray_sample_mask.squeeze()]
        valid_sdf = pred_sdf[ray_sample_mask.squeeze()]
        
        grad_norm = torch.norm(valid_grad, dim=-1, keepdim=True)
        heat = torch.exp(-heat_lambda * valid_sdf.abs()).unsqueeze(1)
        
        heat_loss = (0.5 * heat ** 2 * (grad_norm ** 2 + 1)).mean()
            
        return heat_loss
    
    def find_nearest_neighbor(self, points_xyz, surface_mask, perturbation_mask, batch_size=10000):
        """
        找到points_xyz中每个space点最近的surface点
        使用分批计算防止OOM
        
        Args:
            points_xyz: 所有点的坐标
            surface_mask: surface点的mask
            perturbation_mask: perturbation点的mask  
            batch_size: 分批大小，默认1000
        """
        valid_mask = ~surface_mask.squeeze() & ~perturbation_mask.squeeze()
        X_surf = points_xyz[surface_mask.squeeze()]
        X_space = points_xyz[valid_mask]
        
        n_space = X_space.shape[0]
        
        # 分批计算
        nn_indices = []
        with torch.no_grad():
            for i in range(0, n_space, batch_size):
                end_idx = min(i + batch_size, n_space)
                X_space_batch = X_space[i:end_idx]
                
                # 计算当前批次与所有surface点的距离
                d2_batch = torch.cdist(X_space_batch, X_surf)
                nn_idx_batch = d2_batch.argmin(dim=1)
                nn_indices.append(nn_idx_batch)
        
        # 合并所有批次的结果
        nn_idx_space = torch.cat(nn_indices, dim=0)
        X_nn_space = X_surf[nn_idx_space]
        dir_gt_space = X_space - X_nn_space
        return dir_gt_space
    
    def compute_projection_loss(self, dir_gt_space, pred_sdf, surface_mask, perturbation_mask):
        """
        计算投影loss - 只在非surface且非perturbation上计算
        """
        # 排除surface和perturbation的点
        valid_mask = ~surface_mask.squeeze() & ~perturbation_mask.squeeze()
        dist_space = dir_gt_space.norm(dim=1)  # (N_space,)
        pred_sdf_space = pred_sdf[valid_mask]
        projection_loss = (pred_sdf_space - dist_space.detach()).abs().mean()
        # print(f"projection_loss: {projection_loss.item()}")
        return projection_loss

    def compute_grad_dir_loss(self, grad, dir_gt_space, surface_mask):
        """
        计算grad_dir loss - 只在非surface上计算
        """
        dir_gt_space_normed = dir_gt_space / (dir_gt_space.norm(dim=1, keepdim=True) + 1e-8)
        grad_space = grad[~surface_mask.squeeze()]
        grad_space_normed = grad_space / (grad_space.norm(dim=1, keepdim=True) + 1e-8)
        grad_dir_loss = (1.0 - (grad_space_normed * dir_gt_space_normed).sum(dim=1)).mean()
        return grad_dir_loss
