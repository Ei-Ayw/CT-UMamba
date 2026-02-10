"""
Dynamic-Mamba Loss Functions
支持三大创新点的损失函数：
1. ILSE Loss: 水平集演化损失 (Level Set Evolution Loss)
2. OCD-Contrast Loss: 物体-背景解耦对比损失 (Object-Context Disentangled Contrast Loss)
3. Dir-MoE Loss: 路由稀疏性损失 (Router Sparsity Loss)

Author: Generated for SCI Journal Publication (TGRS/TIP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LevelSetLoss(nn.Module):
    """
    水平集演化损失 (Level Set Evolution Loss)
    用于ILSE头，优化SDF预测
    """
    def __init__(self, beta: float = 10.0, lambda_length: float = 0.1, lambda_area: float = 0.01):
        super().__init__()
        self.beta = beta
        self.lambda_length = lambda_length
        self.lambda_area = lambda_area
        
    def forward(self, sdf_pred: torch.Tensor, mask_gt: torch.Tensor, ignore_index: int = 255):
        """
        Args:
            sdf_pred: (B, num_classes, H, W) 预测的SDF
            mask_gt: (B, H, W) 真实标签
            ignore_index: 忽略的类别索引
        Returns:
            loss: 水平集损失
        """
        B, num_classes, H, W = sdf_pred.shape
        device = sdf_pred.device
        
        # 将mask转换为one-hot编码
        mask_onehot = torch.zeros(B, num_classes, H, W, device=device)
        valid_mask = (mask_gt != ignore_index)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        mask_onehot.scatter_(1, mask_gt.unsqueeze(1) * valid_mask.unsqueeze(1), 1.0)
        mask_onehot = mask_onehot * valid_mask.unsqueeze(1)
        
        # 数据项：SDF与真实边界的匹配
        # 对于物体内部（mask=1），SDF应该>0；对于物体外部（mask=0），SDF应该<0
        mask_from_sdf = torch.sigmoid(self.beta * sdf_pred)  # (B, num_classes, H, W)
        data_term = F.mse_loss(mask_from_sdf * valid_mask.unsqueeze(1), 
                               mask_onehot, reduction='sum') / (valid_mask.sum() * num_classes + 1e-8)
        
        # 长度正则化项：鼓励边界光滑
        # 计算SDF的梯度
        grad_x = torch.abs(sdf_pred[:, :, :, 1:] - sdf_pred[:, :, :, :-1])
        grad_y = torch.abs(sdf_pred[:, :, 1:, :] - sdf_pred[:, :, :-1, :])
        length_term = (grad_x.mean() + grad_y.mean()) * self.lambda_length
        
        # 面积项：鼓励SDF在边界附近接近0
        area_term = torch.abs(sdf_pred * valid_mask.unsqueeze(1)).mean() * self.lambda_area
        
        total_loss = data_term + length_term + area_term
        
        return total_loss


class DisentangledContrastLoss(nn.Module):
    """
    物体-背景解耦对比损失 (Object-Context Disentangled Contrast Loss)
    用于OCD-Contrast，确保物体特征和环境特征真正解耦
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        f_obj_a: torch.Tensor,
        f_ctx_a: torch.Tensor,
        f_obj_b: torch.Tensor,
        f_ctx_b: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
    ):
        """
        计算解耦对比损失
        
        Args:
            f_obj_a: (B, C//2, H, W) A图的物体特征
            f_ctx_a: (B, C//2, H, W) A图的环境特征
            f_obj_b: (B, C//2, H, W) B图的物体特征
            f_ctx_b: (B, C//2, H, W) B图的环境特征
            mask_a: (B, H, W) A图的标签
            mask_b: (B, H, W) B图的标签
        Returns:
            loss: 解耦对比损失
        """
        B, C_half, H, W = f_obj_a.shape
        
        # 将特征展平为 (B, C//2, H*W)
        f_obj_a_flat = f_obj_a.view(B, C_half, -1)  # (B, C//2, H*W)
        f_ctx_a_flat = f_ctx_a.view(B, C_half, -1)
        f_obj_b_flat = f_obj_b.view(B, C_half, -1)
        f_ctx_b_flat = f_ctx_b.view(B, C_half, -1)
        
        # L2归一化
        f_obj_a_norm = F.normalize(f_obj_a_flat, p=2, dim=1)  # (B, C//2, H*W)
        f_ctx_a_norm = F.normalize(f_ctx_a_flat, p=2, dim=1)
        f_obj_b_norm = F.normalize(f_obj_b_flat, p=2, dim=1)
        f_ctx_b_norm = F.normalize(f_ctx_b_flat, p=2, dim=1)
        
        # 一致性约束：交换后的特征应该能预测出原始物体的mask
        # 拼接A的物体特征和B的环境特征
        f_swapped_ab = torch.cat([f_obj_a_norm, f_ctx_b_norm], dim=1)  # (B, C, H*W)
        
        # 计算相似度矩阵（简化：使用平均池化后的特征）
        f_obj_a_pool = f_obj_a_norm.mean(dim=2)  # (B, C//2)
        f_ctx_a_pool = f_ctx_a_norm.mean(dim=2)
        f_obj_b_pool = f_obj_b_norm.mean(dim=2)
        f_ctx_b_pool = f_ctx_b_norm.mean(dim=2)
        
        # 物体特征应该与物体mask相关，环境特征应该与环境相关
        # 这里简化处理：鼓励物体特征在物体区域激活，环境特征在背景区域激活
        
        # 提取物体区域和背景区域
        mask_a_obj = (mask_a > 0).float().view(B, -1)  # (B, H*W)
        mask_a_bg = (mask_a == 0).float().view(B, -1)
        mask_b_obj = (mask_b > 0).float().view(B, -1)
        mask_b_bg = (mask_b == 0).float().view(B, -1)
        
        # 物体特征应该在物体区域激活
        obj_activation_a = (f_obj_a_norm * mask_a_obj.unsqueeze(1)).sum(dim=2).mean()  # 标量
        obj_activation_b = (f_obj_b_norm * mask_b_obj.unsqueeze(1)).sum(dim=2).mean()
        
        # 环境特征应该在背景区域激活
        ctx_activation_a = (f_ctx_a_norm * mask_a_bg.unsqueeze(1)).sum(dim=2).mean()
        ctx_activation_b = (f_ctx_b_norm * mask_b_bg.unsqueeze(1)).sum(dim=2).mean()
        
        # 对比损失：鼓励物体特征与环境特征分离
        # 物体特征与环境特征的相似度应该低
        obj_ctx_sim_a = (f_obj_a_pool * f_ctx_a_pool).sum(dim=1).mean()
        obj_ctx_sim_b = (f_obj_b_pool * f_ctx_b_pool).sum(dim=1).mean()
        
        # 总损失：最大化物体激活，最大化环境激活，最小化物体-环境相似度
        loss = -obj_activation_a - obj_activation_b - ctx_activation_a - ctx_activation_b + \
               0.5 * (obj_ctx_sim_a + obj_ctx_sim_b)
        
        return loss


class RouterSparsityLoss(nn.Module):
    """
    路由稀疏性损失 (Router Sparsity Loss)
    用于Dir-MoE，鼓励路由决策的稀疏性
    """
    def __init__(self, lambda_sparse: float = 0.01):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        
    def forward(self, router_weights: torch.Tensor):
        """
        Args:
            router_weights: (B, H, W, num_directions) 路由权重
        Returns:
            loss: 稀疏性损失
        """
        # 计算熵：鼓励权重分布集中（低熵）
        # 熵越高，分布越均匀，越不稀疏
        router_probs = router_weights + 1e-8  # 避免log(0)
        entropy = -(router_probs * torch.log(router_probs)).sum(dim=-1)  # (B, H, W)
        
        # 稀疏性损失：最小化熵（鼓励高稀疏性）
        sparsity_loss = entropy.mean() * self.lambda_sparse
        
        return sparsity_loss


class DynamicMambaLoss(nn.Module):
    """
    Dynamic-Mamba完整损失函数
    集成所有创新点的损失
    
    Args:
        ignore_index: 忽略的类别索引
        use_ilse: 是否使用ILSE损失
        use_ocd_contrast: 是否使用OCD-Contrast损失
        use_dir_moe: 是否使用Dir-MoE路由稀疏性损失
        lambda_ilse: ILSE损失权重
        lambda_ocd: OCD-Contrast损失权重
        lambda_router: Router稀疏性损失权重
        class_weights: 类别权重（可选）
    """
    def __init__(
        self,
        ignore_index: int = 255,
        use_ilse: bool = True,
        use_ocd_contrast: bool = True,
        use_dir_moe: bool = True,
        lambda_ilse: float = 1.0,
        lambda_ocd: float = 0.5,
        lambda_router: float = 0.01,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.use_ilse = use_ilse
        self.use_ocd_contrast = use_ocd_contrast
        self.use_dir_moe = use_dir_moe
        
        self.lambda_ilse = lambda_ilse
        self.lambda_ocd = lambda_ocd
        self.lambda_router = lambda_router
        
        # 基础分割损失（如果不用ILSE，使用标准CE+Dice）
        if not use_ilse:
            from .useful_loss import OHEM_CELoss
            from .dice import DiceLoss
            self.seg_loss = nn.ModuleDict({
                'ce': OHEM_CELoss(thresh=0.7, ignore_index=ignore_index, weight=class_weights),
                'dice': DiceLoss(smooth=0.05, ignore_index=ignore_index),
            })
        else:
            # ILSE损失
            self.ilse_loss = LevelSetLoss(beta=10.0, lambda_length=0.1, lambda_area=0.01)
        
        # OCD-Contrast损失
        if use_ocd_contrast:
            self.ocd_loss = DisentangledContrastLoss(temperature=0.07)
        
        # Router稀疏性损失
        if use_dir_moe:
            self.router_loss = RouterSparsityLoss(lambda_sparse=lambda_router)
    
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        auxiliary_outputs: Optional[dict] = None,
        batch_features: Optional[Tuple] = None,
    ):
        """
        Args:
            prediction: (B, num_classes, H, W) 预测结果
            target: (B, H, W) 真实标签
            auxiliary_outputs: 辅助输出字典，包含：
                - 'sdf': SDF预测（如果使用ILSE）
                - 'router_infos': 路由信息（如果使用Dir-MoE）
                - 'disentangled_features': 解耦特征（如果使用OCD-Contrast）
            batch_features: (f_obj_a, f_ctx_a, f_obj_b, f_ctx_b, mask_a, mask_b) 用于OCD-Contrast
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典（用于记录）
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. 主分割损失
        if self.use_ilse and auxiliary_outputs is not None and 'sdf' in auxiliary_outputs:
            # 使用ILSE损失
            sdf = auxiliary_outputs['sdf']
            ilse_loss = self.ilse_loss(sdf, target)
            total_loss += self.lambda_ilse * ilse_loss
            loss_dict['ilse_loss'] = ilse_loss.item()
        else:
            # 使用标准分割损失
            if not self.use_ilse:
                ce_loss = self.seg_loss['ce'](prediction, target)
                dice_loss = self.seg_loss['dice'](prediction, target)
                seg_loss = ce_loss + dice_loss
            else:
                # 如果使用ILSE但没有SDF，使用prediction作为mask
                from .useful_loss import OHEM_CELoss
                from .dice import DiceLoss
                ce_loss = OHEM_CELoss(thresh=0.7, ignore_index=255)(prediction, target)
                dice_loss = DiceLoss(smooth=0.05, ignore_index=255)(prediction, target)
                seg_loss = ce_loss + dice_loss
            
            total_loss += seg_loss
            loss_dict['seg_loss'] = seg_loss.item()
        
        # 2. OCD-Contrast损失
        if self.use_ocd_contrast and batch_features is not None:
            f_obj_a, f_ctx_a, f_obj_b, f_ctx_b, mask_a, mask_b = batch_features
            ocd_loss = self.ocd_loss(f_obj_a, f_ctx_a, f_obj_b, f_ctx_b, mask_a, mask_b)
            total_loss += self.lambda_ocd * ocd_loss
            loss_dict['ocd_loss'] = ocd_loss.item()
        
        # 3. Router稀疏性损失
        if self.use_dir_moe and auxiliary_outputs is not None and 'router_infos' in auxiliary_outputs:
            router_infos = auxiliary_outputs['router_infos']
            router_loss_sum = 0.0
            count = 0
            for router_info_list in router_infos:
                for router_info in router_info_list:
                    if 'router_weights' in router_info:
                        router_loss = self.router_loss(router_info['router_weights'])
                        router_loss_sum += router_loss
                        count += 1
            if count > 0:
                router_loss_avg = router_loss_sum / count
                total_loss += self.lambda_router * router_loss_avg
                loss_dict['router_loss'] = router_loss_avg.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict

