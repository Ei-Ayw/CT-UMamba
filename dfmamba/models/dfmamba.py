import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
import timm
from dfmamba.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import math
import copy
from functools import partial
from collections import OrderedDict
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from dfmamba.models.ResT import ResT
from dfmamba.models.DFM import DF_Module
from dfmamba.models.ca import CoordAtt
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 尝试导入torchvision的ConvNeXt作为备选
try:
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    TORCHVISION_AVAILABLE = True
    print("检测到torchvision ConvNeXt可用")
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("torchvision ConvNeXt不可用")

def rest_lite(pretrained=True, weight_path='pretrain_weights/rest_lite.pth', **kwargs):
    """加载ResT轻量级模型"""
    model = ResT(embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, **kwargs)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path, map_location='cpu')
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def convnext_tiny_lite(pretrained=True, weight_path='pretrain_weights/convnext_tiny-983f1562.pth', **kwargs):
    """加载ConvNeXt Tiny模型并适配为特征提取器"""
    
    class ConvNeXtFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = None
            self.use_torchvision = False
            
            # 方法1: 尝试使用timm的ConvNeXt模型
            print("尝试使用timm库加载ConvNeXt...")
            convnext_names = [
                'convnext_tiny.fb_in1k',
                'convnext_tiny.in12k_ft_in1k', 
                'convnext_tiny',
                'convnext_tiny_in22k',
                'convnext_tiny.fb_in1k_384'
            ]
            
            for model_name in convnext_names:
                try:
                    self.model = timm.create_model(model_name, features_only=True, pretrained=False, out_indices=(1, 2, 3, 4))
                    print(f"✓ 成功使用timm创建ConvNeXt模型: {model_name}")
                    break
                except Exception as e:
                    print(f"  尝试模型名称 {model_name} 失败: {e}")
                    continue
            
            # 方法2: 如果timm失败，尝试使用torchvision
            if self.model is None and TORCHVISION_AVAILABLE:
                try:
                    print("尝试使用torchvision加载ConvNeXt...")
                    from torchvision.models.feature_extraction import create_feature_extractor
                    base_model = convnext_tiny(weights=None)
                    # 提取特征的节点名称
                    return_nodes = {
                        'features.1': 'stage1',  # [96, H/4, W/4]
                        'features.3': 'stage2',  # [192, H/8, W/8] 
                        'features.5': 'stage3',  # [384, H/16, W/16]
                        'features.7': 'stage4',  # [768, H/32, W/32]
                    }
                    self.model = create_feature_extractor(base_model, return_nodes)
                    self.use_torchvision = True
                    print("✓ 成功使用torchvision创建ConvNeXt模型")
                except Exception as e:
                    print(f"torchvision ConvNeXt创建失败: {e}")
            
            # 方法3: 最后回退到ResNet18
            if self.model is None:
                print("所有ConvNeXt方法都失败，回退到ResNet18")
                try:
                    self.model = timm.create_model('resnet18', features_only=True, pretrained=False, out_indices=(1, 2, 3, 4))
                    print("✓ 回退到ResNet18成功")
                except Exception as e:
                    print(f"ResNet18创建也失败: {e}")
                    # 使用最简单的ResNet18
                    self.model = timm.create_model('resnet18', features_only=True, pretrained=False)
            
            # 设置特征信息
            if hasattr(self.model, 'feature_info'):
                self.feature_info = self.model.feature_info
            else:
                # 为torchvision模型手动设置特征信息
                if self.use_torchvision:
                    from types import SimpleNamespace
                    self.feature_info = SimpleNamespace()
                    self.feature_info.channels = lambda: [96, 192, 384, 768]
                else:
                    # ResNet18的通道数
                    from types import SimpleNamespace
                    self.feature_info = SimpleNamespace()
                    self.feature_info.channels = lambda: [64, 128, 256, 512]
            
        def forward(self, x):
            if self.use_torchvision:
                # torchvision模型返回字典
                features_dict = self.model(x)
                return [features_dict['stage1'], features_dict['stage2'], 
                       features_dict['stage3'], features_dict['stage4']]
            else:
                # timm模型直接返回列表
                return self.model(x)
    
    model = ConvNeXtFeatureExtractor()
    
    # 加载预训练权重
    if pretrained and weight_path is not None and os.path.exists(weight_path) and not model.use_torchvision:
        try:
            print(f"尝试加载ConvNeXt权重: {weight_path}")
            state_dict = torch.load(weight_path, map_location='cpu')
            
            # 检查是否是ConvNeXt模型
            model_state = model.model.state_dict()
            if any('convnext' in str(type(model.model)).lower() or 
                   'stages' in model_state or 'downsample_layers' in model_state 
                   for _ in [None]):
                
                # 尝试加载权重
                try:
                    missing_keys, unexpected_keys = model.model.load_state_dict(state_dict, strict=False)
                    print(f"✓ 成功加载ConvNeXt权重，缺失: {len(missing_keys)}，多余: {len(unexpected_keys)}")
                except Exception as e:
                    print(f"权重加载失败: {e}")
                    # 尝试匹配维度的权重
                    pretrained_dict = {k: v for k, v in state_dict.items() 
                                     if k in model_state and model_state[k].shape == v.shape}
                    model_state.update(pretrained_dict)
                    model.model.load_state_dict(model_state)
                    print(f"✓ 部分加载ConvNeXt权重: {len(pretrained_dict)}/{len(state_dict)} 层")
            else:
                print("检测到非ConvNeXt模型，跳过权重加载")
                
        except Exception as e:
            print(f"加载ConvNeXt权重失败: {e}")
            print("使用未预训练的模型继续...")
    elif model.use_torchvision:
        print("使用torchvision模型，跳过自定义权重加载")
    
    return model


class FeatureFusionModule(nn.Module):
    """特征融合模块：融合CNN和Transformer特征"""

    def __init__(self, cnn_channels: int, trans_channels: int, out_channels: int):
        super().__init__()
        # 调整CNN特征维度
        self.cnn_proj = nn.Sequential(
            nn.Conv2d(cnn_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 调整Transformer特征维度
        self.trans_proj = nn.Sequential(
            nn.Conv2d(trans_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 融合操作
        self.fusion = DF_Module(trans_channels, out_channels)
        # 注意力机制增强特征融合


    def forward(self, cnn_feat: torch.Tensor, trans_feat: torch.Tensor) -> torch.Tensor:
        # 确保特征图尺寸一致
        if cnn_feat.shape[2:] != trans_feat.shape[2:]:
            trans_feat = F.interpolate(trans_feat, size=cnn_feat.shape[2:], mode='bilinear', align_corners=False)

        cnn_proj = self.cnn_proj(cnn_feat)
        # 调整Transformer特征格式以适应LayerNorm
        #trans_feat_2d = trans_feat.permute(0, 2, 3, 1)
        trans_proj = self.trans_proj(trans_feat)

        # 特征拼接
        #fused = torch.cat([cnn_proj, trans_proj], dim=1)
        fused = self.fusion(cnn_proj, trans_proj)



        return fused


class ProjHead(nn.Module):
    """用于对比学习的投影头：将 B,C,H,W -> B,D,H,W (保持空间分辨率)"""

    def __init__(self, in_ch: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出 B, D, H, W
        return self.net(x)


@torch.no_grad()
def _normalize_l2(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    对称 InfoNCE。z1,z2: [N, D]，已归一化。返回标量 loss。
    若 N < 2，返回 0，避免无负样本导致不稳定。
    """
    if z1.size(0) < 2:
        return z1.new_zeros(())
    logits = (z1 @ z2.t()) / tau  # [N, N]
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    return loss


class PatchExpand(nn.Module):
    """特征扩展模块"""

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)

        return x


class FinalPatchExpand_X4(nn.Module):
    """4倍特征扩展模块"""

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        return x


class VSSLayer(nn.Module):
    """基于状态空间模型的层"""

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        # print(f"初始化VSSLayer，特征维度: {dim}，包含{depth}个VSSBlock")

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # 打印进入该层时的输入维度
        # print(f"\nVSSLayer 输入维度: {x.shape}")
        
        for block_idx, blk in enumerate(self.blocks, 1):
            # 打印当前block的输入维度
            # print(f"  VSSBlock {block_idx} 输入维度: {x.shape}")
            
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            
            # 打印当前block的输出维度
            # print(f"  VSSBlock {block_idx} 输出维度: {x.shape}")

        # 处理下采样并打印维度变化
        if self.downsample is not None:
            # print(f"下采样前维度: {x.shape}")
            x = self.downsample(x)
            # print(f"下采样后维度: {x.shape}")
        
        # 打印该层的最终输出维度
        # print(f"VSSLayer 输出维度: {x.shape}")
        return x


class LocalSupervision(nn.Module):
    """局部监督模块"""

    def __init__(self, in_channels=128, num_classes=6):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6())
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6())
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(in_channels, num_classes, kernel_size=1, dilation=1, stride=1, padding=0, bias=False)

    def forward(self, x, h, w):
        local1 = self.conv3(x)
        local2 = self.conv1(x)
        x = self.drop(local1 + local2)
        x = self.conv_out(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x


class MambaSegDecoder(nn.Module):
    """Mamba解码器"""

    def __init__(
            self,
            num_classes: int,
            encoder_channels: Union[Tuple[int, ...], List[int]] = None,
            decode_channels: int = 64,
            drop_path_rate: float = 0.2,
            d_state: int = 16,
    ):
        super().__init__()

        encoder_output_channels = encoder_channels
        self.num_classes = num_classes
        n_stages_encoder = len(encoder_output_channels)

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (n_stages_encoder - 1) * 2)]
        depths = [2, 2, 2, 2]

        stages = []
        expand_layers = []
        lsm_layers = []
        concat_back_dim = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder_output_channels[-s]
            input_features_skip = encoder_output_channels[-(s + 1)]
            expand_layers.append(PatchExpand(
                input_resolution=None,
                dim=input_features_below,
                dim_scale=2,
                norm_layer=nn.LayerNorm,
            ))
            stages.append(VSSLayer(
                dim=input_features_skip,
                depth=2,
                attn_drop=0.,
                drop_path=dpr[sum(depths[:s - 1]):sum(depths[:s])],
                d_state=math.ceil(2 * input_features_skip / 6) if d_state is None else d_state,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False,
            ))
            concat_back_dim.append(nn.Linear(2 * input_features_skip, input_features_skip))
            lsm_layers.append(LocalSupervision(encoder_channels[-(s + 1)], num_classes))

        expand_layers.append(FinalPatchExpand_X4(
            input_resolution=None,
            dim=encoder_output_channels[0],
            dim_scale=4,
            norm_layer=nn.LayerNorm,
        ))
        stages.append(nn.Identity())

        self.stages = nn.ModuleList(stages)
        self.expand_layers = nn.ModuleList(expand_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)
        if self.training:
            self.lsm = nn.ModuleList(lsm_layers)
        self.seg = nn.Conv2d(encoder_channels[-4], num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.init_weight()

    def forward(self, x, h, w):
        """
        Mamba解码器前向传播
        
        Args:
            x: 编码器输出特征列表 (经过CoordAtt处理后的f_out)
               x[0]: Stage1特征, shape [B, 64, H/4, W/4]
               x[1]: Stage2特征, shape [B, 128, H/8, W/8]
               x[2]: Stage3特征, shape [B, 256, H/16, W/16]
               x[3]: Stage4特征, shape [B, 512, H/32, W/32]
            h, w: 原始输入图像的高度和宽度，用于最终上采样
            
        Returns:
            训练模式: (seg_out, sum(ls)) - 分割输出和局部监督损失之和
            推理模式: seg_out - 仅分割输出
            
        解码流程 (从深到浅):
            Stage4(512) --PatchExpand--> Stage3(256) --PatchExpand--> Stage2(128) --PatchExpand--> Stage1(64) --FinalExpand4x--> Output
                              ↑ skip                    ↑ skip                    ↑ skip
        """
        
        # 从最深层特征开始解码 (x[-1] = Stage4, 分辨率最小, 通道最多)
        lres_input = x[-1]  # [B, 512, H/32, W/32]
        
        if self.training:
            ls = []  # 存储各阶段局部监督损失，用于深监督
            
            # 循环遍历解码器的各个阶段 (s=0,1,2,3 对应 512→256→128→64→output)
            for s in range(len(self.stages)):
                
                # ========== Step 1: Patch Expanding (上采样2倍) ==========
                # s=0: 512通道 → 256通道, 分辨率×2 (H/32 → H/16)
                # s=1: 256通道 → 128通道, 分辨率×2 (H/16 → H/8)
                # s=2: 128通道 → 64通道,  分辨率×2 (H/8 → H/4)
                # s=3: 64通道 → 64通道,   分辨率×4 (H/4 → H) [FinalPatchExpand_X4]
                x_expanded = self.expand_layers[s](lres_input)

                # ========== Step 2: Skip Connection (跳跃连接) ==========
                # 除了最后一个阶段(s=3)，其他阶段都需要融合编码器的跳跃连接
                if s < (len(self.stages) - 1):
                    # 获取对应的跳跃连接特征:
                    # s=0: x[-(0+2)] = x[-2] = x[2] = Stage3特征 (256通道)
                    # s=1: x[-(1+2)] = x[-3] = x[1] = Stage2特征 (128通道)
                    # s=2: x[-(2+2)] = x[-4] = x[0] = Stage1特征 (64通道)
                    skip_feat = x[-(s + 2)].permute(0, 2, 3, 1)  # [B,C,H,W] → [B,H,W,C]
                    
                    # 在通道维度拼接: 上采样特征 + 跳跃连接特征
                    x_expanded = torch.cat((x_expanded, skip_feat), -1)  # 通道数翻倍
                    
                    # 通过线性层降维回原通道数
                    x_expanded = self.concat_back_dim[s](x_expanded)
                    
                # ========== Step 3: VSS Block (状态空间模型处理) ==========
                # 通过Mamba的VSS层进行特征变换
                x_stage = self.stages[s](x_expanded)  # VSSLayer处理
                x_stage = x_stage.permute(0, 3, 1, 2)  # [B,H,W,C] → [B,C,H,W]

                # ========== Step 4: 输出/局部监督 ==========
                if s == (len(self.stages) - 1):
                    # 最后一个阶段: 生成最终分割预测
                    # x_stage: [B, 64, H, W] → seg_out: [B, num_classes, H, W]
                    seg_out = self.seg(x_stage)
                else:
                    # 中间阶段: 计算局部监督损失 (深监督策略)
                    # 将中间特征上采样到原图尺寸并计算辅助损失
                    ls.append(self.lsm[s](x_stage, h, w))
                    
                # 将当前阶段输出作为下一阶段的输入
                lres_input = x_stage

            # 返回: 主分割输出 + 局部监督损失之和
            return seg_out, sum(ls)

        else:
            # =============== 推理模式 (无局部监督) ===============
            for s in range(len(self.stages)):
                # Step 1: Patch Expanding
                x_expanded = self.expand_layers[s](lres_input)

                # Step 2: Skip Connection
                if s < (len(self.stages) - 1):
                    x_expanded = torch.cat((x_expanded, x[-(s + 2)].permute(0, 2, 3, 1)), -1)
                    x_expanded = self.concat_back_dim[s](x_expanded)
                    
                # Step 3: VSS Block
                x_stage = self.stages[s](x_expanded).permute(0, 3, 1, 2)

                # Step 4: 最终输出
                if s == (len(self.stages) - 1):
                    seg_out = self.seg(x_stage)
                    
                lres_input = x_stage

            return seg_out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class TCMDFMConvNeXtMamba(nn.Module):
    """
    TCMDFMConvNeXtMamba: 双流并行混合架构语义分割网络
    
    ======================= 网络架构总览 =======================
    
                          ┌─────────────────────────────────────────────────────────────────┐
                          │                        ENCODER (并行双流)                        │
                          │                                                                 │
       Input Image        │    ┌──────────────────────────────────────┐                    │
      [B,3,H,W]           │    │       ConvNeXt Tiny (CNN分支)         │                    │
           │              │    │  输出: [96, 192, 384, 768] @ 4个Stage │                    │
           ├──────────────┼───►│  擅长: 局部细节, 纹理, 边缘           │──┐                │
           │              │    └──────────────────────────────────────┘  │                │
           │              │                                              │   InfoNCE      │
           │              │                        ↓ (对比学习)          │   Loss         │
           │              │                                              │                │
           │              │    ┌──────────────────────────────────────┐  │                │
           │              │    │        ResT-Lite (Transformer分支)    │  │                │
           └──────────────┼───►│  输出: [64, 128, 256, 512] @ 4个Stage │──┘                │
                          │    │  擅长: 全局上下文, 长距离依赖          │                   │
                          │    └──────────────────────────────────────┘                    │
                          └─────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                          ┌─────────────────────────────────────────────────────────────────┐
                          │                      FUSION (特征融合)                          │
                          │                                                                 │
                          │   Stage 1-4: CNN特征 ──┬──► FeatureFusionModule ──► CoordAtt   │
                          │                        │        (DF_Module)          (注意力)   │
                          │              Trans特征 ─┘                                       │
                          │                                                                 │
                          │   输出: f_out = [64, 128, 256, 512] @ 4个Stage                 │
                          └─────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                          ┌─────────────────────────────────────────────────────────────────┐
                          │                      DECODER (Mamba解码器)                      │
                          │                                                                 │
                          │   VSS Block (状态空间模型) + Patch Expanding + Skip Connections │
                          │   深监督: 各阶段 LocalSupervision 输出                          │
                          │                                                                 │
                          │   输出: seg_out [B, num_classes, H, W]                         │
                          └─────────────────────────────────────────────────────────────────┘
    
    ======================= 核心创新点 =======================
    1. 双流并行编码: CNN (局部) + Transformer (全局) 互补
    2. Dense Feature Fusion (DF_Module): 加法+差分混合融合
    3. CoordAtt: 坐标注意力增强空间感知
    4. InfoNCE对比学习: 促进双流特征对齐
    5. Mamba解码器: 状态空间模型高效解码
    """

    def __init__(self,
                 pretrained_cnn=False,
                 pretrained_trans=True,
                 cnn_backbone_path=None,
                 trans_backbone_path='/root/autodl-fs/UnetMamba-main/UNetMamba/pretrain_weights/rest_lite.pth',
                 convnext_weight_path='pretrain_weights/convnext_tiny-983f1562.pth',
                 embed_dim=64,
                 decode_channels=64,
                 num_classes=6,
                 fusion_channels=[64,128,256,512],
                 debug_shapes=False,
                 # 对比学习（方案二）配置
                 contrast_enable=True,
                 contrast_proj_dim=128,
                 contrast_tau=0.07,
                 contrast_stage_weights=(0.1, 0.2, 0.3, 0.4),
                 contrast_pixel=False,
                 contrast_pixel_num=128,
                 **kwargs
                 ):
        """
        初始化 TCMDFMConvNeXtMamba 网络
        
        Args:
            pretrained_cnn: 是否加载预训练CNN权重
            pretrained_trans: 是否加载预训练Transformer权重
            trans_backbone_path: ResT-Lite预训练权重路径
            convnext_weight_path: ConvNeXt Tiny预训练权重路径
            embed_dim: 基础嵌入维度 (64), 各Stage通道数为 [1x, 2x, 4x, 8x]
            decode_channels: 解码器通道数
            num_classes: 分割类别数 (如Vaihingen数据集为6类)
            fusion_channels: 融合后各Stage的通道数 [64, 128, 256, 512]
            debug_shapes: 是否打印调试信息
            
            # 对比学习参数
            contrast_enable: 是否启用InfoNCE对比学习
            contrast_proj_dim: 投影头输出维度 (128)
            contrast_tau: 温度系数τ (0.07), 越小对比越尖锐
            contrast_stage_weights: 各Stage损失权重 (深层权重更大)
            contrast_pixel: 是否启用像素级对比
            contrast_pixel_num: 像素级采样数量
        """
        super().__init__()

        # ==================== 保存配置参数 ====================
        self.debug_shapes = debug_shapes
        self.contrast_enable = contrast_enable
        self.contrast_proj_dim = contrast_proj_dim
        self.contrast_tau = contrast_tau
        self.contrast_stage_weights = contrast_stage_weights
        self.contrast_pixel = contrast_pixel
        self.contrast_pixel_num = contrast_pixel_num
        self.last_contrastive_loss = None  # 存储最近一次的对比损失，供外部Loss使用

        # ==================== 1. 双流编码器初始化 ====================
        # CNN分支: ConvNeXt Tiny
        # - 现代CNN架构，融合Transformer设计理念
        # - Block数量: [3, 3, 9, 3], 输出通道: [96, 192, 384, 768]
        self.cnn_encoder = convnext_tiny_lite(pretrained=pretrained_cnn, weight_path=convnext_weight_path)
        
        # Transformer分支: ResT-Lite
        # - 高效Transformer，带空间降维注意力(SR-Attention)
        # - Block数量: [2, 2, 2, 2], 输出通道: [64, 128, 256, 512]
        self.trans_encoder = rest_lite(weight_path=trans_backbone_path, pretrained=pretrained_trans)

        # 获取编码器通道配置
        # ConvNeXt: [96, 192, 384, 768] - 比ResT通道数多50%
        # ResT:     [64, 128, 256, 512] - 基础通道配置
        cnn_channels = self.cnn_encoder.feature_info.channels()
        trans_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        # 验证两个编码器输出阶段数一致 (都是4阶段)
        assert len(cnn_channels) == len(trans_channels), "CNN和Transformer编码器输出阶段数不一致"
        self.n_stages = len(cnn_channels)

        if self.debug_shapes:
            print('[Debug] Encoder 通道配置:')
            print('  ConvNeXt channels:', cnn_channels)   # [96, 192, 384, 768]
            print('  Trans channels   :', trans_channels)  # [64, 128, 256, 512]
            print('  Fusion channels  :', fusion_channels) # [64, 128, 256, 512]

        # ==================== 2. 对比学习投影头 ====================
        # 用于InfoNCE损失，将CNN和Trans特征投影到统一的对比空间
        # 投影头结构: 1x1Conv -> BN -> ReLU -> 1x1Conv
        if self.contrast_enable:
            # CNN投影头: 96/192/384/768 -> 128
            self.proj_cnn = nn.ModuleList([
                ProjHead(in_ch=cnn_channels[i], out_dim=self.contrast_proj_dim)
                for i in range(self.n_stages)
            ])
            # Trans投影头: 64/128/256/512 -> 128
            self.proj_trans = nn.ModuleList([
                ProjHead(in_ch=trans_channels[i], out_dim=self.contrast_proj_dim)
                for i in range(self.n_stages)
            ])

        # ==================== 3. 特征融合模块 ====================
        # FeatureFusionModule 包含:
        #   - cnn_proj: 1x1 Conv 对齐CNN通道 (96->64, 192->128, ...)
        #   - trans_proj: 1x1 Conv 对齐Trans通道 (实际不变)
        #   - DF_Module: Dense Fusion (加法分支 + 差分分支)
        self.fusion_modules = nn.ModuleList([
            FeatureFusionModule(
                cnn_channels=cnn_channels[i],      # 输入: 96, 192, 384, 768
                trans_channels=trans_channels[i],   # 输入: 64, 128, 256, 512
                out_channels=fusion_channels[i]     # 输出: 64, 128, 256, 512
            )
            for i in range(self.n_stages)
        ])
        
        # ==================== 4. 坐标注意力模块 ====================
        # CoordAtt: 捕获跨通道信息的同时编码长距离空间依赖
        # 分别对H和W方向进行全局池化，再通过1x1卷积生成注意力权重
        self.ca_modules = nn.ModuleList([
            CoordAtt(
                inp=fusion_channels[i],  # 输入通道
                oup=fusion_channels[i]   # 输出通道 (保持不变)
            )
            for i in range(self.n_stages)
        ])

        # ==================== 5. Mamba解码器 ====================
        # 融合后的编码器输出通道 (与Trans一致)
        encoder_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]  # [64, 128, 256, 512]

        # MambaSegDecoder:
        #   - Patch Expanding: 上采样 (类似反卷积)
        #   - VSS Block: 状态空间模型，高效建模长序列
        #   - LocalSupervision: 深监督，各阶段辅助Loss
        self.decoder = MambaSegDecoder(
            num_classes=num_classes, 
            encoder_channels=encoder_channels, 
            decode_channels=decode_channels
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W], 如 [8, 3, 512, 512]
            
        Returns:
            训练模式: (seg_out, lsm)
                - seg_out: 分割预测 [B, num_classes, H, W]
                - lsm: 局部监督损失 (深监督)
            推理模式: seg_out
            
        数据流:
            Input -> [CNN | Trans] -> InfoNCE -> Fusion -> CoordAtt -> Decoder -> Output
        """
        # 保存原始尺寸，用于最终上采样
        h, w = x.size()[-2:]

        # ==================== Stage 1: 双流并行编码 ====================
        # 同一张图像分别送入CNN和Transformer，提取互补特征
        #
        # CNN (ConvNeXt):  擅长局部低级特征 (边缘、纹理、细节)
        # Trans (ResT):    擅长全局高级语义 (长距离关系、上下文)
        #
        # cnn_features[i] shape:
        #   i=0: [B, 96,  H/4,  W/4]
        #   i=1: [B, 192, H/8,  W/8]
        #   i=2: [B, 384, H/16, W/16]
        #   i=3: [B, 768, H/32, W/32]
        #
        # trans_features[i] shape:
        #   i=0: [B, 64,  H/4,  W/4]
        #   i=1: [B, 128, H/8,  W/8]
        #   i=2: [B, 256, H/16, W/16]
        #   i=3: [B, 512, H/32, W/32]
        cnn_features = self.cnn_encoder(x)
        trans_features = self.trans_encoder(x)

        if self.debug_shapes:
            print('[Debug] 输入图像尺寸:', tuple(x.shape))

        # ==================== Stage 2: 对比学习 (训练时) ====================
        # InfoNCE Loss: 拉近同一位置的CNN-Trans特征对，推远不同位置的特征
        # 目的: 促进双流特征对齐，增强融合效果
        #
        # 计算流程 (每个Stage):
        #   1. 投影: CNN/Trans特征 -> 128维向量
        #   2. 全局池化: [B,128,H,W] -> [B,128]
        #   3. L2归一化
        #   4. InfoNCE计算对比损失
        total_contrastive = None
        if self.training and self.contrast_enable:
            # 各Stage权重: 深层(语义丰富)赋予更高权重
            if isinstance(self.contrast_stage_weights, (list, tuple)) and len(self.contrast_stage_weights) == self.n_stages:
                stage_weights = list(self.contrast_stage_weights)  # [0.1, 0.2, 0.3, 0.4]
            else:
                stage_weights = [1.0 / self.n_stages] * self.n_stages  # 均匀权重兜底

            total_contrastive = x.new_zeros(())  # 初始化为0

            for i in range(self.n_stages):
                # Step 1: 投影到对比学习空间
                zc_map = self.proj_cnn[i](cnn_features[i])     # [B, 128, H/s, W/s]
                zt_map = self.proj_trans[i](trans_features[i])  # [B, 128, H/s, W/s]

                # Step 2: 全局平均池化 + 展平
                zc = F.adaptive_avg_pool2d(zc_map, 1).flatten(1)  # [B, 128]
                zt = F.adaptive_avg_pool2d(zt_map, 1).flatten(1)  # [B, 128]
                
                # Step 3: L2归一化 (对比学习标准做法)
                zc = _normalize_l2(zc, dim=1)
                zt = _normalize_l2(zt, dim=1)

                # Step 4: 计算全局InfoNCE Loss
                loss_glb = info_nce(zc, zt, tau=self.contrast_tau)
                loss_stage = loss_glb

                # [可选] 像素级对比: 采样K个空间位置进行细粒度对比
                if self.contrast_pixel and self.contrast_pixel_num > 0:
                    B, D, Hs, Ws = zc_map.shape
                    K = min(self.contrast_pixel_num, Hs * Ws)  # 采样数不超过总像素数
                    # 随机采样K个位置索引
                    idx = torch.randint(low=0, high=Hs * Ws, size=(B, K), device=zc_map.device)
                    
                    def gather_positions(feat_map: torch.Tensor) -> torch.Tensor:
                        """从特征图中采样指定位置的特征向量"""
                        fm = feat_map.view(B, D, -1)                   # [B, 128, H*W]
                        gather_idx = idx.unsqueeze(1).expand(B, D, K)  # [B, 128, K]
                        sel = torch.gather(fm, dim=2, index=gather_idx) # [B, 128, K]
                        sel = sel.permute(0, 2, 1).contiguous().view(B * K, D)  # [B*K, 128]
                        return _normalize_l2(sel, dim=1)
                    
                    zc_pix = gather_positions(zc_map)
                    zt_pix = gather_positions(zt_map)
                    loss_pix = info_nce(zc_pix, zt_pix, tau=self.contrast_tau)
                    loss_stage = loss_stage + loss_pix

                # 加权累加各Stage的对比损失
                total_contrastive = total_contrastive + stage_weights[i] * loss_stage

            # 保存对比损失，供外部loss函数使用
            self.last_contrastive_loss = total_contrastive

        # ==================== Stage 3: 特征融合 (Dense Fusion) ====================
        # 将CNN和Trans的多尺度特征通过DF_Module进行融合
        #
        # 融合过程 (FeatureFusionModule):
        #   1. cnn_proj: 1x1 Conv 对齐通道 (96->64, 192->128, ...)
        #   2. trans_proj: 1x1 Conv 对齐通道 (实际不变)
        #   3. DF_Module: 加法分支 + 差分分支 组合
        #
        # fused_features[i] shape: [B, 64/128/256/512, H/s, W/s]
        fused_features = []
        for i in range(self.n_stages):
            cnn_feat = cnn_features[i]     # ConvNeXt特征
            trans_feat = trans_features[i]  # ResT特征
            
            if self.debug_shapes:
                print(f'[Debug][Stage {i}] ConvNeXt特征: {tuple(cnn_feat.shape)}  Trans特征: {tuple(trans_feat.shape)}')
            
            # Dense Feature Fusion
            fused_feat = self.fusion_modules[i](cnn_feat, trans_feat)
            
            if self.debug_shapes:
                print(f'[Debug][Stage {i}] Fused特征: {tuple(fused_feat.shape)}')
            fused_features.append(fused_feat)

        # ==================== Stage 4: 坐标注意力增强 ====================
        # CoordAtt: 对融合后的特征进行空间注意力增强
        # 通过H/W方向的全局池化捕获长距离空间依赖
        #
        # f_out[i] shape: [B, 64/128/256/512, H/s, W/s] (与输入相同)
        f_out = []
        for i in range(self.n_stages):
            f_feat = self.ca_modules[i](fused_features[i])
            f_out.append(f_feat)
        
        # ==================== Stage 5: Mamba解码器 ====================
        # 将多尺度特征送入Mamba解码器进行上采样和分割预测
        #
        # 解码流程:
        #   f_out[3](512) --Expand--> f_out[2](256) --Expand--> f_out[1](128) --Expand--> f_out[0](64) --4x--> Output
        #                   ↑ skip                   ↑ skip                   ↑ skip
        if self.training:
            # 训练模式: 返回主输出 + 深监督损失
            seg_out, lsm = self.decoder(f_out, h, w)
            if self.debug_shapes:
                print('[Debug] Decoder seg输出:', tuple(seg_out.shape))
                if isinstance(lsm, torch.Tensor):
                    print('[Debug] Decoder lsm输出:', tuple(lsm.shape))
            return seg_out, lsm
        else:
            # 推理模式: 只返回分割结果
            seg_out = self.decoder(f_out, h, w)
            if self.debug_shapes:
                print('[Debug] Decoder seg输出:', tuple(seg_out.shape))
            return seg_out
        
        
        
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = TCMDFMConvNeXtMamba(
        pretrained_cnn=False, 
        pretrained_trans=False, 
        num_classes=6, 
        debug_shapes=True,
        convnext_weight_path='pretrain_weights/convnext_tiny-983f1562.pth'
    ).to(device)
    
    x = torch.randn(1, 3, 1024, 1024, device=device)
    
    # Eval path
    model.eval()
    with torch.no_grad():
        y = model(x)
    print('[Eval] seg shape:', tuple(y.shape))
    
    # Train path
    model.train()
    y_main, y_aux = model(x)
    print('[Train] y main:', tuple(y_main.shape), 'y aux:', None if y_aux is None else tuple(y_aux.shape))
    print("ConvNeXt替换成功!")
        
