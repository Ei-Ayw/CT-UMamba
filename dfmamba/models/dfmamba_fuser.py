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
        lres_input = x[-1]
        if self.training:
            ls = []
            for s in range(len(self.stages)):
                x_expanded = self.expand_layers[s](lres_input)

                if s < (len(self.stages) - 1):
                    x_expanded = torch.cat((x_expanded, x[-(s + 2)].permute(0, 2, 3, 1)), -1)
                    x_expanded = self.concat_back_dim[s](x_expanded)
                x_stage = self.stages[s](x_expanded).permute(0, 3, 1, 2)

                if s == (len(self.stages) - 1):
                    seg_out = self.seg(x_stage)
                else:
                    ls.append(self.lsm[s](x_stage, h, w))
                lres_input = x_stage

            return seg_out, sum(ls)

        else:
            for s in range(len(self.stages)):
                x_expanded = self.expand_layers[s](lres_input)

                if s < (len(self.stages) - 1):
                    x_expanded = torch.cat((x_expanded, x[-(s + 2)].permute(0, 2, 3, 1)), -1)
                    x_expanded = self.concat_back_dim[s](x_expanded)
                x_stage = self.stages[s](x_expanded).permute(0, 3, 1, 2)

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
    """集成ConvNeXt和ResT的并行架构"""

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
                 contrast_enable=False,
                 contrast_proj_dim=128,
                 contrast_tau=0.07,
                 contrast_stage_weights=(0.1, 0.2, 0.3, 0.4),
                 contrast_pixel=False,
                 contrast_pixel_num=128,
                 **kwargs
                 ):
        super().__init__()

        self.debug_shapes = debug_shapes
        self.contrast_enable = contrast_enable
        self.contrast_proj_dim = contrast_proj_dim
        self.contrast_tau = contrast_tau
        self.contrast_stage_weights = contrast_stage_weights
        self.contrast_pixel = contrast_pixel
        self.contrast_pixel_num = contrast_pixel_num
        self.last_contrastive_loss = None

        # 初始化并行的CNN和Transformer编码器
        # 使用ConvNeXt替代ResNet18
        self.cnn_encoder = convnext_tiny_lite(pretrained=pretrained_cnn, weight_path=convnext_weight_path)
        self.trans_encoder = rest_lite(weight_path=trans_backbone_path, pretrained=pretrained_trans)

        # ConvNeXt Tiny编码器输出通道: [96, 192, 384, 768]
        # Transformer编码器输出通道: [64, 128, 256, 512]
        cnn_channels = self.cnn_encoder.feature_info.channels()
        trans_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        # 确保两个编码器输出阶段数一致
        assert len(cnn_channels) == len(trans_channels), "CNN和Transformer编码器输出阶段数不一致"
        self.n_stages = len(cnn_channels)

        if self.debug_shapes:
            print('[Debug] Encoder 通道配置:')
            print('  ConvNeXt channels:', cnn_channels)
            print('  Trans channels   :', trans_channels)
            print('  Fusion channels  :', fusion_channels)

        # 对比学习：每阶段投影头（保持空间分辨率，后续根据需要做 GAP 或像素采样）
        if self.contrast_enable:
            self.proj_cnn = nn.ModuleList([
                ProjHead(in_ch=cnn_channels[i], out_dim=self.contrast_proj_dim)
                for i in range(self.n_stages)
            ])
            self.proj_trans = nn.ModuleList([
                ProjHead(in_ch=trans_channels[i], out_dim=self.contrast_proj_dim)
                for i in range(self.n_stages)
            ])

        # 特征融合模块
        self.fusion_modules = nn.ModuleList([
            FeatureFusionModule(
                cnn_channels=cnn_channels[i],
                trans_channels=trans_channels[i],
                out_channels=fusion_channels[i]
            )
            for i in range(self.n_stages)
        ])

        # 准备融合后的编码器通道列表
        encoder_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        # 初始化解码器
        self.decoder = MambaSegDecoder(num_classes=num_classes, encoder_channels=encoder_channels, decode_channels=decode_channels)

    def forward(self, x):
        h, w = x.size()[-2:]

        # 并行前向传播
        cnn_features = self.cnn_encoder(x)
        trans_features = self.trans_encoder(x)

        if self.debug_shapes:
            print('[Debug] 输入图像尺寸:', tuple(x.shape))

        # 方案二：对比学习损失（InfoNCE，对称；可选像素级）
        total_contrastive = None
        if self.training and self.contrast_enable:
            # 兼容 stage 权重长度
            if isinstance(self.contrast_stage_weights, (list, tuple)) and len(self.contrast_stage_weights) == self.n_stages:
                stage_weights = list(self.contrast_stage_weights)
            else:
                # 均匀权重兜底
                stage_weights = [1.0 / self.n_stages] * self.n_stages

            total_contrastive = x.new_zeros(())

            for i in range(self.n_stages):
                # 投影并做全局 InfoNCE
                zc_map = self.proj_cnn[i](cnn_features[i])   # B,D,H,W
                zt_map = self.proj_trans[i](trans_features[i])

                zc = F.adaptive_avg_pool2d(zc_map, 1).flatten(1)
                zt = F.adaptive_avg_pool2d(zt_map, 1).flatten(1)
                zc = _normalize_l2(zc, dim=1)
                zt = _normalize_l2(zt, dim=1)

                loss_glb = info_nce(zc, zt, tau=self.contrast_tau)
                loss_stage = loss_glb

                # 可选：像素级 InfoNCE（采样 K 个位置）
                if self.contrast_pixel and self.contrast_pixel_num > 0:
                    B, D, Hs, Ws = zc_map.shape
                    K = min(self.contrast_pixel_num, Hs * Ws)
                    # 统一采样索引（每图 K 个点）
                    idx = torch.randint(low=0, high=Hs * Ws, size=(B, K), device=zc_map.device)
                    def gather_positions(feat_map: torch.Tensor) -> torch.Tensor:
                        fm = feat_map.view(B, D, -1)                 # B,D,L
                        gather_idx = idx.unsqueeze(1).expand(B, D, K) # B,D,K
                        sel = torch.gather(fm, dim=2, index=gather_idx)  # B,D,K
                        sel = sel.permute(0, 2, 1).contiguous().view(B * K, D)  # (B*K),D
                        return _normalize_l2(sel, dim=1)
                    zc_pix = gather_positions(zc_map)
                    zt_pix = gather_positions(zt_map)
                    loss_pix = info_nce(zc_pix, zt_pix, tau=self.contrast_tau)
                    loss_stage = loss_stage + loss_pix

                total_contrastive = total_contrastive + stage_weights[i] * loss_stage

            self.last_contrastive_loss = total_contrastive

        # 特征融合
        fused_features = []
        for i in range(self.n_stages):
            cnn_feat = cnn_features[i]
            trans_feat = trans_features[i]
            if self.debug_shapes:
                print(f'[Debug][Stage {i}] ConvNeXt特征: {tuple(cnn_feat.shape)}  Trans特征: {tuple(trans_feat.shape)}')
            fused_feat = self.fusion_modules[i](cnn_feat, trans_feat)
            if self.debug_shapes:
                print(f'[Debug][Stage {i}] Fused特征: {tuple(fused_feat.shape)}')
            fused_features.append(fused_feat)

        # 送入解码器
        if self.training:
            seg_out, lsm = self.decoder(fused_features, h, w)
            if self.debug_shapes:
                print('[Debug] Decoder seg输出:', tuple(seg_out.shape))
                if isinstance(lsm, torch.Tensor):
                    print('[Debug] Decoder lsm输出:', tuple(lsm.shape))
            return seg_out, lsm
        else:
            seg_out = self.decoder(fused_features, h, w)
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
        
