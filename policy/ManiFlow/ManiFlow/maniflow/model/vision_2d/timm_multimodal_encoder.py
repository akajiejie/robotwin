"""
统一的多模态观测编码器，支持RGB图像、触觉传感器和低维状态
复用TimmEncoder的RGB处理逻辑和TimmTactileEncoder的触觉处理逻辑

功能说明:
    - RGB相机: 使用timm预训练模型(ResNet/ViT/SigLIP等)提取视觉特征
    - 触觉传感器: 使用TimmTactileEncoder提取触觉特征
    - 低维状态: 直接展平拼接
    - 最终输出: 所有模态特征按顺序拼接 (RGB -> 触觉 -> 低维)
"""
import copy
import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from typing import Optional, Tuple, Dict, Union
from termcolor import cprint

from maniflow.model.common.module_attr_mixin import ModuleAttrMixin
from maniflow.common.pytorch_util import replace_submodules
from maniflow.model.tactile.tactile_encoder import TimmTactileEncoder

logger = logging.getLogger(__name__)


class AttentionPool2d(nn.Module):
    """注意力池化层，用于RGB特征聚合"""
    
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class TimmMultimodalEncoder(ModuleAttrMixin):
    """
    多模态观测编码器，统一处理RGB、触觉和低维状态
    - RGB图像: 使用TimmEncoder的逻辑
    - 触觉传感器: 使用ResNet18 + SpatialSoftmax
    - 低维状态: 直接展平
    """
    
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            global_pool: str,
            transforms: list,
            # RGB相关参数
            use_group_norm: bool=False,
            share_rgb_model: bool=False,
            imagenet_norm: bool=False,
            feature_aggregation: str='spatial_embedding',
            downsample_ratio: int=32,
            position_encording: str='learnable',
            # 触觉相关参数
            tactile_model_name: str='resnet18',
            tactile_pretrained: bool=False,
            tactile_frozen: bool=False,
            tactile_feature_dim: int=64,  # 🔥 默认值改为64（与robomimic一致）
            tactile_num_kp: int=32,  # 🆕 SpatialSoftmax关键点数量
            tactile_crop_shape: Optional[Union[Tuple[int, int], Dict[str, Tuple[int, int]]]]=None,  # 🆕 触觉裁剪尺寸
            share_tactile_model: bool=False,
            tactile_output_all_patches: bool=False,  # 🔥 触觉是否输出所有patch tokens
            # 🆕 模态级MoE支持
            output_token_sequence: bool=False,
            head_grid_size: int=1,  # 🔥 Head相机的空间重采样网格大小 (NxN), 默认1=单token
        ):
        """
        Args:
            shape_meta: 数据形状元信息
            model_name: RGB编码器的模型名称
            pretrained: RGB模型是否使用预训练权重
            frozen: RGB模型是否冻结
            global_pool: 全局池化类型
            transforms: 数据增强变换列表
            feature_aggregation: RGB特征聚合方式
            downsample_ratio: RGB下采样比率
            tactile_model_name: 触觉编码器模型名称
            tactile_feature_dim: 触觉特征输出维度
            tactile_num_kp: SpatialSoftmax关键点数量（默认32，与robomimic一致）
            tactile_crop_shape: 触觉图像裁剪尺寸，可选
            share_tactile_model: 是否在多个触觉传感器间共享权重
            tactile_output_all_patches: 触觉是否输出所有patch tokens
            output_token_sequence: 是否输出token序列格式（用于模态级MoE）
            head_grid_size: Head相机的输出token网格大小，生成 head_grid_size^2 个tokens
        """
        super().__init__()
        
        # 分类观测key
        rgb_keys = []
        tactile_keys = []
        low_dim_keys = []
        key_shape_map = {}
        
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            obs_type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            
            if obs_type == 'rgb':
                # 区分RGB图像和触觉传感器
                if 'tactile' in key.lower():
                    tactile_keys.append(key)
                else:
                    rgb_keys.append(key)
            elif obs_type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
        
        rgb_keys = sorted(rgb_keys)
        tactile_keys = sorted(tactile_keys)
        low_dim_keys = sorted(low_dim_keys)
        
        cprint(f"RGB相机: {rgb_keys}", 'cyan')
        cprint(f"触觉传感器: {tactile_keys}", 'yellow')
        cprint(f"低维状态: {low_dim_keys}", 'green')
        
        # ============ RGB编码器初始化 ============
        rgb_model_map = nn.ModuleDict()
        rgb_transform_map = nn.ModuleDict()
        rgb_feature_dim = None
        
        if len(rgb_keys) > 0:
            assert global_pool == ''
            
            # 创建RGB模型
            if model_name == "r3m":
                from r3m import load_r3m
                rgb_base_model = load_r3m("resnet18", pretrained=pretrained)
                rgb_base_model.eval()
                cprint(f"使用R3M模型: {model_name}, pretrained={pretrained}", 'green')
            else:
                rgb_base_model = timm.create_model(
                    model_name=model_name,
                    pretrained=pretrained,
                    global_pool=global_pool,
                    num_classes=0
                )
            
            if frozen:
                assert pretrained
                for param in rgb_base_model.parameters():
                    param.requires_grad = False
            
            # 确定RGB特征维度
            if model_name.startswith('resnet'):
                if downsample_ratio == 32:
                    modules = list(rgb_base_model.children())[:-2]
                    rgb_base_model = nn.Sequential(*modules)
                    rgb_feature_dim = 512
                elif downsample_ratio == 16:
                    modules = list(rgb_base_model.children())[:-3]
                    rgb_base_model = nn.Sequential(*modules)
                    rgb_feature_dim = 256
                else:
                    raise NotImplementedError(f"不支持的下采样率: {downsample_ratio}")
            elif model_name.startswith('convnext'):
                if downsample_ratio == 32:
                    modules = list(rgb_base_model.children())[:-2]
                    rgb_base_model = nn.Sequential(*modules)
                    rgb_feature_dim = 1024
                else:
                    raise NotImplementedError(f"不支持的下采样率: {downsample_ratio}")
            elif model_name.startswith('r3m'):
                rgb_feature_dim = 512
            elif 'siglip' in model_name.lower():
                if 'base' in model_name:
                    rgb_feature_dim = 768
                elif 'large' in model_name:
                    rgb_feature_dim = 1024
                elif 'so400m' in model_name:
                    rgb_feature_dim = 1152
                else:
                    rgb_feature_dim = 768
                cprint(f"使用SigLIP模型: {model_name}, feature_dim={rgb_feature_dim}", 'green')
            
            # GroupNorm替换
            if use_group_norm and not pretrained:
                rgb_base_model = replace_submodules(
                    root_module=rgb_base_model,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                        num_channels=x.num_features)
                )
            
            # 获取RGB图像尺寸并创建数据增强
            image_shape = key_shape_map[rgb_keys[0]][1:]  # (H, W)
            if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
                if hasattr(transforms[0], 'type') and transforms[0].type == 'RandomCrop':
                    ratio = transforms[0].ratio
                    transforms = [
                        torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                        torchvision.transforms.Resize(size=image_shape[0], antialias=True)
                    ] + transforms[1:]
            transform = nn.Identity() if transforms is None else nn.Sequential(*transforms)
            
            # 为每个RGB相机分配模型和变换
            for key in rgb_keys:
                this_model = rgb_base_model if share_rgb_model else copy.deepcopy(rgb_base_model)
                rgb_model_map[key] = this_model
                rgb_transform_map[key] = transform
            
            # 初始化特征聚合模块
            feature_map_shape = [x // downsample_ratio for x in image_shape]
            self._init_rgb_aggregation(feature_aggregation, rgb_feature_dim, feature_map_shape, 
                                      position_encording, model_name)
        
        # ============ 触觉编码器初始化 ============
        tactile_encoder = None
        
        if len(tactile_keys) > 0:
            # 构造触觉专用的shape_meta（只需要obs，不需要action）
            tactile_shape_meta = {
                'obs': {k: v for k, v in obs_shape_meta.items() if k in tactile_keys}
            }
            
            tactile_encoder = TimmTactileEncoder(
                shape_meta=tactile_shape_meta,
                model_name=tactile_model_name,
                pretrained=tactile_pretrained,
                frozen=tactile_frozen,
                use_group_norm=use_group_norm,
                share_tactile_model=share_tactile_model,
                feature_dim=tactile_feature_dim,
                num_kp=tactile_num_kp,  # 🆕 传递关键点数量
                crop_shape=tactile_crop_shape,  # 🆕 传递裁剪尺寸
                output_all_patches=tactile_output_all_patches
            )
            
            cprint(f"✓ 触觉编码器: {tactile_encoder.tactile_keys}, "
                   f"特征维度={tactile_feature_dim}, num_kp={tactile_num_kp}, "
                   f"crop={tactile_crop_shape}, 共享权重={share_tactile_model}, "
                   f"输出patch tokens={tactile_output_all_patches}", 'green')
        
        # ============ 触觉投影层（用于维度对齐） ============
        self.left_rgb_keys = []
        self.right_rgb_keys = []
        self.left_tactile_keys = []
        self.right_tactile_keys = []
        
        # 区分左右手的RGB相机和触觉传感器（用于token序列模式）
        for key in rgb_keys:
            if 'left' in key.lower():
                self.left_rgb_keys.append(key)
            elif 'right' in key.lower():
                self.right_rgb_keys.append(key)
        
        for key in tactile_keys:
            if 'left' in key.lower():
                self.left_tactile_keys.append(key)
            elif 'right' in key.lower():
                self.right_tactile_keys.append(key)
        
        # 如果RGB和触觉特征维度不同，创建投影层
        if len(tactile_keys) > 0 and rgb_feature_dim is not None and tactile_feature_dim is not None:
            if rgb_feature_dim != tactile_feature_dim:
                if len(self.left_tactile_keys) > 0:
                    self.left_tactile_proj = nn.Linear(tactile_feature_dim, rgb_feature_dim)
                    cprint(f"✓ 左手触觉投影层: {tactile_feature_dim} -> {rgb_feature_dim}", 'yellow')
                
                if len(self.right_tactile_keys) > 0:
                    self.right_tactile_proj = nn.Linear(tactile_feature_dim, rgb_feature_dim)
                    cprint(f"✓ 右手触觉投影层: {tactile_feature_dim} -> {rgb_feature_dim}", 'yellow')
            else:
                cprint(f"✓ RGB和触觉特征维度相同({rgb_feature_dim})，无需投影", 'green')
        
        # 保存所有属性
        self.model_name = model_name
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.tactile_keys = tactile_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        
        self.rgb_model_map = rgb_model_map
        self.rgb_transform_map = rgb_transform_map
        self.rgb_feature_dim = rgb_feature_dim
        self.feature_aggregation = feature_aggregation
        
        self.tactile_encoder = tactile_encoder
        self.tactile_feature_dim = tactile_feature_dim
        self.tactile_num_kp = tactile_num_kp  # 🆕 保存新参数
        self.tactile_crop_shape = tactile_crop_shape  # 🆕 保存新参数
        
        self.share_rgb_model = share_rgb_model
        self.share_tactile_model = share_tactile_model
        
        # 🆕 模态级MoE支持
        self.output_token_sequence = output_token_sequence
        if output_token_sequence:
            cprint(f"✓ 启用token序列输出模式 (用于模态级MoE)", 'cyan')
            # 创建本体感知投影层（将低维状态投影到RGB特征维度）
            if len(low_dim_keys) > 0:
                total_low_dim = sum(key_shape_map[k][0] if len(key_shape_map[k]) == 1 
                                  else key_shape_map[k][-1] for k in low_dim_keys)
                self.proprio_proj = nn.Linear(total_low_dim, rgb_feature_dim)
                nn.init.xavier_uniform_(self.proprio_proj.weight)
                nn.init.zeros_(self.proprio_proj.bias)
                cprint(f"  ✓ 本体感知投影层: {total_low_dim} -> {rgb_feature_dim}", 'green')
        
        # 🆕 Head相机空间降采样配置
        self.head_grid_size = head_grid_size
        self.downsample_ratio = downsample_ratio
        if output_token_sequence and head_grid_size > 1:
            # 创建空间降采样层 (将特征图聚合为 NxN 个tokens)
            self.head_spatial_pool = nn.AdaptiveAvgPool2d((head_grid_size, head_grid_size))
            cprint(f"✓ Head相机空间重采样: {head_grid_size}x{head_grid_size} = {head_grid_size**2} tokens per timestep", 'cyan')
        elif output_token_sequence:
            cprint(f"✓ Head相机使用单token模式 (head_grid_size=1)", 'cyan')
        
        # 🔥 模态Drop配置（用于训练时的数据增强）
        # 设计理念：模拟真实机器人操作中的传感器失效场景
        self.modality_drop_config = {
            'head': 0.0,        # Head相机drop概率（模拟遮挡）
            'rgb_wrist': 0.0,   # Wrist相机drop概率（近距离视角）
            'tactile': 0.0,     # Tactile drop概率（接触信息）
            'proprio': 0.0,     # Proprio drop概率（本体感知）
        }
        cprint(f"✓ 模态Drop配置已初始化（默认关闭）", 'cyan')
        
        logger.info(f"多模态编码器参数量: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_rgb_aggregation(self, feature_aggregation, feature_dim, feature_map_shape, 
                              position_encording, model_name):
        """初始化RGB特征聚合模块"""
        # ViT模型的特殊处理
        if model_name.startswith('vit'):
            if feature_aggregation == 'all_tokens':
                pass
            elif feature_aggregation is not None:
                logger.warn(f'ViT使用CLS token，feature_aggregation ({feature_aggregation})被忽略')
                self.feature_aggregation = None
        
        # 创建聚合模块
        if feature_aggregation == 'soft_attention':
            self.rgb_attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif feature_aggregation == 'spatial_embedding':
            self.rgb_spatial_embedding = nn.Parameter(
                torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif feature_aggregation == 'transformer':
            if position_encording == 'learnable':
                self.rgb_position_embedding = nn.Parameter(
                    torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim))
            elif position_encording == 'sinusoidal':
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                pos_embed = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * 
                                    (-math.log(2 * num_features) / feature_dim))
                pos_embed[:, 0::2] = torch.sin(position * div_term)
                pos_embed[:, 1::2] = torch.cos(position * div_term)
                self.rgb_position_embedding = pos_embed
            self.rgb_aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4)
        elif feature_aggregation == 'attention_pool_2d':
            self.rgb_attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
    
    def aggregate_rgb_feature(self, feature):
        """聚合RGB特征"""
        if self.model_name == 'r3m':
            return feature
        
        # SigLIP/CLIP模型处理
        if 'siglip' in self.model_name.lower() or 'clip' in self.model_name.lower():
            if self.feature_aggregation == 'all_tokens':
                # 🔥 输出所有tokens: (B, N, D) 其中 N = num_patches
                return feature
            elif self.feature_aggregation == 'avg' or self.feature_aggregation is None:
                # 默认使用mean pooling
                return torch.mean(feature, dim=1)
            else:
                logger.warn(f'SigLIP/CLIP使用mean pooling作为默认聚合方式')
                return torch.mean(feature, dim=1)
        
        # ViT模型处理
        if self.model_name.startswith('vit'):
            if self.feature_aggregation == 'all_tokens':
                # 🔥 输出所有tokens: (B, 1+P, D) - CLS + patches
                return feature
            elif self.feature_aggregation is None or self.feature_aggregation == 'cls_token':
                return feature[:, 0, :]
            else:
                logger.warn(f'ViT使用CLS token作为默认聚合方式')
                return feature[:, 0, :]
        
        # ResNet处理
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.rgb_attention_pool_2d(feature)
        
        feature = torch.flatten(feature, start_dim=-2)  # B, C, H*W
        feature = torch.transpose(feature, 1, 2)  # B, H*W, C
        
        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1])
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1])
        elif self.feature_aggregation == 'soft_attention':
            weight = self.rgb_attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.rgb_spatial_embedding, dim=1)
        elif self.feature_aggregation == 'transformer':
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.rgb_position_embedding.device != feature.device:
                self.rgb_position_embedding = self.rgb_position_embedding.to(feature.device)
            feature_with_pos = torch.cat([zero_feature, feature], dim=1) + self.rgb_position_embedding
            feature_output = self.rgb_aggregation_transformer(feature_with_pos)
            return feature_output[:, 0]
        else:
            # feature_aggregation为None时，展平所有空间特征
            assert self.feature_aggregation is None
            return feature.reshape(feature.shape[0], -1)  # B, H*W*C
    
    def _extract_rgb_tokens(self, obs_dict, key, is_head_cam=False):
        """
        提取RGB图像的token表示（用于交叉注意力）
        
        Args:
            obs_dict: 观测字典
            key: RGB相机的key
            is_head_cam: 是否是Head相机（用于空间重采样）
            
        Returns:
            tokens: (B*T, N, D) - N为token数量
                   - Head相机 (head_grid_size > 1): N = head_grid_size^2 (空间重采样后)
                   - Head相机 (head_grid_size = 1): N = 1 (聚合后的单token)
                   - Wrist相机: N = 1 (聚合后的单token)
            batch_size: B
            time_steps: T
        """
        img = obs_dict[key]
        
        # 归一化
        if img.max() > 1.0:
            img = img / 255.0
        
        # 调整维度顺序: (B,T,H,W,C) -> (B,T,C,H,W)
        if img.shape[-1] == 3:
            if len(img.shape) == 5:
                img = img.permute(0, 1, 4, 2, 3)
            elif len(img.shape) == 4:
                img = img.permute(0, 3, 1, 2)
        
        B, T = img.shape[:2]
        img = img.reshape(B*T, *img.shape[2:])
        
        # Resize到期望尺寸
        if img.shape[1:] != self.key_shape_map[key]:
            target_H, target_W = self.key_shape_map[key][1], self.key_shape_map[key][2]
            img = F.interpolate(img, size=(target_H, target_W), 
                               mode='bilinear', align_corners=False)
        
        # 前向传播获取原始特征（不聚合）
        img = self.rgb_transform_map[key](img).to(self.device)
        img = img.float()
        raw_feature = self.rgb_model_map[key](img).to(self.device)
        
        # 🔥 核心逻辑：根据相机类型和grid_size决定token生成策略
        if is_head_cam and hasattr(self, 'head_spatial_pool') and self.head_grid_size > 1:
            # === Head相机空间重采样模式 (head_grid_size > 1) ===
            # 将特征转换为2D空间特征图，然后重采样为 NxN 个tokens
            
            if self.model_name.startswith('vit') or 'siglip' in self.model_name.lower():
                # ViT/SigLIP: 输出格式为 (B*T, 1+P, D) 或 (B*T, P, D)
                # 需要将patch tokens转换为空间特征图
                
                feature_dim = raw_feature.shape[-1]
                
                # 去掉CLS token（如果存在）
                if raw_feature.shape[1] == 197:  # ViT-Base: 1 CLS + 196 patches (14x14)
                    patch_tokens = raw_feature[:, 1:, :]  # (B*T, 196, D)
                    spatial_size = int(math.sqrt(patch_tokens.shape[1]))  # 14
                elif 'siglip' in self.model_name.lower():
                    # SigLIP输出: (B*T, P, D) 没有CLS token
                    patch_tokens = raw_feature
                    spatial_size = int(math.sqrt(patch_tokens.shape[1]))
                else:
                    # 其他情况：假设没有CLS token
                    patch_tokens = raw_feature
                    spatial_size = int(math.sqrt(patch_tokens.shape[1]))
                
                # 重塑为空间特征图: (B*T, P, D) -> (B*T, D, H, W)
                spatial_feature = patch_tokens.permute(0, 2, 1).reshape(
                    B*T, feature_dim, spatial_size, spatial_size
                )
                
            else:
                # CNN: 已经是空间特征图格式 (B*T, C, H, W)
                spatial_feature = raw_feature
            
            # 🔥 空间重采样: (B*T, D, H, W) -> (B*T, D, N, N)
            downsampled = self.head_spatial_pool(spatial_feature)  # (B*T, D, grid_size, grid_size)
            
            # 拉平为tokens: (B*T, D, N, N) -> (B*T, N^2, D)
            tokens = downsampled.flatten(2).permute(0, 2, 1)  # (B*T, grid_size^2, D)
            
        else:
            # === 单token模式（Wrist相机 或 Head相机grid_size=1） ===
            # 使用mean pooling聚合为单个token
            
            if self.model_name.startswith('vit') or 'siglip' in self.model_name.lower():
                # ViT/SigLIP: 直接对所有tokens做mean pooling
                tokens = torch.mean(raw_feature, dim=1, keepdim=True)  # (B*T, 1, D)
            else:
                # CNN: 先展平空间维度，再mean pooling
                tokens = torch.flatten(raw_feature, start_dim=-2)  # (B*T, C, H*W)
                tokens = torch.transpose(tokens, 1, 2)  # (B*T, H*W, C)
                tokens = torch.mean(tokens, dim=1, keepdim=True)  # (B*T, 1, C)
        
        return tokens, B, T
    
    def forward(self, obs_dict):
        """
        前向传播，统一处理所有模态
        
        Args:
            obs_dict: 观测字典，每个key对应(B, T, ...)的张量
            
        Returns:
            features: 拼接后的特征向量 (B, D_total) 或 token序列 (B, L_tokens, D)
        """
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # 🆕 Token序列模式（用于模态级MoE）
        if self.output_token_sequence:
            return self._forward_token_sequence(obs_dict, batch_size)
        
        # 原始模式：拼接所有特征为一个向量
        features = []
        
        # ============ 处理RGB图像 ============
        for key in self.rgb_keys:
            img = obs_dict[key]
            
            # 归一化
            if img.max() > 1.0:
                img = img / 255.0
            
            # 调整维度顺序: (B,T,H,W,C) -> (B,T,C,H,W)
            if img.shape[-1] == 3:
                if len(img.shape) == 5:
                    img = img.permute(0, 1, 4, 2, 3)
                elif len(img.shape) == 4:
                    img = img.permute(0, 3, 1, 2)
            
            B, T = img.shape[:2]
            assert B == batch_size
            img = img.reshape(B*T, *img.shape[2:])
            
            # Resize到期望尺寸
            if img.shape[1:] != self.key_shape_map[key]:
                target_H, target_W = self.key_shape_map[key][1], self.key_shape_map[key][2]
                img = F.interpolate(img, size=(target_H, target_W), 
                                   mode='bilinear', align_corners=False)
            
            # 前向传播
            img = self.rgb_transform_map[key](img).to(self.device)
            img = img.float()
            raw_feature = self.rgb_model_map[key](img).to(self.device)
            feature = self.aggregate_rgb_feature(raw_feature)
            
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))
        
        # ============ 处理触觉传感器 ============
        if self.tactile_encoder is not None and len(self.tactile_keys) > 0:
            tactile_obs = {k: obs_dict[k] for k in self.tactile_keys if k in obs_dict}
            tactile_features = self.tactile_encoder(tactile_obs)  # Dict[key, (B, T, D)] or (B, T*H*W, D)
            
            for key in self.tactile_keys:
                if key in tactile_features:
                    feat = tactile_features[key]  # (B, T, D) or (B, T*H*W, D)
                    features.append(feat.reshape(batch_size, -1))  # (B, T*D) or (B, T*H*W*D)
        
        # ============ 处理低维状态 ============
        for key in self.low_dim_keys:
            data = obs_dict[key].to(self.device)
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))
        
        # 拼接所有特征
        result = torch.cat(features, dim=-1)
        
        return result
    
    def _forward_token_sequence(self, obs_dict, batch_size):
        """
        🆕 输出token序列格式: (B, L_tokens, D)
        
        模态组织策略（各模态独立输出，支持模态Drop）:
        - head: head_cam tokens (可被Drop，模拟遮挡场景)
        - rgb_wrist: left_wrist_cam + right_wrist_cam tokens (低Drop率，近距离视角重要)
        - tactile: left_tactile + right_tactile tokens (极低Drop率，接触信息关键)
        - proprio: agent_pos tokens (可被Drop，强制学习视觉-触觉依赖)
        
        Drop策略设计理念:
        - Head相机: 在实际操作中容易被机械臂遮挡 → 高Drop率模拟遮挡
        - Wrist相机: 近距离视角，不易遮挡 → 低Drop率
        - Tactile: 接触时的关键信息 → 极低/不Drop
        - Proprio: 可能存在编码器噪声 → 中等Drop率
        
        Args:
            obs_dict: 观测字典
            batch_size: batch大小
            
        Returns:
            result: (B, L_tokens, D) token序列
        """
        head_tokens_list = []
        rgb_wrist_tokens_list = []  # 🔥 RGB相机tokens（分离出来）
        tactile_tokens_list = []     # 🔥 触觉tokens（分离出来）
        proprio_features_list = []
        
        # 获取时间步数（从任意观测中获取）
        time_steps = next(iter(obs_dict.values())).shape[1]
        
        # ============ 处理RGB图像 ============
        for key in self.rgb_keys:
            is_head_cam = 'head' in key.lower() or 'front' in key.lower()
            
            tokens, B, T = self._extract_rgb_tokens(obs_dict, key, is_head_cam)
            
            # 🔥 tokens已经是正确格式：
            # - Head相机 (grid_size > 1): (B*T, grid_size^2, D)
            # - Head相机 (grid_size = 1): (B*T, 1, D)
            # - Wrist相机: (B*T, 1, D)
            
            # 重塑为 (B, T*num_tokens, D)
            num_tokens_per_timestep = tokens.shape[1]  # grid_size^2 或 1
            token_seq = tokens.reshape(B, T * num_tokens_per_timestep, -1)  # (B, T*N, D)
            
            if is_head_cam:
                head_tokens_list.append(token_seq)
            else:
                # 🔥 分离RGB wrist相机
                rgb_wrist_tokens_list.append(token_seq)
        
        # ============ 处理触觉传感器 ============
        if self.tactile_encoder is not None and len(self.tactile_keys) > 0:
            tactile_obs = {k: obs_dict[k] for k in self.tactile_keys if k in obs_dict}
            tactile_features_dict = self.tactile_encoder(tactile_obs)
            
            for key in self.tactile_keys:
                if key in tactile_features_dict:
                    tact_tok = tactile_features_dict[key]  # (B, Q, D) - Q可以是T或T*H*W
                    
                    # 投影到RGB特征维度（如果需要）
                    if 'left' in key.lower() and hasattr(self, 'left_tactile_proj'):
                        tact_tok = self.left_tactile_proj(tact_tok)
                    elif 'right' in key.lower() and hasattr(self, 'right_tactile_proj'):
                        tact_tok = self.right_tactile_proj(tact_tok)
                    
                    # 触觉编码器保留时序维度，直接使用
                    # output_all_patches=True: (B, T*H*W, D) - 保留所有时间步的所有patch
                    # output_all_patches=False: (B, T, D) - 保留所有时间步，每个时间步1个token
                    tactile_tokens_list.append(tact_tok)
        
        # ============ 处理低维状态（本体感知） ============
        for key in self.low_dim_keys:
            data = obs_dict[key].to(self.device)
            B, T = data.shape[:2]
            assert B == batch_size
            proprio_features_list.append(data)  # (B, T, low_dim)
        
        # ============ 组装最终的token序列（应用模态Drop） ============
        all_tokens = []
        modality_info = {'head': 0, 'rgb_wrist': 0, 'tactile': 0, 'proprio': 0}
        
        # 🔥 Head tokens (高Drop率 - 模拟被机械臂遮挡的场景)
        # 实际操作中，头部相机经常被机械臂遮挡，模型需要学会在遮挡时依赖腕部相机和触觉
        if head_tokens_list:
            head_tokens = torch.cat(head_tokens_list, dim=1)  # (B, n_head_cams*T, D)
            head_tokens = self._apply_modality_drop(head_tokens, 'head')
            all_tokens.append(head_tokens)
            modality_info['head'] = head_tokens.shape[1]
        
        # 🔥 RGB Wrist tokens (低Drop率 - 近距离视角，不易遮挡)
        # 腕部相机贴近操作物体，提供关键的近距离视觉信息
        if rgb_wrist_tokens_list:
            rgb_wrist_tokens = torch.cat(rgb_wrist_tokens_list, dim=1)  # (B, n_wrist_cams*T, D)
            rgb_wrist_tokens = self._apply_modality_drop(rgb_wrist_tokens, 'rgb_wrist')
            all_tokens.append(rgb_wrist_tokens)
            modality_info['rgb_wrist'] = rgb_wrist_tokens.shape[1]
        
        # 🔥 Tactile tokens (极低Drop率 - 接触信息是兜底保障)
        # 触觉传感器提供直接接触信息，是视觉失效时的关键信息源
        if tactile_tokens_list:
            tactile_tokens = torch.cat(tactile_tokens_list, dim=1)  # (B, n_tactile*T, D)
            tactile_tokens = self._apply_modality_drop(tactile_tokens, 'tactile')
            all_tokens.append(tactile_tokens)
            modality_info['tactile'] = tactile_tokens.shape[1]
        
        # 🔥 Proprio tokens (中等Drop率 - 强制学习视觉-触觉依赖)
        # 本体感知可能存在编码器噪声，Drop后强制模型从视觉和触觉推断位置
        if proprio_features_list:
            proprio_concat = torch.cat(proprio_features_list, dim=-1)  # (B, T, total_low_dim)
            proprio_tokens = self.proprio_proj(proprio_concat.float())  # (B, T, D)
            proprio_tokens = self._apply_modality_drop(proprio_tokens, 'proprio')
            all_tokens.append(proprio_tokens)
            modality_info['proprio'] = proprio_tokens.shape[1]
        
        result = torch.cat(all_tokens, dim=1)  # (B, L_total, D)
        
        # 保存模态信息供外部使用
        self._last_modality_info = modality_info
        
        return result
    
    def get_modality_info(self):
        """
        🆕 获取最近一次forward的模态长度信息
        
        Returns:
            modality_info: dict {'head': L_head, 'wrist': L_wrist, 'proprio': L_proprio}
        """
        return getattr(self, '_last_modality_info', None)
    
    def set_modality_drop_config(self, head_drop=0.0, rgb_wrist_drop=0.0, tactile_drop=0.0, proprio_drop=0.0):
        """
        🔥 设置模态Drop概率
        
        设计理念：模拟真实机器人操作中的传感器失效场景
        - Head相机: 容易被机械臂遮挡 → 建议高Drop率 (0.2-0.3)
        - Wrist相机: 近距离视角，不易遮挡 → 建议低Drop率 (0.05-0.1)
        - Tactile: 接触信息关键 → 建议极低/不Drop (0.0-0.02)
        - Proprio: 可能有编码器噪声 → 建议中等Drop率 (0.1-0.15)
        
        Args:
            head_drop: Head相机的drop概率 (0.0-1.0)
            rgb_wrist_drop: Wrist相机的drop概率 (0.0-1.0)
            tactile_drop: Tactile传感器的drop概率 (0.0-1.0)
            proprio_drop: Proprio的drop概率 (0.0-1.0)
        """
        self.modality_drop_config = {
            'head': head_drop,
            'rgb_wrist': rgb_wrist_drop,
            'tactile': tactile_drop,
            'proprio': proprio_drop,
        }
        cprint(f"✓ 模态Drop配置已更新: head={head_drop}, rgb_wrist={rgb_wrist_drop}, "
               f"tactile={tactile_drop}, proprio={proprio_drop}", 'yellow')
    
    def _apply_modality_drop(self, tokens, modality_name):
        """
        🔥 对指定模态的tokens应用Drop（仅训练时）
        
        机制说明:
        - 训练时: 以指定概率将整个batch中的某些样本的该模态tokens全部置零
        - 推理时: 自动关闭（通过self.training判断）
        - 梯度流动: 置零的tokens仍然参与反向传播，梯度会流向未被Drop的模态
        
        Args:
            tokens: (B, L, D) 模态的token序列
            modality_name: 'proprio', 'rgb', 'tactile'
        
        Returns:
            tokens: Drop后的token序列（推理时返回原始tokens）
        """
        # 🔥 关键：只在训练模式下才Drop
        if not self.training:
            return tokens
        
        drop_prob = self.modality_drop_config.get(modality_name, 0.0)
        if drop_prob == 0.0:
            return tokens
        
        # 生成Drop mask（每个batch样本独立决定是否Drop）
        B = tokens.shape[0]
        # drop_mask: (B, 1, 1) - 对每个样本独立采样，广播到所有tokens和特征维度
        drop_mask = torch.bernoulli(
            torch.full((B, 1, 1), 1.0 - drop_prob, device=tokens.device, dtype=tokens.dtype)
        )
        
        # 应用mask（置零但保留梯度流动）
        tokens_dropped = tokens * drop_mask
        
        # 统计Drop信息（用于调试）
        if not hasattr(self, '_drop_stats'):
            self._drop_stats = {}
        dropped_samples = (drop_mask.squeeze() == 0).sum().item()
        self._drop_stats[modality_name] = {
            'dropped_samples': dropped_samples,
            'total_samples': B,
            'drop_rate': dropped_samples / B if B > 0 else 0.0
        }
        
        return tokens_dropped
    
    def get_drop_stats(self):
        """
        获取最近一次forward的Drop统计信息
        
        Returns:
            drop_stats: dict {modality_name: {'dropped_samples': int, 'total_samples': int, 'drop_rate': float}}
        """
        stats = getattr(self, '_drop_stats', {})
        # 清空统计信息
        self._drop_stats = {}
        return stats if stats else None
    
    @torch.no_grad()
    def output_shape(self):
        """计算输出特征的形状"""
        example_obs_dict = {}
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        
        example_output = self.forward(example_obs_dict)
        
        # 🆕 支持token序列模式
        if self.output_token_sequence:
            assert len(example_output.shape) == 3  # (B, L, D)
            assert example_output.shape[0] == 1
        else:
            assert len(example_output.shape) == 2  # (B, total_dim)
            assert example_output.shape[0] == 1
        
        return example_output.shape


if __name__ == '__main__':
    print("\n" + "="*80)
    cprint("TimmMultimodalEncoder 测试 (使用TimmTactileEncoder)", "cyan", attrs=["bold"])
    print("="*80 + "\n")
    
    # 构造shape_meta (RoboTwin2.0环境)
    shape_meta = {
        'obs': {
            'head_cam': {'shape': [3, 224, 224], 'type': 'rgb', 'horizon': 2},
            'left_wrist_cam': {'shape': [3, 224, 224], 'type': 'rgb', 'horizon': 2},
            'right_wrist_cam': {'shape': [3, 224, 224], 'type': 'rgb', 'horizon': 2},
            'left_tactile': {'shape': [1, 16, 32], 'type': 'rgb', 'horizon': 2},
            'right_tactile': {'shape': [1, 16, 32], 'type': 'rgb', 'horizon': 2},
            'agent_pos': {'shape': [14], 'type': 'low_dim', 'horizon': 2},
        },
        'action': {'shape': [14], 'horizon': 16}
    }
    
    # 创建编码器（标准模式）
    encoder = TimmMultimodalEncoder(
        shape_meta=shape_meta,
        model_name='resnet18',
        pretrained=False,
        frozen=False,
        global_pool='',
        transforms=None,
        use_group_norm=True,
        share_rgb_model=False,
        feature_aggregation=None,
        downsample_ratio=32,
        tactile_model_name='resnet18',
        tactile_pretrained=False,
        tactile_feature_dim=64,  # 🔥 与robomimic一致
        tactile_num_kp=32,       # 🆕 关键点数量
        tactile_crop_shape=None, # 🆕 不裁剪
        share_tactile_model=True,
    )
    
    # 创建token序列输出模式的编码器
    cprint("\n" + "="*80, "cyan")
    cprint("测试token序列输出模式（用于模态级MoE）", "cyan", attrs=["bold"])
    cprint("="*80, "cyan")
    
    encoder_token_seq = TimmMultimodalEncoder(
        shape_meta=shape_meta,
        model_name='resnet18',
        pretrained=False,
        frozen=False,
        global_pool='',
        transforms=None,
        use_group_norm=True,
        share_rgb_model=False,
        feature_aggregation='all_tokens',
        downsample_ratio=32,
        tactile_model_name='resnet18',
        tactile_pretrained=False,
        tactile_feature_dim=512,  # 与RGB特征维度对齐
        tactile_num_kp=32,        # 🆕 关键点数量
        tactile_crop_shape=None,  # 🆕 不裁剪
        share_tactile_model=True,
        tactile_output_all_patches=True,
        output_token_sequence=True,
    )
    
    print(f"\n模型信息:")
    print(f"  - RGB相机: {encoder.rgb_keys}")
    print(f"  - 触觉传感器: {encoder.tactile_keys}")
    print(f"  - 低维状态: {encoder.low_dim_keys}")
    print(f"  - RGB特征维度: {encoder.rgb_feature_dim}")
    print(f"  - 触觉特征维度: {encoder.tactile_feature_dim}")
    print(f"  - 总参数量: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # 测试前向传播
    batch_size, time_steps = 2, 2
    obs = {
        'head_cam': torch.randn(batch_size, time_steps, 3, 224, 224),
        'left_wrist_cam': torch.randn(batch_size, time_steps, 3, 224, 224),
        'right_wrist_cam': torch.randn(batch_size, time_steps, 3, 224, 224),
        'left_tactile': torch.randn(batch_size, time_steps, 1, 16, 32),
        'right_tactile': torch.randn(batch_size, time_steps, 1, 16, 32),
        'agent_pos': torch.randn(batch_size, time_steps, 14),
    }
    
    cprint("\n前向传播测试:", "yellow")
    with torch.no_grad():
        output = encoder(obs)
    
    print(f"  输出形状: {output.shape}")
    
    # 维度验证 (注意: TimmTactileEncoder保留时序维度)
    # 标准模式下tactile_feature_dim=64
    tactile_feat_dim = encoder.tactile_feature_dim
    rgb_dim = 3 * 512 * 7 * 7 * time_steps  # 3相机 × 512特征 × 7×7 × 2T
    tactile_dim = 2 * tactile_feat_dim * time_steps  # 2传感器 × feature_dim × 2T
    lowdim_dim = 14 * time_steps  # 14维 × 2T
    expected_dim = rgb_dim + tactile_dim + lowdim_dim
    
    print(f"  预期维度: {expected_dim} (RGB:{rgb_dim} + 触觉:{tactile_dim} + 低维:{lowdim_dim})")
    assert output.shape == (batch_size, expected_dim), f"维度不匹配! {output.shape} != ({batch_size}, {expected_dim})"
    
    # 梯度测试
    cprint("\n梯度反向传播测试:", "yellow")
    obs_grad = {k: v.clone().requires_grad_(True) for k, v in obs.items()}
    output = encoder(obs_grad)
    loss = output.sum()
    loss.backward()
    
    print(f"  left_tactile 梯度范数: {obs_grad['left_tactile'].grad.norm().item():.6f}")
    print(f"  head_cam 梯度范数: {obs_grad['head_cam'].grad.norm().item():.6f}")
    
    print("\n" + "="*80)
    cprint("✅ 标准版本测试通过!", "green", attrs=["bold"])
    print("="*80 + "\n")
    
    # 测试token序列输出版本
    cprint("\n前向传播测试（token序列输出）:", "yellow")
    with torch.no_grad():
        output_token_seq = encoder_token_seq(obs)
    
    print(f"  输出形状: {output_token_seq.shape}")
    modality_info = encoder_token_seq.get_modality_info()
    if modality_info:
        print(f"  模态信息: {modality_info}")
    
    # 梯度测试
    cprint("\n梯度反向传播测试（token序列输出）:", "yellow")
    obs_grad2 = {k: v.clone().requires_grad_(True) for k, v in obs.items()}
    output2 = encoder_token_seq(obs_grad2)
    loss2 = output2.sum()
    loss2.backward()
    
    print(f"  left_tactile 梯度范数: {obs_grad2['left_tactile'].grad.norm().item():.6f}")
    print(f"  left_wrist_cam 梯度范数: {obs_grad2['left_wrist_cam'].grad.norm().item():.6f}")
    
    print("\n" + "="*80)
    cprint("✅ 所有测试通过! Token序列输出功能正常", "green", attrs=["bold"])
    print("="*80 + "\n")

