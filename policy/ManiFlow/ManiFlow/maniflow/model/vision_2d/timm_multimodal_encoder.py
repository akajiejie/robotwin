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
class CrossAttentionBlock(nn.Module):
    """
    图像-触觉交叉注意力模块，支持多种注意力模式
    
    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        dropout: dropout比率
        attention_type: 注意力类型，支持：
            - 'cls': 只使用CLS token进行交叉注意力（适合ViT等有CLS token的模型）
            - 'avg': 使用平均池化的image token进行交叉注意力（计算高效）
            - 'all_patch': 使用所有patch tokens进行交叉注意力（最细粒度，计算量大）
            - 'hybrid': CLS token更新 + 部分patch tokens与触觉交互
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.0, attention_type='cls'):
        super().__init__()
        self.attention_type = attention_type
        cprint(f"✓ 初始化CrossAttentionBlock: attention_type={attention_type}, "
               f"embed_dim={embed_dim}, num_heads={num_heads}", 'cyan')
        
        # 触觉→图像的注意力
        self.attn_t2i = nn.MultiheadAttention(embed_dim, num_heads, 
                                              dropout=dropout, batch_first=True)
        # 图像→触觉的注意力
        self.attn_i2t = nn.MultiheadAttention(embed_dim, num_heads,
                                              dropout=dropout, batch_first=True)
        
        # LayerNorm
        self.ln_tact = nn.LayerNorm(embed_dim)
        self.ln_img = nn.LayerNorm(embed_dim)
        
        # 如果是avg模式，需要一个额外的投影层来处理聚合后的token
        if attention_type == 'avg':
            self.img_token_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, image_tokens, tactile_tokens):
        """
        前向传播
        
        Args:
            image_tokens: (B, 1+P, D) - 图像tokens，第一个是CLS token，其余是patch tokens
            tactile_tokens: (B, Q, D) - 触觉tokens
            
        Returns:
            image_tokens: (B, 1+P, D) - 更新后的图像tokens
            tactile_tokens: (B, Q, D) - 更新后的触觉tokens
            attn_weights: 注意力权重（用于可视化）
        """
        B, N, D = image_tokens.shape  # N = 1(CLS) + P(patches)
        cls_tok = image_tokens[:, :1, :]   # (B, 1, D)
        patch_tok = image_tokens[:, 1:, :]  # (B, P, D)
        
        if self.attention_type == "cls":
            # ========== CLS模式：只使用CLS token进行交叉注意力 ==========
            # 1) 触觉 → CLS：让触觉信息关注全局视觉特征
            tact_out, attn_weights = self.attn_t2i(
                query=tactile_tokens,   # (B, Q, D)
                key=cls_tok,            # (B, 1, D)
                value=cls_tok,          # (B, 1, D)
                need_weights=True,
                average_attn_weights=False
            )
            tactile_tokens = self.ln_tact(tactile_tokens + tact_out)

            # 2) CLS → 触觉：让全局视觉特征融合触觉信息
            img_out, _ = self.attn_i2t(
                query=cls_tok,          # (B, 1, D)
                key=tactile_tokens,     # (B, Q, D)
                value=tactile_tokens,   # (B, Q, D)
            )
            cls_tok = self.ln_img(cls_tok + img_out)

            # 3) 重组：patch tokens保持不变
            image_tokens = torch.cat([cls_tok, patch_tok], dim=1)
            
        elif self.attention_type == "avg":
            # ========== AVG模式：使用平均池化的image token ==========
            # 1) 平均池化所有tokens（包括CLS和patches）
            avg_img_tok = torch.mean(image_tokens, dim=1, keepdim=True)  # (B, 1, D)
            avg_img_tok = self.img_token_proj(avg_img_tok)  # 可学习的投影
            
            # 2) 触觉 → 平均图像token
            tact_out, attn_weights = self.attn_t2i(
                query=tactile_tokens,   # (B, Q, D)
                key=avg_img_tok,        # (B, 1, D)
                value=avg_img_tok,      # (B, 1, D)
                need_weights=True,
                average_attn_weights=False
            )
            tactile_tokens = self.ln_tact(tactile_tokens + tact_out)
            
            # 3) 平均图像token → 触觉
            img_out, _ = self.attn_i2t(
                query=avg_img_tok,      # (B, 1, D)
                key=tactile_tokens,     # (B, Q, D)
                value=tactile_tokens,   # (B, Q, D)
            )
            avg_img_tok = self.ln_img(avg_img_tok + img_out)
            
            # 4) 将更新后的信息广播回所有tokens（简单相加）
            image_tokens = image_tokens + avg_img_tok
            
        elif self.attention_type == "all_patch":
            # ========== ALL_PATCH模式：使用所有patch tokens（不含CLS） ==========
            # 1) 触觉 → 所有patches：细粒度的空间交互
            tact_out, attn_weights = self.attn_t2i(
                query=tactile_tokens,   # (B, Q, D)
                key=patch_tok,          # (B, P, D)
                value=patch_tok,        # (B, P, D)
                need_weights=True,
                average_attn_weights=False,
            )
            tactile_tokens = self.ln_tact(tactile_tokens + tact_out)

            # 2) 所有patches → 触觉：让每个patch都能感知触觉信息
            patch_out, _ = self.attn_i2t(
                query=patch_tok,        # (B, P, D)
                key=tactile_tokens,     # (B, Q, D)
                value=tactile_tokens,   # (B, Q, D)
            )
            patch_tok = self.ln_img(patch_tok + patch_out)

            # 3) 重组：CLS保持不变，更新后的patches拼接回去
            image_tokens = torch.cat([cls_tok, patch_tok], dim=1)
            
        elif self.attention_type == "hybrid":
            # ========== HYBRID模式：CLS + Patches都参与交叉注意力 ==========
            # 1) 触觉 → 所有图像tokens（CLS + patches）
            tact_out, attn_weights = self.attn_t2i(
                query=tactile_tokens,   # (B, Q, D)
                key=image_tokens,       # (B, 1+P, D)
                value=image_tokens,     # (B, 1+P, D)
                need_weights=True,
                average_attn_weights=False,
            )
            tactile_tokens = self.ln_tact(tactile_tokens + tact_out)

            # 2) 所有图像tokens → 触觉
            img_out, _ = self.attn_i2t(
                query=image_tokens,     # (B, 1+P, D)
                key=tactile_tokens,     # (B, Q, D)
                value=tactile_tokens,   # (B, Q, D)
            )
            image_tokens = self.ln_img(image_tokens + img_out)
            
        else:
            raise ValueError(f"不支持的attention_type: {self.attention_type}. "
                           f"支持的类型: ['cls', 'avg', 'all_patch', 'hybrid']")

        return image_tokens, tactile_tokens, attn_weights

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
            tactile_feature_dim: int=512,
            share_tactile_model: bool=False,
            # 交叉注意力参数
            use_cross_attention: bool=True,
            cross_attention_type: str='cls',
            cross_attention_num_heads: int=8,
            cross_attention_dropout: float=0.0,
            # 🆕 模态级MoE支持
            output_token_sequence: bool=False,
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
            share_tactile_model: 是否在多个触觉传感器间共享权重
            use_cross_attention: 是否使用图像-触觉交叉注意力
            cross_attention_type: 交叉注意力类型 ('cls', 'avg', 'all_patch', 'hybrid')
            cross_attention_num_heads: 交叉注意力的头数
            cross_attention_dropout: 交叉注意力的dropout比率
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
                feature_dim=tactile_feature_dim
            )
            
            cprint(f"✓ 触觉编码器: {tactile_encoder.tactile_keys}, "
                   f"特征维度={tactile_feature_dim}, 共享权重={share_tactile_model}", 'green')
        
        # ============ 交叉注意力初始化 ============
        self.use_cross_attention = use_cross_attention
        self.cross_attention_left = None
        self.cross_attention_right = None
        self.left_rgb_keys = []
        self.right_rgb_keys = []
        self.left_tactile_keys = []
        self.right_tactile_keys = []
        
        if use_cross_attention and len(rgb_keys) > 0 and len(tactile_keys) > 0:
            # 区分左右手的RGB相机和触觉传感器
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
            
            cprint(f"左手RGB相机: {self.left_rgb_keys}", 'magenta')
            cprint(f"右手RGB相机: {self.right_rgb_keys}", 'magenta')
            cprint(f"左手触觉传感器: {self.left_tactile_keys}", 'magenta')
            cprint(f"右手触觉传感器: {self.right_tactile_keys}", 'magenta')
            
            # 确定特征维度（需要统一RGB和触觉的特征维度）
            # 如果维度不同，需要添加投影层
            assert rgb_feature_dim is not None and tactile_feature_dim is not None
            
            # 创建左手交叉注意力模块
            if len(self.left_rgb_keys) > 0 and len(self.left_tactile_keys) > 0:
                # 如果维度不同，添加投影层
                if rgb_feature_dim != tactile_feature_dim:
                    self.left_tactile_proj = nn.Linear(tactile_feature_dim, rgb_feature_dim)
                    cprint(f"✓ 左手触觉投影层: {tactile_feature_dim} -> {rgb_feature_dim}", 'yellow')
                else:
                    self.left_tactile_proj = nn.Identity()
                
                self.cross_attention_left = CrossAttentionBlock(
                    embed_dim=rgb_feature_dim,
                    num_heads=cross_attention_num_heads,
                    dropout=cross_attention_dropout,
                    attention_type=cross_attention_type
                )
                cprint(f"✓ 左手交叉注意力已创建: {cross_attention_type} 模式", 'green')
            
            # 创建右手交叉注意力模块
            if len(self.right_rgb_keys) > 0 and len(self.right_tactile_keys) > 0:
                # 如果维度不同，添加投影层
                if rgb_feature_dim != tactile_feature_dim:
                    self.right_tactile_proj = nn.Linear(tactile_feature_dim, rgb_feature_dim)
                    cprint(f"✓ 右手触觉投影层: {tactile_feature_dim} -> {rgb_feature_dim}", 'yellow')
                else:
                    self.right_tactile_proj = nn.Identity()
                
                self.cross_attention_right = CrossAttentionBlock(
                    embed_dim=rgb_feature_dim,
                    num_heads=cross_attention_num_heads,
                    dropout=cross_attention_dropout,
                    attention_type=cross_attention_type
                )
                cprint(f"✓ 右手交叉注意力已创建: {cross_attention_type} 模式", 'green')
        
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
        
        # SigLIP模型处理
        if 'siglip' in self.model_name.lower():
            if self.feature_aggregation == 'avg' or self.feature_aggregation is None:
                return torch.mean(feature, dim=1)
            elif self.feature_aggregation == 'all_tokens':
                return feature
            else:
                logger.warn(f'SigLIP使用mean pooling')
                return torch.mean(feature, dim=1)
        
        # ViT模型处理
        if self.model_name.startswith('vit'):
            if self.feature_aggregation is None or self.feature_aggregation == 'cls_token':
                return feature[:, 0, :]
            elif self.feature_aggregation == 'all_tokens':
                return feature
            else:
                logger.warn(f'ViT使用CLS token')
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
    
    def _extract_rgb_tokens(self, obs_dict, key):
        """
        提取RGB图像的token表示（用于交叉注意力）
        
        Args:
            obs_dict: 观测字典
            key: RGB相机的key
            
        Returns:
            tokens: (B*T, N, D) - N为token数量（1+P for ViT, H*W for CNN）
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
        
        # 转换为token格式
        if self.model_name.startswith('vit') or 'siglip' in self.model_name.lower():
            # ViT/SigLIP: 已经是token格式 (B*T, N, D)
            tokens = raw_feature
        else:
            # CNN: 需要转换 (B*T, C, H, W) -> (B*T, H*W, C)
            # 添加一个虚拟的CLS token
            tokens = torch.flatten(raw_feature, start_dim=-2)  # (B*T, C, H*W)
            tokens = torch.transpose(tokens, 1, 2)  # (B*T, H*W, C)
            # 添加CLS token (均值池化)
            cls_token = torch.mean(tokens, dim=1, keepdim=True)  # (B*T, 1, C)
            tokens = torch.cat([cls_token, tokens], dim=1)  # (B*T, 1+H*W, C)
        
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
        
        # ============ 处理RGB图像和触觉传感器（支持交叉注意力） ============
        if self.use_cross_attention and self.tactile_encoder is not None:
            # 先提取所有触觉特征的token表示
            tactile_obs = {k: obs_dict[k] for k in self.tactile_keys if k in obs_dict}
            tactile_features_dict = self.tactile_encoder.forward_tokens(tactile_obs) if hasattr(self.tactile_encoder, 'forward_tokens') else {}
            
            # 如果触觉编码器没有forward_tokens方法，使用普通forward
            if not tactile_features_dict:
                tactile_features_dict = self.tactile_encoder(tactile_obs)  # Dict[key, (B, 1, D)]
                # 转换为token格式
                for k, v in tactile_features_dict.items():
                    if len(v.shape) == 2:
                        v = v.unsqueeze(1)  # (B, D) -> (B, 1, D)
                    tactile_features_dict[k] = v
            
            # ========== 左手交叉注意力 ==========
            if self.cross_attention_left is not None and len(self.left_rgb_keys) > 0 and len(self.left_tactile_keys) > 0:
                # 提取左手RGB tokens
                left_rgb_tokens_list = []
                for key in self.left_rgb_keys:
                    tokens, B, T = self._extract_rgb_tokens(obs_dict, key)
                    left_rgb_tokens_list.append(tokens)
                
                # 合并左手RGB tokens（简单拼接）
                left_rgb_tokens = torch.cat(left_rgb_tokens_list, dim=1)  # (B*T, N_total, D)
                
                # 提取左手触觉tokens
                left_tactile_tokens_list = []
                for key in self.left_tactile_keys:
                    if key in tactile_features_dict:
                        tact_tok = tactile_features_dict[key]  # (B, Q, D)
                        # 扩展到时间维度
                        tact_tok = tact_tok.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, Q, D)
                        tact_tok = tact_tok.reshape(B*T, -1, tact_tok.shape[-1])  # (B*T, Q, D)
                        # 投影到RGB特征维度
                        tact_tok = self.left_tactile_proj(tact_tok)
                        left_tactile_tokens_list.append(tact_tok)
                
                if len(left_tactile_tokens_list) > 0:
                    left_tactile_tokens = torch.cat(left_tactile_tokens_list, dim=1)  # (B*T, Q_total, D)
                    
                    # 应用交叉注意力
                    left_rgb_tokens, left_tactile_tokens, _ = self.cross_attention_left(
                        left_rgb_tokens, left_tactile_tokens
                    )
                    
                    # 聚合左手RGB特征
                    left_rgb_feature = self.aggregate_rgb_feature(left_rgb_tokens)
                    if len(left_rgb_feature.shape) == 2:
                        features.append(left_rgb_feature.reshape(B, -1))
                    else:
                        features.append(left_rgb_feature.reshape(B, -1))
                    
                    # 聚合左手触觉特征
                    left_tactile_feature = torch.mean(left_tactile_tokens, dim=1)  # (B*T, D)
                    features.append(left_tactile_feature.reshape(B, -1))
            
            # ========== 右手交叉注意力 ==========
            if self.cross_attention_right is not None and len(self.right_rgb_keys) > 0 and len(self.right_tactile_keys) > 0:
                # 提取右手RGB tokens
                right_rgb_tokens_list = []
                for key in self.right_rgb_keys:
                    tokens, B, T = self._extract_rgb_tokens(obs_dict, key)
                    right_rgb_tokens_list.append(tokens)
                
                # 合并右手RGB tokens
                right_rgb_tokens = torch.cat(right_rgb_tokens_list, dim=1)  # (B*T, N_total, D)
                
                # 提取右手触觉tokens
                right_tactile_tokens_list = []
                for key in self.right_tactile_keys:
                    if key in tactile_features_dict:
                        tact_tok = tactile_features_dict[key]  # (B, Q, D)
                        # 扩展到时间维度
                        tact_tok = tact_tok.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, Q, D)
                        tact_tok = tact_tok.reshape(B*T, -1, tact_tok.shape[-1])  # (B*T, Q, D)
                        # 投影到RGB特征维度
                        tact_tok = self.right_tactile_proj(tact_tok)
                        right_tactile_tokens_list.append(tact_tok)
                
                if len(right_tactile_tokens_list) > 0:
                    right_tactile_tokens = torch.cat(right_tactile_tokens_list, dim=1)  # (B*T, Q_total, D)
                    
                    # 应用交叉注意力
                    right_rgb_tokens, right_tactile_tokens, _ = self.cross_attention_right(
                        right_rgb_tokens, right_tactile_tokens
                    )
                    
                    # 聚合右手RGB特征
                    right_rgb_feature = self.aggregate_rgb_feature(right_rgb_tokens)
                    if len(right_rgb_feature.shape) == 2:
                        features.append(right_rgb_feature.reshape(B, -1))
                    else:
                        features.append(right_rgb_feature.reshape(B, -1))
                    
                    # 聚合右手触觉特征
                    right_tactile_feature = torch.mean(right_tactile_tokens, dim=1)  # (B*T, D)
                    features.append(right_tactile_feature.reshape(B, -1))
            
            # ========== 处理其他RGB相机（没有配对触觉的） ==========
            other_rgb_keys = [k for k in self.rgb_keys 
                            if k not in self.left_rgb_keys and k not in self.right_rgb_keys]
            
            for key in other_rgb_keys:
                img = obs_dict[key]
                
                # 归一化
                if img.max() > 1.0:
                    img = img / 255.0
                
                # 调整维度顺序
                if img.shape[-1] == 3:
                    if len(img.shape) == 5:
                        img = img.permute(0, 1, 4, 2, 3)
                    elif len(img.shape) == 4:
                        img = img.permute(0, 3, 1, 2)
                
                B, T = img.shape[:2]
                img = img.reshape(B*T, *img.shape[2:])
                
                # Resize
                if img.shape[1:] != self.key_shape_map[key]:
                    target_H, target_W = self.key_shape_map[key][1], self.key_shape_map[key][2]
                    img = F.interpolate(img, size=(target_H, target_W), 
                                       mode='bilinear', align_corners=False)
                
                # 前向传播
                img = self.rgb_transform_map[key](img).to(self.device)
                img = img.float()
                raw_feature = self.rgb_model_map[key](img).to(self.device)
                feature = self.aggregate_rgb_feature(raw_feature)
                
                features.append(feature.reshape(B, -1))
            
            # ========== 处理其他触觉传感器（没有配对RGB的） ==========
            other_tactile_keys = [k for k in self.tactile_keys 
                                 if k not in self.left_tactile_keys and k not in self.right_tactile_keys]
            
            for key in other_tactile_keys:
                if key in tactile_features_dict:
                    feat = tactile_features_dict[key]  # (B, Q, D)
                    feat = torch.mean(feat, dim=1)  # (B, D)
                    features.append(feat.reshape(batch_size, -1))
        
        else:
            # ============ 不使用交叉注意力的标准处理 ============
            # 处理RGB图像
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
            
            # 处理触觉传感器
            if self.tactile_encoder is not None and len(self.tactile_keys) > 0:
                tactile_obs = {k: obs_dict[k] for k in self.tactile_keys if k in obs_dict}
                tactile_features = self.tactile_encoder(tactile_obs)  # Dict[key, (B, 1, D)]
                
                for key in self.tactile_keys:
                    if key in tactile_features:
                        feat = tactile_features[key]  # (B, 1, D)
                        features.append(feat.reshape(batch_size, -1))  # (B, D)
        
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
        
        模态组织策略（触觉融入腕部）:
        - head: head_cam tokens (如果有)
        - wrist: left_wrist_cam + right_wrist_cam + 对应触觉传感器的tokens
                (通过交叉注意力已融合，体现腕部视觉+触觉的完整感知)
        - proprio: agent_pos tokens (投影到RGB特征维度)
        
        Args:
            obs_dict: 观测字典
            batch_size: batch大小
            
        Returns:
            result: (B, L_tokens, D) token序列
        """
        head_tokens_list = []
        wrist_tokens_list = []
        proprio_features_list = []
        
        # 获取时间步数（从任意观测中获取）
        time_steps = next(iter(obs_dict.values())).shape[1]
        
        # ============ 处理RGB图像（使用交叉注意力融合触觉） ============
        if self.use_cross_attention and self.tactile_encoder is not None and len(self.tactile_keys) > 0:
            # 先提取所有触觉特征的token表示
            tactile_obs = {k: obs_dict[k] for k in self.tactile_keys if k in obs_dict}
            tactile_features_dict = self.tactile_encoder.forward_tokens(tactile_obs) if hasattr(self.tactile_encoder, 'forward_tokens') else {}
            
            # 如果没有forward_tokens方法，使用普通forward
            if not tactile_features_dict:
                tactile_features_dict = self.tactile_encoder(tactile_obs)
                for k, v in tactile_features_dict.items():
                    if len(v.shape) == 2:
                        v = v.unsqueeze(1)  # (B, D) -> (B, 1, D)
                    tactile_features_dict[k] = v
            
            # ========== 左手: 腕部相机 + 触觉（交叉注意力融合） ==========
            if self.cross_attention_left is not None and len(self.left_rgb_keys) > 0 and len(self.left_tactile_keys) > 0:
                # 提取左手RGB tokens
                left_rgb_tokens_list = []
                for key in self.left_rgb_keys:
                    tokens, B, T = self._extract_rgb_tokens(obs_dict, key)
                    left_rgb_tokens_list.append(tokens)
                
                left_rgb_tokens = torch.cat(left_rgb_tokens_list, dim=1)  # (B*T, N, D)
                
                # 提取左手触觉tokens
                left_tactile_tokens_list = []
                for key in self.left_tactile_keys:
                    if key in tactile_features_dict:
                        tact_tok = tactile_features_dict[key]  # (B, Q, D)
                        tact_tok = tact_tok.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, Q, D)
                        tact_tok = tact_tok.reshape(B*T, -1, tact_tok.shape[-1])  # (B*T, Q, D)
                        tact_tok = self.left_tactile_proj(tact_tok)
                        left_tactile_tokens_list.append(tact_tok)
                
                if len(left_tactile_tokens_list) > 0:
                    left_tactile_tokens = torch.cat(left_tactile_tokens_list, dim=1)  # (B*T, Q, D)
                    
                    # 🔥 交叉注意力：腕部视觉 ↔ 触觉
                    left_rgb_tokens, left_tactile_tokens, _ = self.cross_attention_left(
                        left_rgb_tokens, left_tactile_tokens
                    )
                    
                    # 聚合为每个时间步一个token (mean pooling)
                    left_rgb_token_agg = torch.mean(left_rgb_tokens, dim=1)  # (B*T, D)
                    left_tactile_token_agg = torch.mean(left_tactile_tokens, dim=1)  # (B*T, D)
                    
                    # 合并腕部+触觉：拼接后再投影，或直接相加（相加更简洁）
                    left_wrist_fused = (left_rgb_token_agg + left_tactile_token_agg) / 2  # (B*T, D)
                    left_wrist_fused = left_wrist_fused.reshape(B, T, -1)  # (B, T, D)
                    
                    wrist_tokens_list.append(left_wrist_fused)
            
            # ========== 右手: 腕部相机 + 触觉（交叉注意力融合） ==========
            if self.cross_attention_right is not None and len(self.right_rgb_keys) > 0 and len(self.right_tactile_keys) > 0:
                # 提取右手RGB tokens
                right_rgb_tokens_list = []
                for key in self.right_rgb_keys:
                    tokens, B, T = self._extract_rgb_tokens(obs_dict, key)
                    right_rgb_tokens_list.append(tokens)
                
                right_rgb_tokens = torch.cat(right_rgb_tokens_list, dim=1)  # (B*T, N, D)
                
                # 提取右手触觉tokens
                right_tactile_tokens_list = []
                for key in self.right_tactile_keys:
                    if key in tactile_features_dict:
                        tact_tok = tactile_features_dict[key]  # (B, Q, D)
                        tact_tok = tact_tok.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, Q, D)
                        tact_tok = tact_tok.reshape(B*T, -1, tact_tok.shape[-1])  # (B*T, Q, D)
                        tact_tok = self.right_tactile_proj(tact_tok)
                        right_tactile_tokens_list.append(tact_tok)
                
                if len(right_tactile_tokens_list) > 0:
                    right_tactile_tokens = torch.cat(right_tactile_tokens_list, dim=1)  # (B*T, Q, D)
                    
                    # 🔥 交叉注意力：腕部视觉 ↔ 触觉
                    right_rgb_tokens, right_tactile_tokens, _ = self.cross_attention_right(
                        right_rgb_tokens, right_tactile_tokens
                    )
                    
                    # 聚合为每个时间步一个token
                    right_rgb_token_agg = torch.mean(right_rgb_tokens, dim=1)  # (B*T, D)
                    right_tactile_token_agg = torch.mean(right_tactile_tokens, dim=1)  # (B*T, D)
                    
                    # 合并腕部+触觉
                    right_wrist_fused = (right_rgb_token_agg + right_tactile_token_agg) / 2  # (B*T, D)
                    right_wrist_fused = right_wrist_fused.reshape(B, T, -1)  # (B, T, D)
                    
                    wrist_tokens_list.append(right_wrist_fused)
            
            # ========== 处理其他RGB相机（没有配对触觉的，如head_cam） ==========
            other_rgb_keys = [k for k in self.rgb_keys 
                            if k not in self.left_rgb_keys and k not in self.right_rgb_keys]
            
            for key in other_rgb_keys:
                # 判断是否为头部相机
                is_head_cam = 'head' in key.lower() or 'front' in key.lower()
                
                tokens, B, T = self._extract_rgb_tokens(obs_dict, key)
                # 聚合为每个时间步一个token
                token_agg = torch.mean(tokens, dim=1)  # (B*T, D)
                token_agg = token_agg.reshape(B, T, -1)  # (B, T, D)
                
                if is_head_cam:
                    head_tokens_list.append(token_agg)
                else:
                    wrist_tokens_list.append(token_agg)
        
        else:
            # ============ 不使用交叉注意力的标准处理 ============
            for key in self.rgb_keys:
                is_head_cam = 'head' in key.lower() or 'front' in key.lower()
                
                tokens, B, T = self._extract_rgb_tokens(obs_dict, key)
                token_agg = torch.mean(tokens, dim=1)  # (B*T, D)
                token_agg = token_agg.reshape(B, T, -1)  # (B, T, D)
                
                if is_head_cam:
                    head_tokens_list.append(token_agg)
                else:
                    wrist_tokens_list.append(token_agg)
        
        # ============ 处理低维状态（本体感知） ============
        for key in self.low_dim_keys:
            data = obs_dict[key].to(self.device)
            B, T = data.shape[:2]
            assert B == batch_size
            proprio_features_list.append(data)  # (B, T, low_dim)
        
        # ============ 组装最终的token序列 ============
        all_tokens = []
        modality_info = {'head': 0, 'wrist': 0, 'proprio': 0}
        
        # Head tokens
        if head_tokens_list:
            head_tokens = torch.cat(head_tokens_list, dim=1)  # (B, n_head_cams*T, D)
            all_tokens.append(head_tokens)
            modality_info['head'] = head_tokens.shape[1]
        
        # Wrist tokens (包含融合后的触觉信息)
        if wrist_tokens_list:
            wrist_tokens = torch.cat(wrist_tokens_list, dim=1)  # (B, n_wrist_cams*T, D)
            all_tokens.append(wrist_tokens)
            modality_info['wrist'] = wrist_tokens.shape[1]
        
        # Proprio tokens (投影到RGB特征维度)
        if proprio_features_list:
            proprio_concat = torch.cat(proprio_features_list, dim=-1)  # (B, T, total_low_dim)
            proprio_tokens = self.proprio_proj(proprio_concat.float())  # (B, T, D)
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
    
    # 创建编码器（不使用交叉注意力）
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
        tactile_feature_dim=512,
        share_tactile_model=True,
        use_cross_attention=False,
    )
    
    # 创建使用交叉注意力的编码器
    cprint("\n" + "="*80, "cyan")
    cprint("测试交叉注意力版本", "cyan", attrs=["bold"])
    cprint("="*80, "cyan")
    
    encoder_with_attn = TimmMultimodalEncoder(
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
        tactile_feature_dim=512,
        share_tactile_model=True,
        # 交叉注意力参数
        use_cross_attention=True,
        cross_attention_type='cls',
        cross_attention_num_heads=8,
        cross_attention_dropout=0.0,
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
    
    # 维度验证 (注意: TimmTactileEncoder对时序维度求平均)
    rgb_dim = 3 * 512 * 7 * 7 * time_steps  # 3相机 × 512特征 × 7×7 × 2T
    tactile_dim = 2 * 512  # 2传感器 × 512特征 (TimmTactileEncoder已对时序求平均)
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
    
    # 测试交叉注意力版本
    cprint("\n前向传播测试（交叉注意力）:", "yellow")
    with torch.no_grad():
        output_with_attn = encoder_with_attn(obs)
    
    print(f"  输出形状: {output_with_attn.shape}")
    print(f"  左手配对: {encoder_with_attn.left_rgb_keys} <-> {encoder_with_attn.left_tactile_keys}")
    print(f"  右手配对: {encoder_with_attn.right_rgb_keys} <-> {encoder_with_attn.right_tactile_keys}")
    
    # 梯度测试
    cprint("\n梯度反向传播测试（交叉注意力）:", "yellow")
    obs_grad2 = {k: v.clone().requires_grad_(True) for k, v in obs.items()}
    output2 = encoder_with_attn(obs_grad2)
    loss2 = output2.sum()
    loss2.backward()
    
    print(f"  left_tactile 梯度范数: {obs_grad2['left_tactile'].grad.norm().item():.6f}")
    print(f"  left_wrist_cam 梯度范数: {obs_grad2['left_wrist_cam'].grad.norm().item():.6f}")
    
    print("\n" + "="*80)
    cprint("✅ 所有测试通过! 交叉注意力功能正常", "green", attrs=["bold"])
    print("="*80 + "\n")

