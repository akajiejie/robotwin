# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ManiFlow: https://github.com/geyan21/ManiFlow_Policy
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional
from termcolor import cprint

from maniflow.model.common.module_attr_mixin import ModuleAttrMixin
from maniflow.model.vision_2d.timm_encoder import TimmEncoder
from maniflow.model.tactile.tactile_encoder import TimmTactileEncoder

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """Cross Attention模块，用于融合视觉和触觉特征"""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, visual_tokens: torch.Tensor, tactile_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_tokens: [B, 1, D] - 视觉特征token
            tactile_tokens: [B, 1, D] - 触觉特征token
        Returns:
            fused_tokens: [B, 1, D] - 融合后的特征token
        """
        # Cross attention: query=visual, key/value=tactile
        attn_out, _ = self.cross_attn(
            query=visual_tokens,
            key=tactile_tokens,
            value=tactile_tokens
        )
        # Residual connection + LayerNorm
        fused = self.norm(visual_tokens + self.dropout(attn_out))
        return fused


class MultiModalObsEncoder(ModuleAttrMixin):
    """
    多模态观测编码器：融合多视角RGB图像和触觉输入
    
    架构：
    - 头部相机 -> TimmEncoder -> head_feature (仅图像特征)
    - 腕部相机 -> TimmEncoder -> wrist_visual_feature (仅图像特征)
    - 腕部触觉 -> TactileEncoder -> wrist_tactile_feature
    - wrist_visual + wrist_tactile -> CrossAttention -> wrist_fused_feature
    - 最终输出: concat(head_feature, wrist_fused_feature, low_dim_feature)
    """
    
    def __init__(
        self,
        shape_meta: dict,
        # RGB编码器配置
        rgb_model_name: str = 'resnet18',
        rgb_pretrained: bool = False,
        rgb_frozen: bool = False,
        rgb_global_pool: str = '',
        rgb_transforms: Optional[list] = None,
        rgb_use_group_norm: bool = True,
        rgb_share_model: bool = False,
        rgb_feature_aggregation: str = 'attention_pool_2d',
        rgb_downsample_ratio: int = 32,
        # 触觉编码器配置
        tactile_model_name: str = 'resnet18',
        tactile_pretrained: bool = False,
        tactile_frozen: bool = False,
        tactile_use_group_norm: bool = True,
        tactile_share_model: bool = True,
        tactile_feature_dim: int = 768,
        # Cross Attention配置
        fusion_num_heads: int = 8,
        fusion_dropout: float = 0.1,
    ):
        """
        Args:
            shape_meta: 数据shape元信息
            rgb_*: RGB图像编码器的配置参数
            tactile_*: 触觉编码器的配置参数
            fusion_*: Cross Attention融合模块的配置参数
        """
        super().__init__()
        
        # 识别观测模态
        obs_shape_meta = shape_meta['obs']
        self.head_keys = []      # 头部相机
        self.wrist_keys = []     # 腕部相机
        self.tactile_keys = []   # 触觉传感器
        self.low_dim_keys = []   # 低维状态
        
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                if 'tactile' in key.lower():
                    self.tactile_keys.append(key)
                elif 'wrist' in key.lower():
                    self.wrist_keys.append(key)
                elif 'head' in key.lower() or 'top' in key.lower():
                    self.head_keys.append(key)
                else:
                    # 默认归类为wrist
                    self.wrist_keys.append(key)
            elif obs_type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    self.low_dim_keys.append(key)
        
        self.head_keys = sorted(self.head_keys)
        self.wrist_keys = sorted(self.wrist_keys)
        self.tactile_keys = sorted(self.tactile_keys)
        self.low_dim_keys = sorted(self.low_dim_keys)
        
        cprint(f"[MultiModalObsEncoder] Head cameras: {self.head_keys}", 'cyan')
        cprint(f"[MultiModalObsEncoder] Wrist cameras: {self.wrist_keys}", 'cyan')
        cprint(f"[MultiModalObsEncoder] Tactile sensors: {self.tactile_keys}", 'cyan')
        cprint(f"[MultiModalObsEncoder] Low-dim states: {self.low_dim_keys}", 'cyan')
        
        # 创建RGB图像编码器的shape_meta（只包含RGB相机）
        rgb_shape_meta = {
            'obs': {k: v for k, v in obs_shape_meta.items() 
                    if k in self.head_keys or k in self.wrist_keys},
            'action': shape_meta['action']
        }
        
        # 初始化RGB编码器（只输出图像特征，不拼接low_dim）
        self.rgb_encoder = TimmEncoder(
            shape_meta=rgb_shape_meta,
            model_name=rgb_model_name,
            pretrained=rgb_pretrained,
            frozen=rgb_frozen,
            global_pool=rgb_global_pool,
            transforms=rgb_transforms,
            use_group_norm=rgb_use_group_norm,
            share_rgb_model=rgb_share_model,
            feature_aggregation=rgb_feature_aggregation,
            downsample_ratio=rgb_downsample_ratio,
        )
        
        # 获取RGB特征维度
        self.rgb_feature_dim = self.rgb_encoder.feature_dim
        
        # 初始化触觉编码器（如果有触觉传感器）
        self.tactile_encoder = None
        if len(self.tactile_keys) > 0:
            self.tactile_encoder = TimmTactileEncoder(
                shape_meta=shape_meta,
                model_name=tactile_model_name,
                pretrained=tactile_pretrained,
                frozen=tactile_frozen,
                use_group_norm=tactile_use_group_norm,
                share_tactile_model=tactile_share_model,
                feature_dim=tactile_feature_dim,
            )
            
            # 确保触觉特征维度与RGB特征维度一致
            if tactile_feature_dim != self.rgb_feature_dim:
                cprint(f"[Warning] Tactile feature dim ({tactile_feature_dim}) != RGB feature dim ({self.rgb_feature_dim}). "
                       f"Will use projection.", 'yellow')
                self.tactile_projection = nn.Linear(tactile_feature_dim, self.rgb_feature_dim)
            else:
                self.tactile_projection = None
        
        # 为每个腕部相机创建Cross Attention融合模块
        self.wrist_fusion_modules = nn.ModuleDict()
        for wrist_key in self.wrist_keys:
            # 匹配对应的触觉传感器（例如：left_wrist_cam <-> left_tactile）
            tactile_key = self._find_matching_tactile(wrist_key)
            if tactile_key is not None:
                self.wrist_fusion_modules[wrist_key] = CrossAttentionFusion(
                    embed_dim=self.rgb_feature_dim,
                    num_heads=fusion_num_heads,
                    dropout=fusion_dropout
                )
                cprint(f"[MultiModalObsEncoder] Created fusion module: {wrist_key} + {tactile_key}", 'green')
        
        self.shape_meta = shape_meta
        
        logger.info(
            f"MultiModalObsEncoder initialized with {sum(p.numel() for p in self.parameters()):,} parameters"
        )
    
    def _find_matching_tactile(self, wrist_key: str) -> Optional[str]:
        """根据腕部相机key查找对应的触觉传感器key"""
        wrist_key_lower = wrist_key.lower()
        
        # 提取方向标识（left/right）
        for direction in ['left', 'right']:
            if direction in wrist_key_lower:
                # 在触觉传感器中查找相同方向的
                for tactile_key in self.tactile_keys:
                    if direction in tactile_key.lower():
                        return tactile_key
        
        # 如果没有找到匹配，返回第一个触觉传感器（如果存在）
        if len(self.tactile_keys) > 0:
            logger.warning(f"No matching tactile sensor found for {wrist_key}, using {self.tactile_keys[0]}")
            return self.tactile_keys[0]
        
        return None
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs_dict: 观测字典，包含所有模态的输入
        Returns:
            fused_feature: [B, D] 融合后的特征向量
        """
        batch_size = next(iter(obs_dict.values())).shape[0]
        features = []
        
        # 1. 处理RGB图像（头部+腕部）
        rgb_obs_dict = {k: v for k, v in obs_dict.items() 
                        if k in self.head_keys or k in self.wrist_keys}
        
        if len(rgb_obs_dict) > 0:
            # TimmObsEncoder返回的是flatten的特征，我们需要分离出每个相机的特征
            # 先临时调用rgb_encoder提取每个相机的单独特征
            head_features = []
            wrist_features_dict = {}
            
            for key in self.head_keys + self.wrist_keys:
                if key not in obs_dict:
                    continue
                    
                img = obs_dict[key]
                
                # 预处理图像（参考TimmObsEncoder的逻辑）
                if img.max() > 1.0:
                    img = img / 255.0
                if img.shape[-1] == 3:
                    if len(img.shape) == 5:
                        img = img.permute(0, 1, 4, 2, 3)
                    elif len(img.shape) == 4:
                        img = img.permute(0, 3, 1, 2)
                
                B, T = img.shape[:2]
                img = img.reshape(B * T, *img.shape[2:])
                
                # Resize到期望shape
                key_shape = self.rgb_encoder.key_shape_map[key]
                if img.shape[2:] != key_shape[1:]:
                    target_H, target_W = key_shape[1], key_shape[2]
                    img = F.interpolate(img, size=(target_H, target_W), mode='bilinear', align_corners=False)
                
                # Transform和编码
                img = self.rgb_encoder.key_transform_map[key](img).to(self.device).float()
                raw_feature = self.rgb_encoder.key_model_map[key](img)
                feature = self.rgb_encoder.aggregate_feature(raw_feature)  # [B*T, D]
                
                # 如果aggregate_feature返回的是[B*T, 1, D]格式，需要squeeze
                if len(feature.shape) == 3:
                    feature = feature.squeeze(1)  # [B*T, D]
                
                feature = feature.reshape(B, T, -1).mean(dim=1)  # [B, D]
                feature_token = feature.unsqueeze(1)  # [B, 1, D]
                
                if key in self.head_keys:
                    head_features.append(feature_token)
                else:
                    wrist_features_dict[key] = feature_token
        
        # 2. 处理触觉传感器
        tactile_features_dict = {}
        if self.tactile_encoder is not None and len(self.tactile_keys) > 0:
            tactile_output = self.tactile_encoder(obs_dict)  # {key: [B, 1, D]}
            for key, feat in tactile_output.items():
                # 投影到RGB特征空间（如果需要）
                if self.tactile_projection is not None:
                    feat = self.tactile_projection(feat)
                tactile_features_dict[key] = feat
        
        # 3. 融合腕部视觉和触觉特征
        wrist_fused_features = []
        for wrist_key in self.wrist_keys:
            if wrist_key not in wrist_features_dict:
                continue
            
            visual_token = wrist_features_dict[wrist_key]  # [B, 1, D]
            
            # 查找匹配的触觉传感器
            if wrist_key in self.wrist_fusion_modules:
                tactile_key = self._find_matching_tactile(wrist_key)
                if tactile_key in tactile_features_dict:
                    tactile_token = tactile_features_dict[tactile_key]  # [B, 1, D]
                    # Cross Attention融合
                    fused_token = self.wrist_fusion_modules[wrist_key](visual_token, tactile_token)
                    wrist_fused_features.append(fused_token.squeeze(1))  # [B, D]
                else:
                    # 没有触觉输入，直接使用视觉特征
                    wrist_fused_features.append(visual_token.squeeze(1))
            else:
                # 没有融合模块，直接使用视觉特征
                wrist_fused_features.append(visual_token.squeeze(1))
        
        # 4. 拼接所有特征
        # 头部特征
        if len(head_features) > 0:
            features.extend([f.squeeze(1) for f in head_features])  # [B, D]
        
        # 腕部融合特征
        if len(wrist_fused_features) > 0:
            features.extend(wrist_fused_features)
        
        # 低维状态特征
        for key in self.low_dim_keys:
            if key in obs_dict:
                data = obs_dict[key].to(self.device)
                B, T = data.shape[:2]
                features.append(data.reshape(B, -1))  # [B, T*D]
        
        # 最终拼接
        result = torch.cat(features, dim=-1)  # [B, total_D]
        
        return result
    
    @torch.no_grad()
    def output_shape(self):
        """计算输出shape"""
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape,
                dtype=self.dtype,
                device=self.device
            )
            example_obs_dict[key] = this_obs
        
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape


if __name__ == '__main__':
    print("\n=== MultiModalObsEncoder 测试 ===\n")
    
    # 构造shape_meta（头部+双腕+双触觉+状态）
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
    
    # 创建编码器
    encoder = MultiModalObsEncoder(
        shape_meta=shape_meta,
        rgb_model_name='resnet18',
        rgb_pretrained=False,
        rgb_feature_aggregation='attention_pool_2d',
        tactile_model_name='resnet18',
        tactile_share_model=True,
        tactile_feature_dim=768,
    )
    
    print(f"\n总参数量: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # 测试前向传播
    obs = {
        'head_cam': torch.randn(4, 2, 3, 224, 224),
        'left_wrist_cam': torch.randn(4, 2, 3, 224, 224),
        'right_wrist_cam': torch.randn(4, 2, 3, 224, 224),
        'left_tactile': torch.randn(4, 2, 1, 16, 32),
        'right_tactile': torch.randn(4, 2, 1, 16, 32),
        'agent_pos': torch.randn(4, 2, 14),
    }
    
    with torch.no_grad():
        output = encoder(obs)
    
    print(f"\n输入:")
    print(f"  - head_cam: {obs['head_cam'].shape}")
    print(f"  - left_wrist_cam: {obs['left_wrist_cam'].shape}")
    print(f"  - right_wrist_cam: {obs['right_wrist_cam'].shape}")
    print(f"  - left_tactile: {obs['left_tactile'].shape}")
    print(f"  - right_tactile: {obs['right_tactile'].shape}")
    print(f"  - agent_pos: {obs['agent_pos'].shape}")
    print(f"\n输出: {output.shape}")
    
    # 测试梯度
    obs_grad = {k: v.requires_grad_(True) for k, v in obs.items()}
    output = encoder(obs_grad)
    loss = output.sum()
    loss.backward()
    
    print(f"\n梯度测试通过")
    print("\n✅ MultiModalObsEncoder 测试完成\n")

