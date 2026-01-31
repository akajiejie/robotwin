# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# 差分感知触觉编码器 - 专注于触觉信号变化检测
# 
# 主要组件:
# - SharpKeypointSpatialSoftmax: 带硬阈值门控和锐利温度的关键点提取
# - KeypointSpatialSoftmax: 基础版关键点提取
# - CompositeTactileEncoder: 复合触觉编码器（全局+关键点+坐标）
# - DiffAwareCompositeTactileEncoder: 差分感知编码器（双流架构）
# --------------------------------------------------------
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from maniflow.common.pytorch_util import replace_submodules
from maniflow.model.tactile.base_sensor import BaseSensoryEncoder


class SharpKeypointSpatialSoftmax(nn.Module):
    """
    🔥 增强版 SpatialSoftmax：带硬阈值门控和锐利温度控制
    
    核心改进：
    1. 硬阈值过滤：Softmax 之前将低于噪声水平的区域强制设为 -inf
    2. 学习化温度：趋向于较小值（如 0.1），强制只提取最突出的点
    3. 使用可学习的 attention heads 提取 K 个关键点坐标
    """
    
    def __init__(self, 
                 in_channels: int, 
                 num_keypoints: int = 4, 
                 init_temperature: float = 0.1,
                 noise_threshold: float = 0.1,
                 learnable_temperature: bool = True,
                 learnable_threshold: bool = True):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.in_channels = in_channels
        
        # 可学习的 attention heads
        self.attention_conv = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        
        # 可学习温度参数（通过 softplus 确保为正，趋向小值增加锐度）
        if learnable_temperature:
            init_temp_log = torch.log(torch.exp(torch.tensor(init_temperature)) - 1 + 1e-8)
            self.temperature_raw = nn.Parameter(init_temp_log.clone().detach())
        else:
            self.register_buffer('temperature_raw', torch.tensor(init_temperature))
        self.learnable_temperature = learnable_temperature
        
        # 可学习噪声阈值（通过 sigmoid 映射到 [0, 1]）
        if learnable_threshold:
            init_thresh_logit = torch.log(torch.tensor(noise_threshold / (1 - noise_threshold + 1e-8)))
            self.threshold_raw = nn.Parameter(init_thresh_logit.clone().detach())
        else:
            self.register_buffer('threshold_raw', torch.tensor(noise_threshold))
        self.learnable_threshold = learnable_threshold
        
        # 坐标网格缓存
        self._coord_cache = {}
    
    @property
    def temperature(self) -> torch.Tensor:
        """获取当前温度值（通过 softplus 确保为正）"""
        if self.learnable_temperature:
            return F.softplus(self.temperature_raw).clamp(min=0.01, max=2.0)
        return self.temperature_raw
    
    @property
    def noise_threshold(self) -> torch.Tensor:
        """获取当前噪声阈值（通过 sigmoid 映射到 [0, 1]）"""
        if self.learnable_threshold:
            return torch.sigmoid(self.threshold_raw).clamp(min=0.01, max=0.5)
        return self.threshold_raw
    
    def _get_coord_grid(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取归一化坐标网格 [-1, 1]"""
        cache_key = (H, W, device)
        if cache_key not in self._coord_cache:
            pos_x = torch.linspace(-1, 1, W, device=device)
            pos_y = torch.linspace(-1, 1, H, device=device)
            grid_x, grid_y = torch.meshgrid(pos_x, pos_y, indexing='xy')
            self._coord_cache[cache_key] = (grid_x.reshape(1, 1, H * W), grid_y.reshape(1, 1, H * W))
        return self._coord_cache[cache_key]
    
    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        输入: feature_map (B, C, H, W)
        输出:
            - keypoint_coords: (B, K, 2)
            - keypoint_features: (B, K, C)
            - attention_weights: (B, K, H, W)
        """
        B, C, H, W = feature_map.shape
        K = self.num_keypoints
        
        # 1. 计算注意力 logits
        attention_logits = self.attention_conv(feature_map)
        attention_flat = attention_logits.reshape(B, K, H * W)
        
        # 2. 硬阈值门控：将低于阈值的区域设为 -inf
        attn_min = attention_flat.min(dim=-1, keepdim=True)[0]
        attn_max = attention_flat.max(dim=-1, keepdim=True)[0]
        attn_normalized = (attention_flat - attn_min) / (attn_max - attn_min + 1e-8)
        
        noise_mask = attn_normalized < self.noise_threshold
        attention_masked = attention_flat.clone()
        attention_masked[noise_mask] = attention_masked[noise_mask] - 1e4
        
        # 3. 使用锐利温度的 Softmax
        attention_weights = F.softmax(attention_masked / self.temperature, dim=-1)
        
        # 4. 获取坐标网格
        pos_x, pos_y = self._get_coord_grid(H, W, feature_map.device)
        
        # 5. 计算关键点期望坐标
        expected_x = (attention_weights * pos_x).sum(dim=-1)
        expected_y = (attention_weights * pos_y).sum(dim=-1)
        keypoint_coords = torch.stack([expected_x, expected_y], dim=-1)
        
        # 6. 从特征图提取关键点特征
        grid = keypoint_coords.unsqueeze(2)
        keypoint_features = F.grid_sample(
            feature_map, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        keypoint_features = keypoint_features.squeeze(-1).transpose(1, 2)
        
        # 7. 重塑 attention_weights
        attention_weights_2d = attention_weights.reshape(B, K, H, W)
        
        return keypoint_coords, keypoint_features, attention_weights_2d
    
    def get_sharpness_stats(self) -> Dict[str, float]:
        """返回当前锐度参数统计（用于监控）"""
        return {
            'temperature': self.temperature.item(),
            'noise_threshold': self.noise_threshold.item() if self.learnable_threshold else float(self.threshold_raw),
        }


class KeypointSpatialSoftmax(nn.Module):
    """
    基础版 SpatialSoftmax：提取多个关键受力点的坐标和对应特征
    
    核心思想：
    1. 使用可学习的 attention heads 从特征图中提取 K 个关键点坐标
    2. 通过双线性插值从特征图中索引这些坐标对应的特征向量
    3. 同时保留关键点坐标信息（物理意义：受力点位置）
    """
    
    def __init__(self, 
                 in_channels: int, 
                 num_keypoints: int = 4, 
                 temperature: float = 1.0,
                 learnable_temperature: bool = True):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.in_channels = in_channels
        
        self.attention_conv = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        self._coord_cache = {}
    
    def _get_coord_grid(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (H, W, device)
        if cache_key not in self._coord_cache:
            pos_x = torch.linspace(-1, 1, W, device=device)
            pos_y = torch.linspace(-1, 1, H, device=device)
            grid_x, grid_y = torch.meshgrid(pos_x, pos_y, indexing='xy')
            self._coord_cache[cache_key] = (grid_x.reshape(1, 1, H * W), grid_y.reshape(1, 1, H * W))
        return self._coord_cache[cache_key]
    
    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = feature_map.shape
        K = self.num_keypoints
        
        attention_logits = self.attention_conv(feature_map)
        attention_flat = attention_logits.reshape(B, K, H * W)
        attention_weights = F.softmax(attention_flat / self.temperature, dim=-1)
        
        pos_x, pos_y = self._get_coord_grid(H, W, feature_map.device)
        
        expected_x = (attention_weights * pos_x).sum(dim=-1)
        expected_y = (attention_weights * pos_y).sum(dim=-1)
        keypoint_coords = torch.stack([expected_x, expected_y], dim=-1)
        
        grid = keypoint_coords.unsqueeze(2)
        keypoint_features = F.grid_sample(
            feature_map, grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        keypoint_features = keypoint_features.squeeze(-1).transpose(1, 2)
        
        attention_weights_2d = attention_weights.reshape(B, K, H, W)
        
        return keypoint_coords, keypoint_features, attention_weights_2d


class CompositeTactileEncoder(BaseSensoryEncoder):
    """
    复合触觉编码器：输出多个互补的 Token
    
    输出结构（每个触觉传感器）:
    - global_token: (B, T, D) - 全局平均池化特征
    - keypoint_tokens: (B, T, K, D) - K个关键受力点的特征
    - coord_token: (B, T, D) - 关键点坐标编码（物理位置信息）
    
    最终合并为: (B, T, 1+K+1, D) 的 token 序列
    """
    
    def __init__(self,
        shape_meta: dict,
        model_name: str = 'resnet18',
        pretrained: bool = False,
        frozen: bool = False,
        use_group_norm: bool = True,
        share_tactile_model: bool = False,
        feature_dim: int = 768,
        num_keypoints: int = 4,
        temperature: float = 1.0,
        include_coord_token: bool = True,
    ):
        super().__init__()
        
        tactile_keys = []
        key_shape_map = {}
        for key, attr in shape_meta['obs'].items():
            if attr.get('type') == 'rgb' and 'tactile' in key.lower():
                tactile_keys.append(key)
                key_shape_map[key] = tuple(attr['shape'])
        
        tactile_keys = sorted(tactile_keys)
        
        self.num_keypoints = num_keypoints
        self.include_coord_token = include_coord_token
        self.feature_dim = feature_dim
        
        key_backbone_map = nn.ModuleDict()
        key_keypoint_extractor_map = nn.ModuleDict()
        
        self.global_proj = nn.Linear(512, feature_dim)
        self.keypoint_proj = nn.Linear(512, feature_dim)
        if include_coord_token:
            self.coord_encoder = nn.Sequential(
                nn.Linear(num_keypoints * 2, 128),
                nn.GELU(),
                nn.Linear(128, feature_dim)
            )
        
        if share_tactile_model and len(tactile_keys) > 0:
            shared_backbone = self._create_backbone(
                key_shape_map[tactile_keys[0]], 
                model_name, pretrained, frozen, use_group_norm
            )
            shared_keypoint_extractor = KeypointSpatialSoftmax(
                in_channels=512, 
                num_keypoints=num_keypoints,
                temperature=temperature
            )
            for key in tactile_keys:
                key_backbone_map[key] = shared_backbone
                key_keypoint_extractor_map[key] = shared_keypoint_extractor
        else:
            for key in tactile_keys:
                key_backbone_map[key] = self._create_backbone(
                    key_shape_map[key],
                    model_name, pretrained, frozen, use_group_norm
                )
                key_keypoint_extractor_map[key] = KeypointSpatialSoftmax(
                    in_channels=512,
                    num_keypoints=num_keypoints,
                    temperature=temperature
                )
        
        self.tactile_keys = tactile_keys
        self.key_backbone_map = key_backbone_map
        self.key_keypoint_extractor_map = key_keypoint_extractor_map
        self.key_shape_map = key_shape_map
        
        self.num_tokens_per_sensor = 1 + num_keypoints + (1 if include_coord_token else 0)
        
        print(f"✓ 复合触觉编码器: {num_keypoints} 关键点 + 1 全局" + 
              (f" + 1 坐标" if include_coord_token else ""))
        print(f"  每个传感器输出 {self.num_tokens_per_sensor} 个 tokens, 维度 {feature_dim}")
    
    def _create_backbone(self, shape, model_name, pretrained, frozen, use_group_norm):
        in_channels = shape[0]
        
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            global_pool='',
            num_classes=0
        )
        
        if frozen:
            for param in model.parameters():
                param.requires_grad = False
        
        if model_name.startswith('resnet'):
            modules = list(model.children())[:-2]
            backbone = nn.Sequential(*modules)
        else:
            raise NotImplementedError(f"Unsupported model: {model_name}")
        
        if use_group_norm and not pretrained:
            backbone = replace_submodules(
                root_module=backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=max(1, x.num_features // 16),
                    num_channels=x.num_features
                )
            )
        
        return backbone
    
    def modalities(self):
        return ['tactile']
    
    def forward(self, obs: Dict[str, torch.Tensor], 
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        output = {}
        attention_output = {}
        
        for key in self.tactile_keys:
            if key not in obs:
                continue
            
            tactile_data = obs[key]
            
            if len(tactile_data.shape) == 5:
                B, T = tactile_data.shape[:2]
                tactile_data = tactile_data.reshape(B * T, *tactile_data.shape[2:])
            else:
                B = tactile_data.shape[0]
                T = 1
            
            if tactile_data.max() > 1.0:
                tactile_data = tactile_data / 255.0
            
            expected_shape = self.key_shape_map[key]
            if tactile_data.shape[1:] != expected_shape:
                target_H, target_W = expected_shape[1], expected_shape[2]
                tactile_data = F.interpolate(
                    tactile_data, 
                    size=(target_H, target_W), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            feature_map = self.key_backbone_map[key](tactile_data)
            
            global_feat = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
            global_token = self.global_proj(global_feat)
            
            keypoint_coords, keypoint_feats, attention_weights = \
                self.key_keypoint_extractor_map[key](feature_map)
            keypoint_tokens = self.keypoint_proj(keypoint_feats)
            
            tokens_list = [global_token.unsqueeze(1)]
            tokens_list.append(keypoint_tokens)
            
            if self.include_coord_token:
                coord_flat = keypoint_coords.reshape(B * T, -1)
                coord_token = self.coord_encoder(coord_flat).unsqueeze(1)
                tokens_list.append(coord_token)
            
            all_tokens = torch.cat(tokens_list, dim=1)
            all_tokens = all_tokens.reshape(B, T, self.num_tokens_per_sensor, self.feature_dim)
            
            output[key] = all_tokens
            
            if return_attention:
                _, _, H_feat, W_feat = feature_map.shape
                attention_output[key] = attention_weights.reshape(B, T, self.num_keypoints, H_feat, W_feat)
        
        if return_attention:
            return output, attention_output
        return output
    
    def output_feature_dim(self):
        return {key: self.feature_dim for key in self.tactile_keys}
    
    def output_num_tokens(self):
        return {key: self.num_tokens_per_sensor for key in self.tactile_keys}


class DiffAwareCompositeTactileEncoder(BaseSensoryEncoder):
    """
    🔥 差分感知复合触觉编码器 - 专注于触觉信号变化
    
    核心改进：
    1. 差分输入分支：将当前帧 I_t 与变化帧 I_diff = I_t - I_{t-1} 分别处理
    2. 使用 SharpKeypointSpatialSoftmax 实现锐利的关键点提取
    3. 双流架构：分别处理静态特征和动态变化特征
    
    输出结构（每个触觉传感器）:
    - global_token: 全局特征（静态 + 动态融合）
    - keypoint_tokens: K 个关键受力点的特征
    - diff_token: 差分/变化特征 token
    - coord_token: 关键点坐标编码
    """
    
    def __init__(self,
        shape_meta: dict,
        model_name: str = 'resnet18',
        pretrained: bool = False,
        frozen: bool = False,
        use_group_norm: bool = True,
        share_tactile_model: bool = False,
        feature_dim: int = 768,
        num_keypoints: int = 4,
        init_temperature: float = 0.1,
        noise_threshold: float = 0.15,
        include_coord_token: bool = True,
        include_diff_token: bool = True,
        diff_amplify: float = 2.0,
    ):
        super().__init__()
        
        tactile_keys = []
        key_shape_map = {}
        for key, attr in shape_meta['obs'].items():
            if attr.get('type') == 'rgb' and 'tactile' in key.lower():
                tactile_keys.append(key)
                key_shape_map[key] = tuple(attr['shape'])
        
        tactile_keys = sorted(tactile_keys)
        
        self.num_keypoints = num_keypoints
        self.include_coord_token = include_coord_token
        self.include_diff_token = include_diff_token
        self.feature_dim = feature_dim
        self.diff_amplify = diff_amplify
        
        # 双流骨干网络
        key_static_backbone_map = nn.ModuleDict()
        key_diff_backbone_map = nn.ModuleDict()
        key_keypoint_extractor_map = nn.ModuleDict()
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )
        
        # 投影层
        self.global_proj = nn.Linear(512, feature_dim)
        self.keypoint_proj = nn.Linear(512, feature_dim)
        
        if include_diff_token:
            self.diff_proj = nn.Linear(512, feature_dim)
        
        if include_coord_token:
            self.coord_encoder = nn.Sequential(
                nn.Linear(num_keypoints * 2, 128),
                nn.GELU(),
                nn.Linear(128, feature_dim)
            )
        
        if share_tactile_model and len(tactile_keys) > 0:
            shared_static_backbone = self._create_backbone(
                key_shape_map[tactile_keys[0]], 
                model_name, pretrained, frozen, use_group_norm
            )
            shared_diff_backbone = self._create_backbone(
                key_shape_map[tactile_keys[0]], 
                model_name, False, frozen, use_group_norm
            )
            shared_keypoint_extractor = SharpKeypointSpatialSoftmax(
                in_channels=512, 
                num_keypoints=num_keypoints,
                init_temperature=init_temperature,
                noise_threshold=noise_threshold,
                learnable_temperature=True,
                learnable_threshold=True
            )
            for key in tactile_keys:
                key_static_backbone_map[key] = shared_static_backbone
                key_diff_backbone_map[key] = shared_diff_backbone
                key_keypoint_extractor_map[key] = shared_keypoint_extractor
        else:
            for key in tactile_keys:
                key_static_backbone_map[key] = self._create_backbone(
                    key_shape_map[key], model_name, pretrained, frozen, use_group_norm
                )
                key_diff_backbone_map[key] = self._create_backbone(
                    key_shape_map[key], model_name, False, frozen, use_group_norm
                )
                key_keypoint_extractor_map[key] = SharpKeypointSpatialSoftmax(
                    in_channels=512,
                    num_keypoints=num_keypoints,
                    init_temperature=init_temperature,
                    noise_threshold=noise_threshold
                )
        
        self.tactile_keys = tactile_keys
        self.key_static_backbone_map = key_static_backbone_map
        self.key_diff_backbone_map = key_diff_backbone_map
        self.key_keypoint_extractor_map = key_keypoint_extractor_map
        self.key_shape_map = key_shape_map
        
        self.num_tokens_per_sensor = (1 + num_keypoints + 
                                       (1 if include_coord_token else 0) +
                                       (1 if include_diff_token else 0))
        
        print(f"✓ 差分感知触觉编码器: {num_keypoints} 关键点 + 1 全局" + 
              (f" + 1 坐标" if include_coord_token else "") +
              (f" + 1 差分" if include_diff_token else ""))
        print(f"  温度={init_temperature}, 噪声阈值={noise_threshold}, 差分放大={diff_amplify}")
        print(f"  每个传感器输出 {self.num_tokens_per_sensor} 个 tokens, 维度 {feature_dim}")
    
    def _create_backbone(self, shape, model_name, pretrained, frozen, use_group_norm):
        in_channels = shape[0]
        
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            global_pool='',
            num_classes=0
        )
        
        if frozen:
            for param in model.parameters():
                param.requires_grad = False
        
        if model_name.startswith('resnet'):
            modules = list(model.children())[:-2]
            backbone = nn.Sequential(*modules)
        else:
            raise NotImplementedError(f"Unsupported model: {model_name}")
        
        if use_group_norm and not pretrained:
            backbone = replace_submodules(
                root_module=backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=max(1, x.num_features // 16),
                    num_channels=x.num_features
                )
            )
        
        return backbone
    
    def modalities(self):
        return ['tactile']
    
    def forward(self, obs: Dict[str, torch.Tensor],
                prev_obs: Optional[Dict[str, torch.Tensor]] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        输入: 
            - obs: 当前帧 {key: (B, T, C, H, W) 或 (B, C, H, W)}
            - prev_obs: 上一帧（可选）。如果为 None，使用零帧或时序内差分
        输出:
            - features: (B, T, num_tokens, D)
            - attention_maps (可选)
        """
        output = {}
        attention_output = {}
        
        for key in self.tactile_keys:
            if key not in obs:
                continue
            
            tactile_data = obs[key]
            
            if len(tactile_data.shape) == 5:
                B, T = tactile_data.shape[:2]
                tactile_data = tactile_data.reshape(B * T, *tactile_data.shape[2:])
            else:
                B = tactile_data.shape[0]
                T = 1
            
            if tactile_data.max() > 1.0:
                tactile_data = tactile_data / 255.0
            
            expected_shape = self.key_shape_map[key]
            if tactile_data.shape[1:] != expected_shape:
                target_H, target_W = expected_shape[1], expected_shape[2]
                tactile_data = F.interpolate(
                    tactile_data, size=(target_H, target_W),
                    mode='bilinear', align_corners=False
                )
            
            # 计算差分帧
            if prev_obs is not None and key in prev_obs:
                prev_data = prev_obs[key]
                if len(prev_data.shape) == 5:
                    prev_data = prev_data.reshape(B * T, *prev_data.shape[2:])
                if prev_data.max() > 1.0:
                    prev_data = prev_data / 255.0
                if prev_data.shape[1:] != expected_shape:
                    prev_data = F.interpolate(
                        prev_data, size=(target_H, target_W),
                        mode='bilinear', align_corners=False
                    )
                diff_data = (tactile_data - prev_data) * self.diff_amplify
            else:
                if T > 1:
                    tactile_seq = tactile_data.reshape(B, T, *tactile_data.shape[1:])
                    diff_seq = torch.zeros_like(tactile_seq)
                    diff_seq[:, 1:] = (tactile_seq[:, 1:] - tactile_seq[:, :-1]) * self.diff_amplify
                    diff_data = diff_seq.reshape(B * T, *tactile_data.shape[1:])
                else:
                    diff_data = torch.zeros_like(tactile_data)
            
            # 双流特征提取
            static_feat_map = self.key_static_backbone_map[key](tactile_data)
            diff_feat_map = self.key_diff_backbone_map[key](diff_data.clamp(-1, 1))
            
            # 特征融合
            static_global = F.adaptive_avg_pool2d(static_feat_map, 1).flatten(1)
            diff_global = F.adaptive_avg_pool2d(diff_feat_map, 1).flatten(1)
            fused_global = self.fusion_layer(torch.cat([static_global, diff_global], dim=-1))
            
            global_token = self.global_proj(fused_global)
            
            # 在融合特征图上提取关键点
            combined_feat_map = static_feat_map + diff_feat_map * 0.5
            
            keypoint_coords, keypoint_feats, attention_weights = \
                self.key_keypoint_extractor_map[key](combined_feat_map)
            keypoint_tokens = self.keypoint_proj(keypoint_feats)
            
            # 组装 tokens
            tokens_list = [global_token.unsqueeze(1)]
            tokens_list.append(keypoint_tokens)
            
            if self.include_diff_token:
                diff_token = self.diff_proj(diff_global).unsqueeze(1)
                tokens_list.append(diff_token)
            
            if self.include_coord_token:
                coord_flat = keypoint_coords.reshape(B * T, -1)
                coord_token = self.coord_encoder(coord_flat).unsqueeze(1)
                tokens_list.append(coord_token)
            
            all_tokens = torch.cat(tokens_list, dim=1)
            all_tokens = all_tokens.reshape(B, T, self.num_tokens_per_sensor, self.feature_dim)
            
            output[key] = all_tokens
            
            if return_attention:
                _, _, H_feat, W_feat = combined_feat_map.shape
                attention_output[key] = attention_weights.reshape(B, T, self.num_keypoints, H_feat, W_feat)
        
        if return_attention:
            return output, attention_output
        return output
    
    def output_feature_dim(self):
        return {key: self.feature_dim for key in self.tactile_keys}
    
    def output_num_tokens(self):
        return {key: self.num_tokens_per_sensor for key in self.tactile_keys}
    
    def get_sharpness_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有关键点提取器的锐度参数"""
        stats = {}
        for key in self.tactile_keys:
            extractor = self.key_keypoint_extractor_map[key]
            if hasattr(extractor, 'get_sharpness_stats'):
                stats[key] = extractor.get_sharpness_stats()
        return stats


# ========== 验证脚本 ==========
if __name__ == '__main__':
    import os
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    parser = argparse.ArgumentParser(description='差分感知触觉编码器训练与验证')
    parser.add_argument('--zarr_path', type=str, 
                        default='/root/autodl-tmp/robotwin/policy/ManiFlow/data/feed_dual-40.zarr',
                        help='zarr 数据路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--num_samples', type=int, default=10000, help='训练样本数')
    parser.add_argument('--feature_dim', type=int, default=128, help='特征维度')
    parser.add_argument('--num_keypoints', type=int, default=4, help='关键点数量')
    parser.add_argument('--init_temperature', type=float, default=0.1, help='初始温度')
    parser.add_argument('--noise_threshold', type=float, default=0.15, help='噪声阈值')
    parser.add_argument('--diff_amplify', type=float, default=2.0, help='差分放大系数')
    parser.add_argument('--save_dir', type=str, 
                        default='/root/autodl-tmp/robotwin/policy/ManiFlow/data',
                        help='保存目录')
    parser.add_argument('--use_diff_encoder', action='store_true', default=True,
                        help='使用差分感知编码器')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔬 DiffAwareCompositeTactileEncoder 验证")
    print("="*70)
    
    # 加载数据
    try:
        import zarr
        z = zarr.open(args.zarr_path, 'r')
        left_tactile = z['data/left_tactile_sensor'][:]
        right_tactile = z['data/right_tactile_sensor'][:]
        episode_ends = z['meta/episode_ends'][:]
        print(f"\n📊 数据加载成功: {left_tactile.shape}")
        USE_REAL_DATA = True
    except Exception as e:
        print(f"\n⚠️ 无法加载数据: {e}")
        left_tactile = np.random.rand(5000, 1, 16, 32).astype(np.float32) * 255
        episode_ends = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
        USE_REAL_DATA = False
    
    # 构建模型
    shape_meta = {
        'obs': {
            'left_tactile_sensor': {'shape': [1, 16, 32], 'type': 'rgb', 'horizon': 2},
            'right_tactile_sensor': {'shape': [1, 16, 32], 'type': 'rgb', 'horizon': 2},
        }
    }
    
    encoder = DiffAwareCompositeTactileEncoder(
        shape_meta=shape_meta,
        model_name='resnet18',
        pretrained=False,
        use_group_norm=True,
        share_tactile_model=True,
        feature_dim=args.feature_dim,
        num_keypoints=args.num_keypoints,
        init_temperature=args.init_temperature,
        noise_threshold=args.noise_threshold,
        include_coord_token=True,
        include_diff_token=True,
        diff_amplify=args.diff_amplify,
    )
    
    print(f"\n🔧 模型参数量: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # 简单测试
    obs = {'left_tactile_sensor': torch.randn(4, 2, 1, 16, 32)}
    prev_obs = {'left_tactile_sensor': torch.randn(4, 2, 1, 16, 32)}
    
    with torch.no_grad():
        output, attn = encoder(obs, prev_obs=prev_obs, return_attention=True)
    
    print(f"输出形状: {output['left_tactile_sensor'].shape}")
    print(f"注意力形状: {attn['left_tactile_sensor'].shape}")
    print(f"锐度参数: {encoder.get_sharpness_stats()}")
    
    print("\n✅ 测试通过!\n")
