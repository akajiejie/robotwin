# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# policyconsensus：https://github.com/policyconsensus/policyconsensus.git
# ManiFlow: https://github.com/geyan21/ManiFlow_Policy
# touch_in_the_wild:https://github.com/YolandaXinyueZhu/touch_in_the_wild.git
# --------------------------------------------------------
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from maniflow.common.pytorch_util import replace_submodules
from maniflow.model.tactile.base_sensor import BaseSensoryEncoder

class TimmTactileEncoder(BaseSensoryEncoder):
    """使用timm库的触觉编码器，复用ResNet18处理触觉数据"""
    
    def __init__(self,
        shape_meta: dict,
        model_name: str = 'resnet18',
        pretrained: bool = False,
        frozen: bool = False,
        use_group_norm: bool = True,
        share_tactile_model: bool = False,
        feature_dim: int = 768,  # 输出特征维度，对应CLIP cls token
        output_all_patches: bool = False,  # 🔥 是否输出所有patch tokens（类似ViT）
    ):
        super().__init__()
        
        # 筛选触觉数据（type='rgb'且key包含'tactile'）
        tactile_keys = []
        key_shape_map = {}
        for key, attr in shape_meta['obs'].items():
            if attr.get('type') == 'rgb' and 'tactile' in key.lower():
                tactile_keys.append(key)
                key_shape_map[key] = tuple(attr['shape'])
        
        tactile_keys = sorted(tactile_keys)
        
        # 🔥 提前保存output_all_patches，因为_create_tactile_model需要使用它
        self.output_all_patches = output_all_patches
        
        # 为每个触觉传感器创建或共享模型
        key_model_map = nn.ModuleDict()
        
        if share_tactile_model and len(tactile_keys) > 0:
            # 共享模型：所有触觉传感器使用同一个网络
            shared_model = self._create_tactile_model(
                key_shape_map[tactile_keys[0]], 
                model_name, pretrained, frozen, use_group_norm, feature_dim
            )
            for key in tactile_keys:
                key_model_map[key] = shared_model
        else:
            # 独立模型：每个触觉传感器有自己的网络
            for key in tactile_keys:
                key_model_map[key] = self._create_tactile_model(
                    key_shape_map[key],
                    model_name, pretrained, frozen, use_group_norm, feature_dim
                )
        
        self.tactile_keys = tactile_keys
        self.key_model_map = key_model_map
        self.key_shape_map = key_shape_map
        self.feature_dim = feature_dim
        # self.output_all_patches 已在前面赋值
        
        print(f"✓ 触觉编码器输出模式: {'all_patches' if output_all_patches else 'aggregated'}", 
               'cyan' if output_all_patches else 'green')
        
    def _create_tactile_model(self, shape, model_name, pretrained, frozen, use_group_norm, feature_dim):
        """创建单个触觉处理模型"""
        in_channels = shape[0]  # 触觉数据的通道数
        
        # 创建ResNet18模型
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,  # 自适应输入通道数
            global_pool='',  # 不使用全局池化
            num_classes=0
        )
        
        if frozen:
            for param in model.parameters():
                param.requires_grad = False
        
        # 提取ResNet18的卷积层（移除最后的池化和FC层）
        if model_name.startswith('resnet'):
            # 保留到layer4，移除avgpool和fc
            modules = list(model.children())[:-2]
            backbone = nn.Sequential(*modules)
        else:
            raise NotImplementedError(f"Unsupported model: {model_name}")
        
        # 替换BatchNorm为GroupNorm
        if use_group_norm and not pretrained:
            backbone = replace_submodules(
                root_module=backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=max(1, x.num_features // 16),
                    num_channels=x.num_features
                )
            )
        
        # 🔥 根据output_all_patches决定输出方式
        if self.output_all_patches:
            # 输出所有空间patch tokens: (B, C, H, W) -> (B, H*W, D)
            # ResNet18 layer4输出: 512通道
            # 添加1x1卷积投影到目标维度，然后reshape为patch tokens
            conv_proj = nn.Conv2d(512, feature_dim, kernel_size=1)
            return nn.Sequential(backbone, conv_proj)
        else:
            # 原始方式: SpatialSoftmax池化 + 线性投影
            # ResNet18的layer4输出是512通道
            spatial_softmax = SpatialSoftmax(temperature=1.0)
            projection = nn.Linear(512 * 2, feature_dim)  # SpatialSoftmax输出 (x,y) 坐标，所以是 C*2
            return nn.Sequential(backbone, spatial_softmax, projection)
    
    def modalities(self):
        return ['tactile']
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        输入: obs字典，每个触觉key对应 (B, T, C, H, W) 或 (B, C, H, W)
        输出: 
            - output_all_patches=False: 每个触觉key对应的token特征 (B, T, D)，保留时序维度
            - output_all_patches=True: 每个触觉key对应的patch tokens (B, T*H*W, D)，保留时序维度
        """
        output = {}
        
        for key in self.tactile_keys:
            if key not in obs:
                continue
            
            tactile_data = obs[key]  # (B, T, C, H, W) 或 (B, C, H, W)
            
            # 处理时序维度
            if len(tactile_data.shape) == 5:
                B, T = tactile_data.shape[:2]
                tactile_data = tactile_data.reshape(B * T, *tactile_data.shape[2:])
            else:
                B = tactile_data.shape[0]
                T = 1
            
            # 归一化到[0,1]
            if tactile_data.max() > 1.0:
                tactile_data = tactile_data / 255.0
            
            # resize到期望的shape（如果需要）
            expected_shape = self.key_shape_map[key]
            if tactile_data.shape[1:] != expected_shape:
                target_H, target_W = expected_shape[1], expected_shape[2]
                tactile_data = F.interpolate(
                    tactile_data, 
                    size=(target_H, target_W), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 前向传播
            feature = self.key_model_map[key](tactile_data)
            
            # 🔥 根据output_all_patches决定输出格式（保留时序维度）
            if self.output_all_patches:
                # 输出所有patch tokens: (B*T, D, H, W) -> (B, T*H*W, D)
                BT, D, H, W = feature.shape
                feature = feature.flatten(2).transpose(1, 2)  # (B*T, H*W, D)
                feature = feature.reshape(B, T * H * W, D)  # 保留时序维度 -> (B, T*H*W, D)
            else:
                # 原始方式: 聚合为token序列 (B*T, D) -> (B, T, D)
                feature = feature.reshape(B, T, -1)  # 保留时序维度 -> (B, T, D)
            
            output[key] = feature
        
        return output
    
    def output_feature_dim(self):
        """
        返回每个触觉传感器的输出特征维度
        - output_all_patches=False: [B, T, D] (保留时序维度)
        - output_all_patches=True: [B, T*H*W, D] (保留时序维度)
        """
        return {key: self.feature_dim for key in self.tactile_keys}


class SpatialSoftmax(nn.Module):
    """Spatial Softmax池化层，输出特征点的(x,y)坐标加权和"""
    
    def __init__(self, temperature=1.0, normalize=False):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, x):
        """
        输入: (B, C, H, W)
        输出: (B, C*2) - 每个通道的加权x,y坐标
        """
        B, C, H, W = x.shape
        
        # 创建坐标网格
        pos_x = torch.linspace(-1, 1, W, device=x.device)
        pos_y = torch.linspace(-1, 1, H, device=x.device)
        pos_x, pos_y = torch.meshgrid(pos_x, pos_y, indexing='xy')
        pos_x = pos_x.reshape(1, 1, H * W)
        pos_y = pos_y.reshape(1, 1, H * W)
        
        # Flatten spatial维度
        x_flat = x.reshape(B, C, H * W)
        
        # Softmax计算权重
        if self.normalize:
            x_flat = x_flat - x_flat.max(dim=-1, keepdim=True)[0]
        weights = F.softmax(x_flat / self.temperature, dim=-1)  # (B, C, H*W)
        
        # 加权求和得到期望坐标
        expected_x = (weights * pos_x).sum(dim=-1)  # (B, C)
        expected_y = (weights * pos_y).sum(dim=-1)  # (B, C)
        
        # 拼接x,y坐标
        output = torch.cat([expected_x, expected_y], dim=-1)  # (B, C*2)
        
        return output


if __name__ == '__main__':
    print("\n=== TimmTactileEncoder 测试 ===\n")
    
    # 构造shape_meta（触觉编码器只需要obs，不需要action）
    shape_meta = {
        'obs': {
            'head_cam': {'shape': [3, 224, 224], 'type': 'rgb', 'horizon': 2},
            'left_wrist_cam': {'shape': [3, 224, 224], 'type': 'rgb', 'horizon': 2},
            'right_wrist_cam': {'shape': [3, 224, 224], 'type': 'rgb', 'horizon': 2},
            'left_tactile': {'shape': [1, 16, 32], 'type': 'rgb', 'horizon': 2},
            'right_tactile': {'shape': [1, 16, 32], 'type': 'rgb', 'horizon': 2},
            'agent_pos': {'shape': [14], 'type': 'low_dim', 'horizon': 2},
        }
    }
    
    # 创建共享权重编码器（使用768维输出，与CLIP cls token对应）
    encoder = TimmTactileEncoder(
        shape_meta=shape_meta,
        model_name='resnet18',
        pretrained=False,
        frozen=False,
        use_group_norm=True,
        share_tactile_model=True,
        feature_dim=768
    )
    
    print(f"触觉传感器: {encoder.tactile_keys}")
    print(f"特征维度: {list(encoder.output_feature_dim().values())[0]}D (token format)")
    print(f"参数量: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"权重共享: {encoder.key_model_map['left_tactile'] is encoder.key_model_map['right_tactile']}")
    
    # 测试前向传播
    obs = {
        'left_tactile': torch.randn(4, 2, 1, 16, 32),
        'right_tactile': torch.randn(4, 2, 1, 16, 32),
    }
    
    with torch.no_grad():
        out = encoder(obs)
    
    print(f"\n输入: [B=4, T=2, C=1, H=16, W=32]")
    print(f"输出 (保留时序): {list(out.values())[0].shape} -> 期望: [B=4, T=2, D=768]")
    assert list(out.values())[0].shape == (4, 2, 768), "输出形状不匹配！"
    
    # 测试梯度
    obs_grad = {
        'left_tactile': torch.randn(2, 2, 1, 16, 32, requires_grad=True),
        'right_tactile': torch.randn(2, 2, 1, 16, 32, requires_grad=True),
    }
    output = encoder(obs_grad)
    
    # 验证输出形状
    for key, feat in output.items():
        assert feat.shape == (2, 2, 768), f"{key} 输出形状错误: {feat.shape}"
    
    loss = sum(v.sum() for v in output.values())
    loss.backward()
    
    print(f"梯度范数: {obs_grad['left_tactile'].grad.norm().item():.6f}")
    print("\n✅ 测试通过 - 输出格式: [B, T, 768] 保留完整时序信息\n")
