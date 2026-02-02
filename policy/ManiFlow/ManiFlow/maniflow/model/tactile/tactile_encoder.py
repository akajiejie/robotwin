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
        feature_dim: int = 768,
        output_all_patches: bool = False,
    ):
        super().__init__()
        
        tactile_keys = []
        key_shape_map = {}
        for key, attr in shape_meta['obs'].items():
            if attr.get('type') == 'rgb' and 'tactile' in key.lower():
                tactile_keys.append(key)
                key_shape_map[key] = tuple(attr['shape'])
        
        tactile_keys = sorted(tactile_keys)
        self.output_all_patches = output_all_patches
        
        key_model_map = nn.ModuleDict()
        
        if share_tactile_model and len(tactile_keys) > 0:
            shared_model = self._create_tactile_model(
                key_shape_map[tactile_keys[0]], 
                model_name, pretrained, frozen, use_group_norm, feature_dim
            )
            for key in tactile_keys:
                key_model_map[key] = shared_model
        else:
            for key in tactile_keys:
                key_model_map[key] = self._create_tactile_model(
                    key_shape_map[key],
                    model_name, pretrained, frozen, use_group_norm, feature_dim
                )
        
        self.tactile_keys = tactile_keys
        self.key_model_map = key_model_map
        self.key_shape_map = key_shape_map
        self.feature_dim = feature_dim
        
        print(f"✓ 触觉编码器输出模式: {'all_patches' if output_all_patches else 'aggregated'}")
        
    def _create_tactile_model(self, shape, model_name, pretrained, frozen, use_group_norm, feature_dim):
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
        
        #use group norm to replace batch norm
        if use_group_norm:
            backbone = replace_submodules(
                root_module=backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=max(1, x.num_features // 16), 
                    num_channels=x.num_features
                )
            )
        
        if self.output_all_patches:
            conv_proj = nn.Conv2d(512, feature_dim, kernel_size=1)
            return nn.Sequential(backbone, conv_proj)
        else:
            # 🔥 使用改进的SpatialSoftmax（参考robomimic）
            # SpatialSoftmax输出: (B, 512) -> (B, 512*2) = (B, 1024)
            spatial_softmax = SpatialSoftmax(temperature=1.0)
            projection = nn.Linear(512 * 2, feature_dim)
            return nn.Sequential(backbone, spatial_softmax, projection)
    
    def modalities(self):
        return ['tactile']
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {}
        
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
            
            # 🔥 修复：归一化时保持梯度连接
            # 使用条件归一化，确保梯度能够反向传播
            # 注意：不能用 with torch.no_grad() 包裹归一化操作本身
            with torch.no_grad():
                max_val = tactile_data.max().item()
            
            if max_val > 1.0:
                # 关键：这个除法操作必须在梯度计算图中
                tactile_data = tactile_data / 255.0
            
            expected_shape = self.key_shape_map[key]
            if tactile_data.shape[-2] < 64:
                 tactile_data = F.interpolate(
                    tactile_data, 
                    size=(64, 128),  # 强制放大
                    mode='bilinear', 
                    align_corners=False
                )
            
            feature = self.key_model_map[key](tactile_data)
            
            if self.output_all_patches:
                BT, D, H, W = feature.shape
                feature = feature.flatten(2).transpose(1, 2)
                feature = feature.reshape(B, T * H * W, D)
            else:
                feature = feature.reshape(B, T, -1)
            
            output[key] = feature
        
        return output
    
    def output_feature_dim(self):
        return {key: self.feature_dim for key in self.tactile_keys}


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax池化层（参考robomimic实现）
    
    输出每个通道的期望坐标(x,y)，可以保留空间信息同时降维
    关键改进：确保梯度能够正确反向传播到输入特征图
    """
    
    def __init__(self, temperature=1.0, normalize=False):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 特征图
        Returns:
            output: (B, C*2) 每个通道的(x,y)坐标
        """
        B, C, H, W = x.shape
        
        # 创建归一化的坐标网格 [-1, 1]
        # 🔥 修复：确保坐标网格正确创建且不断开梯度
        pos_x = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        pos_y = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        # 使用 meshgrid 创建坐标网格，注意输出顺序
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing='ij')  # (H, W)
        
        # Reshape for broadcasting: (1, 1, H, W)
        pos_x = pos_x.reshape(1, 1, H, W)
        pos_y = pos_y.reshape(1, 1, H, W)
        
        # Flatten spatial dimensions: (B, C, H*W)
        x_flat = x.reshape(B, C, -1)
        
        # 数值稳定性：减去最大值（可选）
        if self.normalize:
            x_flat = x_flat - x_flat.max(dim=-1, keepdim=True)[0]
        
        # 计算softmax权重: (B, C, H*W)
        # 🔥 关键：确保temperature参与计算图
        weights = F.softmax(x_flat / self.temperature, dim=-1)
        
        # Reshape weights for spatial operations: (B, C, H, W)
        weights = weights.reshape(B, C, H, W)
        
        # 计算期望坐标（加权平均）
        # 🔥 这里的乘法和求和操作都是可微的
        expected_x = (weights * pos_x).sum(dim=[2, 3])  # (B, C)
        expected_y = (weights * pos_y).sum(dim=[2, 3])  # (B, C)
        
        # 拼接x和y坐标: (B, C*2)
        output = torch.cat([expected_x, expected_y], dim=-1)
        
        return output


if __name__ == '__main__':
    print("\n=== TimmTactileEncoder 测试 ===\n")
    
    # 构造shape_meta
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
    
    # 创建共享权重编码器
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
    print(f"特征维度: {list(encoder.output_feature_dim().values())[0]}D")
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
    print(f"输出: {list(out.values())[0].shape} -> 期望: [B=4, T=2, D=768]")
    assert list(out.values())[0].shape == (4, 2, 768), "输出形状不匹配！"
    
    print("\n=== 梯度测试 ===")
    encoder.train()
    encoder.zero_grad()
    
    obs_grad = {
        'left_tactile': torch.randn(2, 2, 1, 16, 32, requires_grad=True),
        'right_tactile': torch.randn(2, 2, 1, 16, 32, requires_grad=True),
    }
    
    intermediate_outputs = {}
    hooks = []
    
    def save_grad_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                intermediate_outputs[f'{name}_grad_out'] = grad_output[0].norm().item()
            if grad_input[0] is not None:
                intermediate_outputs[f'{name}_grad_in'] = grad_input[0].norm().item()
        return hook
    
    model = encoder.key_model_map['left_tactile']
    for i, module in enumerate(model):
        hook = module.register_full_backward_hook(save_grad_hook(f'module_{i}_{module.__class__.__name__}'))
        hooks.append(hook)
    
    output = encoder(obs_grad)
    loss = sum(v.sum() for v in output.values())
    loss.backward()
    
    for hook in hooks:
        hook.remove()
    
    left_grad_norm = obs_grad['left_tactile'].grad.norm().item()
    right_grad_norm = obs_grad['right_tactile'].grad.norm().item()
    
    print(f"\n输入梯度:")
    print(f"  left_tactile: {left_grad_norm:.6f}")
    print(f"  right_tactile: {right_grad_norm:.6f}")
    
    assert left_grad_norm > 0, "left_tactile梯度为0"
    assert right_grad_norm > 0, "right_tactile梯度为0"
    assert not torch.isnan(obs_grad['left_tactile'].grad).any(), "梯度包含NaN"
    
    print(f"\n中间层梯度流:")
    for name in sorted(intermediate_outputs.keys()):
        print(f"  {name}: {intermediate_outputs[name]:.6f}")
    
    param_grads = []
    for name, param in encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_grads.append((name, grad_norm))
    
    print(f"\n模型参数梯度:")
    print(f"  有梯度参数: {len(param_grads)}/{sum(1 for p in encoder.parameters())}")
    if param_grads:
        avg_grad = sum(g for _, g in param_grads) / len(param_grads)
        max_grad = max(param_grads, key=lambda x: x[1])
        min_grad = min(param_grads, key=lambda x: x[1])
        print(f"  平均梯度: {avg_grad:.6f}")
        print(f"  最大梯度: {max_grad[0]} = {max_grad[1]:.6f}")
        print(f"  最小梯度: {min_grad[0]} = {min_grad[1]:.6f}")
    
    spatial_softmax_found = False
    for name, module in encoder.key_model_map['left_tactile'].named_modules():
        if isinstance(module, SpatialSoftmax):
            spatial_softmax_found = True
            break
    
    print(f"\nSpatialSoftmax检查:")
    print(f"  模块存在: {spatial_softmax_found}")
    if 'module_1_SpatialSoftmax_grad_in' in intermediate_outputs:
        print(f"  输入梯度: {intermediate_outputs['module_1_SpatialSoftmax_grad_in']:.6f}")
    if 'module_1_SpatialSoftmax_grad_out' in intermediate_outputs:
        print(f"  输出梯度: {intermediate_outputs['module_1_SpatialSoftmax_grad_out']:.6f}")
    
    print("\n✅ 梯度测试通过\n")
