# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# policyconsensus：https://github.com/policyconsensus/policyconsensus.git
# ManiFlow: https://github.com/geyan21/ManiFlow_Policy
# touch_in_the_wild:https://github.com/YolandaXinyueZhu/touch_in_the_wild.git
# --------------------------------------------------------
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from maniflow.common.pytorch_util import replace_submodules
from maniflow.model.tactile.base_sensor import BaseSensoryEncoder


class CropRandomizer(nn.Module):
    """随机裁剪数据增强模块"""
    
    def __init__(self, input_shape: Tuple[int, int, int], crop_height: int, crop_width: int):
        super().__init__()
        self.input_shape = input_shape  # (C, H, W)
        self.crop_height = crop_height
        self.crop_width = crop_width
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 或 (B*T, C, H, W)
        Returns:
            cropped: (B, C, crop_H, crop_W)
        """
        _, _, H, W = x.shape
        if self.training:
            top = torch.randint(0, H - self.crop_height + 1, (1,)).item()
            left = torch.randint(0, W - self.crop_width + 1, (1,)).item()
        else:
            top = (H - self.crop_height) // 2
            left = (W - self.crop_width) // 2
        return x[:, :, top:top + self.crop_height, left:left + self.crop_width]


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax池化层（完全参考robomimic实现）
    输出格式: [x0,y0, x1,y1, ...] 交替排列，与robomimic保持一致
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_kp: int = 32, temperature: float = 1.0):
        super().__init__()
        in_c, in_h, in_w = input_shape
        self._in_c = in_c
        self._in_h = in_h
        self._in_w = in_w
        self._num_kp = num_kp
        self.temperature = temperature
        
        # 1x1卷积将通道数映射到关键点数
        self.nets = nn.Conv2d(in_c, num_kp, kernel_size=1)
        
        # 预计算坐标网格并注册为buffer（与robomimic一致）
        import numpy as np
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., in_w),
            np.linspace(-1., 1., in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, in_h * in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, in_h * in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 特征图
        Returns:
            output: (B, num_kp*2) 格式为 [x0,y0, x1,y1, ...] 交替排列
        """
        B = x.shape[0]
        
        # 1x1卷积: (B, num_kp, H, W)
        feature = self.nets(x)
        
        # [B, num_kp, H, W] -> [B * num_kp, H * W]
        feature = feature.reshape(-1, self._in_h * self._in_w)
        
        # Softmax归一化
        attention = F.softmax(feature / self.temperature, dim=-1)
        
        # 计算期望坐标: [B * num_kp, 1]
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        
        # 拼接: [B * num_kp, 2]
        expected_xy = torch.cat([expected_x, expected_y], dim=1)
        
        # 重塑为 [B, num_kp, 2] 然后 flatten 为 [B, num_kp*2]
        # 这样输出格式为 [x0,y0, x1,y1, ...] 与robomimic一致
        return expected_xy.view(B, self._num_kp, 2).reshape(B, -1)


class TimmTactileEncoder(BaseSensoryEncoder):
    """使用timm库的触觉编码器，复用ResNet18处理触觉数据"""
    
    # 小尺寸图像放大的目标尺寸
    UPSCALE_SIZE = (64, 128)
    UPSCALE_THRESHOLD = 64  # 高度小于此值时放大
    
    def __init__(self,
        shape_meta: dict,
        model_name: str = 'resnet18',
        pretrained: bool = False,
        frozen: bool = False,
        use_group_norm: bool = True,
        share_tactile_model: bool = False,
        feature_dim: int = 64,
        num_kp: int = 32,
        crop_shape: Optional[Union[Tuple[int, int], Dict[str, Tuple[int, int]]]] = None,
        output_all_patches: bool = False,
    ):
        super().__init__()
        
        # 解析触觉传感器keys
        tactile_keys = []
        key_shape_map = {}
        for key, attr in shape_meta['obs'].items():
            if attr.get('type') == 'rgb' and 'tactile' in key.lower():
                tactile_keys.append(key)
                key_shape_map[key] = tuple(attr['shape'])
        tactile_keys = sorted(tactile_keys)
        
        self.output_all_patches = output_all_patches
        self.tactile_keys = tactile_keys
        self.key_shape_map = key_shape_map
        self.feature_dim = feature_dim
        self.num_kp = num_kp
        self.crop_shape = crop_shape
        
        # 创建crop randomizer（基于放大后的尺寸）
        key_crop_map = nn.ModuleDict()
        for key in tactile_keys:
            cs = crop_shape[key] if isinstance(crop_shape, dict) else crop_shape
            if cs is not None:
                # CropRandomizer基于放大后的尺寸创建
                upscaled_shape = self._get_upscaled_shape(key_shape_map[key])
                key_crop_map[key] = CropRandomizer(upscaled_shape, cs[0], cs[1])
        self.key_crop_map = key_crop_map
        
        # 创建模型（需要计算最终输入到backbone的有效形状）
        key_model_map = nn.ModuleDict()
        if share_tactile_model and len(tactile_keys) > 0:
            effective_shape = self._compute_effective_shape(key_shape_map[tactile_keys[0]], crop_shape)
            shared_model = self._create_model(effective_shape, model_name, pretrained, frozen, use_group_norm)
            for key in tactile_keys:
                key_model_map[key] = shared_model
        else:
            for key in tactile_keys:
                cs = crop_shape[key] if isinstance(crop_shape, dict) else crop_shape
                effective_shape = self._compute_effective_shape(key_shape_map[key], cs)
                key_model_map[key] = self._create_model(effective_shape, model_name, pretrained, frozen, use_group_norm)
        self.key_model_map = key_model_map
        
        print(f"✓ 触觉编码器: num_kp={num_kp}, feature_dim={feature_dim}, "
              f"crop={'enabled' if crop_shape else 'disabled'}, "
              f"mode={'all_patches' if output_all_patches else 'aggregated'}")
    
    def _get_upscaled_shape(self, shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """计算放大后的形状"""
        c, h, w = shape
        if h < self.UPSCALE_THRESHOLD:
            return (c, self.UPSCALE_SIZE[0], self.UPSCALE_SIZE[1])
        return shape
    
    def _compute_effective_shape(self, shape: Tuple[int, int, int], cs) -> Tuple[int, int, int]:
        """计算经过放大和裁剪后的有效形状"""
        upscaled = self._get_upscaled_shape(shape)
        if cs is not None:
            return (upscaled[0], cs[0], cs[1])
        return upscaled
        
    def _create_model(self, shape, model_name, pretrained, frozen, use_group_norm):
        """
        Args:
            shape: (C, H, W) 输入形状（经过放大和裁剪后的有效形状）
        """
        in_channels, in_h, in_w = shape
        
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
            backbone_out_channels = 512
            # ResNet下采样32倍
            import math
            out_h = int(math.ceil(in_h / 32.))
            out_w = int(math.ceil(in_w / 32.))
        else:
            raise NotImplementedError(f"Unsupported model: {model_name}")
        
        if use_group_norm:
            backbone = replace_submodules(
                root_module=backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=max(1, x.num_features // 16),
                    num_channels=x.num_features,
                    eps=x.eps,
                    affine=x.affine
                )
            )
        
        if self.output_all_patches:
            conv_proj = nn.Conv2d(backbone_out_channels, self.feature_dim, kernel_size=1)
            return nn.Sequential(backbone, conv_proj)
        else:
            # SpatialSoftmax需要知道backbone输出的空间尺寸
            spatial_softmax = SpatialSoftmax(
                input_shape=(backbone_out_channels, out_h, out_w),
                num_kp=self.num_kp
            )
            projection = nn.Linear(self.num_kp * 2, self.feature_dim)
            return nn.Sequential(backbone, spatial_softmax, projection)
    
    def modalities(self):
        return ['tactile']
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {}
        
        for key in self.tactile_keys:
            if key not in obs:
                continue
            
            tactile_data = obs[key]
            
            # 处理时序维度
            if len(tactile_data.shape) == 5:
                B, T = tactile_data.shape[:2]
                tactile_data = tactile_data.reshape(B * T, *tactile_data.shape[2:])
            else:
                B, T = tactile_data.shape[0], 1
            
            # 归一化
            with torch.no_grad():
                max_val = tactile_data.max().item()
            if max_val > 1.0:
                tactile_data = tactile_data / 255.0
            
            # 图像放大（针对小尺寸触觉数据）
            if tactile_data.shape[-2] < self.UPSCALE_THRESHOLD:
                tactile_data = F.interpolate(tactile_data, size=self.UPSCALE_SIZE, mode='bilinear', align_corners=False)
            
            # 随机裁剪
            if key in self.key_crop_map:
                tactile_data = self.key_crop_map[key](tactile_data)
            
            # 特征提取
            feature = self.key_model_map[key](tactile_data)
            
            if self.output_all_patches:
                BT, D, H, W = feature.shape
                feature = feature.flatten(2).transpose(1, 2).reshape(B, T * H * W, D)
            else:
                feature = feature.reshape(B, T, -1)
            
            output[key] = feature
        
        return output
    
    def output_feature_dim(self):
        return {key: self.feature_dim for key in self.tactile_keys}


if __name__ == '__main__':
    import numpy as np
    
    print("\n=== TimmTactileEncoder 测试 ===\n")
    
    shape_meta = {
        'obs': {
            'head_cam': {'shape': [3, 224, 224], 'type': 'rgb', 'horizon': 2},
            'left_tactile': {'shape': [1, 16, 32], 'type': 'rgb', 'horizon': 2},
            'right_tactile': {'shape': [1, 16, 32], 'type': 'rgb', 'horizon': 2},
            'agent_pos': {'shape': [14], 'type': 'low_dim', 'horizon': 2},
        }
    }
    
    # 测试1: 基础功能（无crop）
    print("--- 测试1: 基础功能 ---")
    encoder = TimmTactileEncoder(
        shape_meta=shape_meta,
        model_name='resnet18',
        use_group_norm=True,
        share_tactile_model=True,
        feature_dim=64,
        num_kp=32,
    )
    
    print(f"触觉传感器: {encoder.tactile_keys}")
    print(f"特征维度: {encoder.feature_dim}D, 关键点数: {encoder.num_kp}")
    print(f"参数量: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"权重共享: {encoder.key_model_map['left_tactile'] is encoder.key_model_map['right_tactile']}")
    
    obs = {
        'left_tactile': torch.randn(4, 2, 1, 16, 32),
        'right_tactile': torch.randn(4, 2, 1, 16, 32),
    }
    
    with torch.no_grad():
        out = encoder(obs)
    print(f"输入: [B=4, T=2, C=1, H=16, W=32]")
    print(f"输出: {list(out.values())[0].shape} -> 期望: [B=4, T=2, D=64]")
    assert list(out.values())[0].shape == (4, 2, 64), "输出形状不匹配！"
    
    # 测试2: CropRandomizer
    print("\n--- 测试2: CropRandomizer ---")
    encoder_crop = TimmTactileEncoder(
        shape_meta=shape_meta,
        model_name='resnet18',
        use_group_norm=True,
        share_tactile_model=True,
        feature_dim=64,
        num_kp=32,
        crop_shape=(56, 112),  # 从64x128裁剪到56x112
    )
    
    encoder_crop.train()
    out_train = encoder_crop(obs)
    encoder_crop.eval()
    out_eval = encoder_crop(obs)
    print(f"训练模式输出: {list(out_train.values())[0].shape}")
    print(f"评估模式输出: {list(out_eval.values())[0].shape}")
    
    # 测试3: 梯度流
    print("\n--- 测试3: 梯度流 ---")
    encoder.train()
    encoder.zero_grad()
    
    obs_grad = {
        'left_tactile': torch.randn(2, 2, 1, 16, 32, requires_grad=True),
        'right_tactile': torch.randn(2, 2, 1, 16, 32, requires_grad=True),
    }
    
    output = encoder(obs_grad)
    loss = sum(v.sum() for v in output.values())
    loss.backward()
    
    left_grad = obs_grad['left_tactile'].grad.norm().item()
    right_grad = obs_grad['right_tactile'].grad.norm().item()
    print(f"输入梯度: left={left_grad:.6f}, right={right_grad:.6f}")
    
    param_grads = [(n, p.grad.norm().item()) for n, p in encoder.named_parameters() if p.grad is not None]
    print(f"有梯度参数: {len(param_grads)}/{sum(1 for _ in encoder.parameters())}")
    
    assert left_grad > 0 and right_grad > 0, "梯度为0"
    assert not torch.isnan(obs_grad['left_tactile'].grad).any(), "梯度包含NaN"
    
    # 测试4: SpatialSoftmax输出格式验证（与robomimic对比）
    print("\n--- 测试4: SpatialSoftmax输出格式验证 ---")
    
    # robomimic的SpatialSoftmax实现
    class RobomimicSpatialSoftmax(nn.Module):
        def __init__(self, input_shape, num_kp=32, temperature=1.0):
            super().__init__()
            self._in_c, self._in_h, self._in_w = input_shape
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
            self.temperature = temperature
            pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self._in_w),
                np.linspace(-1., 1., self._in_h)
            )
            pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
            pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
            self.register_buffer('pos_x', pos_x)
            self.register_buffer('pos_y', pos_y)
        
        def forward(self, feature):
            feature = self.nets(feature)
            feature = feature.reshape(-1, self._in_h * self._in_w)
            attention = F.softmax(feature / self.temperature, dim=-1)
            expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
            expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.cat([expected_x, expected_y], 1)
            return expected_xy.view(-1, self._num_kp, 2).reshape(-1, self._num_kp * 2)
    
    # 对比测试
    torch.manual_seed(42)
    test_input = torch.randn(2, 512, 2, 4)
    
    robomimic_ss = RobomimicSpatialSoftmax((512, 2, 4), num_kp=32)
    our_ss = SpatialSoftmax((512, 2, 4), num_kp=32)
    
    # 复制权重
    our_ss.nets.weight.data = robomimic_ss.nets.weight.data.clone()
    our_ss.nets.bias.data = robomimic_ss.nets.bias.data.clone()
    
    robomimic_out = robomimic_ss(test_input)
    our_out = our_ss(test_input)
    
    print(f"robomimic输出形状: {robomimic_out.shape}")
    print(f"我们的输出形状: {our_out.shape}")
    print(f"最大差异: {(robomimic_out - our_out).abs().max().item():.10f}")
    print(f"输出等效: {torch.allclose(robomimic_out, our_out, atol=1e-6)}")
    
    assert torch.allclose(robomimic_out, our_out, atol=1e-6), "SpatialSoftmax输出与robomimic不一致！"
    
    print("\n✅ 所有测试通过\n")
