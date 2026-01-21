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

logger = logging.getLogger(__name__)

class AttentionPool2d(nn.Module):
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
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
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
    

class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            global_pool: str,
            transforms: list,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            feature_aggregation: str='spatial_embedding',
            downsample_ratio: int=32,
            position_encording: str='learnable',
            # 🆕 输出token序列而非拼接向量（用于模态级MoE）
            output_token_sequence: bool=False,

        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        assert global_pool == ''

        if model_name == "r3m":
            from r3m import load_r3m
            model = load_r3m("resnet18", pretrained=pretrained) # resnet18, resnet34
            model.eval()
            cprint(f"Loaded R3M model using {model_name}. pretrained={pretrained}", 'green')
        else:
            model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool, # '' means no pooling
                num_classes=0            # remove classification layer
            )

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        
        feature_dim = None
        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('r3m'):
            # feature_dim = 2048
            feature_dim = 512
        elif 'siglip' in model_name.lower():
            # SigLIP models from timm
            if 'base' in model_name:
                feature_dim = 768  # SigLIP Base (86M parameters)
            elif 'large' in model_name:
                feature_dim = 1024  # SigLIP Large
            elif 'so400m' in model_name:
                feature_dim = 1152  # SigLIP So400M
            else:
                feature_dim = 768  # Default to base size
            cprint(f"Using SigLIP model {model_name} with feature_dim={feature_dim}", 'green')
        
        self.feature_dim = feature_dim

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model

                this_transform = transform
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
            else:
                cprint(f"Skipping obs key {key} with type {type}", 'red')
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('low_dim_keys keys:', low_dim_keys)

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.feature_aggregation = feature_aggregation

        # 🔧 修复: SigLIP模型名也包含'vit'，需要先检查SigLIP
        if 'siglip' in model_name.lower():
            # SigLIP使用mean pooling，不需要特殊处理feature_aggregation
            if self.feature_aggregation not in ['avg', 'all_tokens', None]:
                logger.warn(f'SigLIP uses mean pooling by default. feature_aggregation ({self.feature_aggregation}) may not work as expected!')
        elif model_name.startswith('vit'):
            # 普通ViT模型使用CLS token
            if self.feature_aggregation == 'all_tokens':
                # Use all tokens from ViT
                pass
            elif self.feature_aggregation is not None:
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = None
        
        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'transformer':
            if position_encording == 'learnable':
                self.position_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim))
            elif position_encording == 'sinusoidal':
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                self.position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(2 * num_features) / feature_dim))
                self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                self.position_embedding[:, 1::2] = torch.cos(position * div_term)
            self.aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4)
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        # 🆕 输出token序列模式
        self.output_token_sequence = output_token_sequence
        self.imagenet_norm = imagenet_norm
        
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature):
        if self.model_name == 'r3m':
            return feature
        
        # Handle SigLIP models (no CLS token, use mean pooling)
        if 'siglip' in self.model_name.lower():
            if self.feature_aggregation == 'avg' or self.feature_aggregation is None:
                # SigLIP doesn't have CLS token, use mean pooling over all tokens
                return torch.mean(feature, dim=1)  # B, N, D -> B, D
            elif self.feature_aggregation == 'all_tokens':
                # Return all tokens (for transformer-based downstream processing)
                return feature
            else:
                logger.warn(f'SigLIP uses mean pooling by default. feature_aggregation ({self.feature_aggregation}) may not work as expected!')
                return torch.mean(feature, dim=1)
        
        # Handle ViT models (has CLS token)
        if self.model_name.startswith('vit'):
            if self.feature_aggregation is None or self.feature_aggregation == 'cls_token':
                return feature[:, 0, :]  # Use CLS token
            elif self.feature_aggregation == 'all_tokens':
                # Return all tokens (for transformer-based downstream processing)
                return feature
            else:
                logger.warn(f'ViT will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                return feature[:, 0, :]
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1])
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1])
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif self.feature_aggregation == 'transformer':
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert self.feature_aggregation is None
            return feature
        
    def forward(self, obs_dict):
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # 🆕 Token序列模式：分别收集不同模态的特征
        if self.output_token_sequence:
            return self._forward_token_sequence(obs_dict, batch_size)
        
        # 原始模式：拼接所有特征为一个向量
        features = list()
        
        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            # normalize image by hand
            if img.max() > 1.0:
                # assume input in [0, 255]
                img = img / 255.0
            if img.shape[-1] == 3:
                if len(img.shape) == 5:
                    # B, T, H, W, C --> B, T, C, H, W
                    img = img.permute(0, 1, 4, 2, 3)
                elif len(img.shape) == 4:
                    # B, H, W, C --> B, C, H, W
                    img = img.permute(0, 3, 1, 2)

            B, T = img.shape[:2]
            assert B == batch_size
            img = img.reshape(B*T, *img.shape[2:])

            if img.shape[2:] != self.key_shape_map[key]:
                target_H, target_W = self.key_shape_map[key][1], self.key_shape_map[key][2]
                # do torchvision resize
                # img shape: Bx3xHxW
                # new size: Bx3xnHxnW
                img = F.interpolate(img, size=(target_H, target_W), mode='bilinear', align_corners=False)
            
            assert img.shape[1:] == self.key_shape_map[key]
            img = self.key_transform_map[key](img).to(self.device)
            # Ensure the image is float32 to match model weights
            img = img.float()
            raw_feature = self.key_model_map[key](img).to(self.device)
            feature = self.aggregate_feature(raw_feature)
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key].to(self.device)
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))
        
        # concatenate all features
        result = torch.cat(features, dim=-1)

        return result
    
    def _forward_token_sequence(self, obs_dict, batch_size):
        """
        🆕 输出token序列格式: (B, L_tokens, D)
        
        每个相机输出1个token (mean pooling后)，每个时间步的低维状态输出1个token
        输出顺序: [head_cam_t0, head_cam_t1, ..., wrist_cam_t0, wrist_cam_t1, ..., proprio_t0, proprio_t1, ...]
        
        Returns:
            result: (B, L_tokens, D) token序列
            modality_info: dict 包含模态长度信息
        """
        head_features = []   # 头部相机特征
        wrist_features = []  # 腕部相机特征
        proprio_features = []  # 本体感知特征
        
        # 按相机类型分类
        head_keys = [k for k in self.rgb_keys if 'head' in k.lower() or 'front' in k.lower()]
        wrist_keys = [k for k in self.rgb_keys if 'wrist' in k.lower() or 'hand' in k.lower() or 'left' in k.lower() or 'right' in k.lower()]
        # 如果没有明确分类，默认第一个为head，其余为wrist
        if not head_keys and not wrist_keys:
            head_keys = self.rgb_keys[:1] if len(self.rgb_keys) >= 1 else []
            wrist_keys = self.rgb_keys[1:] if len(self.rgb_keys) > 1 else []
        
        # 处理头部相机
        for key in head_keys:
            feature = self._process_single_rgb(obs_dict[key], key, batch_size)
            head_features.append(feature)  # (B, T, D)
        
        # 处理腕部相机
        for key in wrist_keys:
            feature = self._process_single_rgb(obs_dict[key], key, batch_size)
            wrist_features.append(feature)  # (B, T, D)
        
        # 处理低维状态（本体感知）
        for key in self.low_dim_keys:
            data = obs_dict[key].to(self.device)
            B, T = data.shape[:2]
            assert B == batch_size
            # 将低维状态投影到feature_dim维度
            # data: (B, T, low_dim) -> (B, T, D)
            # 这里需要一个投影层，如果没有则保持原样
            proprio_features.append(data)  # (B, T, low_dim)
        
        # 合并特征
        # head: (B, n_head_cams * T, D)
        if head_features:
            head_tokens = torch.cat(head_features, dim=1)  # (B, n_head * T, D)
        else:
            head_tokens = None
            
        # wrist: (B, n_wrist_cams * T, D)
        if wrist_features:
            wrist_tokens = torch.cat(wrist_features, dim=1)  # (B, n_wrist * T, D)
        else:
            wrist_tokens = None
        
        # proprio: (B, T, low_dim) - 需要投影到D维度
        if proprio_features:
            proprio_concat = torch.cat(proprio_features, dim=-1)  # (B, T, total_low_dim)
            # 投影到feature_dim（如果需要）
            if not hasattr(self, 'proprio_proj'):
                total_low_dim = proprio_concat.shape[-1]
                self.proprio_proj = nn.Linear(total_low_dim, self.feature_dim).to(self.device)
                nn.init.xavier_uniform_(self.proprio_proj.weight)
                nn.init.zeros_(self.proprio_proj.bias)
            proprio_tokens = self.proprio_proj(proprio_concat.float())  # (B, T, D)
        else:
            proprio_tokens = None
        
        # 拼接所有token: [head, wrist, proprio]
        all_tokens = []
        modality_info = {'head': 0, 'wrist': 0, 'proprio': 0}
        
        if head_tokens is not None:
            all_tokens.append(head_tokens)
            modality_info['head'] = head_tokens.shape[1]
        if wrist_tokens is not None:
            all_tokens.append(wrist_tokens)
            modality_info['wrist'] = wrist_tokens.shape[1]
        if proprio_tokens is not None:
            all_tokens.append(proprio_tokens)
            modality_info['proprio'] = proprio_tokens.shape[1]
        
        result = torch.cat(all_tokens, dim=1)  # (B, L_total, D)
        
        # 保存模态信息供外部使用
        self._last_modality_info = modality_info
        
        return result
    
    def _process_single_rgb(self, img, key, batch_size):
        """处理单个RGB输入，返回(B, T, D)格式的特征"""
        # normalize image by hand
        if img.max() > 1.0:
            img = img / 255.0
        if img.shape[-1] == 3:
            if len(img.shape) == 5:
                img = img.permute(0, 1, 4, 2, 3)
            elif len(img.shape) == 4:
                img = img.permute(0, 3, 1, 2)
        
        B, T = img.shape[:2]
        assert B == batch_size
        img = img.reshape(B*T, *img.shape[2:])
        
        if img.shape[2:] != self.key_shape_map[key]:
            target_H, target_W = self.key_shape_map[key][1], self.key_shape_map[key][2]
            img = F.interpolate(img, size=(target_H, target_W), mode='bilinear', align_corners=False)
        
        assert img.shape[1:] == self.key_shape_map[key]
        img = self.key_transform_map[key](img).to(self.device)
        img = img.float()
        raw_feature = self.key_model_map[key](img).to(self.device)
        
        # Mean pooling得到每帧一个token
        feature = self.aggregate_feature(raw_feature)  # (B*T, D)
        feature = feature.reshape(B, T, -1)  # (B, T, D)
        
        return feature
    
    def get_modality_info(self):
        """获取最近一次forward的模态长度信息"""
        return getattr(self, '_last_modality_info', None)
    

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
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


if __name__=='__main__':
    timm_obs_encoder = TimmObsEncoder(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='',
        transforms=None
    )