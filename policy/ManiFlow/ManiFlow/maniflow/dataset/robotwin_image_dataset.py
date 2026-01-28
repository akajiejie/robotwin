from threadpoolctl import threadpool_limits
from typing import Dict, Optional, List
import torch
import numpy as np
import copy
import zarr  
import numcodecs  
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, get_image_range_normalizer
from maniflow.dataset.base_dataset import BaseDataset
from termcolor import cprint

class RoboTwinImageDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            n_obs_steps=2,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            shape_meta: Optional[Dict] = None,
            obs_key_mapping: Optional[Dict[str, str]] = None,  # 🔑 新增：灵活的键名映射
            state_key: str = 'state',  # 🔑 新增：可配置的state键名
            action_key: str = 'action',  # 🔑 新增：可配置的action键名
            **kwargs
            ):
        super().__init__()
        self.task_name = task_name
        self.state_key = state_key
        self.action_key = action_key
        
        cprint(f'Loading RoboTwinDataset from {zarr_path}', 'green')

        # 🔑 动态解析观察空间配置
        self.rgb_keys = []  # 配置中的RGB键名列表
        self.low_dim_keys = []  # 配置中的低维键名列表
        self.obs_key_mapping = {}  # 配置键 -> 缓冲区键的映射
        self.obs_types = {}  # 记录每个键的类型
        
        if shape_meta is not None:
            obs_shape_meta = shape_meta.get('obs', {})
            
            for key, attr in obs_shape_meta.items():
                obs_type = attr.get('type', 'low_dim')
                self.obs_types[key] = obs_type
                
                if obs_type == 'rgb':
                    # 确定缓冲区键名
                    if obs_key_mapping and key in obs_key_mapping:
                        # 使用自定义映射
                        buffer_key = obs_key_mapping[key]
                    else:
                        # 使用默认映射规则：将 _cam 替换为 _camera
                        buffer_key = self._default_key_mapping(key)
                    
                    self.rgb_keys.append(key)
                    self.obs_key_mapping[key] = buffer_key
                    cprint(f"  Registered RGB obs: {key} -> buffer key: {buffer_key}", 'cyan')
                    
                elif obs_type == 'low_dim':
                    # 低维观察通常直接从state读取，但也支持自定义
                    if obs_key_mapping and key in obs_key_mapping:
                        buffer_key = obs_key_mapping[key]
                    else:
                        buffer_key = self.state_key  # 默认使用state_key
                    
                    self.low_dim_keys.append(key)
                    self.obs_key_mapping[key] = buffer_key
                    cprint(f"  Registered low-dim obs: {key} -> buffer key: {buffer_key}", 'cyan')
                
                else:
                    cprint(f"  Warning: Unknown obs type '{obs_type}' for key '{key}', skipping", 'yellow')
        
        # 后备方案：如果没有提供shape_meta，使用默认配置
        if len(self.rgb_keys) == 0 and len(self.low_dim_keys) == 0:
            cprint(f"  Warning: No shape_meta provided, using default configuration", 'yellow')
            self.rgb_keys = ['head_cam']
            self.low_dim_keys = ['agent_pos']
            self.obs_key_mapping = {
                'head_cam': 'head_camera',
                'agent_pos': self.state_key
            }
            self.obs_types = {'head_cam': 'rgb', 'agent_pos': 'low_dim'}
            cprint(f"  Default: head_cam -> head_camera", 'yellow')
            cprint(f"  Default: agent_pos -> {self.state_key}", 'yellow')
        
        # 🔑 动态构建需要加载的缓冲区键列表
        buffer_keys = self._get_required_buffer_keys()
        cprint(f"Loading buffer keys: {buffer_keys}", 'green')
        
        # 🔥 直接从磁盘按需读取,不加载到内存 (数据集78GB,内存只有110GB)
        cprint(f"⚠️  Using on-demand disk loading mode (NO memory copy) - dataset is 78GB", 'red')
        cprint(f"⚠️  Data will be read from disk on-the-fly, may be slower but avoids OOM", 'yellow')
        # 直接打开zarr文件,不复制到内存
        import zarr as zarr_lib
        zarr_group = zarr_lib.open(zarr_path, mode='r')
        
        # 创建过滤后的视图(只包含需要的keys)
        filtered_data = {}
        for key in buffer_keys:
            if key in zarr_group['data']:
                filtered_data[key] = zarr_group['data'][key]
        
        # 构造ReplayBuffer需要的结构
        # meta需要转换为numpy数组(只有元数据,很小)
        meta_dict = {}
        for key, value in zarr_group['meta'].items():
            if hasattr(value, 'shape') and len(value.shape) == 0:
                # 标量
                meta_dict[key] = np.array(value)
            else:
                # 数组(episode_ends等,很小,可以加载到内存)
                meta_dict[key] = value[:]
        
        root = {
            'meta': meta_dict,
            'data': filtered_data
        }
        self.replay_buffer = ReplayBuffer(root=root)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        if max_train_episodes is None:
            max_train_episodes = self.replay_buffer.n_episodes - np.sum(val_mask)
        cprint(f'Maximum training episodes: {max_train_episodes}', 'yellow')
        cprint(f'Validation ratio: {val_ratio}', 'yellow')

        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

        self.zarr_path = zarr_path
        self.train_episodes_num = np.sum(train_mask)
        self.val_episodes_num = np.sum(val_mask)

    def _default_key_mapping(self, config_key: str) -> str:
        """
        默认的键名映射规则
        
        示例:
            'head_cam' -> 'head_camera'
            'left_wrist_cam' -> 'left_wrist_camera'
            'rgb_front' -> 'rgb_front' (不变)
        """
        if config_key.endswith('_cam'):
            return config_key.replace('_cam', '_camera')
        return config_key
    
    def _get_required_buffer_keys(self) -> List[str]:
        """
        获取需要从缓冲区加载的所有唯一键
        """
        buffer_keys = set()
        
        # 添加所有观察键对应的缓冲区键
        for config_key in self.rgb_keys + self.low_dim_keys:
            buffer_key = self.obs_key_mapping[config_key]
            buffer_keys.add(buffer_key)
        
        # 添加action键
        buffer_keys.add(self.action_key)
        
        return sorted(list(buffer_keys))

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    
    def get_normalizer(self, mode='limits', **kwargs):
        """
        动态创建归一化器
        """
        data = {
            'action': self.replay_buffer[self.action_key]
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # 🔑 为所有观察键添加identity normalizer
        for config_key in self.rgb_keys + self.low_dim_keys:
            normalizer[config_key] = SingleFieldLinearNormalizer.create_identity()
        
        return normalizer


    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        将原始样本转换为模型需要的数据格式
        """
        obs_dict = {}
        
        # 🔑 动态加载所有RGB观察
        for config_key in self.rgb_keys:
            buffer_key = self.obs_key_mapping[config_key]
            if buffer_key in sample:
                # RGB数据：假设是uint8 [0-255]，归一化到[0-1]
                raw_data = sample[buffer_key][:,].astype(np.float32)
                # 检查是否需要归一化
                if raw_data.max() > 1.0:
                    raw_data = raw_data / 255.0
                obs_dict[config_key] = raw_data
            else:
                cprint(f"Warning: RGB key '{buffer_key}' not found in sample, skipping", 'red')
        
        # 🔑 动态加载所有低维观察
        for config_key in self.low_dim_keys:
            buffer_key = self.obs_key_mapping[config_key]
            if buffer_key in sample:
                # 低维数据：直接使用
                obs_dict[config_key] = sample[buffer_key][:,].astype(np.float32)
            else:
                cprint(f"Warning: Low-dim key '{buffer_key}' not found in sample, skipping", 'red')
        
        # 🔑 加载action
        if self.action_key not in sample:
            raise KeyError(f"Action key '{self.action_key}' not found in sample!")
        
        data = {
            'obs': obs_dict,
            'action': sample[self.action_key].astype(np.float32)
        }

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)
        data['obs'] = {k: v[T_slice] for k, v in data['obs'].items()}
        torch_data = dict_apply(data, torch.from_numpy)

        return torch_data
