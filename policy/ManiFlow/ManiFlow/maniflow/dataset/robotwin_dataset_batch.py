import sys, os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '..'))
sys.path.append(os.path.join(parent_directory, '../..'))


from typing import Dict
import torch
import numba
import numpy as np
import copy
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, get_image_range_normalizer
from maniflow.dataset.base_dataset import BaseDataset
from maniflow.model.vision_3d.point_process import PointCloudColorJitter
import random
from termcolor import cprint

class RoboTwinDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            use_img=False,
            use_depth=False,
            use_point_cloud=True,
            aug_color_cfg=None,
            batch_size=128,
            **kwargs
            ):
        super().__init__()
        self.task_name = task_name
        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
        zarr_path = os.path.join(parent_directory, zarr_path)
        self.use_img = use_img
        self.use_depth = use_depth
        self.use_point_cloud = use_point_cloud
        self.aug_color_cfg = aug_color_cfg
        cprint(f'Maximum training episodes: {max_train_episodes}', 'yellow')
        cprint(f'Validation ratio: {val_ratio}', 'yellow')
        cprint(f'Loading RoboTwinDataset from {zarr_path}', 'green')
        cprint(f'Using img: {self.use_img}, depth: {self.use_depth}, point cloud: {self.use_point_cloud}', 'green')

        if self.use_point_cloud and self.aug_color_cfg is not None:
            self.aug_color = self.aug_color_cfg['aug_color']
            if self.aug_color:
                aug_color_params = self.aug_color_cfg['params']
                self.aug_prob = self.aug_color_cfg['prob']
                same_on_batch = self.aug_color_cfg.get('same_on_batch', False)
                self.pc_jitter = PointCloudColorJitter(
                    brightness=aug_color_params[0],
                    contrast=aug_color_params[1], 
                    saturation=aug_color_params[2],
                    hue=aug_color_params[3],
                    same_on_batch=same_on_batch,
                    aug_prob=self.aug_prob
                )
                cprint(f'Using point cloud color jitter with params: {aug_color_params} and prob: {self.aug_prob}', 'red')
        else:
            self.aug_color = False
            cprint(f'Not using point cloud color jitter', 'red')

        buffer_keys = [
            'state', 
            'action',]
        if self.use_img:
            buffer_keys.append('head_camera') # default use head_camera
        if self.use_point_cloud:
            buffer_keys.append('point_cloud')

        self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=buffer_keys)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
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

        self.zarr_path = zarr_path
        self.train_episodes_num = np.sum(train_mask)
        self.val_episodes_num = np.sum(val_mask)

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

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
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:]
            }

        if self.use_point_cloud:
            data['point_cloud'] = self.replay_buffer['point_cloud']
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        if self.use_img:
            normalizer['head_cam'] = get_image_range_normalizer()
            normalizer["front_cam"] = get_image_range_normalizer()
            normalizer["left_cam"] = get_image_range_normalizer()
            normalizer["right_cam"] = get_image_range_normalizer()
        
        return normalizer


    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3), float32
        if self.use_img:
            # head_cam = np.moveaxis(sample['head_camera'],-1,1)
            assert np.min(sample['head_camera']) >= 0 and np.max(sample['head_camera']) <= 255 and np.max(sample['head_camera']) > 1, \
                f"Image values should be in [0, 255], got min: {np.min(sample['head_camera'])}, max: {np.max(sample['head_camera'])}"
            # Normalize image to [0, 1]
            head_cam = sample['head_camera'][:,].astype(np.float32) / 255.0 # (T, 3, 240, 320)
        if self.use_point_cloud:
            point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 512, 3), float32

        data = {
            'obs': {
                'agent_pos': agent_pos,
                },
            'action': sample['action'].astype(np.float32)}

        if self.use_img:
            data['obs']['head_cam'] = head_cam
        if self.use_point_cloud:
            data['obs']['point_cloud'] = point_cloud
        
        return data
    
    # def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    #     sample = self.sampler.sample_sequence(idx)
    #     data = self._sample_to_data(sample)
    #     torch_data = dict_apply(data, torch.from_numpy)

    #     # point cloud color jitter augmentation
    #     if self.use_point_cloud and self.aug_color:
    #         if random.random() > self.aug_prob:
    #             return torch_data
    #         else:
    #             T, N, C = torch_data['obs']['point_cloud'].shape
    #             pc_reshaped = torch_data['obs']['point_cloud'].reshape(-1, C)
    #             pc_reshaped = self.pc_jitter(pc_reshaped)
    #             torch_data['obs']['point_cloud'] = pc_reshaped.reshape(T, N, C) 

    #     return torch_data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)
    
    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        if self.use_img:
            head_cam = samples["head_camera"].to(device, non_blocking=True) / 255.0
            # front_cam = samples['front_camera'].to(device, non_blocking=True) / 255.0
            # left_cam = samples['left_camera'].to(device, non_blocking=True) / 255.0
            # right_cam = samples['right_camera'].to(device, non_blocking=True) / 255.0
        if self.use_point_cloud:
            point_cloud = samples["point_cloud"].to(device, non_blocking=True)
            # point cloud color jitter augmentation
            if self.aug_color:
                B, T, N, C = point_cloud.shape
                pc_reshaped = point_cloud.reshape(-1, N, C)
                pc_reshaped = self.pc_jitter(pc_reshaped)
                point_cloud = pc_reshaped.reshape(B, T, N, C)

        action = samples["action"].to(device, non_blocking=True)
        data = {
            "obs": {
                # "head_cam": head_cam,  # B, T, 3, H, W
                # # 'front_cam': front_cam, # B, T, 3, H, W
                # # 'left_cam': left_cam, # B, T, 3, H, W
                # # 'right_cam': right_cam, # B, T, 3, H, W
                "agent_pos": agent_pos,  # B, T, D
            },
            "action": action,  # B, T, D
        }
        if self.use_img:
            data['obs']['head_cam'] = head_cam
        if self.use_point_cloud:
            data['obs']['point_cloud'] = point_cloud

        return data

def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)

def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)


if __name__ == '__main__':
    # Test dataset
    zarr_path = '/gscratch/scrubbed/geyan/projects/RoboTwin/policy/ManiFlow/data/lift_pot-demo_randomized-50.zarr'
    dataset = RoboTwinDataset(zarr_path=zarr_path, 
                            horizon=1, 
                            pad_before=0, 
                            pad_after=0,
                            use_img=True,
                            use_depth=False,
                            use_point_cloud=True)
    print(f"Dataset size: {len(dataset)}")
    print(f"Train episodes: {dataset.train_episodes_num}")
    print(f"Validation episodes: {dataset.val_episodes_num}")
    print(f"Task name: {dataset.task_name}")
    print(f"Use image: {dataset.use_img}")
    print(f"Use depth: {dataset.use_depth}")
    print(f"Use point cloud: {dataset.use_point_cloud}")

    # Test sampling
    sample = dataset[0]
    print("\nSample structure:")
    for key, value in sample.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: shape={subvalue.shape if hasattr(subvalue, 'shape') else 'scalar'}")
        else:
            print(f"{key}: shape={value.shape if hasattr(value, 'shape') else 'scalar'}")
        
        # print image value range if it's an image
        if key == 'obs' and 'head_cam' in value:
            print(f"  head_cam: min={value['head_cam'].min()}, max={value['head_cam'].max()}")


    # Test normalizer
    normalizer = dataset.get_normalizer()
    print("\nNormalizer stats:")
    print(normalizer.get_input_stats())
    print(normalizer.get_output_stats())
    
    # Test validation set
    val_dataset = dataset.get_validation_dataset()
    print(f"\nValidation dataset size: {len(val_dataset)}")