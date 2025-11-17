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
from maniflow.model.vision_3d.point_process import PointCloudColorJitterSingle
import random
from termcolor import cprint

def pc_camera_to_world(pc, extrinsic):
        R = extrinsic[:3, :3]
        T = extrinsic[:3, 3]
        pc = (R @ pc.T).T + T
        return pc

def depth_to_pointcloud(depth_map, camera_intrinsic, cam2world_matrix):
    # Handle both single and batch inputs
    if depth_map.ndim == 2:  # Single depth map (H, W)
        return _single_depth_to_pointcloud(depth_map, camera_intrinsic, cam2world_matrix)
    elif depth_map.ndim == 3:  # Batch of depth maps (B, H, W)
        return _batch_depth_to_pointcloud(depth_map, camera_intrinsic, cam2world_matrix)
    else:
        raise ValueError(f"Unsupported depth_map shape: {depth_map.shape}")

def _single_depth_to_pointcloud(depth_map, camera_intrinsic, cam2world_matrix):
    """Original single depth map processing"""
    depth_map = depth_map.reshape(depth_map.shape[0], depth_map.shape[1])
    rows, cols = depth_map.shape[0], depth_map.shape[1]
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    z = depth_map
    x = (u - camera_intrinsic[0][2]) * z / camera_intrinsic[0][0]
    y = (v - camera_intrinsic[1][2]) * z / camera_intrinsic[1][1]
    points = np.dstack((x, y, z))
    per_point_xyz = points.reshape(-1, 3)
    
    point_xyz = per_point_xyz
    pcd_camera = np.array(point_xyz)
    Rtilt_rot = cam2world_matrix[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Rtilt_trl = cam2world_matrix[:3, 3]
    cam2_wolrd = np.eye(4)
    cam2_wolrd[:3, :3] = Rtilt_rot
    cam2_wolrd[:3, 3] = Rtilt_trl
    pcd_world = pc_camera_to_world(pcd_camera, cam2_wolrd)
    return pcd_world

def _batch_depth_to_pointcloud(depth_batch, intrinsic_batch, cam2world_batch):
    """Batched depth map processing"""
    B, H, W = depth_batch.shape
    
    # Create pixel coordinate grids (same for all batches)
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # (H, W)
    u = u[None, :, :].repeat(B, axis=0)  # (B, H, W)
    v = v[None, :, :].repeat(B, axis=0)  # (B, H, W)
    
    # Convert to camera coordinates
    z = depth_batch  # (B, H, W)
    x = (u - intrinsic_batch[:, 0:1, 2:3]) * z / intrinsic_batch[:, 0:1, 0:1]  # (B, H, W)
    y = (v - intrinsic_batch[:, 1:2, 2:3]) * z / intrinsic_batch[:, 1:2, 1:2]  # (B, H, W)
    
    # Stack and reshape to points
    points_camera = np.stack([x, y, z], axis=-1)  # (B, H, W, 3)
    points_camera = points_camera.reshape(B, -1, 3)  # (B, H*W, 3)
    
    # Apply transformation for each batch
    R_correction = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Rtilt_rot = cam2world_batch[:, :3, :3] @ R_correction[None, :, :]  # (B, 3, 3)
    Rtilt_trl = cam2world_batch[:, :3, 3]  # (B, 3)
    
    # Transform points: (B, 3, 3) @ (B, H*W, 3).T + (B, 3, 1) = (B, 3, H*W)
    points_world = np.matmul(Rtilt_rot, points_camera.transpose(0, 2, 1)) + Rtilt_trl[:, :, None]
    points_world = points_world.transpose(0, 2, 1)  # (B, H*W, 3)
    
    return points_world

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
            use_multi_view_img=False,
            use_depth=False,
            use_point_cloud=True,
            use_point_map=False,
            aug_color_cfg=None,
            img_cond_mask_ratio=0.0,
            point_cond_mask_ratio=0.0,
            use_dynamic_masking=False,
            no_overlap=False,
            **kwargs
            ):
        super().__init__()
        self.task_name = task_name
        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
        zarr_path = os.path.join(parent_directory, zarr_path)
        self.use_img = use_img
        self.use_multi_view_img = use_multi_view_img
        self.use_depth = use_depth
        self.use_point_cloud = use_point_cloud
        self.use_point_map = use_point_map
        self.aug_color_cfg = aug_color_cfg
        self.img_cond_mask_ratio = img_cond_mask_ratio
        self.point_cond_mask_ratio = point_cond_mask_ratio
        self.use_dynamic_masking = use_dynamic_masking
        self.no_overlap = no_overlap

        # For dynamic masking
        self._current_img_mask_ratio = img_cond_mask_ratio
        self._current_point_mask_ratio = point_cond_mask_ratio

        cprint(f'Maximum training episodes: {max_train_episodes}', 'yellow')
        cprint(f'Validation ratio: {val_ratio}', 'yellow')
        cprint(f'Loading RoboTwinDataset from {zarr_path}', 'green')
        cprint(f'Using img: {self.use_img}, multi-view img: {self.use_multi_view_img}, depth: {self.use_depth}, point cloud: {self.use_point_cloud}, point map: {self.use_point_map}', 'green')
        cprint(f'Image condition mask ratio: {self.img_cond_mask_ratio}', 'green')
        cprint(f'Point condition mask ratio: {self.point_cond_mask_ratio}', 'green')
        cprint(f'Using dynamic masking: {self.use_dynamic_masking}', 'green')
        cprint(f'No overlap between image and point cloud masking: {self.no_overlap}', 'green')

        if self.use_point_cloud and self.aug_color_cfg is not None:
            self.aug_color = self.aug_color_cfg['aug_color']
            if self.aug_color:
                aug_color_params = self.aug_color_cfg['params']
                self.aug_prob = self.aug_color_cfg['prob']
                self.pc_jitter = PointCloudColorJitterSingle(
                    brightness=aug_color_params[0],
                    contrast=aug_color_params[1], 
                    saturation=aug_color_params[2],
                    hue=aug_color_params[3],
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
        if self.use_point_map:
            buffer_keys.append('head_depth') # default use head_depth, HxW, float32, in meters
            buffer_keys.append('head_intrinsic_cv') # 3x3, float32
            buffer_keys.append('head_cam2world_gl') # 4x4, float32
        if self.use_multi_view_img:
            buffer_keys.append('left_camera')
            buffer_keys.append('right_camera')
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


    def set_mask_ratios(self, img_ratio, point_ratio):
        """Update mask ratios dynamically during training"""
        self._current_img_mask_ratio = img_ratio
        self._current_point_mask_ratio = point_ratio
        cprint(f'Updated image condition mask ratio: {self._current_img_mask_ratio}', 'yellow')
        cprint(f'Updated point condition mask ratio: {self._current_point_mask_ratio}', 'yellow')

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
        
        if self.use_point_map:
            normalizer['head_depth'] = SingleFieldLinearNormalizer()
            normalizer['head_intrinsic_cv'] = SingleFieldLinearNormalizer()
            normalizer['head_cam2world_gl'] = SingleFieldLinearNormalizer()
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
        if self.use_multi_view_img:
            left_cam = sample['left_camera'][:,].astype(np.float32) / 255.0 # (T, 3, 240, 320)
            right_cam = sample['right_camera'][:,].astype(np.float32) / 255.0 # (T, 3, 240, 320)
        if self.use_point_cloud:
            point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 512, 3), float32
        
        if self.use_point_map:
            head_depth = sample['head_depth'][:,].astype(np.float32) # (T, H, W), float32, in meters
            head_intrinsic_cv = sample['head_intrinsic_cv'][:,].astype(np.float32) # (T, 3, 3), float32
            head_cam2world_gl = sample['head_cam2world_gl'][:,].astype(np.float32) # (T, 4, 4), float32
            # convert depth map to point cloud
            b, h, w = head_depth.shape
            head_point_cloud = depth_to_pointcloud(head_depth, head_intrinsic_cv, head_cam2world_gl) # (B, H*W, 3)
            # reshape to (B, H, W, 3)
            head_point_cloud = head_point_cloud.reshape(b, h, w, 3).astype(np.float32)
            # then to (B, 3, H, W)
            head_point_cloud = np.moveaxis(head_point_cloud, -1, 1) # (B, 3, H, W)
            

        data = {
            'obs': {
                'agent_pos': agent_pos,
                },
            'action': sample['action'].astype(np.float32)}

        if self.use_img:
            data['obs']['head_cam'] = head_cam
        if self.use_multi_view_img:
            data['obs']['left_cam'] = left_cam
            data['obs']['right_cam'] = right_cam
        if self.use_point_cloud:
            data['obs']['point_cloud'] = point_cloud
        if self.use_point_map:
            data['obs']['head_depth'] = head_depth
            data['obs']['head_intrinsic_cv'] = head_intrinsic_cv
            data['obs']['head_cam2world_gl'] = head_cam2world_gl
            data['obs']['head_point_cloud'] = head_point_cloud
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)

        # point cloud color jitter augmentation
        if self.use_point_cloud and self.aug_color:
            if random.random() > self.aug_prob:
                # return torch_data
                pass
            else:
                T, N, C = torch_data['obs']['point_cloud'].shape
                pc_reshaped = torch_data['obs']['point_cloud'].reshape(-1, C)
                pc_reshaped = self.pc_jitter(pc_reshaped)
                torch_data['obs']['point_cloud'] = pc_reshaped.reshape(T, N, C) 
        
        # Use current mask ratios (either static or dynamic)
        img_mask_ratio = self._current_img_mask_ratio if self.use_dynamic_masking else self.img_cond_mask_ratio
        point_mask_ratio = self._current_point_mask_ratio if self.use_dynamic_masking else self.point_cond_mask_ratio
        
        
        # apply image condition mask, set to 0
        if self.use_img:
            if random.random() < img_mask_ratio:
                head_cam_copy = torch_data['obs']['head_cam'].clone()
                torch_data['obs']['head_cam'] = torch.zeros_like(torch_data['obs']['head_cam'])
                # add mask
                torch_data['obs']['head_cam_mask'] = torch.tensor([False], dtype=torch.bool)  # False means masked
            else:
                torch_data['obs']['head_cam_mask'] = torch.tensor([True], dtype=torch.bool)   # True means not masked
        if self.use_point_cloud:
            if random.random() < point_mask_ratio:
                point_cloud_copy = torch_data['obs']['point_cloud'].clone()
                torch_data['obs']['point_cloud'] = torch.zeros_like(torch_data['obs']['point_cloud'])
                # set mask to 0
                torch_data['obs']['point_cloud_mask'] = torch.tensor([False], dtype=torch.bool)  # False means masked
            else:
                torch_data['obs']['point_cloud_mask'] = torch.tensor([True], dtype=torch.bool)   # True means not masked

        # if use both image and point cloud, and mask both, select only one modality to mask
        if self.use_img and self.use_point_cloud and self.no_overlap:
            if (not torch_data['obs']['head_cam_mask']) and (not torch_data['obs']['point_cloud_mask']):
                if random.random() < 0.5:
                    # keep image unmasked
                    torch_data['obs']['head_cam_mask'] = torch.tensor([True], dtype=torch.bool)
                    torch_data['obs']['head_cam'] = head_cam_copy
                else:
                    # keep point cloud unmasked
                    torch_data['obs']['point_cloud_mask'] = torch.tensor([True], dtype=torch.bool)
                    torch_data['obs']['point_cloud'] = point_cloud_copy

        return torch_data

    
if __name__ == '__main__':
    # Test dataset
    zarr_path = '/home/geyan/projects/RoboTwin/policy/ManiFlow/data/lift_pot-demo_randomized-50.zarr'
    dataset = RoboTwinDataset(zarr_path=zarr_path, 
                            horizon=1, 
                            pad_before=0, 
                            pad_after=0,
                            use_img=True,
                            use_depth=True,
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
    import pdb; pdb.set_trace()
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