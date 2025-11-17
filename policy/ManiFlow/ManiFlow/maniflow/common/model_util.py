from termcolor import cprint
import pdb
import sys
import os
import numpy as np
import cv2

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def print_params(model):
    """
    Print the number of parameters in each part of the model.
    """    
    params_dict = {}

    all_num_param = sum(p.numel() for p in model.parameters())

    for name, param in model.named_parameters():
        part_name = name.split('.')[0]
        if part_name not in params_dict:
            params_dict[part_name] = 0
        params_dict[part_name] += param.numel()

    cprint(f'----------------------------------', 'cyan')
    cprint(f'Class name: {model.__class__.__name__}', 'cyan')
    cprint(f'  Number of parameters: {all_num_param / 1e6:.4f}M', 'cyan')
    for part_name, num_params in params_dict.items():
        cprint(f'   {part_name}: {num_params / 1e6:.4f}M ({num_params / all_num_param:.2%})', 'cyan')
    cprint(f'----------------------------------', 'cyan')

def adjust_intrinsics(intrinsic_matrix, original_size, target_size):
    """
    Adjust camera intrinsic parameters when resizing images.
    
    Args:
        intrinsic_matrix: Original 3x3 intrinsic matrix
        original_size: Original image size (height, width)
        target_size: Target image size (height, width)
    
    Returns:
        Adjusted intrinsic matrix
    """
    scale_x = target_size[1] / original_size[1]  # width scaling
    scale_y = target_size[0] / original_size[0]  # height scaling
    
    adjusted_intrinsics = intrinsic_matrix.copy()
    # Scale focal lengths and principal point
    adjusted_intrinsics[0, 0] *= scale_x  # fx
    adjusted_intrinsics[1, 1] *= scale_y  # fy
    adjusted_intrinsics[0, 2] *= scale_x  # cx
    adjusted_intrinsics[1, 2] *= scale_y  # cy
    
    return adjusted_intrinsics

def resize_depth(depth_array, target_size=(224, 224)):
    """
    Resize depth array to target size using nearest neighbor interpolation
    to preserve depth values accurately.
    
    Args:
        depth_array: Input depth array (H, W)
        target_size: Target size as (height, width)
    
    Returns:
        Resized depth array
    """
    return cv2.resize(depth_array, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

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