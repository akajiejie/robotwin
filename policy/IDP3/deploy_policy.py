# import packages and module here
import sys

import torch
import sapien.core as sapien
import traceback
import os
import numpy as np
from envs import *
from hydra import initialize, compose
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra import main as hydra_main
import pathlib
from omegaconf import OmegaConf

import yaml
from datetime import datetime
import importlib
import cv2
import open3d as o3d

from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_directory, '3D-Diffusion-Policy'))

from idp3_policy import *
from idp3_policy import IDP3  


def depth_to_point_cloud(depth, intrinsic, extrinsic, rgb=None, downsample_to=4096, voxel_size=0.01):
    """
    将深度图和RGB图像转换为带颜色的点云并进行降采样
    :param depth: 深度图 (H, W), uint16 类型，单位为毫米
    :param intrinsic: 相机内参 (3, 3)
    :param extrinsic: 相机外参 (4, 4) cam2world_gl
    :param rgb: RGB图像 (H, W, 3), uint8类型
    :param downsample_to: 目标点云点数
    :param voxel_size: 体素降采样大小
    :return: 降采样后的点云 (downsample_to, 6) - [x, y, z, r, g, b]
    """
    # 转换深度图为米制浮点数
    depth = depth.astype(np.float32) / 1000.0
    
    H, W = depth.shape
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    
    # 转换为相机坐标系
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 移除无效点 (深度为0)
    valid_mask = (z > 0)
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    
    # 如果没有有效点，返回全零点云
    if len(x) == 0:
        return np.zeros((downsample_to, 6), dtype=np.float32)
    
    # 创建点云位置
    points_cam = np.stack([x, y, z], axis=1)
    
    # 添加颜色信息（如果提供了RGB图像）
    colors = None
    if rgb is not None:
        # 确保RGB图像与深度图大小匹配
        if rgb.shape[:2] != (H, W):
            try:
                rgb = cv2.resize(rgb, (W, H))
            except:
                # 创建占位符RGB图像
                rgb = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 提取有效点的颜色并归一化到[0, 1]
        colors = rgb.reshape(-1, 3)[valid_mask.flatten()].astype(np.float32) / 255.0
    
    # 转换为齐次坐标
    ones = np.ones((points_cam.shape[0], 1))
    points_cam_hom = np.hstack([points_cam, ones])
    
    # 应用外参变换到世界坐标系
    points_world_hom = (extrinsic @ points_cam_hom.T).T
    points_world = points_world_hom[:, :3] / points_world_hom[:, 3][:, None]
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    
    # 添加颜色信息
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 体素降采样
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 获取降采样后的点云
    down_points = np.asarray(down_pcd.points)
    down_colors = np.asarray(down_pcd.colors) if down_pcd.has_colors() else None
    
    # 如果点云为空，返回全零点云
    if down_points.shape[0] == 0:
        return np.zeros((downsample_to, 6), dtype=np.float32)
    
    # 组合位置和颜色
    if down_colors is not None:
        down_pc_with_color = np.hstack([down_points, down_colors])
    else:
        # 如果没有颜色，使用白色
        down_colors = np.ones((down_points.shape[0], 3), dtype=np.float32)
        down_pc_with_color = np.hstack([down_points, down_colors])
    
    # 随机采样到目标点数
    n_points = down_pc_with_color.shape[0]
    if n_points >= downsample_to:
        indices = np.random.choice(n_points, downsample_to, replace=False)
        sampled_pc = down_pc_with_color[indices]
    else:
        # 如果点数不足，先使用所有点，然后随机重复点补足
        indices = np.random.choice(n_points, downsample_to - n_points, replace=True)
        sampled_pc = np.vstack([down_pc_with_color, down_pc_with_color[indices]])
    
    return sampled_pc.astype(np.float32)


def encode_obs(observation, use_rgb=False):  # Post-Process Observation
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    
    # 检查是否有相机观测数据
    if 'head_camera' in observation:
        camera_obs = observation['head_camera']
        
        # 检查是否有深度图
        if 'depth' in camera_obs:
            depth = camera_obs['depth']
            rgb = camera_obs.get('rgb', None)
            intrinsic = camera_obs.get('intrinsic_cv', None)
            extrinsic = camera_obs.get('cam2world_gl', None)
            
            # 确保数据是有效的NumPy数组
            if isinstance(depth, np.ndarray):
                # 处理RGB
                rgb_input = None
                if use_rgb and rgb is not None and isinstance(rgb, np.ndarray):
                    if rgb.ndim == 3 and rgb.shape[2] == 4:
                        rgb_input = rgb[:, :, :3]
                    elif rgb.ndim == 2:
                        try:
                            rgb_input = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
                        except:
                            rgb_input = np.stack([rgb]*3, axis=-1)
                    else:
                        rgb_input = rgb
                # 生成点云
                if intrinsic is not None and extrinsic is not None:
                    point_cloud = depth_to_point_cloud(depth, intrinsic, extrinsic, rgb=rgb_input)
                    obs['point_cloud'] = point_cloud
                else:
                    obs['point_cloud'] = observation.get('pointcloud', np.zeros((4096, 6), dtype=np.float32))
            else:
                obs['point_cloud'] = observation.get('pointcloud', np.zeros((4096, 6), dtype=np.float32))
        else:
            obs['point_cloud'] = observation.get('pointcloud', np.zeros((4096, 6), dtype=np.float32))
    else:
        obs['point_cloud'] = observation.get('pointcloud', np.zeros((4096, 6), dtype=np.float32))
    
    return obs


def get_model(usr_args):
    config_path = "./3D-Diffusion-Policy/diffusion_policy_i3d/config"
    config_name = f"{usr_args['config_name']}.yaml"

    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)

    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"

    hydra_runtime_cfg = {
        "job": {
            "override_dirname": usr_args['task_name']
        },
        "run": {
            "dir": run_dir
        },
        "sweep": {
            "dir": run_dir,
            "subdir": "0"
        }
    }

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    cfg.policy.use_pc_color = usr_args['use_rgb']
    OmegaConf.set_struct(cfg, True)

    DP3_Model = IDP3(cfg, usr_args)
    return DP3_Model


def eval(TASK_ENV, model, observation, use_rgb=True):
    obs = encode_obs(observation, use_rgb=use_rgb)  # Post-Process Observation
    # instruction = TASK_ENV.get_instruction()

    if len(
            model.env_runner.obs
    ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation, use_rgb=use_rgb)
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.env_runner.reset_obs()
