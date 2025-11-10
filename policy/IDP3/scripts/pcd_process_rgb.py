import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
import time

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

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        return None

    try:
        with h5py.File(dataset_path, "r") as root:
            left_gripper = root["/joint_action/left_gripper"][()]
            left_arm = root["/joint_action/left_arm"][()]
            right_gripper = root["/joint_action/right_gripper"][()]
            right_arm = root["/joint_action/right_arm"][()]
            vector = root["/joint_action/vector"][()]
            
            # 获取时间步数
            T = left_gripper.shape[0]
            
            # 初始化点云列表
            pointcloud_list = []
            
            # 处理每一帧
            for j in range(T):
                # 读取头部相机数据
                depth = root['observation/head_camera/depth'][j]
                rgb_data = root['observation/head_camera/rgb'][j]
                intrinsic = root['observation/head_camera/intrinsic_cv'][j]
                extrinsic = root['observation/head_camera/cam2world_gl'][j]
                
                # 确保深度图是有效的NumPy数组
                if not isinstance(depth, np.ndarray):
                    try:
                        depth = np.array(depth)
                    except:
                        print(f"Invalid depth data in {dataset_path} frame {j}, skipping frame")
                        continue
                
                # 确保RGB数据是有效的NumPy数组
                if not isinstance(rgb_data, np.ndarray):
                    try:
                        rgb = np.array(rgb_data)
                        print(f"Converted non-array RGB data to array in {dataset_path} frame {j}")
                    except:
                        print(f"Failed to convert RGB data in {dataset_path} frame {j}, using placeholder")
                        # 创建占位符RGB图像
                        H, W = depth.shape[:2]
                        rgb = np.zeros((H, W, 3), dtype=np.uint8)
                else:
                    rgb = rgb_data
                
                # 确保RGB图像是正确格式 (H, W, 3)
                if rgb.ndim == 3 and rgb.shape[2] == 3:
                    pass  # 已经是正确格式
                elif rgb.ndim == 3 and rgb.shape[2] == 4:
                    rgb = rgb[:, :, :3]  # 去掉alpha通道
                elif rgb.ndim == 2:
                    # 灰度图转RGB
                    try:
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
                    except Exception as e:
                        print(f"Error converting grayscale to BGR: {str(e)}")
                        # 手动转换
                        rgb = np.stack([rgb]*3, axis=-1)
                else:
                    print(f"Unsupported RGB shape: {rgb.shape} in {dataset_path}")
                    # 创建占位符RGB图像
                    H, W = depth.shape[:2]
                    rgb = np.zeros((H, W, 3), dtype=np.uint8)
                
                # 生成带RGB的点云
                pc = depth_to_point_cloud(depth, intrinsic, extrinsic, rgb=rgb)
                pointcloud_list.append(pc)
            
            pointcloud_all = np.stack(pointcloud_list, axis=0)
        
        return left_gripper, left_arm, right_gripper, right_arm, vector, pointcloud_all
    
    except Exception as e:
        print(f"Error processing {dataset_path}: {str(e)}")
        return None

def process_episode(ep_index, load_dir, num):
    """处理单个episode的线程任务函数"""
    load_path = os.path.join(load_dir, f"data/episode{ep_index}.hdf5")
    
    # 打印进度
    print(f"Processing episode {ep_index+1}/{num}", end="\r")
    
    result = load_hdf5(load_path)
    if result is None:
        print(f"\nSkipping episode {ep_index} due to processing error")
        return None
    
    (
        left_gripper_all,
        left_arm_all,
        right_gripper_all,
        right_arm_all,
        vector_all,
        pointcloud_all,
    ) = result
    
    # 为该episode准备数据
    episode_point_clouds = []
    episode_states = []
    episode_actions = []
    
    for j in range(0, left_gripper_all.shape[0]):
        pointcloud = pointcloud_all[j]
        joint_state = vector_all[j]

        if j != left_gripper_all.shape[0] - 1:
            episode_point_clouds.append(pointcloud)
            episode_states.append(joint_state)
        if j != 0:
            episode_actions.append(joint_state)
    
    # 返回该episode的处理结果
    return {
        "ep_index": ep_index,
        "point_clouds": np.array(episode_point_clouds),
        "states": np.array(episode_states),
        "actions": np.array(episode_actions),
        "frame_count": left_gripper_all.shape[0] - 1
    }

def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = "../../data/" + str(task_name) + "/" + str(task_config)
    print(f"Loading data from {load_dir}")

    save_dir = f"./data/{task_name}-{task_config}-{num}_rgb.zarr"
    print(f"Saving data to {save_dir}")

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # 创建线程安全的数据收集结构
    point_cloud_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    total_frames = 0
    lock = threading.Lock()

    # 创建线程池
    start_time = time.time()
    print(f"Starting processing with 4 threads...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务
        futures = {executor.submit(process_episode, i, load_dir, num): i for i in range(num)}
        
        # 处理完成的任务
        completed_count = 0
        skipped_episodes = []
        for future in as_completed(futures):
            ep_index = futures[future]
            try:
                result = future.result()
                if result is None:
                    skipped_episodes.append(ep_index)
                    continue
                
                # 使用锁保证数据一致性
                with lock:
                    # 添加数据
                    point_cloud_arrays.append(result["point_clouds"])
                    state_arrays.append(result["states"])
                    action_arrays.append(result["actions"])
                    
                    # 更新帧计数
                    total_frames += result["frame_count"]
                    episode_ends_arrays.append(total_frames)
                    
                    completed_count += 1
                    print(f"Completed {completed_count}/{num} episodes | "
                          f"Frames: {total_frames} | "
                          f"Elapsed: {time.time()-start_time:.1f}s", end="\r")
            
            except Exception as e:
                print(f"Error processing episode {ep_index}: {str(e)}")
                skipped_episodes.append(ep_index)
    
    # 处理完成后打印统计信息
    processing_time = time.time() - start_time
    print(f"\nProcessing completed in {processing_time:.1f} seconds")
    print(f"Processed {total_frames} frames across {completed_count} episodes")
    
    if skipped_episodes:
        print(f"Skipped {len(skipped_episodes)} episodes: {skipped_episodes}")
    
    # 如果没有成功处理任何数据，退出
    if completed_count == 0:
        print("No episodes processed successfully. Exiting.")
        return
    
    # 合并所有episode的数据
    try:
        point_cloud_arrays = np.concatenate(point_cloud_arrays, axis=0)
        state_arrays = np.concatenate(state_arrays, axis=0)
        action_arrays = np.concatenate(action_arrays, axis=0)
        episode_ends_arrays = np.array(episode_ends_arrays)
        
        # 打印数组形状以验证
        print(f"Point cloud shape: {point_cloud_arrays.shape} - should be (frames, 4096, 6)")
        print(f"State shape: {state_arrays.shape}")
        print(f"Action shape: {action_arrays.shape}")
        print(f"Episode ends: {episode_ends_arrays.shape}")
        
        # 创建Zarr存储
        zarr_root = zarr.group(save_dir)
        zarr_data = zarr_root.create_group("data")
        zarr_meta = zarr_root.create_group("meta")
        
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        state_chunk_size = (100, state_arrays.shape[1])
        joint_chunk_size = (100, action_arrays.shape[1])
        point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
        
        # 创建Zarr数据集
        zarr_data.create_dataset(
            "point_cloud",
            data=point_cloud_arrays,
            chunks=point_cloud_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "state",
            data=state_arrays,
            chunks=state_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "action",
            data=action_arrays,
            chunks=joint_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_meta.create_dataset(
            "episode_ends",
            data=episode_ends_arrays,
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )
        
        print(f"Successfully saved RGB point cloud data to {save_dir}")
        print(f"Total processing time: {time.time()-start_time:.1f} seconds")
        
    except Exception as e:
        print(f"Error during data consolidation: {str(e)}")
        raise

if __name__ == "__main__":
    main()