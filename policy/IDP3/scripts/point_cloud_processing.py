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
import threading

def depth_to_point_cloud(depth, intrinsic, extrinsic, rgb=None, downsample_to=4096, voxel_size=0.01):
    """
    将深度图转换为点云并进行降采样
    :param depth: 深度图 (H, W), uint16 类型，单位为毫米
    :param intrinsic: 相机内参 (3, 3)
    :param extrinsic: 相机外参 (4, 4) cam2world_gl
    :param rgb: RGB图像 (H, W, 3), 可选
    :param downsample_to: 目标点云点数
    :param voxel_size: 体素降采样大小
    :return: 降采样后的点云 (downsample_to, 3)
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
        return np.zeros((downsample_to, 3), dtype=np.float32)
    
    # 创建点云
    points_cam = np.stack([x, y, z], axis=1)
    
    # 转换为齐次坐标
    ones = np.ones((points_cam.shape[0], 1))
    points_cam_hom = np.hstack([points_cam, ones])
    
    # 应用外参变换到世界坐标系
    points_world_hom = (extrinsic @ points_cam_hom.T).T
    points_world = points_world_hom[:, :3] / points_world_hom[:, 3][:, None]
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    
    # 体素降采样
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 获取降采样后的点云
    down_points = np.asarray(down_pcd.points)
    
    # 如果点云为空，返回全零点云
    if down_points.shape[0] == 0:
        return np.zeros((downsample_to, 3), dtype=np.float32)
    
    # 随机采样到目标点数
    n_points = down_points.shape[0]
    if n_points >= downsample_to:
        indices = np.random.choice(n_points, downsample_to, replace=False)
        sampled_points = down_points[indices]
    else:
        # 如果点数不足，先使用所有点，然后随机重复点补足
        indices = np.random.choice(n_points, downsample_to - n_points, replace=True)
        sampled_points = np.vstack([down_points, down_points[indices]])
    
    return sampled_points.astype(np.float32)

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
                intrinsic = root['observation/head_camera/intrinsic_cv'][j]
                extrinsic = root['observation/head_camera/cam2world_gl'][j]
                
                # 生成点云 (只使用深度信息，忽略RGB)
                pc = depth_to_point_cloud(depth, intrinsic, extrinsic)
                pointcloud_list.append(pc)
            
            pointcloud_all = np.stack(pointcloud_list, axis=0)
        
        return left_gripper, left_arm, right_gripper, right_arm, vector, pointcloud_all
    
    except Exception as e:
        print(f"Error loading HDF5 file {dataset_path}: {e}")
        return None

def process_episode(episode_idx, load_dir, progress_lock, progress_counter, total_episodes):
    """
    处理单个episode的函数，用于多线程处理
    """
    try:
        load_path = os.path.join(load_dir, f"data/episode{episode_idx}.hdf5")
        result = load_hdf5(load_path)
        if result is None:
            return None
        
        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            vector_all,
            pointcloud_all,
        ) = result

        # 准备该episode的数据
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

        # 更新进度
        with progress_lock:
            progress_counter[0] += 1
            print(f"processing episode: {progress_counter[0]} / {total_episodes}", end="\r")

        return {
            'episode_idx': episode_idx,
            'point_clouds': np.array(episode_point_clouds),
            'states': np.array(episode_states),
            'actions': np.array(episode_actions),
            'episode_length': left_gripper_all.shape[0] - 1
        }
    
    except Exception as e:
        print(f"\nError processing episode {episode_idx}: {e}")
        return None

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
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads to use for parallel processing (default: 4)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config
    num_threads = args.num_threads

    load_dir = "../../data/" + str(task_name) + "/" + str(task_config)
    print(f"Loading data from {load_dir}")
    print(f"Using {num_threads} threads for parallel processing")

    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"
    print(f"Saving data to {save_dir}")

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    # 初始化数据收集列表
    point_cloud_arrays = []
    episode_ends_arrays = []
    state_arrays = []
    joint_action_arrays = []

    # 多线程处理episode
    progress_lock = threading.Lock()
    progress_counter = [0]  # 使用列表以便在函数间传递引用
    
    total_count = 0
    
    # 使用ThreadPoolExecutor进行并行处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有episode处理任务
        future_to_episode = {
            executor.submit(process_episode, ep_idx, load_dir, progress_lock, progress_counter, num): ep_idx 
            for ep_idx in range(num)
        }
        
        # 收集结果，按episode顺序排序
        episode_results = {}
        
        for future in as_completed(future_to_episode):
            result = future.result()
            if result is not None:
                episode_results[result['episode_idx']] = result
    
    print()  # 换行
    
    # 按episode顺序处理结果
    for ep_idx in range(num):
        if ep_idx in episode_results:
            result = episode_results[ep_idx]
            
            # 添加数据到总列表（使用append而不是extend，因为现在返回的是numpy数组）
            point_cloud_arrays.append(result['point_clouds'])
            state_arrays.append(result['states'])
            joint_action_arrays.append(result['actions'])
            
            # 更新episode结束位置
            total_count += result['episode_length']
            episode_ends_arrays.append(total_count)
        else:
            print(f"Warning: Episode {ep_idx} failed to process")

    print()
    
    # 检查是否有数据被成功处理
    if len(point_cloud_arrays) == 0:
        print("ERROR: No episodes were successfully processed!")
        return
    
    try:
        # 使用concatenate合并所有episode的数据
        episode_ends_arrays = np.array(episode_ends_arrays)
        state_arrays = np.concatenate(state_arrays, axis=0)
        point_cloud_arrays = np.concatenate(point_cloud_arrays, axis=0)
        joint_action_arrays = np.concatenate(joint_action_arrays, axis=0)
        
        # 打印数据形状以验证
        print(f"Final data shapes:")
        print(f"  Point cloud: {point_cloud_arrays.shape}")
        print(f"  State: {state_arrays.shape}")
        print(f"  Action: {joint_action_arrays.shape}")
        print(f"  Episode ends: {episode_ends_arrays.shape}")
    
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        state_chunk_size = (100, state_arrays.shape[1])
        joint_chunk_size = (100, joint_action_arrays.shape[1])
        point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
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
            data=joint_action_arrays,
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
    except ZeroDivisionError as e:
        print("If you get a `ZeroDivisionError: division by zero`, check that `data/pointcloud` in the task config is set to true.")
        raise 
    except Exception as e:
        print(f"An unexpected error occurred ({type(e).__name__}): {e}")
        raise

if __name__ == "__main__":
    main()