#!/usr/bin/env python3
"""
测试脚本：验证生成的zarr文件结构是否正确
"""
import zarr
import sys
import os

def test_zarr_structure(zarr_path):
    """测试zarr文件结构"""
    if not os.path.exists(zarr_path):
        print(f"Error: Zarr file does not exist: {zarr_path}")
        return False
    
    try:
        # 打开zarr文件
        root = zarr.open(zarr_path, mode='r')
        
        print(f"Testing zarr structure: {zarr_path}")
        print("=" * 50)
        
        # 检查顶层结构
        print("Top level groups:")
        for key in root.keys():
            print(f"  - {key}")
        
        # 检查必需的组
        required_groups = ['data', 'meta']
        for group in required_groups:
            if group not in root:
                print(f"ERROR: Missing required group: {group}")
                return False
            print(f"✓ Found group: {group}")
        
        # 检查data组中的数据集
        print("\nData group contents:")
        data_group = root['data']
        expected_data_keys = ['state', 'action', 'point_cloud']
        
        for key in data_group.keys():
            dataset = data_group[key]
            print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        # 检查必需的数据集
        for key in expected_data_keys:
            if key not in data_group:
                print(f"ERROR: Missing required dataset in data: {key}")
                return False
            print(f"✓ Found dataset: data/{key}")
        
        # 检查meta组中的数据集
        print("\nMeta group contents:")
        meta_group = root['meta']
        for key in meta_group.keys():
            dataset = meta_group[key]
            print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        # 检查episode_ends
        if 'episode_ends' not in meta_group:
            print("ERROR: Missing required dataset: meta/episode_ends")
            return False
        print("✓ Found dataset: meta/episode_ends")
        
        # 验证数据一致性
        print("\nData consistency check:")
        episode_ends = meta_group['episode_ends'][:]
        total_frames = episode_ends[-1] if len(episode_ends) > 0 else 0
        
        print(f"  Total episodes: {len(episode_ends)}")
        print(f"  Total frames: {total_frames}")
        
        # 检查所有数据集的第一维度是否一致
        for key in expected_data_keys:
            dataset = data_group[key]
            if dataset.shape[0] != total_frames:
                print(f"ERROR: Dataset {key} has {dataset.shape[0]} frames, expected {total_frames}")
                return False
            print(f"✓ Dataset {key} has correct number of frames: {dataset.shape[0]}")
        
        print("\n" + "=" * 50)
        print("✅ Zarr structure validation PASSED")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to read zarr file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_zarr_structure.py <zarr_path>")
        sys.exit(1)
    
    zarr_path = sys.argv[1]
    success = test_zarr_structure(zarr_path)
    sys.exit(0 if success else 1)