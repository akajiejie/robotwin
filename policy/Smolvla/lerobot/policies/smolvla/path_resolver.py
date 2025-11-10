#!/usr/bin/env python3
"""
路径解析模块 - 支持多种环境配置
"""

import os

def resolve_model_path(model_path):
    """
    解析模型路径，支持多种环境配置
    
    Args:
        model_path: 相对路径或绝对路径
        
    Returns:
        str: 解析后的绝对路径
    """
    
    # 如果是绝对路径或网络路径，直接返回
    if os.path.isabs(model_path) or model_path.startswith('http'):
        return model_path
    
    # 策略1: 使用环境变量指定基础目录
    base_dir = os.environ.get('ROBOTWIN_BASE_DIR')
    if base_dir and os.path.exists(base_dir):
        resolved_path = os.path.join(base_dir, model_path)
        print(f"Using ROBOTWIN_BASE_DIR: {resolved_path}")
        return resolved_path
    
    # 策略2: 使用环境变量指定权重目录
    weights_dir = os.environ.get('ROBOTWIN_WEIGHTS_DIR')
    if weights_dir and os.path.exists(weights_dir):
        # 如果 model_path 已经是 weights/... 格式，直接拼接
        if model_path.startswith('weights/'):
            resolved_path = os.path.join(weights_dir, model_path[8:])  # 去掉 'weights/' 前缀
        else:
            resolved_path = os.path.join(weights_dir, model_path)
        print(f"Using ROBOTWIN_WEIGHTS_DIR: {resolved_path}")
        return resolved_path
    
    # 策略3: 自动检测目录结构
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 查找可能的策略目录
    possible_dirs = [
        ("Your_Policy", "Your_Policy"),
        ("custom_policy", "custom_policy"), 
        ("policy", "policy"),
        ("lerobot", "lerobot"),
    ]
    
    for dir_name, target_name in possible_dirs:
        search_dir = current_file_dir
        while search_dir != "/":
            if os.path.basename(search_dir) == dir_name:
                # 找到目标目录，检查是否有 weights 子目录
                weights_path = os.path.join(search_dir, "weights")
                if os.path.exists(weights_path):
                    resolved_path = os.path.join(search_dir, model_path)
                    print(f"Found {target_name} directory with weights: {resolved_path}")
                    return resolved_path
                break
            search_dir = os.path.dirname(search_dir)
    
    # 策略4: 检查当前工作目录
    current_dir = os.getcwd()
    weights_in_current = os.path.join(current_dir, "weights")
    if os.path.exists(weights_in_current):
        resolved_path = os.path.join(current_dir, model_path)
        print(f"Found weights in current directory: {resolved_path}")
        return resolved_path
    
    # 策略5: 检查 policy 子目录
    policy_dir = os.path.join(current_dir, "policy")
    if os.path.exists(policy_dir):
        for subdir in os.listdir(policy_dir):
            subdir_path = os.path.join(policy_dir, subdir)
            if os.path.isdir(subdir_path):
                weights_in_subdir = os.path.join(subdir_path, "weights")
                if os.path.exists(weights_in_subdir):
                    resolved_path = os.path.join(subdir_path, model_path)
                    print(f"Found weights in policy subdirectory: {resolved_path}")
                    return resolved_path
    
    # 最终回退: 使用当前目录
    resolved_path = os.path.join(current_dir, model_path)
    print(f"Fallback to current directory: {resolved_path}")
    return resolved_path

def setup_environment():
    """
    设置环境变量，用于路径解析
    """
    # 自动检测并设置环境变量
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 查找 Your_Policy 或 custom_policy 目录
    search_dir = current_file_dir
    while search_dir != "/":
        if os.path.basename(search_dir) in ["Your_Policy", "custom_policy"]:
            weights_path = os.path.join(search_dir, "weights")
            if os.path.exists(weights_path):
                os.environ['ROBOTWIN_BASE_DIR'] = search_dir
                os.environ['ROBOTWIN_WEIGHTS_DIR'] = weights_path
                print(f"Auto-detected base directory: {search_dir}")
                print(f"Auto-detected weights directory: {weights_path}")
                return
        search_dir = os.path.dirname(search_dir)
    
    print("Warning: Could not auto-detect base directory") 