#!/bin/bash
# bash train_real.sh ${task_name} ${alg_name} ${expert_data_num} ${addition_info} ${seed} ${gpu_id}
# Example:
# bash train_real.sh test_feed_dual maniflow_image_timm_siglip_policy_robotwin2 10 real_robot 0 0

train=true

policy_name=ManiFlow
task_name=${1}          # 任务名称，例如 test_feed_dual
alg_name=${2}           # 算法名称，例如 maniflow_image_timm_siglip_policy_robotwin2
expert_data_num=${3}    # 专家数据数量，例如 10
addition_info=${4}      # 附加信息，例如 real_robot
seed=${5}               # 随机种子，例如 0
gpu_id=${6}             # GPU ID，例如 0
ckpt_setting=${task_name}  # 使用任务名称作为检查点设置（真机数据没有task_config）
eval_seed=0             # 评估种子，可以改为 1, 2 等

# 真机数据路径：policy/ManiFlow/data/test_feed_dual-10.zarr
# 数据已存在，不需要处理

if [ "$train" = true ]; then
    echo "Training is enabled."
    echo "Training with real robot data: ./data/${task_name}-${expert_data_num}.zarr"
    
    # 检查数据是否存在
    if [ ! -d "./data/${task_name}-${expert_data_num}.zarr" ]; then
        echo "Error: Real robot data not found at ./data/${task_name}-${expert_data_num}.zarr"
        exit 1
    fi
    
    # 注意：真机数据训练时，setting参数传递为空或特殊标识
    # 因为真机数据没有demo_randomized/demo_clean这样的task_config概念
    bash scripts/train_policy_real.sh ${alg_name} ${task_name} ${expert_data_num} ${addition_info} ${seed} ${gpu_id}
else
    echo "Training is disabled."
fi

echo "Training completed. Model saved to: data/outputs/${task_name}-${alg_name}-${addition_info}_seed${seed}"
