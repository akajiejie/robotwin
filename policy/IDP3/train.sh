#!/bin/bash
set -x  # 打印调试信息

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

# 获取当前脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[DEBUG] 当前脚本路径：$0"
echo "[DEBUG] SCRIPT_DIR=$SCRIPT_DIR"
echo "[DEBUG] 调用数据处理脚本：${SCRIPT_DIR}/process_data.sh"
echo "[DEBUG] 调用训练脚本：${SCRIPT_DIR}/scripts/train_policy.sh"

# 调用数据处理（如果 zarr 数据未存在）
if [ ! -d "${SCRIPT_DIR}/data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash "${SCRIPT_DIR}/process_data.sh" ${task_name} ${task_config} ${expert_data_num}
fi

# 启动训练
bash "${SCRIPT_DIR}/scripts/train_policy.sh" robot_idp3 ${task_name} ${task_config} ${expert_data_num} train ${seed} ${gpu_id}
