DEBUG=False
save_ckpt=True
train=True

alg_name=${1}
task_name=${2}
expert_data_num=${3}
config_name=${alg_name}
addition_info=${4}
seed=${5}
gpu_id=${6}

exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

# 真机数据没有setting(task_config)概念
# 但Hydra配置文件中task.name需要构建为: ${task_name}-${setting}-${expert_data_num}
# 为了匹配真机数据文件名 test_feed_dual-10.zarr，我们需要让setting为空
# 这样 ${task_name}--${expert_data_num} 会生成错误的路径
# 解决方案：直接覆盖task.name为真机数据的zarr文件名(不含.zarr后缀)
setting=""
task_data_name="${task_name}-${expert_data_num}"

# Environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VK_ICD_FILENAMES="${SCRIPT_DIR}/nvidia_icd.json"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

# 🔥 启用HuggingFace离线模式 (使用本地缓存的预训练权重)
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1


# Set wandb mode based on debug flag
if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33m=== DEBUG MODE ===\033[0m"
else
    wandb_mode=online
    echo -e "\033[33m=== TRAINING MODE (Real Robot Data) ===\033[0m"
fi

# Change to workspace directory
cd ManiFlow/maniflow/workspace

# Training phase
echo -e "\033[32m=== Starting Training with Real Robot Data ===\033[0m"
echo -e "\033[36mTask: ${task_name}\033[0m"
echo -e "\033[36mData: ../data/${task_name}-${expert_data_num}.zarr\033[0m"
echo -e "\033[36mOutput: ${run_dir}\033[0m"

python train_maniflow_robotwin2_workspace.py \
    --config-name=${config_name}.yaml \
    task_name=${task_name} \
    task.name=${task_data_name} \
    hydra.run.dir=${run_dir} \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    checkpoint.save_ckpt=${save_ckpt} \
    expert_data_num=${expert_data_num} \
    setting=${setting}
    

