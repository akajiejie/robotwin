
DEBUG=False
save_ckpt=True

alg_name=${1}
# task choices: See TASK.md
task_name=${2}
setting=${3}
expert_data_num=${4}
config_name=${alg_name}
addition_info=${5}
seed=${6}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${7} # Example: "2_4_5"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# Convert to CUDA_VISIBLE_DEVICES format
export CUDA_VISIBLE_DEVICES=$(echo $gpu_id | tr '_' ',')

# Get number of GPUs
IFS='_' read -ra gpu_id_arr <<< "$gpu_id"
num_gpus=${#gpu_id_arr[@]}


export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1 


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd ManiFlow


python train_robotwin_batch_mp.py --config-name=${config_name}.yaml \
                            +training.num_gpus=${num_gpus} \
                            +training.distributed=True \
                            task=demo_task_batch \
                            task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            expert_data_num=${expert_data_num} \
                            setting=${setting}