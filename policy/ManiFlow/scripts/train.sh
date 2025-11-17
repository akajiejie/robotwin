
# bash scripts/train.sh dp3 adroit_door test_dp3 0 4
# bash scripts/train.sh maniflow adroit_door test_maniflow 0 5
# bash scripts/train.sh maniflow adroit_door test_maniflow_qkv_true_R200 0 5
# bash scripts/train.sh maniflow adroit_door test_maniflow_qkv_true_R50 0 5
# bash scripts/train.sh maniflow adroit_door test_maniflow_qkv_false_R50 0 5

export VK_ICD_FILENAMES=/data/geyan21/projects/3D_Generative_Policy/scripts/nvidia_icd.json

DEBUG=False
save_ckpt=False

dataset_path=/data/geyan21/projects/3D_Generative_Policy/3D-Diffusion-Policy/data/adroit_door_expert.zarr

# dataset_path=data/adroit_hammer_expert.zarr
# dataset_path=data/adroit_door_expert.zarr
# dataset_path=data/adroit_pen_expert.zarr

# dataset_path=data/metaworld_basketball_expert.zarr
# dataset_path=data/metaworld_assembly_expert.zarr
# dataset_path=data/metaworld_disassemble_expert.zarr
# dataset_path=data/metaworld_soccer_expert.zarr
# dataset_path=data/metaworld_hand-insert_expert.zarr

# dataset_path=data/dexart_laptop_expert.zarr
# dataset_path=data/dexart_faucet_expert.zarr
# dataset_path=data/dexart_bucket_expert.zarr
# dataset_path=data/dexart_toilet_expert.zarr


alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


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

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
env_device="cuda:${gpu_id}"
python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            task.dataset.zarr_path=${dataset_path} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            training.env_device=${env_device} \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}



                                