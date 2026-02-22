#!/bin/bash

# Set the policy name
policy_name=Your_Policy

# Read command-line arguments
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}
# You can add custom overrides here, for example:
# policy_path_override="your-hf-username/your-finetuned-smolvla"


# Set Hugging Face cache directory to Your_Policy directory
export HF_HOME="$(pwd)/weights"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mHF_HOME: ${HF_HOME}\033[0m"
echo -e "\033[33mUsing policy: ${policy_name} on GPU ID: ${gpu_id}\033[0m"

# Navigate to the project root directory
cd ../.. 

# Run the evaluation script
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}
    # To override the policy path from the command line, uncomment the following line
    # --policy_path ${policy_path_override} \