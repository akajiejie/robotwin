#!/bin/bash
# bash train_eval.sh lift_pot robot_maniflow demo_randomized 50 test 0 0

train=true # set to false if you want to skip training
eval=false # set to true if you want to run evaluation
policy_name=ManiFlow
task_name=${1}
alg_name=${2} # alg_name is the algorithm name, e.g., robot_maniflow
task_config=${3}
expert_data_num=${4}
addition_info=${5}
seed=${6}
gpu_id=${7}
eval_seed=0 # seed for evaluation, can be changed to 1, 2, etc.


if [ "$train" = true ]; then
    echo "Training is enabled."
    if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
        bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
    fi
    bash scripts/train_policy.sh ${alg_name} ${task_name} ${task_config} ${expert_data_num} ${addition_info} ${seed} ${gpu_id}
else
    echo "Training is disabled."
fi



# if eval is false, skip evaluation
if [ "$eval" = false ]; then
    echo "Evaluation is disabled."
    exit 0
else
    echo "Evaluation is enabled."
    echo "Evaluating policy with task: ${task_name}, config: ${task_config}, expert data num: ${expert_data_num}, seed: ${seed}, gpu id: ${gpu_id}"
fi

# Evaluate the trained policy
export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

ckpt_setting=${task_config} # setting for evaluation, can be changed to demo_clean

# task_config=demo_clean # setting for evaluation, can be changed to demo_clean

# PYTHONWARNINGS=ignore::UserWarning \
# python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
#     --overrides \
#     --config_name ${alg_name} \
#     --task_name ${task_name} \
#     --task_config ${task_config} \
#     --ckpt_setting ${ckpt_setting} \
#     --expert_data_num ${expert_data_num} \
#     --seed ${seed} \
#     --policy_name ${policy_name} 


task_config=demo_randomized # setting for evaluation, can be changed to demo_clean

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --config_name ${alg_name} \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${eval_seed} \
    --training_seed ${seed} \
    --policy_name ${policy_name} \
    --addition_info ${addition_info} \
    --alg_name ${alg_name} \
