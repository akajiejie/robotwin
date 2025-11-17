#!/bin/bash
# bash train_eval.sh pick_diverse_bottles demo_clean 50 0 0
# bash train_eval.sh lift_pot demo_clean 50 0 0
# bash train_eval.sh move_can_pot demo_clean 50 0 1
# bash train_eval.sh put_object_cabinet demo_clean 50 0 2

# bash train_eval.sh lift_pot robot_maniflow demo_clean 50 train 0 0 # eval all
# bash train_eval.sh move_can_pot robot_maniflow demo_clean 50 train 0 1 # eval all
# bash train_eval.sh put_object_cabinet robot_maniflow demo_clean 50 train 0 1
# bash train_eval.sh lift_pot robot_maniflow demo_randomized 50 RGB_Aug_128pts 0 0 # eval randomized
# bash train_eval.sh move_can_pot robot_maniflow demo_randomized 50 RGB_Aug_128pts 0 1 # eval randomized
# bash train_eval.sh put_object_cabinet robot_maniflow demo_randomized 50 RGB_Aug_128pts 0 1

# bash train_eval.sh lift_pot robot_maniflow demo_randomized 50 RGB_Aug_512pts 0 0
# bash train_eval.sh lift_pot robot_maniflow demo_randomized 50 RGB_Aug_1024pts_inf1_bs128 0 0
# bash train_eval.sh move_can_pot robot_maniflow demo_randomized 50 RGB_Aug_512pts_inf1_bs256 0 0
# bash train_eval.sh move_can_pot robot_maniflow demo_randomized 50 RGB_Aug_1024pts_inf1_bs128 0 0
# bash train_eval.sh put_object_cabinet robot_maniflow demo_randomized 50 RGB_Aug_1024pts_bs128 0 1
# bash train_eval.sh pick_diverse_bottles robot_maniflow demo_randomized 50 0719_RGB_Aug_512pts_bs256 0 0
# bash train_eval.sh pick_diverse_bottles robot_maniflow demo_randomized 50 0717_RGB_Aug_1024pts_bs128 0 0

# bash train_eval.sh lift_pot robot_meanflow demo_randomized 50 0721_RGB_Aug_512pts_bs128 0 0 # trained, not working
# bash train_eval.sh move_can_pot robot_meanflow demo_randomized 50 0721_RGB_Aug_512pts_bs128 0 1 # trained
# bash train_eval.sh put_object_cabinet robot_meanflow demo_randomized 50 0721_RGB_Aug_512pts_bs128 0 0
# bash train_eval.sh pick_diverse_bottles robot_meanflow demo_randomized 50 0721_RGB_Aug_512pts_bs128 0 0 # trained
# bash train_eval.sh lift_pot robot_meanflow demo_clean 50 0721_RGB_Aug_128pts_bs256 0 0

# bash train_eval.sh open_laptop robot_maniflow demo_randomized 50 0726_RGB_Aug_512pts_bs256 0 0
# bash train_eval.sh handover_mic robot_maniflow demo_randomized 50 0726_RGB_Aug_512pts_bs256 0 0 # trained
# bash train_eval.sh open_laptop robot_maniflow demo_randomized 50 0726_RGB_Aug_128pts_bs384 0 0 # trained
# bash train_eval.sh handover_mic robot_maniflow demo_randomized 50 0726_RGB_Aug_128pts_bs384 0 0 # trained, not working

# bash train_eval.sh lift_pot robot_maniflow demo_randomized 50 0728_RGB_Aug_512pts 1 0
# bash train_eval.sh put_object_cabinet robot_maniflow demo_randomized 50 0728_RGB_Aug_512pts 1 0

# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0805_image_clip_pointcloud_color_MMDiT-G_h1_512pts_bs32_gpus1_demo50 0 0 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0805_image_clip_pcd_color_MMDiT-G_R1_l8e512_h1_512pts_bs32_gpus1_demo50 0 0 # trained 0, bs64 in fact
# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0805_image_clip_pcd_color_DiTX_h1_512pts_bs64_gpus1_demo50 0 0 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0805_image_clip_pcd_color_Norm_DiTX_h1_512pts_bs64_gpus1_demo50 0 0 # trained 0

# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0805_image_r3m_pre_pcd_color_MMDiT-G_R1_l8e512_h1_512pts_bs64_gpus1_demo50 0 0 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0805_image_r3m_pre_pcd_color_DiTX_l8e512_h1_512pts_bs128_gpus1_demo50 0 1 # trained 0

# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0812_MMDiT-G_No_Alter_Mask_Image_0.2_PCD_256pts_h1_bs64_gpus1_demo50 0 6 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0812_MMDiT-G_No_Alter_Mask_Image_PCD_0.2_256pts_h1_bs64_gpus1_demo50 0 7 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0812_MMDiT-G_No_Alter_Mask_Image0.2_PCD_0.2_256pts_h1_bs64_gpus1_demo50 0 5 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0812_MMDiT-G_Alter_Mask_Image0.2_PCD_0.2_256pts_h1_bs64_gpus1_demo50 0 4 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0812_MMDiT-G_Alter_Mask_Image_0.2_PCD_256pts_h1_bs64_gpus1_demo50 0 4 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0812_MMDiT-G_Alter_Mask_Image_PCD_0.8_256pts_h1_bs64_gpus1_demo50 0 4 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0812_MMDiT-G_No_Alter_Mask_Image_PCD_0.8_256pts_h1_bs64_gpus1_demo50 0 4 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0812_MMDiT-G_Alter_Mask_Image_0.1_PCD_256pts_h1_bs64_gpus1_demo50 0 2 # trained 0
# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 0812_MMDiT-G_Alter_No_Mask_Image_PCD_256pts_h1_bs64_gpus1_demo50 0 4 # trained 0

# bash train_eval.sh lift_pot robot_maniflow demo_randomized 50 debug 0 0
# bash train_eval.sh lift_pot robot_unified_maniflow demo_randomized 50 debug_img 0 0
# bash train_eval.sh lift_pot robot_unified_maniflow_pointmap demo_randomized 50 debug_pointmap 0 0

train=false
eval=true
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
#     --seed ${eval_seed} \
#     --training_seed ${seed} \
#     --policy_name ${policy_name} \
#     --addition_info ${addition_info} \
#     --alg_name ${alg_name} \


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
