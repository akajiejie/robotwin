# bash scripts/gen_demonstration_dexart.sh laptop
# bash scripts/gen_demonstration_dexart.sh faucet
# bash scripts/gen_demonstration_dexart.sh bucket
# bash scripts/gen_demonstration_dexart.sh toilet

export VK_ICD_FILENAMES=/data/geyan21/projects/3D_Generative_Policy/scripts/nvidia_icd.json

cd third_party/dexart-release

task_name=${1}
num_episodes=100
root_dir=../../3D-Diffusion-Policy/data/
# root_dir=../../3D-Diffusion-Policy/data/debug
# root_dir=../../3D-Diffusion-Policy/data/dexart_data__4096pts_1019/

CUDA_VISIBLE_DEVICES=1 python examples/gen_demonstration_expert.py --task_name=${task_name} \
            --checkpoint_path assets/rl_checkpoints/${task_name}/${task_name}_nopretrain_0.zip \
            --num_episodes $num_episodes \
            --root_dir $root_dir \
            --img_size 84 \
            --num_points 1024
