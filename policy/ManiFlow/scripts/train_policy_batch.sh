
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
gpu_id=${7}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

export TOKENIZERS_PARALLELISM=false


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
python train_robotwin_batch.py --config-name=${config_name}.yaml \
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

# python eval_robotwin.py --config-name=${config_name}.yaml \
#                             +eval_mode=${eval_mode} \
#                             +task_mode=${task_mode} \
#                             robotwin_task=${task_name} \
#                             robotwin_task.env_runner.eval_episodes=${eval_episode} \
#                             hydra.run.dir=${run_dir} \
#                             training.debug=$DEBUG \
#                             training.seed=${seed} \
#                             training.device="cuda:0" \
#                             policy.num_inference_steps=${num_inference_steps} \
#                             policy.n_obs_steps=${n_obs_steps} \
#                             policy.horizon=${horizon} \
#                             policy.n_action_steps=${n_action_steps} \
#                             exp_name=${exp_name} \
#                             logging.mode=${wandb_mode} \
#                             checkpoint.save_ckpt=${save_ckpt}

# num_inference_steps=1
# python eval_robotwin_multitask.py --config-name=${config_name}.yaml \
#                             +eval_mode=${eval_mode} \
#                             +task_mode=${task_mode} \
#                             robotwin_task=${task_name} \
#                             robotwin_task.env_runner.eval_episodes=${eval_episode} \
#                             hydra.run.dir=${run_dir} \
#                             training.debug=$DEBUG \
#                             training.seed=${seed} \
#                             training.device="cuda:0" \
#                             policy.num_inference_steps=${num_inference_steps} \
#                             policy.n_obs_steps=${n_obs_steps} \
#                             policy.horizon=${horizon} \
#                             policy.n_action_steps=${n_action_steps} \
#                             exp_name=${exp_name} \
#                             logging.mode=${wandb_mode} \
#                             checkpoint.save_ckpt=${save_ckpt}

# num_inference_steps=4
# python eval_robotwin_multitask.py --config-name=${config_name}.yaml \
#                             +eval_mode=${eval_mode} \
#                             +task_mode=${task_mode} \
#                             robotwin_task=${task_name} \
#                             robotwin_task.env_runner.eval_episodes=${eval_episode} \
#                             hydra.run.dir=${run_dir} \
#                             training.debug=$DEBUG \
#                             training.seed=${seed} \
#                             training.device="cuda:0" \
#                             policy.num_inference_steps=${num_inference_steps} \
#                             policy.n_obs_steps=${n_obs_steps} \
#                             policy.horizon=${horizon} \
#                             policy.n_action_steps=${n_action_steps} \
#                             policy.n_layer=${n_layer} \
#                             exp_name=${exp_name} \
#                             logging.mode=${wandb_mode} \
#                             checkpoint.save_ckpt=${save_ckpt}


# num_inference_steps=2
# python eval_robotwin_multitask.py --config-name=${config_name}.yaml \
#                             +eval_mode=${eval_mode} \
#                             +task_mode=${task_mode} \
#                             robotwin_task=${task_name} \
#                             robotwin_task.env_runner.eval_episodes=${eval_episode} \
#                             hydra.run.dir=${run_dir} \
#                             training.debug=$DEBUG \
#                             training.seed=${seed} \
#                             training.device="cuda:0" \
#                             policy.num_inference_steps=${num_inference_steps} \
#                             policy.horizon=${horizon} \
#                             policy.n_action_steps=${n_action_steps} \
#                             policy.n_obs_steps=${n_obs_steps} \
#                             exp_name=${exp_name} \
#                             logging.mode=${wandb_mode} \
#                             checkpoint.save_ckpt=${save_ckpt}

# num_inference_steps=8
# python eval_robotwin_multitask.py --config-name=${config_name}.yaml \
#                             +eval_mode=${eval_mode} \
#                             +task_mode=${task_mode} \
#                             robotwin_task=${task_name} \
#                             robotwin_task.env_runner.eval_episodes=${eval_episode} \
#                             hydra.run.dir=${run_dir} \
#                             training.debug=$DEBUG \
#                             training.seed=${seed} \
#                             training.device="cuda:0" \
#                             policy.num_inference_steps=${num_inference_steps} \
#                             policy.horizon=${horizon} \
#                             policy.n_action_steps=${n_action_steps} \
#                             policy.n_obs_steps=${n_obs_steps} \
#                             exp_name=${exp_name} \
#                             logging.mode=${wandb_mode} \
#                             checkpoint.save_ckpt=${save_ckpt}
