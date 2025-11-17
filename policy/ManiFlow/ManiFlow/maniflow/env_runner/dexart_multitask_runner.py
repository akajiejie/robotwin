import wandb
import numpy as np
import torch
import tqdm
from typing import Dict, List
from maniflow.env_runner.base_runner import BaseRunner
from maniflow.env_runner.dexart_runner import DexArtRunner
from maniflow.policy.base_policy import BasePolicy

class DexArtMultitaskRunner(BaseRunner):
    def __init__(self,
                output_dir,
                 n_train=10,
                 max_steps=250,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_names=None,
                 use_test_env=False,
                 agent_pos_length=33, # 33, to align for multitasks
                ):
        super().__init__(output_dir)
        
        self.task_names = task_names
        self.runners = {}
        
        # Create individual runners for each task
        for task_name in task_names:
            self.runners[task_name] = DexArtRunner(
                output_dir=output_dir,
                n_train=n_train,
                max_steps=max_steps,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                fps=fps,
                crf=crf,
                tqdm_interval_sec=tqdm_interval_sec,
                task_name=task_name,
                use_test_env=use_test_env,
                agent_pos_length=agent_pos_length
            )
            
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BasePolicy):
        results = {}
        avg_success_rate = 0.0
        avg_reward = 0.0
        
        for task_name in tqdm.tqdm(self.task_names, 
                                desc="Evaluating DexArt tasks",
                                leave=False,
                                mininterval=self.tqdm_interval_sec):
            # Run evaluation for this task
            runner = self.runners[task_name]
            task_results = runner.run(policy)
            results[task_name] = task_results
            
            # Calculate metrics for this task
            success_rate = task_results.get('mean_success_rates_train', 0.0)
            reward = task_results.get('mean_returns_train', 0.0)
            avg_success_rate += success_rate
            avg_reward += reward
            
        # Calculate average metrics across all tasks
        avg_success_rate /= len(self.task_names)
        avg_reward /= len(self.task_names)
        
        results['average_success_rate'] = avg_success_rate
        results['average_reward'] = avg_reward

        return results