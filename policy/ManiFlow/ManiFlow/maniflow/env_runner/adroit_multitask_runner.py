import wandb
import numpy as np
import torch
import tqdm
from typing import Dict, List
from maniflow.env_runner.base_runner import BaseRunner
from maniflow.env_runner.adroit_runner import AdroitRunner
from maniflow.policy.base_policy import BasePolicy
from maniflow.dataset.adroit_multitask_dataset import ACTION_LENGTH_DICT

class AdroitMultitaskRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_names=None,
                 use_point_crop=True,
                 action_length_dict=ACTION_LENGTH_DICT
                ):
        super().__init__(output_dir)
        
        self.task_names = task_names
        self.runners = {}
        
        # Create individual runners for each task
        for task_name in task_names:
            self.runners[task_name] = AdroitRunner(
                output_dir=output_dir,
                eval_episodes=eval_episodes,
                max_steps=max_steps,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                fps=fps,
                crf=crf,
                render_size=render_size,
                tqdm_interval_sec=tqdm_interval_sec,
                task_name=task_name,
                use_point_crop=use_point_crop,
                action_length_dict=action_length_dict
            )
            
        self.eval_episodes = eval_episodes
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BasePolicy):
        results = {}
        avg_success_rate = 0.0
        
        for task_name in tqdm.tqdm(self.task_names, 
                                desc="Evaluating Adroit tasks",
                                leave=False,
                                mininterval=self.tqdm_interval_sec):
            # Run evaluation for this task
            runner = self.runners[task_name]
            task_results = runner.run(policy)
            results[task_name] = task_results
            
            # Calculate success rate for this task
            success_rate = task_results.get('mean_success_rates', 0.0)
            avg_success_rate += success_rate
            
        # Calculate average success rate across all tasks
        avg_success_rate /= len(self.task_names)
        results['average_success_rate'] = avg_success_rate

        return results