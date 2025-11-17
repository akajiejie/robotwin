import wandb
import numpy as np
import torch
import tqdm
from typing import Dict, List
from maniflow.policy.base_policy import BasePolicy
from maniflow.common.pytorch_util import dict_apply
from maniflow.env_runner.base_runner import BaseRunner
import maniflow.common.logger_util as logger_util
from maniflow.env_runner.robot_runner import RobotRunner
from termcolor import cprint
import pdb
from queue import deque
import importlib
import pathlib
import re

def get_raw_task_name(task_name):
    # Use regex to check if the task name has a numeric suffix
    if re.search(r'_\d+$', task_name):
        # Remove the numeric suffix
        raw_task_name = re.sub(r'_\d+$', '', task_name)
        return raw_task_name
    else:
        # Return the original task name
        return task_name

class RobotMultitaskRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 task_names: List[str],
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 use_point_crop=True,
                 multitask_config=None,
                 **kwargs
                 ):
        super().__init__(output_dir)
        # task name may have a numeric suffix, so we need to get the raw task name
        task_names = [get_raw_task_name(task_name) for task_name in task_names]
        self.task_names = task_names
        self.runners = {}
        self.eval_episodes = eval_episodes
        self.tqdm_interval_sec = tqdm_interval_sec
        self.multitask_config = multitask_config
        self.use_point_crop = use_point_crop
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.crf = crf
        self.render_size = render_size

    def create_runners(self, task_name):
        # add task name to multitask config and set it as task config for each task
        this_task_config = self.multitask_config.copy()
        this_task_config['task_name'] = task_name
        cprint(f"Creating runner for task: {task_name}", 'green')
        return RobotRunner(
            output_dir=self.output_dir,
            eval_episodes=self.eval_episodes,
            max_steps=self.max_steps,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            fps=self.fps,
            crf=self.crf,
            render_size=self.render_size,
            tqdm_interval_sec=self.tqdm_interval_sec,
            task_name=task_name,
            use_point_crop=self.use_point_crop,
            task_config=this_task_config
        )
    
    def run(self, policy: BasePolicy):
        results = {}
        avg_success_rate = 0.0

        for task_name in tqdm.tqdm(self.task_names, 
                                 desc="Evaluating tasks",
                                 leave=False,
                                 mininterval=self.tqdm_interval_sec):
            runner = self.create_runners(task_name)
            task_results = runner.run(policy)
            results[task_name] = task_results

            # Calculate success rate for this task
            success_rate = np.mean(task_results.get('mean_success_rates', 0.0))
            avg_success_rate += success_rate

            del runner
            # empty cache
            torch.cuda.empty_cache()

        # Calculate average success rate across all tasks
        avg_success_rate /= len(self.task_names)
        
        results['average_success_rate'] = avg_success_rate

        return results 