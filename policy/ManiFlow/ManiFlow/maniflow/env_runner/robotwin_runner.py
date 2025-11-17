import wandb
import numpy as np
import torch
import tqdm

from maniflow.policy.base_policy import BasePolicy
from maniflow.common.pytorch_util import dict_apply
from maniflow.env_runner.base_runner import BaseRunner
import maniflow.common.logger_util as logger_util
from termcolor import cprint
import pdb
from queue import deque


class RobotRunner(BaseRunner):

    def __init__(
        self,
        output_dir,
        eval_episodes=20,
        max_steps=200,
        n_obs_steps=8,
        n_action_steps=8,
        fps=10,
        crf=22,
        render_size=84,
        tqdm_interval_sec=5.0,
        task_name=None,
        use_point_crop=True,
    ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        self.obs = deque(maxlen=n_obs_steps + 1)
        # self.obs = deque(maxlen=2)
        # self.n_action_steps = 15
        cprint(f"RobotRunner initialized with n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}", "green")
        self.env = None

    def stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert len(self.obs) > 0, "no observation is recorded, please update obs first"

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs([obs[key] for obs in self.obs], self.n_obs_steps)

        return result

    def get_action(self, policy: BasePolicy, observaton=None) -> bool:
        device, dtype = policy.device, policy.dtype
        if observaton is not None:
            self.obs.append(observaton)  # update
        obs = self.get_n_steps_obs()

        # create obs dict
        np_obs_dict = dict(obs)
        # device transfer
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        # run policy
        with torch.no_grad():
            obs_dict_input = {}  # flush unused keys
            obs_dict_input["head_cam"] = obs_dict["head_cam"].unsqueeze(0)
            # obs_dict_input['front_cam'] = obs_dict['front_cam'].unsqueeze(0)
            # obs_dict_input["left_cam"] = obs_dict["left_cam"].unsqueeze(0)
            # obs_dict_input["right_cam"] = obs_dict["right_cam"].unsqueeze(0)
            obs_dict_input["point_cloud"] = obs_dict["point_cloud"].unsqueeze(0)
            obs_dict_input["agent_pos"] = obs_dict["agent_pos"].unsqueeze(0)

            obs_dict_input['head_point_cloud'] = obs_dict['head_point_cloud'].unsqueeze(0)
            obs_dict_input['head_depth'] = obs_dict['head_depth'].unsqueeze(0)
            obs_dict_input['head_intrinsic_cv'] = obs_dict['head_intrinsic_cv'].unsqueeze(0)
            obs_dict_input['head_cam2world_gl'] = obs_dict['head_cam2world_gl'].unsqueeze(0)

            action_dict = policy.predict_action(obs_dict_input)

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
        action = np_action_dict["action"].squeeze(0)

        # # apply temporal aggregation
        # if temporal_agg:
        #     action_horizon = 15  # Assuming fixed horizon, adjust as needed, 16 - 1 
        #     action_dim = 14  # Assuming 7-DOF per arm   
        #     ensemble_steps=8  # Number of steps to ensemble over         
            
        #     # Initialize action buffer
        #     all_time_actions = np.zeros((self.step_lim, self.step_lim + action_horizon, action_dim))
        #     print(f"Temporal ensembling enabled with ensemble_steps={ensemble_steps}, action_horizon={action_horizon}")
        #     # Store raw actions in the buffer
        #     all_time_actions[[iter_idx], iter_idx:iter_idx + action_horizon] = raw_actions
            
        #     # Perform temporal ensembling
        #     action_seq_for_curr_step = all_time_actions[:, iter_idx:iter_idx + action_horizon]
        #     target_pose_list = []
            
        #     for i in range(action_horizon):
        #         actions_for_curr_step = action_seq_for_curr_step[max(0, iter_idx - ensemble_steps + 1):iter_idx + 1, i]
        #         actions_populated = np.all(actions_for_curr_step != 0, axis=1)
        #         actions_for_curr_step = actions_for_curr_step[actions_populated]

        #         if len(actions_for_curr_step) > 0:
        #             k = -0.01
        #             exp_weights = np.exp(k * np.arange(len(actions_for_curr_step)))
        #             exp_weights = exp_weights / exp_weights.sum()
                    
        #             # Simple weighted average across all action dimensions
        #             weighted_action = (actions_for_curr_step * exp_weights[:, np.newaxis]).sum(axis=0, keepdims=True)
        #             target_pose_list.append(weighted_action)
        #         else:
        #             # If no valid actions, use the current raw action
        #             target_pose_list.append(raw_actions[[i]])
            
        #     actions = np.concatenate(target_pose_list, axis=0)
        #     print(f"Temporal ensembling applied at step {iter_idx}")
        
        return action

    def run(self, policy: BasePolicy):
        pass


if __name__ == "__main__":
    test = RobotRunner("./")
    print("ready")
