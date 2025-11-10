import os
import sys

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
import torch
from torch import Tensor
import numpy as np

class SmolVLA(SmolVLAPolicy):
    def __init__(
        self,
        config: SmolVLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
        self.observation_window = None  # 新增属性
    
    def set_language(self, instruction):
        self.instruction = instruction
    def update_observation_window(self, img_arr, state):
    
        img_head, img_right, img_left ,img_front= (
            img_arr[0], img_arr[1], img_arr[2],img_arr[3]
        )
        
        def prepare_img(img):
            
            img = np.transpose(img, (2, 0, 1))
            img = img[np.newaxis, ...]
            img = img.astype(np.float32) / 255.0
            
            return torch.from_numpy(img)
            
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
            
        state_tensor = torch.from_numpy(np.array(state, dtype=np.float32))
        state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.to(device)
        
        self.observation_window = {
            "observation.state": state_tensor,  # 机器人状态
            # "observation.images.front_camera": prepare_img(img_front).to(device),    # 前视相机
            "observation.images.cam_left_wrist": prepare_img(img_left).to(device),      # 左侧相机
            "observation.images.cam_right_wrist": prepare_img(img_right).to(device),    # 右侧相机
            "observation.images.cam_high": prepare_img(img_head).to(device),      # 头部相机
            "task": self.instruction,  # 任务指令
        }
    def get_action(self):
        assert self.observation_window is not None, "Update observation_window first!"
        
        action_tensor = self.select_action(self.observation_window)
        
        action_numpy = action_tensor.cpu().numpy()
        action_numpy = action_numpy.squeeze(0)
        
        return action_numpy
    