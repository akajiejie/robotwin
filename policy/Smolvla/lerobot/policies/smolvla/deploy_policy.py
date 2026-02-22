import torch
import os, sys
import numpy as np

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

# Encode observation for the model
# This function should convert the environment observation into the dict format expected by SmolVLAPolicy
# (see SmolVLAPolicy.prepare_images, prepare_state, prepare_language)
def encode_obs(observation):
    # Example: expects observation["observation"]["head_camera"]["rgb"] etc.
    # You may need to adapt these keys to your actual observation structure
    obs_dict = {}
    obs = observation["observation"]
    # Images: convert to numpy arrays, float32, [0,1], channel-first
    obs_dict["observation.images.cam_high"] = (
        np.asarray(obs["head_camera"]["rgb"], dtype=np.float32) / 255.0
    )
    obs_dict["observation.images.cam_right_wrist"] = (
        np.asarray(obs["right_camera"]["rgb"], dtype=np.float32) / 255.0
    )
    obs_dict["observation.images.cam_left_wrist"] = (
        np.asarray(obs["left_camera"]["rgb"], dtype=np.float32) / 255.0
    )
    # State: should be a 1D np.array
    obs_dict["observation.state"] = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    return obs_dict


def get_model(usr_args):
    # usr_args should provide model_path (directory or file), and optionally config_path
    model_path = usr_args["model_path"]
    config_path = usr_args.get("config_path", None)
    device = usr_args.get("device", "cpu")
    # Load config
    if config_path is not None:
        config = SmolVLAConfig.from_pretrained(config_path)
    else:
        config = SmolVLAConfig()
    # Instantiate model
    model = SmolVLAPolicy(config)
    # Load weights
    model = SmolVLAPolicy._load_as_safetensor(model, model_path, map_location=device, strict=False)
    model.eval()
    model.to(device)
    return model


def eval(TASK_ENV, model, observation, task_str=None):
    # Set language/instruction if needed
    if hasattr(model, "set_language"):
        instruction = TASK_ENV.get_instruction() if task_str is None else task_str
        model.set_language(instruction)
    # Encode observation
    obs_dict = encode_obs(observation)
    # Add task string if required by model
    if task_str is not None:
        obs_dict["task"] = task_str
    # Convert to torch tensors (batch dimension)
    for k in obs_dict:
        v = obs_dict[k]
        if v.ndim == 3 and "image" in k:
            v = torch.from_numpy(v).float().unsqueeze(0)  # (1, C, H, W)
        elif v.ndim == 1:
            v = torch.from_numpy(v).float().unsqueeze(0)  # (1, D)
        obs_dict[k] = v
    # Run model
    with torch.no_grad():
        actions = model.select_action(obs_dict)
    # Take action(s) in the environment
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()
    TASK_ENV.take_action(actions)
    # Get new observation
    new_obs = TASK_ENV.get_obs()
    return new_obs


def reset_model(model):
    if hasattr(model, "reset"):
        model.reset() 