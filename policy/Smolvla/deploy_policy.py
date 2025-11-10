# deploy_policy.py for SmolVLA
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
import torch
import numpy as np
from typing import Any, Dict
from smolvla_model import SmolVLA

model: SmolVLA = None
device: str = "cpu"

def get_model(usr_args: Dict) -> SmolVLA:
    global model, device
    
    device = usr_args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SmolVLA model on device: {device}")
    policy_path = usr_args["policy_path"]
    
    # Convert relative path to absolute path
    if not os.path.isabs(policy_path):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        policy_path = os.path.join(script_dir, policy_path)
    
    model = SmolVLA.from_pretrained(policy_path)
    model.to(device)
    model.eval()
    
    print(f"Successfully loaded model: {policy_path}")
    return model

def reset_model(model) -> None:
    if model:
        print("Resetting SmolVLA internal state (action queue).")
        model.reset()

def encode_obs(observation):
    #print(observation)
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
        observation["observation"]["front_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state

def eval(TASK_ENV: Any, model: SmolVLA, observation: Dict) -> None:
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)

    model.update_observation_window(input_rgb_arr, input_state)

    action = model.get_action()  # Get Action according to observation chunk
    #model.reset_observation_window()
    #print(action)
    #print(action.shape)
    TASK_ENV.take_action(action)
