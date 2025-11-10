"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""
import os
os.environ["HF_LEROBOT_HOME"] = "~/lerobot_datasets"

from pathlib import Path

import dataclasses
import shutil
from typing import Literal

import h5py
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch
import re
import cv2

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None

DEFAULT_DATASET_CONFIG = DatasetConfig()

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]

    cameras = [
        "head_camera",
        "left_camera",
        "right_camera",
        "front_camera"
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def get_cameras(hdf5_file: Path) -> list[str]:
    with h5py.File(hdf5_file, "r") as ep:
        return [key for key in ep["/observation"].keys() if "depth" not in key]  # noqa: SIM118

def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observation/{camera}/rgb"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observation/{camera}/rgb"][:]
        else:

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observation/{camera}/rgb"]:
                data = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs_array.append(img)

        imgs_array = [cv2.resize(img, (640, 480)) for img in imgs_array]
        imgs_array = np.stack(imgs_array, axis=0)

        imgs_array = np.transpose(imgs_array, (0, 3, 1, 2))
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam

def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
        dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/joint_action/vector"][:]).float()
        action = torch.from_numpy(ep["/joint_action/vector"][:]).float()

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            get_cameras(ep_path),
        )

    return imgs_per_cam, state, action

def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    instruction_dir: Path,
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        
        imgs_per_cam, state, action = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]
        json_path = os.path.join(instruction_dir, f"episode{ep_idx}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Instruction file not found: {json_path}")
        with open(json_path, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
            instructions = instruction_dict['seen']
            instruction = np.random.choice(instructions)
        
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]
            dataset.add_frame(frame, task=instruction)
        dataset.save_episode()

    return dataset

def port_aloha(
    raw_dir: Path,
    instruction_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    repo_id = f"{repo_id}/{task}"
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        # download_raw(raw_dir, repo_id=raw_repo_id)
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(file_path)
    hdf5_files = sorted(hdf5_files, key = lambda x: int(re.search(r"episode(\d+)\.hdf5", os.path.basename(x)).group(1)))
    
    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha_agilex",
        mode=mode,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        instruction_dir,
        task=task,
        episodes=episodes,
    )
    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()

if __name__ == "__main__":
    tyro.cli(port_aloha)
