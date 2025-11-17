# Code Attribution

## 🔹 Open-Source / Third-Party Code

The following files or modules are adapted from open-source ManiFlow project. 
- Folder: `ManiFlow/ManiFlow/maniflow/common`
- Folder: `ManiFlow/ManiFlow/maniflow/config`, excluding the following files:
  - `ManiFlow/ManiFlow/maniflow/config/task`
  - `ManiFlow/ManiFlow/maniflow/config/robot_unified_maniflow.yaml`
  - `ManiFlow/ManiFlow/maniflow/config/robot_unified_maniflow_pointmap.yaml`
- Folder: `ManiFlow/ManiFlow/maniflow/dataset`, excluding the following files:
  - `ManiFlow/ManiFlow/maniflow/dataset/robotwin_dataset.py`
  - `ManiFlow/ManiFlow/maniflow/dataset/robotwin_pointmap_dataset.py`
  - `ManiFlow/ManiFlow/maniflow/dataset/robotwin_dataset_batch.py`
- Folder: `ManiFlow/ManiFlow/maniflow/envs`
- Folder: `ManiFlow/ManiFlow/maniflow/env_runner`, excluding the following files:
  - `ManiFlow/ManiFlow/maniflow/env_runner/robotwin_runner.py`
- Folder: `ManiFlow/ManiFlow/maniflow/gym_util`
- Folder: `ManiFlow/ManiFlow/maniflow/model`, excluding the following files:
  - `ManiFlow/ManiFlow/maniflow/model/vision_2d/timm_pointmap_encoder.py`
  - `ManiFlow/ManiFlow/maniflow/model/vision_2d/transformer_pointmap_encoder.py`
  - `ManiFlow/ManiFlow/maniflow/model/vision_3d/randla`
  - `ManiFlow/ManiFlow/maniflow/model/vision_3d/fp3_encoder.py`
  - `ManiFlow/ManiFlow/maniflow/model/vision_3d/pointnet2_encoder.py`
  - `ManiFlow/ManiFlow/maniflow/model/vision_3d/randlanet_encoder.py`
  - `ManiFlow/ManiFlow/maniflow/model/diffusion/adaptive_attention.py`
  - `ManiFlow/ManiFlow/maniflow/model/diffusion/hrdt_block.py`
  - `ManiFlow/ManiFlow/maniflow/model/diffusion/mmdit_block.py`
  - `ManiFlow/ManiFlow/maniflow/model/diffusion/mmdit_generalized_block.py`
  - `ManiFlow/ManiFlow/maniflow/model/diffusion/mmdit.py`
  - `ManiFlow/ManiFlow/maniflow/model/diffusion/routing_module.py`
  - `ManiFlow/ManiFlow/maniflow/model/diffusion/unified_ditx.py`

- Folder: `ManiFlow/ManiFlow/maniflow/policy`, excluding the following files:
    - `ManiFlow/ManiFlow/maniflow/policy/maniflow_unified_policy.py`
    - `ManiFlow/ManiFlow/maniflow/policy/maniflow_unified_policy_pointmap.py`
    - `ManiFlow/ManiFlow/maniflow/policy/meanflow_pointcloud_policy.py`
- Train scripts in `ManiFlow/ManiFlow`, excluding the following files:
  - `ManiFlow/ManiFlow/maniflow_policy.py`
  - `ManiFlow/ManiFlow/robotwin_batch_ddp.py`
  - `ManiFlow/ManiFlow/train_robotwin_batch_mp.py`
  - `ManiFlow/ManiFlow/train_robotwin_batch.py`
- More scripts in `ManiFlow/scripts`, excluding the following files:
  - `ManiFlow/scripts/process_data_depth.py`
  - `ManiFlow/scripts/process_data_unified.py`
  - `ManiFlow/scripts/train_policy_batch.sh`
  - `ManiFlow/scripts/train_policy_ddp_batch.sh`
- Third-party Folder: `ManiFlow/third_party`
- Visualizer: `ManiFlow/visualizer`
-- Other sccripts in `ManiFlow/`, excluding the following files:
  - `ManiFlow/deploy_policy.py`
  - `ManiFlow/deploy_policy.yml`
  - `ManiFlow/process_data_depth.sh`
  - `ManiFlow/process_data_unified.sh`
  - `ManiFlow/train_eval_batch.sh`
  - `ManiFlow/train_eval_ddp_demo.sh`
  - `ManiFlow/train_eval_ddp.sh`
  - `ManiFlow/train_eval_demo.sh`
  - `ManiFlow/train_eval.sh`

---

## 🔹 Licensing

- **ManiFlow code in this repository** is licensed under the MIT License.  
- **Third-party components** remain under their original licenses (see `licenses/` directory).

---

## 🔹 Notes on Modifications

Where original ManiFlow code has been adopted, inline comments indicate changes. Example:

```python
# Adapted from https://github.com/geyan21/ManiFlow_Policy with more details
```

