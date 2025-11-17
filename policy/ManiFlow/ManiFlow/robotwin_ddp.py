if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent) # set to the root directory of ManiFlow
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)

import os
import hydra
import torch
import torch.distributed as dist
import dill
from datetime import datetime
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import time
import threading
import sys

MANIFLOW_ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(MANIFLOW_ROOT)
sys.path.append(os.path.join(MANIFLOW_ROOT, 'ManiFlow'))
sys.path.append(os.path.join(MANIFLOW_ROOT, 'ManiFlow', 'maniflow'))

sys.path.insert(0, '../')
sys.path.append('ManiFlow/env_runner')
sys.path.append('ManiFlow/maniflow/policy')
sys.path.append('ManiFlow')
sys.path.append('ManiFlow/maniflow')

from hydra.core.hydra_config import HydraConfig
from maniflow.policy.dp3 import DP3
from maniflow.dataset.base_dataset import BaseDataset
from maniflow.env_runner.base_runner import BaseRunner
from maniflow.common.checkpoint_util import TopKCheckpointManager
from maniflow.common.pytorch_util import dict_apply, optimizer_to
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.model.common.lr_scheduler import get_scheduler
import re

# Register resolvers
OmegaConf.register_new_resolver("eval", eval, replace=True)

def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        return {k: _copy_to_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)

def get_raw_task_name(task_name):
    # Use regex to check if the task name has a numeric suffix
    if re.search(r'_\d+$', task_name):
        # Remove the numeric suffix
        raw_task_name = re.sub(r'_\d+$', '', task_name)
        return raw_task_name
    else:
        # Return the original task name
        return task_name

import pdb
import sys
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class DDPModelWrapper:
    """Wrapper to automatically handle .module access for DDP models"""
    def __init__(self, model):
        self.model = model
        
    def __getattr__(self, name):
        # First try to get the attribute from the DDP model
        try:
            attr = getattr(self.model, name)
            return attr
        except AttributeError:
            # If not found, try to get from .module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                return getattr(self.model.module, name)
            raise

class DDPWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, local_rank: int, output_dir=None):
        self.cfg = cfg
        self.local_rank = local_rank
        self._output_dir = output_dir
        self._saving_thread = None
        
        # Set seed
        seed = cfg.training.seed + local_rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Configure device
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)

        # Configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)
        self.model = self.model.to(self.device)
        
        if cfg.training.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank
            )
            # Wrap the model
            self.model = DDPModelWrapper(self.model)

        # Configure EMA model
        self.ema_model = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:  # minkowski engine could not be copied
                self.ema_model = hydra.utils.instantiate(cfg.policy)
                self.ema_model = self.ema_model.to(self.device)
                if cfg.training.distributed:
                    self.ema_model = torch.nn.parallel.DistributedDataParallel(
                        self.ema_model, device_ids=[local_rank], output_device=local_rank
                    )
                    self.ema_model = DDPModelWrapper(self.ema_model)

        # Configure optimizer
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        
        # Configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        is_main_process = self.local_rank == 0
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_ROLLOUT = False
        RUN_VALIDATION = False # reduce time cost

        # Resume training if needed
        # if cfg.training.resume and is_main_process:
        #     lastest_ckpt_path = self.get_checkpoint_path()
        #     if lastest_ckpt_path.is_file():
        #         print(f"Resuming from checkpoint {lastest_ckpt_path}")
        #         self.load_checkpoint(path=lastest_ckpt_path)
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            
            # Only rank 0 prints - no barrier needed here
            if is_main_process:
                if lastest_ckpt_path.is_file():
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                # else:
                #     print(f"No checkpoint found at {lastest_ckpt_path}")

            # All ranks load the checkpoint if it exists
            if lastest_ckpt_path.is_file():
                self.load_checkpoint(path=lastest_ckpt_path)

            # Single barrier: ensure all ranks have loaded checkpoint
            # before proceeding with training
            # if cfg.training.distributed:
            #     dist.barrier()

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        
        if cfg.training.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=cfg.training.num_gpus,
                rank=self.local_rank
            )
            train_dataloader = DataLoader(
                dataset,
                sampler=train_sampler,
                batch_size=cfg.dataloader.batch_size, # batch size per GPU
                num_workers=cfg.dataloader.num_workers, # workers per GPU
                pin_memory=cfg.dataloader.pin_memory,
                persistent_workers=cfg.dataloader.persistent_workers,
                drop_last=cfg.dataloader.drop_last,
            )
            cprint(f"Rank {self.local_rank} - Dataset: {dataset.__class__.__name__}, batch_size: {cfg.dataloader.batch_size}", 'red')
        else:
            train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # Configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        if is_main_process:
            # print dataset info
            cprint(f"Dataset: {dataset.__class__.__name__}", 'red')
            cprint(f"Dataset Path: {dataset.zarr_path}", 'red')
            cprint(f"Number of training episodes: {dataset.train_episodes_num}", 'red')
            cprint(f"Number of validation episodes: {dataset.val_episodes_num}", 'red')

        # Set normalizer
        normalizer = dataset.get_normalizer()
        # if cfg.training.distributed:
        #     self.model.set_normalizer(normalizer)
        #     if cfg.training.use_ema:
        #         self.ema_model.set_normalizer(normalizer)
        # else:
        #     self.model.set_normalizer(normalizer)
        #     if cfg.training.use_ema:
        #         self.ema_model.set_normalizer(normalizer)
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        # Configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model
            )

        # Configure env runner
        env_runner = None
        
        # Configure logging
        if is_main_process:
            cfg.logging.name = str(cfg.logging.name)
            cprint("-----------------------------", "yellow")
            cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
            cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
            cprint("-----------------------------", "yellow")
            
            # Register time resolver for wandb
            def time_resolver(pattern: str) -> str:
                return datetime.now().strftime(pattern)
            OmegaConf.register_new_resolver("now", time_resolver)
            
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # Configure checkpoint
        if is_main_process:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

        # Save batch for sampling
        train_sampling_batch = None

        # Training loop
        for local_epoch_idx in range(cfg.training.num_epochs):
            if cfg.training.distributed:
                train_sampler.set_epoch(local_epoch_idx)
                
            step_log = dict()
            train_losses = list()
            
            # Training
            self.model.train()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec, 
                    disable=not is_main_process) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    if cfg.policy.use_train_state:
                        # get train state
                        train_state = {
                            'cur_train_epoch': self.epoch,
                            'cur_train_step': self.global_step,
                            'max_train_epoch': cfg.training.num_epochs,
                            'max_train_step': len(train_dataloader) * cfg.training.num_epochs,
                            'ema_model': self.ema_model.module if cfg.training.use_ema else self.ema_model,
                        }
                        # Forward pass
                        raw_loss, loss_dict = self.model.compute_loss(batch, train_state)
                    else:
                        # Forward pass
                        raw_loss, loss_dict = self.model.compute_loss(batch)

                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    
                    # Backward pass
                    loss.backward()
                    
                    # Optimizer step
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    # Update EMA
                    if cfg.training.use_ema:
                        ema.step(self.model)

                    # Logging
                    raw_loss_cpu = raw_loss.item()
                    if is_main_process:
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log.update({
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        })
                        step_log.update(loss_dict)
                    
                    # Log step metrics if not last batch
                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        if is_main_process:
                            wandb_run.log(step_log, step=self.global_step)

                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break
            
            # at the end of each epoch
            if is_main_process:
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
            # Validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                if is_main_process:
                    cprint(f"Validation at epoch {self.epoch}", 'green')
                    policy = self.ema_model if cfg.training.use_ema else self.model
                    policy.eval()
                
                    with torch.no_grad():
                        val_losses = []
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                                if cfg.policy.use_train_state:
                                    # get train state
                                    train_state = {
                                        'cur_train_epoch': self.epoch,
                                        'cur_train_step': self.global_step,
                                        'max_train_epoch': cfg.training.num_epochs,
                                        'max_train_step': len(train_dataloader) * cfg.training.num_epochs,
                                        'ema_model': self.ema_model.module if cfg.training.use_ema else self.ema_model,
                                    }
                                    # Forward pass
                                    val_loss, loss_dict = policy.compute_loss(batch, train_state)
                                else:
                                    # Forward pass
                                    val_loss, loss_dict = policy.compute_loss(batch)
                                val_losses.append(val_loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                    policy.train()
                    
                # # ForkedPdb().set_trace()
                # if cfg.training.distributed:
                #     dist.barrier()

            elif self.epoch == 0 and step_log.get('val_loss', None) is None:
                if is_main_process:
                    step_log['val_loss'] = train_loss
            
            # # run diffusion sampling on a training batch
            # if (self.epoch % cfg.training.sample_every) == 0:
            #     if is_main_process:
            #         with torch.no_grad():
            #             # sample trajectory from training set, and evaluate difference
            #             batch = dict_apply(train_sampling_batch, lambda x: x.to(self.device, non_blocking=True))
            #             obs_dict = batch['obs']
            #             gt_action = batch['action']

            #             policy = self.ema_model.module if cfg.training.use_ema else self.model.module
                        
            #             result = policy.predict_action(obs_dict)
            #             pred_action = result['action_pred']
            #             mse = torch.nn.functional.mse_loss(pred_action, gt_action.to(pred_action.device))
            #             step_log['train_action_mse_error'] = mse.item()
            #             del batch
            #             del obs_dict
            #             del gt_action
            #             del result
            #             del pred_action
            #             del mse

            # Rollout
            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT:
                if is_main_process:
                    cprint(f"Epoch {self.epoch} - Rollout", 'green')
                    policy = self.ema_model.module if cfg.training.use_ema else self.model.module
                    policy.eval()
                    
                    runner_logs = env_runner.run(policy.module)
                    
                    average_success_rate = runner_logs['average_success_rate']
                    step_log.update({'average_success_rate': average_success_rate})
                    # remove average_success_rate from log
                    runner_logs.pop('average_success_rate')

                    average_reward = runner_logs.get('average_reward', None)
                    if average_reward is not None:
                        step_log.update({'average_reward': average_reward})
                        runner_logs.pop('average_reward')
                    
                    for task_name, task_log_dict in list(runner_logs.items()):
                        # Identify keys that contain 'video'
                        keys_to_remove = [key for key in task_log_dict if 'video' in key]
                        
                        for key in keys_to_remove:
                            value = task_log_dict[key]
                            # Update step_log with the video-related entry
                            step_log.update({f"{task_name}/{key}": value})
                            # Remove the key from task_log_dict
                            task_log_dict.pop(key)

                    # Log per-task metrics
                    for task_name in dataset.raw_task_name:
                        task_metrics = runner_logs[task_name]
                        task_log = {f"{task_name}/{k}": v for k, v in task_metrics.items()}
                        step_log.update(task_log)
                    
                    # Calculate and log mean metrics across tasks
                    mean_metrics = {}
                    for metric in runner_logs[dataset.raw_task_name[0]].keys():
                        values = [runner_logs[task][metric] for task in dataset.raw_task_name]
                        mean_metrics[f"mean/{metric}"] = np.mean(values)
                    step_log.update(mean_metrics)
                        
                    cprint(f"Epoch {self.epoch} - Success Rate: {average_success_rate:.4f}", 'green')

                    policy.train()
                
                if cfg.training.distributed:
                    dist.barrier()
            
            elif self.epoch == 0 and step_log.get('average_success_rate', None) is None:
                if is_main_process:
                    runner_logs = dict()
                    runner_logs['mean/test_mean_score'] = 0
                    runner_logs['mean/mean_success_rates'] = 0
                    runner_logs['mean/SR_test_L3'] = 0
                    runner_logs['mean/SR_test_L5'] = 0
                    step_log.update(runner_logs)

            # Checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                if is_main_process:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                        
                    metric_dict = dict()
                    for key, value in step_log.items():
                        # remove mean/ prefix
                        if key.startswith('mean/'):
                            new_key = key.replace('mean/', '')
                        else:
                            new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    try:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                    except Exception as e:
                        print(f"Error saving topk checkpoint: {e}")

                # ========= eval end for this epoch ==========

            # end of epoch
            if is_main_process:
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log
            
        if cfg.training.distributed:
            dist.destroy_process_group()

    # Keep your original checkpoint methods
    @property
    def output_dir(self):
        return self._output_dir or HydraConfig.get().runtime.output_dir

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
         # Handle DDP state dict by removing 'module.' prefix if necessary
        def fix_state_dict(state_dict, is_current_ddp):
            """Helper function to handle DDP state dict prefixes"""
            if not state_dict:  # Handle empty state dict
                return state_dict
                
            is_checkpoint_ddp = list(state_dict.keys())[0].startswith('module.')
            
            if is_checkpoint_ddp and not is_current_ddp:
                # Remove 'module.' prefix
                return {k[7:]: v for k, v in state_dict.items()}
            elif not is_checkpoint_ddp and is_current_ddp:
                # Add 'module.' prefix
                return {f'module.{k}': v for k, v in state_dict.items()}
            return state_dict
        
        # Handle main model state dict
        if 'state_dicts' in payload and 'model' in payload['state_dicts']:
            is_current_ddp = isinstance(self.model.model, torch.nn.parallel.DistributedDataParallel)
            payload['state_dicts']['model'] = fix_state_dict(
                payload['state_dicts']['model'],
                is_current_ddp
            )
        
        # Handle EMA model state dict
        if 'state_dicts' in payload and 'ema_model' in payload['state_dicts']:
            is_current_ddp = (hasattr(self, 'ema_model') and 
                            isinstance(self.ema_model.model, torch.nn.parallel.DistributedDataParallel))
            payload['state_dicts']['ema_model'] = fix_state_dict(
                payload['state_dicts']['ema_model'],
                is_current_ddp
            )
        # if 'state_dicts' in payload and 'model' in payload['state_dicts']:
        #     state_dict = payload['state_dicts']['model']
        #     if list(state_dict.keys())[0].startswith('module.'):
        #         # If current model is not DDP, remove 'module.' prefix
        #         if not isinstance(self.model.model, torch.nn.parallel.DistributedDataParallel):
        #             payload['state_dicts']['model'] = {
        #                 k[7:]: v for k, v in state_dict.items()
        #             }
        #             # import pdb; pdb.set_trace()
        #     else:
        #         # If current model is DDP but checkpoint is not, add 'module.' prefix
        #         if isinstance(self.model.model, torch.nn.parallel.DistributedDataParallel):
        #             payload['state_dicts']['model'] = {
        #                 f'module.{k}': v for k, v in state_dict.items()
        #             }
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)