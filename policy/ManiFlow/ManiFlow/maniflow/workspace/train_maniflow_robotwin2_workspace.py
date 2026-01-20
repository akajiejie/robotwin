if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
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
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
import sys
import gc
from accelerate import Accelerator
from accelerate.utils import set_seed

MANIFLOW_ROOT = str(pathlib.Path(__file__).parent.parent.parent.parent)
sys.path.append(MANIFLOW_ROOT)
sys.path.append(os.path.join(MANIFLOW_ROOT, 'ManiFlow'))
sys.path.append(os.path.join(MANIFLOW_ROOT, 'ManiFlow', 'maniflow'))

sys.path.insert(0, '../../../')
sys.path.append('ManiFlow/env_runner')
sys.path.append('ManiFlow/maniflow/policy')
sys.path.append('ManiFlow')
sys.path.append('ManiFlow/maniflow')


from hydra.core.hydra_config import HydraConfig
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy
from maniflow.dataset.base_dataset import BaseDataset
from maniflow.env_runner.robot_runner import RobotRunner
from maniflow.common.checkpoint_util import TopKCheckpointManager
from maniflow.common.pytorch_util import dict_apply, optimizer_to
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

def print_memory_usage(prefix=""):
    """Print current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix}GPU Memory: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")
    
    # Python memory usage
    import psutil
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024**3
    print(f"{prefix}RAM Usage: {ram_usage:.2f}GB")

def cleanup_memory():
    """Comprehensive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class TrainManiFlowRoboTwinWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # 🔥 Initialize Accelerator if enabled
        self.use_accelerate = cfg.training.get('use_accelerate', False)
        self.accelerator = None
        
        if self.use_accelerate:
            # 从配置中获取梯度累积步数
            gradient_accumulation_steps = cfg.training.get('gradient_accumulate_every', 1)
            
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=cfg.training.get('mixed_precision', 'no'),  # 'no', 'fp16', 'bf16'
                log_with="wandb" if cfg.training.get('use_wandb', True) else None,
                project_dir=output_dir,
            )
            
            # Use accelerate's set_seed for better reproducibility
            set_seed(cfg.training.seed, device_specific=True)
            
            cprint(f"🚀 Accelerate initialized:", 'cyan')
            cprint(f"   - Gradient accumulation steps: {gradient_accumulation_steps}", 'cyan')
            cprint(f"   - Mixed precision: {cfg.training.get('mixed_precision', 'no')}", 'cyan')
            cprint(f"   - Device: {self.accelerator.device}", 'cyan')
            cprint(f"   - Num processes: {self.accelerator.num_processes}", 'cyan')
        else:
            # Original seed setting
            seed = cfg.training.seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # configure model
        self.model: ManiFlowTransformerPointcloudPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: ManiFlowTransformerPointcloudPolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except Exception as e: # minkowski engine could not be copied. recreate it
                print(f"Failed to copy model for EMA: {e}")
                print("Recreating EMA model...")
                torch.cuda.empty_cache()  # Clear memory before recreating
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        # self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        WANDB = True
        
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
        RUN_VALIDATION = True # reduce time cost
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # print dataset info
        cprint(f"Dataset: {dataset.__class__.__name__}", 'red')
        cprint(f"Dataset Path: {dataset.zarr_path}", 'red')
        cprint(f"Number of training episodes: {dataset.train_episodes_num}", 'red')
        cprint(f"Number of validation episodes: {dataset.val_episodes_num}", 'red')


        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            
        # configure env runner
        # env_runner: BaseRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseRunner)

        env_runner = None
        
        cfg.logging.name = str(cfg.task.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        
        # 🔥 Prepare with Accelerator if enabled
        if self.use_accelerate:
            # Prepare model, optimizer, dataloaders with accelerator
            self.model, self.optimizer, train_dataloader, val_dataloader, lr_scheduler = \
                self.accelerator.prepare(
                    self.model, self.optimizer, train_dataloader, val_dataloader, lr_scheduler
                )
            
            # EMA model should stay on the same device but not wrapped
            if self.ema_model is not None:
                self.ema_model.to(self.accelerator.device)
            
            device = self.accelerator.device
            
            # Initialize wandb through accelerator
            if WANDB and self.accelerator.is_main_process:
                wandb_run = wandb.init(
                    dir=str(self.output_dir),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    **cfg.logging
                )
                wandb.config.update({"output_dir": self.output_dir})
            else:
                wandb_run = None
                
            cprint(f"✅ Models and optimizers prepared with Accelerator", 'green')
        else:
            # Original device transfer
            device = torch.device(cfg.training.device)
            self.model.to(device)
            if self.ema_model is not None:
                self.ema_model.to(device)
            optimizer_to(self.optimizer, device)
            
            # configure logging
            if WANDB:
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

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )
        
        # Print initial memory usage
        print_memory_usage("[Initial] ")
        cleanup_memory()

        # save batch for sampling
        train_sampling_batch = None
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    
                    # 🔥 Use Accelerate's gradient accumulation context if enabled
                    if self.use_accelerate:
                        # Accelerator handles device transfer automatically
                        if train_sampling_batch is None:
                            train_sampling_batch = dict_apply(batch, lambda x: x.clone().detach())
                        
                        t1_1 = time.time()
                        
                        # Use accumulate context for automatic gradient accumulation
                        with self.accelerator.accumulate(self.model):
                            # Forward pass
                            raw_loss, loss_dict = self.model.compute_loss(batch, self.ema_model)
                            
                            # Accelerator handles scaling automatically
                            self.accelerator.backward(raw_loss)
                            
                            t1_2 = time.time()
                            
                            # Optimizer step (only when accumulated enough gradients)
                            self.optimizer.step()
                            lr_scheduler.step()
                            self.optimizer.zero_grad()
                            
                            t1_3 = time.time()
                        
                        # Update EMA (outside accumulate context)
                        if cfg.training.use_ema:
                            # Get unwrapped model for EMA update
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            ema.step(unwrapped_model)
                        
                        t1_4 = time.time()
                        
                        # Gather loss from all processes
                        raw_loss_cpu = self.accelerator.gather(raw_loss).mean().item()
                        
                    else:
                        # Original training loop (without accelerate)
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = dict_apply(batch, lambda x: x.clone().detach())
                    
                        # compute loss
                        t1_1 = time.time()
                        
                        # Forward pass
                        raw_loss, loss_dict = self.model.compute_loss(batch, self.ema_model)
                        
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
                        
                        t1_2 = time.time()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        t1_3 = time.time()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)
                        t1_4 = time.time()
                        
                        raw_loss_cpu = raw_loss.item()
                    
                    # logging
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    
                    # Get learning rate safely
                    if self.use_accelerate:
                        current_lr = self.optimizer.param_groups[0]['lr']
                    else:
                        current_lr = lr_scheduler.get_last_lr()[0]
                    
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': current_lr
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()
                    
                    # Periodic memory cleanup
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()
                    
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")
                        if batch_idx % 50 == 0:
                            print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if WANDB and (not self.use_accelerate or self.accelerator.is_main_process):
                            #每100步记录一次 MoE gate 权重的直方图
                            if self.global_step % 100 == 0:
                                if self.use_accelerate:
                                    model_to_log = self.accelerator.unwrap_model(self.model)
                                else:
                                    model_to_log = self.model
                                
                                if hasattr(model_to_log, 'model') and hasattr(model_to_log.model, 'blocks'):
                                    for i, block in enumerate(model_to_log.model.blocks):
                                        if hasattr(block, 'mlp') and hasattr(block.mlp, 'gate'):
                                            gate_weights = block.mlp.gate.weight.detach().cpu().numpy()
                                            step_log[f'moe/block_{i}/gate_weights'] = wandb.Histogram(gate_weights)
                            
                            wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout
            if cfg.training.debug:
                min_epoch_rollout = 0
            else:
                min_epoch_rollout = 300
            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None and self.epoch >= min_epoch_rollout: # and self.epoch > 1, and self.epoch >= 100
                cprint(f"Running rollout for epoch {self.epoch}", 'cyan')
                t3 = time.time()
                # runner_log = env_runner.run(policy, dataset=dataset)
                runner_log = env_runner.run(policy)
                t4 = time.time()
                # print(f"rollout time: {t4-t3:.3f}")
                # log all
                step_log.update(runner_log)
            elif self.epoch == 0:
                runner_log = dict()
                runner_log['test_mean_score'] = 0
                runner_log['mean_success_rates'] = 0
                runner_log['SR_test_L3'] = 0
                runner_log['SR_test_L5'] = 0
                runner_log['sim_video_eval'] = None
                step_log.update(runner_log)

            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            if not self.use_accelerate:
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                            # Forward pass
                            loss, loss_dict = self.model.compute_loss(batch, self.ema_model)
                            
                            # Handle loss gathering for accelerate
                            if self.use_accelerate:
                                loss_value = self.accelerator.gather(loss).mean().item()
                            else:
                                loss_value = loss.item()
                            
                            val_losses.append(loss_value)
                            print(f'epoch {self.epoch}, eval loss: ', loss_value)
                            
                            # Periodic memory cleanup during validation
                            if batch_idx % 20 == 0:
                                torch.cuda.empty_cache()
                            
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = np.mean(val_losses)  # Direct numpy mean of scalars
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss
            
            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    if not self.use_accelerate:
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    else:
                        batch = train_sampling_batch
                    
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    
                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    
                    if self.use_accelerate:
                        mse_value = self.accelerator.gather(mse).mean().item()
                    else:
                        mse_value = mse.item()
                    
                    step_log['train_action_mse_error'] = mse_value
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse
            
            if env_runner is None or step_log.get('test_mean_score', None) is None:
                step_log['test_mean_score'] = - train_loss

            # checkpoint - 定期保存（独立于checkpoint_every）
            # Only save on main process when using accelerate
            should_save = (not self.use_accelerate) or self.accelerator.is_main_process
            
            if should_save:
                save_every = cfg.checkpoint.get('save_every', 100)  # 默认100
                if self.epoch % save_every == 0 and self.epoch > 0 and cfg.checkpoint.save_ckpt:
                    epoch_ckpt_path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{self.epoch:04d}.ckpt')
                    self.save_checkpoint(path=epoch_ckpt_path)
                    cprint(f"Saved epoch checkpoint at epoch {self.epoch} (every {save_every} epochs): {epoch_ckpt_path}", 'green')

                # checkpoint - topk and latest
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:

                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    try:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    except Exception as e:
                        print(f"Error in getting topk ckpt path: {e}")
                        topk_ckpt_path = None
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                # if not os.path.exists(f"checkpoints/{self.cfg.task.name}_{cfg.training.seed}"):
                #     os.makedirs(f"checkpoints/{self.cfg.task.name}_{cfg.training.seed}")
                # save_path = f"checkpoints/{self.cfg.task.name}_{cfg.training.seed}/{self.epoch + 1}.ckpt"

                # self.save_checkpoint(save_path)
                

            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            if WANDB and (not self.use_accelerate or self.accelerator.is_main_process):
                #每个 epoch 结束时记录 MoE 参数的详细统计
                if self.use_accelerate:
                    model_to_log = self.accelerator.unwrap_model(self.model)
                else:
                    model_to_log = self.model
                
                if hasattr(model_to_log, 'model') and hasattr(model_to_log.model, 'blocks'):
                    # 记录每个block的gate权重直方图和统计信息
                    for i, block in enumerate(model_to_log.model.blocks):
                        if hasattr(block, 'mlp') and hasattr(block.mlp, 'gate'):
                            gate_weights = block.mlp.gate.weight.detach().cpu().numpy()
                            step_log[f'moe_epoch/block_{i}/gate_weights_hist'] = wandb.Histogram(gate_weights)
                            step_log[f'moe_epoch/block_{i}/gate_weights_mean'] = float(gate_weights.mean())
                            step_log[f'moe_epoch/block_{i}/gate_weights_std'] = float(gate_weights.std())
                            step_log[f'moe_epoch/block_{i}/gate_weights_max'] = float(gate_weights.max())
                            step_log[f'moe_epoch/block_{i}/gate_weights_min'] = float(gate_weights.min())
                
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log
            
            # Comprehensive memory cleanup at end of epoch
            cleanup_memory()
            
            # Print memory usage every 10 epochs
            # if self.epoch % 10 == 0:
            #     print_memory_usage(f"[Epoch {self.epoch}] ")
    
    def get_policy_and_runner(self, cfg, usr_args, mode='latest'):
        """
        Load policy and runner from checkpoint.
        
        Args:
            cfg: Configuration object
            usr_args: User arguments
            mode: Checkpoint loading mode, can be:
                  - 'latest': Load latest.ckpt
                  - 'best': Load best checkpoint based on monitor_key
                  - epoch number (int or str): Load specific epoch checkpoint (e.g., 500 or '0500')
        """
        cfg = copy.deepcopy(self.cfg)
        output_dir = self.output_dir
        lastest_ckpt_path = self.get_checkpoint_path(tag=mode, monitor_key=cfg.checkpoint.topk.monitor_key)
        
        if lastest_ckpt_path.is_file():
            cprint(f"✅ Loading checkpoint: {lastest_ckpt_path}", 'magenta')
            cprint(f"   Mode: {mode}", 'cyan')
            self.load_checkpoint(path=lastest_ckpt_path) # rewrite self.output_dir
            # print ckpt info
            cprint(f"   Epoch: {self.epoch}, Global Step: {self.global_step}", 'green')
        else:
            cprint(f"❌ Checkpoint not found: {lastest_ckpt_path}", 'red')
            cprint(f"   Requested mode: {mode}", 'yellow')
            raise FileNotFoundError(
                f"Cannot find checkpoint: {lastest_ckpt_path}\n"
                f"Please check:\n"
                f"  1. Training has completed at least one checkpoint_every interval\n"
                f"  2. Checkpoint mode '{mode}' is valid (latest/best/epoch_number)\n"
                f"  3. For epoch number, ensure the checkpoint exists (e.g., 0500.ckpt for mode=500)"
            )
        
        n_obs_steps = cfg['n_obs_steps']
        n_action_steps = cfg['n_action_steps']

        env_runner = RobotRunner(
            output_dir=output_dir,
            n_obs_steps=n_obs_steps, 
            n_action_steps=n_action_steps)
        self._output_dir = output_dir # recover output_dir
        
        
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
    
        policy.eval()
        policy.cuda()
        return policy, env_runner, self.epoch

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        print('saved in ', path)
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
                    # 🔥 Unwrap model if using accelerate
                    if self.use_accelerate and key == 'model':
                        unwrapped_model = self.accelerator.unwrap_model(value)
                        if use_thread:
                            payload['state_dicts'][key] = _copy_to_cpu(unwrapped_model.state_dict())
                        else:
                            payload['state_dicts'][key] = unwrapped_model.state_dict()
                    else:
                        if use_thread:
                            payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                        else:
                            payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        # Wait for previous saving thread to complete
        if use_thread and self._saving_thread is not None:
            self._saving_thread.join()
        
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        cleanup_memory()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest', monitor_key='test_mean_score'):
        """
        获取checkpoint路径
        
        Args:
            tag: checkpoint标识，可以是：
                - 'latest': 返回 latest.ckpt
                - 'best': 根据monitor_key找到最佳checkpoint
                - 具体文件名（如 '0100.ckpt', 'latest.ckpt'）: 直接返回该文件路径
                - epoch数字（如 '100', '0100'）: 返回对应的 {epoch}.ckpt
            monitor_key: 当tag='best'时使用，用于确定最佳checkpoint的指标
        
        Returns:
            pathlib.Path: checkpoint文件路径
        """
        # 将tag转换为字符串，以支持Hydra传入的整数类型
        tag = str(tag)
        
        checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
        
        # 如果tag是'latest'，返回latest.ckpt
        if tag == 'latest':
            return checkpoint_dir.joinpath('latest.ckpt')
        
        # 如果tag是'best'，查找最佳checkpoint
        elif tag == 'best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10 if 'loss' not in monitor_key else float('inf')
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                try:
                    # Extract score for the specified monitor_key
                    score_str = ckpt.split(f'{monitor_key}=')[1].split('.ckpt')[0]
                    score = float(score_str)
                    
                    # Update best score based on whether we're minimizing or maximizing
                    if 'loss' in monitor_key:
                        if score < best_score:
                            best_ckpt = ckpt
                            best_score = score
                    else:
                        if score > best_score:
                            best_ckpt = ckpt
                            best_score = score
                except (IndexError, ValueError):
                    # Skip checkpoints that don't have the monitor_key
                    continue
            
            if best_ckpt is None:
                raise ValueError(f"No checkpoints found with monitor key: {monitor_key}")
            
            return checkpoint_dir.joinpath(best_ckpt)
        
        # 如果tag已经包含.ckpt后缀，直接作为文件名使用
        elif tag.endswith('.ckpt'):
            return checkpoint_dir.joinpath(tag)
        
        # 如果tag是纯数字或数字字符串（如'100', '0100'），添加.ckpt后缀
        else:
            try:
                # 尝试将tag解析为数字（支持带前导零的格式）
                epoch_num = int(tag)
                # 格式化为4位数字的文件名
                ckpt_filename = f'{epoch_num:04d}.ckpt'
                return checkpoint_dir.joinpath(ckpt_filename)
            except ValueError:
                # 如果不是数字，尝试直接作为文件名（自动添加.ckpt后缀）
                return checkpoint_dir.joinpath(f'{tag}.ckpt')
            

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
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config'))
)
def main(cfg):
    workspace = TrainManiFlowRoboTwinWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
