from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops
import sys
from pathlib import Path

from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.policy.base_policy import BasePolicy
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.model_util import print_params
from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder
from maniflow.model.diffusion.ditx import DiTX
from maniflow.model.common.sample_util import *

class ManiFlowTransformerImagePolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_timestep_embed_dim=256,
            diffusion_stepsize_embed_dim=256,
            visual_cond_len=1024,
            n_layer=3,
            n_head=4,
            n_emb=256,
            qkv_bias=False,
            qk_norm=False,
            obs_encoder: TimmObsEncoder = None,
            language_conditioned=False,
            flow_batch_ratio=0.75,
            consistency_batch_ratio=0.25,
            denoise_timesteps=10,
            sample_t_mode_flow="beta", 
            sample_t_mode_consistency="discrete",
            sample_dt_mode_consistency="uniform", 
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])
        
        # create model
        obs_feature_dim = obs_encoder.output_shape()[-1]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim

     
        model = DiTX(
            input_dim=input_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=global_cond_dim,
            visual_cond_len=visual_cond_len,
            diffusion_timestep_embed_dim=diffusion_timestep_embed_dim,
            diffusion_stepsize_embed_dim=diffusion_stepsize_embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            language_conditioned=language_conditioned,
        )
        
        self.obs_encoder = obs_encoder
        self.model = model
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.language_conditioned = language_conditioned
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps
        self.flow_batch_ratio = flow_batch_ratio
        self.consistency_batch_ratio = consistency_batch_ratio
        assert flow_batch_ratio + consistency_batch_ratio == 1.0, "Sum of batch ratios should be equal to 1.0"
        self.denoise_timesteps = denoise_timesteps
        self.sample_t_mode_flow = sample_t_mode_flow
        self.sample_t_mode_consistency = sample_t_mode_consistency
        self.sample_dt_mode_consistency = sample_dt_mode_consistency
        
        cprint(f"[ManiFlowTransformerImagePolicy] Initialized with parameters:", "yellow")
        cprint(f"  - horizon: {self.horizon}", "yellow")
        cprint(f"  - n_action_steps: {self.n_action_steps}", "yellow")
        cprint(f"  - n_obs_steps: {self.n_obs_steps}", "yellow")
        cprint(f"  - num_inference_steps: {self.num_inference_steps}", "yellow")
        cprint(f"  - flow_batch_ratio: {self.flow_batch_ratio}", "yellow")
        cprint(f"  - consistency_batch_ratio: {self.consistency_batch_ratio}", "yellow")
        cprint(f"  - denoise_timesteps: {self.denoise_timesteps}", "yellow")
        cprint(f"  - sample_t_mode_flow: {self.sample_t_mode_flow}", "yellow")
        cprint(f"  - sample_t_mode_consistency: {self.sample_t_mode_consistency}", "yellow")
        cprint(f"  - sample_dt_mode_consistency: {self.sample_dt_mode_consistency}", "yellow")

        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, 
            global_cond=None,
            lang_cond=None,
            **kwargs
            ):
        
        noise = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=None)
        
        ode_traj = self.sample_ode(
            z0 = noise, 
            N = self.num_inference_steps,
            global_cond=global_cond,
            lang_cond=lang_cond,
           **kwargs)
        
        return ode_traj[-1] # sample ode returns the whole traj, return the last one
    


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        global_cond = None
        lang_cond = None

        if self.language_conditioned:
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"
        
        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].to(device))
        nobs_features = self.obs_encoder(this_nobs).to(device) 
        global_cond = nobs_features.reshape(B, -1, Do) # B, self.n_obs_steps*L, Do
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            global_cond=global_cond,
            lang_cond=lang_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float = None,
            obs_encoder_weight_decay: float = None,
            betas: Tuple[float, float] = (0.9, 0.95)
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=weight_decay)
        
        backbone_params = list()
        other_obs_params = list()
        if obs_encoder_lr is not None:
            cprint(f"[ManiFlowTransformerImagePolicy] Use different lr for obs_encoder: {obs_encoder_lr}", "yellow")
            for key, value in self.obs_encoder.named_parameters():
                if key.startswith('key_model_map'):
                    backbone_params.append(value)
                else:
                    other_obs_params.append(value)
            optim_groups.append({
                "params": backbone_params,
                "weight_decay": obs_encoder_weight_decay,
                "lr": obs_encoder_lr # for fine tuning
            })
            optim_groups.append({
                "params": other_obs_params,
                "weight_decay": obs_encoder_weight_decay
            })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas
        )
        return optimizer
    
    def sample_t(self, batch_size, mode="uniform"):
        """
        Sample t for flow matching or consistency training.
        """
        if mode == "uniform":
            t = torch.rand((batch_size,), device=self.device)
        elif mode == "lognorm":
            t = sample_logit_normal(batch_size, m=self.lognorm_m, s=self.lognorm_s, device=self.device)
        elif mode == "mode":
            t = sample_mode(batch_size, s=self.mode_s, device=self.device)
        elif mode == "cosmap":
            t = sample_cosmap(batch_size, device=self.device)
        elif mode == "beta":
            t = sample_beta(batch_size, device=self.device)
        elif mode == "discrete":
            t = torch.randint(low=0, high=self.denoise_timesteps, size=(batch_size,)).float()
            t = t / self.denoise_timesteps
        else:
            raise ValueError(f" Unsupported sample_t_mode {mode}. Choose from 'uniform', 'lognorm', 'mode', 'cosmap', 'beta', 'discrete'.")
        return t

    def sample_dt(self, batch_size, sample_dt_mode="uniform"):
        """
        Sample dt for consistency training.
        """
        if sample_dt_mode == "uniform":
            dt = torch.rand((batch_size,), device=self.device)
        else:
            raise ValueError(f"Unsupported sample_dt_mode {sample_dt_mode}")
        
        return dt
    
    def get_flow_targets(self, actions=None, **model_kwargs):
        info = {}
        target_dict = {}
        
        global_cond = model_kwargs.get('global_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        flow_batchsize = actions.shape[0]
        device = actions.device
        
        t_flow = self.sample_t(flow_batchsize, mode=self.sample_t_mode_flow).to(device)
        t_flow = t_flow.view(-1, 1, 1)
        dt_flow = torch.zeros((flow_batchsize,), device=device)
        z_0_flow = torch.randn_like(actions, device=device) 
        z_1_flow = actions.to(device) 
        z_t_flow = (1.- t_flow) * z_0_flow + t_flow * z_1_flow
        v_t_flow = z_1_flow - z_0_flow

        target_dict['z_t'] = z_t_flow
        target_dict['t'] = t_flow
        target_dict['dt'] = dt_flow
        target_dict['v_target'] = v_t_flow
        target_dict['global_cond'] = global_cond
        target_dict['lang_cond'] = lang_cond

        return target_dict
    
    def get_consistency_targets(self, actions=None, **model_kwargs):
        info = {}
        target_dict = {}
        
        global_cond = model_kwargs.get('global_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        ema_model = model_kwargs.get('ema_model', None)
        consistency_batchsize = actions.shape[0]
        device = actions.device


        t_ct = self.sample_t(consistency_batchsize, mode=self.sample_t_mode_consistency).to(device)
        t_ct = t_ct.view(-1, 1, 1)
        dt_ct = self.sample_dt(consistency_batchsize, sample_dt_mode=self.sample_dt_mode_consistency).to(device)

        t_next = t_ct.squeeze() + dt_ct
        t_next = torch.clamp(t_next, max=1.0)
        t_next = t_next.view(-1, 1, 1)

        z0_ct = torch.randn_like(actions, device=device) 
        z1_ct = actions.to(device) 
        z_t_ct = (1. - t_ct) * z0_ct + t_ct * z1_ct
        z_t_next_ct = (1. - t_next) * z0_ct + t_next * z1_ct

        with torch.no_grad():
            v_next = ema_model.model(
                sample=z_t_next_ct, 
                timestep=t_next.squeeze(),
                stepsize=dt_ct, 
                global_cond=global_cond[-consistency_batchsize:],
                lang_cond=lang_cond[-consistency_batchsize:] if lang_cond is not None else None,
            ) 
        pred_z1_ct = z_t_next_ct + (1 - t_next) * v_next
        v_ct = (pred_z1_ct - z_t_ct) / (1 - t_ct)
        
        target_dict['z_t'] = z_t_ct
        target_dict['t'] = t_ct
        target_dict['dt'] = dt_ct
        target_dict['v_target'] = v_ct

        return target_dict
    
    @torch.no_grad()
    def sample_ode(self, z0=None, N=None, **model_kwargs):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.num_inference_steps
        dt = 1./N
        traj = [] # to store the trajectory
        z = z0.detach().clone()
        batchsize = z.shape[0]
        
        t = torch.arange(0, N, device=z0.device, dtype=z0.dtype) / N 
        traj.append(z.detach().clone())

        for i in range(N):
            ti = torch.ones((batchsize,), device=self.device) * t[i]
            pred = self.model(z, ti, stepsize=dt, **model_kwargs)
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        return traj

    def compute_loss(self, batch, train_state=None, **kwargs):
        # normalize input
        # normalization will set data to cpu!
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action']).to(self.device)

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        lang_cond = None
        ema_model = train_state.get('ema_model', None)

        if self.language_conditioned:
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].to(self.device))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1, self.obs_feature_dim)
        
        """Get flow and consistency targets"""
        flow_batchsize = int(batch_size * self.flow_batch_ratio)
        consistency_batchsize = int(batch_size * self.consistency_batch_ratio)
    

        # Get flow targets
        flow_target_dict = self.get_flow_targets(nactions[:flow_batchsize], 
                                                    global_cond=global_cond[:flow_batchsize],
                                                    lang_cond=lang_cond[:flow_batchsize] if lang_cond is not None else None)
        v_flow_pred = self.model(
            sample=flow_target_dict['z_t'], 
            timestep=flow_target_dict['t'].squeeze(),
            stepsize=flow_target_dict['dt'],
            global_cond=global_cond[:flow_batchsize],
            lang_cond=flow_target_dict['lang_cond'][:flow_batchsize] if lang_cond is not None else None)
        v_flow_Pred_magnitude = torch.sqrt(torch.mean(v_flow_pred ** 2)).item()

        # Get consistency targets
        consistency_target_dict = self.get_consistency_targets(nactions[flow_batchsize:flow_batchsize+consistency_batchsize],
                                                                        global_cond=global_cond[flow_batchsize:flow_batchsize+consistency_batchsize],
                                                                        lang_cond=lang_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if lang_cond is not None else None,
                                                                        ema_model=ema_model
                                                                        )
        v_ct_pred = self.model(
            sample=consistency_target_dict['z_t'], 
            timestep=consistency_target_dict['t'].squeeze(),
            stepsize=consistency_target_dict['dt'],
            global_cond=global_cond[flow_batchsize:flow_batchsize+consistency_batchsize],
            lang_cond=lang_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if lang_cond is not None else None,
            )
        v_ct_pred_magnitude = torch.sqrt(torch.mean(v_ct_pred ** 2)).item()

        """Compute losses"""
        loss = 0.

        # compute flow loss 
        v_flow_target = flow_target_dict['v_target']
        loss_flow = F.mse_loss(v_flow_pred, v_flow_target, reduction='none')
        loss_flow = reduce(loss_flow, 'b ... -> b (...)', 'mean')
        loss += loss_flow.mean()
        loss_flow = loss_flow.mean().item()

        # compute consistency training loss
        v_ct_target = consistency_target_dict['v_target']
        loss_ct = F.mse_loss(v_ct_pred, v_ct_target, reduction='none')
        loss_ct = reduce(loss_ct, 'b ... -> b (...)', 'mean')
        loss += loss_ct.mean()
        loss_ct = loss_ct.mean().item()  

        loss = loss.mean()
        loss_dict = {
                'loss_flow': loss_flow,
                'loss_ct': loss_ct,
                'v_flow_Pred_magnitude': v_flow_Pred_magnitude,
                'v_ct_pred_magnitude': v_ct_pred_magnitude,
                'bc_loss': loss.item(),
        }
        

        return loss, loss_dict
