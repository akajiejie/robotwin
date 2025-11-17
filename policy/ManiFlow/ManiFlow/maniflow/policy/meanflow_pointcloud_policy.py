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
from maniflow.model.vision_3d.pointnet_extractor import DP3Encoder
from maniflow.model.diffusion.ditx import DiTX
from maniflow.model.common.sample_util import *
from torch.autograd.functional import jvp
import torch.func

class MeanFlowLoss:
    """
    Mean Flows loss adapted for robotics manipulation tasks.
    Based on "Mean Flows for One-step Generative Modeling" paper.
    """
    def __init__(self, P_mean=-0.4, P_std=1.0, noise_dist='logit_normal', 
                 data_proportion=0.75, norm_p=1.0, norm_eps=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.data_proportion = data_proportion
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.noise_dist = noise_dist

    def _logit_normal_dist(self, shape, device):
        rnd_normal = torch.randn(shape, device=device)
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def _uniform_dist(self, shape, device):
        return torch.rand(shape, device=device)

    def noise_distribution(self, shape, device):
        if self.noise_dist == 'logit_normal':
            return self._logit_normal_dist(shape, device)
        elif self.noise_dist == 'uniform':
            return self._uniform_dist(shape, device)
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

    def __call__(self, model, actions, global_cond=None, lang_cond=None):
        """
        Compute Mean Flows loss for robotics actions.
        
        Args:
            model: The neural network model (DiTX)
            actions: Ground truth actions [B, T, Da]
            global_cond: Global conditioning [B, ...]
            lang_cond: Language conditioning
        """
        y = actions  # Ground truth actions
        device = y.device
        batch_size = y.shape[0]
        shape = (batch_size, 1, 1)  # Shape for robotics (B, 1, 1) instead of (B, 1, 1, 1) for images

        # Sample t and r from noise distribution
        t = self.noise_distribution(shape, device)
        r = self.noise_distribution(shape, device)
        t, r = torch.max(t, r), torch.min(t, r)

        # Apply data proportion - some samples have r = t (pure data)
        zero_mask = torch.arange(batch_size, device=device) < int(batch_size * self.data_proportion)
        zero_mask = zero_mask.view(shape)
        r = torch.where(zero_mask, t, r)

        # Create noise and corrupted action sequence
        n = torch.randn_like(y)
        z_t = (1 - t) * y + t * n
        v = n - y  # True velocity field

        v_g = v

        # Define model wrapper for JVP computation
        def u_wrapper(z, t_val, r_val):
            h_val = t_val - r_val
            return model(
                sample=z, 
                timestep=t_val.squeeze(),
                stepsize=h_val.squeeze(),
                global_cond=global_cond,
                lang_cond=lang_cond
            )

        # Compute model output and time derivative using JVP
        primals = (z_t, t, r)
        tangents = (v_g, torch.ones_like(t), torch.zeros_like(r))
        
        # u, du_dt = torch.func.jvp(u_wrapper, primals, tangents)
        u, du_dt = jvp(
            func=u_wrapper,
            inputs=primals,
            v=tangents,
            create_graph=True
        )

        # Compute target velocity with Mean Flows formulation
        h = torch.clamp(t - r, min=0.0, max=1.0)
        u_tgt = v_g - h * du_dt
        u_tgt = u_tgt.detach()

        # Adaptive loss weighting
        unweighted_loss = (u - u_tgt).pow(2).sum(dim=[1, 2])  # Sum over time and action dimensions
        
        with torch.no_grad():
            adaptive_weight = 1 / (unweighted_loss + self.norm_eps).pow(self.norm_p)

        loss = unweighted_loss * adaptive_weight
        return loss.mean()


class MeanFlowTransformerPointcloudPolicy(BasePolicy):
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
            encoder_type="DP3Encoder",
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            downsample_points=False,
            language_conditioned=False,
            # Mean Flows specific parameters
            meanflow_P_mean=-0.4,
            meanflow_P_std=1.0,
            meanflow_noise_dist='logit_normal',
            meanflow_data_proportion=0.75,
            meanflow_norm_p=1.0,
            meanflow_norm_eps=1.0,
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        self.encoder_type = encoder_type
        if encoder_type == "DP3Encoder":
            obs_encoder = DP3Encoder(observation_space=obs_dict,
                                   img_crop_shape=crop_shape,
                                   out_channel=encoder_output_dim,
                                   pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                   use_pc_color=use_pc_color,
                                   pointnet_type=pointnet_type,
                                   downsample_points=downsample_points)
        else:
            raise ValueError(f"Unsupported encoder type {encoder_type}")
        
        cprint(f"[Encoder_type] {encoder_type}", "yellow")
        
        # create model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[MeanFlowTransformerPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[MeanFlowTransformerPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")
        
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

        # Initialize Mean Flows loss
        self.meanflow_loss = MeanFlowLoss(
            P_mean=meanflow_P_mean,
            P_std=meanflow_P_std,
            noise_dist=meanflow_noise_dist,
            data_proportion=meanflow_data_proportion,
            norm_p=meanflow_norm_p,
            norm_eps=meanflow_norm_eps
        )
        
        cprint(f"[MeanFlowTransformerPointcloudPolicy] Initialized with Mean Flows:", "yellow")
        cprint(f"  - horizon: {self.horizon}", "yellow")
        cprint(f"  - n_action_steps: {self.n_action_steps}", "yellow")
        cprint(f"  - n_obs_steps: {self.n_obs_steps}", "yellow")
        cprint(f"  - num_inference_steps: {self.num_inference_steps}", "yellow")
        cprint(f"  - meanflow_noise_dist: {meanflow_noise_dist}", "yellow")
        cprint(f"  - meanflow_data_proportion: {meanflow_data_proportion}", "yellow")

        print_params(self)
        
    # ========= inference ============
    def conditional_sample(self, condition_data, global_cond=None, lang_cond=None, **kwargs):
        """
        Mean Flows 1-step generation (Algorithm 2 from paper).
        """
        noise = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=None)
        
        batch_size = noise.shape[0]
        
        # Mean Flows generation: x0 = z - f(z, t=1, h=1)
        t = torch.ones(batch_size, device=self.device)
        h = torch.ones(batch_size, device=self.device)
        
        velocity = self.model(
            sample=noise,
            timestep=t,
            stepsize=h,
            global_cond=global_cond,
            lang_cond=lang_cond,
            **kwargs
        )
        
        x0 = noise - velocity
        return x0

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
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
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]).to(device))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(B, -1, Do)
        
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # run sampling (1-step generation with Mean Flows)
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
        
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training ============
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
            cprint(f"[MeanFlowTransformerPointcloudPolicy] Using different LR for obs_encoder: {obs_encoder_lr}", "yellow")
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

    def compute_loss(self, batch, train_state=None, **kwargs):
        """
        Compute Mean Flows loss instead of flow + consistency losses.
        """
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action']).to(self.device)

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]

        # handle conditioning
        lang_cond = None
        if self.language_conditioned:
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        # encode observations
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]).to(self.device))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1, self.obs_feature_dim)

        # Compute Mean Flows loss
        loss = self.meanflow_loss(
            model=self.model,
            actions=nactions,
            global_cond=global_cond,
            lang_cond=lang_cond
        )

        # Additional metrics for monitoring
        with torch.no_grad():
            # Sample some predictions for monitoring
            t_sample = torch.rand(batch_size, 1, 1, device=self.device)
            noise = torch.randn_like(nactions)
            z_t = (1 - t_sample) * nactions + t_sample * noise
            
            pred_v = self.model(
                sample=z_t,
                timestep=t_sample.squeeze(),
                stepsize=torch.zeros(batch_size, device=self.device),
                global_cond=global_cond,
                lang_cond=lang_cond
            )
            pred_magnitude = torch.sqrt(torch.mean(pred_v ** 2)).item()

        loss_dict = {
            'meanflow_loss': loss.item(),
            'pred_magnitude': pred_magnitude,
            'bc_loss': loss.item(),  # Keep for compatibility
        }

        return loss, loss_dict