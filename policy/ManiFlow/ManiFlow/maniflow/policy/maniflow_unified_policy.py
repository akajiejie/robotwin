# Adapted from https://github.com/geyan21/ManiFlow_Policy/blob/main/ManiFlow/maniflow/policy/maniflow_pointcloud_policy.py
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
from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder
from maniflow.model.diffusion.unified_ditx import UnifiedDiTX
from maniflow.model.common.sample_util import *

class ManiFlowUnifiedTransformerPointcloudPolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_timestep_embed_dim=256,
            diffusion_stepsize_embed_dim=256,
            img_cond_len=1024,
            point_cond_len=1024,
            use_img=True,
            use_point_cloud=True,
            mask_img_inference=False,
            mask_point_cloud_inference=False,
            language_conditioned=False,
            num_modalities=3,  # For MMDiT_Generalized, set to 3 for image and point conditions, action is always the first modality
            img_cond_mask_ratio=0.1,
            point_cond_mask_ratio=0.1,
            lang_cond_mask_ratio=0.1,
            alternate_injection=False,
            use_meta_query=False,
            meta_query_mode="concat", # how to use meta-query, can be 'concat' or 'replace'
            num_queries=1,
            block_type="DiTX", # DiTX, HRDT
            n_layer=3,
            n_head=4,
            n_emb=256,
            qkv_bias=False,
            qk_norm=False,
            add_t_to_action_decoder=False, # whether to add t to action decoder
            image_encoder: TimmObsEncoder = None,
            encoder_type="DP3Encoder", # DP3Encoder, iDP3Encoder
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            downsample_points=False,
            flow_batch_ratio=0.75,
            consistency_batch_ratio=0.25,
            denoise_timesteps=10,
            sample_t_mode_flow="beta", 
            sample_t_mode_consistency="discrete",
            sample_dt_mode_consistency="uniform", 
            # parameters passed to step
            **kwargs):
        super().__init__()
        self.use_img = use_img
        self.use_point_cloud = use_point_cloud
        self.language_conditioned = language_conditioned
        self.img_cond_mask_ratio = img_cond_mask_ratio
        self.point_cond_mask_ratio = point_cond_mask_ratio
        self.lang_cond_mask_ratio = lang_cond_mask_ratio
        self.mask_img_inference = mask_img_inference
        self.mask_point_cloud_inference = mask_point_cloud_inference
        cprint(f"[ManiFlowUnifiedTransformerPointcloudPolicy] use_img: {self.use_img}", "red")
        cprint(f"[ManiFlowUnifiedTransformerPointcloudPolicy] use_point_cloud: {self.use_point_cloud}", "red")
        cprint(f"[ManiFlowUnifiedTransformerPointcloudPolicy] language_conditioned: {self.language_conditioned}", "red")
        cprint(f"[ManiFlowUnifiedTransformerPointcloudPolicy] img_cond_mask_ratio: {self.img_cond_mask_ratio}", "red")
        cprint(f"[ManiFlowUnifiedTransformerPointcloudPolicy] point_cond_mask_ratio: {self.point_cond_mask_ratio}", "red")
        cprint(f"[ManiFlowUnifiedTransformerPointcloudPolicy] lang_cond_mask_ratio: {self.lang_cond_mask_ratio}", "red")
        cprint(f"[ManiFlowUnifiedTransformerPointcloudPolicy] alternate_injection: {alternate_injection}", "red")
        cprint(f"[ManiFlowUnifiedTransformerPointcloudPolicy] mask_img_inference: {self.mask_img_inference}", "red")
        cprint(f"[ManiFlowUnifiedTransformerPointcloudPolicy] mask_point_cloud_inference: {self.mask_point_cloud_inference}", "red")

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

        self.encoder_type = encoder_type
        if encoder_type == "DP3Encoder":
            pointcloud_encoder = DP3Encoder(observation_space=obs_dict,
                                                    img_crop_shape=crop_shape,
                                                    out_channel=encoder_output_dim,
                                                    pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                    use_pc_color=use_pc_color,
                                                    pointnet_type=pointnet_type,
                                                    downsample_points=downsample_points,
                                                    )
        else:
            raise ValueError(f"Unsupported encoder type {encoder_type}")
        cprint(f"[Encoder_type] {encoder_type}", "yellow")
        # create diffusion model
        image_obs_feature_dim = image_encoder.output_shape()[-1]
        point_cloud_obs_feature_dim = pointcloud_encoder.output_shape()
        input_dim = action_dim
        img_cond_dim = image_obs_feature_dim
        point_cond_dim = point_cloud_obs_feature_dim

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")
        
        model = UnifiedDiTX(
            input_dim=input_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            img_cond_dim=img_cond_dim,
            point_cond_dim=point_cond_dim,
            img_cond_len=img_cond_len,
            point_cond_len=point_cond_len,
            num_modalities=num_modalities,
            alternate_injection=alternate_injection,
            use_meta_query=use_meta_query,
            meta_query_mode=meta_query_mode,
            num_queries=num_queries,
            diffusion_timestep_embed_dim=diffusion_timestep_embed_dim,
            diffusion_stepsize_embed_dim=diffusion_stepsize_embed_dim,
            block_type=block_type,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            add_t_to_action_decoder=add_t_to_action_decoder,
            language_conditioned=language_conditioned,
        )
        
        self.image_encoder = image_encoder
        self.pointcloud_encoder = pointcloud_encoder
        self.model = model
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.image_obs_feature_dim = image_obs_feature_dim
        self.point_cloud_obs_feature_dim = point_cloud_obs_feature_dim
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
        
        cprint(f"[ManiFlowTransformerPointcloudPolicy] Initialized with parameters:", "yellow")
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
            img_cond=None,
            point_cond=None,
            lang_cond=None,
            img_mask=None,
            point_mask=None,
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
            img_cond=img_cond,
            point_cond=point_cond,
            lang_cond=lang_cond,
            img_mask=img_mask,
            point_mask=point_mask,
           **kwargs)
        
        return ode_traj[-1] # sample ode returns the whole traj, return the last one
    


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        if self.use_point_cloud and 'point_cloud_mask' not in obs_dict:
            if self.mask_point_cloud_inference:
                # mask the point cloud in inference as zero
                obs_dict['point_cloud'] = torch.zeros_like(obs_dict['point_cloud'], device=self.device, dtype=obs_dict['point_cloud'].dtype)
                B = obs_dict['point_cloud'].shape[0]
                # create a mask for point cloud
                obs_dict['point_cloud_mask'] =  torch.zeros((B, 1), device=self.device, dtype=torch.bool)  # [B, 1]
            else:
                B = obs_dict['point_cloud'].shape[0]
                obs_dict['point_cloud_mask'] = torch.ones((B, 1), device=self.device, dtype=torch.bool)  # [B, 1]

        if self.use_img and 'head_cam_mask' not in obs_dict:
            # only support head_cam for now
            if self.mask_img_inference:
                # mask the image in inference as zero
                obs_dict['head_cam'] = torch.zeros_like(obs_dict['head_cam'], device=self.device, dtype=obs_dict['head_cam'].dtype)
                B = obs_dict['head_cam'].shape[0]
                # create a mask for image
                obs_dict['head_cam_mask'] = torch.zeros((B, 1), device=self.device, dtype=torch.bool)
            else:
                B = obs_dict['head_cam'].shape[0]
                # create a mask for image
                obs_dict['head_cam_mask'] = torch.ones((B, 1), device=self.device, dtype=torch.bool)
                obs_dict['head_cam_mask'] = obs_dict['head_cam_mask'].bool()
        
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color and self.use_point_cloud:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        img_cond = None
        point_cond = None
        lang_cond = None

        if self.language_conditioned:
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"
        
        if self.use_img:
            # image and robot state features
            this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].to(self.device))
            # this_nobs['head_cam'] = nobs['head_cam'][:,:self.n_obs_steps,...].to(self.device) # B, T, C, H, W
            image_features = self.image_encoder(this_nobs)
            img_cond = image_features.reshape(B, -1, self.image_obs_feature_dim) # B, T*L1, C

            base_img_mask = nobs['head_cam_mask'].to(self.device)  # [B, 1]
            img_mask = base_img_mask.expand(-1, img_cond.shape[1])  # [B, T*L1]
            img_mask = img_mask.bool()  # Ensure boolean type

        if self.use_point_cloud:
            # point cloud and robot state features
            this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]).to(self.device))
            # this_nobs['point_cloud'] = nobs['point_cloud'][:,:self.n_obs_steps,...].reshape(-1, *nobs['point_cloud'].shape[2:]).to(self.device) # B*T, N, C
            point_cloud_features = self.pointcloud_encoder(this_nobs)
            point_cond = point_cloud_features.reshape(B, -1, self.point_cloud_obs_feature_dim) # B, T*L2, C
            
            base_point_mask = nobs['point_cloud_mask'].to(self.device)  # [B, 1]
            point_mask = base_point_mask.expand(-1, point_cond.shape[1])  # [B, T*L2]
            point_mask = point_mask.bool()  # Ensure boolean type

        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            img_cond=img_cond,
            point_cond=point_cond,
            lang_cond=lang_cond,
            img_mask=img_mask if self.use_img else None,
            point_mask=point_mask if self.use_point_cloud else None,
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
            cprint(f"[ManiFlowTransformerPointcloudPolicy] Use different lr for obs_encoder: {obs_encoder_lr}", "yellow")
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

        img_cond = model_kwargs.get('img_cond', None)
        point_cond = model_kwargs.get('point_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        img_mask = model_kwargs.get('img_mask', None)
        point_mask = model_kwargs.get('point_mask', None)
        lang_mask = model_kwargs.get('lang_mask', None)
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
        target_dict['img_cond'] = img_cond
        target_dict['point_cond'] = point_cond
        target_dict['lang_cond'] = lang_cond
        target_dict['img_mask'] = img_mask
        target_dict['point_mask'] = point_mask
        target_dict['lang_mask'] = lang_mask

        return target_dict
    
    def get_consistency_targets(self, actions=None, **model_kwargs):
        info = {}
        target_dict = {}
        
        img_cond = model_kwargs.get('img_cond', None)
        point_cond = model_kwargs.get('point_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        img_mask = model_kwargs.get('img_mask', None)
        point_mask = model_kwargs.get('point_mask', None)
        lang_mask = model_kwargs.get('lang_mask', None)
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
                img_cond=img_cond[-consistency_batchsize:] if img_cond is not None else None,
                point_cond=point_cond[-consistency_batchsize:] if point_cond is not None else None,
                lang_cond=lang_cond[-consistency_batchsize:] if lang_cond is not None else None,
                img_mask=img_mask[-consistency_batchsize:] if img_mask is not None else None,
                point_mask=point_mask[-consistency_batchsize:] if point_mask is not None else None,
                lang_mask=lang_mask[-consistency_batchsize:] if lang_mask is not None else None,
            ) 
        pred_z1_ct = z_t_next_ct + (1 - t_next) * v_next
        v_ct = (pred_z1_ct - z_t_ct) / (1 - t_ct)
        v_next_pred_gt = pred_z1_ct - z0_ct

        # concatenate the targets
        sample_multiple_points = False
        if sample_multiple_points:
            z_t_final = torch.cat([z_t_ct, z_t_next_ct], dim=0)
            t_final = torch.cat([t_ct, t_next], dim=0)
            dt_final = torch.cat([dt_ct, dt_ct], dim=0)
            v_final = torch.cat([v_ct, v_next_pred_gt], dim=0)
            img_cond_final = torch.cat([img_cond, img_cond], dim=0) if img_cond is not None else None
            point_cond_final = torch.cat([point_cond, point_cond], dim=0) if point_cond is not None else None
            lang_cond_final = torch.cat([lang_cond, lang_cond], dim=0) if lang_cond is not None else None

            target_dict['z_t'] = z_t_final
            target_dict['t'] = t_final
            target_dict['dt'] = dt_final
            target_dict['v_target'] = v_final
            target_dict['img_cond'] = img_cond_final
            target_dict['point_cond'] = point_cond_final
            target_dict['lang_cond'] = lang_cond_final
            target_dict['img_mask'] = img_mask if img_mask is not None else None
            target_dict['point_mask'] = point_mask if point_mask is not None else None
            target_dict['lang_mask'] = lang_mask if lang_mask is not None else None
        else:
            # use only one point
            target_dict['z_t'] = z_t_ct
            target_dict['t'] = t_ct
            target_dict['dt'] = dt_ct
            target_dict['v_target'] = v_ct
            target_dict['img_cond'] = img_cond[-consistency_batchsize:] if img_cond is not None else None
            target_dict['point_cond'] = point_cond[-consistency_batchsize:] if point_cond is not None else None
            target_dict['lang_cond'] = lang_cond[-consistency_batchsize:] if lang_cond is not None else None
            target_dict['img_mask'] = img_mask[-consistency_batchsize:] if img_mask is not None else None
            target_dict['point_mask'] = point_mask[-consistency_batchsize:] if point_mask is not None else None
            target_dict['lang_mask'] = lang_mask[-consistency_batchsize:] if lang_mask is not None else None

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

        if not self.use_pc_color and self.use_point_cloud:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        img_cond = None
        point_cond = None
        lang_cond = None
        img_mask = None
        point_mask = None
        lang_mask = None
        trajectory = nactions
        cond_data = trajectory
        ema_model = train_state.get('ema_model', None)

        if self.language_conditioned:
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"
            lang_mask = None # lang_mask is not implemented yet, can be used for masking language condition
            

        # if has image:
        if self.use_img:
            # image and robot state features
            this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].to(self.device))
            # this_nobs['head_cam'] = nobs['head_cam'][:,:self.n_obs_steps,...].to(self.device) # B, T, C, H, W
            image_features = self.image_encoder(this_nobs)
            img_cond = image_features.reshape(batch_size, -1, self.image_obs_feature_dim) # B, T*L1, C
            # img_mask = torch.ones_like(img_cond[:, :, 0], dtype=torch.bool, device=self.device)  # B, T*L1
            # mask the image condition with self.cond_mask_ratio, True for valid, False for masked
            # img_mask = img_mask * torch.rand_like(img_mask, dtype=torch.float) > self.img_cond_mask_ratio
            base_img_mask = nobs['head_cam_mask'].to(self.device)  # [B, 1]
            img_mask = base_img_mask.expand(-1, img_cond.shape[1])  # [B, T*L1]
            img_mask = img_mask.bool()  # Ensure boolean type

        if self.use_point_cloud:
            # point cloud and robot state features
            this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]).to(self.device))
            # this_nobs['point_cloud'] = nobs['point_cloud'][:,:self.n_obs_steps,...].reshape(-1, *nobs['point_cloud'].shape[2:]).to(self.device) # B*T, N, C
            point_cloud_features = self.pointcloud_encoder(this_nobs)
            point_cond = point_cloud_features.reshape(batch_size, -1, self.point_cloud_obs_feature_dim) # B, T*L2, C
            # point_mask = torch.ones_like(point_cond[:, :, 0], dtype=torch.bool, device=self.device)  # B, T*L2
            # mask the point condition with self.cond_mask_ratio
            # point_mask = point_mask * torch.rand_like(point_mask, dtype=torch.float) > self.point_cond_mask_ratio

            base_point_mask = nobs['point_cloud_mask'].to(self.device)  # [B, 1]
            point_mask = base_point_mask.expand(-1, point_cond.shape[1])
            point_mask = point_mask.bool()

        """Get flow and consistency targets"""
        flow_batchsize = int(batch_size * self.flow_batch_ratio)
        consistency_batchsize = int(batch_size * self.consistency_batch_ratio)
    

        # Get flow targets
        flow_target_dict = self.get_flow_targets(nactions[:flow_batchsize], 
                                                    img_cond=img_cond[:flow_batchsize] if img_cond is not None else None,
                                                    point_cond=point_cond[:flow_batchsize] if point_cond is not None else None,
                                                    lang_cond=lang_cond[:flow_batchsize] if lang_cond is not None else None,
                                                    img_mask=img_mask[:flow_batchsize] if img_cond is not None else None,
                                                    point_mask=point_mask[:flow_batchsize] if point_cond is not None else None,
                                                    lang_mask=lang_mask[:flow_batchsize] if lang_cond is not None else None,)
        v_flow_pred = self.model(
            sample=flow_target_dict['z_t'], 
            timestep=flow_target_dict['t'].squeeze(),
            stepsize=flow_target_dict['dt'],
            img_cond=img_cond[:flow_batchsize] if img_cond is not None else None,
            point_cond=point_cond[:flow_batchsize] if point_cond is not None else None,
            lang_cond=flow_target_dict['lang_cond'][:flow_batchsize] if lang_cond is not None else None,
            img_mask=flow_target_dict['img_mask'][:flow_batchsize] if img_cond is not None else None,
            point_mask=flow_target_dict['point_mask'][:flow_batchsize] if point_cond is not None else None,
            lang_mask=flow_target_dict['lang_mask'][:flow_batchsize] if lang_cond is not None else None,
            )
        v_flow_Pred_magnitude = torch.sqrt(torch.mean(v_flow_pred ** 2)).item()

        # Get consistency targets
        consistency_target_dict = self.get_consistency_targets(nactions[flow_batchsize:flow_batchsize+consistency_batchsize],
                                                                        img_cond=img_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if img_cond is not None else None,
                                                                        point_cond=point_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if point_cond is not None else None,
                                                                        lang_cond=lang_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if lang_cond is not None else None,
                                                                        img_mask=img_mask[flow_batchsize:flow_batchsize+consistency_batchsize] if img_mask is not None else None,
                                                                        point_mask=point_mask[flow_batchsize:flow_batchsize+consistency_batchsize] if point_mask is not None else None,
                                                                        lang_mask=lang_mask[flow_batchsize:flow_batchsize+consistency_batchsize] if lang_mask is not None else None,
                                                                        ema_model=ema_model
                                                                        )
        v_ct_pred = self.model(
            sample=consistency_target_dict['z_t'], 
            timestep=consistency_target_dict['t'].squeeze(),
            stepsize=consistency_target_dict['dt'],
            img_cond=img_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if img_cond is not None else None,
            point_cond=point_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if point_cond is not None else None,
            lang_cond= consistency_target_dict['lang_cond'] if lang_cond is not None else None,
            img_mask=consistency_target_dict['img_mask'] if img_cond is not None else None,
            point_mask=consistency_target_dict['point_mask'] if point_cond is not None else None,
            lang_mask=consistency_target_dict['lang_mask'] if lang_cond is not None else None,
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
