from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint

from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.policy.base_policy import BasePolicy
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.model_util import print_params
from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder
from maniflow.model.diffusion.ditx import DiTX
from maniflow.model.common.sample_util import *

# Optional import for Gate-Attention support
try:
    from maniflow.model.diffusion.ditx_gateattn import DiTXGateAttn
    GATE_ATTN_AVAILABLE = True
except ImportError:
    GATE_ATTN_AVAILABLE = False
    DiTXGateAttn = None
    cprint("[Warning] DiTXGateAttn not available, falling back to DiTX", "yellow")



class ManiFlowTransformerImagePolicy(BasePolicy):
    def __init__(self, 
             shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_timestep_embed_dim=256,
            diffusion_target_t_embed_dim=256,
            visual_cond_len=1024,
            n_layer=3,
            n_head=4,
            n_emb=256,
            qkv_bias=False,
            qk_norm=False,
            # Gate-Attention configuration
            use_gate_attn=False,  # 是否使用Gate-Attention（DiTXGateAttn）
            gate_type='elementwise',  # Gate-Attention类型: 'none', 'headwise', 'elementwise'
            block_type='DiTX',  # 保留此参数以保持向后兼容
            obs_encoder: TimmObsEncoder = None,
            language_conditioned=False,
            # consistency flow training parameters
            flow_batch_ratio=0.75,
            consistency_batch_ratio=0.25,
            denoise_timesteps=10,
            sample_t_mode_flow="beta", 
            sample_t_mode_consistency="discrete",
            sample_dt_mode_consistency="uniform", 
            sample_target_t_mode="relative", # relative, absolute
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
        
        # create ManiFlow model
        obs_feature_dim = obs_encoder.output_shape()[-1]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim
        
        # 根据use_gate_attn选择模型类型
        if use_gate_attn:
            if not GATE_ATTN_AVAILABLE:
                raise ImportError(
                    "DiTXGateAttn is not available. Please check if ditx_gateattn.py is properly installed."
                )
            cprint(f"[ManiFlowTransformerImagePolicy] 使用DiTXGateAttn模型 (gate_type={gate_type})", "cyan")
            model = DiTXGateAttn(
                input_dim=input_dim,
                output_dim=action_dim,
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                cond_dim=global_cond_dim,
                visual_cond_len=visual_cond_len,
                diffusion_timestep_embed_dim=diffusion_timestep_embed_dim,
                diffusion_target_t_embed_dim=diffusion_target_t_embed_dim,
                n_layer=n_layer,
                n_head=n_head,
                n_emb=n_emb,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                gate_type=gate_type,  # Gate-Attention配置
                language_conditioned=language_conditioned,
            )
        else:
            cprint(f"[ManiFlowTransformerImagePolicy] 使用DiTX模型 (标准版)", "cyan")
            model = DiTX(
                input_dim=input_dim,
                output_dim=action_dim,
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                cond_dim=global_cond_dim,
                visual_cond_len=visual_cond_len,
                diffusion_timestep_embed_dim=diffusion_timestep_embed_dim,
                diffusion_target_t_embed_dim=diffusion_target_t_embed_dim,
                n_layer=n_layer,
                n_head=n_head,
                n_emb=n_emb,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                language_conditioned=language_conditioned,
            )
        
        self.obs_encoder = obs_encoder
        self.model = model
        self.use_gate_attn = use_gate_attn
        self.gate_type = gate_type
        
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
        self.sample_target_t_mode = sample_target_t_mode
        assert self.sample_target_t_mode in ["absolute", "relative"], "sample_target_t_mode must be either 'absolute' or 'relative'"
        
        cprint(f"[ManiFlowTransformerImagePolicy] Initialized with parameters:", "yellow")
        cprint(f"  - model_type: {'DiTXGateAttn' if self.use_gate_attn else 'DiTX'}", "yellow")
        if self.use_gate_attn:
            cprint(f"  - gate_type: {self.gate_type}", "yellow")
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
        cprint(f"  - sample_target_t_mode: {self.sample_target_t_mode}", "yellow")
        
        # 🔥 运行时验证encoder和model的维度对齐
        self._validate_dimensions()

        print_params(self)
    
    def _validate_dimensions(self):
        """运行时验证encoder输出和model输入的维度对齐"""
        # 检查encoder是否支持token序列输出
        if hasattr(self.obs_encoder, 'output_token_sequence'):
            token_seq_mode = self.obs_encoder.output_token_sequence
            cprint(f"[Validation] Encoder token_sequence mode: {token_seq_mode}", "cyan")
            
            # 检查RGB特征维度
            if hasattr(self.obs_encoder, 'rgb_feature_dim'):
                rgb_dim = self.obs_encoder.rgb_feature_dim
                cprint(f"[Validation] RGB feature dim: {rgb_dim}", "cyan")
                
                # 检查model的cond_dim是否匹配
                if hasattr(self.model, 'vis_cond_obs_emb'):
                    model_cond_dim = self.model.vis_cond_obs_emb.in_features
                    if rgb_dim != model_cond_dim:
                        cprint(f"[WARNING] Dimension mismatch! RGB dim ({rgb_dim}) != Model cond_dim ({model_cond_dim})", "red")
                    else:
                        cprint(f"[Validation] ✓ Dimensions aligned: {rgb_dim}", "green")
            
            # 检查触觉特征维度
            if hasattr(self.obs_encoder, 'tactile_feature_dim') and self.obs_encoder.tactile_feature_dim is not None:
                tactile_dim = self.obs_encoder.tactile_feature_dim
                cprint(f"[Validation] Tactile feature dim: {tactile_dim}", "cyan")
                
                if hasattr(self.obs_encoder, 'rgb_feature_dim'):
                    if tactile_dim != self.obs_encoder.rgb_feature_dim:
                        cprint(f"[WARNING] Tactile dim ({tactile_dim}) != RGB dim ({self.obs_encoder.rgb_feature_dim})", "yellow")
                        cprint(f"           Projection layers should be used for alignment", "yellow")
            
            # 检查位置编码长度
            if token_seq_mode and hasattr(self.model, 'vis_cond_pos_embed'):
                pos_embed_len = self.model.vis_cond_pos_embed.shape[1]
                expected_len = self.model.visual_cond_len * self.n_obs_steps
                if pos_embed_len == expected_len:
                    cprint(f"[Validation] ✓ Position embedding length: {pos_embed_len} (visual_cond_len={self.model.visual_cond_len} × n_obs_steps={self.n_obs_steps})", "green")
                else:
                    cprint(f"[WARNING] Position embedding mismatch! {pos_embed_len} != {expected_len}", "red")
    
    # ========= Attention Recording ============
    def set_record_attn(self, record: bool):
        """Enable/disable attention weight recording in DiTX model"""
        if hasattr(self.model, 'set_record_attn'):
            self.model.set_record_attn(record)
    
    def get_attn_stats(self, modality_info: dict = None):
        """
        Get attention statistics from DiTX model
        
        Args:
            modality_info: 模态信息字典，来自encoder.get_modality_info()
        """
        if hasattr(self.model, 'get_attn_stats'):
            return self.model.get_attn_stats(modality_info=modality_info)
        return None
    
    def get_gate_stats(self):
        """
        Get Gate-Attention statistics from DiTXGateAttn model
        
        Returns:
            dict: Global gate statistics including:
                - gate/mean_activation: Average gate activation across all layers
                - gate/saturation_high_ratio: Ratio of highly saturated gates (>0.9)
                - gate/saturation_low_ratio: Ratio of low saturated gates (<0.1)
                - gate/layer_variance: Variance of gate activations across layers
                - gate/modality_{modality}_mean: Per-modality gate values
                - gate/early_vs_late_diff: Difference between early and late layers
        """
        if hasattr(self.model, 'get_gate_stats'):
            return self.model.get_gate_stats()
        return None
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, 
            vis_cond=None,
            lang_cond=None,
            **kwargs
            ):
        
        noise = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=None)
        
        ode_traj = self.sample_ode(
            x0 = noise, 
            N = self.num_inference_steps,
            vis_cond=vis_cond,
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
        vis_cond = None
        lang_cond = None

        if self.language_conditioned:
            # assume nobs has 'task_name' key for language condition
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        # condition through visual feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].to(device))
        nobs_features = self.obs_encoder(this_nobs).to(device)
        
        # 获取模态信息（用于gate bias和位置编码初始化）
        modality_info = None
        head_grid_size = None
        if hasattr(self.obs_encoder, 'get_modality_info'):
            modality_info = self.obs_encoder.get_modality_info()
        if hasattr(self.obs_encoder, 'head_grid_size'):
            head_grid_size = self.obs_encoder.head_grid_size
        
        # 支持token序列输出模式
        if hasattr(self.obs_encoder, 'output_token_sequence') and self.obs_encoder.output_token_sequence:
            # Token序列模式: (B, L_tokens, D)
            vis_cond = nobs_features  # 已经是正确格式
        else:
            # 向量模式: 拼接向量reshape
            vis_cond = nobs_features.reshape(B, -1, Do)  # B, self.n_obs_steps*L, Do
        
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # run sampling
        kwargs_with_modality = {**self.kwargs}
        if modality_info is not None:
            kwargs_with_modality['modality_info'] = modality_info
        if head_grid_size is not None:
            kwargs_with_modality['head_grid_size'] = head_grid_size
        
        nsample = self.conditional_sample(
            cond_data, 
            vis_cond=vis_cond,
            lang_cond=lang_cond,
            **kwargs_with_modality)
        
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
    
    def linear_interpolate(self, noise, target, timestep, epsilon=0.0):
        """
        Linear interpolation between noise and target data with optional noise preservation.
        
        Args:
            noise (Tensor): Initial noise at t=0
            target (Tensor): Target data point at t=1  
            timestep (float): Interpolation parameter in [0, 1]
                            t=0 returns pure noise, t=1 returns target + epsilon*noise
            epsilon (float): Noise preservation factor. Controls minimum noise retained.
                            Default 0.0 for standard linear interpolation.
                            
        Returns:
            Tensor: Interpolated data point at given timestep
            
        Examples:
            >>> # Standard linear interpolation (epsilon=0)
            >>> result = linear_interpolate(noise, data, 0.5)  # 50% noise + 50% data
            
            >>> # With noise preservation (epsilon=0.01) 
            >>> result = linear_interpolate(noise, data, 1.0, epsilon=0.01)  # data + 1% noise
        """
        # Calculate noise coefficient with epsilon adjustment
        noise_coeff = 1.0 - (1.0 - epsilon) * timestep
        
        # Linear combination: preserved_noise + scaled_target
        interpolated_data_point = noise_coeff * noise + timestep * target
        
        return interpolated_data_point

    
    def get_flow_velocity(self, actions, **model_kwargs):
        """
        Get flow velocity targets for training.
        Flow training is used to train the model to predict instantaneous velocity given a timestep.
        """
        target_dict = {}
        
        # get visual and language conditions
        vis_cond = model_kwargs.get('vis_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        flow_batchsize = actions.shape[0]
        device = actions.device
        
        # sample t and dt for flow
        # dt is zero for flow, as we aim to predict the instantaneous velocity at t
        t_flow = self.sample_t(flow_batchsize, mode=self.sample_t_mode_flow).to(device)
        t_flow = t_flow.view(-1, 1, 1)
        dt_flow = torch.zeros((flow_batchsize,), device=device)
        
        # get target timestep
        # target_t_flow is the target timestep for the flow step
        # it can be either absolute or relative to t_flow
        # if absolute, it is t_flow + dt_flow
        # if relative, it is just dt_flow
        if self.sample_target_t_mode == "absolute":
            target_t_flow = t_flow.squeeze() + dt_flow
        elif self.sample_target_t_mode == "relative":
            target_t_flow = dt_flow
        
        # compute interpolated data points at t and predict flow velocity
        x_0_flow = torch.randn_like(actions, device=device) 
        x_1_flow = actions.to(device) 
        x_t_flow = self.linear_interpolate(x_0_flow, x_1_flow, t_flow, epsilon=0.0)
        v_t_flow = x_1_flow - x_0_flow

        target_dict['x_t'] = x_t_flow
        target_dict['t'] = t_flow
        target_dict['target_t'] = target_t_flow
        target_dict['v_target'] = v_t_flow
        target_dict['vis_cond'] = vis_cond
        target_dict['lang_cond'] = lang_cond

        return target_dict
    
    def get_consistency_velocity(self, actions, **model_kwargs):
        """
        Get consistency velocity targets for training.
        Consistency training is used to train the model to be consistent across different timesteps.
        """
        target_dict = {}
        
        # get visual and language conditions
        vis_cond = model_kwargs.get('vis_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        ema_model = model_kwargs.get('ema_model', None)
        modality_info = model_kwargs.get('modality_info', None)
        consistency_batchsize = actions.shape[0]
        device = actions.device

        # sample t and dt for consistency training
        t_ct = self.sample_t(consistency_batchsize, mode=self.sample_t_mode_consistency).to(device)
        t_ct = t_ct.view(-1, 1, 1)
        delta_t1 = self.sample_dt(consistency_batchsize, sample_dt_mode=self.sample_dt_mode_consistency).to(device)
        # delta_t2 = self.sample_dt(consistency_batchsize, sample_dt_mode=self.sample_dt_mode_consistency).to(device)
        delta_t2 = delta_t1.clone() # use the same delta_t or resample a new one

        # compute next timestep
        t_next = t_ct.squeeze() + delta_t1
        t_next = torch.clamp(t_next, max=1.0) # clip t to ensure it does not exceed 1.0
        t_next = t_next.view(-1, 1, 1)
        
        # compute target timestep
        # target_t_next is the target timestep for the next step
        # it can be either absolute or relative to t_next
        # if absolute, it is t_next + delta_t2
        # if relative, it is just delta_t2
        if self.sample_target_t_mode == "absolute":
            target_t_next = t_next.squeeze() + delta_t2
        elif self.sample_target_t_mode == "relative":
            target_t_next = delta_t2

        # compute interpolated data points at timestep t and t_next
        x0_ct = torch.randn_like(actions, device=device) 
        x1_ct = actions.to(device) 
        x_t_ct = self.linear_interpolate(x0_ct, x1_ct, t_ct, epsilon=0.0)
        x_t_next = self.linear_interpolate(x0_ct, x1_ct, t_next, epsilon=0.0)

        # predict the average velocity from t_next toward next target (t_next + delta_t2)
        with torch.no_grad():
            v_avg_to_next_target = ema_model.model(
                sample=x_t_next, 
                timestep=t_next.squeeze(),
                target_t=target_t_next.squeeze(), 
                vis_cond=vis_cond[-consistency_batchsize:],
                lang_cond=lang_cond[-consistency_batchsize:] if lang_cond is not None else None,
                modality_info=modality_info,
                head_grid_size=model_kwargs.get('head_grid_size', None),
            ) 
        # predict the target data point using the average velocity
        pred_x1_ct = x_t_next + (1 - t_next) * v_avg_to_next_target
        # estimate the velocity at t by using the predicted endpoint
        v_ct = (pred_x1_ct - x_t_ct) / (1 - t_ct)

        # target_t_ct is the target timestep for the current timestep t
        target_t_ct = delta_t1 if self.sample_target_t_mode == "relative" else t_next.squeeze()
        
        target_dict['x_t'] = x_t_ct
        target_dict['t'] = t_ct
        target_dict['target_t'] = target_t_ct
        target_dict['v_target'] = v_ct

        return target_dict
    
    @torch.no_grad()
    def sample_ode(self, x0=None, N=None, **model_kwargs):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.num_inference_steps
        dt = 1./N
        traj = [] # to store the trajectory
        x = x0.detach().clone()
        batchsize = x.shape[0]

        t = torch.arange(0, N, device=x0.device, dtype=x0.dtype) / N
        traj.append(x.detach().clone())

        for i in range(N):
            ti = torch.ones((batchsize,), device=self.device) * t[i]
            if self.sample_target_t_mode == "absolute":
                target_t = ti + dt
            elif self.sample_target_t_mode == "relative":
                target_t = dt
            pred = self.model(x, ti, target_t=target_t, **model_kwargs)
            x = x.detach().clone() + pred * dt
            traj.append(x.detach().clone())

        return traj

    def compute_loss(self, batch, ema_model=None, **kwargs):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action']).to(self.device)

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        vis_cond = None
        trajectory = nactions
        cond_data = trajectory
        lang_cond = None
        ema_model = ema_model

        if self.language_conditioned:
            # we assume language condition is passed as 'task_name'
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].to(self.device))
        nobs_features = self.obs_encoder(this_nobs)
        
        # 获取模态信息（用于gate bias和位置编码初始化）
        modality_info = None
        head_grid_size = None
        if hasattr(self.obs_encoder, 'get_modality_info'):
            modality_info = self.obs_encoder.get_modality_info()
        if hasattr(self.obs_encoder, 'head_grid_size'):
            head_grid_size = self.obs_encoder.head_grid_size
        
        # 支持token序列输出模式
        if hasattr(self.obs_encoder, 'output_token_sequence') and self.obs_encoder.output_token_sequence:
            vis_cond = nobs_features  # 已经是 (B, L_tokens, D) 格式
        else:
            vis_cond = nobs_features.reshape(batch_size, -1, self.obs_feature_dim)
        
        """Get flow and consistency targets"""
        flow_batchsize = int(batch_size * self.flow_batch_ratio)
        consistency_batchsize = int(batch_size * self.consistency_batch_ratio)
    

        # Get flow targets
        flow_target_dict = self.get_flow_velocity(nactions[:flow_batchsize], 
                                                    vis_cond=vis_cond[:flow_batchsize],
                                                    lang_cond=lang_cond[:flow_batchsize] if lang_cond is not None else None)
        v_flow_pred = self.model(
            sample=flow_target_dict['x_t'], 
            timestep=flow_target_dict['t'].squeeze(),
            target_t=flow_target_dict['target_t'].squeeze(),
            vis_cond=vis_cond[:flow_batchsize],
            lang_cond=flow_target_dict['lang_cond'][:flow_batchsize] if lang_cond is not None else None,
            modality_info=modality_info,
            head_grid_size=head_grid_size)
        v_flow_pred_magnitude = torch.sqrt(torch.mean(v_flow_pred ** 2)).item()

        # Get consistency targets
        consistency_target_dict = self.get_consistency_velocity(nactions[flow_batchsize:flow_batchsize+consistency_batchsize],
                                                                        vis_cond=vis_cond[flow_batchsize:flow_batchsize+consistency_batchsize],
                                                                        lang_cond=lang_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if lang_cond is not None else None,
                                                                        ema_model=ema_model,
                                                                        modality_info=modality_info,
                                                                        head_grid_size=head_grid_size
                                                                        )
        v_ct_pred = self.model(
            sample=consistency_target_dict['x_t'], 
            timestep=consistency_target_dict['t'].squeeze(),
            target_t=consistency_target_dict['target_t'].squeeze(),
            vis_cond=vis_cond[flow_batchsize:flow_batchsize+consistency_batchsize],
            lang_cond=lang_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if lang_cond is not None else None,
            modality_info=modality_info,
            head_grid_size=head_grid_size)
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
        
        # Get gate statistics if available (for Gate-Attention models)
        gate_stats = self.get_gate_stats()
        
        loss_dict = {
                'loss_flow': loss_flow,
                'loss_ct': loss_ct,
                'v_flow_pred_magnitude': v_flow_pred_magnitude,
                'v_ct_pred_magnitude': v_ct_pred_magnitude,
                'bc_loss': loss.item(),
        }
        
        # Add gate statistics to loss_dict if available
        if gate_stats:
            loss_dict.update(gate_stats)
            
            # 🔥 Add correlation metrics between loss and gate activations (Head-Proprio融合后)
            # This helps understand if certain modalities are more important for reducing loss
            if 'gate/modality_tactile_mean' in gate_stats:
                loss_dict['correlation/loss_vs_tactile_gate'] = loss_flow * gate_stats['gate/modality_tactile_mean']
            if 'gate/modality_head_mean' in gate_stats:
                # Head现在包含融合的proprio信息
                loss_dict['correlation/loss_vs_head_gate'] = loss_flow * gate_stats['gate/modality_head_mean']
            # if 'gate/modality_proprio_mean' in gate_stats:  # 🔥 移除，已融合进head
            #     loss_dict['correlation/loss_vs_proprio_gate'] = loss_flow * gate_stats['gate/modality_proprio_mean']

        return loss, loss_dict
