# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# FastDIT: https://github.com/chuanyangjin/fast-DiT
# --------------------------------------------------------

import re
import math
import logging
from typing import Union, Tuple, List, Optional
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.models.vision_transformer import Mlp, RmsNorm, use_fused_attn
from maniflow.model.diffusion.positional_embedding import SinusoidalPosEmb
from maniflow.model.diffusion.mmdit_block import MMDiTBlock
from maniflow.model.diffusion.mmdit_generalized_block import MMDiTBlock_Generalized
from maniflow.model.diffusion.hrdt_block import HRDTBlock, ActionDecoder
from hyper_connections import (
    HyperConnections,
    Residual
)
from termcolor import cprint

logger = logging.getLogger(__name__)

# def modulate(x, shift, scale):
#     return shift + (x * (scale))

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# Adapted from https://github.com/geyan21/ManiFlow_Policy
class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    """
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,
            proj_drop: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                mask: None) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Prepare attn mask (B, L) to mask the conditioion
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x


# class DiTXBlock(nn.Module):
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, p_drop_attn=0., 
#                     qkv_bias=True, qk_norm=True,
#                  **block_kwargs):
#         super().__init__()
#         self.hidden_size = hidden_size
        
#         # Layers
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # Added missing norm3
        
#         self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, 
#                                               dropout=p_drop_attn, **block_kwargs)
#         self.cross_attn = CrossAttention(dim=hidden_size, num_heads=num_heads,
#                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
#                                         norm_layer=nn.LayerNorm, **block_kwargs)
        
#         # MLP with GELU activation
#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, 
#                        act_layer=lambda: nn.GELU(approximate="tanh"), drop=0.0)

#         # AdaLN modulation
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 9 * hidden_size, bias=True)
#         )

#     def forward(self, x, global_c, context_c, attn_mask=None):
#         # Ensure c is broadcastable to x's shape
#         if len(global_c.shape) == 2:
#             global_c = global_c.unsqueeze(1)

#         # Get all modulation parameters at once (9 groups total)
#         mod = self.adaLN_modulation(global_c)
#         chunk_size = self.hidden_size
        
#         # Extract parameters for each block (shift, scale, gate for each of the 3 components)
#         params = {}
#         for i, name in enumerate(['self', 'cross', 'mlp']):
#             start_idx = i * 3 * chunk_size
#             params[f'shift_{name}'] = mod[:, :, start_idx:start_idx + chunk_size]
#             params[f'scale_{name}'] = mod[:, :, start_idx + chunk_size:start_idx + 2*chunk_size]
#             params[f'gate_{name}'] = mod[:, :, start_idx + 2*chunk_size:start_idx + 3*chunk_size]
        
#         # Self-attention block
#         normed_x = modulate(self.norm1(x), params['shift_self'], params['scale_self'])
#         self_out, _ = self.self_attn(normed_x, normed_x, normed_x, attn_mask=attn_mask)
#         x = x + params['gate_self'] * self_out
        
#         # Cross-attention block
#         normed_x = modulate(self.norm2(x), params['shift_cross'], params['scale_cross'])
#         cross_out = self.cross_attn(normed_x, context_c, mask=None)
#         x = x + params['gate_cross'] * self.cross_attn.attn_drop(cross_out)
        
#         # MLP block
#         normed_x = modulate(self.norm3(x), params['shift_mlp'], params['scale_mlp'])
#         mlp_out = self.mlp(normed_x)
#         x = x + params['gate_mlp'] * mlp_out
        
#         return x

class DiTXBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, p_drop_attn=0., 
                 qkv_bias=True, qk_norm=True, 
                 apply_adaLN_self_attn=True, apply_adaLN_cross_attn=True, apply_adaLN_mlp=True,
                 **block_kwargs):
        super().__init__()
        self.apply_adaLN_self_attn = apply_adaLN_self_attn
        self.apply_adaLN_mlp = apply_adaLN_mlp
        self.apply_adaLN_cross_attn = apply_adaLN_cross_attn
        self.hidden_size = hidden_size

        # cprint(f"DiTXBlock - apply_adaLN_self_attn: {self.apply_adaLN_self_attn}, apply_adaLN_cross_attn: {self.apply_adaLN_cross_attn}, apply_adaLN_mlp: {self.apply_adaLN_mlp}", "red")

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # For self-attention
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # For cross-attention
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # For MLP


        # Self-Attention
        
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, 
                                            dropout=p_drop_attn,
                                            # **block_kwargs
                                            )
        
        # Cross-Attention
        qkv_bias = qkv_bias # default to True for DiTMIXBlock
        qk_norm = qk_norm  # default to True for DiTMIXBlock
        # cprint(f"Using qkv_bias: {qkv_bias}, qk_norm: {qk_norm}", "yellow")
    
        self.cross_attn = CrossAttention(
            dim=hidden_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            norm_layer=nn.LayerNorm, **block_kwargs
        )
       
        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")  # Standard GELU
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)

        modulation_size = 9 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, modulation_size, bias=True)
        )
        
    def forward(self, x, global_c, context_c, attn_mask=None):
        # adaLN modulation for self-attention, cross-attention, and optionally MLP
        modulation = self.adaLN_modulation(global_c)  # Shape varies based on conditioning flags
        current_chunk = 0
        # Self-Attention parameters
        shift_msa, scale_msa, gate_msa = modulation[:, current_chunk:current_chunk + self.hidden_size], \
                                         modulation[:, current_chunk + self.hidden_size:current_chunk + 2 * self.hidden_size], \
                                         modulation[:, current_chunk + 2 * self.hidden_size:current_chunk + 3 * self.hidden_size]
        current_chunk += 3 * self.hidden_size

        # Cross-Attention parameters
        if self.apply_adaLN_cross_attn:
            shift_cross, scale_cross, gate_cross = modulation[:, current_chunk:current_chunk + self.hidden_size], \
                                                   modulation[:, current_chunk + self.hidden_size:current_chunk + 2 * self.hidden_size], \
                                                   modulation[:, current_chunk + 2 * self.hidden_size:current_chunk + 3 * self.hidden_size]
            current_chunk += 3 * self.hidden_size
        else:
            shift_cross, scale_cross, gate_cross = None, None, None

        # Scale and shift for MLP
        shift_mlp, scale_mlp, gate_mlp = modulation[:, current_chunk:current_chunk + self.hidden_size], \
                                            modulation[:, current_chunk + self.hidden_size:current_chunk + 2 * self.hidden_size], \
                                            modulation[:, current_chunk + 2 * self.hidden_size:current_chunk + 3 * self.hidden_size]
        current_chunk += 3 * self.hidden_size

        # Self-Attention with adaLN conditioning
        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)  # Shape: (batch_size, seq_length, hidden_size)
        self_attn_output, _ = self.self_attn(normed_x, normed_x, normed_x, attn_mask=attn_mask)  # Shape: (batch_size, seq_length, hidden_size)
        x = x + gate_msa.unsqueeze(1) * self_attn_output  # Apply gating and residual connection
        
        # Cross-Attention with adaLN conditioning (if applied)
        if self.apply_adaLN_cross_attn:
            normed_x_cross = modulate(self.norm2(x), shift_cross, scale_cross)  # Apply adaLN to x before cross-attn
        else:
            normed_x_cross = self.norm2(x)  # Normalize without adaLN


        cross_attn_output = self.cross_attn(normed_x_cross, context_c, mask=None)  # Shape: (batch_size, seq_length, hidden_size)
        if self.apply_adaLN_cross_attn:
            x = x + gate_cross.unsqueeze(1) * self.cross_attn.attn_drop(cross_attn_output)  # Gated residual connection
        else:
            x = x + self.cross_attn.attn_drop(cross_attn_output)  # Apply residual connection without gating
    

        # MLP with adaLN conditioning
        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)  # Apply adaLN modulation
        mlp_output = self.mlp(normed_x_mlp)  # Pass through MLP
        x = x + gate_mlp.unsqueeze(1) * mlp_output  # Apply gating and residual connection

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DIT-X, adopted from RDT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels, 
            act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x

# Adapted from https://github.com/geyan21/ManiFlow_Policy
class UnifiedDiTX(nn.Module):
    """
    Flow Matching model with a Transformer backbone.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 256,
        img_cond_len: int = 1024,
        point_cond_len: int = 512,
        img_cond_dim: int = 512,
        point_cond_dim: int = 512,
        alternate_injection: bool = False,
        num_modalities: int = 3,  # Number of modalities for MMDiT_Generalized
        # dim_modalities: Tuple[int, ...] = (512, 512, 512),  # For MMDiT_Generalized
        diffusion_timestep_embed_dim: int = 256,
        diffusion_stepsize_embed_dim: int = 256,
        block_type: str = "DiTX",
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        add_t_to_action_decoder: bool = False,
        language_conditioned: bool=False,
        simple_adaptor: bool = False,
        pre_norm: bool = False,
        use_meta_query: bool = False,
        meta_query_mode: str = "concat",  # How to use meta-query, can be 'concat' or 'replace'
        num_queries: int = 1,
    ):
        super().__init__()
        # compute number of tokens for main trunk and condition encoder
        self.n_obs_steps = n_obs_steps
        self.img_cond_len = img_cond_len
        self.point_cond_len = point_cond_len
        self.language_conditioned = language_conditioned
        self.num_modalities = num_modalities
        self.alternate_injection = alternate_injection

        self.use_meta_query = use_meta_query
        self.meta_query_mode = meta_query_mode
        self.num_queries = num_queries
        if self.use_meta_query:
            # Meta-query is used to condition the model on a specific task or context
            self.meta_query = nn.Parameter(torch.randn(num_queries, n_emb))
            cprint(f"Using meta-query {self.meta_query_mode} with {num_queries} queries and embedding size {n_emb}", "yellow")

        # constants
        T = horizon
        self.T = T
        self.horizon = horizon
        
        # input embedding stem
        self.hidden_dim = n_emb
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.vis_cond_obs_emb = nn.Linear(cond_dim, n_emb) # visual condition observation embedding
        self.img_cond_pos_embed = nn.Parameter(
            torch.zeros(1, img_cond_len * n_obs_steps, n_emb)
            )  # learnable image condition positional embedding
        self.point_cond_pos_embed = nn.Parameter(
            torch.zeros(1, point_cond_len * n_obs_steps, n_emb)
            )  # learnable point condition positional embedding

        self.simple_adaptor = simple_adaptor
        if not simple_adaptor:
            self.img_cond_adaptor = self.build_condition_adapter(
                "mlp2x_gelu",  
                in_features=img_cond_dim,
                out_features=n_emb
            )
            self.point_cond_adaptor = self.build_condition_adapter(
                "mlp2x_gelu",
                in_features=point_cond_dim,
                out_features=n_emb
            )
        else:
            # Simple linear adaptors for image and point conditions
            self.img_cond_adaptor = nn.Linear(img_cond_dim, n_emb)
            self.point_cond_adaptor = nn.Linear(point_cond_dim, n_emb)

        self.pre_norm = pre_norm
        if self.pre_norm:
            self.img_norm = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)
            self.point_norm = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)

        # timestep and stepsize cond encoder
        flow_timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_timestep_embed_dim),
            nn.Linear(diffusion_timestep_embed_dim, diffusion_timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_timestep_embed_dim * 4, n_emb),
        )
        flow_stepsize_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_stepsize_embed_dim),
            nn.Linear(diffusion_stepsize_embed_dim, diffusion_stepsize_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_stepsize_embed_dim * 4, self.hidden_dim),
        )
        self.flow_timestep_encoder = flow_timestep_encoder
        self.flow_stepsize_encoder = flow_stepsize_encoder
        self.timestep_stepsize_adaptor = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Language conditioning, use T5-small as default
        if self.language_conditioned:
            self.load_T5_encoder(freeze=True)
            self.lang_adaptor = self.build_condition_adapter(
                "mlp2x_gelu", 
                in_features=self.language_encoder_out_dim, 
                out_features=n_emb
            )
        
        # Building the transformer blocks
        self.block_type = block_type
        if block_type == "DiTX":
            self.blocks = nn.ModuleList([
                DiTXBlock(n_emb, n_head, mlp_ratio=mlp_ratio, p_drop_attn=p_drop_attn, 
                    qkv_bias=qkv_bias, qk_norm=qk_norm) for _ in range(n_layer)
            ])
            cprint(f"[DiTX Transformer] Initialized {n_layer} DiTX blocks with hidden size {n_emb}, "
                    f"num heads {n_head}, mlp ratio {mlp_ratio}, dropout {p_drop_attn}, qkv_bias {qkv_bias}, qk_norm {qk_norm}", "cyan")
        elif block_type == "MMDiT":
            self.blocks = nn.ModuleList([
                MMDiTBlock(
                    dim_text=n_emb,
                    dim_image=n_emb,
                    dim_cond=n_emb,
                    dim_head=n_emb // n_head,
                    heads=n_head,
                    qk_rmsnorm=qk_norm,
                    ) for _ in range(n_layer)
            ])
            cprint(f"[MMDiT Transformer] Initialized {n_layer} MMDiT blocks with hidden size {n_emb}, "
                    f"num heads {n_head}, qk_rmsnorm {qk_norm}, block type {block_type}", "green")
        elif block_type == "MMDiT_Generalized":
            # MMDiT_Generalized uses dim_modalities to define the input dimensions for each modality
            dim_modalities = (n_emb, n_emb, n_emb) if num_modalities == 3 else (n_emb,) * num_modalities
            cprint(f"[MMDiT_Generalized Transformer] Initialized with {num_modalities} modalities, "
                   f"dim_modalities: {dim_modalities}", "green")
            num_residual_streams = 1
            self.expand_streams, self.reduce_streams = HyperConnections.get_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)
            # Initialize MMDiT_Generalized with the specified dimensions
            self.blocks = nn.ModuleList([
                MMDiTBlock_Generalized(
                    dim_modalities=dim_modalities,
                    num_residual_streams=num_residual_streams,
                    qk_rmsnorm=qk_norm,
                    dim_cond=n_emb,
                    dim_head=n_emb // n_head,
                    heads=n_head,
                ) for _ in range(n_layer)
            ])
            norms = [RmsNorm(dim) for dim in dim_modalities]
            self.norms = nn.ModuleList(norms)
            cprint(f"[MMDiT_Generalized Transformer] Initialized {n_layer} MMDiT_Generalized blocks with hidden size {n_emb}, "
                    f"residual streams {num_residual_streams}, "
                    f"num heads {n_head}, head_dim {n_emb // n_head}, qk_rmsnorm {qk_norm}, block type {block_type}", "green")
        elif block_type == "HRDT":
            HRDT_config = {
                "hidden_size": n_emb,
                "num_heads": n_head,
                "num_kv_heads": 8,
                "norm_eps": 0.00001,
                "multiple_of": 256,
                "ffn_dim_multiplier": None,
                "use_flash_attn": True,  # use flash attention
            }
            self.blocks = nn.ModuleList([
                HRDTBlock(layer_idx, config=HRDT_config, 
                        #   training_mode=training_mode
                          ) for layer_idx in range(n_layer)
            ])
            cprint(f"[HRDT Transformer] Initialized {n_layer} HRDT blocks with hidden size {n_emb}, "   
                    f"num heads {n_head}, norm_eps {HRDT_config['norm_eps']}, "
                    f"multiple_of {HRDT_config['multiple_of']}, use_flash_attn {HRDT_config['use_flash_attn']}", "cyan")
        
        self.add_t_to_action_decoder = add_t_to_action_decoder
        if add_t_to_action_decoder:
            # Final Layer
            action_decoder_config = {
                "hidden_size": n_emb,
                "norm_eps": 0.00001,
                "output_size": output_dim,
            }
            self.final_layer = ActionDecoder(
                action_decoder_config,
            )
            cprint(f"[DiTX Transformer] Initialized ActionDecoder with hidden size {n_emb}, output dim {output_dim}", "cyan")
        else: 
            self.final_layer = FinalLayer(n_emb, output_dim)
            cprint(f"[DiTX Transformer] Initialized FinalLayer with hidden size {n_emb}, output dim {output_dim}", "cyan")
        
        
        
        self.initialize_weights()
        cprint(f"[DiTX Transformer] Initialized weights for DiTX Transformer with block {self.block_type}", "green")

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
    
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    # language encoder
    def load_T5_encoder(self, freeze=True):
        from transformers import (
            T5Config,
            T5EncoderModel,
            AutoTokenizer
        )
        T5_model_name = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]
        encoder_name = T5_model_name[0]
        pretrained_model_id = f"google-t5/{encoder_name}"
        encoder_cfg = T5Config()
        self.language_encoder = T5EncoderModel(encoder_cfg).from_pretrained(
                pretrained_model_id
            )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
        if freeze:
            self.language_encoder.eval()
            # freeze the language encoder
            for param in self.language_encoder.parameters():
                param.requires_grad = False

        self.language_encoder_out_dim = 512
        cprint(f"Loaded T5 encoder: {encoder_name}", "green")
    
    def encode_text_input_T5(self, 
                             lang_cond, 
                             norm_lang_embedding=False, 
                             device="cuda",
                             output_type="sentence"
                             ):
        language_inputs = self.tokenizer(
            lang_cond,
            return_tensors="pt",
            padding=True,
            truncation=True,
            )
        input_ids = language_inputs["input_ids"].to(device)
        attention_mask = language_inputs["attention_mask"].to(device)
        encoder_outputs = self.language_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            )
        token_embeddings = encoder_outputs.last_hidden_state
        if output_type == "token":
            return token_embeddings
        # obtain sentence embedding by averaging the token embeddings
        sentence_embedding = torch.mean(token_embeddings, dim=1).squeeze(1) # (B, 512)
        if norm_lang_embedding:
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=-1)

        return sentence_embedding

    def initialize_weights(self):
        for block in self.blocks:
            if self.block_type == "DiTX":
                # Initialize self_attn's in_proj_weight and out_proj
                nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
                if block.self_attn.in_proj_bias is not None:
                    nn.init.zeros_(block.self_attn.in_proj_bias)
                
                nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
                if block.self_attn.out_proj.bias is not None:
                    nn.init.zeros_(block.self_attn.out_proj.bias)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.block_type == "DiTX" or self.block_type == "HRDT":
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        

        # Initialize input emb by normal distribution:
        nn.init.normal_(self.input_emb.weight, std=0.02)
        nn.init.constant_(self.input_emb.bias, 0) if self.input_emb.bias is not None else None

        # Initialize pos emb by normal distribution:
        nn.init.normal_(self.pos_emb, std=0.02)       
        # nn.init.constant_(self.pos_emb, 0) # not used 

        # Initialize diffusion step encoder:
        for layer in self.flow_timestep_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # Initialize diffusion stepsize encoder:
        for layer in self.flow_stepsize_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # Initialize conditional observation embedding:
        nn.init.normal_(self.vis_cond_obs_emb.weight, std=0.02)
        nn.init.constant_(self.vis_cond_obs_emb.bias, 0) if self.vis_cond_obs_emb.bias is not None else None

        if self.simple_adaptor:
            # Initialize image condition adaptor
            nn.init.normal_(self.img_cond_adaptor.weight, std=0.02)
            nn.init.constant_(self.img_cond_adaptor.bias, 0) if self.img_cond_adaptor.bias is not None else None
            
            # Initialize point condition adaptor
            nn.init.normal_(self.point_cond_adaptor.weight, std=0.02)
            nn.init.constant_(self.point_cond_adaptor.bias, 0) if self.point_cond_adaptor.bias is not None else None

        # Initialize the adapter for timestep and stepsize
        nn.init.normal_(self.timestep_stepsize_adaptor.weight, std=0.02)
        nn.init.constant_(self.timestep_stepsize_adaptor.bias, 0)

        # Initialize the image condition positional embedding
        nn.init.normal_(self.img_cond_pos_embed, std=0.02)
        nn.init.constant_(self.img_cond_pos_embed, 0)
        # Initialize the point condition positional embedding
        nn.init.normal_(self.point_cond_pos_embed, std=0.02)
        nn.init.constant_(self.point_cond_pos_embed, 0)

        # Initialize the language condition adaptor
        if self.language_conditioned:
            nn.init.normal_(self.lang_adaptor.weight, std=0.02)
            nn.init.constant_(self.lang_adaptor.bias, 0) if self.lang_adaptor.bias is not None else None

        # Initialize the final layer: zero-out the final linear layer
        if not self.add_t_to_action_decoder:
            nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
            nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        else:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.ffn.fc2.weight, 0)
            nn.init.constant_(self.final_layer.ffn.fc2.bias, 0)
    
    def get_optim_groups(self, weight_decay: float=1e-3, group_params=False):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        if not group_params:
            cprint("Using single parameter group for optimizer, with weight decay applied to all parameters", "yellow")
            return [
                {
                    "params": list(self.parameters()),
                    "weight_decay": weight_decay,  # All parameters get the same weight decay
                }
            ]

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RmsNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        if self.vis_cond_pos_embed is not None:
            # this is a learnable parameter, so we don't want to decay it
            no_decay.add("vis_cond_pos_embed") 

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer


    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            stepsize: Union[torch.Tensor, float, int], 
            img_cond: torch.Tensor = None,
            point_cond: torch.Tensor = None,
            lang_cond: Union[torch.Tensor, list, str] = None,
            img_mask: Optional[torch.Tensor] = None,
            point_mask: Optional[torch.Tensor] = None,  
            lang_mask: Optional[torch.Tensor] = None,
            **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, maniflow time step t
        stepsize: (B,) or float, the step size for the flow matching process
        img_cond: (B, L_img, img_cond_dim), visual condition input
        point_cond: (B, L_point, point_cond_dim), point condition input
        lang_cond: (B,) or list of strings, language condition input
        **kwargs: additional arguments
        output: (B,T,output_dim)
        """

        """ To be implemented: residual condition, dropout, etc. """
        modality_tokens = ()
        modality_masks = ()
        # process input
        input_emb = self.input_emb(sample) # (B, T, n_emb)
        x = input_emb + self.pos_emb # (B, T, n_emb)

        modality_tokens += (x,) # add action tokens to modality tokens
        modality_masks += (None,)  # No mask for action tokens
 
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        timestep_embed = self.flow_timestep_encoder(timesteps) # (B, n_emb)
        # global_c = timestep_embed

        # 2. stepsize
        stepsizes = stepsize
        if not torch.is_tensor(stepsizes): 
            stepsizes = torch.tensor([stepsizes], dtype=torch.float32, device=sample.device)    
        elif torch.is_tensor(stepsizes) and len(stepsizes.shape) == 0:
            stepsizes = stepsizes[None].to(sample.device)
        stepsizes = stepsizes.expand(sample.shape[0])
        stepsize_embed = self.flow_stepsize_encoder(stepsizes) # (B, n_emb)
        global_c = torch.cat([timestep_embed, stepsize_embed], dim=-1) # (B, 2*n_emb)
        global_c = self.timestep_stepsize_adaptor(global_c) # (B, n_emb)
        
        # 3. visual condition
        # vis_con_obs_emb = self.vis_cond_obs_emb(global_cond) # (B, L, n_emb)
        # vis_cond_pos_embed = self.vis_cond_pos_embed[:, :global_cond.shape[1]]
        # context_c = vis_con_obs_emb + vis_cond_pos_embed # (B, L, n_emb)
        context_c = None
        if img_cond is not None:
            img_cond = self.img_cond_adaptor(img_cond) # (B, L_img, n_emb)
            img_cond = img_cond + self.img_cond_pos_embed[:, :img_cond.shape[1]]
            if self.pre_norm:
                img_cond = self.img_norm(img_cond)  # Normalize the image condition
            context_c = img_cond # (B, L_img, n_emb)
            modality_tokens += (img_cond,)
            modality_masks += (img_mask,)  # Add mask for image condition

        if point_cond is not None:
            point_cond = self.point_cond_adaptor(point_cond) # (B, L_point, n_emb)
            point_cond = point_cond + self.point_cond_pos_embed[:, :point_cond.shape[1]]
            point_token_len = point_cond.shape[1]
            if self.use_meta_query:
                # repeat meta_query for batch size
                meta_query = repeat(self.meta_query, 'n d -> b n d', b=point_cond.shape[0])
                if self.meta_query_mode == "concat":
                    # If using meta-query, concatenate it to the point condition
                    point_cond = torch.cat([meta_query, point_cond], dim=1)  # (B, L_point + num_queries, n_emb)
                    meta_query_mask = torch.ones(point_cond.shape[0], self.num_queries, dtype=torch.bool, device=point_cond.device)
                    point_mask = torch.cat([meta_query_mask, point_mask], dim=1)
                elif self.meta_query_mode == "replace":
                    meta_query = meta_query[:, :point_token_len, :]  # Ensure it matches the point token length
                    # assert point_cond.shape[1] == self.num_queries, \
                    #     f"Point condition length {point_cond.shape[1]} does not match num_queries {self.num_queries}"
                    # If using meta-query, replace the first num_queries entries in point_cond
                    fully_masked = (~point_mask).all(dim=1)  # [B] - True where all entries are False
                    point_cond[fully_masked] = meta_query[fully_masked]
                    # Update mask for replaced entries
                    # Clone before in-place operation
                    point_mask = point_mask.clone()
                    point_mask[fully_masked] = True  # Meta queries are not masked
                else:
                    raise ValueError(f"Unknown meta_query_mode: {self.meta_query_mode}. Use 'concat' or 'replace'.")
                
            if self.pre_norm:
                point_cond = self.point_norm(point_cond)  # Normalize the point condition
            if context_c is not None:
                # concatenate image and point conditions
                context_c = torch.cat([context_c, point_cond], dim=1)
            else:
                context_c = point_cond # (B, L_point, n_emb)
            modality_tokens += (point_cond,)
            modality_masks += (point_mask,)  # Add mask for point condition
        # import pdb; pdb.set_trace()
        # 4. language condition
        if self.language_conditioned:
            assert lang_cond is not None
            lang_c = self.encode_text_input_T5(lang_cond, output_type="token")
            lang_c = self.lang_adaptor(lang_c) # (B, L, D) or (B, D)
            context_c = torch.cat([context_c, lang_c], dim=1) # (B, L + L_lang, n_emb)
            modality_tokens += (lang_c,)
            modality_masks += (lang_mask,)  # Add mask for language condition
        
        if self.block_type == "HRDT":
            # HRDT expects context_c as a dictionary
            context_c = {
                'img_c': context_c,
                'lang_c': lang_c if self.language_conditioned else None,
            }

        # modality_tokens_copy = [token.clone() for token in modality_tokens]
        modality_masks_copy = [mask.clone() if mask is not None else mask for mask in modality_masks]

        # 5. transformer blocks
        if self.block_type != "MMDiT_Generalized":
            for i, block in enumerate(self.blocks):
                if self.block_type == "MMDiT":
                    context_c, x = block(
                        image_tokens=x, 
                        text_tokens=context_c, 
                        time_cond=global_c
                        )
                else: # DiTX or HRDT
                    x = block(x, global_c, context_c) # (B, T, n_emb)
        else:
            assert self.block_type == "MMDiT_Generalized", "MMDiT_Generalized block type expected"
            for i, block in enumerate(self.blocks):
                # MMDiT_Generalized expects modality_tokens as a tuple of tensors
                # modality_tokens: (action_tokens, img_cond_tokens, point_cond_tokens, lang_cond_tokens)
                # modality_masks: (None, img_mask, point_mask, lang_mask)
                assert len(modality_tokens) == self.num_modalities, \
                    f"Expected {self.num_modalities} modality tokens, got {len(modality_tokens)}"
                # expand streams
                modality_tokens = [self.expand_streams(modality) for modality in modality_tokens]
                
                if self.alternate_injection:
                    assert self.num_modalities > 2, "num_modalities must be greater than 2 for alternate injection"
                    # Calculate which conditioning modality to use (skip action tokens at index 0)
                    cur_index = (i % (self.num_modalities - 1)) + 1  # Range: 1 to num_modalities-1
                    # Create modified modality lists for this iteration
                    # Include action tokens (index 0) + only the current conditioning modality
                    selected_modality_tokens = [
                        modality_tokens[0],  # Always include action tokens
                        modality_tokens[cur_index]  # Current conditioning modality
                    ]
                    selected_modality_masks = [
                        modality_masks_copy[0],  # Always include action mask (None)
                        modality_masks_copy[cur_index]  # Current conditioning mask
                    ]
                    output_tokens = block(
                        time_cond=global_c,
                        modality_tokens=selected_modality_tokens,
                        modality_masks=selected_modality_masks,
                        # modality_masks=None, 
                    )
                    modality_tokens[0] = output_tokens[0]  # Update action tokens
                    modality_tokens[cur_index] = output_tokens[1]  # Update current conditioning
                else:
                    # modality_masks = None
                    modality_tokens = block(
                        time_cond=global_c,
                        modality_tokens=modality_tokens,
                        modality_masks=modality_masks,
                        # modality_masks=None,
                    )
            modality_tokens = [self.reduce_streams(modality) for modality in modality_tokens]
            modality_tokens = [norm(tokens) for tokens, norm in zip(modality_tokens, self.norms)]
            # take action tokens out
            x = modality_tokens[0]  # action tokens are the first modality

        # 6. head
        if not self.add_t_to_action_decoder:
            x = self.final_layer(x)
        else:
            x = self.final_layer(x, global_c)
       
        # (B, T, output_dim)
        x = x[:, -self.horizon:] # (B, T, out_channels)
        
        return x

if __name__ == "__main__":
    # Example usage of DiTX model
    torch.manual_seed(0)  # For reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.randn(2, 10, 16).to(device)  # Batch size 2, horizon 10, input_dim 16
    timestep = torch.tensor([1, 2]).to(device)  # Example timesteps for each sample in the batch
    stepsize = torch.tensor([0.1, 0.2]).to(device)  # Example stepsizes for each sample in the batch
    img_cond = torch.randn(2, 15, 512).to(device)  # 15 time steps of image condition
    point_cond = torch.randn(2, 5, 256).to(device)  # 5 time steps of point condition
    lang_cond = ["This is a test sentence.", "Another test sentence."]
    model = UnifiedDiTX(
        input_dim=16,
        output_dim=16,
        horizon=10,
        n_obs_steps=5,
        cond_dim=256,
        img_cond_len=15,
        point_cond_len=5,
        img_cond_dim=512,
        point_cond_dim=256,
        diffusion_timestep_embed_dim=256,
        diffusion_stepsize_embed_dim=256,
        block_type="MMDiT_Generalized", # "DiTX" or "MMDiT", "MMDiT_Generalized" or "HRDT"
        num_modalities=3,  # For MMDiT_Generalized, set to 2 for image and point conditions, action is always the first modality
        n_layer=2,  # Reduced for testing
        n_head=8,
        n_emb=768,
        mlp_ratio=4.0,
        p_drop_attn=0.1,
        language_conditioned=False,  # Set to True if you want to test language conditioning
    )
    model = model.to(device)
    output = model(sample, timestep, stepsize, img_cond, point_cond, lang_cond)
    print("Output shape:", output.shape)  # Should be (2, 10, 768)
    assert output.shape == (2, 10, 16), "Output shape mismatch!"
    # Check if the model is initialized correctly
    logger.info("UnifiedDiTX model initialized and tested successfully.")