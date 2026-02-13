# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# RDT: https://github.com/thu-ml/RoboticsDiffusionTransformer
# Qwen3: Gate-Attention mechanism
# --------------------------------------------------------

import re
import math
import logging
from typing import Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, RmsNorm
from maniflow.model.diffusion.positional_embedding import SinusoidalPosEmb
from maniflow.model.diffusion.ditx_block import AdaptiveLayerNorm
from maniflow.model.diffusion.ditx_gateattn_block import DiTXGateAttnBlock
from termcolor import cprint

logger = logging.getLogger(__name__)


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, cls_token=False, temperature=10000.0):
    """
    生成2D正弦余弦位置编码（参考MAE实现）
    
    Args:
        embed_dim: 嵌入维度（必须是偶数）
        grid_h: 网格高度
        grid_w: 网格宽度
        cls_token: 是否添加CLS token位置
        temperature: 温度参数（控制频率范围）
    
    Returns:
        pos_embed: (grid_h*grid_w, embed_dim) 或 (1+grid_h*grid_w, embed_dim) 如果cls_token=True
    """
    grid_h_coords = np.arange(grid_h, dtype=np.float32)
    grid_w_coords = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w_coords, grid_h_coords)  # (H, W)
    grid = np.stack(grid, axis=0)  # (2, H, W)
    
    grid = grid.reshape([2, 1, grid_h, grid_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, temperature)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, temperature=10000.0):
    """
    从2D网格生成正弦余弦位置编码
    
    Args:
        embed_dim: 嵌入维度
        grid: (2, 1, H, W) 网格坐标
        temperature: 温度参数
    
    Returns:
        pos_embed: (H*W, embed_dim)
    """
    assert embed_dim % 2 == 0, "embed_dim必须是偶数"
    
    # 使用一半维度编码y坐标，另一半编码x坐标
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], temperature)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], temperature)  # (H*W, D/2)
    
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, temperature=10000.0):
    """
    从1D位置生成正弦余弦编码
    
    Args:
        embed_dim: 输出维度
        pos: (M,) 或 (1, H, W) 位置数组
        temperature: 温度参数
    
    Returns:
        emb: (M, embed_dim) 位置编码
    """
    assert embed_dim % 2 == 0, "embed_dim必须是偶数"
    
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (temperature ** omega)  # (D/2,)
    
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2) 外积
    
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


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
    
class DiTXGateAttn(nn.Module):
    """
    Consistency Flow Training model with a Diffusion Transformer backbone using Gate-Attention.
    
    核心特性：
    1. 使用 DiTXGateAttnBlock，支持 Gate-Attention 机制（参考Qwen3）
    2. 支持三种 gate 模式：'none', 'headwise', 'elementwise'
    3. Flash Attention 2 加速训练和推理
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 256,
        visual_cond_len: int = 1024,
        diffusion_timestep_embed_dim: int = 256,
        diffusion_target_t_embed_dim: int = 256,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        gate_type: str = 'elementwise',  # Gate-Attention类型: 'none', 'headwise', 'elementwise'
        pre_norm_modality: bool = False,
        language_conditioned: bool=False,
        language_model: str = "t5-small",
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.visual_cond_len = visual_cond_len
        self.language_conditioned = language_conditioned
        self.pre_norm_modality = pre_norm_modality
        self.gate_type = gate_type
        
        # constants
        T = horizon
        self.T = T
        self.horizon = horizon

        # input embedding stem
        self.hidden_dim = n_emb
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.vis_cond_obs_emb = nn.Linear(cond_dim, n_emb) # visual condition observation embedding
        self.vis_cond_pos_embed = nn.Parameter(
            torch.zeros(1, visual_cond_len * n_obs_steps, n_emb)
            )  # learnable visual condition positional embedding
        
        # pre-norm visual modality
        if self.pre_norm_modality:
            # If pre-norm modality is used, apply adaLN modulation before the transformer blocks
            self.vis_norm = AdaptiveLayerNorm(
                dim=n_emb,
                dim_cond=n_emb,
            )

        # timestep and target_t cond encoder
        flow_timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_timestep_embed_dim),
            nn.Linear(diffusion_timestep_embed_dim, diffusion_timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_timestep_embed_dim * 4, n_emb),
        )
        flow_target_t_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_target_t_embed_dim),
            nn.Linear(diffusion_target_t_embed_dim, diffusion_target_t_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_target_t_embed_dim * 4, self.hidden_dim),
        )
        self.flow_timestep_encoder = flow_timestep_encoder
        self.flow_target_t_encoder = flow_target_t_encoder
        self.timestep_target_t_adaptor = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Language conditioning, use T5-small as default
        if self.language_conditioned:
            self.load_T5_encoder(
                model_name=language_model,
                freeze=True)
            self.lang_adaptor = self.build_condition_adapter(
                "mlp2x_gelu", 
                in_features=self.language_encoder_out_dim, 
                out_features=n_emb
            )
            # pre-norm language modality
            if self.pre_norm_modality:
                self.lang_norm = AdaptiveLayerNorm(
                    dim=n_emb,
                    dim_cond=n_emb,
                )
            
        # Building the transformer blocks with Gate-Attention
        self.blocks = nn.ModuleList([
            DiTXGateAttnBlock(
                hidden_size=n_emb, 
                num_heads=n_head, 
                mlp_ratio=mlp_ratio, 
                gate_type=gate_type,  # Gate-Attention配置
                p_drop_attn=p_drop_attn, 
                qkv_bias=qkv_bias, 
                qk_norm=qk_norm
            ) for _ in range(n_layer)
        ])
        cprint(f"[DiTXGateAttn Transformer] Initialized {n_layer} DiTXGateAttn blocks with hidden size {n_emb}, "
                f"num heads {n_head}, mlp ratio {mlp_ratio}, dropout {p_drop_attn}, "
                f"qkv_bias {qkv_bias}, qk_norm {qk_norm}, gate_type {gate_type}", "cyan")
        
        # Final Layer
        self.final_layer = FinalLayer(n_emb, output_dim)

        self.initialize_weights()
        cprint(f"[DiTXGateAttn Transformer] Initialized weights for DiTXGateAttn", "green")

        # Gate bias初始化标志
        self._gate_bias_initialized = False
        
        # 位置编码初始化标志
        self._pos_embed_initialized = False
        
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
    def load_T5_encoder(self, model_name, freeze=True):
        from transformers import (
            T5Config,
            T5EncoderModel,
            AutoTokenizer
        )
        T5_model_name = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]
        assert model_name in T5_model_name, f"Model name {model_name} not in {T5_model_name}"
        encoder_name = model_name
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
                             output_type="sentence",
                             device="cuda" if torch.cuda.is_available() else "cpu"
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

    def _initialize_gate_bias(self, modality_info: dict):
        """
        为所有block初始化gate bias
        
        Args:
            modality_info: 来自encoder.get_modality_info(), 例如 {'head': 100, 'tactile': 4, 'proprio': 16}
        """
        for i, block in enumerate(self.blocks):
            if hasattr(block, 'set_modality_ranges'):
                block.set_modality_ranges(modality_info)
        # logger.info(f"[DiTXGateAttn] Gate bias已初始化，模态信息: {modality_info}")
    
    def _initialize_vis_pos_embed(self, modality_info: dict, head_grid_size: int = None):
        """
        🔥 基于模态感知的2D Sin-Cos位置编码初始化
        
        设计理念：为不同模态注入合适的几何先验
        - Head相机: 2D Sin-Cos编码（利用$N \times N$网格的空间几何结构）
        - Wrist相机: 1D Sin-Cos编码（序列结构）
        - Tactile: 1D Sin-Cos编码（序列结构）
        - Proprio: 1D Sin-Cos编码（序列结构）
        
        Args:
            modality_info: 模态token数量，例如 {'head': 32, 'rgb_wrist': 4, 'tactile': 4, 'proprio': 2}
            head_grid_size: Head相机的网格大小（用于计算2D编码）
        """
        if modality_info is None:
            logger.warning("[DiTXGateAttn] modality_info为None，跳过位置编码初始化")
            return
        
        # 计算token组织: [head, rgb_wrist, tactile, proprio]
        start_idx = 0
        pos_embed_data = self.vis_cond_pos_embed.data  # (1, L_total, n_emb)
        L_total = pos_embed_data.shape[1]
        embed_dim = pos_embed_data.shape[2]
        
        cprint(f"[DiTXGateAttn] 开始初始化位置编码: L_total={L_total}, embed_dim={embed_dim}", 'cyan')
        cprint(f"[DiTXGateAttn] 模态信息: {modality_info}", 'cyan')
        
        # === 1. Head相机: 2D Sin-Cos编码 ===
        if 'head' in modality_info and modality_info['head'] > 0:
            n_head_tokens = modality_info['head']
            
            # 推断grid_size: n_head_tokens = T * grid_size^2
            # 默认n_obs_steps=2，所以 grid_size^2 = n_head_tokens / T
            if head_grid_size is None:
                # 自动推断
                tokens_per_timestep = n_head_tokens // self.n_obs_steps
                head_grid_size = int(math.sqrt(tokens_per_timestep))
                
                if head_grid_size * head_grid_size != tokens_per_timestep:
                    logger.warning(f"[Head] 无法推断完美的网格大小: {tokens_per_timestep} tokens/timestep")
                    head_grid_size = 1  # 回退到1D编码
            
            if head_grid_size > 1:
                # 🔥 2D空间编码
                cprint(f"[Head] 使用2D Sin-Cos编码: {head_grid_size}x{head_grid_size} 网格, {self.n_obs_steps}帧", 'green')
                
                # 生成单帧的2D位置编码
                pos_embed_2d_single = get_2d_sincos_pos_embed(
                    embed_dim=embed_dim,
                    grid_h=head_grid_size,
                    grid_w=head_grid_size,
                    cls_token=False
                )  # (grid_size^2, embed_dim)
                
                # 为多帧叠加时间偏置
                for t in range(self.n_obs_steps):
                    # 时间偏置: 使用1D Sin-Cos编码在频率空间与空间坐标解耦
                    time_embed = get_1d_sincos_pos_embed_from_grid(
                        embed_dim=embed_dim,
                        pos=np.array([t], dtype=np.float32),
                        temperature=10000.0  # 与空间编码相同的温度
                    )  # (1, embed_dim)
                    
                    # 叠加: spatial + temporal
                    # 使用加权叠加：空间0.8 + 时间0.2，确保空间先验占主导
                    pos_embed_frame = 0.8 * pos_embed_2d_single + 0.2 * time_embed
                    
                    # 填充到vis_cond_pos_embed
                    frame_start = start_idx + t * (head_grid_size ** 2)
                    frame_end = frame_start + (head_grid_size ** 2)
                    pos_embed_data[0, frame_start:frame_end, :] = torch.from_numpy(pos_embed_frame).float()
                
                cprint(f"[Head] ✓ 2D位置编码已注入: tokens [{start_idx}:{start_idx + n_head_tokens}]", 'green')
            else:
                # 1D编码回退
                cprint(f"[Head] 使用1D Sin-Cos编码（回退模式）", 'yellow')
                pos_embed_1d = get_1d_sincos_pos_embed_from_grid(
                    embed_dim=embed_dim,
                    pos=np.arange(n_head_tokens, dtype=np.float32),
                    temperature=10000.0
                )
                pos_embed_data[0, start_idx:start_idx + n_head_tokens, :] = torch.from_numpy(pos_embed_1d).float()
            
            start_idx += n_head_tokens
        
        # === 2. Wrist相机: 1D Sin-Cos编码 ===
        if 'rgb_wrist' in modality_info and modality_info['rgb_wrist'] > 0:
            n_wrist_tokens = modality_info['rgb_wrist']
            cprint(f"[Wrist] 使用1D Sin-Cos编码: {n_wrist_tokens} tokens", 'green')
            
            pos_embed_1d = get_1d_sincos_pos_embed_from_grid(
                embed_dim=embed_dim,
                pos=np.arange(n_wrist_tokens, dtype=np.float32),
                temperature=10000.0
            )
            pos_embed_data[0, start_idx:start_idx + n_wrist_tokens, :] = torch.from_numpy(pos_embed_1d).float()
            cprint(f"[Wrist] ✓ 1D位置编码已注入: tokens [{start_idx}:{start_idx + n_wrist_tokens}]", 'green')
            start_idx += n_wrist_tokens
        
        # === 3. Tactile: 1D Sin-Cos编码 ===
        if 'tactile' in modality_info and modality_info['tactile'] > 0:
            n_tactile_tokens = modality_info['tactile']
            cprint(f"[Tactile] 使用1D Sin-Cos编码: {n_tactile_tokens} tokens", 'green')
            
            pos_embed_1d = get_1d_sincos_pos_embed_from_grid(
                embed_dim=embed_dim,
                pos=np.arange(n_tactile_tokens, dtype=np.float32),
                temperature=10000.0
            )
            pos_embed_data[0, start_idx:start_idx + n_tactile_tokens, :] = torch.from_numpy(pos_embed_1d).float()
            cprint(f"[Tactile] ✓ 1D位置编码已注入: tokens [{start_idx}:{start_idx + n_tactile_tokens}]", 'green')
            start_idx += n_tactile_tokens
        
        # === 4. Proprio: 1D Sin-Cos编码 ===
        if 'proprio' in modality_info and modality_info['proprio'] > 0:
            n_proprio_tokens = modality_info['proprio']
            cprint(f"[Proprio] 使用1D Sin-Cos编码: {n_proprio_tokens} tokens", 'green')
            
            pos_embed_1d = get_1d_sincos_pos_embed_from_grid(
                embed_dim=embed_dim,
                pos=np.arange(n_proprio_tokens, dtype=np.float32),
                temperature=10000.0
            )
            pos_embed_data[0, start_idx:start_idx + n_proprio_tokens, :] = torch.from_numpy(pos_embed_1d).float()
            cprint(f"[Proprio] ✓ 1D位置编码已注入: tokens [{start_idx}:{start_idx + n_proprio_tokens}]", 'green')
            start_idx += n_proprio_tokens
        
        # 验证所有token都被覆盖
        if start_idx != L_total:
            logger.warning(f"[DiTXGateAttn] 位置编码覆盖不完整: {start_idx}/{L_total} tokens")
        else:
            cprint(f"[DiTXGateAttn] ✅ 位置编码初始化完成: {L_total} tokens 全部覆盖", 'green')
        
        # 标记已初始化
        self._pos_embed_initialized = True
    
    def initialize_weights(self):
        # DiTXGateAttnBlock 使用 FlashSelfAttention，结构不同，需要适配初始化
        for block in self.blocks:
            # Initialize FlashSelfAttention (qkv projection)
            if hasattr(block.self_attn, 'qkv'):
                nn.init.xavier_uniform_(block.self_attn.qkv.weight)
                if block.self_attn.qkv.bias is not None:
                    nn.init.zeros_(block.self_attn.qkv.bias)
            
            # Initialize self-attention output projection
            if hasattr(block.self_attn, 'proj'):
                nn.init.xavier_uniform_(block.self_attn.proj.weight)
                if block.self_attn.proj.bias is not None:
                    nn.init.zeros_(block.self_attn.proj.bias)
            
            # Initialize CrossAttention (q and kv projections)
            if hasattr(block.cross_attn, 'q'):
                nn.init.xavier_uniform_(block.cross_attn.q.weight)
                if block.cross_attn.q.bias is not None:
                    nn.init.zeros_(block.cross_attn.q.bias)
            
            if hasattr(block.cross_attn, 'kv'):
                nn.init.xavier_uniform_(block.cross_attn.kv.weight)
                if block.cross_attn.kv.bias is not None:
                    nn.init.zeros_(block.cross_attn.kv.bias)
            
            # Initialize cross-attention output projection
            if hasattr(block.cross_attn, 'proj'):
                nn.init.xavier_uniform_(block.cross_attn.proj.weight)
                if block.cross_attn.proj.bias is not None:
                    nn.init.zeros_(block.cross_attn.proj.bias)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        

        # Initialize input emb by normal distribution:
        nn.init.normal_(self.input_emb.weight, std=0.02)
        nn.init.constant_(self.input_emb.bias, 0) if self.input_emb.bias is not None else None

        # Initialize pos emb by normal distribution:
        nn.init.normal_(self.pos_emb, std=0.02)       
        
        # Initialize visual condition pos emb (important for distinguishing different tokens)
        nn.init.normal_(self.vis_cond_pos_embed, std=0.02) 

        # Initialize diffusion step encoder:
        for layer in self.flow_timestep_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # Initialize diffusion target_t encoder:
        for layer in self.flow_target_t_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # Initialize conditional observation embedding:
        nn.init.normal_(self.vis_cond_obs_emb.weight, std=0.02)
        nn.init.constant_(self.vis_cond_obs_emb.bias, 0) if self.vis_cond_obs_emb.bias is not None else None

        # Initialize the adapter for timestep and target_t
        nn.init.normal_(self.timestep_target_t_adaptor.weight, std=0.02)
        nn.init.constant_(self.timestep_target_t_adaptor.bias, 0)

        if self.language_conditioned:
            # Initialize the language condition adapter
            nn.init.normal_(self.lang_adaptor[0].weight, std=0.02)
            nn.init.constant_(self.lang_adaptor[0].bias, 0) if self.lang_adaptor[0].bias is not None else None
            nn.init.normal_(self.lang_adaptor[-1].weight, std=0.02)
            nn.init.constant_(self.lang_adaptor[-1].bias, 0) if self.lang_adaptor[-1].bias is not None else None
        
        if self.pre_norm_modality:
            # Initialize the adaptive layer norm for visual condition
            nn.init.zeros_(self.vis_norm.cond_linear.weight)
            nn.init.constant_(self.vis_norm.cond_linear.bias[:self.hidden_dim], 1.)
            nn.init.zeros_(self.vis_norm.cond_linear.bias[self.hidden_dim:])
            if self.language_conditioned:
                # Initialize the adaptive layer norm for language condition
                nn.init.zeros_(self.lang_norm.cond_linear.weight)
                nn.init.constant_(self.lang_norm.cond_linear.bias[:self.hidden_dim], 1.)
                nn.init.zeros_(self.lang_norm.cond_linear.bias[self.hidden_dim:])

        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        
        🔥 Gate-Attention保护机制（高优先级）:
        为了防止Gate Collapse，以下参数**强制排除**Weight Decay：
        1. cross_attn.q: Query投影层（生成gate score的核心）
        2. modality_bias或_modality_bias_tensor: 模态特定的gate初始化偏置
        3. 任何包含"gate"关键词的参数（门控相关）
        
        这些参数如果被Weight Decay惩罚，会导致门控值趋向于0，引发Gate Collapse。
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RmsNorm)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                # 🔥 **高优先级**: Gate相关参数强制保护（必须排在所有判断之前）
                # 防止Gate Collapse: 如果参数名包含gate关键词，强制加入no_decay
                gate_related_keywords = ['cross_attn.q', 'modality_bias', '_modality_bias_tensor', 'gate']
                is_gate_related = any(keyword in fpn for keyword in gate_related_keywords)
                
                if is_gate_related:
                    # Gate相关参数**绝对不能**应用Weight Decay
                    no_decay.add(fpn)
                    cprint(f"[Gate Protection] 保护参数: {fpn} (强制关闭Weight Decay)", "yellow")
                    continue  # 跳过后续判断，确保不会被加入decay组
                
                # 原有的参数分组逻辑（仅处理非gate参数）
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

        # 🔥 统计Gate保护的参数数量
        gate_protected_params = [pn for pn in no_decay if any(kw in pn for kw in ['cross_attn.q', 'modality_bias', 'gate'])]
        if gate_protected_params:
            cprint(f"[Gate Protection] 共保护 {len(gate_protected_params)} 个Gate相关参数，免受Weight Decay惩罚", "green")
            cprint(f"[Gate Protection] 保护的参数列表:", "green")
            for pn in sorted(gate_protected_params):
                cprint(f"  - {pn}", "green")

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

    # ========= Gate-Attention Statistics =========
    def get_gate_stats(self):
        """
        获取所有 blocks 的 Gate-Attention 全局统计信息（用于wandb记录）
        
        Returns:
            dict: 包含以下全局指标：
                - gate/mean_activation: 所有层的平均门控激活值
                - gate/std_activation: 所有层门控激活的标准差
                - gate/saturation_high_ratio: 高饱和度比例 (>0.9)
                - gate/saturation_low_ratio: 低饱和度比例 (<0.1)
                - gate/layer_variance: 层间门控激活的方差（衡量层间差异）
                - gate/modality_{modality}_mean: 各模态的平均门控值
                - gate/early_vs_late: 早期层(前1/3)与后期层(后1/3)的门控差异
        """
        if self.gate_type == 'none':
            return None
        
        all_layer_stats = []
        modality_stats = {}
        
        # 收集所有层的统计信息
        for i, block in enumerate(self.blocks):
            block_stats = block.get_gate_stats()
            if block_stats:
                all_layer_stats.append({
                    'layer_idx': i,
                    'mean': block_stats.get('gate_activation_mean', 0),
                    'std': block_stats.get('gate_activation_std', 0),
                    'saturation_high': block_stats.get('gate_saturation_high', 0),
                    'saturation_low': block_stats.get('gate_saturation_low', 0),
                })
                
                # 收集模态统计
                for key, value in block_stats.items():
                    if key.startswith('modality_gate_'):
                        modality = key.replace('modality_gate_', '')
                        if modality not in modality_stats:
                            modality_stats[modality] = []
                        modality_stats[modality].append(value)
        
        if not all_layer_stats:
            return None
        
        # 计算全局统计指标
        global_stats = {}
        
        # 1. 全局平均门控激活
        mean_activations = [s['mean'] for s in all_layer_stats]
        global_stats['gate/mean_activation'] = np.mean(mean_activations)
        global_stats['gate/std_activation'] = np.mean([s['std'] for s in all_layer_stats])
        
        # 2. 饱和度统计
        global_stats['gate/saturation_high_ratio'] = np.mean([s['saturation_high'] for s in all_layer_stats])
        global_stats['gate/saturation_low_ratio'] = np.mean([s['saturation_low'] for s in all_layer_stats])
        
        # 3. 层间方差（衡量不同层的门控差异）
        global_stats['gate/layer_variance'] = np.var(mean_activations)
        
        # 4. 早期层 vs 后期层对比
        n_layers = len(all_layer_stats)
        early_third = n_layers // 3
        late_third = 2 * n_layers // 3
        
        early_mean = np.mean([s['mean'] for s in all_layer_stats[:early_third]]) if early_third > 0 else 0
        late_mean = np.mean([s['mean'] for s in all_layer_stats[late_third:]]) if late_third < n_layers else 0
        global_stats['gate/early_vs_late_diff'] = late_mean - early_mean
        global_stats['gate/early_layers_mean'] = early_mean
        global_stats['gate/late_layers_mean'] = late_mean
        
        # 5. 模态特定的门控值
        for modality, values in modality_stats.items():
            global_stats[f'gate/modality_{modality}_mean'] = np.mean(values)
            global_stats[f'gate/modality_{modality}_std'] = np.std(values)
        
        return global_stats
    
    # ========= Attention Weight Recording =========
    def set_record_attn(self, record: bool):
        """Enable/disable cross-attention weight recording for all blocks"""
        for block in self.blocks:
            if hasattr(block, 'set_record_attn'):
                block.set_record_attn(record)
    
    def get_cross_attn_weights(self):
        """
        Get cross-attention weights from all blocks.
        
        Returns:
            List of attention weights, one per block.
            Each weight has shape (B, num_heads, N_action, L_context)
        """
        weights = []
        for block in self.blocks:
            if hasattr(block, 'get_cross_attn_weights'):
                w = block.get_cross_attn_weights()
                if w is not None:
                    weights.append(w)
        return weights
    
    def get_attn_stats(self, modality_info: dict = None):
        """
        Compute simplified global attention statistics for monitoring.
        
        🔥 重要: 此方法需要从encoder获取modality_info来正确解析token位置
        
        Args:
            modality_info: 模态信息字典，来自encoder.get_modality_info()
                          例如: {'head': 2, 'rgb_wrist': 4, 'tactile': 4, 'proprio': 2}
                          表示各模态的token数量
        
        Returns:
            dict with global attention statistics (easy to monitor):
                - entropy: normalized attention entropy (0=focused, 1=uniform)
                - entropy_early_late_diff: early layers - late layers entropy
                - modality_rgb: attention ratio on RGB cameras (head + wrist)
                - modality_tactile: attention ratio on tactile sensors
                - modality_proprio: attention ratio on proprioception
                - modality_head: attention ratio on head camera (详细)
                - modality_wrist: attention ratio on wrist cameras (详细)
        """
        weights = self.get_cross_attn_weights()
        if not weights:
            return None
        
        n_layers = len(weights)
        # Stack all layers: (n_layers, B, num_heads, N_action, L_context)
        all_weights = torch.stack(weights, dim=0)
        L_context = all_weights.shape[-1]
        
        # === 1. Global Entropy ===
        # Average over all dimensions except L_context, then compute entropy
        attn_avg = all_weights.mean(dim=(0, 1, 2, 3))  # (L_context,)
        entropy = -torch.sum(attn_avg * torch.log(attn_avg + 1e-8))
        max_entropy = torch.log(torch.tensor(float(L_context), device=attn_avg.device))
        entropy_normalized = (entropy / max_entropy).item()
        
        # === 2. Early vs Late Layers Entropy Diff ===
        mid = n_layers // 2
        early_attn = torch.stack(weights[:mid], dim=0).mean(dim=(0, 1, 2, 3))  # (L_context,)
        late_attn = torch.stack(weights[mid:], dim=0).mean(dim=(0, 1, 2, 3))   # (L_context,)
        
        early_entropy = -torch.sum(early_attn * torch.log(early_attn + 1e-8))
        late_entropy = -torch.sum(late_attn * torch.log(late_attn + 1e-8))
        entropy_diff = ((early_entropy - late_entropy) / max_entropy).item()
        
        stats = {
            'entropy': entropy_normalized,
            'entropy_early_late_diff': entropy_diff,  # positive = early more uniform
        }
        
        # === 3. Modality-level Attention (使用实际的token组织) ===
        if modality_info is not None:
            # 🔥 Token组织方式（来自TimmMultimodalEncoder._forward_token_sequence）:
            # [head tokens, rgb_wrist tokens, tactile tokens, proprio tokens]
            # 例如: [head×2, wrist×4, tactile×4, proprio×2] = 12 tokens
            
            start_idx = 0
            head_attn = 0.0
            wrist_attn = 0.0
            tactile_attn = 0.0
            proprio_attn = 0.0
            
            # Head相机
            if 'head' in modality_info and modality_info['head'] > 0:
                n_head = modality_info['head']
                head_attn = attn_avg[start_idx:start_idx + n_head].sum().item()
                start_idx += n_head
            
            # Wrist相机 (RGB)
            if 'rgb_wrist' in modality_info and modality_info['rgb_wrist'] > 0:
                n_wrist = modality_info['rgb_wrist']
                wrist_attn = attn_avg[start_idx:start_idx + n_wrist].sum().item()
                start_idx += n_wrist
            
            # 触觉传感器
            if 'tactile' in modality_info and modality_info['tactile'] > 0:
                n_tactile = modality_info['tactile']
                tactile_attn = attn_avg[start_idx:start_idx + n_tactile].sum().item()
                start_idx += n_tactile
            
            # 本体感知
            if 'proprio' in modality_info and modality_info['proprio'] > 0:
                n_proprio = modality_info['proprio']
                proprio_attn = attn_avg[start_idx:start_idx + n_proprio].sum().item()
                start_idx += n_proprio
            
            # 聚合RGB = head + wrist
            rgb_attn = head_attn + wrist_attn
            
            # 保存统计信息
            stats['modality_rgb'] = rgb_attn
            stats['modality_head'] = head_attn
            stats['modality_wrist'] = wrist_attn
            stats['modality_tactile'] = tactile_attn
            stats['modality_proprio'] = proprio_attn
            
            # 验证总和约为1.0
            total_attn = rgb_attn + tactile_attn + proprio_attn
            if abs(total_attn - 1.0) > 0.01:
                logger.warning(f"注意力总和偏离1.0: {total_attn:.4f}")
        
        return stats

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            target_t: Union[torch.Tensor, float, int], 
            vis_cond: torch.Tensor,
            lang_cond: Union[torch.Tensor, list, str] = None,
            **kwargs):
        """
        Forward pass of the DiTXGateAttn model.
        Input:
            x: (B,T,input_dim)
            timestep: (B,) or int, maniflow time step t
            target_t: (B,) or float, the target absolute or relative time for the consistency flow training process
            vis_cond: (B,T, vis_cond_dim) 
            lang_cond: (B,) or list of strings, language condition input
            **kwargs: additional arguments
        output: 
            action: (B,T,output_dim)
        """

        # process input
        input_emb = self.input_emb(sample) # (B, T, n_emb)
        x = input_emb + self.pos_emb # (B, T, n_emb)
 

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
        

        # 2. target_t
        target_ts = target_t
        if not torch.is_tensor(target_ts): 
            target_ts = torch.tensor([target_ts], dtype=torch.float32, device=sample.device)    
        elif torch.is_tensor(target_ts) and len(target_ts.shape) == 0:
            target_ts = target_ts[None].to(sample.device)
        target_ts = target_ts.expand(sample.shape[0])
        target_t_embed = self.flow_target_t_encoder(target_ts) # (B, n_emb)
        
        time_c = torch.cat([timestep_embed, target_t_embed], dim=-1) # (B, 2*n_emb)
        time_c = self.timestep_target_t_adaptor(time_c) # (B, n_emb)
        

        # 3. visual condition
        vis_con_obs_emb = self.vis_cond_obs_emb(vis_cond) # (B, L, n_emb)
        vis_cond_pos_embed = self.vis_cond_pos_embed[:, :vis_cond.shape[1]]
        context_c = vis_con_obs_emb + vis_cond_pos_embed # (B, L, n_emb)
        if self.pre_norm_modality:
            context_c = self.vis_norm(context_c, time_c)


        # 4. language condition
        if self.language_conditioned:
            assert lang_cond is not None
            lang_c = self.encode_text_input_T5(lang_cond, output_type="token", device=sample.device) # (B, L_lang, 512)
            lang_c = self.lang_adaptor(lang_c) # (B, L, D) or (B, D)
            if self.pre_norm_modality:
                lang_c = self.lang_norm(lang_c, time_c)
            context_c = torch.cat([context_c, lang_c], dim=1) # (B, L + L_lang, n_emb)


        # 首次前向传播时自动初始化gate bias和位置编码
        if not self._gate_bias_initialized and self.gate_type != 'none':
            modality_info = kwargs.get('modality_info', None)
            if modality_info is not None:
                self._initialize_gate_bias(modality_info)
                self._gate_bias_initialized = True
        
        # 🔥 延迟初始化vis_cond_pos_embed（基于模态感知的2D Sin-Cos编码）
        if not self._pos_embed_initialized:
            modality_info = kwargs.get('modality_info', None)
            head_grid_size = kwargs.get('head_grid_size', None)
            if modality_info is not None:
                self._initialize_vis_pos_embed(modality_info, head_grid_size)
                self._pos_embed_initialized = True

        # 5. transformer blocks with Gate-Attention
        for block in self.blocks:
            x = block(x, time_c, context_c) # (B, T, n_emb)


        # 6. head
        x = self.final_layer(x)
       
        # (B, T, output_dim)
        x = x[:, -self.horizon:] # (B, T, out_channels)
        
        return x

if __name__ == "__main__":
    # Example usage of DiTXGateAttn model
    torch.manual_seed(0)  # For reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.randn(2, 10, 16).to(device)  # Batch size 2, horizon 10, input_dim 16
    timestep = torch.tensor([1, 2]).to(device)  # Example timesteps for each sample in the batch
    target_t = torch.tensor([0.1, 0.2]).to(device)  # Example target_ts for each sample in the batch
    vis_cond = torch.randn(2, 256, 256).to(device)  # Visual condition
    lang_cond = ["This is a test sentence.", "Another test sentence."]
    model = DiTXGateAttn(
        input_dim=16,
        output_dim=16,
        horizon=10,
        n_obs_steps=2,
        cond_dim=256,
        visual_cond_len=128,
        diffusion_timestep_embed_dim=256,
        diffusion_target_t_embed_dim=256,
        n_layer=2,  # Reduced for testing
        n_head=8,
        n_emb=768,
        mlp_ratio=4.0,
        p_drop_attn=0.1,
        gate_type='elementwise',  # 测试 elementwise gate
        language_conditioned=True,
        pre_norm_modality=True,
    )
    model = model.to(device)
    output = model(sample, timestep, target_t, vis_cond, lang_cond)
    print("Output shape:", output.shape)  # Should be (2, 10, 16)
    assert output.shape == (2, 10, 16), "Output shape mismatch!"
    
    # Test gate stats
    gate_stats = model.get_gate_stats()
    if gate_stats:
        print("Gate statistics:", gate_stats)
    
    # Check if the model is initialized correctly
    logger.info("DiTXGateAttn model initialized and tested successfully.")
