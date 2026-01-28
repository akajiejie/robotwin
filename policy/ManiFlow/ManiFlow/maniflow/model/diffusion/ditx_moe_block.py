# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# RDT: https://github.com/thu-ml/RoboticsDiffusionTransformer
# --------------------------------------------------------
#
# 使用说明:
# 运行测试: python ditx-moe_block.py (需安装依赖: torch, einops, timm)
# --------------------------------------------------------

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Mlp, use_fused_attn
from maniflow.model.diffusion.ditx_block import DiTXBlock
from maniflow.model.gate.MoEgate import SparseMoeBlock

logger = logging.getLogger(__name__)


FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    logger.info("🚀 Flash Attention 2 已启用，训练将显著加速！")
except ImportError:
    logger.info("⚠️ Flash Attention 未安装，使用 PyTorch SDPA 后端")

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FlashSelfAttention(nn.Module):
    """
    Self-Attention with Flash Attention 2 support.
    
    当 flash-attn 可用时使用 Flash Attention 2，否则回退到 PyTorch SDPA。
    比 nn.MultiheadAttention 更快，特别是在长序列上。
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.use_flash_attn = FLASH_ATTN_AVAILABLE
        
    def forward(self, x: torch.Tensor, attn_mask=None):
        """
        Args:
            x: Input tensor of shape (B, N, C)
            attn_mask: Optional attention mask (not supported with Flash Attention)
        
        Returns:
            output: (B, N, C)
            attn_weights: None (Flash Attention doesn't return weights)
        """
        B, N, C = x.shape
        
        # QKV projection: (B, N, 3*C) -> (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        
        if self.use_flash_attn and x.is_cuda and x.dtype in [torch.float16, torch.bfloat16]:
            # 🚀 Flash Attention 2 路径
            # flash_attn_func 需要 (B, N, num_heads, head_dim) 格式
            q, k, v = qkv.unbind(2)  # 3 x (B, N, num_heads, head_dim)
            
            # QK Normalization
            q, k = self.q_norm(q), self.k_norm(k)
            
            # Flash Attention (自动处理 causal=False)
            dropout_p = self.attn_drop.p if self.training else 0.
            out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=False)
            # out: (B, N, num_heads, head_dim)
            
        else:
            # PyTorch SDPA 后端（支持 FP32）
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
            q, k, v = qkv.unbind(0)
            
            # QK Normalization
            q, k = self.q_norm(q), self.k_norm(k)
            
            # Scaled dot-product attention
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            # out: (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim)
            out = out.transpose(1, 2)
        
        # Reshape and project
        out = out.reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out, None  # 返回 None 作为 attn_weights，保持接口兼容


class AdaptiveLayerNorm(nn.Module):
    def __init__(
        self,
        dim,
        dim_cond,
    ):
        super().__init__()

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
 
        self.cond_linear = nn.Linear(dim_cond, dim * 2)

        self.cond_modulation = nn.Sequential(
            Rearrange('b d -> b 1 d'),
            nn.SiLU(),
            self.cond_linear
        )

        # Initialize the weights and biases of the conditional linear layer
        nn.init.zeros_(self.cond_linear.weight)
        nn.init.constant_(self.cond_linear.bias[:dim], 1.)
        nn.init.zeros_(self.cond_linear.bias[dim:])

    def forward(
        self,
        x,
        cond = None
    ):
        x = self.ln(x)
        gamma, beta = self.cond_modulation(cond).chunk(2, dim = -1)
        x = x * gamma + beta

        return x

class CrossAttention(nn.Module):
    """
    Cross-attention layer with flash attention and optional gate mechanism.
    
    支持两种门控模式（参考Qwen3的gated attention）：
    - 'none': 无门控（标准cross-attention）
    - 'headwise': 每个注意力头一个gate值（轻量级）
    - 'elementwise': 每个元素一个gate值（最细粒度，参考Qwen3）
    
    Args:
        gate_type: 门控类型 ('none', 'headwise', 'elementwise')
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
            gate_type: str = 'none',  # 🔥 新增：gate-attention类型
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.use_flash_attn = FLASH_ATTN_AVAILABLE
        self.gate_type = gate_type

        # 🔥 Query projection with optional gate（参考Qwen3）
        if gate_type == 'headwise':
            # 每个头一个gate: q_dim + num_heads
            self.q = nn.Linear(dim, dim + num_heads, bias=qkv_bias)
        elif gate_type == 'elementwise':
            # 每个元素一个gate: q_dim * 2（与Qwen3一致）
            self.q = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            # 标准query
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if gate_type != 'none':
            logger.info(f"[CrossAttention] 🔥 启用Gate-Attention机制: {gate_type}（参考Qwen3）")
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                mask: None) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        
        # 🔥 Query projection with gate extraction（参考Qwen3实现）
        q_output = self.q(x)
        
        if self.gate_type == 'headwise':
            # Headwise gate: 每个头一个gate值
            # q_output: (B, N, dim + num_heads)
            q_output = q_output.view(B, N, self.num_heads, -1)
            q, gate_score = torch.split(q_output, [self.head_dim, 1], dim=-1)
            # gate_score: (B, N, num_heads, 1)
            q = q.permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
            
        elif self.gate_type == 'elementwise':
            # Elementwise gate: 每个元素一个gate值（与Qwen3一致）
            # q_output: (B, N, dim * 2)
            q_output = q_output.view(B, N, self.num_heads, -1)
            q, gate_score = torch.split(q_output, [self.head_dim, self.head_dim], dim=-1)
            # gate_score: (B, N, num_heads, head_dim)
            q = q.permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
            
        else:
            # 标准模式：无gate
            q = q_output.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            gate_score = None
        
        # Key-Value projection
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)  # k, v: (B, L, num_heads, head_dim)
        
        # 🚀 Flash Attention 路径
        if self.use_flash_attn and x.is_cuda and x.dtype in [torch.float16, torch.bfloat16] and mask is None:
            # Flash Attention 需要 (B, N, num_heads, head_dim) 格式
            # q 当前是 (B, num_heads, N, head_dim)，需要转换
            q = q.transpose(1, 2)  # (B, N, num_heads, head_dim)
            
            # QK Normalization
            q, k = self.q_norm(q), self.k_norm(k)
            
            # Flash Attention cross-attention
            dropout_p = self.attn_drop.p if self.training else 0.
            attn_output = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=False)
            # attn_output: (B, N, num_heads, head_dim)
            
        else:
            # PyTorch SDPA 后端
            # 转换 k, v 到 (B, num_heads, L, head_dim)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            
            # QK Normalization
            q, k = self.q_norm(q), self.k_norm(k)

            # Prepare attn mask (B, L) to mask the condition
            if mask is not None:
                mask = mask.reshape(B, 1, 1, L)
                mask = mask.expand(-1, -1, N, -1)
            
            # Attention computation
            if self.fused_attn:
                attn_output = F.scaled_dot_product_attention(
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
                attn_output = attn @ v
            
            # attn_output: (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim)
            attn_output = attn_output.transpose(1, 2)
        
        # 🔥 Gate-Attention: 用sigmoid(gate)调制attention输出（参考Qwen3）
        if gate_score is not None:
            attn_output = attn_output * torch.sigmoid(gate_score)
        
        # Reshape and project
        attn_output = attn_output.reshape(B, N, C)
        attn_output = self.proj(attn_output)
        if self.proj_drop.p > 0:
            attn_output = self.proj_drop(attn_output)
        
        return attn_output


class DiTXMoEBlock(nn.Module):
    """
    DiTX Block with Token-level Mixture of Experts (MoE) and Gate-Attention.
    
    核心改进：
    1. Token级别路由：每个token独立选择专家，细粒度的特征处理
    2. 时间条件感知：MoE门控感知扩散时间步，根据噪声阶段调整专家选择
    3. 专家自动学习：在不同时间步下关注不同模态的token特征
    4. AdaLN协调：context_c在进入MoE前通过AdaLN感知时间条件
    5. Gate-Attention：Cross-attention输出通过可学习的gate调制（参考Qwen3）
    
    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        mlp_ratio: MLP扩展比例
        use_token_moe: 是否使用token级MoE
        num_experts: MoE专家数量
        num_experts_per_tok: 每个token激活的专家数
        n_shared_experts: 共享专家数量
        moe_aux_loss_alpha: MoE辅助损失权重
        enable_grad_accumulation: 是否启用梯度累积友好模式
        gate_type: Gate-Attention类型 ('none', 'headwise', 'elementwise')
        p_drop_attn: Attention dropout概率
        qkv_bias: 是否使用QKV bias
        qk_norm: 是否对Q和K进行归一化
    """
    def __init__(self, 
                hidden_size=768,
                num_heads=12,
                mlp_ratio=4.0,
                
                # MoE配置
                use_token_moe=True,           # 🔥 改名：强调token级别
                num_experts=8,                # Token级MoE建议8-16个专家（比模态级更多）
                num_experts_per_tok=2,
                n_shared_experts=1,
                moe_aux_loss_alpha=0.01,
                enable_grad_accumulation=False,  # 🔥 梯度累积支持
                
                # Gate-Attention配置
                gate_type='elementwise',      # 🔥 'none', 'headwise', 'elementwise'
                
                # 其他参数
                p_drop_attn=0.1,
                qkv_bias=False,
                qk_norm=False,
                **block_kwargs):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_token_moe = use_token_moe
        self.enable_grad_accumulation = enable_grad_accumulation

        # 🚀 Self-Attention with Flash Attention support
        self.self_attn = FlashSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=p_drop_attn,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
        )
        
        # ⭐ Token级别MoE：每个token独立路由，专家自动学习特征模式
        if use_token_moe:
            self.token_moe = SparseMoeBlock(
                embed_dim=hidden_size,
                mlp_ratio=mlp_ratio,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_shared_experts=n_shared_experts,
                aux_loss_alpha=moe_aux_loss_alpha,
                use_time_cond=True,  # 🔥 启用时间条件感知
                enable_grad_accumulation=enable_grad_accumulation
            )
            # context_c的AdaLN：让MoE输入感知时间条件
            self.context_adaln = AdaptiveLayerNorm(dim=hidden_size, dim_cond=hidden_size)
            logger.info(f"[DiTXMoEBlock] 🔥 Initialized Token-level MoE with {num_experts} experts, "
                       f"top-{num_experts_per_tok}, {n_shared_experts} shared, time_cond=True, "
                       f"grad_accum={enable_grad_accumulation}")
        
        # Cross-Attention with Gate-Attention
        self.cross_attn = CrossAttention(
            dim=hidden_size, 
            num_heads=num_heads,
            qkv_bias=qkv_bias, 
            qk_norm=qk_norm,
            norm_layer=nn.LayerNorm,
            gate_type=gate_type,  # 🔥 传递gate-attention配置
            **block_kwargs
        )
       
        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, 
            hidden_features=mlp_hidden_dim, 
            act_layer=approx_gelu, 
            drop=0.0
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # AdaLN modulation
        modulation_size = 9 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, modulation_size, bias=True)
        )
    
    def reset_moe_accumulation(self):
        """重置MoE的累积统计（在optimizer.step()后调用）"""
        if self.use_token_moe and self.enable_grad_accumulation:
            self.token_moe.reset_gate_accumulation()
    
    def get_moe_stats(self):
        """获取MoE统计信息（用于wandb记录）"""
        if self.use_token_moe and hasattr(self.token_moe, 'moe_stats'):
            return self.token_moe.moe_stats
        return None
        
    def forward(self, x, time_c, context_c, attn_mask=None, modality_lens=None):
        """
        Forward pass of the DiTX-MoE block.
        
        Args:
            x: 动作序列 (batch_size, seq_length, hidden_size)
            time_c: 时间步嵌入 (batch_size, hidden_size)
            context_c: 多模态特征 (batch_size, L_total, hidden_size)
                      包含所有模态的token: [head_tokens, wrist_tokens, tactile_tokens, proprio_tokens]
            attn_mask: 可选的注意力mask
            modality_lens: 模态长度信息（保留接口兼容性，但token级MoE不需要）
        
        Returns:
            x: 输出特征 (batch_size, seq_length, hidden_size)
        """
        # adaLN modulation
        modulation = self.adaLN_modulation(time_c)
        chunks = modulation.chunk(9, dim=-1)
        
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        # 1. Self-Attention with adaLN conditioning (🚀 Flash Attention)
        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output, _ = self.self_attn(normed_x, attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * self_attn_output

        # 2. ⭐ Token级别MoE处理多模态输入特征
        # 每个token独立路由，专家自动学习在不同时间步下关注什么特征
        if self.use_token_moe:
            # 先通过AdaLN让context_c感知时间条件
            context_c_normed = self.context_adaln(context_c, time_c)
            # Token级别MoE处理：(B, L_total, D) -> (B, L_total, D)
            # 每个token独立选择专家，门控由时间条件调制
            context_c_processed = self.token_moe(context_c_normed, time_c)
        else:
            context_c_processed = context_c

        # 3. Cross-Attention with adaLN conditioning
        normed_x_cross = modulate(self.norm2(x), shift_cross, scale_cross)
        cross_attn_output = self.cross_attn(normed_x_cross, context_c_processed, mask=None)
        x = x + gate_cross.unsqueeze(1) * cross_attn_output

        # 4. MLP with adaLN conditioning
        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x

if __name__ == "__main__":
    """
    测试DiTX-MoE Block的功能
    运行方式: python ditx_moe_block.py
    """
    
    def test_ditx_moe_block():
        """测试DiTXMoEBlock的基本功能"""
        print("=" * 80)
        print("测试 DiTXMoEBlock (Token级别MoE + 时间条件感知)")
        print("=" * 80)
        
        # 参数设置
        batch_size = 4
        seq_len = 50          # 动作序列长度
        hidden_size = 768
        num_heads = 12
        
        # 多模态特征长度（真实场景：1180 tokens）
        L_head = 392          # 头部相机: 1相机 × 2T × 196patches
        L_wrist = 784         # 腕部相机: 2相机 × 2T × 196patches  
        L_tactile = 2         # 触觉传感器: 2传感器 × 1patch
        L_proprio = 2         # 本体感知: 2时间步
        L_total = L_head + L_wrist + L_tactile + L_proprio  # 1180 tokens
        
        # 模态长度信息（保留接口兼容性）
        modality_lens = {
            'head': L_head, 
            'wrist': L_wrist + L_tactile,  # 腕部视觉+触觉
            'proprio': L_proprio
        }
        
        # 创建DiTXMoEBlock (Token级MoE + Gate-Attention)
        block_moe = DiTXMoEBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=4.0,
            use_token_moe=True,
            num_experts=8,        # 🔥 Token级MoE建议更多专家
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_aux_loss_alpha=0.01,
            gate_type='headwise',  # 🔥 Gate-Attention（参考Qwen3）
            p_drop_attn=0.1
        )
        
        # 创建原始DiTXBlock作为对比
        block_vanilla = DiTXBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=4.0,
            p_drop_attn=0.1
        )
        
        # 输入数据
        x = torch.randn(batch_size, seq_len, hidden_size)
        time_c = torch.randn(batch_size, hidden_size)
        context_c = torch.randn(batch_size, L_total, hidden_size)
        
        print(f"\n输入形状:")
        print(f"  x (动作序列):    {x.shape}")
        print(f"  time_c (时间):   {time_c.shape}")
        print(f"  context_c (多模态): {context_c.shape}")
        print(f"    └─ 头部: {L_head}, 腕部: {L_wrist}, 本体: {L_proprio}")
        
        # 前向传播 - Token级MoE + Gate-Attention版本
        print(f"\n" + "─" * 80)
        print("DiTXMoEBlock 前向传播 (Token级别路由 + Gate-Attention)...")
        print(f"  🔥 每个token独立选择专家，专家自动学习在不同时间步下关注什么特征")
        print(f"  🔥 Gate-Attention调制cross-attention输出（参考Qwen3）")
        block_moe.train()
        output_moe = block_moe(x, time_c, context_c, modality_lens=modality_lens)
        print(f"  输出形状: {output_moe.shape}")
        
        # 检查MoE统计信息
        moe_stats = block_moe.get_moe_stats()
        if moe_stats:
            print(f"  MoE统计:")
            print(f"    - aux_loss: {moe_stats['aux_loss']:.6f}")
            print(f"    - expert_usage: {moe_stats['expert_usage'].cpu().numpy()}")
            print(f"    - topk_weights_mean: {moe_stats['topk_weights_mean']:.4f}")
            print(f"    - topk_weights_std: {moe_stats['topk_weights_std']:.4f}")
        print(f"  ✅ 成功!")
        
        # 前向传播 - 原始版本
        print(f"\n" + "─" * 80)
        print("DiTXBlock (原始) 前向传播...")
        output_vanilla = block_vanilla(x, time_c, context_c)
        print(f"  输出形状: {output_vanilla.shape}")
        print(f"  ✅ 成功!")
        
        # 参数统计
        print(f"\n" + "=" * 80)
        print("参数对比:")
        print("=" * 80)
        
        params_moe = sum(p.numel() for p in block_moe.parameters())
        params_vanilla = sum(p.numel() for p in block_vanilla.parameters())
        params_diff = params_moe - params_vanilla
        
        print(f"  DiTXMoEBlock:  {params_moe:,} 参数")
        print(f"  DiTXBlock:     {params_vanilla:,} 参数")
        print(f"  增加:          {params_diff:,} 参数 (+{params_diff/params_vanilla*100:.1f}%)")
        
        # 检查MoE模块
        if hasattr(block_moe, 'token_moe'):
            moe_params = sum(p.numel() for p in block_moe.token_moe.parameters())
            print(f"\n  Token-level MoE参数: {moe_params:,}")
            print(f"    ├─ 专家数量: {block_moe.token_moe.num_experts}")
            print(f"    ├─ Top-K: {block_moe.token_moe.num_experts_per_tok}")
            print(f"    ├─ 共享专家: {block_moe.token_moe.n_shared_experts}")
            print(f"    ├─ 时间条件: {block_moe.token_moe.use_time_cond}")
            print(f"    └─ 梯度累积: {block_moe.token_moe.enable_grad_accumulation}")
        
        # 检查Gate-Attention
        if hasattr(block_moe, 'cross_attn'):
            cross_attn_params = sum(p.numel() for p in block_moe.cross_attn.parameters())
            print(f"\n  Cross-Attention参数: {cross_attn_params:,}")
            print(f"    ├─ 注意力头数: {block_moe.cross_attn.num_heads}")
            print(f"    ├─ Head维度: {block_moe.cross_attn.head_dim}")
            print(f"    └─ Gate类型: {block_moe.cross_attn.gate_type} 🔥")
        
        print(f"\n" + "=" * 80)
        print("✅ 所有测试通过!")
        print("=" * 80)


    def test_batch_sizes():
        """测试不同batch size"""
        print("\n\n" + "=" * 80)
        print("测试不同Batch Size (Token级MoE)")
        print("=" * 80)
        
        block = DiTXMoEBlock(
            hidden_size=512,
            num_heads=8,
            use_token_moe=True,
            num_experts=8,
            num_experts_per_tok=2
        )
        block.eval()
        
        modality_lens = {'head': 128, 'wrist': 112, 'proprio': 16}
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 32, 512)
            time_c = torch.randn(batch_size, 512)
            context_c = torch.randn(batch_size, 256, 512)
            
            with torch.no_grad():
                output = block(x, time_c, context_c, modality_lens=modality_lens)
            
            print(f"  Batch size {batch_size}: {output.shape} ✅")
        
        print("=" * 80)


    def test_without_moe():
        """测试关闭MoE的情况"""
        print("\n\n" + "=" * 80)
        print("测试关闭MoE (use_token_moe=False)")
        print("=" * 80)
        
        block = DiTXMoEBlock(
            hidden_size=512,
            num_heads=8,
            use_token_moe=False
        )
        
        x = torch.randn(2, 32, 512)
        time_c = torch.randn(2, 512)
        context_c = torch.randn(2, 256, 512)
        
        output = block(x, time_c, context_c)
        print(f"  输出形状: {output.shape}")
        print(f"  ✅ 关闭MoE模式正常工作!")
        print("=" * 80)
    
    
    def test_gradient_flow():
        """测试梯度流动和MoE辅助损失"""
        print("\n\n" + "=" * 80)
        print("测试梯度流动和MoE辅助损失")
        print("=" * 80)
        
        block = DiTXMoEBlock(
            hidden_size=512,
            num_heads=8,
            use_token_moe=True,
            num_experts=8,
            num_experts_per_tok=2,
            moe_aux_loss_alpha=0.01
        )
        block.train()
        
        x = torch.randn(2, 32, 512, requires_grad=True)
        time_c = torch.randn(2, 512, requires_grad=True)
        context_c = torch.randn(2, 256, 512, requires_grad=True)
        
        modality_lens = {'head': 128, 'wrist': 112, 'proprio': 16}
        output = block(x, time_c, context_c, modality_lens=modality_lens)
        
        # 检查MoE统计
        moe_stats = block.get_moe_stats()
        print(f"  MoE aux_loss: {moe_stats['aux_loss'] if moe_stats else 0.0:.6f}")
        
        loss = output.sum()
        loss.backward()
        
        print(f"  x.grad is not None: {x.grad is not None}")
        print(f"  time_c.grad is not None: {time_c.grad is not None}")
        print(f"  context_c.grad is not None: {context_c.grad is not None}")
        print(f"  ✅ 梯度流动正常!")
        print("=" * 80)


    # 运行所有测试
    torch.manual_seed(42)
    
    test_ditx_moe_block()
    test_batch_sizes()
    test_without_moe()
    test_gradient_flow()
    
    print("\n" + "🎉" * 40)
    print("所有测试完成!")
    print("🎉" * 40)

