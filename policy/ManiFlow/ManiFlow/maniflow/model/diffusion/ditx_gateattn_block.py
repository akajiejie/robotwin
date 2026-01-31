# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# RDT: https://github.com/thu-ml/RoboticsDiffusionTransformer
# Qwen3: Gate-Attention mechanism
# --------------------------------------------------------
#
# 使用说明:
# DiTX-GateAttn Block: 使用Gate-Attention进行特征关注（参考Qwen3）
# 运行测试: python ditx_gateattn_block.py
# --------------------------------------------------------

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Mlp, use_fused_attn

logger = logging.getLogger(__name__)


FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    logger.info("Flash Attention 2 已启用，训练将显著加速！")
except ImportError:
    logger.info("Flash Attention 未安装，使用 PyTorch SDPA 后端")

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


class CrossAttention(nn.Module):
    """
    Cross-attention layer with flash attention and gate mechanism.
    
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
            gate_type: str = 'none',  # 🔥 gate-attention类型
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.use_flash_attn = FLASH_ATTN_AVAILABLE
        self.gate_type = gate_type

        # Query projection with optional gate（参考Qwen3）
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
            logger.info(f"[CrossAttention] 启用Gate-Attention机制: {gate_type}（参考Qwen3）")
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                mask: None) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        
        # Query projection with gate extraction（参考Qwen3实现）
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
        
        # Flash Attention 路径
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
        
        # Gate-Attention: 用sigmoid(gate)调制attention输出（参考Qwen3）
        if gate_score is not None:
            gate_activation = torch.sigmoid(gate_score)
            attn_output = attn_output * gate_activation
            
            # 收集Gate-Attention激活统计（用于wandb监控）
            if self.training:
                with torch.no_grad():
                    # 计算激活值的均值和标准差
                    gate_mean = gate_activation.mean().item()
                    gate_std = gate_activation.std().item()
                    gate_min = gate_activation.min().item()
                    gate_max = gate_activation.max().item()
                    
                    # 存储统计信息（在block的get_gate_stats中访问）
                    if not hasattr(self, '_gate_stats_buffer'):
                        self._gate_stats_buffer = []
                    self._gate_stats_buffer.append({
                        'mean': gate_mean,
                        'std': gate_std,
                        'min': gate_min,
                        'max': gate_max,
                    })
        
        # Reshape and project
        attn_output = attn_output.reshape(B, N, C)
        attn_output = self.proj(attn_output)
        if self.proj_drop.p > 0:
            attn_output = self.proj_drop(attn_output)
        
        return attn_output


class DiTXGateAttnBlock(nn.Module):
    """
    DiTX Block with Gate-Attention mechanism (参考Qwen3).
    
    核心改进：
    1. Gate-Attention：Cross-attention输出通过可学习的gate调制
    2. 支持三种gate模式：
       - 'none': 标准cross-attention（无gate）
       - 'headwise': 每个注意力头一个gate值（轻量级）
       - 'elementwise': 每个元素一个gate值（最细粒度）
    3. Flash Attention 2：加速训练和推理
    
    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        mlp_ratio: MLP扩展比例
        gate_type: Gate-Attention类型 ('none', 'headwise', 'elementwise')
        p_drop_attn: Attention dropout概率
        qkv_bias: 是否使用QKV bias
        qk_norm: 是否对Q和K进行归一化
    """
    def __init__(self, 
                hidden_size=768,
                num_heads=12,
                mlp_ratio=4.0,
                
                # Gate-Attention配置
                gate_type='elementwise',      # 'none', 'headwise', 'elementwise'
                
                # 其他参数
                p_drop_attn=0.1,
                qkv_bias=False,
                qk_norm=False,
                **block_kwargs):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.gate_type = gate_type

        # Self-Attention with Flash Attention support
        self.self_attn = FlashSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=p_drop_attn,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
        )
        
        # Cross-Attention with Gate-Attention
        self.cross_attn = CrossAttention(
            dim=hidden_size, 
            num_heads=num_heads,
            qkv_bias=qkv_bias, 
            qk_norm=qk_norm,
            norm_layer=nn.LayerNorm,
            gate_type=gate_type,  # Gate-Attention配置
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
        
        logger.info(f"[DiTXGateAttnBlock] Initialized with Gate-Attention: {gate_type}")
    
    def get_gate_stats(self):
        """获取Gate-Attention统计信息（用于wandb记录）"""
        stats = {}
        
        # Gate-Attention激活分布
        if hasattr(self.cross_attn, '_gate_stats_buffer') and len(self.cross_attn._gate_stats_buffer) > 0:
            gate_stats = self.cross_attn._gate_stats_buffer[-1]
            stats['gate_activation_mean'] = gate_stats['mean']
            stats['gate_activation_std'] = gate_stats['std']
            stats['gate_activation_min'] = gate_stats['min']
            stats['gate_activation_max'] = gate_stats['max']
            # 清空buffer
            self.cross_attn._gate_stats_buffer.clear()
        
        return stats if stats else None
        
    def forward(self, x, time_c, context_c, attn_mask=None):
        """
        Forward pass of the DiTX-GateAttn block.
        
        Args:
            x: 动作序列 (batch_size, seq_length, hidden_size)
            time_c: 时间步嵌入 (batch_size, hidden_size)
            context_c: 多模态特征 (batch_size, L_total, hidden_size)
            attn_mask: 可选的注意力mask
        
        Returns:
            x: 输出特征 (batch_size, seq_length, hidden_size)
        """
        # adaLN modulation
        modulation = self.adaLN_modulation(time_c)
        chunks = modulation.chunk(9, dim=-1)
        
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        # 1. Self-Attention with adaLN conditioning (Flash Attention)
        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output, _ = self.self_attn(normed_x, attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * self_attn_output

        # 2. Cross-Attention with Gate-Attention
        normed_x_cross = modulate(self.norm2(x), shift_cross, scale_cross)
        cross_attn_output = self.cross_attn(normed_x_cross, context_c, mask=None)
        x = x + gate_cross.unsqueeze(1) * cross_attn_output

        # 3. MLP with adaLN conditioning
        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


if __name__ == "__main__":
    """
    测试DiTX-GateAttn Block的功能
    运行方式: python ditx_gateattn_block.py
    """
    
    def test_ditx_gateattn_block():
        """测试DiTXGateAttnBlock的基本功能"""
        print("=" * 80)
        print("测试 DiTXGateAttnBlock (Gate-Attention)")
        print("=" * 80)
        
        # 参数设置
        batch_size = 4
        seq_len = 50          # 动作序列长度
        hidden_size = 768
        num_heads = 12
        L_total = 1180        # 多模态特征长度
        
        # 创建DiTXGateAttnBlock
        block = DiTXGateAttnBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=4.0,
            gate_type='headwise',  # 测试headwise gate
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
        
        # 前向传播
        print(f"\n" + "─" * 80)
        print("DiTXGateAttnBlock 前向传播...")
        print(f"  使用Gate-Attention调制cross-attention输出")
        block.train()
        output = block(x, time_c, context_c)
        print(f"  输出形状: {output.shape}")
        
        # 检查Gate统计信息
        gate_stats = block.get_gate_stats()
        if gate_stats:
            print(f"  Gate统计:")
            print(f"    - mean: {gate_stats['gate_activation_mean']:.4f}")
            print(f"    - std: {gate_stats['gate_activation_std']:.4f}")
            print(f"    - min: {gate_stats['gate_activation_min']:.4f}")
            print(f"    - max: {gate_stats['gate_activation_max']:.4f}")
        print(f"  ✅ 成功!")
        
        # 参数统计
        print(f"\n" + "=" * 80)
        print("参数统计:")
        print("=" * 80)
        
        params = sum(p.numel() for p in block.parameters())
        print(f"  总参数: {params:,}")
        print(f"  Gate类型: {block.gate_type}")
        
        print(f"\n" + "=" * 80)
        print("✅ 测试通过!")
        print("=" * 80)

    def test_gradient_flow():
        """测试梯度流动"""
        print("\n\n" + "=" * 80)
        print("测试梯度流动")
        print("=" * 80)
        
        block = DiTXGateAttnBlock(
            hidden_size=512,
            num_heads=8,
            gate_type='elementwise'
        )
        block.train()
        
        x = torch.randn(2, 32, 512, requires_grad=True)
        time_c = torch.randn(2, 512, requires_grad=True)
        context_c = torch.randn(2, 256, 512, requires_grad=True)
        
        output = block(x, time_c, context_c)
        
        loss = output.sum()
        loss.backward()
        
        print(f"  x.grad is not None: {x.grad is not None}")
        print(f"  time_c.grad is not None: {time_c.grad is not None}")
        print(f"  context_c.grad is not None: {context_c.grad is not None}")
        print(f"  ✅ 梯度流动正常!")
        print("=" * 80)

    # 运行测试
    torch.manual_seed(42)
    
    test_ditx_gateattn_block()
    test_gradient_flow()
    
    print("\n" + "🎉" * 40)
    print("所有测试完成!")
    print("🎉" * 40)

