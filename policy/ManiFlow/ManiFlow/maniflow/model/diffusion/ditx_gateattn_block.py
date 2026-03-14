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
            x: (B, N, C)
            attn_mask: 可选注意力掩码（Flash Attention 不支持）

        Returns:
            output: (B, N, C)
            attn_weights: None
        """
        B, N, C = x.shape

        # (B, N, 3*C) -> (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        if self.use_flash_attn and x.is_cuda and x.dtype in [torch.float16, torch.bfloat16]:
            q, k, v = qkv.unbind(2)  # 3 x (B, N, num_heads, head_dim)
            q, k = self.q_norm(q), self.k_norm(k)
            dropout_p = self.attn_drop.p if self.training else 0.
            out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=False)
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim)
            out = out.transpose(1, 2)

        out = out.reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, None


class CrossAttention(nn.Module):
    """
    Cross-attention with Flash Attention and gate mechanism.

    Gate 位置：在 Value（V）投影后、注意力计算前进行门控调制。
    即 V_gated = V * sigmoid(gate_score)。

    支持三种 gate 模式：
    - 'none':        无门控（标准 cross-attention）
    - 'headwise':    每个注意力头一个 gate 值，V 投影输出 dim + num_heads
    - 'elementwise': 每个元素一个 gate 值，V 投影输出 dim * 2

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
            gate_type: str = 'none',
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.use_flash_attn = FLASH_ATTN_AVAILABLE
        self.gate_type = gate_type

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)

        # Value 投影：根据 gate_type 决定输出维度
        # - 'none':        输出 dim
        # - 'headwise':    输出 dim + num_heads（每头一个 gate logit）
        # - 'elementwise': 输出 dim * 2（每元素一个 gate logit）
        if gate_type == 'headwise':
            self.v = nn.Linear(dim, dim + num_heads, bias=qkv_bias)
        elif gate_type == 'elementwise':
            self.v = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._record_attn = False
        self._last_attn_weights = None

    def set_record_attn(self, record: bool):
        """开启/关闭注意力权重记录"""
        self._record_attn = record
        if not record:
            self._last_attn_weights = None

    def get_cross_attn_weights(self):
        """获取已记录的 cross-attention 权重"""
        return self._last_attn_weights

    def clear_gate_stats_buffer(self):
        """清空 gate 统计缓存"""
        if hasattr(self, '_gate_stats_buffer'):
            self._gate_stats_buffer.clear()

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask=None) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)  # (B, N, H, D)
        k = self.k(c).reshape(B, L, self.num_heads, self.head_dim)  # (B, L, H, D)

        # Value 投影 + gate 提取
        v_output = self.v(c)  # (B, L, dim) or (B, L, dim+H) or (B, L, dim*2)

        if self.gate_type == 'headwise':
            v_output = v_output.reshape(B, L, self.num_heads, -1)
            v, gate_score = torch.split(v_output, [self.head_dim, 1], dim=-1)
            # v: (B, L, H, D)  gate_score: (B, L, H, 1)
        elif self.gate_type == 'elementwise':
            v_output = v_output.reshape(B, L, self.num_heads, -1)
            v, gate_score = torch.split(v_output, [self.head_dim, self.head_dim], dim=-1)
            # v: (B, L, H, D)  gate_score: (B, L, H, D)
        else:
            v = v_output.reshape(B, L, self.num_heads, self.head_dim)
            gate_score = None

        # V_gated = V * sigmoid(gate_score)，在注意力计算前调制
        if gate_score is not None:
            gate_activation = torch.sigmoid(gate_score)  # (B, L, H, 1 or D)
            v = v * gate_activation

            if self.training:
                with torch.no_grad():
                    if not hasattr(self, '_gate_stats_buffer'):
                        self._gate_stats_buffer = []
                    self._gate_stats_buffer.append({
                        'mean': gate_activation.mean().item(),
                        'std': gate_activation.std().item(),
                        'saturation_high': (gate_activation > 0.9).float().mean().item(),
                        'saturation_low': (gate_activation < 0.1).float().mean().item(),
                    })

        # 注意力计算
        attn_weights_for_recording = None

        if self.use_flash_attn and x.is_cuda and x.dtype in [torch.float16, torch.bfloat16] and mask is None and not self._record_attn:
            q, k = self.q_norm(q), self.k_norm(k)
            dropout_p = self.attn_drop.p if self.training else 0.
            attn_output = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=False)
            # attn_output: (B, N, H, D)
        else:
            # 转换为 (B, H, seq, D) 格式
            q = q.permute(0, 2, 1, 3)  # (B, H, N, D)
            k = k.permute(0, 2, 1, 3)  # (B, H, L, D)
            v = v.permute(0, 2, 1, 3)  # (B, H, L, D)
            q, k = self.q_norm(q), self.k_norm(k)

            if mask is not None:
                mask = mask.reshape(B, 1, 1, L).expand(-1, -1, N, -1)

            if self.fused_attn and not self._record_attn:
                attn_output = F.scaled_dot_product_attention(
                    query=q, key=k, value=v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                    attn_mask=mask,
                )
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, N, L)
                if mask is not None:
                    attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
                attn = attn.softmax(dim=-1)
                if self._record_attn:
                    attn_weights_for_recording = attn.detach()
                if self.attn_drop.p > 0:
                    attn = self.attn_drop(attn)
                attn_output = attn @ v  # (B, H, N, D)

            # (B, H, N, D) -> (B, N, H, D)
            attn_output = attn_output.transpose(1, 2)

        if self._record_attn and attn_weights_for_recording is not None:
            self._last_attn_weights = attn_weights_for_recording

        attn_output = attn_output.reshape(B, N, C)
        attn_output = self.proj(attn_output)
        if self.proj_drop.p > 0:
            attn_output = self.proj_drop(attn_output)

        return attn_output


class DiTXGateAttnBlock(nn.Module):
    """
    DiTX Block with Gate-Attention mechanism.

    核心改进：
    1. Gate-Attention：在 Value（V）投影后、注意力计算前进行门控调制
       即 V_gated = V * sigmoid(gate_score)
    2. 支持三种 gate 模式：
       - 'none':        标准 cross-attention（无 gate）
       - 'headwise':    每个注意力头一个 gate 值（轻量级）
       - 'elementwise': 每个元素一个 gate 值（最细粒度）
    3. Flash Attention 2：加速训练和推理

    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        mlp_ratio: MLP 扩展比例
        gate_type: Gate-Attention 类型 ('none', 'headwise', 'elementwise')
        p_drop_attn: Attention dropout 概率
        qkv_bias: 是否使用 QKV bias
        qk_norm: 是否对 Q 和 K 进行归一化
    """
    def __init__(self,
                 hidden_size=768,
                 num_heads=12,
                 mlp_ratio=4.0,
                 gate_type='elementwise',
                 p_drop_attn=0.1,
                 qkv_bias=False,
                 qk_norm=False,
                 **block_kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.gate_type = gate_type

        self.self_attn = FlashSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=p_drop_attn,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
        )

        self.cross_attn = CrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=nn.LayerNorm,
            gate_type=gate_type,
            **block_kwargs
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0.0,
        )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True),
        )

    def get_gate_stats(self):
        """
        获取 Gate-Attention 统计信息（用于 wandb 记录）。

        Returns:
            dict or None: 包含 gate_activation_mean/std/saturation_high/saturation_low，
                          无数据时返回 None。
        """
        if not hasattr(self.cross_attn, '_gate_stats_buffer') or len(self.cross_attn._gate_stats_buffer) == 0:
            return None

        gate_stats = self.cross_attn._gate_stats_buffer[-1]
        self.cross_attn._gate_stats_buffer.clear()

        return {
            'gate_activation_mean': gate_stats['mean'],
            'gate_activation_std': gate_stats['std'],
            'gate_saturation_high': gate_stats['saturation_high'],
            'gate_saturation_low': gate_stats['saturation_low'],
        }

    def set_record_attn(self, record: bool):
        """开启/关闭 cross-attention 权重记录"""
        if hasattr(self.cross_attn, 'set_record_attn'):
            self.cross_attn.set_record_attn(record)

    def get_cross_attn_weights(self):
        """获取已记录的 cross-attention 权重"""
        if hasattr(self.cross_attn, 'get_cross_attn_weights'):
            return self.cross_attn.get_cross_attn_weights()
        return None

    def clear_gate_stats_buffer(self):
        """清空 gate 统计缓存"""
        if hasattr(self.cross_attn, 'clear_gate_stats_buffer'):
            self.cross_attn.clear_gate_stats_buffer()

    def forward(self, x, time_c, context_c, attn_mask=None):
        """
        Args:
            x: 动作序列 (B, N, hidden_size)
            time_c: 时间步嵌入 (B, hidden_size)
            context_c: 多模态特征 (B, L, hidden_size)
            attn_mask: 可选注意力掩码

        Returns:
            x: 输出特征 (B, N, hidden_size)
        """
        chunks = self.adaLN_modulation(time_c).chunk(9, dim=-1)
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        # 1. Self-Attention
        self_attn_output, _ = self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * self_attn_output

        # 2. Cross-Attention with Gate-Attention
        cross_attn_output = self.cross_attn(modulate(self.norm2(x), shift_cross, scale_cross), context_c, mask=None)
        x = x + gate_cross.unsqueeze(1) * cross_attn_output

        # 3. MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))

        return x


if __name__ == "__main__":
    """
    测试 DiTX-GateAttn Block 的功能
    运行方式: python ditx_gateattn_block.py
    """

    def test_ditx_gateattn_block():
        """测试 DiTXGateAttnBlock 基本功能"""
        print("=" * 80)
        print("测试 DiTXGateAttnBlock (Gate-Attention)")
        print("=" * 80)

        batch_size = 4
        seq_len = 50
        hidden_size = 768
        num_heads = 12
        L_total = 1180

        block = DiTXGateAttnBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=4.0,
            gate_type='headwise',
            p_drop_attn=0.1,
        )

        x = torch.randn(batch_size, seq_len, hidden_size)
        time_c = torch.randn(batch_size, hidden_size)
        context_c = torch.randn(batch_size, L_total, hidden_size)

        print(f"\n输入形状:")
        print(f"  x (动作序列):       {x.shape}")
        print(f"  time_c (时间):      {time_c.shape}")
        print(f"  context_c (多模态): {context_c.shape}")

        print(f"\n" + "-" * 80)
        print("DiTXGateAttnBlock 前向传播...")
        block.train()
        output = block(x, time_c, context_c)
        print(f"  输出形状: {output.shape}")

        gate_stats = block.get_gate_stats()
        if gate_stats:
            print(f"  Gate 统计:")
            print(f"    - mean:            {gate_stats['gate_activation_mean']:.4f}")
            print(f"    - std:             {gate_stats['gate_activation_std']:.4f}")
            print(f"    - saturation_high: {gate_stats['gate_saturation_high']:.4f}")
            print(f"    - saturation_low:  {gate_stats['gate_saturation_low']:.4f}")
        print(f"  成功!")

        params = sum(p.numel() for p in block.parameters())
        print(f"\n总参数: {params:,}")
        print(f"Gate 类型: {block.gate_type}")
        print("=" * 80)

    def test_gradient_flow():
        """测试梯度流动"""
        print("\n" + "=" * 80)
        print("测试梯度流动")
        print("=" * 80)

        block = DiTXGateAttnBlock(
            hidden_size=512,
            num_heads=8,
            gate_type='elementwise',
        )
        block.train()

        x = torch.randn(2, 32, 512, requires_grad=True)
        time_c = torch.randn(2, 512, requires_grad=True)
        context_c = torch.randn(2, 256, 512, requires_grad=True)

        output = block(x, time_c, context_c)
        output.sum().backward()

        print(f"  x.grad is not None:         {x.grad is not None}")
        print(f"  time_c.grad is not None:    {time_c.grad is not None}")
        print(f"  context_c.grad is not None: {context_c.grad is not None}")
        print(f"  梯度流动正常!")
        print("=" * 80)

    torch.manual_seed(42)
    test_ditx_gateattn_block()
    test_gradient_flow()
    print("\n所有测试完成!")
