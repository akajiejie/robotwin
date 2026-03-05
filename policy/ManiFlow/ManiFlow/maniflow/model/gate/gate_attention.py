import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import use_fused_attn

class GatedCrossAttention(nn.Module):
    """
    针对具身智能优化的 Gated-Cross-Attention。
    通过时间条件 (time_c) 生成 Head-wise Gate，动态调节各模态信息流。
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
        self.fused_attn = use_fused_attn()

        # 核心投影层
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # ⭐ Head-wise Gate: 受扩散时间步 time_c 调制
        self.gate_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, num_heads) 
        )
        # 初始化 gate bias 为正值 (如 1.0)，确保训练初期所有模态都是开放的
        nn.init.constant_(self.gate_proj[-1].bias, 1.0)
        nn.init.zeros_(self.gate_proj[-1].weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor, time_c: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x: 动作序列 (B, N, D)
            c: 多模态条件序列 (B, L, D) - 来自全量 Token 输出的 Encoder
            time_c: 扩散时间步嵌入 (B, D)
        """
        B, N, C = x.shape
        _, L, _ = c.shape

        # 1. 计算 Q, K, V
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # 2. 计算注意力 (支持 Flash Attention)
        if self.fused_attn:
            attn_out = F.scaled_dot_product_attention(
                query=q, key=k, value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            attn_out = self.attn_drop(attn) @ v

        # 3. ⭐ 核心改进：应用 Head-wise Gating
        # time_c (B, D) -> gate (B, heads, 1, 1)
        gate_score = self.gate_proj(time_c).view(B, self.num_heads, 1, 1)
        attn_out = attn_out * torch.sigmoid(gate_score)

        # 4. 融合输出
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(attn_out)
        x = self.proj_drop(x)
        return x