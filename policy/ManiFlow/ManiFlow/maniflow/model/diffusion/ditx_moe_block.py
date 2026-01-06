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

logger = logging.getLogger(__name__)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

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


class DiTXMoEBlock(nn.Module):
    """
    DiTX Block with Modality-level Mixture of Experts (MoE).
    
    在CrossAttention之前对多模态输入特征(context_c)应用MoE，
    让不同专家学习不同模态组合的重要性权重。
    
    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        mlp_ratio: MLP扩展比例
        use_modality_moe: 是否使用模态MoE
        num_experts: MoE专家数量
        num_experts_per_tok: 每个token激活的专家数
        n_shared_experts: 共享专家数量
        moe_aux_loss_alpha: MoE辅助损失权重
        p_drop_attn: Attention dropout概率
        qkv_bias: 是否使用QKV bias
        qk_norm: 是否对Q和K进行归一化
    """
    def __init__(self, 
                hidden_size=768,              # 隐藏层维度
                num_heads=12,                 # 注意力头数
                mlp_ratio=4.0,               # MLP扩展比例
                
                # MoE配置
                use_modality_moe=True,        # 启用模态MoE
                num_experts=8,                # 8个模态专家
                num_experts_per_tok=2,        # 每次激活2个专家
                n_shared_experts=1,           # 1个共享专家
                moe_aux_loss_alpha=0.01,      # 辅助损失权重
                
                # 其他参数
                p_drop_attn=0.1,              # Attention dropout概率
                qkv_bias=False,               # 是否使用QKV bias
                qk_norm=False,                # 是否对Q和K进行归一化
                **block_kwargs):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_modality_moe = use_modality_moe

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, 
            batch_first=True, 
            dropout=p_drop_attn
        )
        
        # ⭐ 模态专家MoE (处理多模态输入context_c)
        if use_modality_moe:
            from maniflow.model.gate.MoEgate import SparseMoeBlock
            self.modality_moe = SparseMoeBlock(
                embed_dim=hidden_size,
                mlp_ratio=mlp_ratio,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_shared_experts=n_shared_experts,
                aux_loss_alpha=moe_aux_loss_alpha
            )
            logger.info(f"[DiTXMoEBlock] Initialized Modality MoE with {num_experts} experts, "
                       f"top-{num_experts_per_tok} per token, {n_shared_experts} shared experts")
        
        # Cross-Attention
        self.cross_attn = CrossAttention(
            dim=hidden_size, 
            num_heads=num_heads,
            qkv_bias=qkv_bias, 
            qk_norm=qk_norm,
            norm_layer=nn.LayerNorm, 
            **block_kwargs
        )
       
        # MLP (保持不变，使用标准MLP)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, 
            hidden_features=mlp_hidden_dim, 
            act_layer=approx_gelu, 
            drop=0.0
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # For self-attention
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # For cross-attention
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # For MLP

        # AdaLN modulation
        modulation_size = 9 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, modulation_size, bias=True)
        )
        
    def forward(self, x, time_c, context_c, attn_mask=None):
        """
        Forward pass of the DiTX-MoE block.
        
        Args:
            x: 动作序列 (batch_size, seq_length, hidden_size)
            time_c: 时间步嵌入 (batch_size, hidden_size)
            context_c: 多模态特征 (batch_size, L_total, hidden_size)
                      包含: [头部视觉, 腕部视觉+触觉, 本体感知]
            attn_mask: 可选的注意力mask (batch_size, seq_length, seq_length)
        
        Returns:
            x: 输出特征 (batch_size, seq_length, hidden_size)
        """

        # adaLN modulation for self-attention, cross-attention, and MLP
        modulation = self.adaLN_modulation(time_c)

        # Split into 9 chunks of hidden_size each
        chunks = modulation.chunk(9, dim=-1)
        
        # Self-Attention parameters
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        
        # Cross-Attention parameters  
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        
        # MLP parameters
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]


        # 1. Self-Attention with adaLN conditioning
        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output, _ = self.self_attn(normed_x, normed_x, normed_x, 
                                             attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * self_attn_output
        

        # 2. ⭐ 模态MoE处理多模态输入特征
        if self.use_modality_moe:
            context_c_processed = self.modality_moe(context_c)
        else:
            context_c_processed = context_c
        

        # 3. Cross-Attention with adaLN conditioning
        # 使用MoE处理后的多模态特征进行交叉注意力
        normed_x_cross = modulate(self.norm2(x), shift_cross, scale_cross)
        cross_attn_output = self.cross_attn(normed_x_cross, context_c_processed, 
                                            mask=None)
        x = x + gate_cross.unsqueeze(1) * cross_attn_output
       

        # 4. MLP with adaLN conditioning (保持原有的标准MLP)
        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x

if __name__ == "__main__":
    """
    测试DiTX-MoE Block的功能
    运行方式: python ditx-moe_block.py
    """
    
    def test_ditx_moe_block():
        """测试DiTXMoEBlock的基本功能"""
        print("=" * 80)
        print("测试 DiTXMoEBlock")
        print("=" * 80)
        
        # 参数设置
        batch_size = 4
        seq_len = 50          # 动作序列长度
        hidden_size = 768
        num_heads = 12
        
        # 多模态特征长度
        L_head = 256          # 头部相机特征长度
        L_wrist = 256         # 腕部相机+触觉特征长度
        L_proprio = 16        # 本体感知特征长度
        L_total = L_head + L_wrist + L_proprio  # 总特征长度
        
        # 创建DiTXMoEBlock
        block_moe = DiTXMoEBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=4.0,
            use_modality_moe=True,
            num_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_aux_loss_alpha=0.01,
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
        
        # 前向传播 - MoE版本
        print(f"\n" + "─" * 80)
        print("DiTXMoEBlock 前向传播...")
        block_moe.train()  # 训练模式才会计算aux_loss
        output_moe = block_moe(x, time_c, context_c)
        print(f"  输出形状: {output_moe.shape}")
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
        if hasattr(block_moe, 'modality_moe'):
            moe_params = sum(p.numel() for p in block_moe.modality_moe.parameters())
            print(f"\n  MoE模块参数:   {moe_params:,}")
            print(f"    ├─ 专家数量: {block_moe.modality_moe.gate.n_routed_experts}")
            print(f"    ├─ Top-K: {block_moe.modality_moe.gate.top_k}")
            print(f"    └─ 共享专家: {block_moe.modality_moe.n_shared_experts}")
        
        print(f"\n" + "=" * 80)
        print("✅ 所有测试通过!")
        print("=" * 80)


    def test_batch_sizes():
        """测试不同batch size"""
        print("\n\n" + "=" * 80)
        print("测试不同Batch Size")
        print("=" * 80)
        
        block = DiTXMoEBlock(
            hidden_size=512,
            num_heads=8,
            use_modality_moe=True,
            num_experts=8,
            num_experts_per_tok=2
        )
        block.eval()  # 推理模式
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 32, 512)
            time_c = torch.randn(batch_size, 512)
            context_c = torch.randn(batch_size, 256, 512)
            
            with torch.no_grad():
                output = block(x, time_c, context_c)
            
            print(f"  Batch size {batch_size}: {output.shape} ✅")
        
        print("=" * 80)


    def test_without_moe():
        """测试关闭MoE的情况"""
        print("\n\n" + "=" * 80)
        print("测试关闭MoE (use_modality_moe=False)")
        print("=" * 80)
        
        block = DiTXMoEBlock(
            hidden_size=512,
            num_heads=8,
            use_modality_moe=False  # 关闭MoE
        )
        
        x = torch.randn(2, 32, 512)
        time_c = torch.randn(2, 512)
        context_c = torch.randn(2, 256, 512)
        
        output = block(x, time_c, context_c)
        print(f"  输出形状: {output.shape}")
        print(f"  ✅ 关闭MoE模式正常工作!")
        print("=" * 80)


    # 运行所有测试
    torch.manual_seed(42)
    
    test_ditx_moe_block()
    test_batch_sizes()
    test_without_moe()
    
    print("\n" + "🎉" * 40)
    print("所有测试完成!")
    print("🎉" * 40)

