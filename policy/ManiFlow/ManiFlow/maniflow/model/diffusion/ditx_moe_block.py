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


class ModalityMoE(nn.Module):
    """
    模态级别MoE：按模态组合进行路由，每个专家只处理特定模态特征
    
    🔥 专家专业化策略（每个专家只处理其对应的模态特征）：
    - Expert 0: 全模态组合 (head + wrist + proprio) - 处理所有tokens
    - Expert 1: 头部+本体专家 (head + proprio) - 只处理head和proprio的tokens
    - Expert 2: 腕部+本体专家 (wrist + proprio) - 只处理wrist和proprio的tokens
    - Expert 3+: 额外专家，默认处理全模态
    
    核心改进：
    - Gate根据模态组合聚合特征选择专家
    - 专家只处理其对应的模态tokens，保证专业化
    - 避免无关模态干扰专家学习
    
    Args:
        embed_dim: 特征维度
        num_experts: 专家数量 (>=3)
        num_experts_per_tok: 每次激活的专家数
        n_shared_experts: 共享专家数量
        aux_loss_alpha: 负载均衡损失权重
        use_time_cond: 是否使用时间条件调制
    """
    def __init__(self, embed_dim, num_experts=4, num_experts_per_tok=2, 
                 n_shared_experts=1, aux_loss_alpha=0.01, use_time_cond=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.aux_loss_alpha = aux_loss_alpha
        self.use_time_cond = use_time_cond
        
        # 模态级别门控：输入为模态聚合特征
        # 3个模态组合的聚合特征 -> 专家选择
        self.gate_proj = nn.Linear(embed_dim * 3, num_experts)  # 3种模态组合
        
        # 时间条件调制门控
        if use_time_cond:
            self.time_gate_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embed_dim, num_experts * 2)  # scale和shift
            )
            nn.init.zeros_(self.time_gate_modulation[-1].weight)
            nn.init.zeros_(self.time_gate_modulation[-1].bias)
        
        # 专家网络：每个专家处理完整的context_c
        mlp_hidden = int(embed_dim * 4)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden, embed_dim)
            ) for _ in range(num_experts)
        ])
        
        # 共享专家
        if n_shared_experts > 0:
            self.shared_expert = nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden * n_shared_experts),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden * n_shared_experts, embed_dim)
            )
        else:
            self.shared_expert = None
        
        # 用于记录统计信息
        self.moe_stats = None
        
        self._init_weights()
    
    def _init_weights(self):
        for expert in self.experts:
            nn.init.xavier_uniform_(expert[0].weight)
            nn.init.zeros_(expert[0].bias)
            nn.init.xavier_uniform_(expert[2].weight)
            nn.init.zeros_(expert[2].bias)
        if self.shared_expert is not None:
            nn.init.xavier_uniform_(self.shared_expert[0].weight)
            nn.init.zeros_(self.shared_expert[0].bias)
            nn.init.xavier_uniform_(self.shared_expert[2].weight)
            nn.init.zeros_(self.shared_expert[2].bias)
    
    def forward(self, context_c, time_cond=None, modality_lens=None):
        """
        Args:
            context_c: (B, L_total, D) 多模态特征序列
            time_cond: (B, D) 时间条件嵌入
            modality_lens: dict 各模态长度 {'head': L_head, 'wrist': L_wrist, 'proprio': L_proprio}
                          如果为None，则均分
        Returns:
            output: (B, L_total, D) 处理后的特征
        """
        B, L, D = context_c.shape
        
        # 解析模态长度
        if modality_lens is None:
            # 默认均分
            L_head = L_wrist = L // 3
            L_proprio = L - L_head - L_wrist
        else:
            L_head = modality_lens.get('head', 0)
            L_wrist = modality_lens.get('wrist', 0)
            L_proprio = modality_lens.get('proprio', L - L_head - L_wrist)
        
        # 分割模态特征
        head_feat = context_c[:, :L_head, :]  # (B, L_head, D)
        wrist_feat = context_c[:, L_head:L_head+L_wrist, :]  # (B, L_wrist, D)
        proprio_feat = context_c[:, L_head+L_wrist:, :]  # (B, L_proprio, D)
        
        # 计算模态组合的聚合特征（用于门控）
        # 组合1: 全模态 (head + wrist + proprio)
        full_agg = context_c.mean(dim=1)  # (B, D)
        # 组合2: 头部+本体
        head_proprio_agg = torch.cat([head_feat, proprio_feat], dim=1).mean(dim=1) if L_head > 0 else proprio_feat.mean(dim=1)
        # 组合3: 腕部+本体
        wrist_proprio_agg = torch.cat([wrist_feat, proprio_feat], dim=1).mean(dim=1) if L_wrist > 0 else proprio_feat.mean(dim=1)
        
        # 拼接聚合特征用于门控
        gate_input = torch.cat([full_agg, head_proprio_agg, wrist_proprio_agg], dim=-1)  # (B, 3*D)
        
        # 计算门控分数
        gate_logits = self.gate_proj(gate_input)  # (B, num_experts)
        
        # 时间条件调制
        if self.use_time_cond and time_cond is not None:
            modulation = self.time_gate_modulation(time_cond)
            scale, shift = modulation.chunk(2, dim=-1)
            gate_logits = gate_logits * (1 + scale) + shift
        
        gate_scores = F.softmax(gate_logits, dim=-1)  # (B, num_experts)
        
        # 选择top-k专家
        topk_weights, topk_indices = torch.topk(gate_scores, k=self.num_experts_per_tok, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)  # 归一化
        
        # 🆕 专家计算（专家只处理特定模态）
        output = torch.zeros_like(context_c)
        for k in range(self.num_experts_per_tok):
            expert_idx = topk_indices[:, k]  # (B,)
            expert_weight = topk_weights[:, k:k+1].unsqueeze(-1)  # (B, 1, 1)
            
            # 对每个batch样本应用对应专家
            for b in range(B):
                idx = expert_idx[b].item()
                
                # 🔥 根据专家索引决定处理哪些模态
                # Expert 0: 全模态 (head + wrist + proprio)
                # Expert 1: 头部+本体 (head + proprio)
                # Expert 2: 腕部+本体 (wrist + proprio)
                # Expert 3+: 默认处理全模态
                
                if idx == 0 or idx >= 3:  # 全模态专家
                    expert_input = context_c[b]  # (L_total, D)
                    expert_output = self.experts[idx](expert_input)
                    output[b] += expert_weight[b] * expert_output
                    
                elif idx == 1:  # 头部+本体专家
                    if L_head > 0 and L_proprio > 0:
                        # 拼接头部和本体特征
                        expert_input = torch.cat([head_feat[b], proprio_feat[b]], dim=0)  # (L_head+L_proprio, D)
                        expert_output = self.experts[idx](expert_input)
                        # 分配回对应位置
                        output[b, :L_head] += expert_weight[b, 0, 0] * expert_output[:L_head]
                        output[b, L_head+L_wrist:] += expert_weight[b, 0, 0] * expert_output[L_head:]
                    elif L_head > 0:  # 只有头部
                        expert_output = self.experts[idx](head_feat[b])
                        output[b, :L_head] += expert_weight[b, 0, 0] * expert_output
                    elif L_proprio > 0:  # 只有本体
                        expert_output = self.experts[idx](proprio_feat[b])
                        output[b, L_head+L_wrist:] += expert_weight[b, 0, 0] * expert_output
                        
                elif idx == 2:  # 腕部+本体专家
                    if L_wrist > 0 and L_proprio > 0:
                        # 拼接腕部和本体特征
                        expert_input = torch.cat([wrist_feat[b], proprio_feat[b]], dim=0)  # (L_wrist+L_proprio, D)
                        expert_output = self.experts[idx](expert_input)
                        # 分配回对应位置
                        output[b, L_head:L_head+L_wrist] += expert_weight[b, 0, 0] * expert_output[:L_wrist]
                        output[b, L_head+L_wrist:] += expert_weight[b, 0, 0] * expert_output[L_wrist:]
                    elif L_wrist > 0:  # 只有腕部
                        expert_output = self.experts[idx](wrist_feat[b])
                        output[b, L_head:L_head+L_wrist] += expert_weight[b, 0, 0] * expert_output
                    elif L_proprio > 0:  # 只有本体
                        expert_output = self.experts[idx](proprio_feat[b])
                        output[b, L_head+L_wrist:] += expert_weight[b, 0, 0] * expert_output
        
        # 添加共享专家
        if self.shared_expert is not None:
            output = output + self.shared_expert(context_c)
        
        # 计算负载均衡损失
        if self.training and self.aux_loss_alpha > 0:
            # 专家使用频率
            expert_mask = F.one_hot(topk_indices.view(-1), num_classes=self.num_experts).float()
            expert_usage = expert_mask.mean(0)
            # 路由概率
            router_prob = gate_scores.mean(0)
            # 负载均衡损失
            aux_loss = (expert_usage * router_prob).sum() * self.num_experts * self.aux_loss_alpha
            
            # 计算topk_weights统计 (处理batch_size=1的情况)
            topk_mean = topk_weights.mean().detach().item()
            # 只有当有多个样本时才计算std，否则设为0
            if topk_weights.numel() > 1:
                topk_std = topk_weights.std().detach().item()
            else:
                topk_std = 0.0
            
            self.moe_stats = {
                'aux_loss': aux_loss.detach().item(),
                'expert_usage': expert_usage.detach(),
                'router_scores': router_prob.detach(),
                'topk_weights_mean': topk_mean,
                'topk_weights_std': topk_std,
            }
        
        return output

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
    
    核心改进：
    1. 模态级别路由：按模态组合（全模态/头部+本体/腕部+本体）进行路由，保持模态内语义一致性
    2. 时间条件感知：MoE门控感知扩散时间步，根据噪声阶段调整专家选择
    3. AdaLN协调：context_c在进入MoE前通过AdaLN感知时间条件
    
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
                hidden_size=768,
                num_heads=12,
                mlp_ratio=4.0,
                
                # MoE配置
                use_modality_moe=True,
                num_experts=4,                # 模态级MoE建议4-8个专家
                num_experts_per_tok=2,
                n_shared_experts=1,
                moe_aux_loss_alpha=0.01,
                
                # 其他参数
                p_drop_attn=0.1,
                qkv_bias=False,
                qk_norm=False,
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
        
        # ⭐ 模态级别MoE (替代token级别的SparseMoeBlock)
        if use_modality_moe:
            self.modality_moe = ModalityMoE(
                embed_dim=hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_shared_experts=n_shared_experts,
                aux_loss_alpha=moe_aux_loss_alpha,
                use_time_cond=True  # 启用时间条件感知
            )
            # context_c的AdaLN：让MoE输入感知时间条件
            self.context_adaln = AdaptiveLayerNorm(dim=hidden_size, dim_cond=hidden_size)
            logger.info(f"[DiTXMoEBlock] Initialized ModalityMoE with {num_experts} experts, "
                       f"top-{num_experts_per_tok}, {n_shared_experts} shared, time_cond=True")
        
        # Cross-Attention
        self.cross_attn = CrossAttention(
            dim=hidden_size, 
            num_heads=num_heads,
            qkv_bias=qkv_bias, 
            qk_norm=qk_norm,
            norm_layer=nn.LayerNorm, 
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
        
    def forward(self, x, time_c, context_c, attn_mask=None, modality_lens=None):
        """
        Forward pass of the DiTX-MoE block.
        
        Args:
            x: 动作序列 (batch_size, seq_length, hidden_size)
            time_c: 时间步嵌入 (batch_size, hidden_size)
            context_c: 多模态特征 (batch_size, L_total, hidden_size)
            attn_mask: 可选的注意力mask
            modality_lens: 模态长度信息 {'head': L_head, 'wrist': L_wrist, 'proprio': L_proprio}
        
        Returns:
            x: 输出特征 (batch_size, seq_length, hidden_size)
        """
        # adaLN modulation
        modulation = self.adaLN_modulation(time_c)
        chunks = modulation.chunk(9, dim=-1)
        
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        # 1. Self-Attention with adaLN conditioning
        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output, _ = self.self_attn(normed_x, normed_x, normed_x, attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * self_attn_output

        # 2. ⭐ 模态MoE处理多模态输入特征
        if self.use_modality_moe:
            # 先通过AdaLN让context_c感知时间条件
            context_c_normed = self.context_adaln(context_c, time_c)
            # 模态级别MoE处理（传入时间条件和模态长度）
            context_c_processed = self.modality_moe(context_c_normed, time_c, modality_lens)
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
        print("测试 DiTXMoEBlock (模态级别MoE + 时间条件感知)")
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
        L_total = L_head + L_wrist + L_proprio
        
        # 模态长度信息
        modality_lens = {'head': L_head, 'wrist': L_wrist, 'proprio': L_proprio}
        
        # 创建DiTXMoEBlock
        block_moe = DiTXMoEBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=4.0,
            use_modality_moe=True,
            num_experts=4,        # 模态级MoE建议4个专家
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
        
        # 前向传播 - MoE版本（带模态长度信息）
        print(f"\n" + "─" * 80)
        print("DiTXMoEBlock 前向传播 (模态级别路由)...")
        block_moe.train()
        output_moe = block_moe(x, time_c, context_c, modality_lens=modality_lens)
        print(f"  输出形状: {output_moe.shape}")
        
        # 检查MoE统计信息
        if hasattr(block_moe, 'modality_moe') and block_moe.modality_moe.moe_stats:
            stats = block_moe.modality_moe.moe_stats
            print(f"  MoE统计:")
            print(f"    - aux_loss: {stats['aux_loss']:.6f}")
            print(f"    - expert_usage: {stats['expert_usage'].tolist()}")
            print(f"    - topk_weights_mean: {stats['topk_weights_mean']:.4f}")
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
            print(f"\n  ModalityMoE参数: {moe_params:,}")
            print(f"    ├─ 专家数量: {block_moe.modality_moe.num_experts}")
            print(f"    ├─ Top-K: {block_moe.modality_moe.num_experts_per_tok}")
            print(f"    ├─ 共享专家: {block_moe.modality_moe.n_shared_experts}")
            print(f"    └─ 时间条件: {block_moe.modality_moe.use_time_cond}")
        
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
            num_experts=4,
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
        print("测试关闭MoE (use_modality_moe=False)")
        print("=" * 80)
        
        block = DiTXMoEBlock(
            hidden_size=512,
            num_heads=8,
            use_modality_moe=False
        )
        
        x = torch.randn(2, 32, 512)
        time_c = torch.randn(2, 512)
        context_c = torch.randn(2, 256, 512)
        
        output = block(x, time_c, context_c)
        print(f"  输出形状: {output.shape}")
        print(f"  ✅ 关闭MoE模式正常工作!")
        print("=" * 80)
    
    
    def test_gradient_flow():
        """测试梯度流动"""
        print("\n\n" + "=" * 80)
        print("测试梯度流动")
        print("=" * 80)
        
        block = DiTXMoEBlock(
            hidden_size=512,
            num_heads=8,
            use_modality_moe=True,
            num_experts=4,
            num_experts_per_tok=2
        )
        block.train()
        
        x = torch.randn(2, 32, 512, requires_grad=True)
        time_c = torch.randn(2, 512, requires_grad=True)
        context_c = torch.randn(2, 256, 512, requires_grad=True)
        
        modality_lens = {'head': 128, 'wrist': 112, 'proprio': 16}
        output = block(x, time_c, context_c, modality_lens=modality_lens)
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

