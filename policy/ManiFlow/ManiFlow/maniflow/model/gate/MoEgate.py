import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

#################################################################################
#                                MoE Layer.                                     #
#################################################################################


class MoEGate(nn.Module):
    """
    门控网络，用于选择top-k个专家
    
    Args:
        embed_dim: 输入特征维度
        num_experts: 专家总数
        num_experts_per_tok: 每个token选择的专家数量
        aux_loss_alpha: 辅助损失权重
    """
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape    
        
        # 处理空输入的情况（batch_size=0 或 seq_len=0）
        if bsz == 0 or seq_len == 0:
            # 返回空tensor，形状与输入匹配
            device = hidden_states.device
            dtype = hidden_states.dtype
            topk_idx = torch.empty(bsz, seq_len, self.top_k, dtype=torch.long, device=device)
            topk_weight = torch.empty(bsz, seq_len, self.top_k, dtype=dtype, device=device)
            aux_loss = None if not (self.training and self.alpha > 0.0) else torch.tensor(0.0, dtype=dtype, device=device)
            router_probs = None
            return topk_idx, topk_weight, aux_loss, router_probs
        
        # print(bsz, seq_len, h)    
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        router_probs = None  # 用于记录路由概率
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
            
            # 保存路由概率用于wandb记录
            router_probs = {
                'expert_usage': ce.detach(),  # 每个专家被选中的频率
                'router_scores': Pi.detach(),  # 路由概率分布
            }
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss, router_probs


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    添加辅助损失的技巧函数，在反向传播时包含辅助损失的梯度
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MoeMLP(nn.Module):
    """
    单个专家的MLP网络 (SwiGLU架构)
    
    Args:
        hidden_size: 输入和输出的隐藏层大小
        intermediate_size: 中间层大小
        pretraining_tp: 张量并行度 (通常设置为1)
    """
    def __init__(self, hidden_size, intermediate_size, pretraining_tp=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.pretraining_tp = pretraining_tp

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0) 
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=-1)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # SwiGLU: SiLU(gate_proj(x)) * up_proj(x)
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class SparseMoeBlock(nn.Module):
    """
    稀疏混合专家模块，包含路由专家和共享专家
    
    Args:
        embed_dim: 嵌入维度
        mlp_ratio: MLP扩展比例
        num_experts: 路由专家总数
        num_experts_per_tok: 每个token激活的专家数量
        n_shared_experts: 共享专家数量 (默认2)
        pretraining_tp: 张量并行度
        aux_loss_alpha: 辅助损失权重
    """
    def __init__(self, embed_dim, mlp_ratio=4, num_experts=16, num_experts_per_tok=2, 
                 n_shared_experts=2, pretraining_tp=1, aux_loss_alpha=0.01):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.num_experts = num_experts
        
        # 用于累积MoE统计信息
        self.moe_stats = None
        
        # 路由专家
        self.experts = nn.ModuleList([
            MoeMLP(hidden_size=embed_dim, 
                   intermediate_size=int(mlp_ratio * embed_dim), 
                   pretraining_tp=pretraining_tp) 
            for i in range(num_experts)
        ])
        
        # 门控网络
        self.gate = MoEGate(embed_dim=embed_dim, 
                           num_experts=num_experts, 
                           num_experts_per_tok=num_experts_per_tok,
                           aux_loss_alpha=aux_loss_alpha)
        
        # 共享专家
        if self.n_shared_experts is not None and self.n_shared_experts > 0:
            intermediate_size = embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(hidden_size=embed_dim, 
                                        intermediate_size=intermediate_size, 
                                        pretraining_tp=pretraining_tp)
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, embed_dim)
            
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        # 处理空输入的情况
        if hidden_states.numel() == 0:
            return hidden_states
        
        identity = hidden_states
        orig_shape = hidden_states.shape
        
        # 门控路由
        topk_idx, topk_weight, aux_loss, router_probs = self.gate(hidden_states) 
        
        # 保存MoE统计信息用于wandb记录
        # 🔥 优化2: 减少CPU-GPU同步，只保留GPU tensor
        if self.training and router_probs is not None:
            self.moe_stats = {
                'aux_loss': aux_loss.detach().item() if aux_loss is not None else 0.0,
                'expert_usage': router_probs['expert_usage'],  # 保持在GPU上
                'router_scores': router_probs['router_scores'],  # 保持在GPU上
                'topk_weights_mean': topk_weight.mean().detach().item(),
                'topk_weights_std': topk_weight.std().detach().item(),
            }
        
        # 如果gate返回空结果，直接返回identity
        if topk_idx.numel() == 0:
            if self.n_shared_experts is not None and self.n_shared_experts > 0:
                return self.shared_experts(identity)
            return identity
        
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            # 训练模式：并行专家调度（优化版）
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            
            # 🔥 优化1: 批量处理专家，减少for循环开销
            # 将所有专家的计算合并为一个批次
            expert_outputs = []
            for i, expert in enumerate(self.experts):
                mask = (flat_topk_idx == i)
                if mask.any():
                    expert_input = hidden_states[mask]
                    expert_output = expert(expert_input)
                    expert_outputs.append((mask, expert_output))
            
            # 批量赋值
            for mask, output in expert_outputs:
                y[mask] = output
            
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            # 推理模式：优化的专家调度
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 添加共享专家的输出
        if self.n_shared_experts is not None and self.n_shared_experts > 0:
            y = y + self.shared_experts(identity)
        
        return y
    
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理时的高效专家调度
        
        Args:
            x: 输入张量 (num_tokens, hidden_size)
            flat_expert_indices: 专家索引 (num_tokens * num_experts_per_tok,)
            flat_expert_weights: 专家权重 (num_tokens * num_experts_per_tok, 1)
        """
        expert_cache = torch.zeros_like(x) 
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok 
        
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]]) 
            
            # 确保dtype一致
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), 
                                        expert_out, reduce='sum')
        return expert_cache