import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 n_shared_experts=1, aux_loss_alpha=0.01, use_time_cond=True,
                 enable_grad_accumulation=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.aux_loss_alpha = aux_loss_alpha
        self.use_time_cond = use_time_cond
        self.enable_grad_accumulation = enable_grad_accumulation
        
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
        
        # 梯度累积支持：累积多个micro-batch的统计信息
        if enable_grad_accumulation:
            self.register_buffer('_accumulated_expert_usage', torch.zeros(num_experts))
            self.register_buffer('_accumulated_router_prob', torch.zeros(num_experts))
            self.register_buffer('_accumulated_samples', torch.zeros(1))
        
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
    
    def reset_accumulation(self):
        """重置累积的统计信息（在optimizer.step()后调用）"""
        if self.enable_grad_accumulation:
            self._accumulated_expert_usage.zero_()
            self._accumulated_router_prob.zero_()
            self._accumulated_samples.zero_()
    
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
            if self.enable_grad_accumulation:
                # 梯度累积模式：累积统计信息
                with torch.no_grad():
                    # 累积专家使用频率
                    expert_mask = F.one_hot(topk_indices.view(-1), num_classes=self.num_experts).float()
                    expert_usage_batch = expert_mask.sum(0)  # (num_experts,)
                    self._accumulated_expert_usage += expert_usage_batch
                    
                    # 累积路由概率
                    router_prob_batch = gate_scores.sum(0)  # (num_experts,)
                    self._accumulated_router_prob += router_prob_batch
                    
                    # 累积样本数
                    num_samples = B * self.num_experts_per_tok
                    self._accumulated_samples += num_samples
                
                # 基于累积统计计算负载均衡损失
                if self._accumulated_samples > 0:
                    expert_usage = self._accumulated_expert_usage / self._accumulated_samples.clamp(min=1.0)
                    router_prob = self._accumulated_router_prob / (self._accumulated_samples.clamp(min=1.0) / self.num_experts_per_tok)
                    aux_loss = (expert_usage * router_prob).sum() * self.num_experts * self.aux_loss_alpha
                else:
                    aux_loss = torch.tensor(0.0, device=context_c.device, dtype=context_c.dtype)
                    expert_usage = torch.zeros(self.num_experts, device=context_c.device)
                    router_prob = torch.zeros(self.num_experts, device=context_c.device)
            else:
                # 标准模式：每个batch独立计算
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