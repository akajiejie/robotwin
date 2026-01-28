"""
🔥 MoE Gate梯度诊断脚本
用于检查为什么expert_entropy_normalized不下降

使用方法：
python debug_gate_gradient.py

诊断内容：
1. Gate梯度是否为0
2. Aux_loss是否正确计算
3. Aux_loss是否加入总loss
4. Gate权重是否更新
"""

import torch
import torch.nn as nn
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.gate.MoEgate import MoEGate, SparseMoeBlock


def test_gate_gradient_flow():
    """测试gate梯度是否正常流动"""
    print("="*80)
    print("🔍 MoE Gate梯度诊断测试")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建MoE模块
    embed_dim = 768
    num_experts = 3
    num_experts_per_tok = 2
    aux_loss_alpha = 1.0
    
    print(f"\n创建SparseMoeBlock:")
    print(f"  - embed_dim: {embed_dim}")
    print(f"  - num_experts: {num_experts}")
    print(f"  - num_experts_per_tok: {num_experts_per_tok}")
    print(f"  - aux_loss_alpha: {aux_loss_alpha}")
    
    moe_block = SparseMoeBlock(
        embed_dim=embed_dim,
        mlp_ratio=4,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        n_shared_experts=0,
        aux_loss_alpha=aux_loss_alpha,
        enable_grad_accumulation=False
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(moe_block.parameters(), lr=1e-4)
    
    # 模拟训练数据
    batch_size = 16
    seq_len = 12
    hidden_states = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    print(f"\n输入数据形状: {hidden_states.shape}")
    
    # 前向传播
    moe_block.train()
    output = moe_block(hidden_states)
    
    print(f"输出数据形状: {output.shape}")
    
    # 获取MoE统计信息
    if hasattr(moe_block, 'moe_stats'):
        stats = moe_block.moe_stats
        print(f"\nMoE统计信息:")
        print(f"  - aux_loss: {stats['aux_loss']:.6f}")
        print(f"  - expert_usage: {stats['expert_usage'].cpu().numpy()}")
        print(f"  - topk_weights_mean: {stats['topk_weights_mean']:.4f}")
        
        # 计算熵值
        expert_usage = stats['expert_usage']
        entropy = -(expert_usage * torch.log(expert_usage + 1e-10)).sum()
        max_entropy = torch.log(torch.tensor(num_experts, dtype=torch.float32))
        entropy_normalized = entropy / max_entropy
        print(f"  - expert_entropy_normalized: {entropy_normalized:.4f}")
    
    # 🔥 关键测试：检查gate梯度
    print(f"\n" + "="*80)
    print("🔥 关键测试：Gate梯度流动检查")
    print("="*80)
    
    # 检查gate权重的初始值
    gate_weight_before = moe_block.gate.weight.clone().detach()
    print(f"\nGate权重初始统计:")
    print(f"  - 均值: {gate_weight_before.mean():.6f}")
    print(f"  - 标准差: {gate_weight_before.std():.6f}")
    print(f"  - 范围: [{gate_weight_before.min():.4f}, {gate_weight_before.max():.4f}]")
    
    # 计算损失（模拟实际训练）
    target = torch.randn_like(output)
    main_loss = nn.functional.mse_loss(output, target)
    
    # 获取aux_loss
    if hasattr(moe_block, 'moe_stats'):
        # 注意：aux_loss已经通过AddAuxiliaryLoss加入到output的计算图中
        # 但我们需要确保它真的被加入了
        aux_loss = stats['aux_loss']
        print(f"\nLoss统计:")
        print(f"  - main_loss: {main_loss.item():.6f}")
        print(f"  - aux_loss (detached): {aux_loss:.6f}")
        print(f"  ⚠️  注意：aux_loss应该已通过AddAuxiliaryLoss加入计算图")
    
    # 反向传播
    optimizer.zero_grad()
    main_loss.backward()
    
    # 🔥🔥🔥 核心检查：gate梯度是否存在且非零
    print(f"\n" + "="*80)
    print("🔥🔥🔥 核心检查：Gate梯度")
    print("="*80)
    
    if moe_block.gate.weight.grad is not None:
        gate_grad = moe_block.gate.weight.grad
        gate_grad_norm = gate_grad.norm().item()
        gate_grad_mean = gate_grad.mean().item()
        gate_grad_std = gate_grad.std().item()
        
        print(f"✅ Gate梯度存在!")
        print(f"  - 梯度范数: {gate_grad_norm:.6f}")
        print(f"  - 梯度均值: {gate_grad_mean:.6f}")
        print(f"  - 梯度标准差: {gate_grad_std:.6f}")
        
        if gate_grad_norm < 1e-6:
            print(f"❌ 错误：Gate梯度几乎为0！这解释了为什么专家不分化！")
            print(f"  可能原因：")
            print(f"  1. aux_loss没有正确加入计算图")
            print(f"  2. AddAuxiliaryLoss实现有误")
            print(f"  3. 梯度被裁剪到0")
        else:
            print(f"✅ Gate梯度正常流动")
    else:
        print(f"❌ 致命错误：Gate梯度为None！")
        print(f"  这就是为什么专家不分化的根本原因！")
        print(f"  Gate权重没有requires_grad或没有参与计算图")
        return False
    
    # 更新参数
    optimizer.step()
    
    # 检查gate权重是否更新
    gate_weight_after = moe_block.gate.weight.detach()
    weight_change = (gate_weight_after - gate_weight_before).norm().item()
    
    print(f"\n权重更新检查:")
    print(f"  - 权重变化范数: {weight_change:.6f}")
    
    if weight_change < 1e-6:
        print(f"❌ 错误：Gate权重几乎没有更新！")
        return False
    else:
        print(f"✅ Gate权重正常更新")
    
    # 多次迭代测试
    print(f"\n" + "="*80)
    print("📊 多次迭代测试（10次）")
    print("="*80)
    
    entropy_history = []
    for step in range(10):
        hidden_states = torch.randn(batch_size, seq_len, embed_dim, device=device)
        output = moe_block(hidden_states)
        target = torch.randn_like(output)
        loss = nn.functional.mse_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if hasattr(moe_block, 'moe_stats'):
            stats = moe_block.moe_stats
            expert_usage = stats['expert_usage']
            entropy = -(expert_usage * torch.log(expert_usage + 1e-10)).sum()
            max_entropy = torch.log(torch.tensor(num_experts, dtype=torch.float32))
            entropy_normalized = (entropy / max_entropy).item()
            entropy_history.append(entropy_normalized)
            
            if step % 2 == 0:
                print(f"Step {step}: entropy={entropy_normalized:.4f}, usage={expert_usage.cpu().numpy()}")
    
    print(f"\n熵值变化趋势:")
    print(f"  初始: {entropy_history[0]:.4f}")
    print(f"  最终: {entropy_history[-1]:.4f}")
    print(f"  变化: {entropy_history[-1] - entropy_history[0]:.4f}")
    
    if entropy_history[-1] < entropy_history[0]:
        print(f"✅ 熵值下降，专家正在分化！")
        return True
    else:
        print(f"⚠️  熵值未下降，可能需要更多迭代或调整aux_loss_alpha")
        return False


if __name__ == "__main__":
    success = test_gate_gradient_flow()
    
    print(f"\n" + "="*80)
    if success:
        print("✅ 诊断结果：MoE Gate梯度流动正常，专家应该能分化")
        print("   如果实际训练中专家仍不分化，请检查：")
        print("   1. aux_loss_alpha是否足够大（建议>=0.5）")
        print("   2. 视觉编码器是否冻结（减少梯度干扰）")
        print("   3. batch_size和学习率是否合适")
    else:
        print("❌ 诊断结果：发现问题！Gate梯度流动异常")
        print("   建议立即检查：")
        print("   1. AddAuxiliaryLoss实现是否正确")
        print("   2. aux_loss是否真的加入了计算图")
        print("   3. 是否有梯度裁剪过于激进")
    print("="*80)

