# DiTX GateAttn 训练参数详解

本文档详细说明了DiTX GateAttn模型在训练过程中记录的所有参数、含义及参考范围。

---

## 📊 1. 基础训练损失指标

| 参数名 | 含义 | 参考范围 | 说明 |
|--------|------|----------|------|
| `train_loss` | 训练总损失 | 0.001 - 1.0 | 应逐渐下降，收敛时通常<0.1 |
| `val_loss` | 验证集损失 | 0.001 - 1.0 | 应与train_loss接近，差距过大说明过拟合 |
| `loss_flow` | Flow Matching损失 | 0.001 - 0.5 | 瞬时速度预测误差，应稳定下降 |
| `loss_ct` | Consistency Training损失 | 0.001 - 0.5 | 一致性约束损失，确保不同时间步预测一致 |
| `bc_loss` | 行为克隆总损失 | 0.001 - 1.0 | loss_flow + loss_ct的总和 |

### 说明
- **Flow Matching**: 训练模型预测从噪声到数据的瞬时速度场
- **Consistency Training**: 确保模型在不同时间步的预测保持一致性
- 两者结合提供更稳定的训练和更快的推理速度

---

## 📈 2. 速度预测指标

| 参数名 | 含义 | 参考范围 | 说明 |
|--------|------|----------|------|
| `v_flow_pred_magnitude` | Flow速度预测幅值 | 0.1 - 5.0 | 预测速度向量的L2范数，过大可能不稳定 |
| `v_ct_pred_magnitude` | Consistency速度预测幅值 | 0.1 - 5.0 | 应与v_flow_pred_magnitude接近 |

### 说明
- 速度幅值反映模型预测的动作变化强度
- 两个指标应该接近，差异过大说明训练不平衡
- 如果幅值过大(>10)，可能需要调整学习率或归一化策略

---

## 🚪 3. Gate-Attention 核心指标

> **注意**: 仅当配置文件中设置 `use_gate_attn=True` 时才会记录这些指标

### 3.1 全局门控统计

| 参数名 | 含义 | 参考范围 | 说明 |
|--------|------|----------|------|
| `gate/mean_activation` | 平均门控激活值 | 0.3 - 0.7 | 所有层的平均gate值，0.5表示平衡 |
| `gate/std_activation` | 门控激活标准差 | 0.1 - 0.3 | 衡量gate值的分散程度 |
| `gate/saturation_high_ratio` | 高饱和度比例 (>0.9) | 0.0 - 0.3 | 过高说明某些模态被过度依赖 |
| `gate/saturation_low_ratio` | 低饱和度比例 (<0.1) | 0.0 - 0.3 | 过高说明某些模态被忽略 |
| `gate/layer_variance` | 层间门控方差 | 0.01 - 0.1 | 衡量不同层gate值的差异性 |

#### 解读要点
- **mean_activation**: 
  - 接近0.5表示模型在不同模态间保持平衡
  - 过低(<0.3)说明模型倾向抑制信息流
  - 过高(>0.7)说明gate机制未充分发挥作用
  
- **饱和度比例**:
  - `saturation_high_ratio > 0.4`: 警告！某些模态被过度依赖
  - `saturation_low_ratio > 0.4`: 警告！某些模态被严重忽略
  - 理想情况下两者都应<0.2

- **layer_variance**:
  - 适中的方差(0.02-0.08)表示不同层学到了不同的特征层级
  - 过低(<0.01)可能说明层间差异不足
  - 过高(>0.15)可能说明训练不稳定

### 3.2 层级对比指标

| 参数名 | 含义 | 参考范围 | 说明 |
|--------|------|----------|------|
| `gate/early_layers_mean` | 早期层(前1/3)平均gate | 0.3 - 0.7 | 早期层通常更关注低级特征 |
| `gate/late_layers_mean` | 后期层(后1/3)平均gate | 0.3 - 0.7 | 后期层通常更关注高级特征 |
| `gate/early_vs_late_diff` | 早晚层gate差异 | -0.3 - 0.3 | 正值表示后期层gate更高 |

#### 解读要点
- **early_vs_late_diff**:
  - 正值: 后期层更依赖gated信息（常见于精细控制任务）
  - 负值: 早期层更依赖gated信息（常见于感知密集任务）
  - 接近0: 各层均衡使用gate机制

### 3.3 模态特定门控值

| 参数名 | 含义 | 参考范围 | 说明 |
|--------|------|----------|------|
| `gate/modality_head_mean` | Head相机平均gate | 0.0 - 1.0 | 头部相机的重要性权重 |
| `gate/modality_head_std` | Head相机gate标准差 | 0.0 - 0.3 | 层间差异 |
| `gate/modality_wrist_mean` | Wrist相机平均gate | 0.0 - 1.0 | 腕部相机的重要性权重 |
| `gate/modality_wrist_std` | Wrist相机gate标准差 | 0.0 - 0.3 | 层间差异 |
| `gate/modality_tactile_mean` | 触觉传感器平均gate | 0.0 - 1.0 | 触觉信息的重要性权重 |
| `gate/modality_tactile_std` | 触觉gate标准差 | 0.0 - 0.3 | 层间差异 |
| `gate/modality_proprio_mean` | 本体感知平均gate | 0.0 - 1.0 | 关节位置/速度的重要性权重 |
| `gate/modality_proprio_std` | 本体感知gate标准差 | 0.0 - 0.3 | 层间差异 |

#### 解读要点
- **模态均值**:
  - 所有模态均值应在0.2-0.8之间，表示模型有效利用各模态
  - 某模态<0.1: 该模态几乎被忽略，可能需要调整初始化或数据质量
  - 某模态>0.9: 该模态被过度依赖，可能导致过拟合

- **模态标准差**:
  - 适中的标准差(0.1-0.25)表示不同层对该模态的使用策略不同
  - 过低(<0.05)可能说明该模态的使用模式过于固定
  - 过高(>0.35)可能说明训练不稳定

#### 典型任务的模态权重分布

**精细操作任务** (如插入、组装):
- `tactile_mean`: 0.4-0.7 (高)
- `proprio_mean`: 0.5-0.8 (高)
- `wrist_mean`: 0.4-0.6 (中高)
- `head_mean`: 0.2-0.4 (中低)

**视觉导航任务** (如抓取、放置):
- `head_mean`: 0.5-0.8 (高)
- `wrist_mean`: 0.4-0.6 (中高)
- `proprio_mean`: 0.3-0.5 (中)
- `tactile_mean`: 0.1-0.3 (低)

---

## 👁️ 4. Attention 注意力指标

> **注意**: 每500步记录一次，需要额外的前向传播

| 参数名 | 含义 | 参考范围 | 说明 |
|--------|------|----------|------|
| `attn/entropy` | 注意力熵(归一化) | 0.5 - 1.0 | 0=极度集中, 1=均匀分布 |
| `attn/entropy_diff` | 早晚层熵差异 | -0.3 - 0.3 | 正值表示早期层更均匀 |
| `attn/modality_rgb` | RGB相机注意力占比 | 0.3 - 0.8 | head + wrist相机的总注意力 |
| `attn/modality_head` | Head相机注意力占比 | 0.0 - 0.5 | 头部相机的注意力权重 |
| `attn/modality_wrist` | Wrist相机注意力占比 | 0.0 - 0.5 | 腕部相机的注意力权重 |
| `attn/modality_tactile` | 触觉注意力占比 | 0.0 - 0.4 | 触觉传感器的注意力权重 |
| `attn/modality_proprio` | 本体感知注意力占比 | 0.0 - 0.4 | 关节状态的注意力权重 |

### 说明
- **注意力熵**:
  - 高熵(>0.8): 模型关注广泛，适合探索阶段
  - 中熵(0.5-0.8): 平衡状态，通常是最优的
  - 低熵(<0.3): 注意力崩塌，模型过度关注少数token

- **模态注意力占比**:
  - 所有模态占比之和应≈1.0
  - 与gate值对比可以理解"模型想关注什么"(attention)与"实际使用多少"(gate)的差异

### Attention vs Gate 的区别

| 维度 | Attention | Gate |
|------|-----------|------|
| **含义** | 模型"想要"关注的信息 | 模型"实际使用"的信息量 |
| **计算方式** | Softmax(QK^T) | Sigmoid(可学习参数) |
| **作用范围** | Token级别的权重分配 | 模态级别的信息流控制 |
| **典型模式** | 动态变化，依赖输入 | 相对稳定，学习任务特性 |

---

## 🔗 5. 相关性指标

> **注意**: 仅当有gate统计时才会记录

| 参数名 | 含义 | 参考范围 | 说明 |
|--------|------|----------|------|
| `correlation/loss_vs_tactile_gate` | 损失与触觉gate相关性 | 0.0 - 1.0 | loss × tactile_gate，帮助理解模态重要性 |
| `correlation/loss_vs_proprio_gate` | 损失与本体感知gate相关性 | 0.0 - 1.0 | loss × proprio_gate |

### 说明
- 这些指标帮助理解"哪些模态对降低损失更重要"
- 高相关性表示该模态的gate值与任务性能强相关
- 可用于指导模态融合策略的优化

---

## ⚙️ 6. 训练状态指标

| 参数名 | 含义 | 参考范围 | 说明 |
|--------|------|----------|------|
| `global_step` | 全局训练步数 | 0 - ∞ | 累计训练的batch数 |
| `epoch` | 训练轮次 | 0 - num_epochs | 当前训练到第几轮 |
| `lr` | 学习率 | 1e-5 - 1e-3 | 当前优化器学习率 |
| `train_action_mse_error` | 动作预测MSE | 0.001 - 0.1 | 采样batch的动作重建误差 |

### 说明
- `lr`: 通常使用warmup + cosine decay策略
- `train_action_mse_error`: 直接衡量动作预测质量，应与loss趋势一致

---

## 🎯 关键指标解读建议

### ✅ 健康训练的特征

1. **损失下降稳定**
   - `train_loss`和`val_loss`稳定下降且接近
   - `loss_flow`和`loss_ct`比例相对稳定(通常接近1:1)

2. **Gate机制平衡**
   - `gate/mean_activation` ≈ 0.5
   - `gate/saturation_high_ratio` < 0.2
   - `gate/saturation_low_ratio` < 0.2

3. **模态利用多样**
   - 各模态gate值都在0.2-0.8之间
   - 没有极端值(接近0或1)
   - 模态权重分布符合任务特性

4. **注意力分布合理**
   - `attn/entropy` > 0.6
   - 注意力模态占比与gate值相呼应
   - 早晚层熵差异适中(-0.2 to 0.2)

5. **速度预测稳定**
   - `v_flow_pred_magnitude`和`v_ct_pred_magnitude`接近
   - 幅值在合理范围内(0.5-3.0)

### ⚠️ 需要关注的异常情况

#### 1. Gate饱和问题
**症状**: `gate/saturation_high_ratio` > 0.4
- **原因**: Gate bias初始化不当，或某些模态数据质量过高
- **解决方案**:
  - 调整gate bias初始化策略
  - 检查数据质量，确保各模态信息量平衡
  - 降低学习率，让gate更缓慢地调整

#### 2. 模态忽略问题
**症状**: 某个`gate/modality_*_mean` < 0.1
- **原因**: 该模态数据质量差，或特征提取不充分
- **解决方案**:
  - 检查该模态的数据预处理流程
  - 增强该模态的特征编码器
  - 调整gate bias，给予该模态更高的初始权重

#### 3. 注意力崩塌
**症状**: `attn/entropy` < 0.3
- **原因**: 模型过度关注少数token，泛化能力下降
- **解决方案**:
  - 增加dropout比例
  - 使用attention regularization
  - 检查数据多样性

#### 4. 训练不稳定
**症状**: `loss_flow`和`loss_ct`差异过大(比例>3:1)
- **原因**: Flow和Consistency训练不平衡
- **解决方案**:
  - 调整`flow_batch_ratio`和`consistency_batch_ratio`
  - 检查EMA模型更新策略
  - 降低学习率

#### 5. 速度预测异常
**症状**: `v_flow_pred_magnitude` > 10 或两个速度幅值差异>2倍
- **原因**: 训练不稳定，或归一化问题
- **解决方案**:
  - 检查动作归一化策略
  - 降低学习率
  - 增加gradient clipping

---

## 📊 WandB监控建议

### 必看曲线组合

**组合1: 损失监控**
- `train_loss`, `val_loss`, `loss_flow`, `loss_ct`
- 应该: 稳定下降，train/val接近，flow/ct平衡

**组合2: Gate健康度**
- `gate/mean_activation`, `gate/saturation_high_ratio`, `gate/saturation_low_ratio`
- 应该: mean≈0.5，两个饱和度<0.2

**组合3: 模态权重**
- `gate/modality_*_mean` (所有模态)
- 应该: 符合任务特性，无极端值

**组合4: Attention vs Gate**
- `attn/modality_*` vs `gate/modality_*`
- 应该: 趋势一致，但gate更稳定

**组合5: 层级分析**
- `gate/early_layers_mean`, `gate/late_layers_mean`, `gate/early_vs_late_diff`
- 应该: 有一定差异但不极端

---

## 🔧 配置文件参数对应

### Gate-Attention相关配置

```yaml
policy:
  _target_: maniflow.policy.maniflow_image_policy.ManiFlowTransformerImagePolicy
  use_gate_attn: true  # 启用Gate-Attention
  gate_type: 'elementwise'  # 'none', 'headwise', 'elementwise'
  
  # 其他相关参数
  n_layer: 8  # Transformer层数，影响gate统计的层数
  n_head: 8   # 注意力头数
  qkv_bias: false
  qk_norm: false
```

### 训练相关配置

```yaml
training:
  # Flow vs Consistency比例
  flow_batch_ratio: 0.75
  consistency_batch_ratio: 0.25
  
  # 采样策略
  sample_t_mode_flow: "beta"
  sample_t_mode_consistency: "discrete"
  sample_dt_mode_consistency: "uniform"
  sample_target_t_mode: "relative"
  
  # EMA配置(影响consistency training)
  use_ema: true
  
ema:
  _target_: manifold.model.diffusion.ema_model.EMAModel
  decay: 0.999
```

---

## 📝 代码位置索引

### 参数记录位置

1. **基础损失**: `maniflow_image_policy.py:632-651`
2. **Gate统计**: `ditx_gateattn.py:423-497`
3. **Attention统计**: `ditx_gateattn.py:522-624`
4. **训练循环**: `train_maniflow_robotwin2_workspace.py:304-340`

### 关键函数

- `get_gate_stats()`: 获取Gate-Attention统计
- `get_attn_stats(modality_info)`: 获取注意力统计
- `compute_loss()`: 计算损失并收集统计信息
- `set_record_attn(bool)`: 启用/禁用注意力记录

---

## 🚀 快速诊断检查清单

训练开始后，按以下顺序检查：

- [ ] **第1-100步**: 损失是否开始下降？
- [ ] **第100-500步**: Gate均值是否在0.3-0.7？
- [ ] **第500步**: 查看第一次attention统计，熵是否>0.5？
- [ ] **第1000步**: 各模态gate是否都>0.15？
- [ ] **第5000步**: 饱和度比例是否<0.3？
- [ ] **第10000步**: train_loss和val_loss是否接近？
- [ ] **整个训练**: 速度幅值是否稳定在0.5-5.0？

---

## 📚 参考文献

1. **Flow Matching**: [Flow Matching for Generative Modeling (ICLR 2023)](https://arxiv.org/abs/2210.02747)
2. **Consistency Models**: [Consistency Models (ICML 2023)](https://arxiv.org/abs/2303.01469)
3. **Gate-Attention**: Qwen3 Architecture
4. **DiT**: [Scalable Diffusion Models with Transformers (ICCV 2023)](https://arxiv.org/abs/2212.09748)

---

**文档版本**: v1.0  
**最后更新**: 2026-02-02  
**维护者**: ManiFlow Team
