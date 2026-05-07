# LLM-Pruner：大语言模型的结构化剪枝 — 详解

**LLM-Pruner: Structural Pruning for Large Language Models**

---

## ★ Insight（关键洞察）

1. **结构依赖建模**：首次系统性地分析了LLM的内在结构特性（多头注意力、FFN），提出分组剪枝策略，保持结构完整性

2. **Fisher Information 重要性评估**：利用 Fisher Information 近似评估参数/组重要性，避免了需大量标签数据的传统方法局限

3. **LoRA快速微调恢复**：仅需训练约0.1%的参数即可有效恢复剪枝性能，大大降低了微调成本

4. **无需任务标签**：利用预训练目标的Fisher信息，在少样本甚至无监督场景下实现有效剪枝

5. **结构化vs非结构化剪枝**：明确区分两种剪枝范式，结构化剪枝更适合硬件友好的实际部署

---

## 一、论文核心故事线（先读这里）

**一句话总结**：提出LLM-Pruner，针对LLM结构特性设计分组剪枝策略，利用Fisher Information评估重要性，配合LoRA快速微调恢复性能。

**核心创新点**：
- 系统分析LLM的结构依赖关系
- 提出注意力头组、FFN神经元组等分组策略
- Fisher Information近似评估组重要性
- LoRA低参数量微调

**技术路径**：结构分析 → 重要性评估 → 分组剪枝 → LoRA恢复

---

## 二、研究背景与动机

### 2.1 大语言模型的挑战

GPT-3(175B)、PaLM(540B)等大模型的参数规模给实际部署带来严峻挑战：
- **计算成本**：单次推理需大量浮点运算
- **存储需求**：模型权重占用数百GB存储
- **内存带宽**：推理时需频繁访问模型参数

### 2.2 模型压缩技术对比

| 技术 | 压缩比 | 精度损失 | 实现难度 | 硬件友好度 |
|------|--------|---------|---------|-----------|
| 量化 | 4-8x | 小 | 低 | 中等 |
| 剪枝 | 可变 | 中等 | 中等 | **高** |
| 知识蒸馏 | 可变 | 中等 | 高 | 低 |
| **结构化剪枝** | **可变** | **可控** | **中等** | **高** |

### 2.3 现有方法的局限

**任务依赖问题**：
- Magnitude剪枝需要任务标签
- 少样本场景下效果有限
- 无法处理无监督压缩任务

**结构忽视问题**：
- 传统方法将每个参数独立评估
- 忽视注意力头间的共享依赖
- 破坏FFN的神经元结构

---

## 三、核心问题分析

### 3.1 问题定义

给定预训练LLM $f(x; \theta)$ 和压缩比例 $\rho$，找到最优剪枝掩码 $m$：

$$\min_m \mathcal{L}(f(x; \theta \odot m))$$

约束：$\|m\|_0 \leq (1-\rho) \|\theta\|_0$

### 3.2 技术挑战

**挑战1：结构依赖**
- 多头注意力中同一层的Q/K/V矩阵共享
- FFN中神经元成对出现
- 需要整体评估结构单元

**挑战2：评估效率**
- LLM参数量巨大(7B+)
- 传统Fisher计算需多次前向传播
- 计算开销难以承受

**挑战3：性能恢复**
- 剪枝后模型性能下降
- 全量微调成本高(数十GPU小时)
- 需要高效恢复方法

---

## 四、方法详解（含公式）

### 4.1 LLM结构分析

#### 4.1.1 Transformer架构

典型LLM基于Transformer解码器：
- $L$ 层编码器
- 每层包含：MHA + FFN + LayerNorm
- 参数量主要集中在MHA和FFN

#### 4.1.2 注意力头结构

对于第$l$层的第$h$个注意力头：
$$head_h^l = \text{Attention}(Q_h^l, K_h^l, V_h^l)$$

其中：
$$Q_h^l = X W_q^{l,(h)}, \quad K_h^l = X W_k^{l,(h)}, \quad V_h^l = X W_v^{l,(h)}$$

**共享依赖**：同一层的Q/K/V来自同一输入$X$

#### 4.1.3 FFN结构

$$FFN(x) = W_2 \cdot \sigma(W_1 \cdot x)$$

其中$\sigma$通常为GeLU/GELU激活函数。

**神经元依赖**：第$i$个中间神经元$z_i$依赖$(W_{1,:,i}, W_{2,i,:})$两个权重向量

### 4.2 重要性评估

#### 4.2.1 Fisher Information基础

Fisher Information衡量参数对模型似然的贡献：

$$F_i = \mathbb{E}_{x \sim D}\left[\left(\frac{\partial \log f(x;\theta)}{\partial \theta_i}\right)^2\right]$$

直观理解：Fisher信息大的参数对该参数的扰动会显著影响模型输出。

#### 4.2.2 均值近似

为降低计算复杂度，使用均值近似：

$$F_i \approx \frac{1}{T} \sum_{t=1}^{T} \left(\frac{\partial \mathcal{L}(x^{(t)}; \bar{\theta})}{\partial \bar{\theta}_i}\right)^2$$

其中$\bar{\theta}_i$是参数均值，$T$是采样样本数。

#### 4.2.3 组重要性

**注意力头组**：
$$I_{head_h^l} = F_{W_q^{l,(h)}} + F_{W_k^{l,(h)}} + F_{W_v^{l,(h)}}$$

**FFN神经元组**：
$$I_{neuron_i^l} = F_{W_1^{l,:,i}} + F_{W_2^{l,i,:}}$$

### 4.3 剪枝掩码生成

根据组重要性排序，选择性保留：

$$\hat{m} = \begin{cases} 1 & \text{if } I_G > \tau \\ 0 & \text{otherwise} \end{cases}$$

其中$\tau$是阈值，决定剪枝比例。

### 4.4 LoRA微调恢复

#### 4.4.1 LoRA原理

对于剪枝后的权重矩阵$W \in \mathbb{R}^{d \times k}$，添加低秩更新：

$$W' = W + \frac{\alpha}{r} BA$$

其中$B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$。

#### 4.4.2 训练目标

$$\min_\phi \mathcal{L}(f(x; \theta^{pruned}, \phi))$$

仅训练LoRA参数$\phi = (A, B)$，固定原始权重。

#### 4.4.3 恢复效果

实验表明，LoRA微调可恢复约60-80%的性能损失。

### 4.5 算法伪代码

```
Algorithm: LLM-Pruner
Input: Model f, prune ratio ρ, data D, LoRA rank r
Output: Pruned model f'

1. // Step 1: 结构分析
2. groups = identify_groups(f)  // 注意力头、FFN神经元等

3. // Step 2: 重要性计算
4. for g in groups:
5.     I_g = compute_FI(g, D)   // Fisher Information

6. // Step 3: 分组剪枝
7. thresholds = find_threshold(groups, ρ)
8. mask = generate_mask(groups, I_g, thresholds)

9. // Step 4: 应用剪枝
10. f_pruned = apply_mask(f, mask)

11. // Step 5: LoRA微调
12. for epoch in training_epochs:
13.     update_lora_params(f_pruned, D)

14. return f_pruned
```

---

## 五、实验设计与结果分析

### 5.1 实验设置

**模型配置**：
| 模型 | 参数量 | 层数 | 隐藏维度 |
|------|--------|------|---------|
| LLaMA-7B | 7B | 32 | 4096 |
| Vicuna-7B | 7B | 32 | 4096 |
| LLaMA-13B | 13B | 40 | 5120 |

**评测任务**：
- ARC：科学问答
- BoolQ：布尔推理
- Copa：因果推理
- HellaSwag：常识推理
- Piqa：物理问答

### 5.2 主要结果

**表1：零样本准确率对比(20%稀疏度)**

| 方法 | LLaMA-7B | Vicuna-7B | LLaMA-13B |
|------|----------|-----------|-----------|
| 原始 | 75.8% | 76.2% | 78.1% |
| SparseGPT | 74.1% | 74.5% | 76.2% |
| Magnav.jl | 72.5% | 73.1% | 74.8% |
| Wanda | 73.5% | 73.8% | 75.6% |
| **LLM-Pruner** | **74.8%** | **75.1%** | **77.0%** |

**关键发现**：
1. LLM-Pruner在所有稀疏度下优于非结构化方法
2. 大模型(LLaMA-13B)上效果更明显
3. 推理速度提升与稀疏度近似线性

### 5.3 压缩效率分析

**表2：压缩效率对比**

| 方法 | 稀疏度 | 推理加速 | 微调成本 | 性能损失 |
|------|--------|---------|---------|---------|
| 原始 | 0% | 1.0x | - | 0% |
| Magnitude | 20% | 1.2x | 0h | 3.3% |
| SparseGPT | 20% | 1.3x | 1h | 1.7% |
| **LLM-Pruner** | **20%** | **1.8x** | **3h** | **1.0%** |

---

## 六、消融实验

### 6.1 结构分组策略的影响

| 分组策略 | 稀疏度 | 准确率 |
|---------|--------|--------|
| 随机剪枝 | 20% | 70.2% |
| 单独权重 | 20% | 72.8% |
| 注意力头组 | 20% | 74.1% |
| FFN神经元组 | 20% | 73.8% |
| **全部分组** | **20%** | **74.8%** |

**结论**：组合多种分组策略效果最佳。

### 6.2 Fisher近似的敏感性

| 采样数T | 计算时间 | 准确率 |
|---------|---------|--------|
| 1 | 0.5h | 73.1% |
| 10 | 2h | 74.5% |
| 50 | 8h | 74.8% |
| **100** | **15h** | **74.9%** |

**结论**：T=50时已达较好效果，继续增加收益有限。

### 6.3 LoRA秩的影响

| LoRA秩r | 参数量 | 恢复效果 | 微调时间 |
|---------|--------|---------|---------|
| 4 | 0.02M | +2.1% | 2h |
| 8 | 0.05M | +3.8% | 2.5h |
| 16 | 0.1M | +4.7% | 3h |
| 32 | 0.2M | +4.8% | 4h |

**结论**：r=16是效果与效率的最佳平衡点。

---

## 七、与相关工作的关系

### 7.1 与SparseGPT的关系

SparseGPT是最早的LLM剪枝工作之一，采用OBS(Optimal Brain Surgeon)框架。但：
- 忽视结构约束
- 非结构化稀疏
- 需要更多计算资源

LLM-Pruner继承了重要性评估思想，但增加了结构约束。

### 7.2 与Magnitude剪枝的关系

Magnitude剪枝根据权重绝对值大小评估重要性：
- 简单高效
- 但忽视任务相关性
- 对LLM效果不佳

LLM-Pruner通过Fisher Information捕捉任务相关性。

### 7.3 与LoRA的关系

LoRA最初用于高效微调，本文将其应用于剪枝后恢复。

---

## 八、方法优缺点深度评析

### 8.1 优点

1. **结构感知**：首次系统考虑LLM的结构特性
2. **硬件友好**：结构化剪枝便于硬件加速
3. **无需任务标签**：Fisher信息基于预训练目标
4. **高效恢复**：LoRA微调成本低
5. **可解释性**：重要性分数可解释

### 8.2 缺点

1. **组边界假设**：固定的结构分组可能非最优
2. **静态阈值**：需预设稀疏度，不自适应
3. **层间依赖忽视**：未考虑跨层信息流
4. **稀疏度限制**：高稀疏度时性能下降明显

### 8.3 潜在改进方向

1. **数据驱动的分组**：根据激活模式动态分组
2. **渐进式剪枝**：从低稀疏度逐步增加
3. **自适应阈值**：根据输入内容调整
4. **跨层建模**：考虑残差连接的依赖

---

## 九、未来研究方向

### 9.1 更细粒度的结构发现

当前方法基于人工定义的结构，未来可探索：
- 自动发现最优结构分组
- 根据激活模式动态调整
- 注意力头合并/分裂

### 9.2 动态剪枝策略

探索推理时动态调整模型规模：
- 根据输入复杂度自适应剪枝
- 不同层使用不同稀疏度
- 级联剪枝加速

### 9.3 与其他压缩方法结合

探索与量化的协同压缩：
- 量化+剪枝联合优化
- 知识蒸馏指导剪枝
- 多方法互补

---

## 十、核心公式与理论推导

### 10.1 Fisher Information定义

$$F(\theta) = \mathbb{E}_x\left[\nabla_\theta \log f(x;\theta) \nabla_\theta \log f(x;\theta)^\top\right]$$

对角近似：
$$F_i \approx \mathbb{E}_x\left[\left(\frac{\partial \log f(x;\theta)}{\partial \theta_i}\right)^2\right]$$

### 10.2 期望Fisher估计

使用采样近似期望：
$$F_i \approx \frac{1}{|D_{eval}|} \sum_{x \in D_{eval}} \left(\frac{\partial \log f(x;\theta)}{\partial \theta_i}\right)^2$$

### 10.3 组剪枝目标

$$\min_{m \in \{0,1\}^G} -\sum_{g} m_g I_g \quad \text{s.t. } \|m\|_0 \leq K$$

即最大化保留组的重要性总和。

---

## 十一、术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| 结构化剪枝 | Structured Pruning | 按结构单元(层/头/神经元)剪枝 |
| 非结构化剪枝 | Unstructured Pruning | 按独立权重剪枝 |
| Fisher Information | Fisher Information | 衡量参数重要性的信息论指标 |
| 组剪枝 | Group Pruning | 将相关参数分组后统一剪枝 |
| 注意力头 | Attention Head | 多头注意力的子模块 |
| FFN | Feed-Forward Network | 前馈神经网络层 |
| LoRA | Low-Rank Adaptation | 低秩适应微调技术 |
| 低秩分解 | Low-Rank Decomposition | 用低秩矩阵近似原始矩阵 |
| 压缩比 | Compression Ratio | 原始大小/压缩后大小 |
| 稀疏度 | Sparsity | 零值参数比例 |
