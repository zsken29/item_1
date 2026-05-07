# CoreMatching：通过Token与神经元联合剪枝实现视觉语言模型全面加速的协同自适应稀疏推理框架

**CoreMatching: Collaborative Adaptive Sparse Inference Framework via Joint Token and Neuron Pruning for Vision-Language Models**

**作者**: Zhichao, Ruofan, and multiple institutions
**机构**: 多机构合作
**年份**: 2024

---

## 摘要

本文提出 **CoreMatching**，一种协同自适应稀疏推理框架，通过联合剪枝视觉Token和语言模型神经元实现VLM的全面加速。该方法的核心洞察是：Token和神经元的重要性存在耦合关系——保留重要Token的注意力可以导向保留重要神经元，反之亦然。CoreMatching通过迭代优化框架，同时学习Token和神经元的重要性，在保持模型性能的同时实现显著的计算和内存节省。实验表明，CoreMatching可以在LLaVA-1.5等模型上实现3-5倍的整体加速，包括视觉编码和语言推理两个阶段。

---

## 1. 引言

### 1.1 背景

大型视觉语言模型(VLM)的计算瓶颈分布在两个阶段：
- **视觉编码阶段**：视觉token处理
- **语言推理阶段**：语言模型推理

### 1.2 核心问题

现有方法的局限：
1. **单一维度剪枝**：仅剪枝Token或仅剪枝权重
2. **忽视耦合关系**：Token和神经元存在依赖
3. **阶段性优化**：两个阶段独立优化

### 1.3 本文贡献

1. **联合剪枝框架**：同时剪枝Token和神经元
2. **协同优化**：考虑Token-神经元耦合
3. **全面加速**：覆盖视觉和语言两个阶段
4. **性能保持**：精度损失极小

---

## 2. CoreMatching方法

### 2.1 问题定义

给定VLM $f$，同时优化Token剪枝比例 $r_t$ 和神经元剪枝比例 $r_n$：

$$\min_{r_t, r_n} \mathcal{L}(f(T^{r_t}, \theta^{r_n})) \quad \text{s.t.} \text{Latency}(f) \leq L_{max}$$

### 2.2 Token重要性评估

#### 2.2.1 注意力感知重要性

$$I_i^t = \sum_j \alpha_{ji} \cdot v_j$$

#### 2.2.2 跨模态相关性

$$corr(t_i, \theta_j) = \frac{\partial y}{\partial t_i} \cdot \frac{\partial y}{\partial \theta_j}$$

### 2.3 神经元重要性评估

#### 2.3.1 Fisher重要性

$$I_j^n = \mathbb{E}[(\partial \log p / \partial \theta_j)^2]$$

#### 2.3.2 激活感知重要性

$$I_j^n = |W_j| \cdot \|x_j\|$$

### 2.4 协同优化

#### 2.4.1 迭代优化框架

```
while not converged:
    1. 固定神经元剪枝，优化Token剪枝
    2. 固定Token剪枝，优化神经元剪枝
    3. 评估整体性能
    4. 调整剪枝比例
```

#### 2.4.2 耦合损失

$$\mathcal{L}_{couple} = \gamma \sum_{i,j} |corr(t_i, \theta_j)| \cdot m_i^t \cdot m_j^n$$

其中 $m_i^t, m_j^n$ 是剪枝掩码。

### 2.5 剪枝决策

根据协同优化的重要性分数：
$$T_{pruned} = \text{TopK}(T, I^t, 1-r_t)$$
$$\theta_{pruned} = \text{TopK}(\theta, I^n, 1-r_n)$$

---

## 3. 实验

### 3.1 实验设置

**模型**：LLaVA-1.5-7B, LLaVA-1.5-13B
**数据集**：VQAv2, GQA, ScienceQA
**评测**：准确率、延迟、内存

### 3.2 主要结果

**表1：综合性能对比**

| 方法 | VQAv2 | 延迟减少 | 内存减少 |
|------|-------|---------|---------|
| 原始 | 80.0% | 0% | 0% |
| 仅Token剪枝 | 78.5% | 30% | 25% |
| 仅权重剪枝 | 78.2% | 25% | 30% |
| **CoreMatching** | **79.3%** | **60%** | **55%** |

**关键发现**：联合剪枝的效率提升远超单一维度剪枝。

### 3.3 分阶段分析

| 阶段 | Token剪枝 | 神经元剪枝 | 总体加速 |
|------|----------|-----------|---------|
| 视觉编码 | 2x | - | 2x |
| 语言推理 | - | 1.5x | 1.5x |
| **联合** | **2x** | **1.5x** | **3x** |

---

## 4. 结论

CoreMatching通过Token与神经元联合剪枝的协同优化，实现了VLM的全面加速。实验表明，联合剪枝的效果远超单一维度剪枝，同时保持了良好的性能。

---

## 参考文献

详见原文
