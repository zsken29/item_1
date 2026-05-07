# Token-Adaptive Multi-layer Pruning for Multimodal Large Language Models

## TAMP：多模态大语言模型的Token自适应逐层剪枝

---

**作者**: Zhicheng Duan, Jing Wu, Kaiyan Guo, Huawei Huang, Xueqian Wang, Xiaobo Guo

**单位**: 北京通用人工智能研究院 · 北京大学 · 百度

**发表**: ICLR 2025

---

## 摘要

本文提出TAMP（Token-Adaptive Multi-layer Pruning），一种面向多模态大语言模型（MLLM）的Token自适应逐层剪枝框架。核心思想是：不同样本在同一层的最优剪枝率不同，同一样本在不同层的最优剪枝率也不同。TAMP通过学习样本级与层级自适应剪枝率，在保证性能的同时显著降低计算开销。在图像理解和视频理解任务上的实验表明，TAMP在保持精度的前提下实现了显著的推理加速。

---

## 1 引言

多模态大语言模型（如LLaVA、Qwen-VL等）将强大的语言能力与视觉理解相结合，在各类任务上取得了突破性进展。然而，视觉Token数量的快速增长带来了严重的计算负担。以LLaVA-1.5为例，一张224×224的图像经过ViT编码后产生256个视觉Token，经MLP投影后送入LLM处理。这一过程中，视觉Token的处理成本远高于文本Token，成为推理效率的瓶颈。

现有方法多采用统一剪枝率，即对所有样本、所有层采用相同的剪枝比例。这种做法忽略了两个关键事实：

1. **样本间差异**：不同图像的信息密度差异显著。简单图像可能仅需少量视觉Token即可正确理解，而复杂图像需要更多Token。
2. **层间差异**：不同层的Attention模式不同，早期层捕获更多低级视觉特征，后期层更多关注语义整合。统一剪枝会过早丢失关键视觉信息，或在后期保留冗余Token。

TAMP的核心贡献是：为每个样本的每层学习最优的Token剪枝率。具体而言，引入一个轻量级的"剪枝率预测器"，以样本级特征和层级特征为输入，输出各层应保留的Token比例。

---

## 2 方法

### 2.1 预备知识：视觉Token剪枝

给定视觉Token序列 $V = \{v_1, v_2, ..., v_N\}$，剪枝的目标是选择子集 $V' \subseteq V$ 使得 $|V'| < N$，同时最大化任务性能。常见方法包括：

- **统一剪枝**：所有层保留固定比例 $r$ 的Token
- **层级剪枝**：不同层使用不同比例，但各样本相同
- **样本级剪枝**：不同样本使用不同比例，但各层相同

这些方法均无法同时捕捉样本间与层间的差异。

### 2.2 Token自适应剪枝率学习

TAMP的核心是剪枝率预测器（Pruning Ratio Predictor, PRP）。该预测器以两个输入为条件：

1. **样本级条件 $c_s$**：来自ViT编码器最后一层的[CLS]Token embedding，反映图像整体复杂度
2. **层条件 $l_i$**：当前层的深度位置（归一化至[0,1]），反映当前层的语义处理阶段

预测器由一个小型MLP组成：
$$\hat{r}_i = \text{MLP}([c_s; \; l_i])$$

其中 $\hat{r}_i$ 为第 $i$ 层保留Token的预测比例（0到1之间）。

### 2.3 多层Token剪枝

在得到各层预测比例后，TAMP对每一层独立执行Token剪枝。具体步骤：

**Step 1: 重要性评分**
对第 $i$ 层的视觉Token $V_i$，计算每个Token的重要性分数。本文使用Query-Key相似度作为重要性指标：
$$s_j = \sum_k \text{Softmax}(Q_i \cdot K_{i,k})_j$$

即Token $j$ 在Attention中作为Value被关注的加权求和。分数越高，说明该Token对当前层的表示贡献越大。

**Step 2: Top-k 选择**
根据预测比例 $\hat{r}_i$，选择前 $k = \lfloor N \cdot \hat{r}_i \rfloor$ 个Token保留。

**Step 3: 层级递进**
早期层的剪枝结果（保留的Token集合）作为后续层的输入，形成逐层递进的Token流。这一设计确保了信息流的方向性——早期层保留的Token更容易被后续层利用。

### 2.4 优化目标

TAMP的优化目标包含两项：

$$\mathcal{L} = \mathcal{L}_{\text{VLM}} + \lambda \cdot \mathcal{L}_{\text{sparse}}$$

- **任务损失 $\mathcal{L}_{\text{VLM}}$**：标准MLLM的VQA损失或captioning损失
- **稀疏性损失 $\mathcal{L}_{\text{sparse}}$**：鼓励低剪枝率的正则项，防止所有层都保留大量Token

稀疏性损失定义为：
$$\mathcal{L}_{\text{sparse}} = - \frac{1}{L} \sum_{i=1}^{L} \hat{r}_i$$

即最小化平均预测剪枝率，从而最大化Token减少量。

### 2.5 训练流程

TAMP采用两阶段训练：

**阶段1：剪枝率预测器热启动**
冻结MLLM主体，仅训练PRP。采样少量图像-文本对，使PRP学习基本的空间视觉重要性模式。

**阶段2：联合微调**
解冻PRP和MLLM后几层，联合优化任务性能与稀疏性。

---

## 3 实验

### 3.1 实验设置

- **基础模型**: LLaVA-1.5-7B, Vicuna-7B-v1.5
- **视觉编码器**: CLIP-ViT-L/14
- **评估任务**: 图像理解（GQA, VQAv2, VizWiz）和视频理解（MSRVTT QA）
- **剪枝比例范围**: 10%-90%

### 3.2 主要结果

| 方法 | GQA (Acc) | VQAv2 | 视觉Token数 | 加速比 |
|------|-----------|-------|-------------|--------|
| LLaVA-1.5 | 62.0 | 76.8 | 576 | 1.0x |
| TAMP (Ours) | 61.8 | 76.5 | 288 | 1.9x |
| SparseLLM | 60.1 | 74.9 | 288 | 1.9x |
| LLMRazor | 59.8 | 74.5 | 288 | 1.9x |

TAMP在保留50% Token的情况下，性能损失极小（<0.3%），推理速度提升近2倍。

### 3.3 消融实验

- **PRP的贡献**：移除PRP（使用均匀剪枝）导致性能下降1.5-2.0%，说明自适应的重要性。
- **逐层递进 vs 并行剪枝**：逐层递进设计比各层独立剪枝高0.8%，因为早期层的选择对后期有指导作用。
- **层条件的作用**：加入层条件后，PRP能够区分低层（保留更多Token）和高层（保留更少Token），符合视觉语义整合的直觉。

---

## 4 结论

TAMP提出了一种样本级和层级自适应相结合的视觉Token剪枝方法。核心贡献包括：

1. 剪枝率预测器（PRP）实现样本-层级双条件自适应
2. 逐层递进的Token流设计保留信息流方向性
3. 在图像和视频理解任务上验证了方法的有效性

局限性在于PRP本身引入了一定计算开销，且目前仅针对视觉Token剪枝，未探索文本Token剪枝的可能性。

---

## 参考信息

- **原文标题**: Token-Adaptive Multi-layer Pruning for Multimodal Large Language Models
- **会议**: ICLR 2025
- **GitHub**: https://github.com/TAMP-ICLR/TAMP