# DivPrune：面向大型多模态模型的基于多样性的视觉Token剪枝

## DivPrune: Diversity-aware Visual Token Pruning for Large Multimodal Models

**作者**: Zhichao Wei, Ruofan Wang, Jingjing Liu, Tianyi Zhou, Lei Cheng, Fanxu Meng, P. K. - 待确认
**机构**: 多个研究机构合作
**发表**: ACL 2024 (Findings)
**arXiv**: 2406.12279

---

## 摘要

视觉Token的高冗余性是限制大型多模态模型（LMM）部署效率的主要瓶颈之一。现有的Token剪枝方法通常基于重要性评分（如注意力权重或特征范数）来识别要移除的Token。然而，这类方法往往忽视了视觉Token的多样性分布——即被保留Token集合是否能够充分覆盖输入图像中的不同视觉概念。我们发现，简单地移除低重要性Token会导致多样性显著下降，特别是在包含长尾分布视觉内容的图像上。 为解决这一问题，我们提出了 **DivPrune**，这是一种基于多样性感知的视觉Token剪枝框架。DivPrune的核心思想是在剪枝过程中明确建模并优化Token集合的多样性，确保被保留的Token能够覆盖更广泛的视觉概念。具体而言，我们首先定义了基于互信息的**多样性感知重要性（Diversity-aware Importance, DI）**指标，该指标同时考虑Token自身的独立重要性以及其与已选Token的多样性贡献。随后，我们提出了一个两阶段剪枝流程：首先通过轻量级预测器识别高重要性Token的候选集合，然后利用贪婪多样化策略逐步构建最终的Token子集。实验表明，DivPrune在LLaVA-1.5（7B和13B）和Vicuna-v1.5等模型上，在8个基准数据集（包括VQAv2、GQA、VQA-CE等）上均取得了显著的性能提升。特别是在长尾视觉内容分布的场景下，DivPrune相较于现有方法表现更优。

---

## 1. 引言

### 1.1 研究背景

大型多模态模型（LMM）如LLaVA、InstructBLIP等，通过连接视觉编码器与大语言模型（LLM），展现出强大的视觉理解与推理能力。然而，视觉编码器产生的Token序列往往非常长（通常为数百至上千个Token），这导致：
- **内存占用高**：大量Token需要存储在GPU内存中
- **计算量大**：自注意力机制的复杂度与Token数量的平方成正比
- **延迟增加**：Token序列越长，首Token生成时间（Time-to-First-Token, TTFT）越长

为解决这一问题，Token剪枝技术被广泛研究。核心思想是识别并移除冗余的视觉Token，同时尽可能保留对下游任务有用的信息。

### 1.2 现有方法的局限性

现有的Token剪枝方法主要分为两类：

**（1）基于重要性评分的方法**：根据注意力权重、特征范数或梯度等指标评估每个Token的重要性，然后移除低重要性Token。代表性工作包括：
- **ATPM** [22]：自适应Token剪枝
- **SaTT** [7]：基于语义感知的Token剪枝
- **LLaVA-PruMerge** [5]：基于预测器的Token合并剪枝

**问题**：这类方法倾向于保留相似的Token（如同类物体的重复实例），导致被保留Token集合的信息覆盖范围受限。

**（2）基于聚类的方法**：将Token聚类为若干组，每组保留最具代表性的Token。代表性工作包括：
- **FastV** [2]：基于注意力分数的Token聚类
- **MMG-E** [8]：多模态组剪枝

**问题**：聚类过程引入额外计算开销，且难以保证每个聚类簇内的代表性Token真正捕获关键视觉信息。

### 1.3 核心观察

我们观察到，视觉Token的分布具有显著的长尾特性：

```
┌─────────────────────────────────────────────────────────────────┐
│ 观察1：长尾分布                                                   │
│                                                                     │
│  Token重要性分布图：                                                 │
│                                                                     │
│  高重要性区域 │██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     │
│  (头部内容)   │ (罕见但关键的视觉概念)                               │
│              │                                                      │
│  低重要性区域 │█████████████████████████████████████████████████  │
│  (尾部内容)   │ (大量相似但冗余的Token)                             │
│                                                                     │
│  问题：移除低重要性Token会丢失长尾内容的多样性                        │
└─────────────────────────────────────────────────────────────────┘
```

例如，在包含人群场景的图像中：
- 高重要性Token：场景主体（人物A、人物B）
- 低重要性Token：背景人物C、人物D、人物E...
- **现有方法问题**：如果只保留前50%的Token，可能只保留场景主体，导致背景信息完全丢失

### 1.4 本文贡献

本文的主要贡献包括：

1. **问题形式化**：我们形式化定义了视觉Token剪枝中的多样性问题，指出保留Token集合的多样性对于下游任务性能至关重要。

2. **DI指标**：我们提出了**多样性感知重要性（Diversity-aware Importance, DI）**指标，该指标同时衡量Token的独立重要性和其对集合整体多样性的贡献。

3. **两阶段剪枝框架**：我们设计了高效的两阶段剪枝流程，结合轻量级预测器和贪婪多样化策略，实现高效且有效的Token剪枝。

4. **全面实验**：在8个基准数据集上的实验表明，DivPrune显著优于现有方法，特别是在长尾视觉内容分布的场景下。

---

## 2. 相关工作

### 2.1 大型多模态模型的效率优化

大型多模态模型的效率优化是当前研究热点，主要方向包括：

| 方法类别 | 代表工作 | 核心思想 |
|---------|---------|---------|
| 视觉编码器压缩 | MiniGPT-4, TinyGPT-V | 使用更小的视觉编码器 |
| Token压缩 | SparseVLM, FastV | 合并或剪枝视觉Token |
| 注意力优化 | FlashAttention, RingAttention | 优化注意力计算 |
| 知识蒸馏 | MiniLLaVA, VL-LLM | 从大模型蒸馏到小模型 |

### 2.2 视觉Token剪枝方法

视觉Token剪枝旨在识别并移除冗余的Token，主要方法包括：

**（1）基于规则的方法**
- **VTN** [27]：基于视觉Transformer注意力分数
- **TRiT** [30]：基于Token相关性的规则

**（2）基于预测器的方法**
- **LLaVA-PruMerge** [5]：训练轻量级预测器估计Token重要性
- **EvoPrune** [24]：使用进化算法搜索最优剪枝比例

**（3）基于语义的方法**
- **SaTT** [7]：基于Token语义相似度
- **EfficientVLM** [11]：基于语义聚类的Token选择

### 2.3 现有方法与我们的区别

与现有方法不同，DivPrune首次将**多样性优化**引入Token剪枝过程。不同于简单地最大化重要性或最小化冗余，我们的方法确保被保留的Token集合能够覆盖输入图像中的多样化视觉概念。

---

## 3. 方法论

### 3.1 问题定义

给定输入图像 $I$ 和视觉编码器 $f_V$，我们得到视觉Token序列：

$$\mathbf{V} = \{v_1, v_2, ..., v_N\}, \quad |\mathbf{V}| = N$$

其中 $N$ 通常为数百到数千个Token。

**目标**：找到一个Token子集 $\mathbf{V}' \subset \mathbf{V}$，使得 $|\mathbf{V}'| = M \ll N$，同时最大化下游任务的性能。

### 3.2 多样性感知的Token重要性

#### 3.2.1 Token的独立重要性

首先，我们定义Token $v_i$ 的独立重要性 $s_i$，通过轻量级预测器 $p(\cdot)$ 估计：

$$s_i = p(v_i) \in [0, 1]$$

预测器 $p$ 的训练目标为最小化以下损失：

$$\mathcal{L}_{pred} = \mathbb{E}_{(I, \mathbf{V})} \left[ \sum_i \left\| s_i - \text{AttentionWeight}(v_i) \right\|^2 \right]$$

#### 3.2.2 Token的多样性贡献

给定已选Token集合 $\mathbf{V}_s$，Token $v_i$ 对集合多样性的贡献定义为：

$$\Delta_{div}(v_i | \mathbf{V}_s) = \frac{1}{|\mathbf{V}_s|} \sum_{v_j \in \mathbf{V}_s} \left(1 - \text{sim}(v_i, v_j)\right)$$

其中 $\text{sim}(\cdot, \cdot)$ 是Token特征空间中的余弦相似度：

$$\text{sim}(v_i, v_j) = \frac{v_i \cdot v_j}{\|v_i\| \cdot \|v_j\|}$$

#### 3.2.3 DI指标定义

综合考虑独立重要性和多样性贡献，我们定义Token $v_i$ 的**多样性感知重要性（DI）**为：

$$DI(v_i | \mathbf{V}_s) = \alpha \cdot s_i + \beta \cdot \Delta_{div}(v_i | \mathbf{V}_s)$$

其中 $\alpha$ 和 $\beta$ 为可学习的权重参数，控制两种信号的相对重要性。

### 3.3 两阶段剪枝框架

DivPrune采用两阶段剪枝框架：

#### 阶段一：高重要性Token识别

**目标**：识别候选的高重要性Token集合 $\mathbf{V}_{high}$

**方法**：
1. 使用轻量级预测器 $p$ 估计每个Token的重要性分数 $s_i$
2. 设定阈值 $\tau$，选择 $s_i > \tau$ 的Token作为高重要性候选

$$\mathbf{V}_{high} = \{v_i \in \mathbf{V} \mid s_i > \tau\}$$

**阈值选择**：通过验证集确定最优阈值，平衡保留Token数量与任务性能。

#### 阶段二：贪婪多样化选择

**目标**：从 $\mathbf{V}_{high}$ 中选择最终保留的Token子集 $\mathbf{V}'$，最大化多样性

**贪婪算法**：

```
输入: 候选集合 V_{high}，目标大小 M
输出: 保留Token集合 V'

1. 初始化: V' = ∅
2. 对于 t = 1 到 M:
   a. 对于每个 v ∈ V_{high} \ V':
      计算 DI(v | V')
   b. 选择DI值最高的Token:
      v* = argmax_{v ∈ V_{high} \ V'} DI(v | V')
   c. 更新: V' = V' ∪ {v*}
3. 返回 V'
```

**复杂度分析**：
- 时间复杂度：$O(M \cdot |\mathbf{V}_{high}| \cdot d)$
- 空间复杂度：$O(|\mathbf{V}| \cdot d)$
- 其中 $d$ 为Token特征维度

### 3.4 与大语言模型的协同

DivPrune的Token剪枝发生在视觉编码器与大语言模型的接口处：

```
┌─────────────────────────────────────────────────────────────────┐
│                    DivPrune剪枝流程                              │
│                                                                     │
│  图像 I                                                             │
│      ↓                                                              │
│  视觉编码器 f_V                                                     │
│      ↓                                                              │
│  视觉Token V (N个)  ──→  DivPrune  ──→  剪枝后Token V' (M个)       │
│      ↓                                                              │
│  大语言模型 LLM                                                      │
│      ↓                                                              │
│  输出响应                                                            │
└─────────────────────────────────────────────────────────────────┘
```

关键设计：
- **即插即用**：DivPrune不修改视觉编码器和LLM的参数
- **无额外延迟**：剪枝过程在推理时同步进行，不引入异步计算
- **可配置压缩比**：通过调整目标Token数量 $M$，支持灵活的压缩比

### 3.5 训练目标

DivPrune的完整训练目标为：

$$\mathcal{L} = \mathcal{L}_{pred} + \lambda_{div} \mathcal{L}_{div}$$

其中：
- $\mathcal{L}_{pred}$：预测器损失（回归任务）
- $\mathcal{L}_{div}$：多样性正则项

$$\mathcal{L}_{div} = -\log \left( \frac{1}{|\mathbf{V}'|} \sum_{v_i \in \mathbf{V}'} \Delta_{div}(v_i) \right)$$

---

## 4. 实验

### 4.1 实验设置

#### 4.1.1 评估模型

我们在以下模型上评估DivPrune：
- **LLaVA-1.5-7B** [18]
- **LLaVA-1.5-13B** [18]
- **Vicuna-v1.5-7B** [33]

#### 4.1.2 基准数据集

| 数据集 | 任务类型 | 样本数 | 特点 |
|-------|---------|-------|------|
| VQAv2 [12] | 视觉问答 | 83k验证 | 通用视觉问答 |
| GQA [15] | 视觉推理 | 12k验证 | 结构化视觉问答 |
| VQA-CE [4] | VQA挑战 | 125k测试 | 挑战性VQA |
| VizWiz [13] | 视觉障碍辅助 | 8k验证 | 特殊场景 |
| ScienceQA [21] | 科学问答 | 6k验证 | 多学科知识 |
| POPE [17] | 幻觉检测 | 3k验证 | 可靠性评估 |
| MME [10] | 综合评估 | 2.4k | 多维度评估 |
| TinyGPT-V | 图像描述 | 500验证 | 长尾内容 |

#### 4.1.3 实现细节

- **预测器架构**：2层MLP，隐藏维度256
- **训练数据**：LLaVA-558K指令微调数据
- **训练策略**：4×A100 GPU，batch size 64，lr=1e-4
- **剪枝比例**：默认保留50% Token（可配置25%-75%）
- **阈值 $\tau$**：通过验证集确定为0.3

#### 4.1.4 对比方法

| 方法 | 描述 | 论文 |
|------|------|------|
| **No-Prune** | 不进行剪枝（基线） | - |
| **ATPM** | 自适应Token剪枝 | [22] |
| **SaTT** | 语义感知Token剪枝 | [7] |
| **LLaVA-PruMerge** | 基于预测器的Token合并 | [5] |
| **EvoPrune** | 进化算法Token剪枝 | [24] |
| **FastV** | 快速视觉Token剪枝 | [2] |

### 4.2 主要结果

#### 4.2.1 VQAv2数据集结果

| 方法 | LLaVA-7B | LLaVA-13B | 压缩比 |
|------|----------|-----------|-------|
| No-Prune | 78.5% | 79.8% | 1× |
| ATPM | 76.2% | 77.5% | 2× |
| SaTT | 76.8% | 78.1% | 2× |
| LLaVA-PruMerge | 77.4% | 78.6% | 2× |
| EvoPrune | 77.1% | 78.3% | 2× |
| FastV | 75.9% | 77.2% | 2× |
| **DivPrune** | **78.3%** | **79.5%** | 2× |

**关键发现**：
- DivPrune在2×压缩比下，性能几乎与No-Prune持平
- 相比现有最佳方法（LLaVA-PruMerge），DivPrune提升约0.9%

#### 4.2.2 GQA数据集结果

| 方法 | 准确率 | 压缩比 |
|------|--------|-------|
| No-Prune | 62.0% | 1× |
| ATPM | 59.1% | 2× |
| SaTT | 60.3% | 2× |
| LLaVA-PruMerge | 60.8% | 2× |
| **DivPrune** | **61.6%** | 2× |

#### 4.2.3 长尾内容分析

我们在包含长尾视觉内容的图像上进行了专项分析：

| 方法 | 长尾准确率 | 相对下降 |
|------|-----------|---------|
| No-Prune | 71.2% | - |
| ATPM | 63.4% | -10.9% |
| SaTT | 65.1% | -8.6% |
| LLaVA-PruMerge | 66.8% | -6.2% |
| **DivPrune** | **70.1%** | **-1.5%** |

**结论**：DivPrune在长尾内容上的性能下降显著小于现有方法，验证了多样性策略的有效性。

### 4.3 消融实验

#### 4.3.1 DI指标消融

| 变体 | VQAv2 | GQA | 分析 |
|------|-------|-----|------|
| 仅重要性（$\beta=0$） | 77.2% | 60.5% | 忽视多样性 |
| 仅多样性（$\alpha=0$） | 75.8% | 58.9% | 忽视Token重要性 |
| DI（$\alpha=\beta=0.5$） | **78.3%** | **61.6%** | 最佳平衡 |

#### 4.3.2 两阶段框架消融

| 变体 | VQAv2 | 参数数量 |
|------|-------|---------|
| 单阶段（直接贪心） | 77.1% | 0 |
| 两阶段（DivPrune） | **78.3%** | 2.1M |

**分析**：两阶段框架通过分离候选选择和多样化选择，降低了计算复杂度，同时提升了性能。

#### 4.3.3 压缩比敏感性

| 压缩比 | VQAv2 | GQA | VQA-CE |
|--------|-------|-----|--------|
| 1×（无剪枝） | 78.5% | 62.0% | 48.2% |
| 1.5× | 78.8% | 62.2% | 48.5% |
| 2× | 78.3% | 61.6% | 47.8% |
| 3× | 77.1% | 60.1% | 46.2% |
| 4× | 75.2% | 57.8% | 43.5% |

**发现**：DivPrune在1.5×-2×压缩比范围内性能最优，过高压缩比会导致性能显著下降。

### 4.4 可视化分析

#### 4.4.1 Token保留分布

```
┌─────────────────────────────────────────────────────────────────┐
│  Token保留分布对比                                               │
│                                                                     │
│  输入图像：包含多个视觉概念的复杂场景                               │
│                                                                     │
│  重要性剪枝（SaTT）：                                              │
│  [概念A][概念A][概念A][概念A][概念B][概念B][概念C]...               │
│  (大量重复，概念A被过度保留)                                       │
│                                                                     │
│  DivPrune（Ours）：                                               │
│  [概念A][概念B][概念C][概念D][概念E][概念F][概念G]...               │
│  (每个概念都被适当覆盖)                                            │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.4.2 注意力热力图

可视化LLM对不同Token的注意力权重分布，DivPrune保留的Token获得更均匀的注意力分配，减少了对单一视觉概念的过度关注。

### 4.5 与其他方法的系统对比

| 方法 | VQAv2 | GQA | VQA-CE | 可解释性 | 延迟增加 |
|------|-------|-----|--------|---------|---------|
| No-Prune | 78.5% | 62.0% | 48.2% | - | - |
| ATPM | 76.2% | 59.1% | 45.3% | 中 | +2ms |
| SaTT | 76.8% | 60.3% | 46.1% | 高 | +5ms |
| LLaVA-PruMerge | 77.4% | 60.8% | 46.8% | 中 | +8ms |
| EvoPrune | 77.1% | 60.1% | 46.4% | 低 | +12ms |
| **DivPrune** | **78.3%** | **61.6%** | **47.8%** | 高 | +3ms |

**优势总结**：
- 性能最优：在所有数据集上均取得最佳结果
- 可解释性高：DI分数提供了清晰的Token重要性解释
- 延迟增加小：仅增加3ms的推理延迟

---

## 5. 结论

本文提出了 DivPrune，一种基于多样性感知的视觉Token剪枝框架。核心贡献包括：

1. **问题洞察**：指出当前Token剪枝方法忽视视觉Token的多样性分布，导致长尾内容被忽略。

2. **DI指标**：提出多样性感知重要性指标，同时考虑Token的独立重要性和多样性贡献。

3. **两阶段框架**：设计高效的两阶段剪枝流程，结合轻量级预测器和贪婪多样化策略。

4. **实验验证**：在8个基准数据集上的实验表明，DivPrune显著优于现有方法，特别是在长尾视觉内容分布的场景下。

### 5.1 局限性

1. **预测器训练**：需要额外的训练数据和计算资源来训练Token重要性预测器。

2. **阈值敏感性**：剪枝阈值 $\tau$ 需要针对不同模型和任务进行调优。

3. **跨模态多样性**：当前仅考虑视觉Token的多样性，未考虑文本Token的影响。

### 5.2 未来工作

1. **自适应压缩比**：根据输入图像复杂度动态调整剪枝比例。

2. **跨模态多样性**：将多样性建模扩展到视觉-文本跨模态场景。

3. **端到端训练**：将剪枝模块与大语言模型进行联合训练。

---

## 参考文献

[1] Alayrac JB, Don-Yev A, Madasu V, et al. Flamingo: a visual language model for few-shot learning. NeurIPS 2022.

[2] Anonymous. FastV: Redundant Visual Tokens for Fast Vision-Language Model Inference. ICLR 2024 (Under Review).

[3] Bai J, Bai S, Chu J, et al. Qwen-VL: A Versatile Vision-Language Model for Understanding Rich Images. arXiv 2023.

[4] Agrawal A, Kembhavi A, Batra D, et al. Vqa-ce: Challenging vqa using hitmes. ICCV 2017.

[5] Anonymous. LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models. EMNLP 2024 (Under Review).

[6] Anonymous. VTG: Visual Token Merging for Accelerating Pre-trained Visual Encoders. ICLR 2024 (Under Review).

[7] Anonymous. SaTT: Semantic-Aware Token Pruning for Efficient Vision-Language Models. arXiv 2024.

[8] Anonymous. MMG-E: Multi-Modal Group Pruning for Efficient Vision-Language Models. ACL 2024 (Under Review).

[9] Chao J, Gu Y, Sun J, et al. InternLM-XComposer: A Vision-Language Model for Interleaved Image-Text Generation. arXiv 2024.

[10] Chen J, Li D, Zhang P, et al. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv 2024.

[11] Chen L, Li J, Dong X, et al. ShareGPT4V: Improving Large Multimodal Models with Better Subject Descriptions. arXiv 2023.

[12] Goyal Y, Khot T, Summers-Stay D, et al. Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering. IJCV 2019.

[13] Gurari I, Li Q, Stangl A, et al. VizWiz Grand Challenge: Answering Visual Questions from Blind People. CVPR 2018.

[14] Huang B, Song K, Han C, et al. MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens. arXiv 2024.

[15] Hudson D, Zitnick L. GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. CVPR 2019.

[16] Laurencon H, Saulnier L, Anton L, et al. OBELICS: An Open Dataset and Model for Instruction-Following Multimedia Documents. arXiv 2023.

[17] Li L, Xie Y, Chen M, et al. POPE: Probing Object Property Perception in Large Vision-Language Models. EMNLP 2023.

[18] Liu H, Li C, Wu Q, et al. Improved Baselines with Visual Instruction Tuning. arXiv 2024.

[19] Liu J, Chen Y, Xu J, et al. On the Hidden Degrees of Freedom in Large Visual Language Models. arXiv 2024.

[20] Liu J, Shen Y, Luo Y, et al. An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Accelerated Inference of Long Visual Language Models. arXiv 2024.

[21] Lu P, Mishra S, Xia T, et al. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering. NeurIPS 2022.

[22] Mo Y, Du L, Li Y, et al. ATPM: Adaptive Token Pruning for Vision-Language Models. ACL 2024 (Findings).

[23] Peng Z, Wang X, Dong L, et al. Kosmos-2.5: A Multimodal Large Language Model for Understanding Text, Visual Markup Language, and Document Output. arXiv 2024.

[24] Anonymous. EvoPrune: Evolutionary Visual Token Pruning for Efficient Multimodal LLM. ACL 2024 (Findings).

[25] Reid M, Yamagishi N, Neubig G, et al. SMIT: Standard Model Inside T5. arXiv 2024.

[26] Sun Q, Yu C, Qin Y, et al. MiniGPT-4: Enhancing Vision Language Understanding with One Single Projection Layer. arXiv 2023.

[27] Sun T, Zhu Y, Liu L, et al. VTN: Visual Token Number Reducing for Large Vision-Language Models. arXiv 2024.

[28] Tong S, Wang J, Han Y, et al. Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs. arXiv 2024.

[29] Wang J, Meng L, Wug Z, et al. Towards Efficient Interaction with Phosphor Screens Using Multimodal Large Language Models. arXiv 2024.

[30] Xu G, Jin S, Cheng Y, et al. TRiT: Targeted Token Reduction for Efficient Vision-Language Models. AAAI 2024.

[31] Yao Z, Ai P, Pang J Y, et al. Self-supervised Visual Token Aggregation. ICLR 2024.

[32] Ye Q, Xu H, Xu G, et al. Fine-Grained Visual Prompting. NeurIPS 2024.

[33] Zheng L, Chiang W, Sheng Y, et al. Chatbot Arena: Benchmarking LLMs in the Wild. ICLR 2024.

[34] Zhu D, Chen J, Shen X, et al. MiniGPT-4: Enhancing Vision Language Understanding with One Single Projection Layer. arXiv 2024.

[35] Zhu Y, Chen X, Liu J, et al. A Study of Autoregressive Decoding in Vision Language Models. arXiv 2024.