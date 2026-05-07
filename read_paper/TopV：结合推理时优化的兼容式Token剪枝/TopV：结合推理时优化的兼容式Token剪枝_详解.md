# TopV：结合推理时优化的兼容式Token剪枝 — 详解

**TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model**

**作者**: Cheng Yang*, Yang Sui*, Jinqi Xiao, Lingyi Huang, Yu Gong, Chendi Li, Jinghua Yan, Yu Bai, Ponnuswamy Sadayappan, Xia Hu, Bo Yuan
**机构**: Rutgers University, Rice University, The University of Utah, California State University, Fullerton
**会议**: CVPR 2025

---

## ★ Insight（关键洞察）

1. **最优传输框架**：首次将视觉Token剪枝建模为最优传输(Optimal Transport)优化问题，通过Sinkhorn算法求解，而非简单使用注意力分数

2. **三大视觉感知成本因子**：提出特征相似度、相对空间距离、绝对中心距离三个视觉感知成本因子，全面衡量Token重要性

3. **兼容FlashAttention和KV Cache**：无需显式计算注意力分数，完全兼容FlashAttention；一次剪枝，全程生效，有效减少KV Cache大小

4. **推理时优化**：仅需2ms（约占总推理时间的<1%），无需任何训练或微调

5. **性能优异**：在InternVL2上提升1.2%精度，视觉FLOPs减少47%，动态内存减少61%，推理效率提升2.1倍

---

## 一、论文核心故事线（先读这里）

**一句话总结**：提出TopV，将视觉Token剪枝建模为最优传输优化问题，提出视觉感知成本函数，通过Sinkhorn算法求解，同时兼容FlashAttention和KV Cache，实现高效低内存的VLM推理。

**核心创新点**：
- 最优传输建模
- 视觉感知成本函数
- Sinkhorn算法求解
- 兼容FlashAttention和KV Cache
- 无需训练的推理时优化

**技术路径**：Token建模 → 成本函数设计 → Sinkhorn求解 → Token恢复 → 高效推理

---

## 二，研究背景与动机

### 2.1 VLM的Token开销问题

VLM处理大量视觉Token：

| 模型 | 视觉Token数 | 占总输入比例 |
|------|-----------|-------------|
| LLaVA-v1.5 | 576 | 87% |
| InternVL (高分辨率) | 256-1792 | 最高95% |

视觉Token的计算和内存开销成为VLM部署的主要瓶颈。

### 2.2 现有方法的三大缺陷

**缺陷1：贪婪启发式重要性标准**
- FastV、LLaVA-PruMerge依赖注意力分数评估重要性
- VTW不考虑重要性指标直接剪枝
- 无法准确捕捉每个Token对模型的贡献

**缺陷2：不兼容FlashAttention**
- 需显式计算注意力分数
- 抵消了FlashAttention的效率优势

**缺陷3：不兼容KV Cache**
- 在每个解码步骤动态剪枝
- 需存储完整的预填充阶段KV Cache
- 无法获得内存节省

### 2.3 核心动机

设计兼容FlashAttention和KV Cache的Token剪枝方法：
- **优化而非贪婪**：将剪枝建模为优化问题
- **兼容FlashAttention**：避免显式注意力计算
- **兼容KV Cache**：一次剪枝，全程生效
- **无需训练**：纯推理时优化

---

## 三、方法详解（含公式）

### 3.1 Token重要性建模

#### 3.1.1 最优传输问题定义

给定第$L_i$层的输入视觉Token（源Token）和输出Token（目标Token），剪枝目标是保留对后续层影响最大的Token：

**源Token定义**：
$$S = \{s_i \in \mathbb{R}^d | i = 1, 2, ..., N\}$$

**目标Token定义**：
$$T = \{t_j \in \mathbb{R}^d | j = 1, 2, ..., N\}$$

其中$d$是Token维度，$N$是Token数量。

#### 3.1.2 最优传输优化问题

$$\mathbf{P}^* = \arg\min_{\mathbf{P}} \left( \sum_{i=1}^N \sum_{j=1}^N \mathbf{P}_{i,j} \mathbf{C}_v(s_i, t_j) \right)$$

其中：
- $\mathbf{P}$：传输计划（Contribution Matrix）
- $\mathbf{C}_v(\cdot,\cdot)$：视觉感知成本函数
- $P_{i,j}$：源Token $s_i$ 对构建目标Token $t_j$ 的贡献分数

### 3.2 目标Token的位置确定

在Transformer层中，目标Token候选位置：
1. Pre-LN输出
2. Multi-Head Attention输出
3. **Post-LN输出** ← 本文选择
4. MLP输出

**选择Post-LN输出的原因**：
- Pre-LN输出未充分反映Attention模块功能
- Attention和MLP输出经过残差连接，相似性掩盖了源-目标区分
- Post-LN输出既反映Attention模块核心功能，又保持有意义的区分

### 3.3 视觉感知成本函数

成本函数包含三个因子：

#### 3.3.1 特征相似度因子

相似Token间转换成本低：
$$\mathbf{C}_f(s_i, t_j) = \|s_i - t_j\|_F^2$$

低相似度对获得高成本，惩罚不相关匹配。

#### 3.3.2 相对空间距离因子

相邻空间位置的Token相关性更强：
$$\mathbf{C}_s(s_i, t_j) = 1 - \exp\left( -\frac{(x_{s_i} - x_{t_j})^2 + (y_{s_i} - y_{t_j})^2}{2\sigma^2} \right)$$

使用相对高斯距离衡量空间接近度。

#### 3.3.3 绝对中心距离因子

图像中心区域包含最重要的视觉信息：
$$\mathbf{C}_e(s_i) = \sqrt{(x_{s_i} - x_c)^2 + (y_{s_i} - y_c)^2}$$

其中$(x_c, y_c)$是图像中心坐标。

#### 3.3.4 综合成本函数

$$\mathbf{C}_v(s_i, t_j) = \alpha \mathbf{C}_f(s_i, t_j) + \beta \mathbf{C}_s(s_i, t_j) + \gamma \mathbf{C}_e(s_i, t_j)$$

超参数设置：
- LLaVA模型：$\alpha = 1, \beta = 1, \gamma = 0.01$
- InternVL2模型：$\alpha = 1, \beta = 1, \gamma = 0.1$

### 3.4 Sinkhorn算法求解

#### 3.4.1 算法流程

```
Algorithm: Sinkhorn Algorithm for Token Pruning

Inputs: 视觉Token(源Token) S, 目标Token T, 成本矩阵 C_v, 温度参数 ε, 最大迭代 T, 容差 δ
Outputs: Contribution Matrix P, 剪枝后Token

1. p, q ← Norm(S), Norm(T)
2. u, v ← 1[N], 1[N]
3. K ← exp(-C_v/ε)
4. t ← 0
5. repeat
6.     t ← t + 1
7.     v^{t+1} ← q / (K^T exp(u^t/ε))
8.     u^{t+1} ← p / (K exp(v^{t+1}/ε))
9. until ((u^t - u^{t+1} < δ) and (v^t - v^{t+1} < δ)) or t > T
10. P ← u Kv  [Contribution Matrix]
11. I ← Sum(P, dim=1)  [Token Importance]
12. I ← Sort(I)
13. Stop ← TopK(I)  [重要Token]
14. Sp ← S \ Stop  [移除不重要Token]
15. return P, Sp
```

#### 3.4.2 Token重要性计算

行求和得到Token重要性：
$$I_i = \sum_{j=1}^N P_{i,j}, \quad i = 1, 2, ..., N$$

选择重要性分数最高的top-k Token，丢弃其余。

### 3.5 Token恢复策略

为防止视觉Token崩溃，特别是生成Token需关注输入图像不同区域的任务（如OCR），采用均匀采样的Token恢复方法。

```
Token恢复策略：
1. 均匀采样一定间隔的剪枝Token恢复
2. 保持平衡表示
3. 减少视觉崩溃风险

配置：
- LLaVA模型：采样间隔=4，恢复后约35% FLOPs减少
- InternVL2模型：采样间隔=3，恢复后约47% FLOPs减少
```

### 3.6 整体算法流程

```
TopV完整流程：

Prefilling阶段：
1. 选择层L_i的输入视觉Token作为源Token
2. 收集同一层Post-LN后的输出作为目标Token
3. 建立并求解优化问题确定Token重要性
4. 根据重要性剪枝不重要Token
5. 按均匀模式恢复部分剪枝Token

从层L_i+1开始：
- 对应Token持续被剪枝
- 实现更快、低内存的VLM推理

注：L_i = 2（遵循FastV的设置）
```

---

## 四、实验设计与结果分析

### 4.1 实验设置

**模型配置**：
| 模型 | 视觉Token数 | 说明 |
|------|-----------|------|
| LLaVA-v1.5-7B | 576 | 大型语言模型 |
| LLaVA-v1.5-13B | 576 | 大型语言模型 |
| InternVL2-2B | 256-1792 | 动态图像增强 |
| InternVL2-26B | 256-1792 | 动态图像增强 |

**评估任务**：
- 识别任务：AI2D, SQA IMG, MMMU, MMBench
- 图像描述：Nocaps
- 视觉问答：OK-VQA
- OCR任务：OCRBench

### 4.2 LLaVA模型结果

**表1：LLaVA-v1.5-7B 在多任务上的性能**

| 方法 | FLOPs减少 | AI2D | SQA IMG | MMMU | MMBench |
|------|-----------|-------|---------|------|---------|
| 基线 | 0% | 55.18 | 69.51 | 35.1 | 35.90 |
| FastV | 47% | 55.27 | 68.91 | 34.7 | 35.4 |
| **TopV** | **51%** | **55.31** | **69.61** | **35.8** | **35.45** |

**关键发现**：
- TopV在51% FLOPs减少下，性能仍超过基线
- 相比FastV，TopV在更少FLOPs减少下获得更好性能

### 4.3 InternVL2模型结果

**InternVL2-2B结果**

| 方法 | FLOPs减少 | AI2D | 内存减少 | 推理加速 |
|------|-----------|-------|---------|---------|
| 基线 | 0% | 72.67 | 0% | 1.0x |
| FastV | 47% | 71.57 | 47% | 1.2x |
| **TopV** | **48%** | **73.35** | **61%** | **2.1x** |

**InternVL2-26B结果**

| 方法 | FLOPs减少 | AI2D | MMBench |
|------|-----------|-------|---------|
| 基线 | 0% | 83.13 | 80.89 |
| FastV | 46% | 81.27 | 80.86 |
| **TopV** | **47%** | **83.34** | **81.27** |

### 4.4 效率提升详情

**LLaVA-v1.5-7B**：
| 指标 | 基线 | TopV (35%) | 提升 |
|------|-----|------------|------|
| 延迟 | 10'03" | 9'25" | 6.3% |
| 吞吐量 | 5.13 tok/s | 6.07 tok/s | 18.3% |
| 内存 | 14.17 GB | 13.98 GB | 1.3% |

**InternVL2-2B**：
| 指标 | 基线 | TopV (48%) | 提升 |
|------|-----|------------|------|
| 延迟 | 9'10" | 7'50" | 14.5% |
| 吞吐量 | 5.61 tok/s | 6.57 tok/s | 17.1% |
| 内存 | 6.39 GB | 5.57 GB | 12.8% |

---

## 五、消融实验

### 5.1 目标Token位置消融

| 目标位置 | 性能 | 分析 |
|----------|------|------|
| Pre-LN | 较低 | 未反映Attention功能 |
| Attention输出 | 较低 | 残差连接掩盖区分 |
| MLP输出 | 较低 | 偏离核心信息处理 |
| **Post-LN** | **最优** | 最佳功能-区分平衡 |

### 5.2 成本函数因子消融

| 配置 | AI2D | 说明 |
|------|-------|------|
| 仅特征相似度 | 53.8 | 忽视空间信息 |
| 特征+空间距离 | 54.9 | 加入空间感知 |
| **完整成本函数** | **55.3** | 最佳 |

### 5.3 成本权重消融

| γ值 | 说明 | 效果 |
|------|------|------|
| 0 | 无中心距离 | 过度剪枝边缘Token |
| 0.01 (LLaVA) | 适度 | 平衡 |
| 0.1 (InternVL2) | 较高 | 保留更多中心区域 |

### 5.4 Token恢复间隔消融

| 间隔 | 恢复比例 | 性能 |
|------|---------|------|
| 2 | 高 | 可能过于保守 |
| 4 | 中等 | 最佳 |
| 6 | 低 | 可能丢失关键区域 |

---

## 六、与相关工作的关系

### 6.1 与FastV的关系

**FastV**：
- 依赖注意力分数评估重要性
- 每个解码步骤动态剪枝
- 不兼容KV Cache

**TopV**：
- 最优传输优化评估重要性
- Prefill阶段一次剪枝
- 完全兼容KV Cache

### 6.2 与LLaVA-PruMerge的关系

**LLaVA-PruMerge**：
- 训练预测器
- Token合并而非剪枝

**TopV**：
- 无需训练
- 真正的剪枝

### 6.3 与VTW的关系

**VTW**：
- 完全不考虑重要性
- 特定层直接丢弃所有视觉Token
- 仅适用于识别任务

**TopV**：
- 考虑重要性的优化剪枝
- 适用于多种任务

---

## 七、方法优缺点深度评析

### 7.1 优点

1. **无需训练**：纯推理时优化，无额外训练成本
2. **兼容FlashAttention**：无需显式注意力计算
3. **兼容KV Cache**：一次剪枝，全程生效
4. **计算开销极小**：仅2ms（<1%推理时间）
5. **性能优异**：超越基线和现有方法

### 7.2 缺点

1. **Sinkhorn迭代**：虽仅需3次迭代，仍有额外计算
2. **固定L_i层**：遵循FastV设置L_i=2，可能非最优
3. **参数调优**：成本函数权重需针对不同模型调优

### 7.3 潜在改进方向

1. **自适应L_i层**：根据输入动态选择最优剪枝层
2. **任务感知权重**：根据任务类型调整成本函数权重
3. **跨模态扩展**：考虑文本Token的联合剪枝

---

## 八、核心公式与理论推导

### 8.1 最优传输问题

$$\mathbf{P}^* = \arg\min_{\mathbf{P}} \left( \sum_{i,j} \mathbf{P}_{i,j} \mathbf{C}_v(s_i, t_j) \right)$$

### 8.2 视觉感知成本函数

$$\mathbf{C}_v(s_i, t_j) = \alpha \mathbf{C}_f + \beta \mathbf{C}_s + \gamma \mathbf{C}_e$$

### 8.3 Sinkhorn迭代

$$v^{t+1} = \frac{q}{K^T \exp(u^t/\epsilon)}, \quad u^{t+1} = \frac{p}{K \exp(v^{t+1}/\epsilon)}$$

---

## 九、术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| 最优传输 | Optimal Transport | 概率分布间最优映射的数学框架 |
| Sinkhorn算法 | Sinkhorn Algorithm | 求解最优传输的迭代算法 |
| 成本函数 | Cost Function | 衡量Token间转换成本的函数 |
| 贡献矩阵 | Contribution Matrix | 源Token对目标Token贡献的矩阵 |
| 视觉感知 | Visual-Aware | 考虑视觉特性的设计 |
| KV Cache | Key-Value Cache | 存储注意力Key-Value的缓存 |
| Prefilling | Prefilling | 推理的预填充阶段 |
| Token恢复 | Token Recovery | 恢复部分剪枝Token的策略 |