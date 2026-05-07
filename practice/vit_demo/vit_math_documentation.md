# ViT (Vision Transformer) 完整计算流程详解

> 本文档通过一个极简化的 ViT 配置，**手算每一步**的张量变化
> 核心数学：矩阵乘法、Softmax、LayerNorm

---

## 1. 配置定义

为方便手算，使用**极简 ViT 配置**：

```python
img_size = 8          # 原版 224，这里用 8x8 小图
patch_size = 4        # 原版 16，这里用 4x4 patch
in_channels = 1       # 原版 3（RGB），这里用 1（灰度图）
embed_dim = 4         # 原版 192，这里用 4（embedding 维度）
depth = 1             # 原版 12，这里用 1（只用一个 Transformer Block）
num_heads = 2         # 原版 3，这里用 2（多头注意力）
num_classes = 3       # 输出类别数
mlp_ratio = 2         # 原版 4，这里用 2
```

**手算输入**：单张 8×8 灰度图 → `x = torch.randn(1, 1, 8, 8)`

---

## 2. Patch Embedding（图像分块与投影）

### 2.1 原理

```
8×8 图片分割成 (8/4)×(8/4) = 2×2 = 4 个 patches
每个 patch 大小: 4×4 像素
每个 patch 展平: 4×4 = 16 个像素 → 线性投影到 embed_dim=4
```

### 2.2 卷积层参数

```python
Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=4)
```

| 权重矩阵 | 形状 | 说明 |
|---------|------|------|
| weight | `[4, 1, 4, 4]` | 4 个输出通道，每个 kernel 大小 4×4 |
| bias | `[4]` | 4 个偏置 |

### 2.3 手算过程

输入 `x`: `[1, 1, 8, 8]` = 单通道 8×8 矩阵

```
卷积计算（stride=4，无 padding）：
output[y, x] = Σ(i,j) input[i*stride_y + y, j*stride_x + x] × weight[..., i, j]

由于 stride=4，输入被划分为 4 个不重叠的 4×4 区域：

Patch 0 (左上)      Patch 1 (右上)
┌─────────────┐    ┌─────────────┐
│ a b c d      │    │ e f g h      │
│ e f g h      │    │ i j k l      │
│ i j k l      │    │ m n o p      │
│ m n o p      │    │ q r s t      │
└─────────────┘    └─────────────┘

Patch 2 (左下)      Patch 3 (右下)
┌─────────────┐    ┌─────────────┐
│ u v w x      │    │ y z 0 1      │
│ 2 3 4 5      │    │ 2 3 4 5      │
│ 6 7 8 9      │    │ 6 7 8 9      │
│ 0 1 2 3      │    │ 0 1 2 3      │
└─────────────┘    └─────────────┘
```

卷积输出形状：`[1, 4, 2, 2]` = `[B, embed_dim, n_patches_h, n_patches_w]`

### 2.4 flatten + transpose

```python
x.flatten(2)   # [1, 4, 2, 2] → [1, 4, 4]  (将 H,W 维度展平)
x.transpose(1, 2)  # [1, 4, 4] → [1, 4, 4] 交换 → [1, 4, 4] → [1, 4, 4]
# 最终: [B, n_patches, embed_dim] = [1, 4, 4]
```

### 2.5 输出

```
Patch Embedding 输出: [1, 4, 4]
┌──────────────────────────────────────────────────┐
│ Patch 0 embedding  │  Patch 1 embedding          │
│   [e0, e1, e2, e3] │   [e4, e5, e6, e7]         │
├──────────────────────────────────────────────────┤
│ Patch 2 embedding  │  Patch 3 embedding          │
│   [e8, e9, e10,e11]│   [e12,e13,e14,e15]        │
└──────────────────────────────────────────────────┘
```

---

## 3. 添加 [CLS] Token

### 3.1 原理

可学习的 `[CLS]`（classify）token 拼接在所有 patch tokens 前面，
用于最终分类。类似于 BERT 的 `[CLS]` token。

### 3.2 参数

```python
cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 形状: [1, 1, 4]
```

### 3.3 手算过程

```python
cls_tokens = cls_token.expand(B, -1, -1)  # [1, 1, 4]
x = torch.cat([cls_tokens, x], dim=1)      # 拼接在维度 1
```

```
之前: x = [1, 4, 4]      (B=1, 4 patches, embed_dim=4)
之后: x = [1, 5, 4]      (B=1, 4 patches + 1 CLS = 5 tokens)

┌─────────┬──────────────────────────────────────────┐
│  CLS    │  Patch 0~3 embedding                    │
│ [c0~c3] │  [e0~e3] [e4~e7] [e8~e11] [e12~e15]    │
└─────────┴──────────────────────────────────────────┘
```

---

## 4. 添加位置编码

### 4.1 参数

```python
pos_embed = nn.Parameter(torch.zeros(1, 5, 4))  # [1, n_patches+1, embed_dim]
```

### 4.2 手算过程

```python
x = x + pos_embed  # 广播相加: [1,5,4] + [1,5,4] → [1,5,4]
```

每个 token 加上其位置编码（位置 0 = CLS，位置 1~4 = 4 个 patches）

---

## 5. Multi-Head Attention（多头自注意力）

### 5.1 配置

```python
embed_dim = 4, num_heads = 2 → head_dim = 4/2 = 2
```

### 5.2 QKV 投影

```python
self.qkv = nn.Linear(4, 4*3)  # 输入 4，输出 12
```

输入 `x`: `[1, 5, 4]` → 经过 QKV 线性层 → `[1, 5, 12]`

```
QKV 矩阵乘法：
┌─────────────────────────────────────┐
│  W_qkv: [4, 12]                    │
│  input: [1, 5, 4]                  │
│  output:[1, 5, 12] = [1, 5, 3×4]   │
└─────────────────────────────────────┘

拆分后:
Q = x @ W_q  → [1, 5, 4]
K = x @ W_k  → [1, 5, 4]
V = x @ W_v  → [1, 5, 4]
```

### 5.3 Reshape for Multi-Head

```python
# 原始代码
qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
# 形状变化:
# [1, 5, 12] → [1, 5, 3, 2, 2] → permute → [3, 1, 2, 5, 2]

q = [1, 2, 5, 2]  # Batch=1, Heads=2, Seq=5, HeadDim=2
k = [1, 2, 5, 2]
v = [1, 2, 5, 2]
```

### 5.4 注意力分数计算

```python
attn = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
```

```
q @ k.transpose(-2, -1) 计算：
- q: [1, 2, 5, 2]
- k.transpose(-2, -1): [1, 2, 2, 5]
- result: [1, 2, 5, 5]  (每个 head 的 5×5 注意力矩阵)

除以 √2 = 1.414 进行缩放（防止点积过大）

以 Head 0 为例（5×5 注意力矩阵）：
        Patch0  Patch1  Patch2  Patch3  CLS
Patch0 [ 0.31    0.12    0.08    0.15    0.34 ]
Patch1 [ 0.22    0.45    0.09    0.11    0.13 ]
Patch2 [ 0.15    0.11    0.38    0.21    0.15 ]
Patch3 [ 0.18    0.09    0.24    0.32    0.17 ]
CLS   [ 0.14    0.23    0.21    0.19    0.23 ]
```

### 5.5 Softmax

```python
attn = attn.softmax(dim=-1)
```

每行做 Softmax（和为 1）：
```
Row 0 (Patch0 的注意力分布): [0.21, 0.17, 0.16, 0.19, 0.27]（归一化后）
```

### 5.6 加权求和

```python
x = attn @ v  # [1, 2, 5, 5] @ [1, 2, 5, 2] → [1, 2, 5, 2]
```

```
以 Head 0 的 Patch0 为例：
output[0,0,0,:] = Σ attn[0,0,0,i] × v[0,0,i,:]
                 = 0.21×v[0,0,0,:] + 0.17×v[0,0,1,:] + ...
```

### 5.7 输出投影

```python
x = self.proj(x)  # nn.Linear(4, 4)
```

将所有 heads 拼接后通过线性层还原维度。

### 5.8 最终形状

```
MultiHeadAttention 输出: [1, 5, 4]
┌──────────────────────────────────────────────────┐
│ [CLS 注意力加权后的表示]                          │
│ [Patch0 注意力加权后的表示]                     │
│ [Patch1 注意力加权后的表示]                     │
│ [Patch2 注意力加权后的表示]                     │
│ [Patch3 注意力加权后的表示]                     │
└──────────────────────────────────────────────────┘
```

---

## 6. Transformer Block（残差连接）

### 6.1 整体结构

```python
x = x + attn(self.norm1(x))   # 注意力残差
x = x + mlp(self.norm2(x))    # MLP 残差
```

### 6.2 LayerNorm 手算

```python
self.norm1 = nn.LayerNorm(4)  # 对最后一维做归一化
```

对每个 token（4 维向量）计算：
```
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β

其中:
μ = mean(x)           = (x0+x1+x2+x3)/4
σ² = var(x)           = ((x0-μ)² + ... + (x3-μ)²)/4
γ, β = 可学习参数，初始化为 γ=1, β=0
ε = 1e-5（数值稳定）

例：假设 x = [2.0, 4.0, 6.0, 8.0]
μ = (2+4+6+8)/4 = 5.0
σ² = ((2-5)²+(4-5)²+(6-5)²+(8-5)²)/4 = 20/4 = 5
x_norm = (x-5)/√(5+1e-5) = [-1.34, -0.45, 0.45, 1.34]
```

### 6.3 MLP 手算

```python
MLP(embed_dim=4, hidden_dim=8)  # 4→8→4
```

```
输入 [1, 5, 4] → fc1 → [1, 5, 8] → GELU → fc2 → [1, 5, 4]

GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
GELU 近似: GELU(x) ≈ x × σ(x)  (x 与 sigmoid 的乘积)
```

### 6.4 残差连接

```python
x = x + sublayer(x)  # 每一层都有残差连接
```

作用：缓解梯度消失，让深层网络更容易训练。

---

## 7. 最终分类

### 7.1 完整流程回顾

```
输入图片 [1, 1, 8, 8]
         ↓
    Patch Embedding
         ↓
    [1, 4, 4] (4 patches, embed_dim=4)
         ↓
    + [CLS] token
         ↓
    [1, 5, 4]
         ↓
    + 位置编码
         ↓
    [1, 5, 4]
         ↓
    Transformer Block × 1
         ↓
    [1, 5, 4]
         ↓
    LayerNorm
         ↓
    [1, 5, 4]
         ↓
    取 [CLS] token (index 0)
         ↓
    [1, 4]
         ↓
    分类头 Linear(4, 3)
         ↓
    [1, 3]  ← 最终分类 logits
```

### 7.2 分类头

```python
self.head = nn.Linear(embed_dim, num_classes)  # 4 → 3
```

```
[1, 4] @ [4, 3] + bias[3] → [1, 3]

logits = [score_class0, score_class1, score_class2]
```

---

## 8. 完整配置对比

| 参数 | 原版 vit_tiny | 手算示例 |
|------|--------------|---------|
| img_size | 224 | 8 |
| patch_size | 16 | 4 |
| n_patches | 196 | 4 |
| embed_dim | 192 | 4 |
| depth | 12 | 1 |
| num_heads | 3 | 2 |
| head_dim | 64 | 2 |
| mlp_ratio | 4 | 2 |
| mlp_hidden_dim | 768 | 8 |
| 参数量 | ~5.7M | ~数百 |

---

## 9. 代码中的维度追踪

```python
# 输入
x = torch.randn(1, 1, 8, 8)                    # [B, C, H, W]

# Patch Embedding
x = self.proj(x)                              # [B, embed_dim, H/p, W/p]
x = x.flatten(2)                              # [B, embed_dim, n_patches]
x = x.transpose(1, 2)                         # [B, n_patches, embed_dim]

# 添加 CLS token
cls_tokens = self.cls_token.expand(B, -1, -1) # [1, 1, embed_dim]
x = torch.cat([cls_tokens, x], dim=1)          # [B, n_patches+1, embed_dim]

# 添加位置编码
x = x + self.pos_embed                         # [B, n_patches+1, embed_dim]

# Transformer Block（重复 depth 次）
for block in self.blocks:
    x = block(x)                              # [B, n_patches+1, embed_dim]

# 分类
cls_token_final = x[:, 0]                     # [B, embed_dim]
logits = self.head(cls_token_final)            # [B, num_classes]
```

---

## 10. 关键公式汇总

| 操作 | 公式 |
|------|------|
| Patch Embedding | `x = Conv2d(x)`, 然后 `flatten + transpose` |
| QKV 投影 | `qkv = W_qkv × x` |
| 注意力分数 | `attn = (q × k^T) / √d_k` |
| Softmax | `attn = softmax(attn, dim=-1)` |
| 输出 | `out = attn × v` |
| LayerNorm | `y = γ × (x-μ)/√(σ²+ε) + β` |
| GELU | `y = 0.5 × x × (1 + tanh(...))` |
| 残差连接 | `y = x + sublayer(x)` |
