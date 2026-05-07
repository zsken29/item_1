# Qwen2.5-VL 架构详解与代码对照

> 本文档配合 `qwen2_5_vl.py` 代码，逐模块讲解 Qwen2.5-VL 的核心设计。

---

## 一、整体架构

Qwen2.5-VL 是一个**视觉编码器 + 视觉语言融合 + 纯文本解码器**的三阶段模型：

```
Image / Video
    │
    ▼
┌─────────────────────────────┐
│  VisionTransformer          │
│  (动态分辨率 ViT)            │
│  Conv3D PatchEmbed           │
│  ×32 VisionBlock             │
│  (Window/Full Attention)     │
│  PatchMerger                 │
└──────────────┬──────────────┘
               │ 视觉 token 序列 [N_merged, llm_hidden_dim]
               ▼
┌─────────────────────────────┐
│  视觉 token 注入             │
│  <|image_pad|> → visual_emb │
│  (占位符替换，非前插)         │
└──────────────┬──────────────┘
               │ 统一序列 [L, llm_hidden_dim]
               ▼
┌─────────────────────────────┐
│  TextDecoder (Qwen2.5 LLM)  │
│  ×28 DecoderLayer           │
│  (Causal + GQA + MRoPE)      │
└──────────────┬──────────────┘
               │ hidden states
               ▼
┌─────────────────────────────┐
│  LM Head                    │
│  (vocab_size 投影)          │
└──────────────┬──────────────┘
               ▼
         logits [L, vocab_size]
```

**关键洞察**：视觉信息通过 PatchMerger 压缩后，用**占位符替换**的方式注入文本序列，而非简单拼接在序列前面。这保证了视觉 token 与文本 prompt 的位置对应关系。

---

## 二、配置参数（ModelConfig）

位置：`qwen2_5_vl.py:41-113`

```python
@dataclass
class ModelConfig:
    # ===== 视觉编码器 =====
    patch_size: int = 14              # 每 patch 14×14 像素
    temporal_patch_size: int = 2       # 视频：每 2 帧一个时间 patch
    spatial_merge_size: int = 2        # 2×2 相邻 patch 合并成 1 个
    vision_hidden_dim: int = 1280      # ViT hidden dimension
    vision_mlp_hidden_dim: int = 3420  # SwiGLU 中间层
    num_vision_blocks: int = 32        # ViT 层数
    vision_num_heads: int = 16         # ViT 注意力头数
    vision_window_size: int = 112      # 窗口注意力窗口大小
    full_attention_block_indices = (7, 15, 23, 31)  # 这 4 层用全局注意力

    # ===== LLM 解码器 =====
    llm_hidden_dim: int = 3584         # Qwen2.5-7B hidden
    llm_mlp_hidden_dim: int = 18944    # SwiGLU 中间层
    num_llm_blocks: int = 28           # LLM 层数
    llm_num_attention_heads: int = 28  # Query 头数
    llm_num_key_value_heads: int = 4  # Key/Value 头数（GQA）
    vocab_size: int = 152064          # 词表大小

    # ===== RoPE / MRoPE =====
    rope_theta: float = 1_000_000.0    # LLM RoPE theta（很大）
    vision_rope_theta: float = 10_000.0 # ViT RoPE theta（较小）
    mrope_section: Tuple[int, int, int] = (16, 24, 24)  # MRoPE 三维分段
```

**为什么 LLM 的 rope_theta 这么大？**
RoPE 的频率公式是 `θ^(2i/d)`，theta 越大，高频分量衰减越慢，能编码更长的依赖关系。Qwen2.5-7B 支持 128k tokens 的上下文，所以用很大的 theta。

---

## 三、视觉编码器（VisionTransformer）

### 3.1 VisionPatchEmbed：图像 → Patch Tokens

位置：`qwen2_5_vl.py:560-664`

```python
class VisionPatchEmbed(nn.Module):
    def __init__(self, config):
        # 视频用 Conv3D
        self.proj_video = nn.Conv3d(3, embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size))
        # 图像用 Conv2D
        self.proj_image = nn.Conv2d(3, embed_dim,
            kernel_size=patch_size, stride=patch_size)
```

**输入**：图像 `[B, 3, H, W]` 或视频 `[B, 3, T, H, W]`
**输出**：
- embeddings：`[B, num_patches, embed_dim]`
- grid_thw：`(T', H', W')` patch 网格尺寸

**例子**：224×224 图像，patch_size=14 → 16×16 = 256 patches

### 3.2 VisionBlock：Window Attention + Full Attention

位置：`qwen2_5_vl.py:813-887`

```python
class VisionBlock(nn.Module):
    def __init__(self, config, block_idx):
        self.use_full_attention = block_idx in config.full_attention_block_indices
        self.attn = MultiHeadAttention(
            hidden_dim=config.vision_hidden_dim,
            num_heads=config.vision_num_heads,
            is_causal=False,  # 关键：视觉注意力是双向的
        )
        self.mlp = SwiGLU(...)
```

**设计洞察**：
- **为什么视觉注意力是双向的？** 视觉编码器需要看到整个图像/视频的全部 patch 来提取特征，与文本生成不同
- **为什么需要 Window Attention？** 降低计算量：O(N²) → O(w²×N)，其中 w 是窗口大小
- **哪 4 层用 Full Attention？** index 7, 15, 23, 31——间隔 8 层，与 ViT-SAM 等设计类似

### 3.3 PatchMerger：空间压缩 + 维度变换

位置：`qwen2_5_vl.py:890-972`

```python
class PatchMerger(nn.Module):
    def __init__(self, config):
        self.ln_q = nn.RMSNorm(config.vision_hidden_dim)  # 先 RMSNorm
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),  # 5120 → 5120
            nn.GELU(),
            nn.Linear(hidden_size, config.llm_hidden_dim, bias=True),  # 5120 → 3584
        )

    def forward(self, x, grid_thw):
        x = self.ln_q(x)                    # 1. RMSNorm
        x = x.view(bsz, grid_t, grid_h, grid_w, dim)  # 2. reshape 成 grid
        # 3. 2×2 合并：相邻 4 个 patch 变成 1 个
        x = x.view(bsz, grid_t, merged_h, s, merged_w, s, dim)
        x = x.permute(...).contiguous()
        x = x.view(bsz, grid_t * merged_h * merged_w, s*s*dim)  # 4. 展平
        return self.mlp(x)                  # 5. MLP 投影
```

**数据流示例**：
```
输入：[B, 256, 1280]     — 16×16 patches，1280 维
↓ ln_q (RMSNorm)
↓ reshape → [B×16×16, 4, 4, 1280]
↓ flatten → [B×256, 5120]
↓ MLP → [B×256, 3584]
输出：[B, 256, 3584]     — 压缩后 token 数不变，但维度变换
```

**关键洞察**：PatchMerger 保持了 token 数量不变（256 → 256），但将每个 token 的维度从 ViT 的 1280 维压缩到 LLM 的 3584 维。真正压缩的是空间维度（H×W），而非 token 数量。

---

## 四、位置编码：RoPE 与 MRoPE

### 4.1 标准 RoPE

位置：`qwen2_5_vl.py:120-145`

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)  # [-x2, x1]
```

**数学原理**：
对于 2D 向量 `[x1, x2]`，旋转矩阵：
```
[cos(θ)  -sin(θ)] × [x1] = [x1*cos - x2*sin]
[sin(θ)   cos(θ)]   [x2]   [x1*sin + x2*cos]
```

rotate_half + cos/sin 乘法等价于这个矩阵乘法，但无需显式构造复数。

### 4.2 MRoPE（多模态 RoPE）

位置：`qwen2_5_vl.py:212-271`

```python
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section):
    doubled_sections = [s * 2 for s in mrope_section]  # [32, 48, 48]
    cos_chunks = cos.split(doubled_sections, dim=-1)
    sin_chunks = sin.split(doubled_sections, dim=-1)

    # 交错取用：chunk[0]→temporal, chunk[1]→height, chunk[2]→width
    cos_selected = torch.cat([chunk[i % 3] for i, chunk in enumerate(cos_chunks)], dim=-1)
```

**设计目的**：
- **视觉 token**：有 2D/3D 空间结构，需要 (temporal, height, width) 三维位置编码
- **文本 token**：三个维度相同 → 退化为标准 1D RoPE

**mrope_section = (16, 24, 24) 的含义**：
假设 head_dim = 128（实际是 3584/28 = 128）：
- 前 16×2 = 32 维 → temporal（时间）
- 中 24×2 = 48 维 → height（高度）
- 后 24×2 = 48 维 → width（宽度）

---

## 五、注意力机制

### 5.1 GQA（Grouped Query Attention）

位置：`qwen2_5_vl.py:148-178`

```python
def repeat_kv(hidden_states, n_rep):
    # 输入:  [B, 4, N, 80]  (4 个 KV heads)
    # 扩展后: [B, 28, N, 80] (复制 7 份)
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)
```

**为什么需要 GQA？**
- 减少 KV 缓存：28 个 Q heads 只需要 4 个 K/V heads
- 减少计算量：K/V 投影计算量从 28 降到 4
- 保持多头结构：Q 仍然有 28 个独立 head

### 5.2 因果 mask vs 双向 mask

位置：`qwen2_5_vl.py:460-497`

```python
def _causal_mask(self, batch_size, seq_len, device, attention_mask):
    # torch.tril 生成下三角矩阵
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    mask = mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)
    return mask
```

**视觉 vs 文本**：
- **视觉注意力**：双向 (`is_causal=False`)——所有 patch 互相可见
- **文本注意力**：因果 (`is_causal=True`)——只能看前面的 token

---

## 六、视觉 Token 注入机制

位置：`qwen2_5_vl.py:1307-1346`

```python
def _inject_visual_embeddings(self, input_ids, inputs_embeds, visual_embeds, is_video):
    token_id = self.config.video_token_id if is_video else self.config.image_token_id
    visual_mask = input_ids.eq(token_id)  # 找到所有 <|image_pad|> 位置

    out = inputs_embeds.clone()
    for b in range(bsz):
        idx = visual_mask[b].nonzero(as_tuple=False).flatten()
        out[b, idx, :] = visual_embeds[b]  # 用视觉 embedding 替换
    return out
```

**vs 简单前插方案**：
```
简单前插：  [vision_start] [visual_tok×N] [vision_end] [text_toks...]
真实注入：  [vision_start] [<|image_pad|>×N] [vision_end] [text_toks...]
                                      ↓ 替换
              [vision_start] [visual_emb×N] [vision_end] [text_toks...]
```

**优势**：
1. 保持序列长度不变
2. 视觉 token 位置与 prompt 中的占位符一一对应
3. 支持多图/多图视频混合输入

---

## 七、完整前向传播流程

位置：`qwen2_5_vl.py:1348-1401`

```python
def forward(self, input_ids, pixel_values=None, attention_mask=None, is_video=False):
    # 1. 文本 embedding
    inputs_embeds = self.language_model.embed_tokens(input_ids)

    # 2. 视觉编码（如果有图像）
    if pixel_values is not None:
        visual_embeds, visual_grid_thw = self.visual(pixel_values)
        # 注入视觉 embedding 到占位符位置
        inputs_embeds = self._inject_visual_embeddings(...)

    # 3. 构建 MRoPE position_ids（3D）
    position_ids = self._build_multimodal_position_ids(...)

    # 4. LLM Decoder 前向
    hidden_states = self.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    # 5. LM Head 投影到词表
    return self.lm_head(hidden_states)
```

---

## 八、数据流总览

以 224×224 图像为例：

```
输入图像 [1, 3, 224, 224]
    │
    ▼
VisionPatchEmbed
patch grid: 224/14 = 16×16 = 256 patches
输出: [1, 256, 1280]
    │
    ▼
×32 VisionBlock
(ViT attention with Window/Full Attention)
    │
    ▼
PatchMerger
2×2 merge: 16×16 → 8×8 = 64 tokens
维度变换: 1280 → 3584
输出: [1, 64, 3584]
    │
    ▼
Inject into text embeddings at <|image_pad|> positions
    │
    ▼
×28 DecoderLayer
(Causal + GQA + MRoPE)
    │
    ▼
LM Head
输出: [1, seq_len, 152064]
```

---

## 九、与原版 Qwen2.5-VL 的差异

> 本代码是**教学目的**的架构模拟器，不是生产级实现。

| 方面 | 本代码 | 官方 Qwen2.5-VL |
|-----|-------|----------------|
| 权重 | 随机初始化 | 预训练权重 |
| 视觉输入 | `[B,3,H,W]` 原始图像 | Processor 预处理后的 patch 单元 |
| Window Attention | 简化的布尔掩码 | 官方的 cu_window_seqlens + window_index 重排 |
| 视频处理 | 简化版 | Dynamic FPS 采样 + 时间编码 |
| 生成 | 简单贪婪，无 KV cache | 生产级生成 + KV cache |

---

## 十、关键设计洞察

### 1. 为什么用 Window Attention？
标准 attention 复杂度 O(N²)，Window Attention 降为 O(w²×N)。对于 16×16=256 patches，窗口大小 112：
- 标准：256² = 65536
- 窗口：112² × 2 ≈ 25000（如果窗口重叠）

### 2. 为什么需要 MRoPE？
文本 token 用标准 1D 位置编码（一维序列）。但视觉 token 有空间结构：
- 2D 图像：需要 (h, w) 两个维度
- 3D 视频：需要 (t, h, w) 三个维度

MRoPE 把 head_dim 分成三段，分别编码三个维度。

### 3. PatchMerger 的双重作用
1. **空间压缩**：2×2 合并减少 4 倍 token 数量
2. **维度变换**：1280 → 3584，适配 LLM hidden dimension

### 4. GQA 为什么有效？
减少 KV 头数而不减少 Q 头数，意味着：
- 显存节省：KV cache 从 28 个头降到 4 个头
- 计算量变化：Q 投影不变，K/V 投影减少 7 倍

---

## 十一、代码索引

| 模块 | 位置 | 作用 |
|-----|------|-----|
| `ModelConfig` | `:41` | 配置参数 |
| `rotate_half` | `:120` | RoPE 核心操作 |
| `repeat_kv` | `:148` | GQA K/V 扩展 |
| `apply_multimodal_rotary_pos_emb` | `:212` | MRoPE 分段编码 |
| `SwiGLU` | `:274` | 门控 MLP |
| `VisionRotaryEmbedding` | `:310` | ViT RoPE |
| `TextRotaryEmbedding` | `:343` | LLM MRoPE |
| `MultiHeadAttention` | `:407` | MHA/GQA 注意力 |
| `VisionPatchEmbed` | `:560` | 图像→patches |
| `VisionBlock` | `:813` | ViT Transformer 块 |
| `PatchMerger` | `:890` | 空间压缩+维度变换 |
| `VisionTransformer` | `:975` | 完整视觉编码器 |
| `DecoderLayer` | `:1044` | LLM Transformer 块 |
| `TextDecoder` | `:1105` | 完整 LLM 解码器 |
| `Qwen25VLArchitectureSimulator` | `:1182` | 完整多模态模型 |
