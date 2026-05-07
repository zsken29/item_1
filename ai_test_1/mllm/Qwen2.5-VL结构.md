# Qwen2.5-VL 结构详解

## 一句话概括

Qwen2.5-VL 是一个以 **Qwen2.5 decoder-only LLM** 为核心、用**动态分辨率 ViT**编码图像/视频、再通过 **MLP PatchMerger** 把视觉 token 接入 LLM 的统一多模态生成模型。

```
Image / Video
    │
    ▼
┌─────────────────────────┐
│  Native Dynamic-        │
│  resolution ViT          │
│  (Window Attention)      │
└────────────┬────────────┘
             │ 视觉 patch tokens (变长, 依赖图像尺寸)
             ▼
┌─────────────────────────┐
│  Patch Merger / MLP      │
│  Projector              │
│  (空间聚合 + 维度变换)      │
└────────────┬────────────┘
             │ 压缩后的视觉 tokens
             ▼
┌─────────────────────────┐
│  拼接 text tokens        │
│  <|vision_start|> ...   │
│  <|vision_end|>         │
└────────────┬────────────┘
             │ 统一序列
             ▼
┌─────────────────────────┐
│  Qwen2.5 Decoder-only   │
│  Transformer LLM        │
└────────────┬────────────┘
             │ 自回归生成
             ▼
文本 / 坐标 / JSON / 工具调用
```

---

## 1. 总体架构

Qwen2.5-VL 顶层包含两个核心子模块：

| 子模块 | 类型 | 作用 |
|------|------|------|
| `visual` | `Qwen2_5_VisionTransformerPretrainedModel` | 视觉编码 |
| `language_model` | `Qwen2_5_VLTextModel` | 语言生成 |

这不是 BLIP-2 的 Q-Former 架构，也不是 Flamingo 的 cross-attention 架构，而是**视觉 token 注入 decoder-only LLM** 的方案。

公开规模：**3B / 7B / 32B / 72B**，预训练在 4.1T tokens 上。

---

## 2. 数据流动详解

### 2.1 输入层：动态分辨率

| 输入类型 | 处理方式 |
|--------|---------|
| **图像** | 不同尺寸图片 → 不同长度 visual tokens |
| **视频** | 动态 FPS 采样 + absolute time encoding |
| **文本** | 标准 tokenizer |

关键设计：**不 resize 到固定分辨率**，保留原始空间尺度。检测框、点坐标直接使用真实图像尺寸，而非归一化坐标。

### 2.2 视觉编码器：Native Dynamic-resolution ViT

```
图像 → Patch Embedding (3D Conv)
       kernel=[temporal_patch_size, spatial_patch_size, spatial_patch_size]
       stride=[temporal_patch_size, spatial_patch_size, spatial_patch_size]
       默认 patch_size=14, temporal_patch_size=2, spatial_merge_size=2
         ↓
       变长 patch tokens [B, num_patches, hidden_dim]
         ↓
       Window Attention ViT 层 (大多数)
         ↓
       Full Attention 层 (仅 4 层: index [7, 15, 23, 31])
         ↓
       RMSNorm + SwiGLU (结构对齐 Qwen2.5 LLM)
```

**核心特性：**

- **Window Attention**：大多数层用窗口注意力降低计算量，只有 4 层用全局注意力
- **3D Conv**：kernel/stride 同时覆盖时间维和空间维，实现视频-空间统一建模
- **MRoPE 位置编码**：每个视觉 token 有三维位置 `(temporal, height, width)`

### 2.3 视频建模：动态 FPS + 时间编码

视频处理引入两个新机制：

| 机制 | 作用 |
|-----|------|
| **Dynamic FPS Sampling** | 不同帧率视频自适应采样 |
| **Absolute Time Encoding** | 感知真实时间间隔，支持秒级事件定位 |

### 2.4 位置编码：MRoPE / 多模态 RoPE

```
视觉 token 位置编码 (3D):
  - temporal position id
  - height position id
  - width position id

文本 token 位置编码 (1D):
  - 三个 position index 相同 → 退化为标准 1D RoPE
```

**对剪枝的重要性**：视觉 token 的位置是时空结构而非一维序列。剪枝时需保留/重算 `grid_thw`、position ids、MRoPE 关系。

### 2.5 Vision-Language Merger：MLP 压缩投影

```python
Qwen2_5_VLPatchMerger:
  1. RMSNorm(visual_tokens)
  2. 相邻 patch 按 spatial_merge_size 聚合
  3. 两层 MLP → 投影到 LLM hidden_dim
```

**作用：**
- 视觉 token **空间压缩**（相邻 patch 合并）
- 维度对齐（ViT hidden → LLM hidden）

### 2.6 拼接与生成

```
┌─────────────┐    ┌──────────────┐
│ Visual      │    │ Text         │
│ tokens      │ +  │ tokens       │
└─────────────┘    └──────────────┘
         ↓
  <|vision_start|> ...视觉 tokens ... <|vision_end|>
         ↓
  插入文本序列
         ↓
  Causal Decoder LLM 自回归生成
         ↓
  文本 / 坐标 / JSON / 工具调用
```

---

## 3. 与 LLaVA 架构对比

| 维度 | LLaVA-1.5 / NeXT | Qwen2.5-VL |
|-----|------------------|------------|
| 视觉编码器 | CLIP / SigLIP 等现成 | 从头训练 native ViT |
| 分辨率 | 固定或切图策略 | 动态分辨率，token 数随尺寸变化 |
| 视频 | 多帧实现差异大 | 动态 FPS + absolute time |
| 位置编码 | 视觉位置处理简单 | MRoPE (时间/高/宽分离) |
| 视觉压缩 | MLP projector / resampler | PatchMerger (patch 聚合 + MLP) |
| 注意力 | 依模型而定 | Window Attention（仅 4 层全局注意力） |

---

## 4. 关键模块一览

```
Qwen2.5-VL
├── Qwen2_5_VisionTransformerPretrainedModel (视觉编码器)
│   ├── Qwen2_5_VLVisionPatchEmbed (3D patch embedding)
│   ├── Qwen2_5_VLVisionBlock (Window/Full Attention 交替)
│   │   └── VisionMlpBlock (SwiGLU + RMSNorm)
│   └── Qwen2_5_VLVisionMerger (Patch Merger)
│       ├── RMSNorm
│       ├── patch 空间聚合
│       └── 两层 MLP
│
└── Qwen2_5_VLTextModel (Qwen2.5 decoder-only LLM)
    └── Qwen2.5 标准的 Transformer blocks
```

---

## 5. 各位置剪枝的潜在影响

| 剪枝位置 | 影响 |
|---------|------|
| ViT 层 | 影响视觉特征提取质量 |
| 视觉 token | 空间信息丢失，需重算 MRoPE |
| Patch Merger | 影响压缩率和维度对齐 |
| LLM 参数 | 影响生成能力，与视觉压缩解耦 |

> **核心洞察**：Patch Merger 是一个**很关键的位置**——它同时承担视觉压缩和维度变换，不同剪枝策略对性能和速度的影响差异很大。
