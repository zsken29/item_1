# ViT (Vision Transformer) 演示

Vision Transformer 的简洁实现，基于 [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)。

## 文件结构

```
vit_demo/
├── vit_model.py   # ViT 核心模型实现
├── demo.py        # 演示脚本
└── README.md      # 本文件
```

## 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Vision Transformer                        │
├─────────────────────────────────────────────────────────────┤
│  Input: [B, 3, 224, 224]  (一张 224x224 RGB 图像)          │
│                                                              │
│  1. Patch Embedding                                         │
│     └── 将图像分成 16x16 的 patches → [B, 196, 768]          │
│                                                              │
│  2. 添加 [CLS] token 和位置编码                              │
│     └── [B, 197, 768]                                        │
│                                                              │
│  3. Transformer Encoder (×12 blocks)                         │
│     ├── Multi-Head Self-Attention                            │
│     └── MLP (GeLU, Dropout)                                  │
│                                                              │
│  4. 取 [CLS] token → 分类头                                  │
│     └── [B, num_classes]                                    │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

| 组件 | 说明 |
|------|------|
| `PatchEmbedding` | 将图像分割成 patches 并线性投影 |
| `MultiHeadAttention` | 多头自注意力机制 |
| `MLP` | FFN (Feed-Forward Network) |
| `TransformerBlock` | Transformer 编码器块 |
| `VisionTransformer` | 完整模型 |

## 模型规模

| 模型 | Embed Dim | Heads | Depth | 参数量 |
|------|-----------|-------|-------|--------|
| ViT-Tiny | 192 | 3 | 12 | ~5.7M |
| ViT-Small | 384 | 6 | 12 | ~22M |
| ViT-Base | 768 | 12 | 12 | ~86M |

## 使用方法

### 运行演示

```bash
cd C:/codes/python/items_1/practice/vit_demo
python demo.py
```

### 使用预定义模型

```python
import torch
from vit_model import vit_tiny, vit_small, vit_base

model = vit_tiny(num_classes=10)
x = torch.randn(1, 3, 224, 224)
output = model(x)  # [1, 10]
```

### 自定义模型

```python
from vit_model import VisionTransformer

model = VisionTransformer(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    num_classes=1000
)
```

## 依赖

```bash
pip install torch matplotlib numpy
```

## 参考

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Vision Transformer (ViT) 详解](https://ASKAI.CN)
