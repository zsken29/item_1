# VLMCompression 复现：需要下载的内容清单

> 本文档列出复现 VLMCompression 实验需要准备的所有模型、数据集和依赖项，以及每项内容的获取方式。

---

## 一、总览

| 类别 | 项目数 | 下载方式 |
|------|--------|---------|
| 基础模型 | 5 个 | HuggingFace 自动下载 |
| 校准数据集 | 6 个 | HuggingFace `datasets` 自动下载 |
| VLM 训练数据集 | 3 个 | **需手动下载** |
| Python 依赖 | ~50 个包 | `pip install -r requirements.txt` |

---

## 二、HuggingFace 基础模型（首次运行时自动下载）

这些模型在代码中通过 `from_pretrained()` 加载，首次运行时会自动从 HuggingFace Hub 下载到 `HF_HOME` 目录。

### 2.1 三个目标 VLM

| 模型 | HF ID | Backbone | 参数量 | 用途 |
|------|-------|----------|--------|------|
| Bunny | `BAAI/Bunny-v1_0-3B` | Phi-2 | ~3B | Layer/Width 剪枝 + Recovery |
| LLaVA | `liuhaotian/llava-v1.5-7b` | Vicuna-LLaMA | ~7B | Layer/Width 剪枝 + Recovery |
| Mini-InternVL | `OpenGVLab/Mini-InternVL-Chat-4B-V1-5` | Phi-3 | ~4B | Layer/Width 剪枝 + Recovery |

### 2.2 视觉编码器（VLM 内部依赖）

LLaVA 的 vision tower 默认配置为 `openai/clip-vit-large-patch14-336`（[train_pruned.py:67](VLM/llava/train/train_pruned.py#L67)），Bunny 和 InternVL 的视觉编码器嵌入在各自 checkpoint 中。

Bunny 支持多种视觉 backbone（SigLIP / EVA-CLIP / CLIP），具体取决于 HF checkpoint 中的 `config.json`，加载模型时会自动解析并下载对应的 vision encoder。

### 2.3 独立 LLM（可选）

纯文本剪枝脚本 `hf_prune.py` 默认使用 `lmsys/vicuna-7b-v1.5`（[hf_prune.py:276](LLM-Pruner/hf_prune.py#L276)），仅在只做纯 LLM 剪枝实验时需要。

### 2.4 预计存储

| 模型 | 大小（fp16） |
|------|-------------|
| Bunny-v1.0-3B | ~7 GB |
| LLaVA-v1.5-7B | ~15 GB |
| Mini-InternVL-Chat-4B-V1-5 | ~9 GB |
| CLIP-ViT-Large | ~1.7 GB |
| **总计** | **~33 GB** |

可通过 `export HF_HOME=/path/to/large_disk/hf_cache` 将缓存指向大容量磁盘。

---

## 三、校准数据集（`datasets` 库自动下载）

这些是 LLM-Pruner 计算 Taylor 重要性时使用的**纯文本**校准数据，通过 HuggingFace `datasets` 库的 `load_dataset()` 自动下载。

### 3.1 文本校准数据

| 数据集 | `datasets` ID | 用途 | 默认样本数 |
|--------|---------------|------|-----------|
| C4 | `allenai/c4` | InternVL 默认校准集 | 10 |
| BookCorpus | `bookcorpus` | 通用校准集 | 10 |
| Alpaca | `tatsu-lab/alpaca` | 指令数据校准 | 10 |
| ScienceQA (text) | `lmms-lab/ScienceQA` | 科学问答文本校准 | 10 |

这些数据集通过 [example_samples.py](LLM-Pruner/LLMPruner/datasets/example_samples.py) 中的 `get_c4()` / `get_bookcorpus()` / `get_alpaca()` / `get_scienceqa()` 函数加载。

### 3.2 PPL 评测数据

| 数据集 | `datasets` ID | 用途 |
|--------|---------------|------|
| WikiText-2 | `wikitext` | 剪枝前后 PPL 评测 |
| Penn Treebank | `ptb_text_only` | 剪枝前后 PPL 评测 |

`PPLMetric` 在剪枝前后自动下载这两个数据集用于困惑度评估。

### 3.3 文本校准数据预计存储

文本校准数据集总计约 **15-20 GB**（主要是 C4 和 BookCorpus），可设置 `HF_DATASETS_CACHE` 环境变量指向大容量磁盘。

---

## 四、VLM 训练数据集（**需要手动下载**）

这些是多模态 VLM 的微调数据集，包含图像和对话标注，**不会自动下载**，需要按照上游仓库的说明手动获取。

### 4.1 Bunny-695K

| 项目 | 说明 |
|------|------|
| 用途 | ShortGPT 层剪枝校准 + LLM-Pruner 宽度剪枝校准 + Recovery SFT |
| 内容 | `bunny_695k.json` + 对应的 `images/` 目录 |
| 环境变量 | `BUNNY_DATA_PATH`，指向包含 `bunny_695k.json` 和 `images/` 的目录 |
| 参考 | [BAAI-DCAI/Bunny](https://github.com/BAAI-DCAI/Bunny) — 查看其数据处理文档 |

```bash
# 期望的目录结构
$BUNNY_DATA_PATH/
├── bunny_695k.json
└── images/
    ├── xxx.jpg
    └── ...
```

### 4.2 LLaVA-v1.5 mix-665k

| 项目 | 说明 |
|------|------|
| 用途 | ShortGPT 层剪枝校准 + LLM-Pruner 宽度剪枝校准 + Recovery SFT |
| 内容 | `llava_v1_5_mix665k.json` + COCO 等图像数据 |
| 环境变量 | `LLAVA_DATA_PATH`，指向包含 `llava_v1_5_mix665k.json` 和所有图片的目录 |
| 参考 | [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) — 查看 Data Preparation 部分 |

```bash
# 期望的目录结构
$LLAVA_DATA_PATH/
├── llava_v1_5_mix665k.json
├── coco/
│   └── train2017/
├── gqa/
│   └── images/
├── ocr_vqa/
│   └── images/
├── textvqa/
│   └── train_images/
└── vg/
    ├── VG_100K/
    └── VG_100K_2/
```

LLaVA 的 LazySupervisedDataset 将 `image_folder` 设置为 `LLAVA_DATA_PATH` 根目录，图片路径记录在 JSON 的 `"image"` 字段中（如 `"coco/train2017/xxx.jpg"`）。

### 4.3 InternVL-Chat-V1-2-SFT-Data

| 项目 | 说明 |
|------|------|
| 用途 | ShortGPT 层剪枝校准 + Recovery SFT |
| 内容 | 多个子数据集的 `.jsonl` 标注 + 对应图片目录 |
| 环境变量 | `INTERN_META_PATH`，指向 `meta.json` |
| 配置文件 | `VLM/InternVL/internvl_chat/shell/data/internvl_1_2_finetune_custom.json` |

**子数据集列表**（来自 [internvl_1_2_finetune_custom.json](VLM/InternVL/internvl_chat/shell/data/internvl_1_2_finetune_custom.json)）：

| 子数据集 | 样本数 | 说明 |
|----------|--------|------|
| sharegpt4v_instruct_gpt4-vision_cap100k | 102,025 | ShareGPT4V 指令数据 |
| sharegpt4v_mix665k | 665,058 | ShareGPT4V 混合数据 |
| dvqa_train_200k | 200,000 | DocVQA 训练集 |
| chartqa_train_18k | 18,317 | ChartQA 训练集 |
| ai2d_train_12k | 12,413 | AI2D 图表理解 |
| docvqa_train_10k | 10,211 | 文档 VQA |
| geoqa+ | 72,318 | 几何问答 |
| synthdog_en | 29,765 | OCR 合成数据 |

**配置步骤**：

1. 从 [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) 下载 InternVL-Chat-V1-2-SFT-Data
2. 编辑 `VLM/InternVL/internvl_chat/shell/data/internvl_1_2_finetune_custom.json`，将 `<INTERN_DATA_ROOT>` 替换为你的本地路径
3. 创建一个 `meta.json` 指向该配置文件，并设置 `INTERN_META_PATH`

### 4.4 VLM 训练数据预计存储

| 数据集 | 预估大小 |
|--------|---------|
| Bunny-695K | ~150 GB（含图片） |
| LLaVA mix-665k | ~200 GB（含图片） |
| InternVL-SFT-Data | ~300 GB（含图片） |
| **总计** | **~650 GB** |

> **提示**：如果只想做**轻量实验**（如只跑剪枝，不做完整 recovery），可以仅准备少量数据（5%-10% 子集），甚至在剪枝校准时使用纯文本数据集（`--dataset c4` 或 `--dataset bookcorpus`）而无需下载图像数据。

---

## 五、软件环境

### 5.1 Conda 环境

```bash
conda create -n vlmc python=3.10 -y
conda activate vlmc
pip install -r requirements.txt
```

### 5.2 核心依赖版本

| 包 | 版本 | 说明 |
|----|------|------|
| torch | >=2.0 | 深度学习框架 |
| transformers | 4.37.2 | HuggingFace 模型库（**锁定版本**） |
| accelerate | 0.28.0 | 分布式训练 |
| deepspeed | 0.14.4 | ZeRO 优化 |
| bitsandbytes | 0.41.0 | 量化训练/PTQ |
| peft | >=0.4.0 | LoRA 微调 |
| timm | 0.9.12 | 视觉模型（InternVL 依赖） |
| wandb | latest | 实验追踪（可选） |

### 5.3 Flash-Attention（可选，加速 LLaVA 训练）

```bash
pip install flash-attn --no-build-isolation
```

LLaVA 的 [train_pruned.py](VLM/llava/train/train_pruned.py) 接受 `attn_implementation="flash_attention_2"` 参数。

### 5.4 SLURM 环境（仅 InternVL recovery 需要）

InternVL 的 recovery 训练使用 SLURM 模板。如果在非 SLURM 集群上运行，可以直接调用底层的 `internvl_chat_finetune.py`：

```bash
python VLM/InternVL/internvl_chat/internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "OpenGVLab/Mini-InternVL-Chat-4B-V1-5" \
  --pruned_model_path /path/to/pruned_model.bin \
  --meta_path ./shell/data/internvl_1_2_finetune_custom.json \
  ...
```

---

## 六、按实验场景的下载需求

### 场景 A：仅跑 Layer-wise Pruning（ShortGPT）

```
必须：
  ✓ 基础模型（HF 自动下载）
  ✓ VLM 训练数据集 1 个（根据模型选 Bunny/LLaVA/InternVL）

可选：
  ○ PPL 评测数据（HF 自动下载，跳过不影响剪枝）
```

```bash
# 仅需要模型 + 数据路径
export BUNNY_DATA_PATH=/path/to/bunny_data  # Bunny 选这个
export LLAVA_DATA_PATH=/path/to/llava_data   # LLaVA 选这个
export INTERN_META_PATH=/path/to/meta.json   # InternVL 选这个

python ShortGPT/short_gpt/prune.py \
  --model_name BAAI/Bunny-v1_0-3B \
  --num_examples 50 --n_prune_layers 10
```

### 场景 B：仅跑 Width-wise Pruning（LLM-Pruner）

```
必须：
  ✓ 基础模型（HF 自动下载）
  ✓ 校准数据集 1 个（HF 自动下载，如 C4）

可选（如要用多模态校准）：
  ○ VLM 训练数据集（需手动下载）

可选（如要级联 ShortGPT 结果）：
  ○ 先跑 ShortGPT 生成 pruned_model.bin
```

```bash
# 纯文本校准（无需下载 VLM 数据集）
python LLM-Pruner/examples/Bunny.py \
  --pruning_ratio 0.25 --save_model \
  --dataset c4

# 多模态校准（需要 VLM 数据集）
python LLM-Pruner/examples/Bunny.py \
  --pruning_ratio 0.25 --save_model \
  --dataset bunny  # 需要 BUNNY_DATA_PATH
```

### 场景 C：完整复现（剪枝 + Recovery）

```
必须：
  ✓ 基础模型 × 3（HF 自动下载）
  ✓ VLM 训练数据集 × 3（手动下载）
  ✓ 校准数据集 × 2（HF 自动下载）
  ✓ pip install -r requirements.txt
  ✓ 8×GPU（推荐 A100 40G+）

可选：
  ○ Flash-Attention
  ○ W&B API key
```

---

## 七、环境变量汇总

```bash
# === HuggingFace 缓存（可选，默认 ~/.cache/huggingface）===
export HF_HOME=/path/to/large_disk/hf_cache

# === 数据集路径（按需设置）===
export BUNNY_DATA_PATH=/path/to/Bunny-v1_0-data/finetune
export LLAVA_DATA_PATH=/path/to/llava
export INTERN_META_PATH=/path/to/InternVL-Chat-V1-2-SFT-Data/meta.json

# === 仓库根目录（SLURM 模板需要）===
export REPO_ROOT="$(pwd)"

# === 实验追踪（可选）===
export WANDB_API_KEY=<your_key>
```

---

## 八、快速验证环境是否就绪

```bash
# 1. 检查 Python 依赖
python -c "import torch, transformers, accelerate, deepspeed; print('OK')"

# 2. 检查模型能否加载（会触发下载）
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('BAAI/Bunny-v1_0-3B', use_fast=False)
print('Model tokenizer loaded OK')
"

# 3. 检查校准数据集能否加载
python -c "
from datasets import load_dataset
ds = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
print(f'C4 loaded: {len(ds)} samples')
"

# 4. 检查 VLM 数据路径
ls "$BUNNY_DATA_PATH/bunny_695k.json" && echo "Bunny data OK" || echo "Bunny data MISSING"
ls "$LLAVA_DATA_PATH/llava_v1_5_mix665k.json" && echo "LLaVA data OK" || echo "LLaVA data MISSING"
ls "$INTERN_META_PATH" && echo "InternVL meta OK" || echo "InternVL meta MISSING"
```
