# VLMCompression 代码库架构分析

> 论文：**Investigating Structural Pruning and Recovery Techniques for Compressing Multimodal Large Language Models: An Empirical Study**
>
> IJCV 2026 (Extended Version) & GCPR 2025 (Oral Presentation)
>
> 仓库：[YiranHuangIrene/VLMCompression](https://github.com/YiranHuangIrene/VLMCompression)
>
> 许可证：Apache-2.0

---

## 一、论文核心思路

对多模态大语言模型（LLaVA-v1.5-7B、Bunny-v1.0-3B、Mini-InternVL-Chat-4B-V1-5）的**语言 backbone** 进行结构化剪枝，再通过低成本的恢复训练重振性能。研究涵盖：

- **Layer-wise pruning**：ShortGPT 风格的 decoder block 级别删除
- **Width-wise pruning**：LLM-Pruner 风格的 attention head 与 MLP channel 剪枝
- **Recovery**：SFT / hidden-state distillation / projector-only 三种恢复策略，可使用低至 5% 的原训练数据
- **Post-training quantization**：剪枝后的 4/8-bit PTQ

---

## 二、整体 Pipeline

```
原始 VLM
    │
    ├──→ [ShortGPT Layer-wise Pruning] ──→ 删除不重要的 decoder blocks
    │                                         输出: pruned_model.bin（完整 VLM）
    │
    ├──→ [LLM-Pruner Width-wise Pruning] ──→ 删除 attention heads / MLP channels
    │                                         输出: pytorch_model.bin（裸 LLM backbone）
    │
    ├──→ [Recovery Fine-tuning] ──→ SFT / Distillation / MM-only
    │                               恢复剪枝造成的性能损失
    │
    └──→ [Post-training Quantization] ──→ 4/8-bit 量化进一步压缩
```

两种剪枝可以**独立使用**，也可以**级联**（先 ShortGPT 删层，再 LLM-Pruner 剪宽度）。`Bunny.py` 和 `llava-vicuna_prune.py` 通过 `--pruned_model_path` 参数支持级联。

---

## 三、仓库顶层结构

```
VLMCompression/
├── LLM-Pruner/                 # Width-wise pruning（fork of horseee/LLM-Pruner）
│   ├── examples/               #   三个 VLM 的剪枝入口
│   │   ├── Bunny.py            #     Bunny / Phi-2 backbone
│   │   ├── llava-vicuna_prune.py #   LLaVA / LLaMA backbone
│   │   └── InternVL.py         #     Mini-InternVL / Phi-3 backbone
│   ├── LLMPruner/
│   │   ├── models/             #   多架构 HuggingFace 模型实现
│   │   │   ├── hf_llama/       #     LLaMA (LLaVA 的 backbone)
│   │   │   ├── hf_phi2/        #     Phi-2 (Bunny 的 backbone)
│   │   │   ├── hf_phi3/        #     Phi-3 (InternVL 的 backbone)
│   │   │   │   └── fusion.py   #     ★ Phi-3 unfuse/refuse 关键机制
│   │   │   ├── hf_bloom/
│   │   │   ├── hf_baichuan/
│   │   │   └── hf_chatglm/
│   │   ├── pruner/             #   各架构的剪枝工具
│   │   │   ├── hf_llama_pruner.py  # LLaMA 剪枝（Magnitude + Taylor）
│   │   │   ├── phi2_pruner.py      # Phi-2 剪枝
│   │   │   ├── phi3_pruner.py      # Phi-3 剪枝（复用 LLaMA 逻辑）
│   │   │   └── llama_pruner.py
│   │   ├── datasets/           #   校准数据集（bookcorpus, alpaca, c4 等）
│   │   ├── evaluator/          #   PPL 评测
│   │   └── peft/               #   内嵌的 PEFT 库
│   ├── hf_prune.py             #   通用 LLaMA 剪枝脚本
│   ├── post_training.py
│   ├── generate.py
│   └── scripts/                #   SLURM 模板脚本
│
├── ShortGPT/                   # Layer-wise pruning（fork of sramshetty/ShortGPT）
│   ├── short_gpt/
│   │   ├── prune.py            #   ★ 主入口：加载 VLM → 算重要性 → 删层 → 保存
│   │   ├── short_vlm.py        #   ★ ShortVLM 类：重要性计算与层删除的核心逻辑
│   │   ├── metrics.py          #   block_influence() 函数
│   │   ├── utils.py            #   数据 Collate（Bunny/LLaVA/InternVL）
│   │   ├── short_llama.py      #   LLaMA 专用辅助
│   │   ├── short_hf.py         #   HuggingFace 通用辅助
│   │   └── layer_removal.py    #   层删除辅助函数
│   ├── run_bunny.sh            #   Bunny grid search 脚本
│   └── run_llava.sh            #   LLaVA grid search 脚本
│
├── VLM/                        # VLM 训练/推理/评测 glue code
│   ├── bunny/                  #   Bunny 模型
│   │   ├── model/builder.py    #   ★ 模型加载器（load_pruned_bunny_model 等）
│   │   ├── model/bunny_arch.py #   Bunny 架构定义
│   │   ├── model/language_model/ # 各 backbone 的 Bunny 封装
│   │   ├── model/multimodal_encoder/  # SigLIP / EVA-CLIP 等视觉编码器
│   │   ├── model/multimodal_projector/
│   │   ├── train/train_pruned.py  # ★ Recovery 训练入口
│   │   └── util/               #   数据处理工具
│   ├── llava/                  #   LLaVA 模型（结构类似 Bunny）
│   │   ├── model/builder.py    #   ★ load_pruned_llava_model
│   │   ├── train/train_pruned.py  # ★ Recovery 训练入口
│   │   └── ...
│   ├── InternVL/               #   InternVL 模型
│   │   ├── internvl_chat/
│   │   │   ├── internvl/model/internvl_chat/
│   │   │   │   ├── modeling_internvl_chat.py
│   │   │   │   └── builder.py  #   ★ InternVL 剪枝模型加载器
│   │   │   └── shell/internvl1.5/slurm/
│   │   │       └── Mini-InternVL-Chat-4B-V1-5_pruned_*  # SLURM 模板
│   │   └── ...
│   └── quantization/
│       └── quantization.ipynb  #   4/8-bit PTQ notebook
│
├── generate_simplified.py      # 剪枝后 LLaMA backbone 推理示例
├── requirements.txt            # 统一依赖
├── LICENSE / NOTICE            # Apache-2.0 + 上游归因
└── README.md
```

---

## 四、三大核心模块详解

### 4.1 ShortGPT — Layer-wise Pruning（层级别剪枝）

#### 原理

通过 **Block Influence (BI)** 度量相邻 decoder block 之间隐藏表示的差异。BI 值越小的层，说明其输出与输入越接近（近似恒等映射），删除后对模型影响越小。

#### 核心类：`ShortVLM`（[short_vlm.py](ShortGPT/short_gpt/short_vlm.py)）

```python
class ShortVLM:
    def __init__(self, model, tokenizer, layers_path, n_prune_layers):
        self.layers = model.model.layers          # 访问 decoder blocks
        self.importances = [0 for _ in self.layers]  # 每层的重要性分数
        self.n_prune_layers = n_prune_layers

    def eval_importance(self, dataloader):
        """在少量校准样本上计算每层的 BI 分数"""
        for batch in dataloader:
            outputs = model(..., output_hidden_states=True)
            for i in range(len(hidden_states) - 1):
                self.importances[i] += block_influence(
                    hidden_states[i], hidden_states[i+1]
                )

    def remove_layers(self):
        """删除 BI 分数最低的 n_prune_layers 个层"""
        layers_to_remove = np.argsort(importances)[:n_prune_layers]
        for idx in sorted(layers_to_remove, reverse=True):
            del self.layers[idx]
```

#### Block Influence 计算（[metrics.py](ShortGPT/short_gpt/metrics.py)）

```python
def block_influence(input_hidden, output_hidden):
    # 计算 1 - cosine_similarity(input_hidden, output_hidden)
    norm_input  = input_hidden.norm(dim=-1, keepdim=True)
    norm_output = output_hidden.norm(dim=-1, keepdim=True)
    sim = (input_hidden @ output_hidden.T) / (norm_input * norm_output)
    return 1 - sim.diagonal()
```

- `BI ≈ 0`：输入输出高度相似 → 该层可被跳过 → **不重要**
- `BI ≈ 1`：输入输出差异大 → 该层在做关键变换 → **重要**

#### 剪枝入口（[prune.py](ShortGPT/short_gpt/prune.py)）

```
python ShortGPT/short_gpt/prune.py \
  --model_name BAAI/Bunny-v1_0-3B \
  --num_examples 50 \
  --n_prune_layers 10 \
  --device cuda:0
```

1. 加载完整 VLM（vision encoder + projector + LLM）
2. 从校准数据集中随机采样 `num_examples` 条多模态样本
3. 前向传播，收集 `hidden_states`
4. 调用 `ShortVLM.eval_importance()` → `ShortVLM.remove_layers()`
5. 保存完整 VLM 为 `pruned_model.bin`

#### 支持模型与数据加载

| 模型 | HF ID | 数据集 |
|------|-------|--------|
| Bunny | `BAAI/Bunny-v1_0-3B` | Bunny-695K |
| LLaVA | `liuhaotian/llava-v1.5-7b` | LLaVA mix-665k |
| InternVL | `OpenGVLab/Mini-InternVL-Chat-4B-V1-5` | InternVL-Chat-V1-2-SFT-Data |

Shell 脚本 `run_bunny.sh` / `run_llava.sh` 做 grid search：`N_SAMPLES ∈ {50, 100}` × `N_PRUNE ∈ {5, 10, 15, 21}`。

---

### 4.2 LLM-Pruner — Width-wise Pruning（宽度级别剪枝）

#### 原理

在 decoder block 内部，删除不重要的 **attention heads**（通过删除 Q/K/V/O projection 的对应 output channels）和 **MLP intermediate channels**（通过删除 gate/up projection 的对应 output channels）。剪枝决策基于权重重要性估计。

#### 重要性估计策略

| 策略 | 类 | 重要性公式 | 说明 |
|------|-----|-----------|------|
| Random | `RandomImportance` | 随机 | Baseline |
| L1-Norm | `MagnitudeImportance(p=1)` | `‖W_idx‖₁` | 权重绝对值之和 |
| L2-Norm | `MagnitudeImportance(p=2)` | `‖W_idx‖₂` | 权重平方和 |
| Taylor-1st | `TaylorImportance` | `|W ⊙ ∇W|` | 一阶 Taylor 展开 |
| Taylor-2nd | `TaylorImportance` | `|W ⊙ ∇²W ⊙ W|` | 二阶 Fisher 近似 |
| Taylor-mix | `TaylorImportance` | 一阶 + 0.5×二阶 | 混合策略 |

Taylor 重要性（[hf_llama_pruner.py](LLM-Pruner/LLMPruner/pruner/hf_llama_pruner.py#L214-L343)）：

```python
class TaylorImportance:
    def __call__(self, group):
        for layer in group:
            # salience = weight * grad  （一阶）
            salience = layer.weight * layer.weight.grad

            if taylor == 'param_second':
                # salience = weight * acc_grad * weight  （二阶）
                salience = layer.weight * layer.weight.acc_grad * layer.weight

            # 对 out_features 维度聚合得到每个 channel 的重要性
            local_norm = salience.abs().sum(dim=1)
```

对于 `param_second` 和 `param_mix`，需要预先对每个样本单独 backward，累积 `acc_grad = Σ(grad² / n_samples)`。

#### MetaPruner 依赖图

使用 `torch_pruning` 库构建计算图依赖：

- **consecutive_groups**：将 Q/K/V/O 四个 projection 的相同 channel 组绑定，确保耦合剪枝（每个 attention head 对应 Q 的 `head_dim` 个 channel，删 head 时四个 projection 同步删除）
- **root_instances**：指定从哪些层的哪些 projection 开始剪枝
- **ch_sparsity**：目标 channel 稀疏率
- **customized_pruners**：注册 RMSNorm 等特殊层的剪枝函数

#### ★ Phi-3 unfuse/refuse 机制（[fusion.py](LLM-Pruner/LLMPruner/models/hf_phi3/fusion.py)）

这是代码库的一个重要设计创新。Phi-3 使用 fused projections：
- `qkv_proj` 单次投影输出 [Q | K | V]
- `gate_up_proj` 单次投影输出 [gate | up]

LLM-Pruner 的依赖追踪器无法直接处理这种 fused layout。解决方案：

```
unfuse_phi3():
    qkv_proj (rows = Q+K+V) → q_proj + k_proj + v_proj
    gate_up_proj (rows = gate+up) → gate_proj + up_proj
    monkey-patch forward() 使用 unfused 路径

[运行标准 LLaMA 剪枝流程 — hf_llama_pruner 可以直接使用]

refuse_phi3():
    q_proj + k_proj + v_proj → qkv_proj（重新拼接权重）
    gate_proj + up_proj → gate_up_proj（重新拼接权重）
    恢复原始 forward() 方法
    更新 num_heads、num_key_value_heads、hidden_size
```

限制：目前仅支持 MHA（`num_key_value_heads == num_attention_heads`），不支持 GQA/MQA。

#### 三个剪枝入口对比

| 入口 | Backbone | 模型类 | Pruner | 特殊功能 |
|------|----------|--------|--------|---------|
| [Bunny.py](LLM-Pruner/examples/Bunny.py) | Phi-2 | `PhiForCausalLM` | `phi2_pruner` | 支持 `--short` 级联 ShortGPT 结果 |
| [llava-vicuna_prune.py](LLM-Pruner/examples/llava-vicuna_prune.py) | LLaMA | `LlamaForCausalLM` | `hf_llama_pruner` | 支持 `--short` 级联 + DataParallel |
| [InternVL.py](LLM-Pruner/examples/InternVL.py) | Phi-3 | `Phi3ForCausalLM` | `phi3_pruner` | unfuse → prune → refuse 流程 |

#### 剪枝流程（以 InternVL.py 为例）

```
1. 加载 Phi3ForCausalLM（从 Mini-InternVL checkpoint 中提取 language_model）
2. unfuse_phi3(model) — 解包 fused projections
3. 构建 MetaPruner：
   - root_instances: layers[4:30] 的 q_proj + gate_proj
   - consecutive_groups: q_proj 的 head_dim 作为分组粒度
   - ch_sparsity: 0.25 (删除 25% channels)
4. Taylor 重要性估计（在 c4/alpaca 校准数据上 backward）
5. pruner.step() — 执行剪枝
6. 更新 num_heads 等推理相关属性
7. refuse_phi3(model) — 重新打包
8. 保存 {'model': phi3, 'tokenizer': tokenizer}
```

---

### 4.3 VLM — Recovery Fine-tuning（恢复训练）

#### 模型加载机制

核心函数 `load_pruned_bunny_model()`（[builder.py](VLM/bunny/model/builder.py)）：

```python
def load_pruned_bunny_model(bunny_model_path, pruned_model_path):
    # 1. 从 HuggingFace 加载完整 VLM
    model = BunnyPhiForCausalLM.from_pretrained(bunny_model_path)

    # 2. 加载剪枝后的 decoder layers
    pruned_model = torch.load(pruned_model_path)
    model.model.layers = pruned_model['model'].model.layers

    # 3. 更新推理属性
    for layer in model.model.layers:
        layer.self_attn.num_heads = (
            layer.self_attn.q_proj.weight.shape[0] // layer.self_attn.head_dim
        )
    return tokenizer, model
```

LLaVA 和 InternVL 的加载逻辑类似，均在构建时从 `pruned_model.bin` 中取出 LLM decoder 层替换原始模型的对应层，同时保留 vision encoder 和 multimodal projector。

#### 三种 Recovery 模式

以 InternVL 的 SLURM 模板为例（[slurm/](VLM/InternVL/internvl_chat/shell/internvl1.5/slurm/)）：

| 模式 | 训练参数 | 数据量 | 说明 |
|------|---------|--------|------|
| `-ft` (Full FT) | 全部 LLM + projector | 5%/10%/15% | 标准 SFT |
| `-dist` (Distill) | 学生 LLM + projector | 5%/10%/15% | hidden-state KL 蒸馏（teacher = 原始大模型） |
| `-mm` (MM-only) | 仅 projector | 5%/10%/15% | 冻结 LLM，只训练 multimodal projector |

Bunny 和 LLaVA 通过 `train_pruned.py` 脚本支持 SFT 和 Distillation 两种模式，使用 `--distill` 参数切换。

#### 训练基础设施

- **分布式训练**：DeepSpeed ZeRO-3
- **微调方式**：支持 Full fine-tuning 和 LoRA（`--lora_enable`）
- **量化训练**：支持 4/8-bit QLoRA（`--bits 4/8`）
- **实验追踪**：W&B (`wandb`)，自动按 `pruned_model_path` 命名 group
- **检查点恢复**：支持 `resume_from_checkpoint=True`

#### 特殊设计：DistillationTrainer

Bunny 和 LLaVA 各自实现了 `DistillationTrainer`（继承自 HF Trainer），在标准 LM loss 之外添加 distillation loss：

```
loss = lm_loss(student) + dist_alpha * KL(student_logits || teacher_logits) / dist_temperature
```

Teacher 模型为未剪枝的原始 VLM，学生模型在训练时同时输出 `hidden_states` 用于计算与 teacher 的 hidden-state 差异。

---

### 4.4 辅助工具

#### generate_simplified.py

用于验证 LLM-Pruner 输出的剪枝 LLaMA backbone 能否正常推理：

```bash
python generate_simplified.py \
  --base_model liuhaotian/llava-v1.5-7b \
  --model_path /path/to/pytorch_model.bin \
  --input_text "Tell me a funny joke"
```

两种加载模式：
- `hf`（默认）：从 HF 加载原始模型，再替换 `model.model.layers` 为剪枝后的层
- `torch`：直接 `torch.load` 整个剪枝 bundle

#### quantization.ipynb

对剪枝后的模型应用 4/8-bit post-training quantization，使用 `bitsandbytes` 库。

---

## 五、完整实验复现流程

### 数据准备

```bash
export BUNNY_DATA_PATH=/path/to/Bunny-v1_0-data/finetune
export LLAVA_DATA_PATH=/path/to/llava
export INTERN_META_PATH=/path/to/InternVL-Chat-V1-2-SFT-Data/meta.json
export REPO_ROOT="$(pwd)"
```

### Step 1: Layer-wise Pruning

```bash
bash ShortGPT/run_bunny.sh   # Bunny grid search
bash ShortGPT/run_llava.sh   # LLaVA grid search

# InternVL 手动调用
python ShortGPT/short_gpt/prune.py \
  --model_name OpenGVLab/Mini-InternVL-Chat-4B-V1-5 \
  --num_examples 50 --n_prune_layers 5 \
  --intern_meta_path $INTERN_META_PATH
```

输出：`./prune_log/<model>_pruned_<N>_<K>_samples/pruned_model.bin` + JSON 日志

### Step 2: Width-wise Pruning

```bash
python LLM-Pruner/examples/Bunny.py --pruning_ratio 0.25 --save_model
python LLM-Pruner/examples/llava-vicuna_prune.py --pruning_ratio 0.25 --save_model
python LLM-Pruner/examples/InternVL.py --pruning_ratio 0.25 --save_model
```

### Step 3: Recovery Fine-tuning

```bash
# Bunny
python VLM/bunny/train/train_pruned.py \
  --pruned_model_path /path/to/pruned_model.bin \
  --model_name_or_path BAAI/Bunny-v1_0-3B

# LLaVA
python VLM/llava/train/train_pruned.py \
  --pruned_model_path /path/to/pruned_model.bin \
  --model_name_or_path liuhaotian/llava-v1.5-7b

# InternVL（SLURM）
sbatch VLM/InternVL/internvl_chat/shell/internvl1.5/slurm/Mini-InternVL-Chat-4B-V1-5_pruned_5_50_samples-ft.sh
```

### Step 4: Quantization（可选）

使用 `VLM/quantization/quantization.ipynb` 进行 4/8-bit PTQ。

---

## 六、关键设计决策与实现亮点

### 1. Phi-3 unfuse/refuse 机制

在 [fusion.py](LLM-Pruner/LLMPruner/models/hf_phi3/fusion.py) 中，通过临时将 Phi-3 的 fused projections (`qkv_proj`, `gate_up_proj`) 解包为 LLaMA 风格的独立 projections (`q/k/v_proj`, `gate/up_proj`)，使得已验证的 `hf_llama_pruner` 可以**不加修改**地用于 Phi-3 架构。剪枝完成后重新打包回 fused 格式。这避免了为 fused layout 单独设计和验证一套剪枝逻辑。

### 2. 级联剪枝支持

`Bunny.py` 和 `llava-vicuna_prune.py` 的 `--pruned_model_path` 参数允许先加载 ShortGPT 删层后的模型，在其基础上继续做宽度剪枝。实现上，ShortGPT 输出的 `pruned_model.bin` 中 `model.model.layers` 已经是删减后的层列表，width pruner 直接将其替换到新的 `PhiForCausalLM` / `LlamaForCausalLM` 中。

### 3. Attention Head 耦合剪枝

通过 MetaPruner 的 `consecutive_groups` 机制确保 Q/K/V/O 四个 projection 的**同一组 output channels** 被一起删除：

```python
"consecutive_groups": {
    layer.self_attn.q_proj: layer.self_attn.head_dim
    for layer in model.model.layers
}
```

这保证每个 attention head（`head_dim` 维）作为整体被保留或删除，避免产出结构非法的模型。

### 4. 二阶 Taylor 重要性估计

对于 `param_second` 和 `param_mix` 模式，需要对每个校准样本单独 backward 并累积 Fisher 信息：

```python
for each sample:
    loss.backward()
    for param in model.parameters():
        param.acc_grad += param.grad ** 2 / n_samples  # 累积 Fisher 对角线
    model.zero_grad()
```

这使得在少量校准样本（10-50 条）下能够获得比一阶 Taylor 更准确的重要性估计。

### 5. 多粒度恢复策略

代码支持的三种恢复模式覆盖了不同的计算预算：
- **Full FT**：最贵，恢复效果最好
- **Distillation**：需要加载 teacher 模型（额外显存），通过 KL 散度对齐输出分布
- **MM-only**：最便宜（冻结 LLM），但恢复效果有限

---

## 七、文件统计

| 模块 | Python 文件数 | Shell 脚本数 | 说明 |
|------|-------------|-------------|------|
| LLM-Pruner | 40+ | 4 | 完整 torch_pruning 框架 + 多架构 pruner |
| ShortGPT | 7 | 2 | 精简的层剪枝模块 |
| VLM/bunny | 20+ | 0 | Bunny 训练/推理/服务 |
| VLM/llava | 15+ | 0 | LLaVA 训练/推理 |
| VLM/InternVL | 80+ | 50+ | InternVL 全家桶（训练/评测/SLURM 模板） |
| VLM/quantization | 1 (notebook) | 0 | PTQ notebook |
| 根目录 | 1 | 0 | generate_simplified.py |

总计约 **170+ Python 文件**，实质为上游开源项目（LLM-Pruner, ShortGPT, LLaVA, Bunny, InternVL）的修改版 fork，许可证均为 Apache-2.0，归因记录在 `NOTICE` 文件中。

---

## 八、代码阅读建议

对于想要深入理解或修改代码的研究者，推荐的阅读顺序：

1. **[ShortGPT/short_gpt/short_vlm.py](ShortGPT/short_gpt/short_vlm.py)** — 层剪枝核心逻辑，~100 行，最简洁的入口
2. **[ShortGPT/short_gpt/metrics.py](ShortGPT/short_gpt/metrics.py)** — Block Influence 计算，仅 27 行
3. **[LLM-Pruner/LLMPruner/pruner/hf_llama_pruner.py](LLM-Pruner/LLMPruner/pruner/hf_llama_pruner.py)** — Taylor/Magnitude 重要性估计的完整实现
4. **[LLM-Pruner/LLMPruner/models/hf_phi3/fusion.py](LLM-Pruner/LLMPruner/models/hf_phi3/fusion.py)** — Phi-3 unfuse/refuse 机制
5. **[LLM-Pruner/examples/InternVL.py](LLM-Pruner/examples/InternVL.py)** — Phi-3 宽度剪枝的完整流程（包含 unfuse/refuse 调用）
6. **[VLM/bunny/model/builder.py](VLM/bunny/model/builder.py)** — 剪枝后模型加载逻辑
7. **[VLM/bunny/train/train_pruned.py](VLM/bunny/train/train_pruned.py)** — Recovery 训练完整流程
