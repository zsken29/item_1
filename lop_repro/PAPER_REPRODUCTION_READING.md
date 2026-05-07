# LOP 论文复现阅读文档

更新时间：2026-04-28

本文档用于帮助后续人工或 AI 快速理解本项目如何复现论文 `LOP: Learning Optimal Pruning for Efficient On-Demand MLLMs Scaling`。阅读顺序建议是：先读本文档理解论文和代码主线，再看 `PAPER_REPRODUCTION_CHECKLIST.md` 核对 PDF 细节，最后按 `REPRODUCTION_PLAN.md` 执行实验。

## 1. 论文在解决什么问题

多模态大语言模型通常很大，部署到不同设备或延迟预算下时，需要按不同剪枝率得到不同规模的模型。已有剪枝方法往往要针对每个目标剪枝率重新搜索逐层剪枝策略，这个搜索过程很慢。

LOP 的核心目标是把“每次都重新搜索剪枝策略”变成“学习一个剪枝策略预测器”。具体来说：

1. 离线阶段用 MCTS 搜索一些高质量逐层剪枝率配置。
2. 用这些 `(目标全局剪枝率, 逐层剪枝率)` 样本训练一个预测器。
3. 部署或实验时，输入目标全局剪枝率，预测器一次前向就输出每层剪枝率。
4. 根据每层剪枝率裁剪 FFN 中间神经元，再评测剪枝后模型。

论文主模型是 `Qwen2.5-VL-7B`，核心评测数据集是 `MME`、`MMBench`、`MMMU`、`POPE`。

## 2. 论文方法主线

### 2.1 OSP 问题

论文把结构化剪枝形式化为 Optimal Structural Pruning。对第 `l` 层 FFN，设中间神经元数量为 `d_l`，逐层剪枝率为 `theta_l`。剪枝后保留神经元数约为：

```text
floor((1 - theta_l) * d_l)
```

项目里的对应实现是：

- `lop/adapters/ffn.py`：确定每层 FFN 的 `d_l` 和模块路径。
- `lop/pruning/ffn.py`：根据 `theta_l` 真实裁剪 `gate_proj/up_proj/down_proj`。

这里剪的是 FFN 中间维度，不剪 attention head，也不做权重量化。

### 2.2 神经元重要性

论文 Appendix A.2 使用 activation L2 norm 衡量神经元重要性。直观理解是：校准样本中激活越大的 FFN 中间神经元越重要，剪枝时优先保留。

项目实现：

- `lop/importance/activation.py`
  - hook 位置是每层 FFN 的 `down_proj` 输入。
  - 这个张量最后一维正好对应 FFN intermediate neurons。
  - 对每个神经元累计平方和，再开方得到 RMS 重要性。
- `scripts/compute_importance.py`
  - 批量跑样本，保存 `layer_*.pt`。
  - 默认 `--stat rms` 对应 LOP activation L2/RMS。
  - `--stat variance` 保存输入特征波动，用于 FLAP WIFV 基线。

输出位置：

```text
outputs/importance/<model>/<run>
|-- layer_000.pt
|-- layer_001.pt
|-- ...
`-- summary.json
```

### 2.3 MCTS 搜索

论文用 MCTS 为给定目标剪枝率 `b` 搜索高质量逐层剪枝率 `theta`。搜索包含四步：

1. Selection：用 UCB 选择有潜力的状态。
2. Expansion：对当前剪枝率配置做连续扰动，扰动幅度按深度衰减。
3. Evaluation：剪枝模型并在验证集上计算 reward。
4. Backpropagation：把 reward 回传到搜索树。

项目实现：

- `lop/search/mcts.py`
  - `paper_mcts_search`：论文式连续扰动版本。
  - `mcts_search`：离散候选版本，主要用于 smoke test。
- `scripts/search_pruning.py`
  - 默认 `--mode paper`。
  - 当前可用 importance retention proxy 作为快速 reward。
  - 正式复现时应接入真实验证集 reward。
- `scripts/search_pruning_with_eval.py`
  - 每个候选配置都重新加载模型、应用剪枝，并用验证集 accuracy 作为 reward。
  - 这是论文式 reward 的可执行入口，运行成本明显高于 proxy。

注意：论文写到 MCTS 在 MMBench validation set 上评估 reward。当前项目已提供评测入口，但为避免在本地轻量机器上长跑，默认搜索脚本使用 proxy reward。代码结构上可以替换 objective，不需要重写 MCTS。

### 2.4 LOP 预测器

论文的核心是训练预测器：

```text
输入：目标全局剪枝率 b
输出：每层剪枝率 theta_1 ... theta_L
```

论文主结构是 Transformer，附录还比较了 MLP 和 Bi-LSTM。

项目实现：

- `lop/predictor/model.py`
  - `AutoregressivePruningPredictor`：Transformer 预测器。
  - `MlpPruningPredictor`：MLP 消融。
  - `BiLstmPruningPredictor`：Bi-LSTM 消融。
  - `project_to_budget`：把预测出的逐层剪枝率投影到目标全局预算，避免均值偏离目标。
- `scripts/train_predictor.py`
  - 训练预测器。
- `scripts/predict_pruning.py`
  - 从 checkpoint 推理逐层剪枝率。

论文默认 Transformer 配置：

| 项                   | 值         |
| ------------------- | --------- |
| layers              | 2         |
| hidden size         | 128       |
| heads               | 4         |
| activation          | GELU      |
| sequence length     | 28        |
| positional encoding | learnable |

项目脚本默认与上述配置对齐。

### 2.5 评测

论文 Table 1 主要看：

- MME-P
- MME-R
- MMBench
- MMMU
- POPE
- Avg
- Speedup

项目实现：

- `lop/eval/metrics.py`
  - 通用 yes/no、选项题、exact match。
- `lop/eval/mme.py`
  - MME 官方 `accuracy + accuracy_plus`。
  - perception tasks 聚合为 MME-P。
  - cognition tasks 聚合为 MME-R。
- `scripts/evaluate_dataset.py`
  - 数据集级评测入口。
  - 加 `--limit` 是小跑；不加 `--limit` 是全量。

## 3. 项目代码应该怎么读

建议按这个顺序读：

1. `PAPER_REPRODUCTION_CHECKLIST.md`
   - 看 PDF 每个公式、算法、实验配置对应哪个代码文件。
2. `configs/data/datasets.json`
   - 确认四个数据集本地路径和样本规模。
3. `configs/model/models.json`
   - 确认 Qwen2.5-VL、InternVL、DeepSeek 的角色。
4. `lop/adapters/ffn.py`
   - 理解项目如何定位 FFN 层。
5. `lop/importance/activation.py`
   - 理解重要性如何从 activation 得到。
6. `lop/pruning/ffn.py`
   - 理解结构化剪枝如何改模型权重。
7. `lop/search/mcts.py`
   - 理解论文式 MCTS。
8. `lop/predictor/model.py`
   - 理解 Transformer/MLP/Bi-LSTM 预测器。
9. `lop/eval/`
   - 理解指标如何算。
10. `scripts/`
   - 理解每一步实验如何启动。

## 4. 复现流程如何运行

### 4.1 环境检查

所有命令都必须使用 `cv_env`：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.check_environment --strict
```

### 4.2 检查数据和模型

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.inspect_data --dataset mme
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.inspect_model models/Qwen2.5-VL-7B-Instruct
```

### 4.3 计算重要性

论文主线是 Qwen2.5-VL-7B + MMBench 校准样本：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.compute_importance --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --annotation data/mmbench/annotations/en/dev-00000-of-00001.jsonl --samples 500 --layers 28 --max-new-tokens 1 --output-dir outputs/importance/Qwen2.5-VL-7B-Instruct/mmbench500
```

### 4.4 搜索 MCTS 样本

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.search_pruning --importance-dir outputs/importance/Qwen2.5-VL-7B-Instruct/mmbench500 --target-ratio 0.2 --mode paper --iterations 300 --output-dir outputs/mcts_samples/qwen25vl7b_ratio020
```

如果要生成 Table 1 的 20%、30%、50%，需要分别跑：

```text
--target-ratio 0.2
--target-ratio 0.3
--target-ratio 0.5
```

### 4.5 训练预测器

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.train_predictor --samples outputs/mcts_samples/qwen25vl7b_ratio020/samples.jsonl --architecture transformer --epochs 200 --hidden-size 128 --num-layers 2 --num-heads 4 --learning-rate 0.001 --output-dir outputs/predictor_runs/qwen25vl7b_transformer
```

消融实验用：

```text
--architecture mlp
--architecture bilstm
--architecture transformer
```

### 4.6 预测逐层剪枝率

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.predict_pruning --checkpoint outputs/predictor_runs/qwen25vl7b_transformer/checkpoint.pt --target-ratio 0.2 --output outputs/predictor_runs/qwen25vl7b_transformer/prediction_020.json
```

### 4.7 使用预测剪枝率评测 LOP

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset mme --importance-dir outputs/importance/Qwen2.5-VL-7B-Instruct/mmbench500 --layer-ratios outputs/predictor_runs/qwen25vl7b_transformer/prediction_020.json --output outputs/reports/qwen25vl7b_lop020/mme.json
```

`--layer-ratios` 读取预测器输出的逐层剪枝率，`--importance-dir` 决定每层具体保留哪些神经元。

### 4.8 全量评测

Dense 模型：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset mme --output outputs/reports/qwen25vl7b_dense/mme.json
```

其他数据集：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset mmbench --output outputs/reports/qwen25vl7b_dense/mmbench.json
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset mmmu --output outputs/reports/qwen25vl7b_dense/mmmu.json
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset pope --output outputs/reports/qwen25vl7b_dense/pope.json
```

如果只是确认入口是否正常，添加 `--limit 1`。

### 4.9 基线和表格

Magnitude 结构化重要性：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.compute_baseline_importance --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --method magnitude --layers 28 --output-dir outputs/importance/Qwen2.5-VL-7B-Instruct/magnitude
```

WandA 风格结构化重要性：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.compute_baseline_importance --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --method wanda --activation-importance-dir outputs/importance/Qwen2.5-VL-7B-Instruct/mmbench500 --layers 28 --output-dir outputs/importance/Qwen2.5-VL-7B-Instruct/wanda
```

FLAP WIFV 风格结构化重要性：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.compute_importance --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --annotation data/mmbench/annotations/en/dev-00000-of-00001.jsonl --samples 500 --layers 28 --stat variance --max-new-tokens 1 --output-dir outputs/importance/Qwen2.5-VL-7B-Instruct/flap_variance

C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.compute_baseline_importance --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --method flap --activation-importance-dir outputs/importance/Qwen2.5-VL-7B-Instruct/flap_variance --layers 28 --output-dir outputs/importance/Qwen2.5-VL-7B-Instruct/flap
```

生成复现表格：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.build_reproduction_tables --runs outputs/reports/<run_manifest>.json --paper-reference configs/paper/table1_reference.json --output outputs/reports/table1_reproduction.md
```

## 5. 当前已实际跑通什么

当前机器上已经用 InternVL2\_5-1B 跑通过同一条代码链路：

```text
MME 1 sample
2 FFN layers
importance -> paper MCTS -> Transformer predictor -> pruning -> MME accuracy+
```

代表性输出：

- `outputs/importance/InternVL2_5-1B/smoke_2layers`
- `outputs/mcts_samples/paper_smoke_2layers`
- `outputs/predictor_runs/paper_smoke_2layers`
- `outputs/reports/internvl_dataset/mme_limit1.json`

这证明代码链路可以跑通，但不等于已经得到论文 Table 1/2/3 的最终数值。论文级数值需要在 Qwen2.5-VL-7B 上按完整样本规模长跑。

## 6. 如何理解“完整复现”和“本地验证”的边界

项目现在没有明显代码遗漏：论文提到的核心模块、预测器结构、MCTS 流程、MME accuracy+、四个数据集入口和主模型入口都已经有实现。

但论文表格是实验结果，不是静态代码。要真正复现 Table 1/2/3，还需要：

1. 在 Qwen2.5-VL-7B 上跑完整 importance。
2. 用真实验证集 reward 跑 MCTS。
3. 用 MCTS 样本训练预测器。
4. 对 20%、30%、50% 三个剪枝率分别评测四个基准。
5. 记录 speedup，并与 FLAP/Magnitude/WandA 对照。

当前项目已经补了表格生成和 Magnitude/WandA/FLAP 结构化基线入口。FLAP 入口采用官方默认 WIFV 思路：输入特征波动乘以 `down_proj` 权重平方和。

当前 RTX 4060 Laptop GPU 已经适合做 smoke 和代码验证，但不适合承诺完成 Qwen2.5-VL-7B 的全量长跑。论文原文提到实验使用 RTX 3090 和 A100 40G 级别设备。

## 7. 常见误区

### 7.1 只看 `accuracy` 不够

MME 论文指标不是简单准确率。它对每张图通常有两道 yes/no 问题，`accuracy_plus` 只有两题都答对才给该图加分。本项目在 `lop/eval/mme.py` 里实现了这个逻辑。

### 7.2 剪枝不是 mask，而是改结构

本项目的 `apply_ffn_pruning` 会替换 Linear 层，改变 `gate_proj/up_proj/down_proj` 的形状。这和只乘一个 mask 不同，后续推理确实走更小的 FFN。

### 7.3 MCTS 当前可以跑 proxy reward，但论文正式结果要用验证集 reward

`search_pruning.py` 当前能用 importance retention proxy 快速生成样本，这适合 smoke。论文要求是剪枝后在验证集上评估 reward。正式复现时需要把 objective 接到评测子集。

### 7.4 Qwen2.5-VL 是论文主线，InternVL 是本地烟测

InternVL2\_5-1B 用于证明代码链路可运行。最终论文表格必须以 Qwen2.5-VL-7B 为主。

## 8. 后续接手检查清单

接手者开始新实验前应检查：

1. `git status --short` 是否干净。
2. 是否使用 `C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe`。
3. `scripts.check_environment --strict` 是否通过。
4. `scripts.inspect_model models/Qwen2.5-VL-7B-Instruct` 是否显示 28 层 verified。
5. 数据集 `scripts.inspect_data` 是否通过。
6. 实验输出是否写入 `outputs/`。
7. 结果文档是否记录模型、数据集、剪枝率、seed、样本数、命令和 Git commit。
