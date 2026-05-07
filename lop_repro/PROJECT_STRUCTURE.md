# 项目结构说明

更新时间：2026-04-28

本项目位于 `C:\codes\python\items_1\lop_repro`，用于按 PDF 复现 LOP。代码已覆盖论文主流程：数据集、模型 FFN 适配、activation L2 重要性、结构化 FFN 剪枝、论文式 MCTS、Transformer/MLP/Bi-LSTM 预测器、MME accuracy+ 和数据集级评测入口。

## 根目录

| 路径 | 作用 |
| --- | --- |
| `AGENTS.md` | Codex 协作与编码规范。 |
| `ENVIRONMENT.md` | `cv_env` 环境、版本和验证命令。 |
| `PAPER_REPRODUCTION_CHECKLIST.md` | PDF 要求与代码/命令逐条映射。 |
| `PAPER_REPRODUCTION_READING.md` | 面向接手者的论文方法、代码结构和运行流程阅读文档。 |
| `PROJECT_STRUCTURE.md` | 当前文件，说明项目结构。 |
| `REPRODUCTION_PLAN.md` | 复现路线和当前状态。 |
| `configs/` | 数据集、模型和实验配置。 |
| `lop/` | 复现核心代码。 |
| `scripts/` | 命令行入口。 |
| `tests/` | 单元测试。 |
| `data/` | 本地数据集，被 Git 忽略。 |
| `models/` | 本地模型，被 Git 忽略。 |
| `outputs/` | 实验输出，被 Git 忽略。 |
| `论文/` | PDF 和解析材料，被 Git 忽略。 |

## `lop/`

| 路径 | 作用 |
| --- | --- |
| `lop/adapters/ffn.py` | 探测 Qwen2.5-VL、InternVL、DeepSeek-VL2 的 FFN 结构；给剪枝和 hook 提供权威路径。 |
| `lop/data/datasets.py` | 读取数据集配置、列出 annotation、迭代样本、抽样和完整性校验。 |
| `lop/data/prompts.py` | MME/MMBench/MMMU/POPE prompt 构造与答案读取。 |
| `lop/models/internvl.py` | InternVL2_5-1B 加载和图像预处理。 |
| `lop/models/qwen25_vl.py` | Qwen2.5-VL 加载和生成。 |
| `lop/models/chat.py` | 统一 `internvl` 与 `qwen25_vl` 的推理入口。 |
| `lop/importance/activation.py` | FFN `down_proj` 输入处 activation RMS 和 variance 统计采集。 |
| `lop/importance/io.py` | 重要性文件保存和加载。 |
| `lop/importance/weights.py` | Magnitude、WandA 和 FLAP WIFV 风格结构化 FFN 神经元重要性。 |
| `lop/pruning/ffn.py` | dense FFN 神经元结构化剪枝。 |
| `lop/eval/metrics.py` | 通用 yes/no、选项、exact match 评分。 |
| `lop/eval/mme.py` | MME 官方 `accuracy + accuracy_plus` 聚合，输出 MME-P/MME-R。 |
| `lop/search/mcts.py` | 论文式连续扰动 MCTS，以及离散 smoke 搜索。 |
| `lop/predictor/model.py` | Transformer、Bi-LSTM、MLP 三类剪枝率预测器。 |
| `lop/report/tables.py` | 复现表格 Markdown 生成。 |

## `scripts/`

| 脚本 | 作用 |
| --- | --- |
| `check_environment.py` | 检查 CUDA、FlashAttention2、Transformers、Safetensors 等。 |
| `inspect_data.py` | 检查数据集行数、图片和字段。 |
| `inspect_model.py` | 检查模型 FFN 层、维度和权重索引。 |
| `smoke_internvl.py` | InternVL 单样本推理和 hook 预览。 |
| `compute_importance.py` | 对 `internvl` 或 `qwen25_vl` 批量保存 activation RMS 或 variance importance。 |
| `search_pruning.py` | 运行论文式 MCTS，生成 `(b, theta)` 样本。 |
| `search_pruning_with_eval.py` | 用真实验证集 accuracy 作为 reward 运行论文式 MCTS。 |
| `compute_baseline_importance.py` | 生成 Magnitude/WandA/FLAP 结构化基线重要性。 |
| `train_predictor.py` | 训练 Transformer/Bi-LSTM/MLP 预测器。 |
| `predict_pruning.py` | 用 checkpoint 预测逐层剪枝率。 |
| `build_reproduction_tables.py` | 从本地 metrics 和论文参考值生成 Table 1 风格 Markdown。 |
| `smoke_prune_internvl.py` | dense/pruned 烟测，支持 `internvl` 和 `qwen25_vl` runtime。 |
| `evaluate_internvl_subset.py` | 旧的小文件评测入口，保留用于快速 smoke。 |
| `evaluate_dataset.py` | 数据集级评测入口，支持全量或 `--limit`。 |

## `configs/`

| 路径 | 作用 |
| --- | --- |
| `configs/data/datasets.json` | MME、MMBench、MMMU、POPE 的本地路径和样本统计。 |
| `configs/model/models.json` | Qwen2.5-VL-7B、InternVL2_5-1B、DeepSeek-VL2-Tiny 的本地路径和角色。 |

## `tests/`

测试覆盖：

- 数据集读取、prompt、FFN 探测。
- activation hook 和重要性维度。
- FFN 剪枝后模型结构仍可前向。
- MME accuracy+、选项准确率、yes/no 准确率。
- 论文式 MCTS 预算约束。
- Transformer/Bi-LSTM/MLP 预测器和预算投影。

运行：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m unittest discover -v
```

## 输出目录约定

| 目录 | 内容 |
| --- | --- |
| `outputs/importance/<model>/<run>` | `layer_*.pt` 和 `summary.json`。 |
| `outputs/mcts_samples/<run>` | `samples.jsonl` 和 `search_summary.json`。 |
| `outputs/predictor_runs/<run>` | `checkpoint.pt`、`metrics.jsonl`、`train_summary.json`。 |
| `outputs/reports/<run>` | dense/pruned 评测结果。 |
| `outputs/smoke_tests` | 单样本烟测结果。 |
