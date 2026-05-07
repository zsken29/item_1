# LOP 论文复现核对表

更新时间：2026-04-28

依据文件：`C:\codes\python\items_1\lop_repro\论文\LOP：面向高效按需缩放多模态大语言模型的最优剪枝学习.pdf`

## 论文要求到代码映射

| PDF 位置 | 要求 | 当前实现 |
| --- | --- | --- |
| Sec. 3.1, Eq. 1 | FFN 神经元二值保留向量，按每层剪枝率裁剪中间神经元 | `lop/pruning/ffn.py` 真实裁剪 `gate_proj/up_proj/down_proj` |
| Sec. 4.1, Eq. 2 | 从目标全局剪枝率 `b` 自回归预测逐层剪枝率 | `lop/predictor/model.py` 的 `AutoregressivePruningPredictor` |
| Sec. 4.2, Eq. 3-6 | MCTS：UCB、扰动扩展、预算约束、reward 回传 | `lop/search/mcts.py` 的 `paper_mcts_search` |
| Sec. 4.3, Eq. 7-12 | Transformer 预测器、MSE 损失 | `lop/predictor/model.py` 和 `scripts/train_predictor.py` |
| Sec. 5.1 | 主模型 Qwen2.5-VL-7B | `lop/models/qwen25_vl.py`、`configs/model/models.json` |
| Sec. 5.1 | 基准 MME/MMBench/MMMU/POPE | `configs/data/datasets.json`、`scripts/evaluate_dataset.py` |
| Sec. 5.1 | MME 使用 accuracy+ | `lop/eval/mme.py` |
| Appendix A.1.1 | MLP、Bi-LSTM、Transformer 三种预测器消融 | `MlpPruningPredictor`、`BiLstmPruningPredictor`、`AutoregressivePruningPredictor` |
| Appendix A.1.2 | Transformer: 2 layers, hidden 128, 4 heads, GELU, learnable position | `scripts/train_predictor.py` 默认参数 |
| Appendix A.2, Eq. 24-25 | activation L2 神经元重要性 | `lop/importance/activation.py`、`scripts/compute_importance.py` |
| Appendix A.3, Eq. 26-32 | 300 simulations、200 configs、decaying perturbation | `paper_mcts_search` 支持该流程；正式参数见下方命令 |
| Table 1 | Magnitude/WandA/FLAP/LOP 对照 | paper reference 存在 `configs/paper/table1_reference.json`；Magnitude/WandA/FLAP 结构化重要性入口为 `scripts/compute_baseline_importance.py` |
| Table 1/3 | 论文表格生成 | `scripts/build_reproduction_tables.py` 从本地 metrics JSON 和论文参考值生成 Markdown |

## 论文主流程命令

### Qwen2.5-VL-7B 主线

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.compute_importance --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --annotation data/mmbench/annotations/en/dev-00000-of-00001.jsonl --samples 500 --layers 28 --max-new-tokens 1 --output-dir outputs/importance/Qwen2.5-VL-7B-Instruct/mmbench500

C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.search_pruning --importance-dir outputs/importance/Qwen2.5-VL-7B-Instruct/mmbench500 --target-ratio 0.2 --mode paper --iterations 300 --output-dir outputs/mcts_samples/qwen25vl7b_ratio020

C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.train_predictor --samples outputs/mcts_samples/qwen25vl7b_ratio020/samples.jsonl --architecture transformer --epochs 200 --hidden-size 128 --num-layers 2 --num-heads 4 --learning-rate 0.001 --output-dir outputs/predictor_runs/qwen25vl7b_transformer

C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.predict_pruning --checkpoint outputs/predictor_runs/qwen25vl7b_transformer/checkpoint.pt --target-ratio 0.2 --output outputs/predictor_runs/qwen25vl7b_transformer/prediction_020.json
```

预测结果可直接进入评测：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset mme --importance-dir outputs/importance/Qwen2.5-VL-7B-Instruct/mmbench500 --layer-ratios outputs/predictor_runs/qwen25vl7b_transformer/prediction_020.json --output outputs/reports/qwen25vl7b_lop020/mme.json
```

### 全量基准入口

去掉 `--limit` 即为全量；保留 `--limit` 可做短验证。

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset mme --output outputs/reports/qwen25vl7b_dense/mme.json

C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset mmbench --output outputs/reports/qwen25vl7b_dense/mmbench.json

C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset mmmu --output outputs/reports/qwen25vl7b_dense/mmmu.json

C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset pope --output outputs/reports/qwen25vl7b_dense/pope.json
```

## 本地已跑通的最小审计链路

由于当前机器显存无法承诺完成 Qwen2.5-VL-7B 全量长跑，已用 InternVL2_5-1B 验证同一代码链路：

```text
InternVL2_5-1B -> MME 1 sample -> 2 FFN layers
importance -> paper MCTS -> Transformer predictor -> pruning -> MME accuracy+
```

已验证输出：

- `outputs/importance/InternVL2_5-1B/smoke_2layers`
- `outputs/mcts_samples/paper_smoke_2layers`
- `outputs/predictor_runs/paper_smoke_2layers`
- `outputs/reports/internvl_dataset/mme_limit1.json`

## 仍需长时间运行生成的论文表格

这些不是代码遗漏，而是正式实验输出，需要按论文规模实际长跑：

- Table 1：Qwen2.5-VL-7B 在 20%、30%、50% 剪枝率下的 MME-P、MME-R、MMBench、MMMU、POPE。
- Table 2：与紧凑多模态模型的外部结果对比。
- Table 3：Transformer、Bi-LSTM、MLP 三种预测器的全量 MMBench 对比。

当前项目已经提供生成这些结果所需的代码入口、配置、指标和报告输出目录。

## 已补齐的复现支撑入口

- `scripts/search_pruning_with_eval.py`：使用真实验证集准确率作为 MCTS reward。
- `scripts/compute_baseline_importance.py --method magnitude`：结构化 Magnitude FFN 神经元重要性。
- `scripts/compute_baseline_importance.py --method wanda`：结构化 WandA 风格重要性，使用 activation L2 与权重范数组合。
- `scripts/compute_importance.py --stat variance` + `scripts/compute_baseline_importance.py --method flap`：FLAP WIFV 风格结构化重要性。
- `scripts/build_reproduction_tables.py`：把本地评测 JSON 和论文参考值汇总成复现表格。
