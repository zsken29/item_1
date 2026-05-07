# LOP 论文复现计划与状态

更新时间：2026-04-28

目标是严格按 PDF 复现 `LOP: Learning Optimal Pruning for Efficient On-Demand MLLMs Scaling`。当前项目已补齐论文主流程的代码、命令入口和指标聚合；正式论文表格还需要在足够算力上执行全量长跑。

## 已完成

1. 数据集：MME、MMBench、MMMU、POPE 已整理，配置在 `configs/data/datasets.json`。
2. 模型：Qwen2.5-VL-7B、InternVL2_5-1B、DeepSeek-VL2-Tiny 已登记，配置在 `configs/model/models.json`。
3. FFN 适配：Qwen2.5-VL-7B 28 层 dense FFN 权重索引已验证。
4. 神经元重要性：实现 Appendix A.2 的 activation L2/RMS 指标。
5. 结构化剪枝：实现按逐层剪枝率裁剪 FFN 中间神经元。
6. MCTS：实现 Appendix A.3 的 UCB、连续扰动、预算约束、reward 回传流程。
7. 预测器：实现 Appendix A.1 的 Transformer、Bi-LSTM、MLP；Transformer 默认 2 层、hidden 128、4 heads、GELU。
8. 评测：实现 MME 官方 `accuracy + accuracy_plus`，并保留通用选项/yes-no 准确率。
9. Qwen 主线入口：`--runtime qwen25_vl` 已接入 importance、评测、剪枝烟测脚本。
10. 审计文档：`PAPER_REPRODUCTION_CHECKLIST.md` 已逐条映射 PDF 要求。
11. 真实 reward MCTS：`scripts/search_pruning_with_eval.py` 已接入验证集 accuracy。
12. Baseline 支撑：`scripts/compute_baseline_importance.py` 可生成 Magnitude/WandA/FLAP 结构化重要性。
13. 表格生成：`scripts/build_reproduction_tables.py` 可合并本地结果和论文参考值。

## 已本地验证

在当前机器上用 InternVL2_5-1B 跑通最小链路：

```text
MME 1 sample
2 FFN layers
importance -> paper MCTS -> Transformer predictor -> pruning -> MME accuracy+
```

验证命令包括：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.search_pruning --importance-dir outputs/importance/InternVL2_5-1B/smoke_2layers --target-ratio 0.25 --mode paper --iterations 8 --output-dir outputs/mcts_samples/paper_smoke_2layers
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.train_predictor --samples outputs/mcts_samples/paper_smoke_2layers/samples.jsonl --architecture transformer --epochs 5 --hidden-size 16 --num-heads 2 --num-layers 2 --learning-rate 0.01 --output-dir outputs/predictor_runs/paper_smoke_2layers
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --dataset mme --runtime internvl --model-dir models/InternVL2_5-1B --limit 1 --max-new-tokens 4 --output outputs/reports/internvl_dataset/mme_limit1.json
```

## 论文级正式运行

主模型为 Qwen2.5-VL-7B：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.compute_importance --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --annotation data/mmbench/annotations/en/dev-00000-of-00001.jsonl --samples 500 --layers 28 --max-new-tokens 1 --output-dir outputs/importance/Qwen2.5-VL-7B-Instruct/mmbench500
```

MCTS 按论文 Appendix A.3 使用 300 simulations：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.search_pruning --importance-dir outputs/importance/Qwen2.5-VL-7B-Instruct/mmbench500 --target-ratio 0.2 --mode paper --iterations 300 --output-dir outputs/mcts_samples/qwen25vl7b_ratio020
```

预测器按论文 Appendix A.1.2：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.train_predictor --samples outputs/mcts_samples/qwen25vl7b_ratio020/samples.jsonl --architecture transformer --epochs 200 --hidden-size 128 --num-layers 2 --num-heads 4 --learning-rate 0.001 --output-dir outputs/predictor_runs/qwen25vl7b_transformer
```

全量评测入口：

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset mme --output outputs/reports/qwen25vl7b_dense/mme.json
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset mmbench --output outputs/reports/qwen25vl7b_dense/mmbench.json
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset mmmu --output outputs/reports/qwen25vl7b_dense/mmmu.json
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --runtime qwen25_vl --model-dir models/Qwen2.5-VL-7B-Instruct --dataset pope --output outputs/reports/qwen25vl7b_dense/pope.json
```

## 当前限制

当前本地机器是 RTX 4060 Laptop GPU。Qwen2.5-VL-7B 的完整 importance、MCTS reward 评测和四基准全量评测可能需要论文所述 RTX 3090 或 A100 级别显存与运行时间。代码入口已经补齐；若要生成 Table 1/2/3 的最终数值，需要按上方命令在足够算力上长跑。
