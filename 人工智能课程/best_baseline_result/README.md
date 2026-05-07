# 最佳基线模型

本目录只保留原始非调参实验中的最佳模型：`gmm_2_full`。其他模型尝试、调参历史和临时预测结果已清理。

## 模型信息

- 模型名称：`gmm_2_full`
- 模型类型：生成式分类模型
- 模型说明：每个类别单独训练一个全协方差高斯混合模型，每个类别使用 2 个混合成分。
- 固定验证集准确率：`0.8150`
- 重复交叉验证准确率：`0.8113`

## 目录结构

- `train.py`：重新训练并生成最小必要产物。
- `predict.py`：加载最终模型并预测 CSV 数据。
- `custom_models.py`：模型类定义，保证模型可加载。
- `artifacts/final_model.joblib`：最终模型文件。
- `artifacts/metrics.json`：核心评估指标。
- `artifacts/model_info.json`：模型元信息。
- `artifacts/evaluation_report.md`：中文评估摘要。

## 运行方式

在项目根目录运行训练：

```bash
python best_baseline_result/train.py
```

预测数据：

```bash
python best_baseline_result/predict.py --input train_dataset.csv
```

如果不指定 `--output`，预测结果会写入 `best_baseline_result/outputs/`。该目录属于运行输出，不是必须保留的模型文件。
