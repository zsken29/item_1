from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from best_baseline_result.train import TARGET_COLUMN, feature_columns

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "final_model.joblib"
OUTPUTS_DIR = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用保留的最佳基线模型预测 CSV 数据。")
    parser.add_argument("--input", required=True, help="需要预测的 CSV 文件路径。")
    parser.add_argument("--output", default="", help="预测结果输出路径；不填写时保存到 predictions 目录。")
    return parser.parse_args()


def write_prediction_report(path: Path, y_true: pd.Series, y_pred) -> None:
    labels = sorted(pd.Series(y_true).unique().tolist())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    lines = [
        "# 预测评估报告",
        "",
        f"- 准确率：`{accuracy_score(y_true, y_pred):.6f}`",
        "",
        "| 类别 | 精确率 | 召回率 | F1 | 样本数 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for label, p_value, r_value, f_value, s_value in zip(labels, precision, recall, f1, support):
        lines.append(f"| {label} | {p_value:.6f} | {r_value:.6f} | {f_value:.6f} | {int(s_value)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件，请先运行训练脚本：{MODEL_PATH}")

    df = pd.read_csv(input_path)
    features = feature_columns(df)
    missing_columns = [col for col in ["Feature_1", "Feature_2", "Feature_3"] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"输入文件缺少特征列：{missing_columns}")

    model = joblib.load(MODEL_PATH)
    pred = model.predict(df[features])

    result = df.copy()
    result["预测标签"] = pred

    if args.output:
        output_path = Path(args.output)
    else:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUTS_DIR / f"{input_path.stem}_gmm_2_full_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"使用模型：{MODEL_PATH}")
    print(f"预测结果已保存：{output_path}")
    if TARGET_COLUMN in result.columns:
        accuracy = accuracy_score(result[TARGET_COLUMN], result["预测标签"])
        report_path = output_path.with_suffix(".report.md")
        write_prediction_report(report_path, result[TARGET_COLUMN], pred)
        print(f"带标签输入的准确率：{accuracy:.6f}")
        print(f"中文分类报告已保存：{report_path}")


if __name__ == "__main__":
    main()
