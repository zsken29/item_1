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

from advanced_model_result.train import TARGET_COLUMN, feature_columns

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "final_model.joblib"
OUTPUTS_DIR = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict labels with the advanced model.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", default="", help="Output CSV path. Defaults to advanced_model_result/outputs.")
    return parser.parse_args()


def write_prediction_report(path: Path, y_true: pd.Series, y_pred) -> None:
    labels = sorted(pd.Series(y_true).unique().tolist())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    lines = [
        "# Prediction evaluation report",
        "",
        f"- Accuracy: `{accuracy_score(y_true, y_pred):.6f}`",
        "",
        "| class | precision | recall | F1 | support |",
        "|---:|---:|---:|---:|---:|",
    ]
    for label, p_value, r_value, f_value, s_value in zip(labels, precision, recall, f1, support):
        lines.append(f"| {label} | {p_value:.6f} | {r_value:.6f} | {f_value:.6f} | {int(s_value)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found. Run training first: {MODEL_PATH}")

    df = pd.read_csv(input_path)
    features = feature_columns(df)
    model = joblib.load(MODEL_PATH)
    pred = model.predict(df[features])

    result = df.copy()
    result["Predicted_Label"] = pred

    if args.output:
        output_path = Path(args.output)
    else:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUTS_DIR / f"{input_path.stem}_advanced_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Model: {MODEL_PATH}")
    print(f"Predictions saved: {output_path}")
    if TARGET_COLUMN in result.columns:
        accuracy = accuracy_score(result[TARGET_COLUMN], result["Predicted_Label"])
        report_path = output_path.with_suffix(".report.md")
        write_prediction_report(report_path, result[TARGET_COLUMN], pred)
        print(f"Accuracy: {accuracy:.6f}")
        print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
