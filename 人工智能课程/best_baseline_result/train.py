from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from best_baseline_result.custom_models import PerClassGMMClassifier

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL.*")

DATASET_PATH = PROJECT_ROOT / "train_dataset.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

RANDOM_STATE = 42
VALID_SIZE = 0.2
TARGET_COLUMN = "Label"
MODEL_NAME = "gmm_2_full"
EXPECTED_FEATURES = ["Feature_1", "Feature_2", "Feature_3"]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def read_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"找不到训练数据：{DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"训练数据必须包含标签列：{TARGET_COLUMN}")
    missing = [col for col in EXPECTED_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"训练数据缺少特征列：{missing}")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    missing = [col for col in EXPECTED_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"输入数据缺少特征列：{missing}")
    return EXPECTED_FEATURES.copy()


def build_model() -> PerClassGMMClassifier:
    return PerClassGMMClassifier(
        n_components=2,
        covariance_type="full",
        random_state=RANDOM_STATE,
        reg_covar=1e-6,
    )


def score_model(model: Any, X: pd.DataFrame, y: pd.Series, splitter: Any) -> tuple[float, float]:
    scores: list[float] = []
    for train_idx, valid_idx in splitter.split(X, y):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(X.iloc[valid_idx])
        scores.append(float(accuracy_score(y.iloc[valid_idx], pred)))
    return float(np.mean(scores)), float(np.std(scores, ddof=0))


def class_metrics(y_true: pd.Series, y_pred: np.ndarray) -> list[dict[str, Any]]:
    labels = sorted(pd.Series(y_true).unique().tolist())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return [
        {
            "类别": int(label),
            "精确率": float(p_value),
            "召回率": float(r_value),
            "F1": float(f_value),
            "样本数": int(s_value),
        }
        for label, p_value, r_value, f_value, s_value in zip(labels, precision, recall, f1, support)
    ]


def write_evaluation_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# 最佳基线模型评估报告",
        "",
        f"- 模型名称：`{metrics['model_name']}`",
        f"- 固定验证集准确率：`{metrics['valid_accuracy']:.6f}`",
        f"- 5 折交叉验证平均准确率：`{metrics['cv_mean_accuracy']:.6f}`",
        f"- 重复交叉验证平均准确率：`{metrics['repeated_cv_mean_accuracy']:.6f}`",
        "",
        "## 验证集分类指标",
        "",
        "| 类别 | 精确率 | 召回率 | F1 | 样本数 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["valid_class_metrics"]:
        lines.append(
            f"| {row['类别']} | {row['精确率']:.6f} | {row['召回率']:.6f} | {row['F1']:.6f} | {row['样本数']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = read_dataset()
    features = feature_columns(df)
    X = df[features]
    y = df[TARGET_COLUMN]

    train_df, valid_df = train_test_split(
        df,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_df = train_df.sort_index()
    valid_df = valid_df.sort_index()

    model = build_model()
    split_model = clone(model)
    split_model.fit(train_df[features], train_df[TARGET_COLUMN])
    valid_pred = split_model.predict(valid_df[features])
    valid_accuracy = float(accuracy_score(valid_df[TARGET_COLUMN], valid_pred))

    cv_mean, cv_std = score_model(
        model,
        X,
        y,
        StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    )
    repeated_cv_mean, repeated_cv_std = score_model(
        model,
        X,
        y,
        RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE),
    )

    final_model = clone(model)
    final_model.fit(X, y)
    joblib.dump(final_model, ARTIFACTS_DIR / "final_model.joblib")

    metrics = {
        "model_name": MODEL_NAME,
        "description": "每个类别单独训练一个两成分全协方差高斯混合模型。",
        "model_family": "生成式分类模型",
        "source_round": "第二轮基线实验",
        "feature_columns": features,
        "target_column": TARGET_COLUMN,
        "train_size": int(len(train_df)),
        "valid_size": int(len(valid_df)),
        "valid_accuracy": valid_accuracy,
        "cv_mean_accuracy": cv_mean,
        "cv_std_accuracy": cv_std,
        "repeated_cv_mean_accuracy": repeated_cv_mean,
        "repeated_cv_std_accuracy": repeated_cv_std,
        "selection_basis": "5折分层交叉验证重复3次",
        "selection_score": repeated_cv_mean,
        "valid_class_metrics": class_metrics(valid_df[TARGET_COLUMN], valid_pred),
    }
    write_json(ARTIFACTS_DIR / "metrics.json", metrics)
    write_json(
        ARTIFACTS_DIR / "model_info.json",
        {
            "模型名称": MODEL_NAME,
            "模型说明": metrics["description"],
            "输入特征": features,
            "标签列": TARGET_COLUMN,
            "模型参数": model.get_params(deep=True),
            "默认模型文件": "final_model.joblib",
        },
    )
    write_evaluation_report(ARTIFACTS_DIR / "evaluation_report.md", metrics)

    print("训练完成。")
    print(f"模型名称：{MODEL_NAME}")
    print(f"固定验证集准确率：{valid_accuracy:.6f}")
    print(f"重复交叉验证准确率：{repeated_cv_mean:.6f}")
    print(f"最终模型文件：{ARTIFACTS_DIR / 'final_model.joblib'}")


if __name__ == "__main__":
    main()
