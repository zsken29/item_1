from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from advanced_model_result.custom_models import OffsetPerClassGMMClassifier

DATASET_PATH = PROJECT_ROOT / "train_dataset.csv"
OUTPUT_DIR = ROOT / "visualizations"
FEATURES = ["Feature_1", "Feature_2", "Feature_3"]
TARGET = "Label"
RANDOM_STATE = 42
COLORS = {0: "#2563eb", 1: "#dc2626", 2: "#16a34a"}


def current_best_model() -> OffsetPerClassGMMClassifier:
    return OffsetPerClassGMMClassifier(
        n_components=2,
        covariance_type="full",
        random_state=RANDOM_STATE,
        reg_covar=1e-6,
        n_init=3,
        init_params="kmeans",
        max_iter=500,
        class_n_components={0: 1, 1: 7, 2: 11},
        class_offsets={0: -0.1, 1: 0.0, 2: 0.0},
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def covariance_ellipse(ax: Any, x: np.ndarray, y: np.ndarray, color: str, label: str) -> None:
    data = np.column_stack([x, y])
    cov = np.cov(data, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    center = data.mean(axis=0)
    for scale, alpha in [(1.0, 0.22), (2.0, 0.10)]:
        ellipse = Ellipse(
            xy=center,
            width=2 * scale * np.sqrt(vals[0]),
            height=2 * scale * np.sqrt(vals[1]),
            angle=angle,
            facecolor=color,
            edgecolor=color,
            linewidth=1.8,
            alpha=alpha,
            label=label if scale == 1.0 else None,
        )
        ax.add_patch(ellipse)


def plot_pairwise_scatter(df: pd.DataFrame) -> Path:
    pairs = [("Feature_1", "Feature_2"), ("Feature_1", "Feature_3"), ("Feature_2", "Feature_3")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (x_col, y_col) in zip(axes, pairs):
        for label in sorted(df[TARGET].unique()):
            part = df[df[TARGET] == label]
            ax.scatter(
                part[x_col],
                part[y_col],
                s=9,
                alpha=0.23,
                c=COLORS[int(label)],
                label=f"class {label}",
                edgecolors="none",
            )
        ax.set_title(f"{x_col} vs {y_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, markerscale=2)
    path = OUTPUT_DIR / "01_pairwise_scatter.png"
    savefig(path)
    return path


def plot_feature_distributions(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    bins = 60
    for ax, feature in zip(axes, FEATURES):
        for label in sorted(df[TARGET].unique()):
            part = df[df[TARGET] == label][feature]
            ax.hist(
                part,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.20,
                color=COLORS[int(label)],
                label=f"class {label}",
            )
            ax.hist(part, bins=bins, density=True, histtype="step", linewidth=1.4, color=COLORS[int(label)])
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("density")
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False)
    path = OUTPUT_DIR / "02_feature_distributions.png"
    savefig(path)
    return path


def plot_projection(df: pd.DataFrame) -> Path:
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(Xs, y)

    # PCA is implemented directly to avoid adding another plotting dependency.
    _, _, vh = np.linalg.svd(Xs - Xs.mean(axis=0), full_matrices=False)
    pca = (Xs - Xs.mean(axis=0)) @ vh[:2].T

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, values, title in [(axes[0], pca, "PCA projection"), (axes[1], lda, "LDA projection")]:
        for label in sorted(df[TARGET].unique()):
            mask = y == label
            ax.scatter(
                values[mask, 0],
                values[mask, 1],
                s=9,
                alpha=0.25,
                c=COLORS[int(label)],
                label=f"class {label}",
                edgecolors="none",
            )
        ax.set_title(title)
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, markerscale=2)
    path = OUTPUT_DIR / "03_pca_lda_projection.png"
    savefig(path)
    return path


def plot_class_01_overlap(df: pd.DataFrame) -> Path:
    part = df[df[TARGET].isin([0, 1])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for label in [0, 1]:
        data = part[part[TARGET] == label]
        axes[0].scatter(
            data["Feature_2"],
            data["Feature_3"],
            s=10,
            alpha=0.25,
            c=COLORS[label],
            label=f"class {label}",
            edgecolors="none",
        )
        covariance_ellipse(axes[0], data["Feature_2"].to_numpy(), data["Feature_3"].to_numpy(), COLORS[label], "")
    axes[0].set_title("Class 0 and 1 overlap in Feature_2 / Feature_3")
    axes[0].set_xlabel("Feature_2")
    axes[0].set_ylabel("Feature_3")
    axes[0].grid(alpha=0.18)
    axes[0].legend(frameon=False)

    bins = 70
    for feature, alpha in [("Feature_2", 0.28), ("Feature_3", 0.16)]:
        for label in [0, 1]:
            data = part[part[TARGET] == label][feature]
            axes[1].hist(
                data,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.7 if feature == "Feature_2" else 1.1,
                color=COLORS[label],
                alpha=1.0 if feature == "Feature_2" else 0.55,
                label=f"class {label} {feature}" if feature == "Feature_2" else None,
            )
            axes[1].hist(data, bins=bins, density=True, histtype="stepfilled", color=COLORS[label], alpha=alpha)
    axes[1].set_title("0/1 marginal overlap")
    axes[1].set_xlabel("feature value")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False)

    path = OUTPUT_DIR / "04_class_0_1_overlap.png"
    savefig(path)
    return path


def repeated_confusion(df: pd.DataFrame) -> tuple[Path, float, np.ndarray, np.ndarray]:
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    matrix = np.zeros((3, 3), dtype=int)
    scores: list[float] = []

    for train_idx, valid_idx in splitter.split(X, y):
        model = clone(current_best_model())
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[valid_idx])
        scores.append(float(accuracy_score(y[valid_idx], pred)))
        matrix += confusion_matrix(y[valid_idx], pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Repeated 5-fold confusion matrix\nmean accuracy = {np.mean(scores):.6f}")
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = OUTPUT_DIR / "05_repeated_cv_confusion.png"
    savefig(path)
    return path, float(np.mean(scores)), np.asarray(scores), matrix


def pairwise_difficulty(df: pd.DataFrame) -> tuple[Path, list[dict[str, Any]]]:
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = [
        ("QDA", QuadraticDiscriminantAnalysis()),
        ("GMM-2", OffsetPerClassGMMClassifier(n_components=2, covariance_type="full", random_state=RANDOM_STATE, n_init=3)),
        ("GMM-6", OffsetPerClassGMMClassifier(n_components=6, covariance_type="full", random_state=RANDOM_STATE, n_init=3)),
    ]
    rows: list[dict[str, Any]] = []
    for pair in [(0, 1), (0, 2), (1, 2)]:
        mask = np.isin(y, pair)
        X_pair = X[mask]
        y_pair = y[mask]
        best = {"pair": f"{pair[0]} vs {pair[1]}", "model": "", "accuracy": 0.0}
        for name, estimator in candidates:
            scores = []
            for train_idx, valid_idx in splitter.split(X_pair, y_pair):
                model = clone(estimator)
                model.fit(X_pair[train_idx], y_pair[train_idx])
                pred = model.predict(X_pair[valid_idx])
                scores.append(float(accuracy_score(y_pair[valid_idx], pred)))
            score = float(np.mean(scores))
            if score > best["accuracy"]:
                best = {"pair": f"{pair[0]} vs {pair[1]}", "model": name, "accuracy": score}
        rows.append(best)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    labels = [row["pair"] for row in rows]
    values = [row["accuracy"] for row in rows]
    bars = ax.bar(labels, values, color=["#9333ea", "#0ea5e9", "#22c55e"], alpha=0.82)
    ax.axhline(0.84, color="#ef4444", linestyle="--", linewidth=1.4, label="target 0.84")
    ax.set_ylim(0.70, 1.0)
    ax.set_title("Pairwise class separability")
    ax.set_ylabel("best 5-fold binary accuracy")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{row['accuracy']:.3f}\n{row['model']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    path = OUTPUT_DIR / "06_pairwise_difficulty.png"
    savefig(path)
    return path, rows


def local_neighbor_diagnostic(df: pd.DataFrame) -> Path:
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    k_values = [1, 3, 5, 7, 11, 15, 25, 35, 51, 75, 101, 151, 201]
    nearest = NearestNeighbors(n_neighbors=max(k_values) + 1).fit(Xs)
    indices = nearest.kneighbors(Xs, return_distance=False)[:, 1:]
    accuracies = []
    majorities = []
    for k in k_values:
        pred = []
        purity = []
        for neigh in indices[:, :k]:
            counts = np.bincount(y[neigh], minlength=3)
            pred.append(int(np.argmax(counts)))
            purity.append(float(counts.max() / k))
        accuracies.append(float(accuracy_score(y, pred)))
        majorities.append(float(np.mean(purity)))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(k_values, accuracies, marker="o", label="leave-one-out majority accuracy")
    ax.plot(k_values, majorities, marker="s", label="mean local majority share")
    ax.axhline(0.84, color="#ef4444", linestyle="--", linewidth=1.4, label="target 0.84")
    ax.set_xscale("log")
    ax.set_ylim(0.70, 0.86)
    ax.set_title("Local neighbor ambiguity")
    ax.set_xlabel("number of neighbors")
    ax.set_ylabel("score")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    path = OUTPUT_DIR / "07_local_neighbor_ambiguity.png"
    savefig(path)
    return path


def write_report(paths: dict[str, Path], mean_accuracy: float, scores: np.ndarray, matrix: np.ndarray, pair_rows: list[dict[str, Any]]) -> Path:
    recalls = np.diag(matrix) / matrix.sum(axis=1)
    pair_lines = "\n".join(
        f"| {row['pair']} | {row['model']} | {row['accuracy']:.6f} |" for row in pair_rows
    )
    report = f"""# Data visualization report

This report visualizes why classes `0` and `1` are much harder to separate than class `2`.

## Key findings

- Current best model repeated 5-fold accuracy: `{mean_accuracy:.6f}`.
- Repeated 5-fold per-class recall: class 0 `{recalls[0]:.6f}`, class 1 `{recalls[1]:.6f}`, class 2 `{recalls[2]:.6f}`.
- Most mistakes are between class `0` and class `1`; class `2` is comparatively clean.
- The dedicated binary task `0 vs 1` is far below `0 vs 2` and `1 vs 2`.

## Pairwise separability

| pair | best quick binary model | 5-fold binary accuracy |
|---|---|---:|
{pair_lines}

## Figures

| figure | file |
|---|---|
| Pairwise scatter | `{paths['pairwise'].name}` |
| Feature distributions | `{paths['distributions'].name}` |
| PCA/LDA projections | `{paths['projection'].name}` |
| Class 0/1 overlap | `{paths['overlap_01'].name}` |
| Repeated CV confusion | `{paths['confusion'].name}` |
| Pairwise difficulty | `{paths['pairwise_difficulty'].name}` |
| Local neighbor ambiguity | `{paths['local_neighbor'].name}` |
"""
    path = OUTPUT_DIR / "data_visualization_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    paths = {
        "pairwise": plot_pairwise_scatter(df),
        "distributions": plot_feature_distributions(df),
        "projection": plot_projection(df),
        "overlap_01": plot_class_01_overlap(df),
    }
    confusion_path, mean_accuracy, scores, matrix = repeated_confusion(df)
    pairwise_path, pair_rows = pairwise_difficulty(df)
    local_path = local_neighbor_diagnostic(df)
    paths.update(
        {
            "confusion": confusion_path,
            "pairwise_difficulty": pairwise_path,
            "local_neighbor": local_path,
        }
    )
    report_path = write_report(paths, mean_accuracy, scores, matrix, pair_rows)

    print(f"visualizations_dir={OUTPUT_DIR}")
    print(f"report={report_path}")
    print(f"repeated_cv_mean={mean_accuracy:.6f}")
    print("confusion_matrix=")
    print(matrix)
    print("pairwise_rows=")
    for row in pair_rows:
        print(row)


if __name__ == "__main__":
    main()
