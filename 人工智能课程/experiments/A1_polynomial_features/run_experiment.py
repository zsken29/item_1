"""
路径A实验：特征工程增强
A1: 多项式交互特征 (2次多项式扩展)
A2: PCA/ICA降维变换 + GMM
"""

from __future__ import annotations

import copy
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.decomposition._fastica import FastICA as ICA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL.*")

DATASET_PATH = PROJECT_ROOT / "train_dataset.csv"
EXPERIMENT_ROOT = ROOT
ARTIFACTS_DIR = EXPERIMENT_ROOT / "artifacts"

RANDOM_STATE = 42
VALID_SIZE = 0.2
TARGET_COLUMN = "Label"
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


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def read_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset missing target: {TARGET_COLUMN}")
    return df


def add_polynomial_features(df: pd.DataFrame, features: list[str], degree: int = 2) -> tuple[pd.DataFrame, list[str]]:
    """Add polynomial and interaction features. Returns (df, all_feature_names)."""
    result = df.copy()
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_orig = df[features].values
    X_new = poly.fit_transform(X_orig)
    poly_feature_names = [n for n in poly.get_feature_names_out(features) if n not in features]
    for i, name in enumerate(poly_feature_names):
        result[name] = X_new[:, i]
    return result, features + poly_feature_names


class OffsetPerClassGMMClassifier(BaseEstimator, ClassifierMixin):
    """Per-class GMM with optional class-specific components and offsets."""

    _estimator_type = "classifier"

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        random_state: int = 42,
        reg_covar: float = 1e-6,
        n_init: int = 1,
        max_iter: int = 500,
        class_n_components_0: int | None = None,
        class_n_components_1: int | None = None,
        class_n_components_2: int | None = None,
        class_offset_0: float = 0.0,
        class_offset_1: float = 0.0,
        class_offset_2: float = 0.0,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.max_iter = max_iter
        # Per-class n_components as separate params (immutable for clone)
        self.class_n_components_0 = class_n_components_0
        self.class_n_components_1 = class_n_components_1
        self.class_n_components_2 = class_n_components_2
        self.class_offset_0 = class_offset_0
        self.class_offset_1 = class_offset_1
        self.class_offset_2 = class_offset_2

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = {}
        self.log_priors_ = {}

        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            nc = [self.class_n_components_0, self.class_n_components_1, self.class_n_components_2][idx]
            n_c = nc if nc is not None else self.n_components
            model = GaussianMixture(
                n_components=n_c,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                reg_covar=self.reg_covar,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            model.fit(X_cls)
            self.models_[cls] = model
            self.log_priors_[cls] = float(np.log(len(X_cls) / len(X)))

        self.offsets_ = np.array([self.class_offset_0, self.class_offset_1, self.class_offset_2][:len(self.classes_)])
        return self

    def _joint_log_likelihood(self, X) -> np.ndarray:
        X = np.asarray(X)
        return np.column_stack(
            [self.models_[cls].score_samples(X) + self.log_priors_[cls] for cls in self.classes_]
        ) + self.offsets_

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        joint = self._joint_log_likelihood(X)
        joint -= joint.max(axis=1, keepdims=True)
        probs = np.exp(joint)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


class TransformedGMMClassifier(BaseEstimator, ClassifierMixin):
    """Pipeline: Scaler -> Transformer(optional) -> OffsetPerClassGMM"""

    _estimator_type = "classifier"

    def __init__(
        self,
        transformer_type: str = "none",
        n_components: int = 2,
        covariance_type: str = "full",
        random_state: int = 42,
        reg_covar: float = 1e-6,
        n_init: int = 1,
        max_iter: int = 500,
        class_n_components_0: int | None = None,
        class_n_components_1: int | None = None,
        class_n_components_2: int | None = None,
        class_offset_0: float = 0.0,
        class_offset_1: float = 0.0,
        class_offset_2: float = 0.0,
        transformer_n_components: int = 3,
    ) -> None:
        self.transformer_type = transformer_type
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.max_iter = max_iter
        self.class_n_components_0 = class_n_components_0
        self.class_n_components_1 = class_n_components_1
        self.class_n_components_2 = class_n_components_2
        self.class_offset_0 = class_offset_0
        self.class_offset_1 = class_offset_1
        self.class_offset_2 = class_offset_2
        self.transformer_n_components = transformer_n_components

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        self.scaler_ = StandardScaler()
        X_t = self.scaler_.fit_transform(X)

        if self.transformer_type == "pca":
            self.transformer_ = PCA(n_components=self.transformer_n_components, random_state=self.random_state)
            X_t = self.transformer_.fit_transform(X_t)
        elif self.transformer_type == "ica":
            self.transformer_ = ICA(n_components=self.transformer_n_components, random_state=self.random_state, max_iter=500)
            X_t = self.transformer_.fit_transform(X_t)
        else:
            self.transformer_ = None

        self.gmm_ = OffsetPerClassGMMClassifier(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            reg_covar=self.reg_covar,
            n_init=self.n_init,
            max_iter=self.max_iter,
            class_n_components_0=self.class_n_components_0,
            class_n_components_1=self.class_n_components_1,
            class_n_components_2=self.class_n_components_2,
            class_offset_0=self.class_offset_0,
            class_offset_1=self.class_offset_1,
            class_offset_2=self.class_offset_2,
        )
        self.gmm_.fit(X_t, y)
        return self

    def predict(self, X):
        return self.gmm_.predict(self._transform(X))

    def predict_proba(self, X):
        return self.gmm_.predict_proba(self._transform(X))

    def _transform(self, X):
        X = np.asarray(X)
        X_t = self.scaler_.transform(X)
        if self.transformer_ is not None:
            X_t = self.transformer_.transform(X_t)
        return X_t


def make_model(config: dict) -> TransformedGMMClassifier:
    """Factory to create model from flat config dict."""
    config = {k: v for k, v in config.items() if k != "feature_names"}
    return TransformedGMMClassifier(**config)


def score_model(model: TransformedGMMClassifier, X: pd.DataFrame, y: pd.Series, splitter: Any) -> tuple[float, float]:
    scores = []
    for train_idx, valid_idx in splitter.split(X, y):
        fitted = copy.deepcopy(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(X.iloc[valid_idx])
        scores.append(float(accuracy_score(y.iloc[valid_idx], pred)))
    return float(np.mean(scores)), float(np.std(scores, ddof=0))


def run_single_experiment(
    name: str,
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    splitter5,
    splitter_repeated,
) -> dict:
    model = make_model(config)

    # Fixed validation
    split_model = copy.deepcopy(model)
    split_model.fit(X_train, y_train)
    valid_pred = split_model.predict(X_valid)
    valid_acc = float(accuracy_score(y_valid, valid_pred))

    # CV scores
    cv_mean, cv_std = score_model(model, X_full, y_full, splitter5)
    repeated_cv_mean, repeated_cv_std = score_model(model, X_full, y_full, splitter_repeated)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_valid, valid_pred, labels=[0, 1, 2], zero_division=0
    )
    class_metrics = [
        {"class": i, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support))
    ]

    # Final model
    final_model = copy.deepcopy(model)
    final_model.fit(X_full, y_full)
    joblib.dump(final_model, ARTIFACTS_DIR / f"{name}_final_model.joblib")

    return {
        "name": name,
        "valid_accuracy": valid_acc,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "repeated_cv_mean": repeated_cv_mean,
        "repeated_cv_std": repeated_cv_std,
        "class_metrics": class_metrics,
        "config": {k: v for k, v in config.items()},
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading dataset...")
    df = read_dataset()
    y = df[TARGET_COLUMN]

    train_df, valid_df = train_test_split(df, test_size=VALID_SIZE, random_state=RANDOM_STATE, stratify=y)
    train_df = train_df.sort_index()
    valid_df = valid_df.sort_index()

    splitter5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splitter_repeated = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    results = []

    def run(name, config, X_tr, y_tr, X_ful, y_ful, X_valid, y_valid):
        r = run_single_experiment(name, config, X_tr, y_tr, X_ful, y_ful,
                                  X_valid, y_valid,
                                  splitter5, splitter_repeated)
        results.append(r)
        print(f"  {name}: valid={r['valid_accuracy']:.4f}  repCV={r['repeated_cv_mean']:.4f}")

    # ── BASELINE ──────────────────────────────────────────────
    print("\n[A0] Baseline: raw 3 features, GMM n_components=2 ...")
    run("A0_baseline_raw",
        {
            "transformer_type": "none", "n_components": 2, "covariance_type": "full",
            "random_state": RANDOM_STATE, "reg_covar": 1e-6, "n_init": 3, "max_iter": 500,
            "class_n_components_0": None, "class_n_components_1": None, "class_n_components_2": None,
            "class_offset_0": 0.0, "class_offset_1": 0.0, "class_offset_2": 0.0,
            "feature_names": EXPECTED_FEATURES,
        },
        train_df[EXPECTED_FEATURES], train_df[TARGET_COLUMN],
        df[EXPECTED_FEATURES], df[TARGET_COLUMN],
        valid_df[EXPECTED_FEATURES], valid_df[TARGET_COLUMN])

    # ── A1: POLYNOMIAL FEATURES ───────────────────────────────
    print("\n[A1a] Polynomial degree=2 (interactions + squares) ...")
    df_poly, poly_features = add_polynomial_features(df, EXPECTED_FEATURES, degree=2)
    print(f"    {len(poly_features) - len(EXPECTED_FEATURES)} new features: {[f for f in poly_features if f not in EXPECTED_FEATURES]}")

    train_poly = df_poly.loc[train_df.index]
    valid_poly = df_poly.loc[valid_df.index]

    configs_poly = []
    for nc in [2, 3, 4]:
        for cov in ["full", "diag"]:
            for reg in [1e-6, 1e-4, 1e-3]:
                configs_poly.append({
                    "transformer_type": "none", "n_components": nc, "covariance_type": cov,
                    "random_state": RANDOM_STATE, "reg_covar": reg, "n_init": 3, "max_iter": 500,
                    "class_n_components_0": None, "class_n_components_1": None, "class_n_components_2": None,
                    "class_offset_0": 0.0, "class_offset_1": 0.0, "class_offset_2": 0.0,
                    "feature_names": poly_features,
                })

    for cfg in configs_poly:
        run(f"A1_poly2_nc{cfg['n_components']}_{cfg['covariance_type']}_reg{cfg['reg_covar']}", cfg,
            train_poly[poly_features], train_df[TARGET_COLUMN], df_poly[poly_features], df[TARGET_COLUMN],
            valid_poly[poly_features], valid_df[TARGET_COLUMN])

    print("\n[A1b] Polynomial degree=2 + per-class components ...")
    for nc0 in [1, 2]:
        for nc1 in [4, 6, 8]:
            for nc2 in [4, 6, 7]:
                run(f"A1_pc_{nc0}_{nc1}_{nc2}",
                    {
                        "transformer_type": "none", "n_components": 2, "covariance_type": "full",
                        "random_state": RANDOM_STATE, "reg_covar": 1e-4, "n_init": 3, "max_iter": 500,
                        "class_n_components_0": nc0, "class_n_components_1": nc1, "class_n_components_2": nc2,
                        "class_offset_0": 0.0, "class_offset_1": 0.0, "class_offset_2": 0.0,
                        "feature_names": poly_features,
                    },
                    train_poly[poly_features], train_df[TARGET_COLUMN], df_poly[poly_features], df[TARGET_COLUMN],
                    valid_poly[poly_features], valid_df[TARGET_COLUMN])

    print("\n[A1c] Polynomial degree=2 + per-class offsets ...")
    for off0, off1, off2 in [
        (0.05, 0.15, 0.0), (0.1, 0.2, 0.0), (0.0, 0.1, -0.05), (0.08, 0.18, 0.02),
        (0.03, 0.12, -0.02), (0.06, 0.16, 0.01),
    ]:
        run(f"A1_offset_{off0}_{off1}_{off2}",
            {
                "transformer_type": "none", "n_components": 2, "covariance_type": "full",
                "random_state": RANDOM_STATE, "reg_covar": 1e-4, "n_init": 3, "max_iter": 500,
                "class_n_components_0": 1, "class_n_components_1": 6, "class_n_components_2": 7,
                "class_offset_0": off0, "class_offset_1": off1, "class_offset_2": off2,
                "feature_names": poly_features,
            },
            train_poly[poly_features], train_df[TARGET_COLUMN], df_poly[poly_features], df[TARGET_COLUMN],
                    valid_poly[poly_features], valid_df[TARGET_COLUMN])

    # ── A2: PCA TRANSFORM ─────────────────────────────────────
    print("\n[A2a] PCA transforms ...")
    for t_nc in [2, 3]:  # max is min(n_samples, n_features) = 3
        for gmm_nc in [2, 3, 4]:
            run(f"A2_pca{t_nc}_gmm{gmm_nc}",
                {
                    "transformer_type": "pca", "transformer_n_components": t_nc,
                    "n_components": gmm_nc, "covariance_type": "full",
                    "random_state": RANDOM_STATE, "reg_covar": 1e-4, "n_init": 3, "max_iter": 500,
                    "class_n_components_0": None, "class_n_components_1": None, "class_n_components_2": None,
                    "class_offset_0": 0.0, "class_offset_1": 0.0, "class_offset_2": 0.0,
                    "feature_names": EXPECTED_FEATURES,
                },
                train_df[EXPECTED_FEATURES], train_df[TARGET_COLUMN], df[EXPECTED_FEATURES], df[TARGET_COLUMN],
                valid_df[EXPECTED_FEATURES], valid_df[TARGET_COLUMN])

    # ── A2: ICA TRANSFORM ─────────────────────────────────────
    print("\n[A2b] ICA transforms ...")
    for t_nc in [2, 3]:  # max is min(n_samples, n_features) = 3
        for gmm_nc in [2, 3, 4]:
            run(f"A2_ica{t_nc}_gmm{gmm_nc}",
                {
                    "transformer_type": "ica", "transformer_n_components": t_nc,
                    "n_components": gmm_nc, "covariance_type": "full",
                    "random_state": RANDOM_STATE, "reg_covar": 1e-4, "n_init": 3, "max_iter": 500,
                    "class_n_components_0": None, "class_n_components_1": None, "class_n_components_2": None,
                    "class_offset_0": 0.0, "class_offset_1": 0.0, "class_offset_2": 0.0,
                    "feature_names": EXPECTED_FEATURES,
                },
                train_df[EXPECTED_FEATURES], train_df[TARGET_COLUMN], df[EXPECTED_FEATURES], df[TARGET_COLUMN],
                valid_df[EXPECTED_FEATURES], valid_df[TARGET_COLUMN])

    # ── A2c: PCA + Polynomial ──────────────────────────────────
    print("\n[A2c] PCA(3) + Polynomial degree=2 ...")
    df_pca_poly, pca_poly_features = add_polynomial_features(df, EXPECTED_FEATURES, degree=2)
    train_pca_poly = df_pca_poly.loc[train_df.index]
    valid_pca_poly = df_pca_poly.loc[valid_df.index]
    for gmm_nc in [2, 3, 4]:
        run(f"A2c_pca3_poly2_gmm{gmm_nc}",
            {
                "transformer_type": "pca", "transformer_n_components": 3,
                "n_components": gmm_nc, "covariance_type": "full",
                "random_state": RANDOM_STATE, "reg_covar": 1e-4, "n_init": 3, "max_iter": 500,
                "class_n_components_0": None, "class_n_components_1": None, "class_n_components_2": None,
                "class_offset_0": 0.0, "class_offset_1": 0.0, "class_offset_2": 0.0,
                "feature_names": pca_poly_features,
            },
            train_pca_poly[pca_poly_features], train_df[TARGET_COLUMN],
            df_pca_poly[pca_poly_features], df[TARGET_COLUMN],
            valid_pca_poly[pca_poly_features], valid_df[TARGET_COLUMN])

    # ── Results summary ─────────────────────────────────────────
    results_sorted = sorted(results, key=lambda x: x["repeated_cv_mean"], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 15 RESULTS (sorted by repeated_cv_mean)")
    print("=" * 70)
    print(f"{'Rank':<4} {'Name':<40} {'Valid':>7} {'RepCV':>7} {'CvStd':>7}")
    print("-" * 70)
    for i, r in enumerate(results_sorted[:15]):
        print(f"{i+1:<4} {r['name']:<40} {r['valid_accuracy']:>7.4f} {r['repeated_cv_mean']:>7.4f} {r['cv_std']:>7.4f}")

    # Save
    write_json(ARTIFACTS_DIR / "experiment_results.json", results)

    best = results_sorted[0]
    print(f"\n*** BEST: {best['name']} ***")
    print(f"    Valid Acc: {best['valid_accuracy']:.4f}")
    print(f"    Repeated CV: {best['repeated_cv_mean']:.4f} ± {best['repeated_cv_std']:.4f}")
    print("\n  Per-class metrics:")
    for cm in best["class_metrics"]:
        print(f"    Class {cm['class']}: P={cm['precision']:.4f} R={cm['recall']:.4f} F1={cm['f1']:.4f}")

    # Save best model info
    write_json(ARTIFACTS_DIR / "best_config.json", {
        "name": best["name"],
        "config": best["config"],
        "valid_accuracy": best["valid_accuracy"],
        "repeated_cv_mean": best["repeated_cv_mean"],
        "class_metrics": best["class_metrics"],
        "baseline_valid": 0.815,
        "baseline_repeated_cv": 0.8113,
        "improvement": best["repeated_cv_mean"] - 0.8113,
    })


if __name__ == "__main__":
    main()