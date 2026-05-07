"""
路径B实验：Optuna自动调参 GMM - 修复版
"""

from __future__ import annotations

import copy
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

DATASET_PATH = PROJECT_ROOT / "train_dataset.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
RANDOM_STATE = 42
VALID_SIZE = 0.2
TARGET_COLUMN = "Label"
EXPECTED_FEATURES = ["Feature_1", "Feature_2", "Feature_3"]


def jwrite(path, payload):
    def jr(v):
        if isinstance(v, dict): return {str(k): jr(v) for k, v in v.items()}
        if isinstance(v, (list, tuple)): return [jr(x) for x in v]
        if hasattr(v, "tolist") and not isinstance(v, (str, bytes)): return v.tolist()
        if hasattr(v, "item"): return v.item()
        return v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)
    path.write_text(json.dumps(jr(payload), indent=2, ensure_ascii=False), encoding="utf-8")


class PerClassGMM(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(
        self,
        n_components=2,
        covariance_type="full",
        random_state=42,
        reg_covar=1e-6,
        n_init=1,
        max_iter=200,
        # per-class overrides (by index in self.classes_)
        class_n_components=None,  # list of 3 ints or None
        class_offsets=None,        # list of 3 floats
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.max_iter = max_iter
        self.class_n_components = class_n_components or [None, None, None]
        self.class_offsets = class_offsets or [0.0, 0.0, 0.0]

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = {}
        self.log_priors_ = {}

        for idx, cls in enumerate(self.classes_):
            mask = (y == cls)
            X_cls = X[mask]
            n_c = self.class_n_components[idx]
            if n_c is None:
                n_c = self.n_components
            gmm = GaussianMixture(
                n_components=n_c,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                reg_covar=self.reg_covar,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            gmm.fit(X_cls)
            self.models_[cls] = gmm
            self.log_priors_[cls] = float(np.log(mask.sum() / len(y)))

        self.offsets_ = np.array(self.class_offsets[:len(self.classes_)])
        return self

    def _jll(self, X):
        X = np.asarray(X, dtype=np.float64)
        cols = []
        for cls in self.classes_:
            s = self.models_[cls].score_samples(X)
            cols.append(s + self.log_priors_[cls])
        return np.column_stack(cols) + self.offsets_

    def predict(self, X):
        return self.classes_[np.argmax(self._jll(X), axis=1)]

    def predict_proba(self, X):
        j = self._jll(X)
        j -= j.max(axis=1, keepdims=True)
        p = np.exp(j)
        return p / p.sum(axis=1, keepdims=True)


def load_data():
    df = pd.read_csv(DATASET_PATH)
    tr_df, va_df = train_test_split(df, test_size=VALID_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN])
    tr_df = tr_df.sort_index()
    va_df = va_df.sort_index()
    return df, tr_df, va_df


def cv_score(model, X, y, n_splits=5, n_repeats=3, seed=42):
    """Repeated stratified K-fold CV."""
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = []
    for tr_idx, va_idx in splitter.split(X, y):
        m = PerClassGMM(**model.get_params())
        m.fit(X[tr_idx], y[tr_idx])
        scores.append(accuracy_score(y[va_idx], m.predict(X[va_idx])))
    return float(np.mean(scores))


def cv_score_from_params(params, X, y, n_splits=5, n_repeats=3, seed=42):
    """Build model from dict params and run repeated CV."""
    nc = [params['nc0'], params['nc1'], params['nc2']]
    offsets = [params['off0'], params['off1'], params['off2']]
    model = PerClassGMM(
        n_components=2,
        covariance_type=params['covariance_type'],
        random_state=RANDOM_STATE,
        reg_covar=params['reg_covar'],
        n_init=params['n_init'],
        max_iter=params['max_iter'],
        class_n_components=nc,
        class_offsets=offsets,
    )
    return cv_score(model, X, y, n_splits, n_repeats, seed)


def objective(trial, X, y):
    params = {
        'nc0': trial.suggest_int('nc0', 1, 8),
        'nc1': trial.suggest_int('nc1', 1, 12),
        'nc2': trial.suggest_int('nc2', 1, 12),
        'off0': trial.suggest_float('off0', -0.5, 0.5),
        'off1': trial.suggest_float('off1', -0.5, 0.5),
        'off2': trial.suggest_float('off2', -0.5, 0.5),
        'covariance_type': trial.suggest_categorical('covariance_type', ['full', 'diag']),
        'reg_covar': trial.suggest_float('reg_covar', 1e-6, 1e-1, log=True),
        'n_init': trial.suggest_categorical('n_init', [1, 3]),
        'max_iter': trial.suggest_categorical('max_iter', [200, 500]),
    }
    return cv_score_from_params(params, X, y, n_splits=5, n_repeats=1, seed=RANDOM_STATE)


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    df, tr_df, va_df = load_data()
    Xf = df[EXPECTED_FEATURES].values
    yf = df[TARGET_COLUMN].values
    Xt = tr_df[EXPECTED_FEATURES].values
    yt = tr_df[TARGET_COLUMN].values
    Xv = va_df[EXPECTED_FEATURES].values
    yv = va_df[TARGET_COLUMN].values

    print(f"Dataset: {len(df)} rows, Train: {len(tr_df)}, Valid: {len(va_df)}")

    # ── Optuna search ────────────────────────────────────────
    print("\n[Optuna] 200 trials with MedianPruner...")
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=15)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(lambda t: objective(t, Xf, yf), n_trials=200, show_progress_bar=True)

    bp = study.best_params
    print(f"\nBest 5-fold CV (n_reps=1): {study.best_value:.4f}")
    print(f"Best params: {bp}")

    # ── Evaluate with full repCV (n_repeats=3) ───────────────
    print("\n[Final] RepCV (5x3) evaluation...")
    rep_cv = cv_score_from_params(bp, Xf, yf, n_splits=5, n_repeats=3, seed=RANDOM_STATE)

    # Fixed validation
    nc = [bp['nc0'], bp['nc1'], bp['nc2']]
    off = [bp['off0'], bp['off1'], bp['off2']]
    model_on_train = PerClassGMM(
        n_components=2,
        covariance_type=bp['covariance_type'],
        random_state=RANDOM_STATE,
        reg_covar=bp['reg_covar'],
        n_init=bp['n_init'],
        max_iter=bp['max_iter'],
        class_n_components=nc,
        class_offsets=off,
    )
    model_on_train.fit(Xt, yt)
    valid_pred = model_on_train.predict(Xv)
    valid_acc = float(accuracy_score(yv, valid_pred))

    # 5-fold CV for std
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for tr_idx, va_idx in cv_splitter.split(Xf, yf):
        m = PerClassGMM(
            n_components=2,
            covariance_type=bp['covariance_type'],
            random_state=RANDOM_STATE,
            reg_covar=bp['reg_covar'],
            n_init=bp['n_init'],
            max_iter=bp['max_iter'],
            class_n_components=nc,
            class_offsets=off,
        )
        m.fit(Xf[tr_idx], yf[tr_idx])
        cv_scores.append(float(accuracy_score(yf[va_idx], m.predict(Xf[va_idx]))))
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores, ddof=0))

    print(f"\n  Fixed Valid Accuracy: {valid_acc:.4f}")
    print(f"  5-fold CV: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  Repeated 5-fold CV (3x): {rep_cv:.4f}")

    # Per-class
    prec, rec, f1, sup = precision_recall_fscore_support(yv, valid_pred, labels=[0, 1, 2], zero_division=0)
    class_metrics = [
        {"class": i, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for i, (p, r, f, s) in enumerate(zip(prec, rec, f1, sup))
    ]
    print("\n  Per-class metrics:")
    for cm in class_metrics:
        print(f"    Class {cm['class']}: P={cm['precision']:.4f} R={cm['recall']:.4f} F1={cm['f1']:.4f}")

    # Final model on all data
    print("\nTraining final model on all data...")
    final_model = PerClassGMM(
        n_components=2,
        covariance_type=bp['covariance_type'],
        random_state=RANDOM_STATE,
        reg_covar=bp['reg_covar'],
        n_init=bp['n_init'],
        max_iter=bp['max_iter'],
        class_n_components=nc,
        class_offsets=off,
    )
    final_model.fit(Xf, yf)
    joblib.dump(final_model, ARTIFACTS_DIR / "optuna_best_model.joblib")

    result = {
        "valid_accuracy": valid_acc,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "repeated_cv_mean": rep_cv,
        "best_params": bp,
        "best_5fold_cv": study.best_value,
        "class_metrics": class_metrics,
        "baseline_valid": 0.815,
        "baseline_repeated_cv": 0.8113,
        "improvement_valid": valid_acc - 0.815,
        "improvement_repCV": rep_cv - 0.8113,
    }
    jwrite(ARTIFACTS_DIR / "optuna_result.json", result)
    jwrite(ARTIFACTS_DIR / "optuna_study.json", {
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": bp,
    })

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Baseline:  valid=0.8150,  repCV=0.8113")
    print(f"Optuna:    valid={valid_acc:.4f},  repCV={rep_cv:.4f}")
    print(f"Improvement: valid {valid_acc - 0.815:+.4f},  repCV {rep_cv - 0.8113:+.4f}")
    print(f"\nBest params found:")
    for k, v in bp.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()