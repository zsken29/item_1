from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted


class OffsetPerClassGMMClassifier(BaseEstimator, ClassifierMixin):
    """Per-class Gaussian mixtures with fixed additive class log-score offsets."""

    _estimator_type = "classifier"

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        random_state: int = 42,
        reg_covar: float = 1e-6,
        n_init: int = 1,
        init_params: str = "kmeans",
        max_iter: int = 500,
        class_n_components: Mapping[int, int] | None = None,
        class_offsets: Mapping[int, float] | None = None,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.init_params = init_params
        self.max_iter = max_iter
        self.class_n_components = class_n_components
        self.class_offsets = class_offsets

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_: dict[int, GaussianMixture] = {}
        self.log_priors_: dict[int, float] = {}
        class_n_components = self.class_n_components or {}

        for cls in self.classes_:
            X_cls = X[y == cls]
            n_components = int(class_n_components.get(int(cls), self.n_components))
            model = GaussianMixture(
                n_components=n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                reg_covar=self.reg_covar,
                n_init=self.n_init,
                init_params=self.init_params,
                max_iter=self.max_iter,
            )
            model.fit(X_cls)
            self.models_[cls] = model
            self.log_priors_[cls] = float(np.log(len(X_cls) / len(X)))

        offsets = self.class_offsets or {}
        self.class_offsets_ = np.array([float(offsets.get(int(cls), 0.0)) for cls in self.classes_])
        return self

    def _raw_joint_log_likelihood(self, X) -> np.ndarray:
        check_is_fitted(self, ["classes_", "models_", "log_priors_"])
        X = np.asarray(X)
        return np.column_stack(
            [self.models_[cls].score_samples(X) + self.log_priors_[cls] for cls in self.classes_]
        )

    def _joint_log_likelihood(self, X) -> np.ndarray:
        check_is_fitted(self, ["class_offsets_"])
        return self._raw_joint_log_likelihood(X) + self.class_offsets_

    def predict_proba(self, X) -> np.ndarray:
        joint = self._joint_log_likelihood(X)
        joint -= joint.max(axis=1, keepdims=True)
        probs = np.exp(joint)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]
