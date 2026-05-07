from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted


class PerClassGMMClassifier(BaseEstimator, ClassifierMixin):
    """每个类别单独训练一个高斯混合模型，然后按后验概率完成分类。"""

    _estimator_type = "classifier"

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        random_state: int = 42,
        reg_covar: float = 1e-6,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.reg_covar = reg_covar

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_: dict[int, GaussianMixture] = {}
        self.log_priors_: dict[int, float] = {}

        for cls in self.classes_:
            X_cls = X[y == cls]
            model = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                reg_covar=self.reg_covar,
            )
            model.fit(X_cls)
            self.models_[cls] = model
            self.log_priors_[cls] = float(np.log(len(X_cls) / len(X)))

        return self

    def _joint_log_likelihood(self, X) -> np.ndarray:
        check_is_fitted(self, ["classes_", "models_", "log_priors_"])
        X = np.asarray(X)
        return np.column_stack(
            [self.models_[cls].score_samples(X) + self.log_priors_[cls] for cls in self.classes_]
        )

    def predict_proba(self, X) -> np.ndarray:
        joint = self._joint_log_likelihood(X)
        joint -= joint.max(axis=1, keepdims=True)
        probs = np.exp(joint)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]
