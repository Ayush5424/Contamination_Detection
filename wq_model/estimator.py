"""Hybrid CWQI estimator - paper Section III-C, equation (3), and SHAP attribution, equation (4).

    W_t = W^b_t + beta_0 + sum_m beta_m * f_m(x_t)

The six tree-based learners are stacked with a linear meta-learner, but they only
learn the residual W_t - W^b_t, so the estimate can follow severe contamination that
lies outside the range the trees saw during training.
"""
import numpy as np
import shap
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from .cwqi import SENSORS
from .features import FEATURE_COLUMNS

NAMES = {"xgb": "XGBoost", "cat": "CatBoost", "rf": "Random Forest",
         "gb": "Gradient Boosting", "et": "Extra Trees", "ada": "AdaBoost"}


def base_learners(size="demo"):
    """The six base regressors of the paper.

    size="paper" uses the settings of the published experiment; size="demo" uses fewer
    and shallower trees so the model trains in minutes and the saved files stay small.
    """
    if size == "paper":
        return [
            ("xgb", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, n_jobs=4, random_state=0)),
            ("cat", CatBoostRegressor(iterations=400, depth=6, learning_rate=0.08, verbose=0, allow_writing_files=False,
                                      random_seed=0, thread_count=4)),
            ("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=4, random_state=0)),
            ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                                             subsample=0.8, random_state=0)),
            ("et", ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=4, random_state=0)),
            ("ada", AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=100,
                                      learning_rate=0.1, random_state=0)),
        ]
    return [
        ("xgb", XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, n_jobs=4, random_state=0)),
        ("cat", CatBoostRegressor(iterations=300, depth=6, learning_rate=0.08, verbose=0, allow_writing_files=False,
                                  random_seed=0, thread_count=4)),
        ("rf", RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_leaf=5, n_jobs=4,
                                     random_state=0)),
        ("gb", GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.1,
                                         subsample=0.8, random_state=0)),
        ("et", ExtraTreesRegressor(n_estimators=60, max_depth=12, min_samples_leaf=5, n_jobs=4,
                                   random_state=0)),
        ("ada", AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=50,
                                  learning_rate=0.1, random_state=0)),
    ]


class HybridCWQIEstimator:
    def __init__(self, size="demo", folds=5):
        self.size = size
        self.stack = StackingRegressor(estimators=base_learners(size), final_estimator=LinearRegression(),
                                       cv=KFold(folds, shuffle=False), n_jobs=1)
        self.bg_contrib = None
        self._explainers = None

    # ---------------------------------------------------------------- training
    def fit(self, X, base, target, base_contrib):
        """X: 21 features, base: W^b_t, target: true CWQI, base_contrib: w_s q_s of smoothed readings."""
        self.stack.fit(X, target - base)                     # learn only the residual
        self.bg_contrib = base_contrib.mean(axis=0)          # E[w_s q_s] for equation (4)
        return self

    def predict(self, X, base):
        return base + self.stack.predict(X)                  # equation (3)

    @property
    def meta_weights(self):
        return {NAMES[k]: float(c) for (k, _), c in zip(self.stack.estimators, self.stack.final_estimator_.coef_)}

    # ---------------------------------------------------------------- explanation, equation (4)
    def _get_explainers(self):
        if self._explainers is None:
            coef = dict(zip([k for k, _ in self.stack.estimators], self.stack.final_estimator_.coef_))
            # boosted members: exact and fast TreeSHAP (Section III-D)
            self._explainers = [(shap.TreeExplainer(self.stack.named_estimators_[k]), coef[k])
                                for k in ("xgb", "cat", "gb")]
        return self._explainers

    def attribute(self, X, base_contrib):
        """Sensor-level attribution Phi_s for each row of X (index points, one column per sensor)."""
        X = np.atleast_2d(X)
        phi = np.zeros((X.shape[0], len(FEATURE_COLUMNS)))
        for explainer, beta in self._get_explainers():
            phi += beta * np.asarray(explainer.shap_values(X, check_additivity=False))
        groups = [[FEATURE_COLUMNS.index(c) for c in FEATURE_COLUMNS if c.split("_")[0] == s] for s in SENSORS]
        residual = np.column_stack([phi[:, g].sum(axis=1) for g in groups])
        return (np.atleast_2d(base_contrib) - self.bg_contrib) + residual

    def __getstate__(self):                                  # explainers are rebuilt after loading
        state = self.__dict__.copy()
        state["_explainers"] = None
        return state
