"""Anomaly detection - paper Section III-F, equation (5).

An Isolation Forest scores 21 standardized features (value, first difference and
30-min rolling standard deviation of each sensor).  A sample is flagged when its score
exceeds tau, and the anomaly evidence a_t is the fraction of flagged samples in the
last 30 min.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score


class AnomalyDetector:
    def __init__(self, n_estimators=300, max_samples=4096, seed=0):
        self.forest = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                      contamination="auto", random_state=seed, n_jobs=4)
        self.mu = self.sd = None
        self.tau = None

    def _z(self, A):
        return (A - self.mu) / self.sd

    def fit(self, A_train):
        self.mu, self.sd = A_train.mean(0), A_train.std(0) + 1e-9
        self.forest.fit(self._z(A_train))
        return self

    def score(self, A):
        """Higher = more anomalous (the paper's s(x))."""
        return -self.forest.score_samples(self._z(np.atleast_2d(A)))

    def tune_threshold(self, A_val, is_anomaly_val):
        """Pick tau on validation data by maximizing the F1-score (Section IV-B)."""
        sc = self.score(A_val)
        candidates = np.quantile(sc, np.linspace(0.5, 0.995, 120))
        f1 = [f1_score(is_anomaly_val, sc >= q) for q in candidates]
        self.tau = float(candidates[int(np.argmax(f1))])
        return self.tau

    def flags(self, A):
        return self.score(A) >= self.tau

    @staticmethod
    def evidence(flag_history):
        """Equation (5): a_t = fraction of the last 6 samples (30 min) that were flagged."""
        recent = np.asarray(flag_history, dtype=float)[-6:]
        return float(recent.mean()) if len(recent) else 0.0
