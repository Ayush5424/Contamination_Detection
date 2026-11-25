"""Risk fusion and classification - paper Section III-G, equation (6).

    R_t = max(W_t, W^f_t) + lambda * a_t

The level rises as soon as either the present water (W_t) or the water expected within
the next hour (W^f_t) is unacceptable.  Validation selected lambda = 0 in the paper,
so anomalies are reported on a separate notification channel.
"""
from .cwqi import LEVELS, risk_level

LAMBDA = 0.0


def risk_score(current_cwqi, forecast_cwqi, anomaly_evidence=0.0, lam=LAMBDA):
    return max(float(current_cwqi), float(forecast_cwqi)) + lam * float(anomaly_evidence)


def classify(score):
    return LEVELS[int(risk_level(score))]
