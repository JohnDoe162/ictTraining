"""L2-regularised logistic regression (numpy Newton-Raphson). No third-party ML dependency.

Loss = sum of log losses + (1/(2C)) * ||w||^2 over the non-intercept weights (C = 1.0, a model
constant, SPEC section 10). Features are standardised on the TRAINING window only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_REG = 1.0
MAX_ITER = 200
TOL = 1e-10


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class LogitModel:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray  # intercept first, then standardised-feature weights

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (x - self.mean) / self.std
        z = self.coef[0] + xs @ self.coef[1:]
        return _sigmoid(z)


def fit(x: np.ndarray, y: np.ndarray, c_reg: float = C_REG) -> LogitModel:
    """y in {0,1}. Newton-Raphson on the penalised log likelihood; intercept unpenalised."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0
    xs = np.column_stack([np.ones(len(x)), (x - mean) / std])
    n, d = xs.shape
    w = np.zeros(d)
    lam = np.zeros(d)
    lam[1:] = 1.0 / c_reg
    for _ in range(MAX_ITER):
        p = _sigmoid(xs @ w)
        grad = xs.T @ (p - y) + lam * w
        s = np.clip(p * (1 - p), 1e-12, None)
        hess = (xs * s[:, None]).T @ xs + np.diag(lam)
        step = np.linalg.solve(hess, grad)
        w -= step
        if float(np.max(np.abs(step))) < TOL:
            break
    return LogitModel(mean=mean, std=std, coef=w)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))
