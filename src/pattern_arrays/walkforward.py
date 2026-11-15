"""Stage 1 walk-forward for Array A: 18-month train / 6-month test, purge and embargo.

A training row is used only if its episode RESOLVED before the test window starts, so no label
can leak across the cut. Verdict thresholds are NOT applied here: they are the
owner's and were not pre-registered before this run (reported, SPEC QA5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import logit
from .dol import FEATURES

TRAIN_MONTHS = 18
TEST_MONTHS = 6


def windows(data_start: str, data_end: str) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """(train_start, test_start, test_end) triples; the last partial test window is included
    (configs/validation.yaml include_partial_last_oos_window)."""
    start = pd.Timestamp(data_start, tz="UTC")
    end = pd.Timestamp(data_end, tz="UTC")
    out = []
    test_start = start + pd.DateOffset(months=TRAIN_MONTHS)
    while test_start < end:
        test_end = min(test_start + pd.DateOffset(months=TEST_MONTHS), end)
        out.append((test_start - pd.DateOffset(months=TRAIN_MONTHS), test_start, test_end))
        test_start = test_start + pd.DateOffset(months=TEST_MONTHS)
    return out


def _metrics(y: np.ndarray, p: np.ndarray, p_ref: float) -> dict:
    """Scores vs the constant reference model that always predicts the TRAIN base rate."""
    ref = np.full_like(p, p_ref)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "log_loss": logit.log_loss(y, p),
        "log_loss_ref": logit.log_loss(y, ref),
        "brier": logit.brier(y, p),
        "brier_ref": logit.brier(y, ref),
        "brier_skill": 1.0 - logit.brier(y, p) / logit.brier(y, ref),
        "auc": _auc(y, p),
    }


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC with tie-averaged ranks."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def decile_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(p, kind="mergesort")
    bins = np.array_split(order, n_bins)
    rows = []
    for i, b in enumerate(bins):
        rows.append(
            {
                "decile": i + 1,  # 10 = highest scores
                "n": len(b),
                "mean_score": float(p[b].mean()),
                "event_rate": float(y[b].mean()),
            }
        )
    return pd.DataFrame(rows)


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        rows.append(
            {
                "bin_lo": edges[i],
                "bin_hi": edges[i + 1],
                "n": int(m.sum()),
                "mean_pred": float(p[m].mean()),
                "obs_rate": float(y[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def run(rows: pd.DataFrame, data_start: str, data_end: str) -> dict:
    """Train/test over every window. Returns per-window metrics, pooled OOS scores and tables."""
    df = rows[rows["label"].isin(["event", "invalidated"])].copy()
    df["y"] = (df["label"] == "event").astype(float)
    per_window = []
    oos_parts = []
    for train_start, test_start, test_end in windows(data_start, data_end):
        tr = df[(df["t_dec"] >= train_start) & (df["t_dec"] < test_start) & (df["resolution_ts"] < test_start)]
        te = df[(df["t_dec"] >= test_start) & (df["t_dec"] < test_end)]
        if len(tr) < 200 or len(te) < 50:
            per_window.append({"test_start": test_start, "skipped": True, "n_train": len(tr), "n_test": len(te)})
            continue
        model = logit.fit(tr[FEATURES].to_numpy(float), tr["y"].to_numpy())
        p = model.predict(te[FEATURES].to_numpy(float))
        y = te["y"].to_numpy()
        p_ref = float(tr["y"].mean())
        m = _metrics(y, p, p_ref)
        dec = decile_table(y, p)
        m.update(
            {
                "test_start": test_start,
                "test_end": test_end,
                "n_train": len(tr),
                "train_base_rate": p_ref,
                "top_decile_rate": float(dec["event_rate"].iloc[-1]),
                "top_decile_lift": float(dec["event_rate"].iloc[-1] / y.mean()) if y.mean() > 0 else float("nan"),
                "skipped": False,
                "coef": model.coef.tolist(),
            }
        )
        per_window.append(m)
        part = te[["t_dec", "y", "age_sessions"]].copy()
        part["p"] = p
        part["p_ref"] = p_ref
        oos_parts.append(part)
    oos = pd.concat(oos_parts, ignore_index=True)
    y, p = oos["y"].to_numpy(), oos["p"].to_numpy()
    pooled = _metrics(y, p, float(oos["p_ref"].mean()))
    first = oos[oos["age_sessions"] == 0]
    pooled_first = _metrics(first["y"].to_numpy(), first["p"].to_numpy(), float(first["p_ref"].mean()))
    return {
        "per_window": per_window,
        "pooled": pooled,
        "pooled_first_rows": pooled_first,
        "decile": decile_table(y, p),
        "decile_first_rows": decile_table(first["y"].to_numpy(), first["p"].to_numpy()),
        "reliability": reliability_table(y, p),
        "oos": oos,
    }
