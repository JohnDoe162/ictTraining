"""Stage 1 walk-forward for a conditional (outcome-surface) array model — Array B.

SPEC: SPEC.md Part B sections B.7-B.8. Differences from Array A's `walkforward.py`, which is left
untouched:

1. One row per (episode, x, y) cell, so EPISODES are the unit of independence. Every gate, window
   size and count below is quoted in episodes, never in rows.
2. Purge and embargo use the episode's SESSION CAP - the latest moment any of its cells can
   resolve - so all 30 cells of an episode always fall on the same side of a cut.
3. The baseline is the PER-CELL base rate of the training window. A model that had only learned
   the geometry of the grid would already beat a single pooled rate; it cannot beat this one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import logit
from .arrayb import FEATURES, X_GRID, Y_GRID, expand
from .walkforward import _auc, decile_table, reliability_table, windows

MIN_TRAIN_EPISODES = 200
MIN_TEST_EPISODES = 50
CELL = ["x_atr", "y_atr"]


def _cell_base_rates(df: pd.DataFrame) -> pd.Series:
    return df.groupby(CELL)["y"].mean()


def _ref(df: pd.DataFrame, rates: pd.Series) -> np.ndarray:
    """Per-cell baseline prediction for each row, from the TRAINING window's cell base rates."""
    idx = pd.MultiIndex.from_arrays([df["x_atr"], df["y_atr"]])
    return rates.reindex(idx).to_numpy(dtype=float)


def _skill(y: np.ndarray, p: np.ndarray, ref: np.ndarray) -> dict:
    return {
        "n_rows": int(len(y)),
        "base_rate": float(y.mean()),
        "log_loss": logit.log_loss(y, p),
        "log_loss_ref": logit.log_loss(y, ref),
        "brier": logit.brier(y, p),
        "brier_ref": logit.brier(y, ref),
        "brier_skill": 1.0 - logit.brier(y, p) / logit.brier(y, ref),
        "log_loss_skill": 1.0 - logit.log_loss(y, p) / logit.log_loss(y, ref),
    }


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell OOS behaviour: base rate printed beside every score (SPEC B.7)."""
    rows = []
    for (x, y_lvl), g in df.groupby(CELL):
        yy, pp = g["y"].to_numpy(), g["p"].to_numpy()
        base = float(yy.mean())
        if yy.min() == yy.max():  # a degenerate cell cannot be ranked
            top, lift, auc = float("nan"), float("nan"), float("nan")
        else:
            dec = decile_table(yy, pp)
            top = float(dec["event_rate"].iloc[-1])
            lift = top / base if base > 0 else float("nan")
            auc = _auc(yy, pp)
        rows.append(
            {
                "x_atr": x,
                "y_atr": y_lvl,
                "n_episodes": int(len(g)),
                "base_rate": base,
                "mean_pred": float(pp.mean()),
                "top_decile_rate": top,
                "top_decile_lift": lift,
                "auc": auc,
                "brier_skill": 1.0 - logit.brier(yy, pp) / logit.brier(yy, np.full_like(pp, base)),
            }
        )
    return pd.DataFrame(rows).sort_values(CELL).reset_index(drop=True)


def monotone_share(df: pd.DataFrame) -> float:
    """Share of OOS episodes whose PREDICTED surface falls with x and rises with y (SPEC V4).

    Every episode contributes exactly one row per grid cell, so sorting by (episode, x, y) and
    reshaping gives one (x, y) matrix per episode without a per-episode groupby.
    """
    nx, ny = len(X_GRID), len(Y_GRID)
    g = df.sort_values(["cand_id", "t_dec", "x_atr", "y_atr"])
    n = len(g) // (nx * ny)
    if n == 0 or len(g) != n * nx * ny:
        raise ValueError(f"expected {nx * ny} cells per episode, got {len(g)} rows for {n} episodes")
    a = g["p"].to_numpy(dtype=float).reshape(n, nx, ny)
    ok = (np.diff(a, axis=1) <= 1e-12).all(axis=(1, 2)) & (np.diff(a, axis=2) >= -1e-12).all(axis=(1, 2))
    return float(ok.mean())


def run(episodes: pd.DataFrame, data_start: str, data_end: str) -> dict:
    """`episodes` is one row per sweep with the stored surface; each window is expanded to
    (episode, x, y) rows only while it is being used."""
    df = episodes
    per_window = []
    oos_parts = []
    for train_start, test_start, test_end in windows(data_start, data_end):
        tr_ep = df[(df["t_dec"] >= train_start) & (df["t_dec"] < test_start) & (df["cap_ts"] < test_start)]
        te_ep = df[(df["t_dec"] >= test_start) & (df["t_dec"] < test_end)]
        n_tr_ep = len(tr_ep)
        n_te_ep = len(te_ep)
        if n_tr_ep < MIN_TRAIN_EPISODES or n_te_ep < MIN_TEST_EPISODES:
            per_window.append(
                {"test_start": test_start, "skipped": True, "train_episodes": n_tr_ep, "test_episodes": n_te_ep}
            )
            continue
        tr, te = expand(tr_ep), expand(te_ep)
        model = logit.fit(tr[FEATURES].to_numpy(float), tr["y"].to_numpy())
        p = model.predict(te[FEATURES].to_numpy(float))
        y = te["y"].to_numpy()
        rates = _cell_base_rates(tr)
        ref = _ref(te, rates)
        m = _skill(y, p, ref)
        part = te[["cand_id", "t_dec"] + CELL + ["y"]].copy()
        part["p"] = p
        part["p_ref"] = ref
        part["test_start"] = test_start
        ct = cell_table(part)
        m.update(
            {
                "test_start": test_start,
                "test_end": test_end,
                "train_episodes": n_tr_ep,
                "test_episodes": n_te_ep,
                "mean_cell_lift": float(np.nansum(ct["top_decile_lift"] * ct["n_episodes"]) / ct["n_episodes"].sum()),
                "monotone_share": monotone_share(part),
                "skipped": False,
                "coef": model.coef.tolist(),
            }
        )
        per_window.append(m)
        oos_parts.append(part)
    if not oos_parts:
        return {"per_window": per_window, "pooled": None, "oos": pd.DataFrame()}
    oos = pd.concat(oos_parts, ignore_index=True)
    y, p, ref = oos["y"].to_numpy(), oos["p"].to_numpy(), oos["p_ref"].to_numpy()
    ct = cell_table(oos)
    pooled = _skill(y, p, ref)
    pooled.update(
        {
            "episodes": int(oos["cand_id"].nunique()),
            "mean_cell_lift": float(np.nansum(ct["top_decile_lift"] * ct["n_episodes"]) / ct["n_episodes"].sum()),
            "monotone_share": monotone_share(oos),
            "auc_within_cells": float(np.nansum(ct["auc"] * ct["n_episodes"]) / ct["n_episodes"].sum()),
        }
    )
    done = [w for w in per_window if not w.get("skipped")]
    pooled["positive_window_share"] = float(np.mean([w["brier_skill"] > 0 for w in done])) if done else float("nan")
    return {
        "per_window": per_window,
        "pooled": pooled,
        "cells": ct,
        "reliability": reliability_table(y, p),
        "oos": oos,
        "grid": {"x": list(X_GRID), "y": list(Y_GRID)},
    }


def verdict(pooled: dict, thresholds: dict) -> dict:
    """Apply the PRE-REGISTERED Stage 1 tests (SPEC B.8, configs/validation.yaml)."""
    v = {
        "V1_ranking": pooled["mean_cell_lift"] >= thresholds["top_decile_lift"],
        "V2_brier_skill": pooled["brier_skill"] > thresholds["min_brier_skill"],
        "V3_consistency": pooled["positive_window_share"] >= thresholds["min_positive_window_share"],
        "V4_monotone": pooled["monotone_share"] >= thresholds["min_monotone_share"],
    }
    v["PASS"] = all(bool(x) for x in v.values())
    return v
