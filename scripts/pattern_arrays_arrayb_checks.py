"""Array B Stage 1 cross-checks. Diagnostics only: NOTHING is fitted or selected here, so these
add no trials. Run after scripts/pattern_arrays_arrayb_stage1.py.

Run:  uv run python scripts/pattern_arrays_arrayb_checks.py

1. Correlated episodes: ~100 sweeps happen in a trading day and overlap the same price moves, so
   162,099 OOS episodes are not 162,099 independent pieces of evidence. The metrics are recomputed
   on one episode per session - the least correlated subset available.
2. Triviality: how much of the ranking is carried by ONE raw feature, and how much by "time left
   in the session". Both are raw measurements, scored directly, with no model fitted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pattern_arrays import walkforward_cond as wfc  # noqa: E402
from pattern_arrays.dol import session_date  # noqa: E402
from pattern_arrays.walkforward import _auc  # noqa: E402

OUT = ROOT / "results"
SOLO = ["close_back_frac", "minutes_to_cap", "sweep_depth_atr"]


def weighted(ct: pd.DataFrame, col: str) -> float:
    return float(np.nansum(ct[col] * ct["n_episodes"]) / ct["n_episodes"].sum())


def brier_skill(df: pd.DataFrame) -> float:
    return float(1 - ((df["p"] - df["y"]) ** 2).mean() / ((df["p_ref"] - df["y"]) ** 2).mean())


def main() -> int:
    oos = pd.read_parquet(OUT / "arrayB_oos_scores.parquet")
    eps = pd.read_parquet(OUT / "arrayB_episodes.parquet")
    feats = eps[["cand_id", "t_dec"] + SOLO].copy()
    feats["sess"] = session_date(feats["t_dec"].dt.tz_convert("America/New_York"))

    first = set(feats.sort_values("t_dec").groupby("sess")["cand_id"].first())
    sub = oos[oos["cand_id"].isin(first)]
    ct_all, ct_sub = wfc.cell_table(oos), wfc.cell_table(sub)

    m = oos.merge(feats[["cand_id"] + SOLO], on="cand_id", how="left")
    rows = []
    for (x, y), g in m.groupby(["x_atr", "y_atr"]):
        yy = g["y"].to_numpy()
        r = {"x_atr": x, "y_atr": y, "n": len(g), "auc_model": _auc(yy, g["p"].to_numpy())}
        for f in SOLO:
            r[f"auc_{f}"] = _auc(yy, g[f].to_numpy())
        rows.append(r)
    solo = pd.DataFrame(rows)
    solo.to_csv(OUT / "arrayB_check_single_features.csv", index=False)

    out = {
        "correlation_cross_check": {
            "all_episodes": {
                "episodes": int(oos["cand_id"].nunique()),
                "mean_cell_lift": weighted(ct_all, "top_decile_lift"),
                "auc_within_cells": weighted(ct_all, "auc"),
                "brier_skill": brier_skill(oos),
            },
            "first_episode_per_session": {
                "episodes": int(sub["cand_id"].nunique()),
                "mean_cell_lift": weighted(ct_sub, "top_decile_lift"),
                "auc_within_cells": weighted(ct_sub, "auc"),
                "brier_skill": brier_skill(sub),
            },
        },
        "ranking_power_auc": {
            "model": float(np.average(solo["auc_model"], weights=solo["n"])),
            **{f: float(np.average(solo[f"auc_{f}"], weights=solo["n"])) for f in SOLO},
        },
    }
    (OUT / "arrayB_checks.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nper cell:")
    print(solo.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
