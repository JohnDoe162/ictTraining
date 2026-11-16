"""Parity harness for Array B: does the model still work on only what TradingView can see?

The research labeller (arrayb.build_rows) uses 1-minute bars, the roll calendar and
no_entry_sessions.csv. A TradingView indicator on a 5-minute chart has none of those: it sees
5m bars of a continuous contract and resolves everything at bar close. This script rebuilds the
SAME episodes under those restrictions, runs the SAME purged walk-forward, and then fits the
model on the chart-visible features so the constants embedded in the .pine are the ones the
chart actually computes.

Run: uv run python scripts/pattern_arrays_arrayb_pine_parity.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pattern_arrays import arrayb, logit, walkforward_cond  # noqa: E402
from pattern_arrays.data import load_bars  # noqa: E402
from pattern_arrays.arrayb import (  # noqa: E402
    CAP_HOUR,
    EPS,
    NY,
    RANGE_BARS,
    STRENGTH_CAP,
    TESTS_HELD_BAND_ATR,
    TICK,
    X_GRID,
    Y_GRID,
    _block_max,
    _first_ge,
    find_candidates,
    resample_5m,
    session_bucket,
)

START = "2018-01-01"
END = "2025-10-17"
BARS = os.environ.get("MGC_BARS", "")  # path to your own 1-minute bars; see src/pattern_arrays/data.py
OUT = ROOT / "results"


def build_rows_pine(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Array B episodes using ONLY 5m bars: no 1m resolution, no roll drops, no no-entry
    calendar. Exactly the information a TradingView indicator has.

    The sweep extreme is the sweep bar's high. That is NOT an approximation: the first minute
    to trade through the level cannot precede the bar's high, so the 1m maximum from the
    through-trade to the bar's close is the bar's high either way.
    """
    bars = bars.reset_index(drop=True)
    cands = find_candidates(bars)
    hi = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    cl = bars["close"].to_numpy()
    vol = bars["volume"].to_numpy()
    atr = bars["atr"].to_numpy()
    med = bars["atr_med"].to_numpy()
    vmed = bars["vol_med"].to_numpy()
    sess = bars["session"].to_numpy()
    close_ts = bars["close_ts"]
    n = len(bars)

    sess_end = {}
    for s, g in bars.groupby("session", sort=True):
        sess_end[s] = int(g.index[-1])
    sess_start = {}
    for s, g in bars.groupby("session", sort=True):
        sess_start[s] = int(g.index[0])

    counts = {"candidates": len(cands), "episodes": 0, "never_swept": 0, "no_opposite": 0, "at_cap": 0}
    # Sweep index per candidate: first 5m bar that trades through, detected at bar close, with no
    # contract bound because a chart has none. Block maxima so a never-swept level does not drag a
    # scan across the whole history; the answer is the same as a bar-by-bar scan (tested).
    bmax_h, bmax_l = _block_max(hi), _block_max(-lo)
    sweep = {}
    for c in cands:
        if c.side == 1:
            s = _first_ge(hi, bmax_h, c.conf_idx + 1, n, c.level + TICK - EPS)
        else:
            s = _first_ge(-lo, bmax_l, c.conf_idx + 1, n, -(c.level - TICK + EPS))
        sweep[c.cid] = s
        if s < 0:
            counts["never_swept"] += 1

    by_conf = sorted(cands, key=lambda c: c.conf_idx)
    conf_idx_arr = np.array([c.conf_idx for c in by_conf])

    def reference_opposite(c, si: int):
        k = int(np.searchsorted(conf_idx_arr, si, side="right"))
        for r in reversed(by_conf[:k]):
            if r.side == -c.side and (sweep[r.cid] < 0 or sweep[r.cid] >= si):
                return r
        return None

    rows = []
    for c in cands:
        si = sweep[c.cid]
        if si < 0 or np.isnan(atr[si]) or np.isnan(med[si]):
            continue
        s = sess[si]
        end = sess_end[s]
        if si >= end:
            counts["at_cap"] += 1
            continue
        ref = reference_opposite(c, si)
        if ref is None:
            counts["no_opposite"] += 1
            continue
        a = float(atr[si])
        S = float(hi[si]) if c.side == 1 else float(lo[si])
        w = slice(si + 1, end + 1)
        adv = (hi[w] - S) if c.side == 1 else (S - lo[w])
        cb = (c.level - cl[w]) if c.side == 1 else (cl[w] - c.level)
        radv = np.maximum.accumulate(adv)
        rcb = np.maximum.accumulate(cb)
        labels = {}
        for yi, yv in enumerate(Y_GRID):
            j = int(np.searchsorted(radv, yv * a - EPS, side="left"))
            for xi, xv in enumerate(X_GRID):
                k = int(np.searchsorted(rcb, xv * a - EPS, side="left"))
                # strict: a bar that both closes back and breaches the adverse level is a loss
                labels[f"lab_{xi}_{yi}"] = np.int8(k < len(rcb) and k < j)
        rng = hi[si] - lo[si]
        cbf = ((c.level - cl[si]) if c.side == 1 else (cl[si] - c.level)) / rng if rng > EPS else 0.0
        w0 = max(0, si - RANGE_BARS + 1)
        rh, rl = float(np.nanmax(hi[w0 : si + 1])), float(np.nanmin(lo[w0 : si + 1]))
        rr = rh - rl
        lo_i, hi_i = c.conf_idx + 1, si
        if hi_i > lo_i:
            ab = atr[lo_i:hi_i]
            seg = hi[lo_i:hi_i] if c.side == 1 else lo[lo_i:hi_i]
            near = (
                (seg >= c.level - TESTS_HELD_BAND_ATR * ab) & (seg < c.level + TICK - EPS)
                if c.side == 1
                else (seg <= c.level + TESTS_HELD_BAND_ATR * ab) & (seg > c.level - TICK + EPS)
            )
            th = int(np.count_nonzero(near & ~np.isnan(ab)))
        else:
            th = 0
        a0 = sess_start[s]
        s_hi, s_lo = float(np.nanmax(hi[a0 : si + 1])), float(np.nanmin(lo[a0 : si + 1]))
        s_rng = s_hi - s_lo
        t_dec = close_ts.iloc[si]
        cap = pd.Timestamp(s).tz_convert(NY).normalize() + pd.Timedelta(hours=CAP_HOUR)
        bucket = session_bucket(t_dec.tz_convert(NY))
        rows.append(
            {
                "cand_id": c.cid,
                "t_dec": t_dec,
                "cap_ts": cap.tz_convert("UTC"),
                "sweep_depth_atr": (S - c.level) * c.side / a,
                "sweep_depth_ticks": (S - c.level) * c.side / TICK,
                "close_back_frac": cbf,
                "sweep_bar_range_atr": rng / a,
                "left_strength": c.left_strength,
                "prominence_atr": c.prominence_atr,
                "age_bars": si - c.conf_idx,
                "tests_held": th,
                "atr_norm": a / float(med[si]),
                "rvol_sweep": float(vol[si]) / float(vmed[si]) if vmed[si] > 0 else 1.0,
                "room_to_opposite_atr": abs(S - ref.level) / a,
                "level_rank_session": (c.level - s_lo) / s_rng if s_rng > EPS else 0.5,
                "range_pos": (cl[si] - rl) / rr if rr > EPS else 0.5,
                "minutes_to_cap": (cap.tz_convert("UTC") - t_dec).total_seconds() / 60.0,
                "side": c.side,
                "bucket_london": 1.0 if bucket == "london" else 0.0,
                "bucket_ny_am": 1.0 if bucket == "ny_am" else 0.0,
                "bucket_ny_pm": 1.0 if bucket == "ny_pm" else 0.0,
                **labels,
            }
        )
        counts["episodes"] += 1
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("t_dec").reset_index(drop=True)
    return out, counts


def main() -> int:
    if not BARS:
        raise SystemExit("Set MGC_BARS to a Parquet or CSV of 1-minute bars (see src/pattern_arrays/data.py).")
    df1m = load_bars(BARS, START, END)
    bars = resample_5m(df1m)
    eps, counts = build_rows_pine(bars)
    print("chart-visible episodes:", json.dumps(counts, indent=2))
    eps.to_parquet(OUT / "arrayB_episodes_pine.parquet", index=False)

    res = walkforward_cond.run(eps, START, END)
    pooled = res["pooled"]
    print("\nchart-visible pooled OOS:", json.dumps(pooled, indent=2))
    research = json.loads((OUT / "arrayB_summary.json").read_text())["pooled"]
    parity = {
        "research": {k: research[k] for k in ("episodes", "mean_cell_lift", "brier_skill", "auc_within_cells")},
        "chart_visible": {k: pooled[k] for k in ("episodes", "mean_cell_lift", "brier_skill", "auc_within_cells")},
        "windows_positive": pooled["positive_window_share"],
        "monotone_share": pooled["monotone_share"],
    }
    (OUT / "arrayB_pine_parity.json").write_text(json.dumps(parity, indent=2))
    print("\nparity:", json.dumps(parity, indent=2))

    # --- the model the chart will carry: fitted on the chart-visible features, full sample ---
    long = arrayb.expand(eps)
    x = long[arrayb.FEATURES].to_numpy(float)
    y = long["y"].to_numpy()
    model = logit.fit(x, y)
    payload = {
        "features": arrayb.FEATURES,
        "intercept": float(model.coef[0]),
        "coef": [float(v) for v in model.coef[1:]],
        "mean": [float(v) for v in model.mean],
        "std": [float(v) for v in model.std],
        "episodes": int(len(eps)),
        "rows": int(len(long)),
        "base_rate": float(y.mean()),
        "grid": {"x": list(X_GRID), "y": list(Y_GRID)},
        "fitted_on": "chart-visible (5m-only) features, full sample 2018-01-01..2025-10-17",
        "strength_cap": STRENGTH_CAP,
    }
    (OUT / "arrayB_model_pine.json").write_text(json.dumps(payload, indent=2))
    print(f"\nmodel for the chart: {len(eps)} episodes, base rate {y.mean():.4f}")
    for n_, w_ in sorted(zip(arrayb.FEATURES, payload["coef"]), key=lambda t: -abs(t[1])):
        print(f"  {n_:24s} {w_:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
