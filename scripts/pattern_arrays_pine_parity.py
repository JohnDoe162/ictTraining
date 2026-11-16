"""Parity harness: does the Array A model still work when it can only see what TradingView sees?

The research pipeline (dol.build_rows) uses 1-minute bars, the roll calendar and
no_entry_sessions.csv. A TradingView indicator has none of those: it sees 1h bars of a
continuous contract and resolves levels at bar close. This script rebuilds the SAME dataset
under those restrictions and runs the SAME walk-forward, so the two can be compared honestly.

Run: uv run python scripts/pattern_arrays_pine_parity.py
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

from pattern_arrays import dol, walkforward  # noqa: E402
from pattern_arrays.data import load_bars  # noqa: E402
from pattern_arrays.dol import (  # noqa: E402
    EPS,
    RANGE_BARS,
    RETIRE_SESSIONS,
    TESTS_HELD_BAND_ATR,
    TICK,
    find_candidates,
    resample_1h,
    session_bucket,
)

START = "2018-01-01"
END = "2025-10-17"
OUT = ROOT / "results"
BARS = os.environ.get("MGC_BARS", "")  # path to your own 1-minute bars; see src/pattern_arrays/data.py


def build_rows_pine(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Array A rows using ONLY 1h bars: no 1m resolution, no roll drops, no no-entry calendar.
    This is exactly the information a TradingView indicator has."""
    bars = bars.reset_index(drop=True)
    cands = find_candidates(bars)
    hi = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    atr = bars["atr"].to_numpy()
    med = bars["atr_med"].to_numpy()
    close = bars["close"].to_numpy()
    valid = bars["valid"].to_numpy()
    sess = bars["session"].to_numpy()
    close_ts = bars["close_ts"].to_numpy()

    def through(idx: int, level: float, side: int) -> bool:
        return hi[idx] >= level + TICK - EPS if side == 1 else lo[idx] <= level - TICK + EPS

    # death index: first bar AFTER confirmation that trades through the level (bar-close detection)
    death = {}
    for c in cands:
        d = None
        for b in range(c.conf_idx, min(c.conf_idx + 24 * 45, len(bars))):
            if through(b, c.level, c.side):
                d = b
                break
        death[c.cid] = d

    # session -> ordered bar indices
    sess_idx: dict = {}
    for i in range(len(bars)):
        if valid[i]:
            sess_idx.setdefault(sess[i], []).append(i)
    sessions = sorted(sess_idx)
    spos = {s: i for i, s in enumerate(sessions)}

    cs = sorted(cands, key=lambda c: c.conf_idx)
    conf_idx_arr = np.array([c.conf_idx for c in cs])

    def ref_opposite(c, dec_idx: int):
        j = int(np.searchsorted(conf_idx_arr, dec_idx, side="right"))
        for r in reversed(cs[:j]):
            if r.side == -c.side and (death[r.cid] is None or death[r.cid] > dec_idx):
                return r
        return None

    rows, counts = [], {"candidates": len(cands), "rows": 0, "event": 0, "invalidated": 0, "neither": 0}
    for c in cands:
        start_pos = spos.get(sess[c.conf_idx])
        if start_pos is None:
            continue
        for age in range(RETIRE_SESSIONS):
            pos = start_pos + age
            if pos >= len(sessions):
                break
            idxs = sess_idx[sessions[pos]]
            dec_idx = c.conf_idx if age == 0 else idxs[0]
            if death[c.cid] is not None and death[c.cid] <= dec_idx:
                break
            if np.isnan(atr[dec_idx]) or np.isnan(med[dec_idx]):
                continue
            ref = ref_opposite(c, dec_idx)
            if ref is None:
                continue
            rest = [b for b in idxs if b > dec_idx]
            if not rest:
                continue
            label = "neither"
            for b in rest:
                ev, iv = through(b, c.level, c.side), through(b, ref.level, ref.side)
                if iv:  # same-bar tie -> worst case
                    label = "invalidated"
                    break
                if ev:
                    label = "event"
                    break
            a = float(atr[dec_idx])
            cl = float(close[dec_idx])
            dist = (c.level - cl) if c.side == 1 else (cl - c.level)
            w0 = max(0, dec_idx - RANGE_BARS + 1)
            rh, rl = float(np.nanmax(hi[w0 : dec_idx + 1])), float(np.nanmin(lo[w0 : dec_idx + 1]))
            rng = rh - rl
            th = 0
            for b in range(c.conf_idx + 1, dec_idx + 1):
                if np.isnan(atr[b]):
                    continue
                if c.side == 1 and c.level - TESTS_HELD_BAND_ATR * atr[b] <= hi[b] < c.level + TICK - EPS:
                    th += 1
                elif c.side == -1 and c.level - TICK + EPS < lo[b] <= c.level + TESTS_HELD_BAND_ATR * atr[b]:
                    th += 1
            t_dec = pd.Timestamp(close_ts[dec_idx])
            bucket = session_bucket(t_dec.tz_convert(dol.NY))
            rows.append(
                {
                    "cand_id": c.cid,
                    "t_dec": t_dec,
                    "label": label,
                    "resolution_ts": t_dec,  # resolution within the same session; not used for purging here
                    "dist_to_level_atr": dist / a,
                    "dist_to_level_ticks": dist / TICK,
                    "left_strength": c.left_strength,
                    "prominence_atr": c.prominence_atr,
                    "age_sessions": age,
                    "tests_held": th,
                    "atr_norm": a / float(med[dec_idx]),
                    "side": c.side,
                    "range_pos": (cl - rl) / rng if rng > EPS else 0.5,
                    "bucket_london": 1.0 if bucket == "london" else 0.0,
                    "bucket_ny_am": 1.0 if bucket == "ny_am" else 0.0,
                    "bucket_ny_pm": 1.0 if bucket == "ny_pm" else 0.0,
                }
            )
            counts["rows"] += 1
            counts[label] += 1
            if label in ("event", "invalidated"):
                break
    out = pd.DataFrame(rows)
    out["resolution_ts"] = out["t_dec"] + pd.Timedelta(hours=24)  # conservative purge margin
    return out, counts


def main() -> int:
    if not BARS:
        raise SystemExit("Set MGC_BARS to a Parquet or CSV of 1-minute bars (see src/pattern_arrays/data.py).")
    df1m = load_bars(BARS, START, END)
    bars = resample_1h(df1m)
    rows_p, counts_p = build_rows_pine(bars)
    print("pine-equivalent counts:", counts_p)
    res_p = walkforward.run(rows_p, START, END)
    rows_r = pd.read_parquet(OUT / "arrayA_rows.parquet")
    res_r = walkforward.run(rows_r, START, END)

    def line(tag, r):
        p = r["pooled"]
        d = r["decile"]
        return {
            "pipeline": tag,
            "oos_rows": p["n"],
            "base_rate": round(p["base_rate"], 4),
            "auc": round(p["auc"], 4),
            "brier_skill": round(p["brier_skill"], 4),
            "log_loss": round(p["log_loss"], 4),
            "top_decile": round(float(d["event_rate"].iloc[-1]), 4),
            "bottom_decile": round(float(d["event_rate"].iloc[0]), 4),
        }

    cmp = pd.DataFrame(
        [line("research (1m, rolls, no-entry cal)", res_r), line("TradingView-equivalent (1h only)", res_p)]
    )
    cmp.to_csv(OUT / "arrayA_pine_parity.csv", index=False)
    print(cmp.to_string(index=False))
    per = pd.DataFrame(
        [{k: v for k, v in w.items() if k != "coef"} for w in res_p["per_window"] if not w.get("skipped")]
    )
    print("\nTradingView-equivalent per-window top-decile rate:")
    print(per[["test_start", "n", "base_rate", "auc", "brier_skill", "top_decile_rate"]].to_string(index=False))
    (OUT / "arrayA_pine_parity.json").write_text(json.dumps(cmp.to_dict("records"), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
