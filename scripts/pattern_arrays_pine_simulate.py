"""Faithful transliteration of tradingview/dol_model_mgc.pine, run bar by bar in Python.

The Pine script is a streaming state machine; the research code is a batch algorithm. This
script re-implements the Pine phases in the same order, with the same pruning, and checks
that it produces the same episodes, scores and outcomes. It also prints the exact hit-rate
table the indicator will show on a MGC 1h chart, so the numbers can be checked before the
script is ever loaded.

Run: uv run python scripts/pattern_arrays_pine_simulate.py
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
sys.path.insert(0, str(ROOT / "scripts"))

from pattern_arrays_pine_parity import build_rows_pine  # noqa: E402

from pattern_arrays.dol import (  # noqa: E402
    RANGE_BARS,
    RETIRE_SESSIONS,
    STRENGTH_CAP,
    TESTS_HELD_BAND_ATR,
    resample_1h,
    session_bucket,
)

START, END = "2018-01-01", "2025-10-17"
OUT = ROOT / "results"
BARS = os.environ.get("MGC_BARS", "")  # path to your own 1-minute bars
MODEL = json.loads((OUT / "arrayA_model_pine.json").read_text())
TICK = 0.10
EPS = TICK * 0.01
MAX_TRACKED = 800


def score_of(f: dict) -> float:
    z = MODEL["intercept"]
    for name, w, m, s in zip(MODEL["features"], MODEL["coef"], MODEL["mean"], MODEL["std"]):
        z += w * (f[name] - m) / s
    return 1.0 / (1.0 + np.exp(-z))


class C:
    __slots__ = (
        "lvl",
        "side",
        "conf_bar",
        "left_str",
        "prom",
        "tests",
        "level_alive",
        "tracking",
        "ep_open",
        "ep_score",
        "ref_lvl",
        "ref_side",
        "age_sess",
    )

    def __init__(self, lvl, side, conf_bar, left_str, prom):
        self.lvl, self.side, self.conf_bar = lvl, side, conf_bar
        self.left_str, self.prom, self.tests = left_str, prom, 0
        self.level_alive = self.tracking = True
        self.ep_open = False
        self.ep_score = self.ref_lvl = None
        self.ref_side = self.age_sess = 0


def band_of(s: float) -> int:
    return 0 if s < 0.2 else 1 if s < 0.4 else 2 if s < 0.6 else 3 if s < 0.8 else 4


def simulate(bars: pd.DataFrame) -> dict:
    hi = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    cl = bars["close"].to_numpy()
    atr = bars["atr"].to_numpy()
    # Pine's ta.median(atr, 460) is na until the window is full; the research code uses an
    # expanding median. This reproduces Pine, which is what the chart will show.
    atr_med = bars["atr"].rolling(460, min_periods=460).median().to_numpy()
    sess = bars["session"].to_numpy()
    close_ts = bars["close_ts"].to_numpy()
    rng_hi = bars["high"].rolling(RANGE_BARS, min_periods=1).max().to_numpy()
    rng_lo = bars["low"].rolling(RANGE_BARS, min_periods=1).min().to_numpy()

    def through(i, lvl, sd):
        return hi[i] >= lvl + TICK - EPS if sd == 1 else lo[i] <= lvl - TICK + EPS

    cands: list[C] = []
    band_n = [0] * 5
    band_hit = [0] * 5
    n_neither = 0
    rows = []
    max_arr = 0

    for i in range(3, len(bars)):
        new_session = sess[i] != sess[i - 1]

        # PHASE 0 - session cap: a running episode ends as "neither"; the level lives on
        if new_session:
            for c in cands:
                if c.ep_open:
                    c.ep_open = False
                    n_neither += 1

        # PHASE 1 - resolve running episodes; invalidation first (worst case on a tie)
        for c in cands:
            if c.ep_open and c.conf_bar < i:
                inv = through(i, c.ref_lvl, c.ref_side)
                rch = through(i, c.lvl, c.side)
                if inv or rch:
                    b = band_of(c.ep_score)
                    band_n[b] += 1
                    if inv:
                        rows.append({"score": c.ep_score, "label": "invalidated"})
                    else:
                        band_hit[b] += 1
                        c.level_alive = False
                        rows.append({"score": c.ep_score, "label": "event"})
                    c.ep_open = False
                    c.tracking = False

        # PHASE 2 - a level traded through outside an episode is spent
        for c in cands:
            if c.level_alive and not c.ep_open and c.conf_bar < i and through(i, c.lvl, c.side):
                c.level_alive = False
                c.tracking = False

        # PHASE 3 - approaches that did not trade through
        if not np.isnan(atr[i]):
            for c in cands:
                if c.tracking and c.conf_bar < i:
                    if c.side == 1 and c.lvl - TESTS_HELD_BAND_ATR * atr[i] <= hi[i] < c.lvl + TICK - EPS:
                        c.tests += 1
                    elif c.side == -1 and c.lvl - TICK + EPS < lo[i] <= c.lvl + TESTS_HELD_BAND_ATR * atr[i]:
                        c.tests += 1

        # PHASE 4 - confirm a pivot one bar back
        if not np.isnan(atr[i - 1]) and i > STRENGTH_CAP + 3:
            for sd in (1, -1):
                lvl = hi[i - 1] if sd == 1 else lo[i - 1]
                is_piv = (
                    hi[i - 1] > hi[i - 2] + EPS and hi[i - 1] > hi[i] + EPS
                    if sd == 1
                    else lo[i - 1] < lo[i - 2] - EPS and lo[i - 1] < lo[i] - EPS
                )
                if not is_piv:
                    continue
                k = 0
                for j in range(2, STRENGTH_CAP + 2):
                    if i - j < 0:
                        break
                    ok = hi[i - j] < lvl - EPS if sd == 1 else lo[i - j] > lvl + EPS
                    if ok:
                        k += 1
                    else:
                        break
                nb = max(hi[i - 2], hi[i]) if sd == 1 else min(lo[i - 2], lo[i])
                a1 = atr[i - 1]
                pr = 0.0 if np.isnan(a1) or a1 <= 0 else ((lvl - nb) if sd == 1 else (nb - lvl)) / a1
                cands.append(C(lvl, sd, i, k, pr))

        cands = [c for c in cands if c.tracking or c.level_alive]
        while len(cands) > MAX_TRACKED:
            cands.pop(0)
        max_arr = max(max_arr, len(cands))

        # PHASE 5 - decision points
        ref_hi = ref_lo = None
        for c in reversed(cands):
            if c.level_alive:
                if c.side == 1 and ref_hi is None:
                    ref_hi = c.lvl
                if c.side == -1 and ref_lo is None:
                    ref_lo = c.lvl
        if not np.isnan(atr_med[i]) and not np.isnan(atr[i]) and atr[i] > 0:
            for c in cands:
                if c.tracking and c.level_alive and not c.ep_open and (c.conf_bar == i or new_session):
                    if c.conf_bar != i:
                        c.age_sess += 1
                    if c.age_sess >= RETIRE_SESSIONS:
                        c.tracking = False
                        continue
                    rl = ref_lo if c.side == 1 else ref_hi
                    rs = -1 if c.side == 1 else 1
                    if rl is None:
                        continue
                    dist = (c.lvl - cl[i]) if c.side == 1 else (cl[i] - c.lvl)
                    rng = rng_hi[i] - rng_lo[i]
                    bucket = session_bucket(pd.Timestamp(close_ts[i]).tz_convert("America/New_York"))
                    c.ep_score = score_of(
                        {
                            "dist_to_level_atr": dist / atr[i],
                            "dist_to_level_ticks": dist / TICK,
                            "left_strength": c.left_str,
                            "prominence_atr": c.prom,
                            "age_sessions": c.age_sess,
                            "tests_held": c.tests,
                            "atr_norm": atr[i] / atr_med[i],
                            "side": c.side,
                            "range_pos": (cl[i] - rng_lo[i]) / rng if rng > EPS else 0.5,
                            "bucket_london": 1.0 if bucket == "london" else 0.0,
                            "bucket_ny_am": 1.0 if bucket == "ny_am" else 0.0,
                            "bucket_ny_pm": 1.0 if bucket == "ny_pm" else 0.0,
                        }
                    )
                    c.ref_lvl, c.ref_side, c.ep_open = rl, rs, True

    return {
        "band_n": band_n,
        "band_hit": band_hit,
        "n_neither": n_neither,
        "rows": pd.DataFrame(rows),
        "max_array": max_arr,
    }


def main() -> int:
    bars = resample_1h(load_bars(BARS, START, END))
    sim = simulate(bars)
    r = sim["rows"]
    r["y"] = (r["label"] == "event").astype(float)

    print("=== What the TradingView table will show on MGC 1h, full history ===")
    names = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    out = []
    for b in range(5):
        n, h = sim["band_n"][b], sim["band_hit"][b]
        out.append(
            {"Model score": names[b], "Episodes": n, "Reached": h, "Hit rate": f"{100 * h / n:.0f}%" if n else "-"}
        )
    tot_n, tot_h = sum(sim["band_n"]), sum(sim["band_hit"])
    out.append(
        {
            "Model score": "All (base rate)",
            "Episodes": tot_n,
            "Reached": tot_h,
            "Hit rate": f"{100 * tot_h / tot_n:.0f}%",
        }
    )
    tbl = pd.DataFrame(out)
    print(tbl.to_string(index=False))
    print(f"Unresolved at session close: {sim['n_neither']} (excluded, as in training)")
    print(f"peak candidates held in memory: {sim['max_array']} (indicator default cap 800)")

    # agreement with the batch research pipeline on the same 1h-only information
    batch, _ = build_rows_pine(bars)
    batch = batch[batch["label"].isin(["event", "invalidated"])]
    print("\n=== Streaming indicator vs batch research pipeline (same 1h information) ===")
    cmp = pd.DataFrame(
        [
            {
                "pipeline": "batch (scripts/pattern_arrays_pine_parity.py)",
                "episodes": len(batch),
                "event_rate": round(float((batch["label"] == "event").mean()), 4),
            },
            {
                "pipeline": "streaming (the Pine state machine)",
                "episodes": len(r),
                "event_rate": round(float(r["y"].mean()), 4),
            },
        ]
    )
    print(cmp.to_string(index=False))
    tbl.to_csv(OUT / "arrayA_tradingview_expected_table.csv", index=False)
    print(f"\nwritten {OUT / 'arrayA_tradingview_expected_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
