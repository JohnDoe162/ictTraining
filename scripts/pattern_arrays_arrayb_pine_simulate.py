"""Transliterate the Array B Pine state machine in Python and produce its expected on-chart table.

tradingview/sweep_model_mgc.pine keeps running state: a candidate list, per-episode levels, a
hit-rate tally. The parity harness works in batch, so it cannot catch a bug in that state machine.
This script walks the bars one at a time in exactly the Pine phase order and prints the table the
indicator should show, so the owner can compare the chart against a number computed here.

The table is IN-SAMPLE (the weights were fitted on this data). It confirms the indicator computes
the model faithfully; it is not fresh evidence.

Run: uv run python scripts/pattern_arrays_arrayb_pine_simulate.py [--x 0.50] [--y 1.00]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pattern_arrays.data import load_bars  # noqa: E402
from pattern_arrays.arrayb import (  # noqa: E402
    ATR_NORM_BARS,
    CAP_HOUR,
    NY,
    RANGE_BARS,
    RVOL_BARS,
    STRENGTH_CAP,
    TESTS_HELD_BAND_ATR,
    TICK,
    resample_5m,
)

START = "2018-01-01"
END = "2025-10-17"
BARS = os.environ.get("MGC_BARS", "")  # path to your own 1-minute bars; see src/pattern_arrays/data.py
OUT = ROOT / "results"
MAX_TRACKED = 800  # the .pine input of the same name


@dataclass
class Cand:
    lvl: float
    side: int
    conf: int
    left: int
    prom: float
    tests: int = 0
    alive: bool = True
    ep_open: bool = False
    score: float = float("nan")
    sx: float = float("nan")
    atr_dec: float = float("nan")
    dec: int = -1
    _f: dict = field(default_factory=dict)


def main() -> int:
    if not BARS:
        raise SystemExit("Set MGC_BARS to a Parquet or CSV of 1-minute bars (see src/pattern_arrays/data.py).")
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=0.50)
    ap.add_argument("--y", type=float, default=1.00)
    args = ap.parse_args()

    m = json.loads((OUT / "arrayB_model_pine.json").read_text())
    w = np.array(m["coef"])
    mu = np.array(m["mean"])
    sd = np.array(m["std"])
    b0 = m["intercept"]

    bars = resample_5m(load_bars(BARS, START, END)).reset_index(drop=True)
    hi = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    cl = bars["close"].to_numpy()
    vol = bars["volume"].to_numpy().astype(float)
    atr = bars["atr"].to_numpy()
    sess = bars["session"].to_numpy()
    ny = bars["open_ny"]
    close_ts = bars["close_ts"]
    n = len(bars)
    eps = TICK * 0.01

    atr_med = pd.Series(atr).rolling(ATR_NORM_BARS, min_periods=1).median().to_numpy()
    vol_med = pd.Series(vol).shift(1).rolling(RVOL_BARS, min_periods=1).median().to_numpy()
    rng_hi = pd.Series(hi).rolling(RANGE_BARS, min_periods=1).max().to_numpy()
    rng_lo = pd.Series(lo).rolling(RANGE_BARS, min_periods=1).min().to_numpy()

    cands: list[Cand] = []
    band_n = [0] * 5
    band_hit = [0] * 5
    n_neither = 0
    sess_hi = sess_lo = float("nan")

    def through(i: int, lvl: float, side: int) -> bool:
        return hi[i] >= lvl + TICK - eps if side == 1 else lo[i] <= lvl - TICK + eps

    def band(s: float) -> int:
        return 0 if s < 0.2 else 1 if s < 0.4 else 2 if s < 0.6 else 3 if s < 0.8 else 4

    for i in range(n):
        new_sess = i == 0 or sess[i] != sess[i - 1]
        # PHASE 0 - the session cap
        if new_sess:
            for c in cands:
                if c.ep_open:
                    c.ep_open = False
                    n_neither += 1
            sess_hi, sess_lo = hi[i], lo[i]
        else:
            sess_hi, sess_lo = max(sess_hi, hi[i]), min(sess_lo, lo[i])

        # PHASE 1 - resolve running episodes; the heat is checked first (worst case)
        for c in cands:
            if c.ep_open and c.dec < i:
                heat = c.sx + args.y * c.atr_dec * c.side
                took = hi[i] >= heat - eps if c.side == 1 else lo[i] <= heat + eps
                back_lvl = c.lvl - args.x * c.atr_dec * c.side
                came = cl[i] <= back_lvl + eps if c.side == 1 else cl[i] >= back_lvl - eps
                if took or came:
                    b = band(c.score)
                    band_n[b] += 1
                    if not took:
                        band_hit[b] += 1
                    c.ep_open = False

        # PHASE 2 - approaches that came close but did not trade through
        if not math.isnan(atr[i]):
            for c in cands:
                if c.alive and c.conf < i:
                    if c.side == 1 and hi[i] >= c.lvl - TESTS_HELD_BAND_ATR * atr[i] and hi[i] < c.lvl + TICK - eps:
                        c.tests += 1
                    elif c.side == -1 and lo[i] <= c.lvl + TESTS_HELD_BAND_ATR * atr[i] and lo[i] > c.lvl - TICK + eps:
                        c.tests += 1

        # PHASE 3 - confirm a new pivot (it sits one bar back)
        if i >= 2 and not math.isnan(atr[i - 1]) and i > STRENGTH_CAP + 3:
            for side in (1, -1):
                if side == 1:
                    ok = hi[i - 1] > hi[i - 2] + eps and hi[i - 1] > hi[i] + eps
                    lvl = hi[i - 1]
                    nb = max(hi[i - 2], hi[i])
                else:
                    ok = lo[i - 1] < lo[i - 2] - eps and lo[i - 1] < lo[i] - eps
                    lvl = lo[i - 1]
                    nb = min(lo[i - 2], lo[i])
                if not ok:
                    continue
                k = 0
                for j in range(2, STRENGTH_CAP + 2):
                    if i - j < 0 or not (hi[i - j] < lvl - eps if side == 1 else lo[i - j] > lvl + eps):
                        break
                    k += 1
                a1 = atr[i - 1]
                prom = 0.0 if math.isnan(a1) or a1 <= 0 else (lvl - nb if side == 1 else nb - lvl) / a1
                cands.append(Cand(lvl=float(lvl), side=side, conf=i, left=k, prom=float(prom)))
        cands = [c for c in cands if c.alive or c.ep_open]
        while len(cands) > MAX_TRACKED:
            cands.pop(0)

        # PHASE 4 - the decision: a level swept on THIS bar is scored at this close
        if not (math.isnan(atr[i]) or atr[i] <= 0 or math.isnan(atr_med[i]) or math.isnan(vol_med[i])):
            ref_hi = next((c.lvl for c in reversed(cands) if c.alive and c.side == 1), None)
            ref_lo = next((c.lvl for c in reversed(cands) if c.alive and c.side == -1), None)
            cap = (pd.Timestamp(sess[i]).tz_convert(NY).normalize() + pd.Timedelta(hours=CAP_HOUR)).tz_convert("UTC")
            mins = (cap - close_ts.iloc[i]).total_seconds() / 60.0
            h = ny.iloc[i].hour + ny.iloc[i].minute / 60.0
            for c in list(cands):
                if c.alive and c.conf < i and through(i, c.lvl, c.side):
                    c.alive = False
                    rl = ref_lo if c.side == 1 else ref_hi
                    if rl is None or mins <= 0:
                        continue
                    sx = hi[i] if c.side == 1 else lo[i]
                    rng = hi[i] - lo[i]
                    depth = (sx - c.lvl) * c.side
                    cbf = ((c.lvl - cl[i]) * c.side / rng) if rng > 0 else 0.0
                    lrank = (c.lvl - sess_lo) / (sess_hi - sess_lo) if sess_hi > sess_lo else 0.5
                    rpos = (cl[i] - rng_lo[i]) / (rng_hi[i] - rng_lo[i]) if rng_hi[i] > rng_lo[i] else 0.5
                    f = np.array(
                        [
                            depth / atr[i],
                            depth / TICK,
                            cbf,
                            rng / atr[i],
                            c.left,
                            c.prom,
                            i - c.conf,
                            c.tests,
                            atr[i] / atr_med[i],
                            vol[i] / vol_med[i] if vol_med[i] > 0 else 1.0,
                            abs(sx - rl) / atr[i],
                            lrank,
                            rpos,
                            mins,
                            c.side,
                            1.0 if 3.0 <= h < 9.5 else 0.0,
                            1.0 if 9.5 <= h < 12.0 else 0.0,
                            1.0 if 12.0 <= h < 17.0 else 0.0,
                            args.x,
                            args.y,
                        ]
                    )
                    z = b0 + float(np.dot(w, (f - mu) / sd))
                    c.score = 1.0 / (1.0 + math.exp(-z))
                    c.sx = float(sx)
                    c.atr_dec = float(atr[i])
                    c.dec = i
                    c.ep_open = True

    names = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    rows = [
        {
            "band": names[b],
            "episodes": band_n[b],
            "came_back": band_hit[b],
            "hit_rate": round(100.0 * band_hit[b] / band_n[b], 1) if band_n[b] else None,
        }
        for b in range(5)
    ]
    tot_n, tot_h = sum(band_n), sum(band_hit)
    rows.append(
        {
            "band": "All (base rate)",
            "episodes": tot_n,
            "came_back": tot_h,
            "hit_rate": round(100.0 * tot_h / tot_n, 1) if tot_n else None,
        }
    )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "arrayB_tradingview_expected_table.csv", index=False)
    print(f"question: come back {args.x:g} ATR before {args.y:g} ATR of heat")
    print(tab.to_string(index=False))
    print(f"unresolved at the cap (excluded, as in training): {n_neither}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
