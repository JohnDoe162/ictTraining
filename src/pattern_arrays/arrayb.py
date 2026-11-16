"""Array B labeller: 5m sweep-and-reversal episodes, outcome surface and features.

Specification: METHOD.md, "Array B". Approved by the project owner on 2026-11-16. The label carries NO threshold: each sweep records the largest close-back achieved
before each level of adverse excursion, on a fixed grid, and the grid levels enter the model as
inputs. Only the candidate floor, the decision time (sweep bar close) and the 17:00 ET cap are
fixed by the owner; everything about array quality is a continuous feature with a learned weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dol import session_bucket, session_date

NY = "America/New_York"
TICK = 0.10
EPS = 1e-9
INF_TS = np.datetime64("2262-04-11")  # far-future sentinel inside datetime64[ns] range

# Engineering constants (SPEC B.6). Changing any is a new version and new trials.
ATR_LEN = 14
ATR_NORM_BARS = 5520  # ~20 trading days of 5m bars
STRENGTH_CAP = 20
RANGE_BARS = 20
TESTS_HELD_BAND_ATR = 1.0
RVOL_BARS = 20
CAP_HOUR = 17  # 17:00 ET session cap (owner approved)

# Outcome grid (SPEC B.5), in ATR at the decision time. Declared before any run.
X_GRID = (0.10, 0.25, 0.50, 0.75, 1.00, 1.50)  # close-back
Y_GRID = (0.25, 0.50, 1.00, 2.00, 4.00)  # adverse excursion beyond the sweep extreme

EPISODE_FEATURES = [
    "sweep_depth_atr",
    "sweep_depth_ticks",
    "close_back_frac",
    "sweep_bar_range_atr",
    "left_strength",
    "prominence_atr",
    "age_bars",
    "tests_held",
    "atr_norm",
    "rvol_sweep",
    "room_to_opposite_atr",
    "level_rank_session",
    "range_pos",
    "minutes_to_cap",
    "side",
    "bucket_london",
    "bucket_ny_am",
    "bucket_ny_pm",
]
# The two grid inputs are appended per cell; the model sees the surface as part of its input.
FEATURES = EPISODE_FEATURES + ["x_atr", "y_atr"]


def resample_5m(df1m: pd.DataFrame) -> pd.DataFrame:
    """NY-clock-aligned 5m bars from 1m bars.

    A 5m bar exists if it holds >= 1 traded minute (Array A amendment 1: a missing minute on MGC
    means no trades, not bad data). A bar mixing two contracts is invalid. Usable from `close_ts`.
    """
    ny = df1m["ts_event"].dt.tz_convert(NY)
    slot = ny.dt.floor("5min")
    g = df1m.groupby(slot, sort=True)
    bars = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
            "n_min": g.size(),
            "contract_first": g["contract"].first(),
            "contract_last": g["contract"].last(),
        }
    )
    bars.index.name = "open_ny"
    bars = bars.reset_index()
    bars["close_ts"] = (bars["open_ny"] + pd.Timedelta(minutes=5)).dt.tz_convert("UTC")
    bars["valid"] = bars["contract_first"] == bars["contract_last"]
    bars["contract"] = bars["contract_last"]
    bars["session"] = session_date(bars["open_ny"])
    pc = bars["close"].shift(1)
    tr = np.maximum(
        bars["high"] - bars["low"],
        np.maximum((bars["high"] - pc).abs(), (bars["low"] - pc).abs()),
    )
    bars["atr"] = tr.rolling(ATR_LEN, min_periods=ATR_LEN).mean()
    bars["atr_med"] = bars["atr"].rolling(ATR_NORM_BARS, min_periods=1).median()
    bars["vol_med"] = bars["volume"].shift(1).rolling(RVOL_BARS, min_periods=1).median()
    return bars


@dataclass
class Candidate:
    cid: int
    side: int  # +1 pivot high, -1 pivot low
    level: float
    bar_idx: int
    conf_idx: int
    conf_ts: pd.Timestamp
    contract: str
    left_strength: int
    prominence_atr: float
    sweep_ts: pd.Timestamp | None = None  # ts_event of the 1m bar that first trades through


def find_candidates(bars: pd.DataFrame) -> list[Candidate]:
    """Candidate floor (SPEC B.3, owner-approved): strictly higher high (lower low) than the single
    5m bar on each side; ties disqualify. Confirmed at the next bar's close. The pivot triple must
    sit inside one contract. Strength is NOT filtered - it is measured (features 5-6)."""
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    atr = bars["atr"].to_numpy()
    valid = bars["valid"].to_numpy()
    contract = bars["contract"].to_numpy()
    out: list[Candidate] = []
    cid = 0
    for i in range(1, len(bars) - 1):
        if not (valid[i - 1] and valid[i] and valid[i + 1]):
            continue
        if not (contract[i - 1] == contract[i] == contract[i + 1]):
            continue
        if np.isnan(atr[i]):
            continue
        for side in (1, -1):
            if side == 1:
                is_pivot = h[i] > h[i - 1] + EPS and h[i] > h[i + 1] + EPS
                level = h[i]
                prom = (h[i] - max(h[i - 1], h[i + 1])) / atr[i]
            else:
                is_pivot = lo[i] < lo[i - 1] - EPS and lo[i] < lo[i + 1] - EPS
                level = lo[i]
                prom = (min(lo[i - 1], lo[i + 1]) - lo[i]) / atr[i]
            if not is_pivot:
                continue
            k = 0
            j = i - 1
            while j >= 0 and k < STRENGTH_CAP:
                if side == 1 and h[j] < level - EPS:
                    k += 1
                elif side == -1 and lo[j] > level + EPS:
                    k += 1
                else:
                    break
                j -= 1
            out.append(
                Candidate(
                    cid=cid,
                    side=side,
                    level=float(level),
                    bar_idx=i,
                    conf_idx=i + 1,
                    conf_ts=bars["close_ts"].iloc[i + 1],
                    contract=str(contract[i]),
                    left_strength=k,
                    prominence_atr=float(prom),
                )
            )
            cid += 1
    return out


def _utc64(s: pd.Series) -> np.ndarray:
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()


def _as64(t: pd.Timestamp) -> np.datetime64:
    return t.tz_convert("UTC").tz_localize(None).to_datetime64()


BLOCK = 512


def _first_ge(vals: np.ndarray, bmax: np.ndarray, start: int, end: int, target: float) -> int:
    """First index in [start, end) with vals[i] >= target, or -1. Block maxima skip whole blocks,
    so the answer is identical to a bar-by-bar scan but does not re-read quiet stretches."""
    i = start
    while i < end:
        blk = i // BLOCK
        blk_end = min((blk + 1) * BLOCK, end)
        if bmax[blk] >= target:
            seg = vals[i:blk_end]
            hit = np.flatnonzero(seg >= target)
            if len(hit):
                return i + int(hit[0])
        i = blk_end
    return -1


def _block_max(a: np.ndarray) -> np.ndarray:
    n = int(np.ceil(len(a) / BLOCK))
    pad = np.full(n * BLOCK, -np.inf)
    pad[: len(a)] = a
    return pad.reshape(n, BLOCK).max(axis=1)


def mark_sweeps(cands: list[Candidate], df1m: pd.DataFrame) -> None:
    """First 1m bar that trades through the level by >= 1 tick, while the level's contract is still
    the front contract. A touch is not a sweep. The contract run bounds the scan, so no episode can
    span a roll (CLAUDE.md 5.5) and no extra retirement constant is needed."""
    ts = _utc64(df1m["ts_event"])
    highs = df1m["high"].to_numpy()
    lows = df1m["low"].to_numpy()
    contract = df1m["contract"].to_numpy()
    change = np.flatnonzero(contract[1:] != contract[:-1]) + 1
    ends = np.concatenate([change, [len(contract)]])
    run_end = ends[np.searchsorted(ends, np.arange(len(contract)), side="right")]
    bmax_h = _block_max(highs)
    bmax_l = _block_max(-lows)
    for c in cands:
        j0 = int(np.searchsorted(ts, _as64(c.conf_ts)))
        if j0 >= len(ts) or contract[j0] != c.contract:
            continue
        b = int(run_end[j0])
        if c.side == 1:
            k = _first_ge(highs, bmax_h, j0, b, c.level + TICK - EPS)
        else:
            k = _first_ge(-lows, bmax_l, j0, b, -(c.level - TICK + EPS))
        if k >= 0:
            c.sweep_ts = pd.Timestamp(ts[k], tz="UTC")


def _running_max(a: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(a) if len(a) else a


def cell_cols() -> list[str]:
    """Column names of the stored outcome surface, one per (x, y) grid cell."""
    return [f"lab_{xi}_{yi}" for xi in range(len(X_GRID)) for yi in range(len(Y_GRID))]


def expand(episodes: pd.DataFrame, keep: list[str] | None = None) -> pd.DataFrame:
    """Long form: one row per (episode, x, y), with `x_atr`, `y_atr` and the 0/1 label `y`.

    Kept out of `build_rows` on purpose: 30 rows per episode over 7.8 years is millions of rows,
    and every one of them repeats the same episode features. The walk-forward expands one window
    at a time instead of holding the whole thing.
    """
    if episodes.empty:
        return pd.DataFrame(columns=(keep or []) + ["x_atr", "y_atr", "y"])
    keep = keep if keep is not None else ["cand_id", "t_dec", "cap_ts"] + EPISODE_FEATURES
    base = episodes[keep]
    parts = []
    for xi, x in enumerate(X_GRID):
        for yi, y in enumerate(Y_GRID):
            part = base.copy()
            part["x_atr"] = x
            part["y_atr"] = y
            part["y"] = episodes[f"lab_{xi}_{yi}"].to_numpy(dtype=float)
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_rows(df1m: pd.DataFrame, no_entry: set) -> tuple[pd.DataFrame, dict]:
    """Episodes with features and the outcome surface.

    Returns (episodes, counts): one row per sweep, with the surface stored as 30 `lab_<xi>_<yi>`
    columns (`expand` turns them into training rows). All gates and counts are quoted in EPISODES
    (SPEC B.10.2), never in rows.
    """
    bars = resample_5m(df1m)
    cands = find_candidates(bars)
    mark_sweeps(cands, df1m)

    ny1m = df1m["ts_event"].dt.tz_convert(NY)
    df1m = df1m.assign(_sess=session_date(ny1m), _ny_hour=ny1m.dt.hour)
    d1 = df1m[df1m["_ny_hour"] != CAP_HOUR]
    ts1 = _utc64(d1["ts_event"])
    hi1 = d1["high"].to_numpy()
    lo1 = d1["low"].to_numpy()
    sess1 = d1["_sess"].to_numpy()
    contract1 = d1["contract"].to_numpy()
    sess_start1 = np.searchsorted(sess1, np.unique(sess1), side="left")
    sess_end1 = np.searchsorted(sess1, np.unique(sess1), side="right")
    sess_keys = np.unique(sess1)

    b_close = _utc64(bars["close_ts"])
    b_closep = bars["close"].to_numpy()
    b_sess = bars["session"].to_numpy()
    b_high = bars["high"].to_numpy()
    b_low = bars["low"].to_numpy()
    b_atr = bars["atr"].to_numpy()

    counts = {
        "candidates": len(cands),
        "swept": 0,
        "episodes": 0,
        "never_swept": 0,
        "no_entry_skipped": 0,
        "decision_at_cap": 0,
        "roll_dropped": 0,
        "truncated_at_data_end": 0,
        "no_opposite": 0,
        "bad_bar": 0,
    }

    # opposite-side reference: most recent confirmed opposite pivot not yet swept at t
    cs = sorted(cands, key=lambda c: c.conf_ts)
    cs_conf = np.array([_as64(c.conf_ts) for c in cs])

    def reference_opposite(c: Candidate, t: pd.Timestamp) -> Candidate | None:
        hi = int(np.searchsorted(cs_conf, _as64(t), side="right"))
        for r in reversed(cs[:hi]):
            if r.side == -c.side and r.contract == c.contract and (r.sweep_ts is None or r.sweep_ts >= t):
                return r
        return None

    data_end = df1m["ts_event"].iloc[-1] + pd.Timedelta(minutes=1)
    eps: list[dict] = []
    for c in cands:
        if c.sweep_ts is None:
            counts["never_swept"] += 1
            continue
        counts["swept"] += 1
        # the 5m bar containing the first through-trade; decision time = its close
        si = int(np.searchsorted(b_close, _as64(c.sweep_ts), side="right"))
        if si >= len(bars):
            counts["truncated_at_data_end"] += 1
            continue
        t_dec = bars["close_ts"].iloc[si]
        sess = bars["session"].iloc[si]
        if not bars["valid"].iloc[si] or np.isnan(b_atr[si]) or np.isnan(bars["atr_med"].iloc[si]):
            counts["bad_bar"] += 1
            continue
        if sess in no_entry:
            counts["no_entry_skipped"] += 1
            continue
        atr = float(b_atr[si])
        cap = pd.Timestamp(sess).tz_convert(NY).normalize() + pd.Timedelta(hours=CAP_HOUR)
        cap_utc = cap.tz_convert("UTC")
        if cap_utc > data_end:
            counts["truncated_at_data_end"] += 1
            continue

        # 1m window: same session, bars opening at or after the decision close
        k = int(np.searchsorted(sess_keys, sess))
        if k >= len(sess_keys) or sess_keys[k] != sess:
            counts["bad_bar"] += 1
            continue
        a0, a1 = int(sess_start1[k]), int(sess_end1[k])
        j0 = a0 + int(np.searchsorted(ts1[a0:a1], _as64(t_dec)))
        if j0 >= a1:
            counts["decision_at_cap"] += 1
            continue
        if not (contract1[j0:a1] == c.contract).all() or contract1[j0] != c.contract:
            counts["roll_dropped"] += 1
            continue

        # sweep extreme S: furthest price from the first through-trade to the decision close
        s0 = a0 + int(np.searchsorted(ts1[a0:a1], _as64(c.sweep_ts)))
        if s0 >= j0:
            counts["bad_bar"] += 1
            continue
        S = float(hi1[s0:j0].max()) if c.side == 1 else float(lo1[s0:j0].min())

        ref = reference_opposite(c, t_dec)
        if ref is None:
            counts["no_opposite"] += 1
            continue

        # outcome surface -----------------------------------------------------------------
        adv = (hi1[j0:a1] - S) if c.side == 1 else (S - lo1[j0:a1])
        radv = _running_max(adv)
        m0 = int(np.searchsorted(b_close, _as64(t_dec), side="right"))
        m1 = int(np.searchsorted(b_close, _as64(cap_utc), side="right"))
        m1 = min(m1, m0 + int(np.searchsorted(b_sess[m0:], sess, side="right")))
        cb = (c.level - b_closep[m0:m1]) if c.side == 1 else (b_closep[m0:m1] - c.level)
        rcb = _running_max(cb)
        cb_close_ts = b_close[m0:m1]

        # The cell is 1 when the close-back level x is first reached at or before the first 1m bar
        # that exceeds the adverse level y. A 1m bar inside the same 5m bar as the qualifying close
        # carries an earlier timestamp, so the adverse side wins the tie (SPEC B.5.4).
        first_adv = []
        for yv in Y_GRID:
            jj = int(np.searchsorted(radv, yv * atr - EPS, side="left"))
            first_adv.append(ts1[j0 + jj] if jj < len(radv) else INF_TS)
        first_cb = []
        for xv in X_GRID:
            kk = int(np.searchsorted(rcb, xv * atr - EPS, side="left"))
            first_cb.append(cb_close_ts[kk] if kk < len(rcb) else INF_TS)
        labels = {
            f"lab_{xi}_{yi}": np.int8(tx != INF_TS and tx <= ta)
            for xi, tx in enumerate(first_cb)
            for yi, ta in enumerate(first_adv)
        }

        # features ------------------------------------------------------------------------
        sb_high, sb_low = float(b_high[si]), float(b_low[si])
        sb_close = float(b_closep[si])
        rng = sb_high - sb_low
        cbf = ((c.level - sb_close) if c.side == 1 else (sb_close - c.level)) / rng if rng > EPS else 0.0
        w0 = max(0, si - RANGE_BARS + 1)
        rh = float(np.nanmax(b_high[w0 : si + 1]))
        rl = float(np.nanmin(b_low[w0 : si + 1]))
        rr = rh - rl
        lo_i, hi_i = c.conf_idx + 1, si
        if hi_i > lo_i:
            a_b = b_atr[lo_i:hi_i]
            if c.side == 1:
                seg = b_high[lo_i:hi_i]
                near = (seg >= c.level - TESTS_HELD_BAND_ATR * a_b) & (seg < c.level + TICK - EPS)
            else:
                seg = b_low[lo_i:hi_i]
                near = (seg <= c.level + TESTS_HELD_BAND_ATR * a_b) & (seg > c.level - TICK + EPS)
            th = int(np.count_nonzero(near & ~np.isnan(a_b)))
        else:
            th = 0
        s_hi = float(hi1[a0:j0].max())
        s_lo = float(lo1[a0:j0].min())
        s_rng = s_hi - s_lo
        vol_med = float(bars["vol_med"].iloc[si])
        bucket = session_bucket(t_dec.tz_convert(NY))
        eps.append(
            {
                "cand_id": c.cid,
                "t_dec": t_dec,
                "session": sess,
                "cap_ts": cap_utc,
                "level": c.level,
                "sweep_extreme": S,
                "atr": atr,
                "mfe_back_atr": float(rcb[-1] / atr) if len(rcb) else float("nan"),
                "mae_beyond_atr": float(radv[-1] / atr) if len(radv) else float("nan"),
                "mfe_back_ticks": float(rcb[-1] / TICK) if len(rcb) else float("nan"),
                "mae_beyond_ticks": float(radv[-1] / TICK) if len(radv) else float("nan"),
                "sweep_depth_atr": (S - c.level) * c.side / atr,
                "sweep_depth_ticks": (S - c.level) * c.side / TICK,
                "close_back_frac": cbf,
                "sweep_bar_range_atr": rng / atr,
                "left_strength": c.left_strength,
                "prominence_atr": c.prominence_atr,
                "age_bars": si - c.conf_idx,
                "tests_held": th,
                "atr_norm": atr / float(bars["atr_med"].iloc[si]),
                "rvol_sweep": float(bars["volume"].iloc[si]) / vol_med if vol_med > 0 else 1.0,
                "room_to_opposite_atr": abs(S - ref.level) / atr,
                "level_rank_session": (c.level - s_lo) / s_rng if s_rng > EPS else 0.5,
                "range_pos": (sb_close - rl) / rr if rr > EPS else 0.5,
                "minutes_to_cap": (cap_utc - t_dec).total_seconds() / 60.0,
                "side": c.side,
                "bucket_london": 1.0 if bucket == "london" else 0.0,
                "bucket_ny_am": 1.0 if bucket == "ny_am" else 0.0,
                "bucket_ny_pm": 1.0 if bucket == "ny_pm" else 0.0,
                **labels,
            }
        )
        counts["episodes"] += 1

    episodes = pd.DataFrame(eps)
    if not episodes.empty:
        episodes = episodes.sort_values("t_dec").reset_index(drop=True)
    return episodes, counts
