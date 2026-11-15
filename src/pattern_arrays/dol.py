"""Array A labeller: 1h draw-on-liquidity candidates, episodes, labels and features.

Specification: METHOD.md in this repository.
The label is fixed by the owner (event = trade-through >= 1 tick; invalidation = reference
opposite pivot traded through; session cap 17:00 ET; worst case on same-1m-bar ties).
Everything about array quality is a continuous feature with a learned weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NY = "America/New_York"
TICK = 0.10
EPS = 1e-9
# Engineering constants (SPEC section 10): changing any is a new version.
ATR_LEN = 14
ATR_NORM_BARS = 460  # ~20 trading days of 1h bars
STRENGTH_CAP = 20
RANGE_BARS = 20
TESTS_HELD_BAND_ATR = 1.0
RETIRE_SESSIONS = 30
CAP_HOUR = 17  # 17:00 ET session cap (owner approved)

FEATURES = [
    "dist_to_level_atr",
    "dist_to_level_ticks",
    "left_strength",
    "prominence_atr",
    "age_sessions",
    "tests_held",
    "atr_norm",
    "side",
    "range_pos",
    "bucket_london",
    "bucket_ny_am",
    "bucket_ny_pm",  # Asia is the one-hot base
]


def session_date(ts_ny: pd.Series) -> pd.Series:
    """Trading day of each NY timestamp: bars at/after 18:00 belong to the next calendar day."""
    return (ts_ny + pd.Timedelta(hours=6)).dt.normalize()


def session_bucket(ts_ny: pd.Timestamp) -> str:
    """Bucket of a decision time by NY wall clock (SPEC feature 8)."""
    h = ts_ny.hour + ts_ny.minute / 60.0
    if 3.0 <= h < 9.5:
        return "london"
    if 9.5 <= h < 12.0:
        return "ny_am"
    if 12.0 <= h < 17.0:
        return "ny_pm"
    return "asia"  # 17:00(gap)/18:00-03:00


def resample_1h(df1m: pd.DataFrame) -> pd.DataFrame:
    """NY-clock-aligned 1h bars from 1m bars.

    A 1h bar exists if it holds >= 1 traded minute (missing minutes are hours without trades,
    normal for MGC 2018-2020; coverage is reported by the audit). A bar mixing two contracts
    (a roll inside the hour) is marked invalid. Bars are usable only from `close_ts`.
    """
    ny = df1m["ts_event"].dt.tz_convert(NY)
    hour = ny.dt.floor("h")
    g = df1m.groupby(hour, sort=True)
    bars = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "n_min": g.size(),
            "contract_first": g["contract"].first(),
            "contract_last": g["contract"].last(),
        }
    )
    bars.index.name = "open_ny"
    bars = bars.reset_index()
    bars["close_ts"] = (bars["open_ny"] + pd.Timedelta(hours=1)).dt.tz_convert("UTC")
    bars["valid"] = bars["contract_first"] == bars["contract_last"]
    bars["contract"] = bars["contract_last"]
    bars["session"] = session_date(bars["open_ny"])
    # ATR(14): previous-existing-bar close true range, simple mean.
    pc = bars["close"].shift(1)
    tr = np.maximum(
        bars["high"] - bars["low"],
        np.maximum((bars["high"] - pc).abs(), (bars["low"] - pc).abs()),
    )
    bars["atr"] = tr.rolling(ATR_LEN, min_periods=ATR_LEN).mean()
    # normalisation denominator: median of whatever ATR history exists (>= 1 value)
    bars["atr_med"] = bars["atr"].rolling(ATR_NORM_BARS, min_periods=1).median()
    return bars


@dataclass
class Candidate:
    cid: int
    side: int  # +1 pivot high, -1 pivot low
    level: float
    bar_idx: int  # pivot bar position in the bars frame
    conf_idx: int  # confirmation bar position (bar_idx + 1 in existing sequence)
    conf_ts: pd.Timestamp  # usable from this close (UTC)
    conf_session: pd.Timestamp
    contract: str
    left_strength: int
    prominence_atr: float
    death_ts: pd.Timestamp | None = None  # close of the 1m bar that trades through


def find_candidates(bars: pd.DataFrame) -> list[Candidate]:
    """Weakest-pivot floor (owner-approved): strictly higher high (lower low) than the single
    existing bar on each side; ties disqualify. Confirmed at the next bar's close. The pivot
    triple must sit inside one contract. Strength is NOT filtered - it is measured."""
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
            # left_strength: consecutive prior existing bars the pivot extreme exceeds.
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
                    conf_session=bars["session"].iloc[i + 1],
                    contract=str(contract[i]),
                    left_strength=k,
                    prominence_atr=float(prom),
                )
            )
            cid += 1
    return out


def _utc64(s: pd.Series) -> np.ndarray:
    """tz-aware UTC series -> naive datetime64[ns] numpy array (UTC wall time)."""
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()


def _as64(t: pd.Timestamp) -> np.datetime64:
    return t.tz_convert("UTC").tz_localize(None).to_datetime64()


def _first_through(highs: np.ndarray, lows: np.ndarray, level: float, side: int) -> int:
    """Index of the first 1m bar that trades through `level` on `side` (+1 high, -1 low); -1 if none."""
    hit = highs >= level + TICK - EPS if side == 1 else lows <= level - TICK + EPS
    idx = int(np.argmax(hit))
    return idx if hit[idx] else -1


def mark_deaths(cands: list[Candidate], df1m: pd.DataFrame) -> None:
    """A level dies permanently at the close of the first 1m bar that trades through it,
    episode or not (SPEC 5.5). Checked on 1m bars so the label is bar-boundary independent."""
    ts = _utc64(df1m["ts_event"])
    highs = df1m["high"].to_numpy()
    lows = df1m["low"].to_numpy()
    for c in cands:
        j0 = int(np.searchsorted(ts, _as64(c.conf_ts)))
        if j0 >= len(ts):
            continue
        # Bounded scan: 60 calendar days covers the 30-trading-day retirement with margin;
        # a death beyond it can never affect an episode or a reference choice.
        j1 = int(np.searchsorted(ts, _as64(c.conf_ts + pd.Timedelta(days=60))))
        k = _first_through(highs[j0:j1], lows[j0:j1], c.level, c.side)
        if k >= 0:
            c.death_ts = pd.Timestamp(ts[j0 + k], tz="UTC") + pd.Timedelta(minutes=1)


def build_rows(df1m: pd.DataFrame, no_entry: set) -> tuple[pd.DataFrame, dict]:
    """Decision rows with features and labels for Array A. Returns (rows, audit counts)."""
    bars = resample_1h(df1m)
    cands = find_candidates(bars)
    mark_deaths(cands, df1m)

    ny1m = df1m["ts_event"].dt.tz_convert(NY)
    sess1m = session_date(ny1m)
    df1m = df1m.assign(_sess=sess1m, _ny_hour=ny1m.dt.hour)
    # bars inside the session cap only (before 17:00 ET; there is no 17:xx trading hour)
    sess_groups = {s: g for s, g in df1m[df1m["_ny_hour"] != CAP_HOUR].groupby("_sess", sort=True)}
    sessions = sorted(sess_groups)
    sess_pos = {s: i for i, s in enumerate(sessions)}
    # first valid 1h bar close per session, and bar positions per session
    bars_by_session = {s: g for s, g in bars[bars["valid"]].groupby("session", sort=True)}

    counts = {
        "candidates": len(cands),
        "rows": 0,
        "event": 0,
        "invalidated": 0,
        "neither": 0,
        "no_opposite": 0,
        "died_outside_episode": 0,
        "roll_dropped": 0,
        "retired": 0,
        "no_entry_skipped": 0,
        "decision_at_cap": 0,
        "unresolved_at_data_end": 0,
    }

    # Alive-candidate lookup for the invalidation reference: sort by confirmation time.
    cands_sorted = sorted(cands, key=lambda c: c.conf_ts)
    conf_ts_arr = np.array([_as64(c.conf_ts) for c in cands_sorted])

    def reference_opposite(c: Candidate, t_dec: pd.Timestamp) -> Candidate | None:
        """Most recently confirmed opposite-side candidate alive at t_dec, same contract."""
        hi = int(np.searchsorted(conf_ts_arr, _as64(t_dec), side="right"))
        for r in reversed(cands_sorted[:hi]):
            if r.side == -c.side and r.contract == c.contract and (r.death_ts is None or r.death_ts > t_dec):
                return r
        return None

    data_end = df1m["ts_event"].iloc[-1] + pd.Timedelta(minutes=1)
    rows: list[dict] = []
    bars_high = bars["high"].to_numpy()
    bars_low = bars["low"].to_numpy()
    bars_atr = bars["atr"].to_numpy()

    for c in cands:
        start_pos = sess_pos.get(c.conf_session)
        if start_pos is None:
            continue
        for age in range(RETIRE_SESSIONS + 1):
            pos = start_pos + age
            if pos >= len(sessions):
                break
            if age == RETIRE_SESSIONS:
                counts["retired"] += 1
                break
            sess = sessions[pos]
            g = sess_groups[sess]
            if age == 0:
                t_dec = c.conf_ts
                dec_bar_idx = c.conf_idx
            else:
                sb = bars_by_session.get(sess)
                if sb is None or sb.empty:
                    continue
                t_dec = sb["close_ts"].iloc[0]
                dec_bar_idx = sb.index[0]
            # roll: candidate dies when its contract leaves the data
            sess_contract = g["contract"].iloc[-1]
            if sess_contract != c.contract or (age > 0 and g["contract"].iloc[0] != c.contract):
                counts["roll_dropped"] += 1
                break
            if c.death_ts is not None and c.death_ts <= t_dec:
                if age > 0:
                    counts["died_outside_episode"] += 1
                break
            if sess in no_entry:
                counts["no_entry_skipped"] += 1
                continue
            dec_bar = bars.iloc[dec_bar_idx]
            if not dec_bar["valid"] or np.isnan(dec_bar["atr"]) or np.isnan(dec_bar["atr_med"]):
                continue
            # episode 1m slice: bars opening at/after the decision close, same session
            ts_g = _utc64(g["ts_event"])
            j0 = int(np.searchsorted(ts_g, _as64(t_dec)))
            if j0 >= len(ts_g):
                counts["decision_at_cap"] += 1
                continue
            ref = reference_opposite(c, t_dec)
            if ref is None:
                counts["no_opposite"] += 1
                continue
            hi_g = g["high"].to_numpy()[j0:]
            lo_g = g["low"].to_numpy()[j0:]
            ev = _first_through(hi_g, lo_g, c.level, c.side)
            iv = _first_through(hi_g, lo_g, ref.level, ref.side)
            if ev < 0 and iv < 0:
                if ts_g[-1] + np.timedelta64(1, "m") >= data_end.to_datetime64():
                    counts["unresolved_at_data_end"] += 1
                    break
                label, res_idx = "neither", len(ts_g) - 1 - j0
            elif iv >= 0 and (ev < 0 or iv <= ev):  # same-1m-bar tie -> worst case
                label, res_idx = "invalidated", iv
            else:
                label, res_idx = "event", ev
            res_ts = pd.Timestamp(ts_g[j0 + res_idx], tz="UTC") + pd.Timedelta(minutes=1)

            # features (bars closed at or before t_dec only)
            atr = float(dec_bar["atr"])
            close = float(dec_bar["close"])
            dist = (c.level - close) if c.side == 1 else (close - c.level)
            w0 = max(0, dec_bar_idx - RANGE_BARS + 1)
            rh = float(np.nanmax(bars_high[w0 : dec_bar_idx + 1]))
            rl = float(np.nanmin(bars_low[w0 : dec_bar_idx + 1]))
            rng = rh - rl
            range_pos = (close - rl) / rng if rng > EPS else 0.5
            th = 0
            for b in range(c.conf_idx + 1, dec_bar_idx + 1):
                a_b = bars_atr[b]
                if np.isnan(a_b):
                    continue
                if c.side == 1:
                    if bars_high[b] >= c.level - TESTS_HELD_BAND_ATR * a_b and bars_high[b] < c.level + TICK - EPS:
                        th += 1
                else:
                    if bars_low[b] <= c.level + TESTS_HELD_BAND_ATR * a_b and bars_low[b] > c.level - TICK + EPS:
                        th += 1
            bucket = session_bucket(t_dec.tz_convert(NY))
            rows.append(
                {
                    "cand_id": c.cid,
                    "t_dec": t_dec,
                    "session": sess,
                    "label": label,
                    "resolution_ts": res_ts,
                    "level": c.level,
                    "dist_to_level_atr": dist / atr,
                    "dist_to_level_ticks": dist / TICK,
                    "left_strength": c.left_strength,
                    "prominence_atr": c.prominence_atr,
                    "age_sessions": age,
                    "tests_held": th,
                    "atr_norm": atr / float(dec_bar["atr_med"]),
                    "side": c.side,
                    "range_pos": range_pos,
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
    return out, counts


def load_no_entry(path) -> set:
    df = pd.read_csv(path)
    return set(pd.to_datetime(df["trading_day"]).dt.tz_localize(NY).dt.normalize())
