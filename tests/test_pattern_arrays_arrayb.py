"""Known-answer and lookahead tests for Array B (5m sweep and reversal).

Specification: METHOD.md, "Array B".

THE ARITHMETIC USED BY EVERY TEST BELOW (worked out by hand, not read off the code)
-----------------------------------------------------------------------------------
A "base" 5m bar is five identical 1m bars with high 101.0, low 100.0, close 100.5.
  -> 5m bar: high 101.0, low 100.0, close 100.5. Previous close is also 100.5, so
     true range = max(101.0-100.0, |101.0-100.5|, |100.0-100.5|) = max(1.0, 0.5, 0.5) = 1.0.
  A run of base bars therefore gives ATR(14) = 1.0 exactly.

The pivot bar lifts one minute to high 102.0 (low 100.0, close 100.5):
  -> level = 102.0; its neighbours' highs are 101.0, so it is a strict pivot high (SPEC B.3).
  -> its true range is max(2.0, |102.0-100.5|, |100.0-100.5|) = 2.0.
  15 base bars follow, so the pivot's true range has left the trailing 14-bar ATR window
  before the sweep bar. ATR over those 14 bars is 1.0 again.

The sweep bar trades through 102.0 by >= 1 tick (0.10), i.e. reaches 102.1 or higher.
  Its minutes are: minute 1 high 102.5 low 100.5 close 102.4, minutes 2-5 high 102.4 low 102.0
  close 102.2.  -> 5m bar high 102.5, low 100.5, close 102.2.
  Sweep extreme S = 102.5 (highest 1m high from the through-trade to the decision close).
  True range = max(102.5-100.5, |102.5-100.5|, |100.5-100.5|) = 2.0.
  ATR at the sweep bar = (13 x 1.0 + 2.0) / 14 = 15/14 = 1.0714285714...

GRID LEVELS IN PRICE AT THAT ATR (SPEC B.5: x = close-back, y = adverse beyond S):
  x: 0.10 -> 0.1071   0.25 -> 0.2679   0.50 -> 0.5357
     0.75 -> 0.8036   1.00 -> 1.0714   1.50 -> 1.6071
     so a close at or below 102.0 - 1.6071 = 100.3929 reaches EVERY x level.
  y: 0.25 -> S+0.2679 = 102.7679     0.50 -> S+0.5357 = 103.0357
     1.00 -> S+1.0714 = 103.5714     2.00 -> S+2.1429 = 104.6429
     4.00 -> S+4.2857 = 106.7857
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pattern_arrays import arrayb

NY = "America/New_York"
BASE = [(101.0, 100.0, 100.5)] * 5
PIVOT = [(102.0, 100.0, 100.5)] + [(101.0, 100.0, 100.5)] * 4
# An opposite-side pivot must exist for `room_to_opposite_atr` (SPEC B.6 feature 11); a session of
# identical bars has none, so one pivot low at 99.0 is placed 17 bars before the sweep bar - far
# enough back that its true range (2.0) has left the trailing 14-bar ATR window.
LOWPIV = [(101.0, 99.0, 100.5)] + [(101.0, 100.0, 100.5)] * 4
SWEEP = [(102.5, 100.5, 102.4)] + [(102.4, 102.0, 102.2)] * 4
ATR_SWEEP = 15 / 14


def frame(slots, start_utc="2024-05-01 13:00", contracts=None) -> pd.DataFrame:
    """1m bars from a list of 5m slots; each slot is five (high, low, close) triples."""
    rows = []
    t0 = pd.Timestamp(start_utc, tz="UTC")
    n = 0
    for slot in slots:
        for h, lo, c in slot:
            rows.append(
                {
                    "ts_event": t0 + pd.Timedelta(minutes=n),
                    "open": c,
                    "high": h,
                    "low": lo,
                    "close": c,
                    "volume": 10,
                    "contract": "MGCM4",
                    "roll": False,
                }
            )
            n += 1
    df = pd.DataFrame(rows)
    if contracts is not None:
        df["contract"] = contracts(df)
    return df


def session_09_to_17(middle, pad=BASE):
    """A full 09:00-17:00 New York session (96 five-minute slots), `middle` placed after the
    warm-up. 13:00 UTC = 09:00 EDT on 2024-05-01."""
    slots = [BASE] * 15 + [LOWPIV] + list(middle)
    slots += [pad] * (96 - len(slots))
    return slots[:96]


def build(slots, **kw):
    df = frame(slots, **kw)
    eps, counts = arrayb.build_rows(df, set())
    return eps, arrayb.expand(eps), counts


def episode(episodes, level=102.0):
    e = episodes[np.isclose(episodes["level"], level)]
    assert len(e) == 1, f"expected exactly one episode at {level}, got {len(e)}"
    return e.iloc[0]


def surface(cells, cand_id):
    c = cells[cells["cand_id"] == cand_id]
    return {(round(x, 2), round(y, 2)): v for x, y, v in zip(c["x_atr"], c["y_atr"], c["y"])}


# ---------------------------------------------------------------------------------------------
# 1. Plain close-back: price closes far back inside and never runs beyond the sweep extreme.
#    Post-decision bars close at 100.0, i.e. 2.00 below the level = 1.87 ATR, which clears every x
#    level (the largest is 1.6071). Highs are 101.0, far below S + 0.25 ATR = 102.7679, so no y
#    level is ever reached. => ALL 30 CELLS ARE 1.
# ---------------------------------------------------------------------------------------------
BACK = [(101.0, 100.0, 100.0)] * 5


def test_plain_close_back_fills_the_whole_surface():
    eps, cells, counts = build(session_09_to_17([PIVOT] + [BASE] * 15 + [SWEEP] + [BACK] * 4, pad=BACK))
    e = episode(eps)
    assert e["sweep_extreme"] == pytest.approx(102.5)
    assert e["atr"] == pytest.approx(ATR_SWEEP)
    s = surface(cells, e["cand_id"])
    assert len(s) == 30
    assert all(v == 1.0 for v in s.values())
    # raw outcomes: best close-back 2.00 price = 1.8667 ATR; adverse never positive.
    assert e["mfe_back_atr"] == pytest.approx(2.0 / ATR_SWEEP)
    assert e["mfe_back_ticks"] == pytest.approx(20.0)
    assert e["mae_beyond_atr"] < 0
    assert counts["episodes"] >= 1


# ---------------------------------------------------------------------------------------------
# 2. Plain adverse run: price leaves to 107.0, which is past S + 4.00 ATR = 106.7857, and never
#    closes back inside. Every y level is breached before any close-back. => ALL 30 CELLS ARE 0.
# ---------------------------------------------------------------------------------------------
AWAY = [(107.0, 106.5, 107.0)] * 5


def test_plain_adverse_run_empties_the_whole_surface():
    eps, cells, _ = build(session_09_to_17([PIVOT] + [BASE] * 15 + [SWEEP] + [AWAY] * 4, pad=AWAY))
    e = episode(eps)
    s = surface(cells, e["cand_id"])
    assert len(s) == 30
    assert all(v == 0.0 for v in s.values())
    assert e["mae_beyond_atr"] == pytest.approx((107.0 - 102.5) / ATR_SWEEP)


# ---------------------------------------------------------------------------------------------
# 3. WORST CASE, the owner's rule (SPEC B.5.4). Two different same-bar situations, both scored
#    against the trader:
#    (a) EARLIER IN THE BAR: minute 1 trades to 102.8 - past S + 0.25 ATR = 102.7679 but short of
#        S + 0.50 ATR = 103.0357 - and the bar then closes at 100.0 (a close-back of 2.00 = 1.87
#        ATR, clearing every x level). The adverse print genuinely came first, so y = 0.25 is 0.
#    (b) THE TRUE TIE: the 102.8 print happens in the LAST minute of the bar, the very minute whose
#        close is the qualifying close. Nothing in 1-minute data can say which came first, and no
#        1-second data is bought, so the tie goes against the trader: y = 0.25 is still 0.
#    In both, every y >= 0.50 cell is 1, because 102.8 never reached those levels.
# ---------------------------------------------------------------------------------------------
TIE_EARLY = [(102.8, 100.0, 100.0)] + [(101.0, 100.0, 100.0)] * 4
TIE_LAST_MINUTE = [(101.0, 100.0, 100.0)] * 4 + [(102.8, 100.0, 100.0)]


@pytest.mark.parametrize("tie_bar", [TIE_EARLY, TIE_LAST_MINUTE], ids=["earlier_in_bar", "same_minute"])
def test_same_bar_tie_goes_to_the_adverse_side(tie_bar):
    eps, cells, _ = build(session_09_to_17([PIVOT] + [BASE] * 15 + [SWEEP] + [tie_bar] + [BACK] * 3, pad=BACK))
    e = episode(eps)
    s = surface(cells, e["cand_id"])
    for (x, y), v in s.items():
        assert v == (0.0 if y == 0.25 else 1.0), (x, y, v)


# ---------------------------------------------------------------------------------------------
# 4. Adverse breach BETWEEN two 5m closes. The first post-decision bar does nothing (high 102.4,
#    close 102.2 - no close-back, no adverse). The second bar spikes to 103.1 in minute 1: past
#    S + 0.50 ATR = 103.0357 but short of S + 1.00 ATR = 103.5714, then closes at 100.0.
#    So y = 0.25 and y = 0.50 are dead before the close-back; y >= 1.00 are not. The later
#    qualifying close does NOT rescue the smaller y cells.
# ---------------------------------------------------------------------------------------------
QUIET = [(102.4, 102.0, 102.2)] * 5
SPIKE = [(103.1, 100.0, 100.0)] + [(101.0, 100.0, 100.0)] * 4


def test_adverse_between_closes_kills_only_the_smaller_y_cells():
    eps, cells, _ = build(session_09_to_17([PIVOT] + [BASE] * 15 + [SWEEP] + [QUIET, SPIKE] + [BACK] * 2, pad=BACK))
    e = episode(eps)
    s = surface(cells, e["cand_id"])
    for (x, y), v in s.items():
        assert v == (0.0 if y in (0.25, 0.5) else 1.0), (x, y, v)


# ---------------------------------------------------------------------------------------------
# 5. Neither: price hovers between the level and the sweep extreme for the rest of the session.
#    Closes are 102.2, which is ABOVE the level, so the close-back never reaches even x = 0.10
#    (which needs a close at or below 102.0 - 0.1071 = 101.8929). Highs are 102.4, below
#    S + 0.25 ATR = 102.7679. => ALL CELLS 0, and the cap-only outcome is recorded as negative.
# ---------------------------------------------------------------------------------------------
def test_neither_at_the_cap():
    eps, cells, _ = build(session_09_to_17([PIVOT] + [BASE] * 15 + [SWEEP] + [QUIET] * 4, pad=QUIET))
    e = episode(eps)
    s = surface(cells, e["cand_id"])
    assert all(v == 0.0 for v in s.values())
    assert e["mfe_back_atr"] < 0  # best close was still above the level
    assert e["mae_beyond_atr"] < 0  # never beyond the sweep extreme
    assert e["cap_ts"].tz_convert(NY).hour == 17


# ---------------------------------------------------------------------------------------------
# 6. A touch is not a sweep (SPEC B.4). Price reaches exactly 102.0, never 102.1, so the level is
#    never traded through and no episode exists.
# ---------------------------------------------------------------------------------------------
TOUCH = [(102.0, 100.5, 101.5)] + [(101.0, 100.0, 100.5)] * 4


def test_touch_without_trade_through_is_not_a_sweep():
    eps, cells, counts = build(session_09_to_17([PIVOT] + [BASE] * 15 + [TOUCH] + [BASE] * 4))
    assert eps.empty or not np.isclose(eps["level"], 102.0).any()
    assert counts["never_swept"] >= 1


# ---------------------------------------------------------------------------------------------
# 7. Monotonicity (SPEC B.5.5): by construction a cell can only fall as x rises and only rise as y
#    rises. Checked on every episode of a mixed session.
# ---------------------------------------------------------------------------------------------
def test_surface_is_monotone_in_x_and_y():
    eps, cells, _ = build(
        session_09_to_17([PIVOT] + [BASE] * 15 + [SWEEP] + [QUIET, SPIKE, TIE_EARLY] + [BACK], pad=BACK)
    )
    assert not eps.empty
    for cid, c in cells.groupby("cand_id"):
        g = c.pivot(index="x_atr", columns="y_atr", values="y").sort_index()
        g = g[sorted(g.columns)]
        assert (g.diff().dropna() <= 0).all().all(), f"not non-increasing in x: {cid}"
        assert (g.diff(axis=1).dropna(axis=1) >= 0).all().all(), f"not non-decreasing in y: {cid}"


# ---------------------------------------------------------------------------------------------
# 8. A sweep in the last 5m bar of the session leaves no window: the decision time IS the cap, so
#    there is nothing to measure. The episode is dropped and counted, never half-labelled.
# ---------------------------------------------------------------------------------------------
def test_sweep_in_the_final_bar_is_dropped_and_counted():
    slots = [BASE] * 15 + [LOWPIV] + [PIVOT] + [BASE] * 78 + [SWEEP]
    assert len(slots) == 96  # the sweep bar closes exactly at the 17:00 cap
    eps, cells, counts = build(slots)
    assert eps.empty or not np.isclose(eps["level"], 102.0).any()
    assert counts["decision_at_cap"] >= 1


# ---------------------------------------------------------------------------------------------
# 9. No-entry sessions (roll rule C) produce no rows at all.
# ---------------------------------------------------------------------------------------------
def test_no_entry_session_is_dropped_and_counted():
    df = frame(session_09_to_17([PIVOT] + [BASE] * 15 + [SWEEP] + [BACK] * 4, pad=BACK))
    sess = set(arrayb.session_date(df["ts_event"].dt.tz_convert(NY)))
    eps, counts = arrayb.build_rows(df, sess)
    assert eps.empty
    assert counts["no_entry_skipped"] >= 1


# ---------------------------------------------------------------------------------------------
# 10. No episode may span a roll (CLAUDE.md 5.5). The contract changes after the decision time, so
#     the episode is dropped and counted.
# ---------------------------------------------------------------------------------------------
def test_roll_inside_the_window_drops_the_episode():
    slots = session_09_to_17([PIVOT] + [BASE] * 15 + [SWEEP] + [BACK] * 4, pad=BACK)
    cut = (16 + 1 + 15 + 1 + 2) * 5  # a few minutes after the decision close

    def contracts(df):
        return np.where(np.arange(len(df)) < cut, "MGCM4", "MGCQ4")

    df = frame(slots, contracts=contracts)
    eps, counts = arrayb.build_rows(df, set())
    assert eps.empty or not np.isclose(eps["level"], 102.0).any()
    assert counts["roll_dropped"] >= 1


# ---------------------------------------------------------------------------------------------
# 11. A 5m bar that mixes two contracts is invalid and confirms no pivot (SPEC B.2).
# ---------------------------------------------------------------------------------------------
def test_mixed_contract_bar_confirms_no_pivot():
    slots = session_09_to_17([PIVOT] + [BASE] * 15 + [SWEEP] + [BACK] * 4, pad=BACK)
    mid = 16 * 5 + 2  # inside the pivot bar itself

    def contracts(df):
        return np.where(np.arange(len(df)) < mid, "MGCM4", "MGCQ4")

    df = frame(slots, contracts=contracts)
    bars = arrayb.resample_5m(df)
    assert not bars["valid"].iloc[16]
    cands = arrayb.find_candidates(bars)
    assert not any(np.isclose(c.level, 102.0) and c.side == 1 for c in cands)


# ---------------------------------------------------------------------------------------------
# 12. Both DST transition weeks. The session cap must land on 17:00 New York local time on each
#     side of the change, not on a fixed UTC hour.
#     2024-03-10 (spring forward): the session runs 18:00 EST Sunday to 17:00 EDT Monday = 22h.
#     2024-11-03 (fall back):      18:00 EDT Sunday to 17:00 EST Monday = 24h.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "start_utc,n_slots",
    [("2024-03-10 23:00", 22 * 12), ("2024-11-03 22:00", 24 * 12)],
)
def test_dst_transition_sessions_cap_at_17_new_york(start_utc, n_slots):
    middle = [PIVOT] + [BASE] * 15 + [SWEEP] + [BACK] * 4
    slots = [BASE] * 15 + [LOWPIV] + middle
    slots += [BACK] * (n_slots - len(slots))
    df = frame(slots[:n_slots], start_utc=start_utc)
    eps, counts = arrayb.build_rows(df, set())
    cells = arrayb.expand(eps)
    e = episode(eps)
    cap_ny = e["cap_ts"].tz_convert(NY)
    assert (cap_ny.hour, cap_ny.minute) == (17, 0)
    assert e["minutes_to_cap"] > 0
    assert surface(cells, e["cand_id"])[(0.1, 0.25)] == 1.0


# ---------------------------------------------------------------------------------------------
# 13. No lookahead. Perturbing every 1m bar after the decision time must leave the decision time,
#     the level, the sweep extreme, the ATR and every feature at or before t unchanged (SPEC B.9).
# ---------------------------------------------------------------------------------------------
def test_no_lookahead_in_features():
    slots = session_09_to_17([PIVOT] + [BASE] * 15 + [SWEEP] + [BACK] * 4, pad=BACK)
    df = frame(slots)
    eps, _ = arrayb.build_rows(df, set())
    e = episode(eps)
    t = e["t_dec"]

    bumped = df.copy()
    m = bumped["ts_event"] >= t
    for col in ("open", "high", "low", "close"):
        bumped.loc[m, col] = bumped.loc[m, col] * 1.5 + 7.0
    bumped.loc[m, "volume"] = 999

    eps2, _ = arrayb.build_rows(bumped, set())
    e2 = episode(eps2)
    assert e2["t_dec"] == t
    for col in ["level", "sweep_extreme", "atr"] + arrayb.EPISODE_FEATURES:
        assert e2[col] == pytest.approx(e[col]), col


# ---------------------------------------------------------------------------------------------
# 14. The block-maximum sweep search is only a speed-up: on random data it must return exactly the
#     bar a plain bar-by-bar scan returns, including "no sweep".
# ---------------------------------------------------------------------------------------------
def test_block_search_matches_a_plain_scan():
    rng = np.random.default_rng(20261116)
    vals = rng.normal(size=5000).cumsum() + 100.0
    bmax = arrayb._block_max(vals)
    for _ in range(300):
        start = int(rng.integers(0, 4900))
        end = int(rng.integers(start + 1, 5000))
        target = float(vals[start:end].mean() + rng.normal() * 0.5)
        hit = np.flatnonzero(vals[start:end] >= target)
        expected = start + int(hit[0]) if len(hit) else -1
        assert arrayb._first_ge(vals, bmax, start, end, target) == expected


# ---------------------------------------------------------------------------------------------
# 15. The walk-forward's monotonicity check must count a surface that rises with x (or falls with
#     y) as a failure, and a well-behaved one as a pass.
# ---------------------------------------------------------------------------------------------
def test_monotone_share_counts_bad_surfaces():
    from pattern_arrays import walkforward_cond as wfc

    rows = []
    for cid, kind in enumerate(["good", "bad_x", "bad_y"]):
        for xi, x in enumerate(arrayb.X_GRID):
            for yi, y in enumerate(arrayb.Y_GRID):
                if kind == "good":
                    p = 0.9 - 0.1 * xi + 0.01 * yi
                elif kind == "bad_x":
                    p = 0.1 + 0.1 * xi + 0.01 * yi  # rises with x: wrong
                else:
                    p = 0.9 - 0.1 * xi - 0.05 * yi  # falls with y: wrong
                rows.append(
                    {"cand_id": cid, "t_dec": pd.Timestamp("2024-01-01", tz="UTC"), "x_atr": x, "y_atr": y, "p": p}
                )
    assert wfc.monotone_share(pd.DataFrame(rows)) == pytest.approx(1 / 3)
