"""Known-answer and lookahead tests for the Array A (1h DOL) labeller, model and metrics.

Every scenario is a hand-built 1m bar sequence whose correct candidates and labels were worked
out by hand in the comments (CLAUDE.md section 5 rule 11). MGC tick = 0.10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pattern_arrays import dol, logit, walkforward

NY = "America/New_York"


def bars_1m(spec, start, contract="MGCZ5", step_min=1):
    """spec: list of (offset_minutes, open, high, low, close). start: NY-time string."""
    t0 = pd.Timestamp(start, tz=NY).tz_convert("UTC")
    rows = []
    for off, o, h, lo, c in spec:
        rows.append(
            {
                "ts_event": t0 + pd.Timedelta(minutes=off),
                "instrument_id": 1,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": 1,
                "contract": contract,
                "roll": False,
            }
        )
    return pd.DataFrame(rows)


def flat_hours(start, n_hours, price, contract="MGCZ5"):
    """One 1m bar per hour at a flat price - keeps ATR warm-up simple."""
    spec = [(60 * i, price, price, price, price) for i in range(n_hours)]
    return bars_1m(spec, start, contract)


def hour_bar(start_ny, o, h, lo, c, contract="MGCZ5"):
    """A single 1m bar representing one whole hour (open minute only)."""
    return bars_1m([(0, o, h, lo, c)], start_ny, contract)


def build(df1m, no_entry=frozenset()):
    df1m = df1m.sort_values("ts_event").reset_index(drop=True)
    return dol.build_rows(df1m, set(no_entry))


# ---------------------------------------------------------------------------
# Scenario base: 16 warm-up hours at 1800.0 starting Mon 2025-01-06 18:00 NY,
# then a pivot-high triple and a pivot-low triple, then an outcome day.
#
# Hand-worked structure (1h bars, one 1m bar per hour unless stated):
#   bars 0..15 : flat 1800/1800/1800/1800  (warm-up; ATR(14) ready from bar 14, = 0 range
#                but previous-close TR = 0 ... so use small ranges to keep ATR > 0)
# To keep ATR > 0 the warm-up bars have high = price + 0.5, low = price - 0.5 (TR = 1.0,
# ATR = 1.0 exactly once 14 TRs exist).
# ---------------------------------------------------------------------------


def warmup(start="2025-01-06 18:00", n=16, price=1800.0, contract="MGCZ5"):
    spec = [(60 * i, price, price + 0.5, price - 0.5, price) for i in range(n)]
    return bars_1m(spec, start, contract)


def scenario(outcome_minutes, extra_hours=(), contract="MGCZ5"):
    """Warm-up Mon 18:00 .. Tue 09:00 NY (16 bars, ATR=1.0), then:
    bar16 Tue 10:00: 1800/1800.5/1799.5/1800
    bar17 Tue 11:00: 1800/1803.0/1799.5/1802   <- pivot HIGH bar (high 1803.0)
    bar18 Tue 12:00: 1800/1801.0/1796.0/1797   <- confirms pivot high at 13:00 close;
                                                  also pivot LOW bar (low 1796.0)
    bar19 Tue 13:00: 1797/1798.0/1797.0/1797.5 <- confirms pivot low at 14:00 close
    Then `outcome_minutes`: 1m bars from Tue 14:00 NY.
    Candidate A (side +1): level 1803.0, conf 13:00 NY close (18:00 UTC winter).
      left_strength: bars 16,15,... highs 1800.5 < 1803 -> capped run of warm-up bars = 17? cap 20;
      bars before pivot: bar16 high 1800.5, bars 15..0 high 1800.5 -> all < 1803, so k = 17.
      prominence = (1803.0 - max(1800.5, 1801.0)) / ATR = (1803-1801)/1.0 = 2.0.
    Candidate B (side -1): level 1796.0, conf 14:00 NY close.
      invalidation reference for A at its row-2+ decisions = B (most recent opposite alive).
      At A's FIRST decision (13:00) the most recent opposite-side pivot is the warm-up's only
      pivot low if any: warm-up bars are all equal lows (ties disqualify) -> NO pivot low yet
      -> A's first row is dropped (no_opposite).
    Candidate A event: a 1m bar with high >= 1803.1. B event: low <= 1795.9.
    A invalidated (via B): low <= 1795.9. So after B exists, A's invalidation = B's event level.
    """
    parts = [warmup(contract=contract)]
    parts.append(hour_bar("2025-01-07 10:00", 1800, 1800.5, 1799.5, 1800, contract))
    parts.append(hour_bar("2025-01-07 11:00", 1800, 1803.0, 1799.5, 1802, contract))
    parts.append(hour_bar("2025-01-07 12:00", 1800, 1801.0, 1796.0, 1797, contract))
    parts.append(hour_bar("2025-01-07 13:00", 1797, 1798.0, 1797.0, 1797.5, contract))
    if outcome_minutes:
        parts.append(bars_1m(outcome_minutes, "2025-01-07 14:00", contract))
    for hb in extra_hours:
        parts.append(hb)
    return pd.concat(parts, ignore_index=True)


def get_row(rows, side, age=None):
    m = rows[rows["side"] == side]
    if age is not None:
        m = m[m["age_sessions"] == age]
    return m


def test_structure_candidates_and_features():
    # Quiet outcome bar, plus a Wednesday bar so Tuesday's episode resolves "neither" at the
    # cap instead of being dropped as unresolved-at-data-end.
    wed = [hour_bar("2025-01-07 18:00", 1797.5, 1798.0, 1797.0, 1797.2)]
    rows, counts = build(scenario([(0, 1797.5, 1797.6, 1797.4, 1797.5)], extra_hours=wed))
    assert counts["candidates"] == 2
    # Candidate A's first row was dropped: no opposite pivot existed at its decision time.
    assert counts["no_opposite"] == 1
    # Candidate B's first row (decision Tue 14:00 close): reference = A (alive).
    b = get_row(rows, -1, 0)
    assert len(b) == 1
    r = b.iloc[0]
    # Hand-worked features for B at 14:00 close (close 1797.5, ATR = mean of last 14 TRs):
    # TRs of bars 6..19: bars 6..15 TR=1.0; bar16 TR=1.0; bar17 max(3.5,|1803-1800|,|1799.5-1800|)=3.5;
    # bar18 max(5.0,|1801-1802|,|1796-1802|)=6.0; bar19 max(1.0,|1798-1797|,|1797-1797|)=1.0
    # ATR = (10*1.0 + 1.0 + 3.5 + 6.0 + 1.0)/14 = 21.5/14 = 1.535714...
    atr = 21.5 / 14
    assert r["level"] == 1796.0
    assert abs(r["dist_to_level_atr"] - (1797.5 - 1796.0) / atr) < 1e-9
    assert abs(r["dist_to_level_ticks"] - 15.0) < 1e-9
    # left_strength of B: bars before pivot bar18: bar17 low 1799.5 > 1796 (k=1), bar16 low
    # 1799.5 (k=2), warm-up lows 1799.5 (k up to cap) -> 18 bars before bar18 -> k = 18.
    assert r["left_strength"] == 18
    # prominence of B = (min(1799.5, 1797.0) - 1796.0)/ATR(bar18).
    # ATR(bar18) = mean TRs bars 5..18 = (11*1.0 + 1.0 + 3.5 + 6.0)/14 = 21.5/14? bars 5..18 is
    # 14 TRs: bars5..15 are 1.0 (11 bars), bar16 1.0, bar17 3.5, bar18 6.0 -> 21.5/14.
    assert abs(r["prominence_atr"] - (1797.0 - 1796.0) / atr) < 1e-9
    assert r["age_sessions"] == 0
    assert r["side"] == -1
    # decision 14:00 NY -> bucket ny_pm
    assert r["bucket_ny_pm"] == 1.0 and r["bucket_ny_am"] == 0.0


def test_plain_event():
    # Outcome: at 14:05 a bar trades DOWN through B's level 1796.0 by >= 1 tick (low 1795.9).
    # For candidate B (side -1) that is the EVENT. Candidate A's rows: first dropped
    # (no_opposite); A gets no later row before data ends (same session).
    out = [(0, 1797.5, 1797.6, 1797.4, 1797.5), (5, 1797.0, 1797.0, 1795.9, 1796.5)]
    rows, counts = build(scenario(out))
    b = get_row(rows, -1, 0).iloc[0]
    assert b["label"] == "event"
    # resolution = close of the 14:05 bar = 14:06 NY
    assert b["resolution_ts"] == pd.Timestamp("2025-01-07 14:06", tz=NY).tz_convert("UTC")


def test_touch_does_not_count():
    # low exactly 1796.0 = touch, and low 1795.95 = half a tick through: neither is the event.
    out = [(0, 1797.0, 1797.0, 1796.0, 1796.5), (5, 1796.5, 1796.6, 1795.95, 1796.2)]
    wed = [hour_bar("2025-01-07 18:00", 1797.5, 1798.0, 1797.0, 1797.2)]
    rows, _ = build(scenario(out, extra_hours=wed))
    b = get_row(rows, -1, 0).iloc[0]
    assert b["label"] == "neither"  # resolved at the session cap


def test_plain_invalidation_and_same_bar_worst_case():
    # B's invalidation reference is A (level 1803.0, side +1): a 1m bar with high >= 1803.1
    # invalidates B. Hand-worked: 14:10 bar high 1803.1 -> B invalidated.
    out = [(0, 1797.5, 1797.6, 1797.4, 1797.5), (10, 1800.0, 1803.1, 1799.9, 1802.0)]
    rows, _ = build(scenario(out))
    assert get_row(rows, -1, 0).iloc[0]["label"] == "invalidated"

    # Same single 1m bar reaches BOTH 1795.9 (event) and 1803.1 (invalidation):
    # worst case -> invalidated (no 1s data).
    out2 = [(0, 1797.5, 1797.6, 1797.4, 1797.5), (10, 1800.0, 1803.1, 1795.9, 1802.0)]
    rows2, _ = build(scenario(out2))
    assert get_row(rows2, -1, 0).iloc[0]["label"] == "invalidated"


def test_neither_then_rescored_next_session():
    # No event, no invalidation before Tue 17:00 -> label "neither". Next session (Wed) the
    # candidate is re-scored at the first 1h bar close with age_sessions = 1, and Wed at 10:30
    # a bar trades through 1796.0 -> that second row is the EVENT.
    out = [(0, 1797.5, 1797.6, 1797.4, 1797.5)]
    wed = [
        hour_bar("2025-01-07 18:00", 1797.5, 1798.0, 1797.0, 1797.2),  # Wed session 1st bar
        hour_bar("2025-01-08 10:00", 1797.0, 1797.2, 1795.9, 1796.0),
    ]
    rows, counts = build(scenario(out, extra_hours=wed))
    b = get_row(rows, -1)
    assert list(b["label"]) == ["neither", "event"]
    assert list(b["age_sessions"]) == [0, 1]
    # second decision = close of Wed session's first 1h bar = 19:00 Tue NY
    assert b.iloc[1]["t_dec"] == pd.Timestamp("2025-01-07 19:00", tz=NY).tz_convert("UTC")


def test_level_killed_outside_episode():
    # "neither" on Tue; Wed's FIRST 1h bar itself trades through 1796.0 BEFORE the decision
    # close -> the level died outside an episode: no Wed row for B.
    out = [(0, 1797.5, 1797.6, 1797.4, 1797.5)]
    wed = [hour_bar("2025-01-07 18:00", 1797.0, 1797.2, 1795.9, 1796.0)]
    rows, counts = build(scenario(out, extra_hours=wed))
    b = get_row(rows, -1)
    assert list(b["label"]) == ["neither"]
    assert counts["died_outside_episode"] == 1


def test_no_entry_session_skipped():
    # Wednesday is a no-entry session: no decision row that day, but the candidate survives
    # and is scored Thursday with age_sessions = 2.
    out = [(0, 1797.5, 1797.6, 1797.4, 1797.5)]
    wed_thu = [
        hour_bar("2025-01-07 18:00", 1797.5, 1798.0, 1797.0, 1797.2),
        hour_bar("2025-01-08 18:00", 1797.2, 1797.6, 1796.8, 1797.0),
        hour_bar("2025-01-09 10:00", 1797.0, 1797.2, 1795.9, 1796.0),
    ]
    ne = {pd.Timestamp("2025-01-08", tz=NY)}
    rows, counts = build(scenario(out, extra_hours=wed_thu), no_entry=ne)
    b = get_row(rows, -1)
    assert list(b["label"]) == ["neither", "event"]
    assert list(b["age_sessions"]) == [0, 2]
    assert counts["no_entry_skipped"] >= 1


def test_roll_drop():
    # Wednesday's bars are the NEXT contract: the candidate is dropped, no Wed row.
    out = [(0, 1797.5, 1797.6, 1797.4, 1797.5)]
    wed = [hour_bar("2025-01-07 18:00", 1797.5, 1798.0, 1797.0, 1797.2, contract="MGCG6")]
    rows, counts = build(scenario(out, extra_hours=wed))
    b = get_row(rows, -1)
    assert list(b["label"]) == ["neither"]
    assert counts["roll_dropped"] >= 1


def test_retirement_bound():
    # A candidate alive for > 30 sessions is retired. Build 33 quiet sessions after the setup.
    out = [(0, 1797.5, 1797.6, 1797.4, 1797.5)]
    extra = []
    d0 = pd.Timestamp("2025-01-07", tz=NY)
    for i in range(34):
        evening = str((d0 + pd.DateOffset(days=i)).date())
        morning = str((d0 + pd.DateOffset(days=i + 1)).date())
        # session day i+1: an 18:00 bar (its first 1h bar = the decision bar) and a quiet
        # 10:00 bar the next morning, so the episode has bars after the decision close.
        extra.append(hour_bar(f"{evening} 18:00", 1797.5, 1798.0, 1797.0, 1797.5))
        extra.append(hour_bar(f"{morning} 10:00", 1797.5, 1798.0, 1797.0, 1797.5))
    rows, counts = build(scenario(out, extra_hours=extra))
    b = get_row(rows, -1)
    assert b["age_sessions"].max() == dol.RETIRE_SESSIONS - 1
    assert counts["retired"] >= 1


def test_dst_spring_and_fall_sessions():
    # Spring forward 2025-03-09 and fall back 2025-11-02: the session cap at 17:00 NY and the
    # 18:00 NY session start must hold in both weeks. Hand-check: a bar at 16:59 NY belongs to
    # that session; a bar at 18:00 NY belongs to the next.
    for day in ["2025-03-09", "2025-11-02"]:
        d = pd.Timestamp(day, tz=NY)
        # wall-clock construction (NOT absolute-hour addition, which breaks on the 23h/25h days)
        ts = pd.Series(
            [
                pd.Timestamp(f"{day} 16:59", tz=NY).tz_convert("UTC"),
                pd.Timestamp(f"{day} 18:00", tz=NY).tz_convert("UTC"),
            ]
        )
        sd = dol.session_date(ts.dt.tz_convert(NY))
        assert sd.iloc[0] == d.normalize()
        # wall-clock next midnight (NOT +24 absolute hours: these are 23h/25h days)
        assert sd.iloc[1] == pd.Timestamp(str((d + pd.DateOffset(days=1)).date()), tz=NY)


def test_lookahead_perturbation():
    # Perturb every bar strictly after t: the set of rows with decision <= t and ALL their
    # features are unchanged; labels of rows RESOLVED <= t are unchanged.
    out = [(0, 1797.5, 1797.6, 1797.4, 1797.5), (5, 1797.0, 1797.0, 1795.9, 1796.5)]
    base = scenario(out)
    t = pd.Timestamp("2025-01-07 14:00", tz=NY).tz_convert("UTC")  # B's decision close
    pert = base.copy()
    after = pert["ts_event"] > t
    assert after.any()
    pert.loc[after, ["open", "high", "low", "close"]] += 50.0  # move the future wildly
    r1, _ = build(base)
    r2, _ = build(pert)
    k1 = r1[r1["t_dec"] <= t].reset_index(drop=True)
    k2 = r2[r2["t_dec"] <= t].reset_index(drop=True)
    feat_cols = ["t_dec", "side", "level"] + dol.FEATURES
    pd.testing.assert_frame_equal(k1[feat_cols], k2[feat_cols])
    res1 = k1[k1["resolution_ts"] <= t]
    res2 = k2.loc[res1.index]
    assert list(res1["label"]) == list(res2["label"])


# --------------------------- model and metrics ----------------------------


def test_logit_known_answer_gradient_zero_and_recovery():
    # (a) At the fitted optimum the penalised gradient is ~0 (the defining condition).
    rng = np.random.default_rng(7)
    x = rng.normal(size=(500, 3))
    true_w = np.array([1.5, -2.0, 0.0])
    p = 1 / (1 + np.exp(-(x @ true_w)))
    y = (rng.uniform(size=500) < p).astype(float)
    m = logit.fit(x, y)
    xs = np.column_stack([np.ones(len(x)), (x - m.mean) / m.std])
    pr = 1 / (1 + np.exp(-(xs @ m.coef)))
    lam = np.array([0.0, 1.0, 1.0, 1.0])
    grad = xs.T @ (pr - y) + lam * m.coef
    assert np.max(np.abs(grad)) < 1e-6
    # (b) sign recovery: w1 > 0, w2 < 0, |w3| small
    assert m.coef[1] > 0.5 and m.coef[2] < -0.5 and abs(m.coef[3]) < 0.3


def test_metrics_known_answer():
    # Hand-worked: y = [1,0,1,0], p = [0.8,0.2,0.6,0.4]
    # Brier = ((0.2)^2 + (0.2)^2 + (0.4)^2 + (0.4)^2)/4 = (0.04+0.04+0.16+0.16)/4 = 0.10
    # log loss = -(ln0.8 + ln0.8 + ln0.6 + ln0.6)/4
    y = np.array([1.0, 0, 1, 0])
    p = np.array([0.8, 0.2, 0.6, 0.4])
    assert abs(logit.brier(y, p) - 0.10) < 1e-12
    assert abs(logit.log_loss(y, p) + (2 * np.log(0.8) + 2 * np.log(0.6)) / 4) < 1e-12
    # AUC: pairs (0.8 vs 0.2, 0.8 vs 0.4, 0.6 vs 0.2, 0.6 vs 0.4) all correctly ordered -> 1.0
    assert walkforward._auc(y, p) == 1.0
    # deciles with 4 rows, 2 bins: low bin rates
    dec = walkforward.decile_table(y, p, n_bins=2)
    # sorted p: 0.2(y0), 0.4(y0) | 0.6(y1), 0.8(y1) -> rates 0.0 and 1.0
    assert list(dec["event_rate"]) == [0.0, 1.0]


def test_walkforward_purge():
    # A training row resolving AFTER the test start must be excluded. Window maths:
    # data 2018-01-01..2019-12-31 -> first test starts 2019-07-01.
    w = walkforward.windows("2018-01-01", "2020-01-01")
    assert w[0][1] == pd.Timestamp("2019-07-01", tz="UTC")
    assert w[0][0] == pd.Timestamp("2018-01-01", tz="UTC")
    # a last partial window is included
    w2 = walkforward.windows("2018-01-01", "2019-10-15")
    assert w2[-1][2] == pd.Timestamp("2019-10-15", tz="UTC")
