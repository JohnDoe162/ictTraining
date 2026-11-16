# Method — Array B: sweep and reversal (5-minute)

Array A asks whether a resting level gets reached. **Array B asks what happens after one is run.**

A prior 5-minute swing high or low is traded through by at least one tick — the stops resting
there are taken. At the close of that bar, a trader has a decision to make. Array B scores it.

## 1. The question, and why it has no threshold in it

The obvious way to pose this is a yes/no: *does price close back inside the level by X ticks
before it runs Y ticks further against me?* That version was drafted and **withdrawn**, because X
and Y are exactly the kind of numbers that get chosen after seeing which value works.

Instead the label carries no threshold at all. For every sweep, the outcome recorded is

> **g(y) = the largest close-back achieved before price first ran y against you**

on a fixed grid of both quantities, in ATR at the moment of the decision:

| | values (ATR) |
|---|---|
| come back by (`x`) | 0.10, 0.25, 0.50, 0.75, 1.00, 1.50 |
| heat allowed (`y`) | 0.25, 0.50, 1.00, 2.00, 4.00 |

30 cells per sweep, each a 0 or a 1. **`x` and `y` are then fed to the model as inputs**, so it
predicts the whole surface rather than one point on it. Nobody picks a threshold; a trader picks
an operating point later, when there is money on the table to pick it with.

The grid was fixed in writing before the first run. Re-slicing the stored outcomes onto a
different grid later is a new version of the model, not a free second look.

## 2. The three rules that are fixed, and everything that is not

Fixed in advance, never tuned:

1. **The candidate floor.** A 5-minute bar whose high is strictly above the single bar on each
   side (mirrored for lows; ties disqualify). The weakest usable swing — anything less is not a
   turning point. Without a floor there is nothing to score.
2. **The decision time.** The close of the bar that sweeps the level. Everything the model sees
   is closed by then; everything it predicts happens after. This is also what makes the awkward
   "what if it closes back inside on the sweep bar itself" case disappear: measurement starts at
   the decision, so an outcome reached before it does not exist.
3. **The session cap.** 17:00 New York. An outcome measured over an unbounded horizon is not
   defined.

Where a close-back and the heat happen inside the same minute, the sweep is scored as a loss:
the worst case for the trader. No 1-second data is bought to break the tie.

**Everything else is learned.** No minimum sweep depth, no strength cutoff, no age limit, no
volume filter. Each is a continuous measurement with a weight the model chooses.

## 3. The sweep extreme

All heat is measured from the furthest price reached between the first trade through the level
and the decision close. On 5-minute bars that is simply the sweep bar's extreme — not an
approximation, because the first minute to trade through the level cannot come after the bar's
high was set.

## 4. Features

Eighteen measurements, plus the two grid inputs. All use only bars closed at or before the
decision. ATR is a 14-bar simple mean of true range on 5-minute bars.

| | feature | what the model may learn from it |
|---|---|---|
| 1 | `sweep_depth_atr` | shallow stop-run or a real break |
| 2 | `sweep_depth_ticks` | the same in absolute terms |
| 3 | `close_back_frac` | how far back inside the level the sweep bar closed, as a fraction of its range — the rejection |
| 4 | `sweep_bar_range_atr` | how violent the run was |
| 5 | `left_strength` | swing strength: earlier bars the pivot towers over |
| 6 | `prominence_atr` | swing strength: sharpness against its two neighbours |
| 7 | `age_bars` | fresh or stale liquidity |
| 8 | `tests_held` | approaches that came close and held |
| 9 | `atr_norm` | volatility regime |
| 10 | `rvol_sweep` | was the run on real participation |
| 11 | `room_to_opposite_atr` | room for the reversal to travel |
| 12 | `level_rank_session` | was this the session's extreme liquidity |
| 13 | `range_pos` | where price sits in the recent range |
| 14 | `minutes_to_cap` | how much time the outcome has |
| 15 | `side` | swept high +1, swept low −1 |
| 16–18 | `bucket_london`, `bucket_ny_am`, `bucket_ny_pm` | time of day (Asia is the baseline) |
| 19–20 | `x_atr`, `y_atr` | the grid inputs |

Constants, fixed and not tuned: ATR length 14, volatility-median window 20 trading days, strength
cap 20 bars, range window 20 bars, tested-and-held band 1 ATR, relative-volume window 20 bars.

## 5. The model

L2-regularised logistic regression (C = 1.0), fitted in numpy — the same `logit.py` Array A uses,
unchanged. Features are standardised on the training window only. One row per (sweep, cell).

Being linear in `x` and `y` makes it an approximation to a curved surface. It is the baseline, and
it is reported as one: a richer family must beat it out of sample on the same walk-forward.

## 6. How it is judged

- **18 months train, 6 months test, rolled forward**, 13 windows, nothing overlapping.
- **Purge and embargo on the session cap** — the latest moment any cell of an episode can
  resolve — so all 30 cells of a sweep always sit on the same side of a cut.
- **The baseline is the base rate of the individual cell.** Cell base rates run from 28% to 85%;
  a model that had only learned the shape of the grid would beat a single pooled rate while
  knowing nothing about sweeps. That loophole is closed by construction.
- **Episodes, not rows, are the unit of independence.** 30 rows come from one sweep. Every count
  and every gate in this repository is quoted in episodes.

### Registered before the run

Unlike Array A, Array B's thresholds were written down before it was first run
([`thresholds.json`](thresholds.json)) and are applied by the runner:

| | test | needed |
|---|---|---|
| V1 | within-cell top-decile lift over that cell's base rate | ≥ 1.20 |
| V2 | pooled Brier skill against the per-cell baseline | > 0 |
| V3 | share of walk-forward windows with positive skill | ≥ 0.50 |
| V4 | predicted surface falls with `x` and rises with `y` | ≥ 99% of episodes |
| gate | independent episodes, before any training | ≥ 500, and ≥ 50 in each year |

A failed array is shelved, not tuned.

## 7. Known limits

1. **Part of the edge is proximity, not prediction.** A sweep bar that already closed back inside
   starts nearer the close-back levels it is asked to reach. The ranking does decay as more is
   asked of it (AUC 0.72 at the smallest close-back, 0.62 at the largest), which says there is
   something beyond geometry — but not how much.
2. **No costs anywhere.** This predicts price, not profit.
3. **The linear surface is weakest at its edges**, and the published calibration shows it.
4. **One market**, MGC, 5-minute bars, 2018 to the seal in 2025.
