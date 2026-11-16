# TradingView indicators

Two scripts, one per array. Both exist to check a model by eye, and neither places orders.

| File | Chart | What it scores |
|---|---|---|
| `dol_model_mgc.pine` | MGC, **1 hour** | every live draw on liquidity: will this level be reached? |
| `sweep_model_mgc.pine` | MGC, **5 minute** | every sweep: will price come back inside before taking heat? |

The Array B script is documented in its own section at the end.

---

## Array A — `dol_model_mgc.pine`

`dol_model_mgc.pine` draws every live draw on liquidity on an MGC 1-hour chart and scores it
0–100% using the model's own weights. It exists to check the model by eye. It places no orders.

## Loading it

1. Open an **MGC** chart and set the timeframe to **1 hour**. The script refuses any other
   timeframe, because that is the only one the model was trained on.
2. Pine Editor → paste `dol_model_mgc.pine` → Save → Add to chart.
3. **If the script appears in its own panel below the price chart, remove the indicator and add
   it again.** TradingView keeps an on-chart instance in whatever pane it was first placed in;
   pressing Save updates the code but never moves it. The drawings are pinned to the price pane
   either way, but the table belongs to whichever pane owns the script and cannot be moved by
   code.
4. Scroll back a few years. Scoring begins once about 460 one-hour bars have loaded, because the
   volatility-regime feature needs that much history.

Colours follow the chart theme automatically; there is an explicit Light/Dark override in the
settings.

## What you see

| On the chart | Meaning |
|---|---|
| A horizontal line | A prior swing high or low not yet traded through: a draw |
| Line colour and width | The score. Faint and thin is low, blue and thick is high |
| `72%` | The model's estimate that this level is reached before the opposite side |
| `✓ 81%` green | Reached. The model had scored it 81% |
| `✗ 44%` orange | Invalidated: the opposite side went first |
| `– 38%` grey | Neither, by the 17:00 New York session close |

A level counts as reached only when price trades **through** it by at least one tick. A touch
does not count.

## The table is the point

The corner table tallies every resolved draw in the loaded history by score band. Expected values
on MGC 1h over 2018-01-01 to 2025-10-17, computed ahead of time by
`scripts/pattern_arrays_pine_simulate.py`:

| Model score | Episodes | Reached | Hit rate |
|---|---|---|---|
| 0–20% | 1,367 | 247 | **18%** |
| 20–40% | 2,858 | 951 | **33%** |
| 40–60% | 7,059 | 3,335 | **47%** |
| 60–80% | 7,673 | 5,390 | **70%** |
| 80–100% | 391 | 364 | **93%** |
| All (base rate) | 19,348 | 10,287 | 53% |

**If your chart shows roughly this ladder, the indicator is working.** A flat line across the
bands means something is wrong — check the symbol and the timeframe first.

Always read a band against the 53% base rate on the bottom row. The evidence is the spread
between bands, not the level of any one.

## Limits

1. **The table is in-sample.** The weights were fitted on the same years it tallies, so it
   confirms the indicator computes the model faithfully — it is not fresh evidence that the model
   predicts. The out-of-sample evidence is the walk-forward in `../RESULTS.md`.
2. **TradingView data is not the research data.** The research uses a roll calendar, a
   degraded-session list and an early-close calendar. A continuous contract on TradingView has
   none of that, so expect small differences around contract rolls. How much is lost was measured:
   see the parity table in `../RESULTS.md`.
3. **Extreme scores are rare.** Only about 2% of episodes score above 80%; most sit between 40%
   and 80%.
4. **No costs, no fills, no profit claim.**
5. **MGC only.** The tick-size input exists so the distance feature can be rescaled, but nothing
   else has been checked.

## Keeping it honest

The weights are hardcoded in the `.pine` because Pine cannot read a file.
`tests/test_pattern_arrays_pine.py` fails if they drift from `model/arrayA_model_pine.json`, if
rounding could move a score, if the script starts using `request.security` (which can repaint),
if any table cell loses its background, or if the drawings stop being pinned to the price pane.

If the model is refitted, regenerate the JSON, update the twelve `W_/M_/S_` blocks in the
`.pine`, and run the tests.


---

## Array B — `sweep_model_mgc.pine`

`sweep_model_mgc.pine` marks every 5-minute swing high or low that gets **run** — traded through
by at least one tick — and scores, at the close of that bar, whether price comes back inside the
level before running further against you.

### You choose the question

Two dropdowns at the top of the settings:

| Setting | Meaning |
|---|---|
| **Come back by (ATR)** | how far back inside the level a 5-minute bar must *close* to count |
| **Heat allowed (ATR)** | how far beyond the sweep's extreme price may run first — where a stop would sit |

Only the six and five values the model was trained on are offered. Changing either re-scores
every sweep; it does not refit anything, because both are inputs to the model. This is the whole
point of Array B: **nobody picked those thresholds**, so you can move them and watch the scores
and the hit rates move together.

### Loading it

1. Open an **MGC** chart and set the timeframe to **5 minutes**. The script refuses any other
   timeframe.
2. Pine Editor → paste `sweep_model_mgc.pine` → Save → Add to chart.
3. **If the script appears in its own panel, remove the indicator and add it again** — same
   TradingView quirk as Array A, explained above.
4. Scroll back. The volatility-regime feature uses a 20-trading-day window of 5-minute bars
   (5,520 bars); if your browser struggles or the script errors, lower it in the settings, at the
   cost of matching the research exactly.

### What you see

| On the chart | Meaning |
|---|---|
| A horizontal line | A level that has just been swept, with an episode running |
| Line colour and width | The score. Faint and thin is low, blue and thick is high |
| `68%` | The model's estimate that price comes back by your chosen amount before taking your chosen heat |
| `ok 81%` green | It came back first |
| `x 44%` orange | It took the heat first |
| `- 38%` grey | Neither, by the 17:00 New York cap |

Where a bar both takes the heat and closes back inside, it counts as heat. Five-minute bars
cannot say which came first, so the tie goes against the trader — the same rule the model was
trained on.

### The table

Expected values on MGC 5m over 2018-01-01 to 2025-10-17, for the default question (come back
0.50 ATR before 1.00 ATR of heat), computed ahead of time by
`scripts/pattern_arrays_arrayb_pine_simulate.py`:

| Model score | Sweeps | Came back | Hit rate |
|---|---|---|---|
| 0–20% | 1,431 | 545 | 38% |
| 20–40% | 21,646 | 8,741 | 40% |
| 40–60% | 102,976 | 54,858 | 53% |
| 60–80% | 73,621 | 54,287 | 74% |
| 80–100% | 9,444 | 8,458 | 90% |
| All | 209,118 | 126,889 | 61% |

Your chart will differ a little: TradingView's continuous contract is not stitched the way the
research data is, and the script holds a capped number of candidates in memory.

**This table is in-sample** — the weights were fitted on this data. It confirms the indicator
computes the model faithfully; it is not evidence that the model predicts. That evidence is the
walk-forward in [`../RESULTS_ARRAY_B.md`](../RESULTS_ARRAY_B.md).

### One caveat worth knowing before you trust the score

Almost all of this model's ranking comes from a single measurement: how far back inside the level
the sweep bar itself closed. The full twenty-input model scores AUC 0.673 out of sample; that one
measurement alone scores 0.671. Part of its power is simply that a bar which already closed back
inside starts nearer the target. The score is not reading anything subtle about swing quality —
there was nothing subtle to read.
