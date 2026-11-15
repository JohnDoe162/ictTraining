# TradingView indicator

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
