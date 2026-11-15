# Results

All figures are **out-of-sample**: 13 walk-forward windows, each model trained on the preceding
18 months and tested on the following 6, 2019-07 to 2025-10, MGC 1-hour bars. 15,094 resolved
episodes. Source files are named beside each table.

## Headline

| Measure | Result | Reading |
|---|---|---|
| Base rate | **53.9%** | how often any candidate draw was reached |
| Ranking (AUC) | **0.721** | 0.5 is a coin flip, 1.0 is perfect ordering |
| Calibration (Brier skill) | **+0.141** | 14% better than always guessing the base rate |
| Log loss | **0.6175** vs 0.6902 | against the constant base-rate model |
| Top-decile hit rate | **87.4%** | the highest-scored tenth of draws |
| Bottom-decile hit rate | **21.3%** | the lowest-scored tenth |
| Windows with positive skill | **13 of 13** | no window failed |

The model separates a 21% group from an 87% group, out of sample, around a 54% base rate.

## Hit rate by score decile

`results/arrayA_decile.csv`, charted in `results/arrayA_deciles.png`.

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Event rate | 21% | 33% | 40% | 47% | 50% | 56% | 59% | 68% | 76% | 87% |

Monotonic across all ten. The ranking is not driven by a single extreme bucket.

## Stability across windows

`results/arrayA_per_window.csv`, charted in `results/arrayA_windows.png`.

| Test window opens | Episodes | Base rate | AUC | Brier skill | Top decile |
|---|---|---|---|---|---|
| 2019-07 | 1,239 | 56.0% | 0.677 | +0.083 | 82.1% |
| 2020-01 | 1,154 | 54.0% | 0.708 | +0.102 | 82.6% |
| 2020-07 | 1,225 | 53.7% | 0.713 | +0.129 | 86.9% |
| 2021-01 | 1,164 | 52.9% | 0.706 | +0.127 | 87.1% |
| 2021-07 | 1,214 | 53.9% | 0.711 | +0.135 | 90.1% |
| 2022-01 | 1,204 | 54.7% | 0.757 | +0.186 | 90.0% |
| 2022-07 | 1,205 | 52.4% | 0.736 | +0.155 | 89.2% |
| 2023-01 | 1,189 | 53.7% | 0.718 | +0.142 | 83.1% |
| 2023-07 | 1,208 | 55.0% | 0.710 | +0.133 | 84.2% |
| 2024-01 | 1,217 | 52.4% | 0.738 | +0.154 | 86.0% |
| 2024-07 | 1,240 | 54.7% | 0.747 | +0.176 | 91.1% |
| 2025-01 | 1,152 | 53.1% | 0.747 | +0.176 | 90.4% |
| 2025-07 | 683 | 53.4% | 0.714 | +0.124 | 91.2% |

The weakest window is the first, which has the least training history. 2020 and 2022 — very
different volatility regimes — are not outliers.

## Calibration

`results/arrayA_reliability.csv`, charted in `results/arrayA_reliability.png`. Predicted
probabilities track observed frequencies closely through the middle of the range, with mild
over-confidence at both tails: scores below 20% are reached slightly more often than claimed, and
scores above 80% slightly less.

## Correlated rows

One level is re-scored on successive sessions, so pooled rows are not fully independent. Scoring
only each candidate's **first** appearance (11,384 episodes, 58.0% base rate) gives
AUC **0.706** and Brier skill **+0.123** — slightly lower, same conclusion.
`results/arrayA_decile_first_rows.csv`.

## What the model learned

Weights on standardised features, fitted over all pre-cutoff data (`model/arrayA_model_full.json`):

| Feature | Weight | Reading |
|---|---|---|
| `dist_to_level_atr` | **−1.009** | nearby draws are reached far more often. Partly mechanical |
| `left_strength` | **+0.234** | **stronger swings attract price more** |
| `bucket_london` | +0.178 | London-session decisions score higher |
| `bucket_ny_pm` | +0.149 | as do New York afternoon ones |
| `age_sessions` | +0.111 | older surviving levels do slightly better |
| `tests_held` | −0.091 | repeatedly approached and held is a mild negative |
| `dist_to_level_ticks` | −0.084 | absolute distance, beyond the ATR-scaled term |
| `range_pos`, `atr_norm`, `prominence_atr`, `side` | < 0.03 | near zero |

Distance dominating is expected. The result worth noting is **`left_strength`**: a swing that
towers over more prior bars is more likely to be taken out. That is the ICT claim about liquidity
resting beyond significant highs and lows, recovered from data rather than assumed. It is stable
across windows (+0.163 to +0.286 in all 13).

Conversely `prominence_atr` — sharpness against the two adjacent bars — is near zero. Breadth
matters; local spikiness does not.

## Episode counts

`results/arrayA_audit_by_year.csv`. 20,192 candidates produced 22,474 decision rows, of which
18,537 resolved. Between 2,195 and 2,950 rows per year, evenly spread, with no year dominating.
The event-versus-invalidation base rate sits between 53.2% and 54.9% every year.

## The chart version

A TradingView indicator (`tradingview/`) carries the same weights. Because a chart sees only
1-hour bars of a continuous contract — no roll calendar, no vendor-degraded-session list, and
resolution at bar close rather than minute by minute — the whole dataset was rebuilt under those
restrictions and the walk-forward re-run. `results/arrayA_pine_parity.csv`:

| Pipeline | Episodes | Base rate | AUC | Brier skill | Top decile |
|---|---|---|---|---|---|
| Research (1-minute, roll calendar) | 15,094 | 53.9% | 0.721 | +0.141 | 87.4% |
| Chart-equivalent (1-hour only) | 15,912 | 52.6% | 0.715 | +0.134 | 85.4% |

Almost nothing is lost. The indicator's own state machine was then transliterated bar by bar in
Python (`scripts/pattern_arrays_pine_simulate.py`) and agrees with the batch pipeline to within
1.1% on episode count.

## Reading this honestly

A 0.72 AUC on a 54% base rate is a real but moderate signal. It says the model knows which draws
are likely to be reached. It does **not** say that trading them makes money: there are no costs,
no slippage and no fill model anywhere in this repository. That is a different question and it
has not been answered here.
