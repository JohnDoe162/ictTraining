# ICT draw on liquidity: a concept model

Can a model learn which **draws on liquidity** actually get reached?

An ICT draw on liquidity is a prior swing high or low that price has not yet traded through — a
resting level that price is supposedly drawn towards. This repository trains a model to score
each such level, on 1-hour MGC (Micro Gold) bars, and tests whether the score means anything on
data the model never saw.

**It does.** Out of sample, across 13 walk-forward windows and 15,094 resolved episodes:

| | |
|---|---|
| Draws reached overall (base rate) | **53.9%** |
| Reached, among the model's **top-scored tenth** | **87.4%** |
| Reached, among the model's **bottom-scored tenth** | **21.3%** |
| Ranking (AUC) | **0.721** |
| Calibration (Brier skill vs base rate) | **+0.141** |
| Walk-forward windows with positive skill | **13 of 13** |

![Hit rate by model score decile](results/arrayA_deciles.png)

Every window is trained only on the 18 months before it and tested on the 6 months after, with
episodes purged so no outcome can leak across the cut. The most recent 12 months of data are
withheld entirely.

## The idea being tested

Most published ICT work specifies quality by hand: a swing must be *this* strong, a gap *this*
wide, a level *this* fresh. Those thresholds are where overfitting hides, because they are
usually chosen after seeing what works.

This model does the opposite. **Three things are fixed in advance** and never tuned — what counts
as the level being reached (traded through by at least one tick), what counts as invalidation
(the opposite swing goes first), and when the clock runs out (17:00 New York). Where both happen
inside the same minute, the episode is scored as a loss: the worst case for the trader.

**Everything else is learned.** There are no minimum sizes, no strength cutoffs, no age limits.
Swing strength, distance, age and volatility regime go in as continuous measurements and the
model decides what they are worth.

The candidate floor is deliberately the weakest usable swing — a bar higher than the single bar
on each side. Strength is measured, not filtered.

## What it learned

Distance dominates, which is partly mechanical: near levels get hit more. The interesting result
is the second-largest weight.

> **`left_strength` = +0.234.** A swing that towers over more prior bars is more likely to be
> taken out — and the effect holds in all 13 windows (+0.163 to +0.286).

That is the ICT claim about liquidity resting beyond significant highs and lows, recovered from
data rather than assumed. Meanwhile `prominence_atr` — how sharply a pivot stands above its two
immediate neighbours — is near zero. **Breadth matters; local spikiness does not.**

Full table in [RESULTS.md](RESULTS.md).

## What this is not

Read this part before using anything here.

1. **It is not a profit result.** The model predicts where price goes, not whether trading it
   makes money. There are no costs, no slippage, no fill model and no position sizing anywhere in
   this repository. A 0.72 AUC can still lose money after costs.
2. **It is not a pass or a verdict.** Pass thresholds were not registered before the run, and
   thresholds chosen after seeing results are not evidence. These numbers are a report.
3. **The base rate is high.** At 54%, guessing "it gets hit" is right more often than not. The
   evidence is the *spread* between score bands, not the level of any one of them.
4. **One market.** MGC only, 1-hour bars, 2018 to 2025.

## Repository

| Path | What it holds |
|---|---|
| [`METHOD.md`](METHOD.md) | the definitions, features, model and evaluation in full |
| [`RESULTS.md`](RESULTS.md) | every number, per window and pooled |
| `src/pattern_arrays/` | the labeller, the logistic regression and the walk-forward harness |
| `scripts/` | the runner, the chart generator and the chart-parity checks |
| `tests/` | 23 tests: hand-worked known answers, a lookahead test, and indicator checks |
| `model/` | the fitted weights |
| `results/` | result tables and charts |
| `tradingview/` | a Pine v6 indicator carrying the same weights |

**No price data ships here.** The research runs on a licensed Databento feed that cannot be
redistributed. `src/pattern_arrays/data.py` documents exactly what bar format to supply.

## Running it

```bash
uv run --with pytest --with pandas --with numpy --with pyarrow python -m pytest tests -q
```

The tests need no market data — every case is a hand-built bar sequence whose correct answer is
worked out by hand in the test file's comments, including a lookahead test that perturbs all
future bars and asserts nothing before that point changes.

With your own 1-minute bars:

```bash
python scripts/pattern_arrays_stage1.py --bars /path/to/mgc_1m.parquet
```

## On the chart

`tradingview/dol_model_mgc.pine` draws every live draw on an MGC 1-hour chart, scores it, marks
whether it was reached, and tallies a hit-rate-by-score table over the loaded history. Expected
table, computed in advance so the indicator can be checked rather than trusted:

| Model score | Episodes | Reached | Hit rate |
|---|---|---|---|
| 0–20% | 1,367 | 247 | 18% |
| 20–40% | 2,858 | 951 | 33% |
| 40–60% | 7,059 | 3,335 | 47% |
| 60–80% | 7,673 | 5,390 | 70% |
| 80–100% | 391 | 364 | 93% |
| All | 19,348 | 10,287 | 53% |

That table is in-sample — it confirms the indicator computes the model faithfully, not that the
model predicts. The out-of-sample evidence is the walk-forward above.

See [`tradingview/README.md`](tradingview/README.md) for setup.
