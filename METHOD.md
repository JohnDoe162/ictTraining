# Method

How the model is defined, trained and judged. Every number quoted here comes from the files in
`results/`, produced by the scripts in `scripts/`.

## 1. What the model predicts

A **draw on liquidity** is a prior swing high or low that price has not yet traded through — a
resting level that, in ICT terms, price is drawn towards. The model scores each such level with
the probability that it gets reached before the opposite side does.

Formally, for each candidate level at each decision time, the outcome is one of:

| Label | Meaning |
|---|---|
| `event` | the level was traded through by at least one tick |
| `invalidated` | the opposite-side reference level was traded through first |
| `neither` | neither happened before the session cap at 17:00 New York |

The model is trained on `event` versus `invalidated`. `neither` episodes are counted and reported
but excluded from training, because they carry no information about which side won.

## 2. The three rules that are fixed, and the one thing that is not

A model that chooses its own definition of success will always look good. Three things are
therefore fixed in advance and never tuned:

1. **The event.** A level counts as reached only when price trades *through* it by at least one
   tick. A touch does not count.
2. **The invalidation.** The most recent confirmed opposite-side pivot still alive at the decision
   time. If that is traded through first, the episode is lost.
3. **The session cap.** 17:00 New York. A CME trading day runs 18:00 to 17:00.

Where both the event and the invalidation occur inside the same one-minute bar, the episode is
labelled **invalidated** — the worst case for the trader. No one-second data is used to break the
tie, so the label can never be flattered by an optimistic assumption.

Everything else is **learned, not specified**. There are no hand-set quality gates: no minimum
swing size, no strength cutoff, no age limit. Such measurements enter as continuous features and
the model decides what they are worth. This is deliberate — the question being asked is what
*the data* says makes a good draw, not whether a hand-tuned filter can be made to look profitable.

## 3. The candidate floor

Something must define the weakest structure that still counts, or there is no list of candidates
to score. The floor is the weakest usable swing: **a bar whose high is strictly greater than the
single bar on each side** (mirrored for lows), with ties disqualified. It is confirmed at the
close of the following bar and is usable only from then, so nothing looks ahead.

Swing *strength* is deliberately **not** part of the definition. It is measured by two features
(`left_strength` and `prominence_atr`) and the model learns how much it matters.

A level dies permanently once it is traded through, whether or not an episode was running at the
time. A candidate is re-scored at the first bar of each later session while it survives, which is
how a draw is actually used: yesterday's untouched high is today's target.

## 4. Features

Ten features, all computed only from bars closed at or before the decision time, on 1-hour bars.
ATR is a simple 14-bar mean of true range.

| Feature | What it measures |
|---|---|
| `dist_to_level_atr` | distance from price to the level, in ATR |
| `dist_to_level_ticks` | the same distance in ticks |
| `left_strength` | how many earlier bars the pivot towers over (capped at 20) |
| `prominence_atr` | how sharply the pivot stands above its neighbours, in ATR |
| `age_sessions` | trading days since the level was confirmed |
| `tests_held` | times price came within 1 ATR without breaking through |
| `atr_norm` | current ATR against its own 20-day median: the volatility regime |
| `side` | +1 for a high, −1 for a low |
| `range_pos` | where price sits in the last 20 bars' range |
| session bucket | Asia / London / New York morning / New York afternoon (one-hot, Asia the baseline) |

## 5. The model

L2-regularised logistic regression (`C = 1.0`), fitted with Newton-Raphson in numpy — about 60
lines, no machine-learning dependency. Features are standardised using the training window's own
mean and standard deviation.

The loss is **log loss**, a proper scoring rule: a confident correct call gains most, a confident
wrong call costs most, and a hedged call moves the score little either way. This matters, because
a model that is merely well-ranked but badly calibrated would still score poorly.

A linear model was chosen first on purpose. If simple measurements carry the signal, a linear
model finds it; anything more complex has to beat this baseline out-of-sample before it earns its
place.

## 6. How it is judged

**Walk-forward with purging.** Train on 18 months, test on the next 6, roll forward, repeat —
13 windows over 2019-07 to 2025-10. Each window's model sees only data from before its test
period. A training row is used only if its episode *resolved* before the test window opens, so no
label can leak across the cut.

Reported per window and pooled:

- **Base rate** — how often any candidate is reached. Printed beside every result, because price
  reaches nearby levels often and "most draws get hit" is not evidence of anything.
- **Ranking** — event rate by score decile, and AUC.
- **Calibration** — a reliability table and the Brier skill score against the base rate.
- **Log-loss lift** over a constant model that always predicts the base rate.

The most recent 12 months of data are withheld from everything above.

## 7. Known limits

1. **No verdict is claimed.** Pass thresholds were not registered before the run, and thresholds
   chosen after seeing results are not evidence. These numbers are a report.
2. **This is not a profit result.** The model predicts where price goes, not whether trading it
   makes money. No costs, no slippage, no fill model, no position sizing.
3. **Correlated rows.** One level is re-scored on successive sessions, so rows are not fully
   independent. A first-rows-only cross-check is reported alongside every pooled figure.
4. **One market.** MGC only. Nothing here has been checked on another instrument.
5. **A high base rate flatters everything.** At 54%, a naive "it'll get hit" guess is right more
   often than not. The evidence is the *spread* between score bands, not the level of any one.
