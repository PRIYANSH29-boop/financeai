# Cross-Sectional Prediction — the core idea behind RankAlpha

## Why the single-stock model died

The AAPL model predicted **absolute** 5-day direction: "will the price be higher in 5
days?" After purging look-ahead leakage and charging 10 bps transaction costs, it
collapsed to a coin flip:

- ROC-AUC **0.503** (random = 0.500)
- Strategy **+22.7%** vs buy-and-hold **+47%** over 5 years (loses by ~24 pts)
- The earlier "edge" (Sharpe 0.72) was almost entirely leakage.

This is a **real finding, not a failure.** Technical features (RSI, MACD, Bollinger, etc.)
do not predict the absolute direction of a single name. Absolute return is dominated by
the market itself — a factor no single-stock technical signal can forecast.

## The shift: time-series → cross-section

Scaling the *same question* to 500 stocks ("will each one go up?") just hits the
coin-flip wall 500 times. The point of a universe is to change **what** we predict.

| | Single stock (old) | Universe (new) |
|---|---|---|
| Question | "Will AAPL go up?" | "Will AAPL **outperform the other 499** over the next 5 days?" |
| Frame | absolute, time-series | **relative, cross-sectional** |
| Output | up / down | a **rank** across all names, each day |
| Trade | long if up | long the top, short/avoid the bottom |

Each day we *rank* every stock by predicted relative strength, go long the top decile and
short (or simply avoid) the bottom. This is **cross-sectional momentum / relative
strength** — where quant edge actually tends to live, because ranking nets out the common
market move and isolates the differences between stocks.

## Why `future_return > 0` is the wrong label now

In a cross-sectional world the label must be **relative to the same day's universe**, not
an absolute threshold.

- `future_return > 0` measures each stock against zero — i.e. against the **market drift**.
  On an up day almost everything is positive (all label = 1); on a down day almost
  everything is negative (all label = 0). The label mostly encodes *what the market did
  that day*, which we can't predict and don't want to bet on.
- The right target is a **within-day cross-sectional rank** of forward return: on each
  date, rank all ~500 names by their forward return and label by relative position (top vs
  bottom, or a normalized rank in [0,1]). This is invariant to the market's daily move —
  it only asks **which names beat which**, which is the thing a scoring model can actually
  learn.

In one line: **cross-sectional** means the label is defined *across the stocks on a given
day*, not *across time for one stock* — so the target is a rank within each date, and an
absolute `> 0` threshold is meaningless because it just re-imports the market factor we're
trying to remove.

## New leakage risk that didn't exist with single-stock AAPL

With one ticker, the only leakage axis was *time* (labels peeking forward). With 500
stocks there is a new **cross-sectional** axis:

- **Within-day normalization leakage.** Normalizing/ranking a feature "within each day"
  must use only that day's cross-section. If a feature is z-scored using stats pooled over
  the *full* sample (including future days), every row leaks information from the future.
  Normalize strictly per-date, fit on training dates only.
- (Related) **Universe/survivorship leakage** — using *today's* S&P 500 membership for
  historical dates silently drops the names that were delisted, which is forward-looking
  knowledge. v1 accepts this with a documented caveat; the fix is point-in-time
  constituents.

See [ROADMAP.md](ROADMAP.md) for the phased build plan.
