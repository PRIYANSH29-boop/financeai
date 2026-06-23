# LIMITATIONS — RankAlpha v1

An honest account of what this system can and cannot claim. Read this before trusting any
number in the README or the backtests. Written at the end of Phase 6 (honest evaluation).

## 1. Survivorship bias (no point-in-time constituents)
The universe is the **current** S&P 500 membership applied across the whole 2019–2026
history. Companies that were dropped or delisted are silently excluded, so the panel only
contains *survivors*. This biases **all** absolute return/Sharpe levels **upward** — for
both the model and the baseline. The *relative* comparison (model vs baseline on the same
universe) is far more trustworthy than any absolute level. Fix deferred: real point-in-time
index membership.

## 2. Weak `size` and `liquidity` proxies
- `size` = `log(adj_close)` — share **price**, not market cap. Price is a poor size proxy
  (a $500 share isn't a big company). True size needs shares outstanding × price.
- `liquidity` = `volume ÷ 20-day average volume` — this is a volume **surge** measure, not
  a liquidity *level*. A dollar-volume measure (price × volume) would be better.
Both are flagged in the Phase-2 backlog for a later fundamentals-backed revision.

## 3. Tiny sample, not statistically significant
Only **47 monthly rebalances** in the OOS window. The model's mean Rank IC is +0.050 with a
**t-stat of 1.64 — below the conventional 2.0 threshold**. The edge *looks* real and is
consistent across most years, but we cannot claim statistical significance from this sample.

## 4. Untested through a momentum crash
The OOS window (2022-06 → 2026-05) contains **no major momentum crash**. The brutal late-2020
crash (the vaccine-news loser rally that gave the full-sample baseline its −46% drawdown)
sits inside the **initial training window**, so the model has never been *tested* through a
regime like it. Its −27% OOS drawdown is the worst we've observed, but a real momentum crash
could be worse. The strategy's behavior in a crash is **unknown**, not validated.

## 5. Single universe (large-cap US only)
Large-cap S&P 500 only. No mid/small caps, no other regions, no other asset classes. Factor
behavior (especially size and liquidity) differs materially outside large-cap US equities;
none of these results generalize beyond this universe without re-testing.

## 6. Short-side realism not modelled
Costs are a flat **10 bps per side on turnover** — no **borrow costs**, no **hard-to-borrow**
fees, no short-availability constraints. Real short books on the bottom-decile names (often
small, volatile, distressed) can be expensive or impossible to short. The short leg's
realized economics are likely **worse** than modelled.

## 7. Edge is concentrated on the long leg
The model's decile spread is a clean staircase on top (D9 +4.04%/21d) but the bottom decile
(D0, the short leg) still has a *positive* +1.12% forward return — shorting it loses money on
average. The model **reduced** the momentum-crash contamination in the short leg (baseline D0
was +1.68%) but did **not** eliminate it. Most of the alpha is long-side; a long-only or
long-tilted version may be the more honest product.

## 8. The model leans on the volatility factor (importance ≠ direction)
Feature importance (LightGBM gain and SHAP agree): **`vol_6m_rank` ≈ 60%**, `size_rank` ≈ 14%,
`mom_12_1m` only ≈ 10%. So `vol_6m` is by far the dominant feature — the ranker beats the
momentum baseline largely by using **volatility and size**, not by doing momentum better
(a genuine multi-factor blend, not re-derived momentum).
**Important caveat on direction:** importance is not direction. Per-holding SHAP on the long
(top-decile) book shows it tilts toward **high-`vol_6m_rank`** names (typically paired with
strong momentum) — i.e. the long picks are higher-volatility, *not* the classic low-volatility
anomaly. An earlier draft mislabelled this as a "low-volatility effect"; corrected. The result
rides heavily on the volatility factor, which is crowded and regime-dependent either way.

## 9. Other
- Daily-bar `fwd_ret_1m` uses a per-ticker 21-row shift; tickers with missing days have a
  slightly irregular forward window. Minor at the panel level.
- No hyperparameter tuning was done (deliberately, to avoid overfitting the test window), so
  the config is reasonable-but-arbitrary; a properly nested CV could move the numbers.
- Costs/turnover assume weights reset to target each rebalance (no intra-period drift).

## What we CAN claim
On a single survivorship-biased large-cap universe, over a 4-year out-of-sample window with
no momentum crash, a frozen LightGBM LambdaMART ranker on 7 price-based factors **beat a pure
12-1 momentum baseline on the same window** (after-cost Sharpe 1.14 vs 0.82; Rank IC 0.050 vs
0.041), with a more monotonic decile spread and an improved (not fixed) short leg, and the
edge **survived costs up to 30 bps/side**. The improvement is interpretable: the model blends
a dominant volatility signal (the long book tilts high-vol, paired with momentum) with size.

## What we CANNOT claim
That this is a deployable, profitable strategy. The absolute returns are survivorship-inflated,
the edge is not statistically significant (t < 2), it is untested through a crash, the short
side is not realistically costed, and the alpha is long-concentrated and volatility-driven.
This is a **methodology demonstration** — a clean, leakage-controlled research pipeline — not
investment advice or a live track record.
