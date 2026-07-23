# RankAlpha — value factor A/B (Phase 18)

*Reproducible via `python scripts/value_factor.py`. Educational SIMULATION — no promised returns, survivorship-biased universe, results DIRECTIONAL.*

Fundamentals: **33,364 point-in-time records** over **501 tickers**, SEC EDGAR XBRL, publication-date lagged 1d. #17 audit verdict: **GO**.

## Composite spec

| ratio | definition | orientation |
|---|---|---|
| earnings yield | EPS (TTM) / price | higher = cheaper |
| book-to-market | book value / market cap | higher = cheaper |
| EBITDA/EV | EBITDA (TTM) / enterprise value | higher = cheaper |
| FCF yield | free cash flow (TTM) / market cap | higher = cheaper |

Each ratio winsorized at (0.01, 0.99), z-scored cross-sectionally per rebalance, then averaged; a name needs ≥2 of the 4. Value coverage across rebalance dates: **96.0%** of the eligible cross-section; uncovered names get a neutral score (0.0) so the A/B universe is identical.

## Frozen paper-track window — 2024-06-14 → 2026-04-21 (23 months)

| metric | A · Momentum | B · Momentum + value | C · Value only | Equal-weight universe |
|---|---|---|---|---|
| Total return | 46.23% | 43.62% | 39.07% | 36.62% |
| CAGR | 21.93% | 20.79% | 18.78% | 17.68% |
| Volatility (ann) | 13.49% | 13.72% | 15.18% | 14.02% |
| Sharpe | 1.55 | 1.46 | 1.22 | 1.24 |
| Sortino | 1.82 | 2.19 | 2.24 | 2.05 |
| Max drawdown | -13.09% | -11.06% | -9.85% | -10.79% |
| Drawdown duration (periods) | 9 | 6 | 8 | 4 |
| Hit rate | 73.91% | 73.91% | 65.22% | 60.87% |
| Beta | 0.75 | 0.88 | 0.84 | 1.00 |
| Alpha (ann) | 7.83% | 4.64% | 3.88% | 0.00% |
| N periods | 23 | 23 | 23 | 23 |

- **corr(value, momentum)**: mean **-0.203** (range -0.283 … -0.096), cross-sectional Spearman per rebalance.
- **turnover** (mean per rebalance, sum |Δw|): A 0.47 → B 0.51 (Δ +0.04, ≈ 0.4 bps/month more cost at 10 bps/side).

## Full labelled history — 2020-06-16 → 2026-04-23 (71 months)

| metric | A · Momentum | B · Momentum + value | C · Value only | Equal-weight universe |
|---|---|---|---|---|
| Total return | 155.81% | 132.19% | 123.38% | 173.60% |
| CAGR | 17.20% | 15.30% | 14.55% | 18.54% |
| Volatility (ann) | 12.18% | 11.86% | 12.27% | 15.26% |
| Sharpe | 1.37 | 1.27 | 1.17 | 1.20 |
| Sortino | 2.37 | 2.40 | 2.10 | 1.92 |
| Max drawdown | -14.13% | -12.50% | -9.37% | -18.21% |
| Drawdown duration (periods) | 19 | 17 | 12 | 16 |
| Hit rate | 64.79% | 60.56% | 57.75% | 71.83% |
| Beta | 0.61 | 0.65 | 0.65 | 1.00 |
| Alpha (ann) | 5.55% | 3.11% | 2.61% | 0.00% |
| N periods | 71 | 71 | 71 | 71 |

- **corr(value, momentum)**: mean **-0.151** (range -0.418 … +0.190), cross-sectional Spearman per rebalance.
- **turnover** (mean per rebalance, sum |Δw|): A 0.47 → B 0.42 (Δ -0.05, ≈ 0.5 bps/month less cost at 10 bps/side).

## Full history, matched vol (14%) — 2020-06-16 → 2026-04-23 (71 months)

| metric | A · Momentum | B · Momentum + value | C · Value only | Equal-weight universe |
|---|---|---|---|---|
| Total return | 155.81% | 146.49% | 126.71% | 173.60% |
| CAGR | 17.20% | 16.47% | 14.84% | 18.54% |
| Volatility (ann) | 12.18% | 12.38% | 12.49% | 15.26% |
| Sharpe | 1.37 | 1.30 | 1.18 | 1.20 |
| Sortino | 2.37 | 2.55 | 2.13 | 1.92 |
| Max drawdown | -14.13% | -12.50% | -9.37% | -18.21% |
| Drawdown duration (periods) | 19 | 17 | 12 | 16 |
| Hit rate | 64.79% | 60.56% | 57.75% | 71.83% |
| Beta | 0.61 | 0.67 | 0.65 | 1.00 |
| Alpha (ann) | 5.55% | 3.90% | 2.84% | 0.00% |
| N periods | 71 | 71 | 71 | 71 |

- **corr(value, momentum)**: mean **-0.151** (range -0.418 … +0.190), cross-sectional Spearman per rebalance.
- **turnover** (mean per rebalance, sum |Δw|): A 0.47 → B 0.44 (Δ -0.02, ≈ 0.2 bps/month less cost at 10 bps/side).

## Verdict

- **Frozen paper-track window** (23 months): Sharpe 1.55 → 1.46 (-0.09); max drawdown -13.09% → -11.06% (+2.03%); corr(value, momentum) -0.203.
- **Full labelled history** (71 months): Sharpe 1.37 → 1.27 (-0.10); max drawdown -14.13% → -12.50% (+1.63%); corr(value, momentum) -0.151.
- **Full history, matched vol (14%)** (71 months): Sharpe 1.37 → 1.30 (-0.07); max drawdown -14.13% → -12.50% (+1.63%); corr(value, momentum) -0.151.

**DROP for now — but a genuine near-miss, and the reason is worth stating precisely.** Value passes the independence test (the hard one) and it does cut drawdown in every window. What it does not do is pay for itself: Sharpe falls in every window because the return it gives up exceeds the risk it removes. That is the opposite of the low-vol result in #14, which held return while halving drawdown and so earned its place. Under the survival-chain rule — uncorrelated AND improves the scorecard — one out of two is a DROP. Keep the harness and the point-in-time data; revisit value when it can be tested on a universe where cheap names are not systematically the ones survivorship deleted.

## Caveats

- Short windows: the frozen-track window is ~2 years of monthly observations; no Sharpe difference over that span is statistically significant.
- Survivorship: today's S&P 500 membership applied to all history — the cheapest names are precisely those most likely to have been deleted, so a survivorship-biased universe flatters value more than it flatters momentum.
- Coverage is uneven by sector: EBITDA/EV and FCF yield are undefined for most banks (no capex line), so financials effectively score on E/P and B/M only.
- Educational SIMULATION. Not investment advice.