# RankAlpha — universe expansion to mid + large cap (Phase 16)

*Reproducible via `python scripts/expand_universe.py`. Educational SIMULATION — survivorship-biased universe, results DIRECTIONAL, no promised returns.*

## 1. The universe

| filter | names remaining |
|---|---|
| registrants_listed_with_shares | 5,346 |
| with_market_data | 5,344 |
| after price >= $1.00 | 4,779 |
| after market cap > $2.0B | 2,138 |
| after median $vol >= $1.0M | 1,802 |
| universe_size | 1,200 |

- Median market cap **$14.1B**, floor **$2.01B**, aggregate **$89.8T**.
- Built 2026-07-23 from SEC registrant + share-count data and a yfinance price/liquidity screen. A liquidity cap kept the top 1200 by dollar volume (602 names dropped by that cap alone).
- Panel actually downloaded: **1200 tickers**, 2,017,848 rows, 2019-07-23 → 2026-07-22.

## 2. Retrained model vs no-ML baseline

Same architecture and hyper-parameters as the S&P 500 ranker (`lambdarank`, 300 trees, lr 0.02), same walk-forward (504d initial train, 126d step, 21d embargo, never shuffled). This is a NEW frozen model; the S&P 500 one is untouched.

| universe | model | Sharpe | Rank IC | IC t | max DD | total ret | turnover |
|---|---|---|---|---|---|---|---|
| S&P 500 (501 names) | LGBMRanker | +1.140 | +0.0505 | +1.64 | -27.2% | +166.7% | 1.34 |
| S&P 500 (501 names) | 12-1 momentum | +0.823 | +0.0414 | +1.58 | -20.1% | +72.6% | 1.29 |
| Mid + large cap (1185 names) | LGBMRanker | +1.812 | +0.0276 | +1.03 | -13.3% | +549.0% | 1.10 |
| Mid + large cap (1185 names) | 12-1 momentum | +0.507 | +0.0307 | +1.41 | -29.1% | +40.8% | 1.27 |

- **Model − baseline Sharpe**: S&P 500 +0.317 → mid+large +1.305.
- OOS windows: S&P 500 2022-06-15 → 2026-05-15; mid+large 2022-07-22 → 2026-06-22.

## 3. Did the edge survive?

On the wider universe the ranker posts Sharpe +1.81 and Rank IC +0.0276 (t=+1.03) against a no-ML momentum baseline of Sharpe +0.51 / IC +0.0307. **The edge survived on paper — but do NOT bank this number.** Model − baseline Sharpe went +0.317 → +1.305, yet Rank IC *fell* +0.0505 → +0.0276 and got less significant (t +1.64 → +1.03). Those two facts point in opposite directions: the model's ability to ORDER the cross-section got worse while the payoff of its top decile got much better. That is what survivorship-inclusion bias looks like on a universe screened by today's market cap — names that were small in 2019 and compounded their way past the $2B floor are present for the whole history, and a volatility-and-momentum ranker concentrates in exactly them. Rank IC is the bias-resistant measure here, and Rank IC says the skill did not improve. Treat the S&P 500 model as the shipped one; treat this universe as a demonstration that the pipeline retrains cleanly, not as evidence of a bigger edge. Rank IC t-statistics on ~4 years of monthly rebalances are not strong evidence either way; treat a small Sharpe difference as noise, and a large one as a question about the data before it is an answer about the model.

## 4. Benchmark

The scorecard benchmark is the **equal-weight new universe** (mean realized forward return across eligible names), which is the right field for a mid+large cap book — an S&P 500 benchmark would be measuring against a different universe than the one traded. The long/short decile book is roughly market-neutral by construction, so the benchmark is a reference, not the thing being beaten.

## 5. Pointing the pie engine at the new universe

`portfolio.beta_engine.build_portfolio` and `portfolio.engine.score_book` are already path-parameterised, so #15's pie runs on this universe with no code change:

```python
from portfolio.beta_engine import build_portfolio
build_portfolio(10_000, target_beta=1.0,
                panel_path='data/midlarge_panel.parquet',
                tickers_path='data/universe_midlarge.csv',
                features_path='data/midlarge_features.parquet',
                labeled_path='data/midlarge_labeled.parquet')
```
Note the sector column: `data/universe_midlarge.csv` carries no GICS sector (SEC does not publish one), so the pie's per-sector caps degrade to a single '?' bucket until a sector mapping is attached. That is a real gap, not a rounding detail — the ≤5-per-sector and 30%-per-sector constraints are inert without it.

## 6. Caveats

- **Survivorship is worse here, not better — in both directions.** A current $2B floor applied to all history *deletes* the names that fell below it, and *includes* from day one the names that started small and compounded past it. The second half is the more dangerous one for a long/short decile book: the top decile gets to hold 2019's future ten-baggers. The S&P 500 panel has the same disease; this universe has a much larger dose, which is the first thing to suspect about any headline number in §2.
- **Foreign issuers are included** — US-listed ADRs (e.g. AMBEV, Abivax) pass an exchange + market-cap screen. A US-domicile filter would change the field.
- **Short history for younger names**: features need 253 trading days, so the early cross-section is much smaller than the late one.
- Educational SIMULATION. Not investment advice.