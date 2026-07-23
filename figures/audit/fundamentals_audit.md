# RankAlpha — fundamentals data-quality audit (Phase 17)

*Reproducible via `python scripts/audit_fundamentals.py --source sec`. Educational SIMULATION. Source: SEC EDGAR XBRL `companyfacts` (primary source, real `filed` dates); yfinance cross-check only.*

## VERDICT: **GO**

- Coverage WARN (70–90%): ebitda 76%, enterprise_value 89%, free_cash_flow 82%, market_cap 89%
- Survivorship: universe is NOT point-in-time — GO covers DIRECTIONAL / educational use only, never live/point-in-time claims.

Universe: 503/503 tickers returned data · 19513 statement records (40 quarters/name).

## 1. Accuracy (vs yfinance, TTM-matched)
- status: `ran` · comparable fields: 10 · max relative discrepancy: 2.53% · median: 0.00%

| ticker | field | ours | yfinance | rel. diff | status |
|---|---|---|---|---|---|
| AAPL | revenue | 4.514e+11 | 4.514e+11 | 0.00% | `ok` |
| AAPL | eps | 8.26 | 8.25 | 0.12% | `ok` |
| AAPL | book_value | 1.065e+11 | 1.065e+11 | 0.00% | `ok` |
| MSFT | revenue | 3.183e+11 | 3.183e+11 | 0.00% | `ok` |
| MSFT | eps | 16.8 | 16.47 | 2.00% | `ok` |
| MSFT | book_value | 4.144e+11 | 4.144e+11 | 0.00% | `ok` |
| NVDA | revenue | 2.535e+11 | 2.535e+11 | 0.00% | `ok` |
| JPM | book_value | 3.64e+11 | 3.64e+11 | 0.00% | `ok` |
| XOM | revenue | 3.342e+11 | 3.26e+11 | 2.53% | `ok` |
| KO | eps | 3.18 | 3.19 | 0.31% | `ok` |

## 2. Coverage (% of universe with each value input)
| field | coverage |
|---|---|
| eps | 94.0% |
| price | 90.4% |
| book_value | 100.0% |
| ebitda | 76.0% |
| enterprise_value | 89.4% |
| free_cash_flow | 81.8% |
| market_cap | 89.4% |

## 3. Point-in-time integrity (leakage gate)
- records: 19513 · OK: 19513 · missing publication date: 0 · publication ≤ period-end: 0 · **pass: True**

## 4. Outliers
- counts: `{'negative_equity': 33}` · flagged names: 33
- handling: winsorize ratios to (0.01, 0.99) pct; drop undefined (neg equity / nonpositive denominator) before z-scoring (#18).

## 5. Consistency
- currencies: `['USD']` · all-USD: True · periods: `['TTM']`
- split adjustment applied: **True** · names with splits in window: 124 (e.g. AAPL, ACGL, AFL, AMCR, AMZN, ANET, AOS, APD, APH, APTV, AVGO, BALL)
- Absolute-$ line items are as reported (XBRL facts are already in whole units, not thousands). SPLITS: XBRL facts are as-reported and never retro-adjusted (we keep the earliest-filed value on purpose), while the yfinance price panel rewrites history onto today's split basis — so EPS and share counts are multiplied by the cumulative post-period split factor before any ratio is formed. Without that step NVDA's pre-June-2024 earnings yield would read 10x too high. Fiscal calendars differ across filers by design; every record is keyed to its own fiscal period end and joined by publication date, never by calendar date. Non-USD filers are flagged above.

## 6. Survivorship
- Universe is today's members (survivorship-biased). History omits delisted names; a point-in-time source (e.g. Sharadar) is required for unbiased backtests. Results are DIRECTIONAL / educational only. Note the FUNDAMENTALS themselves are point-in-time (EDGAR `filed` dates, earliest-filed value per period, no restatements) — the survivorship caveat is about which TICKERS are in the list, not about the data.
