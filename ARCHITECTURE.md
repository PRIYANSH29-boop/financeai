# ARCHITECTURE — RankAlpha

*Written for someone who has never seen this repo: a recruiter, an interviewer, or a future
contributor. Every entry says what a thing does, what it reads, what it writes, which `make`
target exercises it, and which tests cover it. Every claim describes the code **as it is**
(post-#28), not as it is meant to become.*

⚠️ **Educational simulation. No real money, no investment advice, no forward projections.**

---

## What RankAlpha is, in one paragraph

RankAlpha ranks S&P 500 stocks each month with a machine-learned **ranking** model (LightGBM
LambdaMART), turns the top of that ranking into a long-only, risk-capped portfolio at a
user-chosen market beta, and publishes the whole thing — including its own weaknesses — as a
backend-free static website. The mission is *"rank by alpha, strip beta"*: find the
stock-picking signal, then control market exposure deliberately. The model is **frozen** (fit
once, 2024-05-15); everything built since only *measures* or *constructs* on top of it.

## Data flow

```
                         ┌──────────────────── SEC EDGAR XBRL ──────────────────┐
                         │  audit/sec_provider.py  (point-in-time `filed` dates)│
                         └───────────────┬──────────────────────────────────────┘
                                         │
                                   AUDIT GATE  ── audit/fundamentals.py ──► GO / NO-GO
                                         │        (7 checks; NO-GO blocks the factor)
                                         ▼
  yfinance ──► utils/sp500_data.py ──► data/sp500_panel.parquet
                                         │
                                         ├──► utils/sp500_features.py ──► sp500_features.parquet
                                         │           (cross-sectional ranks, ≥253d eligibility)
                                         └──► utils/sp500_labels.py   ──► sp500_labeled.parquet
                                                     (forward-return RANK, 21-day embargo)
                                         │
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
        signals/baseline_momentum.py            signals/lgbm_ranker.py  ──► FROZEN MODEL
             (the bar ML must beat)                          │
                     └──────────► signals/evaluate.py ◄──────┘
                                  (walk-forward, embargoed, after-cost)
                                         │
                     ┌───────────────────┼────────────────────────┐
                     ▼                   ▼                        ▼
             lab/  (survival chain)   portfolio/  (pie engines)   analytics/  (the referee)
             strategy_lab             engine.py     — vol-target   metrics / charts / compare
             value_factor             beta_engine.py — beta-target
             style_lab                paper_trade.py — forward track
             regime_backtest
                                         │
                                         ▼
                        scripts/export_web_bundle.py  ──► web/public/bundle/*.json
                            (precomputes EVERY number; validators gate the export)
                                         │
                                         ▼
                        web/  (static Next.js export) ──► Cloudflare Pages
                            the browser only multiplies weights by capital
```

The one-line rule that holds the right-hand side together: **if a number is not in the
bundle, the UI cannot show it.** Honesty by architecture rather than by discipline.

---

## Directories and modules

### `utils/` — phases 1–3, the base pipeline

Builds the three artifacts everything else consumes. Rebuilt with `make pipeline`.

| Module | What it does | Reads | Writes |
|---|---|---|---|
| `sp500_data.py` | Scrapes current S&P 500 constituents and downloads ~7 years of daily OHLCV into one tidy panel. Class `SP500DataBuilder`. Sanity-gates the result. | yfinance, Wikipedia | `data/sp500_panel.parquet`, `data/sp500_tickers.csv` |
| `sp500_features.py` | Per stock/day: momentum (1/3/6/12m), 1-month reversal, 6m vol, liquidity, size — then **percentile-ranks within each date's cross-section**. `MIN_HISTORY = 253` trading days makes "eligible" mean "every feature computable". Public: `sanity_gate`, `build_raw_features`, `cross_sectional_rank`, `build_features`. | `sp500_panel.parquet` | `data/sp500_features.parquet` |
| `sp500_labels.py` | The label is a **rank**, not a return: forward 21-day return, ranked within its date. Public: `build_labels`, `print_report`. | panel + features | `data/sp500_labeled.parquet` |

**Make:** `make panel` · `make features` · `make labels` · `make pipeline` (all three, in order).
**Tests:** `test/test_utils_pipeline_f2.py` — 24 tests pinning the *leakage geometry*
(backward-only, per-ticker, within-date, exactly one deliberate 21-day forward look), plus the
sanity gate and three that pin the A-1 fix (#31 Arm 1: `size` is the raw traded price,
never the retro-adjusted one). Every assertion is
mutation-verified: the builders were deliberately broken five ways and each break was caught.

### `signals/` — phases 4–6, the model and its honest evaluation

| Module | What it does | Reads | Writes |
|---|---|---|---|
| `baseline_momentum.py` | The zero-ML bar: classic 12-1 momentum, long top decile, inverse-vol sized, monthly rebalance, costs. If the model can't beat this, the model is a bug. Public: `rebalance_dates`, `backtest_scores`, `run_backtest`, `compute_metrics`. | labeled | backtest artifacts |
| `lgbm_ranker.py` | LightGBM **LambdaMART** (`rank:` objective) trained walk-forward with a 21-day embargo. Public: `walk_forward` (accepts a `fit_fold` override so a rival library runs the identical protocol), `run`. **This is the module that fits the frozen model; nothing else may refit it.** | labeled | frozen model + `data/sp500_oos_walkforward.parquet` |
| `xgb_ranker.py` | #31 Arm 2 — the controlled XGBoost rival. Swaps only the estimator into the shared `walk_forward`. Research only; never shipped, never the product engine. | labeled v2 | `data/sp500_oos_walkforward_xgb.parquet` |
| `evaluate.py` | Rank IC, decile spreads, after-cost Sharpe, sub-period stability, cost sensitivity, feature importance. Public: `ic_timeseries`, `sharpe`, `subperiod_stability`, `cost_sensitivity`, `feature_importance`. | OOS predictions | figures |
| `sic_sectors.py` | SIC → GICS-ish sector table, the #20 fallback (source B) when yfinance has no sector. Public: `sic_to_sector`. | — | — |

**Tests:** `portfolio/tests/test_sector_mapping.py` covers `sic_sectors`.

### `audit/` — phase 17, the data gate

| Module | What it does | Reads | Writes |
|---|---|---|---|
| `sec_provider.py` | EDGAR XBRL client. Caches a slim per-ticker extraction rather than the multi-MB companyfacts blob. Every fact carries the real `filed` date — the point-in-time weapon. Public: `SECClient` (`.ledger`, `.statements`, `.fetch_splits`, `.cik_candidates`), `split_factor`, `ttm`, and the `SplitBasisUnavailable` exception. | data.sec.gov | `data/sec_cache/` |
| `fundamentals.py` | The seven-check GO/NO-GO audit: accuracy, coverage, point-in-time, outliers, consistency, survivorship, reproducibility. Also the shared `winsorize` / `zscore` / `value_ratios` helpers. Single-source since #30 — the unreachable FMP vendor client was removed (F-4). | provider | `figures/audit/fundamentals_audit.md` |

**Make:** `make audit` (needs network) · `make audit-self-test` (offline, proves the harness).
**Tests:** `audit/tests/test_fundamentals.py`, `test_sec_provider.py`, `test_split_basis_a2.py`.

**Invariant:** a missing split basis **raises** (`SplitBasisUnavailable`). An empty split map
means "fetched, no splits exist" and can never mean "we could not find out" — pairing
as-reported per-share facts with split-adjusted prices made NVDA look 10× cheaper than it was.

### `analytics/` — phase 12, the measuring instrument

A standalone library that **imports nothing from the model code** — the referee and the player
are different people. Conventions: population std (`ddof=0`), monthly periodicity ×√12.

| Module | What it does |
|---|---|
| `metrics.py` | `analyse()` plus `equity_curve`, `total_return`, `cagr`, `volatility`, `sharpe`, `sortino`, `max_drawdown`, `beta`, `alpha`, `hit_rate`. |
| `charts.py` | Five charts (equity, drawdown, rolling, histogram, overlay) → `figures/analytics/`. |
| `compare.py` | Side-by-side scorecards for two return streams. |
| `data.py` | Autopilot loader — any ticker or benchmark, so the library is usable off this repo. |
| `rankalpha.py` | Wires the library onto RankAlpha's own paper-track returns. |

**Make:** `make analyse` · **Tests:** `analytics/tests/test_metrics.py`, `test/test_analytics.py`
(hand-checked fixtures, e.g. `vol([+12,−8,+4,+8]) = 7.48%`, so a convention can never drift silently).

### `lab/` — the survival chain (phases 14, 18, 21, 26)

A candidate factor is **KEPT** only if it is uncorrelated with what is already traded **and**
improves the risk-adjusted scorecard. One out of two is a DROP.

| Module | What it does | Verdict |
|---|---|---|
| `strategy_lab.py` | Combine one candidate factor with momentum by equal-weight percentile rank (deliberately no ML, so any effect is attributable). | low-vol **KEEP** |
| `value_factor.py` | E/P + book-to-market + EBITDA/EV + FCF yield, winsorized → z-scored → averaged, joined by **publication** date. `as_of()` is the leakage gate. | value **DROP** |
| `style_lab.py` | Cross-sectional percentile style rules (growth/value/dividend/blue-chip/cyclical/defensive/speculative) + `is_non_equity` (the SIC + name rule that keeps trusts out). | census |
| `regime_backtest.py` | Slices the *existing* history into calm/normal/stressed and recomputes per pile. Nothing trained, nothing predicted. | β-drift measured |
| `signal_duel.py` | #27 — momentum vs the frozen ML through identical construction, only the score varying, on the walk-forward OOS frame. Applies a decision rule fixed *before* the run. | **trade momentum** |
| `last_stand.py` | #31 — the ML's three-arm final campaign: clean the model, change the library, change the vehicle. Writes all three verdicts and the closure statement. | **momentum keeps the engine** |
| `long_short.py` | #31 Arm 3 — decile long/short research sleeve with a charged, sensitivity-tested borrow assumption. Research only; the retail pie stays long-only. | ML wins the vehicle |

**Make:** `make lab` · `make value` · `make styles` · `make duel` · `make v2` · `make xgb` · `make last-stand` · `make regimes-backtest`
**Tests:** `lab/tests/` (4 files). `regime_backtest` publishes **two** momentum columns since
#30 — the repaired one and the retracted uncapped one — because a correction that erases what
it corrects is not a correction.

### `portfolio/` — the pie engines

| Module | What it does | Notes |
|---|---|---|
| `engine.py` | Phase 7 **vol-targeted** engine. `score_book()` (frozen scoring, cached), `build_portfolio(amount, target_vol=…)`. **Shipped and frozen — do not mutate.** | wired into `app.py` |
| `beta_engine.py` | Phase 15 **beta-targeted** engine, added alongside rather than over the top. `build_portfolio(capital, target_beta, …)` hits the target by construction: cash sleeve to lower beta, bounded bisected tilt to raise it, and impossible targets are **capped and disclosed** (max ≈1.89), never faked. | the web product's engine |
| `paper_trade.py` | The forward, never-refit track record: `load_track`, `update_track` (idempotent), `compute_stats`. | 23 months |
| `llm_explainer.py` | Optional per-slice narration (`USE_LLM=1`, local Ollama or Groq). Template text is the default. | never a number source |
| `make_bundle.py` | Regenerates the committed v1.1 hosted-demo bundle. | legacy path |

**Construction funnel (beta_engine):** frozen scores → top-N with ≤5 names/sector →
inverse-vol sizing → **caps** (≤8%/name, ≤30%/sector) → hit target beta → score vs benchmark.

**Caps fail CLOSED (post-#28).** `_apply_caps` raises `CapsInfeasibleError` rather than
returning weights that breach the caps it exists to enforce. It checks joint capacity up
front: a pool concentrated in few sectors can hold at most `Σ min(sector_cap, n×name_cap)`,
and if that is under 100% there is no correct answer to return.

**Tests:** `portfolio/tests/` — `test_beta_engine.py`, `test_caps_fail_closed_b1.py`,
`test_web_bundle.py`, `test_sector_mapping.py`.

### `scripts/` — the runnable entry points

Each script is the CLI for one phase and writes a committed report under `figures/`.

| Script | Phase | Produces |
|---|---|---|
| `analyse.py` | 13 | `figures/analytics/rankalpha_scorecard.md` |
| `audit_fundamentals.py` | 17 | `figures/audit/fundamentals_audit.md` (+ `--self-test`) |
| `strategy_lab.py` | 14 | `figures/lab/strategy_lab_*.md/png` |
| `value_factor.py` | 18 | `figures/lab/value_factor.md/png` |
| `expand_universe.py` | 16 | wide universe + panel/features/labels + **a second frozen model**, `figures/lab/universe_expansion.md` |
| `map_sectors.py` / `sector_report.py` | 20 | `data/universe_midlarge_sectors.csv`, `figures/lab/sector_mapping.md` |
| `regime_stress_test.py` | 14-ext | 2008 GFC + COVID stress windows |
| `style_season_report.py` | 26 | `figures/lab/style_season_report.md` |
| `export_web_bundle.py` | 19A | `web/public/bundle/*.json` — **the only writer of UI numbers** |

`universe.py` (repo root) builds the 1,200-name mid+large universe from SEC registrant data
plus a liquidity screen: `shares_outstanding`, `candidates`, `is_non_equity`, `fallback_shares`.

### `web/` — the static product (phase 19B, 23)

`output: 'export'` Next.js. No server, no login, no runtime data fetch.

- **`web/lib/`** — pure, `node:test`-verified logic. `basket.js` (client-side equal-weight
  scorecard; **no forward-projection function exists, by design**), `explore.js` (the
  ordering rule that sinks statistically implausible rows), `disclosure.js` (the partial-month
  caveat), `format.js`.
- **`web/components/`** — `PieApp` + `Panels` (control bar, why-this-holding, drift, the
  "How honest is this?" receipts), `ExploreView` (1,200-name table), `BasketView`,
  `Charts`/`Donut` (hand-rolled SVG, no chart library), `App` (view switch).
- **`web/public/bundle/`** — the exported JSON. **Gitignored**: it is a build artifact.

**Make:** `make web-bundle` · `make web-dev` · `make deploy` · **Tests:** `npm test` (32).

### `test/`, `figures/`, `data/`

- **`test/`** — cross-cutting suites that do not belong to one package: bundle contracts
  (`test_web_bundle_hotfix24.py`, `test_web_bundle_v2.py`), universe rules
  (`test_universe_non_equity.py`, `test_universe_share_gap.py`), and the #28 fix-pass tests
  (`test_partial_month_a3.py`, `test_requirements_pinned_d1.py`, `test_make_pipeline_d2.py`).
- **`figures/`** — **committed on purpose.** These are the receipts: every published number
  traces to a markdown report here. `analytics/`, `audit/`, `lab/`, `portfolio/`.
- **`data/`** — **gitignored** (`*.parquet`, `*.csv`, `*.json`, `cache/`, `sec_cache/`).
  Regenerate with `make pipeline`, then `make universe` / `make sectors` for the wide-universe
  artifacts. Nothing in `data/` is required to read the repo — only to re-run it.

---

## Make targets

| Target | Does | Needs | Produces |
|---|---|---|---|
| `panel` / `features` / `labels` | Phases 1–3 base pipeline | network (panel only) | `data/sp500_*.parquet` |
| `pipeline` | all three, in order | network | the above |
| `analyse` | analyser scorecard + charts | data | `figures/analytics/` |
| `lab` | momentum vs momentum+low-vol | data | `figures/lab/strategy_lab_*` |
| `value` | value-factor A/B (blocked unless #17 says GO) | data | `figures/lab/value_factor.*` |
| `styles` | style census + season grids | data | `figures/lab/style_season_report.md` |
| `regimes` | 2008 + COVID stress test | network | `figures/lab/` |
| `regimes-backtest` | regime-segmented backtest | data | `figures/lab/regime_report.md` |
| `duel` | #27 signal duel: ML vs momentum | data | `figures/lab/signal_duel.md` |
| `v2` / `xgb` | #31 build the A-1-fixed model / the XGBoost rival | data | `data/*_v2, *_xgb.parquet` |
| `last-stand` | #31 all three arms + closure | data | `figures/lab/last_stand.md` |
| `audit` | fundamentals GO/NO-GO | network | `figures/audit/` |
| `audit-self-test` | proves the harness offline | — | exit code |
| `universe` | wide universe + retrain | network, slow | `data/midlarge_*` |
| `sectors` | sector map + cap-binding report | network | `data/universe_midlarge_sectors.csv` |
| `web-bundle` | export every UI number | data | `web/public/bundle/` |
| `web-dev` | bundle + Next dev server | Node | — |
| `deploy` | build + ship to Cloudflare Pages | Node, CF token | live site |
| `test` / `test-all` | analyser tests / every suite | — | — |

**There is deliberately no `train` target,** and a test asserts there never is one. The ranker
is frozen; a `make train` sitting next to `make pipeline` is an invitation to refit it.

---

## Invariants (do not break these)

1. **The model is FROZEN.** Fit 2024-05-15, walk-forward + 21-day embargo. Feature, analytics
   and portfolio phases only measure or construct on top of the same scores. The single
   exception is #16, which trained a *separate, clearly-labelled* model on a wider universe
   and kept the old one — and whose headline was then rejected.
2. **Educational simulation.** No real money, no advice, survivorship caveat on every report.
3. **No forward projections, ever.** The only permitted answer to "what will I get?" is the
   distribution of past outcomes. A test asserts no forward-return field exists in the bundle.
4. **The audit gate.** No factor is built on data that has not passed a GO/NO-GO audit.
5. **No number is typed into the frontend.** The exporter computes every figure; the browser
   only multiplies weights by capital.
6. **The caps fail closed** (post-#28). A safety rule that fails open is worse than no rule,
   because the UI still says "max 8% per stock".
7. **Fail loud, never silently wrong.** Missing split basis raises; a future-dated `as_of`
   fails the export; a partial final month must be disclosed, not merely flagged.
8. **Negative results are published.** The rejected Sharpe 1.81, the dropped value factor, the
   208-cell null grid. They are the credibility.

---

## Branches

| Branch | Purpose | State | Recommendation |
|---|---|---|---|
| `main` | Trunk | Tip `aa3f1f7` (Phase 23, 2026-07-24). **10 commits behind `phase-12-analytics`**, 0 ahead. | **Fast-forward from `phase-12-analytics`** — the merge is trivial (no divergence). |
| `phase-12-analytics` | Opened for phase 12 (the analyser) | **The de-facto trunk.** 10 unpushed commits ahead of `main`, carrying phases 24, 25, 26, 26b–d and 28. Checked out. | **Keep, but merge to `main` and stop working on it.** The name is 16 phases stale. |
| `phase-11-hosted-demo` | The v1.1 hosted Streamlit demo | Tip `7d27893` (2026-06-24), **fully merged into `main`** (0 commits not in main). | **Delete**, local and remote. Nothing is lost. |
| `origin/main` | — | Identical to local `main`. | — |
| `origin/phase-12-analytics` | — | At `aa3f1f7` — **10 commits behind local**. GitHub is blocked from the build machine, so pushes happen from a hotspot. | Push. |
| `origin/phase-11-hosted-demo` | — | Merged. | Delete with the local one. |

**It is not main's twin — it is main's future.** `main` has not moved since Phase 23 while
every phase since (#24 hotfix → #28 fix-pass) landed on `phase-12-analytics`. That is the
cleanup worth doing: fast-forward `main`, delete the stale demo branch, and either continue on
`main` or open a branch whose name matches the work.

---

## Findings raised while writing this document

*Per the #29 rails: behaviour that surprised me is reported here, not quietly fixed.*

**F-1 — `make regimes-backtest` did not run. → CLOSED in #30 Part A.** Its momentum comparison
book was 20 names across 4 sectors (13 Information Technology), which can hold at most 86%
under the caps, so once #28 made the caps fail closed it raised rather than returning a
cap-violating book. Selection now applies the pies' own ≤5-names/sector rule and the book is
cap-feasible by construction (7 sectors, largest 27.8%). The retracted column is still
published beside the repaired one, and tests pin **both**: that the new book satisfies the
caps, and that the old one still reproduces the exact published breach. The pie rows are
unchanged — asserted against the published #21 betas, not assumed.

**F-2 — `utils/` had no dedicated test file. → CLOSED (`5b56b44`).** It is the module set that
builds features and labels — the highest leakage-risk code in the repo — and it was covered
only indirectly. It was also missing from #25's own sweep-A directory list and from the Notion
documentation map (finding C-2): three separate maps of this project omitted the same
directory. Now covered by `test/test_utils_pipeline_f2.py` (24 tests, mutation-verified).
The *pattern* remains the finding worth keeping — a directory can be invisible to a
documentation map, an audit scope and a test suite simultaneously, and nothing complains.

**F-3 — the docstring deliverable was nearly a no-op, which is good news.** 60 of 66 Python
modules already carried a top-of-file docstring; the 6 without were all *empty*
`__init__.py` package markers. Those 6 now have one, so coverage is **66 of 66**, pinned by
`test/test_module_docstrings.py`. Reported because "added 6 docstrings" would otherwise read
as a thin result when the real finding is that the codebase was already documented.

**F-4 — dead code with no live purpose. → RESOLVED in #30 Part C, and I was half wrong.**
`FMPClient` (94 lines) is **deleted**: unreachable host, unused API key, and a second
implementation of the record parser that no test exercised. Git history preserves it.
But `portfolio/make_bundle.py` is **kept** — the reviewer's ruling, and correctly: it is not
dead, it regenerates the bundle the **live v1.1 Streamlit demo** loads, which is the README's
first link. Its docstring now says so, because the flag itself proved the file was easy to
mistake for the Cloudflare product's exporter.
