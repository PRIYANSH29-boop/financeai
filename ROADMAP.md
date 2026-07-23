# LOCKED ROADMAP — RankAlpha (cross-sectional momentum ranking)

_Locked 2026-06-15. See [CONCEPT.md](CONCEPT.md) for the why._

**What we're building:** a market-neutral long/short equity system that *ranks* S&P 500
stocks each month and trades the tails. ML is a **scoring function for ranking**, not a
price/direction predictor.

**v1 = Phases 1–6. Each phase gates the next. No skipping ahead.**

> **Status (2026-06-23): Phases 1–10 complete ✅. v1 shipped.**

1. **Data** — daily prices for the S&P 500 universe, ~7 years. Survivorship: v1 uses
   *current* constituents + a documented caveat. **Done when:** clean panel dataset saved.
2. **Features** — per stock/day: 3/6/12-month return, volatility, volume/liquidity, price
   (size), 1-month reversal. Normalize within each day. **Done when:** leakage-checked
   feature table.
3. **Labels** — forward return over holding window, ranked within each day. **Done when:**
   every row has a within-day rank label.
4. **Baseline (no ML)** — classical 12-month momentum, long top decile / short bottom,
   inverse-vol sized, monthly rebalance, costs. Purged time split. **Done when:** baseline
   Sharpe + Rank IC. _This is the number ML must beat._
5. **Ranking model** — LightGBM LambdaMART (`rank:` objective). Same portfolio machinery.
   **Done when:** model Rank IC + Sharpe vs baseline.
6. **Honest eval** — purged+embargoed CV, Rank IC, decile/quantile plot, after-cost Sharpe
   + max drawdown, vs baseline and market. **Done when:** every number defensible +
   limitations written.

**Phases 7–10 (after v1):**

7. ✅ Risk engine — long-only pie engine (inverse-vol + position-cap + vol-target).
8. ✅ LLM analyst layer — local Ollama `phi3:mini` explainer (template default, `USE_LLM=1`
   opt-in; hosted Groq swap noted). _Folded the product-page work in here._
9. ✅ Streamlit dashboard + live paper-trading track record — model frozen 2024-05-15, traded
   forward (single frozen model, never refit), marked to realized returns; idempotent
   `update_track`, "Track record" tab. Accepted 2026-06-23.
10. ✅ Ship — honest README rewrite (accurate architecture, headline numbers, embedded
    figures, quickstart), `requirements.txt`, MIT `LICENSE`, and removal of the aspirational
    legacy scaffold (XGBoost/RAG/Telegram stubs that nothing imported). Done 2026-06-23.

**Phases 12–18 (post-ship research track):**

12. ✅ Analyser — model-agnostic metrics library + charts, unit-tested.
13. ✅ Analyser scorecard on the frozen 23-month paper track.
14. ✅ Strategy Lab v0 — momentum vs momentum+low-vol. **VERDICT: KEEP.** At matched vol,
    low-vol earns ~the same return with half the drawdown; a shock absorber, not a Sharpe
    booster.
15. ✅ Beta-targeted pie engine — `portfolio/beta_engine.py::build_portfolio(capital,
    target_beta)`; hits the target beta by construction, caps impossible targets instead of
    faking them.
16. ✅ Universe expansion — `universe.py` builds a US mid+large-cap universe (1,200 names,
    market cap > $2B) from SEC registrant data + a yfinance liquidity screen; then
    `scripts/expand_universe.py` REBUILDS the panel/features/labels and **retrains** the
    ranker on it. A frozen model is only valid on the universe it was fit on, so this is a
    second frozen model, not a config swap. **VERDICT: pipeline retrains cleanly; the
    headline number is not trustworthy.** Sharpe rises 1.14 → 1.81 while Rank IC *falls*
    0.0505 → 0.0276 — payoff up, ranking skill down, which is the signature of
    survivorship-INCLUSION bias (names that were small in 2019 and compounded past the $2B
    floor sit in the panel from day one). The S&P 500 model stays the shipped one.
    Report: `figures/lab/universe_expansion.md`.
17. ✅ Fundamentals data audit — the GO/NO-GO gate before any fundamental factor. Seven
    checks (accuracy, coverage, point-in-time, outliers, consistency, survivorship,
    reproducibility) against **SEC EDGAR XBRL** rather than FMP: EDGAR is the primary
    source, carries the real `filed` publication date, is free and keyless, and was
    reachable when FMP was not. **VERDICT: GO** — 19,513 records, 100% carrying a
    publication date strictly after period end, max cross-source discrepancy 2.5%.
    Report: `figures/audit/fundamentals_audit.md`.
18. ✅ Value factor — earnings yield + book-to-market + EBITDA/EV + FCF yield, winsorized →
    z-scored → averaged, joined by publication date. **VERDICT: DROP.** It passes the
    independence test (corr with momentum −0.15 to −0.20) and cuts drawdown ~1.6–2.0pp, but
    Sharpe falls in all three windows tested including at matched vol — the return it gives
    up exceeds the risk it removes. Uncorrelated is necessary, not sufficient. Report:
    `figures/lab/value_factor.md`.

**Phase 6 experiment backlog (banked, build v1 first):**

- Sector/industry-neutral ranking (rank within sector to remove hidden sector bets).
  A/B test vs global ranking.
- News/NLP sentiment as an added feature (FinBERT). A/B test. ⚠️ High lookahead-leakage
  risk — strict point-in-time only.
