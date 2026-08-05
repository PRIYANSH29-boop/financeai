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

**Phases 19–26 (product + audit track):** *(added in #28 — the roadmap stopped at 18 while
the work ran to 26, so these existed only in commit messages and `figures/`.)*

19. ✅ Static web product — **A:** `scripts/export_web_bundle.py` precomputes every number the
    frontend can display; **B:** `web/`, a backend-free Next.js static export. Hard rule: no
    number is typed into the frontend; the browser only multiplies weights by capital. Built
    on the SHIPPED S&P 500 beta engine, explicitly not the #16 mid+large model. Live at
    https://rankalpha.pages.dev via `make deploy`.
20. ✅ Wide-universe sector mapping (yfinance + SEC SIC). 100% coverage, and the pie's sector
    caps went from inert to **binding** (Health 7→5, Tech 6→5). Report:
    `figures/lab/sector_mapping.md`.
21. ✅ Regime-segmented backtest — slice the committed history into calm/normal/stressed.
    **Beta drift measured:** the β0.75 pie realises 0.26 in calm months and 0.78 in stress.
    In-sample, 2022 is the sole stress episode — directional only. Report:
    `figures/lab/regime_report.md`. **Corrected in #30:** the momentum comparison book was
    selected without a per-sector name limit and breached the sector cap (Info Tech 44% vs
    30%); it is rebuilt under the pies' own ≤5-names/sector rule, with the retracted column
    still published beside it. The pie rows were unaffected and are unchanged.
22. ✅ Ship-ready riders — README ship-window, live demo + artifact links.
23. ✅ Frontend v2 — Explore (1,200-name browser), Basket (client-side equal-weight
    scorecard), pie label upgrades. Deployed.
24. ✅ Explore data hotfix — honest `as_of` (the true data date, never the resampled
    month-end label, which was a FUTURE date on a live page), a sanity band for artifact
    stats, and real basket eligibility gated in one place.
25. ✅ Adversarial self-audit — `AUDIT_FINDINGS.md`, 16 suspicions (6×S1, 6×S2, 4×S3) raised
    as suspicions for triage, deliberately with **no fixes** in the same commit.
26. ✅ The forest expedition — wide-panel rebuild + Style & Season Lab. **Headline result is
    a NULL:** 208 cells tested at a |t| ≥ 3 bar; momentum does not live in a particular style
    or season. A null grid is a real research result and is kept. Riders 26b/26c/26d excluded
    non-equities from the universe (SIC + name rule), rebuilt the forest (499/503 S&P, zero
    non-equities), and filtered commodity/crypto trusts out of the shipped Explore bundle.
    Report: `figures/lab/style_season_report.md`.
27. ✅ **The Signal Duel** — frozen ML vs plain 12-1 momentum, identical construction, only
    the score varying, scored on the walk-forward OOS frame (47 months). **VERDICT: TRADE
    MOMENTUM, and it is CONCLUSIVE.** After-cost Sharpe 1.79 (momentum) vs 1.41 (ML); the
    ML's Rank IC edge is real on the full window (0.0505 vs 0.0414) but rests on 2 of 5
    years, so it fails the pre-stated consistency clause. Conclusive because the A-1
    contamination *favours* the ML — it lost while carrying an advantage, so a clean retrain
    can only widen the gap. The ML also turns over LESS (30% vs 48%), so the gap is
    selection, not cost. Report: `figures/lab/signal_duel.md`.
28. ✅ **Audit fix-pass one** (Part A) — the six #25 findings the reviewer accepted as the fix
    list, each with tests. **A-3:** the partial final month is now disclosed, not just
    flagged (the exporter ships the sentence; a validator refuses a raised flag with nothing
    to render; one helper renders it on all three surfaces). **A-2:** a missing split basis
    RAISES — an empty split map can only mean "fetched, none exist". **B-1/B-2/B-3/B-4:** the
    caps fail CLOSED, with an up-front joint-capacity check; shipped pies verified
    bit-identical (max |Δw| = 0.0). **D-1:** all deps pinned exactly. **D-2:** `make
    panel/features/labels/pipeline`, and deliberately no train target. **S3 sweep** cleared.
    Tests 190 → 243. **Finding:** B-1/B-2 was not latent — #21's momentum book is
    cap-infeasible (86% max investable) and its published numbers came from a book with
    Information Technology at 44% against a 30% cap. `make regimes-backtest` now refuses.
29. 🔄 **The complete project description** — `ARCHITECTURE.md` (every directory, every
    module, the make-target table, the invariants, the branch map), module docstrings, and
    this roadmap + the README brought in sync. Documentation only, zero behaviour changes.

**Phase 6 experiment backlog (banked, build v1 first):**

- Sector/industry-neutral ranking (rank within sector to remove hidden sector bets).
  A/B test vs global ranking.
- News/NLP sentiment as an added feature (FinBERT). A/B test. ⚠️ High lookahead-leakage
  risk — strict point-in-time only.
