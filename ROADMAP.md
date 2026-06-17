# LOCKED ROADMAP — RankAlpha (cross-sectional momentum ranking)

_Locked 2026-06-15. See [CONCEPT.md](CONCEPT.md) for the why._

**What we're building:** a market-neutral long/short equity system that *ranks* S&P 500
stocks each month and trades the tails. ML is a **scoring function for ranking**, not a
price/direction predictor.

**v1 = Phases 1–6. Each phase gates the next. No skipping ahead.**

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

7. Risk engine
8. LLM analyst layer (Groq)
9. Streamlit dashboard + live paper-trading track record
10. Ship (README, methodology, demo, applications)

**Phase 6 experiment backlog (banked, build v1 first):**

- Sector/industry-neutral ranking (rank within sector to remove hidden sector bets).
  A/B test vs global ranking.
- News/NLP sentiment as an added feature (FinBERT). A/B test. ⚠️ High lookahead-leakage
  risk — strict point-in-time only.
