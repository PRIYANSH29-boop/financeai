#!/usr/bin/env python3
"""
Style & Season report — RankAlpha Phase 26 Part B (census) + Part C (the grid).

Assembles per-name characteristics from committed data only, classifies with the rules in
`lab.style_lab`, and writes `figures/lab/style_season_report.md` with the FULL grids.

Nothing is trained here and the frozen model is never refit — the ML grid reads the committed
walk-forward OOS scores (`data/sp500_oos_walkforward.parquet`).

    python scripts/style_season_report.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lab.style_lab import (  # noqa: E402
    STYLES, T_FINDING, EARNINGS_MONTHS, BLUE_CHIP_MIN_MONTHS,
    dividend_yield, price_characteristics, eps_growth, latest_eps,
    classify, census, overlap_matrix, build_grid, grid_summary, is_non_equity,
)
import lab.value_factor as vf  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("style_season")

OUT_MD = Path("figures/lab/style_season_report.md")

WIDE_UNIVERSE = Path("data/universe_midlarge.csv")
WIDE_SECTORS = Path("data/universe_midlarge_sectors.csv")
WIDE_PANEL = Path("data/midlarge_panel.parquet")
WIDE_LABELED = Path("data/midlarge_labeled.parquet")

SP_PANEL = Path("data/sp500_panel.parquet")
SP_TICKERS = Path("data/sp500_tickers.csv")
SP_OOS = Path("data/sp500_oos_walkforward.parquet")

FUND = Path("data/sec_fundamentals.parquet")


# ─────────────────────────────────────────────────────────────────── characteristics assembly
def value_scores(fund: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """The #18 value composite struck at the panel's last date (higher = cheaper)."""
    last = panel["date"].max()
    px = (panel.loc[panel["date"] == last, ["ticker", "close"]]
               .dropna().set_index("ticker")["close"])
    ratios = vf.ratios_on(fund, px, last)
    if ratios.empty:
        return pd.Series(dtype="float64", name="value_score")
    return vf.composite(ratios)


def build_chars(panel: pd.DataFrame, caps: pd.Series, sectors: pd.Series,
                fund: pd.DataFrame | None) -> pd.DataFrame:
    """Per-name characteristics for one universe, from committed data only."""
    logger.info("price characteristics for %d tickers…", panel["ticker"].nunique())
    chars = price_characteristics(panel)
    chars["div_yield"] = dividend_yield(panel)
    chars["market_cap"] = caps.reindex(chars.index)
    chars["sector"] = sectors.reindex(chars.index).fillna("?")
    if fund is not None and len(fund):
        chars["eps_growth"] = eps_growth(fund).reindex(chars.index)
        chars["ttm_eps"] = latest_eps(fund).reindex(chars.index)
        chars["value_score"] = value_scores(fund, panel).reindex(chars.index)
    return chars


def _md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "—" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    rule = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [fmt(v) for v in df.loc[i]]) + " |" for i in df.index]
    return "\n".join([head, rule, *rows])


def _grid_table(grid: pd.DataFrame) -> str:
    """EVERY cell, with n. Honesty rail 1 — nothing is dropped for looking bad or being empty."""
    g = grid.copy()
    g["mean_ic"] = g["mean_ic"].map(lambda v: "—" if not np.isfinite(v) else f"{v:+.4f}")
    g["t"] = g["t"].map(lambda v: "—" if not np.isfinite(v) else f"{v:+.2f}")
    lines = ["| style | period | n names | n months | mean Rank IC | t | verdict |",
             "|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        lines.append(f"| {r['style']} | {r['period']} | {r['n_names']} | {r['n_months']} | "
                     f"{r['mean_ic']} | {r['t']} | {r['verdict']} |")
    return "\n".join(lines)


def main() -> int:
    # ── wide universe ────────────────────────────────────────────────────────────────────
    uni = pd.read_csv(WIDE_UNIVERSE)
    # Drop index funds the #24 recovery let in (SPY/QQQ/DIA/MDY) — see `is_non_equity`.
    _sic_cache = Path("data/cache/sector_sic_cache.json")
    _sics = (uni["cik"].astype(str).map(__import__("json").loads(_sic_cache.read_text()))
             if _sic_cache.exists() else None)
    etf_mask = is_non_equity(uni["name"], _sics)
    etfs = sorted(uni.loc[etf_mask, "ticker"])
    if etfs:
        logger.warning("excluding %d non-equity issuers from the universe: %s", len(etfs), etfs)
        uni = uni[~etf_mask]
    caps_w = uni.set_index("ticker")["market_cap"]
    sec_w = pd.read_csv(WIDE_SECTORS).set_index("ticker")["sector"]
    panel_w = pd.read_parquet(WIDE_PANEL, columns=["date", "ticker", "close", "adj_close"])
    panel_w["date"] = pd.to_datetime(panel_w["date"])
    panel_w = panel_w[panel_w["ticker"].isin(set(uni["ticker"]))]
    fund = pd.read_parquet(FUND) if FUND.exists() else None
    if fund is not None:
        fund["publication_date"] = pd.to_datetime(fund["publication_date"])
        fund["period_end"] = pd.to_datetime(fund["period_end"])

    chars_w = build_chars(panel_w, caps_w, sec_w, fund)
    cls_w = classify(chars_w)

    # ── S&P 500 ──────────────────────────────────────────────────────────────────────────
    panel_s = pd.read_parquet(SP_PANEL, columns=["date", "ticker", "close", "adj_close"])
    panel_s["date"] = pd.to_datetime(panel_s["date"])
    sp_meta = pd.read_csv(SP_TICKERS)
    tcol = sp_meta.columns[0]
    sec_s = (sp_meta.set_index(tcol)["sector"] if "sector" in sp_meta.columns
             else pd.Series(dtype=object))
    # market cap for the S&P set: shares from the ledger x last close, else fall back to wide.
    caps_s = caps_w.reindex(sorted(set(panel_s["ticker"])))
    chars_s = build_chars(panel_s, caps_s, sec_s, fund)
    cls_s = classify(chars_s)

    # ── grids ────────────────────────────────────────────────────────────────────────────
    lab_w = pd.read_parquet(WIDE_LABELED, columns=["date", "ticker", "mom_12_1m", "fwd_ret_1m"])
    lab_w["date"] = pd.to_datetime(lab_w["date"])
    lab_w = lab_w[lab_w["ticker"].isin(set(uni["ticker"]))]
    logger.info("momentum grid on the wide universe (%d rows)…", len(lab_w))
    grid_mom_w = build_grid(lab_w, cls_w, "mom_12_1m")

    lab_s = pd.read_parquet(SP_OOS, columns=["date", "ticker", "mom_12_1m", "model_score",
                                             "fwd_ret_1m"])
    lab_s["date"] = pd.to_datetime(lab_s["date"])
    logger.info("momentum + ML grids on the S&P OOS window (%d rows)…", len(lab_s))
    grid_mom_s = build_grid(lab_s, cls_s, "mom_12_1m")
    grid_ml_s = build_grid(lab_s, cls_s, "model_score")

    s_mom_w, s_mom_s, s_ml_s = (grid_summary(g) for g in (grid_mom_w, grid_mom_s, grid_ml_s))
    total_tested = s_mom_w["n_cells_tested"] + s_mom_s["n_cells_tested"] + s_ml_s["n_cells_tested"]

    # ── report ───────────────────────────────────────────────────────────────────────────
    L: list[str] = []
    A = L.append
    A("# Style & Season Lab — where does momentum actually live? (Phase 26)\n")
    A("⚠️ **EDUCATIONAL SIMULATION.** Both universes are CURRENT membership screens applied to "
      "all history, so every number here is survivorship-biased and **DIRECTIONAL**. Styles are "
      "computed from current data applied backwards. Nothing here is a forecast, and nothing "
      "was trained — the frozen model is read, never refit.\n")
    A(f"**Multiple-testing bar:** a cell is a FINDING only at **|t| ≥ {T_FINDING:g}**. Below that "
      f"it is *suggestive* at best. **{total_tested} cells were tested in total** across the three "
      f"grids; at the |t| ≥ 2 level alone you would expect ~{0.0455 * total_tested:.1f} false "
      "positives by luck. Read every highlight against that number.\n")

    A("\n## Coverage limit — read before the census\n")
    A(f"`data/sec_fundamentals.parquet` covers **{0 if fund is None else fund['ticker'].nunique()} "
      f"tickers**, not the full wide universe, and it has **no `revenue` column**. So:\n")
    A("- **GROWTH** is *EPS* growth, not revenue growth — only the earnings half of the "
      "instruction's \"revenue/earnings growth\" exists offline.")
    A("- **GROWTH**, **VALUE** and the *no-earnings* leg of **SPECULATIVE** are evaluable only "
      "for names with a SEC ledger. Names without one are **not** defaulted into or out of "
      "those styles — that would invent a census. They are reported as not evaluable.")
    A(f"- Wide universe: **{int(cls_w['fundamentals_available'].sum())} of {len(cls_w)}** names "
      f"have usable fundamentals. S&P: **{int(cls_s['fundamentals_available'].sum())} of "
      f"{len(cls_s)}**.\n")

    A("\n## Part A finding — index funds in the stock universe\n")
    A(f"The #24 share-gap recovery reopened the universe to **non-equities**: an ETF trust "
      f"has no share count in the SEC `frames` endpoint, so it landed in the 2,303-name gap, "
      f"and the price provider answered `sharesOutstanding` for it. **{len(etfs)} index funds "
      f"({', '.join(etfs)}) cleared the $2B and liquidity screens** — SPY entered as the "
      f"single largest name in the 1,200. They are excluded from everything below by SEC "
      f"entity name (`lab.style_lab.is_non_equity`). **`universe.py` itself is NOT fixed** — "
      f"the committed universe CSV still contains them until a rebuild with a permanent "
      f"exclusion rule, which needs its own instruction.\n")
    A("\n## Part B — the style census\n")
    A("Rules are committed in `lab/style_lab.py` as cross-sectional percentile thresholds, so "
      "no name is hand-picked. A name may hold at most 2 labels, resolved by a fixed priority "
      "order (most-specific first), never by eyeball.\n")
    A("### Wide universe\n")
    A(_md_table(census(cls_w), "{:.1f}"))
    A(f"\n*{int((cls_w['n_styles'] == 0).sum())} of {len(cls_w)} names carry no style at all "
      f"(they clear no rule); {int((cls_w['n_styles'] == 2).sum())} carry the maximum 2.*\n")
    A("\n### S&P 500\n")
    A(_md_table(census(cls_s), "{:.1f}"))
    A(f"\n*{int((cls_s['n_styles'] == 0).sum())} of {len(cls_s)} names carry no style; "
      f"{int((cls_s['n_styles'] == 2).sum())} carry the maximum 2.*\n")
    A("\n### Style overlaps — wide universe\n")
    A(_md_table(overlap_matrix(cls_w), "{:.0f}"))

    A("\n\n## Part C — the grids (every cell, with n)\n")
    for title, grid, summ, note in [
        ("C1 · Simple 12-1 momentum — WIDE universe (the thesis test)", grid_mom_w, s_mom_w,
         "The mid-cap question lives here: if momentum is stronger in less-watched names, the "
         "speculative/small-end cohorts should beat the blue-chip cohort and the ALL control."),
        ("C2 · Simple 12-1 momentum — S&P 500 OOS window", grid_mom_s, s_mom_s,
         "The same signal on the big efficient names, over the model's OOS window only — the "
         "like-for-like comparison against C1."),
        ("C3 · FROZEN ML score — S&P 500 only (its valid universe)", grid_ml_s, s_ml_s,
         "The model is never scored outside the cross-section it was fit on."),
    ]:
        A(f"\n### {title}\n")
        A(note + "\n")
        A(f"*{summ['n_cells_tested']} testable cells · {summ['n_findings']} at |t| ≥ "
          f"{T_FINDING:g} · {summ['n_suggestive']} suggestive (2 ≤ |t| < {T_FINDING:g}) · "
          f"~{summ['expected_false_at_t2']} expected false at |t| ≥ 2 by luck.*\n")
        A(_grid_table(grid))

    A("\n\n## Honesty rails applied\n")
    A(f"- Every cell above is printed with its n. No cell was dropped for being empty or weak.")
    A(f"- `ALL (control)` rows are the null: a style effect that matches the control is not a "
      f"style effect.")
    A(f"- Monthly rebalance rows only (`month_end_slice`) — daily rows would count the same bet "
      f"~21 times and inflate every t-stat.")
    A(f"- Earnings season = calendar months {list(EARNINGS_MONTHS)}, committed in code, not fitted.")
    A(f"- No intraday, no bonds/commodities, no 15-day slicing — all out of scope by instruction.")
    A(f"- Blue-chip requires ≥ {BLUE_CHIP_MIN_MONTHS} months of history, so it cannot be earned "
      f"by a recent listing.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L))
    logger.info("wrote %s", OUT_MD)

    print(f"\nWide census:\n{census(cls_w).to_string()}")
    print(f"\nS&P census:\n{census(cls_s).to_string()}")
    for name, s in [("momentum/wide", s_mom_w), ("momentum/S&P", s_mom_s), ("ML/S&P", s_ml_s)]:
        print(f"\n{name}: {s['n_cells_tested']} tested · {s['n_findings']} findings "
              f"(|t|>={T_FINDING:g}) · {s['n_suggestive']} suggestive")
        if len(s["findings"]):
            print(s["findings"][["style", "period", "n_months", "mean_ic", "t"]].to_string(index=False))
    print(f"\nReport: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
