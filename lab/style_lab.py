#!/usr/bin/env python3
"""
Style & Season Lab — RankAlpha Phase 26 (Parts B + C).

The research question (Priyansh's thesis): our signal is tested on big, efficient stocks over
uniform periods. Does momentum behave differently by stock TYPE and by PERIOD — and is it
stronger in less-watched mid-caps?

This module answers it WITHOUT training anything. It classifies names into styles using rules
committed here in code, then measures Rank IC per (style x period) cell for two signals:

  * SIMPLE 12-1 momentum (`mom_12_1m`) — no ML, so any effect is attributable to the signal.
  * The FROZEN ML score — S&P 500 names ONLY, its valid universe. It is never refit here.

────────────────────────────────────────────────────────────────────────────────────────────
HONESTY RAILS (from the instruction; enforced in code, not just documented)
────────────────────────────────────────────────────────────────────────────────────────────
1. EVERY cell is reported with its n. Nothing is filtered out for looking bad.
2. Multiple testing: with 30+ cells several WILL look good by luck. A cell is a FINDING only
   at |t| >= 3 (`T_FINDING`); below that it is "suggestive" at most. `n_cells_tested` travels
   with every highlight — see `grid_summary`.
3. No intraday (no data, and no action point at a monthly rebalance). No bonds/commodities.
   No 15-day slicing — monthly data makes 15-day cells noise.
4. Survivorship carries over: both universes are CURRENT membership screens applied to all
   history. Styles are computed from current data applied backwards wherever the underlying
   fundamentals are missing. Every number here is DIRECTIONAL.

────────────────────────────────────────────────────────────────────────────────────────────
COVERAGE LIMIT — read before interpreting the census
────────────────────────────────────────────────────────────────────────────────────────────
`data/sec_fundamentals.parquet` covers only the ~501 S&P names, so the three fundamentals-gated
legs — GROWTH, VALUE, and the "no earnings" leg of SPECULATIVE — are computable ONLY for names
with a SEC ledger. Wide-universe names without one are reported as `unclassifiable_fundamental`
rather than silently defaulted into or out of those styles. Defaulting them would invent a
census. Additionally there is no `revenue` column in the committed ledger, so GROWTH is EPS
growth only, not revenue growth — the instruction asked for "revenue/earnings growth" and only
the earnings half is available offline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("style_lab")

# ── committed rule constants (no per-name tuning; all cross-sectional percentiles) ──────────
TOP_TERCILE = 2 / 3           # "high" = at or above this cross-sectional percentile
BOT_TERCILE = 1 / 3           # "low"  = at or below this
TOP_QUINTILE = 0.8            # "mega-cap" for the blue-chip rule
BLUE_CHIP_MIN_MONTHS = 60     # >= 5y of monthly history
DEFENSIVE_MAX_BETA = 1.0      # a defensive name must actually be low-beta, not just low-beta sector
MAX_LABELS = 2                # a name may hold at most 2 styles
T_FINDING = 3.0               # |t| >= 3 is a FINDING; below is suggestive at best
EARNINGS_MONTHS = (1, 4, 7, 10)   # the calendar reporting months — committed, not fitted
PPY = 12

CYCLICAL_SECTORS = {"Industrials", "Consumer Cyclical", "Energy", "Basic Materials"}
DEFENSIVE_SECTORS = {"Consumer Defensive", "Utilities", "Healthcare"}

# The two universes carry DIFFERENT sector taxonomies — the wide file is yfinance
# ("Consumer Cyclical", "Healthcare", "Basic Materials"), the S&P file is GICS
# ("Consumer Discretionary", "Health Care", "Materials"). Applying a sector rule to the raw
# strings silently matches only the overlap and undercounts every sector-keyed style on one
# side. This is #20's mixed-taxonomy bug, so everything is normalised to the yfinance labels
# used above BEFORE any rule is evaluated.
SECTOR_ALIASES = {
    "Health Care": "Healthcare",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Materials": "Basic Materials",
    "Information Technology": "Technology",
    "Financials": "Financial Services",
    "Telecommunication Services": "Communication Services",
}


def normalize_sector(s: pd.Series) -> pd.Series:
    """Map any supported taxonomy onto the canonical (yfinance) labels the rules are written in."""
    return s.astype(object).replace(SECTOR_ALIASES)


# The #24 share-gap recovery reopened the universe to non-equities: an ETF trust has no
# `EntityCommonStockSharesOutstanding` in the SEC frames endpoint, so it fell into the gap, and
# the price provider answers `sharesOutstanding` for it happily. SPY/QQQ/DIA/MDY then cleared
# the $2B and liquidity screens trivially — SPY entered as the single largest "name" in the
# 1,200. They are index funds, not stocks: they carry no sector, and a market proxy sitting
# inside a cross-sectional momentum ranking is a contaminant, not a name. Excluded here by
# SEC-registered entity name, which is the field that actually identifies them.
NON_EQUITY_NAME_PATTERNS = ("ETF TRUST", "ETF, SERIES", "TRUST, SERIES", "INDEX TRUST",
                            "INDEX FUND", " ETF")


def is_non_equity(entity_names: pd.Series) -> pd.Series:
    """True where the SEC-registered entity name identifies a fund/trust rather than a company."""
    up = entity_names.astype(str).str.upper()
    hit = pd.Series(False, index=entity_names.index)
    for pat in NON_EQUITY_NAME_PATTERNS:
        hit |= up.str.contains(pat, regex=False, na=False)
    return hit

STYLES = ["growth", "value", "dividend", "blue_chip", "cyclical", "defensive", "speculative"]

# Priority when a name qualifies for more than MAX_LABELS styles. Ordered most-specific first:
# a rule keyed on a name's own fundamentals says more about it than a sector bucket does.
STYLE_PRIORITY = ["speculative", "blue_chip", "value", "growth", "dividend",
                  "defensive", "cyclical"]


# ══════════════════════════════════════════════════════════════════ per-name characteristics
def dividend_yield(panel: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """Trailing dividend yield per ticker, derived from the committed panel alone.

    `close` and `adj_close` are both split-adjusted, but only `adj_close` is dividend-adjusted.
    Their ratio therefore drifts by exactly the cumulative dividend factor, so the growth of
    (adj_close / close) over a window is the dividend return over that window. This needs no
    vendor dividend feed — it falls out of two columns we already committed.
    """
    p = panel.sort_values(["ticker", "date"])
    r = (p["adj_close"] / p["close"]).replace([np.inf, -np.inf], np.nan)
    p = p.assign(_ratio=r).dropna(subset=["_ratio"])
    out = {}
    for tk, g in p.groupby("ticker", sort=False):
        s = g["_ratio"].tail(lookback + 1)
        if len(s) < 2 or s.iloc[0] <= 0:
            continue
        out[tk] = float(s.iloc[-1] / s.iloc[0] - 1.0)
    return pd.Series(out, name="div_yield", dtype="float64")


def price_characteristics(panel: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Per-ticker ann vol, beta vs an equal-weight proxy, and months of history."""
    wide = (panel.pivot_table(index="date", columns="ticker", values="adj_close")
                 .sort_index().resample("ME").last())
    rets = wide.pct_change()
    bench = rets.mean(axis=1, skipna=True) if benchmark is None else benchmark
    rows = {}
    for tk in rets.columns:
        r = rets[tk].dropna()
        if len(r) < 2:
            continue
        b = bench.reindex(r.index)
        ok = r.notna() & b.notna()
        var = float(b[ok].var(ddof=0)) if ok.sum() >= 2 else np.nan
        beta = float(r[ok].cov(b[ok]) / var) if var and var > 0 else np.nan
        rows[tk] = {"ann_vol": float(r.std(ddof=0) * np.sqrt(PPY)),
                    "beta": beta, "n_months": int(len(r))}
    return pd.DataFrame.from_dict(rows, orient="index")


def eps_growth(fund: pd.DataFrame, min_quarters: int = 8) -> pd.Series:
    """Latest TTM EPS vs the prior TTM, per ticker, from the committed SEC ledger.

    The ledger has no `revenue` column, so this is the earnings half of the instruction's
    "revenue/earnings growth" and the report says so.
    """
    f = fund.dropna(subset=["eps"]).sort_values(["ticker", "publication_date", "period_end"])
    out = {}
    for tk, g in f.groupby("ticker", sort=False):
        e = g["eps"].to_numpy(dtype="float64")
        if len(e) < min_quarters:
            continue
        recent, prior = e[-4:].sum(), e[-8:-4].sum()
        if prior <= 0:                      # growth off a negative base is not a number
            continue
        out[tk] = float(recent / prior - 1.0)
    return pd.Series(out, name="eps_growth", dtype="float64")


def latest_eps(fund: pd.DataFrame) -> pd.Series:
    """Latest TTM EPS per ticker — the 'no earnings' leg of the speculative rule."""
    f = fund.dropna(subset=["eps"]).sort_values(["ticker", "publication_date", "period_end"])
    return (f.groupby("ticker")["eps"].apply(lambda s: float(s.tail(4).sum()))
             .rename("ttm_eps"))


# ══════════════════════════════════════════════════════════════════════════ classification
def _pct(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile in (0, 1]. NaN stays NaN — never treated as 0."""
    return s.rank(pct=True)


def classify(chars: pd.DataFrame) -> pd.DataFrame:
    """Assign <= MAX_LABELS styles per name from `chars`, using only the committed rules.

    `chars` needs: market_cap, ann_vol, beta, n_months, sector, and optionally div_yield,
    eps_growth, ttm_eps, value_score. Missing fundamentals do NOT default a name into or out
    of a fundamentals-gated style — the style is simply not evaluable for that name, and the
    census reports it as such.
    """
    c = chars.copy()
    p_cap = _pct(c["market_cap"]) if "market_cap" in c else pd.Series(np.nan, index=c.index)
    p_vol = _pct(c["ann_vol"]) if "ann_vol" in c else pd.Series(np.nan, index=c.index)
    p_div = _pct(c["div_yield"]) if "div_yield" in c else pd.Series(np.nan, index=c.index)
    p_gro = _pct(c["eps_growth"]) if "eps_growth" in c else pd.Series(np.nan, index=c.index)
    # value_score is a z-score where HIGHER = cheaper (#18 convention), so top tercile = value.
    p_val = _pct(c["value_score"]) if "value_score" in c else pd.Series(np.nan, index=c.index)

    sector = normalize_sector(c["sector"]) if "sector" in c else pd.Series("?", index=c.index)
    beta = c["beta"] if "beta" in c else pd.Series(np.nan, index=c.index)
    n_mo = c["n_months"] if "n_months" in c else pd.Series(0, index=c.index)
    eps = c["ttm_eps"] if "ttm_eps" in c else pd.Series(np.nan, index=c.index)

    flags = pd.DataFrame(index=c.index)
    flags["growth"] = p_gro >= TOP_TERCILE
    flags["value"] = p_val >= TOP_TERCILE
    flags["dividend"] = p_div >= TOP_TERCILE
    flags["blue_chip"] = (p_cap >= TOP_QUINTILE) & (p_vol <= BOT_TERCILE) & (n_mo >= BLUE_CHIP_MIN_MONTHS)
    flags["cyclical"] = sector.isin(CYCLICAL_SECTORS)
    flags["defensive"] = sector.isin(DEFENSIVE_SECTORS) & (beta < DEFENSIVE_MAX_BETA)
    # Speculative: high vol AND small-end AND (no earnings, where earnings are known at all).
    no_earnings = eps.isna() | (eps <= 0)
    flags["speculative"] = (p_vol >= TOP_TERCILE) & (p_cap <= BOT_TERCILE) & no_earnings

    flags = flags.fillna(False).astype(bool)

    # Cap at MAX_LABELS by the committed priority order — deterministic, never by eyeball.
    kept = []
    for tk, row in flags.iterrows():
        on = [s for s in STYLE_PRIORITY if bool(row.get(s, False))][:MAX_LABELS]
        kept.append({"ticker": tk, "styles": on, "n_styles": len(on)})
    out = pd.DataFrame(kept).set_index("ticker")
    for s in STYLES:
        out[s] = out["styles"].apply(lambda lst, s=s: s in lst)

    # Evaluability — which fundamentals-gated styles could even be judged for this name.
    out["fundamentals_available"] = (~p_gro.isna()) | (~p_val.isna()) | (~eps.isna())
    return out.join(c, how="left")


def census(classified: pd.DataFrame) -> pd.DataFrame:
    """Names per style + overlap counts, with the unclassifiable tail stated explicitly."""
    rows = []
    for s in STYLES:
        sel = classified[classified[s]]
        rows.append({"style": s, "n_names": int(len(sel)),
                     "pct_of_universe": round(100 * len(sel) / max(len(classified), 1), 1)})
    df = pd.DataFrame(rows).set_index("style")
    return df


def overlap_matrix(classified: pd.DataFrame) -> pd.DataFrame:
    m = pd.DataFrame(0, index=STYLES, columns=STYLES, dtype=int)
    for a in STYLES:
        for b in STYLES:
            m.loc[a, b] = int((classified[a] & classified[b]).sum())
    return m


# ══════════════════════════════════════════════════════════════════════════════ the grid
def rank_ic_by_date(df: pd.DataFrame, signal: str, fwd: str = "fwd_ret_1m") -> pd.Series:
    """Spearman rank IC within each date. Dates with < 5 names are dropped (not silently —
    the caller reports n)."""
    out = {}
    for d, g in df.groupby("date", sort=True):
        g = g[[signal, fwd]].dropna()
        if len(g) < 5:
            continue
        out[d] = float(g[signal].corr(g[fwd], method="spearman"))
    return pd.Series(out, dtype="float64").sort_index()


def cell_stats(ic: pd.Series) -> dict:
    """mean IC, t-stat, n periods. t = mean / (sd / sqrt(n)) — the standard IC t-test."""
    ic = ic.dropna()
    n = int(len(ic))
    if n == 0:
        return {"mean_ic": np.nan, "t": np.nan, "n_months": 0, "verdict": "empty"}
    mean = float(ic.mean())
    sd = float(ic.std(ddof=1)) if n > 1 else np.nan
    t = float(mean / (sd / np.sqrt(n))) if sd and sd > 0 and n > 1 else np.nan
    if np.isfinite(t) and abs(t) >= T_FINDING:
        verdict = "FINDING"
    elif np.isfinite(t) and abs(t) >= 2:
        verdict = "suggestive"
    else:
        verdict = "noise"
    return {"mean_ic": mean, "t": t, "n_months": n, "verdict": verdict}


def month_end_slice(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, month) at the LAST available date in each month — the monthly
    rebalance point. Daily rows would count the same bet 21 times and fake the t-stat."""
    d = df.copy()
    d["_ym"] = d["date"].dt.to_period("M")
    last = d.groupby("_ym")["date"].transform("max")
    return d[d["date"] == last].drop(columns="_ym")


def build_grid(labeled: pd.DataFrame, classified: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Rank IC for every (style x period) cell. EVERY cell is emitted, including empty ones."""
    df = month_end_slice(labeled)
    df = df.merge(classified[STYLES].reset_index(), on="ticker", how="inner")
    df["year"] = df["date"].dt.year
    df["earnings_season"] = df["date"].dt.month.isin(EARNINGS_MONTHS)

    years = sorted(df["year"].unique())
    periods = [("full window", lambda x: x)]
    periods += [(str(y), (lambda x, y=y: x[x["year"] == y])) for y in years]
    periods += [("earnings months", lambda x: x[x["earnings_season"]]),
                ("non-earnings months", lambda x: x[~x["earnings_season"]])]

    rows = []
    for style in STYLES:
        cohort = df[df[style]]
        for pname, pfn in periods:
            sub = pfn(cohort)
            st = cell_stats(rank_ic_by_date(sub, signal))
            rows.append({"style": style, "period": pname, "n_names": int(sub["ticker"].nunique()),
                         **st})
    # The all-styles row is the control: if a "style effect" matches it, there is no style effect.
    for pname, pfn in periods:
        sub = pfn(df)
        st = cell_stats(rank_ic_by_date(sub, signal))
        rows.append({"style": "ALL (control)", "period": pname,
                     "n_names": int(sub["ticker"].nunique()), **st})
    return pd.DataFrame(rows)


def grid_summary(grid: pd.DataFrame) -> dict:
    """Findings with the multiple-testing context attached — never a highlight on its own."""
    tested = int(grid["t"].notna().sum())
    findings = grid[(grid["t"].abs() >= T_FINDING) & grid["t"].notna()]
    suggestive = grid[(grid["t"].abs() >= 2) & (grid["t"].abs() < T_FINDING)]
    return {
        "n_cells_total": int(len(grid)),
        "n_cells_tested": tested,
        "n_findings": int(len(findings)),
        "n_suggestive": int(len(suggestive)),
        "expected_false_at_t2": round(0.0455 * tested, 1),
        "findings": findings.sort_values("t", key=lambda s: s.abs(), ascending=False),
    }
