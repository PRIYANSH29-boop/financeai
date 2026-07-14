"""
Strategy Lab v0 — the survival-chain harness. Phase 14.

`run_strategy(spec)` takes a scoring *recipe* and runs it through the frozen RankAlpha
book-construction pipeline, returning a monthly returns Series. Everything about the book
— universe, rebalance schedule, embargo, top-N long-only selection, inverse-`vol_6m`
weights, weight cap, vol target, cash buffer, and 10 bps/side turnover cost — is imported
verbatim from `portfolio/paper_trade.py`, so the recipe is the ONLY moving part.

A spec is a dict::

    {
      "name": "Momentum + low-vol",
      "factors": [("mom_12_1m", True), ("vol_6m", False)],  # (column, higher_is_better)
      "combine": "rank_avg",     # equal-weight average of cross-sectional percentile ranks
      "long_only": True,
      "rebalance": "monthly",
    }

Factors are combined by **equal-weight average of cross-sectional percentile ranks** (NOT
a fitted model), so an improvement is attributable to the factor. `higher_is_better=False`
(e.g. volatility) means LOW values score high — the low-vol tilt.

The model stays frozen: this module never fits or tunes anything. The only place a fitted
model appears is `frozen_lgbm_score`, used purely to sanity-check that this harness's book
plumbing reproduces the committed paper track before we trust the A/B comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Frozen-track settings + book construction — imported, never redefined, so the harness is
# apples-to-apples with the committed paper track by construction.
from portfolio.paper_trade import (
    FREEZE_DATE, TOP_N, MAX_WEIGHT, TARGET_VOL, COST_PER_SIDE,
    _rebalance_dates,
)
from portfolio.engine import _cap_weights, _book_vol

LABELED_PATH = Path("data/sp500_labeled.parquet")
PANEL_PATH = Path("data/sp500_panel.parquet")
PERIODS_PER_YEAR = 12


# ------------------------------------------------------------------ factor scoring
def factor_score(day: pd.DataFrame, factors) -> pd.Series:
    """Equal-weight average of cross-sectional percentile ranks for one rebalance day.

    factors : list of (column, higher_is_better). Each column is percentile-ranked across
    the day's cross-section; `higher_is_better=False` flips it so LOW values rank high
    (the low-vol tilt). Returns a Series indexed by ticker; higher = more attractive.
    """
    ranks = []
    for col, higher_is_better in factors:
        r = day[col].rank(pct=True, ascending=bool(higher_is_better))
        ranks.append(r)
    score = pd.concat(ranks, axis=1).mean(axis=1)
    return pd.Series(score.values, index=day["ticker"].values)


def frozen_lgbm_score(day: pd.DataFrame, model) -> pd.Series:
    """Score a day with the frozen LGBM ranker — for the plumbing sanity check only."""
    from signals.lgbm_ranker import FEATURES
    return pd.Series(model.predict(day[FEATURES]), index=day["ticker"].values)


# ------------------------------------------------------------- one month, one book
def _build_and_mark(day: pd.DataFrame, score: pd.Series, panel, t, prev_w: pd.Series,
                    allow_leverage: bool = False):
    """Build the frozen-track product book from `score` and mark to realized fwd return.

    Mirrors `portfolio.paper_trade._build_and_mark` exactly (top-N long-only, inverse-vol
    weights, cap, vol target, cash buffer, turnover cost) — only the score source differs.

    allow_leverage : the frozen book only ever DE-risks (`k = min(1, target/book_vol)`),
        leaving low-vol books under the target with a cash buffer. With this flag the vol
        target is two-sided (`k = target/book_vol`), levering the book UP to hit the target
        so a low-vol book is compared at MATCHED risk. Borrowing earns/costs 0 (same as the
        cash convention) and per-name caps scale with leverage — an optimistic, documented
        simplification used only to separate factor efficiency from de-risking.
    """
    day = day.copy()
    day["score"] = score.reindex(day["ticker"].values).to_numpy()
    ranked = day.sort_values("score", ascending=False)

    holdings = ranked.head(TOP_N)                       # LONG-ONLY top names
    inv = 1.0 / holdings["vol_6m"]
    base = pd.Series((inv / inv.sum()).values, index=holdings["ticker"].values)
    capped = _cap_weights(base, MAX_WEIGHT)             # fully invested (sum=1)

    book_vol = _book_vol(capped.index, capped, panel, t)
    if book_vol > 0:
        k = TARGET_VOL / book_vol if allow_leverage else min(1.0, TARGET_VOL / book_vol)
    else:
        k = 1.0
    weights = capped * k                                # vol-targeted (levered if allowed)
    cash_w = float(1.0 - weights.sum())

    fwd = holdings.set_index("ticker")["fwd_ret_1m"]
    gross = float((weights * fwd.reindex(weights.index)).sum())   # cash earns 0

    tk = prev_w.index.union(weights.index)
    turnover = float((weights.reindex(tk, fill_value=0.0)
                      - prev_w.reindex(tk, fill_value=0.0)).abs().sum())
    cost = COST_PER_SIDE * turnover
    net = gross - cost
    bench = float(day["fwd_ret_1m"].mean())             # equal-weight universe

    row = {
        "date": pd.Timestamp(t), "net_ret": net, "gross_ret": gross, "cost": cost,
        "turnover": turnover, "bench_ret": bench, "invested_frac": float(weights.sum()),
        "cash_frac": cash_w, "book_vol": float(book_vol), "n_holdings": int(len(weights)),
    }
    return row, weights


# --------------------------------------------------------------------- run a spec
def monthly_rebalances(labeled, start=None, end=None, step: int = 21):
    """Non-overlapping ~monthly rebalance dates within [start, end] (every `step` days).

    Used for regime windows (2008 GFC, COVID) that don't key off the 2024 freeze date.
    """
    dates = np.sort(labeled["date"].unique())
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    return [pd.Timestamp(d) for d in dates[::step]]


def run_strategy(spec: dict, labeled=None, panel=None, score_fn=None,
                 allow_leverage: bool = False, rebalances=None) -> pd.DataFrame:
    """Run a strategy spec through the frozen book pipeline → per-month portfolio frame.

    spec : {name, factors, combine, long_only, rebalance}. Only `factors`/`combine` and
           `long_only` are honored here (rebalance is monthly, matching the frozen track).
    score_fn : optional override `score_fn(day) -> Series` (used by the plumbing check to
               inject the frozen LGBM score); otherwise the score comes from `factors`.
    allow_leverage : two-sided vol target (lever a low-vol book UP to 14%) for a
               matched-risk comparison. See `_build_and_mark`. Default False.
    rebalances : optional explicit list of rebalance dates (e.g. a regime window). Defaults
               to the frozen-track schedule `_rebalance_dates(labeled, FREEZE_DATE)`.

    Returns a DataFrame with the same columns as the committed paper track, one row per
    rebalance month.
    """
    if spec.get("combine", "rank_avg") != "rank_avg":
        raise ValueError(f"only 'rank_avg' combine is supported (got {spec.get('combine')!r})")
    if not spec.get("long_only", True):
        raise ValueError("Strategy Lab v0 is long-only (matches the pie product)")

    if labeled is None:
        labeled = pd.read_parquet(LABELED_PATH); labeled["date"] = pd.to_datetime(labeled["date"])
    if panel is None:
        panel = pd.read_parquet(PANEL_PATH); panel["date"] = pd.to_datetime(panel["date"])

    if score_fn is None:
        factors = spec["factors"]
        score_fn = lambda day: factor_score(day, factors)  # noqa: E731

    if rebalances is not None:
        rebals = [pd.Timestamp(d) for d in rebalances]
    else:
        rebals = [pd.Timestamp(d) for d in _rebalance_dates(labeled, FREEZE_DATE)]
    labeled_dates = set(labeled["date"].unique())
    rebals = [d for d in rebals if d in labeled_dates]

    rows, prev_w = [], pd.Series(dtype=float)
    for t in rebals:
        day = labeled[labeled["date"] == t]
        if len(day) < TOP_N:
            continue
        row, prev_w = _build_and_mark(day, score_fn(day), panel, t, prev_w, allow_leverage)
        rows.append(row)

    return pd.DataFrame(rows)


def strategy_returns(spec: dict, **kw) -> pd.Series:
    """Convenience: monthly net-return Series (date-indexed, named by the spec)."""
    pf = run_strategy(spec, **kw)
    return pd.Series(pf["net_ret"].to_numpy(),
                     index=pd.to_datetime(pf["date"]), name=spec["name"])


# ------------------------------------------------------- factor independence check
def signal_correlation(factors_a, factors_b, labeled=None) -> float:
    """Mean cross-sectional Spearman correlation between two factor scores.

    Averaged over the frozen-track rebalance dates. ~0 → the factors carry independent
    information; ~±1 → redundant. Used to judge whether low-vol adds anything to momentum.
    """
    if labeled is None:
        labeled = pd.read_parquet(LABELED_PATH); labeled["date"] = pd.to_datetime(labeled["date"])
    rebals = [pd.Timestamp(d) for d in _rebalance_dates(labeled, FREEZE_DATE)]
    corrs = []
    for t in rebals:
        day = labeled[labeled["date"] == t]
        if len(day) < TOP_N:
            continue
        a = factor_score(day, factors_a)
        b = factor_score(day, factors_b)
        corrs.append(a.corr(b, method="spearman"))
    return float(np.nanmean(corrs)) if corrs else float("nan")


# ----------------------------------------------------------------- strategy specs
MOMENTUM = {
    "name": "A · Momentum",
    "factors": [("mom_12_1m", True)],
    "combine": "rank_avg", "long_only": True, "rebalance": "monthly",
}

MOMENTUM_LOWVOL = {
    "name": "B · Momentum + low-vol",
    "factors": [("mom_12_1m", True), ("vol_6m", False)],
    "combine": "rank_avg", "long_only": True, "rebalance": "monthly",
}
