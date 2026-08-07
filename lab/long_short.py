"""
The long/short research sleeve — #31 Arm 3.

⚠️ EDUCATIONAL SIMULATION. **This is research, not the product.** The retail pie stays
long-only by product definition (#30 ruling); nothing here competes for the product engine.
It competes only for a place on the site as a second, clearly-labelled research track.

Why this arm exists
-------------------
The #30 duel found the ML loses the long-only product book while having the HIGHER Rank IC,
and the short-sleeve diagnostic located the reason: the ML orders the whole cross-section,
and roughly half that skill sits in the bottom decile — which a 20-name long-only book
cannot reach. This arm gives the signal its natural vehicle and measures what it does there.

Construction
------------
`signals.baseline_momentum.backtest_scores` verbatim: long the top decile, short the bottom
decile, inverse-vol weights within each leg normalised to |sum| = 1 (so ±100% per leg, 200%
gross, market-neutral by construction), monthly rebalance, 10 bps/side on turnover. Only the
score column differs between books — the same one-variable discipline as the duel.

The borrow cost, stated rather than assumed away
------------------------------------------------
A short book is not free. `BORROW_ANNUAL` is charged on the short leg's gross exposure every
month. **It is an assumption, not a measurement** — real borrow is per-name, time-varying,
and worst exactly where a short signal is strongest (crowded, hard-to-borrow names). 100 bps
is a benign large-cap number; the S&P 500 is the easiest universe to borrow in, which is the
one thing that argues for the assumption being generous rather than optimistic.

`sensitivity()` re-runs the verdict across a range, because a result that survives only at
the assumed rate is not a result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from signals.baseline_momentum import (
    COST_PER_SIDE, DECILE, PERIODS_PER_YEAR, SIGNAL, backtest_scores, rebalance_dates,
)

OOS_V2 = Path("data/sp500_oos_walkforward_v2.parquet")

BORROW_ANNUAL = 0.01        # 100 bps/yr on short-leg gross exposure — an ASSUMPTION
SHORT_GROSS = 1.0           # |sum| of the short leg by construction


def borrow_drag(annual: float = BORROW_ANNUAL) -> float:
    """Monthly cost of carrying the short leg."""
    return annual * SHORT_GROSS / PERIODS_PER_YEAR


def decile_profile(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Mean forward return by score decile — the diagnostic that motivated this arm.

    Reported for the v2 (clean) model specifically: if the near-monotonicity measured on v1
    was leak-driven, it degrades here, and the whole long/short case weakens with it.
    """
    d = df.copy()
    d["dec"] = d.groupby("date")[score_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    return d.groupby("dec")["fwd_ret_1m"].mean()


def score_book(df: pd.DataFrame, score_col: str, rebals, annual_borrow: float = BORROW_ANNUAL):
    """Long/short decile book on one score, with the borrow charge applied."""
    bt = backtest_scores(df, score_col, rebals, cost=COST_PER_SIDE)
    res = bt["res"].copy()
    res["borrow"] = borrow_drag(annual_borrow)
    res["net_ret"] = res["net_ret"] - res["borrow"]      # after borrow
    r = res["net_ret"]
    eq = (1 + r).cumprod()
    stats = {
        "n": int(len(r)),
        "ann_ret": float((1 + r).prod() ** (PERIODS_PER_YEAR / len(r)) - 1) if len(r) else None,
        "vol": float(r.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR)),
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(r) > 1 else None,
        "maxdd": float((eq / eq.cummax() - 1).min()),
        "turnover": float(res["turnover"].mean()),
        "mean_ic": float(bt["ic"].mean()),
        "hit": float((r > 0).mean()),
    }
    return stats, res, bt


def sensitivity(df, score_col, rebals, rates=(0.0, 0.005, 0.01, 0.02, 0.05)) -> pd.DataFrame:
    """Sharpe across borrow assumptions. A book that only works at one rate is not a book."""
    rows = []
    for a in rates:
        s, _, _ = score_book(df, score_col, rebals, annual_borrow=a)
        rows.append({"borrow_annual": a, "sharpe": s["sharpe"], "ann_ret": s["ann_ret"]})
    return pd.DataFrame(rows)


def run(oos_path: Path = OOS_V2) -> dict:
    df = pd.read_parquet(oos_path)
    df["date"] = pd.to_datetime(df["date"])
    rebals = rebalance_dates(df)

    books = {}
    for label, col in (("ML v2", "model_score"), ("Momentum", SIGNAL)):
        stats, res, bt = score_book(df, col, rebals)
        books[label] = {"stats": stats, "res": res, "decile_spread": bt["decile_spread"]}

    return {
        "window": f"{df['date'].min().date()} → {df['date'].max().date()}",
        "n_rebalances": int(len(rebals)),
        "decile": DECILE,
        "borrow_annual": BORROW_ANNUAL,
        "books": books,
        "profiles": {"ML v2": decile_profile(df, "model_score"),
                     "Momentum": decile_profile(df, SIGNAL)},
        "sensitivity": {"ML v2": sensitivity(df, "model_score", rebals),
                        "Momentum": sensitivity(df, SIGNAL, rebals)},
    }
