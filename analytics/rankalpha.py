"""
Wire the base analyser onto RankAlpha's own returns — Phase 12, step 6.

Reads the committed paper-trading ledger (`data/paper_track_portfolio.parquet`, written
by `portfolio/paper_trade.py`) and hands its realized monthly returns to the same generic
`analyse` / `make_charts` used for any ticker. The book's benchmark is the equal-weight
investable universe recorded alongside it (`bench_ret`) — NOT SPY, matching how the paper
track defines its benchmark. This module only READS the frozen model's output; it never
imports or refits it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .charts import make_charts
from .metrics import analyse

PORTFOLIO_PATH = Path("data/paper_track_portfolio.parquet")
PERIODS_PER_YEAR = 12  # monthly rebalances


def rankalpha_returns(path: Path = PORTFOLIO_PATH):
    """(book_returns, benchmark_returns) as date-indexed monthly Series."""
    pf = pd.read_parquet(path).sort_values("date")
    idx = pd.to_datetime(pf["date"])
    book = pd.Series(pf["net_ret"].to_numpy(), index=idx, name="RankAlpha")
    bench = pd.Series(pf["bench_ret"].to_numpy(), index=idx, name="Equal-weight universe")
    return book, bench


def report(path: Path = PORTFOLIO_PATH, make_figs: bool = True) -> dict:
    """Full performance report on RankAlpha's paper track: metrics + (optional) charts."""
    book, bench = rankalpha_returns(path)
    metrics = analyse(book, benchmark=bench, periods_per_year=PERIODS_PER_YEAR)
    figs = {}
    if make_figs:
        figs = make_charts(book, bench, prefix="rankalpha",
                           periods_per_year=PERIODS_PER_YEAR,
                           label="RankAlpha paper book",
                           bench_label="Equal-weight universe")
    return {"metrics": metrics, "figures": {k: str(v) for k, v in figs.items()},
            "n_months": len(book)}
