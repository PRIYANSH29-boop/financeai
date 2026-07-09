"""
RankAlpha analytics — Phase 12 base analyser module.

A reusable, model-agnostic performance analyser. It takes a per-period returns series
(loaded from yfinance for any ticker, or handed in directly — e.g. RankAlpha's own
paper-track returns) and produces metrics, charts, and comparison tables. It imports
NOTHING from `signals/` or `portfolio/`: the frozen model stays frozen; data only flows
in.

Quick start
-----------
    from analytics import analyse, load_returns, compare, make_charts

    rets, ppy = load_returns("MSTR", benchmark="SPY", period="5y", interval="1mo")
    print(analyse(rets["MSTR"], rets["SPY"], periods_per_year=ppy))
    compare(rets, benchmark=rets["SPY"], periods_per_year=ppy, pretty=True)
    make_charts(rets["MSTR"], rets["SPY"], prefix="mstr")

    # …or on RankAlpha's realized paper track:
    from analytics.rankalpha import report
    report()
"""

from .metrics import (
    analyse,
    total_return,
    cagr,
    volatility,
    sharpe,
    sortino,
    max_drawdown,
    beta,
    alpha,
    equity_curve,
)
from .data import load_returns, load_prices, to_returns
from .compare import compare
from .charts import make_charts

__all__ = [
    "analyse", "total_return", "cagr", "volatility", "sharpe", "sortino",
    "max_drawdown", "beta", "alpha", "equity_curve",
    "load_returns", "load_prices", "to_returns",
    "compare", "make_charts",
]
