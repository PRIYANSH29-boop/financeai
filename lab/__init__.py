"""
RankAlpha Strategy Lab — Phase 14.

A thin research harness for testing scoring *recipes* on top of the frozen RankAlpha
pipeline. A "strategy" is a spec — which factors to rank by, how to combine them — and
`run_strategy(spec)` runs it through the SAME book construction as the frozen paper track
(`portfolio/paper_trade.py`): same universe, rebalance dates, embargo, top-N long-only
selection, inverse-vol weights, weight cap, vol target, and turnover cost. The ONLY thing
that changes between strategies is the ranking factor, so any performance difference is
attributable to the factor — not to a refit.

This layer is allowed to import from `signals/`, `portfolio/`, and `analytics/` (it is the
orchestration layer); `analytics/` itself stays model-agnostic and imports none of them.
"""

from .strategy_lab import (
    run_strategy,
    factor_score,
    strategy_returns,
    signal_correlation,
    MOMENTUM,
    MOMENTUM_LOWVOL,
)

__all__ = [
    "run_strategy", "factor_score", "strategy_returns", "signal_correlation",
    "MOMENTUM", "MOMENTUM_LOWVOL",
]
