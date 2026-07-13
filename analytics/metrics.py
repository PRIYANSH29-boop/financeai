"""
Core performance metrics — Phase 12 base analyser.

Pure functions over a returns array/Series. No I/O, no model imports: this module
knows nothing about RankAlpha's frozen ranker. Formulas follow the reviewer's spec
exactly.

Convention: we use POPULATION standard deviation (ddof=0) throughout, so that
    vol([+12%, -8%, +4%, +8%]) == 7.48%   (√(0.0224/4)); ddof=1 would give 8.64%.
This is a deliberate, documented choice for internal consistency with the spec's
hand-checks. (`portfolio/paper_trade.py` uses ddof=1; the two are not blended.)
Beta is invariant to the choice since cov and var share the same ddof.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def coerce_returns(returns) -> np.ndarray:
    """Accept a list / np.ndarray / pd.Series of per-period simple returns → 1-D float array.

    NaNs are dropped (e.g. the first row of a pct_change). Values are assumed to be
    fractional returns (0.12 == +12%), not percents.
    """
    if isinstance(returns, pd.Series):
        r = returns.to_numpy(dtype="float64")
    else:
        r = np.asarray(returns, dtype="float64")
    r = r[~np.isnan(r)]
    return r


def equity_curve(returns) -> np.ndarray:
    """Cumulative growth of $1: equity = cumprod(1 + r)."""
    r = coerce_returns(returns)
    return np.cumprod(1.0 + r)


# ------------------------------------------------------------------ scalar metrics
def total_return(returns) -> float:
    """∏(1 + r) − 1."""
    r = coerce_returns(returns)
    if r.size == 0:
        return float("nan")
    return float(np.prod(1.0 + r) - 1.0)


def cagr(returns, periods_per_year: int = 12) -> float:
    """(∏(1 + r))^(1/years) − 1, with years = n / periods_per_year."""
    r = coerce_returns(returns)
    n = r.size
    if n == 0:
        return float("nan")
    years = n / periods_per_year
    growth = float(np.prod(1.0 + r))
    if years <= 0 or growth <= 0:
        return float("nan")
    return growth ** (1.0 / years) - 1.0


def volatility(returns, periods_per_year: int = 12) -> float:
    """Annualised volatility = std(r) × √N (population std, ddof=0)."""
    r = coerce_returns(returns)
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=0) * np.sqrt(periods_per_year))


def sharpe(returns, periods_per_year: int = 12, rf: float = 0.0) -> float:
    """Annualised Sharpe = (mean(r) − rf) / std(r) × √N. rf is per-period."""
    r = coerce_returns(returns)
    if r.size < 2:
        return float("nan")
    sd = np.std(r, ddof=0)
    if sd == 0:
        return float("nan")
    return float((np.mean(r) - rf) / sd * np.sqrt(periods_per_year))


def sortino(returns, periods_per_year: int = 12, rf: float = 0.0) -> float:
    """Annualised Sortino = (mean(r) − rf) / std(negative r only) × √N.

    Downside deviation uses only the periods with r < 0 (population std of that subset).
    """
    r = coerce_returns(returns)
    if r.size < 2:
        return float("nan")
    downside = r[r < 0]
    if downside.size == 0:
        return float("inf")  # no losing periods → undefined downside risk
    dd = np.std(downside, ddof=0)
    if dd == 0:
        return float("nan")
    return float((np.mean(r) - rf) / dd * np.sqrt(periods_per_year))


def max_drawdown(equity) -> tuple[float, int]:
    """Max drawdown and its duration from an EQUITY / price level series.

    max_dd = min(equity / cummax(equity) − 1). Duration = the longest run (in periods)
    spent below a prior peak, i.e. the length of the worst peak→recovery (or peak→end)
    underwater stretch. Pass either an equity curve from returns or raw price levels —
    drawdown is scale-invariant.

        max_drawdown([100, 150, 90, 120, 200, 140]) → (-0.40, 2)
    """
    eq = np.asarray(equity, dtype="float64")
    eq = eq[~np.isnan(eq)]
    if eq.size == 0:
        return float("nan"), 0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    max_dd = float(dd.min())

    # duration = longest consecutive stretch under water (dd < 0)
    longest = cur = 0
    for x in dd:
        if x < 0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return max_dd, int(longest)


def beta(returns, benchmark) -> float:
    """Cov(r, r_bench) / Var(r_bench). Series are aligned by dropping NaN pairwise."""
    r, b = _align(returns, benchmark)
    if r.size < 2:
        return float("nan")
    var_b = np.var(b, ddof=0)
    if var_b == 0:
        return float("nan")
    cov = np.mean((r - r.mean()) * (b - b.mean()))
    return float(cov / var_b)


def alpha(returns, benchmark, periods_per_year: int = 12) -> float:
    """Annualised alpha = (mean(r) − beta × mean(r_bench)) × N."""
    r, b = _align(returns, benchmark)
    if r.size < 2:
        return float("nan")
    bta = beta(r, b)
    if np.isnan(bta):
        return float("nan")
    return float((np.mean(r) - bta * np.mean(b)) * periods_per_year)


def _align(returns, benchmark) -> tuple[np.ndarray, np.ndarray]:
    """Align two return series to a common index / length, dropping any NaN pair."""
    if isinstance(returns, pd.Series) and isinstance(benchmark, pd.Series):
        df = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()
        return df["r"].to_numpy("float64"), df["b"].to_numpy("float64")
    r = np.asarray(returns, dtype="float64")
    b = np.asarray(benchmark, dtype="float64")
    n = min(r.size, b.size)
    r, b = r[:n], b[:n]
    mask = ~(np.isnan(r) | np.isnan(b))
    return r[mask], b[mask]


# ------------------------------------------------------------------ the one entry point
def analyse(returns, benchmark=None, periods_per_year: int = 12, rf: float = 0.0) -> dict:
    """Full metric dict for a per-period returns series.

    Parameters
    ----------
    returns : list | np.ndarray | pd.Series of per-period simple returns (0.12 == +12%).
    benchmark : optional same-shape benchmark returns; enables beta & alpha.
    periods_per_year : 12 for monthly (default), 252 for daily, 52 weekly.
    rf : per-period risk-free rate for Sharpe/Sortino (default 0).

    Returns a dict of scalar metrics (see keys below). Benchmark-relative metrics are
    NaN when no benchmark is given.
    """
    r = coerce_returns(returns)
    eq = equity_curve(r)
    max_dd, dd_dur = max_drawdown(eq) if eq.size else (float("nan"), 0)

    out = {
        "n_periods": int(r.size),
        "periods_per_year": periods_per_year,
        "total_return": total_return(r),
        "cagr": cagr(r, periods_per_year),
        "volatility": volatility(r, periods_per_year),
        "sharpe": sharpe(r, periods_per_year, rf),
        "sortino": sortino(r, periods_per_year, rf),
        "max_drawdown": max_dd,
        "max_drawdown_duration": dd_dur,
        "hit_rate": float(np.mean(r > 0)) if r.size else float("nan"),
        "beta": float("nan"),
        "alpha": float("nan"),
    }
    if benchmark is not None:
        out["beta"] = beta(returns, benchmark)
        out["alpha"] = alpha(returns, benchmark, periods_per_year)
    return out
