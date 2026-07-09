"""
Repo-generated performance charts — Phase 12.

Five committed figures, all driven off a per-period returns Series (ideally with a
DatetimeIndex; a plain RangeIndex works too):

  1. equity curve (strategy vs benchmark overlay)
  2. drawdown (underwater) chart
  3. rolling Sharpe & rolling volatility (window = one year)
  4. return-distribution histogram
  5. stock-vs-benchmark price/equity overlay (normalised to 1.0 at inception)

Everything is written to `figures/analytics/` with a caller-chosen prefix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .metrics import sharpe, volatility  # noqa: E402

FIG_DIR = Path("figures/analytics")
_STRAT_C = "#2ca02c"
_BENCH_C = "#7f7f7f"


def _as_series(returns) -> pd.Series:
    s = returns if isinstance(returns, pd.Series) else pd.Series(np.asarray(returns, "float64"))
    return s.dropna()


def _equity(s: pd.Series) -> pd.Series:
    return (1.0 + s).cumprod()


def equity_curve_chart(returns, benchmark=None, path=None,
                       label="Strategy", bench_label="Benchmark") -> Path:
    s = _as_series(returns)
    eq = _equity(s)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(eq.index, eq.values, lw=2, color=_STRAT_C, label=label)
    if benchmark is not None:
        b = _as_series(benchmark)
        ax.plot(_equity(b).index, _equity(b).values, lw=1.5, ls="--",
                color=_BENCH_C, label=bench_label)
    ax.set_title("Equity curve — growth of $1")
    ax.set_ylabel("Cumulative growth (×)")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def drawdown_chart(returns, path=None, label="Strategy") -> Path:
    s = _as_series(returns)
    eq = _equity(s)
    dd = eq / eq.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.35)
    ax.plot(dd.index, dd.values, lw=1, color="#d62728", label=label)
    ax.set_title(f"Drawdown — worst {dd.min():.1%}")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def rolling_chart(returns, window: int | None = None, periods_per_year: int = 12,
                  path=None) -> Path:
    """Rolling Sharpe & volatility. Default window = one year (periods_per_year)."""
    s = _as_series(returns)
    w = window or periods_per_year
    roll_sh = s.rolling(w).apply(lambda x: sharpe(x.values, periods_per_year), raw=False)
    roll_vol = s.rolling(w).apply(lambda x: volatility(x.values, periods_per_year), raw=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(roll_sh.index, roll_sh.values, color="#1f77b4")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_title(f"Rolling {w}-period Sharpe")
    ax1.grid(alpha=0.3)
    ax2.plot(roll_vol.index, roll_vol.values, color="#ff7f0e")
    ax2.set_title(f"Rolling {w}-period volatility (annualised)")
    ax2.grid(alpha=0.3)
    return _save(fig, path)


def return_hist_chart(returns, path=None) -> Path:
    s = _as_series(returns)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s.values, bins=max(10, int(np.sqrt(s.size) * 2)),
            color=_STRAT_C, alpha=0.8, edgecolor="white")
    ax.axvline(s.mean(), color="k", ls="--", lw=1, label=f"mean {s.mean():.2%}")
    ax.axvline(0, color="#d62728", lw=1)
    ax.set_title("Return distribution")
    ax.set_xlabel("Per-period return")
    ax.set_ylabel("Count")
    ax.legend()
    return _save(fig, path)


def overlay_chart(stock_returns, benchmark_returns, path=None,
                  stock_label="Stock", bench_label="Benchmark") -> Path:
    """Normalised stock-vs-benchmark overlay (both start at 1.0)."""
    s, b = _as_series(stock_returns), _as_series(benchmark_returns)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(_equity(s).index, _equity(s).values, lw=2, color=_STRAT_C, label=stock_label)
    ax.plot(_equity(b).index, _equity(b).values, lw=2, color=_BENCH_C, label=bench_label)
    ax.set_title(f"{stock_label} vs {bench_label} — normalised")
    ax.set_ylabel("Growth of $1 (×)")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def make_charts(returns, benchmark=None, outdir=FIG_DIR, prefix="strategy",
                periods_per_year: int = 12, label="Strategy",
                bench_label="Benchmark") -> dict:
    """Generate all applicable charts, return {name: Path}."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figs = {
        "equity": equity_curve_chart(returns, benchmark, out / f"{prefix}_equity.png",
                                     label, bench_label),
        "drawdown": drawdown_chart(returns, out / f"{prefix}_drawdown.png", label),
        "rolling": rolling_chart(returns, None, periods_per_year,
                                 out / f"{prefix}_rolling.png"),
        "histogram": return_hist_chart(returns, out / f"{prefix}_hist.png"),
    }
    if benchmark is not None:
        figs["overlay"] = overlay_chart(returns, benchmark, out / f"{prefix}_overlay.png",
                                        label, bench_label)
    return figs


def _save(fig, path) -> Path:
    if path is None:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        path = FIG_DIR / "chart.png"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
