"""
Autopilot data loader — Phase 12.

Pulls adjusted prices via yfinance for any ticker(s) + a benchmark (default SPY) and
converts to per-period returns. Also passes through a returns Series you already have
(e.g. RankAlpha's paper-track `net_ret`), so the analyser runs on either source.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("analytics.data")

# yfinance interval → periods per year, for annualising downstream metrics.
INTERVAL_PPY = {"1d": 252, "1wk": 52, "1mo": 12}


def to_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Adjusted-price levels → simple per-period returns (first NaN row dropped)."""
    return prices.pct_change().dropna(how="all")


def load_prices(tickers, benchmark: str | None = "SPY",
                start=None, end=None, period: str = "5y",
                interval: str = "1mo") -> pd.DataFrame:
    """Adjusted-close price panel for `tickers` (+ benchmark) as a DataFrame.

    tickers : a single ticker string or an iterable of tickers.
    benchmark : appended as an extra column if not already present; pass None to skip.
    start/end : explicit ISO dates (override `period` when given).
    period/interval : yfinance shorthand, e.g. period='5y', interval='1mo'.
    """
    import yfinance as yf  # imported lazily so metrics/tests need no network dep

    if isinstance(tickers, str):
        tickers = [tickers]
    symbols = list(dict.fromkeys(tickers))  # de-dupe, preserve order
    if benchmark and benchmark not in symbols:
        symbols.append(benchmark)

    kw = dict(interval=interval, auto_adjust=True, progress=False)
    if start is not None:
        kw.update(start=start, end=end)
    else:
        kw.update(period=period)

    logger.info("yfinance download: %s (%s)", symbols, interval)
    raw = yf.download(symbols, **kw)

    # With auto_adjust=True the adjusted level lives in "Close".
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    # Restore requested column order and drop all-NaN columns (bad tickers).
    close = close.reindex(columns=[s for s in symbols if s in close.columns])
    return close.dropna(how="all")


def load_returns(tickers, benchmark: str | None = "SPY",
                 start=None, end=None, period: str = "5y",
                 interval: str = "1mo"):
    """Per-period return panel for `tickers` (+ benchmark).

    Returns (returns_df, periods_per_year). `returns_df` columns are the tickers with
    the benchmark last (when requested). Pair with `analytics.metrics.analyse`.
    """
    prices = load_prices(tickers, benchmark, start, end, period, interval)
    return to_returns(prices), INTERVAL_PPY.get(interval, 12)
