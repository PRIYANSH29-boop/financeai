"""
Example run — Phase 12.

    python -m analytics              # MSTR vs SPY (yfinance) + RankAlpha paper track
    python -m analytics AAPL NVDA    # any tickers vs SPY

Prints the metric table, writes the charts to figures/analytics/, then runs the same
analyser on RankAlpha's own realized returns.
"""

from __future__ import annotations

import sys

from .compare import compare
from .charts import make_charts


def main(argv=None):
    argv = argv or sys.argv[1:]
    tickers = argv or ["MSTR"]
    benchmark = "SPY"

    print(f"Loading {tickers} vs {benchmark} (monthly, 5y) via yfinance …")
    from .data import load_returns
    rets, ppy = load_returns(tickers, benchmark=benchmark, period="5y", interval="1mo")
    print(f"  {len(rets)} monthly observations.\n")

    print("=== Metric table ===")
    print(compare(rets, benchmark=rets[benchmark], periods_per_year=ppy, pretty=True)
          .to_string())

    print("\n=== Charts ===")
    for t in tickers:
        if t not in rets.columns:
            print(f"  ! {t}: no data, skipped")
            continue
        figs = make_charts(rets[t], rets[benchmark], prefix=t.lower(),
                           periods_per_year=ppy, label=t, bench_label=benchmark)
        for name, path in figs.items():
            print(f"  {t} {name:10s} -> {path}")

    print("\n=== RankAlpha paper track (same analyser) ===")
    try:
        from .rankalpha import report
        rep = report()
        m = rep["metrics"]
        print(f"  {rep['n_months']} months | Sharpe {m['sharpe']:.2f} | "
              f"CAGR {m['cagr']:.1%} | maxDD {m['max_drawdown']:.1%} | "
              f"alpha {m['alpha']:.1%} | beta {m['beta']:.2f}")
        for name, path in rep["figures"].items():
            print(f"  figure {name:10s} -> {path}")
    except FileNotFoundError:
        print("  (paper-track ledger not found — run portfolio/paper_trade.py first)")


if __name__ == "__main__":
    main()
