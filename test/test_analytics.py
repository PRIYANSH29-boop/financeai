"""
Phase 12 unit checks — the two reviewer hand-checks, plus a couple of guards.

Run:  python -m pytest test/test_analytics.py     (or)   python test/test_analytics.py
These are offline (no yfinance / network needed).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.metrics import volatility, max_drawdown, analyse  # noqa: E402


def test_volatility_hand_check():
    """vol([+12%, -8%, +4%, +8%]) == 7.48% monthly (population std, ddof=0)."""
    monthly = volatility([0.12, -0.08, 0.04, 0.08], periods_per_year=1)
    assert math.isclose(monthly, 0.0748331, rel_tol=1e-4), monthly


def test_max_drawdown_hand_check():
    """maxDD of 100→150→90→120→200→140 == -40% (peak-tracking)."""
    dd, duration = max_drawdown([100, 150, 90, 120, 200, 140])
    assert math.isclose(dd, -0.40, abs_tol=1e-9), dd
    # longest underwater run is 90→120 (recovers at 200); the later 140 is a separate run
    assert duration == 2, duration


def test_analyse_shape():
    """analyse returns every documented key and a sane drawdown."""
    m = analyse([0.02, -0.01, 0.03, 0.00, 0.015], benchmark=[0.01, 0.00, 0.02, 0.005, 0.01])
    for k in ("total_return", "cagr", "volatility", "sharpe", "sortino",
              "max_drawdown", "max_drawdown_duration", "beta", "alpha", "hit_rate"):
        assert k in m
    assert -1.0 <= m["max_drawdown"] <= 0.0


if __name__ == "__main__":
    test_volatility_hand_check()
    test_max_drawdown_hand_check()
    test_analyse_shape()
    print("volatility hand-check ...... PASS  (7.48% monthly)")
    print("max-drawdown hand-check .... PASS  (-40%, duration 2)")
    print("analyse shape .............. PASS")
    print("\nAll Phase 12 analytics checks passed.")
