"""
Unit tests for the Phase 12 base analyser — Phase 13 integration.

Six tests, anchored on the reviewer's two hand-checks:
  * vol([+12%, -8%, +4%, +8%]) == 7.48%   (population std, ddof=0)
  * max drawdown of [100,150,90,120,200,140] == -40%

The remaining four pin down the rest of the metric surface (total return / CAGR,
Sharpe & Sortino, beta/alpha self-consistency, and the analyse() contract) so the
numbers the RankAlpha scorecard is built on can't silently drift.

Run: pytest analytics/tests/ -q
"""

import math

import numpy as np
import pytest

from analytics.metrics import (
    analyse,
    alpha,
    beta,
    cagr,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
    volatility,
)


# ---------------------------------------------------------------- hand-check #1
def test_volatility_handcheck():
    """std([+12%, -8%, +4%, +8%]) = sqrt(0.0224/4) = 7.4833%  (ddof=0).

    periods_per_year=1 so there is no annualisation factor — this is the raw
    per-period dispersion the spec hand-checked.
    """
    vol = volatility([0.12, -0.08, 0.04, 0.08], periods_per_year=1)
    assert vol == pytest.approx(0.0748, abs=5e-5)


# ---------------------------------------------------------------- hand-check #2
def test_max_drawdown_handcheck():
    """Equity [100,150,90,120,200,140]: worst peak-to-trough is 150 -> 90 = -40%.

    Duration is the longest underwater run (dd < 0): indices 2 and 3 -> 2 periods.
    """
    dd, dur = max_drawdown([100, 150, 90, 120, 200, 140])
    assert dd == pytest.approx(-0.40, abs=1e-9)
    assert dur == 2


# ---------------------------------------------------------------- total return / CAGR
def test_total_return_and_cagr_agree():
    """total_return is the compounded growth; CAGR over exactly `periods_per_year`
    periods must equal the total return (one year of data)."""
    r = [0.10, -0.05, 0.10, 0.05, 0.02, 0.03, -0.01, 0.04, 0.06, -0.02, 0.01, 0.07]
    growth = math.prod(1 + x for x in r) - 1
    assert total_return(r) == pytest.approx(growth, rel=1e-12)
    # 12 monthly periods == 1 year -> CAGR collapses to the total return.
    assert cagr(r, periods_per_year=12) == pytest.approx(growth, rel=1e-12)


# ---------------------------------------------------------------- Sharpe & Sortino
def test_sharpe_and_sortino():
    """Sharpe uses full-sample std; Sortino only downside std, so with any losing
    period present Sortino must exceed Sharpe (smaller denominator)."""
    r = [0.02, -0.01, 0.03, 0.00, -0.02, 0.04]
    sh = sharpe(r, periods_per_year=12)
    so = sortino(r, periods_per_year=12)
    mean = np.mean(r)
    sd = np.std(r, ddof=0)
    assert sh == pytest.approx(mean / sd * math.sqrt(12), rel=1e-12)
    assert so > sh
    # No losing periods -> downside risk undefined -> +inf by convention.
    assert sortino([0.01, 0.02, 0.03]) == float("inf")


# ---------------------------------------------------------------- beta / alpha
def test_beta_alpha_vs_self():
    """A series regressed on itself has beta exactly 1 and alpha exactly 0."""
    r = [0.02, -0.01, 0.03, 0.00, -0.02, 0.04]
    assert beta(r, r) == pytest.approx(1.0, abs=1e-12)
    assert alpha(r, r, periods_per_year=12) == pytest.approx(0.0, abs=1e-12)
    # Beta to a 2x-levered version of itself is 0.5 (Var scales 4x, Cov 2x).
    r2 = [2 * x for x in r]
    assert beta(r, r2) == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------- analyse() contract
def test_analyse_contract():
    """analyse() returns the full key set; beta/alpha are NaN without a benchmark
    and finite once one is supplied."""
    r = [0.02, -0.01, 0.03, 0.00, -0.02, 0.04]
    expected = {
        "n_periods", "periods_per_year", "total_return", "cagr", "volatility",
        "sharpe", "sortino", "max_drawdown", "max_drawdown_duration", "hit_rate",
        "beta", "alpha",
    }
    m = analyse(r)  # no benchmark
    assert expected <= set(m)
    assert m["n_periods"] == len(r)
    assert m["hit_rate"] == pytest.approx(3 / 6)  # 3 of 6 periods strictly positive
    assert math.isnan(m["beta"]) and math.isnan(m["alpha"])

    mb = analyse(r, benchmark=r)  # self-benchmark -> beta 1, alpha 0
    assert mb["beta"] == pytest.approx(1.0, abs=1e-12)
    assert mb["alpha"] == pytest.approx(0.0, abs=1e-12)
