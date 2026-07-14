"""
Tests for the Strategy Lab harness — Phase 14.

Two fast unit tests on `factor_score` (no data needed), plus one integration test that
the harness's book plumbing reproduces the committed paper track when fed the frozen LGBM
score. The integration test is skipped if the committed data / cached model are absent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lab.strategy_lab import factor_score, monthly_rebalances

_LABELED = "data/sp500_labeled.parquet"
_PANEL = "data/sp500_panel.parquet"
_COMMITTED = "data/paper_track_portfolio.parquet"


def _day():
    # Four names; A has the highest momentum, D the lowest volatility.
    return pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "mom_12_1m": [0.40, 0.20, 0.00, -0.10],
        "vol_6m": [0.030, 0.025, 0.020, 0.010],
    })


def test_momentum_direction():
    """higher_is_better=True → the highest-momentum name gets the top score."""
    s = factor_score(_day(), [("mom_12_1m", True)])
    assert s.idxmax() == "A" and s.idxmin() == "D"


def test_lowvol_direction():
    """higher_is_better=False → the LOWEST-vol name gets the top score."""
    s = factor_score(_day(), [("vol_6m", False)])
    assert s.idxmax() == "D" and s.idxmin() == "A"


def test_equal_weight_combine():
    """Combined score is the mean of the two percentile ranks (equal weight)."""
    day = _day()
    mom = day["mom_12_1m"].rank(pct=True, ascending=True).to_numpy()
    lov = day["vol_6m"].rank(pct=True, ascending=False).to_numpy()
    combined = factor_score(day, [("mom_12_1m", True), ("vol_6m", False)]).to_numpy()
    assert np.allclose(combined, (mom + lov) / 2)


def test_monthly_rebalances_window_and_step():
    """monthly_rebalances tiles a window every `step` trading days, respecting bounds."""
    dates = pd.bdate_range("2020-01-01", periods=100)
    labeled = pd.DataFrame({"date": list(dates) * 2, "ticker": ["A"] * 100 + ["B"] * 100})
    reb = monthly_rebalances(labeled, "2020-02-01", "2020-04-01", step=21)
    assert reb == sorted(reb)
    assert all(pd.Timestamp("2020-02-01") <= d <= pd.Timestamp("2020-04-01") for d in reb)
    # 21-trading-day spacing between consecutive rebalances.
    gaps = [(reb[i + 1] - reb[i]).days for i in range(len(reb) - 1)]
    assert all(g >= 28 for g in gaps)  # 21 business days ≈ 29 calendar days


@pytest.mark.skipif(
    not Path(_COMMITTED).exists() or not Path(_LABELED).exists(),
    reason="committed track / labeled data not present",
)
def test_plumbing_reproduces_committed_track():
    """Frozen LGBM score through the harness reproduces the committed paper track exactly."""
    from portfolio.paper_trade import _frozen_model, FREEZE_DATE
    from lab.strategy_lab import run_strategy, frozen_lgbm_score, MOMENTUM

    labeled = pd.read_parquet(_LABELED); labeled["date"] = pd.to_datetime(labeled["date"])
    panel = pd.read_parquet(_PANEL); panel["date"] = pd.to_datetime(panel["date"])
    model = _frozen_model(labeled, FREEZE_DATE)

    plumb = run_strategy(MOMENTUM, labeled=labeled, panel=panel,
                         score_fn=lambda d: frozen_lgbm_score(d, model))
    committed = pd.read_parquet(_COMMITTED).sort_values("date").reset_index(drop=True)
    m = plumb[["date", "net_ret"]].merge(committed[["date", "net_ret"]], on="date",
                                         suffixes=("_lab", "_c"))
    assert len(m) == len(committed)
    assert (m["net_ret_lab"] - m["net_ret_c"]).abs().max() < 1e-9
