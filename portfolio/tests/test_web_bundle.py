"""
Unit tests for the Phase 19 web-bundle exporter — offline, no engine run, no network.

The frontend trusts this bundle completely: there is no backend to sanity-check it and no
way for the UI to notice that a number is wrong. So these tests pin the invariants the
validator enforces (weights sum to 1 with cash, nothing above the 8% cap, drift deltas
self-consistent, presets present) and the pure transforms that produce them.

Run: pytest portfolio/tests/ -q
"""

import numpy as np
import pandas as pd
import pytest

from scripts.export_web_bundle import (
    beta_grid, beta_key, validate, drift_after_one_period, beta_in_drawdown,
    _drawdown, _drawdown_window, _series, PRESETS, SCHEMA_VERSION,
)
from portfolio.beta_engine import NAME_CAP, SECTOR_CAP


# ------------------------------------------------------------------ the beta grid
def test_grid_covers_the_slider_range_on_exact_2dp_steps():
    g = beta_grid()
    assert g[0] == 0.0 and g[-1] == 1.85
    assert len(g) == 38
    # floats built by repeated addition drift (0.30000000000000004); these must not
    assert all(round(b, 2) == b for b in g)


def test_every_preset_is_a_grid_point():
    """A preset chip that is not on the grid would have no bundle file to load."""
    g = set(beta_grid())
    for label, b in PRESETS:
        assert b in g, f"preset {label} (beta {b}) is not on the exported grid"


def test_beta_key_is_stable_and_unique():
    g = beta_grid()
    keys = [beta_key(b) for b in g]
    assert len(set(keys)) == len(keys)
    assert beta_key(0.0) == "b000"
    assert beta_key(0.75) == "b075"
    assert beta_key(1.85) == "b185"


# ------------------------------------------------------------------ payload fixtures
def _payload(holdings=None, cash=0.90, drift=None):
    # Realistic shape: a real pie is ~20 names each well under the 8% cap, with a large
    # cash sleeve at low betas. Fixture weights must respect the caps or every test that
    # uses the default payload picks up spurious cap errors.
    holdings = holdings if holdings is not None else [
        {"ticker": "AAA", "name": "A", "sector": "Tech", "weight": 0.06,
         "reasons": ["momentum up"], "cap": NAME_CAP, "pct_of_cap": 0.75, "at_cap": False},
        {"ticker": "BBB", "name": "B", "sector": "Health", "weight": 0.04,
         "reasons": ["low vol"], "cap": NAME_CAP, "pct_of_cap": 0.50, "at_cap": False},
    ]
    return {
        "schema_version": SCHEMA_VERSION, "target_beta": 1.0, "achieved_beta": 1.0,
        "cash_weight": cash, "holdings": holdings,
        "equity": [{"d": "2024-01-31", "v": 1.0}],
        "drift": drift if drift is not None else {"available": False, "reason": "n/a"},
    }


# ------------------------------------------------------------------ validator
def test_valid_payload_passes():
    assert validate(_payload()) == []


def test_weights_plus_cash_must_sum_to_one():
    """A donut whose slices do not sum to 100% is a visible defect with no UI-side fix."""
    errs = validate(_payload(cash=0.80))         # 0.06 + 0.04 + 0.80 = 0.90
    assert any("expected 1" in e for e in errs)


def test_holding_above_the_name_cap_is_rejected():
    over = [{"ticker": "AAA", "name": "A", "sector": "Tech", "weight": NAME_CAP + 0.01,
             "reasons": ["x"], "cap": NAME_CAP, "pct_of_cap": 1.1, "at_cap": True}]
    errs = validate(_payload(holdings=over, cash=1.0 - (NAME_CAP + 0.01)))
    assert any("exceeds the" in e and "cap" in e for e in errs)


def test_sector_concentration_above_the_cap_is_rejected():
    same_sector = [
        {"ticker": "AAA", "name": "A", "sector": "Tech", "weight": 0.07,
         "reasons": ["x"], "cap": NAME_CAP, "pct_of_cap": 0.9, "at_cap": False},
        {"ticker": "BBB", "name": "B", "sector": "Tech", "weight": 0.07,
         "reasons": ["x"], "cap": NAME_CAP, "pct_of_cap": 0.9, "at_cap": False},
        {"ticker": "CCC", "name": "C", "sector": "Tech", "weight": 0.07,
         "reasons": ["x"], "cap": NAME_CAP, "pct_of_cap": 0.9, "at_cap": False},
        {"ticker": "DDD", "name": "D", "sector": "Tech", "weight": 0.07,
         "reasons": ["x"], "cap": NAME_CAP, "pct_of_cap": 0.9, "at_cap": False},
        {"ticker": "EEE", "name": "E", "sector": "Tech", "weight": 0.07,
         "reasons": ["x"], "cap": NAME_CAP, "pct_of_cap": 0.9, "at_cap": False},
    ]
    errs = validate(_payload(holdings=same_sector, cash=1.0 - 0.35))
    assert any("sector Tech" in e for e in errs)
    assert SECTOR_CAP == 0.30


def test_holding_without_a_reason_is_rejected():
    """Zone 3 of the UI exists to explain every slice — a blank reason breaks the promise."""
    mute = [{"ticker": "AAA", "name": "A", "sector": "Tech", "weight": 0.05,
             "reasons": [""], "cap": NAME_CAP, "pct_of_cap": 0.625, "at_cap": False}]
    errs = validate(_payload(holdings=mute, cash=0.95))
    assert any("no reason text" in e for e in errs)


def test_inconsistent_drift_delta_is_rejected():
    bad = {"available": True, "period_end": "2026-05-31",
           "holdings": [{"ticker": "AAA", "target_weight": 0.30,
                         "drifted_weight": 0.34, "delta_pp": 1.0}],  # should be 4.0
           "cash": {}}
    errs = validate(_payload(drift=bad))
    assert any("drift delta" in e for e in errs)


# ------------------------------------------------------------------ drift
def test_drift_conserves_total_weight_and_deltas_reconcile():
    w = pd.Series({"AAA": 0.3, "BBB": 0.2})
    rets = pd.DataFrame({"AAA": [0.10], "BBB": [-0.05]},
                        index=pd.to_datetime(["2026-05-31"]))
    d = drift_after_one_period(w, rets, cash_weight=0.5)
    assert d["available"]
    total = sum(r["drifted_weight"] for r in d["holdings"]) + d["cash"]["drifted_weight"]
    assert total == pytest.approx(1.0, abs=1e-6)
    for r in d["holdings"]:
        assert r["target_weight"] + r["delta_pp"] / 100.0 == pytest.approx(
            r["drifted_weight"], abs=1e-7)


def test_the_winner_gains_weight_and_the_loser_loses_it():
    w = pd.Series({"UP": 0.25, "DOWN": 0.25})
    rets = pd.DataFrame({"UP": [0.20], "DOWN": [-0.20]},
                        index=pd.to_datetime(["2026-05-31"]))
    d = {r["ticker"]: r for r in drift_after_one_period(w, rets, 0.5)["holdings"]}
    assert d["UP"]["delta_pp"] > 0
    assert d["DOWN"]["delta_pp"] < 0


def test_cash_share_rises_when_the_book_falls():
    """Cash earns 0, so it cannot lose share to a falling book — this is the mechanism the
    'what happens after you invest?' panel is explaining."""
    w = pd.Series({"AAA": 0.5})
    rets = pd.DataFrame({"AAA": [-0.20]}, index=pd.to_datetime(["2026-05-31"]))
    d = drift_after_one_period(w, rets, cash_weight=0.5)
    assert d["cash"]["delta_pp"] > 0


def test_drift_reports_unavailable_rather_than_guessing():
    w = pd.Series({"AAA": 0.5})
    assert drift_after_one_period(w, pd.DataFrame(), 0.5)["available"] is False
    assert drift_after_one_period(w, None, 0.5)["available"] is False


# ------------------------------------------------------------------ drawdown + beta
def test_drawdown_is_zero_at_a_new_high_and_negative_below_one():
    r = pd.Series([0.10, -0.20, 0.05])
    dd = _drawdown(r)
    assert dd[0] == pytest.approx(0.0)
    assert dd[1] < 0 and dd[2] < 0


def test_drawdown_window_runs_from_the_peak_to_the_trough():
    r = pd.Series([0.10, 0.10, -0.30, -0.10, 0.40])   # peak at idx 1, trough at idx 3
    start, end = _drawdown_window(r)
    assert start == 1 and end == 3


def test_beta_in_drawdown_is_measured_only_inside_that_window():
    """The whole point of the disclosure: it must NOT be the full-history beta."""
    idx = pd.date_range("2024-01-31", periods=8, freq="ME")
    port = pd.Series([0.02, 0.02, -0.10, -0.08, -0.05, 0.04, 0.03, 0.02], index=idx)
    bench = pd.Series([0.01, 0.01, -0.02, -0.02, -0.01, 0.02, 0.01, 0.01], index=idx)
    res = beta_in_drawdown(port, bench)
    assert res["beta"] is not None
    assert res["n_months"] < len(port)
    assert res["start"] is not None and res["end"] is not None


def test_beta_in_drawdown_refuses_a_too_short_window():
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    res = beta_in_drawdown(pd.Series([-0.1, 0.1], index=idx),
                           pd.Series([-0.05, 0.05], index=idx))
    assert res["beta"] is None and "too short" in res["note"]


# ------------------------------------------------------------------ series encoding
def test_series_drops_non_finite_points_rather_than_emitting_null():
    """JSON NaN is not valid JSON and would break the chart at parse time."""
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    out = _series(idx, np.array([1.0, np.nan, 1.2]))
    assert len(out) == 2
    assert all(np.isfinite(p["v"]) for p in out)
    assert out[0]["d"] == "2024-01-31"
