"""
Unit tests for the Phase 17 fundamentals-audit PURE logic — offline, no network, no FMP
key. These pin the checks the GO/NO-GO gate relies on so it can't silently rot:
the point-in-time leakage assertion, coverage, outlier flags, value-ratio guards, and the
GO/NO-GO decision. The FMP network path is intentionally NOT tested here (no fabrication).

Run: pytest audit/tests/ -q
"""

import math

import numpy as np

from audit.fundamentals import (
    winsorize, zscore, value_ratios, detect_outliers,
    assert_point_in_time, coverage_map, go_no_go, self_test, CORE_INPUTS,
)


def test_self_test_all_pass():
    assert self_test()["all_passed"] is True


def test_winsorize_clips_tails():
    w = winsorize([1, 2, 3, 4, 1000], pct=(0.0, 0.8))
    assert w.max() < 1000 and w.min() == 1


def test_zscore_is_mean_zero():
    z = zscore([10.0, 20.0, 30.0])
    assert abs(float(np.mean(z))) < 1e-9


def test_value_ratio_orientation_and_guards():
    rec = {"eps": 5.0, "price": 100.0, "book_value": 40e9, "market_cap": 200e9,
           "ebitda": 25e9, "enterprise_value": 210e9, "free_cash_flow": 15e9}
    vr = value_ratios(rec)
    assert math.isclose(vr["earnings_yield"], 0.05)
    assert math.isclose(vr["book_to_market"], 0.2)
    # negative equity ⇒ book/market undefined (NaN), not a bogus number
    assert math.isnan(value_ratios({**rec, "book_value": -1e9})["book_to_market"])
    # zero price ⇒ earnings yield undefined
    assert math.isnan(value_ratios({**rec, "price": 0})["earnings_yield"])


def test_outlier_flags():
    flags = set(detect_outliers({"book_value": -5, "price": 0, "market_cap": 1e9,
                                 "enterprise_value": 1e9}))
    assert "negative_equity" in flags
    assert "nonpositive_price" in flags


def test_point_in_time_leakage_gate():
    recs = [
        {"ticker": "A", "period_end": "2024-03-31", "publication_date": "2024-05-02"},  # ok
        {"ticker": "B", "period_end": "2024-03-31", "publication_date": "2024-03-31"},  # leak
        {"ticker": "C", "period_end": "2024-03-31", "publication_date": None},          # missing
    ]
    pit = assert_point_in_time(recs)
    assert pit["n_ok"] == 1
    assert pit["n_publication_not_after_period"] == 1
    assert pit["n_missing_publication_date"] == 1
    assert pit["pass"] is False


def test_point_in_time_all_clean_passes():
    recs = [{"ticker": "A", "period_end": "2024-03-31", "publication_date": "2024-05-02"},
            {"ticker": "B", "period_end": "2024-06-30", "publication_date": "2024-08-01"}]
    assert assert_point_in_time(recs)["pass"] is True


def test_coverage_fraction():
    cov = coverage_map([{"eps": 1.0}, {"eps": None}, {"eps": 3.0}, {"eps": float("nan")}],
                       fields=["eps"])
    assert math.isclose(cov["eps"], 0.5)


def test_gonogo_blocks_leakage_and_thin_coverage():
    leaking = assert_point_in_time(
        [{"ticker": "B", "period_end": "2024-03-31", "publication_date": "2024-03-31"}])
    report = {
        "checks": {
            "accuracy": {"status": "no_comparisons", "max_rel_discrepancy": None},
            "coverage": {k: 0.5 for k in CORE_INPUTS},   # below the 70% hard floor
            "point_in_time": leaking,
        },
    }
    assert go_no_go(report)["verdict"] == "NO-GO"


def test_gonogo_go_when_clean():
    clean = assert_point_in_time(
        [{"ticker": "A", "period_end": "2024-03-31", "publication_date": "2024-05-02"}])
    report = {
        "checks": {
            "accuracy": {"status": "ran", "max_rel_discrepancy": 0.01},
            "coverage": {k: 0.95 for k in CORE_INPUTS},
            "point_in_time": clean,
        },
    }
    assert go_no_go(report)["verdict"] == "GO"
