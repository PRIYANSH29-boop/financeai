"""
Unit tests for the SEC EDGAR provider's PURE logic — offline, no network.

These pin the transformations that everything downstream trusts: how raw XBRL duration
facts become a clean quarterly series (including the Q4 that only exists inside the 10-K),
how a trailing-twelve-month sum refuses to span a gap, how restatements are excluded so the
factor stays point-in-time, and how the split factor reconciles as-reported per-share facts
with a retro-adjusted price panel.

Run: pytest audit/tests/ -q
"""

import math

import numpy as np
import pandas as pd

from audit.sec_provider import (
    quarterly_from_duration_facts, ttm, split_factor, enterprise_value,
    _dedupe_earliest_filed,
)


def _f(start, end, val, filed, form="10-Q", rank=0):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form,
            "rank": rank}


# ------------------------------------------------------------------ quarterly extraction
def test_direct_quarterly_facts_are_kept():
    facts = [
        _f("2023-01-01", "2023-03-31", 100.0, "2023-05-01"),
        _f("2023-04-01", "2023-06-30", 110.0, "2023-08-01"),
    ]
    q = quarterly_from_duration_facts(facts)
    assert [r["end"] for r in q] == ["2023-03-31", "2023-06-30"]
    assert [r["val"] for r in q] == [100.0, 110.0]
    assert all(r["source"] == "direct" for r in q)


def test_q4_is_derived_from_the_annual_cumulative():
    """The classic gap: filers report Q1/Q2/Q3 as quarters and Q4 only inside the FY total.
    Differencing the year-to-date cumulatives has to recover it, or every 4th quarter is
    missing and TTM never computes."""
    facts = [
        _f("2023-01-01", "2023-03-31", 100.0, "2023-05-01"),               # Q1 (direct)
        _f("2023-01-01", "2023-06-30", 210.0, "2023-08-01"),               # H1 cumulative
        _f("2023-01-01", "2023-09-30", 330.0, "2023-11-01"),               # 9M cumulative
        _f("2023-01-01", "2023-12-31", 460.0, "2024-02-15", form="10-K"),  # FY
    ]
    q = {r["end"]: r for r in quarterly_from_duration_facts(facts)}
    assert math.isclose(q["2023-06-30"]["val"], 110.0)      # 210 - 100
    assert math.isclose(q["2023-09-30"]["val"], 120.0)      # 330 - 210
    assert math.isclose(q["2023-12-31"]["val"], 130.0)      # 460 - 330  ← the Q4 recovery
    # a derived quarter becomes public only when the LATER cumulative is filed
    assert q["2023-12-31"]["filed"] == "2024-02-15"


def test_restatement_is_ignored_in_favour_of_the_original_filing():
    facts = [
        _f("2023-01-01", "2023-03-31", 100.0, "2023-05-01"),   # as originally reported
        _f("2023-01-01", "2023-03-31", 95.0, "2024-05-01"),    # restated a year later
    ]
    q = quarterly_from_duration_facts(facts)
    assert len(q) == 1 and q[0]["val"] == 100.0, "restatement leaked into the PIT series"


def test_lower_ranked_tag_wins_when_two_tags_cover_the_same_period():
    facts = [
        _f("2023-01-01", "2023-03-31", 100.0, "2023-05-01", rank=1),
        _f("2023-01-01", "2023-03-31", 101.0, "2023-05-01", rank=0),   # preferred tag
    ]
    kept = _dedupe_earliest_filed(facts, key=lambda f: (f["start"], f["end"]))
    assert len(kept) == 1 and kept[0]["val"] == 101.0


# ------------------------------------------------------------------ TTM
def _series(ends, vals):
    return pd.DataFrame({"val": vals}, index=ends)


def test_ttm_sums_four_consecutive_quarters():
    s = _series(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"],
                [10.0, 20.0, 30.0, 40.0])
    t = ttm(s)
    assert np.isnan(t.iloc[:3]).all(), "TTM must not emit before 4 quarters exist"
    assert math.isclose(t.iloc[3], 100.0)


def test_ttm_refuses_to_span_a_missing_quarter():
    """Q3 absent: the 4-row window covers ~15 months, which is not a year. Summing it would
    silently overstate the flow, so it must be NaN instead."""
    s = _series(["2023-03-31", "2023-06-30", "2023-12-31", "2024-03-31"],
                [10.0, 20.0, 40.0, 50.0])
    assert np.isnan(ttm(s).iloc[3])


def test_ttm_propagates_a_missing_value_as_nan():
    s = _series(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"],
                [10.0, np.nan, 30.0, 40.0])
    assert np.isnan(ttm(s).iloc[3])


# ------------------------------------------------------------------ splits
def test_split_factor_only_counts_splits_after_the_period():
    splits = [("2024-06-10", 10.0)]
    assert split_factor(splits, "2024-03-31") == 10.0   # period predates the split
    assert split_factor(splits, "2024-09-30") == 1.0    # period follows it
    assert split_factor([], "2024-03-31") == 1.0


def test_split_factor_compounds_and_handles_reverse_splits():
    splits = [("2021-01-01", 2.0), ("2023-01-01", 5.0), ("2024-01-01", 0.2)]
    assert math.isclose(split_factor(splits, "2020-06-30"), 2.0 * 5.0 * 0.2)
    assert math.isclose(split_factor(splits, "2022-06-30"), 5.0 * 0.2)


def test_split_adjustment_puts_eps_on_the_price_panel_basis():
    """The bug this guards: as-reported EPS of $5.98 against a 10:1-split-adjusted price of
    $90 reads as a 6.6% earnings yield instead of 0.66% — a 10x error that would rank the
    name as the cheapest in the index."""
    eps_reported, price_adjusted = 5.98, 90.0
    fac = split_factor([("2024-06-10", 10.0)], "2024-03-31")
    assert math.isclose(eps_reported / fac / price_adjusted, 0.0066, abs_tol=1e-4)


# ------------------------------------------------------------------ enterprise value
def test_enterprise_value_adds_debt_and_subtracts_cash():
    assert math.isclose(enterprise_value(100.0, 30.0, 10.0, 20.0), 120.0)


def test_enterprise_value_treats_missing_legs_as_zero_but_needs_a_market_cap():
    assert math.isclose(enterprise_value(100.0, None, None, None), 100.0)
    assert math.isnan(enterprise_value(None, 30.0, 10.0, 20.0))
    assert math.isnan(enterprise_value(0.0, 30.0, 10.0, 20.0))
