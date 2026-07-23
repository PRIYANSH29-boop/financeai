"""
Unit tests for the Phase 18 value factor's PURE logic — offline, no network.

The thing most worth pinning here is the leakage gate: `as_of` must never hand a rebalance
a filing that was not yet public. The rest pins the composite's guards — negative equity,
too-few-ratios, and the neutral fill that keeps the A/B universe identical.

Run: pytest lab/tests/ -q
"""

import numpy as np
import pandas as pd
import pytest

from lab.value_factor import (
    as_of, ratios_on, composite, attach, RATIOS, MIN_RATIOS, PUBLICATION_LAG_DAYS,
)


@pytest.fixture
def fund():
    """Two tickers, two filings each, with deliberately different publication dates."""
    return pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
        "period_end": pd.to_datetime(["2023-03-31", "2023-06-30",
                                      "2023-03-31", "2023-06-30"]),
        "publication_date": pd.to_datetime(["2023-05-01", "2023-08-01",
                                            "2023-05-10", "2023-08-10"]),
        "eps": [4.0, 5.0, 2.0, 2.5],
        "book_value": [50.0, 55.0, 20.0, 22.0],
        "ebitda": [30.0, 33.0, 12.0, 13.0],
        "free_cash_flow": [20.0, 22.0, 8.0, 9.0],
        "cash": [10.0, 10.0, 5.0, 5.0],
        "debt_lt": [40.0, 40.0, 15.0, 15.0],
        "debt_st": [0.0, 0.0, 0.0, 0.0],
        "shares": [10.0, 10.0, 4.0, 4.0],
    })


# ------------------------------------------------------------------ the leakage gate
def test_as_of_takes_the_newest_filing_that_was_already_public(fund):
    f = as_of(fund, "2023-09-01")
    assert list(f.index) == ["AAA", "BBB"]
    assert f.loc["AAA", "period_end"] == pd.Timestamp("2023-06-30")


def test_as_of_excludes_a_filing_published_after_the_rebalance(fund):
    """On 2023-06-01 the Q2 report (filed August) does not exist yet — using it would be
    reading the future, which is the entire failure mode #17 was built to prevent."""
    f = as_of(fund, "2023-06-01")
    assert f.loc["AAA", "period_end"] == pd.Timestamp("2023-03-31")
    assert f.loc["BBB", "period_end"] == pd.Timestamp("2023-03-31")


def test_as_of_applies_the_publication_lag_on_the_filing_day_itself(fund):
    """A filing that hit EDGAR on the rebalance date is not tradable that same day."""
    assert as_of(fund, "2023-05-01", lag_days=PUBLICATION_LAG_DAYS).empty
    assert not as_of(fund, "2023-05-01", lag_days=0).empty


def test_as_of_is_empty_before_any_filing_exists(fund):
    assert as_of(fund, "2023-01-01").empty


# ------------------------------------------------------------------ ratios
def test_ratios_use_the_rebalance_price_with_lagged_fundamentals(fund):
    prices = pd.Series({"AAA": 20.0, "BBB": 10.0})
    r = ratios_on(fund, prices, "2023-09-01")
    # AAA: eps 5 / price 20 = 0.25 ; mcap = 10 shares * 20 = 200 ; book 55 / 200 = 0.275
    assert r.loc["AAA", "earnings_yield"] == pytest.approx(0.25)
    assert r.loc["AAA", "book_to_market"] == pytest.approx(0.275)
    # EV = 200 + 40 - 10 = 230 ; ebitda 33 / 230
    assert r.loc["AAA", "ebitda_to_ev"] == pytest.approx(33.0 / 230.0)
    assert r.loc["AAA", "fcf_yield"] == pytest.approx(22.0 / 200.0)


def test_negative_equity_makes_book_to_market_undefined(fund):
    f = fund.copy()
    f.loc[f["ticker"] == "AAA", "book_value"] = -5.0
    r = ratios_on(f, pd.Series({"AAA": 20.0, "BBB": 10.0}), "2023-09-01")
    assert np.isnan(r.loc["AAA", "book_to_market"])
    assert not np.isnan(r.loc["AAA", "earnings_yield"])   # the other ratios survive


def test_missing_price_yields_no_ratios_rather_than_a_wrong_one(fund):
    r = ratios_on(fund, pd.Series({"BBB": 10.0}), "2023-09-01")
    assert r.loc["AAA"].isna().all()


# ------------------------------------------------------------------ composite
def _ratio_frame(n=6):
    idx = [f"T{i}" for i in range(n)]
    return pd.DataFrame({c: np.linspace(0.01, 0.10, n) for c in RATIOS}, index=idx)


def test_composite_is_mean_zero_across_the_cross_section():
    z = composite(_ratio_frame())
    assert float(np.nanmean(z)) == pytest.approx(0.0, abs=1e-9)


def test_composite_ranks_cheaper_names_higher():
    r = _ratio_frame()
    z = composite(r)
    assert z.idxmax() == r.index[-1]     # highest yields = cheapest = highest score


def test_composite_requires_at_least_two_ratios():
    r = _ratio_frame()
    r.loc["T0", RATIOS[1:]] = np.nan     # only one ratio left for T0
    z = composite(r)
    assert np.isnan(z.loc["T0"])
    assert z.notna().sum() == len(r) - 1


def test_composite_averages_the_ratios_that_are_present():
    r = _ratio_frame()
    r.loc["T0", RATIOS[2:]] = np.nan     # two ratios present — still above the threshold
    assert not np.isnan(composite(r).loc["T0"])
    assert MIN_RATIOS == 2


# ------------------------------------------------------------------ attachment
def test_attach_fills_uncovered_names_neutrally_and_keeps_the_universe_intact():
    """A name with no fundamentals must stay in the book with a neutral opinion. Dropping it
    would change the universe between A and B; leaving it NaN would let the rank-average
    hand it its momentum rank twice, a hidden momentum overweight."""
    labeled = pd.DataFrame({
        "date": pd.to_datetime(["2023-09-01"] * 3),
        "ticker": ["AAA", "BBB", "CCC"],
        "mom_12_1m": [0.1, 0.2, 0.3],
    })
    vpanel = pd.DataFrame({
        "date": pd.to_datetime(["2023-09-01"] * 2),
        "ticker": ["AAA", "BBB"],
        "value_score": [1.0, -1.0],
    })
    out = attach(labeled, vpanel)
    assert len(out) == 3                                  # universe unchanged
    assert out.set_index("ticker").loc["CCC", "value_score"] == 0.0
    assert not out.set_index("ticker").loc["CCC", "value_covered"]
    assert out.set_index("ticker").loc["AAA", "value_covered"]
