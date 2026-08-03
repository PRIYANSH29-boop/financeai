"""Tests for the Style & Season Lab (#26 Parts B + C).

Two jobs: pin the style RULES on synthetic fixtures (so a rule change is a visible diff, not a
quiet census shift), and hand-check the GRID MATH (so a t-stat can never be a formula typo).
Nothing here touches the network or the committed panels.
"""

import numpy as np
import pandas as pd
import pytest

from lab.style_lab import (
    BLUE_CHIP_MIN_MONTHS, MAX_LABELS, STYLES, T_FINDING, EARNINGS_MONTHS,
    dividend_yield, eps_growth, latest_eps, classify, census, overlap_matrix,
    rank_ic_by_date, cell_stats, month_end_slice, build_grid, grid_summary,
)


# ────────────────────────────────────────────────────────────── per-name characteristics
def test_dividend_yield_reads_the_adj_close_over_close_drift():
    """A payer's adj/close ratio drifts up by exactly the dividend return; a non-payer's is flat."""
    dates = pd.bdate_range("2025-01-01", periods=253)
    payer = pd.DataFrame({"date": dates, "ticker": "PAY", "close": 100.0,
                          "adj_close": np.linspace(100.0, 105.0, len(dates))})
    flat = pd.DataFrame({"date": dates, "ticker": "NOPAY", "close": 100.0, "adj_close": 100.0})
    dy = dividend_yield(pd.concat([payer, flat], ignore_index=True))
    assert dy["PAY"] == pytest.approx(0.05, abs=1e-6)
    assert dy["NOPAY"] == pytest.approx(0.0, abs=1e-12)


def test_dividend_yield_skips_names_with_too_little_history():
    one = pd.DataFrame({"date": [pd.Timestamp("2025-01-02")], "ticker": ["X"],
                        "close": [10.0], "adj_close": [10.0]})
    assert "X" not in dividend_yield(one).index


def _ledger(ticker, eps_values):
    n = len(eps_values)
    return pd.DataFrame({
        "ticker": ticker,
        "period_end": pd.date_range("2023-03-31", periods=n, freq="QE"),
        "publication_date": pd.date_range("2023-05-01", periods=n, freq="QE"),
        "eps": eps_values,
    })


def test_eps_growth_compares_ttm_against_the_prior_ttm():
    # prior TTM = 4.0, recent TTM = 6.0 -> +50%
    f = _ledger("G", [1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5])
    assert eps_growth(f)["G"] == pytest.approx(0.5)


def test_eps_growth_refuses_a_negative_prior_base():
    """Growth off a loss is not a number — the name must drop out, not report a huge figure."""
    f = _ledger("L", [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    assert "L" not in eps_growth(f).index


def test_eps_growth_needs_eight_quarters():
    assert "S" not in eps_growth(_ledger("S", [1.0] * 7)).index


def test_latest_eps_sums_the_trailing_four_quarters():
    assert latest_eps(_ledger("E", [9.0, 1.0, 1.0, 1.0, 1.0]))["E"] == pytest.approx(4.0)


# ─────────────────────────────────────────────────────────────────────────── classification
def _chars(**over):
    """Ten names spanning the cross-section, so percentile rules have something to rank."""
    base = pd.DataFrame({
        "market_cap": np.linspace(1e9, 1e12, 10),
        "ann_vol": np.linspace(0.10, 1.00, 10),
        "beta": np.linspace(0.2, 2.0, 10),
        "n_months": [72] * 10,
        "sector": ["Technology"] * 10,
        "div_yield": np.linspace(0.0, 0.09, 10),
        "eps_growth": np.linspace(-0.5, 0.5, 10),
        "ttm_eps": np.linspace(-2.0, 8.0, 10),
        "value_score": np.linspace(-2.0, 2.0, 10),
    }, index=[f"T{i}" for i in range(10)])
    for k, v in over.items():
        base[k] = v
    return base


def test_a_name_never_carries_more_than_two_styles():
    out = classify(_chars())
    assert out["n_styles"].max() <= MAX_LABELS
    assert (out[STYLES].sum(axis=1) == out["n_styles"]).all()


def test_cyclical_and_defensive_are_sector_rules():
    c = _chars(sector=["Energy"] * 5 + ["Utilities"] * 5, beta=[0.5] * 10)
    out = classify(c)
    assert out.loc["T0", "cyclical"] or out.loc["T0", "n_styles"] == MAX_LABELS
    assert out.loc["T9", "defensive"] or out.loc["T9", "n_styles"] == MAX_LABELS


def test_gics_and_yfinance_taxonomies_classify_identically():
    """#20's mixed-taxonomy bug, guarded. The S&P file is GICS ("Health Care", "Consumer
    Discretionary", "Materials"); the wide file is yfinance ("Healthcare", "Consumer Cyclical",
    "Basic Materials"). A sector rule applied to raw strings matches only the overlap and
    silently undercounts one side, so both must normalise to the same answer."""
    gics = _chars(sector=["Health Care", "Consumer Discretionary", "Consumer Staples",
                          "Materials", "Information Technology", "Financials"] + ["Energy"] * 4,
                  beta=[0.5] * 10)
    yf_ = _chars(sector=["Healthcare", "Consumer Cyclical", "Consumer Defensive",
                         "Basic Materials", "Technology", "Financial Services"] + ["Energy"] * 4,
                 beta=[0.5] * 10)
    a, b = classify(gics), classify(yf_)
    for s in STYLES:
        assert a[s].tolist() == b[s].tolist(), f"{s} differs between taxonomies"
    assert a["cyclical"].sum() > 0 and a["defensive"].sum() > 0


def test_normalize_sector_leaves_canonical_labels_alone():
    from lab.style_lab import normalize_sector
    canon = pd.Series(["Healthcare", "Energy", "Utilities", "Technology"])
    assert normalize_sector(canon).tolist() == canon.tolist()
    assert normalize_sector(pd.Series(["Health Care"])).tolist() == ["Healthcare"]


def test_defensive_requires_low_beta_not_just_a_defensive_sector():
    """A utility with beta 2 is not defensive — the sector alone must not earn the label."""
    c = _chars(sector=["Utilities"] * 10, beta=[2.0] * 10)
    assert not classify(c)["defensive"].any()


def test_blue_chip_requires_all_three_legs():
    """Mega-cap + low vol + long history. Drop the history leg and the label must vanish."""
    c = _chars(n_months=[BLUE_CHIP_MIN_MONTHS - 1] * 10)
    assert not classify(c)["blue_chip"].any()


def test_speculative_requires_high_vol_small_cap_and_no_earnings():
    """A high-vol small cap that is solidly profitable is not speculative."""
    c = _chars(ttm_eps=[5.0] * 10)
    assert not classify(c)["speculative"].any()


def test_missing_fundamentals_do_not_default_a_name_into_or_out_of_a_gated_style():
    """The whole point of the coverage caveat: NaN must mean 'not evaluable', never 'False-ish
    but counted'. A name with no ledger must not appear in growth or value."""
    c = _chars()
    c.loc["T9", ["eps_growth", "value_score", "ttm_eps"]] = np.nan
    out = classify(c)
    assert not out.loc["T9", "growth"]
    assert not out.loc["T9", "value"]
    assert not out.loc["T9", "fundamentals_available"]
    assert out.loc["T0", "fundamentals_available"]


def test_census_and_overlap_are_consistent_with_the_flags():
    out = classify(_chars())
    cen = census(out)
    for s in STYLES:
        assert cen.loc[s, "n_names"] == int(out[s].sum())
    ov = overlap_matrix(out)
    for s in STYLES:
        assert ov.loc[s, s] == int(out[s].sum())          # diagonal is the style's own count
    assert (ov.values == ov.values.T).all()               # overlaps are symmetric


# ─────────────────────────────────────────────────────────────────────────────── grid math
def test_cell_stats_t_is_hand_checked():
    """mean 0.04, sd(ddof=1) 0.02, n 3 -> t = 0.04 / (0.02/sqrt(3)) = 3.4641."""
    st = cell_stats(pd.Series([0.02, 0.04, 0.06]))
    assert st["mean_ic"] == pytest.approx(0.04)
    assert st["t"] == pytest.approx(3.4641, abs=1e-4)
    assert st["n_months"] == 3
    assert st["verdict"] == "FINDING"          # 3.46 >= T_FINDING


def test_cell_stats_grades_against_the_multiple_testing_bar():
    assert cell_stats(pd.Series([0.02, 0.04, 0.06]))["verdict"] == "FINDING"
    # mean 0.02, sd 0.02, n 3 -> t = 1.73 -> noise
    assert cell_stats(pd.Series([0.0, 0.02, 0.04]))["verdict"] == "noise"
    assert cell_stats(pd.Series([], dtype="float64"))["verdict"] == "empty"
    assert cell_stats(pd.Series([0.05]))["n_months"] == 1     # n=1 has no t, must not crash


def test_cell_stats_t_threshold_matches_the_documented_constant():
    assert T_FINDING == 3.0


def test_rank_ic_is_a_spearman_within_each_date():
    d = pd.Timestamp("2025-01-31")
    df = pd.DataFrame({"date": [d] * 6, "ticker": list("ABCDEF"),
                       "sig": [1, 2, 3, 4, 5, 6], "fwd_ret_1m": [10, 20, 30, 40, 50, 60]})
    assert rank_ic_by_date(df, "sig")[d] == pytest.approx(1.0)
    df["fwd_ret_1m"] = [60, 50, 40, 30, 20, 10]
    assert rank_ic_by_date(df, "sig")[d] == pytest.approx(-1.0)


def test_rank_ic_drops_dates_too_thin_to_correlate():
    d = pd.Timestamp("2025-01-31")
    df = pd.DataFrame({"date": [d] * 4, "ticker": list("ABCD"),
                       "sig": [1, 2, 3, 4], "fwd_ret_1m": [1, 2, 3, 4]})
    assert len(rank_ic_by_date(df, "sig")) == 0


def test_month_end_slice_keeps_one_rebalance_row_per_month():
    """Daily rows would count the same monthly bet ~21 times and inflate every t-stat."""
    dates = pd.bdate_range("2025-01-01", "2025-02-28")
    df = pd.DataFrame({"date": list(dates) * 2,
                       "ticker": ["A"] * len(dates) + ["B"] * len(dates)})
    out = month_end_slice(df)
    assert out["date"].nunique() == 2
    assert set(out["date"].dt.month) == {1, 2}
    # one row per (ticker, month) — 2 tickers x 2 months = 2 rows each, never 21.
    assert out.groupby(["ticker", out["date"].dt.month]).size().eq(1).all()
    assert len(out) == 4


def _grid_fixture(n_months=30, n_names=12):
    dates = pd.date_range("2023-01-31", periods=n_months, freq="ME")
    rows = []
    rng = np.random.default_rng(0)
    for d in dates:
        for i in range(n_names):
            rows.append({"date": d, "ticker": f"T{i}",
                         "mom_12_1m": rng.normal(), "fwd_ret_1m": rng.normal()})
    return pd.DataFrame(rows)


def test_build_grid_emits_every_cell_including_the_empty_ones():
    """Honesty rail 1: no cell is dropped for being empty or for looking bad."""
    labeled = _grid_fixture()
    chars = _chars().reindex([f"T{i}" for i in range(12)]).ffill()
    chars.index = [f"T{i}" for i in range(12)]
    classified = classify(chars)
    grid = build_grid(labeled, classified, "mom_12_1m")
    n_periods = grid["period"].nunique()
    assert len(grid) == (len(STYLES) + 1) * n_periods        # +1 for the ALL control row
    assert "ALL (control)" in set(grid["style"])
    assert {"full window", "earnings months", "non-earnings months"} <= set(grid["period"])
    assert grid["n_names"].notna().all()


def test_grid_periods_split_earnings_season_by_the_committed_months():
    labeled = _grid_fixture()
    chars = _chars().reindex([f"T{i}" for i in range(12)]).ffill()
    chars.index = [f"T{i}" for i in range(12)]
    grid = build_grid(labeled, classify(chars), "mom_12_1m")
    ctrl = grid[grid["style"] == "ALL (control)"].set_index("period")
    e, ne = ctrl.loc["earnings months", "n_months"], ctrl.loc["non-earnings months", "n_months"]
    assert e + ne == ctrl.loc["full window", "n_months"]     # the split is a partition
    assert set(EARNINGS_MONTHS) == {1, 4, 7, 10}


def test_grid_summary_carries_the_multiple_testing_context():
    """A finding may never travel without the number of cells it was selected from."""
    labeled = _grid_fixture()
    chars = _chars().reindex([f"T{i}" for i in range(12)]).ffill()
    chars.index = [f"T{i}" for i in range(12)]
    s = grid_summary(build_grid(labeled, classify(chars), "mom_12_1m"))
    assert {"n_cells_total", "n_cells_tested", "n_findings", "expected_false_at_t2"} <= set(s)
    assert s["n_cells_tested"] <= s["n_cells_total"]
    assert s["expected_false_at_t2"] == round(0.0455 * s["n_cells_tested"], 1)


def test_pure_noise_produces_no_findings_at_the_t3_bar():
    """The bar has to actually stop luck: random signal vs random forward return, 30 months."""
    s = grid_summary(build_grid(_grid_fixture(n_months=30),
                                classify(_chars().reindex([f"T{i}" for i in range(12)]).ffill()
                                         .set_axis([f"T{i}" for i in range(12)])),
                                "mom_12_1m"))
    assert s["n_findings"] == 0
