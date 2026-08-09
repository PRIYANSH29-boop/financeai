"""
Phase 24 — the Explore data hotfix, tested at both levels.

The four defects #24 fixes were all *silent*: a future as-of date, a beta of 364 sorting to
the top of the page, a "basket →" button for a name the basket page had no data for, and a
scored count that nothing reconciled. None of them crashed anything, which is exactly why
they shipped. So these tests come in pairs — a unit test that pins the RULE (runs anywhere,
no data needed) and a bundle test that pins the SHIPPED ARTIFACT (skipped if the bundle is
absent). A rule with no artifact check can pass while the live site stays wrong.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from scripts.export_web_bundle import (
    BASKET_MIN_HISTORY, BETA_ABS_MAX, ANN_VOL_MAX,
    band_flags, history_flags, panel_data_date, reason_text,
    validate_as_of, validate_explore, validate_scored_consistency,
)

BUNDLE = Path("web/public/bundle")
STOCKS = BUNDLE / "stocks.json"
EXPLORE = BUNDLE / "explore.json"
INDEX = BUNDLE / "index.json"

needs_bundle = pytest.mark.skipif(not STOCKS.exists(),
                                  reason="web bundle not built (run make web-bundle)")


@pytest.fixture(scope="module")
def stocks():
    return json.loads(STOCKS.read_text())


@pytest.fixture(scope="module")
def explore():
    return json.loads(EXPLORE.read_text()) if EXPLORE.exists() else None


def _axis(n=30, end="2026-06-30"):
    return pd.date_range(end=end, periods=n, freq="ME")


# ============================================================ 1. the as-of bound
def test_panel_data_date_is_the_last_data_date_not_the_month_end_bucket():
    """The bug in one assertion: a panel ending 2026-07-22 must not stamp 2026-07-31."""
    panel = pd.DataFrame({"date": pd.to_datetime(["2026-07-20", "2026-07-22", "2026-06-30"]),
                          "ticker": "AAA", "adj_close": [1.0, 2.0, 3.0]})
    assert panel_data_date(panel) == "2026-07-22"


def test_validate_as_of_rejects_a_future_stamp():
    errs = validate_as_of("explore.json", "2026-07-31", today="2026-07-25")
    assert len(errs) == 1 and "FUTURE" in errs[0]


def test_validate_as_of_accepts_today_and_the_past():
    assert validate_as_of("x", "2026-07-25", today="2026-07-25") == []
    assert validate_as_of("x", "2026-06-16", today="2026-07-25") == []


def test_validate_as_of_rejects_missing_and_unparseable():
    assert validate_as_of("x", None) and "missing" in validate_as_of("x", None)[0]
    assert validate_as_of("x", "not-a-date", today="2026-07-25")


@needs_bundle
def test_shipped_as_of_stamps_are_not_in_the_future(stocks, explore):
    today = pd.Timestamp.today().normalize()
    for label, payload in (("stocks.json", stocks), ("explore.json", explore),
                           ("index.json", json.loads(INDEX.read_text()))):
        if payload is None:
            continue
        assert pd.Timestamp(payload["as_of"]) <= today, \
            f"{label} as_of {payload['as_of']} is in the future"


@needs_bundle
def test_shipped_explore_as_of_matches_the_wide_panel_data_date(explore):
    """Guards the specific regression: as_of must be the data date, not the axis label."""
    if explore is None:
        pytest.skip("explore.json absent")
    panel = Path("data/midlarge_panel.parquet")
    if not panel.exists():
        pytest.skip("wide panel is a gitignored build input")
    p = pd.read_parquet(panel, columns=["date"])
    assert explore["as_of"] == panel_data_date(p)

    # The original assertion here was `axis_end_label >= as_of`: the axis label is the
    # resampled month-end, which sits at or beyond the true data date, and publishing it as
    # the as-of stamp was the #24 regression.
    #
    # #30-B's min-day rule adds the other direction. When the final bucket is too short to
    # annualise it is dropped, and the surviving axis then ends BEFORE the data date on
    # purpose. Both are correct; what must never happen is the two being confused, so the
    # assertion is now on the relationship the ruling defines rather than on one inequality.
    if explore.get("axis_last_month_action") == "dropped":
        assert explore["axis_end_label"] < explore["as_of"], \
            "a dropped final bucket must leave the axis ending before the data date"
        assert explore["axis_last_month_partial"] is False
    else:
        assert explore["axis_end_label"] >= explore["as_of"], \
            "a kept axis label is the resampled month-end, at or beyond the data date"


# ============================================================ 2. the sanity band
def test_band_flags_pass_ordinary_values():
    assert band_flags(0.28, 1.05) == []
    assert band_flags(ANN_VOL_MAX, BETA_ABS_MAX) == []          # boundary is INCLUSIVE


def test_band_flags_catch_the_chrd_and_sndk_shapes():
    assert band_flags(1.26, 364.5) == ["beta_out_of_band", "vol_out_of_band"]
    assert band_flags(1.60, 6.00) == ["beta_out_of_band", "vol_out_of_band"]
    assert band_flags(0.30, -4.0) == ["beta_out_of_band"]       # sign-symmetric
    assert band_flags(1.01, 1.0) == ["vol_out_of_band"]


def test_band_flags_treat_missing_stats_as_unusable_not_as_passing():
    assert "beta_unavailable" in band_flags(0.3, None)
    assert "beta_unavailable" in band_flags(0.3, float("nan"))
    assert "vol_unavailable" in band_flags(float("inf"), 1.0)


def test_reason_text_is_present_for_every_flag_and_absent_when_clean():
    assert reason_text([]) is None
    for f in ("beta_out_of_band", "vol_out_of_band", "insufficient_history",
              "discontinuous", "stale_history", "no_history"):
        assert reason_text([f]) and reason_text([f]) != f, f"{f} has no human reason"


@needs_bundle
def test_shipped_rows_are_flagged_but_never_winsorised(explore):
    """The band must not have changed a single displayed number — flag, don't fudge."""
    if explore is None:
        pytest.skip("explore.json absent")
    flagged = [r for r in explore["rows"] if r["stat_quality"] == "unreliable"]
    assert flagged, "expected some out-of-band rows in the shipped wide universe"
    for r in flagged:
        assert r["stat_flags"], f"{r['ticker']} flagged with no reason codes"
        assert r["stat_note"]
        # the out-of-band value is still there, unaltered
        out_of_band = ((r["beta"] is not None and abs(r["beta"]) > BETA_ABS_MAX)
                       or (r["ann_vol"] is not None and r["ann_vol"] > ANN_VOL_MAX)
                       or r["beta"] is None or r["ann_vol"] is None)
        assert out_of_band, f"{r['ticker']} flagged but its stats are inside the band"
    for r in explore["rows"]:
        if r["stat_quality"] == "ok":
            assert r["beta"] is not None and abs(r["beta"]) <= BETA_ABS_MAX
            assert r["ann_vol"] is not None and r["ann_vol"] <= ANN_VOL_MAX


@needs_bundle
def test_movers_cards_carry_no_flagged_rows(explore):
    if explore is None:
        pytest.skip("explore.json absent")
    for r in explore["movers"]["winners"] + explore["movers"]["losers"]:
        assert r["stat_quality"] == "ok", f"mover {r['ticker']} has unreliable stats"


# ============================================================ 3. scored gating
def test_history_flags_accept_a_long_clean_series():
    ax = _axis(30)
    col = pd.Series(np.full(30, 0.01), index=ax)
    assert history_flags(col, ax[-1]) == []


def test_history_flags_reject_a_short_history_spinoff():
    """The SNDK fixture: a 2025 spinoff with 17 months, current but too short."""
    ax = _axis(30)
    col = pd.Series(np.nan, index=ax, dtype="float64")
    col.iloc[-17:] = 0.02
    assert history_flags(col, ax[-1]) == ["insufficient_history"]


def test_history_flags_count_a_trailing_run_not_a_total():
    """40 observations total, but only 10 unbroken at the end, is not 24 months of history."""
    ax = _axis(60)
    col = pd.Series(np.nan, index=ax, dtype="float64")
    col.iloc[0:30] = 0.01          # long-ago block
    col.iloc[-10:] = 0.01          # short recent run
    flags = history_flags(col, ax[-1])
    assert "insufficient_history" in flags
    assert "discontinuous" in flags


def test_history_flags_reject_a_delisted_name_with_a_long_old_history():
    ax = _axis(60)
    col = pd.Series(np.nan, index=ax, dtype="float64")
    col.iloc[:40] = 0.01           # 40 clean months, then stops trading
    flags = history_flags(col, ax[-1])
    assert "stale_history" in flags and "insufficient_history" in flags


def test_history_flags_reject_an_internal_gap():
    """A gap 20 months back also shortens the trailing run below 24, so both flags fire —
    "≥24 observations" is deliberately read as 24 months of UNBROKEN recent history."""
    ax = _axis(40)
    col = pd.Series(np.full(40, 0.01), index=ax)
    col.iloc[20] = np.nan          # one missing month mid-history
    assert history_flags(col, ax[-1]) == ["discontinuous", "insufficient_history"]


def test_history_flags_reject_an_early_gap_on_the_discontinuity_rule_alone():
    """Gap far enough back that the trailing run is fine: discontinuity must still reject."""
    ax = _axis(40)
    col = pd.Series(np.full(40, 0.01), index=ax)
    col.iloc[3] = np.nan
    assert history_flags(col, ax[-1]) == ["discontinuous"]


def test_history_flags_boundary_is_exactly_min_history():
    ax = _axis(40)
    for n, expect_flag in ((BASKET_MIN_HISTORY, False), (BASKET_MIN_HISTORY - 1, True)):
        col = pd.Series(np.nan, index=ax, dtype="float64")
        col.iloc[-n:] = 0.01
        flags = history_flags(col, ax[-1])
        assert ("insufficient_history" in flags) is expect_flag, f"n={n}"


def test_history_flags_reject_an_empty_series():
    ax = _axis(30)
    assert history_flags(pd.Series(np.nan, index=ax, dtype="float64"), ax[-1]) == ["no_history"]


@needs_bundle
def test_every_shipped_series_actually_clears_the_gate(stocks):
    """No name in stocks.json may violate the rules that are supposed to have filtered it."""
    for tk, st in stocks["stats"].items():
        assert st["n"] >= BASKET_MIN_HISTORY, f"{tk} shipped with {st['n']} months"
        assert st["beta"] is not None and abs(st["beta"]) <= BETA_ABS_MAX, \
            f"{tk} shipped with beta {st['beta']}"
        assert st["ann_vol"] is not None and st["ann_vol"] <= ANN_VOL_MAX, \
            f"{tk} shipped with vol {st['ann_vol']}"


@needs_bundle
def test_every_refusal_is_recorded_with_a_reason(stocks):
    assert stocks["excluded"], "expected some refused names"
    for tk, info in stocks["excluded"].items():
        assert info["flags"] and info["reason"], f"{tk} refused with no reason"
        assert tk not in stocks["returns"], f"{tk} both refused and shipped"


@needs_bundle
def test_no_explore_row_offers_a_basket_button_without_a_series(stocks, explore):
    """The #23 defect, pinned: Q and SNDK were marked basket-eligible with no series."""
    if explore is None:
        pytest.skip("explore.json absent")
    have = set(stocks["returns"])
    claimed = {r["ticker"] for r in explore["rows"] if r["scored"]}
    orphans = sorted(claimed - have)
    assert not orphans, f"basket-eligible with no series: {orphans}"


@needs_bundle
def test_unscored_rows_always_say_why(explore):
    if explore is None:
        pytest.skip("explore.json absent")
    for r in explore["rows"]:
        if not r["scored"]:
            assert r["scored_reason"], f"{r['ticker']} refused with no reason"
        if r["stat_quality"] == "unreliable":
            assert not r["scored"], f"{r['ticker']} is flagged but basket-eligible"


# ============================================================ 4. count reconciliation
@needs_bundle
def test_reconciliation_adds_up(explore):
    if explore is None:
        pytest.skip("explore.json absent")
    rec = explore["scored_reconciliation"]
    assert (rec["with_explore_row"] + rec["dropped"]["absent_from_wide_universe"]
            == rec["model_universe"])
    assert rec["basket_eligible"] == explore["n_scored"]
    assert rec["basket_eligible"] <= rec["with_explore_row"]
    # the funnel names its casualties, it does not just count them
    assert len(rec["absent_from_wide_universe_tickers"]) == \
        rec["dropped"]["absent_from_wide_universe"]
    assert rec["why_absent_from_wide_universe"], "the gap has no stated cause"


@needs_bundle
def test_reconciliation_matches_the_shipped_ticker_lists(explore, stocks):
    if explore is None:
        pytest.skip("explore.json absent")
    rec = explore["scored_reconciliation"]
    tickers = Path("data/sp500_tickers.csv")
    if not tickers.exists():
        pytest.skip("S&P ticker list is a gitignored build input")
    model = set(pd.read_csv(tickers)["ticker"])
    rows = {r["ticker"] for r in explore["rows"]}
    assert rec["model_universe"] == len(model)
    assert set(rec["absent_from_wide_universe_tickers"]) == model - rows
    assert rec["eligible_without_explore_row"] == len(set(stocks["returns"]) - rows)


def test_validate_scored_consistency_catches_an_orphan():
    """A synthetic regression: Explore promises AAA, stocks.json has never heard of it."""
    explore = {"rows": [{"ticker": "AAA", "scored": True}], "n_scored": 1}
    errs = validate_scored_consistency(explore, {"returns": {"BBB": []}})
    assert any("AAA" in e for e in errs)


def test_validate_scored_consistency_passes_a_coherent_pair():
    explore = {"rows": [{"ticker": "AAA", "scored": True},
                        {"ticker": "BBB", "scored": False}], "n_scored": 1}
    assert validate_scored_consistency(explore, {"returns": {"AAA": []}}) == []


def test_validate_scored_consistency_catches_a_funnel_that_does_not_add_up():
    explore = {
        "rows": [], "n_scored": 5,
        "scored_reconciliation": {"model_universe": 503, "with_explore_row": 451,
                                  "basket_eligible": 5,
                                  "dropped": {"absent_from_wide_universe": 10}},
    }
    errs = validate_scored_consistency(explore, {"returns": {}})
    assert any("does not add up" in e for e in errs)


def test_validate_explore_rejects_a_flagged_but_eligible_row():
    payload = {
        "as_of": "2026-06-16", "movers": {"winners": [{"ticker": "AAA", "stat_quality": "ok"}]},
        "rows": [{"ticker": "AAA", "sector": "Tech", "cap_bucket": "large",
                  "stat_quality": "unreliable", "scored": True, "scored_reason": None}],
    }
    errs = validate_explore(payload)
    assert any("flagged unreliable but still marked basket-eligible" in e for e in errs)
