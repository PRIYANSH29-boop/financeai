"""The #30-B minimum-day rule for the final monthly bucket.

A month-end bucket built from one or two trading days is not a month. Annualising its
volatility (×√12) or fitting a beta to it yields a number with the shape of a statistic and
none of the content — and the exported explore bundle really did ship a one-day bucket whose
stats were computed and displayed like every other month's.

The ruling has three outcomes and this file pins all three, plus the validator that stops a
future call site from quietly skipping the rule.
"""

import pandas as pd
import pytest

from scripts.export_web_bundle import (
    MIN_FINAL_MONTH_DAYS,
    apply_min_day_rule,
    final_month_days,
    partial_month_disclosure,
    validate_partial_month,
)


def _panel(dates):
    """A minimal panel: the exporter's rule only ever reads the date column."""
    return pd.DataFrame({"date": pd.to_datetime(list(dates))})


def _rets(month_ends):
    """Monthly returns indexed at month-end labels, one dummy column."""
    idx = pd.to_datetime(list(month_ends))
    return pd.DataFrame({"AAA": [0.01] * len(idx)}, index=idx)


def _trading_days(start, n):
    """`n` business days from `start` inclusive."""
    return pd.bdate_range(start=start, periods=n)


# ------------------------------------------------------------------ day counting

def test_a_complete_final_month_counts_as_nothing_to_disclose():
    panel = _panel(_trading_days("2026-05-01", 21))
    # panel reaches the axis label itself
    panel = _panel(list(panel["date"]) + [pd.Timestamp("2026-05-31")])
    assert final_month_days(panel, pd.Timestamp("2026-05-31")) is None


def test_a_partial_final_month_counts_its_trading_days():
    days = list(_trading_days("2026-06-01", 12))
    panel = _panel(list(_trading_days("2026-05-01", 21)) + days)
    assert final_month_days(panel, pd.Timestamp("2026-06-30")) == 12


def test_no_axis_means_no_count():
    assert final_month_days(_panel(["2026-05-04"]), None) is None


# ------------------------------------------------------------------ the three outcomes

def test_complete_month_is_kept_and_says_nothing():
    idx = ["2026-04-30", "2026-05-31"]
    panel = _panel(list(_trading_days("2026-04-01", 22)) + [pd.Timestamp("2026-05-31")])
    rets, note = apply_min_day_rule(panel, _rets(idx))
    assert len(rets) == 2
    assert note["axis_last_month_partial"] is False
    assert note["axis_last_month_action"] is None
    assert note["axis_last_month_text"] is None


def test_a_long_enough_partial_month_is_kept_and_disclosed():
    """Ten days or more: unchanged behaviour, and the disclosure still fires."""
    panel = _panel(list(_trading_days("2026-05-01", 21)) + list(_trading_days("2026-06-01", 12)))
    rets, note = apply_min_day_rule(panel, _rets(["2026-05-31", "2026-06-30"]))
    assert len(rets) == 2, "a 12-day bucket must survive"
    assert note["axis_last_month_action"] == "kept"
    assert note["axis_last_month_partial"] is True
    assert note["axis_last_month_days"] == 12
    assert "stats include it" in note["axis_last_month_text"]


def test_a_too_short_final_month_is_dropped_from_the_axis():
    panel = _panel(list(_trading_days("2026-07-01", 22)) + [pd.Timestamp("2026-08-03")])
    rets, note = apply_min_day_rule(panel, _rets(["2026-07-31", "2026-08-31"]))
    assert len(rets) == 1, "the one-day bucket must not survive"
    assert str(rets.index[-1].date()) == "2026-07-31"
    assert note["axis_last_month_action"] == "dropped"
    assert note["axis_last_month_days"] == 1
    # The surviving axis ends on a complete month, so the partial flag is FALSE.
    assert note["axis_last_month_partial"] is False


def test_the_dropped_case_still_discloses():
    """Silently trimming would be A-3 pointed the other way.

    The as-of date would advertise data more recent than anything the figures came from,
    with nothing on the page to explain the gap.
    """
    panel = _panel(list(_trading_days("2026-07-01", 22)) + [pd.Timestamp("2026-08-03")])
    _, note = apply_min_day_rule(panel, _rets(["2026-07-31", "2026-08-31"]))
    text = note["axis_last_month_text"]
    assert text and "excluded" in text.lower()
    assert "2026-08-03" in text, "the reader must see how recent the data actually is"
    assert "2026-07-31" in text, "and where the figures actually stop"


@pytest.mark.parametrize("days", [1, 2, 5, 9])
def test_everything_below_the_floor_is_dropped(days):
    panel = _panel(list(_trading_days("2026-07-01", 22)) + list(_trading_days("2026-08-03", days)))
    rets, note = apply_min_day_rule(panel, _rets(["2026-07-31", "2026-08-31"]))
    assert note["axis_last_month_action"] == "dropped"
    assert len(rets) == 1


@pytest.mark.parametrize("days", [10, 11, 15, 20])
def test_everything_at_or_above_the_floor_is_kept(days):
    panel = _panel(list(_trading_days("2026-07-01", 22)) + list(_trading_days("2026-08-03", days)))
    rets, note = apply_min_day_rule(panel, _rets(["2026-07-31", "2026-08-31"]))
    assert note["axis_last_month_action"] == "kept"
    assert len(rets) == 2


def test_the_floor_is_ten():
    """The boundary is the ruling, so it is pinned rather than inferred from behaviour."""
    assert MIN_FINAL_MONTH_DAYS == 10


# ------------------------------------------------------------------ the validator

def test_validator_rejects_a_short_month_that_was_kept():
    """The enforcement half. Without it the rule is a convention, and a call site that
    forgets apply_min_day_rule ships annualised one-day stats exactly as before."""
    errs = validate_partial_month("explore.json", {
        "axis_last_month_partial": True,
        "axis_last_month_days": 1,
        "axis_last_month_text": "Final month is partial (1 trading day to X) — stats include it.",
        "axis_last_month_action": "kept",
    })
    assert errs, "a 1-day bucket kept on the axis must fail the export"
    assert "below the 10-day floor" in errs[0]


def test_validator_accepts_a_long_enough_kept_month():
    assert validate_partial_month("stocks.json", {
        "axis_last_month_partial": True,
        "axis_last_month_days": 12,
        "axis_last_month_text": "Final month is partial (12 trading days to X) — stats include it.",
        "axis_last_month_action": "kept",
    }) == []


def test_validator_requires_the_dropped_case_to_say_so():
    errs = validate_partial_month("explore.json", {
        "axis_last_month_partial": False,
        "axis_last_month_days": 1,
        "axis_last_month_text": "Final month is partial — stats include it.",
        "axis_last_month_action": "dropped",
    })
    assert errs and "saying so" in errs[0]


def test_validator_rejects_dropped_plus_partial():
    """Both flags at once describes an axis that cannot exist."""
    errs = validate_partial_month("explore.json", {
        "axis_last_month_partial": True,
        "axis_last_month_days": 1,
        "axis_last_month_text": "… excluded from every statistic here.",
        "axis_last_month_action": "dropped",
    })
    assert errs and "must be false" in errs[0]


def test_validator_rejects_a_dropped_claim_for_a_long_month():
    errs = validate_partial_month("explore.json", {
        "axis_last_month_partial": False,
        "axis_last_month_days": 14,
        "axis_last_month_text": "… excluded from every statistic here.",
        "axis_last_month_action": "dropped",
    })
    assert errs and "not below" in errs[0]


def test_validator_stays_quiet_on_a_complete_month():
    assert validate_partial_month("index.json", {
        "axis_last_month_partial": False,
        "axis_last_month_days": None,
        "axis_last_month_text": None,
        "axis_last_month_action": None,
    }) == []


# ------------------------------------------------------------------ shape

def test_every_bundle_carries_the_same_disclosure_shape():
    """The frontend must never branch on a missing key to tell the three cases apart."""
    keys = {"axis_last_month_partial", "axis_last_month_days",
            "axis_last_month_text", "axis_last_month_action"}
    complete = _panel(["2026-05-31"])
    partial = _panel(list(_trading_days("2026-06-01", 12)))
    short = _panel([pd.Timestamp("2026-08-03")])
    for panel, axis_end in [(complete, pd.Timestamp("2026-05-31")),
                            (partial, pd.Timestamp("2026-06-30")),
                            (short, pd.Timestamp("2026-08-31"))]:
        assert set(partial_month_disclosure(panel, axis_end)) >= keys - {"axis_last_month_action"}
    for panel, idx in [(complete, ["2026-05-31"]),
                       (partial, ["2026-05-31", "2026-06-30"]),
                       (short, ["2026-07-31", "2026-08-31"])]:
        _, note = apply_min_day_rule(panel, _rets(idx))
        assert set(note) == keys
