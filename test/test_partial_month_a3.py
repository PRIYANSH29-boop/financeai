"""#28 A-3 — the partial final month must be DISCLOSED, not merely flagged.

#24 exported `axis_last_month_partial` and no component read it, so the disclosure was true,
machine-readable and invisible for a full release (#25 finding A-3). These tests pin both
halves of the fix: the exporter emits a renderable sentence with a real day count, and the
export refuses to ship a raised flag that carries nothing to render.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.export_web_bundle import (  # noqa: E402
    partial_month_disclosure, validate_partial_month,
)

BUNDLE = ROOT / "web" / "public" / "bundle"


def panel(dates):
    return pd.DataFrame({"date": pd.to_datetime(dates), "ticker": "AAA"})


# ------------------------------------------------------------------ the computation
def test_complete_final_month_discloses_nothing():
    p = panel(["2026-06-01", "2026-06-15", "2026-06-30"])
    d = partial_month_disclosure(p, pd.Timestamp("2026-06-30"))
    assert d["axis_last_month_partial"] is False
    assert d["axis_last_month_days"] is None
    assert d["axis_last_month_text"] is None


def test_partial_final_month_counts_the_days_actually_present():
    # The live case: panel stops 2026-06-16, bucket labelled 2026-06-30.
    days = pd.bdate_range("2026-06-01", "2026-06-16")
    d = partial_month_disclosure(panel(days), pd.Timestamp("2026-06-30"))
    assert d["axis_last_month_partial"] is True
    assert d["axis_last_month_days"] == len(days) == 12
    assert "partial" in d["axis_last_month_text"].lower()
    assert "12 trading days" in d["axis_last_month_text"]
    assert "2026-06-16" in d["axis_last_month_text"]


def test_day_count_ignores_earlier_months_and_duplicate_tickers():
    # Many tickers share each date, and the panel holds years of history — neither may
    # inflate the count, which is a count of DATES inside the final bucket only.
    rows = []
    for t in ("AAA", "BBB", "CCC"):
        for dt in list(pd.bdate_range("2026-04-01", "2026-05-29")) + \
                  list(pd.bdate_range("2026-06-01", "2026-06-03")):
            rows.append({"date": dt, "ticker": t})
    d = partial_month_disclosure(pd.DataFrame(rows), pd.Timestamp("2026-06-30"))
    assert d["axis_last_month_days"] == 3


def test_one_day_final_month_reads_as_a_day_not_days():
    # Not cosmetic: the rebuilt wide panel really does end with a ONE-day final bucket, so
    # this is the sentence the live site would print.
    d = partial_month_disclosure(panel(["2026-08-03"]), pd.Timestamp("2026-08-31"))
    assert d["axis_last_month_days"] == 1
    assert "1 trading day to 2026-08-03" in d["axis_last_month_text"]
    assert "trading days" not in d["axis_last_month_text"]


def test_no_axis_end_is_not_a_partial_month():
    d = partial_month_disclosure(panel(["2026-06-16"]), None)
    assert d["axis_last_month_partial"] is False


# ------------------------------------------------------------------ the export gate
def test_validator_passes_a_complete_month():
    assert validate_partial_month("x.json", {"axis_last_month_partial": False}) == []


def test_validator_refuses_a_flag_with_nothing_to_render():
    errs = validate_partial_month("x.json", {"axis_last_month_partial": True})
    assert len(errs) == 2                       # missing day count AND missing text
    assert any("axis_last_month_text" in e for e in errs)
    assert any("axis_last_month_days" in e for e in errs)


def test_validator_refuses_a_zero_day_count():
    errs = validate_partial_month("x.json", {
        "axis_last_month_partial": True, "axis_last_month_days": 0,
        "axis_last_month_text": "Final month is partial — stats include it."})
    assert any("axis_last_month_days" in e for e in errs)


def test_validator_accepts_the_real_shape():
    assert validate_partial_month("x.json", partial_month_disclosure(
        panel(pd.bdate_range("2026-06-01", "2026-06-16")),
        pd.Timestamp("2026-06-30"))) == []


# ------------------------------------------------------------------ the shipped artifact
@pytest.mark.parametrize("name", ["index.json", "stocks.json", "explore.json"])
def test_shipped_bundle_discloses_its_partial_month(name):
    p = BUNDLE / name
    if not p.exists():
        pytest.skip(f"{name} is a gitignored build artifact")
    b = json.loads(p.read_text())
    # The pre-fix bundle carries the bare boolean and no `axis_last_month_days` KEY at all;
    # after the fix the key is always present (int or null). That absence is the only honest
    # discriminator, and it is why this skips rather than fails: the shipped bundle is
    # deliberately pinned to the DEPLOYED data state (#26d), so it cannot be regenerated here
    # without shipping the deferred mega-caps. It goes green on the next approved re-export.
    if "axis_last_month_days" not in b:
        pytest.skip(f"{name} predates the A-3 fix — goes green on the next re-export")
    assert validate_partial_month(name, b) == []
