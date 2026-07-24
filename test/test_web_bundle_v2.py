"""
Phase 23 — validate the committed web bundle's new files (stocks.json, explore.json) and
cross-check the exported per-stock stats against analytics.metrics. Runs anywhere: the
bundle JSON is committed, so no source parquets / network are needed.

This is the Python half of the #23 test pair; the JS half (web/lib/basket.test.js, run via
`npm test` in web/) verifies the client math mirrors these same analyser conventions.
"""

from pathlib import Path
import json

import numpy as np
import pytest

from analytics.metrics import volatility, beta as _beta, sharpe

BUNDLE = Path("web/public/bundle")
STOCKS = BUNDLE / "stocks.json"
EXPLORE = BUNDLE / "explore.json"

pytestmark = pytest.mark.skipif(not STOCKS.exists(),
                                reason="web bundle not built (run make web-bundle)")


@pytest.fixture(scope="module")
def stocks():
    return json.loads(STOCKS.read_text())


@pytest.fixture(scope="module")
def explore():
    return json.loads(EXPLORE.read_text()) if EXPLORE.exists() else None


# ------------------------------------------------------------------ stocks.json
def test_stocks_has_as_of_and_aligned_series(stocks):
    assert stocks["as_of"], "stocks.json missing as_of stamp"
    n = len(stocks["dates"])
    assert n > 0
    assert len(stocks["benchmark_returns"]) == n
    for tk, arr in stocks["returns"].items():
        assert len(arr) == n, f"{tk} series length {len(arr)} != {n}"
        assert tk in stocks["stats"], f"{tk} has a series but no stats"


def test_exported_stats_match_analytics_metrics(stocks):
    # Recompute ann_vol / beta from the shipped series and confirm the exporter's stats
    # agree with analytics.metrics — the exporter did not fudge them.
    bench = np.array([np.nan if v is None else v for v in stocks["benchmark_returns"]])
    checked = 0
    for tk in list(stocks["returns"])[:25]:
        r = np.array([np.nan if v is None else v for v in stocks["returns"][tk]])
        mask = ~np.isnan(r)
        rr = r[mask]
        if rr.size < 24:
            continue
        # tolerance 1e-4: the bundle rounds each return to 6dp, so recomputing from the
        # shipped (rounded) series drifts slightly from the exporter's full-precision stat.
        # A fudged stat would be off by far more than this.
        assert abs(volatility(rr) - stocks["stats"][tk]["ann_vol"]) < 1e-4
        b_ref = _beta(rr, bench[mask])
        assert abs(b_ref - stocks["stats"][tk]["beta"]) < 1e-4
        checked += 1
    assert checked > 0, "no stocks with enough history to cross-check"


def test_scored_universe_is_nonempty_and_reasonable(stocks):
    assert len(stocks["returns"]) >= 400   # ~500 S&P names expected


# ------------------------------------------------------------------ explore.json
def test_explore_rows_complete(explore):
    if explore is None:
        pytest.skip("explore.json absent (wide universe not built)")
    assert explore["as_of"]
    assert explore["n_names"] == len(explore["rows"])
    for row in explore["rows"]:
        assert row["sector"] not in (None, "", "?"), f"{row['ticker']} unlabelled sector"
        assert row["cap_bucket"] in ("mid", "large")


def test_explore_movers_sorted_and_scored_flag(explore):
    if explore is None:
        pytest.skip("explore.json absent")
    winners = explore["movers"]["winners"]
    rets = [w["last_return"] for w in winners]
    assert rets == sorted(rets, reverse=True), "winners not sorted desc"
    # at least some names are model-scored (basket-eligible), and fewer than the full universe
    assert 0 < explore["n_scored"] < explore["n_names"]
