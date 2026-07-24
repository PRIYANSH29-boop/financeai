"""
Tests for the Phase 21 regime-segmented backtest.

Two fast unit tests on the committed pieces (no committed data / network needed):
  1. `classify_regimes` on a SYNTHETIC benchmark with a hand-known answer — a >10%
     drawdown month must land in 'stressed'; a flat low-vol near-peak month must be 'calm'.
  2. Per-regime beta hand-check: a book that is exactly 2× the benchmark must score β = 2.0
     inside a regime, via `regime_stats`.
Plus one integration smoke test (skipped if committed data is absent).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lab.regime_backtest import classify_regimes, regime_stats, run, DD_STRESS


def _monthly_index(n):
    return pd.date_range("2021-01-31", periods=n, freq="ME")


# --------------------------------------------------------------------- 1. regime labelling
def test_classify_regimes_known_answer():
    # 8 gently-wiggling low-vol months (price climbing → near peak), then a −20% crash,
    # then a partial recovery. The wiggle (not a flat line) keeps rolling vol small but
    # DISTINCT so the tercile split is non-degenerate. Hand-known: the crash month is a
    # >10% drawdown → stressed; an early low-vol near-peak month sits in the bottom vol
    # tercile within 3% of peak → calm.
    rets = [0.006, 0.004] * 4 + [-0.20, 0.05, 0.03]
    bench = pd.Series(rets, index=_monthly_index(len(rets)))
    df = classify_regimes(bench)

    crash = df.index[8]                       # the −20% month
    assert df.loc[crash, "regime"] == "stressed"
    assert df.loc[crash, "drawdown"] < DD_STRESS
    assert df.loc[crash, "trigger"] in ("dd", "dd+vol")

    # at least one of the flat early months is calm (bottom-tercile vol AND near peak)
    early = df.iloc[:8]
    assert (early["regime"] == "calm").any()

    # only the three labels are ever emitted
    assert set(df["regime"]) <= {"calm", "normal", "stressed"}


def test_classify_regimes_drawdown_leg_is_deterministic():
    # A clean −12% single month from an all-time high is unambiguously stressed regardless
    # of the vol tercile — the drawdown leg does not depend on the tercile split.
    bench = pd.Series([0.03, 0.02, 0.01, -0.12, 0.02],
                      index=_monthly_index(5))
    df = classify_regimes(bench)
    assert df.iloc[3]["regime"] == "stressed"
    assert df.iloc[3]["drawdown"] < DD_STRESS


# ------------------------------------------------------------------- 2. per-regime beta
def test_regime_beta_hand_check():
    # port = exactly 2× benchmark ⇒ cov/var = 2.0 on any set of months.
    idx = _monthly_index(4)
    bench = pd.Series([0.02, -0.03, 0.01, 0.04], index=idx)
    port = 2.0 * bench
    labels = pd.Series(["stressed"] * 4, index=idx)   # all four months one regime

    stats = regime_stats(port, bench, labels)
    assert stats["stressed"]["n"] == 4
    assert stats["stressed"]["beta"] == pytest.approx(2.0, abs=1e-9)
    # benchmark against itself is β = 1.0
    self_stats = regime_stats(bench, bench, labels)
    assert self_stats["stressed"]["beta"] == pytest.approx(1.0, abs=1e-9)


def test_regime_stats_hit_rate_and_empty_regime():
    idx = _monthly_index(4)
    port = pd.Series([0.01, -0.02, 0.03, 0.00], index=idx)
    bench = pd.Series([0.01, -0.01, 0.02, 0.01], index=idx)
    labels = pd.Series(["calm", "calm", "stressed", "stressed"], index=idx)
    stats = regime_stats(port, bench, labels)
    assert stats["calm"]["n"] == 2 and stats["stressed"]["n"] == 2
    assert stats["normal"]["n"] == 0 and stats["normal"]["beta"] is None
    # calm months: one +, one − ⇒ hit rate 50%
    assert stats["calm"]["hit"] == pytest.approx(0.5)


# --------------------------------------------------------------------- 3. integration smoke
@pytest.mark.skipif(not Path("data/sp500_panel.parquet").exists(),
                    reason="committed panel absent")
def test_run_smoke_and_sanity_gate():
    res = run(make_report=False)
    # the sanity gate inside run() would have raised if 2022 were not stressed
    assert 2022 in {int(m[:4]) for m in res["stressed_months"]}
    assert res["n_months"] > 40
    for reg in ("calm", "normal", "stressed"):
        assert res["stats"]["EW benchmark"][reg]["beta"] in (None, pytest.approx(1.0, abs=1e-9))
