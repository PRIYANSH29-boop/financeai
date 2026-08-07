"""#30 Part D — harness tests for the Signal Duel (#27).

The duel's credibility rests on three things, so those are what is tested:

  1. **One variable.** Both books must run through identical construction, differing only in
     the score. If the harness let anything else vary, the result would be unattributable.
  2. **The rule was pre-stated and is applied as arithmetic.** `decide()` is tested against
     synthetic inputs, so it is provably not a narration of whatever the data happened to
     show — including the case where the ML wins, which the real data does not produce.
  3. **The ML score is out-of-sample.** Scoring the frozen model over its own training window
     would rig the duel in book B's favour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import lab.signal_duel as sd  # noqa: E402


# ------------------------------------------------------------------ the decision rule
def _fake(sharpe_a, sharpe_b, ic_by_year):
    """(full, full_ic, per_year) shaped like run() produces, from chosen inputs."""
    full = {sd.BOOK_A: {"sharpe": sharpe_a}, sd.BOOK_B: {"sharpe": sharpe_b}}
    full_ic = {sd.BOOK_A: {"mean_ic": 0.0}, sd.BOOK_B: {"mean_ic": 0.0}}
    per_year = {y: {sd.BOOK_A: {"mean_ic": a}, sd.BOOK_B: {"mean_ic": b}}
                for y, (a, b) in ic_by_year.items()}
    return full, full_ic, per_year


def test_ml_must_win_BOTH_tests_to_earn_its_keep():
    """Sharpe win alone is not enough — the IC edge has to be consistent too."""
    v = sd.decide(*_fake(1.0, 2.0, {2023: (0.05, 0.01), 2024: (0.05, 0.01),
                                    2025: (0.01, 0.05)}))
    assert v["ml_wins_sharpe"] is True
    assert v["ic_consistent"] is False          # ML won IC in 1 of 3 years
    assert v["ml_earns_keep"] is False
    assert "TRADE MOMENTUM" in v["headline"]


def test_consistent_ic_alone_is_not_enough_either():
    v = sd.decide(*_fake(2.0, 1.0, {2023: (0.01, 0.05), 2024: (0.01, 0.05),
                                    2025: (0.01, 0.05)}))
    assert v["ic_consistent"] is True
    assert v["ml_wins_sharpe"] is False
    assert v["ml_earns_keep"] is False


def test_an_ml_win_is_reported_as_INCONCLUSIVE_not_as_a_win():
    """The asymmetric ruling. This branch never fires on the real data, so it is tested
    synthetically — otherwise the ruling would be untested prose."""
    v = sd.decide(*_fake(1.0, 2.0, {2023: (0.01, 0.05), 2024: (0.01, 0.05),
                                    2025: (0.01, 0.05)}))
    assert v["ml_earns_keep"] is True
    assert "INCONCLUSIVE" in v["strength"]
    assert "A-1" in v["strength"]


def test_an_ml_loss_is_reported_as_CONCLUSIVE():
    v = sd.decide(*_fake(2.0, 1.0, {2023: (0.05, 0.01), 2024: (0.05, 0.01)}))
    assert v["ml_earns_keep"] is False
    assert "CONCLUSIVE" in v["strength"]
    assert "advantage" in v["strength"]


def test_a_tie_on_sharpe_goes_to_momentum():
    """#27: 'Ties or momentum wins ⇒ trade momentum'. A tie must not be an ML win."""
    v = sd.decide(*_fake(1.5, 1.5, {2023: (0.01, 0.05), 2024: (0.01, 0.05)}))
    assert v["ml_wins_sharpe"] is False
    assert v["ml_earns_keep"] is False


def test_the_consistency_threshold_is_a_strict_majority():
    """Exactly half the years is NOT consistent — the rule says 'more than'."""
    v = sd.decide(*_fake(1.0, 2.0, {2023: (0.01, 0.05), 2024: (0.05, 0.01)}))
    assert v["ic_consistency"] == pytest.approx(0.5)
    assert v["ic_consistent"] is False


# ------------------------------------------------------------------ rank IC
def test_rank_ic_is_perfect_for_a_perfect_score():
    day = pd.DataFrame({"date": pd.Timestamp("2024-01-31"),
                        "ticker": [f"T{i:02d}" for i in range(30)],
                        "fwd_ret_1m": np.linspace(-0.1, 0.1, 30)})
    ic = sd.rank_ic(day, lambda d: pd.Series(d["fwd_ret_1m"].to_numpy(),
                                             index=d["ticker"].to_numpy()),
                    [pd.Timestamp("2024-01-31")])
    assert ic.iloc[0] == pytest.approx(1.0)


def test_rank_ic_is_inverted_for_a_perfectly_wrong_score():
    day = pd.DataFrame({"date": pd.Timestamp("2024-01-31"),
                        "ticker": [f"T{i:02d}" for i in range(30)],
                        "fwd_ret_1m": np.linspace(-0.1, 0.1, 30)})
    ic = sd.rank_ic(day, lambda d: pd.Series(-d["fwd_ret_1m"].to_numpy(),
                                             index=d["ticker"].to_numpy()),
                    [pd.Timestamp("2024-01-31")])
    assert ic.iloc[0] == pytest.approx(-1.0)


def test_ic_summary_reports_n_and_a_t_stat():
    s = sd.ic_summary(pd.Series([0.05, 0.05, 0.05, 0.05]))
    assert s["n"] == 4 and s["mean_ic"] == pytest.approx(0.05)
    s2 = sd.ic_summary(pd.Series([0.5, -0.4, 0.3, -0.2]))
    assert abs(s2["t_stat"]) < 1.0, "a noisy IC must not produce a confident t"


# ------------------------------------------------------------------ the shipped result
@pytest.mark.skipif(not sd.OOS_PATH.exists(), reason="walk-forward OOS parquet is gitignored")
def test_the_ml_score_column_is_the_walk_forward_oos_one():
    """Book B must be scored from walk-forward predictions, not from re-running the frozen
    model over its own training window."""
    df = sd.load_oos()
    assert "model_score" in df.columns
    day = df[df["date"] == df["date"].max()]
    s = sd.ml_score(day)
    assert len(s) == len(day)
    assert np.isfinite(s.to_numpy()).all()


@pytest.mark.skipif(not sd.OOS_PATH.exists(), reason="walk-forward OOS parquet is gitignored")
def test_both_books_see_an_identical_universe_on_every_date():
    """'One variable' is the whole claim. Both scores are computed on the same frame, so any
    difference in the books can only come from the score."""
    df = sd.load_oos()
    for t in sd.monthly_rebalances(df)[:6]:
        day = df[df["date"] == t]
        a, b = sd.momentum_score(day), sd.ml_score(day)
        assert set(a.index) == set(b.index) == set(day["ticker"])


@pytest.mark.skipif(not sd.REPORT_PATH.exists(), reason="report not generated yet")
def test_the_report_states_the_rule_and_the_asymmetry():
    """A verdict without its pre-stated rule and its known contaminant printed beside it is
    not a result anyone can audit."""
    t = sd.REPORT_PATH.read_text()
    assert "pre-stated decision rule" in t.lower()
    assert "A-1" in t
    assert "CONCLUSIVE" in t and "INCONCLUSIVE" in t
    assert "EDUCATIONAL SIMULATION" in t
    assert "urvivorship" in t


# ============================================================ #31 Arm 1 — the v2 rematch

@pytest.mark.skipif(not Path("data/sp500_oos_walkforward_v2.parquet").exists(),
                    reason="v2 OOS parquet is a gitignored build artifact")
def test_the_harness_can_score_a_different_model_version():
    """Arm 1's whole method: the SAME duel harness, a different model, so the rematch is
    comparable to the original by construction rather than by re-implementation."""
    res = sd.run(make_report=False, oos_path=Path("data/sp500_oos_walkforward_v2.parquet"),
                 ml_label="B · v2")
    assert res["ml_label"] == "B · v2"
    assert set(res["books"]) == {sd.BOOK_A, "B · v2"}
    assert res["n_months"] > 40


def test_the_defaults_still_reproduce_the_published_duel():
    """Parameterising run() must not have moved the #30 result. The momentum control is the
    same series either way, so its Sharpe is the tell."""
    import inspect
    sig = inspect.signature(sd.run)
    assert sig.parameters["oos_path"].default == sd.OOS_PATH
    assert sig.parameters["ml_label"].default == sd.BOOK_B
    assert sig.parameters["report_path"].default == sd.REPORT_PATH
