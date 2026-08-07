"""#31 Arms 2 and 3 — harness tests.

Arm 2's claim is "identical protocol, different library". Arm 3's is "market-neutral, and
the borrow cost is charged, not assumed away". Both are structural claims, so both are
tested structurally rather than by eyeballing the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import lab.long_short as ls  # noqa: E402
from signals import lgbm_ranker as lg  # noqa: E402
from signals import xgb_ranker as xg  # noqa: E402


# ------------------------------------------------------------------ arm 2: same protocol
def test_xgb_shares_the_lgbm_walk_forward_not_a_copy_of_it():
    """The protocol is shared CODE. A second implementation would have to be trusted to
    match; this one cannot drift because there is only one fold loop."""
    p = xg.protocol_matches_lgbm()
    assert p["shared_walk_forward"] == "signals.lgbm_ranker.walk_forward"
    assert p["features"] == list(lg.FEATURES)
    assert p["label"] == lg.LABEL
    assert (p["initial_train"], p["step"], p["embargo"]) == (lg.INITIAL_TRAIN, lg.STEP, lg.EMBARGO)


def test_walk_forward_accepts_a_rival_estimator_and_defaults_to_lgbm():
    import inspect
    assert inspect.signature(lg.walk_forward).parameters["fit_fold"].default is None


@pytest.mark.parametrize("key,lgbm_key", [
    ("n_estimators", "n_estimators"), ("learning_rate", "learning_rate"),
    ("max_depth", "max_depth"), ("subsample", "subsample"),
    ("colsample_bytree", "colsample_bytree"), ("reg_lambda", "reg_lambda"),
    ("reg_alpha", "reg_alpha"), ("random_state", "random_state"),
])
def test_the_mappable_hyperparameters_are_actually_copied(key, lgbm_key):
    """Every parameter that CAN carry across must, or 'identical configuration' is a story."""
    assert xg.XGB_PARAMS[key] == lg.PARAMS[lgbm_key]


def test_the_unmappable_parameters_are_documented_not_silently_dropped():
    """num_leaves and min_child_samples do not map. The honest move is to name them."""
    doc = xg.__doc__
    assert "num_leaves" in doc and "min_child_samples" in doc
    assert "min_child_weight" in doc
    assert xg.XGB_PARAMS["min_child_weight"] == lg.PARAMS["min_child_samples"]


def test_no_hyperparameter_search_crept_in():
    """#31 rails: a search is a different experiment with its own multiple-testing cost."""
    assert "No tuning" in xg.__doc__ or "no tuning" in xg.__doc__.lower()


# ------------------------------------------------------------------ arm 3: the short book
def test_borrow_is_charged_per_month_not_annually():
    assert ls.borrow_drag(0.12) == pytest.approx(0.01)
    assert ls.borrow_drag(0.0) == 0.0


def test_borrow_reduces_the_return_it_is_supposed_to_reduce():
    """A cost that does not change the answer is not being applied."""
    df = pd.read_parquet(ls.OOS_V2) if ls.OOS_V2.exists() else None
    if df is None:
        pytest.skip("v2 OOS parquet is a gitignored build artifact")
    df["date"] = pd.to_datetime(df["date"])
    from signals.baseline_momentum import rebalance_dates
    rebals = rebalance_dates(df)
    free, _, _ = ls.score_book(df, "model_score", rebals, annual_borrow=0.0)
    paid, _, _ = ls.score_book(df, "model_score", rebals, annual_borrow=0.05)
    assert paid["sharpe"] < free["sharpe"]
    assert paid["ann_ret"] < free["ann_ret"]


@pytest.mark.skipif(not ls.OOS_V2.exists(), reason="v2 OOS parquet is gitignored")
def test_the_decile_profile_discriminates_the_two_scores():
    """The whole Arm 3 case rests on the ML's profile being monotone and momentum's not."""
    df = pd.read_parquet(ls.OOS_V2)
    df["date"] = pd.to_datetime(df["date"])
    ml = ls.decile_profile(df, "model_score")
    mom = ls.decile_profile(df, "mom_12_1m")
    rank = pd.Series(range(10))
    ml_mono = pd.Series(ml.values).corr(rank, method="spearman")
    mom_mono = pd.Series(mom.values).corr(rank, method="spearman")
    assert ml_mono > 0.9, "the ML profile should be near-monotone"
    assert mom_mono < ml_mono, "momentum's profile is the non-monotone one"
    # momentum's bottom decile is not a short candidate — it is one of its best deciles
    assert mom.iloc[0] > mom.iloc[1:5].max()


def test_the_short_leg_convention_is_market_neutral():
    """+100% long / -100% short by construction — the claim the book rests on."""
    from signals.baseline_momentum import _leg_weights
    sub = pd.DataFrame({"ticker": list("ABCDE"), "vol_6m": [0.01, 0.02, 0.03, 0.04, 0.05]})
    assert _leg_weights(sub, +1.0).sum() == pytest.approx(1.0)
    assert _leg_weights(sub, -1.0).sum() == pytest.approx(-1.0)
