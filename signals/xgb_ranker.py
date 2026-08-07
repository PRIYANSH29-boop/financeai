"""
XGBoost learning-to-rank challenger — #31 Arm 2, the controlled rival.

⚠️ EDUCATIONAL SIMULATION. Nothing here is advice.

The question this arm answers is narrow and worth answering exactly: **is the ML's loss to
momentum a property of the LIBRARY, or of the signal?** So this is deliberately not a better
model — it is the same model family, same features, same label, same folds, same embargo,
built by a different implementation.

It reuses `signals.lgbm_ranker.walk_forward` with a swapped `fit_fold`, so the fold
boundaries, expanding window, 21-day embargo, group structure, feature list and label are
*literally the same code*. Only the estimator differs. A second implementation of the
protocol would have had to be trusted to match; this one cannot drift.

Hyperparameter mapping — honest about what "identical" can mean across libraries
--------------------------------------------------------------------------------
LightGBM and XGBoost do not share a parameter space, so an exact copy is impossible. Each
value below is the nearest equivalent of the frozen v1 config, and the two that do NOT map
cleanly are named rather than buried:

  * `num_leaves=15` has no XGBoost equivalent. XGBoost grows depth-wise; LightGBM grows
    leaf-wise. With `max_depth=4` a depth-wise tree can hold up to 16 leaves, so the
    capacities are close but the SHAPES differ. This is the main irreducible difference.
  * `min_child_samples=100` (a row count) becomes `min_child_weight=100`. For a ranking
    objective these are not the same quantity; the row count is the intent.

Everything else — n_estimators, learning_rate, max_depth, subsample, colsample_bytree,
reg_lambda, reg_alpha, random_state — carries across directly.

**No tuning.** The #31 rails put a hyperparameter search explicitly out of scope: that would
be a different experiment carrying its own multiple-testing cost. If XGBoost loses here, the
claim is "the same configuration in another library does not rescue the signal" — NOT "no
XGBoost configuration could".
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from signals.lgbm_ranker import EMBARGO, FEATURES, INITIAL_TRAIN, LABEL, PARAMS, STEP

logger = logging.getLogger("xgb_ranker")

# Nearest equivalent of the frozen v1 LightGBM config. See the module docstring for the two
# parameters that do not map cleanly.
XGB_PARAMS = dict(
    objective="rank:ndcg",
    n_estimators=PARAMS["n_estimators"],          # 300
    learning_rate=PARAMS["learning_rate"],        # 0.02
    max_depth=PARAMS["max_depth"],                # 4
    min_child_weight=PARAMS["min_child_samples"],  # 100 — row count -> weight, see docstring
    subsample=PARAMS["subsample"],                # 0.8
    colsample_bytree=PARAMS["colsample_bytree"],  # 0.8
    reg_lambda=PARAMS["reg_lambda"],              # 5.0
    reg_alpha=PARAMS["reg_alpha"],                # 1.0
    random_state=PARAMS["random_state"],          # 42
    n_jobs=-1,
    verbosity=0,
)


def fit_fold(train: pd.DataFrame) -> XGBRanker:
    """Fit one XGBoost ranker on a fold. Group = rows per date, as LightGBM's does."""
    group = train.groupby("date", sort=False).size().to_numpy()
    model = XGBRanker(**XGB_PARAMS)
    model.fit(train[FEATURES], train[LABEL], group=group)
    return model


def protocol_matches_lgbm() -> dict:
    """The claim 'identical protocol' as machine-checkable facts, not prose."""
    return {"features": list(FEATURES), "label": LABEL, "initial_train": INITIAL_TRAIN,
            "step": STEP, "embargo": EMBARGO,
            "shared_walk_forward": "signals.lgbm_ranker.walk_forward"}
