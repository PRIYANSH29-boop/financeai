#!/usr/bin/env python3
"""
Build the v2 (A-1-fixed) artifacts — #31 Arm 1.

⚠️ EDUCATIONAL SIMULATION. This trains a SECOND, clearly-labelled frozen model. The shipped
v1 model and its stored walk-forward predictions are the historical record and are **not
touched**: every v2 artifact is written to its own `*_v2` path.

What differs from v1, and nothing else
--------------------------------------
Exactly one thing: `size` is now `log(close)` (the price that actually traded) instead of
`log(adj_close)` (a retroactively re-written series). That was finding **A-1** — the value at
a past date moved whenever a later split or dividend landed, so the feature leaked future
information. Architecture, hyperparameters, feature list, label, walk-forward schedule and
embargo are all imported from `signals.lgbm_ranker` unchanged, so the rematch is attributable
to the fix and to nothing else.

    python scripts/build_v2.py            # features -> labels -> retrain -> OOS parquet

Writes:
    data/sp500_features_v2.parquet
    data/sp500_labeled_v2.parquet
    data/sp500_oos_walkforward_v2.parquet
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.lgbm_ranker import FEATURES, LABEL, PARAMS, walk_forward  # noqa: E402
from utils.sp500_features import build_features  # noqa: E402
from utils.sp500_labels import build_labels  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_v2")

PANEL = "data/sp500_panel.parquet"
FEATURES_V2 = "data/sp500_features_v2.parquet"
LABELED_V2 = "data/sp500_labeled_v2.parquet"
OOS_V2 = "data/sp500_oos_walkforward_v2.parquet"

V1_FEATURES = Path("data/sp500_features.parquet")
V1_OOS = Path("data/sp500_oos_walkforward.parquet")


def main() -> int:
    for p in (Path(PANEL), V1_OOS):
        if not p.exists():
            print(f"ERROR: {p} missing — run `make pipeline` first.", file=sys.stderr)
            return 2

    v1_mtime = V1_FEATURES.stat().st_mtime if V1_FEATURES.exists() else None

    logger.info("v2 step 1/3 — features (size = log(close), the A-1 fix)")
    build_features(panel_path=PANEL, out_path=FEATURES_V2)

    logger.info("v2 step 2/3 — labels")
    build_labels(features_path=FEATURES_V2, panel_path=PANEL, out_path=LABELED_V2)

    logger.info("v2 step 3/3 — walk-forward retrain (identical architecture to v1)")
    labeled = pd.read_parquet(LABELED_V2)
    labeled["date"] = pd.to_datetime(labeled["date"])
    oos = walk_forward(labeled)
    oos.to_parquet(OOS_V2, index=False)

    # The whole point of writing to *_v2 paths: prove v1 was not disturbed.
    if v1_mtime is not None:
        assert V1_FEATURES.stat().st_mtime == v1_mtime, \
            "v1 feature table was modified — v2 must never overwrite the historical record"

    v1 = pd.read_parquet(V1_OOS, columns=["date", "ticker", "model_score"])
    logger.info("v2 OOS rows %d (v1 %d) | dates %s -> %s",
                len(oos), len(v1),
                pd.to_datetime(oos["date"]).min().date(),
                pd.to_datetime(oos["date"]).max().date())
    print(f"\nv2 artifacts written:\n  {FEATURES_V2}\n  {LABELED_V2}\n  {OOS_V2}")
    print(f"v1 untouched: {V1_FEATURES} mtime unchanged, {V1_OOS} not rewritten")
    print(f"features: {FEATURES}\nlabel: {LABEL}\nparams identical to v1: "
          f"{PARAMS['objective']}, n_estimators={PARAMS['n_estimators']}, "
          f"lr={PARAMS['learning_rate']}, seed={PARAMS['random_state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
