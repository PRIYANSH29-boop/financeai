"""
Unit tests for the Phase 15 beta-targeted pie engine — the PURE construction logic
(weight caps + beta targeting). These do NOT fit the frozen model, so they are fast and
run offline; the model-scoring path is exercised by `python -m portfolio.beta_engine`.

Pins the two invariants the #15 deliverable rests on:
  * weights obey the per-name and per-sector caps and sum to 1;
  * portfolio beta is linear in weights, so a cash sleeve hits target exactly
    (target ≤ book beta) and an impossible target resets rather than fakes.

Run: pytest portfolio/tests/ -q
"""

import numpy as np
import pandas as pd

from portfolio.beta_engine import _apply_caps, _hit_target_beta, NAME_CAP, SECTOR_CAP


def _sectors(n_by_sector):
    idx, sec = [], []
    for s, n in n_by_sector.items():
        for i in range(n):
            idx.append(f"{s}{i}")
            sec.append(s)
    return pd.Series(sec, index=idx)


# ------------------------------------------------------------------ weight caps
# NB: with a 30% sector cap the constraint is only feasible for ≥4 sectors
# (4 × 0.30 = 1.20 ≥ 1). The real path guarantees this: `_select` caps names per
# sector at SECTOR_MAX_NAMES=5, so 20 holdings always span ≥4 sectors.
def test_apply_caps_respects_name_and_sector_caps():
    sectors = _sectors({"Tech": 5, "Fin": 5, "Health": 5, "Energy": 5})  # 20 names, 4 sectors
    raw = pd.Series(1.0, index=sectors.index)                # start equal-weight
    w = _apply_caps(raw, sectors)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w <= NAME_CAP + 1e-6).all()
    sec_tot = w.groupby(sectors).sum()
    assert (sec_tot <= SECTOR_CAP + 1e-6).all()


def test_apply_caps_concentrated_input_still_diversifies():
    """A hugely skewed input must still be pulled under both caps."""
    sectors = _sectors({"Tech": 5, "Fin": 5, "Health": 5, "Energy": 5})
    raw = pd.Series(np.linspace(1, 100, 20), index=sectors.index)
    w = _apply_caps(raw, sectors)
    assert (w <= NAME_CAP + 1e-6).all()
    assert (w.groupby(sectors).sum() <= SECTOR_CAP + 1e-6).all()


# ------------------------------------------------------------------ beta targeting
def test_cash_sleeve_hits_target_exactly():
    """target ≤ book beta → cash sleeve; achieved == target by linearity."""
    idx = [f"n{i}" for i in range(10)]
    w = pd.Series(0.1, index=idx)                            # fully invested
    betas = pd.Series(np.linspace(0.8, 1.6, 10), index=idx)  # book beta 1.2
    sectors = pd.Series("Tech", index=idx)
    final, cash, achieved, note = _hit_target_beta(w, betas, sectors, target=0.6)
    assert note is None
    assert abs(achieved - 0.6) < 1e-9
    assert abs(float((final * betas).sum()) - 0.6) < 1e-9   # realised == target
    assert abs(cash - (1 - final.sum())) < 1e-9
    assert cash > 0


def test_impossible_target_resets_not_fakes():
    """target above the max achievable long-only beta → cap + reset note, no leverage."""
    idx = [f"n{i}" for i in range(8)]
    w = pd.Series(0.125, index=idx)
    betas = pd.Series(np.linspace(0.9, 1.3, 8), index=idx)   # max any single name = 1.3
    sectors = pd.Series("Tech", index=idx)
    final, cash, achieved, note = _hit_target_beta(w, betas, sectors, target=3.0)
    assert note is not None and "exceeds" in note
    assert achieved < 3.0
    assert achieved <= betas.max() + 1e-9                    # can't beat the highest-beta name
    assert cash == 0.0
