"""#28 B-1/B-2/B-3/B-4 — the caps must fail CLOSED.

#25 findings: `_apply_caps` alternated for 200 passes and then `return w / w.sum()`
unconditionally, so a pool too small to absorb the deficit got back a weight vector that
violated the caps the function exists to enforce — silently, with no flag. An all-zero book
returned NaN weights, and a negative-beta book returned NaN weights while reporting
`achieved_beta = 0.0`.

The caps are the product's central safety claim. A safety rule that fails open is worse than
no rule, because the UI still says "max 8% per stock".

The audit's own probes are the fixtures here — each test is the exact repro from
AUDIT_FINDINGS.md, now asserting a refusal instead of documenting a violation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from portfolio.beta_engine import (  # noqa: E402
    NAME_CAP, SECTOR_CAP, CapsInfeasibleError, _apply_caps, _caps_violations,
    _hit_target_beta,
)


def healthy(n=20, sectors=6):
    """A pool that CAN satisfy the caps — the control for every refusal below."""
    tk = [f"T{i:02d}" for i in range(n)]
    w = pd.Series({t: 1.0 for t in tk})
    sec = pd.Series({t: f"S{i % sectors}" for i, t in enumerate(tk)})
    return w, sec


# ------------------------------------------------------------------ B-1: one name
def test_single_name_pool_is_refused_not_given_100_percent():
    """AUDIT B-1: `_apply_caps({A: 1.0})` returned `{A: 1.0}` — 100% in one name vs an 8% cap."""
    with pytest.raises(CapsInfeasibleError) as ei:
        _apply_caps(pd.Series({"A": 1.0}), pd.Series({"A": "Tech"}))
    assert "1 name " in str(ei.value)          # singular, and names the arithmetic
    assert "8%" in str(ei.value)


# ------------------------------------------------------------------ B-2: one sector
def test_three_names_one_sector_is_refused_not_given_a_third_each():
    """AUDIT B-2: returned 0.333 each — 100% in one sector vs a 30% cap."""
    w = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
    sec = pd.Series({"A": "T", "B": "T", "C": "T"})
    with pytest.raises(CapsInfeasibleError):
        _apply_caps(w, sec)


def test_large_single_sector_pool_is_refused_on_the_sector_cap():
    """Enough names to clear the per-name arithmetic, but all in one sector: the refusal has
    to come from the SECTOR cap, not from the pool-size shortcut."""
    w, _ = healthy(n=20)
    one_sector = pd.Series({t: "Tech" for t in w.index})
    with pytest.raises(CapsInfeasibleError) as ei:
        _apply_caps(w, one_sector)
    assert "sector" in str(ei.value).lower()


# ------------------------------------------------------------------ B-3: all-zero / non-finite
def test_all_zero_weights_are_refused_not_returned_as_nan():
    """AUDIT B-3: returned `{A: nan, B: nan}` — NaN weights, no raise."""
    w, sec = healthy()
    with pytest.raises(CapsInfeasibleError) as ei:
        _apply_caps(w * 0.0, sec)
    assert "sum" in str(ei.value).lower()


def test_empty_pool_is_refused():
    with pytest.raises(CapsInfeasibleError):
        _apply_caps(pd.Series(dtype=float), pd.Series(dtype=object))


def test_nan_weights_in_are_refused():
    w, sec = healthy()
    w.iloc[0] = np.nan
    with pytest.raises(CapsInfeasibleError):
        _apply_caps(w, sec)


# ------------------------------------------------------------------ B-4: beta targeting
def test_negative_book_beta_is_refused_not_reported_as_zero():
    """AUDIT B-4: returned NaN weights while reporting `achieved_beta = 0.0, cash = 0.0`."""
    w = pd.Series({"A": 0.5, "B": 0.5})
    betas = pd.Series({"A": -1.0, "B": -1.0})
    sec = pd.Series({"A": "T", "B": "T"})
    with pytest.raises(CapsInfeasibleError) as ei:
        _hit_target_beta(w, betas, sec, 0.5)
    assert "positive beta" in str(ei.value)


def test_nan_book_beta_is_refused():
    w, sec = healthy()
    w = w / w.sum()
    betas = pd.Series({t: np.nan for t in w.index})
    with pytest.raises(CapsInfeasibleError):
        _hit_target_beta(w, betas, sec, 0.5)


def test_a_reachable_target_still_works_after_the_guards():
    """The refusals must not have made the normal path stricter."""
    w, sec = healthy()
    w = _apply_caps(w, sec)
    betas = pd.Series({t: 1.0 for t in w.index})
    final, cash, achieved, _ = _hit_target_beta(w, betas, sec, 0.5)
    assert achieved == pytest.approx(0.5, abs=1e-9)
    assert cash == pytest.approx(0.5, abs=1e-9)
    assert np.isfinite(final.to_numpy(dtype="float64")).all()


# ------------------------------------------------------------------ the happy path is intact
def test_a_healthy_pool_is_capped_and_returned():
    w, sec = healthy()
    out = _apply_caps(w, sec)
    assert out.sum() == pytest.approx(1.0)
    assert out.max() <= NAME_CAP + 1e-9
    assert out.groupby(sec).sum().max() <= SECTOR_CAP + 1e-9
    assert _caps_violations(out, sec, NAME_CAP, SECTOR_CAP) == []


def test_the_verifier_actually_detects_a_planted_violation():
    """A verifier that never fires would make every test above vacuous."""
    w, sec = healthy()
    bad = _apply_caps(w, sec).copy()
    bad.iloc[0] = 0.5                                   # 50% in one name, 8% cap
    assert _caps_violations(bad, sec, NAME_CAP, SECTOR_CAP)
    assert any("per-name" in v for v in _caps_violations(bad, sec, NAME_CAP, SECTOR_CAP))


def test_the_live_case_the_audit_thought_was_only_latent():
    """#25 judged B-1/B-2 "latent — they need a degenerate pool the shipped 503-name universe
    never produces". It produces one: the #21 regime-backtest momentum book is 20 names in
    just 4 sectors, 13 of them Information Technology, which can hold at most 86% under the
    caps. The old code returned it with Information Technology at 44% against a 30% cap.

    Pinned as a fixture (not by re-running the backtest) so the arithmetic is checked without
    a parquet dependency.
    """
    counts = {"Information Technology": 13, "Communication Services": 3,
              "Industrials": 3, "Materials": 1}
    idx, sec = [], {}
    for s, n in counts.items():
        for i in range(n):
            tk = f"{s[:3]}{i}"
            idx.append(tk)
            sec[tk] = s
    w = pd.Series(1.0, index=idx)
    sectors = pd.Series(sec)
    capacity = sum(min(SECTOR_CAP, n * NAME_CAP) for n in counts.values())
    assert capacity == pytest.approx(0.86)              # 30% + 24% + 24% + 8%
    with pytest.raises(CapsInfeasibleError) as ei:
        _apply_caps(w, sectors)
    assert "86.0%" in str(ei.value)


def test_boundary_weight_exactly_at_the_cap_is_allowed():
    """`<` vs `<=` at the 1e-9 tolerance — an exactly-at-cap book is legal, not a breach.
    (AUDIT sweep-B listed this cell as not probed.)"""
    tk = [f"T{i:02d}" for i in range(int(round(1 / NAME_CAP)) + 1)]
    w = pd.Series({t: 1.0 for t in tk})
    sec = pd.Series({t: f"S{i}" for i, t in enumerate(tk)})
    out = _apply_caps(w, sec)
    assert _caps_violations(out, sec, NAME_CAP, SECTOR_CAP) == []
