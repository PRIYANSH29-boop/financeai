"""#29 F-2 — the missing tests for `utils/`, the feature and label builders.

`utils/` builds every feature the frozen ranker consumes and the one column that deliberately
looks into the future. It is the highest leakage-risk code in the repo, and it had **no
dedicated test file** — it was also absent from #25's sweep-A scope and from the Notion
documentation map (finding C-2). Three independent maps of this project skipped the same
directory, which is why these tests exist.

What is pinned here is the *leakage geometry*, not the numbers:

  * backward-only  — a feature at date t must not move when a LATER price changes
  * per-ticker     — a grouped shift must never borrow another ticker's history
  * within-date    — a cross-sectional rank must not see other dates
  * the forward look is exactly one, is deliberate, and is 21 trading days

One test pins a known DEFECT rather than a guarantee — see `test_size_is_a_bare_price_level`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.sp500_features import (  # noqa: E402
    FEATURE_COLS, MIN_HISTORY, build_raw_features, cross_sectional_rank, sanity_gate,
)
from utils.sp500_labels import HORIZON, _forward_returns  # noqa: E402

N = 320          # comfortably over MIN_HISTORY so eligibility switches on mid-panel


def panel(tickers=("AAA",), n=N, seed=0, start="2020-01-01"):
    """A synthetic daily panel with the columns the builders actually read."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    # `close` and `adj_close` must DIFFER, or any test meant to tell them apart passes
    # vacuously. (They were identical here until #31 Arm 1, which is exactly how one A-1
    # tripwire stayed green through the fix it was built to detect.) The factor rises to 1.0
    # at the right edge, mimicking a real adjusted series: history is marked down, today is not.
    adj = np.linspace(0.85, 1.0, n)
    rows = []
    for i, tk in enumerate(tickers):
        px = 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, n)) + i
        for d, p, f in zip(dates, px, adj):
            rows.append({"date": d, "ticker": tk, "open": p, "high": p * 1.01,
                         "low": p * 0.99, "close": p, "adj_close": p * f,
                         "volume": 1_000_000 + i})
    return pd.DataFrame(rows)


def feats(df):
    out, _ = build_raw_features(df.copy())
    return out.set_index(["ticker", "date"])


# ====================================================================== backward-only
def test_a_later_price_cannot_change_an_earlier_feature():
    """The core leakage property. Change the FINAL price by 10x; every feature on every
    earlier date must be bit-identical. If any feature peeked forward, this moves."""
    base = panel()
    tampered = base.copy()
    tampered.loc[tampered.index[-1], "adj_close"] *= 10.0

    a, b = feats(base), feats(tampered)
    earlier = a.index[:-1]                       # everything except the tampered day
    for c in FEATURE_COLS:
        pd.testing.assert_series_equal(a.loc[earlier, c], b.loc[earlier, c],
                                       check_names=False)


def test_the_whole_tail_can_be_rewritten_without_moving_the_head():
    """Stronger version: rewrite the last 60 days entirely. The first 200 must not move."""
    base = panel()
    tampered = base.copy()
    tail = tampered.index[-60:]
    tampered.loc[tail, "adj_close"] = 999.0
    tampered.loc[tail, "volume"] = 5_000_000

    a, b = feats(base), feats(tampered)
    head = a.index[:200]
    for c in FEATURE_COLS:
        pd.testing.assert_series_equal(a.loc[head, c], b.loc[head, c], check_names=False)


# ====================================================================== per-ticker
def test_a_grouped_shift_never_borrows_another_ticker_history():
    """AAA has a full history, BBB starts late. BBB's early rows must be NaN — never AAA's
    prices bleeding across the group boundary, which a bare (ungrouped) shift would do."""
    a = panel(("AAA",), n=N)
    b = panel(("BBB",), n=40, start="2021-01-01")
    out = feats(pd.concat([a, b], ignore_index=True))

    bbb = out.loc["BBB"]
    assert bbb["mom_12_1m"].isna().all(), "BBB has 40 days; a 252-day lookback cannot exist"
    assert bbb["mom_6m"].isna().all(), "BBB has 40 days; a 126-day lookback cannot exist"
    assert not bbb["eligible"].any(), "40 days of history cannot be eligible"


def test_each_ticker_first_row_has_no_return():
    """A grouped pct_change must give NaN at each ticker's first observation, not a jump
    from the previous ticker's last price."""
    out = feats(panel(("AAA", "BBB", "CCC"), n=60))
    for tk in ("AAA", "BBB", "CCC"):
        assert np.isnan(out.loc[tk, "ret_1d"].iloc[0]), f"{tk} first row must have no return"


# ====================================================================== formulas
def test_momentum_12_1_skips_the_most_recent_month():
    """mom_12_1m is t-252 → t-21, deliberately excluding the last month (the reversal
    window). If it ever became t-252 → t it would silently double-count reversal."""
    df = panel()
    out = feats(df)
    ac = df.set_index("date")["adj_close"]
    t = 300
    expected = ac.iloc[t - 21] / ac.iloc[t - 252] - 1
    got = out.loc["AAA"].iloc[t]["mom_12_1m"]
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("col,lag", [("mom_3m", 63), ("mom_6m", 126), ("reversal_1m", 21)])
def test_trailing_windows_are_the_documented_lengths(col, lag):
    df = panel()
    out = feats(df)
    ac = df.set_index("date")["adj_close"]
    t = 300
    expected = ac.iloc[t] / ac.iloc[t - lag] - 1
    assert out.loc["AAA"].iloc[t][col] == pytest.approx(expected, rel=1e-12)


def test_vol_6m_is_a_126_day_rolling_window():
    df = panel()
    out = feats(df)
    r = df["adj_close"].pct_change()
    t = 300
    assert out.loc["AAA"].iloc[t]["vol_6m"] == pytest.approx(r.iloc[t - 125:t + 1].std(),
                                                             rel=1e-12)


# ====================================================================== eligibility
def test_eligibility_turns_on_at_exactly_min_history():
    out = feats(panel()).loc["AAA"]
    assert not out["eligible"].iloc[MIN_HISTORY - 2]
    assert out["eligible"].iloc[MIN_HISTORY - 1], f"row {MIN_HISTORY} must be the first eligible"


def test_eligible_means_every_feature_is_computable():
    """The claim the constant exists to make: MIN_HISTORY = 253 (not 252) because the
    longest feature needs adj_close[t-252], so the boundary row must have no NaN."""
    out = feats(panel()).loc["AAA"]
    first = out[out["eligible"]].iloc[0]
    for c in FEATURE_COLS:
        assert np.isfinite(first[c]), f"{c} is NaN on the first eligible row"


def test_min_history_is_253_not_252():
    """Pins the off-by-one the docstring used to contradict (#25 D-3)."""
    assert MIN_HISTORY == 253


# ====================================================================== within-date rank
def test_ranks_are_computed_within_each_date_only():
    """A rank that saw other dates would be a global scaler — a classic panel leak. Two
    dates with IDENTICAL cross-sections must produce identical ranks even though their
    raw levels differ by 100x."""
    rows = []
    for d, scale in [("2024-01-02", 1.0), ("2024-01-03", 100.0)]:
        for i, tk in enumerate("ABCDE"):
            rows.append({"date": pd.Timestamp(d), "ticker": tk,
                         **{c: (i + 1) * scale for c in FEATURE_COLS}})
    ranked = cross_sectional_rank(pd.DataFrame(rows))
    d1 = ranked[ranked["date"] == "2024-01-02"].set_index("ticker")
    d2 = ranked[ranked["date"] == "2024-01-03"].set_index("ticker")
    for c in FEATURE_COLS:
        pd.testing.assert_series_equal(d1[c + "_rank"], d2[c + "_rank"], check_names=False)


def test_ranks_are_percentiles_in_the_unit_interval():
    rows = [{"date": pd.Timestamp("2024-01-02"), "ticker": tk,
             **{c: float(i) for c in FEATURE_COLS}} for i, tk in enumerate("ABCDEFGHIJ")]
    ranked = cross_sectional_rank(pd.DataFrame(rows))
    for c in FEATURE_COLS:
        r = ranked[c + "_rank"]
        assert r.min() > 0 and r.max() == pytest.approx(1.0)


# ====================================================================== labels
def test_the_forward_look_is_exactly_21_trading_days():
    """The embargo rule in the module docstring is only correct if this constant is. A
    change here silently invalidates every train/test split downstream."""
    assert HORIZON == 21


def test_forward_return_is_the_documented_shift():
    df = panel()
    fwd = _forward_returns(df.copy()).set_index("date")["fwd_ret_1m"]
    ac = df.set_index("date")["adj_close"]
    t = 100
    expected = ac.iloc[t + HORIZON] / ac.iloc[t] - 1
    assert fwd.iloc[t] == pytest.approx(expected, rel=1e-12)


def test_forward_returns_never_cross_tickers():
    """The one deliberate forward look must still respect the group boundary — otherwise
    the last 21 rows of each ticker borrow the NEXT ticker's opening prices."""
    df = pd.concat([panel(("AAA",), n=60), panel(("BBB",), n=60, seed=1)], ignore_index=True)
    fwd = _forward_returns(df.copy())
    for tk in ("AAA", "BBB"):
        tail = fwd[fwd["ticker"] == tk].sort_values("date").tail(HORIZON)
        assert tail["fwd_ret_1m"].isna().all(), f"{tk}'s last {HORIZON} rows must have no future"


def test_only_the_label_looks_forward():
    """Belt and braces on the central claim: of every column the builders produce, exactly
    one moves when a future price changes."""
    base = panel()
    tampered = base.copy()
    tampered.loc[tampered.index[-1], "adj_close"] *= 10.0

    a, b = feats(base), feats(tampered)
    moved = [c for c in FEATURE_COLS
             if not a[c].iloc[:-1].equals(b[c].iloc[:-1])]
    assert moved == [], f"these features saw the future: {moved}"

    fa = _forward_returns(base.copy())["fwd_ret_1m"]
    fb = _forward_returns(tampered.copy())["fwd_ret_1m"]
    assert not fa.equals(fb), "fwd_ret_1m is SUPPOSED to move — it is the label"


# ====================================================================== sanity gate
def test_sanity_gate_passes_clean_data(capsys):
    sanity_gate(panel())
    assert "PASSED" in capsys.readouterr().out


@pytest.mark.parametrize("break_it,why", [
    (lambda d: d.assign(close=d["close"].mask(d.index == 5, -1.0)), "negative price"),
    (lambda d: d.assign(adj_close=d["adj_close"].mask(d.index == 5, np.nan)), "null adj_close"),
    (lambda d: pd.concat([d, d.iloc[[5]]], ignore_index=True), "duplicate (date, ticker)"),
])
def test_sanity_gate_aborts_on_bad_data(break_it, why):
    """The gate exits rather than warning: a panel with a negative price or a duplicate key
    produces features that look plausible and are wrong."""
    with pytest.raises(SystemExit) as ei:
        sanity_gate(break_it(panel()))
    assert ei.value.code != 0, f"gate must exit non-zero on {why}"


# ============================================================ A-1: FIXED in #31 Arm 1
def test_size_is_the_raw_traded_price_not_the_adjusted_one():
    """A-1 is CLOSED. `size` is `log(close)` — the price that actually traded that day.

    It was `log(adj_close)`, a retroactively re-written series: the value at date t changed
    whenever a split or dividend landed AFTER t, so the cross-sectional rank of `size` on a
    historical date depended on the future. Momentum was immune (the adjustment factor
    cancels in a price ratio); a bare level did not cancel.

    The fixture's `close` and `adj_close` deliberately differ, so this test discriminates.
    They were identical until #31 — which is exactly how the other tripwire in this pair
    stayed green through the very change it existed to catch.

    Still true and still documented: raw close is a price LEVEL, not a size. A $500 stock is
    not "bigger" than a $50 one, and a name that splits drops in this ranking overnight
    without changing. "Should `size` be market cap?" is a different experiment (#31 rails).
    """
    df = panel()
    out = feats(df)
    got = out.loc["AAA", "size"].iloc[100]
    assert got == pytest.approx(np.log(df["close"].iloc[100]), rel=1e-12)
    assert got != pytest.approx(np.log(df["adj_close"].iloc[100]), rel=1e-9), \
        "size must no longer read the retroactively adjusted series"


def test_size_no_longer_moves_when_history_is_retro_adjusted():
    """The fix, demonstrated rather than described — the inverse of the test it replaces.

    A 10-for-1 split re-adjusts the WHOLE adj_close history, which is what a price provider
    does. Every feature must now be unmoved: momentum because ratios cancel the factor, and
    `size` because it no longer reads that series at all.
    """
    base = panel()
    split = base.copy()
    split["adj_close"] = split["adj_close"] / 10.0        # retroactive re-adjustment

    a, b = feats(base), feats(split)
    t = 300
    assert a["size"].iloc[t] == pytest.approx(b["size"].iloc[t], rel=1e-12), \
        "A-1 has regressed: size moved when only the adjusted series was rewritten"
    for c in ("mom_3m", "mom_6m", "mom_12_1m", "reversal_1m"):
        assert a[c].iloc[t] == pytest.approx(b[c].iloc[t], rel=1e-12), \
            f"{c} is a ratio and must be immune to re-adjustment"


def test_a_real_split_in_the_raw_series_does_move_size():
    """The other side of the fix: `size` must still RESPOND to the raw price actually
    changing. A test that only asserts immobility would also pass if `size` were a constant.
    """
    base = panel()
    real = base.copy()
    real["close"] = real["close"] / 10.0                  # the traded price itself changed

    a, b = feats(base), feats(real)
    assert a["size"].iloc[300] != pytest.approx(b["size"].iloc[300], rel=1e-9)
