"""
Phase 20 — sector-mapping tests.

  1. SIC→sector table completeness: every 4-digit SIC resolves to a valid sector; spot-checks
     of the hand-checked overrides.
  2. Loader round-trip: `_load_sector_map` reads an inline `sector` column (S&P) and a
     companion `<stem>_sectors.csv` (wide universe), and returns empty when neither exists.
  3. Cap-binding kill-test (the acceptance gate): `build_portfolio` on the wide universe at
     β0.75 and β1.0 has zero '?' holdings, ≤5 names/sector, ≤30% weight/sector. Skipped if
     the committed wide-universe parquets / sectors file are absent.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from signals.sic_sectors import sic_to_sector, SECTORS, DEFAULT_SECTOR
from portfolio.beta_engine import _load_sector_map, build_portfolio, SECTOR_CAP, SECTOR_MAX_NAMES

_WIDE = {
    "panel": "data/midlarge_panel.parquet",
    "tickers": "data/universe_midlarge.csv",
    "features": "data/midlarge_features.parquet",
    "labeled": "data/midlarge_labeled.parquet",
    "sectors": "data/universe_midlarge_sectors.csv",
}


# ---------------------------------------------------------------- 1. table completeness
def test_every_sic_maps_to_a_valid_sector():
    # total over the entire 4-digit space — the completeness gate: no holes.
    for sic in range(0, 10000):
        assert sic_to_sector(sic) in SECTORS
    # invalid / junk inputs fall back to the documented default, never crash
    for junk in (None, "", "abc", -1, 99999, float("nan")):
        assert sic_to_sector(junk) in SECTORS
    assert DEFAULT_SECTOR in SECTORS


@pytest.mark.parametrize("sic,expected", [
    (3571, "Technology"), (7372, "Technology"), (3826, "Technology"),
    (2836, "Healthcare"), (8000, "Healthcare"), (3841, "Healthcare"),
    (1311, "Energy"), (2911, "Energy"),
    (6021, "Financial Services"), (6798, "Real Estate"), (6500, "Real Estate"),
    (4911, "Utilities"), (3711, "Consumer Cyclical"), (5411, "Consumer Defensive"),
    (4813, "Communication Services"), (3334, "Basic Materials"),
])
def test_sic_override_spotchecks(sic, expected):
    assert sic_to_sector(sic) == expected


# ------------------------------------------------------------------- 2. loader round-trip
def test_load_sector_map_inline_column(tmp_path):
    # S&P style: sector is a column in the tickers file itself.
    f = tmp_path / "sp.csv"
    pd.DataFrame({"ticker": ["A", "B"], "sector": ["Technology", "Energy"]}).to_csv(f, index=False)
    m = _load_sector_map(f)
    assert m.loc["A"] == "Technology" and m.loc["B"] == "Energy"


def test_load_sector_map_companion_file(tmp_path):
    # Wide style: tickers file has NO sector; a `<stem>_sectors.csv` companion supplies it.
    tickers = tmp_path / "universe.csv"
    pd.DataFrame({"ticker": ["A", "B"], "cik": [1, 2]}).to_csv(tickers, index=False)
    (tmp_path / "universe_sectors.csv").write_text(
        "ticker,sector,source\nA,Healthcare,A\nB,Industrials,B\n")
    m = _load_sector_map(tickers)
    assert m.loc["A"] == "Healthcare" and m.loc["B"] == "Industrials"


def test_load_sector_map_absent_returns_empty(tmp_path):
    tickers = tmp_path / "bare.csv"
    pd.DataFrame({"ticker": ["A"], "cik": [1]}).to_csv(tickers, index=False)
    assert _load_sector_map(tickers).empty


# ------------------------------------------------------------- 3. cap-binding kill-test
@pytest.mark.skipif(not all(Path(p).exists() for p in _WIDE.values()),
                    reason="wide-universe parquets / sectors file absent")
@pytest.mark.parametrize("target_beta", [0.75, 1.0])
def test_wide_universe_caps_active_and_binding(target_beta):
    p = build_portfolio(10_000, target_beta=target_beta, top_n=20,
                        panel_path=_WIDE["panel"], tickers_path=_WIDE["tickers"],
                        features_path=_WIDE["features"], labeled_path=_WIDE["labeled"],
                        make_figures=False)
    secs = [sl["sector"] for sl in p["slices"].values()]
    assert secs, "no holdings"
    # zero holdings in the '?' bucket — caps are no longer inert
    assert "?" not in secs
    # ≤ SECTOR_MAX_NAMES names per sector
    by_name = Counter(secs)
    assert max(by_name.values()) <= SECTOR_MAX_NAMES
    # ≤ SECTOR_CAP weight per sector (small tolerance for rounding)
    by_wt: dict = {}
    for sl in p["slices"].values():
        by_wt[sl["sector"]] = by_wt.get(sl["sector"], 0.0) + sl["weight"]
    assert max(by_wt.values()) <= SECTOR_CAP + 1e-6
