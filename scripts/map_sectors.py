"""
Phase 20 — attach a sector to every name in the wide universe so the pie engine's sector
caps become active. Educational plumbing only: no model, no refit, no scoring.

Two sources (reviewer decision A-primary, B-fallback):
  A. yfinance `.info` sector — the 11-bucket Yahoo taxonomy, labels most names.
  B. SEC SIC via the CIK already in the universe file → `signals.sic_sectors.sic_to_sector`.
     Every name has a CIK, so B is a 100%-coverage backbone; A upgrades the label where it
     responds. Final label = A if valid else B; `source` records which won.

Both sources are cached to `data/cache/` so re-runs are fast and resumable (yfinance
rate-limits under load; SEC is ~10 req/s). Writes `data/universe_midlarge_sectors.csv`
(ticker, sector, source) and prints coverage stats (A / B / missing).

Caveat (documented, same class as survivorship): labels are TODAY's classification applied
backwards. Sector reclassification is rare but nonzero; the caps only need clustering
prevention, not point-in-time GICS.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from signals.sic_sectors import sic_to_sector, SECTORS

UNIVERSE = Path("data/universe_midlarge.csv")
OUT = Path("data/universe_midlarge_sectors.csv")
CACHE_DIR = Path("data/cache")
YF_CACHE = CACHE_DIR / "sector_yf_cache.json"
SIC_CACHE = CACHE_DIR / "sector_sic_cache.json"
UA = {"User-Agent": "RankAlpha research (educational) priyansh2005p@gmail.com"}
_VALID = set(SECTORS)


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=0, sort_keys=True))


# ------------------------------------------------------------------- source B: SEC SIC
def fetch_sic(cik: int, cache: dict, pause: float = 0.12) -> str | None:
    key = str(int(cik))
    if key in cache:
        return cache[key]
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    try:
        req = urllib.request.Request(url, headers=UA)
        data = json.load(urllib.request.urlopen(req, timeout=15))
        sic = str(data.get("sic") or "").strip() or None
    except Exception:
        sic = None
    cache[key] = sic
    time.sleep(pause)
    return sic


# ------------------------------------------------------------------- source A: yfinance
def fetch_yf_sector(ticker: str, cache: dict, pause: float = 0.25) -> str | None:
    if ticker in cache:
        return cache[ticker]
    sector = None
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        s = (info or {}).get("sector")
        if s and s in _VALID:
            sector = s
    except Exception:
        sector = None
    cache[ticker] = sector
    time.sleep(pause)
    return sector


# ------------------------------------------------------------------------------- build
def build(limit: int | None = None, skip_yf: bool = False,
          save_every: int = 50) -> pd.DataFrame:
    uni = pd.read_csv(UNIVERSE)
    if limit:
        uni = uni.head(limit)
    yf_cache, sic_cache = _load(YF_CACHE), _load(SIC_CACHE)

    rows = []
    for i, r in enumerate(uni.itertuples(index=False), 1):
        tk, cik = r.ticker, r.cik
        # B backbone first (reliable, gives every name a label)
        sic = fetch_sic(cik, sic_cache)
        b_sector = sic_to_sector(sic) if sic else None
        # A overlay (preferred label where yfinance responds)
        a_sector = None if skip_yf else fetch_yf_sector(tk, yf_cache)

        if a_sector:
            sector, source = a_sector, "A"
        elif b_sector:
            sector, source = b_sector, "B"
        else:
            sector, source = "?", "missing"
        rows.append({"ticker": tk, "sector": sector, "source": source, "sic": sic})

        if i % save_every == 0:
            _save(YF_CACHE, yf_cache)
            _save(SIC_CACHE, sic_cache)
            print(f"  … {i}/{len(uni)} processed")

    _save(YF_CACHE, yf_cache)
    _save(SIC_CACHE, sic_cache)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only first N names (testing)")
    ap.add_argument("--skip-yf", action="store_true",
                    help="SEC-only (source B backbone), skip the slow yfinance overlay")
    args = ap.parse_args()

    df = build(limit=args.limit, skip_yf=args.skip_yf)
    df[["ticker", "sector", "source"]].to_csv(OUT, index=False)

    n = len(df)
    a = int((df["source"] == "A").sum())
    b = int((df["source"] == "B").sum())
    miss = df[df["source"] == "missing"]
    print("\n===== sector-mapping coverage =====")
    print(f"total names       : {n}")
    print(f"source A (yfinance): {a}  ({a / n * 100:.1f}%)")
    print(f"source B (SEC SIC) : {b}  ({b / n * 100:.1f}%)")
    print(f"labelled          : {n - len(miss)}  ({(n - len(miss)) / n * 100:.1f}%)")
    print(f"missing           : {len(miss)}")
    if len(miss):
        print("  missing tickers:", ", ".join(miss["ticker"].tolist()))
    print(f"\nsector distribution:\n{df['sector'].value_counts().to_string()}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
