"""
Universe builder — Phase 16. S&P 500 → US mid + large cap (market cap > $2B).

Why this is not a config swap
-----------------------------
The frozen LightGBM ranker was trained and validated on the S&P 500 cross-section. Features
are cross-sectional ranks *within the universe*, so widening the universe changes what every
feature means. A model scored on a universe it was never fit on produces ranks that look
fine and mean nothing. Phase 16 therefore ships a NEW frozen model for the new universe and
keeps the S&P 500 one untouched for comparison — see `scripts/expand_universe.py`.

How the universe is built (all from sources reachable without a paid key)
-------------------------------------------------------------------------
1. **Registrants** — SEC `company_tickers_exchange.json`: every SEC filer with a ticker and
   an exchange. Filtered to NYSE / Nasdaq / NYSE American.
2. **Share counts** — SEC XBRL `frames` API for `dei:EntityCommonStockSharesOutstanding`,
   most recent frame per CIK. One request per quarterly frame covers every filer at once,
   instead of one request per name.
3. **Prices and liquidity** — yfinance, a short recent window: last close and median dollar
   volume. Names with no recent trading are dropped (delisted / halted).
4. **Filters** — market cap = shares × last close > `MIN_MARKET_CAP` ($2B); price > $1
   (drops sub-penny survivors); a minimum median dollar volume; optional cap to the top
   `max_names` by dollar volume so the universe stays liquid and the price download stays
   finite.

⚠️ SURVIVORSHIP: this is a CURRENT market-cap screen applied to the WHOLE history. Names
that fell below $2B, were acquired, or delisted are absent, which biases any backtest
upward — the same caveat the S&P 500 universe already carries, and the reason Phase 16
results are DIRECTIONAL. A point-in-time constituent source (e.g. Sharadar) is the fix.
Educational SIMULATION only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from audit.sec_provider import SECClient, SEC_CACHE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("universe")

FRAMES_URL = ("https://data.sec.gov/api/xbrl/frames/dei/"
              "EntityCommonStockSharesOutstanding/shares/{frame}.json")

# Recent instantaneous frames, newest first. A filer appears in the frame matching its
# fiscal quarter end, so several are needed to cover every fiscal calendar.
DEFAULT_FRAMES = ["CY2026Q1I", "CY2025Q4I", "CY2025Q3I", "CY2025Q2I", "CY2025Q1I"]

EXCHANGES = {"NYSE", "Nasdaq", "NYSE American", "NYSEAmerican", "NYSE MKT"}

MIN_MARKET_CAP = 2e9          # the #16 spec: mid + large cap
MIN_PRICE = 1.0
MIN_DOLLAR_VOLUME = 1e6       # $1M/day median — tradable enough to model
DEFAULT_MAX_NAMES = 1200      # liquidity cap; ~the S&P 900 / Russell 1000 field

UNIVERSE_PATH = Path("data/universe_midlarge.csv")


# ------------------------------------------------------------------ share counts
def shares_outstanding(client: SECClient | None = None, frames=DEFAULT_FRAMES,
                       cache: Path | None = None) -> pd.DataFrame:
    """{cik, entity, shares, as_of} — most recent tagged share count per registrant.

    Frames are ordered newest-first and the first hit wins, so a filer that stopped filing
    keeps its last known count rather than dropping out.
    """
    client = client or SECClient()
    cache = Path(cache or (SEC_CACHE / "shares_frames.json"))
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        payload = {}
        for fr in frames:
            try:
                d = client._get_json(FRAMES_URL.format(frame=fr))
            except Exception as e:                  # noqa: BLE001
                logger.warning("frame %s failed: %s", fr, e)
                continue
            for row in d.get("data", []):
                payload.setdefault(str(row["cik"]), {
                    "entity": row.get("entityName"), "shares": row.get("val"),
                    "as_of": row.get("end"), "frame": fr})
            logger.info("frame %s: %d cumulative registrants", fr, len(payload))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))
    rows = [{"cik": int(k), **v} for k, v in payload.items()]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ candidates
def candidates(client: SECClient | None = None) -> pd.DataFrame:
    """Exchange-listed registrants with a share count: ticker, cik, name, exchange, shares."""
    client = client or SECClient()
    cmap = client.cik_candidates()
    rows = []
    for tk, cands in cmap.items():
        for info in cands:
            if info.get("exchange") in EXCHANGES:
                rows.append({"ticker": tk, "cik": info["cik"], "name": info["name"],
                             "exchange": info["exchange"]})
                break
    listed = pd.DataFrame(rows)
    sh = shares_outstanding(client)
    out = listed.merge(sh[["cik", "shares", "as_of"]], on="cik", how="inner")
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce")
    out = out[out["shares"] > 0]
    logger.info("candidates: %d exchange-listed registrants with a share count", len(out))
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ market data screen
def market_screen(tickers, lookback_days: int = 90, batch: int = 200) -> pd.DataFrame:
    """{ticker, last_close, median_dollar_volume} from a short recent yfinance window."""
    import yfinance as yf
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    tickers = sorted(set(tickers))
    frames = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        try:
            d = yf.download(chunk, start=start, end=end + pd.Timedelta(days=1),
                            progress=False, auto_adjust=False, group_by="column",
                            threads=True)
        except Exception as e:                       # noqa: BLE001
            logger.warning("price screen chunk %d failed: %s", i // batch, e)
            continue
        if d.empty or "Close" not in d.columns.get_level_values(0):
            continue
        close, vol = d["Close"], d["Volume"]
        if isinstance(close, pd.Series):
            close, vol = close.to_frame(chunk[0]), vol.to_frame(chunk[0])
        dv = (close * vol).median(axis=0, skipna=True)
        frames.append(pd.DataFrame({"ticker": close.columns,
                                    "last_close": close.ffill().iloc[-1].to_numpy(),
                                    "median_dollar_volume": dv.to_numpy()}))
        logger.info("price screen: %d/%d", min(i + batch, len(tickers)), len(tickers))
    if not frames:
        raise RuntimeError("price screen returned nothing — refusing to build a universe")
    return pd.concat(frames, ignore_index=True).dropna(subset=["last_close"])


# ------------------------------------------------------------------ build
def build_universe(min_market_cap: float = MIN_MARKET_CAP, max_names: int = DEFAULT_MAX_NAMES,
                   min_price: float = MIN_PRICE, min_dollar_volume: float = MIN_DOLLAR_VOLUME,
                   client: SECClient | None = None, out: Path | None = UNIVERSE_PATH) -> dict:
    """Build the mid+large-cap universe and return {universe, stats} with filter counts."""
    cand = candidates(client)
    scr = market_screen(cand["ticker"].tolist())
    df = cand.merge(scr, on="ticker", how="inner")
    stats = {"registrants_listed_with_shares": len(cand), "with_market_data": len(df)}

    df["market_cap"] = df["shares"] * df["last_close"]
    steps = [
        ("price >= $%.2f" % min_price, df["last_close"] >= min_price),
        ("market cap > $%.1fB" % (min_market_cap / 1e9), df["market_cap"] > min_market_cap),
        ("median $vol >= $%.1fM" % (min_dollar_volume / 1e6),
         df["median_dollar_volume"] >= min_dollar_volume),
    ]
    mask = pd.Series(True, index=df.index)
    for label, m in steps:
        before = int(mask.sum())
        mask &= m.fillna(False)
        stats[f"after {label}"] = int(mask.sum())
        logger.info("filter %-24s %d -> %d", label, before, int(mask.sum()))
    df = df[mask].copy()

    df = df.sort_values("median_dollar_volume", ascending=False)
    if max_names and len(df) > max_names:
        stats["liquidity_cap_dropped"] = int(len(df) - max_names)
        df = df.head(max_names)
    df = df.sort_values("ticker").reset_index(drop=True)

    stats["universe_size"] = len(df)
    stats["median_market_cap"] = float(df["market_cap"].median())
    stats["min_market_cap"] = float(df["market_cap"].min())
    stats["total_market_cap"] = float(df["market_cap"].sum())
    stats["built_as_of"] = str(pd.Timestamp.today().date())

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        # The filter funnel is the audit trail for the universe — keep it next to the list
        # so a later report can show what was screened out rather than just what survived.
        Path(out).with_name("universe_stats.json").write_text(json.dumps(stats, indent=2))
        logger.info("wrote %d names -> %s", len(df), out)
    return {"universe": df, "stats": stats}


if __name__ == "__main__":
    res = build_universe()
    print(json.dumps(res["stats"], indent=2))
    print(res["universe"].head(20).to_string(index=False))
