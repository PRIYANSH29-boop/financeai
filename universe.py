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

   ⚠️ **The frames API only returns facts that carry no dimensions**, and multi-class filers
   tag their share count *per share class*. So Alphabet, Meta, Berkshire, Visa, Mastercard,
   Ford, Comcast — roughly 50 S&P 500 members — have **no** share count in any frame, and
   the pre-#24 build dropped every one of them silently on an inner join. That is why the
   shipped 1,200-name universe contains 451 of the 503 S&P names rather than ~500. Two
   things changed in #24: `candidates()` now returns the dropped names instead of discarding
   them (persisted to `universe_share_gap.csv`, counted in `universe_stats.json`), and
   `fallback_shares()` recovers a share count for the liquid ones so they survive the
   screen. Per-class counts are unavailable from XBRL entirely — companyconcept 404s for
   Alphabet and Meta, and Berkshire's only undimensioned share fact is a stale 2015
   Class-A-equivalent — so the fallback is a listed-class count from the price provider,
   tagged in `shares_source`. That is the right basis for a per-ticker universe anyway:
   this row is GOOGL-the-listing at GOOGL's price.
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

# "CBOE" appears as an exchange in SEC's map for 27 tickers, CBOE itself among them; leaving
# it out excluded a real S&P 500 constituent for no methodological reason (#24).
EXCHANGES = {"NYSE", "Nasdaq", "NYSE American", "NYSEAmerican", "NYSE MKT", "CBOE"}

MIN_MARKET_CAP = 2e9          # the #16 spec: mid + large cap
MIN_PRICE = 1.0
MIN_DOLLAR_VOLUME = 1e6       # $1M/day median — tradable enough to model
DEFAULT_MAX_NAMES = 1200      # liquidity cap; ~the S&P 900 / Russell 1000 field
RECOVERY_MIN_RATE = 0.5       # below this share of the liquid gap, the recovery is degraded
SHARES_CACHE = Path("data/cache/fallback_shares.json")

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
def candidates(client: SECClient | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exchange-listed registrants, split into those WITH and those WITHOUT a share count.

    Returns `(with_shares, without_shares)`. The second frame is the whole point: before #24
    this function inner-joined the share counts and returned only the survivors, so the ~50
    multi-class S&P 500 filers that XBRL frames cannot see left no trace at all. A universe
    builder that cannot say which names it dropped, and why, cannot be audited.
    """
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
    out = listed.merge(sh[["cik", "shares", "as_of"]], on="cik", how="left")
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce")

    have = out["shares"] > 0
    with_shares = out[have].copy()
    with_shares["shares_source"] = "sec_xbrl_frames"
    gap = out[~have].drop(columns=["shares", "as_of"]).copy()
    logger.info("candidates: %d exchange-listed registrants with an SEC share count, "
                "%d without (multi-class filers are invisible to the frames API)",
                len(with_shares), len(gap))
    return with_shares.reset_index(drop=True), gap.reset_index(drop=True)


# ------------------------------------------------------------------ non-equity exclusion (#26)
# The #24 share-gap recovery reopened this universe to things that are not companies. An ETF or
# commodity trust has no `EntityCommonStockSharesOutstanding` in the frames endpoint, so it fell
# into the gap; the price provider then answered `sharesOutstanding` for it happily, and it
# cleared the $2B and liquidity screens trivially. SPY entered as the single largest name in the
# 1,200, alongside gold, silver, bitcoin and leveraged-volatility trusts.
#
# Two signals, because neither alone is sufficient:
#   * SIC — the SEC's own structural classification. 6221 is where commodity and crypto trusts
#     live; 6722/6726 are the fund codes. This is the principled test, and it is precise: REITs
#     are 6798 and banks 6021/6022, so property trusts and "…Trust Corp" banks are untouched.
#   * NAME — SPY/QQQ/DIA/MDY are unit investment trusts that carry NO SIC at all, so structure
#     cannot see them. `\bETF\b` is word-bounded on purpose: a bare "ETF" substring matches
#     N-ETF-LIX.
NON_EQUITY_SIC = {"6221", "6722", "6726"}
NON_EQUITY_NAME_RE = r"\bETF\b|TRUST,\s*SERIES|TRUST II\b"


def is_non_equity(names: pd.Series, sics: pd.Series | None = None) -> pd.Series:
    """True where a registrant is a fund/trust vehicle rather than an operating company."""
    hit = names.astype(str).str.contains(NON_EQUITY_NAME_RE, case=False, regex=True, na=False)
    if sics is not None:
        hit = hit | sics.astype(str).isin(NON_EQUITY_SIC)
    return hit


def sic_codes(ciks, cache_path: Path = Path("data/cache/sector_sic_cache.json")) -> pd.Series:
    """{cik -> SIC} from the SEC submissions endpoint, sharing `map_sectors`' on-disk cache.

    Called only on the names that already survived the value filters, so this is a few hundred
    cached lookups, not a sweep of every registrant.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_map_sectors", Path(__file__).with_name("scripts") / "map_sectors.py")
    ms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms)
    cache = ms._load(cache_path)
    before = len(cache)
    out = {int(c): ms.fetch_sic(int(c), cache) for c in pd.unique(pd.Series(list(ciks)))}
    if len(cache) != before:
        ms._save(cache_path, cache)
    logger.info("SIC codes: %d resolved (%d newly fetched)", len(out), len(cache) - before)
    return pd.Series(out, dtype=object)


def fallback_shares(tickers, batch: int = 1, cache_path: Path | None = None,
                    pause: float = 0.3, max_retries: int = 3, circuit_break: int = 25,
                    cool_down: float = 5.0, refresh: bool = False) -> pd.DataFrame:
    """{ticker, shares} from the price provider, for names SEC frames cannot supply.

    Used ONLY for the share-count gap above. yfinance reports the *listed class's* shares
    outstanding (GOOGL → 5.87B Class A, not the 12.2B A+B+C aggregate), which is the correct
    basis for a per-ticker universe: this row is GOOGL-the-listing at GOOGL's price. Every
    recovered name is tagged `shares_source="price_provider"` so a later audit can separate
    SEC-sourced market caps from provider-sourced ones.

    `fast_info.shares` would be ~4x cheaper but reports the A+B+C AGGREGATE (GOOGL 12.2B),
    which paired with the Class-A price would inflate market cap by ~2x. The heavier
    `get_info` call is the one with the right basis, so the cost is paid deliberately and
    managed with pacing, backoff and an on-disk cache instead of being optimised away.

    Results are cached per ticker (including misses, as null) so an interrupted or
    rate-limited run resumes instead of restarting — a 20-minute loop that dies at 90% used
    to lose everything.
    """
    import time
    import yfinance as yf

    cache_path = Path(cache_path or SHARES_CACHE)
    cache: dict = json.loads(cache_path.read_text()) if cache_path.exists() and not refresh else {}
    wanted = sorted(set(tickers))
    todo = [t for t in wanted if t not in cache]
    if len(todo) < len(wanted):
        logger.info("fallback shares: %d of %d already cached, fetching %d",
                    len(wanted) - len(todo), len(wanted), len(todo))

    def _save():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, sort_keys=True))

    # The bulk price screen runs immediately before this loop and hammers the same host. On
    # 2026-08-03 an unpaced 853-name burst straight after it got 853/853 refusals; the same
    # calls paced at 0.3s answered 30/30. Breathe first, then pace, then back off.
    if todo and cool_down:
        logger.info("cooling down %.0fs before per-name lookups", cool_down)
        time.sleep(cool_down)

    consecutive = 0
    for i, tk in enumerate(todo, 1):
        so = None
        for attempt in range(max_retries):
            try:
                so = yf.Ticker(tk).get_info().get("sharesOutstanding")
                break
            except Exception as e:                       # noqa: BLE001
                logger.debug("fallback shares failed for %s (attempt %d): %s", tk, attempt + 1, e)
                time.sleep(pause * (2 ** attempt))       # exponential backoff
        cache[tk] = float(so) if pd.notna(pd.to_numeric(so, errors="coerce")) else None
        consecutive = 0 if cache[tk] else consecutive + 1
        # A provider that has stopped answering will not start again inside this run. Bail out
        # rather than burn 20 silent minutes to recover nothing.
        if consecutive >= circuit_break:
            logger.warning("fallback shares: %d consecutive failures — the provider is refusing. "
                           "Aborting the loop early with %d/%d fetched; progress is cached, so a "
                           "later re-run resumes instead of restarting.",
                           consecutive, i, len(todo))
            break
        if i % 50 == 0:
            _save()
            logger.info("fallback shares: %d/%d fetched (%d recovered so far)",
                        i, len(todo), sum(1 for v in cache.values() if v))
        time.sleep(pause)
    _save()

    rows = [{"ticker": t, "shares": cache[t]} for t in wanted if cache.get(t)]
    failures = sum(1 for t in wanted if t in cache and not cache[t])
    n = len(wanted)
    logger.info("fallback share counts: recovered %d of %d names from the price provider "
                "(%d lookups failed)", len(rows), n, failures)
    # A provider that rate-limits mid-run answers nothing and raises nothing, so a degraded
    # recovery used to look exactly like a healthy one: an INFO line, then a universe quietly
    # missing every multi-class mega-cap. This happened for real on 2026-08-03 — 0 of 852
    # recovered, S&P coverage silently back to 451/503. Say so loudly.
    if n and len(rows) < RECOVERY_MIN_RATE * n:
        logger.warning("share-count recovery DEGRADED: only %d of %d names (%.0f%%) came back — "
                       "the price provider is probably rate-limiting. The universe built from "
                       "this run will be missing multi-class names.", len(rows), n,
                       100 * len(rows) / n)
    return pd.DataFrame(rows, columns=["ticker", "shares"])


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
                   client: SECClient | None = None, out: Path | None = UNIVERSE_PATH,
                   recover_share_gap: bool = True, exclude_non_equity: bool = True,
                   use_sic: bool = True, fail_on_degraded_recovery: bool = True) -> dict:
    """Build the mid+large-cap universe and return {universe, stats} with filter counts."""
    cand, gap = candidates(client)

    # One price screen covers both cohorts — the names with an SEC share count and the gap —
    # so recovering the gap costs no extra price downloads, only the per-name share lookups.
    screen_for = cand["ticker"].tolist() + (gap["ticker"].tolist() if recover_share_gap else [])
    scr = market_screen(screen_for)

    # The gap is pre-filtered on price and liquidity, so only plausible names cost a per-name
    # request. Most of the ~2,300 registrants without an SEC share count are tiny or barely
    # traded and would fail the $2B floor regardless; the ones that matter are the multi-class
    # large caps, and they clear the liquidity pre-filter comfortably.
    recovered = pd.DataFrame(columns=["ticker", "shares"])
    if recover_share_gap and len(gap):
        gap_scr = scr[scr["ticker"].isin(set(gap["ticker"]))]
        liquid = gap_scr[(gap_scr["last_close"] >= min_price)
                         & (gap_scr["median_dollar_volume"] >= min_dollar_volume)]
        recovered = fallback_shares(liquid["ticker"].tolist())
        # Refuse to write a universe the recovery silently gutted. Overwriting a good universe
        # with a degraded one is worse than not building at all — the artifact looks normal and
        # every downstream phase inherits the hole.
        if fail_on_degraded_recovery and len(liquid) and \
                len(recovered) < RECOVERY_MIN_RATE * len(liquid):
            raise RuntimeError(
                f"share-count recovery degraded: {len(recovered)} of {len(liquid)} liquid gap "
                f"names recovered (<{RECOVERY_MIN_RATE:.0%}). The price provider is likely "
                f"rate-limiting. Refusing to build a universe missing its multi-class names — "
                f"retry later, or pass fail_on_degraded_recovery=False to build anyway.")
        if len(recovered):
            add = gap.merge(recovered, on="ticker", how="inner")
            add["as_of"] = pd.NaT
            add["shares_source"] = "price_provider"
            cand = pd.concat([cand, add], ignore_index=True)

    df = cand.merge(scr, on="ticker", how="inner")
    stats = {"registrants_listed_with_shares": int((cand["shares_source"]
                                                    == "sec_xbrl_frames").sum()),
             "share_count_gap": len(gap),
             "share_count_recovered_from_price_provider": len(recovered),
             "with_market_data": len(df)}

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

    # Drop funds/trusts BEFORE the liquidity cap, so the freed slots refill with real companies
    # rather than simply shrinking the universe.
    if exclude_non_equity and len(df):
        sics = sic_codes(df["cik"]) if use_sic else None
        non_eq = is_non_equity(df["name"], df["cik"].map(sics) if sics is not None else None)
        stats["excluded_non_equity"] = int(non_eq.sum())
        stats["excluded_non_equity_tickers"] = sorted(df.loc[non_eq, "ticker"].tolist())
        if int(non_eq.sum()):
            logger.info("excluded %d non-equity issuers (funds/commodity/crypto trusts): %s",
                        int(non_eq.sum()), stats["excluded_non_equity_tickers"])
        df = df[~non_eq]

    if max_names and len(df) > max_names:
        stats["liquidity_cap_dropped"] = int(len(df) - max_names)
        df = df.head(max_names)
    df = df.sort_values("ticker").reset_index(drop=True)

    stats["universe_size"] = len(df)
    stats["median_market_cap"] = float(df["market_cap"].median())
    stats["min_market_cap"] = float(df["market_cap"].min())
    stats["total_market_cap"] = float(df["market_cap"].sum())
    stats["built_as_of"] = str(pd.Timestamp.today().date())
    stats["shares_source_counts"] = {k: int(v) for k, v in
                                     df["shares_source"].value_counts().items()}

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        # The names that never got a share count, kept next to the universe: this is the
        # artifact whose absence let the multi-class drop hide for four phases.
        gap.to_csv(Path(out).with_name("universe_share_gap.csv"), index=False)
        # The filter funnel is the audit trail for the universe — keep it next to the list
        # so a later report can show what was screened out rather than just what survived.
        Path(out).with_name("universe_stats.json").write_text(json.dumps(stats, indent=2))
        logger.info("wrote %d names -> %s", len(df), out)
    return {"universe": df, "stats": stats}


if __name__ == "__main__":
    res = build_universe()
    print(json.dumps(res["stats"], indent=2))
    print(res["universe"].head(20).to_string(index=False))
