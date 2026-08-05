#!/usr/bin/env python3
"""
Fundamentals data-quality audit CLI — Phase 17 (the GO/NO-GO gate before the value factor).

Thin wrapper over the `audit/` package (mirrors `scripts/analyse.py` over `analytics/`).

    # verify the harness offline (no network, no key) — validates the checks themselves:
    python scripts/audit_fundamentals.py --self-test

    # run the real audit against SEC EDGAR XBRL (default; free, keyless, needs network):
    python scripts/audit_fundamentals.py --sample 50            # smoke
    python scripts/audit_fundamentals.py                        # full S&P 500 → GO/NO-GO


Writes `figures/audit/fundamentals_audit.md` with the seven-check report and a computed
GO/NO-GO verdict. It NEVER fabricates data: if the source is unreachable it errors out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from audit.fundamentals import (  # noqa: E402
    run_audit, self_test, write_report, REPORT_PATH, CACHE_DIR,
)


def _load_tickers(path: str, sample: int | None) -> list[str]:
    df = pd.read_csv(path)
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    tks = df[col].astype(str).str.strip().tolist()
    return tks[:sample] if sample else tks


def main() -> int:
    ap = argparse.ArgumentParser(description="RankAlpha fundamentals data-quality audit (#17)")
    ap.add_argument("--source", default="sec", choices=["sec"],
                    help="fundamentals source (default: sec = EDGAR XBRL, free/keyless)")
    ap.add_argument("--tickers", default="data/sp500_tickers.csv", help="ticker list CSV")
    ap.add_argument("--sample", type=int, default=None, help="audit only the first N tickers")
    ap.add_argument("--quarters", type=int, default=12, help="quarters of history per name")
    ap.add_argument("--cache-dir", default=None, help="source JSON cache dir")
    ap.add_argument("--panel", default="data/sp500_panel.parquet",
                    help="price panel, for point-in-time prices / market cap (sec source)")
    ap.add_argument("--out", default=str(REPORT_PATH), help="report markdown path")
    ap.add_argument("--self-test", action="store_true",
                    help="run the offline logic self-test only (no network)")
    args = ap.parse_args()

    if args.self_test:
        res = self_test()
        pprint(res)
        print("\nSELF-TEST:", "PASS ✅" if res["all_passed"] else "FAIL ❌")
        return 0 if res["all_passed"] else 1

    key = None      # EDGAR is keyless; the keyed vendor path was removed in #30 (F-4).

    panel = None
    if args.source == "sec":
        pp = Path(args.panel)
        if not pp.exists():
            print(f"ERROR: price panel {pp} not found — needed for point-in-time prices.",
                  file=sys.stderr)
            return 2
        panel = pd.read_parquet(pp)
        panel["date"] = pd.to_datetime(panel["date"])

    tickers = _load_tickers(args.tickers, args.sample)
    print(f"Auditing {len(tickers)} tickers via {args.source.upper()} ({args.quarters}q each)…")
    try:
        report = run_audit(tickers, api_key=key, quarters=args.quarters,
                           cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                           source=args.source, price_panel=panel)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: audit could not run against {args.source.upper()}: {e}", file=sys.stderr)
        return 3

    path = write_report(report, Path(args.out))
    g = report["go_no_go"]
    print(f"\nVERDICT: {g['verdict']}")
    for r in g["reasons"]:
        print("  -", r)
    print(f"\nReport: {path}")
    return 0 if g["verdict"] == "GO" else 4


if __name__ == "__main__":
    raise SystemExit(main())
