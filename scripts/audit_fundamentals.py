#!/usr/bin/env python3
"""
Fundamentals data-quality audit CLI — Phase 17 (the GO/NO-GO gate before the value factor).

Thin wrapper over the `audit/` package (mirrors `scripts/analyse.py` over `analytics/`).

    # verify the harness offline (no network, no key) — validates the checks themselves:
    python scripts/audit_fundamentals.py --self-test

    # run the real audit (needs FMP free-tier key + connectivity):
    python scripts/audit_fundamentals.py --fmp-key $FMP_KEY --tickers data/sp500_tickers.csv
    python scripts/audit_fundamentals.py --sample 50           # key from .env / FMP_API_KEY

Writes `figures/audit/fundamentals_audit.md` with the seven-check report and a computed
GO/NO-GO verdict. It NEVER fabricates data: with no key/network it self-tests and exits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from audit.fundamentals import (  # noqa: E402
    run_audit, self_test, write_report, _load_key, REPORT_PATH, CACHE_DIR,
)


def _load_tickers(path: str, sample: int | None) -> list[str]:
    df = pd.read_csv(path)
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    tks = df[col].astype(str).str.strip().tolist()
    return tks[:sample] if sample else tks


def main() -> int:
    ap = argparse.ArgumentParser(description="RankAlpha fundamentals data-quality audit (#17)")
    ap.add_argument("--fmp-key", default=None, help="FMP API key (else FMP_API_KEY / .env)")
    ap.add_argument("--tickers", default="data/sp500_tickers.csv", help="ticker list CSV")
    ap.add_argument("--sample", type=int, default=None, help="audit only the first N tickers")
    ap.add_argument("--quarters", type=int, default=12, help="quarters of history per name")
    ap.add_argument("--cache-dir", default=str(CACHE_DIR), help="FMP JSON cache dir")
    ap.add_argument("--out", default=str(REPORT_PATH), help="report markdown path")
    ap.add_argument("--self-test", action="store_true",
                    help="run the offline logic self-test only (no network)")
    args = ap.parse_args()

    if args.self_test:
        res = self_test()
        pprint(res)
        print("\nSELF-TEST:", "PASS ✅" if res["all_passed"] else "FAIL ❌")
        return 0 if res["all_passed"] else 1

    key = _load_key(args.fmp_key)
    if not key:
        print("ERROR: no FMP API key (pass --fmp-key, or set FMP_API_KEY / .env).\n"
              "       Run `--self-test` to verify the harness offline.", file=sys.stderr)
        return 2

    tickers = _load_tickers(args.tickers, args.sample)
    print(f"Auditing {len(tickers)} tickers via FMP ({args.quarters}q each)…")
    try:
        report = run_audit(tickers, api_key=key, quarters=args.quarters,
                           cache_dir=Path(args.cache_dir))
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: audit could not run against FMP: {e}", file=sys.stderr)
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
