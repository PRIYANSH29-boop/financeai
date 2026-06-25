"""
Regenerate the committed hosted-demo bundle — RankAlpha v1.1.

A free cloud host (Streamlit Community Cloud) has none of the laptop's `data/` parquets
and must NOT refit the model or download 500 tickers on boot. This script writes the
SMALL, self-contained, TRACKED artifacts the hosted app loads directly:

    portfolio/bundle/score_book.joblib              # frozen fitted long book (~24 KB)
                                                    # — the exact input finalize_portfolio()
                                                    #   needs: holdings+factors, capped
                                                    #   weights, book vol, per-holding
                                                    #   explanations w/ sectors, OOS risk.
    portfolio/bundle/paper_track_portfolio.parquet  # realized monthly track (~8 KB)
    portfolio/bundle/paper_track_holdings.parquet   # per-month holdings  (~50 KB)

The paper-track figures (figures/paper_track_*.png) are already tracked; the build-tab
figures are regenerated at runtime. Total committed bundle ≈ 80 KB — the 131 MB of source
parquets stay gitignored and never reach the host.

Run locally AFTER refreshing the data parquets and the paper-track ledger
(`python -m portfolio.paper_trade`):

    python -m portfolio.make_bundle
"""

import shutil
from pathlib import Path

import joblib

from portfolio.engine import score_book, BUNDLE_DIR

PAPER_TRACK_FILES = ("paper_track_portfolio.parquet", "paper_track_holdings.parquet")


def main():
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    # Fit the frozen book from the real parquets (use_cache=False = always fresh).
    book = score_book(use_cache=False)
    out = BUNDLE_DIR / "score_book.joblib"
    joblib.dump(book, out)
    print(f"wrote {out}  (as_of {book['as_of']}, {len(book['capped'])} holdings)")

    # Copy the paper-track ledger so the hosted track tab renders the same realized record.
    for name in PAPER_TRACK_FILES:
        src = Path("data") / name
        if src.exists():
            shutil.copy2(src, BUNDLE_DIR / name)
            print(f"copied {src} -> {BUNDLE_DIR / name}")
        else:
            print(f"WARNING: {src} missing — run `python -m portfolio.paper_trade` first")

    print("bundle ready:", BUNDLE_DIR)


if __name__ == "__main__":
    main()
