# AUDIT_FINDINGS — RankAlpha adversarial self-audit (#25)

*Run 2026-08-03 against `9f267a0`. Role: hostile external auditor, not the builder. These are
**SUSPICIONS for reviewer triage, not verdicts**. **Nothing was fixed** — this file is the only
change in the commit. Severity: **S1** silent-wrong-answer (user sees a wrong number, nothing
crashes) > **S2** loud failure > **S3** cosmetic/doc drift.*

**Counts: 6 × S1 · 6 × S2 · 4 × S3 = 16 suspicions.** Coverage and the parts I did NOT complete
are stated in the last section — read it before treating any sweep as exhaustive.

---

## A. Time-travel sweep (lens 1)

`grep -E '\.(merge|join|shift|resample|reindex|asof|concat|align)\('` over `signals/ lab/
portfolio/ scripts/ audit/` returns 84 hits, of which ~50 are `str.join` false positives. The
~34 genuine pandas temporal ops are below. **`utils/` is not in the instruction's directory list
but is where features and labels are actually built** — the highest-leakage-risk code in the
repo — so I swept it too.

### A-1 — S1 — `utils/sp500_features.py` — **CLOSED in #31 Arm 1**
*Fixed: `size` is now `log(close)`, the price that actually traded, never retro-adjusted. The
v2 rematch showed the contamination was immaterial to the ML's performance — see
`figures/lab/last_stand.md`. Original finding preserved below.*

**Lens:** period vs publication basis. **Suspicion:** `df["size"] = np.log(ac)` is `log(adj_close)`
— (a) that is price level, not size (a $500 stock is not "bigger" than a $50 one), and (b)
`adj_close` is *retroactively* re-adjusted for every later split and dividend, so the value at
date *t* changes whenever a corporate action happens **after** *t*. Momentum features are immune
(the adjustment factor cancels in a price ratio); a bare level does not cancel. The
cross-sectional rank of `size` on a historical date therefore depends on the future.
**Verify:** `python -c "import yfinance,numpy as np; h=yfinance.Ticker('NVDA').history(start='2023-01-01',end='2023-06-01',auto_adjust=False)['Adj Close']; print(np.log(h).head(3))"` — re-run after any split and compare; the level moves, the ratios do not.

### A-2 — S1 — `audit/sec_provider.py:539-543`
**Lens:** both sides on the same basis. **Suspicion:** if `import yfinance` fails, `fetch_splits`
logs a warning and returns `{}`, so `ledger()` skips the `shares`/`eps` split adjustment
(`:415-419`) entirely and as-filed per-share figures get paired with split-adjusted prices. That
is exactly the NVDA-looks-10×-cheaper bug the #17 audit gate caught, reintroduced silently by a
missing import — a warning line, no exception, and every downstream ratio is wrong.
**Verify:** `python -c "import audit.sec_provider as s; s.SECClient.fetch_splits.__doc__" ` then rerun `make value` with yfinance uninstalled and diff `figures/lab/value_factor.md`.

### A-3 — S1 — `portfolio/beta_engine.py:70-76`
**Lens:** period date vs data date. **Suspicion:** `_monthly_returns` does
`.resample("ME").last().pct_change()`. The panel ends mid-month (S&P 2026-06-16), so the final
bucket is labelled `2026-06-30` but built from **16 days of data, presented as a full month**.
Every displayed stat downstream — `ann_vol` (×√12), `beta`, `sharpe`, `max_drawdown`, the Explore
`last_return`, the movers cards, every basket scorecard — is computed on that mixed-length series.
#24 added `axis_last_month_partial: true` to the bundle, but **`grep -rn axis_last_month_partial
web/components web/app web/lib` returns nothing** — the flag is exported and never rendered, so
the disclosure does not reach the user.
**Verify:** `venv/bin/python -c "import pandas as pd,sys; sys.path.insert(0,'.'); from portfolio.beta_engine import _monthly_returns; p=pd.read_parquet('data/sp500_panel.parquet',columns=['date','ticker','adj_close']); print(_monthly_returns(p,['AAPL']).tail(2)); print('panel ends', p.date.max())"`

### A-4 — S1 — `portfolio/beta_engine.py:112-131`
**Lens:** silent constraint violation (carried into sweep B, listed there as B-1/B-2/B-3).

### Swept clean — the coverage evidence
| Site | Date basis | Verdict |
|---|---|---|
| `utils/sp500_features.py:105-108` grouped `shift(21/63/126/252)` | period, backward only, `groupby('ticker')` never crosses names | **CLEAN** |
| `utils/sp500_features.py:144` `groupby('date').rank(pct=True)` | that day's cross-section only, no global scaler | **CLEAN** |
| `utils/sp500_labels.py:51` `groupby('ticker').shift(-21)` | the one deliberate forward look; 21-day embargo documented at module level | **CLEAN by design** |
| `utils/sp500_labels.py:67` `feats.merge(fwd, on=['date','ticker'])` | both sides from the same panel, same basis | **CLEAN** |
| `lab/value_factor.py:103-114` `as_of()` | `publication_date <= t − 1d` — a real publication-date gate, not a period-end gate | **CLEAN** |
| `lab/value_factor.py:117-143` `ratios_on()` | uses `close` **not** `adj_close`, because share counts are split- but not dividend-adjusted; reasoning is written down at `:167-171` | **CLEAN — notably careful** |
| `audit/sec_provider.py:413-419` split factor | puts as-reported per-share figures on the price panel's basis, so ratios are split-neutral | **CLEAN** (but see A-2) |
| `scripts/regime_stress_test.py:121`, `scripts/strategy_lab.py:130` | `(date,ticker)` / `date` merges within one already-aligned frame | **CLEAN** |

---

## B. Edge-case matrix (lens 6)

Probed directly, not read. `TESTED` = a named test covers it; `PROBED` = I ran it in this audit.

### `portfolio/beta_engine.py`
| Input / case | Result | Status |
|---|---|---|
| `capital=0` | raises `ValueError: capital must be positive` | PROBED — correct, loud |
| `capital=1`, `capital=1e9` | fine, weights scale | PROBED — clean |
| `target_beta=0.0` | 100% cash, achieved 0.0 | PROBED — clean |
| `target_beta=1.89` (max) | achieved 1.89, cash 0.0 | PROBED — clean |
| **1-name pool** | `_apply_caps` returns `{A: 1.0}` — 100% in one name vs `NAME_CAP=0.08` | **B-1, UNTESTED** |
| **3 names, one sector** | returns `0.333` each — 100% in one sector vs `SECTOR_CAP=0.30` | **B-2, UNTESTED** |
| **all-zero weights** | returns `{A: nan, B: nan}` — NaN weights, no raise | **B-3, UNTESTED** |
| **negative book beta** | `_hit_target_beta` returns NaN weights, reports `achieved_beta=0.0`, `cash=0.0` | **B-4, UNTESTED** |
| **all sectors unknown** | `_select` returns 10 picks all in `'?'` vs `SECTOR_MAX_NAMES=5` | **B-5, UNTESTED** |
| cap-boundary equality (`<` vs `<=`) | `1e-9` tolerances at `:120,:127` | not probed — see coverage |

**B-1 / B-2 — S1 — `portfolio/beta_engine.py:112-131`.** `_apply_caps` alternates cap-and-redistribute
for 200 passes and then does `return w / w.sum()` **unconditionally**. When the pool is too small
to absorb the deficit, `head.sum() <= 0` breaks the loop (`:128-129`) and the function returns a
weight vector that **violates the caps it exists to enforce**, with no exception and no flag. The
caps are the product's central safety claim ("humility encoded as rules"); here they fail open.
The shipped bundle is protected only because `export_web_bundle.py`'s validator re-checks caps —
any direct `build_portfolio` caller gets the violation silently.
**Verify:** `venv/bin/python -c "import sys,pandas as pd; sys.path.insert(0,'.'); from portfolio.beta_engine import _apply_caps; print(_apply_caps(pd.Series({'A':1.,'B':1.,'C':1.}), pd.Series({'A':'T','B':'T','C':'T'})))"`

**B-3 / B-4 — S1 — `portfolio/beta_engine.py:116,168`.** All-zero weights → `w/w.sum()` = NaN.
Negative book beta → `max(0.0, target/book_beta)` with `book_beta<0` gives a negative `k`, guarded
to `0.0` only when `book_beta == 0`, so the function returns **NaN weights while reporting
`achieved_beta = 0.0`** — a confident-looking number attached to an unusable book.
**Verify:** `venv/bin/python -c "import sys,pandas as pd; sys.path.insert(0,'.'); from portfolio.beta_engine import _hit_target_beta as h; w=pd.Series({'A':.5,'B':.5}); print(h(w,pd.Series({'A':-1.,'B':-1.}),pd.Series({'A':'T','B':'T'}),0.5)[:3])"`

**B-5 — S2 — `portfolio/beta_engine.py:150-155`.** The "relax the sector cap if starved" fallback
appends names **ignoring `SECTOR_MAX_NAMES` entirely**. With an empty/unknown sector map every
name is `'?'`, the greedy loop stops at 5, and the fallback silently fills the rest. #20's
kill-test proved the caps bind *on the mapped universe*; this path is how they stop binding.
**Verify:** `venv/bin/python -c "import sys,pandas as pd; sys.path.insert(0,'.'); from portfolio.beta_engine import _select; h=pd.DataFrame({'ticker':list('ABCDEFGHIJKL'),'model_score':range(12,0,-1)}); print(_select({'holdings':h},10,pd.Series({t:1. for t in 'ABCDEFGHIJKL'}),pd.Series(dtype=object)))"`

### `web/lib/basket.js`
| Input / case | Result | Status |
|---|---|---|
| 0 picks | `n_months=0`, `beta=NaN`, **`total_return=0`** | **B-6, UNTESTED** |
| 1 pick | correct | PROBED — clean |
| duplicate pick ×2 | stats correct (mean of `[x,x] = x`) but **`nPicks=2`** | **B-8, UNTESTED** |
| unknown ticker only | identical to empty basket, no signal | **B-7, UNTESTED** |
| known + unknown | unknown silently dropped, `nPicks=1` | **B-7, UNTESTED** |
| 11 picks (UI cap 10) | not enforced in the library | not probed |
| ≤24-month history | gated upstream by #24's `scored`; library itself ungated | not probed |
| month where every return is negative | not probed — see coverage |

**B-6 — S2 — `web/lib/basket.js:24,89-103`.** `totalReturn([])` returns `0`, so an empty or
all-invalid basket scores **"+0.00%" rather than "no data"**. `n_months=0` sits right next to it,
so the UI has the information to refuse — it just isn't obliged to.
**Verify:** `node -e "import('./web/lib/basket.js').then(m=>console.log(m.totalReturn([])))"`

**B-7 — S2 — `web/lib/basket.js:76`.** `picks.filter(tk => Array.isArray(returns[tk]))` silently
discards tickers with no series. A user picking 5 names can get a 2-name scorecard with no
indication that 3 were dropped — and #24's own reconciliation says **49 basket-eligible names have
no Explore row**, so pick-list/series mismatches are a live condition, not hypothetical.

**B-8 — S3 — `web/lib/basket.js:76-85`.** Duplicates are not deduped. For a pure duplicate the
stats stay right, but `nPicks` over-counts; with **one duplicate plus any other name** the repeated
name gets double weight in `mean(vals)`. I confirmed the harmless case and am inferring the
weighted case — reviewer should confirm the UI cannot emit duplicates.

**B-9 — S2 — `web/lib/basket.js:79-84`.** Months where only *some* picks have data are kept and
equal-weighted **over the survivors only**, so a basket's composition silently changes month to
month and the scorecard blends different portfolios. Nothing tells the user which months were
partial.

---

## C. Claims audit (lens 5)

| Claim (README.md) | Source artifact | Verdict |
|---|---|---|
| After-cost Sharpe **1.14** vs baseline 0.82 (`:40`) | `figures/` OOS eval, #12 | **OK** |
| Mean Rank IC **0.050** (t=1.64) (`:41`) | same | **OK** — note t=1.64 is below the \|t\|≥3 bar #26 now imposes |
| Book Sharpe **1.25**, ann ret +19.0%, vol 14.8%, MDD −15.0% (`:53-56`) | Phase-13 scorecard | **OK** (1.28 ddof=0 × 0.978 = 1.25) |
| #16 Sharpe 1.14→**1.81**, Rank IC 0.0505→**0.0276** (`:230`) | `figures/lab/universe_expansion.md` | **OK** |
| #17 **19,513 records, 100% point-in-time, ≤2.5%** discrepancy (`:231`) | `figures/audit/fundamentals_audit.md` | **OK** |
| #18 value uncorrelated **−0.15…−0.20**, Sharpe falls (`:232`) | `figures/lab/value_factor.md` | **OK** |
| #20 sector caps binding **Health 7→5, Tech 6→5** (`:234`) | `figures/lab/sector_mapping.md` | **OK** |
| #21 β0.75 pie **0.26 calm → 0.78 stress** (`:235`) | `figures/lab/regime_report.md` — calm 0.263, stressed 0.781 | **OK** |
| All 6 linked `figures/**` artifacts exist | `git ls-files` | **OK** |
| Live-site numbers | all three bundle files md5-match local (#24 RESULT) | **OK — architecturally backed** |
| **Streamlit demo link** (`:3`) | `curl` returns `http=000` | **UNVERIFIABLE — see C-1** |

**C-1 — S2 — `README.md:3`.** The headline "Try the live demo (Streamlit)" link returns `http=000`
from this machine. **This is NOT evidence the demo is down** — egress here is host-allowlisted and
`streamlit.app` has never been on the allowlist. It is unverifiable from the audit environment,
which means **no one has verified it since the allowlist was mapped**. A dead headline link on an
honesty-branded README is a claim failure regardless of cause; someone on an unrestricted network
must load it.
**Verify:** open the URL from a normal network, or `curl -sI <url>` from the hotspot.

**C-2 — S3 — Notion "Complete Documentation" §2.** Cites `signals/feature_engineering.py` as the
feature builder. **No such file exists**; features are built in `utils/sp500_features.py`. The same
`utils/` directory is missing from #25's own sweep-A list, so the repo's most leakage-sensitive
module is absent from both the documentation map and the audit scope.
**Verify:** `ls signals/` and `grep -rn feature_engineering --include=*.py .`

---

## D. Hygiene sweep (lens 7)

**Secrets — swept, nothing found in scope.** `.env` is **not** tracked (`git ls-files | grep .env`
→ empty). A `git grep -IE` for `sk-…`, `AKIA…`, `ghp_…`, `fut_…` and `api_key = "…"` patterns over
the whole tracked tree returns **zero hits**. *Scope limit:* I scanned the working tree and the
47-commit log, **not** the full blob history — see coverage.

**D-1 — S2 — `requirements.txt`.** **0 of 28 dependencies are pinned** — every line is `>=`
(`pandas>=2.2`, `lightgbm>=4.3`, `numpy>=1.26`, …). The project's central claim is a **frozen**
model whose scores must not move, yet a fresh `pip install` can pull a different LightGBM and
change the ranking. `web/package.json` pins exactly (`next 15.5.4`, `react 19.1.1`) — so the
discipline exists on the JS side and not the Python side, where it matters more.
**Verify:** `grep -c '==' requirements.txt` → `0`

**D-2 — S2 — `Makefile` / reproducibility.** `data/*.parquet` is gitignored, and **no make target
regenerates the base artifacts**: `data/sp500_panel.parquet`, `sp500_features.parquet`,
`sp500_labeled.parquet` and the frozen model have no target. `utils/sp500_data.py`,
`sp500_features.py`, `sp500_labels.py` are runnable only via `if __name__ == "__main__"`. Every
documented target (`make analyse|lab|value|universe|sectors|regimes-backtest|web-bundle`) *consumes*
these files. **A fresh clone cannot rebuild the pipeline from documented targets alone** — the
first three phases are undocumented tribal knowledge.
**Verify:** `grep -E "^[a-z-]+:" Makefile` and confirm no panel/features/labels/train target.

**D-3 — S3 — `utils/sp500_features.py:11` vs `:38`.** Docstring says "point-in-time eligibility:
>=252 trading days"; the constant is `MIN_HISTORY = 253`. The code comment at `:35-37` explains
253 correctly, so the module contradicts itself.

**D-4 — S3 — `lab/value_factor.py:104`.** Docstring says "public **strictly before** `when`"; the
code is `publication_date <= when − 1 day`. Same-day filings are excluded by the 1-day lag rather
than by the comparison, so the behaviour is right and the description is not. If
`PUBLICATION_LAG_DAYS` is ever set to 0, the docstring becomes an active lie.

---

## Coverage — what I did NOT complete

Stated because a silent partial sweep reads as full coverage.

- **Sweep A.** I hand-verified the ~34 genuine pandas temporal ops and read 6 modules end to end
  (`sp500_features`, `sp500_labels`, `value_factor`, `sec_provider.ledger`, `beta_engine`,
  `export_web_bundle` validators). I did **not** line-by-line verify the `reindex` sites in
  `lab/regime_backtest.py` (7), `portfolio/paper_trade.py` (5), `signals/baseline_momentum.py` (4)
  and `lab/strategy_lab.py` (5). They are same-index alignments inside one already-built frame and
  I judged them low-risk **by inspection of the surrounding call, not by proof**.
- **Sweep B.** Cap-boundary equality (`<` vs `<=` at the `1e-9` tolerances), the ≤24-month history
  cell, the 11-picks-vs-UI-cap-10 cell, and "a month where every return is negative" were **not
  probed**. The all-negative-month cell matters most: `maxDrawdown` and `sortino` are the likely
  failure sites.
- **Sweep C.** I traced every **numeric** claim in README and the bundle-backed site numbers. I did
  **not** walk all ~260 lines of README prose, nor every string rendered by the six web components.
  Sweep C is therefore complete for numbers and partial for prose.
- **Sweep D.** The secrets scan covered the tracked working tree and the commit log, **not** the
  full blob history. A secret in a deleted-but-still-reachable blob would not have been caught.
  Run `git rev-list --objects --all | git cat-file --batch-check | grep blob` + grep to close it.

## The one I would fix first

**A-3 (partial final month).** B-1/B-2 are worse in principle, but they need a degenerate pool that
the shipped 503-name universe never produces — they are latent. A-3 is **live right now on every
number the site displays**: the most recent month is 16 days of data annualised as if it were 30,
and the flag that would disclose it (`axis_last_month_partial`) is exported but rendered nowhere.
It is the exact failure class #24 was opened for — a plausible-looking date-derived number that
nothing crashes on — one layer deeper than the as-of stamp #24 fixed.
