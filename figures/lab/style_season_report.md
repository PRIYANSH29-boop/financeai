# Style & Season Lab — where does momentum actually live? (Phase 26)

⚠️ **EDUCATIONAL SIMULATION.** Both universes are CURRENT membership screens applied to all history, so every number here is survivorship-biased and **DIRECTIONAL**. Styles are computed from current data applied backwards. Nothing here is a forecast, and nothing was trained — the frozen model is read, never refit.

**Multiple-testing bar:** a cell is a FINDING only at **|t| ≥ 3**. Below that it is *suggestive* at best. **208 cells were tested in total** across the three grids; at the |t| ≥ 2 level alone you would expect ~9.5 false positives by luck. Read every highlight against that number.


## Coverage limit — read before the census

`data/sec_fundamentals.parquet` covers **501 tickers**, not the full wide universe, and it has **no `revenue` column**. So:

- **GROWTH** is *EPS* growth, not revenue growth — only the earnings half of the instruction's "revenue/earnings growth" exists offline.
- **GROWTH**, **VALUE** and the *no-earnings* leg of **SPECULATIVE** are evaluable only for names with a SEC ledger. Names without one are **not** defaulted into or out of those styles — that would invent a census. They are reported as not evaluable.
- Wide universe: **445 of 1178** names have usable fundamentals. S&P: **490 of 502**.


## Part A finding — index funds in the stock universe

The #24 share-gap recovery reopened the universe to **non-equities**: an ETF trust has no share count in the SEC `frames` endpoint, so it landed in the 2,303-name gap, and the price provider answered `sharesOutstanding` for it. **21 index funds (AGQ, BITB, BOIL, BTC, ETHA, GBTC, GLD, GLDM, IAU, IAUM, IBIT, KOLD, PHYS, PSLV, SCO, SGOL, SIVR, SLV, UCO, UGL, ZSL) cleared the $2B and liquidity screens** — SPY entered as the single largest name in the 1,200. They are excluded from everything below by SEC entity name (`lab.style_lab.is_non_equity`). **`universe.py` itself is NOT fixed** — the committed universe CSV still contains them until a rebuild with a permanent exclusion rule, which needs its own instruction.


## Part B — the style census

Rules are committed in `lab/style_lab.py` as cross-sectional percentile thresholds, so no name is hand-picked. A name may hold at most 2 labels, resolved by a fixed priority order (most-specific first), never by eyeball.

### Wide universe

| style | n_names | pct_of_universe |
|---|---|---|
| growth | 127.0 | 10.8 |
| value | 148.0 | 12.6 |
| dividend | 349.0 | 29.6 |
| blue_chip | 131.0 | 11.1 |
| cyclical | 393.0 | 33.4 |
| defensive | 164.0 | 13.9 |
| speculative | 211.0 | 17.9 |

*172 of 1178 names carry no style at all (they clear no rule); 517 carry the maximum 2.*


### S&P 500

| style | n_names | pct_of_universe |
|---|---|---|
| growth | 151.0 | 30.1 |
| value | 163.0 | 32.5 |
| dividend | 138.0 | 27.5 |
| blue_chip | 35.0 | 7.0 |
| cyclical | 143.0 | 28.5 |
| defensive | 60.0 | 12.0 |
| speculative | 9.0 | 1.8 |

*67 of 502 names carry no style; 264 carry the maximum 2.*


### Style overlaps — wide universe

|  | growth | value | dividend | blue_chip | cyclical | defensive | speculative |
|---|---|---|---|---|---|---|---|
| growth | 127 | 42 | 16 | 28 | 17 | 7 | 0 |
| value | 42 | 148 | 50 | 24 | 18 | 4 | 0 |
| dividend | 16 | 50 | 349 | 41 | 70 | 38 | 37 |
| blue_chip | 28 | 24 | 41 | 131 | 20 | 5 | 0 |
| cyclical | 17 | 18 | 70 | 20 | 393 | 0 | 68 |
| defensive | 7 | 4 | 38 | 5 | 0 | 164 | 32 |
| speculative | 0 | 0 | 37 | 0 | 68 | 32 | 211 |


## Part C — the grids (every cell, with n)


### C1 · Simple 12-1 momentum — WIDE universe (the thesis test)

The mid-cap question lives here: if momentum is stronger in less-watched names, the speculative/small-end cohorts should beat the blue-chip cohort and the ALL control.

*80 testable cells · 0 at |t| ≥ 3 · 0 suggestive (2 ≤ |t| < 3) · ~3.6 expected false at |t| ≥ 2 by luck.*

| style | period | n names | n months | mean Rank IC | t | verdict |
|---|---|---|---|---|---|---|
| growth | full window | 127 | 72 | -0.0078 | -0.28 | noise |
| growth | 2020 | 123 | 6 | -0.1353 | -1.58 | noise |
| growth | 2021 | 123 | 12 | -0.0729 | -1.06 | noise |
| growth | 2022 | 124 | 12 | +0.0095 | +0.12 | noise |
| growth | 2023 | 124 | 12 | +0.0799 | +1.27 | noise |
| growth | 2024 | 125 | 12 | +0.0239 | +0.41 | noise |
| growth | 2025 | 127 | 12 | -0.0538 | -0.75 | noise |
| growth | 2026 | 127 | 6 | +0.0678 | +0.51 | noise |
| growth | earnings months | 127 | 24 | +0.0144 | +0.31 | noise |
| growth | non-earnings months | 127 | 48 | -0.0190 | -0.53 | noise |
| value | full window | 148 | 72 | -0.0046 | -0.18 | noise |
| value | 2020 | 144 | 6 | -0.1268 | -1.06 | noise |
| value | 2021 | 145 | 12 | -0.0269 | -0.36 | noise |
| value | 2022 | 146 | 12 | +0.0033 | +0.04 | noise |
| value | 2023 | 147 | 12 | +0.0412 | +0.81 | noise |
| value | 2024 | 147 | 12 | +0.0462 | +0.89 | noise |
| value | 2025 | 148 | 12 | -0.0660 | -1.36 | noise |
| value | 2026 | 148 | 6 | +0.0753 | +1.73 | noise |
| value | earnings months | 148 | 24 | +0.0198 | +0.40 | noise |
| value | non-earnings months | 148 | 48 | -0.0169 | -0.54 | noise |
| dividend | full window | 347 | 72 | -0.0122 | -0.49 | noise |
| dividend | 2020 | 336 | 6 | -0.1455 | -1.17 | noise |
| dividend | 2021 | 338 | 12 | -0.0415 | -0.71 | noise |
| dividend | 2022 | 342 | 12 | -0.0002 | -0.00 | noise |
| dividend | 2023 | 344 | 12 | +0.0281 | +0.44 | noise |
| dividend | 2024 | 346 | 12 | +0.0625 | +1.52 | noise |
| dividend | 2025 | 347 | 12 | -0.0296 | -0.52 | noise |
| dividend | 2026 | 347 | 6 | -0.0392 | -0.89 | noise |
| dividend | earnings months | 347 | 24 | +0.0059 | +0.12 | noise |
| dividend | non-earnings months | 347 | 48 | -0.0212 | -0.73 | noise |
| blue_chip | full window | 131 | 72 | -0.0211 | -0.85 | noise |
| blue_chip | 2020 | 131 | 6 | -0.1343 | -1.59 | noise |
| blue_chip | 2021 | 131 | 12 | -0.0783 | -1.41 | noise |
| blue_chip | 2022 | 131 | 12 | -0.0593 | -0.78 | noise |
| blue_chip | 2023 | 131 | 12 | +0.0765 | +1.24 | noise |
| blue_chip | 2024 | 131 | 12 | +0.0286 | +0.46 | noise |
| blue_chip | 2025 | 131 | 12 | -0.0608 | -1.30 | noise |
| blue_chip | 2026 | 131 | 6 | +0.0677 | +1.18 | noise |
| blue_chip | earnings months | 131 | 24 | +0.0165 | +0.38 | noise |
| blue_chip | non-earnings months | 131 | 48 | -0.0399 | -1.33 | noise |
| cyclical | full window | 388 | 72 | -0.0081 | -0.39 | noise |
| cyclical | 2020 | 355 | 6 | -0.0732 | -0.75 | noise |
| cyclical | 2021 | 363 | 12 | -0.0594 | -1.63 | noise |
| cyclical | 2022 | 372 | 12 | -0.0195 | -0.32 | noise |
| cyclical | 2023 | 374 | 12 | +0.0281 | +0.73 | noise |
| cyclical | 2024 | 381 | 12 | +0.0642 | +1.50 | noise |
| cyclical | 2025 | 387 | 12 | +0.0107 | +0.19 | noise |
| cyclical | 2026 | 388 | 6 | -0.0721 | -0.76 | noise |
| cyclical | earnings months | 388 | 24 | +0.0139 | +0.37 | noise |
| cyclical | non-earnings months | 388 | 48 | -0.0191 | -0.75 | noise |
| defensive | full window | 164 | 72 | -0.0077 | -0.40 | noise |
| defensive | 2020 | 153 | 6 | +0.0055 | +0.13 | noise |
| defensive | 2021 | 158 | 12 | -0.0477 | -0.84 | noise |
| defensive | 2022 | 160 | 12 | +0.0247 | +0.55 | noise |
| defensive | 2023 | 163 | 12 | -0.0239 | -0.53 | noise |
| defensive | 2024 | 163 | 12 | +0.0258 | +0.51 | noise |
| defensive | 2025 | 164 | 12 | -0.0400 | -0.85 | noise |
| defensive | 2026 | 164 | 6 | +0.0247 | +0.35 | noise |
| defensive | earnings months | 164 | 24 | +0.0190 | +0.76 | noise |
| defensive | non-earnings months | 164 | 48 | -0.0210 | -0.82 | noise |
| speculative | full window | 204 | 72 | +0.0021 | +0.11 | noise |
| speculative | 2020 | 160 | 6 | -0.0036 | -0.05 | noise |
| speculative | 2021 | 177 | 12 | -0.0257 | -0.72 | noise |
| speculative | 2022 | 192 | 12 | +0.0525 | +0.86 | noise |
| speculative | 2023 | 194 | 12 | +0.0567 | +1.16 | noise |
| speculative | 2024 | 197 | 12 | -0.0060 | -0.15 | noise |
| speculative | 2025 | 202 | 12 | -0.0249 | -0.57 | noise |
| speculative | 2026 | 204 | 6 | -0.0765 | -1.34 | noise |
| speculative | earnings months | 204 | 24 | +0.0217 | +0.74 | noise |
| speculative | non-earnings months | 204 | 48 | -0.0077 | -0.31 | noise |
| ALL (control) | full window | 1164 | 72 | -0.0057 | -0.30 | noise |
| ALL (control) | 2020 | 1064 | 6 | -0.0554 | -0.70 | noise |
| ALL (control) | 2021 | 1098 | 12 | -0.0566 | -1.48 | noise |
| ALL (control) | 2022 | 1126 | 12 | +0.0155 | +0.26 | noise |
| ALL (control) | 2023 | 1134 | 12 | +0.0287 | +0.74 | noise |
| ALL (control) | 2024 | 1146 | 12 | +0.0423 | +1.06 | noise |
| ALL (control) | 2025 | 1159 | 12 | -0.0260 | -0.55 | noise |
| ALL (control) | 2026 | 1164 | 6 | -0.0204 | -0.23 | noise |
| ALL (control) | earnings months | 1163 | 24 | +0.0178 | +0.62 | noise |
| ALL (control) | non-earnings months | 1164 | 48 | -0.0174 | -0.69 | noise |

### C2 · Simple 12-1 momentum — S&P 500 OOS window

The same signal on the big efficient names, over the model's OOS window only — the like-for-like comparison against C1.

*64 testable cells · 0 at |t| ≥ 3 · 1 suggestive (2 ≤ |t| < 3) · ~2.9 expected false at |t| ≥ 2 by luck.*

| style | period | n names | n months | mean Rank IC | t | verdict |
|---|---|---|---|---|---|---|
| growth | full window | 151 | 48 | +0.0187 | +0.55 | noise |
| growth | 2022 | 148 | 7 | -0.0429 | -0.34 | noise |
| growth | 2023 | 148 | 12 | +0.0551 | +0.88 | noise |
| growth | 2024 | 149 | 12 | +0.0394 | +0.64 | noise |
| growth | 2025 | 151 | 12 | -0.0561 | -0.82 | noise |
| growth | 2026 | 151 | 5 | +0.1476 | +1.60 | noise |
| growth | earnings months | 151 | 16 | +0.0528 | +1.04 | noise |
| growth | non-earnings months | 151 | 32 | +0.0017 | +0.04 | noise |
| value | full window | 163 | 48 | -0.0028 | -0.09 | noise |
| value | 2022 | 161 | 7 | -0.0812 | -0.57 | noise |
| value | 2023 | 162 | 12 | +0.0358 | +0.78 | noise |
| value | 2024 | 162 | 12 | +0.0285 | +0.51 | noise |
| value | 2025 | 163 | 12 | -0.0497 | -1.03 | noise |
| value | 2026 | 163 | 5 | +0.0518 | +1.52 | noise |
| value | earnings months | 163 | 16 | +0.0490 | +1.11 | noise |
| value | non-earnings months | 163 | 32 | -0.0287 | -0.75 | noise |
| dividend | full window | 138 | 48 | +0.0036 | +0.10 | noise |
| dividend | 2022 | 137 | 7 | -0.0256 | -0.17 | noise |
| dividend | 2023 | 137 | 12 | +0.0387 | +0.64 | noise |
| dividend | 2024 | 138 | 12 | +0.0745 | +1.50 | noise |
| dividend | 2025 | 138 | 12 | -0.0671 | -1.03 | noise |
| dividend | 2026 | 138 | 5 | -0.0401 | -0.36 | noise |
| dividend | earnings months | 138 | 16 | +0.0335 | +0.59 | noise |
| dividend | non-earnings months | 138 | 32 | -0.0113 | -0.26 | noise |
| blue_chip | full window | 35 | 48 | -0.0185 | -0.53 | noise |
| blue_chip | 2022 | 35 | 7 | -0.1669 | -1.73 | noise |
| blue_chip | 2023 | 35 | 12 | +0.1059 | +1.97 | noise |
| blue_chip | 2024 | 35 | 12 | +0.0295 | +0.37 | noise |
| blue_chip | 2025 | 35 | 12 | -0.1051 | -1.56 | noise |
| blue_chip | 2026 | 35 | 5 | -0.0162 | -0.44 | noise |
| blue_chip | earnings months | 35 | 16 | +0.0638 | +1.13 | noise |
| blue_chip | non-earnings months | 35 | 32 | -0.0596 | -1.41 | noise |
| cyclical | full window | 143 | 48 | +0.0227 | +0.77 | noise |
| cyclical | 2022 | 141 | 7 | -0.0556 | -0.49 | noise |
| cyclical | 2023 | 141 | 12 | +0.0344 | +0.73 | noise |
| cyclical | 2024 | 142 | 12 | +0.0934 | +2.20 | suggestive |
| cyclical | 2025 | 143 | 12 | -0.0436 | -0.68 | noise |
| cyclical | 2026 | 143 | 5 | +0.0940 | +1.02 | noise |
| cyclical | earnings months | 143 | 16 | +0.0566 | +1.05 | noise |
| cyclical | non-earnings months | 143 | 32 | +0.0058 | +0.16 | noise |
| defensive | full window | 60 | 48 | +0.0100 | +0.28 | noise |
| defensive | 2022 | 60 | 7 | -0.0568 | -0.55 | noise |
| defensive | 2023 | 60 | 12 | +0.0201 | +0.25 | noise |
| defensive | 2024 | 60 | 12 | +0.0900 | +1.39 | noise |
| defensive | 2025 | 60 | 12 | -0.0442 | -0.61 | noise |
| defensive | 2026 | 60 | 5 | +0.0173 | +0.18 | noise |
| defensive | earnings months | 60 | 16 | +0.0667 | +1.03 | noise |
| defensive | non-earnings months | 60 | 32 | -0.0184 | -0.43 | noise |
| speculative | full window | 9 | 48 | -0.0097 | -0.18 | noise |
| speculative | 2022 | 9 | 7 | -0.0690 | -0.50 | noise |
| speculative | 2023 | 9 | 12 | +0.1097 | +0.97 | noise |
| speculative | 2024 | 9 | 12 | -0.0069 | -0.06 | noise |
| speculative | 2025 | 9 | 12 | +0.0139 | +0.16 | noise |
| speculative | 2026 | 9 | 5 | -0.2767 | -1.52 | noise |
| speculative | earnings months | 9 | 16 | +0.0187 | +0.18 | noise |
| speculative | non-earnings months | 9 | 32 | -0.0240 | -0.37 | noise |
| ALL (control) | full window | 501 | 48 | +0.0250 | +0.84 | noise |
| ALL (control) | 2022 | 494 | 7 | -0.0442 | -0.35 | noise |
| ALL (control) | 2023 | 496 | 12 | +0.0631 | +1.29 | noise |
| ALL (control) | 2024 | 498 | 12 | +0.0772 | +1.47 | noise |
| ALL (control) | 2025 | 500 | 12 | -0.0442 | -0.78 | noise |
| ALL (control) | 2026 | 501 | 5 | +0.0711 | +1.17 | noise |
| ALL (control) | earnings months | 501 | 16 | +0.0663 | +1.54 | noise |
| ALL (control) | non-earnings months | 501 | 32 | +0.0043 | +0.11 | noise |

### C3 · FROZEN ML score — S&P 500 only (its valid universe)

The model is never scored outside the cross-section it was fit on.

*64 testable cells · 0 at |t| ≥ 3 · 1 suggestive (2 ≤ |t| < 3) · ~2.9 expected false at |t| ≥ 2 by luck.*

| style | period | n names | n months | mean Rank IC | t | verdict |
|---|---|---|---|---|---|---|
| growth | full window | 151 | 48 | +0.0589 | +1.73 | noise |
| growth | 2022 | 148 | 7 | +0.0529 | +0.48 | noise |
| growth | 2023 | 148 | 12 | +0.0176 | +0.26 | noise |
| growth | 2024 | 149 | 12 | +0.0513 | +0.94 | noise |
| growth | 2025 | 151 | 12 | +0.1164 | +1.40 | noise |
| growth | 2026 | 151 | 5 | +0.0461 | +0.60 | noise |
| growth | earnings months | 151 | 16 | +0.0661 | +1.18 | noise |
| growth | non-earnings months | 151 | 32 | +0.0552 | +1.28 | noise |
| value | full window | 163 | 48 | +0.0299 | +0.96 | noise |
| value | 2022 | 161 | 7 | +0.1018 | +0.97 | noise |
| value | 2023 | 162 | 12 | +0.0039 | +0.05 | noise |
| value | 2024 | 162 | 12 | +0.0107 | +0.44 | noise |
| value | 2025 | 163 | 12 | +0.0512 | +0.68 | noise |
| value | 2026 | 163 | 5 | -0.0137 | -0.22 | noise |
| value | earnings months | 163 | 16 | +0.0445 | +0.85 | noise |
| value | non-earnings months | 163 | 32 | +0.0226 | +0.58 | noise |
| dividend | full window | 138 | 48 | +0.0169 | +0.50 | noise |
| dividend | 2022 | 137 | 7 | +0.0918 | +0.88 | noise |
| dividend | 2023 | 137 | 12 | -0.0259 | -0.31 | noise |
| dividend | 2024 | 138 | 12 | -0.0296 | -0.83 | noise |
| dividend | 2025 | 138 | 12 | +0.0531 | +0.71 | noise |
| dividend | 2026 | 138 | 5 | +0.0397 | +0.46 | noise |
| dividend | earnings months | 138 | 16 | +0.0321 | +0.60 | noise |
| dividend | non-earnings months | 138 | 32 | +0.0093 | +0.22 | noise |
| blue_chip | full window | 35 | 48 | +0.0108 | +0.38 | noise |
| blue_chip | 2022 | 35 | 7 | -0.0147 | -0.17 | noise |
| blue_chip | 2023 | 35 | 12 | -0.0781 | -1.82 | noise |
| blue_chip | 2024 | 35 | 12 | +0.0333 | +0.48 | noise |
| blue_chip | 2025 | 35 | 12 | +0.0931 | +1.84 | noise |
| blue_chip | 2026 | 35 | 5 | +0.0082 | +0.13 | noise |
| blue_chip | earnings months | 35 | 16 | -0.0229 | -0.46 | noise |
| blue_chip | non-earnings months | 35 | 32 | +0.0277 | +0.81 | noise |
| cyclical | full window | 143 | 48 | +0.0615 | +2.32 | suggestive |
| cyclical | 2022 | 141 | 7 | +0.1225 | +1.50 | noise |
| cyclical | 2023 | 141 | 12 | +0.0189 | +0.38 | noise |
| cyclical | 2024 | 142 | 12 | +0.0781 | +1.69 | noise |
| cyclical | 2025 | 143 | 12 | +0.0662 | +1.02 | noise |
| cyclical | 2026 | 143 | 5 | +0.0274 | +0.42 | noise |
| cyclical | earnings months | 143 | 16 | +0.0667 | +1.76 | noise |
| cyclical | non-earnings months | 143 | 32 | +0.0589 | +1.66 | noise |
| defensive | full window | 60 | 48 | +0.0378 | +1.15 | noise |
| defensive | 2022 | 60 | 7 | +0.0368 | +0.39 | noise |
| defensive | 2023 | 60 | 12 | +0.0605 | +1.39 | noise |
| defensive | 2024 | 60 | 12 | -0.0171 | -0.37 | noise |
| defensive | 2025 | 60 | 12 | +0.0964 | +1.22 | noise |
| defensive | 2026 | 60 | 5 | -0.0241 | -0.13 | noise |
| defensive | earnings months | 60 | 16 | -0.0043 | -0.07 | noise |
| defensive | non-earnings months | 60 | 32 | +0.0589 | +1.48 | noise |
| speculative | full window | 9 | 48 | -0.0674 | -1.26 | noise |
| speculative | 2022 | 9 | 7 | -0.0524 | -0.46 | noise |
| speculative | 2023 | 9 | 12 | -0.1611 | -1.33 | noise |
| speculative | 2024 | 9 | 12 | -0.0250 | -0.22 | noise |
| speculative | 2025 | 9 | 12 | -0.0056 | -0.05 | noise |
| speculative | 2026 | 9 | 5 | -0.1133 | -0.64 | noise |
| speculative | earnings months | 9 | 16 | -0.1167 | -1.09 | noise |
| speculative | non-earnings months | 9 | 32 | -0.0427 | -0.70 | noise |
| ALL (control) | full window | 501 | 48 | +0.0532 | +1.71 | noise |
| ALL (control) | 2022 | 494 | 7 | +0.0827 | +0.79 | noise |
| ALL (control) | 2023 | 496 | 12 | +0.0196 | +0.30 | noise |
| ALL (control) | 2024 | 498 | 12 | +0.0425 | +1.12 | noise |
| ALL (control) | 2025 | 500 | 12 | +0.0793 | +1.04 | noise |
| ALL (control) | 2026 | 501 | 5 | +0.0555 | +0.59 | noise |
| ALL (control) | earnings months | 501 | 16 | +0.0608 | +1.19 | noise |
| ALL (control) | non-earnings months | 501 | 32 | +0.0494 | +1.24 | noise |


## Honesty rails applied

- Every cell above is printed with its n. No cell was dropped for being empty or weak.
- `ALL (control)` rows are the null: a style effect that matches the control is not a style effect.
- Monthly rebalance rows only (`month_end_slice`) — daily rows would count the same bet ~21 times and inflate every t-stat.
- Earnings season = calendar months [1, 4, 7, 10], committed in code, not fitted.
- No intraday, no bonds/commodities, no 15-day slicing — all out of scope by instruction.
- Blue-chip requires ≥ 60 months of history, so it cannot be earned by a recent listing.
