# Style & Season Lab — where does momentum actually live? (Phase 26)

⚠️ **EDUCATIONAL SIMULATION.** Both universes are CURRENT membership screens applied to all history, so every number here is survivorship-biased and **DIRECTIONAL**. Styles are computed from current data applied backwards. Nothing here is a forecast, and nothing was trained — the frozen model is read, never refit.

**Multiple-testing bar:** a cell is a FINDING only at **|t| ≥ 3**. Below that it is *suggestive* at best. **208 cells were tested in total** across the three grids; at the |t| ≥ 2 level alone you would expect ~9.5 false positives by luck. Read every highlight against that number.


## Coverage limit — read before the census

`data/sec_fundamentals.parquet` covers **501 tickers**, not the full wide universe, and it has **no `revenue` column**. So:

- **GROWTH** is *EPS* growth, not revenue growth — only the earnings half of the instruction's "revenue/earnings growth" exists offline.
- **GROWTH**, **VALUE** and the *no-earnings* leg of **SPECULATIVE** are evaluable only for names with a SEC ledger. Names without one are **not** defaulted into or out of those styles — that would invent a census. They are reported as not evaluable.
- Wide universe: **486 of 1169** names have usable fundamentals. S&P: **490 of 502**.


## Part A finding — index funds in the stock universe

The #24 share-gap recovery reopened the universe to **non-equities**: an ETF trust has no share count in the SEC `frames` endpoint, so it landed in the 2,303-name gap, and the price provider answered `sharesOutstanding` for it. **25 index funds (AGQ, BOIL, DIA, ETHA, FBTC, GBTC, GLD, GLDM, IAU, IAUM, IBIT, MDY, PHYS, PSLV, QQQ, SCO, SGOL, SLV, SPY, SVXY, UCO, UGL, UVIX, UVXY, ZSL) cleared the $2B and liquidity screens** — SPY entered as the single largest name in the 1,200. They are excluded from everything below by SEC entity name (`lab.style_lab.is_non_equity`). **`universe.py` itself is NOT fixed** — the committed universe CSV still contains them until a rebuild with a permanent exclusion rule, which needs its own instruction.


## Part B — the style census

Rules are committed in `lab/style_lab.py` as cross-sectional percentile thresholds, so no name is hand-picked. A name may hold at most 2 labels, resolved by a fixed priority order (most-specific first), never by eyeball.

### Wide universe

| style | n_names | pct_of_universe |
|---|---|---|
| growth | 143.0 | 12.2 |
| value | 161.0 | 13.8 |
| dividend | 339.0 | 29.0 |
| blue_chip | 129.0 | 11.0 |
| cyclical | 368.0 | 31.5 |
| defensive | 150.0 | 12.8 |
| speculative | 217.0 | 18.6 |

*172 of 1169 names carry no style at all (they clear no rule); 510 carry the maximum 2.*


### S&P 500

| style | n_names | pct_of_universe |
|---|---|---|
| growth | 151.0 | 30.1 |
| value | 163.0 | 32.5 |
| dividend | 139.0 | 27.7 |
| blue_chip | 37.0 | 7.4 |
| cyclical | 143.0 | 28.5 |
| defensive | 60.0 | 12.0 |
| speculative | 10.0 | 2.0 |

*65 of 502 names carry no style; 266 carry the maximum 2.*


### Style overlaps — wide universe

|  | growth | value | dividend | blue_chip | cyclical | defensive | speculative |
|---|---|---|---|---|---|---|---|
| growth | 143 | 43 | 20 | 32 | 15 | 9 | 0 |
| value | 43 | 161 | 63 | 21 | 18 | 4 | 0 |
| dividend | 20 | 63 | 339 | 44 | 68 | 35 | 20 |
| blue_chip | 32 | 21 | 44 | 129 | 14 | 5 | 0 |
| cyclical | 15 | 18 | 68 | 14 | 368 | 0 | 77 |
| defensive | 9 | 4 | 35 | 5 | 0 | 150 | 22 |
| speculative | 0 | 0 | 20 | 0 | 77 | 22 | 217 |


## Part C — the grids (every cell, with n)


### C1 · Simple 12-1 momentum — WIDE universe (the thesis test)

The mid-cap question lives here: if momentum is stronger in less-watched names, the speculative/small-end cohorts should beat the blue-chip cohort and the ALL control.

*80 testable cells · 0 at |t| ≥ 3 · 0 suggestive (2 ≤ |t| < 3) · ~3.6 expected false at |t| ≥ 2 by luck.*

| style | period | n names | n months | mean Rank IC | t | verdict |
|---|---|---|---|---|---|---|
| growth | full window | 143 | 72 | -0.0096 | -0.34 | noise |
| growth | 2020 | 136 | 5 | -0.1019 | -0.98 | noise |
| growth | 2021 | 137 | 12 | -0.0733 | -1.18 | noise |
| growth | 2022 | 140 | 12 | +0.0154 | +0.21 | noise |
| growth | 2023 | 140 | 12 | +0.0616 | +1.02 | noise |
| growth | 2024 | 141 | 12 | +0.0391 | +0.64 | noise |
| growth | 2025 | 143 | 12 | -0.0458 | -0.69 | noise |
| growth | 2026 | 143 | 7 | -0.0210 | -0.15 | noise |
| growth | earnings months | 143 | 24 | +0.0114 | +0.25 | noise |
| growth | non-earnings months | 143 | 48 | -0.0201 | -0.56 | noise |
| value | full window | 161 | 72 | -0.0013 | -0.05 | noise |
| value | 2020 | 157 | 5 | -0.1248 | -0.83 | noise |
| value | 2021 | 158 | 12 | -0.0235 | -0.32 | noise |
| value | 2022 | 159 | 12 | -0.0016 | -0.02 | noise |
| value | 2023 | 160 | 12 | +0.0442 | +0.93 | noise |
| value | 2024 | 160 | 12 | +0.0586 | +1.17 | noise |
| value | 2025 | 161 | 12 | -0.0547 | -1.20 | noise |
| value | 2026 | 161 | 7 | +0.0362 | +0.99 | noise |
| value | earnings months | 161 | 24 | +0.0108 | +0.23 | noise |
| value | non-earnings months | 161 | 48 | -0.0074 | -0.24 | noise |
| dividend | full window | 339 | 72 | -0.0120 | -0.50 | noise |
| dividend | 2020 | 325 | 5 | -0.1451 | -0.97 | noise |
| dividend | 2021 | 329 | 12 | -0.0577 | -1.05 | noise |
| dividend | 2022 | 331 | 12 | -0.0015 | -0.02 | noise |
| dividend | 2023 | 335 | 12 | +0.0338 | +0.59 | noise |
| dividend | 2024 | 337 | 12 | +0.0558 | +1.41 | noise |
| dividend | 2025 | 338 | 12 | -0.0301 | -0.59 | noise |
| dividend | 2026 | 339 | 7 | -0.0206 | -0.58 | noise |
| dividend | earnings months | 339 | 24 | +0.0043 | +0.09 | noise |
| dividend | non-earnings months | 339 | 48 | -0.0202 | -0.72 | noise |
| blue_chip | full window | 129 | 72 | -0.0250 | -1.00 | noise |
| blue_chip | 2020 | 129 | 5 | -0.1338 | -1.15 | noise |
| blue_chip | 2021 | 129 | 12 | -0.0663 | -1.17 | noise |
| blue_chip | 2022 | 129 | 12 | -0.0472 | -0.60 | noise |
| blue_chip | 2023 | 129 | 12 | +0.0424 | +0.70 | noise |
| blue_chip | 2024 | 129 | 12 | +0.0373 | +0.62 | noise |
| blue_chip | 2025 | 129 | 12 | -0.0639 | -1.20 | noise |
| blue_chip | 2026 | 129 | 7 | +0.0056 | +0.10 | noise |
| blue_chip | earnings months | 129 | 24 | +0.0075 | +0.17 | noise |
| blue_chip | non-earnings months | 129 | 48 | -0.0413 | -1.36 | noise |
| cyclical | full window | 358 | 72 | -0.0076 | -0.36 | noise |
| cyclical | 2020 | 316 | 5 | -0.0776 | -0.66 | noise |
| cyclical | 2021 | 328 | 12 | -0.0609 | -1.53 | noise |
| cyclical | 2022 | 342 | 12 | -0.0060 | -0.10 | noise |
| cyclical | 2023 | 343 | 12 | +0.0459 | +1.04 | noise |
| cyclical | 2024 | 350 | 12 | +0.0441 | +1.00 | noise |
| cyclical | 2025 | 355 | 12 | +0.0101 | +0.17 | noise |
| cyclical | 2026 | 358 | 7 | -0.0794 | -1.02 | noise |
| cyclical | earnings months | 358 | 24 | +0.0084 | +0.22 | noise |
| cyclical | non-earnings months | 358 | 48 | -0.0156 | -0.60 | noise |
| defensive | full window | 148 | 72 | -0.0151 | -0.74 | noise |
| defensive | 2020 | 138 | 5 | +0.0054 | +0.10 | noise |
| defensive | 2021 | 141 | 12 | -0.0360 | -0.66 | noise |
| defensive | 2022 | 142 | 12 | -0.0153 | -0.30 | noise |
| defensive | 2023 | 145 | 12 | -0.0147 | -0.30 | noise |
| defensive | 2024 | 147 | 12 | +0.0415 | +0.77 | noise |
| defensive | 2025 | 147 | 12 | -0.0451 | -0.85 | noise |
| defensive | 2026 | 148 | 7 | -0.0393 | -0.56 | noise |
| defensive | earnings months | 148 | 24 | +0.0149 | +0.49 | noise |
| defensive | non-earnings months | 148 | 48 | -0.0300 | -1.13 | noise |
| speculative | full window | 207 | 72 | -0.0046 | -0.24 | noise |
| speculative | 2020 | 143 | 5 | -0.0076 | -0.12 | noise |
| speculative | 2021 | 161 | 12 | -0.0073 | -0.21 | noise |
| speculative | 2022 | 190 | 12 | +0.0643 | +1.08 | noise |
| speculative | 2023 | 193 | 12 | +0.0280 | +0.66 | noise |
| speculative | 2024 | 197 | 12 | -0.0117 | -0.32 | noise |
| speculative | 2025 | 200 | 12 | -0.0306 | -0.65 | noise |
| speculative | 2026 | 207 | 7 | -0.1149 | -1.41 | noise |
| speculative | earnings months | 207 | 24 | +0.0161 | +0.57 | noise |
| speculative | non-earnings months | 207 | 48 | -0.0149 | -0.60 | noise |
| ALL (control) | full window | 1148 | 72 | -0.0062 | -0.30 | noise |
| ALL (control) | 2020 | 1009 | 5 | -0.0300 | -0.30 | noise |
| ALL (control) | 2021 | 1045 | 12 | -0.0556 | -1.42 | noise |
| ALL (control) | 2022 | 1095 | 12 | +0.0258 | +0.40 | noise |
| ALL (control) | 2023 | 1106 | 12 | +0.0296 | +0.76 | noise |
| ALL (control) | 2024 | 1121 | 12 | +0.0379 | +0.93 | noise |
| ALL (control) | 2025 | 1135 | 12 | -0.0219 | -0.44 | noise |
| ALL (control) | 2026 | 1148 | 7 | -0.0695 | -0.73 | noise |
| ALL (control) | earnings months | 1148 | 24 | +0.0078 | +0.25 | noise |
| ALL (control) | non-earnings months | 1148 | 48 | -0.0132 | -0.49 | noise |

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
| dividend | full window | 139 | 48 | +0.0045 | +0.13 | noise |
| dividend | 2022 | 138 | 7 | -0.0263 | -0.18 | noise |
| dividend | 2023 | 138 | 12 | +0.0382 | +0.63 | noise |
| dividend | 2024 | 139 | 12 | +0.0734 | +1.48 | noise |
| dividend | 2025 | 139 | 12 | -0.0626 | -0.97 | noise |
| dividend | 2026 | 139 | 5 | -0.0374 | -0.33 | noise |
| dividend | earnings months | 139 | 16 | +0.0359 | +0.64 | noise |
| dividend | non-earnings months | 139 | 32 | -0.0112 | -0.25 | noise |
| blue_chip | full window | 37 | 48 | -0.0255 | -0.74 | noise |
| blue_chip | 2022 | 37 | 7 | -0.1549 | -1.65 | noise |
| blue_chip | 2023 | 37 | 12 | +0.0971 | +1.56 | noise |
| blue_chip | 2024 | 37 | 12 | +0.0314 | +0.42 | noise |
| blue_chip | 2025 | 37 | 12 | -0.1245 | -1.87 | noise |
| blue_chip | 2026 | 37 | 5 | -0.0374 | -1.96 | noise |
| blue_chip | earnings months | 37 | 16 | +0.0620 | +1.08 | noise |
| blue_chip | non-earnings months | 37 | 32 | -0.0692 | -1.68 | noise |
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
| speculative | full window | 10 | 48 | -0.0076 | -0.14 | noise |
| speculative | 2022 | 10 | 7 | -0.0840 | -0.60 | noise |
| speculative | 2023 | 10 | 12 | +0.1465 | +1.33 | noise |
| speculative | 2024 | 10 | 12 | +0.0051 | +0.04 | noise |
| speculative | 2025 | 10 | 12 | -0.0051 | -0.06 | noise |
| speculative | 2026 | 10 | 5 | -0.3067 | -1.69 | noise |
| speculative | earnings months | 10 | 16 | +0.0295 | +0.31 | noise |
| speculative | non-earnings months | 10 | 32 | -0.0261 | -0.39 | noise |
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
| dividend | full window | 139 | 48 | +0.0162 | +0.48 | noise |
| dividend | 2022 | 138 | 7 | +0.0926 | +0.88 | noise |
| dividend | 2023 | 138 | 12 | -0.0263 | -0.31 | noise |
| dividend | 2024 | 139 | 12 | -0.0318 | -0.90 | noise |
| dividend | 2025 | 139 | 12 | +0.0525 | +0.70 | noise |
| dividend | 2026 | 139 | 5 | +0.0398 | +0.47 | noise |
| dividend | earnings months | 139 | 16 | +0.0329 | +0.62 | noise |
| dividend | non-earnings months | 139 | 32 | +0.0079 | +0.18 | noise |
| blue_chip | full window | 37 | 48 | +0.0168 | +0.62 | noise |
| blue_chip | 2022 | 37 | 7 | -0.0073 | -0.10 | noise |
| blue_chip | 2023 | 37 | 12 | -0.0689 | -1.97 | noise |
| blue_chip | 2024 | 37 | 12 | +0.0299 | +0.45 | noise |
| blue_chip | 2025 | 37 | 12 | +0.1070 | +1.89 | noise |
| blue_chip | 2026 | 37 | 5 | +0.0081 | +0.14 | noise |
| blue_chip | earnings months | 37 | 16 | -0.0216 | -0.44 | noise |
| blue_chip | non-earnings months | 37 | 32 | +0.0360 | +1.10 | noise |
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
| speculative | full window | 10 | 48 | -0.0465 | -0.90 | noise |
| speculative | 2022 | 10 | 7 | +0.0147 | +0.14 | noise |
| speculative | 2023 | 10 | 12 | -0.1778 | -1.67 | noise |
| speculative | 2024 | 10 | 12 | -0.0071 | -0.06 | noise |
| speculative | 2025 | 10 | 12 | +0.0242 | +0.22 | noise |
| speculative | 2026 | 10 | 5 | -0.0812 | -0.61 | noise |
| speculative | earnings months | 10 | 16 | -0.0909 | -0.94 | noise |
| speculative | non-earnings months | 10 | 32 | -0.0242 | -0.40 | noise |
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
