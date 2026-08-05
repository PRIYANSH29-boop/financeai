"""
Signal generation and model evaluation — RankAlpha phases 4-6.

Holds the no-ML momentum baseline that sets the bar (`baseline_momentum`), the frozen
LightGBM LambdaMART ranker that had to beat it (`lgbm_ranker`), the honest walk-forward
evaluation of that ranker (`evaluate`), and the SIC -> sector fallback table used by the
pie engine's sector caps (`sic_sectors`).

`lgbm_ranker` is the ONLY module permitted to fit the ranker. It is frozen (2024-05-15);
every later phase reads its scores and never refits. See ARCHITECTURE.md.
"""
