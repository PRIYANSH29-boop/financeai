"""
The base data pipeline — RankAlpha phases 1-3.

Builds the three artifacts every other package consumes: the daily S&P 500 price panel
(`sp500_data`), the leakage-checked cross-sectional feature table (`sp500_features`), and
the within-day ranked forward-return labels (`sp500_labels`). Run all three with
`make pipeline`.

This is the most leakage-sensitive code in the repo: features are ranked within each date's
cross-section, and the one deliberate forward look lives in `sp500_labels` behind a 21-day
embargo. See ARCHITECTURE.md.
"""
