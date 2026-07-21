"""
RankAlpha fundamentals-audit package — Phase 17.

A reproducible data-quality gate that MUST pass before the value factor (#18) is built.
It verifies fundamentals are real, accurate, complete, and *point-in-time honest*
(every value lagged to its publication date, never its period-end).

Public surface
--------------
    from audit.fundamentals import run_audit, self_test, FMPClient

`scripts/audit_fundamentals.py` is the thin CLI wrapper (like `scripts/analyse.py` over
`analytics/`). Pure-logic helpers are unit-tested in `audit/tests/`; the network path
(FMP) is exercised only when an API key + connectivity are present — this module never
fabricates fundamentals.
"""

from .fundamentals import (
    run_audit,
    self_test,
    write_report,
    go_no_go,
    winsorize,
    zscore,
    value_ratios,
    detect_outliers,
    assert_point_in_time,
    coverage_map,
    FMPClient,
    CORE_INPUTS,
)

__all__ = [
    "run_audit", "self_test", "write_report", "go_no_go",
    "winsorize", "zscore", "value_ratios", "detect_outliers",
    "assert_point_in_time", "coverage_map", "FMPClient", "CORE_INPUTS",
]
