"""
Side-by-side comparison helper — Phase 12.

`compare([...])` runs `analyse` on several instruments/strategies and returns a tidy
metric table (one column per instrument), ready to print or drop into a report.
"""

from __future__ import annotations

import pandas as pd

from .metrics import analyse

# Display order + human labels for the metric rows.
_ROWS = [
    ("total_return", "Total return", "pct"),
    ("cagr", "CAGR", "pct"),
    ("volatility", "Volatility (ann)", "pct"),
    ("sharpe", "Sharpe", "num"),
    ("sortino", "Sortino", "num"),
    ("max_drawdown", "Max drawdown", "pct"),
    ("max_drawdown_duration", "Drawdown duration (periods)", "int"),
    ("hit_rate", "Hit rate", "pct"),
    ("beta", "Beta", "num"),
    ("alpha", "Alpha (ann)", "pct"),
    ("n_periods", "N periods", "int"),
]


def compare(items, benchmark=None, periods_per_year: int = 12, rf: float = 0.0,
            pretty: bool = False) -> pd.DataFrame:
    """Metric table for several return series.

    items : dict {label: returns} OR a list of pandas Series (their `.name` is the label,
            falling back to positional labels) OR a list of (label, returns) tuples.
    benchmark : optional shared benchmark returns → enables beta/alpha for every column.
    pretty : if True, format cells as human strings (12.3%, 1.45); else raw floats.

    Returns a DataFrame indexed by metric label, one column per instrument.
    """
    named = _normalise(items)
    cols = {}
    for label, returns in named:
        cols[label] = analyse(returns, benchmark, periods_per_year, rf)

    table = pd.DataFrame(
        {label: {disp: metrics[key] for key, disp, _ in _ROWS}
         for label, metrics in cols.items()}
    )
    table = table.reindex([disp for _, disp, _ in _ROWS])
    return _format(table) if pretty else table


def _normalise(items) -> list[tuple[str, object]]:
    if isinstance(items, pd.DataFrame):
        return [(str(c), items[c]) for c in items.columns]
    if isinstance(items, dict):
        return list(items.items())
    out = []
    for i, it in enumerate(items):
        if isinstance(it, tuple) and len(it) == 2:
            out.append((str(it[0]), it[1]))
        elif hasattr(it, "name") and it.name is not None:
            out.append((str(it.name), it))
        else:
            out.append((f"series_{i}", it))
    return out


def _format(table: pd.DataFrame) -> pd.DataFrame:
    fmt = {disp: kind for _, disp, kind in _ROWS}

    def cell(disp, v):
        if v is None or (isinstance(v, float) and (v != v)):
            return "—"
        kind = fmt[disp]
        if kind == "pct":
            return f"{v:.2%}"
        if kind == "int":
            return f"{int(v)}"
        return f"{v:.2f}"

    return pd.DataFrame(
        {col: {disp: cell(disp, table.at[disp, col]) for disp in table.index}
         for col in table.columns}
    ).reindex(table.index)
