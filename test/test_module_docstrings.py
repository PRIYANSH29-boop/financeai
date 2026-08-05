"""#29 — every Python module carries a top-of-file docstring.

The #29 deliverable was "every module missing a docstring gets one". Coverage was already
60 of 66; the six without were empty `__init__.py` package markers. This test keeps it at
66 of 66 rather than letting it decay back, which is the only way a documentation deliverable
stays true after the phase that produced it.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP = {"venv", "node_modules", ".git", "__pycache__", ".next", "out", ".wrangler",
        ".pytest_cache", "scratchpad", "notebooks"}


def modules():
    return sorted(p for p in ROOT.rglob("*.py") if not any(s in p.parts for s in SKIP))


def test_there_are_modules_to_check():
    assert len(modules()) > 50


@pytest.mark.parametrize("path", modules(), ids=lambda p: str(p.relative_to(ROOT)))
def test_module_has_a_docstring(path):
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    rel = path.relative_to(ROOT)
    assert doc, f"{rel} has no top-of-file docstring (see ARCHITECTURE.md)"
    assert len(doc.strip()) >= 20, f"{rel}'s docstring is too thin to be useful"
