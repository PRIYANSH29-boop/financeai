"""#28 D-1 — every Python dependency must be pinned exactly.

#25 finding D-1: 0 of the declared dependencies were pinned; every line was `>=`. A frozen
model whose scores must not move cannot rest on a dependency set that `pip install` is free
to re-resolve — a different LightGBM produces a different ranking while every doc still says
"frozen".
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REQ = Path(__file__).resolve().parent.parent / "requirements.txt"
FROZEN_MODEL_CRITICAL = "lightgbm"


def requirement_lines():
    return [ln.strip() for ln in REQ.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def test_there_are_requirements_to_check():
    assert len(requirement_lines()) >= 10


@pytest.mark.parametrize("line", requirement_lines())
def test_every_requirement_is_an_exact_pin(line):
    spec = line.split("#")[0].strip()
    assert "==" in spec, f"{spec!r} is not an exact pin"
    for loose in (">=", "<=", "~=", ">", "<", "!="):
        assert loose not in spec.replace("==", ""), f"{spec!r} still carries a {loose} range"
    name, _, version = spec.partition("==")
    assert name and version, f"{spec!r} is malformed"
    # A pin has to be a real version, not `pandas==*` or a stray marker.
    assert re.fullmatch(r"[0-9][0-9A-Za-z.\-+]*", version), f"{version!r} is not a version"


def test_the_frozen_model_dependency_is_pinned_and_says_why():
    """LightGBM is part of the model artifact, not just a library — the file must say so, or
    the next person 'tidies up' the pin without knowing what it protects."""
    text = REQ.read_text()
    assert re.search(rf"^{FROZEN_MODEL_CRITICAL}==", text, re.M), "lightgbm is not pinned"
    assert "FROZEN-MODEL CRITICAL" in text
