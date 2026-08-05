"""#28 D-2 — a fresh clone must be able to rebuild the pipeline from make targets.

#25 finding D-2: `data/*.parquet` is gitignored and no make target regenerated the base
artifacts, so every documented target (`analyse`, `lab`, `value`, `web-bundle`, …) consumed
files nothing could rebuild.

The second half matters as much as the first: there must be NO train target. The ranker is
frozen, and a `make train` sitting next to `make pipeline` is an invitation to refit it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAKEFILE = ROOT / "Makefile"
README = ROOT / "README.md"


def targets():
    return set(re.findall(r"^([a-zA-Z0-9_-]+):", MAKEFILE.read_text(), re.M))


@pytest.mark.parametrize("target", ["panel", "features", "labels", "pipeline"])
def test_base_stage_has_a_make_target(target):
    assert target in targets(), f"`make {target}` is missing — D-2 is not fixed"


def test_pipeline_runs_the_three_stages_in_order():
    body = re.search(r"^pipeline:(.*)$", MAKEFILE.read_text(), re.M).group(1).split()
    assert body == ["panel", "features", "labels"], f"pipeline deps out of order: {body}"


@pytest.mark.parametrize("module", ["utils.sp500_data", "utils.sp500_features",
                                    "utils.sp500_labels"])
def test_the_target_invokes_the_real_module(module):
    assert f"-m {module}" in MAKEFILE.read_text(), f"no target runs {module}"


def test_there_is_no_train_target():
    """The frozen-model invariant, enforced in the build system rather than in prose."""
    assert "train" not in targets()
    assert "retrain" not in targets()


def test_the_makefile_says_why_there_is_no_train_target():
    """Without the reason written down, the next person adds `make train` as an oversight."""
    text = MAKEFILE.read_text()
    assert "NO train target" in text or "no train target" in text
    assert "FROZEN" in text or "frozen" in text


def test_readme_documents_the_fresh_clone_rebuild():
    assert "make pipeline" in README.read_text()
