# RankAlpha — convenience targets. Uses the repo venv if present.
PY ?= $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python3)

.PHONY: analyse test

## Regenerate the analyser scorecard + charts from committed data (no refit, no network).
analyse:
	$(PY) scripts/analyse.py

## Run the base-analyser unit tests.
test:
	$(PY) -m pytest analytics/tests/ -q
