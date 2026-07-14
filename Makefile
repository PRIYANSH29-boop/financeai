# RankAlpha — convenience targets. Uses the repo venv if present.
PY ?= $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python3)

.PHONY: analyse lab regimes test

## Regenerate the analyser scorecard + charts from committed data (no refit, no network).
analyse:
	$(PY) scripts/analyse.py

## Strategy Lab v0: score momentum vs momentum+low-vol from committed data (no refit, no network).
lab:
	$(PY) scripts/strategy_lab.py

## Regime stress-test (2008 GFC + COVID). NEEDS NETWORK on first run (yfinance); survivorship-biased.
regimes:
	$(PY) scripts/regime_stress_test.py

## Run the base-analyser unit tests.
test:
	$(PY) -m pytest analytics/tests/ -q
