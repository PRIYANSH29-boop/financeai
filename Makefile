# RankAlpha — convenience targets. Uses the repo venv if present.
PY ?= $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python3)

.PHONY: analyse lab regimes audit value universe test test-all

## Regenerate the analyser scorecard + charts from committed data (no refit, no network).
analyse:
	$(PY) scripts/analyse.py

## Strategy Lab v0: score momentum vs momentum+low-vol from committed data (no refit, no network).
lab:
	$(PY) scripts/strategy_lab.py

## Regime stress-test (2008 GFC + COVID). NEEDS NETWORK on first run (yfinance); survivorship-biased.
regimes:
	$(PY) scripts/regime_stress_test.py

## Phase 17 — fundamentals data-quality audit against SEC EDGAR XBRL. NEEDS NETWORK.
## Emits figures/audit/fundamentals_audit.md with a computed GO/NO-GO. Exits non-zero on NO-GO.
audit:
	$(PY) scripts/audit_fundamentals.py

## Phase 17 — verify the audit harness itself offline (no network, no key).
audit-self-test:
	$(PY) scripts/audit_fundamentals.py --self-test

## Phase 18 — value-factor A/B. Blocked unless the #17 audit verdict is GO.
## Add --build on first run to fetch the point-in-time fundamentals.
value:
	$(PY) scripts/value_factor.py

## Phase 16 — mid+large-cap universe, panel, features, labels, RETRAIN, report. NEEDS NETWORK, slow.
universe:
	$(PY) scripts/expand_universe.py

## Run the base-analyser unit tests.
test:
	$(PY) -m pytest analytics/tests/ -q

## Run every unit-test suite (analyser, strategy lab, value factor, pie engine, data audit).
test-all:
	$(PY) -m pytest analytics/tests/ lab/tests/ portfolio/tests/ audit/tests/ test/ -q
