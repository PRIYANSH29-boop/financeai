# RankAlpha — convenience targets. Uses the repo venv if present.
PY ?= $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python3)

.PHONY: analyse lab regimes regimes-backtest audit value universe sectors web-bundle web-dev deploy test test-all

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

## Phase 20 — map a sector to every wide-universe name (yfinance A + SEC SIC B), then the
## cap-binding report. NEEDS NETWORK (yfinance query2 + data.sec.gov); cached/resumable.
## Writes data/universe_midlarge_sectors.csv + figures/lab/sector_mapping.md.
sectors:
	$(PY) -m scripts.map_sectors
	$(PY) -m scripts.sector_report

## Phase 26 — style census + style x season Rank IC grids (no training; reads the frozen OOS).
styles:
	$(PY) scripts/style_season_report.py

## Phase 21 — regime-segmented backtest: slice the committed history by market weather.
## Offline, no refit, no prediction. Writes figures/lab/regime_report.md.
regimes-backtest:
	$(PY) -m lab.regime_backtest

## Phase 19 — export the static web bundle (every number the frontend shows). Offline.
web-bundle:
	$(PY) scripts/export_web_bundle.py

## Phase 19 — run the Next.js dev server (needs Node + `cd web && npm install` once).
web-dev: web-bundle
	cd web && npm run dev

## Phase 19 — build the static export and deploy it to Cloudflare Pages (project: rankalpha).
## Requires Node and a Cloudflare API token with Account · Cloudflare Pages · Edit:
##   export CLOUDFLARE_API_TOKEN=<token>   (account id below is not secret)
## Live at https://rankalpha.pages.dev
CLOUDFLARE_ACCOUNT_ID ?= 1d83e7785436264464abe428a70bc94c
deploy:
	cd web && npm run build
	CLOUDFLARE_ACCOUNT_ID=$(CLOUDFLARE_ACCOUNT_ID) \
		npx --yes wrangler@latest pages deploy web/out \
		--project-name=rankalpha --branch=main --commit-dirty=true

## Run the base-analyser unit tests.
test:
	$(PY) -m pytest analytics/tests/ -q

## Run every unit-test suite (analyser, strategy lab, value factor, pie engine, data audit).
test-all:
	$(PY) -m pytest analytics/tests/ lab/tests/ portfolio/tests/ audit/tests/ test/ -q
