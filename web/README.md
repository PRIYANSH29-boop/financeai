# RankAlpha — web pie product (Phase 19)

A static, backend-free frontend for the beta-targeted pie engine. Pick a capital amount and
a risk level (target beta); see the resulting pie, why each holding is in it, what happens
to it after one rebalance, and an honest scorecard against the benchmark.

> ⚠️ **Educational simulation. No real money. Not investment advice.** Every number is a
> historical characterisation of today's weights, never a forecast. The universe is
> survivorship-biased. See the repo root's `LIMITATIONS.md`.

## The one rule

**No number is typed into the frontend.** Everything the UI shows comes from a precomputed
bundle (`public/bundle/`) emitted by `scripts/export_web_bundle.py`, which is the only code
that ever calls the portfolio engine. The React app just multiplies weights by the user's
capital (a client-side scalar) and draws what the bundle already computed. If a figure is
not in the bundle, the UI cannot display it — there is no fetch-at-runtime, no API key, no
backend.

This is what makes the export trustworthy: the browser cannot invent a number, and the
engine is the single source of truth for every one it shows.

## Run it

```bash
# 1. from the repo root — generate the bundle (offline; reads committed data + frozen model)
make web-bundle                 # → web/public/bundle/{index.json, beta/b###.json}

# 2. in web/ — install once, then dev or static-build
cd web
npm install
npm run dev                     # http://localhost:3000
npm run build                   # static export → web/out/  (deploy this folder)
```

`make web-dev` does steps 1 and the dev server in one go.

The build is `output: 'export'` (see `next.config.mjs`) — `npm run build` writes a fully
static `out/` with no server component at runtime. Deploy `out/` to Vercel (or any static
host) exactly like the Streamlit demo: **the deploy is Priyansh's click, not automated
here.**

## Layout

```
web/
├── app/
│   ├── page.js         # server component: reads the bundle at BUILD time → first paint has real numbers
│   ├── layout.js       # root layout + metadata
│   └── globals.css     # design tokens (neutral base + teal accent; red = losses only)
├── components/
│   ├── PieApp.js       # state (capital, target beta); fetches other betas on demand, memoised
│   ├── Donut.js        # zone 2 — the pie; cash is a first-class slice
│   ├── Panels.js       # zones 1/3/4/5 + the "How it works" modal
│   └── Charts.js       # hand-rolled SVG growth + drawdown charts (no chart library)
├── lib/format.js       # money / pct / pp / beta formatting, decided once
└── public/bundle/      # gitignored — regenerate with `make web-bundle`
```

## Design rules (from the spec)

- Neutral base + **one** accent (teal). **Red is reserved for losses and warnings only** —
  the drawdown fill, the max-drawdown tile, negative figures. A weight-drift delta is
  *information, not a loss*, so it renders in a neutral tone and is labelled in percentage
  points (`+1.0pp`).
- Numbers are monospace and tabular, so columns line up and digits don't jitter as the
  slider moves.
- The beta slider greys out everything above the engine's **max achievable long-only beta**
  (exported, not hardcoded — currently ≈1.89) and snaps to the bundle's grid. If a
  slider position has no bundle entry it snaps to the nearest; it never interpolates an
  invented pie.
- Cash is a first-class pie slice with its own explanation — a low-beta pie is mostly cash,
  and hiding that would misrepresent the product.
- The **realised beta inside the worst-drawdown window** is shown next to the headline beta.
  A beta estimated over calm months is not a promise about a crash, and the UI says so.

## Scope (v1)

No accounts, no chat, no tax/currency handling, and **no forward projections of any kind**.
The bundle deliberately exports no field a frontend could misread as a predicted return.
