
# Finance AI System

A production-grade financial analysis engine that combines live market data, classical ML, and large language models to produce structured, explainable stock analysis. Built to run on a single Linux machine — no cloud dependencies required.

> This is not a toy demo. It follows the architecture of production AI systems used in fintech and quant funds, scaled down to one machine.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FINANCE AI SYSTEM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐ │
│  │   DATA   │──▶│ SIGNALS  │──▶│   RISK   │──▶│  LLM BRAIN  │ │
│  │  LAYER   │   │  LAYER   │   │  ENGINE  │   │  (RAG/LLM)  │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────┬──────┘ │
│       │              │              │                 │         │
│  Live prices    XGBoost +       Kelly          Phi-3 / Llama   │
│  Indicators     SHAP            Criterion      3.3 70B         │
│  Fundamentals   Walk-forward    Confidence     Cites sources   │
│  Validation     validation      thresholds     Never halluci-  │
│                                 Drawdown       nates           │
│                                 protection                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    INTERFACE LAYER                          ││
│  │         Streamlit Dashboard  +  Telegram Bot                ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  SELF-EVALUATION LOOP                       ││
│  │    Logs every prediction → Checks if past calls correct     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## The 5 layers

### Layer 1: Data (`/utils`)
- Pulls live stock prices from **yfinance**
- Computes technical indicators: RSI, MACD, Bollinger Bands, moving averages, volume ratios
- Fetches fundamentals: P/E, debt-to-equity, margins, growth rates
- Validates all data for anomalies and missing values
- Outputs clean, structured datasets ready for modelling
- **S&P 500 panel** (`utils/sp500_data.py`): scrapes current constituents and builds a
  tidy ~7-year daily OHLCV panel (`data/sp500_panel.parquet`) for the RankAlpha
  cross-sectional model. See [ROADMAP.md](ROADMAP.md) and [CONCEPT.md](CONCEPT.md).

> ⚠️ **SURVIVORSHIP BIAS (known v1 limitation).** The S&P 500 panel uses the **current**
> index membership applied across the entire history. Companies that were dropped or
> delisted over the lookback window are silently excluded, which biases any backtest
> **upward** (we only keep the survivors). This is accepted for v1 and will be fixed later
> with **point-in-time constituents** (true index membership as of each historical date).

---

## RankAlpha — cross-sectional momentum ranker + demo product

A market-neutral **learning-to-rank** system over the S&P 500: a LightGBM LambdaMART model
scores the cross-section each month; the research backtest trades the tails long/short, and a
long-only **demo product** turns the scores into a transparent, risk-managed "pie." Full
methodology in [CONCEPT.md](CONCEPT.md) and [ROADMAP.md](ROADMAP.md); the honest write-up of
what it can and cannot claim is in [LIMITATIONS.md](LIMITATIONS.md).

Pipeline (`utils/` data → `signals/` model → `portfolio/` product):
- `utils/sp500_data.py` → panel · `utils/sp500_features.py` → 7 point-in-time rank features ·
  `utils/sp500_labels.py` → forward-return rank/decile labels (21-day horizon).
- `signals/baseline_momentum.py` → no-ML 12-1 momentum baseline · `signals/lgbm_ranker.py` →
  walk-forward ranker (21-day embargo) · `signals/evaluate.py` → honest evaluation.
- `portfolio/engine.py` → `build_portfolio()` (long-only, inverse-vol + position-cap +
  volatility-target) · `portfolio/llm_explainer.py` → plain-English explainer.

**Headline (out-of-sample 2022–2026, after 10 bps/side):** model Sharpe **1.14** vs momentum
baseline **0.82**; Rank IC **0.050** vs 0.041; edge survives costs to 30 bps. The model's
dominant feature is **6-month volatility** by importance — and per-holding SHAP shows the long
book tilts to **higher-volatility** names (importance ≠ direction; it is *not* a low-vol tilt).

### Run the demo product (Streamlit)

```bash
streamlit run app.py
```

Enter an amount and a risk level (**Conservative ≈ 10% vol / Balanced ≈ 14% / Aggressive ≈
20%**); the page shows the allocation pie, a self-explaining holdings table, an honest
(historical, not forecast) risk panel, and a plain-English summary. The risk slider updates
the **invested-vs-cash** split live (higher target → more invested, never levered).

- **Local LLM explainer:** uses **Ollama `phi3:mini`** if running (`ollama serve` +
  `ollama pull phi3:mini`); otherwise it falls back to a deterministic templated summary, so
  the app runs with no LLM. The explainer is fed **only** the model's real factor/SHAP
  reasons + stats and is instructed never to predict, advise, or invent numbers. Note:
  `phi3:mini` on a CPU/low-RAM laptop is slow (~1 tok/s), so the app uses a 150 s budget and
  will often show the (equally factual) template; the local-LLM path is proven to work given
  more time or better hardware.
- **Deploy notes:** the first build fits the frozen model (~1–2 min, cached for the session).
  A **free cloud host cannot run a local LLM** — a hosted version would swap Ollama for
  **Groq** (same prompt, same no-invent guardrail); not built yet.

> ⚠️ **EDUCATIONAL SIMULATION — NOT investment advice. No real money. Past backtest does not
> predict future returns.** The pie is a methodology demo on survivorship-biased data.

### Layer 2: Signals (`/signals`)
- Trains an **XGBoost** classifier on years of historical data
- Uses **walk-forward validation** to prevent look-ahead bias — no data leakage
- Outputs a probability score (not just up/down) with calibrated confidence
- **SHAP explainability** on every prediction — shows exactly which features drove the decision
- Tracks feature importance drift over time

### Layer 3: Risk (`/risk`)
- Applies **Kelly Criterion** for mathematically optimal position sizing
- Enforces **confidence thresholds** — the system refuses to act when uncertain
- Drawdown protection — automatically reduces exposure during losing streaks
- Separates signal quality from position size — a good signal with low confidence = small position

### Layer 4: LLM Brain (`/llm`)
- Uses **local Phi-3** via Ollama or **cloud Llama 3.3 70B** via Groq's free API
- Receives all signals as **structured context through a RAG pipeline**
- Produces analyst-grade reasoning that **cites every number**
- Never hallucinating — the LLM only reasons over data it actually received
- Never gives buy/sell advice — provides analysis, not recommendations

### Layer 5: Interface (`/interface`)
- **Streamlit dashboard** — full visual interface for analysis, charts, and signals
- **Telegram bot** — real-time alerts when significant signals are detected
- Clean, responsive UI that surfaces the right information at the right time

### Self-Evaluation Loop
- Logs every prediction with timestamp, confidence, and reasoning
- Periodically checks whether past calls were directionally correct
- Maintains a transparent track record — no hiding bad predictions
- Uses evaluation results to flag when model retraining is needed

---

## Tech stack

| Category | Technologies |
|----------|-------------|
| **ML/AI** | Python, XGBoost, SHAP, scikit-learn, pandas, NumPy |
| **Deep Learning** | PyTorch, LSTM (for sequence modelling) |
| **LLM** | LangChain, Ollama (Phi-3), Groq API (Llama 3.3 70B), ChromaDB |
| **Data** | yfinance, financial APIs, SQL |
| **Interface** | Streamlit, python-telegram-bot, Plotly |
| **Infrastructure** | Linux, Git, cron scheduling |

---

## Project structure

```
financeai/
├── signals/              # ML signal generation
│   ├── model.py          # XGBoost training + walk-forward validation
│   ├── features.py       # Feature engineering pipeline
│   └── explainer.py      # SHAP explainability
├── risk/                 # Risk management engine
│   ├── kelly.py          # Kelly Criterion position sizing
│   ├── thresholds.py     # Confidence thresholds + drawdown protection
│   └── portfolio.py      # Portfolio-level risk aggregation
├── llm/                  # LLM reasoning layer
│   ├── agent.py          # LangChain agent + RAG orchestration
│   ├── prompts.py        # System prompts + output formatting
│   └── rag.py            # Document embedding + retrieval
├── interface/            # User-facing layer
│   ├── dashboard.py      # Streamlit app
│   └── telegram_bot.py   # Telegram alert bot
├── utils/                # Shared utilities
│   ├── data_pipeline.py  # Market data fetching + cleaning
│   ├── indicators.py     # Technical indicator calculations
│   └── validators.py     # Data quality checks
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Key design decisions

**Why XGBoost over deep learning for signals?**
Gradient boosted trees outperform neural nets on structured tabular data with <100k rows. Our feature set is hand-engineered financial indicators, not raw sequences — XGBoost is the right tool. LSTM is used separately for sequence modelling where appropriate.

**Why Kelly Criterion?**
Most ML projects predict direction but never address "how much." Kelly provides a mathematically optimal answer to position sizing given your edge and confidence. The system never bets more than Kelly suggests, and often bets less.

**Why does the system refuse to trade?**
A system that always has an opinion is a dangerous system. When confidence is below threshold, the most profitable action is inaction. This design choice alone separates this from 99% of student projects.

**Why RAG instead of fine-tuning the LLM?**
Fine-tuning creates a static model. RAG lets the LLM reason over today's data — live prices, fresh news, current indicators. The LLM never answers from memory; it only answers from what it can see and cite.

**Why local + cloud LLM options?**
Phi-3 runs locally for privacy and zero cost. Llama 3.3 70B via Groq's free API provides higher quality reasoning when needed. The system works with either — no vendor lock-in.

---

## What this project demonstrates

- **Linux engineering** — runs continuously on a personal machine
- **Financial data engineering** — live market data, indicators, fundamentals
- **Classical ML** — XGBoost with proper temporal validation
- **Model explainability** — SHAP on every prediction (EU AI Act compliant)
- **Risk management** — Kelly Criterion, confidence thresholds, drawdown protection
- **LLM integration** — RAG pipeline, structured prompting, citation enforcement
- **Production patterns** — logging, self-evaluation, deployment, alerting
- **Honest evaluation** — tracks past predictions, documents where the model fails

---

## Status

🟡 **In active development** — architecture complete, building each layer incrementally with daily commits.

---

## Disclaimer

This system is for **educational and research purposes only**. It does not provide financial advice. Past performance of the model does not indicate future results. Never make investment decisions based solely on algorithmic output.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built by [Priyansh Patel](https://github.com/PRIYANSH29-boop) — CS student in London, building at the intersection of finance and AI.
