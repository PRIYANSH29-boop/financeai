
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
