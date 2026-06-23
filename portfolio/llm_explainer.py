"""
Local LLM explainer — RankAlpha Phase 8.

Turns the engine's REAL per-holding factor/SHAP reasons + portfolio stats into a short
plain-English paragraph, using a local Ollama model (phi3:mini). The model is told to
EXPLAIN, never to predict or advise, and never to invent numbers.

Guardrails:
  * The prompt contains ONLY facts produced by the engine (weights, sectors, factor
    reasons, backtested risk stats). The model is asked to rephrase, not to add data.
  * Hard instruction: no forecasts, no advice, no invented tickers/numbers.
  * If Ollama is unreachable, a deterministic TEMPLATED summary (built from the same
    facts) is returned instead — the app never hard-depends on an LLM.

For a hosted deploy (free cloud can't run a local LLM), swap the Ollama call for Groq —
same prompt, same guardrails. Not built here.
"""

import json
import logging
import os

import requests

logger = logging.getLogger("llm_explainer")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

SYSTEM = (
    "You explain an EDUCATIONAL, SIMULATED stock portfolio in plain English. "
    "Use ONLY the facts given. Do NOT predict returns, do NOT give buy/sell advice, "
    "do NOT invent any tickers, numbers, or facts. Explain WHY the portfolio looks the "
    "way it does (which factors drove the picks, the risk posture, the cash level). "
    "Write 3-4 sentences. End by reminding the reader it is an educational simulation, "
    "not investment advice."
)


def _facts(portfolio: dict) -> dict:
    """Distil the portfolio dict into a compact, factual context for the LLM."""
    r = portfolio["risk_stats"]
    holdings = list(portfolio["weights"].items())
    top = []
    for tk, w in holdings[:6]:
        ex = portfolio["explanations"].get(tk, {})
        top.append({
            "ticker": tk,
            "weight_pct": round(w * 100, 1),
            "sector": ex.get("sector", "?"),
            "reasons": ex.get("reasons", [])[:2],
        })
    return {
        "as_of": portfolio["as_of"],
        "amount_usd": portfolio["amount"],
        "n_holdings": len(holdings),
        "invested_pct": round((1 - portfolio["cash_weight"]) * 100, 1),
        "cash_pct": round(portfolio["cash_weight"] * 100, 1),
        "target_vol_pct": round(r["target_vol"] * 100, 1),
        "backtested_ann_vol_pct": round(r["ann_vol"] * 100, 1),
        "historical_max_drawdown_pct": round(r["max_drawdown"] * 100, 1),
        "dominant_factor": "6-month volatility (long book tilts to higher-volatility names, "
                           "often with strong momentum)",
        "top_holdings": top,
    }


def _templated(facts: dict) -> str:
    """Deterministic fallback summary built from the same facts (no LLM)."""
    secs = sorted({h["sector"] for h in facts["top_holdings"] if h["sector"] != "?"})
    sec_str = ", ".join(secs[:4]) if secs else "several sectors"
    return (
        f"This simulated portfolio puts {facts['invested_pct']}% of "
        f"${facts['amount_usd']:,.0f} into {facts['n_holdings']} stocks and holds "
        f"{facts['cash_pct']}% in cash to meet a {facts['target_vol_pct']}% volatility "
        f"target. The model's picks are driven mainly by the 6-month volatility factor "
        f"(the long book tilts toward higher-volatility names, often with strong "
        f"momentum), spanning {sec_str}. Backtested annualized volatility was "
        f"{facts['backtested_ann_vol_pct']}% with a historical max drawdown of "
        f"{facts['historical_max_drawdown_pct']}% (historical, not a forecast). "
        f"Educational simulation only — not investment advice."
    )


def _trim_to_sentence(text: str) -> str:
    """Drop a dangling final sentence if generation was cut off by num_predict."""
    if not text or text[-1] in ".!?\"":
        return text
    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    return text[:cut + 1] if cut > 0 else text


def explain(portfolio: dict, timeout: float = 150.0) -> dict:
    """Return {'text': summary, 'source': 'ollama'|'template'}. Never raises.

    The deterministic TEMPLATE is the default — it is equally factual and instant, so the
    app never hangs on a RAM-starved box. The local LLM path is opt-in: set USE_LLM=1 to
    call Ollama (phi3:mini). timeout is generous because phi3:mini on CPU has a slow cold
    start; generation is bounded with num_predict and the model is kept warm for the session.
    """
    facts = _facts(portfolio)

    if os.environ.get("USE_LLM") != "1":
        return {"text": _templated(facts), "source": "template"}

    prompt = (
        SYSTEM + "\n\nFACTS (JSON):\n" + json.dumps(facts, indent=2)
        + "\n\nWrite the explanation now."
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "keep_alive": "10m",
                  "options": {"temperature": 0.2, "num_predict": 130}},
            timeout=timeout,
        )
        resp.raise_for_status()
        text = _trim_to_sentence(resp.json().get("response", "").strip())
        if text:
            return {"text": text, "source": "ollama"}
        logger.warning("Ollama returned empty response; using template")
    except Exception as e:  # noqa: BLE001 - any failure -> deterministic fallback
        logger.warning("Ollama unavailable (%s); using templated summary", e)
    return {"text": _templated(facts), "source": "template"}


if __name__ == "__main__":
    from portfolio.engine import build_portfolio
    p = build_portfolio(10_000, target_vol=0.14)
    out = explain(p)
    print(f"\n[source: {out['source']}]\n{out['text']}\n")
