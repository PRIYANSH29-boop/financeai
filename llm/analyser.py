"""
LLM Analyser - takes structured stock data and produces analysis.
Supports both local Ollama and Groq cloud API.
"""

import os
import logging
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful, honest financial analyst.

Your job is to analyse stock data the user gives you. Strict rules:
1. ONLY use the numbers in the data provided. Never invent data.
2. If a value is missing, say "data unavailable" - don't guess.
3. Explain WHY you think what you think. Cite specific numbers.
4. Express uncertainty clearly. Say "this suggests" not "this means".
5. NEVER give buy/sell/hold recommendations. You analyse, you don't advise.
6. Keep responses focused and structured.
7. Flag any signals that contradict each other.

Output format:
- Summary (2-3 sentences)
- Technical signals (what indicators show)
- Fundamental signals (what fundamentals show)
- Risks and uncertainties
- Overall picture (no recommendation, just synthesis)
"""


class LLMAnalyser:
    """Routes analysis requests to local Ollama or Groq cloud."""

    def __init__(self, backend: Literal["ollama", "groq", "auto"] = "auto"):
        self.backend = self._choose_backend(backend)
        logger.info(f"LLMAnalyser using backend: {self.backend}")

    def _choose_backend(self, preference):
        """Pick which LLM to use."""
        if preference == "groq":
            return "groq"
        if preference == "ollama":
            return "ollama"

        # Auto: prefer Groq if API key is set, fall back to Ollama
        if os.getenv("GROQ_API_KEY"):
            return "groq"
        return "ollama"

    def analyse(self, summary: dict) -> Optional[str]:
        """
        Analyse a stock summary and return text analysis.

        Args:
            summary: dict from DataFetcher (the 'summary' field of the result)

        Returns:
            Analysis text, or None if it failed
        """
        prompt = self._build_prompt(summary)

        try:
            if self.backend == "groq":
                return self._call_groq(prompt)
            else:
                return self._call_ollama(prompt)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return None

    def _build_prompt(self, summary: dict) -> str:
        """Format the summary into a clear analyst prompt."""
        f = summary.get('fundamentals', {})

        prompt = f"""Analyse the following stock data:

TICKER: {summary['ticker']}
COMPANY: {f.get('name', 'Unknown')}
SECTOR: {f.get('sector', 'Unknown')} / {f.get('industry', 'Unknown')}
TIMESTAMP: {summary['timestamp']}

PRICE DATA:
- Current price: ${summary['current_price']}
- Today's change: {summary['price_change_pct']}%
- 52-week range: ${f.get('52w_low')} to ${f.get('52w_high')}

TECHNICAL INDICATORS:
- RSI (14): {summary['rsi']}    [<30 oversold, >70 overbought]
- MACD: {summary['macd']}
- MACD signal: {summary['macd_signal']}
- MACD above signal? {summary['macd_above_signal']}
- Above 20-day SMA? {summary['above_sma20']}
- Above 50-day SMA? {summary['above_sma50']}
- Volume vs 20-day average: {summary['volume_ratio']}x
- Bollinger Band position: {summary['bb_position']} [0=lower band, 1=upper band]

FUNDAMENTALS:
- P/E ratio: {f.get('pe_ratio')}
- P/B ratio: {f.get('pb_ratio')}
- Market cap: ${f.get('market_cap')}
- Revenue growth (YoY): {f.get('revenue_growth')}
- Profit margin: {f.get('profit_margins')}
- Debt-to-equity: {f.get('debt_to_equity')}
- Beta: {f.get('beta')}
- Dividend yield: {f.get('dividend_yield')}

Provide your analysis following the format in your instructions."""

        return prompt

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API."""
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3  # Low = more focused, less creative
        )
        return response.choices[0].message.content

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama."""
        import ollama

        response = ollama.chat(
            model="phi3:mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.3}
        )
        return response['message']['content']


# Quick test when run directly
if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    # Add parent to path so we can import utils
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_fetcher import DataFetcher

    ticker = input("Ticker: ").upper().strip()
    backend = input("Backend (groq/ollama/auto) [auto]: ").strip() or "auto"

    print(f"\nFetching data for {ticker}...")
    fetcher = DataFetcher()
    result = fetcher.fetch(ticker)

    if not result:
        print("Failed to fetch data.")
        sys.exit(1)

    print(f"\nAnalysing with {backend}...\n")
    analyser = LLMAnalyser(backend=backend)
    analysis = analyser.analyse(result['summary'])

    if analysis:
        print("=" * 70)
        print(f"ANALYSIS: {ticker}")
        print("=" * 70)
        print(analysis)
        print("=" * 70)
    else:
        print("Analysis failed.")