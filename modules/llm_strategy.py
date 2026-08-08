"""LLM-powered trade rationale via LiteLLM proxy (localhost:4000)."""
import json
import urllib.request
import urllib.error

LITELLM_BASE = "http://localhost:4000"
LITELLM_KEY  = "sk-1234"
MODEL        = "mistral-7b"
TIMEOUT      = 12  # seconds — fail fast if LLM slow


def _chat(prompt: str, system: str) -> str | None:
    """POST to LiteLLM /v1/chat/completions. Returns content string or None on failure."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        f"{LITELLM_BASE}/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LITELLM_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def generate_trade_rationale(
    ticker: str,
    verdict: str,
    confidence_pct: int,
    limit_entry: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
    risk_reward: float,
    conviction: str,
    breakdown: dict,
    thesis_oneliner: str,
    momentum_trend: str,
    sector_trend: str,
    piotroski: int,
    altman_zone: str,
    earnings_days: int | None,
    analyst_upside: float | None,
) -> dict:
    """
    Returns dict:
      - entry_rationale: str   — why this entry level makes sense
      - risk_rationale: str    — key risks to this trade
      - timing_note: str       — when/how to execute
    Returns None values on failure (fallback to algo-generated text).
    """
    system = (
        "You are a concise, experienced equity trader. "
        "Give plain-English, actionable trade commentary. "
        "No fluff. No disclaimers. 1-2 sentences per field max."
    )

    signal_summary = (
        f"Technical: {breakdown.get('technical', 0):+.2f}, "
        f"Fundamental: {breakdown.get('fundamental', 0):+.2f}, "
        f"Sentiment: {breakdown.get('sentiment', 0):+.2f}"
    )

    prompt = f"""Ticker: {ticker}
Verdict: {verdict} ({confidence_pct}% confident)
Signals: {signal_summary}
Thesis: {thesis_oneliner}
Entry: ${limit_entry:.2f} | Stop: ${stop_loss:.2f} | TP1: ${tp1:.2f} | TP2: ${tp2:.2f}
R:R = 1:{risk_reward:.2f} | Conviction: {conviction}
Momentum: {momentum_trend} | Sector: {sector_trend}
Piotroski: {piotroski}/9 | Altman: {altman_zone}
Earnings: {f'in {earnings_days}d' if earnings_days else 'N/A'}
Analyst upside: {f'{analyst_upside:+.1f}%' if analyst_upside else 'N/A'}

Respond ONLY as valid JSON with exactly these 3 keys:
{{
  "entry_rationale": "...",
  "risk_rationale": "...",
  "timing_note": "..."
}}"""

    raw = _chat(prompt, system)
    if not raw:
        return {"entry_rationale": None, "risk_rationale": None, "timing_note": None}

    # Extract JSON from response (model may wrap in markdown)
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"entry_rationale": None, "risk_rationale": None, "timing_note": None}
