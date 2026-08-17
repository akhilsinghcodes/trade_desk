"""LLM-powered trade rationale.

Backends:
  "local"  — LiteLLM proxy at localhost:4000 (default, zero API cost)
  "haiku"  — Claude Haiku via `claude` CLI (uses Claude Code subscription)
             Falls back to ANTHROPIC_API_KEY env var if CLI unavailable.

Cache: api_cache table (key=llm_rationale:{ticker}, TTL 8h).
Batch pre-computed results (from llm_batch.py) are served from cache automatically.
"""
import json
import os
import subprocess
import urllib.request
import urllib.error

from modules.db import cache_get, cache_set, log_llm_call

LLM_CACHE_TTL = 8 * 3600

LITELLM_BASE = "http://localhost:4000"
LITELLM_KEY  = "sk-1234"
LOCAL_MODEL  = "gemma2:9b"
HAIKU_MODEL  = "claude-haiku-4-5-20251001"
CLAUDE_CLI   = os.path.expanduser("~/.local/bin/claude")
TIMEOUT      = 35


def _chat_local(prompt: str, system: str) -> str | None:
    payload = json.dumps({
        "model": LOCAL_MODEL,
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


# Haiku pricing per million tokens
_HAIKU_IN  = 1.00 / 1_000_000
_HAIKU_OUT = 5.00 / 1_000_000
_BATCH_IN  = 0.50 / 1_000_000
_BATCH_OUT = 2.50 / 1_000_000


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _chat_haiku_cli(prompt: str, system: str) -> tuple[str, float, int, int] | None:
    """Returns (text, cost_usd, in_tok, out_tok) or None. Cost estimated (CLI doesn't return usage)."""
    if not os.path.exists(CLAUDE_CLI):
        return None
    full_prompt = f"{system}\n\n{prompt}"
    try:
        result = subprocess.run(
            [CLAUDE_CLI, "-p", full_prompt, "--model", HAIKU_MODEL,
             "--output-format", "text"],
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        if result.returncode == 0 and result.stdout.strip():
            text = result.stdout.strip()
            in_tok  = _estimate_tokens(full_prompt)
            out_tok = _estimate_tokens(text)
            # subscription call — no marginal $ cost, unlike metered API/batch calls
            cost = 0.0
            return text, cost, in_tok, out_tok
    except Exception:
        pass
    return None


def _chat_haiku_api(prompt: str, system: str) -> tuple[str, float, int, int] | None:
    """Returns (text, cost_usd, in_tok, out_tok) with exact token counts from API response."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    payload = json.dumps({
        "model": HAIKU_MODEL,
        "max_tokens": 300,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read())
            text = data["content"][0]["text"].strip()
            usage = data.get("usage", {})
            in_tok  = usage.get("input_tokens",  _estimate_tokens(prompt))
            out_tok = usage.get("output_tokens", _estimate_tokens(text))
            cost = in_tok * _HAIKU_IN + out_tok * _HAIKU_OUT
            return text, cost, in_tok, out_tok
    except Exception:
        return None


def _chat(prompt: str, system: str, backend: str = "local") -> tuple[str | None, float, str, int, int]:
    """Returns (text, cost_usd, source, in_tok, out_tok)."""
    if backend == "haiku":
        res = _chat_haiku_cli(prompt, system)
        if res:
            return res[0], res[1], "haiku_cli", res[2], res[3]
        res = _chat_haiku_api(prompt, system)
        if res:
            return res[0], res[1], "haiku_api", res[2], res[3]
        return None, 0.0, "haiku_fail", 0, 0
    text = _chat_local(prompt, system)
    return text, 0.0, "local", 0, 0


def _format_signals(signals: list) -> str:
    """Format raw signal tuples into bullish/bearish/neutral sections."""
    bullish, bearish, neutral = [], [], []
    for sig in signals:
        if not sig or len(sig) < 3:
            continue
        label, status, desc = sig[0], sig[1], sig[2]
        line = f"  [{label}] {desc}"
        if status == "good":
            bullish.append(line)
        elif status == "warning":
            bearish.append(line)
        else:
            neutral.append(line)
    parts = []
    if bullish:
        parts.append("BULLISH signals:\n" + "\n".join(bullish))
    if bearish:
        parts.append("BEARISH signals:\n" + "\n".join(bearish))
    if neutral:
        parts.append("NEUTRAL signals:\n" + "\n".join(neutral))
    return "\n\n".join(parts)


def _format_signals_compact(signals: list) -> str:
    """Compressed signal format for smaller local models."""
    bullish = [f"{s[0]}:{s[2][:60]}" for s in signals if len(s) >= 3 and s[1] == "good"]
    bearish = [f"{s[0]}:{s[2][:60]}" for s in signals if len(s) >= 3 and s[1] == "warning"]
    parts = []
    if bullish:
        parts.append("BULL: " + " | ".join(bullish[:6]))
    if bearish:
        parts.append("BEAR: " + " | ".join(bearish[:6]))
    return "\n".join(parts)


def _build_prompt(ticker, signals, earnings_days, backend):
    """LLM is a pure explainer here — no trade_action/conviction/position_size.
    Those were the model inventing decisions with no data edge over the algo;
    it only gets asked to summarize and flag contradictions in signals that
    already exist, nothing it's asked to output should be treated as advice."""
    if backend == "local":
        signals_text = _format_signals_compact(signals)
        system = "You summarize stock signals in plain English. Output only valid JSON. No explanation outside JSON. Never recommend an action or position size."
        prompt = f"""Stock: {ticker}
Earnings: {f'in {earnings_days}d' if earnings_days else 'N/A'}

{signals_text}

Output this JSON and nothing else:
```json
{{
  "entry_rationale": "one sentence summarizing what the bullish signals suggest, no recommendation",
  "risk_rationale": "one sentence summarizing what the bearish signals suggest, no recommendation",
  "key_conflict": "one sentence naming the biggest bull vs bear contradiction, if any"
}}
```"""
    else:
        signals_text = _format_signals(signals)
        system = (
            "You summarize equity signals in plain English for a reader who will decide for themselves. "
            "Read ALL signals carefully. Identify the most important conflicts. "
            "Do NOT recommend a trade action, conviction level, or position size — those are not yours to decide. "
            "Be specific and non-obvious. 1-2 sentences per field. No disclaimers."
        )
        prompt = f"""Ticker: {ticker}
Earnings: {f'in {earnings_days}d' if earnings_days else 'N/A'}

{signals_text}

Summarize only — do not recommend a trade. Respond ONLY as valid JSON with exactly these 3 keys:
{{
  "entry_rationale": "what the bullish signals suggest, citing specifics — not a recommendation",
  "risk_rationale": "what the bearish signals suggest, citing specifics — not a recommendation",
  "key_conflict": "the most important contradiction in the signal set and what it implies"
}}"""
    return system, prompt


def generate_trade_rationale(
    ticker: str,
    signals: list,
    earnings_days: int | None,
    backend: str = "local",
) -> dict:
    system, prompt = _build_prompt(ticker, signals, earnings_days, backend)

    _empty = {"entry_rationale": None, "risk_rationale": None, "key_conflict": None}

    # Cache key includes backend — local and haiku results stored separately
    cache_key = f"llm_rationale:{ticker}:{backend}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    raw, cost_usd, source, in_tok, out_tok = _chat(prompt, system, backend=backend)
    if not raw:
        return _empty

    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        result = json.loads(raw[start:end])
        result["_cost_usd"] = round(cost_usd, 6)
        result["_source"]   = source
        log_llm_call(ticker, backend, source, cost_usd, in_tok, out_tok)
        cache_set(cache_key, result, ttl_seconds=LLM_CACHE_TTL)
        return result
    except Exception:
        return _empty
