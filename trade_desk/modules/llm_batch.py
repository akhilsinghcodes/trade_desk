"""Anthropic Batches API for portfolio LLM pre-computation.

On server start: submit one batch for all portfolio tickers.
Results stored in api_cache as llm_rationale:{ticker} (TTL 8h).
Requires ANTHROPIC_API_KEY env var.
"""
import json
import os
import threading
import time
import urllib.request
import urllib.error

from modules.db import cache_get, cache_set, log_llm_call

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HAIKU_MODEL = "claude-haiku-4-5-20251001"
BATCH_URL = "https://api.anthropic.com/v1/messages/batches"
LLM_CACHE_TTL = 8 * 3600
POLL_INTERVAL = 30

_BATCH_IN  = 0.50 / 1_000_000
_BATCH_OUT = 2.50 / 1_000_000

_SYSTEM = (
    "You are a concise equity trader. Give plain-English, actionable trade commentary. "
    "No fluff. No disclaimers. 1-2 sentences per field max."
)


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "message-batches-2024-09-24",
    }


def _make_prompt(ticker: str, close: float, pct_chg_5d: float) -> str:
    return f"""Ticker: {ticker}
Current price: ${close:.2f}
5-day change: {pct_chg_5d:+.1f}%

Respond ONLY as valid JSON with exactly these 3 keys:
{{
  "entry_rationale": "...",
  "risk_rationale": "...",
  "timing_note": "..."
}}"""


def _parse_result(raw: str) -> dict:
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"entry_rationale": None, "risk_rationale": None, "timing_note": None}


def _submit_batch(requests: list[dict]) -> str | None:
    payload = json.dumps({"requests": requests}).encode()
    req = urllib.request.Request(
        BATCH_URL, data=payload, headers=_headers(), method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("id")
    except Exception as e:
        print(f"[llm_batch] submit failed: {e}")
        return None


def _poll_batch(batch_id: str) -> str | None:
    """Returns processing_status string or None on error."""
    req = urllib.request.Request(
        f"{BATCH_URL}/{batch_id}", headers=_headers(), method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("processing_status")
    except Exception:
        return None


def _fetch_results(batch_id: str) -> list[dict]:
    req = urllib.request.Request(
        f"{BATCH_URL}/{batch_id}/results", headers=_headers(), method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            # Results are newline-delimited JSON
            lines = resp.read().decode().strip().split("\n")
            return [json.loads(line) for line in lines if line.strip()]
    except Exception as e:
        print(f"[llm_batch] fetch results failed: {e}")
        return []


def _poll_and_store(batch_id: str, custom_id_map: dict[str, str]):
    """Background thread: polls until ended, writes results to cache."""
    while True:
        time.sleep(POLL_INTERVAL)
        status = _poll_batch(batch_id)
        if status is None:
            print("[llm_batch] poll error, will retry")
            continue
        if status != "ended":
            print(f"[llm_batch] batch {batch_id} status={status}, waiting...")
            continue

        # Batch ended — fetch and store results
        results = _fetch_results(batch_id)
        ticker_map = {v: k for k, v in custom_id_map.items()}  # custom_id → ticker
        for item in results:
            custom_id = item.get("custom_id", "")
            ticker = ticker_map.get(custom_id)
            if not ticker:
                continue
            result_obj = item.get("result", {})
            if result_obj.get("type") != "succeeded":
                continue
            msg = result_obj.get("message", {})
            content = msg.get("content", [])
            text = next((b["text"] for b in content if b.get("type") == "text"), None)
            if text:
                parsed = _parse_result(text)
                usage = msg.get("usage", {})
                in_tok  = usage.get("input_tokens",  len(text) // 4)
                out_tok = usage.get("output_tokens", len(text) // 4)
                parsed["_cost_usd"] = round(in_tok * _BATCH_IN + out_tok * _BATCH_OUT, 6)
                parsed["_source"]   = "batch"
                log_llm_call(ticker, "haiku", "batch", parsed["_cost_usd"], in_tok, out_tok)
                cache_set(f"llm_rationale:{ticker}:haiku", parsed, ttl_seconds=LLM_CACHE_TTL)
                print(f"[llm_batch] cached rationale for {ticker} cost=${parsed['_cost_usd']:.6f}")
        break


def submit_portfolio_batch(tickers_with_data: list[dict]) -> None:
    """
    Submit a Haiku batch for portfolio tickers. Runs poll+store in background thread.

    tickers_with_data: list of {"ticker": str, "close": float, "pct_chg_5d": float}
    Skips tickers already in cache. No-ops if no API key or nothing to submit.
    """
    if not ANTHROPIC_API_KEY:
        return

    # Only submit for tickers not already cached
    to_submit = [
        t for t in tickers_with_data
        if cache_get(f"llm_rationale:{t['ticker']}:haiku") is None
    ]
    if not to_submit:
        return

    custom_id_map: dict[str, str] = {}  # ticker → custom_id
    requests = []
    for i, t in enumerate(to_submit):
        custom_id = f"port-{t['ticker']}-{i}"
        custom_id_map[t["ticker"]] = custom_id
        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": HAIKU_MODEL,
                "max_tokens": 300,
                "system": _SYSTEM,
                "messages": [{"role": "user", "content": _make_prompt(
                    t["ticker"], t["close"], t["pct_chg_5d"]
                )}],
            },
        })

    batch_id = _submit_batch(requests)
    if not batch_id:
        return

    print(f"[llm_batch] submitted batch {batch_id} for {[t['ticker'] for t in to_submit]}")
    thread = threading.Thread(
        target=_poll_and_store,
        args=(batch_id, custom_id_map),
        daemon=True,
    )
    thread.start()
