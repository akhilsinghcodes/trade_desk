"""Options put/call ratio from yfinance — contrarian sentiment signal."""
import yfinance as yf


def get_options_sentiment(ticker: str) -> dict:
    """
    Fetches nearest expiry options chain and computes put/call ratio.
    Returns dict with:
    - put_call_ratio: float or None
    - put_volume: int
    - call_volume: int
    - signal: "bullish" / "bearish" / "neutral"
    - expiry: str (date used)
    """
    result = {
        "put_call_ratio": None,
        "put_volume": 0,
        "call_volume": 0,
        "signal": "neutral",
        "expiry": None,
    }

    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return result

        # Use nearest expiry
        expiry = expirations[0]
        result["expiry"] = expiry

        chain = t.option_chain(expiry)
        calls = chain.calls
        puts = chain.puts

        call_vol = int(calls["volume"].fillna(0).sum())
        put_vol = int(puts["volume"].fillna(0).sum())

        result["call_volume"] = call_vol
        result["put_volume"] = put_vol

        if call_vol > 0:
            pcr = put_vol / call_vol
            result["put_call_ratio"] = round(pcr, 2)

            # Contrarian interpretation:
            # PCR > 1.2 = lots of puts = fear = contrarian bullish
            # PCR < 0.6 = lots of calls = greed = contrarian bearish
            if pcr > 1.2:
                result["signal"] = "bullish"   # market too fearful = buy signal
            elif pcr < 0.6:
                result["signal"] = "bearish"   # market too greedy = caution
            else:
                result["signal"] = "neutral"

    except Exception:
        pass

    return result


def score_options(data: dict) -> tuple[str, str, str]:
    """Returns (label, status, text) for signals list."""
    pcr = data.get("put_call_ratio")
    signal = data.get("signal", "neutral")
    call_vol = data.get("call_volume", 0)
    put_vol = data.get("put_volume", 0)
    expiry = data.get("expiry", "")

    if pcr is None:
        return ("Options", "neutral", "No options data available.")

    if signal == "bullish":
        status = "good"
        text = (f"Put/Call ratio {pcr} (high puts vs calls) — market is fearful. "
                f"Contrarian signal: potential bounce. Expiry {expiry}.")
    elif signal == "bearish":
        status = "warning"
        text = (f"Put/Call ratio {pcr} (lots of call buying) — market is greedy. "
                f"Contrarian caution: may be overbought. Expiry {expiry}.")
    else:
        status = "neutral"
        text = (f"Put/Call ratio {pcr} — balanced options activity. "
                f"{call_vol:,} calls vs {put_vol:,} puts. Expiry {expiry}.")

    return ("Options", status, text)
