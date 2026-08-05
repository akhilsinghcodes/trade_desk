"""Generate plain-English signal summaries from indicator data."""
import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    last = df.dropna().iloc[-1]
    signals = []
    score = 0  # positive = bullish, negative = bearish

    # RSI
    rsi = last.get("rsi", 50)
    if rsi > 70:
        signals.append(("RSI", "warning", f"RSI is {rsi:.0f} — stock is OVERBOUGHT. May be due for a pullback."))
        score -= 1
    elif rsi < 30:
        signals.append(("RSI", "good", f"RSI is {rsi:.0f} — stock is OVERSOLD. May be due for a bounce."))
        score += 1
    else:
        signals.append(("RSI", "neutral", f"RSI is {rsi:.0f} — neutral zone, no extreme pressure either way."))

    # SMA crossover
    sma20 = last.get("sma20")
    sma50 = last.get("sma50")
    if sma20 and sma50:
        if sma20 > sma50:
            signals.append(("Trend", "good", "Short-term average is ABOVE long-term average — uptrend in place."))
            score += 1
        else:
            signals.append(("Trend", "warning", "Short-term average is BELOW long-term average — downtrend in place."))
            score -= 1

    # MACD
    macd = last.get("macd")
    signal = last.get("macd_signal")
    if macd is not None and signal is not None:
        if macd > signal:
            signals.append(("Momentum", "good", "MACD above signal line — momentum is building upward."))
            score += 1
        else:
            signals.append(("Momentum", "warning", "MACD below signal line — momentum is slowing or reversing."))
            score -= 1

    # Bollinger Bands
    close = last.get("close")
    bb_upper = last.get("bb_upper")
    bb_lower = last.get("bb_lower")
    bb_mid = last.get("bb_mid")
    if close and bb_upper and bb_lower:
        if close > bb_upper:
            signals.append(("Volatility", "warning", "Price is ABOVE upper Bollinger Band — extended move, possible reversal."))
            score -= 1
        elif close < bb_lower:
            signals.append(("Volatility", "good", "Price is BELOW lower Bollinger Band — could be a buying opportunity."))
            score += 1
        elif bb_mid and close > bb_mid:
            signals.append(("Volatility", "good", "Price is in upper half of Bollinger Bands — mild bullish bias."))
            score += 0.5
        else:
            signals.append(("Volatility", "neutral", "Price is in lower half of Bollinger Bands — mild bearish bias."))

    # Overall verdict
    if score >= 2:
        verdict = ("BULLISH", "green", "Most indicators point UP. Technically a reasonable time to consider buying — but do your own research.")
    elif score <= -2:
        verdict = ("BEARISH", "red", "Most indicators point DOWN. Technically caution territory — possible sell or avoid.")
    else:
        verdict = ("MIXED", "orange", "Signals are mixed. No strong technical case for buy or sell right now.")

    return {"signals": signals, "verdict": verdict, "score": score}
