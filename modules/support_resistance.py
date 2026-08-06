"""Support & resistance levels + suggested entry/stop-loss."""
import pandas as pd


def pivot_points(df: pd.DataFrame) -> dict:
    """Classic pivot points from last completed session."""
    last = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    h, lo, c = last["high"], last["low"], last["close"]
    pp = (h + lo + c) / 3
    return {
        "PP": round(pp, 2),
        "R1": round(2 * pp - lo, 2),
        "R2": round(pp + ( h - lo), 2),
        "R3": round(h + 2 * (pp - lo), 2),
        "S1": round(2 * pp - h, 2),
        "S2": round(pp - ( h - lo), 2),
        "S3": round(lo - 2 * (h - pp), 2),
    }


def swing_levels(df: pd.DataFrame, window: int = 10) -> dict:
    """Find recent swing highs and lows using rolling window."""
    highs = df["high"].rolling(window, center=True).max()
    lows = df["low"].rolling(window, center=True).min()

    swing_highs = df["high"][df["high"] == highs].dropna()
    swing_lows = df["low"][df["low"] == lows].dropna()

    # Keep top 3 most recent distinct levels
    resistance = sorted(swing_highs.tail(5).unique(), reverse=True)[:3]
    support = sorted(swing_lows.tail(5).unique())[:3]

    return {
        "resistance": [round(r, 2) for r in resistance],
        "support": [round(s, 2) for s in support],
    }


def suggest_trade(df: pd.DataFrame, verdict: str) -> dict:
    """
    Suggest entry, stop-loss, and take-profit based on S/R levels.
    Uses ATR for stop-loss sizing.
    """
    close = df["close"].iloc[-1]
    atr = _atr(df, 14)

    pivots = pivot_points(df)
    swings = swing_levels(df)

    if verdict == "BUY":
        entry = round(close, 2)
        stop_loss = round(close - 1.5 * atr, 2)
        # TP1 = 2× risk, TP2 = 3× risk (guarantees R:R ≥ 2)
        risk_amt = close - stop_loss
        take_profit_1 = round(close + 2 * risk_amt, 2)
        take_profit_2 = round(close + 3 * risk_amt, 2)
        risk = round(risk_amt, 2)
        reward = round(take_profit_1 - close, 2)
    elif verdict in ("SELL / AVOID", "SELL"):
        entry = round(close, 2)
        stop_loss = round(close + 1.5 * atr, 2)
        risk_amt = stop_loss - close
        take_profit_1 = round(close - 2 * risk_amt, 2)
        take_profit_2 = round(close - 3 * risk_amt, 2)
        risk = round(risk_amt, 2)
        reward = round(close - take_profit_1, 2)
    else:  # HOLD
        return {
            "action": "HOLD / WATCH",
            "note": "No trade suggested. Wait for a clearer signal.",
            "current_price": round(close, 2),
            "key_support": swings["support"],
            "key_resistance": swings["resistance"],
        }

    rr = round(reward / risk, 2) if risk > 0 else 0

    return {
        "action": verdict,
        "current_price": round(close, 2),
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "risk_per_share": risk,
        "reward_per_share": reward,
        "risk_reward_ratio": rr,
        "atr": round(atr, 2),
        "key_support": swings["support"],
        "key_resistance": swings["resistance"],
        "pivots": pivots,
    }


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]
