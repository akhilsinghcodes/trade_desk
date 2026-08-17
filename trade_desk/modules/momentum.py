import yfinance as yf
import pandas as pd


def get_momentum(ticker: str) -> dict:
    """
    Computes price momentum (return) across multiple timeframes.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        dict with keys:
        - ret_1mo: float or None  (% return over last 21 trading days)
        - ret_3mo: float or None  (% return over last 63 trading days)
        - ret_6mo: float or None  (% return over last 126 trading days)
        - ret_1yr: float or None  (% return over last 252 trading days)
        - momentum_score: float  (-1.0 to 1.0, average of normalized signals)
        - trend: "strong_up" | "up" | "neutral" | "down" | "strong_down"
    """
    try:
        # Fetch 1.5 years of daily OHLCV data
        df = yf.download(ticker, period="18mo", auto_adjust=True, progress=False)

        # Handle MultiIndex columns from newer yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Use lowercase column names
        df.columns = [c.lower() for c in df.columns]

        # Get close prices
        close = df["close"]

        # Calculate returns for different timeframes
        ret_1mo = None
        ret_3mo = None
        ret_6mo = None
        ret_1yr = None

        if len(close) >= 22:
            ret_1mo = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100

        if len(close) >= 64:
            ret_3mo = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63] * 100

        if len(close) >= 127:
            ret_6mo = (close.iloc[-1] - close.iloc[-126]) / close.iloc[-126] * 100

        if len(close) >= 253:
            ret_1yr = (close.iloc[-1] - close.iloc[-252]) / close.iloc[-252] * 100

        # Normalize signals and compute momentum score
        signals = []

        if ret_1mo is not None:
            if ret_1mo > 5:
                signal = 1.0
            elif ret_1mo < -5:
                signal = -1.0
            else:
                signal = ret_1mo / 5  # proportional within [-5, 5]
            signals.append(max(-1.0, min(1.0, signal)))

        if ret_3mo is not None:
            if ret_3mo > 15:
                signal = 1.0
            elif ret_3mo < -15:
                signal = -1.0
            else:
                signal = ret_3mo / 15  # proportional within [-15, 15]
            signals.append(max(-1.0, min(1.0, signal)))

        if ret_6mo is not None:
            if ret_6mo > 25:
                signal = 1.0
            elif ret_6mo < -25:
                signal = -1.0
            else:
                signal = ret_6mo / 25  # proportional within [-25, 25]
            signals.append(max(-1.0, min(1.0, signal)))

        if ret_1yr is not None:
            if ret_1yr > 40:
                signal = 1.0
            elif ret_1yr < -40:
                signal = -1.0
            else:
                signal = ret_1yr / 40  # proportional within [-40, 40]
            signals.append(max(-1.0, min(1.0, signal)))

        # Compute average momentum score
        momentum_score = sum(signals) / len(signals) if signals else 0.0

        # Determine trend
        if momentum_score > 0.5:
            trend = "strong_up"
        elif momentum_score > 0.1:
            trend = "up"
        elif momentum_score < -0.5:
            trend = "strong_down"
        elif momentum_score < -0.1:
            trend = "down"
        else:
            trend = "neutral"

        return {
            "ret_1mo": ret_1mo,
            "ret_3mo": ret_3mo,
            "ret_6mo": ret_6mo,
            "ret_1yr": ret_1yr,
            "momentum_score": momentum_score,
            "trend": trend,
        }

    except Exception:
        # Return default values on failure
        return {
            "ret_1mo": None,
            "ret_3mo": None,
            "ret_6mo": None,
            "ret_1yr": None,
            "momentum_score": 0.0,
            "trend": "neutral",
        }


def score_momentum(data: dict) -> tuple[str, str, str]:
    """
    Formats momentum data into a label, status, and description.

    Args:
        data: dict returned from get_momentum()

    Returns:
        tuple of (label, status, text) where:
        - label: "Momentum"
        - status: "good" if strong_up or up, "warning" if down or strong_down, "neutral" otherwise
        - text: Human-readable description of momentum across available timeframes
    """
    label = "Momentum"

    trend = data.get("trend", "neutral")
    if trend in ("strong_up", "up"):
        status = "good"
    elif trend in ("down", "strong_down"):
        status = "warning"
    else:
        status = "neutral"

    # Build text description
    parts = []

    if data.get("ret_1mo") is not None:
        parts.append(f"{data['ret_1mo']:+.1f}% (1mo)")

    if data.get("ret_3mo") is not None:
        parts.append(f"{data['ret_3mo']:+.1f}% (3mo)")

    if data.get("ret_6mo") is not None:
        parts.append(f"{data['ret_6mo']:+.1f}% (6mo)")

    if data.get("ret_1yr") is not None:
        parts.append(f"{data['ret_1yr']:+.1f}% (1yr)")

    if parts:
        # Determine description based on trend
        if trend == "strong_up":
            prefix = "Strong upward momentum: "
        elif trend == "up":
            prefix = "Upward momentum: "
        elif trend == "strong_down":
            prefix = "Strong downward momentum: "
        elif trend == "down":
            prefix = "Downward momentum: "
        else:
            prefix = "Neutral momentum: "

        text = prefix + ", ".join(parts) + "."
    else:
        text = "Insufficient data to calculate momentum."

    return (label, status, text)
