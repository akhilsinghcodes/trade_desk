"""Price Momentum Oscillator (PMO) with relative strength vs SPY."""
import pandas as pd
from modules.fetch import get_ohlcv


def compute_pmo(df: pd.DataFrame) -> pd.Series:
    """
    Compute PMO (Price Momentum Oscillator) for a DataFrame.

    Algorithm (double-smoothed ROC):
    1. ROC = (close / close.shift(1) - 1) * 100
    2. Smooth1 = ROC.ewm(span=34, adjust=False).mean() * 10
    3. PMO = Smooth1.ewm(span=19, adjust=False).mean()

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with PMO values
    """
    close = df["close"].astype(float)

    # 1. Rate of Change
    roc = (close / close.shift(1) - 1) * 100

    # 2. First smoothing (34-period EMA * 10)
    smooth1 = roc.ewm(span=34, adjust=False).mean() * 10

    # 3. PMO (19-period EMA of smooth1)
    pmo = smooth1.ewm(span=19, adjust=False).mean()

    return pmo


def compute_pmo_rs(ticker: str, period: str = "1y") -> dict:
    """
    Compute PMO relative strength signal vs SPY.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        period: Period for OHLCV data (default "1y")

    Returns:
        dict with keys:
        - pmo: float (latest PMO value for ticker)
        - pmo_signal: float (latest PMO signal line)
        - spy_pmo: float (latest PMO value for SPY)
        - spy_pmo_signal: float (latest PMO signal line for SPY)
        - rs_score: float (ticker_pmo - spy_pmo, positive = outperforming)
        - rs_norm: float (normalized to [-1, 1], clipped [-5,5] / 5)
        - trend: str ("leading" if rs_norm > 0.2, "lagging" if rs_norm < -0.2, else "neutral")
    """
    try:
        # Fetch OHLCV data for ticker and SPY
        df_ticker = get_ohlcv(ticker, period=period)
        df_spy = get_ohlcv("SPY", period=period)

        if df_ticker.empty or df_spy.empty:
            return _pmo_default()

        # Compute PMO for both
        pmo_ticker = compute_pmo(df_ticker)
        pmo_spy = compute_pmo(df_spy)

        # Compute PMO signal (9-period EMA)
        pmo_signal_ticker = pmo_ticker.ewm(span=9, adjust=False).mean()
        pmo_signal_spy = pmo_spy.ewm(span=9, adjust=False).mean()

        # Get latest values
        pmo_val = float(pmo_ticker.iloc[-1])
        pmo_sig_val = float(pmo_signal_ticker.iloc[-1])
        spy_pmo_val = float(pmo_spy.iloc[-1])
        spy_pmo_sig_val = float(pmo_signal_spy.iloc[-1])

        # Compute relative strength score
        rs_score = pmo_val - spy_pmo_val

        # Normalize to [-1, 1] by clipping to [-5, 5] and dividing by 5
        rs_norm = max(-1.0, min(1.0, rs_score / 5.0))

        # Determine trend
        if rs_norm > 0.2:
            trend = "leading"
        elif rs_norm < -0.2:
            trend = "lagging"
        else:
            trend = "neutral"

        return {
            "pmo": pmo_val,
            "pmo_signal": pmo_sig_val,
            "spy_pmo": spy_pmo_val,
            "spy_pmo_signal": spy_pmo_sig_val,
            "rs_score": rs_score,
            "rs_norm": rs_norm,
            "trend": trend,
        }

    except Exception:
        return _pmo_default()


def _pmo_default() -> dict:
    """Return default/empty PMO result on error."""
    return {
        "pmo": None,
        "pmo_signal": None,
        "spy_pmo": None,
        "spy_pmo_signal": None,
        "rs_score": None,
        "rs_norm": None,
        "trend": "neutral",
    }


def score_pmo_rs(data: dict) -> tuple:
    """
    Format PMO RS data into a label, status, and description.

    Args:
        data: dict returned from compute_pmo_rs()

    Returns:
        tuple of (label, status, text) where:
        - label: "PMO Relative Strength"
        - status: "good" if leading, "warning" if lagging, "neutral" otherwise
        - text: Human-readable description
    """
    label = "PMO Relative Strength"

    trend = data.get("trend", "neutral")
    if trend == "leading":
        status = "good"
    elif trend == "lagging":
        status = "warning"
    else:
        status = "neutral"

    rs_norm = data.get("rs_norm")
    rs_score = data.get("rs_score")

    if rs_norm is None:
        text = "No PMO data available."
    else:
        score_str = f"{rs_score:+.2f}"

        if trend == "leading":
            text = f"Stock PMO is outperforming SPY by {score_str} points — relative strength trending upward (momentum leader)."
        elif trend == "lagging":
            text = f"Stock PMO is underperforming SPY by {score_str} points — relative strength trending downward (momentum laggard)."
        else:
            text = f"Stock PMO is trading near SPY ({score_str} points diff) — momentum parity with broader market."

    return (label, status, text)
