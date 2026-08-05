"""Technical indicators via FinTA."""
import pandas as pd
from finta import TA


def add_common_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, BB, SMA20, SMA50 to OHLCV dataframe."""
    out = df.copy()
    # FinTA expects columns: open, high, low, close, volume
    ohlcv = df.rename(columns=str.upper)

    out["sma20"] = TA.SMA(ohlcv, 20)
    out["sma50"] = TA.SMA(ohlcv, 50)
    out["rsi"] = TA.RSI(ohlcv)

    macd = TA.MACD(ohlcv)
    out["macd"] = macd["MACD"]
    out["macd_signal"] = macd["SIGNAL"]

    bb = TA.BBANDS(ohlcv)
    out["bb_upper"] = bb["BB_UPPER"]
    out["bb_lower"] = bb["BB_LOWER"]
    out["bb_mid"] = bb["BB_MIDDLE"]

    return out
