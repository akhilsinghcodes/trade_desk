"""Momentum signals: Coppock Curve and Ridge Regression slope."""
import pandas as pd
import numpy as np


def compute_coppock(df: pd.DataFrame) -> dict:
    """
    Compute Coppock Curve indicator.

    Coppock Curve uses rate of change and weighted moving average.
    Signal: bullish when coppock > 0 and rising, bearish when < 0 and falling.

    Args:
        df: pandas DataFrame with 'close' column

    Returns:
        dict with keys:
        - value: float, current Coppock Curve value
        - signal: float, normalized signal (-1.0 to 1.0)
        - trend: str, "bullish", "bearish", or "neutral"
    """
    try:
        close = df["close"].astype(float)

        # Need at least 15 bars for ROC14
        if len(close) < 15:
            return {"value": 0.0, "signal": 0.0, "trend": "neutral"}

        # Calculate ROC14 and ROC11
        roc14 = (close / close.shift(14) - 1) * 100
        roc11 = (close / close.shift(11) - 1) * 100

        # Sum for WMA input
        roc_sum = roc14 + roc11

        # WMA with period 10: weighted moving average
        # Weight: (10, 9, 8, 7, 6, 5, 4, 3, 2, 1) for last 10 bars
        def wma_10(series):
            if len(series) < 10:
                return np.nan
            weights = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
            return np.sum(series.iloc[-10:].values * weights) / np.sum(weights)

        # Apply rolling WMA
        coppock = roc_sum.rolling(window=10).apply(
            lambda x: wma_10(pd.Series(x)),
            raw=False
        )

        if len(coppock) < 1 or pd.isna(coppock.iloc[-1]):
            return {"value": 0.0, "signal": 0.0, "trend": "neutral"}

        current_val = float(coppock.iloc[-1])

        # Determine trend — value sign is primary; direction is secondary confirmation
        if current_val > 0:
            trend = "bullish"
            signal = float(np.clip(current_val / 50.0, 0.3, 1.0))
        elif current_val < 0:
            trend = "bearish"
            signal = float(np.clip(current_val / -50.0, 0.3, 1.0)) * -1
        else:
            trend = "neutral"
            signal = 0.0

        return {
            "value": current_val,
            "signal": signal,
            "trend": trend
        }

    except Exception:
        return {"value": 0.0, "signal": 0.0, "trend": "neutral"}


def compute_ridge_slope(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Compute Ridge Regression price slope indicator.

    Uses a linear regression slope over the last N bars (window).
    Higher slope = stronger uptrend, lower slope = stronger downtrend.

    Args:
        df: pandas DataFrame with 'close' column
        window: lookback period (default 20)

    Returns:
        dict with keys:
        - slope: float, raw slope value
        - signal: float, normalized signal (-1.0 to 1.0)
        - interpretation: str, "uptrend", "downtrend", or "neutral"
    """
    try:
        close = df["close"].astype(float)

        if len(close) < window:
            return {"slope": 0.0, "signal": 0.0, "interpretation": "neutral"}

        # Get last window bars
        y = close.tail(window).values
        X = np.arange(len(y)).reshape(-1, 1)

        # Normalize y
        y_mean = y.mean()
        y_std = y.std()

        # Avoid division by zero
        if y_std < 1e-9:
            return {"slope": 0.0, "signal": 0.0, "interpretation": "neutral"}

        y_norm = (y - y_mean) / (y_std + 1e-9)

        # Use numpy polyfit as fallback (no sklearn dependency)
        try:
            coeffs = np.polyfit(X.flatten(), y_norm, 1)
            slope = float(coeffs[0])
        except Exception:
            return {"slope": 0.0, "signal": 0.0, "interpretation": "neutral"}

        # Normalize slope to [-1, 1]
        # Assume typical slope range is ~[-0.2, 0.2] in normalized units
        signal = float(np.clip(slope / 0.2, -1.0, 1.0))

        # Determine interpretation
        if signal > 0.05:
            interpretation = "uptrend"
        elif signal < -0.05:
            interpretation = "downtrend"
        else:
            interpretation = "neutral"

        return {
            "slope": slope,
            "signal": signal,
            "interpretation": interpretation
        }

    except Exception:
        return {"slope": 0.0, "signal": 0.0, "interpretation": "neutral"}
