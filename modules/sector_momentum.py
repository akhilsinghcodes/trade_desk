"""
Sector momentum analysis module.

Checks whether a ticker's sector ETF is in an uptrend or downtrend
by comparing SMA20 and SMA50 on 3 months of historical data.
"""

import yfinance as yf
import pandas as pd


SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}


def get_sector_momentum(sector: str) -> dict:
    """
    Analyze sector ETF momentum for a given sector.

    Args:
        sector: Sector name (e.g., "Technology")

    Returns:
        dict with keys:
        - sector: str
        - etf: str or None
        - etf_price: float or None (current price)
        - etf_sma20: float or None
        - etf_sma50: float or None
        - trend: "uptrend" | "downtrend" | "neutral" | "unknown"
        - etf_1mo_return: float or None (percentage)
    """
    result = {
        "sector": sector,
        "etf": None,
        "etf_price": None,
        "etf_sma20": None,
        "etf_sma50": None,
        "trend": "unknown",
        "etf_1mo_return": None,
    }

    # Check if sector exists in map
    if sector not in SECTOR_ETF_MAP:
        return result

    etf_ticker = SECTOR_ETF_MAP[sector]
    result["etf"] = etf_ticker

    try:
        # Download 3 months of daily OHLCV data
        df = yf.download(
            etf_ticker, period="3mo", auto_adjust=True, progress=False
        )

        # Handle MultiIndex columns from newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure we have enough data
        if df.empty or len(df) < 50:
            return result

        # Extract close price
        close = df["Close"]

        # Compute SMAs
        sma20 = close.rolling(window=20).mean().iloc[-1]
        sma50 = close.rolling(window=50).mean().iloc[-1]

        # Current price (last close)
        current_price = close.iloc[-1]

        # Determine trend
        if pd.notna(sma20) and pd.notna(sma50):
            if sma20 > sma50:
                trend = "uptrend"
            elif sma20 < sma50:
                trend = "downtrend"
            else:
                trend = "neutral"
        else:
            trend = "unknown"

        # Calculate 1-month return (approximately 21 trading days)
        if len(close) >= 21:
            price_21_days_ago = close.iloc[-21]
            one_mo_return = (current_price - price_21_days_ago) / price_21_days_ago * 100
        else:
            one_mo_return = None

        # Update result
        result.update({
            "etf_price": float(current_price),
            "etf_sma20": float(sma20) if pd.notna(sma20) else None,
            "etf_sma50": float(sma50) if pd.notna(sma50) else None,
            "trend": trend,
            "etf_1mo_return": float(one_mo_return) if one_mo_return is not None else None,
        })

    except Exception:
        # Return partial result with unknown trend on any error
        result["trend"] = "unknown"

    return result


def score_sector_momentum(data: dict) -> tuple[str, str, str]:
    """
    Score sector momentum and generate a readable summary.

    Args:
        data: dict returned from get_sector_momentum()

    Returns:
        tuple of (label, status, text):
        - label: "Sector Trend"
        - status: "good" if uptrend, "warning" if downtrend, "neutral" if unknown
        - text: Formatted description of sector momentum
    """
    label = "Sector Trend"

    sector = data.get("sector", "Unknown")
    etf = data.get("etf")
    trend = data.get("trend", "unknown")

    if trend == "unknown" or etf is None:
        status = "neutral"
        text = f"Sector momentum unknown for {sector}."
        return label, status, text

    sector_etf_str = f"{sector} ({etf})"

    if trend == "uptrend":
        status = "good"
        text = f"{sector_etf_str} in uptrend — SMA20 above SMA50. Sector tailwind."
    elif trend == "downtrend":
        status = "warning"
        text = f"{sector_etf_str} in downtrend — sector headwind. Be cautious."
    else:  # neutral
        status = "neutral"
        text = f"{sector_etf_str} in neutral trend — SMA20 near SMA50."

    return label, status, text
