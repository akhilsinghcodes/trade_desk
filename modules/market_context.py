"""
Market context module for fetching VIX and computing 52-week rank metrics.
"""

import yfinance as yf
from typing import Optional


def get_market_context(ticker: str) -> dict:
    """
    Fetch market context data including VIX and 52-week rank for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        dict with keys:
        - vix: float or None (current VIX level)
        - vix_regime: "high_fear" | "neutral" | "complacency"
                      (>25 / 15-25 / <15)
        - week52_high: float (52-week high price)
        - week52_low: float (52-week low price)
        - week52_rank: float (0.0-1.0, where 1.0 = at 52W high)
        - week52_pct: float (0-100, human readable percentage)
    """
    defaults = {
        "vix": None,
        "vix_regime": "neutral",
        "week52_high": 0.0,
        "week52_low": 0.0,
        "week52_rank": 0.0,
        "week52_pct": 0.0,
    }

    try:
        # Fetch VIX data
        vix_ticker = yf.Ticker("^VIX")
        vix = vix_ticker.info.get("regularMarketPrice")

        # Determine VIX regime
        if vix is not None:
            if vix > 25:
                vix_regime = "high_fear"
            elif vix >= 15:
                vix_regime = "neutral"
            else:
                vix_regime = "complacency"
        else:
            vix_regime = "neutral"

        # Fetch ticker info for 52-week data
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        week52_high = info.get("fiftyTwoWeekHigh")
        week52_low = info.get("fiftyTwoWeekLow")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        # Calculate 52-week rank
        if (
            week52_high is not None
            and week52_low is not None
            and current_price is not None
            and week52_high > week52_low
        ):
            week52_rank = (current_price - week52_low) / (week52_high - week52_low)
            # Clamp to [0, 1]
            week52_rank = max(0.0, min(1.0, week52_rank))
            week52_pct = week52_rank * 100
        else:
            week52_rank = 0.0
            week52_pct = 0.0

        return {
            "vix": vix,
            "vix_regime": vix_regime,
            "week52_high": week52_high or 0.0,
            "week52_low": week52_low or 0.0,
            "week52_rank": week52_rank,
            "week52_pct": week52_pct,
        }

    except Exception as e:
        print(f"Error fetching market context for {ticker}: {e}")
        return defaults


def score_market_context(data: dict) -> tuple[str, str, str]:
    """
    Score market context and return signal for display in signals list.

    Args:
        data: dict returned from get_market_context()

    Returns:
        tuple of (label, status, text)
        - label: "Market Context"
        - status: "good" (high fear = contrarian opportunity),
                  "warning" (complacency = elevated risk),
                  "neutral" (normal conditions)
        - text: Human-readable description including VIX regime and 52W rank
    """
    label = "Market Context"

    vix = data.get("vix")
    vix_regime = data.get("vix_regime", "neutral")
    week52_rank = data.get("week52_rank", 0.0)
    week52_pct = data.get("week52_pct", 0.0)

    # Determine status and VIX text based on regime
    if vix_regime == "high_fear":
        status = "good"
        vix_text = f"VIX {vix:.1f} — high fear (contrarian opportunity)"
    elif vix_regime == "complacency":
        status = "warning"
        vix_text = f"VIX {vix:.1f} — complacency (elevated risk)"
    else:  # neutral
        status = "neutral"
        vix_text = f"VIX {vix:.1f}" if vix is not None else "VIX unavailable"

    # Add 52-week rank interpretation
    if week52_rank > 0.80:
        rank_text = (
            f"at {week52_pct:.0f}% of 52W range — "
            "near 52W high, momentum strong but watch for resistance"
        )
    elif week52_rank < 0.20:
        rank_text = (
            f"at {week52_pct:.0f}% of 52W range — "
            "near 52W low, potential value or falling knife"
        )
    else:
        rank_text = f"at {week52_pct:.0f}% of 52W range"

    text = f"{vix_text}. {rank_text}"

    return (label, status, text)
