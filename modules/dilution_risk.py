"""
Dilution Risk Module

Two signals:
1. Share Count Trend (buybacks vs dilution)
2. Earnings Proximity Risk
"""

import yfinance as yf
from typing import Optional


def get_dilution_risk(ticker: str) -> dict:
    """
    Analyze share count trend to detect buybacks vs dilution.

    Returns:
    - shares_current: int or None
    - shares_prior: int or None (1 year ago)
    - share_change_pct: float or None (negative = buyback, positive = dilution)
    - share_trend: "buyback" | "stable" | "diluting" | "unknown"
    - interpretation: str
    """
    try:
        t = yf.Ticker(ticker)
        bs = t.balance_sheet

        if bs is None or bs.empty:
            return {
                "shares_current": None,
                "shares_prior": None,
                "share_change_pct": None,
                "share_trend": "unknown",
                "interpretation": "Balance sheet data unavailable."
            }

        # Normalize index to lowercase with underscores
        bs.index = bs.index.str.lower().str.replace(" ", "_")

        # Look for share count row
        share_cols = ["ordinary_shares_number", "share_issued", "common_stock", "shares_outstanding"]
        share_row = None

        for col in share_cols:
            if col in bs.index:
                share_row = bs.loc[col]
                break

        if share_row is None or len(share_row) < 2:
            return {
                "shares_current": None,
                "shares_prior": None,
                "share_change_pct": None,
                "share_trend": "unknown",
                "interpretation": "Share count data not found in balance sheet."
            }

        # Get current and prior year shares
        shares_current_raw = share_row.iloc[0]
        shares_prior_raw = share_row.iloc[1]

        if shares_current_raw is None or shares_prior_raw is None or shares_current_raw <= 0 or shares_prior_raw <= 0:
            return {
                "shares_current": None,
                "shares_prior": None,
                "share_change_pct": None,
                "share_trend": "unknown",
                "interpretation": "Incomplete share count history."
            }

        shares_current = int(shares_current_raw)
        shares_prior = int(shares_prior_raw)

        # Calculate percentage change
        share_change_pct = ((shares_current - shares_prior) / shares_prior) * 100

        # Determine trend
        if share_change_pct < -1:
            share_trend = "buyback"
        elif share_change_pct > 2:
            share_trend = "diluting"
        else:
            share_trend = "stable"

        return {
            "shares_current": shares_current,
            "shares_prior": shares_prior,
            "share_change_pct": round(share_change_pct, 2),
            "share_trend": share_trend,
            "interpretation": f"Share change: {share_change_pct:.1f}% YoY"
        }

    except Exception as e:
        return {
            "shares_current": None,
            "shares_prior": None,
            "share_change_pct": None,
            "share_trend": "unknown",
            "interpretation": f"Error fetching dilution risk: {str(e)}"
        }


def score_dilution(data: dict) -> tuple[str, str, str]:
    """
    Score share count health.

    Args:
        data: dict from get_dilution_risk()

    Returns:
        (label, status, text) tuple
        label = "Share Count"
        status: "good" (buyback), "warning" (dilution >2%), "neutral" (stable/unknown)
        text: human-readable interpretation
    """
    label = "Share Count"
    share_trend = data.get("share_trend", "unknown")
    share_change_pct = data.get("share_change_pct")

    if share_trend == "unknown":
        return (label, "neutral", "Share count data unavailable.")

    if share_trend == "buyback" and share_change_pct is not None:
        abs_pct = abs(share_change_pct)
        text = f"Shares reduced {abs_pct:.1f}% YoY — active buybacks. Company returning capital."
        return (label, "good", text)

    if share_trend == "diluting" and share_change_pct is not None:
        text = f"Shares increased {share_change_pct:.1f}% YoY — significant dilution. Earnings per share hurt."
        return (label, "warning", text)

    # stable
    if share_change_pct is not None:
        abs_pct = abs(share_change_pct)
        text = f"Share count stable (±{abs_pct:.1f}% YoY). No major buybacks or dilution."
        return (label, "neutral", text)

    return (label, "neutral", "Share count unchanged.")


def get_earnings_proximity(earnings_info: dict) -> dict:
    """
    Assess earnings announcement proximity risk.

    Args:
        earnings_info: dict with keys:
            - "date": str "YYYY-MM-DD" or None
            - "days_away": int or None

    Returns:
        dict with:
        - days_away: int or None
        - risk_level: "high" (<=14 days) | "medium" (15-30 days) | "low" (>30 days) | "unknown"
        - within_2_weeks: bool
    """
    days_away = earnings_info.get("days_away")

    if days_away is None:
        return {
            "days_away": None,
            "risk_level": "unknown",
            "within_2_weeks": False
        }

    within_2_weeks = days_away <= 14

    if days_away <= 14:
        risk_level = "high"
    elif days_away <= 30:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "days_away": days_away,
        "risk_level": risk_level,
        "within_2_weeks": within_2_weeks
    }


def score_earnings_proximity(data: dict) -> tuple[str, str, str]:
    """
    Score earnings announcement proximity risk.

    Args:
        data: dict from get_earnings_proximity()

    Returns:
        (label, status, text) tuple
        label = "Earnings Risk"
        status: "warning" (<=14 days), "neutral" (15-30 days), "good" (>30 days or unknown)
        text: human-readable warning or context
    """
    label = "Earnings Risk"
    days_away = data.get("days_away")
    risk_level = data.get("risk_level", "unknown")

    if risk_level == "unknown":
        return (label, "good", "Earnings date unknown.")

    if risk_level == "high" and days_away is not None:
        text = f"Earnings in {days_away} days — elevated volatility expected. Size positions carefully."
        return (label, "warning", text)

    if risk_level == "medium" and days_away is not None:
        text = f"Earnings in {days_away} days — moderate event risk."
        return (label, "neutral", text)

    # low risk (>30 days)
    if days_away is not None:
        text = f"Earnings in {days_away} days — no immediate event risk."
        return (label, "good", text)

    return (label, "good", "No near-term earnings risk.")
