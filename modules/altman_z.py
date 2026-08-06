"""
Altman Z-Score module for bankruptcy risk prediction.

Altman Z-Score (public company formula):
Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

Where:
- X1 = Working Capital / Total Assets
- X2 = Retained Earnings / Total Assets
- X3 = EBIT / Total Assets
- X4 = Market Cap / Total Liabilities
- X5 = Revenue / Total Assets

Zones:
- Z > 2.99: Safe (low bankruptcy risk)
- 1.81-2.99: Grey (moderate risk)
- Z < 1.81: Distress (high bankruptcy risk)
"""

import yfinance as yf
from typing import Tuple


def _get(series, *keys):
    """
    Helper: substring match in index, return value of first match.
    Return None if not found.

    Args:
        series: pandas Series with string index
        *keys: list of substring keys to search for

    Returns:
        Value of first matching index entry, or None if not found
    """
    if series is None or len(series) == 0:
        return None

    try:
        # Normalize index to lowercase for matching
        index_lower = series.index.astype(str).str.lower()

        for key in keys:
            key_lower = key.lower()
            matches = index_lower.str.contains(key_lower, regex=False, na=False)
            if matches.any():
                val = series[matches].iloc[0]
                return float(val) if val is not None else None
    except Exception:
        pass

    return None


def get_altman_z(ticker: str) -> dict:
    """
    Calculate Altman Z-Score for bankruptcy risk prediction (public company formula).

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        dict with keys:
        - z_score: float or None
        - zone: "safe" | "grey" | "distress" | "unknown"
        - components: dict {x1, x2, x3, x4, x5} — the 5 ratios
        - interpretation: str describing the result
    """
    try:
        t = yf.Ticker(ticker)

        # Fetch financial statements
        bs = t.balance_sheet
        inc = t.income_stmt

        if bs is None or bs.empty or inc is None or inc.empty:
            return {
                "z_score": None,
                "zone": "unknown",
                "components": {},
                "interpretation": "Unable to fetch financial data."
            }

        # Normalize index: lowercase and replace spaces with underscores
        bs.index = bs.index.str.lower().str.replace(" ", "_")
        inc.index = inc.index.str.lower().str.replace(" ", "_")

        # Use most recent column (column 0)
        bs_latest = bs.iloc[:, 0]
        inc_latest = inc.iloc[:, 0]

        # ===== Extract X1: Working Capital / Total Assets =====
        current_assets = _get(bs_latest, "current_assets", "total_current_assets")
        current_liabilities = _get(bs_latest, "current_liabilities", "total_current_liabilities")
        total_assets = _get(bs_latest, "total_assets")

        x1 = None
        if (current_assets is not None and
            current_liabilities is not None and
            total_assets is not None and
            total_assets != 0):
            working_capital = current_assets - current_liabilities
            x1 = working_capital / total_assets

        # ===== Extract X2: Retained Earnings / Total Assets =====
        retained_earnings = _get(bs_latest, "retained_earnings", "accumulated_deficit")

        x2 = None
        if retained_earnings is not None and total_assets is not None and total_assets != 0:
            x2 = retained_earnings / total_assets

        # ===== Extract X3: EBIT / Total Assets =====
        ebit = _get(inc_latest, "ebit", "operating_income")
        if ebit is None:
            ebit = _get(inc_latest, "ebitda")

        x3 = None
        if ebit is not None and total_assets is not None and total_assets != 0:
            x3 = ebit / total_assets

        # ===== Extract X4: Market Cap / Total Liabilities =====
        market_cap = t.info.get("marketCap")
        total_liabilities = _get(bs_latest, "total_liabilities", "total_liabilities_net_minority_interest")

        x4 = None
        if market_cap is not None and total_liabilities is not None and total_liabilities != 0:
            x4 = market_cap / total_liabilities

        # ===== Extract X5: Revenue / Total Assets =====
        revenue = _get(inc_latest, "total_revenue", "revenue")

        x5 = None
        if revenue is not None and total_assets is not None and total_assets != 0:
            x5 = revenue / total_assets

        # ===== Calculate Z-Score =====
        components_available = [x for x in [x1, x2, x3, x4, x5] if x is not None]

        if not components_available:
            return {
                "z_score": None,
                "zone": "unknown",
                "components": {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5},
                "interpretation": "Insufficient data to calculate Z-Score."
            }

        # Weighted sum (only include available components)
        z_score = 0.0
        if x1 is not None:
            z_score += 1.2 * x1
        if x2 is not None:
            z_score += 1.4 * x2
        if x3 is not None:
            z_score += 3.3 * x3
        if x4 is not None:
            z_score += 0.6 * x4
        if x5 is not None:
            z_score += 1.0 * x5

        # ===== Determine Zone =====
        if z_score > 2.99:
            zone = "safe"
        elif z_score >= 1.81:
            zone = "grey"
        else:
            zone = "distress"

        # ===== Generate Interpretation =====
        if zone == "safe":
            interpretation = f"Z-Score {z_score:.2f} — safe zone. Low bankruptcy risk."
        elif zone == "grey":
            interpretation = f"Z-Score {z_score:.2f} — grey zone. Moderate bankruptcy risk."
        else:
            interpretation = f"Z-Score {z_score:.2f} — distress zone. High bankruptcy risk."

        if len(components_available) < 5:
            interpretation += f" (Based on {len(components_available)}/5 components)"

        return {
            "z_score": z_score,
            "zone": zone,
            "components": {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5},
            "interpretation": interpretation
        }

    except Exception as e:
        return {
            "z_score": None,
            "zone": "unknown",
            "components": {},
            "interpretation": f"Error calculating Z-Score: {str(e)}"
        }


def score_altman_z(data: dict) -> Tuple[str, str, str]:
    """
    Format Altman Z-Score result for dashboard display.

    Args:
        data: dict returned from get_altman_z()

    Returns:
        tuple of (label, status, text)
        - label: "Altman Z-Score"
        - status: "good" if safe (>2.99), "warning" if distress (<1.81), "neutral" if grey
        - text: formatted interpretation string
    """
    z_score = data.get("z_score")
    zone = data.get("zone")
    interpretation = data.get("interpretation", "")

    label = "Altman Z-Score"

    # Determine status based on zone
    if z_score is None or zone == "unknown":
        status = "neutral"
    elif zone == "safe":
        status = "good"
    elif zone == "distress":
        status = "warning"
    else:  # grey zone
        status = "neutral"

    text = interpretation

    return (label, status, text)
