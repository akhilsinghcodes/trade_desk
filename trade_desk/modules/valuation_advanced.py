"""Advanced valuation metrics: FCF yield and EV/EBITDA analysis."""

import yfinance as yf


def get_advanced_valuation(ticker: str) -> dict:
    """
    Fetch advanced valuation metrics from Yahoo Finance.

    Returns:
    - fcf_yield: float or None  (FCF / market cap, as percentage e.g. 3.5 means 3.5%)
    - ev_ebitda: float or None
    - fcf_per_share: float or None
    - enterprise_value: float or None (raw $)
    - ebitda: float or None (raw $)
    - fcf_interpretation: str  ("cheap" | "fair" | "expensive" | "unknown")
    - ev_ebitda_interpretation: str  ("cheap" | "fair" | "expensive" | "unknown")
    """
    result = {
        "fcf_yield": None,
        "ev_ebitda": None,
        "fcf_per_share": None,
        "enterprise_value": None,
        "ebitda": None,
        "fcf_interpretation": "unknown",
        "ev_ebitda_interpretation": "unknown",
    }

    try:
        info = yf.Ticker(ticker).info

        # Extract raw values
        free_cashflow = info.get("freeCashflow")
        market_cap = info.get("marketCap")
        enterprise_value = info.get("enterpriseValue")
        ebitda = info.get("ebitda")
        shares_outstanding = info.get("sharesOutstanding")

        # Store enterprise value and ebitda
        result["enterprise_value"] = enterprise_value
        result["ebitda"] = ebitda

        # Calculate FCF yield
        if free_cashflow and market_cap and market_cap != 0:
            result["fcf_yield"] = (free_cashflow / market_cap) * 100
            # Interpret FCF yield: >5% = cheap, 2-5% = fair, <2% = expensive
            if result["fcf_yield"] > 5:
                result["fcf_interpretation"] = "cheap"
            elif 2 <= result["fcf_yield"] <= 5:
                result["fcf_interpretation"] = "fair"
            else:
                result["fcf_interpretation"] = "expensive"

        # Calculate FCF per share
        if free_cashflow and shares_outstanding and shares_outstanding != 0:
            result["fcf_per_share"] = free_cashflow / shares_outstanding

        # Calculate EV/EBITDA
        if enterprise_value and ebitda and ebitda != 0:
            result["ev_ebitda"] = enterprise_value / ebitda
            # Interpret EV/EBITDA: <10 = cheap, 10-20 = fair, >20 = expensive
            if result["ev_ebitda"] < 10:
                result["ev_ebitda_interpretation"] = "cheap"
            elif 10 <= result["ev_ebitda"] <= 20:
                result["ev_ebitda_interpretation"] = "fair"
            else:
                result["ev_ebitda_interpretation"] = "expensive"

    except Exception as e:
        print(f"Error fetching advanced valuation for {ticker}: {e}")

    return result


def score_valuation_advanced(data: dict) -> tuple[str, str, str]:
    """
    Score advanced valuation metrics and return a human-readable assessment.

    Args:
        data: dict returned by get_advanced_valuation()

    Returns:
        tuple[str, str, str]: (label, status, text)
        - label: "Advanced Valuation"
        - status: "good" | "neutral" | "warning" | "unknown"
        - text: human-readable explanation
    """
    label = "Advanced Valuation"

    fcf_interp = data.get("fcf_interpretation", "unknown")
    ev_interp = data.get("ev_ebitda_interpretation", "unknown")

    fcf_yield = data.get("fcf_yield")
    ev_ebitda = data.get("ev_ebitda")

    # Build status based on combination of metrics
    if fcf_interp == "unknown" and ev_interp == "unknown":
        status = "unknown"
        text = "Insufficient data for advanced valuation."
    elif fcf_interp == "cheap" and ev_interp == "cheap":
        status = "good"
        text = f"FCF yield {fcf_yield:.1f}% (cheap) · EV/EBITDA {ev_ebitda:.1f} (cheap valuation)."
    elif fcf_interp == "expensive" and ev_interp == "expensive":
        status = "warning"
        text = f"FCF yield {fcf_yield:.1f}% (expensive) · EV/EBITDA {ev_ebitda:.1f} (expensive valuation)."
    else:
        # Mixed signals: fair valuation, needs closer look
        status = "neutral"
        fcf_str = f"FCF yield {fcf_yield:.1f}% ({fcf_interp})" if fcf_yield is not None else "FCF yield N/A"
        ev_str = f"EV/EBITDA {ev_ebitda:.1f} ({ev_interp})" if ev_ebitda is not None else "EV/EBITDA N/A"
        text = f"{fcf_str} · {ev_str} (mixed signals)."

    return (label, status, text)
