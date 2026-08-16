"""Short interest data from yfinance — squeeze potential signal."""
import yfinance as yf


def get_short_interest(ticker: str) -> dict:
    """
    Returns dict:
    - short_pct_float: float (% of float that is shorted, e.g. 5.2 means 5.2%)
    - short_ratio: float (days to cover)
    - shares_short: int
    - squeeze_potential: str ("high" / "moderate" / "low" / "unknown")
    """
    result = {
        "short_pct_float": None,
        "short_ratio": None,
        "shares_short": None,
        "squeeze_potential": "unknown",
    }

    try:
        info = yf.Ticker(ticker).info

        raw_pct = info.get("shortPercentOfFloat")
        if raw_pct is not None:
            # yfinance returns decimal (0.052 = 5.2%)
            result["short_pct_float"] = round(float(raw_pct) * 100, 2)

        short_ratio = info.get("shortRatio")
        if short_ratio is not None:
            result["short_ratio"] = round(float(short_ratio), 1)

        shares_short = info.get("sharesShort")
        if shares_short is not None:
            result["shares_short"] = int(shares_short)

        pct = result["short_pct_float"]
        if pct is not None:
            if pct >= 20:
                result["squeeze_potential"] = "high"
            elif pct >= 10:
                result["squeeze_potential"] = "moderate"
            else:
                result["squeeze_potential"] = "low"

    except Exception:
        pass

    return result


def score_short_interest(data: dict) -> tuple[str, str, str]:
    """Returns (label, status, text) for signals list."""
    pct = data.get("short_pct_float")
    ratio = data.get("short_ratio")
    squeeze = data.get("squeeze_potential", "unknown")

    if pct is None:
        return ("Short Interest", "neutral", "No short interest data available.")

    ratio_str = f" ({ratio} days to cover)" if ratio else ""
    pct_str = f"{pct:.1f}% of float shorted{ratio_str}."

    if squeeze == "high":
        return ("Short Interest", "warning",
                f"{pct_str} HIGH short interest — strong bearish bet by market. "
                f"Risk of short squeeze if price rises sharply.")
    elif squeeze == "moderate":
        return ("Short Interest", "neutral",
                f"{pct_str} Moderate short interest — some bearish pressure but not extreme.")
    else:
        return ("Short Interest", "neutral",
                f"{pct_str} Low short interest — market not heavily betting against this stock.")
