"""Analyst recommendations and price targets from yfinance."""
import yfinance as yf
import pandas as pd


def get_analyst_data(ticker: str) -> dict:
    """
    Returns dict with:
    - recommendations: DataFrame or None (columns: period, strongBuy, buy, hold, sell, strongSell)
    - price_targets: dict with keys: mean, high, low, current, upside_pct
    - consensus: str like "Strong Buy" / "Buy" / "Hold" / "Sell"
    - analyst_count: int
    """
    t = yf.Ticker(ticker)
    result = {
        "recommendations": None,
        "price_targets": {},
        "consensus": "N/A",
        "analyst_count": 0,
    }

    # Price targets
    try:
        info = t.info
        current = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        mean_target = info.get("targetMeanPrice")
        high_target = info.get("targetHighPrice")
        low_target = info.get("targetLowPrice")
        num_analysts = info.get("numberOfAnalystOpinions", 0)
        recommendation = info.get("recommendationKey", "")  # e.g. "buy", "strong_buy", "hold"

        if mean_target and current:
            upside = (mean_target - current) / current * 100
        else:
            upside = None

        result["price_targets"] = {
            "current": round(current, 2) if current else None,
            "mean": round(mean_target, 2) if mean_target else None,
            "high": round(high_target, 2) if high_target else None,
            "low": round(low_target, 2) if low_target else None,
            "upside_pct": round(upside, 1) if upside is not None else None,
        }
        result["analyst_count"] = num_analysts or 0

        # Map recommendationKey to human label
        key_map = {
            "strong_buy": "Strong Buy",
            "buy": "Buy",
            "hold": "Hold",
            "underperform": "Underperform",
            "sell": "Sell",
        }
        result["consensus"] = key_map.get(recommendation.lower(), recommendation.title() if recommendation else "N/A")

    except Exception:
        pass

    # Recent recommendations summary (last 4 periods)
    try:
        recs = t.recommendations
        if recs is not None and not recs.empty:
            result["recommendations"] = recs.tail(4)
    except Exception:
        pass

    return result


def score_analyst(data: dict) -> tuple[str, str, str]:
    """
    Returns (label, status, text) signal for combined_score signals list.
    status: "good" / "warning" / "neutral"
    """
    targets = data.get("price_targets", {})
    consensus = data.get("consensus", "N/A")
    upside = targets.get("upside_pct")
    mean = targets.get("mean")
    count = data.get("analyst_count", 0)

    if count == 0 or mean is None:
        return ("Analyst", "neutral", "No analyst coverage found.")

    consensus_lower = consensus.lower()
    if "strong buy" in consensus_lower or (upside and upside > 20):
        status = "good"
    elif "sell" in consensus_lower or (upside and upside < -10):
        status = "warning"
    else:
        status = "neutral"

    upside_str = f"+{upside:.1f}%" if upside and upside >= 0 else f"{upside:.1f}%" if upside else "N/A"
    text = (f"{count} analysts say {consensus}. "
            f"Average target ${mean} ({upside_str} from current price).")
    return ("Analyst", status, text)
