"""Auto-suggest price alerts based on analysis data."""


def suggest_alerts(
    ticker: str,
    current_price: float,
    swing_levels: dict,
    pivot_points: dict,
    analyst_data: dict,
    earnings_info: dict,
) -> list[dict]:
    """
    Returns a list of suggested alerts with:
    - ticker: str
    - target_price: float
    - direction: "above" | "below"
    - reason: str  (plain English)
    - priority: "high" | "medium" | "low"

    Sources:
    1. Analyst mean/high/low price targets
    2. Key support levels (alert if price drops below)
    3. Key resistance levels (alert if price breaks above)
    4. Earnings date warning (if within 14 days)
    """
    suggestions = []

    # ── Analyst targets ──
    targets = analyst_data.get("price_targets", {})
    mean_target = targets.get("mean")
    high_target = targets.get("high")
    low_target = targets.get("low")

    if mean_target and mean_target > current_price * 1.02:
        suggestions.append({
            "ticker": ticker,
            "target_price": mean_target,
            "direction": "above",
            "reason": f"Analyst avg target ${mean_target} — alert when price reaches consensus",
            "priority": "high",
        })

    if high_target and high_target > current_price * 1.05:
        suggestions.append({
            "ticker": ticker,
            "target_price": high_target,
            "direction": "above",
            "reason": f"Analyst high target ${high_target} — most bullish scenario",
            "priority": "low",
        })

    if low_target and low_target < current_price * 0.98:
        suggestions.append({
            "ticker": ticker,
            "target_price": low_target,
            "direction": "below",
            "reason": f"Analyst low target ${low_target} — alert if price drops to most bearish estimate",
            "priority": "medium",
        })

    # ── Support levels (alert if breaks below) ──
    supports = swing_levels.get("support", [])
    for i, s in enumerate(supports[:2]):  # top 2 support levels
        if s < current_price * 0.99:
            priority = "high" if i == 0 else "medium"
            suggestions.append({
                "ticker": ticker,
                "target_price": s,
                "direction": "below",
                "reason": f"Key support ${s} — price breaking below this is a bearish signal",
                "priority": priority,
            })

    # ── Resistance levels (alert if breaks above) ──
    resistances = swing_levels.get("resistance", [])
    for i, r in enumerate(resistances[:2]):  # top 2 resistance levels
        if r > current_price * 1.01:
            priority = "high" if i == 0 else "medium"
            suggestions.append({
                "ticker": ticker,
                "target_price": r,
                "direction": "above",
                "reason": f"Key resistance ${r} — breaking above signals potential breakout",
                "priority": priority,
            })

    # ── Pivot point ──
    pp = pivot_points.get("PP")
    if pp:
        if pp < current_price * 0.98:
            suggestions.append({
                "ticker": ticker,
                "target_price": pp,
                "direction": "below",
                "reason": f"Pivot point ${pp} — dropping below pivot is bearish",
                "priority": "medium",
            })
        elif pp > current_price * 1.02:
            suggestions.append({
                "ticker": ticker,
                "target_price": pp,
                "direction": "above",
                "reason": f"Pivot point ${pp} — reclaiming pivot is bullish",
                "priority": "medium",
            })

    # Deduplicate by target_price (keep highest priority)
    seen = {}
    for s in suggestions:
        key = (s["direction"], s["target_price"])
        if key not in seen:
            seen[key] = s
        else:
            # Keep higher priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            if priority_order[s["priority"]] < priority_order[seen[key]["priority"]]:
                seen[key] = s

    # Sort: high priority first, then by proximity to current price
    result = sorted(
        seen.values(),
        key=lambda x: ({"high": 0, "medium": 1, "low": 2}[x["priority"]],
                       abs(x["target_price"] - current_price))
    )

    return result[:6]  # max 6 suggestions
