"""Combined BUY/HOLD/SELL score from technical + fundamental + sentiment signals."""


def combined_score(
    tech_signals: list,
    fund_signals: list,
    news_articles: list,
) -> dict:
    # --- Technical score (-1 to +1 per signal) ---
    tech_map = {"good": 1, "neutral": 0, "warning": -1}
    tech_raw = sum(tech_map.get(s, 0) for _, s, _ in tech_signals)
    tech_max = len(tech_signals) if tech_signals else 1
    tech_score = tech_raw / tech_max  # normalised -1 to +1

    # --- Fundamental score ---
    fund_raw = sum(tech_map.get(s, 0) for _, s, _ in fund_signals)
    fund_max = len(fund_signals) if fund_signals else 1
    fund_score = fund_raw / fund_max

    # --- Sentiment score ---
    sent_map = {"positive": 1, "negative": -1, "neutral": 0}
    valid = [a for a in news_articles if a.get("sentiment", "unknown") != "unknown"]
    if valid:
        sent_raw = sum(sent_map.get(a["sentiment"].lower(), 0) for a in valid)
        sent_score = sent_raw / len(valid)
    else:
        sent_score = 0.0

    # --- Weighted total (tech 40%, fund 35%, sentiment 25%) ---
    total = (tech_score * 0.40) + (fund_score * 0.35) + (sent_score * 0.25)

    # --- Verdict ---
    if total >= 0.25:
        verdict = "BUY"
        color = "green"
        reason = "Majority of technical, fundamental, and sentiment signals are positive."
    elif total <= -0.25:
        verdict = "SELL / AVOID"
        color = "red"
        reason = "Majority of signals are negative. High risk of further downside."
    else:
        verdict = "HOLD / WATCH"
        color = "orange"
        reason = "Signals are mixed. No strong edge in either direction right now."

    confidence = min(abs(total) * 2, 1.0)  # 0-100%

    return {
        "verdict": verdict,
        "color": color,
        "reason": reason,
        "confidence": confidence,
        "total": total,
        "breakdown": {
            "technical": round(tech_score, 2),
            "fundamental": round(fund_score, 2),
            "sentiment": round(sent_score, 2),
        },
    }
