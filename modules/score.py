"""Combined BUY/HOLD/SELL score from technical + fundamental + sentiment signals."""

from typing import Dict, List, Tuple, Any
import statistics


def signal_to_float(label: str) -> float:
    """
    Convert a signal label to a continuous float value in [-1.0, 1.0].

    Maps:
    - "good", "bullish", "beat" → +1.0
    - "neutral", "mixed" → 0.0
    - "warning", "bearish", "miss" → -1.0

    Case-insensitive, partial matching.

    Args:
        label: Signal status label

    Returns:
        Float value in [-1.0, 1.0]
    """
    if not isinstance(label, str):
        return 0.0

    label_lower = label.lower().strip()

    # Bullish signals
    if any(x in label_lower for x in ["good", "bullish", "beat", "positive"]):
        return 1.0

    # Bearish signals
    if any(x in label_lower for x in ["warning", "bearish", "miss", "negative"]):
        return -1.0

    # Neutral signals
    return 0.0


def combined_score(
    tech_signals: List[Tuple[str, str, str]],
    fund_signals: List[Tuple[str, str, str]],
    news_articles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a weighted composite score with confidence and conflict detection.

    Args:
        tech_signals: List of (label, status, text) technical signal tuples
        fund_signals: List of (label, status, text) fundamental signal tuples
        news_articles: List of article dicts with "sentiment" key

    Returns:
        Dict with keys:
        - score: float [-1, 1] weighted composite
        - verdict: "BUY" / "HOLD" / "SELL"
        - confidence: float [0, 1]
        - conflicted: bool (True if tech/fund signals conflict)
        - component_scores: dict with tech/fund/sent scores
        - breakdown: dict with tech/fund/sent (backward compat)
        - color: string (backward compat)
        - reason: string (backward compat)
        - total: float (backward compat, same as score)
    """
    from modules.config import get_weights, get_thresholds

    weights = get_weights()
    thresholds = get_thresholds()

    # --- Convert signals to continuous floats ---
    tech_floats = [signal_to_float(status) for _, status, _ in tech_signals]
    fund_floats = [signal_to_float(status) for _, status, _ in fund_signals]

    # --- Technical score [-1, 1] ---
    if tech_floats:
        tech_score = sum(tech_floats) / len(tech_floats)
    else:
        tech_score = 0.0
    tech_score = max(-1.0, min(1.0, tech_score))

    # --- Fundamental score [-1, 1] ---
    if fund_floats:
        fund_score = sum(fund_floats) / len(fund_floats)
    else:
        fund_score = 0.0
    fund_score = max(-1.0, min(1.0, fund_score))

    # --- Sentiment score [-1, 1] ---
    sent_floats = []
    valid_articles = [a for a in news_articles if a.get("sentiment", "unknown") != "unknown"]
    if valid_articles:
        sent_floats = [signal_to_float(a["sentiment"]) for a in valid_articles]
        sent_score = sum(sent_floats) / len(sent_floats)
    else:
        sent_score = 0.0
    sent_score = max(-1.0, min(1.0, sent_score))

    # --- Weighted composite score ---
    w_tech = weights.get("technical", 0.40)
    w_fund = weights.get("fundamental", 0.35)
    w_sent = weights.get("sentiment", 0.25)

    composite_score = (tech_score * w_tech) + (fund_score * w_fund) + (sent_score * w_sent)
    composite_score = max(-1.0, min(1.0, composite_score))

    # --- Signal conflict detection ---
    conflicted = (
        tech_score * fund_score < 0 and
        abs(tech_score) > 0.3 and
        abs(fund_score) > 0.3
    )

    # --- Confidence metric: 1 - stddev(all_signal_floats) ---
    all_floats = tech_floats + fund_floats + sent_floats
    if len(all_floats) > 1:
        stddev = statistics.stdev(all_floats)
        confidence = max(0.0, 1.0 - stddev)
    elif len(all_floats) == 1:
        confidence = 1.0  # Single signal = high confidence
    else:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    # --- Verdict based on configurable thresholds ---
    bullish_threshold = thresholds.get("bullish", 0.25)
    bearish_threshold = thresholds.get("bearish", -0.25)

    if composite_score >= bullish_threshold:
        verdict = "BUY"
        color = "green"
        reason = "Majority of technical, fundamental, and sentiment signals are positive."
    elif composite_score <= bearish_threshold:
        verdict = "SELL / AVOID"
        color = "red"
        reason = "Majority of signals are negative. High risk of further downside."
    else:
        verdict = "HOLD / WATCH"
        color = "orange"
        reason = "Signals are mixed. No strong edge in either direction right now."

    # --- Return comprehensive result dict ---
    return {
        # New fields (v2)
        "score": composite_score,
        "conflicted": conflicted,
        "component_scores": {
            "technical": round(tech_score, 3),
            "fundamental": round(fund_score, 3),
            "sentiment": round(sent_score, 3),
        },
        # Backward compatibility fields
        "verdict": verdict,
        "color": color,
        "reason": reason,
        "confidence": confidence,
        "total": composite_score,
        "breakdown": {
            "technical": round(tech_score, 2),
            "fundamental": round(fund_score, 2),
            "sentiment": round(sent_score, 2),
        },
    }
