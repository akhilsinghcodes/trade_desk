"""
Smart trade strategy: limit entry, take profit, stop loss, exit conditions.
Uses all available signals — technical, fundamental, momentum, analyst targets,
support/resistance levels, earnings proximity, setup detector, and conviction score.
"""
from typing import Optional


def get_smart_trade_strategy(
    current_price: float,
    atr: float,
    confidence: float,          # 0.0–1.0
    verdict: str,               # "BUY" | "HOLD / WATCH" | "SELL / AVOID"
    breakdown: dict,            # {"technical": float, "fundamental": float, "sentiment": float}
    support_levels: list,       # sorted desc (closest first), prices below current
    resistance_levels: list,    # sorted asc (closest first), prices above current
    pivot: Optional[float],
    analyst_target: Optional[float],
    analyst_upside_pct: Optional[float],
    week52_high: Optional[float],
    week52_low: Optional[float],
    week52_rank: Optional[float],   # 0–1, where 1 = at 52W high
    piotroski_score: int,           # 0–9
    altman_zone: str,               # "safe" | "grey" | "distress" | "unknown"
    momentum_trend: str,            # "strong_up" | "up" | "neutral" | "down" | "strong_down"
    ret_1yr: Optional[float],       # % return over 1yr
    short_pct: Optional[float],     # short interest % of float
    earnings_days_away: Optional[int],
    sector_trend: str,              # "uptrend" | "downtrend" | "unknown"
    vix: Optional[float] = None,    # VIX level for position sizing
    setup_data: Optional[dict] = None,  # from modules.setup_detector.analyze_setup
) -> dict:
    """
    Returns dict with:
    - limit_entry: float — suggested limit buy price
    - limit_entry_reason: str
    - take_profit_1: float — conservative target
    - take_profit_2: float — extended target
    - take_profit_reason: str
    - stop_loss: float
    - stop_loss_reason: str
    - risk_reward: float
    - exit_conditions: list[str] — plain-English exit triggers
    - time_horizon: str
    - conviction: str — "high" | "medium" | "low"
    - position_size_pct: int — VIX-scaled position sizing (0-100%, rounded to 5%)
    """

    # ── Conviction tier ──────────────────────────────────────────────────────────
    if confidence >= 0.65:
        conviction = "high"
    elif confidence >= 0.45:
        conviction = "medium"
    else:
        conviction = "low"

    fund_score  = breakdown.get("fundamental", 0)

    # ── Risk adjustments ────────────────────────────────────────────────────────
    near_earnings = earnings_days_away is not None and 0 < earnings_days_away <= 14
    financial_distress = altman_zone == "distress"
    strong_momentum = momentum_trend in ("strong_up", "up")
    sector_headwind = sector_trend == "downtrend"
    high_short      = short_pct is not None and short_pct >= 20

    # ── SETUP-AWARE ENTRY/EXIT ────────────────────────────────────────────────────
    # When setup_detector provides structural levels, use them as primary anchors.
    # Fall back to percentage-based heuristics only when no structural data exists.

    is_buy = "BUY" in verdict.upper() and "AVOID" not in verdict.upper()
    supports_below = [s for s in support_levels if s < current_price]

    setup_type  = (setup_data or {}).get("setup_type", "RANGE")
    ee          = (setup_data or {}).get("entry_exit", {}) or {}
    struct_entry = ee.get("entry")
    struct_sl    = ee.get("stop_loss")
    struct_tp1   = ee.get("tp1")
    struct_tp2   = ee.get("tp2")
    ee_entry_basis = ee.get("entry_basis", "")
    ee_sl_basis    = ee.get("sl_basis", "")
    ee_tp_basis    = ee.get("tp_basis", "")

    # ── LIMIT ENTRY ──────────────────────────────────────────────────────────────
    limit_reason_parts = []

    # BREAKOUT entries are above current price (buy the break); all others must be at/below
    _entry_valid = (
        struct_entry is not None
        and abs(struct_entry - current_price) / current_price < 0.08
        and (setup_type == "BREAKOUT" or struct_entry <= current_price * 1.001)
    )

    if _entry_valid:
        # Structural level exists and is within 8% — use it
        limit_entry = struct_entry
        limit_reason_parts.append(f"{setup_type} setup: {ee_entry_basis}")
        base_discount_pct = (current_price - limit_entry) / current_price
    else:
        # Fallback: percentage-based with support snap
        if is_buy:
            if conviction == "high" and strong_momentum:
                base_discount_pct = 0.003
            elif conviction == "high":
                base_discount_pct = 0.007
            elif conviction == "medium":
                base_discount_pct = 0.010
            else:
                base_discount_pct = 0.015
        else:
            if conviction == "high":
                base_discount_pct = 0.010
            elif conviction == "medium":
                base_discount_pct = 0.018
            else:
                base_discount_pct = 0.025

        adj = 0.0
        if near_earnings:
            adj += 0.005
        if sector_headwind:
            adj += 0.003
        if financial_distress:
            adj += 0.005
        if high_short:
            adj += 0.003
        if not is_buy and week52_rank is not None and week52_rank > 0.90:
            adj += 0.008

        max_discount = 0.03 if is_buy else 0.05
        base_discount_pct = min(base_discount_pct + adj, max_discount)
        naive_limit = current_price * (1 - base_discount_pct)

        max_snap_distance = current_price * (base_discount_pct * 1.5)
        snap_floor = current_price - max_snap_distance
        limit_entry = naive_limit

        snapped = False
        for sup in sorted(supports_below, reverse=True):
            if sup < snap_floor:
                break
            if abs(sup - naive_limit) / current_price < 0.015:
                limit_entry = sup * 0.997
                limit_reason_parts.append(f"near support ${sup:.2f}")
                snapped = True
                break

        if not snapped:
            limit_reason_parts.append(f"{base_discount_pct*100:.1f}% discount from current")

    limit_reason_parts.append(f"{conviction} conviction")
    if near_earnings:
        limit_reason_parts.append("earnings buffer added")
    if week52_rank is not None and week52_rank > 0.85:
        limit_reason_parts.append("near 52W high")
    limit_entry_reason = "; ".join(limit_reason_parts)

    # ── STOP LOSS ───────────────────────────────────────────────────────────────
    if struct_sl and struct_sl < limit_entry and struct_sl > limit_entry * 0.90:
        # Structural swing-low stop — preferred
        stop_loss = struct_sl
        stop_reason = f"{setup_type}: {ee_sl_basis}"
    else:
        atr_stop = limit_entry - (1.5 * atr)
        stop_snapped = False
        stop_loss = atr_stop
        stop_reason = f"1.5× ATR (${atr:.2f}) below entry"

        for sup in sorted(supports_below, reverse=True):
            if sup < limit_entry * 0.995:
                candidate = sup * 0.995
                if candidate > atr_stop:
                    stop_loss = candidate
                    stop_reason = f"just below support level ${sup:.2f}"
                    stop_snapped = True
                    break

        if not stop_snapped and atr_stop < limit_entry * 0.92:
            stop_loss = limit_entry * 0.95
            stop_reason = "5% hard floor (ATR too wide)"

    if near_earnings and stop_loss < limit_entry * 0.97:
        stop_loss = limit_entry * 0.97
        stop_reason = f"tightened to 3% before earnings in {earnings_days_away}d"

    # ── TAKE PROFIT ─────────────────────────────────────────────────────────────
    risk_dist = limit_entry - stop_loss
    min_tp1 = limit_entry + max(risk_dist * 1.0, atr)
    min_tp2 = limit_entry + max(risk_dist * 1.5, atr * 2.0)

    resistances_above = [r for r in resistance_levels if r > limit_entry]

    # Use structural TP when it clears R:R floor
    if struct_tp1 and struct_tp1 >= min_tp1:
        tp1 = struct_tp1
        tp_source = f"{setup_type}: {ee_tp_basis}"
    else:
        tp1_candidates = [r for r in resistances_above if r >= min_tp1]
        tp1_from_resistance = min(tp1_candidates) if tp1_candidates else min_tp1

        tp1_from_analyst = None
        if analyst_target and analyst_target > current_price:
            tp1_from_analyst = analyst_target * 0.95

        if tp1_from_analyst and tp1_from_analyst >= min_tp1:
            tp1 = min(tp1_from_resistance, tp1_from_analyst)
        else:
            tp1 = tp1_from_resistance
        tp_source = f"resistance ${tp1:.2f}" if resistances_above else "ATR floor"

    if struct_tp2 and struct_tp2 > tp1 * 1.01 and struct_tp2 >= min_tp2:
        tp2 = struct_tp2
    else:
        tp2_candidates = [r for r in resistances_above if r > tp1 * 1.01 and r >= min_tp2]
        if tp2_candidates:
            tp2 = min(tp2_candidates)
        elif analyst_target and analyst_target >= min_tp2:
            tp2 = analyst_target
        else:
            tp2 = max(min_tp2, limit_entry + (3 * atr))

    tp_parts = [tp_source]
    if analyst_target and analyst_upside_pct is not None:
        tp_parts.append(f"analyst consensus ${analyst_target:.2f} ({analyst_upside_pct:+.1f}%)")
    take_profit_reason = "; ".join(tp_parts)

    # ── RISK/REWARD ─────────────────────────────────────────────────────────────
    risk = limit_entry - stop_loss
    reward = tp1 - limit_entry
    risk_reward = round(reward / risk, 2) if risk > 0 else 0

    # ── EXIT CONDITIONS (plain English) ─────────────────────────────────────────
    exits = []

    # Price-based
    exits.append(f"Price drops below stop ${stop_loss:.2f} — close position")

    # Thesis invalidation
    if fund_score > 0.3:
        exits.append("Fundamentals deteriorate (earnings miss >10%, guidance cut) — re-evaluate")
    if piotroski_score >= 6:
        exits.append(f"Piotroski score drops below 5 (currently {piotroski_score}/9) — financial health weakening")

    # Profit taking
    exits.append(f"Take 50% off at TP1 (${tp1:.2f}), let rest run to TP2 (${tp2:.2f})".replace("*", "\\*"))

    # Momentum reversal
    if strong_momentum:
        exits.append("MACD crosses below signal AND RSI drops below 45 — momentum broken, exit remaining")
    else:
        exits.append("RSI drops below 35 and price closes below 50-day SMA — trend broken")

    # Earnings event
    if earnings_days_away and earnings_days_away <= 30:
        exits.append(f"Earnings in {earnings_days_away}d — consider taking profit before if up >5%, re-enter after if thesis intact")

    # Sector rotation
    if sector_headwind:
        exits.append("Sector still in downtrend — size smaller, exit faster on any weakness")

    # Time-based
    exits.append("Re-evaluate thesis in 4–8 weeks if price hasn't moved toward TP1")

    # ── TIME HORIZON ─────────────────────────────────────────────────────────────
    if strong_momentum and conviction == "high":
        time_horizon = "2–6 weeks (momentum trade)"
    elif conviction == "high" and fund_score > 0.4:
        time_horizon = "2–4 months (fundamental + momentum)"
    elif conviction == "medium":
        time_horizon = "4–8 weeks (wait for catalyst)"
    else:
        time_horizon = "1–2 weeks (speculative; tight stops)"

    # ── POSITION SIZE SCALING BY VIX ─────────────────────────────────────────────
    # Interpolate: VIX ≤ 15 → 100%, VIX ≥ 25 → 0%, linear between
    # Round to nearest 5%
    position_size_pct = 100  # default to full size
    if vix is not None:
        if vix <= 15:
            position_size_pct = 100.0
        elif vix >= 25:
            position_size_pct = 0.0
        else:
            # Linear interpolation between 15 and 25
            position_size_pct = 100.0 - ((vix - 15) / (25 - 15)) * 100.0
        # Round to nearest 5%
        position_size_pct = round(position_size_pct / 5) * 5
    # Clamp to [0, 100]
    position_size_pct = max(0, min(100, position_size_pct))

    return {
        "limit_entry": round(limit_entry, 2),
        "limit_entry_reason": limit_entry_reason,
        "take_profit_1": round(tp1, 2),
        "take_profit_2": round(tp2, 2),
        "take_profit_reason": take_profit_reason,
        "stop_loss": round(stop_loss, 2),
        "stop_loss_reason": stop_reason,
        "risk_reward": risk_reward,
        "exit_conditions": exits,
        "time_horizon": time_horizon,
        "conviction": conviction,
        "discount_pct": round(base_discount_pct * 100, 1),
        "position_size_pct": int(position_size_pct),
        "setup_type": setup_type,
    }
