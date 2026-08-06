"""
Smart trade strategy: limit entry, take profit, stop loss, exit conditions.
Uses all available signals — technical, fundamental, momentum, analyst targets,
support/resistance levels, earnings proximity, and combined conviction score.
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

    # ── LIMIT ENTRY ─────────────────────────────────────────────────────────────
    # Discount is verdict-first, then conviction-adjusted.
    # BUY  → meaningful discount (waiting for dip makes sense)
    # HOLD → small discount (stock is fairly valued; enter near current)
    # SELL → no entry suggested (handled below)
    is_buy  = "BUY" in verdict.upper() and "AVOID" not in verdict.upper()

    if is_buy:
        if conviction == "high" and strong_momentum:
            base_discount_pct = 0.008   # 0.8% — strong trend, don't miss entry
        elif conviction == "high":
            base_discount_pct = 0.015   # 1.5%
        elif conviction == "medium":
            base_discount_pct = 0.020   # 2.0%
        else:
            base_discount_pct = 0.025   # 2.5% — low conviction BUY needs cushion
    else:
        # HOLD/WATCH — enter near current or on small dip; no big discount hunting
        if conviction == "high":
            base_discount_pct = 0.005   # 0.5%
        elif conviction == "medium":
            base_discount_pct = 0.010   # 1.0%
        else:
            base_discount_pct = 0.015   # 1.5% — low conviction HOLD, be patient

    # Risk adjustments (additive, but cap total at 4% for HOLD, 6% for BUY)
    adj = 0.0
    if near_earnings:
        adj += 0.010
    if sector_headwind:
        adj += 0.005
    if financial_distress:
        adj += 0.010
    if week52_rank is not None and week52_rank > 0.90:
        adj += 0.010  # near 52W high — extra patience
    if high_short:
        adj += 0.005

    max_discount = 0.06 if is_buy else 0.04
    base_discount_pct = min(base_discount_pct + adj, max_discount)

    naive_limit = current_price * (1 - base_discount_pct)

    # Snap to nearest support below naive_limit (within 5% slack)
    supports_below = [s for s in support_levels if s < current_price]
    limit_entry = naive_limit
    limit_reason_parts = []

    # Only snap to support if it's within 1.5× the base discount distance from current price
    # Prevents snapping to far-away supports that require unrealistic pullbacks
    max_snap_distance = current_price * (base_discount_pct * 1.5)
    snap_floor = current_price - max_snap_distance

    snapped = False
    for sup in sorted(supports_below, reverse=True):  # closest first
        if sup < snap_floor:
            break  # too far below — don't snap here
        if abs(sup - naive_limit) / current_price < 0.015:  # support within 1.5% of naive → snap to it
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
        limit_reason_parts.append("near 52W high — extra patience")

    limit_entry_reason = "; ".join(limit_reason_parts)

    # ── STOP LOSS ───────────────────────────────────────────────────────────────
    # Default: 1.5× ATR below entry
    atr_stop = limit_entry - (1.5 * atr)

    # Try to snap to support below limit_entry
    stop_snapped = False
    stop_loss = atr_stop
    stop_reason = f"1.5× ATR (${atr:.2f}) below entry"

    for sup in sorted(supports_below, reverse=True):
        if sup < limit_entry * 0.995:  # below entry
            candidate = sup * 0.995    # just below support
            if candidate > atr_stop:   # tighter than ATR stop → use support-based
                stop_loss = candidate
                stop_reason = f"just below support level ${sup:.2f}"
                stop_snapped = True
                break

    if not stop_snapped and atr_stop < limit_entry * 0.92:
        # ATR stop is >8% away — use 5% hard floor instead
        stop_loss = limit_entry * 0.95
        stop_reason = "5% hard floor (ATR too wide)"

    # Tighten stop near earnings — don't hold through binary event at risk
    if near_earnings and stop_loss < limit_entry * 0.97:
        stop_loss = limit_entry * 0.97
        stop_reason = f"tightened to 3% before earnings in {earnings_days_away}d"

    # ── TAKE PROFIT ─────────────────────────────────────────────────────────────
    # Conservative TP1: nearest resistance above entry
    resistances_above = [r for r in resistance_levels if r > limit_entry]
    tp1_from_resistance = min(resistances_above) if resistances_above else limit_entry + (2 * atr)

    # Analyst target as ceiling check
    tp1_from_analyst = None
    if analyst_target and analyst_target > current_price:
        tp1_from_analyst = analyst_target * 0.95  # 5% haircut on analyst target

    # TP1: min of resistance and 95% of analyst target (take the more conservative)
    if tp1_from_analyst:
        tp1 = min(tp1_from_resistance, tp1_from_analyst)
        tp1 = max(tp1, limit_entry + atr)  # at least 1 ATR above entry
    else:
        tp1 = tp1_from_resistance

    # TP2: next resistance level, or analyst target, or 3× ATR
    tp2_candidates = [r for r in resistances_above if r > tp1 * 1.01]
    if tp2_candidates:
        tp2 = min(tp2_candidates)
    elif analyst_target and analyst_target > tp1:
        tp2 = analyst_target
    else:
        tp2 = limit_entry + (3 * atr)

    # Build TP reason
    tp_parts = []
    if resistances_above and tp1 == tp1_from_resistance:
        tp_parts.append(f"TP1 at resistance ${tp1_from_resistance:.2f}")
    elif tp1_from_analyst:
        tp_parts.append(f"TP1 at 95% of analyst target ${analyst_target:.2f}")
    if analyst_target:
        tp_parts.append(f"analyst consensus ${analyst_target:.2f} ({analyst_upside_pct:+.1f}%)")
    take_profit_reason = "; ".join(tp_parts) if tp_parts else "ATR-based targets"

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
    }
