"""
Thesis Module: Synthesize stock signals into structured investment thesis.

Pure Python logic — no external API calls, no ML model needed.
Converts quantitative signals into human-readable investment narrative.
"""



def generate_thesis(
    ticker: str,
    verdict: str,           # "BUY" | "HOLD / WATCH" | "SELL / AVOID"
    confidence: float,      # 0-1
    breakdown: dict,        # {"technical": float, "fundamental": float, "sentiment": float}
    signals: list,          # list of (label, status, text) tuples
    fundamentals: dict,     # raw fundamentals dict
    analyst_data: dict,
    piotroski_data: dict,
    valuation_adv: dict,
    market_ctx: dict,
    short_data: dict,
    earnings_info: dict,    # {"days_away": int or None, "date": str or None}
    sector_mom: dict,
    momentum_data: dict,
    altman_data: dict,
    company_name: str,
) -> dict:
    """
    Synthesize all stock signals into a structured, plain-English investment thesis.

    Args:
        ticker: Stock ticker symbol
        verdict: Investment recommendation (BUY, HOLD / WATCH, SELL / AVOID)
        confidence: Confidence level (0-1)
        breakdown: Signal strength breakdown by category
        signals: List of (label, status, text) tuples from analysis
        fundamentals: Raw fundamentals dictionary
        analyst_data: Analyst ratings and targets
        piotroski_data: Piotroski F-Score data
        valuation_adv: Advanced valuation metrics
        market_ctx: Market context (sector, VIX, etc.)
        short_data: Short interest data
        earnings_info: Earnings info with days_away and date
        sector_mom: Sector momentum data
        momentum_data: Price momentum data
        altman_data: Altman Z-Score data
        company_name: Full company name

    Returns:
        Dictionary with thesis components:
        - headline: str (1 punchy sentence)
        - bull_points: list[str] (3-5 points)
        - bear_points: list[str] (3-5 points)
        - key_risk: str (single most important risk)
        - key_catalyst: str (single most important catalyst)
        - one_liner: str (tweet-length summary)
    """

    # Extract signal groups by status
    good_signals = [(label, text) for label, status, text in signals if status == "good"]
    warning_signals = [(label, text) for label, status, text in signals if status == "warning"]

    # Generate each thesis component
    headline = _generate_headline(verdict, good_signals, warning_signals, ticker)
    bull_points = _generate_bull_points(good_signals, analyst_data, piotroski_data,
                                        fundamentals, momentum_data, short_data)
    bear_points = _generate_bear_points(warning_signals, valuation_adv, earnings_info,
                                        sector_mom, short_data)
    key_risk = _identify_key_risk(warning_signals, earnings_info, altman_data,
                                  valuation_adv, sector_mom, short_data)
    key_catalyst = _identify_key_catalyst(good_signals, analyst_data, momentum_data,
                                          short_data, fundamentals)
    one_liner = _generate_one_liner(company_name, ticker, verdict, fundamentals,
                                    analyst_data, sector_mom, breakdown)

    return {
        "headline": headline,
        "bull_points": bull_points,
        "bear_points": bear_points,
        "key_risk": key_risk,
        "key_catalyst": key_catalyst,
        "one_liner": one_liner,
    }


def _generate_headline(verdict: str, good_signals: list, warning_signals: list, ticker: str) -> str:
    """
    Generate punchy headline based on verdict and strongest signals.

    BUY + high fundamental: "Strong fundamentals at reasonable valuation"
    BUY + momentum: "Momentum building with improving technicals"
    HOLD: "Mixed picture — strength offset by [top warning]"
    SELL: "Deteriorating [fundamentals/technicals] with limited upside"
    """

    if "BUY" in verdict:
        # Check strongest signal category
        if good_signals:
            top_signal = good_signals[0][0].lower()
            if "fundamental" in top_signal or "valuation" in top_signal:
                return "Strong fundamentals at reasonable valuation"
            elif "momentum" in top_signal or "technical" in top_signal:
                return "Momentum building with improving technicals"
            elif "growth" in top_signal or "earnings" in top_signal:
                return "Exceptional growth engine with strong execution"
        return "Compelling risk-reward setup"

    elif "HOLD" in verdict or "WATCH" in verdict:
        if warning_signals:
            top_warning = warning_signals[0][0].lower()
            if "valuation" in top_warning:
                return "Mixed picture — strength offset by premium valuation"
            elif "technical" in top_warning:
                return "Mixed picture — fundamentals strong but technicals weakening"
            elif "sector" in top_warning:
                return "Mixed picture — company strength offset by sector headwind"
        return "Mixed signals warrant cautious stance"

    elif "SELL" in verdict or "AVOID" in verdict:
        if warning_signals:
            top_warning = warning_signals[0][0].lower()
            if "fundamental" in top_warning or "earnings" in top_warning:
                return "Deteriorating fundamentals with limited upside"
            elif "valuation" in top_warning:
                return "Stretched valuation with deteriorating momentum"
            elif "technical" in top_warning:
                return "Technical breakdown with bearish implications"
        return "Risk-reward heavily skewed to downside"

    return "Neutral setup awaiting clarity"


def _generate_bull_points(good_signals: list, analyst_data: dict, piotroski_data: dict,
                          fundamentals: dict, momentum_data: dict, short_data: dict) -> list:
    """
    Generate 3-5 bull points from good signals and supporting metrics.
    Format: "✅ [label]: [abbreviated text, max 80 chars]"
    """
    bull_points = []

    # Add good signals first (priority)
    for label, text in good_signals[:5]:
        abbreviated = text[:77] if len(text) > 77 else text
        bull_points.append(f"✅ {label}: {abbreviated}")

    # Add supplementary bullish metrics if not already covered
    covered_labels = [label.lower() for label, _ in good_signals]

    # Analyst upside
    if analyst_data and analyst_data.get("upside_pct", 0) > 20:
        if not any("analyst" in c for c in covered_labels):
            upside = analyst_data.get("upside_pct", 0)
            bull_points.append(f"✅ Analyst Coverage: {upside:.1f}% upside target implies significant room")

    # Piotroski score
    if piotroski_data and piotroski_data.get("score", 0) >= 7:
        if not any("piotroski" in c for c in covered_labels):
            score = piotroski_data.get("score", 0)
            bull_points.append(f"✅ Piotroski F-Score: {int(score)}/9 shows strong financial health")

    # FCF yield
    if fundamentals and fundamentals.get("fcf_yield", 0) > 0.05:
        if not any("fcf" in c for c in covered_labels):
            fcf = fundamentals.get("fcf_yield", 0) * 100
            bull_points.append(f"✅ Free Cash Flow: {fcf:.1f}% yield provides downside cushion")

    # Positive momentum
    if momentum_data and momentum_data.get("trend", "").lower() in ("up", "strong_up"):
        if not any("momentum" in c for c in covered_labels):
            ret_1yr = momentum_data.get("ret_1yr")
            yr_str = f" (+{ret_1yr:.1f}% 1yr)" if ret_1yr else ""
            bull_points.append(f"✅ Momentum: Price momentum accelerating{yr_str}")

    # Low short interest
    if short_data and short_data.get("short_pct_float", 0) < 3:
        if not any("short" in c for c in covered_labels):
            si = short_data.get("short_pct_float", 0)
            bull_points.append(f"✅ Short Interest: Low {si:.1f}% reduces squeeze risk downside")

    # Return top 5
    return bull_points[:5]


def _generate_bear_points(warning_signals: list, valuation_adv: dict, earnings_info: dict,
                          sector_mom: dict, short_data: dict) -> list:
    """
    Generate 3-5 bear points from warning signals and risk metrics.
    Format: "⚠️ [label]: [abbreviated text, max 80 chars]"
    """
    bear_points = []

    # Add warning signals first
    for label, text in warning_signals[:5]:
        abbreviated = text[:77] if len(text) > 77 else text
        bear_points.append(f"⚠️ {label}: {abbreviated}")

    # Add supplementary bearish metrics
    covered_labels = [label.lower() for label, _ in warning_signals]

    # High valuation
    if valuation_adv and (valuation_adv.get("ev_ebitda") or 0) > 25:
        if not any("valuation" in c for c in covered_labels):
            ev = valuation_adv.get("ev_ebitda", 0)
            bear_points.append(f"⚠️ Valuation: EV/EBITDA {ev:.1f}x well above historical average")

    # Earnings proximity
    if earnings_info and earnings_info.get("days_away"):
        days = earnings_info.get("days_away", 999)
        if 0 < days < 14 and not any("earnings" in c for c in covered_labels):
            bear_points.append(f"⚠️ Earnings Event: Earnings in {int(days)} days adds near-term volatility")

    # Sector downtrend
    if sector_mom and sector_mom.get("trend", "").lower() == "down":
        if not any("sector" in c for c in covered_labels):
            bear_points.append(f"⚠️ Sector Headwind: {sector_mom.get('name', 'Sector')} momentum negative")

    # High short interest
    if short_data and short_data.get("short_pct_float", 0) > 15:
        if not any("short" in c for c in covered_labels):
            si = short_data.get("short_pct_float", 0)
            bear_points.append(f"⚠️ Short Interest: High {si:.1f}% indicates significant bearish positioning")

    # Return top 5
    return bear_points[:5]


def _identify_key_risk(warning_signals: list, earnings_info: dict, altman_data: dict,
                       valuation_adv: dict, sector_mom: dict, short_data: dict) -> str:
    """
    Identify single most urgent risk.
    Priority: earnings <14 days > Altman distress > high valuation > sector downtrend > high short interest
    """

    # Earnings proximity (highest priority)
    if earnings_info and earnings_info.get("days_away"):
        days = earnings_info.get("days_away", 999)
        if 0 < days < 14:
            return f"Earnings release in {int(days)} days introduces event risk; volatility spike likely"

    # Altman Z-Score distress
    if altman_data and altman_data.get("zone") == "distress" and altman_data.get("z_score") is not None:
        score = altman_data.get("z_score", 0)
        return f"Altman Z-Score {score:.2f} signals financial distress; bankruptcy risk elevated"

    # High valuation
    if valuation_adv and (valuation_adv.get("ev_ebitda") or 0) > 25:
        ev = valuation_adv.get("ev_ebitda", 0)
        return f"Valuation premium (EV/EBITDA {ev:.1f}x) leaves little room for disappointment"

    # Sector downtrend
    if sector_mom and sector_mom.get("trend", "").lower() == "down":
        sector_name = sector_mom.get("name", "Sector")
        return f"{sector_name} sector momentum turning negative; sector rotation risk"

    # Insider selling
    if warning_signals:
        for label, text in warning_signals:
            if "insider" in label.lower() and "sell" in text.lower():
                return text

    # High short interest
    if short_data and short_data.get("short_pct_float", 0) > 15:
        si = short_data.get("short_pct_float", 0)
        return f"High short interest ({si:.1f}%) indicates bearish positioning and crowding"

    # Fallback
    if warning_signals:
        return warning_signals[0][1]

    return "No material risks identified; upside risk moderate"


def _identify_key_catalyst(good_signals: list, analyst_data: dict, momentum_data: dict,
                           short_data: dict, fundamentals: dict) -> str:
    """
    Identify single strongest upside catalyst.
    Priority: analyst upside >30% > strong momentum > earnings beat history > low short + rising > FCF yield >5%
    """

    # Analyst upside (highest priority)
    if analyst_data and analyst_data.get("upside_pct", 0) > 30:
        upside = analyst_data.get("upside_pct", 0)
        target = analyst_data.get("target_price", 0)
        return f"Analyst consensus {upside:.1f}% upside to ${target:.2f} target reflects significant value"

    # Strong momentum
    if momentum_data and momentum_data.get("trend", "").lower() in ("up", "strong_up"):
        ret_1yr = momentum_data.get("ret_1yr")
        yr_str = f" (+{ret_1yr:.1f}% 1yr)" if ret_1yr else ""
        return f"Strong price momentum{yr_str} suggests trend continuation"

    # Earnings beat history
    if fundamentals and fundamentals.get("earnings_growth_pct", 0) > 20:
        growth = fundamentals.get("earnings_growth_pct", 0)
        return f"Earnings growth of {growth:.1f}% YoY with track record of beat-and-raise"

    # Low short interest + rising price
    if short_data and short_data.get("short_pct_float", 0) < 3:
        if momentum_data and momentum_data.get("trend", "").lower() in ("up", "strong_up"):
            return "Low short interest with rising price reduces squeeze upside but confirms trend"

    # FCF yield
    if fundamentals and fundamentals.get("fcf_yield", 0) > 0.05:
        fcf = fundamentals.get("fcf_yield", 0) * 100
        return f"Strong free cash flow yield ({fcf:.1f}%) supports buybacks and dividend growth"

    # Fallback to good signals
    if good_signals:
        return good_signals[0][1]

    return "Lack of near-term catalysts; monitor for fundamental inflection"


def _generate_one_liner(company_name: str, ticker: str, verdict: str, fundamentals: dict,
                        analyst_data: dict, sector_mom: dict, breakdown: dict) -> str:
    """
    Generate tweet-length summary (1-2 sentences).
    Format: "TICKER: [2-3 key facts], [verdict]."
    Example: "NVDA: exceptional margins (63%) and growth (85%) but EV/EBITDA 30x with XLK headwind. Analysts bullish (+42.9%). HOLD."
    """

    facts = []

    # Key fundamental fact
    if fundamentals:
        margin = fundamentals.get("profit_margin", 0) * 100
        if margin > 20:
            facts.append(f"{margin:.0f}% margins")

        growth = fundamentals.get("revenue_growth_pct", 0)
        if growth > 0:
            facts.append(f"{growth:.0f}% revenue growth")

    # Valuation concern
    if fundamentals:
        pe = fundamentals.get("pe_ratio", 0)
        if pe > 30:
            facts.append(f"{pe:.0f}x P/E (pricey)")

    # Sector headwind
    if sector_mom and sector_mom.get("trend", "").lower() == "down":
        sector = sector_mom.get("symbol", "sector")
        facts.append(f"{sector} headwind")

    # Analyst view
    analyst_view = ""
    if analyst_data and analyst_data.get("upside_pct", 0) > 0:
        upside = analyst_data.get("upside_pct", 0)
        if upside > 20:
            analyst_view = f"Analysts bullish (+{upside:.1f}% target)."
        elif upside < -10:
            analyst_view = f"Analysts bearish ({upside:.1f}% target)."

    # Assemble one-liner
    facts_str = ", ".join(facts) if facts else "mixed fundamentals"
    one_liner = f"{ticker}: {facts_str}. {analyst_view} {verdict}."

    # Clean up
    one_liner = one_liner.replace("..", ".").strip()

    return one_liner


def format_thesis_markdown(thesis: dict) -> str:
    """
    Format thesis dictionary into clean markdown string for Streamlit display.

    Args:
        thesis: Dictionary returned by generate_thesis()

    Returns:
        Formatted markdown string
    """

    markdown = ""

    # Headline
    if thesis.get("headline"):
        markdown += f"## {thesis['headline']}\n\n"

    # Confidence/Verdict section (if available in context)

    # Bull points
    if thesis.get("bull_points"):
        markdown += "### Bull Case\n"
        for point in thesis["bull_points"]:
            markdown += f"- {point}\n"
        markdown += "\n"

    # Bear points
    if thesis.get("bear_points"):
        markdown += "### Bear Case\n"
        for point in thesis["bear_points"]:
            markdown += f"- {point}\n"
        markdown += "\n"

    # Key risk
    if thesis.get("key_risk"):
        markdown += f"### Key Risk\n{thesis['key_risk']}\n\n"

    # Key catalyst
    if thesis.get("key_catalyst"):
        markdown += f"### Key Catalyst\n{thesis['key_catalyst']}\n\n"

    # One-liner
    if thesis.get("one_liner"):
        markdown += f"### Summary\n> {thesis['one_liner']}\n"

    return markdown
