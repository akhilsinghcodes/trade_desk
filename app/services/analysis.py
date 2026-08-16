"""Analysis service — orchestrates all data fetching and scoring."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.indicators import add_common_indicators
from modules.summary import summarize
from modules.support_resistance import swing_levels, pivot_points, suggest_trade
from modules.earnings import get_next_earnings
from modules.score import combined_score
from modules.volume import add_volume_indicators, volume_signal
from modules.thesis import generate_thesis
from modules.trade_strategy import get_smart_trade_strategy as _smart_strat
from modules.llm_strategy import generate_trade_rationale
from modules.alert_suggestions import suggest_alerts
from modules.fundamentals import score_fundamentals
from modules.analyst import score_analyst
from modules.insider import score_insider
from modules.ownership import score_ownership
from modules.options_sentiment import score_options
from modules.earnings_history import score_earnings_history
from modules.relative_performance import score_relative_performance
from modules.short_interest import score_short_interest
from modules.balance_sheet_trends import score_balance_sheet
from modules.market_context import score_market_context
from modules.piotroski import score_piotroski
from modules.valuation_advanced import score_valuation_advanced
from modules.sector_momentum import score_sector_momentum
from modules.dilution_risk import score_dilution, score_earnings_proximity, get_earnings_proximity
from modules.altman_z import score_altman_z
from modules.momentum import score_momentum
from modules.vol_ratio import score_vol_ratio
from modules.momentum_signals import compute_coppock, compute_ridge_slope
from modules.setup_detector import analyze_setup
from modules.cached_fetch import (
    cached_ohlcv, cached_info, cached_fundamentals,
    cached_analyst, cached_insider, cached_ownership, cached_options,
    cached_earnings_history, cached_short_interest, cached_balance_sheet,
    cached_rel_perf, cached_news, cached_market_context, cached_piotroski,
    cached_valuation_advanced, cached_sector_momentum, cached_dilution_risk,
    cached_altman_z, cached_momentum, cached_pmo_rs, cached_vol_ratio,
)


def run_analysis(ticker: str, period: str, interval: str = "1d", llm_backend: str = "local") -> dict:
    """
    Fetch all data and compute signals for a ticker.

    Args:
        ticker: Stock ticker symbol
        period: Period for price data (e.g., "1y", "3mo")
        interval: Interval for price data (default "1d")

    Returns:
        Dict with keys: df, info, df_vol, fundamentals, fund_signals, earnings_info,
        analyst_data, insider_data, ownership_data, options_data, earnings_hist,
        short_data, balance_data, rel_perf, rel_series, market_ctx, piotroski_data,
        valuation_adv, sector_mom, dilution_data, altman_data, momentum_data, vol_ratio_data,
        pmo_rs_data, coppock_data, ridge_slope_data, scored_articles, tech_summary,
        verdict_result, swings, pivots, smart_trade, thesis, alert_suggestions, close_price,
        prev_close, price_change, price_change_pct.
    """
    # ── FETCH ALL DATA (cached) ────────────────────────────────────────────────────
    df = cached_ohlcv(ticker, period)
    info = cached_info(ticker)
    df = add_common_indicators(df)
    df = add_volume_indicators(df)

    fundamentals = cached_fundamentals(ticker)
    fund_signals = score_fundamentals(fundamentals)
    earnings_info = get_next_earnings(ticker)
    sector = info.get("sector", "")

    analyst_data = cached_analyst(ticker)
    insider_data = cached_insider(ticker)
    ownership_data = cached_ownership(ticker)
    options_data = cached_options(ticker)
    earnings_hist = cached_earnings_history(ticker)
    short_data = cached_short_interest(ticker)
    balance_data = cached_balance_sheet(ticker)
    rel_perf, rel_series = cached_rel_perf(ticker, period)
    market_ctx = cached_market_context(ticker)
    piotroski_data = cached_piotroski(ticker)
    valuation_adv = cached_valuation_advanced(ticker)
    sector_mom = cached_sector_momentum(sector)
    dilution_data = cached_dilution_risk(ticker)
    altman_data = cached_altman_z(ticker)
    momentum_data = cached_momentum(ticker)
    pmo_rs_data = cached_pmo_rs(ticker, period)
    vol_ratio_data = cached_vol_ratio(ticker, df)

    # News — use SQLite cache
    scored_articles = cached_news(ticker, limit=10, company=info.get("shortName", ""))

    # Compute scores
    tech_summary = summarize(df)
    coppock_data = compute_coppock(df)
    ridge_slope_data = compute_ridge_slope(df)
    setup_data = analyze_setup(df)
    vol_sig = volume_signal(df)
    tech_summary["signals"].append(vol_sig)
    tech_summary["signals"].append(score_analyst(analyst_data))
    tech_summary["signals"].append(score_insider(insider_data))
    tech_summary["signals"].append(score_ownership(ownership_data))
    tech_summary["signals"].append(score_options(options_data))
    tech_summary["signals"].append(score_earnings_history(earnings_hist))
    tech_summary["signals"].append(score_relative_performance(rel_perf, ticker))
    tech_summary["signals"].append(score_short_interest(short_data))
    tech_summary["signals"].append(score_balance_sheet(balance_data))
    tech_summary["signals"].append(score_market_context(market_ctx))
    tech_summary["signals"].append(score_piotroski(piotroski_data))
    tech_summary["signals"].append(score_valuation_advanced(valuation_adv))
    tech_summary["signals"].append(score_sector_momentum(sector_mom))
    tech_summary["signals"].append(score_dilution(dilution_data))
    earnings_prox = get_earnings_proximity(earnings_info)
    tech_summary["signals"].append(score_earnings_proximity(earnings_prox))
    tech_summary["signals"].append(score_altman_z(altman_data))
    tech_summary["signals"].append(score_momentum(momentum_data))
    tech_summary["signals"].append(score_vol_ratio(vol_ratio_data))

    verdict_result = combined_score(
        tech_signals=tech_summary["signals"],
        fund_signals=fund_signals,
        news_articles=scored_articles,
    )

    trade = suggest_trade(df, verdict_result["verdict"])
    swings = swing_levels(df)
    pivots = pivot_points(df)

    # Smart trade strategy using all signals
    _supports_flat = sorted(
        [s for s in (swings.get("support", []) + [pivots.get("s1"), pivots.get("s2")]) if s],
        reverse=True
    )
    _resists_flat = sorted(
        [r for r in (swings.get("resistance", []) + [pivots.get("r1"), pivots.get("r2")]) if r]
    )
    _atr_val = trade.get("atr") or 0
    if not _atr_val and len(df) >= 14:
        _hi = df["high"].astype(float)
        _lo = df["low"].astype(float)
        _cl = df["close"].astype(float).shift(1)
        _tr = (_hi - _lo).combine((_hi - _cl).abs(), max).combine((_lo - _cl).abs(), max)
        _atr_val = float(_tr.rolling(14).mean().iloc[-1] or 0)

    smart_trade = _smart_strat(
        current_price=float(df["close"].iloc[-1]),
        atr=float(_atr_val),
        confidence=verdict_result["confidence"],
        verdict=verdict_result["verdict"],
        breakdown=verdict_result.get("breakdown", {}),
        support_levels=_supports_flat,
        resistance_levels=_resists_flat,
        pivot=pivots.get("pivot"),
        analyst_target=analyst_data.get("target_price"),
        analyst_upside_pct=analyst_data.get("upside_pct"),
        week52_high=market_ctx.get("week52_high"),
        week52_low=market_ctx.get("week52_low"),
        week52_rank=market_ctx.get("week52_rank"),
        piotroski_score=piotroski_data.get("score", 0),
        altman_zone=altman_data.get("zone", "unknown"),
        momentum_trend=momentum_data.get("trend", "neutral"),
        ret_1yr=momentum_data.get("ret_1yr"),
        short_pct=short_data.get("short_pct_float"),
        earnings_days_away=earnings_info.get("days_away"),
        sector_trend=sector_mom.get("trend", "unknown"),
        vix=market_ctx.get("vix"),
        setup_data=setup_data,
    )

    # Investment thesis
    thesis = generate_thesis(
        ticker=ticker,
        verdict=verdict_result["verdict"],
        confidence=verdict_result["confidence"],
        breakdown=verdict_result.get("breakdown", {}),
        signals=tech_summary["signals"],
        fundamentals=fundamentals,
        analyst_data=analyst_data,
        piotroski_data=piotroski_data,
        valuation_adv=valuation_adv,
        market_ctx=market_ctx,
        short_data=short_data,
        earnings_info=earnings_info,
        sector_mom=sector_mom,
        momentum_data=momentum_data,
        altman_data=altman_data,
        company_name=info.get("shortName", ticker),
    )

    # LLM signal summary — plain-English explainer only, no trade decisions
    llm_rationale = generate_trade_rationale(
        ticker=ticker,
        signals=tech_summary["signals"],
        earnings_days=earnings_info.get("days_away"),
        backend=llm_backend,
    )

    # Auto-suggest alerts
    alert_suggestions = suggest_alerts(
        ticker=ticker,
        current_price=df["close"].iloc[-1],
        swing_levels=swings,
        pivot_points=pivots,
        analyst_data=analyst_data,
        earnings_info=earnings_info,
    )

    # Price data
    close_price = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    price_change = close_price - prev_close
    price_change_pct = price_change / prev_close * 100

    return {
        "df": df,
        "info": info,
        "fundamentals": fundamentals,
        "fund_signals": fund_signals,
        "earnings_info": earnings_info,
        "analyst_data": analyst_data,
        "insider_data": insider_data,
        "ownership_data": ownership_data,
        "options_data": options_data,
        "earnings_hist": earnings_hist,
        "short_data": short_data,
        "balance_data": balance_data,
        "rel_perf": rel_perf,
        "rel_series": rel_series,
        "market_ctx": market_ctx,
        "piotroski_data": piotroski_data,
        "valuation_adv": valuation_adv,
        "sector_mom": sector_mom,
        "dilution_data": dilution_data,
        "altman_data": altman_data,
        "momentum_data": momentum_data,
        "vol_ratio_data": vol_ratio_data,
        "pmo_rs_data": pmo_rs_data,
        "coppock_data": coppock_data,
        "ridge_slope_data": ridge_slope_data,
        "setup_data": setup_data,
        "scored_articles": scored_articles,
        "tech_summary": tech_summary,
        "verdict_result": verdict_result,
        "swings": swings,
        "pivots": pivots,
        "trade": trade,
        "smart_trade": smart_trade,
        "thesis": thesis,
        "llm_rationale": llm_rationale,
        "alert_suggestions": alert_suggestions,
        "close_price": close_price,
        "prev_close": prev_close,
        "price_change": price_change,
        "price_change_pct": price_change_pct,
    }
