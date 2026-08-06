"""Analyze page."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.backtest import sma_crossover_backtest, get_backtest_stats
from modules.sector import get_sector_comparison
from modules.news import overall_sentiment
from modules.db import alerts_load, alerts_add
from app.services.analysis import run_analysis


color_map = {"green": "#00c853", "red": "#ff1744", "orange": "#ff9100"}
icon_map = {"good": "✅", "warning": "⚠️", "neutral": "➖", "unknown": "➖"}
sent_icon = {"positive": "🟢", "negative": "🔴", "neutral": "🟡", "unknown": "⚪"}


def render_analyze_page(st_obj, ticker, period, show_bb, show_sma, run_backtest, fast_win, slow_win):
    """Render analyze page."""
    with st_obj.spinner(f"Loading {ticker}..."):
        try:
            analysis = run_analysis(ticker, period)
        except Exception as e:
            st_obj.error(f"Failed to fetch data: {e}")
            st_obj.stop()

    df = analysis["df"]
    if df.empty:
        st_obj.warning("No data returned. Check ticker symbol.")
        st_obj.stop()

    # Unpack analysis results
    info = analysis["info"]
    fund_signals = analysis["fund_signals"]
    analyst_data = analysis["analyst_data"]
    insider_data = analysis["insider_data"]
    ownership_data = analysis["ownership_data"]
    options_data = analysis["options_data"]
    earnings_hist = analysis["earnings_hist"]
    short_data = analysis["short_data"]
    balance_data = analysis["balance_data"]
    rel_perf = analysis["rel_perf"]
    rel_series = analysis["rel_series"]
    market_ctx = analysis["market_ctx"]
    piotroski_data = analysis["piotroski_data"]
    valuation_adv = analysis["valuation_adv"]
    sector_mom = analysis["sector_mom"]
    altman_data = analysis["altman_data"]
    momentum_data = analysis["momentum_data"]
    earnings_info = analysis["earnings_info"]
    scored_articles = analysis["scored_articles"]
    tech_summary = analysis["tech_summary"]
    verdict_result = analysis["verdict_result"]
    swings = analysis["swings"]
    pivots = analysis["pivots"]
    smart_trade = analysis["smart_trade"]
    thesis = analysis["thesis"]
    alert_suggestions = analysis["alert_suggestions"]
    close_price = analysis["close_price"]
    prev_close = analysis["prev_close"]
    price_change = analysis["price_change"]
    price_change_pct = analysis["price_change_pct"]

    # Price data
    price_color = "#00c853" if price_change >= 0 else "#ff1744"
    price_sign = "+" if price_change >= 0 else ""

    v_hex = color_map.get(verdict_result["color"], "#888888")
    v_bg = f"{v_hex}14"
    confidence_pct = int(verdict_result["confidence"] * 100)
    bd = verdict_result["breakdown"]

    # ── SECTION 1: COMPANY HEADER ─────────────────────────────────────────────────
    mc = f"${info.get('marketCap',0)/1e9:.1f}B" if info.get("marketCap") else "N/A"
    pe = f"{info.get('trailingPE'):.1f}" if isinstance(info.get("trailingPE"), float) else "N/A"
    week_high = info.get("fiftyTwoWeekHigh", "N/A")
    week_low = info.get("fiftyTwoWeekLow", "N/A")
    sector = info.get("sector", "")

    st_obj.markdown(f"""
<div class="company-name">{info.get('shortName', ticker)} &nbsp;·&nbsp; {ticker}
  {f'&nbsp;·&nbsp; {sector}' if sector else ''}
</div>
<div class="price-header">
  <span class="price-big">${close_price:.2f}</span>
  <span class="price-change" style="color:{price_color}">{price_sign}{price_change:.2f} ({price_sign}{price_change_pct:.2f}%)</span>
</div>
<div style="margin-bottom:20px">
  <span class="meta-chip">Mkt Cap {mc}</span>
  <span class="meta-chip">P/E {pe}</span>
  <span class="meta-chip">52W {week_low} – {week_high}</span>
</div>
""", unsafe_allow_html=True)

    # Earnings warning banner
    if earnings_info.get("warning"):
        st_obj.warning(f"⚠️ **Earnings Alert:** {earnings_info['text']} (Date: {earnings_info['date']})")
    elif earnings_info.get("days_away") and earnings_info["days_away"] <= 30:
        st_obj.info(f"📅 {earnings_info['text']}")

    # ── SECTION 2: VERDICT HERO ───────────────────────────────────────────────────
    # Build 3 top signal bullets (most impactful signals across all layers)
    all_signals = tech_summary["signals"] + fund_signals
    top_bullets = []
    for label, status, text in all_signals:
        if status in ("good", "warning") and len(top_bullets) < 3:
            icon = "✅" if status == "good" else "⚠️"
            top_bullets.append((icon, text))
    top_bullets = top_bullets[:3]

    bullets_html = "".join(
        f'<div class="reason-bullet"><span>{icon}</span><span>{text}</span></div>'
        for icon, text in top_bullets
    )

    # Score bars
    def bar_html(score, label, weight):
        pct = int((score + 1) / 2 * 100)
        clr = "#00c853" if score > 0.2 else "#ff1744" if score < -0.2 else "#ff9100"
        return (f'<div class="score-bar-item">'
                f'<div class="score-bar-label">{label} <span style="opacity:0.4">({weight})</span></div>'
                f'<div class="score-bar-track"><div class="score-bar-fill" style="width:{pct}%;background:{clr}"></div></div>'
                f'<div class="score-bar-value">{score:+.2f}</div>'
                f'</div>')

    bars = (
        bar_html(bd["technical"], "Technical", "40%") +
        bar_html(bd["fundamental"], "Fundamental", "35%") +
        bar_html(bd["sentiment"], "Sentiment", "25%")
    )

    st_obj.markdown(f"""
<div class="verdict-hero" style="background:{v_bg}; border: 1px solid {v_hex}30;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:24px;">
    <div>
      <div class="verdict-word" style="color:{v_hex}">{verdict_result["verdict"]}</div>
      <div class="verdict-confidence">Confidence {confidence_pct}%</div>
    </div>
    <div style="flex:1; min-width:260px; max-width:480px;">
      {bullets_html}
    </div>
  </div>
  <div class="score-bar-row" style="margin-top:24px">
    {bars}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── SECTION 3: SMART TRADE STRATEGY ──────────────────────────────────────────
    _rr = smart_trade["risk_reward"]
    _rr_color = "#00c853" if _rr >= 2 else "#ff9100" if _rr >= 1 else "#ff1744"
    _cv = smart_trade["conviction"]
    _cv_color = "#00c853" if _cv == "high" else "#ff9100" if _cv == "medium" else "#ff1744"
    _entry_disc = smart_trade["discount_pct"]
    st_obj.markdown(f"""
<div class="trade-box">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
    <div style="font-size:0.75rem;opacity:0.45;text-transform:uppercase;letter-spacing:0.06em">Smart Trade Strategy</div>
    <div style="font-size:0.75rem">Conviction: <span style="color:{_cv_color};font-weight:600">{_cv.upper()}</span>
    &nbsp;·&nbsp; Horizon: <span style="opacity:0.7">{smart_trade['time_horizon']}</span></div>
  </div>
  <div class="trade-row">
    <div class="trade-item">
      <div class="trade-label">Limit Entry</div>
      <div class="trade-value" style="color:#4fc3f7">${smart_trade['limit_entry']:.2f}</div>
      <div class="trade-sub">−{_entry_disc:.1f}% from now</div>
    </div>
    <div class="trade-item">
      <div class="trade-label">Stop Loss</div>
      <div class="trade-value" style="color:#ff1744">${smart_trade['stop_loss']:.2f}</div>
      <div class="trade-sub" style="font-size:0.7rem;opacity:0.5">{smart_trade['stop_loss_reason'][:30]}</div>
    </div>
    <div class="trade-item">
      <div class="trade-label">Take Profit 1</div>
      <div class="trade-value" style="color:#00c853">${smart_trade['take_profit_1']:.2f}</div>
      <div class="trade-sub">Take 50% here</div>
    </div>
    <div class="trade-item">
      <div class="trade-label">Take Profit 2</div>
      <div class="trade-value" style="color:#00c853">${smart_trade['take_profit_2']:.2f}</div>
      <div class="trade-sub">Let rest run</div>
    </div>
    <div class="trade-item">
      <div class="trade-label">Risk / Reward</div>
      <div class="trade-value" style="color:{_rr_color}">1 : {_rr}</div>
      <div class="trade-sub">{'Good' if _rr >= 2 else 'Acceptable' if _rr >= 1 else 'Poor'}</div>
    </div>
  </div>
  <div style="font-size:0.75rem;opacity:0.45;margin-top:10px">
    Entry rationale: {smart_trade['limit_entry_reason']}
  </div>
</div>
""", unsafe_allow_html=True)

    # Exit strategy
    with st_obj.expander("🚪 Exit Strategy & Conditions"):
        st_obj.markdown(f"**TP rationale:** {smart_trade['take_profit_reason']}" if smart_trade['take_profit_reason'] else "")
        for cond in smart_trade["exit_conditions"]:
            icon = "🛑" if "stop" in cond.lower() or "drops below" in cond.lower() else "💰" if "profit" in cond.lower() or "tp" in cond.lower() else "⚠️"
            safe_cond = cond.replace("$", "\\$")
            st_obj.markdown(f"{icon} {safe_cond}")

    # ── ALERT SUGGESTIONS ────────────────────────────────────────────────────
    with st_obj.expander(f"🧠 AI Thesis — {thesis.get('headline', ticker)}"):
        one_liner = thesis.get("one_liner", "")
        if one_liner:
            st_obj.markdown(f"*{one_liner}*")
            st_obj.divider()
        col_bull, col_bear = st_obj.columns(2)
        with col_bull:
            st_obj.caption("Bull case")
            for pt in thesis.get("bull_points", []):
                st_obj.markdown(pt)
        with col_bear:
            st_obj.caption("Bear case")
            for pt in thesis.get("bear_points", []):
                st_obj.markdown(pt)
        if thesis.get("key_catalyst") or thesis.get("key_risk"):
            st_obj.divider()
            kc1, kc2 = st_obj.columns(2)
            if thesis.get("key_catalyst"):
                kc1.markdown(f"**🚀 Key catalyst:** {thesis['key_catalyst']}")
            if thesis.get("key_risk"):
                kc2.markdown(f"**⚠️ Key risk:** {thesis['key_risk']}")

    if alert_suggestions:
        with st_obj.expander(f"🔔 Suggested Alerts for {ticker} ({len(alert_suggestions)} suggestions)"):
            st_obj.caption("Based on support/resistance levels and analyst targets. Click to add.")
            existing_alert_prices = {a["target_price"] for a in alerts_load() if a["ticker"] == ticker}
            for sug in alert_suggestions:
                priority_color = "#00c853" if sug["priority"] == "high" else "#ff9100" if sug["priority"] == "medium" else "#888"
                direction_arrow = "▲" if sug["direction"] == "above" else "▼"
                already_set = sug["target_price"] in existing_alert_prices
                c1, c2 = st_obj.columns([4, 1])
                c1.markdown(
                    f'<span style="color:{priority_color};font-size:0.8rem;font-weight:600">{sug["priority"].upper()}</span> '
                    f'<span style="font-family:\'JetBrains Mono\',monospace">{direction_arrow} ${sug["target_price"]}</span> '
                    f'— {sug["reason"]}',
                    unsafe_allow_html=True
                )
                if already_set:
                    c2.caption("✓ Set")
                elif c2.button("Add", key=f"sug_{sug['target_price']}_{sug['direction']}"):
                    alerts_add(ticker, sug["target_price"], sug["direction"],
                               note=f"Auto: {sug['reason'][:50]}")
                    st_obj.session_state.alerts = alerts_load()
                    st_obj.rerun()

    # ── SECTION 4: DRILL-DOWNS ────────────────────────────────────────────────────
    with st_obj.expander("📈 Price chart & technical indicators"):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="OHLC"
        ))
        # Volume bars on secondary y-axis
        vol_colors = ["rgba(0,200,83,0.25)" if df["close"].iloc[i] >= df["open"].iloc[i] else "rgba(255,23,68,0.25)"
                      for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
                             marker_color=vol_colors, yaxis="y2", showlegend=False))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                      showticklabels=False, range=[0, df["volume"].max() * 5]))

        if show_sma:
            fig.add_trace(go.Scatter(x=df.index, y=df["sma20"], name="SMA20", line=dict(color="#ff9100", width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df["sma50"], name="SMA50", line=dict(color="#4fc3f7", width=1)))
        if show_bb:
            fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper", line=dict(color="rgba(150,150,150,0.5)", dash="dot", width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower", line=dict(color="rgba(150,150,150,0.5)", dash="dot", width=1), fill="tonexty", fillcolor="rgba(150,150,150,0.05)"))
        # S/R lines
        for r in swings["resistance"]:
            fig.add_hline(y=r, line_dash="dash", line_color="rgba(255,71,87,0.4)", annotation_text=f"R {r}", annotation_position="right")
        for s in swings["support"]:
            fig.add_hline(y=s, line_dash="dash", line_color="rgba(0,200,83,0.4)", annotation_text=f"S {s}", annotation_position="right")
        fig.update_layout(
            xaxis_rangeslider_visible=False, height=480,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk"), margin=dict(t=20, b=20),
            legend=dict(orientation="h", y=1.02)
        )
        st_obj.plotly_chart(fig, width="stretch")

        col_rsi, col_macd = st_obj.columns(2)
        with col_rsi:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="#b39ddb", width=1.5)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="rgba(255,71,87,0.5)", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="rgba(0,200,83,0.5)", annotation_text="Oversold")
            fig_rsi.update_layout(
                title="RSI", height=220, margin=dict(t=30,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Space Grotesk")
            )
            st_obj.plotly_chart(fig_rsi, width="stretch")
        with col_macd:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="#4fc3f7", width=1.5)))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="#ff7043", width=1.5)))
            fig_macd.update_layout(
                title="MACD", height=220, margin=dict(t=30,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Space Grotesk")
            )
            st_obj.plotly_chart(fig_macd, width="stretch")

        st_obj.divider()
        st_obj.caption("Technical signals")
        for label, status, text in tech_summary["signals"]:
            st_obj.markdown(f"{icon_map.get(status, '➖')} **{label}:** {text}")

    with st_obj.expander("🏢 Fundamentals — is the company healthy?"):
        if fund_signals:
            for label, status, text in fund_signals:
                st_obj.markdown(f"{icon_map.get(status, '➖')} **{label}:** {text}")
        else:
            st_obj.info("No fundamental data available.")

        # Sector comparison
        if sector:
            st_obj.divider()
            st_obj.caption("Sector comparison")
            with st_obj.spinner(f"Comparing {ticker} vs {sector} peers..."):
                sec = get_sector_comparison(ticker, sector)
            if sec.get("signal"):
                s_status, s_label, s_text = sec["signal"]
                st_obj.markdown(f"{icon_map.get(s_status, '➖')} **{s_label}:** {s_text}")
                median_pe = sec.get('sector_median_pe')
                median_pe_str = f"{median_pe:.1f}" if isinstance(median_pe, (int, float)) else "N/A"
                st_obj.caption(f"Based on {sec.get('peers_used', 0)} sector peers · Sector median P/E: {median_pe_str}")

        st_obj.divider()
        fmt_map = {
            "pe_ratio": ("P/E Ratio", lambda v: f"{v:.1f}"),
            "forward_pe": ("Forward P/E", lambda v: f"{v:.1f}"),
            "eps": ("EPS", lambda v: f"${v:.2f}"),
            "revenue": ("Revenue", lambda v: f"${v/1e9:.2f}B"),
            "revenue_growth": ("Revenue Growth", lambda v: f"{v*100:.1f}%"),
            "earnings_growth": ("Earnings Growth", lambda v: f"{v*100:.1f}%"),
            "profit_margin": ("Profit Margin", lambda v: f"{v*100:.1f}%"),
            "roe": ("Return on Equity", lambda v: f"{v*100:.1f}%"),
            "debt_to_equity": ("Debt / Equity", lambda v: f"{v/100:.2f}"),
            "current_ratio": ("Current Ratio", lambda v: f"{v:.2f}"),
            "free_cash_flow": ("Free Cash Flow", lambda v: f"${v/1e9:.2f}B"),
            "dividend_yield": ("Dividend Yield", lambda v: f"{v:.2f}%" if v < 1 else f"{v/100:.2f}%"),
            "beta": ("Beta", lambda v: f"{v:.2f}"),
            "short_ratio": ("Short Ratio", lambda v: f"{v:.1f} days"),
        }
        fundamentals = analysis["fundamentals"]
        rows = []
        for key, (label, fmt) in fmt_map.items():
            val = fundamentals.get(key)
            if val is not None:
                try:
                    rows.append({"Metric": label, "Value": fmt(val)})
                except Exception:
                    pass
        if rows:
            st_obj.dataframe(pd.DataFrame(rows).set_index("Metric"), width="stretch")

        # ── Balance sheet trends ──
        if balance_data.get("years"):
            st_obj.divider()
            st_obj.caption("Balance sheet trends (YoY)")
            from modules.balance_sheet_trends import score_balance_sheet
            bs_label, bs_status, bs_text = score_balance_sheet(balance_data)
            st_obj.markdown(f"{icon_map.get(bs_status, '➖')} {bs_text}")

            years = balance_data["years"]
            debt = balance_data.get("total_debt", [])
            cash = balance_data.get("cash", [])
            revenue = balance_data.get("revenue", [])

            fig_bs = go.Figure()
            if any(v is not None for v in debt):
                fig_bs.add_trace(go.Bar(name="Total Debt ($B)", x=years, y=debt, marker_color="rgba(255,71,87,0.7)"))
            if any(v is not None for v in cash):
                fig_bs.add_trace(go.Bar(name="Cash ($B)", x=years, y=cash, marker_color="rgba(0,200,83,0.7)"))
            if any(v is not None for v in revenue):
                fig_bs.add_trace(go.Scatter(name="Revenue ($B)", x=years, y=revenue,
                                            line=dict(color="#4fc3f7", width=2), yaxis="y2"))
            fig_bs.update_layout(
                barmode="group", height=220, margin=dict(t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Space Grotesk"),
                legend=dict(orientation="h", y=1.1),
                xaxis=dict(type="category"),
                yaxis=dict(title="$B"),
                yaxis2=dict(overlaying="y", side="right", title="Revenue $B", showgrid=False),
            )
            st_obj.plotly_chart(fig_bs, width="stretch")

        # ── Piotroski F-Score ──
        if piotroski_data.get("score") is not None:
            st_obj.divider()
            st_obj.caption("Piotroski F-Score — financial health (0-9)")
            from modules.piotroski import score_piotroski
            p_label, p_status, p_text = score_piotroski(piotroski_data)
            st_obj.markdown(f"{icon_map.get(p_status, '➖')} {p_text}")
            components = piotroski_data.get("components", {})
            if components:
                criteria_groups = {
                    "Profitability": ["roa_positive", "ocf_positive", "roa_improving", "accruals"],
                    "Leverage / Liquidity": ["debt_ratio_decreasing", "current_ratio_improving", "no_new_shares"],
                    "Efficiency": ["gross_margin_improving", "asset_turnover_improving"],
                }
                cols = st_obj.columns(3)
                for i, (group, keys) in enumerate(criteria_groups.items()):
                    with cols[i]:
                        st_obj.caption(group)
                        for k in keys:
                            val = components.get(k)
                            icon = "✅" if val else "❌"
                            _plabels = {"roa_positive": "ROA Positive", "ocf_positive": "OCF Positive", "roa_improving": "ROA Improving YoY", "accruals": "Cash Earnings Quality", "debt_ratio_decreasing": "Debt Ratio Decreasing", "current_ratio_improving": "Current Ratio Improving", "no_new_shares": "No Share Dilution", "gross_margin_improving": "Gross Margin Improving", "asset_turnover_improving": "Asset Turnover Improving"}
                            label = _plabels.get(k, k.replace("_", " ").title())
                            st_obj.markdown(f"{icon} {label}")

        # ── Advanced valuation ──
        st_obj.divider()
        st_obj.caption("Advanced valuation")
        from modules.valuation_advanced import score_valuation_advanced
        av_label, av_status, av_text = score_valuation_advanced(valuation_adv)
        st_obj.markdown(f"{icon_map.get(av_status, '➖')} {av_text}")
        fcf_yield = valuation_adv.get("fcf_yield")
        ev_ebitda = valuation_adv.get("ev_ebitda")
        fcf_interp = valuation_adv.get("fcf_interpretation", "unknown")
        ev_interp = valuation_adv.get("ev_ebitda_interpretation", "unknown")
        interp_color = {"cheap": "#00c853", "fair": "#ff9100", "expensive": "#ff1744", "unknown": "#888"}
        if fcf_yield is not None or ev_ebitda is not None:
            st_obj.markdown(
                f'<div class="trade-box"><div class="trade-row">'
                f'<div class="trade-item"><div class="trade-label">FCF Yield</div>'
                f'<div class="trade-value" style="color:{interp_color.get(fcf_interp,"#888")}">'
                f'{f"{fcf_yield:.1f}%" if fcf_yield is not None else "N/A"}</div>'
                f'<div style="font-size:0.7rem;opacity:0.5">{fcf_interp}</div></div>'
                f'<div class="trade-item"><div class="trade-label">EV / EBITDA</div>'
                f'<div class="trade-value" style="color:{interp_color.get(ev_interp,"#888")}">'
                f'{f"{ev_ebitda:.1f}x" if ev_ebitda is not None else "N/A"}</div>'
                f'<div style="font-size:0.7rem;opacity:0.5">{ev_interp}</div></div>'
                f'</div></div>', unsafe_allow_html=True
            )

        # ── Altman Z-Score ──
        st_obj.divider()
        st_obj.caption("Altman Z-Score — bankruptcy risk")
        from modules.altman_z import score_altman_z
        az_label, az_status, az_text = score_altman_z(altman_data)
        st_obj.markdown(f"{icon_map.get(az_status, '➖')} {az_text}")
        az_score = altman_data.get("z_score")
        az_zone = altman_data.get("zone", "unknown")
        if az_score is not None:
            zone_color = {"safe": "#00c853", "grey": "#ff9100", "distress": "#ff1744"}.get(az_zone, "#888")
            bar_pct = min(100, max(0, (az_score / 5.0) * 100))
            st_obj.markdown(
                f'<div style="margin:8px 0">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:1.2rem;font-weight:600;color:{zone_color}">{az_score:.2f}</span>'
                f'<span style="color:{zone_color};font-size:0.85rem;font-weight:600">{az_zone.upper()}</span></div>'
                f'<div style="background:rgba(255,255,255,0.08);border-radius:4px;height:6px">'
                f'<div style="width:{bar_pct:.0f}%;background:{zone_color};height:6px;border-radius:4px"></div></div>'
                f'<div style="display:flex;justify-content:space-between;font-size:0.65rem;opacity:0.4;margin-top:2px">'
                f'<span>Distress &lt;1.81</span><span>Grey 1.81–2.99</span><span>Safe &gt;2.99</span></div>'
                f'</div>', unsafe_allow_html=True
            )

        # ── Momentum across timeframes ──
        st_obj.divider()
        st_obj.caption("Price momentum")
        from modules.momentum import score_momentum
        mo_label, mo_status, mo_text = score_momentum(momentum_data)
        st_obj.markdown(f"{icon_map.get(mo_status, '➖')} {mo_text}")
        mo_fields = [("1 Month", "ret_1mo"), ("3 Month", "ret_3mo"), ("6 Month", "ret_6mo"), ("1 Year", "ret_1yr")]
        mo_cols = st_obj.columns(4)
        for i, (lbl, key) in enumerate(mo_fields):
            val = momentum_data.get(key)
            if val is not None:
                c = "#00c853" if val >= 0 else "#ff1744"
                s = "+" if val >= 0 else ""
                mo_cols[i].markdown(
                    f'<div style="text-align:center;padding:8px;background:rgba(255,255,255,0.03);border-radius:8px">'
                    f'<div style="font-size:0.7rem;opacity:0.45">{lbl}</div>'
                    f'<div style="color:{c};font-family:\'JetBrains Mono\',monospace;font-weight:600">{s}{val:.1f}%</div>'
                    f'</div>', unsafe_allow_html=True
                )

    with st_obj.expander("🔬 Analyst, Insider & Market Intelligence"):
        # ── Analyst targets ──
        st_obj.caption("Analyst coverage")
        from modules.analyst import score_analyst
        targets = analyst_data.get("price_targets", {})
        consensus = analyst_data.get("consensus", "N/A")
        count = analyst_data.get("analyst_count", 0)
        if targets.get("mean"):
            upside = targets.get("upside_pct", 0) or 0
            u_color = "#00c853" if upside >= 0 else "#ff1744"
            u_sign = "+" if upside >= 0 else ""
            st_obj.markdown(f"""
<div class="trade-box" style="margin-bottom:12px">
  <div class="trade-row">
    <div class="trade-item"><div class="trade-label">Consensus ({count} analysts)</div>
      <div class="trade-value">{consensus}</div></div>
    <div class="trade-item"><div class="trade-label">Avg Target</div>
      <div class="trade-value">${targets['mean']}</div></div>
    <div class="trade-item"><div class="trade-label">Upside</div>
      <div class="trade-value" style="color:{u_color}">{u_sign}{upside:.1f}%</div></div>
    <div class="trade-item"><div class="trade-label">High Target</div>
      <div class="trade-value" style="color:#00c853">${targets.get('high','N/A')}</div></div>
    <div class="trade-item"><div class="trade-label">Low Target</div>
      <div class="trade-value" style="color:#ff1744">${targets.get('low','N/A')}</div></div>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st_obj.info("No analyst price targets available.")

        # ── Options sentiment ──
        st_obj.divider()
        st_obj.caption("Options market sentiment (contrarian)")
        from modules.options_sentiment import score_options
        pcr = options_data.get("put_call_ratio")
        if pcr is not None:
            o_sig, o_status, o_text = score_options(options_data)
            st_obj.markdown(f"{icon_map.get(o_status, '➖')} **Put/Call Ratio {pcr}** — {o_text}")
            st_obj.caption("Ratio > 1.2 = fearful market (contrarian buy). Ratio < 0.6 = greedy market (contrarian caution).")
        else:
            st_obj.info("No options data available.")

        # ── Insider transactions ──
        st_obj.divider()
        st_obj.caption("Recent insider transactions")
        from modules.insider import score_insider
        insider_txns = insider_data.get("transactions", [])
        ins_sig, ins_status, ins_text = score_insider(insider_data)
        st_obj.markdown(f"{icon_map[ins_status]} {ins_text}")
        if insider_txns:
            for txn in insider_txns[:5]:
                action_color = "#00c853" if txn["action"] == "BUY" else "#ff1744" if txn["action"] == "SELL" else "#888"
                shares_str = f"{txn['shares']:,} shares" if txn["shares"] else ""
                st_obj.markdown(
                    f'<span style="color:{action_color};font-weight:600">{txn["action"]}</span> '
                    f'— **{txn["insider"]}** ({txn["position"]}) {shares_str}',
                    unsafe_allow_html=True
                )

        # ── Institutional ownership ──
        st_obj.divider()
        st_obj.caption("Institutional ownership")
        from modules.ownership import score_ownership
        own_sig, own_status, own_text = score_ownership(ownership_data)
        st_obj.markdown(f"{icon_map[own_status]} {own_text}")
        top_holders = ownership_data.get("top_holders", [])
        if top_holders:
            holder_rows = [{"Institution": h["name"], "% Held": f"{h['pct_held']:.2f}%" if h.get("pct_held") else "N/A"}
                           for h in top_holders]
            st_obj.dataframe(pd.DataFrame(holder_rows).set_index("Institution"), width="stretch")

        # ── Short interest ──
        st_obj.divider()
        st_obj.caption("Short interest")
        from modules.short_interest import score_short_interest
        si_label, si_status, si_text = score_short_interest(short_data)
        st_obj.markdown(f"{icon_map[si_status]} {si_text}")
        pct_float = short_data.get("short_pct_float")
        short_ratio = short_data.get("short_ratio")
        shares_short = short_data.get("shares_short")
        squeeze = short_data.get("squeeze_potential", "unknown")
        if pct_float is not None:
            squeeze_color = "#ff1744" if squeeze == "high" else "#ff9100" if squeeze == "moderate" else "#00c853"
            st_obj.markdown(
                f'<div class="trade-box"><div class="trade-row">'
                f'<div class="trade-item"><div class="trade-label">Short % of Float</div><div class="trade-value">{pct_float:.1f}%</div></div>'
                f'<div class="trade-item"><div class="trade-label">Days to Cover</div><div class="trade-value">{short_ratio if short_ratio else "N/A"}</div></div>'
                f'<div class="trade-item"><div class="trade-label">Shares Short</div><div class="trade-value">{f"{shares_short:,}" if shares_short else "N/A"}</div></div>'
                f'<div class="trade-item"><div class="trade-label">Squeeze Potential</div><div class="trade-value" style="color:{squeeze_color}">{squeeze.upper()}</div></div>'
                f'</div></div>', unsafe_allow_html=True
            )

        # ── Earnings surprise history ──
        st_obj.divider()
        st_obj.caption("Earnings surprise history (last 4 quarters)")
        from modules.earnings_history import score_earnings_history
        e_sig, e_status, e_text = score_earnings_history(earnings_hist)
        st_obj.markdown(f"{icon_map[e_status]} {e_text}")
        quarters = earnings_hist.get("quarters", [])
        if quarters:
            eq_cols = st_obj.columns(len(quarters))
            for i, q in enumerate(quarters):
                beat = q.get("beat")
                b_icon = "✅" if beat is True else "❌" if beat is False else "➖"
                surp = q.get("surprise_pct")
                surp_str = (f"+{surp:.1f}%" if surp and surp >= 0 else f"{surp:.1f}%") if surp is not None else "N/A"
                surp_color = "#00c853" if surp and surp > 0 else "#ff1744" if surp and surp < 0 else "#888"
                eq_cols[i].markdown(
                    f'<div style="text-align:center;padding:8px;background:rgba(255,255,255,0.03);border-radius:8px">'
                    f'<div style="font-size:0.7rem;opacity:0.45">{q["date"]}</div>'
                    f'<div style="font-size:1.3rem">{b_icon}</div>'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.85rem">EPS {q.get("actual","N/A")}</div>'
                    f'<div style="font-size:0.75rem;opacity:0.5">Est {q.get("estimate","N/A")}</div>'
                    f'<div style="color:{surp_color};font-size:0.8rem;font-weight:600">{surp_str}</div>'
                    f'</div>', unsafe_allow_html=True
                )

        # ── Market context (VIX + 52W rank) ──
        st_obj.divider()
        st_obj.caption("Market context")
        from modules.market_context import score_market_context
        mc_label, mc_status, mc_text = score_market_context(market_ctx)
        st_obj.markdown(f"{icon_map[mc_status]} {mc_text}")
        vix_val = market_ctx.get("vix")
        vix_regime = market_ctx.get("vix_regime", "unknown")
        w52_pct = market_ctx.get("week52_pct")
        vix_regime_color = {"high_fear": "#ff1744", "neutral": "#ff9100", "complacency": "#00c853"}.get(vix_regime, "#888")
        rank_color = "#00c853" if w52_pct and w52_pct > 60 else "#ff1744" if w52_pct and w52_pct < 20 else "#ff9100"
        if vix_val is not None or w52_pct is not None:
            st_obj.markdown(
                f'<div class="trade-box"><div class="trade-row">'
                f'<div class="trade-item"><div class="trade-label">VIX (Fear Gauge)</div>'
                f'<div class="trade-value" style="color:{vix_regime_color}">{f"{vix_val:.1f}" if vix_val else "N/A"}</div>'
                f'<div style="font-size:0.7rem;opacity:0.5">{vix_regime.replace("_"," ")}</div></div>'
                f'<div class="trade-item"><div class="trade-label">52W Rank</div>'
                f'<div class="trade-value" style="color:{rank_color}">{f"{w52_pct:.0f}%" if w52_pct is not None else "N/A"}</div>'
                f'<div style="font-size:0.7rem;opacity:0.5">of 52W range</div></div>'
                f'<div class="trade-item"><div class="trade-label">52W High</div>'
                f'<div class="trade-value">${market_ctx.get("week52_high","N/A")}</div></div>'
                f'<div class="trade-item"><div class="trade-label">52W Low</div>'
                f'<div class="trade-value">${market_ctx.get("week52_low","N/A")}</div></div>'
                f'</div></div>', unsafe_allow_html=True
            )

        # ── Sector momentum ──
        st_obj.divider()
        st_obj.caption("Sector trend")
        from modules.sector_momentum import score_sector_momentum
        sm_label, sm_status, sm_text = score_sector_momentum(sector_mom)
        st_obj.markdown(f"{icon_map[sm_status]} {sm_text}")
        sm_etf = sector_mom.get("etf")
        sm_ret = sector_mom.get("etf_1mo_return")
        if sm_etf and sm_ret is not None:
            ret_color = "#00c853" if sm_ret >= 0 else "#ff1744"
            ret_sign = "+" if sm_ret >= 0 else ""
            st_obj.caption(f"{sm_etf} 1-month return: {ret_sign}{sm_ret:.1f}%")

    with st_obj.expander("📈 Performance vs S&P 500"):
        from modules.relative_performance import score_relative_performance
        r_sig, r_status, r_text = score_relative_performance(rel_perf, ticker)
        st_obj.markdown(f"{icon_map[r_status]} {r_text}")

        if rel_series.get("dates"):
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(
                x=rel_series["dates"], y=rel_series["ticker_series"],
                name=ticker, line=dict(color="#4fc3f7", width=2)
            ))
            fig_rel.add_trace(go.Scatter(
                x=rel_series["dates"], y=rel_series["spy_series"],
                name="S&P 500", line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot")
            ))
            fig_rel.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.1)")
            fig_rel.update_layout(
                height=280, margin=dict(t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Space Grotesk"),
                legend=dict(orientation="h", y=1.02),
                yaxis_title="Base 100",
            )
            st_obj.plotly_chart(fig_rel, width="stretch")
        else:
            st_obj.info("Could not load comparison data.")

    with st_obj.expander("📰 News & sentiment"):
        if not scored_articles:
            st_obj.info("No recent news found.")
        else:
            sent_label, sent_color = overall_sentiment(scored_articles)
            sent_hex = color_map.get(sent_color, "#888")
            st_obj.markdown(f'<div style="color:{sent_hex};font-weight:600;margin-bottom:12px">{sent_label}</div>', unsafe_allow_html=True)
            for a in scored_articles:
                sentiment = a.get("sentiment", "unknown").lower()
                score = a.get("sentiment_score", 0)
                icon = sent_icon.get(sentiment, "⚪")
                with st_obj.expander(f"{icon} {a['title']}"):
                    st_obj.caption(f"{a['source']}  ·  {a['published']}  ·  **{sentiment.upper()}** ({score:.0%})")
                    if a.get("summary"):
                        st_obj.write(a["summary"])
                    if a.get("url"):
                        st_obj.markdown(f"[Read article]({a['url']})")

    with st_obj.expander("🔁 Backtest — how did a simple strategy do?"):
        st_obj.caption("Buys when 20-day average crosses above 50-day average. Sells when it crosses below.")
        if not run_backtest:
            st_obj.info("Enable 'Run SMA Crossover' in the sidebar to run.")
        else:
            with st_obj.spinner("Running backtest..."):
                try:
                    pf = sma_crossover_backtest(df["close"], fast=int(fast_win), slow=int(slow_win))
                    stats = get_backtest_stats(pf)
                    total_return = float(stats.get("Total Return [%]", 0) or 0)
                    n_trades = int(stats.get("Total Trades", 0) or 0)
                    if n_trades == 0:
                        st_obj.warning(
                            f"No crossover trades triggered in this period. "
                            f"The SMA{int(fast_win)} never crossed the SMA{int(slow_win)} — "
                            f"stock was in a sustained trend. Try a longer period (1y or 2y)."
                        )
                    else:
                        st_obj.dataframe(stats.to_frame("Value").astype(str), width="stretch")
                        st_obj.plotly_chart(pf.plot(), width="stretch")
                except Exception as e:
                    st_obj.error(f"Backtest failed: {e}")
