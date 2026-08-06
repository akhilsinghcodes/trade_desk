"""Streamlit dashboard — Trade Lab."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.fetch import get_ohlcv, get_info
from modules.indicators import add_common_indicators
from modules.backtest import sma_crossover_backtest, get_backtest_stats
from modules.summary import summarize
from modules.fundamentals import get_fundamentals, score_fundamentals
from modules.news import overall_sentiment
from modules.score import combined_score
from modules.support_resistance import swing_levels, pivot_points, suggest_trade
from modules.earnings_history import score_earnings_history
from modules.relative_performance import score_relative_performance
from modules.analyst import score_analyst
from modules.insider import score_insider
from modules.ownership import score_ownership
from modules.options_sentiment import score_options
from modules.short_interest import score_short_interest
from modules.balance_sheet_trends import score_balance_sheet
from modules.volume import add_volume_indicators, volume_signal
from modules.earnings import get_next_earnings
from modules.sector import get_sector_comparison
from modules.alert_suggestions import suggest_alerts
from modules.notifications import alert_checker, notify_alert_triggered
from modules.market_context import score_market_context
from modules.piotroski import score_piotroski
from modules.valuation_advanced import score_valuation_advanced
from modules.sector_momentum import score_sector_momentum
from modules.dilution_risk import score_dilution, score_earnings_proximity, get_earnings_proximity
from modules.altman_z import score_altman_z
from modules.momentum import score_momentum
from modules.thesis import generate_thesis
from modules.trade_strategy import get_smart_trade_strategy as _smart_strat
from modules.eli5 import CATEGORIES, get_by_category, search_terms as eli5_search
from modules.db import (
    wl_load, wl_add, wl_remove, port_load, port_add, port_remove,
    alerts_load, alerts_add, alerts_remove, alerts_check,
    cache_invalidate_ticker, init_db,
)
from modules.cached_fetch import (
    cached_ohlcv, cached_info, cached_fundamentals,
    cached_analyst, cached_insider, cached_ownership, cached_options,
    cached_earnings_history, cached_short_interest, cached_balance_sheet,
    cached_rel_perf, cached_news,
    cached_market_context, cached_piotroski, cached_valuation_advanced,
    cached_sector_momentum, cached_dilution_risk,
    cached_altman_z, cached_momentum,
)
from modules.portfolio import portfolio_summary

st.set_page_config(page_title="Trade Lab", layout="wide", initial_sidebar_state="expanded")

# ── GLOBAL STYLES ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.verdict-hero {
    padding: 32px 36px;
    border-radius: 12px;
    margin-bottom: 24px;
}
.verdict-word {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(3rem, 8vw, 5.5rem);
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.03em;
}
.verdict-reason { font-size: 1rem; margin-top: 10px; opacity: 0.85; max-width: 65ch; }
.verdict-confidence { font-size: 0.85rem; margin-top: 6px; opacity: 0.55; font-family: 'JetBrains Mono', monospace; }

.price-header {
    display: flex; align-items: baseline; gap: 16px;
    margin-bottom: 4px;
}
.price-big {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem; font-weight: 600;
}
.price-change {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem; font-weight: 500;
}
.company-name { font-size: 0.9rem; opacity: 0.55; margin-bottom: 20px; }

.score-bar-row { display: flex; gap: 20px; margin: 20px 0; }
.score-bar-item { flex: 1; }
.score-bar-label { font-size: 0.75rem; opacity: 0.55; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.score-bar-track {
    height: 6px; background: rgba(255,255,255,0.08);
    border-radius: 3px; overflow: hidden; margin-bottom: 4px;
}
.score-bar-fill { height: 100%; border-radius: 3px; transition: width 0.6s ease; }
.score-bar-value { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; opacity: 0.7; }

.reason-bullet {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.9rem;
}
.reason-bullet:last-child { border-bottom: none; }

.trade-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 20px 24px;
    margin: 16px 0;
}
.trade-row { display: flex; gap: 0; }
.trade-item { flex: 1; padding: 8px 16px; border-right: 1px solid rgba(255,255,255,0.06); }
.trade-item:first-child { padding-left: 0; }
.trade-item:last-child { border-right: none; }
.trade-label { font-size: 0.7rem; opacity: 0.45; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
.trade-value { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 600; }
.trade-sub { font-size: 0.75rem; opacity: 0.5; margin-top: 2px; }

.meta-chip {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    margin-right: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* tighten Streamlit expander */
.streamlit-expanderHeader { font-size: 0.9rem !important; }
div[data-testid="stExpander"] { border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 8px !important; }

/* hide default streamlit header */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
color_map = {"green": "#00c853", "red": "#ff1744", "orange": "#ff9100"}

_TICKER_ALIASES = {"BRKB": "BRK-B", "BRKA": "BRK-A"}
def _norm_ticker(t: str) -> str:
    t = t.upper().strip()
    return _TICKER_ALIASES.get(t, t)
icon_map = {"good": "✅", "warning": "⚠️", "neutral": "➖", "unknown": "➖"}
sent_icon = {"positive": "🟢", "negative": "🔴", "neutral": "🟡", "unknown": "⚪"}

# ── INIT DB + PERSISTENT STATE ─────────────────────────────────────────────────
init_db()

# ── DISCLAIMER BANNER (shown once per session) ─────────────────────────────────
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

if not st.session_state.disclaimer_accepted:
    with st.container():
        st.markdown("""
        <div style="background:rgba(255,145,0,0.08);border:1px solid rgba(255,145,0,0.3);
                    border-radius:10px;padding:20px 24px;margin-bottom:24px">
          <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.1em;color:#ff9100;margin-bottom:8px">
            ⚠️ For Informational Use Only
          </div>
          <div style="font-size:0.85rem;opacity:0.85;line-height:1.6">
            TradeDesk is a <strong>personal research tool</strong>, not financial advice.
            Verdicts, trade levels, signals, and any output do <strong>not</strong> constitute
            a recommendation to buy, sell, or hold any security. All investment decisions
            are solely your own. Past performance does not guarantee future results.
            Investing involves risk, including possible loss of principal.
            Consult a licensed financial advisor before making investment decisions.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("I understand — continue to TradeDesk"):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    st.stop()

if "watchlist" not in st.session_state:
    st.session_state.watchlist = wl_load()
if "portfolio" not in st.session_state:
    st.session_state.portfolio = port_load()
if "alerts" not in st.session_state:
    st.session_state.alerts = alerts_load()

# Start background alert checker (desktop notifications every 5 min)
if not alert_checker.is_running:
    def _get_prices():
        tickers = list({a["ticker"] for a in alerts_load() if not a.get("triggered")})
        prices = {}
        for t in tickers:
            try:
                df = get_ohlcv(t, period="5d")
                prices[t] = float(df["close"].iloc[-1])
            except Exception:
                pass
        return prices
    alert_checker.start(_get_prices, alerts_check)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    pages = ["📊 Analyze", "⭐ Watchlist", "💼 Portfolio", "🔔 Alerts", "📚 ELI5"]
    default_page_idx = 0
    page = st.radio("Page", pages, label_visibility="collapsed", index=default_page_idx)
    if "page_override" in st.session_state:
        page = st.session_state.pop("page_override")
    st.divider()
    if page == "📊 Analyze":
        default_ticker = st.session_state.pop("analyze_ticker", "AAPL")
        ticker = _norm_ticker(st.text_input("Ticker", value=default_ticker))
        if st.button("⭐ Add to Watchlist"):
            if ticker and ticker not in st.session_state.watchlist:
                wl_add(ticker)
                st.session_state.watchlist = wl_load()
                st.success(f"{ticker} added!")
            else:
                st.info("Already in watchlist.")
        if st.button("🗑 Clear Cache", help="Force re-fetch all data for this ticker"):
            cache_invalidate_ticker(ticker)
            st.rerun()
        period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
        st.divider()
        st.caption("Chart options")
        show_bb = st.checkbox("Bollinger Bands", value=True)
        show_sma = st.checkbox("SMA 20/50", value=True)
        st.divider()
        st.caption("Backtest")
        run_backtest = st.checkbox("Run SMA Crossover", value=True)
        fast_win = st.number_input("Fast SMA", value=20, min_value=2)
        slow_win = st.number_input("Slow SMA", value=50, min_value=3)

# ══════════════════════════════════════════════════════════════════════════════
# WATCHLIST PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "⭐ Watchlist":
    from streamlit_autorefresh import st_autorefresh
    col_wl_title, col_wl_refresh = st.columns([3, 1])
    col_wl_title.title("Watchlist")
    refresh_interval = col_wl_refresh.selectbox(
        "Auto-refresh", ["Off", "1 min", "5 min", "15 min"],
        index=2, label_visibility="collapsed"
    )
    interval_map = {"1 min": 60_000, "5 min": 300_000, "15 min": 900_000}
    if refresh_interval != "Off":
        st_autorefresh(interval=interval_map[refresh_interval], limit=None, key="wl_autorefresh")

    col_add, col_btn = st.columns([4, 1])
    with col_add:
        new_ticker = st.text_input("Add ticker", placeholder="e.g. NVDA", label_visibility="collapsed").upper().strip()
    with col_btn:
        if st.button("Add") and new_ticker:
            if new_ticker not in st.session_state.watchlist:
                wl_add(new_ticker)
                st.session_state.watchlist = wl_load()
                st.rerun()

    if not st.session_state.watchlist:
        st.info("No tickers yet. Add one above.")
        st.stop()

    st.divider()
    verdict_icons = {"BUY": "🟢", "HOLD / WATCH": "🟡", "SELL / AVOID": "🔴"}
    to_remove = None

    for t in st.session_state.watchlist:
        with st.spinner(f"Loading {t}..."):
            try:
                _df = get_ohlcv(t, period="3mo")
                _info = get_info(t)
                _df = add_common_indicators(_df)
                _fund = score_fundamentals(get_fundamentals(t))
                _tech = summarize(_df)
                _score = combined_score(_tech["signals"], _fund, [])
                close = _df["close"].iloc[-1]
                prev_close = _df["close"].iloc[-2]
                chg = (close - prev_close) / prev_close * 100
                chg_color = "#00c853" if chg > 0 else "#ff1744"
                chg_str = f"+{chg:.2f}%" if chg > 0 else f"{chg:.2f}%"
                mc = f"${_info.get('marketCap',0)/1e9:.1f}B" if _info.get("marketCap") else "N/A"
                pe = f"{_info.get('trailingPE'):.1f}" if _info.get("trailingPE") else "N/A"
                v = _score["verdict"]
                conf = int(_score["confidence"] * 100)
                v_hex = color_map.get(_score["color"], "#888")

                c1, c2, c3, c4, c5, c6, c7 = st.columns([1.2, 3, 1.5, 1.5, 2.5, 0.8, 0.5])
                if c1.button(f"**{t}**", key=f"analyze_{t}", help=f"Analyze {t}"):
                    st.session_state["analyze_ticker"] = t
                    st.session_state["page_override"] = "📊 Analyze"
                    st.rerun()
                c2.caption(_info.get("shortName", t))
                c3.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace;font-weight:600">${close:.2f}</span>', unsafe_allow_html=True)
                c4.markdown(f'<span style="color:{chg_color};font-family:\'JetBrains Mono\',monospace">{chg_str}</span>', unsafe_allow_html=True)
                c5.markdown(f'<span style="color:{v_hex};font-weight:600">{verdict_icons.get(v,"⚪")} {v}</span> <span style="opacity:0.45;font-size:0.8rem">({conf}%)</span>', unsafe_allow_html=True)
                c6.markdown(f'<span style="opacity:0.4;font-size:0.75rem">P/E {pe}</span>', unsafe_allow_html=True)
                if c7.button("✕", key=f"rm_{t}"):
                    to_remove = t
            except Exception:
                st.error(f"Failed to load {t}")
        st.divider()

    if to_remove:
        wl_remove(to_remove)
        st.session_state.watchlist = wl_load()
        st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "💼 Portfolio":
    st.title("Portfolio")

    with st.expander("➕ Add a position", expanded=False):
        with st.form("add_position_form"):
            pc1, pc2, pc3, pc4 = st.columns([1.5, 1, 1.5, 2])
            p_ticker = pc1.text_input("Ticker").upper().strip()
            p_shares = pc2.number_input("Shares", min_value=0.0001, value=1.0, step=0.001, format="%.4f")
            p_price  = pc3.number_input("Buy Price ($)", min_value=0.01, value=100.0)
            p_note   = pc4.text_input("Note (optional)")
            if st.form_submit_button("Add Position") and p_ticker:
                port_add(p_ticker, p_shares, p_price, note=p_note)
                st.session_state.portfolio = port_load()
                st.rerun()

    positions = st.session_state.portfolio
    if not positions:
        st.info("No positions yet. Add one above.")
        st.stop()

    st.divider()

    # Fetch current prices + previous close for all positions (cached)
    tickers_needed = [p["ticker"] for p in positions]
    current_prices = {}
    prev_prices = {}
    with st.spinner("Fetching current prices..."):
        for t in tickers_needed:
            try:
                _df = cached_ohlcv(t, period="5d")
                current_prices[t] = float(_df["close"].iloc[-1])
                if len(_df) >= 2:
                    prev_prices[t] = float(_df["close"].iloc[-2])
            except Exception:
                pass

    summary = portfolio_summary(positions, current_prices)

    # Compute total daily change across all positions
    _total_daily = 0.0
    for _pos in summary["positions"]:
        _t = _pos["ticker"]
        _prev = prev_prices.get(_t)
        if _prev and _prev != 0:
            _total_daily += (_pos["current_price"] - _prev) * _pos["shares"]
    _day_color = "#00c853" if _total_daily >= 0 else "#ff1744"
    _day_sign = "+" if _total_daily >= 0 else ""

    # Summary row
    total_pnl = summary["total_pnl_dollars"]
    total_pnl_pct = summary["total_pnl_pct"]
    pnl_color = "#00c853" if total_pnl >= 0 else "#ff1744"
    pnl_sign = "+" if total_pnl >= 0 else ""
    st.markdown(f"""
    <div class="trade-box" style="margin-bottom:24px">
      <div class="trade-row">
        <div class="trade-item"><div class="trade-label">Total Invested</div>
          <div class="trade-value">${summary['total_cost']:,.2f}</div></div>
        <div class="trade-item"><div class="trade-label">Current Value</div>
          <div class="trade-value">${summary['total_value']:,.2f}</div></div>
        <div class="trade-item"><div class="trade-label">Today's Change</div>
          <div class="trade-value" style="color:{_day_color}">{_day_sign}${_total_daily:,.2f}</div></div>
        <div class="trade-item"><div class="trade-label">Total P&L</div>
          <div class="trade-value" style="color:{pnl_color}">{pnl_sign}${total_pnl:,.2f}</div></div>
        <div class="trade-item"><div class="trade-label">Return</div>
          <div class="trade-value" style="color:{pnl_color}">{pnl_sign}{total_pnl_pct:.2f}%</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Allocation pie chart
    if summary["positions"]:
        _palette = ["#4fc3f7","#ff9100","#b39ddb","#00c853","#ff7043","#f06292",
                    "#ffeb3b","#26c6da","#ef5350","#ab47bc","#66bb6a","#ffa726"]
        pie_labels = [p["ticker"] for p in summary["positions"]]
        pie_values = [p["current_value"] for p in summary["positions"]]
        pie_colors = [_palette[i % len(_palette)] for i in range(len(pie_labels))]
        n = len(pie_labels)
        fig_pie = go.Figure(go.Pie(
            labels=pie_labels, values=pie_values,
            hole=0.55,
            marker=dict(colors=pie_colors),
            textinfo="label+percent" if n <= 8 else "percent",
            textposition="outside" if n <= 8 else "inside",
            textfont=dict(family="Space Grotesk", size=11),
            insidetextorientation="radial",
        ))
        fig_pie.update_layout(
            showlegend=True if n > 8 else False,
            legend=dict(
                orientation="v", x=1.02, y=0.5,
                font=dict(family="Space Grotesk", size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=320,
            margin=dict(t=20, b=20, l=20, r=160 if n > 8 else 20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"${summary['total_value']:,.0f}", x=0.5, y=0.5,
                              font=dict(size=16, family="JetBrains Mono"), showarrow=False)]
        )
        st.plotly_chart(fig_pie, width="stretch")

    # Column headers
    _h1, _h2, _h3, _h4, _h5, _h6, _h7, _h8, _h9 = st.columns([1, 1.5, 1, 1.3, 1.3, 1.3, 1.5, 1.1, 0.5])
    _h1.caption("Ticker")
    _h2.caption("Buy → Now")
    _h3.caption("Shares")
    _h4.caption("Cost Basis")
    _h5.caption("Value")
    _h6.caption("Today's Chg")
    _h7.caption("Total P&L")
    _h8.caption("Signal")

    to_remove = None
    total_daily_chg = 0.0
    for pnl in summary["positions"]:
        t = pnl["ticker"]
        p_color = "#00c853" if pnl["pnl_dollars"] >= 0 else "#ff1744"
        p_sign = "+" if pnl["pnl_dollars"] >= 0 else ""

        # Daily change
        shares = pnl["shares"]
        curr_p = pnl["current_price"]
        prev_p = prev_prices.get(t)
        if prev_p and prev_p != 0:
            day_chg_dollars = (curr_p - prev_p) * shares
            day_chg_pct = (curr_p - prev_p) / prev_p * 100
            total_daily_chg += day_chg_dollars
            day_color = "#00c853" if day_chg_dollars >= 0 else "#ff1744"
            day_sign = "+" if day_chg_dollars >= 0 else ""
            day_str = f'{day_sign}${day_chg_dollars:.2f} ({day_sign}{day_chg_pct:.1f}%)'
        else:
            day_color = "#888"
            day_str = "—"

        # Quick verdict for this position
        try:
            _df2 = cached_ohlcv(t, period="3mo")
            _df2 = add_common_indicators(_df2)
            _fund2 = score_fundamentals(cached_fundamentals(t))
            _tech2 = summarize(_df2)
            _sc2 = combined_score(_tech2["signals"], _fund2, [])
            v2 = _sc2["verdict"]
            v2_hex = color_map.get(_sc2["color"], "#888")
            verdict_chip = f'<span style="color:{v2_hex};font-weight:600;font-size:0.85rem">{v2}</span>'
        except Exception:
            verdict_chip = '<span style="opacity:0.4;font-size:0.85rem">—</span>'

        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns([1, 1.5, 1, 1.3, 1.3, 1.3, 1.5, 1.1, 0.5])
        if c1.button(f"**{t}**", key=f"port_analyze_{t}", help=f"Analyze {t}"):
            st.session_state["analyze_ticker"] = t
            st.session_state["page_override"] = "📊 Analyze"
            st.rerun()
        c2.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace">${pnl["buy_price"]:.2f} → ${pnl["current_price"]:.2f}</span>', unsafe_allow_html=True)
        c3.caption(f'{pnl["shares"]:g}')
        c4.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace">${pnl["cost_basis"]:,.2f}</span>', unsafe_allow_html=True)
        c5.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace">${pnl["current_value"]:,.2f}</span>', unsafe_allow_html=True)
        c6.markdown(f'<span style="color:{day_color};font-family:\'JetBrains Mono\',monospace;font-size:0.85rem">{day_str}</span>', unsafe_allow_html=True)
        c7.markdown(f'<span style="color:{p_color};font-family:\'JetBrains Mono\',monospace">{p_sign}${pnl["pnl_dollars"]:.2f} ({p_sign}{pnl["pnl_pct"]:.1f}%)</span>', unsafe_allow_html=True)
        c8.markdown(verdict_chip, unsafe_allow_html=True)
        if c9.button("✕", key=f"rm_pos_{t}"):
            to_remove = t
        st.divider()

    if to_remove:
        port_remove(to_remove)
        st.session_state.portfolio = port_load()
        st.rerun()

    # ── Correlation matrix ──
    port_tickers = [p["ticker"] for p in summary["positions"]]
    if len(port_tickers) >= 2:
        st.divider()
        st.subheader("📐 Holdings Correlation")
        st.markdown(
            "**How to read this:** Each cell shows how similarly two stocks move day-to-day (past 1 year). "
            "**+1.0** = always move together. **0.0** = move independently. **−1.0** = always move opposite. "
            "Green = good diversification. Red = overlapping bets — if one drops, the other likely does too.",
            unsafe_allow_html=False
        )
        with st.spinner("Computing correlations..."):
            try:
                _corr_dfs = [cached_ohlcv(t, "1y")[["close"]].rename(columns={"close": t}) for t in port_tickers]
                raw = pd.concat(_corr_dfs, axis=1).dropna()
                returns = raw.pct_change().dropna()
                corr = returns.corr()

                # Build heatmap
                z = corr.values.tolist()
                labels = corr.columns.tolist()
                text = [[f"{v:.2f}" for v in row] for row in z]

                fig_corr = go.Figure(go.Heatmap(
                    z=z, x=labels, y=labels, text=text, texttemplate="%{text}",
                    colorscale=[
                        [0.0, "rgba(255,23,68,0.8)"],
                        [0.5, "rgba(40,40,40,0.6)"],
                        [1.0, "rgba(0,200,83,0.8)"],
                    ],
                    zmin=-1, zmax=1,
                    showscale=True,
                    colorbar=dict(title="r", thickness=12),
                ))
                fig_corr.update_layout(
                    height=max(300, len(labels) * 60),
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Space Grotesk"),
                    xaxis=dict(side="bottom"),
                )
                st.plotly_chart(fig_corr, width="stretch")

                # Plain-English interpretation
                high_corr_pairs = []
                low_corr_pairs = []
                for i in range(len(labels)):
                    for j in range(i + 1, len(labels)):
                        r = corr.iloc[i, j]
                        if r > 0.75:
                            high_corr_pairs.append((labels[i], labels[j], r))
                        elif r < 0.25:
                            low_corr_pairs.append((labels[i], labels[j], r))

                if high_corr_pairs:
                    pairs_str = ", ".join(f"**{a}/{b}** ({r:.2f})" for a, b, r in high_corr_pairs)
                    st.warning(f"⚠️ High correlation (>0.75): {pairs_str} — these move together, limited diversification benefit.")
                if low_corr_pairs:
                    pairs_str = ", ".join(f"**{a}/{b}** ({r:.2f})" for a, b, r in low_corr_pairs)
                    st.success(f"✅ Low correlation (<0.25): {pairs_str} — good diversification.")
            except Exception as e:
                st.info(f"Could not compute correlations: {e}")

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ALERTS PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔔 Alerts":
    st.title("Price Alerts")
    st.caption("Desktop notifications fire every 5 min even when this tab is closed.")
    checker_status = "🟢 Running" if alert_checker.is_running else "🔴 Stopped"
    st.caption(f"Background checker: {checker_status}")

    with st.form("add_alert_form"):
        st.caption("Set a new alert")
        ac1, ac2, ac3, ac4 = st.columns([1.5, 1.5, 1.5, 2])
        a_ticker    = ac1.text_input("Ticker").upper().strip()
        a_price     = ac2.number_input("Target Price ($)", min_value=0.01, value=100.0)
        a_direction = ac3.selectbox("Trigger when price goes", ["above", "below"])
        a_note      = ac4.text_input("Note (optional)")
        if st.form_submit_button("Add Alert") and a_ticker:
            alerts_add(a_ticker, a_price, a_direction, note=a_note)
            st.session_state.alerts = alerts_load()
            st.rerun()

    alerts = st.session_state.alerts
    if not alerts:
        st.info("No alerts set. Add one above.")
        st.stop()

    st.divider()

    # Check alerts + send desktop notifications for newly triggered ones
    alert_tickers = list({a["ticker"] for a in alerts if not a.get("triggered")})
    current_prices = {}
    with st.spinner("Checking prices..."):
        for t in alert_tickers:
            try:
                _df = cached_ohlcv(t, "5d")
                current_prices[t] = float(_df["close"].iloc[-1])
            except Exception:
                pass

    triggered = alerts_check(current_prices)
    if triggered:
        for ta in triggered:
            st.success(f"🔔 **{ta['ticker']}** hit your target of ${ta['target_price']} ({ta['direction']})")
            notify_alert_triggered(ta["ticker"], ta["target_price"], ta["direction"],
                                   current_prices.get(ta["ticker"], ta["target_price"]))
        st.session_state.alerts = alerts_load()

    # Display all alerts
    st.subheader("Active Alerts")
    to_remove_alert_id = None
    for a in alerts:
        t = a["ticker"]
        cp = current_prices.get(t)
        cp_str = f"Current: ${cp:.2f}" if cp else "Price unavailable"
        status_icon = "✅" if a.get("triggered") else "🔔"
        triggered_label = " — **TRIGGERED**" if a.get("triggered") else ""
        c1, c2, c3, c4, c5 = st.columns([1, 1.5, 1.5, 2, 0.5])
        c1.markdown(f"**{t}**")
        c2.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace">${a["target_price"]:.2f} {a["direction"]}</span>', unsafe_allow_html=True)
        c3.caption(cp_str)
        c4.markdown(f"{status_icon} {a.get('note', '')}{triggered_label}")
        if c5.button("✕", key=f"rm_alert_{a['id']}"):
            to_remove_alert_id = a["id"]
        st.divider()

    if to_remove_alert_id is not None:
        alerts_remove(to_remove_alert_id)
        st.session_state.alerts = alerts_load()
        st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ELI5 PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "📚 ELI5":
    st.title("📚 Explain Like I'm 5")
    st.caption("Every term, chart, and signal in plain English. No jargon.")

    search_q = st.text_input("🔍 Search a term...", placeholder="e.g. RSI, short squeeze, P/E ratio")
    st.divider()

    category_colors = {
        "Technical": "#4fc3f7",
        "Fundamental": "#00c853",
        "Valuation": "#ff9100",
        "Market Intelligence": "#b39ddb",
        "Risk": "#ff1744",
        "Verdict": "#ffd740",
    }

    def render_term_card(term, data):
        cat = data.get("category", "")
        cat_color = category_colors.get(cat, "#888")
        with st.expander(f"{data.get('emoji','📌')} **{term}** — {data.get('full_name', '')}"):
            st.markdown(
                f'<span style="background:{cat_color}22;color:{cat_color};padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:600">{cat}</span>',
                unsafe_allow_html=True
            )
            st.markdown(f"### 🧒 Simple version\n{data['eli5']}")
            st.markdown(f"**What it really means:** {data['what_it_means']}")
            col_g, col_b = st.columns(2)
            col_g.success(f"✅ Good: {data['good_sign']}")
            col_b.error(f"⚠️ Watch out: {data['bad_sign']}")

    if search_q.strip():
        results = eli5_search(search_q.strip())
        if results:
            st.caption(f"{len(results)} result(s) for '{search_q}'")
            for term, data in results:
                render_term_card(term, data)
        else:
            st.info("No matching terms. Try a shorter keyword.")
    else:
        for cat in CATEGORIES:
            entries = get_by_category(cat)
            if not entries:
                continue
            cat_color = category_colors.get(cat, "#888")
            st.markdown(
                f'<h3 style="color:{cat_color};margin-top:24px">{cat}</h3>',
                unsafe_allow_html=True
            )
            for term, data in entries:
                render_term_card(term, data)

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ANALYZE PAGE
# ══════════════════════════════════════════════════════════════════════════════

# ── FETCH ALL DATA (cached) ────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker}..."):
    try:
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

    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

if df.empty:
    st.warning("No data returned. Check ticker symbol.")
    st.stop()

# News — use SQLite cache (handled inside cached_news)
scored_articles = cached_news(ticker, limit=10, company=info.get("shortName", ""))

# Compute scores
tech_summary = summarize(df)
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
verdict_result = combined_score(
    tech_signals=tech_summary["signals"],
    fund_signals=fund_signals,
    news_articles=scored_articles,
)
trade = suggest_trade(df, verdict_result["verdict"])
swings = swing_levels(df)
pivots = pivot_points(df)

# Smart trade strategy using all signals
_supports_flat  = sorted([s for s in (swings.get("support", []) + [pivots.get("s1"), pivots.get("s2")]) if s], reverse=True)
_resists_flat   = sorted([r for r in (swings.get("resistance", []) + [pivots.get("r1"), pivots.get("r2")]) if r])
_atr_val = trade.get("atr") or 0
if not _atr_val and len(df) >= 14:
    _hi = df["high"].astype(float)
    _lo = df["low"].astype(float)
    _cl = df["close"].astype(float).shift(1)
    _tr = (_hi - _lo).combine((_hi - _cl).abs(), max).combine((_lo - _cl).abs(), max)
    _atr_val = float(_tr.rolling(14).mean().iloc[-1] or 0)
smart_trade = _smart_strat(
    current_price       = float(df["close"].iloc[-1]),
    atr                 = float(_atr_val),
    confidence          = verdict_result["confidence"],
    verdict             = verdict_result["verdict"],
    breakdown           = verdict_result.get("breakdown", {}),
    support_levels      = _supports_flat,
    resistance_levels   = _resists_flat,
    pivot               = pivots.get("pivot"),
    analyst_target      = analyst_data.get("target_price"),
    analyst_upside_pct  = analyst_data.get("upside_pct"),
    week52_high         = market_ctx.get("week52_high"),
    week52_low          = market_ctx.get("week52_low"),
    week52_rank         = market_ctx.get("week52_rank"),
    piotroski_score     = piotroski_data.get("score", 0),
    altman_zone         = altman_data.get("zone", "unknown"),
    momentum_trend      = momentum_data.get("trend", "neutral"),
    ret_1yr             = momentum_data.get("ret_1yr"),
    short_pct           = short_data.get("short_pct_float"),
    earnings_days_away  = earnings_info.get("days_away"),
    sector_trend        = sector_mom.get("trend", "unknown"),
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

st.markdown(f"""
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
    st.warning(f"⚠️ **Earnings Alert:** {earnings_info['text']} (Date: {earnings_info['date']})")
elif earnings_info.get("days_away") and earnings_info["days_away"] <= 30:
    st.info(f"📅 {earnings_info['text']}")

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

st.markdown(f"""
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
st.markdown(f"""
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
with st.expander("🚪 Exit Strategy & Conditions"):
    st.markdown(f"**TP rationale:** {smart_trade['take_profit_reason']}" if smart_trade['take_profit_reason'] else "")
    for cond in smart_trade["exit_conditions"]:
        icon = "🛑" if "stop" in cond.lower() or "drops below" in cond.lower() else "💰" if "profit" in cond.lower() or "tp" in cond.lower() else "⚠️"
        # escape $ to prevent markdown from treating price ranges as math/italic
        safe_cond = cond.replace("$", "\\$")
        st.markdown(f"{icon} {safe_cond}")

# ── ALERT SUGGESTIONS ────────────────────────────────────────────────────────
with st.expander(f"🧠 AI Thesis — {thesis.get('headline', ticker)}"):
    one_liner = thesis.get("one_liner", "")
    if one_liner:
        st.markdown(f"*{one_liner}*")
        st.divider()
    col_bull, col_bear = st.columns(2)
    with col_bull:
        st.caption("Bull case")
        for pt in thesis.get("bull_points", []):
            st.markdown(pt)
    with col_bear:
        st.caption("Bear case")
        for pt in thesis.get("bear_points", []):
            st.markdown(pt)
    if thesis.get("key_catalyst") or thesis.get("key_risk"):
        st.divider()
        kc1, kc2 = st.columns(2)
        if thesis.get("key_catalyst"):
            kc1.markdown(f"**🚀 Key catalyst:** {thesis['key_catalyst']}")
        if thesis.get("key_risk"):
            kc2.markdown(f"**⚠️ Key risk:** {thesis['key_risk']}")

if alert_suggestions:
    with st.expander(f"🔔 Suggested Alerts for {ticker} ({len(alert_suggestions)} suggestions)"):
        st.caption("Based on support/resistance levels and analyst targets. Click to add.")
        existing_alert_prices = {a["target_price"] for a in alerts_load() if a["ticker"] == ticker}
        for sug in alert_suggestions:
            priority_color = "#00c853" if sug["priority"] == "high" else "#ff9100" if sug["priority"] == "medium" else "#888"
            direction_arrow = "▲" if sug["direction"] == "above" else "▼"
            already_set = sug["target_price"] in existing_alert_prices
            c1, c2 = st.columns([4, 1])
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
                st.session_state.alerts = alerts_load()
                st.rerun()

# ── SECTION 4: DRILL-DOWNS ────────────────────────────────────────────────────
with st.expander("📈 Price chart & technical indicators"):
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
    st.plotly_chart(fig, width="stretch")

    col_rsi, col_macd = st.columns(2)
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
        st.plotly_chart(fig_rsi, width="stretch")
    with col_macd:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="#4fc3f7", width=1.5)))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="#ff7043", width=1.5)))
        fig_macd.update_layout(
            title="MACD", height=220, margin=dict(t=30,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk")
        )
        st.plotly_chart(fig_macd, width="stretch")

    st.divider()
    st.caption("Technical signals")
    for label, status, text in tech_summary["signals"]:
        st.markdown(f"{icon_map.get(status, '➖')} **{label}:** {text}")

with st.expander("🏢 Fundamentals — is the company healthy?"):
    if fund_signals:
        for label, status, text in fund_signals:
            st.markdown(f"{icon_map.get(status, '➖')} **{label}:** {text}")
    else:
        st.info("No fundamental data available.")

    # Sector comparison
    if sector:
        st.divider()
        st.caption("Sector comparison")
        with st.spinner(f"Comparing {ticker} vs {sector} peers..."):
            sec = get_sector_comparison(ticker, sector)
        if sec.get("signal"):
            s_status, s_label, s_text = sec["signal"]
            st.markdown(f"{icon_map.get(s_status, '➖')} **{s_label}:** {s_text}")
            median_pe = sec.get('sector_median_pe')
            median_pe_str = f"{median_pe:.1f}" if isinstance(median_pe, (int, float)) else "N/A"
            st.caption(f"Based on {sec.get('peers_used', 0)} sector peers · Sector median P/E: {median_pe_str}")

    st.divider()
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
    rows = []
    for key, (label, fmt) in fmt_map.items():
        val = fundamentals.get(key)
        if val is not None:
            try:
                rows.append({"Metric": label, "Value": fmt(val)})
            except Exception:
                pass
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Metric"), width="stretch")

    # ── Balance sheet trends ──
    if balance_data.get("years"):
        st.divider()
        st.caption("Balance sheet trends (YoY)")
        bs_label, bs_status, bs_text = score_balance_sheet(balance_data)
        st.markdown(f"{icon_map.get(bs_status, '➖')} {bs_text}")

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
        st.plotly_chart(fig_bs, width="stretch")

    # ── Piotroski F-Score ──
    if piotroski_data.get("score") is not None:
        st.divider()
        st.caption("Piotroski F-Score — financial health (0-9)")
        p_label, p_status, p_text = score_piotroski(piotroski_data)
        st.markdown(f"{icon_map.get(p_status, '➖')} {p_text}")
        components = piotroski_data.get("components", {})
        if components:
            criteria_groups = {
                "Profitability": ["roa_positive", "ocf_positive", "roa_improving", "accruals"],
                "Leverage / Liquidity": ["debt_ratio_decreasing", "current_ratio_improving", "no_new_shares"],
                "Efficiency": ["gross_margin_improving", "asset_turnover_improving"],
            }
            cols = st.columns(3)
            for i, (group, keys) in enumerate(criteria_groups.items()):
                with cols[i]:
                    st.caption(group)
                    for k in keys:
                        val = components.get(k)
                        icon = "✅" if val else "❌"
                        _plabels = {"roa_positive": "ROA Positive", "ocf_positive": "OCF Positive", "roa_improving": "ROA Improving YoY", "accruals": "Cash Earnings Quality", "debt_ratio_decreasing": "Debt Ratio Decreasing", "current_ratio_improving": "Current Ratio Improving", "no_new_shares": "No Share Dilution", "gross_margin_improving": "Gross Margin Improving", "asset_turnover_improving": "Asset Turnover Improving"}
                        label = _plabels.get(k, k.replace("_", " ").title())
                        st.markdown(f"{icon} {label}")

    # ── Advanced valuation ──
    st.divider()
    st.caption("Advanced valuation")
    av_label, av_status, av_text = score_valuation_advanced(valuation_adv)
    st.markdown(f"{icon_map.get(av_status, '➖')} {av_text}")
    fcf_yield = valuation_adv.get("fcf_yield")
    ev_ebitda = valuation_adv.get("ev_ebitda")
    fcf_interp = valuation_adv.get("fcf_interpretation", "unknown")
    ev_interp = valuation_adv.get("ev_ebitda_interpretation", "unknown")
    interp_color = {"cheap": "#00c853", "fair": "#ff9100", "expensive": "#ff1744", "unknown": "#888"}
    if fcf_yield is not None or ev_ebitda is not None:
        st.markdown(
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
    st.divider()
    st.caption("Altman Z-Score — bankruptcy risk")
    az_label, az_status, az_text = score_altman_z(altman_data)
    st.markdown(f"{icon_map.get(az_status, '➖')} {az_text}")
    az_score = altman_data.get("z_score")
    az_zone = altman_data.get("zone", "unknown")
    if az_score is not None:
        zone_color = {"safe": "#00c853", "grey": "#ff9100", "distress": "#ff1744"}.get(az_zone, "#888")
        bar_pct = min(100, max(0, (az_score / 5.0) * 100))
        st.markdown(
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
    st.divider()
    st.caption("Price momentum")
    mo_label, mo_status, mo_text = score_momentum(momentum_data)
    st.markdown(f"{icon_map.get(mo_status, '➖')} {mo_text}")
    mo_fields = [("1 Month", "ret_1mo"), ("3 Month", "ret_3mo"), ("6 Month", "ret_6mo"), ("1 Year", "ret_1yr")]
    mo_cols = st.columns(4)
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

with st.expander("🔬 Analyst, Insider & Market Intelligence"):
    # ── Analyst targets ──
    st.caption("Analyst coverage")
    targets = analyst_data.get("price_targets", {})
    consensus = analyst_data.get("consensus", "N/A")
    count = analyst_data.get("analyst_count", 0)
    if targets.get("mean"):
        upside = targets.get("upside_pct", 0) or 0
        u_color = "#00c853" if upside >= 0 else "#ff1744"
        u_sign = "+" if upside >= 0 else ""
        st.markdown(f"""
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
        st.info("No analyst price targets available.")

    # ── Options sentiment ──
    st.divider()
    st.caption("Options market sentiment (contrarian)")
    pcr = options_data.get("put_call_ratio")
    if pcr is not None:
        o_sig, o_status, o_text = score_options(options_data)
        st.markdown(f"{icon_map.get(o_status, '➖')} **Put/Call Ratio {pcr}** — {o_text}")
        st.caption("Ratio > 1.2 = fearful market (contrarian buy). Ratio < 0.6 = greedy market (contrarian caution).")
    else:
        st.info("No options data available.")

    # ── Insider transactions ──
    st.divider()
    st.caption("Recent insider transactions")
    insider_txns = insider_data.get("transactions", [])
    ins_sig, ins_status, ins_text = score_insider(insider_data)
    st.markdown(f"{icon_map[ins_status]} {ins_text}")
    if insider_txns:
        for txn in insider_txns[:5]:
            action_color = "#00c853" if txn["action"] == "BUY" else "#ff1744" if txn["action"] == "SELL" else "#888"
            shares_str = f"{txn['shares']:,} shares" if txn["shares"] else ""
            st.markdown(
                f'<span style="color:{action_color};font-weight:600">{txn["action"]}</span> '
                f'— **{txn["insider"]}** ({txn["position"]}) {shares_str}',
                unsafe_allow_html=True
            )

    # ── Institutional ownership ──
    st.divider()
    st.caption("Institutional ownership")
    own_sig, own_status, own_text = score_ownership(ownership_data)
    st.markdown(f"{icon_map[own_status]} {own_text}")
    top_holders = ownership_data.get("top_holders", [])
    if top_holders:
        holder_rows = [{"Institution": h["name"], "% Held": f"{h['pct_held']:.2f}%" if h.get("pct_held") else "N/A"}
                       for h in top_holders]
        st.dataframe(pd.DataFrame(holder_rows).set_index("Institution"), width="stretch")

    # ── Short interest ──
    st.divider()
    st.caption("Short interest")
    si_label, si_status, si_text = score_short_interest(short_data)
    st.markdown(f"{icon_map[si_status]} {si_text}")
    pct_float = short_data.get("short_pct_float")
    short_ratio = short_data.get("short_ratio")
    shares_short = short_data.get("shares_short")
    squeeze = short_data.get("squeeze_potential", "unknown")
    if pct_float is not None:
        squeeze_color = "#ff1744" if squeeze == "high" else "#ff9100" if squeeze == "moderate" else "#00c853"
        st.markdown(
            f'<div class="trade-box"><div class="trade-row">'
            f'<div class="trade-item"><div class="trade-label">Short % of Float</div><div class="trade-value">{pct_float:.1f}%</div></div>'
            f'<div class="trade-item"><div class="trade-label">Days to Cover</div><div class="trade-value">{short_ratio if short_ratio else "N/A"}</div></div>'
            f'<div class="trade-item"><div class="trade-label">Shares Short</div><div class="trade-value">{f"{shares_short:,}" if shares_short else "N/A"}</div></div>'
            f'<div class="trade-item"><div class="trade-label">Squeeze Potential</div><div class="trade-value" style="color:{squeeze_color}">{squeeze.upper()}</div></div>'
            f'</div></div>', unsafe_allow_html=True
        )

    # ── Earnings surprise history ──
    st.divider()
    st.caption("Earnings surprise history (last 4 quarters)")
    e_sig, e_status, e_text = score_earnings_history(earnings_hist)
    st.markdown(f"{icon_map[e_status]} {e_text}")
    quarters = earnings_hist.get("quarters", [])
    if quarters:
        eq_cols = st.columns(len(quarters))
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
    st.divider()
    st.caption("Market context")
    mc_label, mc_status, mc_text = score_market_context(market_ctx)
    st.markdown(f"{icon_map[mc_status]} {mc_text}")
    vix_val = market_ctx.get("vix")
    vix_regime = market_ctx.get("vix_regime", "unknown")
    w52_pct = market_ctx.get("week52_pct")
    vix_regime_color = {"high_fear": "#ff1744", "neutral": "#ff9100", "complacency": "#00c853"}.get(vix_regime, "#888")
    rank_color = "#00c853" if w52_pct and w52_pct > 60 else "#ff1744" if w52_pct and w52_pct < 20 else "#ff9100"
    if vix_val is not None or w52_pct is not None:
        st.markdown(
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
    st.divider()
    st.caption("Sector trend")
    sm_label, sm_status, sm_text = score_sector_momentum(sector_mom)
    st.markdown(f"{icon_map[sm_status]} {sm_text}")
    sm_etf = sector_mom.get("etf")
    sm_ret = sector_mom.get("etf_1mo_return")
    if sm_etf and sm_ret is not None:
        ret_color = "#00c853" if sm_ret >= 0 else "#ff1744"
        ret_sign = "+" if sm_ret >= 0 else ""
        st.caption(f"{sm_etf} 1-month return: {ret_sign}{sm_ret:.1f}%")

with st.expander("📈 Performance vs S&P 500"):
    r_sig, r_status, r_text = score_relative_performance(rel_perf, ticker)
    st.markdown(f"{icon_map[r_status]} {r_text}")

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
        st.plotly_chart(fig_rel, width="stretch")
    else:
        st.info("Could not load comparison data.")

with st.expander("📰 News & sentiment"):
    if not scored_articles:
        st.info("No recent news found.")
    else:
        sent_label, sent_color = overall_sentiment(scored_articles)
        sent_hex = color_map.get(sent_color, "#888")
        st.markdown(f'<div style="color:{sent_hex};font-weight:600;margin-bottom:12px">{sent_label}</div>', unsafe_allow_html=True)
        for a in scored_articles:
            sentiment = a.get("sentiment", "unknown").lower()
            score = a.get("sentiment_score", 0)
            icon = sent_icon.get(sentiment, "⚪")
            with st.expander(f"{icon} {a['title']}"):
                st.caption(f"{a['source']}  ·  {a['published']}  ·  **{sentiment.upper()}** ({score:.0%})")
                if a.get("summary"):
                    st.write(a["summary"])
                if a.get("url"):
                    st.markdown(f"[Read article]({a['url']})")

with st.expander("🔁 Backtest — how did a simple strategy do?"):
    st.caption("Buys when 20-day average crosses above 50-day average. Sells when it crosses below.")
    if not run_backtest:
        st.info("Enable 'Run SMA Crossover' in the sidebar to run.")
    else:
        with st.spinner("Running backtest..."):
            try:
                pf = sma_crossover_backtest(df["close"], fast=int(fast_win), slow=int(slow_win))
                stats = get_backtest_stats(pf)
                total_return = float(stats.get("Total Return [%]", 0) or 0)
                n_trades = int(stats.get("Total Trades", 0) or 0)
                if n_trades == 0:
                    st.warning(
                        f"No crossover trades triggered in this period. "
                        f"The SMA{int(fast_win)} never crossed the SMA{int(slow_win)} — "
                        f"stock was in a sustained trend. Try a longer period (1y or 2y)."
                    )
                else:
                    st.dataframe(stats.to_frame("Value").astype(str), width="stretch")
                    st.plotly_chart(pf.plot(), width="stretch")
            except Exception as e:
                st.error(f"Backtest failed: {e}")
