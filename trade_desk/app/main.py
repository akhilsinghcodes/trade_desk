"""Streamlit dashboard — Trade Lab."""
import streamlit as st
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.db import wl_load, wl_add, cache_invalidate_ticker, init_db, alerts_load, port_load, alerts_check
from modules.fetch import get_ohlcv
from modules.notifications import alert_checker
from modules.llm_batch import submit_portfolio_batch
from app.views import watchlist, portfolio, alerts, eli5, analyze, screener, backtest, telemetry, top_movers, paper_trading

# Streamlit's hot-reload watcher walks every loaded module's submodules to
# find file paths. Once modules.news lazily imports `transformers` (its
# sentiment pipeline), the watcher trips on transformers' optional vision
# submodules that need torchvision (not installed, not needed — we never
# use those models). Harmless probe failures, but noisy; silenced here.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
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
.score-bar-label { font-size: 0.75rem; opacity: 0.55; margin-bottom: 4px; letter-spacing: 0.01em; }
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
.trade-label { font-size: 0.7rem; opacity: 0.55; letter-spacing: 0.01em; margin-bottom: 4px; }
.trade-value { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 600; }
.trade-sub { font-size: 0.75rem; opacity: 0.5; margin-top: 2px; }

.meta-chip {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 0.75rem;
    margin-right: 6px;
    border: 0.5px solid rgba(255,255,255,0.1);
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
_TICKER_ALIASES = {"BRKB": "BRK-B", "BRKA": "BRK-A"}
def _norm_ticker(t: str) -> str:
    t = t.upper().strip()
    return _TICKER_ALIASES.get(t, t)

# ── INIT DB + PERSISTENT STATE ─────────────────────────────────────────────────
init_db()

# ── PRE-COMPUTE LLM RATIONALE FOR PORTFOLIO (batch, fully background) ──────────
# Runs once per server start in a daemon thread — never blocks page load.
@st.cache_resource
def _kick_portfolio_batch():
    import threading
    def _run():
        try:
            positions = port_load()
            if not positions:
                return
            from modules.cached_fetch import cached_ohlcv
            batch_data = []
            for pos in positions:
                t = pos["ticker"]
                try:
                    df = cached_ohlcv(t, "5d")
                    close = float(df["close"].iloc[-1])
                    pct_chg_5d = float((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100) if len(df) >= 2 else 0.0
                    batch_data.append({"ticker": t, "close": close, "pct_chg_5d": pct_chg_5d})
                except Exception:
                    pass
            submit_portfolio_batch(batch_data)
        except Exception as e:
            print(f"[portfolio batch] error: {e}")
    threading.Thread(target=_run, daemon=True).start()

_kick_portfolio_batch()

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
    pages = ["📊 Analyze", "⭐ Watchlist", "🔍 Screener", "📈 Top Movers", "📝 Paper Trading", "💼 Portfolio", "🔔 Alerts", "📚 ELI5", "🧪 Backtest", "💰 LLM Usage"]
    default_page_idx = 0
    page = st.radio("Page", pages, label_visibility="collapsed", index=default_page_idx)
    if "page_override" in st.session_state:
        page = st.session_state.pop("page_override")
    st.divider()

    if page == "📊 Analyze":
        st.sidebar.markdown("### 🔍 Ticker")
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
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📅 Period")
        period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=3)
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Chart Options")
        show_bb = st.checkbox("Bollinger Bands", value=True)
        show_sma = st.checkbox("SMA 20/50", value=True)
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔁 Backtest")
        run_backtest = st.checkbox("Run SMA Crossover", value=True)
        fast_win = st.number_input("Fast SMA", value=20, min_value=2)
        slow_win = st.number_input("Slow SMA", value=50, min_value=3)

    elif page == "🧪 Backtest":
        st.sidebar.markdown("### 🔍 Ticker")
        bt_ticker = _norm_ticker(st.text_input("Ticker", value="AAPL", key="bt_ticker"))
        st.sidebar.markdown("### 📅 Period")
        bt_period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=1, key="bt_period")
        ticker = bt_ticker
        period = bt_period

# ── PAGE ROUTING ───────────────────────────────────────────────────────────────
if page == "⭐ Watchlist":
    watchlist.render_watchlist_page(st)
elif page == "🔍 Screener":
    screener.render_screener_page(st)
elif page == "📈 Top Movers":
    top_movers.render_top_movers_page(st)
elif page == "📝 Paper Trading":
    paper_trading.render_paper_trading_page(st)
elif page == "💼 Portfolio":
    portfolio.render_portfolio_page(st)
elif page == "🔔 Alerts":
    alerts.render_alerts_page(st, alert_checker)
elif page == "📚 ELI5":
    eli5.render_eli5_page(st)
elif page == "📊 Analyze":
    analyze.render_analyze_page(st, ticker, period, show_bb, show_sma, run_backtest, fast_win, slow_win)
elif page == "🧪 Backtest":
    backtest.render_backtest_page(st, ticker, period)
elif page == "💰 LLM Usage":
    telemetry.render_telemetry_page(st)
