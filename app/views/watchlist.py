"""Watchlist page."""
from streamlit_autorefresh import st_autorefresh
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.fetch import get_ohlcv, get_info
from modules.indicators import add_common_indicators
from modules.fundamentals import get_fundamentals, score_fundamentals
from modules.summary import summarize
from modules.score import combined_score
from modules.db import wl_load, wl_add, wl_remove


color_map = {"green": "#00c853", "red": "#ff1744", "orange": "#ff9100"}


def render_watchlist_page(st_obj):
    """Render watchlist page."""
    col_wl_title, col_wl_refresh = st_obj.columns([3, 1])
    col_wl_title.title("Watchlist")
    refresh_interval = col_wl_refresh.selectbox(
        "Auto-refresh", ["Off", "1 min", "5 min", "15 min"],
        index=2, label_visibility="collapsed"
    )
    interval_map = {"1 min": 60_000, "5 min": 300_000, "15 min": 900_000}
    if refresh_interval != "Off":
        st_autorefresh(interval=interval_map[refresh_interval], limit=None, key="wl_autorefresh")

    col_add, col_btn = st_obj.columns([4, 1])
    with col_add:
        new_ticker = st_obj.text_input("Add ticker", placeholder="e.g. NVDA", label_visibility="collapsed").upper().strip()
    with col_btn:
        if st_obj.button("Add") and new_ticker:
            if new_ticker not in st_obj.session_state.watchlist:
                wl_add(new_ticker)
                st_obj.session_state.watchlist = wl_load()
                st_obj.rerun()

    if not st_obj.session_state.watchlist:
        st_obj.info("No tickers yet. Add one above.")
        st_obj.stop()

    st_obj.divider()
    verdict_icons = {"BUY": "🟢", "HOLD / WATCH": "🟡", "SELL / AVOID": "🔴"}
    to_remove = None

    for t in st_obj.session_state.watchlist:
        with st_obj.spinner(f"Loading {t}..."):
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
                pe =f"{_info.get('trailingPE'):.1f}" if _info.get("trailingPE") else "N/A"
                v = _score["verdict"]
                conf = int(_score["confidence"] * 100)
                v_hex = color_map.get(_score["color"], "#888")

                c1, c2, c3, c4, c5, c6, c7 = st_obj.columns([1.2, 3, 1.5, 1.5, 2.5, 0.8, 0.5])
                if c1.button(f"**{t}**", key=f"analyze_{t}", help=f"Analyze {t}"):
                    st_obj.session_state["analyze_ticker"] = t
                    st_obj.session_state["page_override"] = "📊 Analyze"
                    st_obj.rerun()
                c2.caption(_info.get("shortName", t))
                c3.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace;font-weight:600">${close:.2f}</span>', unsafe_allow_html=True)
                c4.markdown(f'<span style="color:{chg_color};font-family:\'JetBrains Mono\',monospace">{chg_str}</span>', unsafe_allow_html=True)
                c5.markdown(f'<span style="color:{v_hex};font-weight:600">{verdict_icons.get(v,"⚪")} {v}</span> <span style="opacity:0.45;font-size:0.8rem">({conf}%)</span>', unsafe_allow_html=True)
                c6.markdown(f'<span style="opacity:0.4;font-size:0.75rem">P/E {pe}</span>', unsafe_allow_html=True)
                if c7.button("✕", key=f"rm_{t}"):
                    to_remove = t
            except Exception:
                st_obj.error(f"Failed to load {t}")
        st_obj.divider()

    if to_remove:
        wl_remove(to_remove)
        st_obj.session_state.watchlist = wl_load()
        st_obj.rerun()
    st_obj.stop()
