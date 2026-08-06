"""Alerts page."""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.cached_fetch import cached_ohlcv
from modules.db import alerts_load, alerts_add, alerts_remove, alerts_check
from modules.notifications import notify_alert_triggered


def render_alerts_page(st_obj, alert_checker):
    """Render alerts page."""
    st_obj.title("Price Alerts")
    st_obj.caption("Desktop notifications fire every 5 min even when this tab is closed.")
    checker_status = "🟢 Running" if alert_checker.is_running else "🔴 Stopped"
    st_obj.caption(f"Background checker: {checker_status}")

    with st_obj.form("add_alert_form"):
        st_obj.caption("Set a new alert")
        ac1, ac2, ac3, ac4 = st_obj.columns([1.5, 1.5, 1.5, 2])
        a_ticker = ac1.text_input("Ticker").upper().strip()
        a_price = ac2.number_input("Target Price ($)", min_value=0.01, value=100.0)
        a_direction = ac3.selectbox("Trigger when price goes", ["above", "below"])
        a_note = ac4.text_input("Note (optional)")
        if st_obj.form_submit_button("Add Alert") and a_ticker:
            alerts_add(a_ticker, a_price, a_direction, note=a_note)
            st_obj.session_state.alerts = alerts_load()
            st_obj.rerun()

    alerts = st_obj.session_state.alerts
    if not alerts:
        st_obj.info("No alerts set. Add one above.")
        st_obj.stop()

    st_obj.divider()

    # Check alerts + send desktop notifications for newly triggered ones
    alert_tickers = list({a["ticker"] for a in alerts if not a.get("triggered")})
    current_prices = {}
    with st_obj.spinner("Checking prices..."):
        for t in alert_tickers:
            try:
                _df = cached_ohlcv(t, "5d")
                current_prices[t] = float(_df["close"].iloc[-1])
            except Exception:
                pass

    triggered = alerts_check(current_prices)
    if triggered:
        for ta in triggered:
            st_obj.success(f"🔔 **{ta['ticker']}** hit your target of ${ta['target_price']} ({ta['direction']})")
            notify_alert_triggered(ta["ticker"], ta["target_price"], ta["direction"],
                                   current_prices.get(ta["ticker"], ta["target_price"]))
        st_obj.session_state.alerts = alerts_load()

    # Display all alerts
    st_obj.subheader("Active Alerts")
    to_remove_alert_id = None
    for a in alerts:
        t = a["ticker"]
        cp = current_prices.get(t)
        cp_str = f"Current: ${cp:.2f}" if cp else "Price unavailable"
        status_icon = "✅" if a.get("triggered") else "🔔"
        triggered_label = " — **TRIGGERED**" if a.get("triggered") else ""
        c1, c2, c3, c4, c5 = st_obj.columns([1, 1.5, 1.5, 2, 0.5])
        c1.markdown(f"**{t}**")
        c2.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace">${a["target_price"]:.2f} {a["direction"]}</span>', unsafe_allow_html=True)
        c3.caption(cp_str)
        c4.markdown(f"{status_icon} {a.get('note', '')}{triggered_label}")
        if c5.button("✕", key=f"rm_alert_{a['id']}"):
            to_remove_alert_id = a["id"]
        st_obj.divider()

    if to_remove_alert_id is not None:
        alerts_remove(to_remove_alert_id)
        st_obj.session_state.alerts = alerts_load()
        st_obj.rerun()
    st_obj.stop()
