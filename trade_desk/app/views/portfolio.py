"""Portfolio page."""
import plotly.graph_objects as go
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.indicators import add_common_indicators
from modules.fundamentals import score_fundamentals
from modules.summary import summarize
from modules.score import combined_score
from modules.cached_fetch import cached_ohlcv, cached_fundamentals
from modules.db import port_load, port_add, port_remove
from modules.portfolio import portfolio_summary


color_map = {"green": "#00c853", "red": "#ff1744", "orange": "#ff9100"}


def render_portfolio_page(st_obj):
    """Render portfolio page."""
    st_obj.title("Portfolio")

    with st_obj.expander("➕ Add a position", expanded=False):
        with st_obj.form("add_position_form"):
            pc1, pc2, pc3, pc4 = st_obj.columns([1.5, 1, 1.5, 2])
            p_ticker = pc1.text_input("Ticker").upper().strip()
            p_shares = pc2.number_input("Shares", min_value=0.0001, value=1.0, step=0.001, format="%.4f")
            p_price = pc3.number_input("Buy Price ($)", min_value=0.01, value=100.0)
            p_note = pc4.text_input("Note (optional)")
            if st_obj.form_submit_button("Add Position") and p_ticker:
                port_add(p_ticker, p_shares, p_price, note=p_note)
                st_obj.session_state.portfolio = port_load()
                st_obj.rerun()

    positions = st_obj.session_state.portfolio
    if not positions:
        st_obj.info("No positions yet. Add one above.")
        st_obj.stop()

    st_obj.divider()

    # Fetch current prices + previous close for all positions (cached)
    tickers_needed = [p["ticker"] for p in positions]
    current_prices = {}
    prev_prices = {}
    with st_obj.spinner("Fetching current prices..."):
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
    st_obj.markdown(f"""
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
        st_obj.plotly_chart(fig_pie, width="stretch")

    # Column headers
    _h1, _h2, _h3, _h4, _h5, _h6, _h7, _h8, _h9 = st_obj.columns([1, 1.5, 1, 1.3, 1.3, 1.3, 1.5, 1.1, 0.5])
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

        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st_obj.columns([1, 1.5, 1, 1.3, 1.3, 1.3, 1.5, 1.1, 0.5])
        if c1.button(f"**{t}**", key=f"port_analyze_{t}", help=f"Analyze {t}"):
            st_obj.session_state["analyze_ticker"] = t
            st_obj.session_state["page_override"] = "📊 Analyze"
            st_obj.rerun()
        c2.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace">${pnl["buy_price"]:.2f} → ${pnl["current_price"]:.2f}</span>', unsafe_allow_html=True)
        c3.caption(f'{pnl["shares"]:g}')
        c4.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace">${pnl["cost_basis"]:,.2f}</span>', unsafe_allow_html=True)
        c5.markdown(f'<span style="font-family:\'JetBrains Mono\',monospace">${pnl["current_value"]:,.2f}</span>', unsafe_allow_html=True)
        c6.markdown(f'<span style="color:{day_color};font-family:\'JetBrains Mono\',monospace;font-size:0.85rem">{day_str}</span>', unsafe_allow_html=True)
        c7.markdown(f'<span style="color:{p_color};font-family:\'JetBrains Mono\',monospace">{p_sign}${pnl["pnl_dollars"]:.2f} ({p_sign}{pnl["pnl_pct"]:.1f}%)</span>', unsafe_allow_html=True)
        c8.markdown(verdict_chip, unsafe_allow_html=True)
        if c9.button("✕", key=f"rm_pos_{t}"):
            to_remove = t
        st_obj.divider()

    if to_remove:
        port_remove(to_remove)
        st_obj.session_state.portfolio = port_load()
        st_obj.rerun()

    # ── Correlation matrix ──
    port_tickers = [p["ticker"] for p in summary["positions"]]
    if len(port_tickers) >= 2:
        st_obj.divider()
        st_obj.subheader("📐 Holdings Correlation")
        st_obj.markdown(
            "**How to read this:** Each cell shows how similarly two stocks move day-to-day (past 1 year). "
            "**+1.0** = always move together. **0.0** = move independently. **−1.0** = always move opposite. "
            "Green = good diversification. Red = overlapping bets — if one drops, the other likely does too.",
            unsafe_allow_html=False
        )
        with st_obj.spinner("Computing correlations..."):
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
                st_obj.plotly_chart(fig_corr, width="stretch")

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
                    st_obj.warning(f"⚠️ High correlation (>0.75): {pairs_str} — these move together, limited diversification benefit.")
                if low_corr_pairs:
                    pairs_str = ", ".join(f"**{a}/{b}** ({r:.2f})" for a, b, r in low_corr_pairs)
                    st_obj.success(f"✅ Low correlation (<0.25): {pairs_str} — good diversification.")
            except Exception as e:
                st_obj.info(f"Could not compute correlations: {e}")

    st_obj.stop()
