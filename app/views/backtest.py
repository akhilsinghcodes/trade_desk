"""Backtest & Signal Validation page."""
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.cached_fetch import cached_ohlcv
from modules.indicators import add_common_indicators
from modules.volume import add_volume_indicators

# Lazy imports — modules may not exist yet during first load
def _load_factor_analysis():
    from modules.factor_analysis import compute_signals_df, compute_ic, compute_rolling_ic, ic_summary
    return compute_signals_df, compute_ic, compute_rolling_ic, ic_summary

def _load_setup_backtest():
    from modules.setup_backtest import backtest_all, compare_summary
    return backtest_all, compare_summary


def _color_ic(val):
    if isinstance(val, float):
        if abs(val) >= 0.05:
            return f"color: {'#4ade80' if val > 0 else '#f87171'}"
    return ""

def _color_return(val):
    if isinstance(val, (int, float)):
        return f"color: {'#4ade80' if val > 0 else '#f87171'}"
    return ""


def render_backtest_page(st_obj, ticker: str, period: str = "2y"):
    st_obj.markdown("## 🧪 Backtest & Signal Validation")
    st_obj.markdown(
        "<div style='font-size:0.85rem;opacity:0.5;margin-bottom:24px'>"
        "Does our strategy actually work? Signal IC measures predictive power. "
        "Backtest shows real P&L vs buy-and-hold."
        "</div>",
        unsafe_allow_html=True,
    )

    tab_ic, tab_bt = st_obj.tabs(["📊 Signal IC (Factor Analysis)", "📈 Strategy Backtest"])

    # ── Fetch data ────────────────────────────────────────────────────────────
    with st_obj.spinner(f"Loading {ticker} data…"):
        try:
            df = cached_ohlcv(ticker, period)
            df = add_common_indicators(df)
            df = add_volume_indicators(df)
        except Exception as e:
            st_obj.error(f"Data fetch failed: {e}")
            return

    if df is None or len(df) < 63:
        st_obj.warning(f"Need at least 63 bars — only {len(df) if df is not None else 0} available. Try a longer period.")
        return

    close = df["close"].astype(float)
    n_bars = len(df)
    date_range = f"{df.index[0].date()} → {df.index[-1].date()}"

    st_obj.caption(f"{ticker} · {n_bars} bars · {date_range}")

    # ── TAB 1: Signal IC ──────────────────────────────────────────────────────
    with tab_ic:
        st_obj.markdown("### Which signals predict forward returns?")
        st_obj.markdown(
            "<div style='font-size:0.8rem;opacity:0.5;margin-bottom:16px'>"
            "IC = Spearman correlation between signal value and N-day forward return. "
            "|IC| > 0.05 = meaningful. |IC| > 0.10 = strong. t-stat > 2 = statistically significant."
            "</div>",
            unsafe_allow_html=True,
        )

        try:
            compute_signals_df, compute_ic, compute_rolling_ic, ic_summary_fn = _load_factor_analysis()
        except ImportError as e:
            st_obj.error(f"factor_analysis module not found: {e}")
            return

        with st_obj.spinner("Computing signal ICs…"):
            try:
                signals_df = compute_signals_df(df)
                ic_df = compute_ic(signals_df, close, horizons=[1, 5, 10, 21])
                summary = ic_summary_fn(ic_df)
            except Exception as e:
                st_obj.error(f"IC computation failed: {e}")
                return

        if summary.empty:
            st_obj.warning("Insufficient data for IC analysis.")
            return

        # IC summary table
        st_obj.markdown("#### IC by signal & horizon")

        display_cols = []
        for h in [1, 5, 10, 21]:
            col = (h, "ic")
            if col in ic_df.columns:
                display_cols.append(col)

        if display_cols:
            ic_table = ic_df[display_cols].copy()
            ic_table.columns = [f"{h}d IC" for (h, _) in display_cols]
            ic_table["Mean |IC|"] = ic_table.abs().mean(axis=1)
            ic_table = ic_table.sort_values("Mean |IC|", ascending=False)
            styled = ic_table.style.format("{:.3f}").applymap(_color_ic)
            st_obj.dataframe(styled, use_container_width=True)

        # t-stat table
        with st_obj.expander("t-statistics (|t| > 2 = significant at 95%)"):
            tstat_cols = [(h, "tstat") for h in [1, 5, 10, 21] if (h, "tstat") in ic_df.columns]
            if tstat_cols:
                ts_table = ic_df[tstat_cols].copy()
                ts_table.columns = [f"{h}d t-stat" for (h, _) in tstat_cols]
                ts_table = ts_table.reindex(ic_table.index)
                st_obj.dataframe(ts_table.style.format("{:.2f}"), use_container_width=True)

        # Rolling IC chart for best signal
        best_signal = summary.index[0] if not summary.empty else None
        if best_signal and best_signal in signals_df.columns:
            st_obj.markdown(f"#### Rolling 63-day IC: `{best_signal}` → 5d forward return")
            rolling_ic = compute_rolling_ic(signals_df[best_signal], close, horizon=5, window=63)
            if rolling_ic is not None and not rolling_ic.dropna().empty:
                chart_df = rolling_ic.dropna().to_frame("Rolling IC")
                chart_df["zero"] = 0.0
                st_obj.line_chart(chart_df[["Rolling IC"]], use_container_width=True)
                mean_ic = rolling_ic.dropna().mean()
                st_obj.caption(f"Mean rolling IC: {mean_ic:.3f} over {len(rolling_ic.dropna())} windows")

        # Insight callout
        top_signals = summary.head(3).index.tolist()
        dead_signals = summary[summary["mean_ic_abs"] < 0.02].index.tolist()
        if top_signals:
            st_obj.markdown(
                f"<div style='background:rgba(74,222,128,0.07);border:1px solid rgba(74,222,128,0.2);"
                f"border-radius:8px;padding:12px 16px;margin-top:16px;font-size:0.85rem'>"
                f"<strong style='color:#4ade80'>✓ Strongest signals:</strong> {', '.join(f'<code>{s}</code>' for s in top_signals)}"
                f"</div>",
                unsafe_allow_html=True,
            )
        if dead_signals:
            st_obj.markdown(
                f"<div style='background:rgba(248,113,113,0.07);border:1px solid rgba(248,113,113,0.2);"
                f"border-radius:8px;padding:12px 16px;margin-top:8px;font-size:0.85rem'>"
                f"<strong style='color:#f87171'>✗ Near-zero IC (likely noise):</strong> {', '.join(f'<code>{s}</code>' for s in dead_signals[:5])}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── TAB 2: Strategy Backtest ──────────────────────────────────────────────
    with tab_bt:
        st_obj.markdown("### Setup strategies vs buy-and-hold")
        st_obj.markdown(
            "<div style='font-size:0.8rem;opacity:0.5;margin-bottom:16px'>"
            "Backtest of BREAKOUT, PULLBACK, MEAN_REVERSION strategies using vectorbt. "
            "0.1% commission per trade. No slippage modeled."
            "</div>",
            unsafe_allow_html=True,
        )

        try:
            backtest_all_fn, compare_summary_fn = _load_setup_backtest()
        except ImportError as e:
            st_obj.error(f"setup_backtest module not found: {e}")
            return

        with st_obj.spinner("Running backtests (this takes ~10s)…"):
            try:
                results = backtest_all_fn(df)
            except Exception as e:
                st_obj.error(f"Backtest failed: {e}")
                return

        # Summary table
        try:
            summary_df = compare_summary_fn(results)
        except Exception as e:
            st_obj.error(f"Summary failed: {e}")
            return

        if not summary_df.empty:
            st_obj.markdown("#### Performance summary")
            styled_summary = summary_df.style.format({
                c: "{:.1f}%" for c in summary_df.columns if "%" in c or "Return" in c or "Drawdown" in c or "Rate" in c
            } | {
                c: "{:.2f}" for c in summary_df.columns if "Sharpe" in c
            } | {
                "# Trades": "{:.0f}"
            }).applymap(
                lambda v: f"color: {'#4ade80' if v > 0 else '#f87171'}" if isinstance(v, float) else "",
                subset=[c for c in summary_df.columns if "Return" in c or "Sharpe" in c]
            )
            st_obj.dataframe(styled_summary, use_container_width=True)

        # Equity curves
        st_obj.markdown("#### Equity curves")
        equity_data = {}


        for name, res in results.items():
            if res.get("error"):
                continue
            eq = res.get("equity_curve")
            if eq is not None and not eq.empty:
                normalized = eq / eq.iloc[0] * 100  # index to 100
                equity_data[name] = normalized

        if equity_data:
            eq_df = pd.DataFrame(equity_data)
            eq_df = eq_df.dropna(how="all")
            st_obj.line_chart(eq_df, use_container_width=True)
            st_obj.caption("Indexed to 100 at start. Same initial capital ($10,000) for all strategies.")

        # Per-strategy stats
        for name, res in results.items():
            if res.get("error"):
                with st_obj.expander(f"⚠️ {name} — failed"):
                    st_obj.code(res["error"])
                continue

            trades = res.get("trades", [])
            n = res.get("n_trades", 0)
            ret = res.get("total_return_pct", 0)
            sharpe = res.get("sharpe", 0)

            with st_obj.expander(f"{name}  ·  {ret:+.1f}%  ·  {n} trades  ·  Sharpe {sharpe:.2f}"):
                c1, c2, c3, c4 = st_obj.columns(4)
                c1.metric("Total Return", f"{ret:+.1f}%")
                c2.metric("CAGR", f"{res.get('cagr_pct', 0):+.1f}%")
                c3.metric("Max Drawdown", f"{res.get('max_drawdown_pct', 0):.1f}%")
                c4.metric("Win Rate", f"{res.get('win_rate_pct', 0):.1f}%")

                if trades:
                    trades_df = pd.DataFrame(trades[:20])
                    if not trades_df.empty:
                        st_obj.dataframe(trades_df, use_container_width=True)
