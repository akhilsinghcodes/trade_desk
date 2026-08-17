"""Top Movers page — ranks live per-ticker ML 5-day return predictions.

Uses modules.ml_verdict.get_ml_verdict (same live inference path as the
Analyze page) instead of trade_ml's pre-published latest_predictions.json,
so numbers here always match what Analyze shows for the same ticker.

Ticker universe comes from modules.sp500_universe (fetched from Wikipedia,
cached) — deliberately NOT trade_ml/tickers.txt, which is train.py's input
and must stay decoupled from what this page scans.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.services.analysis import get_ml_verdict
from modules.sp500_universe import get_sp500_tickers


def _load_tickers() -> list[str]:
    return get_sp500_tickers()


def _run_live_predictions(tickers: list[str], progress_cb) -> pd.DataFrame:
    rows = []
    total = len(tickers)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_ml_verdict, t): t for t in tickers}
        completed = 0
        for future in as_completed(futures):
            ticker = futures[future]
            completed += 1
            progress_cb(completed, total, ticker)
            try:
                verdict = future.result(timeout=30)
            except Exception:
                continue
            if verdict is None:
                continue
            rows.append({
                "Ticker": ticker,
                "Pred 5D Return": verdict["pred_return_5d"] * 100,
                "As Of": verdict["as_of"],
            })
    return pd.DataFrame(rows)


def render_top_movers_page(st_obj) -> None:
    st_obj.title("📈 Top Movers")
    st_obj.markdown("Live per-ticker ML 5-day return predictions — same model + feature path as the Analyze page.")

    tickers = _load_tickers()
    if not tickers:
        st_obj.warning("Couldn't fetch the S&P 500 ticker list (Wikipedia fetch failed, no cache available).")
        return

    if "top_movers_df" not in st_obj.session_state:
        st_obj.session_state.top_movers_df = None

    if st_obj.button("🔄 Run live predictions", key="run_top_movers_btn"):
        progress_bar = st_obj.progress(0)
        status_text = st_obj.empty()

        def _progress(completed, total, ticker):
            status_text.info(f"Predicting {completed}/{total}: {ticker}")
            progress_bar.progress(completed / total)

        df = _run_live_predictions(tickers, _progress)
        progress_bar.empty()
        status_text.empty()
        st_obj.session_state.top_movers_df = df

    df = st_obj.session_state.top_movers_df
    if df is None:
        st_obj.info("Click **Run live predictions** to score all tracked tickers live.")
        return
    if df.empty:
        st_obj.warning("No tickers scored successfully.")
        return

    st_obj.caption(f"{len(df)}/{len(tickers)} tickers scored · as of {df['As Of'].iloc[0]}")

    n = st_obj.slider("Show top N per side", min_value=5, max_value=100, value=25, step=5)

    df_sorted = df.sort_values("Pred 5D Return", ascending=False)
    top = df_sorted.head(n).reset_index(drop=True)
    bottom = df_sorted.tail(n).sort_values("Pred 5D Return").reset_index(drop=True)

    col_pos, col_neg = st_obj.columns(2)
    col_config = {
        "Pred 5D Return": st_obj.column_config.NumberColumn(format="%.2f%%", width="small"),
        "Ticker": st_obj.column_config.TextColumn(width="small"),
        "As Of": st_obj.column_config.TextColumn(width="small"),
    }
    row_h = 35

    def table_h(n_rows):
        return row_h * (n_rows + 1) + 3

    with col_pos:
        st_obj.subheader(f"🟢 Top {len(top)} Positive")
        st_obj.dataframe(top, column_config=col_config, hide_index=True, use_container_width=True, height=table_h(len(top)))

    with col_neg:
        st_obj.subheader(f"🔴 Top {len(bottom)} Negative")
        st_obj.dataframe(bottom, column_config=col_config, hide_index=True, use_container_width=True, height=table_h(len(bottom)))
