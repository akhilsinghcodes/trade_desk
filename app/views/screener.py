"""Multi-ticker screener page."""
import streamlit as st
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.services.analysis import run_analysis


def render_screener_page(st_obj) -> None:
    """Render the multi-ticker screener page."""
    st_obj.title("🔍 Screener")
    st_obj.markdown("Analyze multiple tickers at once and compare signals.")

    # Initialize session state for screener results
    if "screener_results" not in st_obj.session_state:
        st_obj.session_state.screener_results = None
    if "screener_tickers" not in st_obj.session_state:
        st_obj.session_state.screener_tickers = ""

    # ── INPUT SECTION ──────────────────────────────────────────────────────────
    st_obj.subheader("Input Tickers")
    col_input, col_count = st_obj.columns([4, 1])
    with col_input:
        tickers_input = st_obj.text_area(
            "Enter tickers (one per line or comma-separated)",
            value=st_obj.session_state.screener_tickers,
            placeholder="AAPL\nMSFT\nGOOGL",
            height=100,
            label_visibility="collapsed"
        )

    # Parse tickers
    ticker_list = [t.strip().upper() for t in tickers_input.replace(",", "\n").split("\n") if t.strip()]
    ticker_list = list(dict.fromkeys(ticker_list))  # Remove duplicates, preserve order
    ticker_list = ticker_list[:20]  # Limit to 20 tickers

    with col_count:
        st_obj.metric("Tickers", len(ticker_list))

    # ── RUN SCREENER BUTTON ────────────────────────────────────────────────────
    col_btn1, col_btn2 = st_obj.columns(2)
    with col_btn1:
        run_screener = st_obj.button("🚀 Run Screener", key="run_screener_btn")
    with col_btn2:
        clear_results = st_obj.button("Clear Results", key="clear_screener_btn")

    if clear_results:
        st_obj.session_state.screener_results = None
        st_obj.session_state.screener_tickers = ""
        st_obj.rerun()

    if run_screener and ticker_list:
        st_obj.session_state.screener_tickers = tickers_input
        _run_screener_analysis(st_obj, ticker_list)

    # ── RESULTS SECTION ────────────────────────────────────────────────────────
    if st_obj.session_state.screener_results is not None:
        st_obj.divider()
        st_obj.subheader("Results")
        results_df = st_obj.session_state.screener_results

        if not results_df.empty:
            # Sort controls
            col_sort1, col_sort2 = st_obj.columns(2)
            with col_sort1:
                sort_col = st_obj.selectbox(
                    "Sort by",
                    ["Score", "Ticker", "Verdict", "Confidence"],
                    index=0
                )
            with col_sort2:
                sort_desc = st_obj.checkbox("Descending", value=True)

            sort_order = sort_desc

            if sort_col == "Score":
                results_sorted = results_df.sort_values("Score", ascending=not sort_order)
            elif sort_col == "Confidence":
                results_sorted = results_df.sort_values("Confidence", ascending=not sort_order)
            elif sort_col == "Verdict":
                verdict_order = {"BUY": 3, "HOLD / WATCH": 2, "SELL / AVOID": 1}
                results_sorted = results_df.copy()
                results_sorted["_verdict_rank"] = results_sorted["Verdict"].map(verdict_order)
                results_sorted = results_sorted.sort_values("_verdict_rank", ascending=sort_order)
                results_sorted = results_sorted.drop("_verdict_rank", axis=1)
            else:
                results_sorted = results_df.sort_values("Ticker", ascending=not sort_order)

            # Display dataframe with column config
            st_obj.dataframe(
                results_sorted,
                column_config={
                    "Score": st_obj.column_config.NumberColumn(format="%.3f"),
                    "Confidence": st_obj.column_config.NumberColumn(format="%.1%"),
                    "Tech Score": st_obj.column_config.NumberColumn(format="%.3f"),
                    "Fund Score": st_obj.column_config.NumberColumn(format="%.3f"),
                    "Sent Score": st_obj.column_config.NumberColumn(format="%.3f"),
                    "Analyze": st_obj.column_config.CheckboxColumn(),
                },
                hide_index=True,
                use_container_width=True,
            )

            # Analyze buttons below table
            st_obj.markdown("---")
            st_obj.subheader("Quick Actions")
            col_export, col_space = st_obj.columns([2, 3])

            with col_export:
                csv = results_sorted.drop("Analyze", axis=1, errors="ignore").to_csv(index=False)
                st_obj.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="screener_results.csv",
                    mime="text/csv"
                )

            # Action buttons for each row
            st_obj.markdown("---")
            for idx, row in results_sorted.iterrows():
                col_ticker, col_verdict, col_score, col_analyze = st_obj.columns([1.5, 2, 1.5, 1])
                with col_ticker:
                    st_obj.markdown(f"**{row['Ticker']}**")
                with col_verdict:
                    st_obj.caption(row["Verdict"])
                with col_score:
                    st_obj.caption(f"Score: {row['Score']:.3f}")
                with col_analyze:
                    if st_obj.button("→ Analyze", key=f"analyze_screener_{row['Ticker']}"):
                        st_obj.session_state["analyze_ticker"] = row["Ticker"]
                        st_obj.session_state["page_override"] = "📊 Analyze"
                        st_obj.rerun()

        else:
            st_obj.warning("No results to display.")

    elif st_obj.session_state.screener_results is not None and st_obj.session_state.screener_results.empty:
        st_obj.warning("No tickers analyzed successfully.")


def _run_screener_analysis(st_obj, ticker_list: list) -> None:
    """Fetch and score multiple tickers in parallel."""
    progress_bar = st_obj.progress(0)
    status_text = st_obj.empty()

    results = []
    total = len(ticker_list)

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        futures = {
            executor.submit(run_analysis, ticker, "3mo", "1d"): ticker
            for ticker in ticker_list
        }

        completed = 0
        for future in as_completed(futures):
            ticker = futures[future]
            completed += 1
            progress_pct = completed / total

            status_text.info(f"Analyzing {completed}/{total}: {ticker}")
            progress_bar.progress(progress_pct)

            try:
                result = future.result(timeout=30)
                if result is None:
                    st_obj.warning(f"⚠️ {ticker}: No data")
                    continue

                verdict_result = result.get("verdict_result", {})
                component_scores = verdict_result.get("component_scores", {})

                row = {
                    "Ticker": ticker,
                    "Verdict": verdict_result.get("verdict", "N/A"),
                    "Score": verdict_result.get("score", 0.0),
                    "Confidence": verdict_result.get("confidence", 0.0),
                    "Tech Score": component_scores.get("technical", 0.0),
                    "Fund Score": component_scores.get("fundamental", 0.0),
                    "Sent Score": component_scores.get("sentiment", 0.0),
                    "Conflicted": verdict_result.get("conflicted", False),
                }
                results.append(row)

            except Exception as e:
                st_obj.warning(f"⚠️ {ticker}: Error — {str(e)[:50]}")

    progress_bar.empty()
    status_text.empty()

    if results:
        results_df = pd.DataFrame(results)

        # Color coding logic for display (note: Streamlit DataFrame doesn't directly support
        # conditional cell coloring in the simple API, but we can use column config)
        st_obj.session_state.screener_results = results_df
        st_obj.success(f"✅ Analyzed {len(results)}/{total} tickers")
    else:
        st_obj.error("No tickers analyzed successfully.")
        st_obj.session_state.screener_results = pd.DataFrame()
