"""Model Ledger page — predictions and watchlist management."""
import sqlite3
import os
import sys
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

PREDICTIONS_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "trade_ml", "predictions.db"
)


def _get_db_connection():
    """Open connection to trade_ml/predictions.db."""
    if not os.path.exists(PREDICTIONS_DB_PATH):
        return None
    return sqlite3.connect(PREDICTIONS_DB_PATH)


def _load_predict_watchlist() -> list[str]:
    """Load tickers from predict_watchlist table."""
    conn = _get_db_connection()
    if conn is None:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker FROM predict_watchlist ORDER BY ticker")
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


def _add_ticker_to_watchlist(ticker: str) -> bool:
    """Add ticker to predict_watchlist. Returns True if successful."""
    conn = _get_db_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO predict_watchlist (ticker) VALUES (?)", (ticker.upper(),))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def _remove_ticker_from_watchlist(ticker: str) -> bool:
    """Remove ticker from predict_watchlist."""
    conn = _get_db_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predict_watchlist WHERE ticker = ?", (ticker,))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def _load_predictions() -> pd.DataFrame | None:
    """Load all predictions from predictions table."""
    conn = _get_db_connection()
    if conn is None:
        return None
    try:
        df = pd.read_sql_query(
            "SELECT date, ticker, pred_return, weight, leg, model_version, price_at_prediction, "
            "checked, realized_return FROM predictions ORDER BY date DESC",
            conn
        )
        return df if not df.empty else None
    finally:
        conn.close()


def _compute_hit_rate(checked_df: pd.DataFrame) -> float | None:
    """
    Compute hit rate: for each date, split checked rows into above-median vs below-median
    pred_return, compare avg realized_return, return % of dates where above-median beat below-median.
    """
    if len(checked_df) < 10:
        return None

    hit_count = 0
    date_count = 0

    for date, group in checked_df.groupby("date"):
        if len(group) < 2:
            continue
        median_score = group["pred_return"].median()
        above_median = group[group["pred_return"] >= median_score]
        below_median = group[group["pred_return"] < median_score]

        if len(above_median) > 0 and len(below_median) > 0:
            avg_above = above_median["realized_return"].mean()
            avg_below = below_median["realized_return"].mean()
            if avg_above > avg_below:
                hit_count += 1
            date_count += 1

    return hit_count / date_count * 100 if date_count > 0 else None


def render_model_ledger_page(st_obj):
    """Render model ledger page."""
    st_obj.title("🧾 Model Ledger")

    # Check if predictions.db exists
    if not os.path.exists(PREDICTIONS_DB_PATH):
        st_obj.error(
            f"Predictions database not found at {PREDICTIONS_DB_PATH}. "
            "Run trade_ml's log_predictions.py first."
        )
        st_obj.stop()

    # ── WATCHLIST EDITOR ────────────────────────────────────────────────────────
    st_obj.subheader("📋 Prediction Watchlist")
    watchlist_tickers = _load_predict_watchlist()

    col_add, col_btn = st_obj.columns([4, 1])
    with col_add:
        new_ticker = st_obj.text_input(
            "Add ticker to predict_watchlist",
            placeholder="e.g. NVDA",
            label_visibility="collapsed"
        ).upper().strip()
    with col_btn:
        if st_obj.button("Add", key="add_ticker") and new_ticker:
            if _add_ticker_to_watchlist(new_ticker):
                st_obj.success(f"Added {new_ticker}")
                st_obj.rerun()
            else:
                st_obj.error(f"Failed to add {new_ticker}")

    if watchlist_tickers:
        st_obj.markdown("**Current tickers:**")
        for ticker in watchlist_tickers:
            col_name, col_btn = st_obj.columns([3, 1])
            col_name.text(ticker)
            if col_btn.button("✕", key=f"rm_{ticker}"):
                if _remove_ticker_from_watchlist(ticker):
                    st_obj.success(f"Removed {ticker}")
                    st_obj.rerun()
    else:
        st_obj.info(
            "No custom watchlist configured. "
            "log_predictions.py will use DEFAULT_TICKERS until you add tickers here."
        )

    st_obj.divider()

    # ── PREDICTIONS TABLE ───────────────────────────────────────────────────────
    st_obj.subheader("📊 Predictions")
    predictions_df = _load_predictions()

    if predictions_df is None or predictions_df.empty:
        st_obj.info("No predictions logged yet.")
        st_obj.stop()

    # Split into pending and checked
    pending_df = predictions_df[predictions_df["checked"] == 0].copy()
    checked_df = predictions_df[predictions_df["checked"] == 1].copy()

    # Pending predictions
    if not pending_df.empty:
        st_obj.markdown("**Pending** (awaiting outcome)")
        pending_display = pending_df[["date", "ticker", "pred_return", "weight", "leg",
                                       "model_version", "price_at_prediction"]].copy()
        pending_display["pred_return"] = pending_display["pred_return"].apply(lambda x: f"{x:.4f}")
        pending_display["weight"] = pending_display["weight"].apply(lambda x: f"{x:+.3f}")
        pending_display["price_at_prediction"] = pending_display["price_at_prediction"].apply(lambda x: f"${x:.2f}")
        st_obj.dataframe(pending_display, use_container_width=True, hide_index=True)
    else:
        st_obj.info("No pending predictions.")

    st_obj.markdown("")  # Spacer

    # Checked predictions
    if not checked_df.empty:
        st_obj.markdown("**Checked** (outcomes realized)")
        checked_display = checked_df[["date", "ticker", "pred_return", "weight", "leg",
                                       "model_version", "price_at_prediction", "realized_return"]].copy()
        checked_display["pred_return"] = checked_display["pred_return"].apply(lambda x: f"{x:.4f}")
        checked_display["weight"] = checked_display["weight"].apply(lambda x: f"{x:+.3f}")
        checked_display["price_at_prediction"] = checked_display["price_at_prediction"].apply(lambda x: f"${x:.2f}")
        checked_display["realized_return"] = checked_display["realized_return"].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
        )
        st_obj.dataframe(checked_display, use_container_width=True, hide_index=True)
    else:
        st_obj.info("No checked predictions yet.")

    st_obj.divider()

    # ── SUMMARY STATS ───────────────────────────────────────────────────────────
    st_obj.subheader("📈 Summary Statistics")

    if len(checked_df) < 10:
        st_obj.warning(
            f"Not enough data yet ({len(checked_df)}/10 checked predictions). "
            "Return after checking more predictions."
        )
        st_obj.stop()

    # Spearman correlation
    valid_checked = checked_df[
        checked_df["realized_return"].notna()
    ].copy()

    col_corr, col_hitrate = st_obj.columns(2)

    if len(valid_checked) >= 3:
        corr, pval = spearmanr(valid_checked["pred_return"], valid_checked["realized_return"])
        col_corr.metric(
            "Spearman ρ (pred_return vs realized_return)",
            f"{corr:.3f}",
            delta=f"p-value: {pval:.3f}" if pval < 0.05 else "p > 0.05"
        )
    else:
        col_corr.warning("Need ≥3 checked predictions with realized_return to compute correlation")

    # Hit rate
    hit_rate = _compute_hit_rate(checked_df[checked_df["realized_return"].notna()])
    if hit_rate is not None:
        col_hitrate.metric(
            "Hit Rate (above-median pred_return beat below-median by date)",
            f"{hit_rate:.1f}%"
        )
    else:
        col_hitrate.warning("Need ≥2 predictions per date to compute hit rate")
