"""
Check predictions: find old rows, fetch realized returns, compute correlation and hit rate.

Usage:
  python check_predictions.py
"""
import sqlite3
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


DB_PATH = "predictions.db"


def _fetch_current_price(ticker: str) -> float | None:
    """Fetch the latest closing price for a ticker."""
    try:
        data = yf.download(ticker, period="1d", progress=False, auto_adjust=True)
        if data.empty:
            return None
        close_val = data["Close"].squeeze()
        if isinstance(close_val, pd.Series):
            close_val = close_val.iloc[-1]
        return float(close_val)
    except Exception:
        return None


def _days_since_date(date_str: str) -> int:
    """Compute number of days between date_str and today."""
    # Handle both "%Y-%m-%d" and "%Y-%m-%d %H:%M:%S" formats
    date_part = date_str.split()[0]
    pred_date = datetime.strptime(date_part, "%Y-%m-%d").date()
    today = datetime.now().date()
    return (today - pred_date).days


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Find all unchecked predictions that are at least 7 days old
    cursor.execute("""
        SELECT * FROM predictions
        WHERE checked = 0
        ORDER BY date ASC
    """)
    rows_to_check = cursor.fetchall()

    updated_count = 0
    for row in rows_to_check:
        days_old = _days_since_date(row["date"])
        if days_old < 7:
            continue

        # Fetch current price
        current_price = _fetch_current_price(row["ticker"])
        if current_price is None or current_price == 0:
            continue

        # Compute realized return
        realized_return = (current_price - row["price_at_prediction"]) / row["price_at_prediction"]
        checked_date = datetime.now().strftime("%Y-%m-%d")

        # Update row
        cursor.execute("""
            UPDATE predictions
            SET realized_return = ?, realized_price = ?, checked_date = ?, checked = 1
            WHERE id = ?
        """, (realized_return, current_price, checked_date, row["id"]))
        updated_count += 1

    conn.commit()

    # Compute statistics over all checked rows
    cursor.execute("SELECT COUNT(*) as n FROM predictions WHERE checked = 1")
    n_checked = cursor.fetchone()["n"]

    if n_checked < 10:
        print(f"\n{'='*60}")
        print(f"  Check Results")
        print(f"{'='*60}")
        print(f"  Updated this run: {updated_count}")
        print(f"  Total checked: {n_checked}")
        print(f"  Status: Not enough data yet (need >= 10 checked rows)")
        print(f"{'='*60}\n")
        conn.close()
        return

    # Fetch all checked rows for analysis
    cursor.execute("""
        SELECT date, pred_return, realized_return
        FROM predictions
        WHERE checked = 1
        ORDER BY date ASC
    """)
    checked_rows = cursor.fetchall()

    # Compute Spearman correlation
    pred_returns = np.array([row["pred_return"] for row in checked_rows])
    realized_returns = np.array([row["realized_return"] for row in checked_rows])
    rho, pval = spearmanr(pred_returns, realized_returns)

    # Compute hit rate (top half vs bottom half by pred_return, grouped by date)
    df_checked = pd.DataFrame([dict(row) for row in checked_rows])
    df_checked["date"] = pd.to_datetime(df_checked["date"])

    hit_results = []
    for date, group in df_checked.groupby("date"):
        if len(group) < 2:
            continue
        group = group.sort_values("pred_return")
        n = len(group)
        half = n // 2
        top_half = group.tail(half)["realized_return"].mean()
        bottom_half = group.head(half)["realized_return"].mean()
        spread = top_half - bottom_half
        hit_results.append({
            "date": date,
            "top_half_return": top_half,
            "bottom_half_return": bottom_half,
            "spread": spread,
        })

    if hit_results:
        hit_df = pd.DataFrame(hit_results)
        avg_top = hit_df["top_half_return"].mean()
        avg_bottom = hit_df["bottom_half_return"].mean()
        avg_spread = hit_df["spread"].mean()
        win_rate = (hit_df["spread"] > 0).mean()
    else:
        avg_top = avg_bottom = avg_spread = win_rate = 0.0

    conn.close()

    # Print report
    print(f"\n{'='*60}")
    print(f"  Check Results")
    print(f"{'='*60}")
    print(f"  Updated this run: {updated_count}")
    print(f"  Total checked: {n_checked}")
    print(f"{'='*60}\n")

    print(f"  Predictive Power (across all {n_checked} checked rows):")
    print(f"    Spearman rho: {rho:+.4f} (p={pval:.3f})")
    print()

    if hit_results:
        print(f"  Hit Rate (top half vs bottom half by date):")
        print(f"    Top half avg return: {avg_top:+.2%}")
        print(f"    Bottom half avg return: {avg_bottom:+.2%}")
        print(f"    Spread (top - bottom): {avg_spread:+.2%}")
        print(f"    Win rate (spread > 0): {win_rate:.0%} ({sum(1 for r in hit_results if r['spread'] > 0)}/{len(hit_results)} dates)")
    else:
        print(f"  Hit Rate: Insufficient data (need groups per date)")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
