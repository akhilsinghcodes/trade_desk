"""
Log predictions to SQLite: run the production engine (XGBoost 5d-return
regressor, quintile selection, inverse-vol weighting), store each ticker's
score/weight + current price for later scoring against realized returns.

Production engine per the session's own findings, in order:
- return_model.pkl (XGBoost, 5d forward return) — the only model with a
  real, walk-forward-validated, bootstrap-CI-excludes-zero edge. NOT the
  ridge master_model — that one's near-zero betas, unvalidated, demoted.
- Quintile (20%) selection, not decile — the only book-size configuration
  whose bootstrap Sharpe CI excluded zero.
- Inverse-vol weighting (std_20) — confirmed the real lever behind every
  positive result this session; equal-weight was never once a winner.

Every logged row is stamped with a model_version (return_model.pkl's file
hash) so historical predictions stay attributable to the exact model that
made them, even after retraining.

Usage:
  python log_predictions.py                    # All DEFAULT_TICKERS
  python log_predictions.py [space-separated tickers]
"""
import sys
import os
import sqlite3
import hashlib

import yfinance as yf
import pandas as pd
import numpy as np
import pickle

sys.path.insert(0, os.path.dirname(__file__))

from modules.ml_dataset import build_live_features, get_feature_cols, DEFAULT_TICKERS

DB_PATH = "predictions.db"
MODEL_PATH = "models/return_model.pkl"
QUINTILE_FRAC = 0.20


def _init_db():
    """Create predictions and predict_watchlist tables if they don't exist,
    and add columns from later revisions to an existing table (idempotent)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            date TEXT,
            ticker TEXT,
            pred_return REAL,
            weight REAL,
            leg TEXT,
            model_version TEXT,
            price_at_prediction REAL,
            checked INTEGER DEFAULT 0,
            realized_return REAL,
            realized_price REAL,
            checked_date TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predict_watchlist (
            ticker TEXT PRIMARY KEY
        )
    """)
    # Migrate older DBs (had momentum/squeeze/catalyst/safety/math_score instead)
    cursor.execute("PRAGMA table_info(predictions)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col, decl in [("pred_return", "REAL"), ("weight", "REAL"), ("leg", "TEXT"),
                       ("model_version", "TEXT")]:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col} {decl}")
    conn.commit()
    conn.close()


def _model_version(model_path: str) -> str:
    """Short hash of the model file — identifies exactly which trained
    weights produced a given prediction, survives retraining."""
    with open(model_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _fetch_current_price(ticker: str) -> float | None:
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


def rank_and_weight(live_df: pd.DataFrame, preds: np.ndarray, quintile_frac: float = QUINTILE_FRAC) -> pd.DataFrame:
    """
    Quintile selection + inverse-vol weighting, same construction validated
    in modules/portfolio_backtest.py's "C" configuration. Returns live_df
    with pred_return, weight (signed, long positive / short negative, 0 if
    excluded from both legs), and leg ("long"/"short"/"none") columns.
    """
    df = live_df.copy()
    df["pred_return"] = preds
    n = len(df)
    q = max(1, int(n * quintile_frac))
    ranked = df.sort_values("pred_return")
    long_names = set(ranked.tail(q)["ticker"])
    short_names = set(ranked.head(q)["ticker"])

    def _inv_vol_weights(names):
        sub = df[df["ticker"].isin(names)]
        v = sub.set_index("ticker")["std_20"].replace(0, np.nan)
        inv = (1 / v).dropna()
        if inv.empty:
            return pd.Series(1.0 / len(names), index=list(names))
        return inv / inv.sum()

    w_long = _inv_vol_weights(long_names) if long_names else pd.Series(dtype=float)
    w_short = _inv_vol_weights(short_names) if short_names else pd.Series(dtype=float)

    df["weight"] = 0.0
    df["leg"] = "none"
    df.loc[df["ticker"].isin(long_names), "weight"] = df["ticker"].map(w_long).fillna(0.0)
    df.loc[df["ticker"].isin(long_names), "leg"] = "long"
    df.loc[df["ticker"].isin(short_names), "weight"] = -df["ticker"].map(w_short).fillna(0.0)
    df.loc[df["ticker"].isin(short_names), "leg"] = "short"
    return df


def main():
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
    else:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT ticker FROM predict_watchlist ORDER BY ticker")
        watchlist_rows = cursor.fetchall()
        conn.close()
        tickers = [row[0] for row in watchlist_rows] if watchlist_rows else DEFAULT_TICKERS

    print(f"\n{'='*60}")
    print("  Log Predictions — production engine (XGBoost + quintile + inv-vol)")
    print(f"{'='*60}")
    print(f"  Tickers : {len(tickers)} ({', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''})")
    print(f"{'='*60}\n")

    _init_db()

    print("Step 1/3: Building live features (same 54-column feature set the model trained on)...")
    try:
        live_df = build_live_features(tickers=tickers, period="1y")
    except ValueError as e:
        print(f"\nError: {e}")
        return
    print(f"  Features built: {len(live_df)} tickers\n")

    print("Step 2/3: Loading return_model.pkl...")
    if not os.path.exists(MODEL_PATH):
        print("\n" + "="*60)
        print(f"  No trained model found at {MODEL_PATH}")
        print("  Run 'python train_return.py' first.")
        print("="*60 + "\n")
        return
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    version = _model_version(MODEL_PATH)
    print(f"  Model loaded (version {version})\n")

    if len(live_df) < 10:
        print(f"  Only {len(live_df)} tickers scored — quintile selection needs a real universe "
              f"(10+ names) to mean anything. Add more tickers to predict_watchlist or use DEFAULT_TICKERS.")
        return

    X = live_df[get_feature_cols()].fillna(0).values.astype(np.float32)
    preds = model.predict(X)
    ranked_df = rank_and_weight(live_df, preds)

    print("Step 3/3: Fetching current prices...")
    prices = []
    for ticker in ranked_df["ticker"]:
        price = _fetch_current_price(ticker)
        prices.append(price if price is not None else 0.0)
    ranked_df["price_at_prediction"] = prices

    print("  Inserting into database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    count = 0
    for _, row in ranked_df.iterrows():
        if row["price_at_prediction"] == 0.0:
            continue
        cursor.execute("""
            INSERT INTO predictions
            (date, ticker, pred_return, weight, leg, model_version, price_at_prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(row["date"]), row["ticker"], float(row["pred_return"]),
            float(row["weight"]), row["leg"], version, float(row["price_at_prediction"]),
        ))
        count += 1
    conn.commit()
    conn.close()

    n_long = (ranked_df["leg"] == "long").sum()
    n_short = (ranked_df["leg"] == "short").sum()
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  Rows written: {count}  (long: {n_long}, short: {n_short}, excluded: {count - n_long - n_short})")
    print(f"  Model version: {version}")
    print(f"  Date: {live_df['date'].max()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
