"""
Signal backtest: how much money did BUY signals actually make?

Uses the historical dataset (already built with real TP/stop outcomes).
Tests ML model vs rule-based score vs random baseline vs SPY buy-and-hold.

Run:
  python -m modules.backtest_signal
  python -m modules.backtest_signal --model-dir models/ --period 15y
"""
import argparse
import pickle
import json
import numpy as np
import pandas as pd


# ── Rule-based score (simplified combined_score from features) ─────────────────
# Mimics trade_experimentation combined_score() logic using available feature columns.
# Each condition maps to a +1 (bullish) or -1 (bearish) signal.
# Score = mean of signals. BUY if > 0.15, SELL if < -0.15.

def rule_score(row: pd.Series) -> float:
    signals = []

    # Technical
    rsi = row.get("rsi", 0.5)  # normalized 0-1
    if rsi < 0.35:   signals.append(1.0)   # oversold = bullish
    elif rsi > 0.70: signals.append(-1.0)  # overbought = bearish
    else:            signals.append(0.0)

    signals.append(1.0 if row.get("macd_above_signal", 0) > 0.5 else -1.0)
    signals.append(1.0 if row.get("sma20_vs_sma50", 0) > 0 else -1.0)
    signals.append(1.0 if row.get("bb_position", 0.5) < 0.3 else
                  -1.0 if row.get("bb_position", 0.5) > 0.7 else 0.0)
    signals.append(1.0 if row.get("cmf_20d", 0) > 0.05 else
                  -1.0 if row.get("cmf_20d", 0) < -0.05 else 0.0)
    signals.append(1.0 if row.get("volume_surge_20d", 0) > 0.2 else 0.0)
    signals.append(1.0 if row.get("alpha_21d", 0) > 0.01 else
                  -1.0 if row.get("alpha_21d", 0) < -0.01 else 0.0)
    signals.append(1.0 if row.get("high_52w_proximity", 0) > 0.85 else
                  -1.0 if row.get("high_52w_proximity", 0) < 0.60 else 0.0)

    # Fundamental
    signals.append(1.0 if row.get("revenue_growth", 0) > 0.05 else
                  -1.0 if row.get("revenue_growth", 0) < 0 else 0.0)
    signals.append(1.0 if row.get("profit_margin", 0) > 0.10 else
                  -1.0 if row.get("profit_margin", 0) < 0 else 0.0)
    signals.append(1.0 if row.get("roe", 0) > 0.10 else
                  -1.0 if row.get("roe", 0) < 0 else 0.0)
    signals.append(1.0 if row.get("fcf_yield", 0) > 0.02 else
                  -1.0 if row.get("fcf_yield", 0) < 0 else 0.0)
    piotroski = row.get("piotroski_score", 0.5)
    signals.append(1.0 if piotroski > 0.6 else -1.0 if piotroski < 0.3 else 0.0)

    # Sentiment
    signals.append(1.0 if row.get("analyst_score", 0) > 0.3 else
                  -1.0 if row.get("analyst_score", 0) < -0.3 else 0.0)
    signals.append(1.0 if row.get("insider_net_ratio", 0) > 0.1 else
                  -1.0 if row.get("insider_net_ratio", 0) < -0.1 else 0.0)

    return float(np.mean(signals))


# ── Trade simulation ───────────────────────────────────────────────────────────

def simulate_trades(df: pd.DataFrame, signal_col: str, threshold: float = 0.55) -> dict:
    """
    For rows where signal >= threshold (BUY), check actual outcome (WIN/LOSS label).
    Returns trade stats.
    """
    buy_signals = df[df[signal_col] >= threshold]
    if len(buy_signals) == 0:
        return {"n_trades": 0, "win_rate": 0, "avg_return": 0, "sharpe": 0}

    wins = (buy_signals["label"] == "WIN").sum()
    losses = (buy_signals["label"] == "LOSS").sum()
    n = len(buy_signals)

    # Approximate return per trade:
    # WIN → +3×ATR / close ≈ atr_pct * 3 (avg across trades)
    # LOSS → -1.5×ATR / close ≈ atr_pct * 1.5
    avg_atr = buy_signals["atr_pct"].mean()
    avg_win_return = avg_atr * 3.0
    avg_loss_return = -avg_atr * 1.5
    win_rate = wins / n

    avg_return = win_rate * avg_win_return + (1 - win_rate) * avg_loss_return

    # Monthly returns for Sharpe (group by year-month)
    buy_signals = buy_signals.copy()
    buy_signals["ym"] = pd.to_datetime(buy_signals["date"]).dt.to_period("M")
    monthly_wins = buy_signals.groupby("ym").apply(
        lambda g: (g["label"] == "WIN").mean() * g["atr_pct"].mean() * 3.0 +
                  (g["label"] == "LOSS").mean() * g["atr_pct"].mean() * -1.5,
        include_groups=False
    )
    sharpe = float(monthly_wins.mean() / (monthly_wins.std() + 1e-8) * np.sqrt(12)) if len(monthly_wins) > 1 else 0.0

    return {
        "n_trades": int(n),
        "win_rate": float(win_rate),
        "avg_return_per_trade": float(avg_return),
        "annualized_return_est": float(avg_return * 12),  # rough: ~1 trade/month per stock
        "sharpe": float(sharpe),
        "wins": int(wins),
        "losses": int(losses),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run_backtest(model_dir: str = "models/", test_start: str = "2022-01-01") -> dict:
    # Load model
    model_path = f"{model_dir}/model.pkl"
    cols_path = f"{model_dir}/feature_cols.json"
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(cols_path) as f:
            feature_cols = json.load(f)
        has_model = True
    except FileNotFoundError:
        has_model = False
        print("  No trained model found — run train.py first")

    # Load dataset
    import yfinance as yf
    from modules.ml_dataset import build_training_dataset, DEFAULT_TICKERS

    print("Building dataset for backtest (this uses cached data if available)...")
    df = build_training_dataset(tickers=DEFAULT_TICKERS, period="15y")

    # Test window only (walk-forward: only evaluate on out-of-sample period)
    df["date"] = pd.to_datetime(df["date"])
    test_df = df[df["date"] >= test_start].copy()
    print(f"\nTest window: {test_start} → {df['date'].max().date()}  ({len(test_df):,} rows)\n")

    results = {}

    # ── 1. Baseline: random (buy everything) ──────────────────────────────────
    baseline_win = (test_df["label"] == "WIN").mean()
    baseline_atr = test_df["atr_pct"].mean()
    baseline_ret = baseline_win * baseline_atr * 3.0 + (1 - baseline_win) * baseline_atr * -1.5
    results["baseline_random"] = {
        "n_trades": len(test_df),
        "win_rate": float(baseline_win),
        "avg_return_per_trade": float(baseline_ret),
        "annualized_return_est": float(baseline_ret * 12),
        "sharpe": 0.0,
    }
    print(f"Baseline (buy everything):")
    print(f"  Win rate     : {baseline_win:.1%}")
    print(f"  Avg return   : {baseline_ret:.2%} per trade")

    # ── 2. Rule-based score (combined_score proxy) ───────────────────────────
    print("\nComputing rule-based scores...")
    test_df["rule_score"] = test_df.apply(rule_score, axis=1)

    # Normalize to 0-1 for threshold comparison
    r_min, r_max = test_df["rule_score"].min(), test_df["rule_score"].max()
    test_df["rule_signal"] = (test_df["rule_score"] - r_min) / (r_max - r_min + 1e-8)

    rule_result = simulate_trades(test_df, "rule_signal", threshold=0.60)
    results["rule_based"] = rule_result
    print(f"\nRule-based (combined_score proxy, threshold=0.60):")
    print(f"  Trades       : {rule_result['n_trades']:,}")
    print(f"  Win rate     : {rule_result['win_rate']:.1%}  (baseline {baseline_win:.1%})")
    print(f"  Avg return   : {rule_result['avg_return_per_trade']:.2%} per trade")
    print(f"  Sharpe       : {rule_result['sharpe']:.2f}")

    # ── 3. ML model — walk-forward only (true out-of-sample) ─────────────────
    # CRITICAL: must NOT use final model on test data — final model trained on all
    # data including test window. Instead, retrain walk-forward and collect fold predictions.
    if has_model:
        print("\nComputing ML model scores (walk-forward, true out-of-sample)...")
        from modules.ml_models import _fold_dates, _make_xgb

        folds = _fold_dates(df)
        oos_rows = []

        for tr_start, tr_end, te_start, te_end in folds:
            if te_start < test_start:
                continue
            train_mask = (df["date"] >= tr_start) & (df["date"] <= tr_end)
            test_mask  = (df["date"] >= te_start) & (df["date"] <= te_end)
            tr = df[train_mask]
            te = df[test_mask]
            if len(tr) < 500 or len(te) < 50:
                continue

            X_tr = tr[feature_cols].fillna(0).values.astype(np.float32)
            y_tr = tr["label_binary"].values
            X_te = te[feature_cols].fillna(0).values.astype(np.float32)

            pos_weight = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 0.5)
            fold_model = _make_xgb(pos_weight)
            fold_model.fit(X_tr, y_tr)

            probs = fold_model.predict_proba(X_te)[:, 1]
            te = te.copy()
            te["ml_prob"] = probs
            oos_rows.append(te)
            print(f"  Fold [{te_start}→{te_end}]: {len(te):,} rows")

        if oos_rows:
            oos_df = pd.concat(oos_rows, ignore_index=True)
            for thresh in [0.55, 0.60, 0.65]:
                ml_result = simulate_trades(oos_df, "ml_prob", threshold=thresh)
                results[f"ml_model_t{int(thresh*100)}"] = ml_result
                print(f"\nML model OOS (threshold={thresh}):")
                print(f"  Trades       : {ml_result['n_trades']:,}")
                print(f"  Win rate     : {ml_result['win_rate']:.1%}  (baseline {baseline_win:.1%})")
                print(f"  Avg return   : {ml_result['avg_return_per_trade']:.2%} per trade")
                print(f"  Sharpe       : {ml_result['sharpe']:.2f}")
        else:
            print("  No OOS folds found — check test_start date")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  BACKTEST SUMMARY (test window: {test_start}+)")
    print(f"{'='*55}")
    print(f"  {'Strategy':<35} {'Win%':>6} {'Avg Ret':>8} {'Sharpe':>7}")
    print(f"  {'-'*55}")
    print(f"  {'Baseline (buy everything)':<35} {baseline_win:>6.1%} {baseline_ret:>8.2%}   n/a")
    print(f"  {'Rule-based (score>0.60)':<35} {rule_result['win_rate']:>6.1%} {rule_result['avg_return_per_trade']:>8.2%} {rule_result['sharpe']:>7.2f}")
    if has_model:
        for thresh in [0.55, 0.60, 0.65]:
            r = results[f"ml_model_t{int(thresh*100)}"]
            print(f"  {f'ML model (prob>{thresh})':<35} {r['win_rate']:>6.1%} {r['avg_return_per_trade']:>8.2%} {r['sharpe']:>7.2f}")
    print(f"{'='*55}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/")
    parser.add_argument("--test-start", default="2022-01-01")
    args = parser.parse_args()
    run_backtest(model_dir=args.model_dir, test_start=args.test_start)
