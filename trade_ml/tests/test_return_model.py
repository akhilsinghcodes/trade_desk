"""
Extended tests for forward return regression model.

1. Lookahead sweep: 1d/3d/5d/10d/21d — where does signal peak?
2. Quintile breakdown: are all 5 buckets monotonically ordered?
3. Feature ablation: technical vs fundamental vs sentiment — what drives rho?

Run:
  .venv/bin/python tests/test_return_model.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor

from modules.ml_dataset import build_training_dataset, DEFAULT_TICKERS
from modules.ml_dataset import TECHNICAL_FEATURES, FUNDAMENTAL_FEATURES, SENTIMENT_FEATURES
from modules.return_model import _fold_dates, _make_xgb


def _walk_forward_rho(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[float, int, int]:
    """Returns (avg_rho, n_sig, n_folds)."""
    folds = _fold_dates(df)
    rhos = []
    n_sig = 0
    for tr_start, tr_end, te_start, te_end in folds:
        train = df[(df["date"] >= tr_start) & (df["date"] <= tr_end)]
        test = df[(df["date"] >= te_start) & (df["date"] <= te_end)]
        if len(train) < 2000 or len(test) < 200:
            continue
        X_tr = train[feature_cols].fillna(0).values.astype(np.float32)
        y_tr = train[target_col].values
        X_te = test[feature_cols].fillna(0).values.astype(np.float32)
        y_te = test[target_col].values
        model = _make_xgb()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        rho, pval = spearmanr(preds, y_te)
        rhos.append(rho)
        if pval < 0.05:
            n_sig += 1
    avg_rho = float(np.mean(rhos)) if rhos else 0
    return avg_rho, n_sig, len(rhos)


def add_returns(df: pd.DataFrame, days_list: list[int]) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    def _fwd(grp):
        grp = grp.sort_values("date")
        for d in days_list:
            grp[f"fwd_{d}d"] = grp["close"].pct_change(d).shift(-d).clip(-0.25, 0.25)
        return grp

    return df.groupby("ticker", group_keys=False).apply(_fwd)


def test_lookahead_sweep(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  TEST 1: Lookahead sweep")
    print(f"{'='*60}")
    print(f"  {'Days':<6} {'Avg Rho':>10} {'Sig Folds':>12} {'Signal?':>10}")
    print(f"  {'-'*45}")

    ALL_FEATS = TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES + SENTIMENT_FEATURES
    for days in [1, 3, 5, 10, 21]:
        col = f"fwd_{days}d"
        sub = df.dropna(subset=[col])
        rho, n_sig, n_folds = _walk_forward_rho(sub, ALL_FEATS, col)
        flag = "YES" if rho > 0.03 and n_sig >= n_folds // 3 else ("weak" if rho > 0.01 else "none")
        print(f"  {days}d     {rho:>+10.4f} {n_sig:>5}/{n_folds:<6} {flag:>10}")


def test_quintile_breakdown(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  TEST 2: Quintile breakdown (5-day, OOS)")
    print(f"{'='*60}")

    ALL_FEATS = TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES + SENTIMENT_FEATURES
    col = "fwd_5d"
    sub = df.dropna(subset=[col])
    folds = _fold_dates(sub)
    oos_rows = []

    for tr_start, tr_end, te_start, te_end in folds:
        train = sub[(sub["date"] >= tr_start) & (sub["date"] <= tr_end)]
        test = sub[(sub["date"] >= te_start) & (sub["date"] <= te_end)]
        if len(train) < 2000 or len(test) < 200:
            continue
        X_tr = train[ALL_FEATS].fillna(0).values.astype(np.float32)
        y_tr = train[col].values
        X_te = test[ALL_FEATS].fillna(0).values.astype(np.float32)
        model = _make_xgb()
        model.fit(X_tr, y_tr)
        test = test.copy()
        test["pred"] = model.predict(X_te)
        oos_rows.append(test)

    oos = pd.concat(oos_rows, ignore_index=True)

    # Quintile by date (rank within each day's cross-section)
    oos["quintile"] = oos.groupby("date")["pred"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    )

    print(f"  {'Quintile':<12} {'Avg 5d Ret':>12} {'N':>8}")
    print(f"  {'-'*35}")
    for q in [1, 2, 3, 4, 5]:
        bucket = oos[oos["quintile"] == q]
        avg_ret = bucket[col].mean()
        print(f"  Q{q} ({'bottom' if q==1 else 'top' if q==5 else '      '}){avg_ret:>+12.3%} {len(bucket):>8,}")

    # Monotonic check
    q_rets = [oos[oos["quintile"] == q][col].mean() for q in [1, 2, 3, 4, 5]]
    monotonic = all(q_rets[i] <= q_rets[i+1] for i in range(4))
    print(f"\n  Monotonically increasing Q1→Q5: {'YES' if monotonic else 'NO'}")


def test_feature_ablation(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  TEST 3: Feature group ablation (5-day)")
    print(f"{'='*60}")
    print(f"  {'Group':<25} {'Avg Rho':>10} {'Sig Folds':>12}")
    print(f"  {'-'*50}")

    col = "fwd_5d"
    sub = df.dropna(subset=[col])

    groups = {
        "Technical only": TECHNICAL_FEATURES,
        "Fundamental only": FUNDAMENTAL_FEATURES,
        "Sentiment only": SENTIMENT_FEATURES,
        "Tech + Sentiment": TECHNICAL_FEATURES + SENTIMENT_FEATURES,
        "All features": TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES + SENTIMENT_FEATURES,
    }

    for name, feats in groups.items():
        rho, n_sig, n_folds = _walk_forward_rho(sub, feats, col)
        print(f"  {name:<25} {rho:>+10.4f} {n_sig:>5}/{n_folds}")


if __name__ == "__main__":
    print("Loading dataset (using cached prices)...")
    raw = build_training_dataset(tickers=DEFAULT_TICKERS, period="15y")
    df = add_returns(raw, days_list=[1, 3, 5, 10, 21])
    print(f"Dataset: {len(df):,} rows\n")

    test_lookahead_sweep(df)
    test_quintile_breakdown(df)
    test_feature_ablation(df)

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")
