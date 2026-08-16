"""
Meta-labeling: secondary filter on primary model's positive predictions.

The primary model (return_model.pkl) predicts 5-day forward returns and ranks stocks.
Meta-labeling trains a secondary classifier to predict whether a specific primary-model
"long" call would have been profitable (hit take-profit before stop-loss) using
trade-quality features like volatility, liquidity, execution conditions.

Input: feature panel with high/low/close + features
Output: meta_label column (1=TP hit, 0=stop/neither)

Technique from Lopez de Prado's "Advances in Financial Machine Learning"
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from modules.ml_dataset import LABEL_LOOKAHEAD_DAYS

# Meta-labeling uses "execution quality" features, not directional signals
META_LABEL_FEATURES = [
    "atr_pct", "std_5", "std_10", "std_20", "std_60",
    "vma_20", "cmf_20d", "volume_surge_20d"
]


def _fold_dates(df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """
    Quarterly expanding windows, matching return_model.py's fold structure.
    Purge/embargo gap of LABEL_LOOKAHEAD_DAYS before each test start to avoid leakage.
    """
    quarters = pd.date_range("2015-01-01", df["date"].max(), freq="QS")
    folds = []
    for q in quarters[4:]:  # need at least 1y training
        test_start = q
        train_end = test_start - pd.Timedelta(days=LABEL_LOOKAHEAD_DAYS + 1)
        test_end = q + pd.offsets.QuarterEnd()
        train_start = df["date"].min()
        folds.append((str(train_start.date()), str(train_end.date()),
                      str(test_start.date()), str(test_end.date())))
    return folds


def build_meta_labels(
    df: pd.DataFrame,
    primary_model,
    feature_cols: list[str],
    atr_col: str = "atr_pct",
    tp_mult: float = 2.0,
    stop_mult: float = 1.5,
    lookahead_days: int = 21
) -> pd.DataFrame:
    """
    Create meta-labels for primary model's positive predictions.

    Process:
    1. Run primary model predictions
    2. Filter for positive predictions ("long" calls only)
    3. For each such row, check if trade would hit TP before stop
    4. Label meta_label=1 if TP hit first, 0 otherwise

    Args:
        df: Feature panel (same as ml_dataset.build_training_dataset output)
            Must have: ticker, date, close, high, low, and feature columns
        primary_model: Trained XGBoost regressor (return_model.pkl loaded)
        feature_cols: List of feature column names for primary model
        atr_col: Column name for ATR percentages (default "atr_pct")
        tp_mult: Take-profit multiplier relative to stop distance (default 2.0)
        stop_mult: Stop-loss multiplier on ATR (default 1.5)
        lookahead_days: Forward window for triple-barrier check (default 21)

    Returns:
        DataFrame with original "long" rows plus meta_label column (WIN=1, LOSS=0)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Primary model predictions
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    df["pred_return"] = primary_model.predict(X)

    # Triple-barrier labeling MUST run per-ticker, on the full contiguous
    # daily series, BEFORE filtering to "long" rows — np.roll(-lag) means
    # "lag trading days later for this ticker" only if the array is one
    # ticker's unbroken chronological series. Filtering first and rolling
    # after (the original bug here) mixes rows across tickers and skips
    # filtered-out days, so "future price" ends up being some other
    # stock's price or a random later date — not a real label.
    all_labeled = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        c = g["close"].values
        h = g["high"].values
        l = g["low"].values
        atr_pct = g[atr_col].values

        atr_abs = atr_pct * c
        n = len(c)
        labels = np.full(n, "NEUTRAL", dtype=object)

        stop = c - stop_mult * atr_abs
        tp = c + tp_mult * stop_mult * atr_abs

        for lag in range(1, lookahead_days + 1):
            future_high = np.roll(h, -lag).astype(float)
            future_low = np.roll(l, -lag).astype(float)
            if lag <= n:
                future_high[-lag:] = np.nan
                future_low[-lag:] = np.nan
            undecided = labels == "NEUTRAL"
            labels[undecided & (future_high >= tp)] = "WIN"
            labels[undecided & (future_low <= stop)] = "LOSS"

        if n > lookahead_days:
            labels[-lookahead_days:] = "NEUTRAL"
        else:
            labels[:] = "NEUTRAL"

        g["meta_label"] = (labels == "WIN").astype(int)
        g["_label_raw"] = labels
        all_labeled.append(g)

    labeled = pd.concat(all_labeled, ignore_index=True)

    # NOW filter: only rows where primary model said "long" AND the
    # triple-barrier actually resolved (not NEUTRAL/no-lookahead-window)
    result = labeled[(labeled["pred_return"] > 0) & (labeled["_label_raw"] != "NEUTRAL")].copy()
    result = result.drop(columns=["_label_raw"])

    if len(result) == 0:
        print("  No positive predictions with resolved labels found")

    return result


def train_meta_model(
    df_with_meta_labels: pd.DataFrame,
    feature_cols: list[str] = None
) -> XGBClassifier:
    """
    Train secondary XGBoost classifier on meta-labeled data.

    Uses walk-forward validation matching return_model.py's fold structure.
    Reports accuracy/precision/recall/AUC per fold.

    Args:
        df_with_meta_labels: DataFrame from build_meta_labels()
            Must have: date, meta_label, and feature columns
        feature_cols: Features to use. Default: META_LABEL_FEATURES

    Returns:
        Trained XGBoost classifier
    """
    if feature_cols is None:
        feature_cols = META_LABEL_FEATURES

    df = df_with_meta_labels.copy()
    df["date"] = pd.to_datetime(df["date"])

    print(f"\n  Meta-Model Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    print(f"  Class balance: {(df['meta_label']==1).mean():.1%} WIN / {(df['meta_label']==0).mean():.1%} LOSS")

    # Walk-forward validation
    folds = _fold_dates(df)
    fold_results = []

    print(f"  Walk-forward: {len(folds)} quarterly folds\n")

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        train_df = df[(df["date"] >= tr_start) & (df["date"] <= tr_end)]
        test_df = df[(df["date"] >= te_start) & (df["date"] <= te_end)]

        if len(train_df) < 200 or len(test_df) < 30:
            continue

        X_tr = train_df[feature_cols].fillna(0).values.astype(np.float32)
        y_tr = train_df["meta_label"].values
        X_te = test_df[feature_cols].fillna(0).values.astype(np.float32)
        y_te = test_df["meta_label"].values

        # Class balance handling
        pos_weight = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 0.5)

        model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_pred_proba = model.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_te, y_pred_proba)
        except Exception:
            auc = 0.5

        fold_results.append({
            "fold": i, "te_start": te_start, "te_end": te_end,
            "n_train": len(train_df), "n_test": len(test_df),
            "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "auc": float(auc),
            "win_pct": float(y_te.mean())
        })

        print(f"  [{te_start[:7]}] train={len(train_df):>6,}  test={len(test_df):>5,}  "
              f"acc={acc:>5.1%}  prec={prec:.1%}  rec={rec:.1%}  auc={auc:.3f}  WIN%={y_te.mean():.0%}")

    if fold_results:
        avg_acc = np.mean([f["accuracy"] for f in fold_results])
        avg_auc = np.mean([f["auc"] for f in fold_results])
        avg_prec = np.mean([f["precision"] for f in fold_results])
        print(f"\n  Avg accuracy: {avg_acc:.1%}  avg AUC: {avg_auc:.3f}  avg precision: {avg_prec:.1%}")
    else:
        print("  No valid folds (insufficient training or test data)")

    # Train final model on all data
    print("\n  Training final meta model on all data...")
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df["meta_label"].values
    pos_weight = max((y == 0).sum() / max((y == 1).sum(), 1), 0.5)

    final_model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    final_model.fit(X, y)

    return final_model


if __name__ == "__main__":
    from modules.ml_dataset import build_training_dataset, get_feature_cols

    # Check if primary model exists
    model_path = "models/return_model.pkl"
    if not os.path.exists(model_path):
        print(f"Primary model not found at {model_path}")
        print("Build it first with: python -m train_return")
        sys.exit(0)

    print("Loading primary model...")
    with open(model_path, "rb") as f:
        primary_model = pickle.load(f)

    print("Building dataset...")
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "JPM", "JNJ", "XOM"
    ]
    df = build_training_dataset(tickers, period="5y")

    feature_cols = get_feature_cols()

    print("\nBuilding meta-labels...")
    df_meta = build_meta_labels(df, primary_model, feature_cols)

    if df_meta.empty:
        print("\nNo meta-labels generated (no positive predictions or data issues)")
        sys.exit(0)

    n_long = df_meta.shape[0]
    n_win = (df_meta["meta_label"] == 1).sum()
    n_loss = (df_meta["meta_label"] == 0).sum()

    print(f"\nMeta-labeling results:")
    print(f"  Primary model 'long' calls: {n_long:,}")
    print(f"  Would hit TP (meta_label=1): {n_win:,} ({n_win/max(n_long, 1):.1%})")
    print(f"  Would hit stop (meta_label=0): {n_loss:,} ({n_loss/max(n_long, 1):.1%})")

    if len(df_meta) < 100:
        print("\nWarning: too few labeled rows for walk-forward validation")
        sys.exit(0)

    print("\nTraining meta model...")
    meta_model = train_meta_model(df_meta)
    print("\n✓ Meta-model training complete")
