"""
PEAD model: walk-forward XGBoost on earnings events.
One model, expanding window, quarterly folds.
"""
import json
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from modules.pead_dataset import PEAD_FEATURES

MODEL_DIR = "models/"


def _make_xgb(pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def walk_forward_pead(df: pd.DataFrame) -> dict:
    """
    Walk-forward CV on earnings events. Folds by calendar year of earnings_date.
    Train: all events before test year. Test: events in that year.
    Uses 2017+ for first test fold (need >=3y training data).
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["earnings_date"]).dt.year

    test_years = sorted(df["year"].unique())
    # Need at least 3 years of training, so first test year is min+3
    min_year = test_years[0]
    test_years = [y for y in test_years if y >= min_year + 3]

    all_preds = []
    fold_results = []

    for test_year in test_years:
        train = df[df["year"] < test_year]
        test = df[df["year"] == test_year]

        if len(train) < 50 or len(test) < 10:
            continue

        X_tr = train[PEAD_FEATURES].fillna(0).values.astype(np.float32)
        y_tr = train["label_binary"].values
        X_te = test[PEAD_FEATURES].fillna(0).values.astype(np.float32)
        y_te = test["label_binary"].values

        pos_weight = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 0.5)
        model = _make_xgb(pos_weight)
        model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_te, preds)
        try:
            auc = roc_auc_score(y_te, probs)
        except Exception:
            auc = 0.5

        fold_results.append({"year": test_year, "n": len(test), "acc": acc, "auc": auc})
        test = test.copy()
        test["pead_prob"] = probs
        all_preds.append(test)

        print(f"  Fold {test_year}: train={len(train):,} test={len(test):,}  acc={acc:.1%}  auc={auc:.3f}")

    avg_acc = np.mean([r["acc"] for r in fold_results]) if fold_results else 0
    avg_auc = np.mean([r["auc"] for r in fold_results]) if fold_results else 0
    print(f"\n  Walk-forward avg: acc={avg_acc:.1%}  auc={avg_auc:.3f}")

    oos_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return {"fold_results": fold_results, "avg_acc": avg_acc, "avg_auc": avg_auc, "oos_df": oos_df}


def backtest_pead(oos_df: pd.DataFrame, threshold: float = 0.55) -> dict:
    """
    Signal backtest: take earnings events where pead_prob >= threshold.
    Outcome is relative_ret_30d (stock return minus SPY return over 30d).
    """
    if oos_df.empty:
        return {}

    buy = oos_df[oos_df["pead_prob"] >= threshold].copy()
    sell = oos_df[oos_df["pead_prob"] < (1 - threshold)].copy()
    baseline = oos_df.copy()

    def stats(subset, label):
        if len(subset) == 0:
            return
        win_rate = (subset["label"] == "WIN").mean()
        avg_ret = subset["relative_ret_30d"].mean()
        std_ret = subset["relative_ret_30d"].std()
        sharpe = avg_ret / (std_ret + 1e-8) * np.sqrt(4)  # quarterly-ish
        print(f"  {label:<30}: n={len(subset):>4}  win={win_rate:.1%}  avg_rel_ret={avg_ret:+.2%}  sharpe={sharpe:.2f}")

    print(f"\n  PEAD Backtest (OOS only):")
    print(f"  {'Strategy':<30}: {'n':>4}  {'win%':>6}  {'avg_rel_ret':>11}  {'sharpe':>7}")
    stats(baseline, "Baseline (all events)")
    stats(buy, f"PEAD BUY (prob>={threshold})")
    stats(sell, f"PEAD SELL (prob<{1-threshold:.2f})")

    return {
        "n_buy": len(buy),
        "n_sell": len(sell),
        "baseline_win_rate": float((baseline["label"] == "WIN").mean()),
        "buy_win_rate": float((buy["label"] == "WIN").mean()) if len(buy) else 0,
        "buy_avg_ret": float(buy["relative_ret_30d"].mean()) if len(buy) else 0,
    }


def train_final_pead(df: pd.DataFrame, model_dir: str = MODEL_DIR) -> dict:
    """Train final model on all data and save."""
    import os
    os.makedirs(model_dir, exist_ok=True)

    X = df[PEAD_FEATURES].fillna(0).values.astype(np.float32)
    y = df["label_binary"].values
    pos_weight = max((y == 0).sum() / max((y == 1).sum(), 1), 0.5)

    model = _make_xgb(pos_weight)
    model.fit(X, y)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X[:500])
        mean_abs = np.abs(sv).mean(axis=0)
        top_feats = [PEAD_FEATURES[i] for i in np.argsort(mean_abs)[::-1][:5]]
    except Exception:
        top_feats = PEAD_FEATURES[:5]

    with open(f"{model_dir}/pead_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{model_dir}/pead_features.json", "w") as f:
        json.dump(PEAD_FEATURES, f)
    with open(f"{model_dir}/pead_summary.json", "w") as f:
        json.dump({"top_shap_features": top_feats, "n_train": len(df)}, f)

    print(f"\n  Final PEAD model saved → {model_dir}/pead_model.pkl")
    print(f"  Top features: {top_feats}")
    return {"top_shap_features": top_feats}
