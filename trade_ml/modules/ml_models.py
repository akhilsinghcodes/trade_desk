"""
XGBoost classifier: trade outcome (WIN/LOSS) prediction.
WIN = price hits TP (entry+3×ATR) before stop (entry-1.5×ATR) within 21 days.
Walk-forward validation with expanding quarterly windows.
Bear market detection at inference: if SPY 63d < -3%, return HOLD (don't trade bears).
"""
import os
import json
import pickle
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
import shap

from modules.ml_dataset import get_feature_cols, TECHNICAL_FEATURES, FUNDAMENTAL_FEATURES, LABEL_LOOKAHEAD_DAYS
from modules.validation import passes_validation_gate


def _fold_dates(df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """
    Quarterly test windows with a purge/embargo gap of LABEL_LOOKAHEAD_DAYS
    before each test start. The WIN/LOSS label looks 21 days into the future,
    so training rows dated within 21 days of test_start have labels computed
    using price data that falls inside the test window — leakage across the
    fold boundary without this gap (Lopez de Prado, purged CV).
    """
    test_quarters = pd.date_range("2022-01-01", periods=12, freq="QS")
    folds = []
    for q_start in test_quarters:
        q_end = q_start + pd.offsets.QuarterEnd()
        train_end = q_start - pd.Timedelta(days=LABEL_LOOKAHEAD_DAYS + 1)
        if train_end < pd.Timestamp("2021-01-01"):
            continue
        folds.append(("2010-01-01", str(train_end.date()), str(q_start.date()), str(q_end.date())))
    return folds


def _make_xgb(pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=30,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _walk_forward(df: pd.DataFrame, feature_cols: list, label: str) -> list[dict]:
    folds = _fold_dates(df)
    fold_results = []
    for fold_i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        train_df = df[(df["date"] >= tr_start) & (df["date"] <= tr_end)]
        test_df  = df[(df["date"] >= te_start) & (df["date"] <= te_end)]

        if len(train_df) < 200 or len(test_df) < 30:
            print(f"  [{label}] Fold {fold_i+1}: skip (train={len(train_df)}, test={len(test_df)})")
            continue

        X_tr = train_df[feature_cols].values.astype(np.float32)
        y_tr = train_df["label_binary"].values
        X_te = test_df[feature_cols].values.astype(np.float32)
        y_te = test_df["label_binary"].values

        pos_weight = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 0.5)
        model = _make_xgb(pos_weight)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec  = recall_score(y_te, y_pred, zero_division=0)

        print(f"  [{label}] Fold {fold_i+1:2d} [{te_start}→{te_end}]: "
              f"acc={acc:.1%}  prec={prec:.1%}  rec={rec:.1%}  "
              f"n={len(test_df):,}  WIN%={y_te.mean():.0%}")

        fold_results.append({
            "fold": fold_i + 1, "regime": label,
            "train_start": tr_start, "train_end": tr_end,
            "test_start": te_start, "test_end": te_end,
            "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
            "n_test": len(test_df), "n_train": len(train_df),
        })
    return fold_results


def _train_final(df: pd.DataFrame, feature_cols: list, label: str, model_dir: str, filename: str) -> dict:
    X = df[feature_cols].values.astype(np.float32)
    y = df["label_binary"].values
    pos_weight = max((y == 0).sum() / max((y == 1).sum(), 1), 0.5)

    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=20,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)
    print(f"  [{label}] Trained 400 rounds on {len(df):,} rows ({time.time()-t0:.0f}s)")

    # SHAP
    sample_idx = np.random.choice(len(X), min(2000, len(X)), replace=False)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X[sample_idx])
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_ranking = sorted(zip(feature_cols, mean_abs_shap.tolist()), key=lambda x: x[1], reverse=True)

    print(f"  [{label}] Top 5 SHAP: {', '.join(f for f, _ in shap_ranking[:5])}")

    with open(os.path.join(model_dir, filename), "wb") as f:
        pickle.dump(model, f)

    return {"shap_ranking": shap_ranking, "label_balance": {"WIN": float((y==1).mean()), "LOSS": float((y==0).mean())}}


def train_models(df: pd.DataFrame, model_dir: str = "models/") -> dict:
    os.makedirs(model_dir, exist_ok=True)

    feature_cols = get_feature_cols()
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    print(f"\n  Features ({len(feature_cols)}): {', '.join(feature_cols[:8])}...")

    # ── Walk-forward CV ──────────────────────────────────────────────────────
    folds = _fold_dates(df)
    print(f"  Walk-forward: {len(folds)} quarterly folds\n")

    fold_results = _walk_forward(df, feature_cols, "model")

    if fold_results:
        avg_acc = np.mean([f["accuracy"] for f in fold_results])
        print(f"\n  Walk-forward avg accuracy: {avg_acc:.1%}")

        # Validation gate on accuracy
        gate_result = passes_validation_gate(fold_results, metric_key="accuracy", combined_threshold=0.55, half_threshold=0.5)
        status = "PASS" if gate_result["passed"] else "FAIL"
        print(f"  Validation gate [{status}]: {gate_result['reason']}")
    else:
        gate_result = {"passed": False, "reason": "No fold results"}

    # ── Final model ──────────────────────────────────────────────────────────
    print("\n  Training final model on all data...")
    info = _train_final(df, feature_cols, "model", model_dir, "model.pkl")

    # ── Save artifacts ───────────────────────────────────────────────────────
    with open(os.path.join(model_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    feature_meta = {}
    for feat in feature_cols:
        if feat in TECHNICAL_FEATURES:
            ftype = "technical"
        elif feat in FUNDAMENTAL_FEATURES:
            ftype = "fundamental"
        else:
            ftype = "sentiment"
        feature_meta[feat] = {
            "type": ftype,
            "shap_importance": dict(info["shap_ranking"]).get(feat, 0.0),
        }
    feature_meta["_walk_forward_accuracy"] = [f["accuracy"] for f in fold_results]
    with open(os.path.join(model_dir, "feature_meta.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)

    training_summary = {
        "n_samples": len(df),
        "n_tickers": df["ticker"].nunique() if "ticker" in df.columns else None,
        "label_balance": info["label_balance"],
        "walk_forward_folds": fold_results,
        "walk_forward_avg_accuracy": float(avg_acc) if fold_results else None,
        "validation_gate": gate_result,
        "top_shap_features": [f for f, _ in info["shap_ranking"][:10]],
    }
    with open(os.path.join(model_dir, "training_summary.json"), "w") as f:
        json.dump(training_summary, f, indent=2)

    print(f"  Saved: {model_dir}/model.pkl, feature_cols.json, feature_meta.json, training_summary.json")
    return training_summary


def load_models(model_dir: str = "models/") -> dict | None:
    model_path = os.path.join(model_dir, "model.pkl")
    cols_path = os.path.join(model_dir, "feature_cols.json")
    if not os.path.exists(model_path) or not os.path.exists(cols_path):
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(cols_path) as f:
        feature_cols = json.load(f)
    return {"model": model, "feature_cols": feature_cols}
