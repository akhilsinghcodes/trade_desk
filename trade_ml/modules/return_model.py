"""
Forward return regression model.

Input : stock + date + 25 signal columns
Output: predicted 5-day forward return (pct)
Eval  : Spearman rank correlation OOS — does model rank stocks correctly?

Walk-forward expanding window. No TP/stop. No path dependency. No arbitrary thresholds.
"""
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor

from modules.ml_dataset import FEATURE_COLS
from modules.validation import passes_validation_gate

LOOKAHEAD_DAYS = 5
MODEL_FILE = "return_model.pkl"


def add_return_target(df: pd.DataFrame, days: int = LOOKAHEAD_DAYS) -> pd.DataFrame:
    """Add forward_return_Nd column. Groups by ticker to avoid cross-ticker leakage."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    close_col = "close" if "close" in df.columns else "Close"
    # groupby(...).apply(fn) that returns the whole group can silently drop the
    # grouping column depending on pandas version (confirmed on pandas 3.0) —
    # use transform on a single column instead, which never touches the rest.
    df["fwd_return"] = df.groupby("ticker")[close_col].transform(
        lambda s: s.pct_change(days).shift(-days)
    )
    df = df.dropna(subset=["fwd_return"])
    # Clip extreme returns (earnings gaps, halts) so they don't dominate
    df["fwd_return"] = df["fwd_return"].clip(-0.25, 0.25)
    return df


def _make_xgb() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _fold_dates(df: pd.DataFrame, embargo_days: int = LOOKAHEAD_DAYS) -> list:
    """
    Quarterly expanding folds from 2015 onwards, with a purge/embargo gap of
    embargo_days before each test start. Without this, training rows dated
    right before train_end have fwd_return labels computed using price data
    that falls inside the test window — direct leakage across the fold
    boundary (Lopez de Prado, purged CV).

    embargo_days MUST match whatever horizon add_return_target() was called
    with — a 126-day forward-return label needs a 126+ day gap, not the
    5-day default. Passing the wrong value here silently reintroduces the
    exact leakage bug this gap was built to fix, just at a longer horizon.
    """
    quarters = pd.date_range("2015-01-01", df["date"].max(), freq="QS")
    folds = []
    for q in quarters[4:]:  # need at least 1y training
        test_start = q
        train_end = test_start - pd.Timedelta(days=embargo_days + 1)
        test_end = q + pd.offsets.QuarterEnd()
        train_start = df["date"].min()
        folds.append((str(train_start.date()), str(train_end.date()),
                      str(test_start.date()), str(test_end.date())))
    return folds


def walk_forward_return(df: pd.DataFrame, embargo_days: int = LOOKAHEAD_DAYS) -> dict:
    df["date"] = pd.to_datetime(df["date"])
    folds = _fold_dates(df, embargo_days=embargo_days)
    oos_rows = []
    fold_results = []

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        train = df[(df["date"] >= tr_start) & (df["date"] <= tr_end)]
        test = df[(df["date"] >= te_start) & (df["date"] <= te_end)]
        if len(train) < 2000 or len(test) < 200:
            continue

        X_tr = train[FEATURE_COLS].fillna(0).values.astype(np.float32)
        y_tr = train["fwd_return"].values
        X_te = test[FEATURE_COLS].fillna(0).values.astype(np.float32)
        y_te = test["fwd_return"].values

        model = _make_xgb()
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        rho, pval = spearmanr(preds, y_te)
        mse = float(np.mean((preds - y_te) ** 2))

        fold_results.append({
            "fold": i, "te_start": te_start, "te_end": te_end,
            "n_train": len(train), "n_test": len(test),
            "spearman_rho": float(rho), "pval": float(pval), "mse": mse
        })

        test = test.copy()
        test["pred_return"] = preds
        oos_rows.append(test)

        sig = "*" if pval < 0.05 else " "
        print(f"  [{te_start[:7]}] train={len(train):>6,}  test={len(test):>5,}  "
              f"rho={rho:+.3f}{sig}  p={pval:.3f}")

    avg_rho = np.mean([r["spearman_rho"] for r in fold_results]) if fold_results else 0
    n_sig = sum(1 for r in fold_results if r["pval"] < 0.05)
    print(f"\n  Avg Spearman rho: {avg_rho:+.4f}  ({n_sig}/{len(fold_results)} folds significant p<0.05)")

    # Validation gate on Spearman rho (use as IC proxy). IC=0.10 is unrealistic for
    # single-name 5d equity returns; 0.03 combined + majority of folds positive is
    # the recalibrated bar (long-short spread is checked separately in train_return.py).
    gate_result = passes_validation_gate(fold_results, metric_key="spearman_rho",
                                          combined_threshold=0.03, half_threshold=0.0,
                                          min_frac_positive=0.5)
    status = "PASS" if gate_result["passed"] else "FAIL"
    print(f"  Validation gate [{status}]: {gate_result['reason']}")

    oos_df = pd.concat(oos_rows, ignore_index=True) if oos_rows else pd.DataFrame()
    return {"fold_results": fold_results, "avg_rho": avg_rho, "n_sig": n_sig,
            "n_folds": len(fold_results), "oos_df": oos_df, "validation_gate": gate_result}


def backtest_long_short(oos_df: pd.DataFrame, horizon_days: int = LOOKAHEAD_DAYS) -> dict:
    """
    Each day: rank all stocks by pred_return. Buy top quintile, short bottom quintile.
    Measure if top quintile actually outperforms bottom quintile.

    horizon_days MUST match the actual forward-return horizon the model was
    trained on — it drives the annualization factor (252/horizon_days
    "rebalances" per year). Passing the wrong value silently mis-annualizes
    the spread (e.g. treating a 126-day return as if rebalanced every 5 days
    would inflate the annualized number ~25x).
    """
    if oos_df.empty:
        return {}

    oos_df = oos_df.copy()
    oos_df["date"] = pd.to_datetime(oos_df["date"])

    results = []
    for date, group in oos_df.groupby("date"):
        if len(group) < 5:
            continue
        group = group.sort_values("pred_return")
        n = len(group)
        q = max(1, n // 5)
        top = group.tail(q)["fwd_return"].mean()
        bottom = group.head(q)["fwd_return"].mean()
        results.append({"date": date, "top_ret": top, "bottom_ret": bottom,
                        "spread": top - bottom})

    if not results:
        return {}

    res_df = pd.DataFrame(results)
    avg_spread = res_df["spread"].mean()
    avg_top = res_df["top_ret"].mean()
    avg_bottom = res_df["bottom_ret"].mean()
    hit_rate = (res_df["spread"] > 0).mean()

    print(f"\n  Long-Short Backtest (OOS, top vs bottom quintile):")
    print(f"  Top quintile avg {horizon_days}d return    : {avg_top:+.3%}")
    print(f"  Bottom quintile avg {horizon_days}d return : {avg_bottom:+.3%}")
    print(f"  Spread (top - bottom)            : {avg_spread:+.3%}")
    print(f"  Days where top > bottom          : {hit_rate:.1%}")

    # Annualize: ~252/horizon_days rebalances per year
    ann_spread = avg_spread * (252 / horizon_days)
    print(f"  Annualized spread (rough)        : {ann_spread:+.1%}")

    return {
        "avg_top_ret": float(avg_top),
        "avg_bottom_ret": float(avg_bottom),
        "avg_spread": float(avg_spread),
        "hit_rate": float(hit_rate),
        "ann_spread": float(ann_spread),
    }


def analyze_turnover_and_costs(oos_df: pd.DataFrame, decile_frac: float = 0.1,
                                bps_sweep: tuple = (5, 10, 20, 50),
                                horizon_days: int = LOOKAHEAD_DAYS) -> dict:
    """
    Top-decile-minus-bottom-decile spread and hit rate — the primary metric
    now (Spearman rho stays secondary; flat rho with moving hit-rate/spread
    means improvement concentrated in the tails, which rho is insensitive
    to by construction).

    Then: measure one-way turnover of decile membership PER REBALANCE (this
    backtest re-picks deciles every trading day — an overlapping-tranche
    horizon_days-return bet opened daily, not a buy-and-hold rebalance — so
    turnover is measured at that actual cadence: rebalances happen daily
    (252/yr), but each rebalance's spread is a horizon_days-forward return,
    so spread and turnover annualize by DIFFERENT factors — turnover by 252
    (rebalance frequency), spread by 252/horizon_days (payoff frequency).
    Getting this wrong silently inflates the spread ~horizon_days-fold; see
    the identical bug already fixed once in backtest_long_short.

    Sweeps round-trip transaction costs (5/10/20/50bps) against the
    per-rebalance spread to find where the gross edge actually survives.
    This is the real gate before anything is worth running live: a spread
    that dies at 20bps needs slower rebalancing or wider bands, not more
    features.
    """
    if oos_df.empty:
        return {}

    oos_df = oos_df.copy()
    oos_df["date"] = pd.to_datetime(oos_df["date"])

    daily = []
    prev_top, prev_bottom = None, None
    for date, group in oos_df.groupby("date"):
        if len(group) < 10:
            continue
        group = group.sort_values("pred_return")
        n = len(group)
        q = max(1, int(n * decile_frac))
        top_set = set(group.tail(q)["ticker"])
        bottom_set = set(group.head(q)["ticker"])
        top_ret = group.tail(q)["fwd_return"].mean()
        bottom_ret = group.head(q)["fwd_return"].mean()

        turnover = np.nan
        if prev_top is not None:
            # one-way turnover: fraction of each leg's names that changed,
            # averaged across both legs
            top_changed = len(top_set - prev_top) / q
            bottom_changed = len(bottom_set - prev_bottom) / q
            turnover = (top_changed + bottom_changed) / 2

        daily.append({"date": date, "top_ret": top_ret, "bottom_ret": bottom_ret,
                       "spread": top_ret - bottom_ret, "turnover": turnover})
        prev_top, prev_bottom = top_set, bottom_set

    if not daily:
        return {}

    res = pd.DataFrame(daily)
    gross_spread_per_rebalance = res["spread"].mean()  # a horizon_days-forward return, not a daily one
    hit_rate = (res["spread"] > 0).mean()
    turnover_per_rebalance = res["turnover"].mean()  # NaN-skipping mean
    annual_turnover = turnover_per_rebalance * 252  # rebalances happen daily regardless of horizon
    ann_factor = 252 / horizon_days  # but payoffs only realize every horizon_days

    print(f"\n  Top-Decile vs Bottom-Decile (primary metric, {decile_frac:.0%} deciles):")
    print(f"  Gross spread per rebalance ({horizon_days}d fwd): {gross_spread_per_rebalance:+.3%}")
    print(f"  Hit rate                  : {hit_rate:.1%}")
    print(f"  Gross annualized spread    : {gross_spread_per_rebalance * ann_factor:+.1%}")
    print(f"  Turnover per rebalance     : {turnover_per_rebalance:.1%}  (rebalance freq annualized: {annual_turnover:.0%})")

    print(f"\n  Cost sweep (round-trip bps applied once per rebalance):")
    print(f"  {'bps':>6}  {'net per rebalance':>18}  {'net annualized':>15}")
    cost_results = {}
    for bps in bps_sweep:
        cost_drag = turnover_per_rebalance * (bps / 10000)
        net_per_rebalance = gross_spread_per_rebalance - cost_drag
        net_annual = net_per_rebalance * ann_factor
        cost_results[bps] = {"net_per_rebalance_spread": float(net_per_rebalance), "net_annual_spread": float(net_annual)}
        print(f"  {bps:>4}bps  {net_per_rebalance:>+17.3%}  {net_annual:>+14.1%}")

    return {
        "gross_spread_per_rebalance": float(gross_spread_per_rebalance),
        "gross_annual_spread": float(gross_spread_per_rebalance * ann_factor),
        "hit_rate": float(hit_rate),
        "turnover_per_rebalance": float(turnover_per_rebalance),
        "annual_turnover": float(annual_turnover),
        "cost_sweep": cost_results,
    }


def train_final(df: pd.DataFrame, model_dir: str = "models/") -> dict:
    import os
    os.makedirs(model_dir, exist_ok=True)

    X = df[FEATURE_COLS].fillna(0).values.astype(np.float32)
    y = df["fwd_return"].values

    model = _make_xgb()
    model.fit(X, y)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X[:1000])
        mean_abs = np.abs(sv).mean(axis=0)
        top_feats = [FEATURE_COLS[i] for i in np.argsort(mean_abs)[::-1][:8]]
    except Exception:
        top_feats = FEATURE_COLS[:8]

    with open(f"{model_dir}/{MODEL_FILE}", "wb") as f:
        pickle.dump(model, f)
    with open(f"{model_dir}/return_features.json", "w") as f:
        json.dump(FEATURE_COLS, f)
    with open(f"{model_dir}/return_summary.json", "w") as f:
        json.dump({"top_shap_features": top_feats, "lookahead_days": LOOKAHEAD_DAYS}, f)

    print(f"\n  Return model saved → {model_dir}/{MODEL_FILE}")
    print(f"  Top SHAP features: {top_feats}")
    return {"top_shap_features": top_feats}
