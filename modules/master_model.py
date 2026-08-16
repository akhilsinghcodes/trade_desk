"""
Master model — live scoring only (inference side, Math Ranker stage).

Sub-Models 1-4 (momentum, squeeze, catalyst, safety) are fit offline in
trade_ml against historical forward returns; betas are persisted to a small
JSON file (see trade_ml/modules/master_model.py). This module loads that
file and applies it to trade_desk's own real-time features (already
fetched/cached by app/services/analysis.py — nothing here re-fetches
anything). Safety (altman_z) is a z-scored pillar like the other 3, not a
hard gate — the fitted beta decides how much it matters.

The LLM Arbiter (Model 5's second stage) runs offline in trade_ml as part of
its batch judge run, not here — this module only computes math_score.

Pillar definitions here are trade_desk's own signals, not a feature-for-
feature match to trade_ml's training set — they're conceptually parallel
(both directional composites) but not identical columns, so treat betas
trained there as an approximation, not an exact transfer.
"""
import json
import os
import numpy as np
import pandas as pd

PILLARS = ("momentum", "squeeze", "catalyst", "safety")
DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "trade_ml", "models", "master_model_weights.json"
)


def load_weights(path: str = DEFAULT_WEIGHTS_PATH) -> dict | None:
    """Load betas fit offline by trade_ml. Returns None if not yet trained."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def features_from_analysis(analysis: dict) -> dict:
    """Pull raw numeric features out of app.services.analysis.run_analysis()'s
    output — trade_desk already fetched/cached all of this per ticker."""
    short_data = analysis.get("short_data") or {}
    analyst_data = analysis.get("analyst_data") or {}
    options_data = analysis.get("options_data") or {}
    altman_data = analysis.get("altman_data") or {}

    df = analysis.get("df")
    vol_ratio = 1.0
    if df is not None and "vol_ratio" in df and pd.notna(df["vol_ratio"].iloc[-1]):
        vol_ratio = float(df["vol_ratio"].iloc[-1])

    return {
        "coppock": (analysis.get("coppock_data") or {}).get("signal", 0.0),
        "ridge_slope": (analysis.get("ridge_slope_data") or {}).get("signal", 0.0),
        "vol_ratio": vol_ratio,
        "short_pct_float": short_data.get("short_pct_float") or 0.0,
        "short_ratio": short_data.get("short_ratio") or 0.0,
        "analyst_upside": analyst_data.get("price_targets", {}).get("upside_pct") or 0.0,
        "put_call_ratio": options_data.get("put_call_ratio") or 1.0,
        "altman_z": altman_data.get("z_score"),
    }


def cross_sectional_zscores(features_by_ticker: dict[str, dict]) -> dict[str, dict]:
    """Z-score each feature column across the tickers being scored together."""
    df = pd.DataFrame(features_by_ticker).T.apply(pd.to_numeric, errors="coerce")
    z = (df - df.mean()) / df.std(ddof=0).replace(0, np.nan)
    return z.fillna(0.0).T.to_dict()


def _pillar_score(z: dict, feature_names: list[str]) -> float:
    vals = [z[f] for f in feature_names if f in z]
    return float(np.mean(vals)) if vals else 0.0


def compute_pillar_scores(z_by_ticker: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for ticker, z in z_by_ticker.items():
        out[ticker] = {
            "momentum": _pillar_score(z, ["coppock", "ridge_slope", "vol_ratio"]),
            "squeeze": _pillar_score(z, ["short_pct_float", "short_ratio"]),
            "catalyst": _pillar_score(z, ["analyst_upside", "put_call_ratio"]),
            "safety": _pillar_score(z, ["altman_z"]),
        }
    return out


def score_tickers(tickers: list[str], period: str = "1y",
                   weights_path: str = DEFAULT_WEIGHTS_PATH) -> pd.DataFrame:
    """
    Math Ranker only (live). Runs trade_desk's existing run_analysis() per
    ticker (cached), z-scores cross-sectionally within this batch (including
    altman_z as the safety pillar — no hard gating), applies betas loaded
    from trade_ml's offline fit (or falls back to equal weights if no
    weights file exists yet). The LLM Arbiter stage doesn't run here — that's
    trade_ml's offline judge batch.
    """
    from app.services.analysis import run_analysis

    raw = {t: features_from_analysis(run_analysis(t, period)) for t in tickers}
    z = cross_sectional_zscores(raw)
    pillars = compute_pillar_scores(z)

    weights = load_weights(weights_path)
    if weights:
        betas = dict(zip(weights["pillars"], weights["betas"]))
        intercept = weights["intercept"]
    else:
        betas = {p: 1.0 / len(PILLARS) for p in PILLARS}
        intercept = 0.0

    rows = {}
    for t, p in pillars.items():
        math_score = intercept + sum(betas.get(pl, 0.0) * p[pl] for pl in PILLARS)
        rows[t] = {"math_score": math_score, **p}

    return pd.DataFrame(rows).T.sort_values("math_score", ascending=False)
