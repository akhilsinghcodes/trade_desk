"""ML model prediction for the analyze page — runs the published model
in-process, live, for any ticker. The trained model (models/return_model.pkl)
is published by trade_ml; trade_desk owns feature computation and inference
from that point on (see modules/ml_features.py), no dependency on trade_ml
at request time.

Per-ticker track record (models/ticker_track_record.json) only exists for
the 149 tickers trade_ml has walk-forward backtested — shown when available,
falls back to the model's aggregate stats otherwise.
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from modules.ml_features import compute_live_features, compute_live_features_batch, FEATURE_COLS

_MODELS_DIR = Path(__file__).parent.parent / "models"

# Above this fraction of missing features, treat the prediction as unreliable —
# XGBoost doesn't crash on NaN inputs (routes them down a learned default
# branch), so a mostly-empty feature vector (e.g. from a Yahoo API outage
# mid-fetch) still returns a confident-looking number with no signal it's
# built on garbage. Flagged, not silently trusted.
_LOW_CONFIDENCE_NAN_THRESHOLD = 0.25


def _load_json(name: str) -> dict:
    path = _MODELS_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(_MODELS_DIR / "return_model.pkl")


def _score_from_features(ticker: str, live: pd.DataFrame | None) -> dict | None:
    """Shared scoring logic — given an already-computed feature DataFrame,
    run the model and assemble the result dict. Used by both the single-
    ticker and batch entry points so they stay in sync."""
    if live is None or live.empty:
        return None

    model = _load_model()
    X = live[FEATURE_COLS].tail(1).apply(pd.to_numeric, errors="coerce")
    nan_pct = float(X.isna().mean(axis=1).iloc[0])
    pred = float(model.predict(X)[0])
    as_of = str(pd.Timestamp(live.iloc[-1]["date"]).date())

    track_record = _load_json("ticker_track_record.json")
    track = track_record.get(ticker)

    result = {
        "pred_return_5d": round(pred, 5),
        "as_of": as_of,
        "low_confidence": nan_pct > _LOW_CONFIDENCE_NAN_THRESHOLD,
        "missing_feature_pct": round(nan_pct, 3),
    }

    if track is not None:
        result.update({
            "has_track_record": True,
            "track_rho": track["rho"],
            "track_pval": track["pval"],
            "track_significant": track["significant"],
            "track_direction": track["direction"],
            "track_n_folds": track["n_folds"],
        })
    else:
        agg = _load_json("aggregate_stats.json")
        result.update({
            "has_track_record": False,
            "agg_n_tickers_tested": agg.get("n_tickers_tested"),
            "agg_n_positive": agg.get("n_positive"),
            "agg_n_significant": agg.get("n_significant"),
            "agg_median_rho": agg.get("median_rho"),
        })
    return result


def get_ml_verdict(ticker: str) -> dict | None:
    """Live prediction for any ticker with enough price history. Returns
    None only if feature computation fails (e.g. brand-new listing).
    No caching here — this module stays streamlit-free/testable in CI.
    Callers in the Streamlit app layer wrap this with st.cache_data."""
    model_path = _MODELS_DIR / "return_model.pkl"
    if not model_path.exists():
        return None
    live = compute_live_features(ticker, period="1y")
    return _score_from_features(ticker, live)


def get_ml_verdict_batch(tickers: list[str], progress_cb=None) -> dict[str, dict | None]:
    """Batch scan entry point (e.g. Top Movers) — fetches all tickers' OHLCV
    in one request via compute_live_features_batch instead of each ticker
    issuing its own, cutting the request volume that was triggering Yahoo's
    rate limiter under concurrent per-ticker scans. Fundamentals calls are
    still one-per-ticker (no batch API for those in yfinance).

    progress_cb(completed, total, ticker), if given, fires as each ticker
    finishes its fundamentals step."""
    model_path = _MODELS_DIR / "return_model.pkl"
    if not model_path.exists():
        return {t: None for t in tickers}
    features = compute_live_features_batch(tickers, period="1y", progress_cb=progress_cb)
    return {t: _score_from_features(t, features.get(t)) for t in tickers}
