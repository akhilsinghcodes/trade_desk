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
import streamlit as st

from modules.ml_features import compute_live_features, FEATURE_COLS

_MODELS_DIR = Path(__file__).parent.parent / "models"


def _load_json(name: str) -> dict:
    path = _MODELS_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(_MODELS_DIR / "return_model.pkl")


@st.cache_data(ttl=900, show_spinner=False)
def get_ml_verdict(ticker: str) -> dict | None:
    """Live prediction for any ticker with enough price history. Returns
    None only if feature computation fails (e.g. brand-new listing).
    Cached 15min — same-session reruns/re-clicks don't re-hit yfinance."""
    model_path = _MODELS_DIR / "return_model.pkl"
    if not model_path.exists():
        return None

    live = compute_live_features(ticker, period="1y")
    if live is None or live.empty:
        return None

    model = _load_model()
    X = live[FEATURE_COLS].tail(1).apply(pd.to_numeric, errors="coerce")
    pred = float(model.predict(X)[0])
    as_of = str(pd.Timestamp(live.iloc[-1]["date"]).date())

    track_record = _load_json("ticker_track_record.json")
    track = track_record.get(ticker)

    result = {"pred_return_5d": round(pred, 5), "as_of": as_of}

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
