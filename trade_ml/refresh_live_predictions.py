"""Refresh today's live feature snapshot + predictions (does NOT touch
atr_dataset_cache.parquet or retrain — just today's inference input).
Run this before publish_artifacts.py to get current-day predictions instead
of the training cache's last labeled date (which lags ~3 weeks due to the
21-day label lookahead truncation)."""
import json
from pathlib import Path

import joblib
import pandas as pd

from modules.ml_dataset import build_live_features, get_feature_cols

TRADE_ML = Path(__file__).parent
TRADE_DESK_MODELS = TRADE_ML.parent / "trade_desk" / "models"

with open(TRADE_ML / "tickers.txt" if (TRADE_ML / "tickers.txt").exists() else "/tmp/ticker_universe.txt") as f:
    TICKERS = [t.strip() for t in f if t.strip()]

FEATURE_COLS = get_feature_cols()


def main():
    live = build_live_features(TICKERS, period="1y")
    live.to_parquet(TRADE_ML / "live_features_cache.parquet")

    model = joblib.load(TRADE_ML / "models" / "return_model.pkl")
    X = live[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    live["pred_return"] = model.predict(X)

    out = {
        row.ticker: {
            "pred_return_5d": round(float(row.pred_return), 5),
            "as_of": str(pd.Timestamp(row.date).date()),
        }
        for row in live.itertuples()
    }
    TRADE_DESK_MODELS.mkdir(parents=True, exist_ok=True)
    (TRADE_DESK_MODELS / "latest_predictions.json").write_text(json.dumps(out, indent=2))
    print(f"\nPublished LIVE predictions for {len(out)} tickers -> {TRADE_DESK_MODELS / 'latest_predictions.json'}")


if __name__ == "__main__":
    main()
