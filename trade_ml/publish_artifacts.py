"""Publish trained artifacts + per-ticker track record into trade_desk/models/.

No live cross-repo imports — trade_desk only ever reads these finished files.
Rerun this after retraining return_model.pkl or refreshing single_stock_check.csv
to push updated predictions/track-record into the app.
"""
import json
import shutil
from pathlib import Path

import pandas as pd

TRADE_ML = Path(__file__).parent
TRADE_DESK_MODELS = TRADE_ML.parent / "trade_desk" / "models"

FEATURE_COLS = json.loads((TRADE_ML / "models" / "return_features.json").read_text())


def publish_model():
    TRADE_DESK_MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy(TRADE_ML / "models" / "return_model.pkl", TRADE_DESK_MODELS / "return_model.pkl")
    shutil.copy(TRADE_ML / "models" / "return_features.json", TRADE_DESK_MODELS / "return_features.json")
    print(f"Copied return_model.pkl + return_features.json -> {TRADE_DESK_MODELS}")


def publish_track_record():
    track = pd.read_csv(TRADE_ML / "single_stock_check.csv")
    # Per-ticker time-series Spearman rho from the walk-forward check (per-ticker
    # evaluation, not cross-sectional — see /Users/akhil/.claude single_stock_check2).
    track["significant"] = track["pval"] < 0.05
    track["direction"] = track["rho"].apply(lambda r: "positive" if r > 0 else "negative")
    out = {
        row.ticker: {
            "n_folds": int(row.n),
            "rho": round(float(row.rho), 4),
            "pval": round(float(row.pval), 4),
            "significant": bool(row.significant),
            "direction": row.direction,
        }
        for row in track.itertuples()
    }
    (TRADE_DESK_MODELS / "ticker_track_record.json").write_text(json.dumps(out, indent=2))
    print(f"Published track record for {len(out)} tickers -> {TRADE_DESK_MODELS / 'ticker_track_record.json'}")


def publish_aggregate_stats():
    """Model-wide stats (across the tested 149-ticker universe) — shown as a
    fallback for tickers with no per-ticker track record of their own."""
    track = pd.read_csv(TRADE_ML / "single_stock_check.csv")
    n = len(track)
    n_positive = int((track["rho"] > 0).sum())
    n_significant = int((track["pval"] < 0.05).sum())
    median_rho = float(track["rho"].median())
    out = {
        "n_tickers_tested": n,
        "n_positive": n_positive,
        "n_significant": n_significant,
        "median_rho": round(median_rho, 4),
    }
    (TRADE_DESK_MODELS / "aggregate_stats.json").write_text(json.dumps(out, indent=2))
    print(f"Published aggregate stats -> {TRADE_DESK_MODELS / 'aggregate_stats.json'}")


if __name__ == "__main__":
    publish_model()
    publish_track_record()
    publish_aggregate_stats()
    print(
        "\nSkipped predictions — run refresh_live_predictions.py separately for "
        "current-day predictions (this script would otherwise overwrite them with "
        "the stale training-cache snapshot)."
    )
