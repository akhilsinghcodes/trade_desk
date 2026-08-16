"""
Alphalens IC decay sweep — per-feature, per-horizon Spearman IC.

Answers: which of the 54 features carry real signal, at what holding
horizon (1/5/21/63/126 days), so we know what to keep, what to prune,
and what horizon this feature set is actually suited to (short vs long).

Usage:
  .venv/bin/python alphalens_sweep.py
  .venv/bin/python alphalens_sweep.py --tickers AAPL MSFT NVDA --period 5y
"""
import argparse
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import alphalens as al  # noqa: E402

from modules.ml_dataset import build_training_dataset, get_feature_cols, DEFAULT_TICKERS  # noqa: E402

PERIODS = (1, 5, 21, 63, 126)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--period", default="7y")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS

    print(f"Building dataset: {len(tickers)} tickers, {args.period}...")
    df = build_training_dataset(tickers=tickers, period=args.period)
    feature_cols = get_feature_cols()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Wide close-price panel: date x ticker (alphalens needs this to compute
    # forward returns itself, at whatever horizons we ask for)
    prices = df.pivot(index="date", columns="ticker", values="close").sort_index()

    results = []
    for feat in feature_cols:
        try:
            factor = df.set_index(["date", "ticker"])[feat]
            factor.index = factor.index.set_names(["date", "asset"])
            prices.columns.name = "asset"

            clean = al.utils.get_clean_factor_and_forward_returns(
                factor, prices, periods=PERIODS, quantiles=5, max_loss=0.5
            )
            ic = al.performance.factor_information_coefficient(clean)
            mean_ic = ic.mean()
            row = {"feature": feat}
            for p in PERIODS:
                col = f"{p}D"
                row[col] = mean_ic.get(col, np.nan)
            results.append(row)
            print(f"  {feat:<22} " + "  ".join(f"{row.get(f'{p}D', float('nan')):+.4f}" for p in PERIODS))
        except Exception as e:
            print(f"  {feat:<22} FAILED — {e}")
            results.append({"feature": feat, **{f"{p}D": np.nan for p in PERIODS}})

    out = pd.DataFrame(results).set_index("feature")
    out.to_csv("alphalens_ic_sweep.csv")

    print(f"\n{'='*70}")
    print("IC Decay Sweep — mean Spearman IC by holding horizon")
    print(f"{'='*70}")
    print(out.round(4).to_string())

    print("\nSaved: alphalens_ic_sweep.csv")

    print(f"\n{'='*70}")
    print("Best horizon per feature (max |IC|):")
    print(f"{'='*70}")
    best_horizon = out.abs().idxmax(axis=1)
    print(best_horizon.value_counts().to_string())

    print(f"\n{'='*70}")
    print("Near-zero features (|IC| < 0.01 at ALL horizons — pruning candidates):")
    print(f"{'='*70}")
    dead = out[(out.abs() < 0.01).all(axis=1)]
    print(dead.round(4).to_string() if not dead.empty else "  none")


if __name__ == "__main__":
    main()
