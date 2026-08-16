"""
4-pillar Master Model training pipeline.

Input : 25 signals (technical + fundamental + sentiment) per stock per day,
        rolled into 4 sub-models (momentum, squeeze, catalyst, safety)
Output: pillar betas fit via closed-form ridge regression against 5-day
        forward return, persisted to models/master_model_weights.json

Usage:
  .venv/bin/python train_master.py
  .venv/bin/python train_master.py --tickers AAPL MSFT NVDA --period 5y
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from modules.ml_dataset import build_training_dataset, DEFAULT_TICKERS
from modules.master_model import train_master_model, PILLARS


def main():
    parser = argparse.ArgumentParser(description="Train the 4-pillar Master Model")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--period", default="7y")
    parser.add_argument("--output", default="models/")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS

    print(f"\n{'='*60}")
    print("  Master Model Training (4 Sub-Models + Math Ranker)")
    print(f"{'='*60}")
    print(f"  Tickers : {len(tickers)} ({', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''})")
    print(f"  Period  : {args.period}")
    print(f"  Pillars : {', '.join(PILLARS)}")
    print("  Target  : 5-day forward return (ridge regression)")
    print(f"{'='*60}\n")

    print("Step 1/2: Building dataset...")
    t0 = time.time()
    df = build_training_dataset(tickers=tickers, period=args.period)
    print(f"\n  Dataset: {df.shape[0]:,} samples × {df.shape[1]} cols ({time.time()-t0:.0f}s)\n")

    print("Step 2/2: Fitting pillar betas (ridge)...")
    t0 = time.time()
    payload = train_master_model(df, model_dir=args.output, alpha=args.alpha)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Training Complete ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Samples : {payload['n_samples']:,}")
    print(f"  Betas   : {dict(zip(payload['pillars'], [round(b, 4) for b in payload['betas']]))}")
    print(f"  Intercept: {payload['intercept']:.4f}")
    print(f"  Weights saved to: {os.path.abspath(args.output)}/master_model_weights.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
