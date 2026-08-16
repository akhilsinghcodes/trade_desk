"""
Trade ML — 21-day direction model with walk-forward validation.

Usage:
  python train.py                          # 50+ diverse tickers, 7y history
  python train.py --tickers AAPL MSFT      # specific tickers
  python train.py --tickers-file tickers.txt
  python train.py --period 5y              # shorter history
  python train.py --output /path/to/dir
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from modules.ml_dataset import build_training_dataset, DEFAULT_TICKERS
from modules.ml_models import train_models


def main():
    parser = argparse.ArgumentParser(description="Train Trade ML walk-forward XGBoost model")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--tickers-file", default=None,
                        help="Text file with one ticker per line")
    parser.add_argument("--period", default="7y",
                        help="History period per ticker (default: 7y)")
    parser.add_argument("--output", default="models/")
    args = parser.parse_args()

    if args.tickers_file:
        with open(args.tickers_file) as fh:
            tickers = [line.strip().upper() for line in fh if line.strip()]
    else:
        tickers = args.tickers or DEFAULT_TICKERS

    print(f"\n{'='*60}")
    print("  Trade ML Walk-Forward Training Pipeline")
    print(f"{'='*60}")
    print(f"  Tickers : {len(tickers)} ({', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''})")
    print(f"  Period  : {args.period}")
    print(f"  Output  : {args.output}")
    print("  Target  : trade outcome — WIN if TP hit before stop in 21d, LOSS otherwise")
    print("  CV      : Walk-forward expanding quarterly windows 2022+")
    print(f"{'='*60}\n")

    print("Step 1/2: Building dataset...")
    t0 = time.time()
    df = build_training_dataset(tickers=tickers, period=args.period)
    print(f"\n  Dataset: {df.shape[0]:,} samples × {df.shape[1]} cols ({time.time()-t0:.0f}s)\n")

    print("Step 2/2: Training model with walk-forward CV...")
    t0 = time.time()
    summary = train_models(df, model_dir=args.output)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Training Complete ({elapsed:.0f}s)")
    print(f"{'='*60}")
    if summary.get("walk_forward_avg_accuracy"):
        print(f"  Walk-forward avg accuracy: {summary['walk_forward_avg_accuracy']:.1%}")
    print(f"  Top features: {', '.join(summary.get('top_shap_features', [])[:5])}")
    print(f"  Models saved to: {os.path.abspath(args.output)}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
