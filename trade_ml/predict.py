"""
Live scoring: build current-day features across a universe and rank by math_score.

Usage:
  python predict.py                    # First 15 default tickers
  python predict.py [space-separated tickers]
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from modules.ml_dataset import build_live_features, DEFAULT_TICKERS
from modules.master_model import load_master_model, math_rank


def main():
    # Default to first 15 tickers for speed (can override via CLI)
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS[:15]

    print(f"\n{'='*60}")
    print("  Live Scoring Pipeline")
    print(f"{'='*60}")
    print(f"  Tickers : {len(tickers)} ({', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''})")
    print(f"{'='*60}\n")

    print("Step 1/2: Building live features...")
    t0 = time.time()
    try:
        live_df = build_live_features(tickers=tickers, period="1y")
    except ValueError as e:
        print(f"\nError: {e}")
        return
    print(f"  Features built: {len(live_df)} tickers ({time.time()-t0:.0f}s)\n")

    print("Step 2/2: Loading model weights...")
    weights = load_master_model("models/")
    if weights is None:
        print("\n" + "="*60)
        print("  No trained model found at models/master_model_weights.json")
        print("  Run 'python train.py' first to train the model.")
        print("="*60 + "\n")
        return

    print(f"  Model loaded: {len(weights['betas'])} pillar betas\n")

    # Rank by math_score
    ranked = math_rank(live_df, weights)
    ranked_df = ranked.reset_index()
    ranked_df["ticker"] = live_df["ticker"].values
    ranked_df = ranked_df.sort_values("math_score", ascending=False)

    # Print table
    print("="*60)
    print("  Ranked by Math Score")
    print("="*60)
    print(f"{'Ticker':<8} {'Momentum':<10} {'Squeeze':<10} {'Catalyst':<10} {'Safety':<10} {'Math Score':<12}")
    print("-"*60)
    for _, row in ranked_df.iterrows():
        print(
            f"{row['ticker']:<8} "
            f"{row['momentum']:>9.3f}  "
            f"{row['squeeze']:>9.3f}  "
            f"{row['catalyst']:>9.3f}  "
            f"{row['safety']:>9.3f}  "
            f"{row['math_score']:>11.4f}"
        )
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
