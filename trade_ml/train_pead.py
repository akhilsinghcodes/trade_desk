"""
PEAD earnings model training pipeline.

Usage:
  python train_pead.py
  python train_pead.py --tickers AAPL MSFT NVDA ...
  python train_pead.py --output models/
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from modules.ml_dataset import DEFAULT_TICKERS
from modules.pead_dataset import build_pead_dataset
from modules.pead_model import walk_forward_pead, backtest_pead, train_final_pead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--output", default="models/")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS

    print(f"\n{'='*60}")
    print("  PEAD Earnings Model Pipeline")
    print(f"{'='*60}")
    print(f"  Tickers : {len(tickers)}")
    print("  Target  : stock outperforms SPY >3% in 30d post-earnings")
    print("  CV      : Walk-forward by year (expanding window)\n")

    print("Step 1/3: Building earnings dataset...")
    t0 = time.time()
    df = build_pead_dataset(tickers)
    print(f"\n  {len(df):,} events × {df.shape[1]} cols ({time.time()-t0:.0f}s)")

    print("\nStep 2/3: Walk-forward validation...")
    wf = walk_forward_pead(df)

    print("\nStep 2b: OOS backtest...")
    bt = backtest_pead(wf["oos_df"], threshold=0.55)
    backtest_pead(wf["oos_df"], threshold=0.60)

    print("\nStep 3/3: Training final model on all data...")
    summary = train_final_pead(df, model_dir=args.output)

    print(f"\n{'='*60}")
    print("  PEAD Training Complete")
    print(f"{'='*60}")
    print(f"  Walk-forward avg acc  : {wf['avg_acc']:.1%}")
    print(f"  Walk-forward avg AUC  : {wf['avg_auc']:.3f}")
    print(f"  Baseline win rate     : {bt.get('baseline_win_rate', 0):.1%}")
    print(f"  BUY signal win rate   : {bt.get('buy_win_rate', 0):.1%}")
    print(f"  Top features          : {summary.get('top_shap_features', [])}")
    print(f"{'='*60}\n")

    if wf["avg_auc"] > 0.55:
        print("  ✓ AUC > 0.55 — signal looks real. Validate before using.")
    elif wf["avg_auc"] > 0.52:
        print("  ~ AUC 0.52-0.55 — marginal signal, more data might help.")
    else:
        print("  ✗ AUC ≤ 0.52 — no real edge detected on this feature set.")


if __name__ == "__main__":
    main()
