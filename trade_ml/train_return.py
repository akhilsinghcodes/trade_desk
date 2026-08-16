"""
Forward return regression pipeline.

Input : 25 signals (technical + fundamental + sentiment) per stock per day
Output: predicted 5-day forward return
Metric: Spearman rank correlation (does model correctly rank stocks?)

Usage:
  .venv/bin/python train_return.py
  .venv/bin/python train_return.py --tickers AAPL MSFT NVDA
  .venv/bin/python train_return.py --days 10
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from modules.ml_dataset import build_training_dataset, DEFAULT_TICKERS
from modules.return_model import add_return_target, walk_forward_return, backtest_long_short, analyze_turnover_and_costs, train_final, LOOKAHEAD_DAYS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--period", default="15y")
    parser.add_argument("--days", type=int, default=LOOKAHEAD_DAYS)
    parser.add_argument("--output", default="models/")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS

    print(f"\n{'='*60}")
    print(f"  Forward Return Regression Pipeline")
    print(f"{'='*60}")
    print(f"  Tickers : {len(tickers)}")
    print(f"  Period  : {args.period}")
    print(f"  Target  : {args.days}-day forward return (regression)")
    print(f"  Metric  : Spearman rank correlation OOS")
    print(f"  CV      : Walk-forward quarterly expanding window\n")

    print("Step 1/3: Building dataset...")
    t0 = time.time()
    raw_df = build_training_dataset(tickers=tickers, period=args.period)
    df = add_return_target(raw_df, days=args.days)
    print(f"\n  {len(df):,} rows × {df.shape[1]} cols ({time.time()-t0:.0f}s)")
    print(f"  Forward return range: {df['fwd_return'].min():.2%} to {df['fwd_return'].max():.2%}")
    print(f"  Mean: {df['fwd_return'].mean():.3%}  Std: {df['fwd_return'].std():.3%}\n")

    print("Step 2/3: Walk-forward validation...")
    wf = walk_forward_return(df, embargo_days=args.days)

    print("\nStep 2b: Long-short backtest...")
    bt = backtest_long_short(wf["oos_df"], horizon_days=args.days)

    print("\nStep 2c: Turnover and cost sweep (decides if this is worth running live)...")
    tc = analyze_turnover_and_costs(wf["oos_df"], horizon_days=args.days)

    print("\nStep 3/3: Training final model...")
    summary = train_final(df, model_dir=args.output)

    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    if tc:
        print(f"  [PRIMARY]  Top-decile spread (annualized, gross): {tc['gross_annual_spread']:+.1%}")
        print(f"  [PRIMARY]  Hit rate                             : {tc['hit_rate']:.1%}")
        print(f"  [PRIMARY]  Turnover (rebalance freq. annualized) : {tc['annual_turnover']:.0%}")
    print(f"  [secondary] Avg Spearman rho     : {wf['avg_rho']:+.4f}")
    print(f"  [secondary] Significant folds    : {wf['n_sig']}/{wf['n_folds']} (p<0.05)")
    print(f"  Top features      : {summary['top_shap_features'][:5]}")
    print(f"{'='*60}")

    gate_passed = wf["validation_gate"]["passed"]
    spread_ok = bool(bt) and bt["avg_spread"] > 0
    overall = gate_passed and spread_ok

    print(f"\n  Validation gate  : {'PASS' if gate_passed else 'FAIL'} ({wf['validation_gate']['reason'].strip()})")
    print(f"  Long-short spread: {'PASS' if spread_ok else 'FAIL'} ({bt.get('avg_spread', 0):+.3%})")
    print(f"  Overall          : {'PASS — model cleared for use' if overall else 'FAIL — do not deploy'}")

    if tc:
        print(f"\n  {'='*60}")
        print(f"  THE REAL QUESTION: does the edge survive real trading costs?")
        print(f"  {'='*60}")
        for bps, r in tc["cost_sweep"].items():
            verdict = "still worth running" if r["net_annual_spread"] > 0.06 else \
                      "marginal" if r["net_annual_spread"] > 0 else "dead — costs eat the whole edge"
            print(f"  At {bps}bps round-trip: net annualized spread {r['net_annual_spread']:+.1%} — {verdict}")

    if wf["avg_rho"] > 0.05 and wf["n_sig"] >= wf["n_folds"] // 3:
        print("\n  Signal detected. Worth investigating further.")
    elif wf["avg_rho"] > 0.02:
        print("\n  Weak signal. Marginal.")
    else:
        print("\n  No signal. Features don't predict return ranking.")


if __name__ == "__main__":
    main()
