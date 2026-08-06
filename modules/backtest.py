"""Expanded backtesting module with multiple strategies and validation methods."""
import warnings
import pandas as pd
import numpy as np
import vectorbt as vbt
import yfinance as yf
from typing import Callable, List, Dict, Any

warnings.filterwarnings("ignore", category=UserWarning, module="vectorbt")


def sma_crossover_backtest(
    close: pd.Series,
    fast: int = 20,
    slow: int = 50,
    init_cash: float = 10_000,
    commission: float = 0.001,
) -> vbt.Portfolio:
    """SMA crossover strategy with optional transaction costs.

    Args:
        close: Price series
        fast: Fast MA period (default 20)
        slow: Slow MA period (default 50)
        init_cash: Initial capital (default 10000)
        commission: Transaction cost as decimal (default 0.001 = 0.1%)

    Returns:
        vectorbt Portfolio object
    """
    fast_ma = vbt.MA.run(close, fast)
    slow_ma = vbt.MA.run(close, slow)

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(
        close, entries, exits, init_cash=init_cash, freq="D",
        fees=commission, fixed_fees=0
    )
    return pf


def rsi_strategy_backtest(
    df: pd.DataFrame,
    rsi_buy: int = 30,
    rsi_sell: int = 70,
    rsi_period: int = 14,
    init_cash: float = 10_000,
    commission: float = 0.001,
) -> vbt.Portfolio:
    """RSI mean-reversion strategy.

    Buy when RSI crosses above rsi_buy from below.
    Sell when RSI crosses below rsi_sell from above.

    Args:
        df: DataFrame with 'close' column
        rsi_buy: RSI buy threshold (default 30 = oversold)
        rsi_sell: RSI sell threshold (default 70 = overbought)
        rsi_period: RSI period (default 14)
        init_cash: Initial capital (default 10000)
        commission: Transaction cost as decimal (default 0.001 = 0.1%)

    Returns:
        vectorbt Portfolio object
    """
    close = df["close"] if isinstance(df, pd.DataFrame) else df

    # Calculate RSI
    rsi = vbt.RSI.run(close, rsi_period).rsi

    # Detect crossovers manually
    # Buy when RSI crosses above rsi_buy from below
    rsi_above_buy = rsi > rsi_buy
    rsi_below_buy = rsi <= rsi_buy
    entries = rsi_above_buy & rsi_below_buy.shift(1)

    # Sell when RSI crosses below rsi_sell from above
    rsi_below_sell = rsi < rsi_sell
    rsi_above_sell = rsi >= rsi_sell
    exits = rsi_below_sell & rsi_above_sell.shift(1)

    pf = vbt.Portfolio.from_signals(
        close, entries, exits, init_cash=init_cash, freq="D",
        fees=commission, fixed_fees=0
    )
    return pf


def walk_forward(
    df: pd.DataFrame,
    strategy_fn: Callable,
    n_splits: int = 5,
    train_pct: float = 0.8,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Walk-forward validation of a strategy.

    Splits data into n_splits windows. For each window, trains on first 80%,
    tests on last 20%, and returns per-split + aggregate statistics.

    Args:
        df: DataFrame with 'close' column and datetime index
        strategy_fn: Strategy function(close_series, **kwargs) -> vbt.Portfolio
        n_splits: Number of windows (default 5)
        train_pct: Fraction of each window for training (default 0.8)
        **kwargs: Strategy parameters (rsi_buy, fast, slow, commission, etc.)

    Returns:
        List of dicts, each with:
        - split: split number (1, 2, ...)
        - train_dates: (start, end) of training period
        - test_dates: (start, end) of testing period
        - stats: dict of backtest stats for test period
        - plus aggregate dict with mean/median/std of all splits
    """
    close = df["close"] if isinstance(df, pd.DataFrame) else df
    n_total = len(close)
    window_size = n_total // n_splits

    results = []
    all_returns = []
    all_sharpe = []
    all_drawdown = []

    for i in range(n_splits):
        split_start = i * window_size
        split_end = (i + 1) * window_size if i < n_splits - 1 else n_total

        train_size = int((split_end - split_start) * train_pct)
        train_end = split_start + train_size
        test_end = split_end

        # Extract test data
        test_close = close.iloc[train_end:test_end]

        if len(test_close) < 2:
            continue

        # Run strategy on test period
        try:
            pf = strategy_fn(test_close, **kwargs)
            stats = get_backtest_stats(pf)

            train_date = close.index[split_start]
            train_end_date = close.index[train_end - 1] if train_end > 0 else train_date
            test_start_date = close.index[train_end]
            test_end_date = close.index[test_end - 1]

            total_ret = float(stats.get("Total Return [%]", 0) or 0) / 100
            sharpe = float(stats.get("Sharpe Ratio", 0) or 0)
            max_dd = float(stats.get("Max. Drawdown [%]", 0) or 0) / 100

            all_returns.append(total_ret)
            all_sharpe.append(sharpe)
            all_drawdown.append(max_dd)

            results.append({
                "split": i + 1,
                "train_dates": (str(train_date.date()), str(train_end_date.date())),
                "test_dates": (str(test_start_date.date()), str(test_end_date.date())),
                "test_return": total_ret,
                "test_sharpe": sharpe,
                "test_max_drawdown": max_dd,
                "stats": stats.to_dict() if hasattr(stats, 'to_dict') else dict(stats),
            })
        except Exception as e:
            print(f"Split {i+1} failed: {e}")
            continue

    # Add aggregate statistics
    if results:
        agg = {
            "aggregate": {
                "n_splits": len(results),
                "avg_return": float(np.mean(all_returns)) if all_returns else 0,
                "median_return": float(np.median(all_returns)) if all_returns else 0,
                "std_return": float(np.std(all_returns)) if all_returns else 0,
                "avg_sharpe": float(np.mean(all_sharpe)) if all_sharpe else 0,
                "median_sharpe": float(np.median(all_sharpe)) if all_sharpe else 0,
                "avg_max_drawdown": float(np.mean(all_drawdown)) if all_drawdown else 0,
            }
        }
        results.append(agg)

    return results


def compare_to_benchmark(
    df: pd.DataFrame,
    strategy_stats: pd.Series,
    benchmark: str = "SPY",
) -> Dict[str, Any]:
    """Compare strategy performance to a benchmark (e.g., SPY).

    Fetches benchmark data for the same date range, computes buy-and-hold
    return, and returns comparison metrics.

    Args:
        df: DataFrame with datetime index and 'close' column
        strategy_stats: Output from get_backtest_stats()
        benchmark: Ticker to compare against (default "SPY")

    Returns:
        Dict with:
        - strategy_cagr: Compound annual growth rate (%)
        - benchmark_cagr: Benchmark CAGR (%)
        - strategy_sharpe: Sharpe ratio
        - benchmark_sharpe: Benchmark Sharpe ratio
        - strategy_max_drawdown: Max drawdown (%)
        - benchmark_max_drawdown: Max drawdown (%)
        - outperformance: Strategy CAGR - Benchmark CAGR (%)
    """
    try:
        # Get benchmark data
        start_date = df.index[0]
        end_date = df.index[-1]

        bench_data = yf.download(benchmark, start=start_date, end=end_date, progress=False)
        if bench_data is None or len(bench_data) == 0:
            return {"error": f"Could not fetch benchmark data for {benchmark}"}

        # Handle both DataFrame and Series returns from yfinance
        if isinstance(bench_data["Close"], pd.DataFrame):
            bench_close = bench_data["Close"][benchmark]
        else:
            bench_close = bench_data["Close"]

        # Align dates
        common_dates = df.index.intersection(bench_close.index)
        if len(common_dates) < 2:
            return {"error": "Insufficient overlapping dates"}

        # Get values safely
        total_ret_pct = strategy_stats.get("Total Return [%]", 0)
        if total_ret_pct is None:
            total_ret_pct = 0
        strategy_ret = float(total_ret_pct) / 100.0

        bench_start = float(bench_close.loc[common_dates[0]])
        bench_end = float(bench_close.loc[common_dates[-1]])
        bench_ret = (bench_end / bench_start - 1) * 100

        strategy_sharpe = float(strategy_stats.get("Sharpe Ratio", 0) or 0)
        strategy_dd = float(strategy_stats.get("Max. Drawdown [%]", 0) or 0)

        # Compute benchmark Sharpe and drawdown
        bench_returns = bench_close.pct_change().dropna()
        if len(bench_returns) > 0 and bench_returns.std() > 0:
            bench_sharpe = float(bench_returns.mean() / bench_returns.std() * np.sqrt(252))
        else:
            bench_sharpe = 0.0

        # Benchmark max drawdown
        cummax = bench_close.expanding().max()
        drawdown = (bench_close - cummax) / cummax
        bench_dd = float(drawdown.min() * 100)

        # CAGR calculation
        years = (common_dates[-1] - common_dates[0]).days / 365.25
        if years > 0:
            strategy_cagr = ((1 + strategy_ret) ** (1 / years) - 1) * 100
            bench_cagr = ((1 + bench_ret / 100) ** (1 / years) - 1) * 100
        else:
            strategy_cagr = 0.0
            bench_cagr = 0.0

        return {
            "strategy_total_return": strategy_ret * 100,
            "benchmark_total_return": bench_ret,
            "strategy_cagr": strategy_cagr,
            "benchmark_cagr": bench_cagr,
            "strategy_sharpe": float(strategy_sharpe),
            "benchmark_sharpe": float(bench_sharpe),
            "strategy_max_drawdown": float(strategy_dd),
            "benchmark_max_drawdown": bench_dd,
            "outperformance_cagr": strategy_cagr - bench_cagr,
            "benchmark_ticker": benchmark,
        }
    except Exception as e:
        return {"error": f"Benchmark comparison failed: {str(e)}"}


def get_backtest_stats(pf: vbt.Portfolio) -> pd.Series:
    """Get extended backtest statistics.

    Returns original vectorbt stats plus:
    - cagr: Compound Annual Growth Rate (%)
    - max_drawdown: Maximum drawdown (%)
    - sharpe_ratio: Sharpe ratio
    - win_rate: Percentage of winning trades
    - avg_win: Average win size ($)
    - avg_loss: Average loss size ($)

    Args:
        pf: vectorbt Portfolio object

    Returns:
        pd.Series with all statistics
    """
    stats = pf.stats()

    try:
        # Calculate CAGR
        total_ret = stats.get("Total Return [%]", 0)
        start_val = pf.init_cash
        end_val = start_val * (1 + total_ret / 100)
        years = (pf.close.index[-1] - pf.close.index[0]).days / 365.25
        if years > 0:
            cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
        else:
            cagr = 0
        stats["CAGR [%]"] = cagr

        # Max drawdown (already in stats but ensure it's there)
        if "Max. Drawdown [%]" not in stats or pd.isna(stats["Max. Drawdown [%]"]):
            stats["Max. Drawdown [%]"] = 0

        # Sharpe ratio (already in stats)
        if "Sharpe Ratio" not in stats or pd.isna(stats["Sharpe Ratio"]):
            stats["Sharpe Ratio"] = 0

        # Calculate win rate and avg win/loss
        trades = pf.trades.records
        if len(trades) > 0:
            pnl = trades["pnl"]
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]

            win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0

            stats["Win Rate [%]"] = win_rate
            stats["Avg. Win [$]"] = avg_win
            stats["Avg. Loss [$]"] = avg_loss
        else:
            stats["Win Rate [%]"] = 0
            stats["Avg. Win [$]"] = 0
            stats["Avg. Loss [$]"] = 0

    except Exception:
        # If any calculation fails, set defaults
        if "CAGR [%]" not in stats:
            stats["CAGR [%]"] = 0
        if "Win Rate [%]" not in stats:
            stats["Win Rate [%]"] = 0
        if "Avg. Win [$]" not in stats:
            stats["Avg. Win [$]"] = 0
        if "Avg. Loss [$]" not in stats:
            stats["Avg. Loss [$]"] = 0

    return stats
