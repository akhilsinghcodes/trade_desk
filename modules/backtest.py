"""Simple backtesting via vectorbt."""
import warnings
import pandas as pd
import vectorbt as vbt

warnings.filterwarnings("ignore", category=UserWarning, module="vectorbt")


def sma_crossover_backtest(
    close: pd.Series,
    fast: int = 20,
    slow: int = 50,
    init_cash: float = 10_000,
) -> vbt.Portfolio:
    """SMA crossover strategy. Returns vectorbt Portfolio for stats/plots."""
    fast_ma = vbt.MA.run(close, fast)
    slow_ma = vbt.MA.run(close, slow)

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=init_cash, freq="D")
    return pf


def get_backtest_stats(pf: vbt.Portfolio) -> pd.Series:
    return pf.stats()
