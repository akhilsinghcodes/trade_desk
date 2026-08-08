"""Backtest setup-type strategies (BREAKOUT, PULLBACK, MEAN_REVERSION) vs BUY_AND_HOLD."""
import warnings

import numpy as np
import pandas as pd
import vectorbt as vbt

warnings.filterwarnings("ignore", category=UserWarning, module="vectorbt")
warnings.filterwarnings("ignore", category=FutureWarning)

_STRATEGIES = ("BREAKOUT", "PULLBACK", "MEAN_REVERSION", "RANGE", "BREAKDOWN", "VALIDATED", "BUY_AND_HOLD")
_MIN_BARS = 100
# Strategies that trade the short side — direction is set accordingly in backtest_strategy.
_SHORT_STRATEGIES = ("BREAKDOWN",)


def _build_signals(df: pd.DataFrame, strategy: str) -> tuple[pd.Series, pd.Series]:
    close = df["close"]
    rsi = df["rsi"]
    volume = df["volume"]
    sma20 = df["sma20"]
    sma50 = df["sma50"]
    bb_lower = df["bb_lower"]
    bb_mid = df["bb_mid"]

    if strategy == "BREAKOUT":
        prev_20_high = close.shift(1).rolling(20).max()
        prev_10_low = close.shift(1).rolling(10).min()
        vol_surge = volume > volume.rolling(20).mean() * 1.3

        entries = (close > prev_20_high) & vol_surge & (rsi < 75)
        exits = (close < prev_10_low) | (rsi > 75)

    elif strategy == "PULLBACK":
        near_sma20 = (close > sma20 * 0.98) & (close < sma20 * 1.02)
        entries = (
            (sma20 > sma50)
            & (close > close.shift(1))
            & (rsi > rsi.shift(1))
            & rsi.between(35, 60)
            & near_sma20
        )
        exits = (rsi > 70) | (close < sma20 * 0.96)

    elif strategy == "MEAN_REVERSION":
        entries = (rsi < 32) & (close < bb_lower)
        exits = (rsi > 55) | (close > bb_mid)

    elif strategy == "RANGE":
        # Mirrors detect_setup's RANGE fallback: no strong trend (SMA20/50 close
        # together), not RSI-extreme. Trades the range itself — buy near the
        # lower band, sell near the upper band, stop out if it actually breaks.
        flat_trend = ((sma20 - sma50).abs() / close) < 0.03
        entries = (close <= bb_lower * 1.01) & flat_trend & rsi.between(30, 55)
        exits = (close >= bb_mid) | (close < bb_lower * 0.97)

    elif strategy == "BREAKDOWN":
        # Mirrors BREAKOUT but short: price breaks below a recent swing low
        # with bearish structure. Short-only — covers on an oversold bounce
        # rather than riding it to zero.
        prev_10_low = close.shift(1).rolling(10).min()
        entries = (close < prev_10_low) & (sma20 < sma50) & (rsi < 45)
        exits = (rsi < 25) | (close > sma20)

    elif strategy == "VALIDATED":
        # The composite of ONLY the ~10 price signals that show real
        # predictive power, weighted by their own measured IC — not a
        # hand-picked verdict. See modules/factor_analysis.compute_composite_signal.
        from modules.factor_analysis import compute_composite_signal
        composite, _, _ = compute_composite_signal(df)
        entries = composite > 0.15
        exits = composite < -0.05

    elif strategy == "BUY_AND_HOLD":
        entries = pd.Series([True] + [False] * (len(close) - 1), index=close.index)
        exits = pd.Series(False, index=close.index)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return entries.fillna(False), exits.fillna(False)


def _extract_trades(pf: vbt.Portfolio) -> list[dict]:
    trades = []
    try:
        records = pf.trades.records_readable
        for _, row in records.iterrows():
            trades.append({
                "entry_date": row.get("Entry Timestamp", row.get("entry_idx")),
                "exit_date": row.get("Exit Timestamp", row.get("exit_idx")),
                "return_pct": float(row.get("Return", row.get("return", 0))) * 100,
                "pnl": float(row.get("PnL", row.get("pnl", 0))),
            })
    except Exception:
        # Fall back to raw records array when readable format is unavailable.
        try:
            for rec in pf.trades.records:
                trades.append({
                    "entry_date": rec["entry_idx"],
                    "exit_date": rec["exit_idx"],
                    "return_pct": float(rec.get("return", 0)) * 100,
                    "pnl": float(rec.get("pnl", 0)),
                })
        except Exception:
            pass
    return trades


def _cagr(start_val: float, end_val: float, n_days: int) -> float:
    if start_val <= 0 or n_days <= 0:
        return 0.0
    years = n_days / 365.25
    return ((end_val / start_val) ** (1.0 / years) - 1) * 100


def backtest_strategy(
    df: pd.DataFrame,
    strategy: str,
    init_cash: float = 10_000,
    commission: float = 0.001,
) -> dict:
    """Run a single strategy backtest on OHLCV+indicator dataframe.

    Returns a dict with performance metrics, equity curve, and trade list.
    On failure the dict has a non-None 'error' field and zeroed metrics.
    """
    base: dict = {
        "strategy": strategy,
        "total_return_pct": 0.0,
        "cagr_pct": 0.0,
        "sharpe": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate_pct": 0.0,
        "n_trades": 0,
        "avg_trade_days": 0.0,
        "equity_curve": pd.Series(dtype=float),
        "trades": [],
        "error": None,
    }

    try:
        if len(df) < _MIN_BARS:
            raise ValueError(f"Need at least {_MIN_BARS} bars, got {len(df)}")

        close = df["close"]
        entries, exits = _build_signals(df, strategy)
        direction = "shortonly" if strategy in _SHORT_STRATEGIES else "longonly"

        pf = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            init_cash=init_cash,
            freq="D",
            fees=commission,
            direction=direction,
        )

        equity = pf.value()
        start_val = float(equity.iloc[0])
        end_val = float(equity.iloc[-1])
        n_days = (equity.index[-1] - equity.index[0]).days

        stats = pf.stats()

        total_return = (end_val / start_val - 1) * 100

        # Sharpe from vectorbt stats; fall back to 0 if missing.
        sharpe = float(stats.get("Sharpe Ratio", 0) or 0)

        # Max drawdown is returned as a negative fraction by vectorbt.
        mdd_raw = stats.get("Max Drawdown [%]", stats.get("Max Drawdown", 0)) or 0
        max_dd = abs(float(mdd_raw))

        n_trades = int(stats.get("Total Trades", 0) or 0)
        win_rate = float(stats.get("Win Rate [%]", 0) or 0)

        # Average trade duration in calendar days.
        avg_days = 0.0
        trades_list = _extract_trades(pf)
        if trades_list:
            durations = []
            for t in trades_list:
                try:
                    d = (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days
                    if d >= 0:
                        durations.append(d)
                except Exception:
                    pass
            avg_days = float(np.mean(durations)) if durations else 0.0

        base.update({
            "total_return_pct": round(total_return, 2),
            "cagr_pct": round(_cagr(start_val, end_val, n_days), 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate_pct": round(win_rate, 2),
            "n_trades": n_trades,
            "avg_trade_days": round(avg_days, 1),
            "equity_curve": equity,
            "trades": trades_list,
        })

    except Exception as exc:
        base["error"] = str(exc)

    return base


def backtest_stability(
    df: pd.DataFrame,
    strategy: str,
    init_cash: float = 10_000,
    commission: float = 0.001,
) -> dict:
    """Split the period in half and backtest each half independently.

    A single blended backtest over the whole period can look like an edge
    purely from one lucky stretch. Requiring the edge to show up in BOTH
    halves independently is a cheap, honest proxy for out-of-sample
    validation — it won't catch everything walk-forward testing would, but
    it catches the common case of a strategy that only worked in one regime.
    """
    mid = len(df) // 2
    first_half = backtest_strategy(df.iloc[:mid], strategy, init_cash, commission)
    second_half = backtest_strategy(df.iloc[mid:], strategy, init_cash, commission)

    both_have_trades = first_half["n_trades"] >= 3 and second_half["n_trades"] >= 3
    both_profitable = first_half["sharpe"] > 0 and second_half["sharpe"] > 0
    stable = both_have_trades and both_profitable

    return {
        "first_half": first_half,
        "second_half": second_half,
        "stable": stable,
        "reason": (
            "not enough trades in one half to check" if not both_have_trades
            else "edge did not hold in both halves" if not both_profitable
            else "edge held in both halves independently"
        ),
    }


def backtest_all(
    df: pd.DataFrame,
    init_cash: float = 10_000,
    commission: float = 0.001,
) -> dict[str, dict]:
    """Run all four strategies. Per-strategy errors are captured in result['error']."""
    return {
        s: backtest_strategy(df, s, init_cash=init_cash, commission=commission)
        for s in _STRATEGIES
    }


def compare_summary(results: dict) -> pd.DataFrame:
    """Build a comparison DataFrame sorted by Sharpe descending."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Strategy": name,
            "Total Return %": r["total_return_pct"],
            "CAGR %": r["cagr_pct"],
            "Sharpe": r["sharpe"],
            "Max Drawdown %": r["max_drawdown_pct"],
            "Win Rate %": r["win_rate_pct"],
            "# Trades": r["n_trades"],
        })
    return (
        pd.DataFrame(rows)
        .set_index("Strategy")
        .sort_values("Sharpe", ascending=False)
    )
