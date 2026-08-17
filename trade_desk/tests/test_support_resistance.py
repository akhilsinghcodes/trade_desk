import pandas as pd
import numpy as np
import pytest

from modules.support_resistance import (
    suggest_trade, pivot_points, swing_levels,
    ATR_STOP_MULT, ATR_TARGET_MULT,
)


def _synthetic_df(n=60, base=100.0, seed=0):
    """Deterministic OHLCV with real intrabar range, so ATR is nonzero."""
    rng = np.random.RandomState(seed)
    close = base + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 1.5, n)
    low = close - rng.uniform(0.5, 1.5, n)
    return pd.DataFrame({"high": high, "low": low, "close": close})


def test_buy_uses_validated_multipliers():
    df = _synthetic_df()
    result = suggest_trade(df, "BUY")
    close = df["close"].iloc[-1]
    risk = close - result["stop_loss"]
    reward = result["take_profit"] - close
    # target distance should be ATR_TARGET_MULT/ATR_STOP_MULT x the stop distance
    # (2dp rounding on each leg means a few cents of slack, not exact equality)
    assert reward == pytest.approx(risk * (ATR_TARGET_MULT / ATR_STOP_MULT), abs=0.05)
    assert result["risk_reward_ratio"] == 3.0


def test_sell_mirrors_buy():
    df = _synthetic_df()
    result = suggest_trade(df, "SELL")
    close = df["close"].iloc[-1]
    assert result["stop_loss"] > close
    assert result["take_profit"] < close
    assert result["risk_reward_ratio"] == 3.0


def test_sell_avoid_alias_behaves_like_sell():
    df = _synthetic_df()
    a = suggest_trade(df, "SELL")
    b = suggest_trade(df, "SELL / AVOID")
    assert a["stop_loss"] == b["stop_loss"]
    assert a["take_profit"] == b["take_profit"]


def test_hold_returns_no_trade_plan():
    df = _synthetic_df()
    result = suggest_trade(df, "HOLD")
    assert result["action"] == "HOLD / WATCH"
    assert "stop_loss" not in result
    assert "take_profit" not in result


def test_backtest_stats_attached_to_buy_sell_only():
    df = _synthetic_df()
    buy = suggest_trade(df, "BUY")
    assert buy["backtest_win_rate"] > 0
    assert buy["backtest_ev_r"] > 0
    assert "R:R" not in buy.get("backtest_note", "") or "3.0" in buy["backtest_note"]


def test_stop_never_negative_on_cheap_stock():
    # A stock near $1 with high volatility shouldn't produce a negative stop.
    df = _synthetic_df(base=1.5, seed=1)
    df["low"] = df["low"].clip(lower=0.01)
    result = suggest_trade(df, "BUY")
    assert result["stop_loss"] >= 0


def test_pivot_points_uses_second_to_last_row():
    df = pd.DataFrame({
        "high": [10, 20, 999],
        "low": [5, 15, 1],
        "close": [8, 18, 500],
    })
    p = pivot_points(df)
    # Should use row index -2 (20, 15, 18), not the last row's extreme values
    expected_pp = round((20 + 15 + 18) / 3, 2)
    assert p["PP"] == expected_pp


def test_swing_levels_returns_at_most_three_each():
    df = _synthetic_df(n=100)
    levels = swing_levels(df)
    assert len(levels["support"]) <= 3
    assert len(levels["resistance"]) <= 3
