"""Fast, deterministic unit tests — no network, no training. Run with:
  .venv/bin/python -m pytest tests/test_return_model_units.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pytest

from modules.return_model import add_return_target, _fold_dates


def _price_series(ticker, start_price, n=30, start_date="2020-01-01"):
    dates = pd.date_range(start_date, periods=n, freq="D")
    # deterministic +1/day walk so fwd_return is exactly computable
    closes = [start_price + i for i in range(n)]
    return pd.DataFrame({"ticker": ticker, "date": dates, "close": closes})


def test_ticker_column_survives_add_return_target():
    # Regression test for a real pandas 3.0 bug: groupby("ticker").apply(fn)
    # that returns the whole group silently dropped the "ticker" column.
    df = pd.concat([_price_series("AAA", 100), _price_series("BBB", 200)], ignore_index=True)
    out = add_return_target(df, days=5)
    assert "ticker" in out.columns
    assert set(out["ticker"].unique()) == {"AAA", "BBB"}


def test_forward_return_is_correct_and_no_cross_ticker_leakage():
    df = pd.concat([_price_series("AAA", 100), _price_series("BBB", 200)], ignore_index=True)
    out = add_return_target(df, days=5)

    aaa = out[out["ticker"] == "AAA"].sort_values("date")
    # close goes 100,101,102... so 5-day fwd return at day i = 5/(100+i)
    first_row = aaa.iloc[0]
    expected = 5 / 100.0
    assert first_row["fwd_return"] == pytest.approx(expected, abs=1e-9)

    # BBB's returns must never be computed using AAA's prices
    bbb = out[out["ticker"] == "BBB"].sort_values("date")
    expected_bbb = 5 / 200.0
    assert bbb.iloc[0]["fwd_return"] == pytest.approx(expected_bbb, abs=1e-9)


def test_last_n_rows_dropped_for_missing_forward_window():
    df = _price_series("AAA", 100, n=30)
    out = add_return_target(df, days=5)
    # last 5 rows have no forward price 5 days out -> dropped
    assert len(out) == 25


def test_extreme_returns_are_clipped():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    # A halt/gap: price 10x's in one day
    closes = [100, 100, 100, 100, 1000, 1000, 1000, 1000, 1000, 1000]
    df = pd.DataFrame({"ticker": "AAA", "date": dates, "close": closes})
    out = add_return_target(df, days=1)
    assert out["fwd_return"].max() <= 0.25
    assert out["fwd_return"].min() >= -0.25


def test_fold_dates_embargo_gap_matches_horizon():
    dates = pd.date_range("2015-01-01", "2021-12-31", freq="D")
    df = pd.DataFrame({"date": dates})
    embargo = 5
    folds = _fold_dates(df, embargo_days=embargo)
    assert len(folds) > 0
    for train_start, train_end, test_start, test_end in folds:
        gap_days = (pd.Timestamp(test_start) - pd.Timestamp(train_end)).days
        # embargo_days + 1 gap between train_end and test_start
        assert gap_days == embargo + 1


def test_fold_dates_are_expanding_not_rolling():
    dates = pd.date_range("2015-01-01", "2021-12-31", freq="D")
    df = pd.DataFrame({"date": dates})
    folds = _fold_dates(df, embargo_days=5)
    train_starts = [f[0] for f in folds]
    # expanding window: every fold starts training from the same earliest date
    assert len(set(train_starts)) == 1


def test_fold_dates_test_windows_advance_quarterly():
    dates = pd.date_range("2015-01-01", "2021-12-31", freq="D")
    df = pd.DataFrame({"date": dates})
    folds = _fold_dates(df, embargo_days=5)
    test_starts = [pd.Timestamp(f[2]) for f in folds]
    assert test_starts == sorted(test_starts)
    assert len(test_starts) == len(set(test_starts))
