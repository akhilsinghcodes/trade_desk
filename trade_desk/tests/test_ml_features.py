import numpy as np

from modules.ml_features import (
    _rolling_ols_metrics, FEATURE_COLS,
    TECHNICAL_FEATURES, FUNDAMENTAL_FEATURES, SENTIMENT_FEATURES,
)


def test_feature_cols_is_union_of_groups():
    assert FEATURE_COLS == TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES + SENTIMENT_FEATURES
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))  # no duplicates


def test_feature_cols_matches_published_model_contract():
    # This list must stay byte-for-byte in sync with trade_ml's
    # return_features.json — a mismatch here means predictions silently
    # use the wrong feature at the wrong position.
    assert len(FEATURE_COLS) == 57


def test_rolling_ols_perfect_line_has_zero_residual():
    window = 10
    x = np.arange(window, dtype=float) * 2.0 + 5.0  # perfectly linear
    slope, r_squared, mean_abs_resid = _rolling_ols_metrics(x, window)
    assert slope == 2.0
    assert r_squared == 1.0
    assert mean_abs_resid < 1e-9


def test_rolling_ols_flat_line_has_zero_slope():
    window = 10
    x = np.full(window, 7.0)
    slope, r_squared, mean_abs_resid = _rolling_ols_metrics(x, window)
    assert slope == 0.0


def test_rolling_ols_short_window_returns_nan():
    slope, r_squared, resid = _rolling_ols_metrics(np.array([1.0, 2.0]), window=10)
    assert np.isnan(slope)
    assert np.isnan(r_squared)
    assert np.isnan(resid)
