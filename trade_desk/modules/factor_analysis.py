import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_signals_df(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    signals = pd.DataFrame(index=df.index)

    signals["rsi_signal"] = (50 - df["rsi"]) / 50

    macd_diff = df["macd"] - df["macd_signal"]
    signals["macd_signal_val"] = np.tanh(macd_diff / close * 100)

    signals["momentum_5d"] = (close / close.shift(5) - 1).clip(-0.2, 0.2)
    signals["momentum_21d"] = (close / close.shift(21) - 1).clip(-0.3, 0.3)

    bb_range = df["bb_upper"] - df["bb_lower"]
    bb_pos_raw = (close - df["bb_lower"]) / bb_range.replace(0, np.nan)
    signals["bb_position"] = (bb_pos_raw * 2 - 1).clip(-1, 1)

    vol_mean = df["volume"].rolling(20).mean()
    signals["volume_surge"] = (df["volume"] / vol_mean.replace(0, np.nan) - 1).clip(-1, 2)

    signals["price_vs_sma20"] = ((close - df["sma20"]) / df["sma20"].replace(0, np.nan)).clip(-0.15, 0.15)
    signals["price_vs_sma50"] = ((close - df["sma50"]) / df["sma50"].replace(0, np.nan)).clip(-0.20, 0.20)

    def _linreg_slope(series: pd.Series) -> float:
        y = series.values
        if np.isnan(y).any():
            return np.nan
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return slope

    raw_slope = close.rolling(10).apply(_linreg_slope, raw=False)
    signals["trend_strength"] = (raw_slope / close.replace(0, np.nan)).clip(-0.05, 0.05)

    rets = close.pct_change()
    std_10 = rets.rolling(10).std()
    std_60 = rets.rolling(60).std()
    signals["volatility_regime"] = (std_10 / std_60.replace(0, np.nan) - 1).clip(-1, 1)

    return signals


def compute_ic(
    signals_df: pd.DataFrame,
    close: pd.Series,
    horizons: list = [1, 5, 10, 21],
) -> pd.DataFrame:
    cols = pd.MultiIndex.from_product([horizons, ["ic", "tstat"]], names=["horizon", "metric"])
    ic_df = pd.DataFrame(index=signals_df.columns, columns=cols, dtype=float)

    for horizon in horizons:
        fwd_ret = close.shift(-horizon) / close - 1
        combined = signals_df.copy()
        combined["__fwd__"] = fwd_ret

        for signal in signals_df.columns:
            valid = combined[[signal, "__fwd__"]].dropna()
            if len(valid) < 10:
                ic_df.loc[signal, (horizon, "ic")] = np.nan
                ic_df.loc[signal, (horizon, "tstat")] = np.nan
                continue

            ic_val, _ = spearmanr(valid[signal], valid["__fwd__"])
            n = len(valid)
            # t-stat derivation from Fisher's z is unstable near |IC|=1; clamp denominator
            denom = np.sqrt(max(1 - ic_val**2, 1e-10))
            tstat = ic_val * np.sqrt(n) / denom

            ic_df.loc[signal, (horizon, "ic")] = ic_val
            ic_df.loc[signal, (horizon, "tstat")] = tstat

    return ic_df.astype(float)


def compute_rolling_ic(
    signal: pd.Series,
    close: pd.Series,
    horizon: int = 5,
    window: int = 63,
) -> pd.Series:
    fwd_ret = close.shift(-horizon) / close - 1
    combined = pd.concat([signal.rename("sig"), fwd_ret.rename("fwd")], axis=1)

    rolling_ic = pd.Series(index=combined.index, dtype=float)

    for i in range(window, len(combined) + 1):
        window_slice = combined.iloc[i - window : i].dropna()
        if len(window_slice) < 30:
            continue
        ic_val, _ = spearmanr(window_slice["sig"], window_slice["fwd"])
        rolling_ic.iloc[i - 1] = ic_val

    return rolling_ic


def compute_composite_signal(
    df: pd.DataFrame,
    horizons: list = [1, 5, 10, 21],
    min_abs_ic: float = 0.05,
) -> tuple[pd.Series, list, pd.Series]:
    """Build a composite signal from ONLY the price-based signals that show
    real predictive power (|IC| >= min_abs_ic), weighted by their own measured
    IC instead of hand-picked percentages. This is the only kind of signal in
    this codebase that CAN be validated this way — it needs price history,
    which fundamentals/insider/sentiment signals don't have point-in-time.

    Returns (composite_series, signal_names_used, mean_ic_abs_per_signal).
    composite_series is roughly bounded in [-1, 1]; NaN where insufficient
    history exists (early bars).
    """
    signals_df = compute_signals_df(df)
    ic_df = compute_ic(signals_df, df["close"], horizons=horizons)
    summary = ic_summary(ic_df)
    mean_ic_abs = summary["mean_ic_abs"]

    ic_cols = [c for c in ic_df.columns if c[1] == "ic"]
    mean_ic_signed = ic_df[ic_cols].astype(float).mean(axis=1)

    # Normalize each signal to comparable units via its own expanding z-score —
    # raw signals live on wildly different scales (RSI ~[-1,1] vs trend_strength
    # ~[-0.05,0.05]) and can't be combined without this.
    z = (signals_df - signals_df.expanding(min_periods=30).mean()) / signals_df.expanding(min_periods=30).std()
    z = z.clip(-3, 3) / 3

    composite = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    used = []
    for sig in signals_df.columns:
        ic_abs = mean_ic_abs.get(sig, 0)
        if pd.isna(ic_abs) or ic_abs < min_abs_ic:
            continue
        sign = 1 if mean_ic_signed.get(sig, 0) >= 0 else -1
        composite = composite + sign * ic_abs * z[sig].fillna(0)
        total_weight += ic_abs
        used.append(sig)

    if total_weight > 0:
        composite = composite / total_weight
    else:
        composite = pd.Series(np.nan, index=df.index)

    return composite, used, mean_ic_abs


def ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    summary = ic_df.copy()

    ic_cols = [col for col in ic_df.columns if col[1] == "ic"]
    tstat_cols = [col for col in ic_df.columns if col[1] == "tstat"]

    ic_vals = ic_df[ic_cols].astype(float)
    tstat_vals = ic_df[tstat_cols].astype(float)

    summary["mean_ic_abs"] = ic_vals.abs().mean(axis=1)
    summary["max_tstat"] = tstat_vals.abs().max(axis=1)

    ic_signs = np.sign(ic_vals)
    # consistent = dominant sign appears in at least 3 of the 4 horizons
    pos_count = (ic_signs > 0).sum(axis=1)
    neg_count = (ic_signs < 0).sum(axis=1)
    summary["consistent"] = (pos_count >= 3) | (neg_count >= 3)

    summary = summary.sort_values("mean_ic_abs", ascending=False)

    return summary
