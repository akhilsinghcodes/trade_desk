import pandas as pd


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume indicators to the dataframe.

    Adds:
    - vol_ma20: 20-day rolling average of volume
    - vol_ratio: volume / vol_ma20 (1.0 = average, 2.0 = 2x average)
    - vol_price_confirm: bool, True if volume confirms price move

    Args:
        df: pandas DataFrame with columns: open, high, low, close, volume

    Returns:
        DataFrame with volume indicator columns added
    """
    df = df.copy()

    # Calculate 20-day rolling average of volume
    df['vol_ma20'] = df['volume'].rolling(window=20).mean()

    # Calculate volume ratio
    df['vol_ratio'] = df['volume'] / df['vol_ma20']

    # Calculate vol_price_confirm: True if volume above average AND price moved (up or down)
    price_up = df['close'] > df['close'].shift(1)
    price_down = df['close'] < df['close'].shift(1)
    volume_above_avg = df['volume'] > df['vol_ma20']

    df['vol_price_confirm'] = (price_up & volume_above_avg) | (price_down & volume_above_avg)

    return df


def volume_signal(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Generate a volume signal based on the last row of the dataframe.

    Returns a tuple of (status, label, plain_english_text).
    - status: "good", "warning", or "neutral"
    - label: "Volume"
    - plain_english_text: description of the signal

    Logic:
    - vol_ratio > 1.5 AND close > prev close → strong bullish volume
    - vol_ratio > 1.5 AND close < prev close → strong bearish volume
    - vol_ratio < 0.7 → weak volume, low conviction
    - else → normal volume, neutral signal

    Args:
        df: pandas DataFrame with volume indicators (must have vol_ratio column)

    Returns:
        Tuple of (status, label, plain_english_text)
    """
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    vol_ratio = last_row['vol_ratio']
    close = last_row['close']
    prev_close = prev_row['close']

    # vol_ratio > 1.5 AND close > prev close
    if vol_ratio > 1.5 and close > prev_close:
        return ("Volume", "good", "Price rising on HIGH volume — strong conviction move. Buyers are serious.")

    if vol_ratio > 1.5 and close < prev_close:
        return ("Volume", "warning", "Price falling on HIGH volume — strong selling pressure. Take caution.")

    if vol_ratio < 0.7:
        return ("Volume", "neutral", f"Volume is LOW ({vol_ratio:.1f}x average) — weak conviction. Don't trust this move.")

    return ("Volume", "neutral", f"Volume is normal ({vol_ratio:.1f}x average) — no strong confirmation signal.")
