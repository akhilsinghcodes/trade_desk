"""IV30/RV30 volatility ratio — implied vs realized volatility signal."""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


def compute_vol_ratio(symbol: str, df: pd.DataFrame) -> dict:
    """
    Computes IV30/RV30 volatility ratio signal.

    Args:
        symbol: Ticker symbol (e.g., "AAPL")
        df: OHLCV DataFrame with at least 'close' column and 30+ rows

    Returns:
        dict with keys:
        - iv30: float or None (ATM implied volatility ~30 days out, annualized as decimal)
        - rv30: float or None (30-day realized volatility, annualized as decimal)
        - ratio: float or None (iv30 / rv30)
        - interpretation: str ("Vol overpriced", "Vol underpriced", "Vol fairly priced")
        - signal: float (-1.0 to 1.0, where 1.0 = vol overpriced, -1.0 = vol underpriced)
    """
    result = {
        "iv30": None,
        "rv30": None,
        "ratio": None,
        "interpretation": "No data",
        "signal": 0.0,
    }

    # Calculate RV30 first (this is reliable)
    if df is None or df.empty or len(df) < 30:
        return result

    try:
        # RV30: Close-to-close log returns std * sqrt(252)
        close_prices = pd.to_numeric(df["close"], errors="coerce")
        if close_prices.isna().all():
            return result

        pct_changes = close_prices.pct_change().dropna()
        if len(pct_changes) < 30:
            return result

        recent_returns = pct_changes.tail(30)
        rv30 = float(recent_returns.std() * np.sqrt(252))
        result["rv30"] = rv30
    except Exception:
        result["rv30"] = None
        return result

    # Get IV30 from options (may fail gracefully if no options)
    try:
        ticker_obj = yf.Ticker(symbol)

        # Get available expiration dates
        expirations = ticker_obj.options
        if not expirations or len(expirations) == 0:
            # No options available for this ticker
            result["interpretation"] = "No options data available"
            return result

        # Find expiry closest to 30 days out — minimum 14 DTE to avoid near-expiry IV collapse
        now = datetime.now()
        expiry_dates = [datetime.strptime(exp, "%Y-%m-%d") for exp in expirations]
        eligible = [e for e in expiry_dates if (e - now).days >= 14]
        if not eligible:
            eligible = expiry_dates  # fallback if nothing qualifies

        closest_expiry = min(eligible, key=lambda x: abs((x - now).days - 30))
        expiry_str = closest_expiry.strftime("%Y-%m-%d")

        # Fetch option chain for that expiry
        chain = ticker_obj.option_chain(expiry_str)
        calls = chain.calls
        puts = chain.puts

        # Get current stock price
        current_price = float(ticker_obj.info.get("currentPrice") or ticker_obj.info.get("regularMarketPrice", 0))
        if current_price <= 0:
            return result

        # yfinance options IV is unreliable: returns discrete placeholder values
        # (0.0625, 0.125, 0.25) with openInterest=0 when real data isn't populated.
        # Detect bad data: if >80% of non-zero IVs are identical, it's placeholders.
        all_chain_ivs = pd.concat([
            calls["impliedVolatility"], puts["impliedVolatility"]
        ]).dropna()
        nonzero_ivs = all_chain_ivs[all_chain_ivs > 0.001]

        if len(nonzero_ivs) == 0:
            result["interpretation"] = "IV unavailable — options data not populated"
            return result

        # Check if IVs are discrete placeholders (top value accounts for >60% of rows)
        top_freq = nonzero_ivs.value_counts().iloc[0] / len(nonzero_ivs)
        if top_freq > 0.60:
            result["interpretation"] = "IV unreliable — showing realized vol only"
            return result

        # Data looks real — use ATM range (within 5% of price)
        atm_mask_c = (calls["strike"].between(current_price * 0.95, current_price * 1.05) &
                      calls["impliedVolatility"].notna() & (calls["impliedVolatility"] > 0.05))
        atm_mask_p = (puts["strike"].between(current_price * 0.95, current_price * 1.05) &
                      puts["impliedVolatility"].notna() & (puts["impliedVolatility"] > 0.05))
        atm_ivs = pd.concat([
            calls.loc[atm_mask_c, "impliedVolatility"],
            puts.loc[atm_mask_p, "impliedVolatility"]
        ])
        if atm_ivs.empty:
            result["interpretation"] = "IV unavailable for ATM strikes"
            return result

        iv30 = float(atm_ivs.mean())
        result["iv30"] = iv30

        # Compute ratio
        if result["rv30"] is not None and result["rv30"] > 0 and iv30 is not None:
            ratio = iv30 / result["rv30"]
            result["ratio"] = ratio

            # Interpretation
            if ratio > 1.3:
                result["interpretation"] = "Vol overpriced — premium selling setup"
                # Signal: ratio > 1.3 = overpriced = bearish options = -1 (down pressure from options)
                # But if vol is overpriced, that's a selling opportunity, so from sentiment it's bearish
                # Let's normalize: ratio > 1.3 should give signal closer to -1 (bearish)
                signal = -((ratio - 1.0) / 1.0)  # Center at 1.0, invert
                signal = max(-1.0, min(1.0, signal))
                result["signal"] = signal
            elif ratio < 0.8:
                result["interpretation"] = "Vol underpriced — directional play"
                # Signal: ratio < 0.8 = underpriced = bullish options
                signal = (0.8 - ratio) / 0.8
                signal = max(-1.0, min(1.0, signal))
                result["signal"] = signal
            else:
                result["interpretation"] = "Vol fairly priced"
                result["signal"] = 0.0

        return result

    except Exception as e:
        # Gracefully return partial data (RV30 only)
        result["interpretation"] = f"Options unavailable: {type(e).__name__}"
        return result


def score_vol_ratio(data: dict) -> tuple[str, str, str]:
    """
    Returns (label, status, text) for signals list.

    Args:
        data: dict from compute_vol_ratio

    Returns:
        (label, status, text) tuple for signals list
    """
    iv30 = data.get("iv30")
    rv30 = data.get("rv30")
    ratio = data.get("ratio")
    interpretation = data.get("interpretation", "No data")

    if iv30 is None or rv30 is None:
        return ("Vol Ratio", "neutral", f"{interpretation}")

    if ratio is None:
        return ("Vol Ratio", "neutral", f"RV30: {rv30:.1%} (No IV available)")

    if ratio > 1.3:
        status = "warning"
        text = f"IV {iv30:.1%} / RV {rv30:.1%} = {ratio:.2f} — {interpretation}"
    elif ratio < 0.8:
        status = "good"
        text = f"IV {iv30:.1%} / RV {rv30:.1%} = {ratio:.2f} — {interpretation}"
    else:
        status = "neutral"
        text = f"IV {iv30:.1%} / RV {rv30:.1%} = {ratio:.2f} — {interpretation}"

    return ("Vol Ratio", status, text)
