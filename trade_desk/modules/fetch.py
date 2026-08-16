"""Data fetching — yfinance as primary source (offline-safe, no API key needed)."""
import time
import yfinance as yf
import pandas as pd


def _is_rate_limit(e: Exception) -> bool:
    """Check if exception is a rate-limit error."""
    msg = str(e).lower()
    return "429" in msg or "rate" in msg or "too many" in msg


def _retry_once(func, *args, **kwargs):
    """Call func once, retry once on rate limit (2s backoff)."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if _is_rate_limit(e):
            time.sleep(2)
            return func(*args, **kwargs)
        raise


def get_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data for a ticker. period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"""
    def _fetch():
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0).str.lower()
        else:
            df.columns = df.columns.str.lower()
        return df

    return _retry_once(_fetch)


def get_info(ticker: str) -> dict:
    """Fetch company fundamentals."""
    def _fetch():
        t = yf.Ticker(ticker)
        return t.info

    return _retry_once(_fetch)
