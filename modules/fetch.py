"""Data fetching — yfinance as primary source (offline-safe, no API key needed)."""
import yfinance as yf
import pandas as pd


def get_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data for a ticker. period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0).str.lower()
    else:
        df.columns = df.columns.str.lower()
    return df


def get_info(ticker: str) -> dict:
    """Fetch company fundamentals."""
    t = yf.Ticker(ticker)
    return t.info
