"""Relative performance vs S&P 500 (SPY) from yfinance."""
import yfinance as yf
import pandas as pd


def get_relative_performance(ticker: str, period: str = "1y") -> dict:
    """
    Compares ticker return vs SPY over the given period.
    Returns dict:
    - ticker_return_pct: float
    - spy_return_pct: float
    - alpha_pct: float (ticker - spy)
    - outperforming: bool
    - period: str
    """
    result = {
        "ticker_return_pct": None,
        "spy_return_pct": None,
        "alpha_pct": None,
        "outperforming": None,
        "period": period,
    }

    try:
        ticker_df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        spy_df = yf.download("SPY", period=period, progress=False, auto_adjust=True)

        if ticker_df.empty or spy_df.empty:
            return result

        # Flatten MultiIndex if present
        if isinstance(ticker_df.columns, pd.MultiIndex):
            ticker_df.columns = ticker_df.columns.get_level_values(0)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)

        ticker_df.columns = ticker_df.columns.str.lower()
        spy_df.columns = spy_df.columns.str.lower()

        t_start = ticker_df["close"].iloc[0]
        t_end = ticker_df["close"].iloc[-1]
        s_start = spy_df["close"].iloc[0]
        s_end = spy_df["close"].iloc[-1]

        t_ret = (t_end - t_start) / t_start * 100
        s_ret = (s_end - s_start) / s_start * 100
        alpha = t_ret - s_ret

        result["ticker_return_pct"] = round(float(t_ret), 1)
        result["spy_return_pct"] = round(float(s_ret), 1)
        result["alpha_pct"] = round(float(alpha), 1)
        result["outperforming"] = alpha > 0

    except Exception:
        pass

    return result


def get_relative_performance_series(ticker: str, period: str = "1y") -> dict:
    """
    Returns normalized price series for both ticker and SPY (base=100).
    Useful for plotting.
    Returns dict:
    - dates: list of str
    - ticker_series: list of float (normalized)
    - spy_series: list of float (normalized)
    """
    result = {"dates": [], "ticker_series": [], "spy_series": []}

    try:
        ticker_df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        spy_df = yf.download("SPY", period=period, progress=False, auto_adjust=True)

        if ticker_df.empty or spy_df.empty:
            return result

        if isinstance(ticker_df.columns, pd.MultiIndex):
            ticker_df.columns = ticker_df.columns.get_level_values(0)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)

        ticker_df.columns = ticker_df.columns.str.lower()
        spy_df.columns = spy_df.columns.str.lower()

        # Align on common dates
        common = ticker_df.index.intersection(spy_df.index)
        t_close = ticker_df.loc[common, "close"]
        s_close = spy_df.loc[common, "close"]

        t_norm = (t_close / t_close.iloc[0] * 100).round(2)
        s_norm = (s_close / s_close.iloc[0] * 100).round(2)

        result["dates"] = [str(d)[:10] for d in common]
        result["ticker_series"] = t_norm.tolist()
        result["spy_series"] = s_norm.tolist()

    except Exception:
        pass

    return result


def score_relative_performance(data: dict, ticker: str = "") -> tuple[str, str, str]:
    """Returns (label, status, text) for signals list."""
    alpha = data.get("alpha_pct")
    t_ret = data.get("ticker_return_pct")
    s_ret = data.get("spy_return_pct")
    period = data.get("period", "1y")

    if alpha is None:
        return ("vs Market", "neutral", "Relative performance data unavailable.")

    t_str = f"+{t_ret:.1f}%" if t_ret and t_ret >= 0 else f"{t_ret:.1f}%"
    s_str = f"+{s_ret:.1f}%" if s_ret and s_ret >= 0 else f"{s_ret:.1f}%"
    a_str = f"+{alpha:.1f}%" if alpha >= 0 else f"{alpha:.1f}%"

    label = ticker if ticker else "Stock"
    if alpha > 5:
        status = "good"
        text = f"{label} {t_str} vs S&P 500 {s_str} over {period} — outperforming by {a_str}."
    elif alpha < -5:
        status = "warning"
        text = f"{label} {t_str} vs S&P 500 {s_str} over {period} — underperforming by {a_str}."
    else:
        status = "neutral"
        text = f"{label} {t_str} vs S&P 500 {s_str} over {period} — roughly in line ({a_str})."

    return ("vs Market", status, text)
