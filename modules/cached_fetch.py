"""Cached wrappers around all expensive API calls. Uses SQLite cache with TTL."""
import logging
from modules.db import cache_get, cache_set, cache_get_stale
from modules.config import get_ttl

logger = logging.getLogger(__name__)


def _df_to_records(df):
    """Serialize DataFrame for JSON storage."""
    if df is None or df.empty:
        return None
    return {"index": [str(i) for i in df.index], "data": df.to_dict(orient="list")}


def _records_to_df(records):
    """Deserialize DataFrame from JSON storage."""
    import pandas as pd
    if records is None:
        return pd.DataFrame()
    df = pd.DataFrame(records["data"])
    df.index = pd.to_datetime(records["index"])
    return df


def cached_ohlcv(ticker: str, period: str):
    from modules.fetch import get_ohlcv
    key = f"{ticker}:ohlcv:{period}"
    cached = cache_get(key)
    if cached is not None:
        return _records_to_df(cached)
    try:
        df = get_ohlcv(ticker, period=period)
        if not df.empty:
            cache_set(key, _df_to_records(df), get_ttl("price"))
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch OHLCV for {ticker}: {e}")
        if cached is not None:
            logger.warning(f"Returning stale OHLCV for {ticker}")
            return _records_to_df(cached)
        return None


def cached_info(ticker: str):
    from modules.fetch import get_info
    key = f"{ticker}:info"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_info(ticker)
        if data:
            cache_set(key, data, get_ttl("fundamentals"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch info for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale info for {ticker}")
            return stale
        return None


def cached_fundamentals(ticker: str):
    from modules.fundamentals import get_fundamentals
    key = f"{ticker}:fundamentals"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_fundamentals(ticker)
        cache_set(key, data, get_ttl("fundamentals"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale fundamentals for {ticker}")
            return stale
        return None


def cached_analyst(ticker: str):
    from modules.analyst import get_analyst_data
    key = f"{ticker}:analyst"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_analyst_data(ticker)
        cache_set(key, data, get_ttl("analyst"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch analyst data for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale analyst data for {ticker}")
            return stale
        return None


def cached_insider(ticker: str):
    from modules.insider import get_insider_transactions
    key = f"{ticker}:insider"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_insider_transactions(ticker)
        cache_set(key, data, get_ttl("insider"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch insider data for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale insider data for {ticker}")
            return stale
        return None


def cached_ownership(ticker: str):
    from modules.ownership import get_ownership
    key = f"{ticker}:ownership"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_ownership(ticker)
        cache_set(key, data, get_ttl("ownership"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch ownership for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale ownership for {ticker}")
            return stale
        return None


def cached_options(ticker: str):
    from modules.options_sentiment import get_options_sentiment
    key = f"{ticker}:options"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_options_sentiment(ticker)
        cache_set(key, data, get_ttl("options"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch options data for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale options data for {ticker}")
            return stale
        return None


def cached_earnings_history(ticker: str):
    from modules.earnings_history import get_earnings_history
    key = f"{ticker}:earnings_history"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_earnings_history(ticker)
        cache_set(key, data, get_ttl("earnings"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch earnings history for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale earnings history for {ticker}")
            return stale
        return None


def cached_short_interest(ticker: str):
    from modules.short_interest import get_short_interest
    key = f"{ticker}:short_interest"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_short_interest(ticker)
        cache_set(key, data, get_ttl("short"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch short interest for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale short interest for {ticker}")
            return stale
        return None


def cached_balance_sheet(ticker: str):
    from modules.balance_sheet_trends import get_balance_sheet_trends
    key = f"{ticker}:balance_sheet"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_balance_sheet_trends(ticker)
        cache_set(key, data, get_ttl("balance_sheet"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch balance sheet for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale balance sheet for {ticker}")
            return stale
        return None


def cached_rel_perf(ticker: str, period: str):
    from modules.relative_performance import get_relative_performance, get_relative_performance_series
    key_perf = f"{ticker}:rel_perf:{period}"
    key_series = f"{ticker}:rel_series:{period}"
    perf = cache_get(key_perf)
    series = cache_get(key_series)
    ttl = get_ttl("relative_performance")
    if perf is None:
        try:
            perf = get_relative_performance(ticker, period)
            cache_set(key_perf, perf, ttl)
        except Exception as e:
            logger.warning(f"Failed to fetch relative performance for {ticker}: {e}")
            perf = cache_get_stale(key_perf)
            if perf is None:
                perf = {}
    if series is None:
        try:
            series = get_relative_performance_series(ticker, period)
            cache_set(key_series, series, ttl)
        except Exception as e:
            logger.warning(f"Failed to fetch relative performance series for {ticker}: {e}")
            series = cache_get_stale(key_series)
            if series is None:
                series = []
    return perf, series


def cached_market_context(ticker: str):
    from modules.market_context import get_market_context
    key = f"{ticker}:market_context"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_market_context(ticker)
        cache_set(key, data, get_ttl("market_context"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch market context for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale market context for {ticker}")
            return stale
        return None


def cached_piotroski(ticker: str):
    from modules.piotroski import get_piotroski
    key = f"{ticker}:piotroski"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_piotroski(ticker)
        cache_set(key, data, get_ttl("piotroski"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch piotroski score for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale piotroski score for {ticker}")
            return stale
        return None


def cached_valuation_advanced(ticker: str):
    from modules.valuation_advanced import get_advanced_valuation
    key = f"{ticker}:valuation_advanced"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_advanced_valuation(ticker)
        cache_set(key, data, get_ttl("valuation_advanced"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch advanced valuation for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale advanced valuation for {ticker}")
            return stale
        return None


def cached_sector_momentum(sector: str):
    from modules.sector_momentum import get_sector_momentum
    key = f"sector:{sector}:momentum"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_sector_momentum(sector)
        cache_set(key, data, get_ttl("sector_momentum"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch sector momentum for {sector}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale sector momentum for {sector}")
            return stale
        return None


def cached_dilution_risk(ticker: str):
    from modules.dilution_risk import get_dilution_risk
    key = f"{ticker}:dilution_risk"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_dilution_risk(ticker)
        cache_set(key, data, get_ttl("dilution_risk"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch dilution risk for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale dilution risk for {ticker}")
            return stale
        return None


def cached_altman_z(ticker: str):
    from modules.altman_z import get_altman_z
    key = f"{ticker}:altman_z"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_altman_z(ticker)
        cache_set(key, data, get_ttl("altman_z"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch altman z score for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale altman z score for {ticker}")
            return stale
        return None


def cached_momentum(ticker: str):
    from modules.momentum import get_momentum
    key = f"{ticker}:momentum"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = get_momentum(ticker)
        cache_set(key, data, get_ttl("momentum"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch momentum for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale momentum for {ticker}")
            return stale
        return None


def cached_news(ticker: str, limit: int, company: str):
    from modules.news import get_news, analyze_sentiment
    key = f"{ticker}:news:{limit}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        articles = get_news(ticker, limit=limit, company=company)
        articles = analyze_sentiment(articles)
        cache_set(key, articles, get_ttl("news"))
        return articles
    except Exception as e:
        logger.warning(f"Failed to fetch news for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale news for {ticker}")
            return stale
        return None


def cached_pmo_rs(ticker: str, period: str = "1y"):
    from modules.pmo import compute_pmo_rs
    key = f"{ticker}:pmo_rs:{period}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = compute_pmo_rs(ticker, period=period)
        cache_set(key, data, get_ttl("pmo_rs"))
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch PMO RS for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale PMO RS for {ticker}")
            return stale
        return None


def cached_vol_ratio(ticker: str, df):
    """Cache wrapper for IV30/RV30 volatility ratio."""
    from modules.vol_ratio import compute_vol_ratio
    key = f"{ticker}:vol_ratio"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        data = compute_vol_ratio(ticker, df)
        cache_set(key, data, get_ttl("options"))
        return data
    except Exception as e:
        logger.warning(f"Failed to compute vol ratio for {ticker}: {e}")
        stale = cache_get_stale(key)
        if stale is not None:
            logger.warning(f"Returning stale vol ratio for {ticker}")
            return stale
        return None
