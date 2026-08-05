"""Cached wrappers around all expensive API calls. Uses SQLite cache with TTL."""
from modules.db import cache_get, cache_set

# TTLs in seconds
TTL_PRICE = 900        # 15 min — price data
TTL_FUNDAMENTALS = 86400   # 24 hr
TTL_NEWS = 1800        # 30 min
TTL_ANALYST = 86400    # 24 hr
TTL_INSIDER = 86400    # 24 hr
TTL_OWNERSHIP = 86400  # 24 hr
TTL_OPTIONS = 900      # 15 min
TTL_EARNINGS = 86400   # 24 hr
TTL_SHORT = 3600       # 1 hr
TTL_BALANCE = 86400    # 24 hr
TTL_REL_PERF = 900     # 15 min


def _df_to_records(df):
    """Serialize DataFrame for JSON storage."""
    import pandas as pd
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
    df = get_ohlcv(ticker, period=period)
    if not df.empty:
        cache_set(key, _df_to_records(df), TTL_PRICE)
    return df


def cached_info(ticker: str):
    from modules.fetch import get_info
    key = f"{ticker}:info"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_info(ticker)
    if data:
        cache_set(key, data, TTL_FUNDAMENTALS)
    return data


def cached_fundamentals(ticker: str):
    from modules.fundamentals import get_fundamentals
    key = f"{ticker}:fundamentals"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_fundamentals(ticker)
    cache_set(key, data, TTL_FUNDAMENTALS)
    return data


def cached_analyst(ticker: str):
    from modules.analyst import get_analyst_data
    key = f"{ticker}:analyst"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_analyst_data(ticker)
    cache_set(key, data, TTL_ANALYST)
    return data


def cached_insider(ticker: str):
    from modules.insider import get_insider_transactions
    key = f"{ticker}:insider"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_insider_transactions(ticker)
    cache_set(key, data, TTL_INSIDER)
    return data


def cached_ownership(ticker: str):
    from modules.ownership import get_ownership
    key = f"{ticker}:ownership"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_ownership(ticker)
    cache_set(key, data, TTL_OWNERSHIP)
    return data


def cached_options(ticker: str):
    from modules.options_sentiment import get_options_sentiment
    key = f"{ticker}:options"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_options_sentiment(ticker)
    cache_set(key, data, TTL_OPTIONS)
    return data


def cached_earnings_history(ticker: str):
    from modules.earnings_history import get_earnings_history
    key = f"{ticker}:earnings_history"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_earnings_history(ticker)
    cache_set(key, data, TTL_EARNINGS)
    return data


def cached_short_interest(ticker: str):
    from modules.short_interest import get_short_interest
    key = f"{ticker}:short_interest"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_short_interest(ticker)
    cache_set(key, data, TTL_SHORT)
    return data


def cached_balance_sheet(ticker: str):
    from modules.balance_sheet_trends import get_balance_sheet_trends
    key = f"{ticker}:balance_sheet"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_balance_sheet_trends(ticker)
    cache_set(key, data, TTL_BALANCE)
    return data


def cached_rel_perf(ticker: str, period: str):
    from modules.relative_performance import get_relative_performance, get_relative_performance_series
    key_perf = f"{ticker}:rel_perf:{period}"
    key_series = f"{ticker}:rel_series:{period}"
    perf = cache_get(key_perf)
    series = cache_get(key_series)
    if perf is None:
        perf = get_relative_performance(ticker, period)
        cache_set(key_perf, perf, TTL_REL_PERF)
    if series is None:
        series = get_relative_performance_series(ticker, period)
        cache_set(key_series, series, TTL_REL_PERF)
    return perf, series


def cached_market_context(ticker: str):
    from modules.market_context import get_market_context
    key = f"{ticker}:market_context"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_market_context(ticker)
    cache_set(key, data, TTL_PRICE)
    return data


def cached_piotroski(ticker: str):
    from modules.piotroski import get_piotroski
    key = f"{ticker}:piotroski"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_piotroski(ticker)
    cache_set(key, data, TTL_BALANCE)
    return data


def cached_valuation_advanced(ticker: str):
    from modules.valuation_advanced import get_advanced_valuation
    key = f"{ticker}:valuation_advanced"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_advanced_valuation(ticker)
    cache_set(key, data, TTL_FUNDAMENTALS)
    return data


def cached_sector_momentum(sector: str):
    from modules.sector_momentum import get_sector_momentum
    key = f"sector:{sector}:momentum"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_sector_momentum(sector)
    cache_set(key, data, TTL_PRICE)
    return data


def cached_dilution_risk(ticker: str):
    from modules.dilution_risk import get_dilution_risk
    key = f"{ticker}:dilution_risk"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_dilution_risk(ticker)
    cache_set(key, data, TTL_BALANCE)
    return data


def cached_altman_z(ticker: str):
    from modules.altman_z import get_altman_z
    key = f"{ticker}:altman_z"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_altman_z(ticker)
    cache_set(key, data, TTL_BALANCE)
    return data


def cached_momentum(ticker: str):
    from modules.momentum import get_momentum
    key = f"{ticker}:momentum"
    cached = cache_get(key)
    if cached is not None:
        return cached
    data = get_momentum(ticker)
    cache_set(key, data, TTL_PRICE)
    return data


def cached_news(ticker: str, limit: int, company: str):
    from modules.news import get_news, analyze_sentiment
    key = f"{ticker}:news:{limit}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    articles = get_news(ticker, limit=limit, company=company)
    articles = analyze_sentiment(articles)
    cache_set(key, articles, TTL_NEWS)
    return articles
