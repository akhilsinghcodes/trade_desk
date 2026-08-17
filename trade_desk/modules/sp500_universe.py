"""S&P 500 ticker universe for live-scan pages (e.g. Top Movers).

Fetched from Wikipedia and cached — NOT a static file, NOT the same list
trade_ml/tickers.txt uses for training. This module has no relationship
to trade_ml; it exists so pages that need "scan the whole index" have a
ticker list without touching anything training-related.
"""
import logging
from io import StringIO

import pandas as pd
import urllib.request

from modules.db import cache_get, cache_set, cache_get_stale

logger = logging.getLogger(__name__)

_CACHE_KEY = "sp500_universe"
_TTL_SECONDS = 86400  # 1 day
_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers() -> list[str]:
    """Sorted list of current S&P 500 tickers. Cached 1 day; falls back to
    stale cache on fetch failure so a Wikipedia hiccup doesn't break the page."""
    cached = cache_get(_CACHE_KEY)
    if cached is not None:
        return cached
    try:
        req = urllib.request.Request(_WIKI_URL, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
        df = pd.read_html(StringIO(html))[0]
        tickers = sorted(df["Symbol"].str.replace(".", "-", regex=False).tolist())
        cache_set(_CACHE_KEY, tickers, _TTL_SECONDS)
        return tickers
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500 list: {e}")
        stale = cache_get_stale(_CACHE_KEY)
        if stale is not None:
            logger.warning("Returning stale S&P 500 list")
            return stale
        return []
