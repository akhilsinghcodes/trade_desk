"""Feature computation for the published ML return model — ported from
trade_ml/modules/ml_dataset.py so trade_desk can run the model in-process,
live, for any ticker (not just the ones trade_ml has backtested).

This is a real code copy, not an import across repos — trade_ml and
trade_desk both have a top-level package named `modules`, so importing
across them would collide. When trade_ml's feature engineering changes,
this file needs to be re-synced by hand.

Skips the EDGAR point-in-time overlay trade_ml's training pipeline uses:
that only matters for historical rows (avoiding lookahead bias when
training on past dates). For a live snapshot of today, the current-value
fundamentals already computed below are already point-in-time correct.
"""
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress

from modules.indicators import add_common_indicators
from modules.db import cache_get, cache_set

warnings.filterwarnings("ignore")

# SPY and each sector ETF get refetched independently by every ticker that
# needs them — under a concurrent batch (e.g. Top Movers' 10-worker scan)
# that's 10+ simultaneous duplicate requests for the identical data, which
# both wastes calls and makes Yahoo's rate limiting/crumb errors more likely
# in the first place. Cached here so a batch run shares one fetch instead.
_MARKET_CLOSE_CACHE_TTL = 1800  # 30min — daily-bar indicators don't need fresher

SECTOR_ETF_MAP = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Basic Materials": "XLB", "Real Estate": "XLRE",
    "Utilities": "XLU", "Communication Services": "XLC",
}

TECHNICAL_FEATURES = [
    "rsi", "macd_above_signal", "bb_position", "sma20_vs_sma50",
    "volume_surge_20d", "atr_pct", "cmf_20d", "roc_21d",
    "alpha_21d", "sector_alpha_21d", "high_52w_proximity",
    "kmid", "klen", "kup", "klow", "ksft",
    "roc_5", "roc_10", "roc_60",
    "ma_5", "ma_10", "ma_20", "ma_60",
    "std_5", "std_10", "std_20", "std_60",
    "beta_5", "beta_10", "beta_20",
    "rsqr_10", "rsqr_20",
    "resi_10", "resi_20",
    "max_20", "min_20",
    "qtlu_10", "qtld_10",
    "rank_10",
    "corr_10",
    "cntp_10", "cntn_10",
    "vma_20",
]

FUNDAMENTAL_FEATURES = [
    "pe_ratio", "forward_pe", "profit_margin", "roe", "debt_to_equity",
    "revenue_growth", "earnings_growth", "fcf_yield", "gross_profitability",
    "piotroski_score", "altman_z",
]

SENTIMENT_FEATURES = [
    "short_interest_pct", "insider_net_ratio", "analyst_score",
]

FEATURE_COLS = TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES + SENTIMENT_FEATURES


def _fetch_close(symbol: str, period: str) -> pd.Series | None:
    cache_key = f"ml_features_close:{symbol}:{period}"
    cached = cache_get(cache_key)
    if cached is not None:
        s = pd.Series(cached["values"], index=pd.to_datetime(cached["index"]))
        return s

    try:
        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        s = df["Close"].squeeze()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        cache_set(
            cache_key,
            {"index": [d.isoformat() for d in s.index], "values": s.tolist()},
            _MARKET_CLOSE_CACHE_TTL,
        )
        return s
    except Exception:
        return None


def _piotroski(t: yf.Ticker) -> int:
    score = 0
    try:
        inc = t.financials
        bs = t.balance_sheet
        cf = t.cashflow
        if inc is None or inc.empty:
            return 0

        def row(df, *keys):
            if df is None or df.empty:
                return None
            for k in keys:
                for idx in df.index:
                    if k.lower() in str(idx).lower():
                        return df.loc[idx]
            return None

        def v(s, i=0):
            if s is None or len(s) <= i:
                return None
            val = s.iloc[i]
            return None if pd.isna(val) else float(val)

        ni = row(inc, "net income")
        ta = row(bs, "total assets")
        ocf = row(cf, "operating cash flow", "total cash from operations")
        ltd = row(bs, "long term debt")
        ca = row(bs, "current assets")
        cl = row(bs, "current liabilities")
        gp = row(inc, "gross profit")
        rev = row(inc, "total revenue", "revenue")

        ni0, ni1 = v(ni, 0), v(ni, 1)
        ta0, ta1 = v(ta, 0), v(ta, 1)
        ocf0 = v(ocf, 0)
        ltd0, ltd1 = v(ltd, 0), v(ltd, 1)
        ca0, ca1 = v(ca, 0), v(ca, 1)
        cl0, cl1 = v(cl, 0), v(cl, 1)
        gp0, gp1 = v(gp, 0), v(gp, 1)
        rev0, rev1 = v(rev, 0), v(rev, 1)

        if ta0 and ta0 != 0:
            roa = ni0 / ta0 if ni0 is not None else None
            if roa and roa > 0:
                score += 1
            if ocf0 and ocf0 > 0:
                score += 1
            if roa is not None and ta1 and ni1 is not None and ta1 != 0:
                if roa > ni1 / ta1:
                    score += 1
            if ocf0 and roa is not None and ocf0 / ta0 > roa:
                score += 1
            if ltd0 is not None and ltd1 is not None and ta1:
                if (ltd0 / ta0) < (ltd1 / ta1):
                    score += 1
            if ca0 and cl0 and ca1 and cl1 and cl0 != 0 and cl1 != 0:
                if (ca0 / cl0) > (ca1 / cl1):
                    score += 1
            if rev0 and rev1 and ta1 and rev0 != 0 and ta1 != 0:
                if (rev0 / ta0) > (rev1 / ta1):
                    score += 1
            if gp0 and rev0 and gp1 is not None and rev1 and rev0 != 0 and rev1 != 0:
                if (gp0 / rev0) > (gp1 / rev1):
                    score += 1
    except Exception:
        pass
    return min(score, 9)


def _altman_z(t: yf.Ticker, info: dict) -> float | None:
    try:
        bs = t.balance_sheet
        inc = t.financials
        if bs is None or bs.empty:
            return None

        def row(df, *keys):
            if df is None or df.empty:
                return None
            for k in keys:
                for idx in df.index:
                    if k.lower() in str(idx).lower():
                        return df.loc[idx]
            return None

        def v(s, i=0):
            if s is None or len(s) <= i:
                return None
            val = s.iloc[i]
            return None if pd.isna(val) else float(val)

        ta = v(row(bs, "total assets"))
        if not ta or ta == 0:
            return None

        ca = v(row(bs, "current assets"))
        cl = v(row(bs, "current liabilities"))
        re = v(row(bs, "retained earnings"))
        ebit = v(row(inc, "ebit", "operating income"))
        tl = v(row(bs, "total liabilities"))
        rev = v(row(inc, "total revenue", "revenue"))
        eq = v(row(bs, "stockholders equity", "total equity"))

        x1 = ((ca or 0) - (cl or 0)) / ta
        x2 = (re or 0) / ta
        x3 = (ebit or 0) / ta
        x4 = (eq or 0) / (tl or ta)
        x5 = (rev or 0) / ta

        return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    except Exception:
        return None


def _get_fundamentals(ticker: str) -> dict:
    result = {k: np.nan for k in FUNDAMENTAL_FEATURES}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        pe = info.get("trailingPE")
        result["pe_ratio"] = float(np.clip(pe, -100, 100)) if pe is not None else np.nan

        fpe = info.get("forwardPE")
        result["forward_pe"] = float(np.clip(fpe, -100, 100)) if fpe is not None else np.nan

        pm = info.get("profitMargins")
        result["profit_margin"] = float(pm) if pm is not None else np.nan

        roe = info.get("returnOnEquity")
        result["roe"] = float(roe) if roe is not None else np.nan

        de = info.get("debtToEquity")
        result["debt_to_equity"] = float(np.clip(de, -50, 50)) if de is not None else np.nan

        rg = info.get("revenueGrowth")
        result["revenue_growth"] = float(rg) if rg is not None else np.nan

        eg = info.get("earningsGrowth")
        result["earnings_growth"] = float(eg) if eg is not None else np.nan

        fcf = info.get("freeCashflow")
        mktcap = info.get("marketCap")
        if fcf and mktcap and mktcap > 0:
            result["fcf_yield"] = float(np.clip(fcf / mktcap, -0.5, 0.5))

        try:
            gp = info.get("grossProfits")
            bs = t.balance_sheet
            total_assets = None
            if bs is not None and not bs.empty:
                for idx in bs.index:
                    if "total assets" in str(idx).lower():
                        total_assets = float(bs.loc[idx].iloc[0])
                        break
            if gp is not None and total_assets and total_assets > 0:
                result["gross_profitability"] = float(np.clip(gp / total_assets, -1, 2))
        except Exception:
            pass

        try:
            result["piotroski_score"] = _piotroski(t) / 9.0
        except Exception:
            pass

        try:
            result["altman_z"] = float(np.clip(_altman_z(t, info), -5, 15))
        except Exception:
            pass

    except Exception:
        pass
    return result


def _insider_series(ticker: str, dates: pd.DatetimeIndex) -> pd.Series:
    result = pd.Series(0.0, index=dates)
    try:
        t = yf.Ticker(ticker)
        trans = t.insider_transactions
        if trans is None or trans.empty:
            return result

        date_col = next((c for c in ["startDate", "Start Date", "Date", "date"] if c in trans.columns), None)
        if date_col is None:
            return result
        trans = trans.copy()
        trans["_date"] = pd.to_datetime(trans[date_col], errors="coerce").dt.tz_localize(None)
        trans = trans.dropna(subset=["_date"])

        text_col = next((c for c in ["Text", "Transaction", "transaction"] if c in trans.columns), None)
        if text_col is None:
            return result

        trans["_buy"] = trans[text_col].str.contains("Purchase|Buy", case=False, na=False).astype(int)
        trans["_sell"] = trans[text_col].str.contains("Sale|Sell", case=False, na=False).astype(int)

        for i, date in enumerate(dates):
            cutoff = date - pd.Timedelta(days=90)
            mask = (trans["_date"] >= cutoff) & (trans["_date"] <= date)
            window = trans[mask]
            if window.empty:
                continue
            buys = window["_buy"].sum()
            sells = window["_sell"].sum()
            total = buys + sells
            result.iloc[i] = float((buys - sells) / total) if total > 0 else 0.0
    except Exception:
        pass
    return result


def _analyst_series(ticker: str, dates: pd.DatetimeIndex) -> pd.Series:
    result = pd.Series(0.0, index=dates)
    score_map = {
        "strong buy": 2, "buy": 1, "outperform": 1, "overweight": 1, "positive": 1,
        "hold": 0, "neutral": 0, "market perform": 0, "equal weight": 0, "in-line": 0,
        "underperform": -1, "sell": -1, "underweight": -1, "negative": -1, "strong sell": -2,
    }
    try:
        t = yf.Ticker(ticker)
        ud = t.upgrades_downgrades
        if ud is None or ud.empty:
            return result

        ud = ud.copy()
        ud.index = pd.to_datetime(ud.index).tz_localize(None)
        if "ToGrade" not in ud.columns:
            return result

        ud["_score"] = ud["ToGrade"].str.lower().map(score_map)

        for i, date in enumerate(dates):
            cutoff = date - pd.Timedelta(days=180)
            window = ud[(ud.index >= cutoff) & (ud.index <= date)]["_score"].dropna()
            if not window.empty:
                result.iloc[i] = float(window.mean())
    except Exception:
        pass
    return result


def _rolling_ols_metrics(x, window):
    if len(x) < window:
        return np.nan, np.nan, np.nan
    try:
        t = np.arange(window)
        result = linregress(t, x)
        slope = result.slope
        r_squared = result.rvalue ** 2
        fitted = result.intercept + result.slope * t
        residuals = x - fitted
        mean_abs_residual = np.mean(np.abs(residuals)) / (np.mean(x) + 1e-8)
        return slope, r_squared, mean_abs_residual
    except Exception:
        return np.nan, np.nan, np.nan


def _fetch_batch_ohlcv(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """One yf.download call for many tickers' OHLCV instead of one call per
    ticker — the single biggest source of redundant concurrent requests in a
    full-universe scan (e.g. Top Movers), which was helping trigger Yahoo's
    rate limiter. Fundamentals/insider/analyst data still go through
    per-ticker calls inside compute_live_features — yfinance has no batch API
    for those."""
    try:
        raw = yf.download(list(tickers), period=period, progress=False,
                           auto_adjust=True, group_by="ticker")
    except Exception:
        return {}
    if raw.empty:
        return {}
    result = {}
    for t in tickers:
        try:
            result[t] = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            result[t] = pd.DataFrame()
    return result


def compute_live_features_batch(tickers: list[str], period: str = "1y",
                                 max_workers: int = 10,
                                 progress_cb: Callable[[int, int, str], None] | None = None,
                                 ) -> dict[str, pd.DataFrame | None]:
    """Batch entry point for scanning many tickers — fetches all OHLCV in one
    request (eliminating N redundant price downloads), then computes each
    ticker's remaining per-ticker-only features (fundamentals, insider,
    analyst — no batch API exists for these in yfinance) concurrently, same
    as before. Use this instead of calling compute_live_features in a loop
    when scoring more than a handful of tickers.

    progress_cb(completed, total, ticker), if given, fires as each ticker's
    fundamentals step finishes — the OHLCV batch call itself is fast enough
    (one request) that it isn't worth granular progress on its own."""
    batch = _fetch_batch_ohlcv(tickers, period)
    results: dict[str, pd.DataFrame | None] = {}
    total = len(tickers)
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_live_features, t, period, batch.get(t)): t
            for t in tickers
        }
        for future in as_completed(futures):
            t = futures[future]
            completed += 1
            if progress_cb:
                progress_cb(completed, total, t)
            try:
                results[t] = future.result()
            except Exception:
                results[t] = None
    return results


def compute_live_features(ticker: str, period: str = "1y",
                           _prefetched_raw: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Full feature-complete DataFrame for one ticker (all rows in period).
    Caller takes the last row for a live prediction. Returns None if the
    ticker has insufficient price history.

    _prefetched_raw: pass OHLCV already fetched via _fetch_batch_ohlcv (batch
    scans) to skip this ticker's own yf.download call for price data."""
    try:
        if _prefetched_raw is not None:
            raw = _prefetched_raw
        else:
            raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if raw is None or raw.empty or len(raw) < 100:
            return None

        df = raw.reset_index()
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)
        df = add_common_indicators(df)

        c = df["close"].values
        h = df["high"].values
        lo = df["low"].values
        vol = df["volume"].values
        vol_ma20 = pd.Series(vol).rolling(20, min_periods=1).mean().values

        df["rsi"] = df["rsi"].clip(0, 100) / 100.0
        df["macd_above_signal"] = (df["macd"] > df["macd_signal"]).astype(float)
        bb_u, bb_l = df["bb_upper"].values, df["bb_lower"].values
        df["bb_position"] = np.clip((c - bb_l) / (bb_u - bb_l + 1e-8), 0, 1)
        df["sma20_vs_sma50"] = (df["sma20"] / (df["sma50"] + 1e-8) - 1).clip(-0.5, 0.5)
        df["volume_surge_20d"] = np.clip(vol / (vol_ma20 + 1e-8) - 1, -1, 3)

        tr = np.maximum(h - lo, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
        tr[0] = h[0] - lo[0]
        df["atr_pct"] = (pd.Series(tr).rolling(14, min_periods=1).mean().values / (c + 1e-8)).clip(0, 0.2)

        mf_mult = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-8)
        vol_sum = df["volume"].rolling(20, min_periods=20).sum().replace(0, np.nan)
        df["cmf_20d"] = ((mf_mult * df["volume"]).rolling(20, min_periods=20).sum() / vol_sum).clip(-1, 1)

        df["roc_21d"] = pd.Series(c).pct_change(21).clip(-0.5, 0.5).values

        spy_close = _fetch_close("SPY", period)
        stock_ret_21d = pd.Series(c).pct_change(21).values
        if spy_close is not None:
            spy_ret_21d = spy_close.pct_change(21)
            spy_a = spy_ret_21d.reindex(df["date"]).values
            df["alpha_21d"] = (stock_ret_21d - spy_a).clip(-0.5, 0.5)
        else:
            # SPY fetch failed (e.g. Yahoo rate-limited under concurrent load) —
            # NaN, not a silent substitute. Raw return is NOT the same signal as
            # market-relative alpha and was previously fed to the model as if it
            # were, producing wrong-but-plausible-looking predictions with no
            # trace it happened.
            df["alpha_21d"] = np.nan

        try:
            sector = (yf.Ticker(ticker).info or {}).get("sector")
            etf = SECTOR_ETF_MAP.get(sector)
            sec_close = _fetch_close(etf, period) if etf else None
            if sec_close is not None:
                sec_ret = sec_close.pct_change(21).reindex(df["date"]).values
                df["sector_alpha_21d"] = (stock_ret_21d - sec_ret).clip(-0.5, 0.5)
            else:
                df["sector_alpha_21d"] = np.nan
        except Exception:
            df["sector_alpha_21d"] = np.nan

        high_52w = df["close"].rolling(252, min_periods=252).max()
        df["high_52w_proximity"] = (df["close"] / high_52w.clip(lower=1e-8)).clip(0, 1)

        close_s = pd.Series(c)
        high_s = pd.Series(h)
        low_s = pd.Series(lo)
        vol_s = pd.Series(vol)

        df["kmid"] = ((c - df["open"].values) / (df["open"].values + 1e-8)).clip(-0.5, 0.5)
        df["klen"] = ((h - lo) / (df["open"].values + 1e-8)).clip(-0.5, 0.5)
        open_arr, close_arr = df["open"].values, c
        df["kup"] = ((h - np.maximum(open_arr, close_arr)) / (df["open"].values + 1e-8)).clip(-0.5, 0.5)
        df["klow"] = ((np.minimum(open_arr, close_arr) - lo) / (df["open"].values + 1e-8)).clip(-0.5, 0.5)
        df["ksft"] = ((c * 2 - h - lo) / (df["open"].values + 1e-8)).clip(-0.5, 0.5)

        df["roc_5"] = close_s.pct_change(5).clip(-0.5, 0.5).values
        df["roc_10"] = close_s.pct_change(10).clip(-0.5, 0.5).values
        df["roc_60"] = close_s.pct_change(60).clip(-0.5, 0.5).values

        for w in [5, 10, 20, 60]:
            ma = close_s.rolling(w, min_periods=w).mean()
            df[f"ma_{w}"] = (ma / close_s).clip(-0.5, 1.5).values

        for w in [5, 10, 20, 60]:
            std = close_s.rolling(w, min_periods=w).std()
            df[f"std_{w}"] = (std / close_s).clip(0, 1).values

        for w in [5, 10, 20]:
            slope_vals = close_s.rolling(w, min_periods=w).apply(
                lambda x: _rolling_ols_metrics(x.values, w)[0], raw=False
            )
            df[f"beta_{w}"] = (slope_vals / close_s).clip(-0.5, 0.5).values

        for w in [10, 20]:
            r2_vals = close_s.rolling(w, min_periods=w).apply(
                lambda x: _rolling_ols_metrics(x.values, w)[1], raw=False
            )
            df[f"rsqr_{w}"] = r2_vals.clip(0, 1).values
            resi_vals = close_s.rolling(w, min_periods=w).apply(
                lambda x: _rolling_ols_metrics(x.values, w)[2], raw=False
            )
            df[f"resi_{w}"] = resi_vals.clip(0, 1).values

        df["max_20"] = (high_s.rolling(20, min_periods=20).max() / close_s).clip(0, 2).values
        df["min_20"] = (low_s.rolling(20, min_periods=20).min() / close_s).clip(0, 1).values

        df["qtlu_10"] = (close_s.rolling(10, min_periods=10).quantile(0.8) / close_s).clip(0, 2).values
        df["qtld_10"] = (close_s.rolling(10, min_periods=10).quantile(0.2) / close_s).clip(0, 1).values

        df["rank_10"] = close_s.rolling(10, min_periods=10).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) == 10 else np.nan
        ).values

        close_chg = close_s.pct_change()
        vol_chg = vol_s.pct_change()
        df["corr_10"] = close_chg.rolling(10, min_periods=10).corr(vol_chg).clip(-1, 1).values

        df["cntp_10"] = (close_s.rolling(10, min_periods=10).apply(
            lambda x: (np.diff(x) > 0).sum() if len(x) == 10 else np.nan
        ) / 10).clip(0, 1).values
        df["cntn_10"] = (close_s.rolling(10, min_periods=10).apply(
            lambda x: (np.diff(x) < 0).sum() if len(x) == 10 else np.nan
        ) / 10).clip(0, 1).values

        vma_20 = vol_s.rolling(20, min_periods=20).mean()
        df["vma_20"] = (vma_20 / (close_s * 1e4)).clip(0, 10).values

        fund = _get_fundamentals(ticker)
        for k, v_val in fund.items():
            df[k] = v_val

        dates_dt = pd.DatetimeIndex(df["date"].values)
        df["insider_net_ratio"] = _insider_series(ticker, dates_dt).values
        df["analyst_score"] = _analyst_series(ticker, dates_dt).values

        try:
            info = yf.Ticker(ticker).info or {}
            si = info.get("shortPercentOfFloat")
            df["short_interest_pct"] = float(si) if si is not None else np.nan
        except Exception:
            df["short_interest_pct"] = np.nan

        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = np.nan

        df["ticker"] = ticker
        return df[["ticker", "date", "high", "low", "close"] + FEATURE_COLS].copy()

    except Exception:
        return None
