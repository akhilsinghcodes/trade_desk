"""
ML dataset: trade-outcome classification.

Features:
  Technical  : daily OHLCV-derived signals
  Fundamental: current snapshot from yf.info (static per ticker — company character
               changes slowly; ranking by P/E/ROE/Piotroski is stable over years)
  Sentiment  : insider buy ratio and analyst score computed time-aware from
               historical transaction/recommendation dates; short interest static

Target: WIN if price hits take-profit (entry + 3*ATR) before stop (entry - 1.5*ATR)
        within 21 trading days. LOSS otherwise. NEUTRAL (neither hit) excluded.
        Mirrors suggest_trade() TP1 logic exactly.
"""
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import linregress
from modules.indicators import add_common_indicators
from modules import edgar_fundamentals

warnings.filterwarnings("ignore")

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "BAC", "GS", "MS", "V", "MA", "WFC",
    "JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK",
    "XOM", "CVX", "COP", "SLB",
    "PG", "KO", "PEP", "WMT", "COST",
    "HD", "NKE", "MCD", "SBUX", "TGT",
    "BA", "CAT", "HON", "GE",
    "NEE", "DUK",
    "AMT", "PLD",
    "CRWD", "DDOG", "SNOW", "PANW",
    "AMD", "MU", "NFLX", "PYPL",
]

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
    # Alpha158-style candlestick patterns (5 factors)
    "kmid", "klen", "kup", "klow", "ksft",
    # Rate of change at multiple windows (5, 10, 60d; 21d already above)
    "roc_5", "roc_10", "roc_60",
    # Moving averages at multiple windows (normalized by close)
    "ma_5", "ma_10", "ma_20", "ma_60",
    # Rolling volatility (std dev) at multiple windows
    "std_5", "std_10", "std_20", "std_60",
    # Regression-based factors (slope/R-squared/residual from rolling OLS)
    "beta_5", "beta_10", "beta_20",
    "rsqr_10", "rsqr_20",
    "resi_10", "resi_20",
    # Running extrema normalized by current close
    "max_20", "min_20",
    # Quantile-based features (80th/20th percentiles)
    "qtlu_10", "qtld_10",
    # Percentile rank of close within window
    "rank_10",
    # Price-volume correlation
    "corr_10",
    # Count of up/down days in window
    "cntp_10", "cntn_10",
    # Volume-based features
    "vma_20",
]

FUNDAMENTAL_FEATURES = [
    "pe_ratio", "forward_pe", "profit_margin", "roe", "debt_to_equity",
    "revenue_growth", "earnings_growth", "fcf_yield", "gross_profitability",
    "piotroski_score", "altman_z",
]

SENTIMENT_FEATURES = [
    "short_interest_pct", "insider_net_ratio", "analyst_score",
    # put_call_ratio removed: no historical options chain data source exists,
    # it was hardcoded to 0.0 for every ticker/date — a dead, zero-information
    # feature masquerading as real signal. Don't re-add without a real
    # historical options data source.
]

FEATURE_COLS = TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES + SENTIMENT_FEATURES

LABEL_LOOKAHEAD_DAYS = 21  # WIN/LOSS label window; also the walk-forward CV embargo gap


def get_feature_cols() -> list[str]:
    return FEATURE_COLS


def _impute_missing(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Fill NaN feature values with the cross-sectional median for that date
    (the day's other tickers), instead of a fabricated constant like 0 —
    a missing PE ratio becomes "the universe's median PE that day", not
    "zero PE", which was silently wrong. Falls back to the column's global
    median if an entire date has no valid values (e.g. a single-ticker
    live scoring call), then to 0.0 only as a last resort.
    """
    df = df.copy()
    for col in feature_cols:
        df[col] = df.groupby("date")[col].transform(lambda s: s.fillna(s.median()))
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


# ── Fundamental snapshot ───────────────────────────────────────────────────────

def _get_fundamentals(ticker: str) -> dict:
    """
    Fetch current fundamentals from yf.info + annual statements for Piotroski/Altman.
    Applied as a static snapshot to all historical rows for this ticker.
    Rationale: P/E quintile, profitability class, leverage level are stable company
    characteristics over the 5-7y training window. Ranking signal persists.
    """
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

        # Novy-Marx gross profitability: gross_profit / total_assets — replicates
        # better than most fundamental factors (Novy-Marx, 2013, "The Other Side
        # of Value").
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

        # Piotroski from annual statements
        try:
            result["piotroski_score"] = _piotroski(t) / 9.0
        except Exception:
            pass

        # Altman Z from annual statements
        try:
            result["altman_z"] = float(np.clip(_altman_z(t, info), -5, 15))
        except Exception:
            pass

    except Exception:
        pass
    return result


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

        return 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
    except Exception:
        return None


# ── Time-aware sentiment ───────────────────────────────────────────────────────

def _insider_series(ticker: str, dates: pd.DatetimeIndex) -> pd.Series:
    """
    Compute rolling 90d net insider buy ratio at each date.
    Returns Series indexed by date, values in [-1, 1].
    """
    result = pd.Series(0.0, index=dates)
    try:
        t = yf.Ticker(ticker)
        trans = t.insider_transactions
        if trans is None or trans.empty:
            return result

        # Normalize date column
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
    """
    Rolling analyst consensus score from historical upgrades/downgrades.
    strong buy=2, buy=1, hold=0, sell=-1, strong sell=-2
    Uses 180-day rolling window.
    """
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


# ── Reference data ─────────────────────────────────────────────────────────────

def _fetch_close(symbol: str, period: str) -> pd.Series | None:
    try:
        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        s = df["Close"].squeeze()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception:
        return None


# ── Per-ticker feature builder (shared by training and live inference) ────────

def _compute_ticker_features(
    ticker: str,
    period: str,
    spy_close: pd.Series | None,
    spy_ret_21d: pd.Series | None,
    sector_etf_closes: dict[str, pd.Series] = None,
) -> pd.DataFrame | None:
    """
    Build feature-complete DataFrame for one ticker (all rows in period).
    Computes technical, fundamental, and sentiment features without labels.
    Returns DataFrame with date + FEATURE_COLS columns.
    Returns None if ticker has insufficient data.
    """
    if sector_etf_closes is None:
        sector_etf_closes = {}

    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if raw.empty or len(raw) < 100:
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

        # ── Technical ────────────────────────────────────────────────────
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

        stock_ret_21d = pd.Series(c).pct_change(21).values
        if spy_ret_21d is not None:
            spy_a = spy_ret_21d.reindex(df["date"]).values
            df["alpha_21d"] = (stock_ret_21d - spy_a).clip(-0.5, 0.5)
        else:
            df["alpha_21d"] = stock_ret_21d

        try:
            sector = yf.Ticker(ticker).info.get("sector")
            etf = SECTOR_ETF_MAP.get(sector)
            if etf:
                sec_close = sector_etf_closes.get(etf)
                if sec_close is None:
                    sec_close = _fetch_close(etf, period)
                    if sec_close is not None:
                        sector_etf_closes[etf] = sec_close
                if sec_close is not None:
                    sec_ret = sec_close.pct_change(21).reindex(df["date"]).values
                    df["sector_alpha_21d"] = (stock_ret_21d - sec_ret).clip(-0.5, 0.5)
                else:
                    df["sector_alpha_21d"] = df["alpha_21d"]
            else:
                df["sector_alpha_21d"] = df["alpha_21d"]
        except Exception:
            df["sector_alpha_21d"] = df["alpha_21d"]

        # min_periods=252 (full window) on purpose: a partial window doesn't compute
        # a genuine 52-week high, it silently returns a distorted proxy. Real NaN
        # for young tickers is correct here — filled later by cross-sectional median.
        high_52w = df["close"].rolling(252, min_periods=252).max()
        df["high_52w_proximity"] = (df["close"] / high_52w.clip(lower=1e-8)).clip(0, 1)

        # ── Alpha158-style factors ───────────────────────────────────────────────
        close_s = pd.Series(c)
        high_s = pd.Series(h)
        low_s = pd.Series(lo)
        vol_s = pd.Series(vol)

        # Candlestick patterns (OHLC-based)
        df["kmid"] = ((c - df["open"].values) / (df["open"].values + 1e-8)).clip(-0.5, 0.5)
        df["klen"] = ((h - lo) / (df["open"].values + 1e-8)).clip(-0.5, 0.5)
        open_arr, close_arr = df["open"].values, c
        df["kup"] = ((h - np.maximum(open_arr, close_arr)) / (df["open"].values + 1e-8)).clip(-0.5, 0.5)
        df["klow"] = ((np.minimum(open_arr, close_arr) - lo) / (df["open"].values + 1e-8)).clip(-0.5, 0.5)
        df["ksft"] = ((c * 2 - h - lo) / (df["open"].values + 1e-8)).clip(-0.5, 0.5)

        # Rate of change at multiple windows (5, 10, 60 days; 21d already computed above)
        df["roc_5"] = close_s.pct_change(5).clip(-0.5, 0.5).values
        df["roc_10"] = close_s.pct_change(10).clip(-0.5, 0.5).values
        df["roc_60"] = close_s.pct_change(60).clip(-0.5, 0.5).values

        # Moving averages (normalized by close): mean(close, w) / close
        for w in [5, 10, 20, 60]:
            ma = close_s.rolling(w, min_periods=w).mean()
            df[f"ma_{w}"] = (ma / close_s).clip(-0.5, 1.5).values

        # Volatility (standard deviation normalized by close)
        for w in [5, 10, 20, 60]:
            std = close_s.rolling(w, min_periods=w).std()
            df[f"std_{w}"] = (std / close_s).clip(0, 1).values

        # Rolling OLS regression: y = close, x = time index [0, 1, ..., w-1]
        # Beta: regression slope (close change per day) / current close
        # Rsqr: coefficient of determination (real R²) from the regression
        # Resi: mean absolute residual from the fit, normalized by mean close

        def _rolling_ols_metrics(x, window):
            """
            Compute OLS regression metrics for a rolling window.
            Returns (slope, r_squared, mean_abs_residual) normalized appropriately.
            """
            if len(x) < window:
                return np.nan, np.nan, np.nan
            try:
                # Fit y = a + b*t where t = [0, 1, ..., window-1]
                t = np.arange(window)
                result = linregress(t, x)
                slope = result.slope
                r_squared = result.rvalue ** 2

                # Fitted values and residuals
                fitted = result.intercept + result.slope * t
                residuals = x - fitted
                mean_abs_residual = np.mean(np.abs(residuals)) / (np.mean(x) + 1e-8)

                return slope, r_squared, mean_abs_residual
            except Exception:
                return np.nan, np.nan, np.nan

        # Beta (regression slope normalized by close)
        for w in [5, 10, 20]:
            slope_vals = close_s.rolling(w, min_periods=w).apply(
                lambda x: _rolling_ols_metrics(x.values, w)[0], raw=False
            )
            # Normalize by close price
            df[f"beta_{w}"] = (slope_vals / close_s).clip(-0.5, 0.5).values

        # R-squared and residual (real OLS metrics from rolling regression)
        for w in [10, 20]:
            # Extract R² values from rolling regression
            r2_vals = close_s.rolling(w, min_periods=w).apply(
                lambda x: _rolling_ols_metrics(x.values, w)[1], raw=False
            )
            df[f"rsqr_{w}"] = r2_vals.clip(0, 1).values

            # Extract mean absolute residual (already normalized in the function)
            resi_vals = close_s.rolling(w, min_periods=w).apply(
                lambda x: _rolling_ols_metrics(x.values, w)[2], raw=False
            )
            df[f"resi_{w}"] = resi_vals.clip(0, 1).values

        # Running max/min normalized by close
        df["max_20"] = (high_s.rolling(20, min_periods=20).max() / close_s).clip(0, 2).values
        df["min_20"] = (low_s.rolling(20, min_periods=20).min() / close_s).clip(0, 1).values

        # Quantile-based features: 80th and 20th percentile of close over window
        def _quantile(s, q):
            return s.rolling(10, min_periods=10).quantile(q)

        df["qtlu_10"] = (_quantile(close_s, 0.8) / close_s).clip(0, 2).values
        df["qtld_10"] = (_quantile(close_s, 0.2) / close_s).clip(0, 1).values

        # Percentile rank: what percentile is today's close within the 10-day window?
        def _rank(s):
            return s.rolling(10, min_periods=10).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) == 10 else np.nan
            )

        df["rank_10"] = _rank(close_s).values

        # Price-volume correlation: rolling correlation between close and volume
        def _corr_vol(close, vol):
            return close.rolling(10, min_periods=10).apply(
                lambda x_: vol.iloc[x_.index[-10]:x_.index[-1]+1].corr(
                    close.iloc[x_.index[-10]:x_.index[-1]+1]
                ) if len(x_) == 10 else np.nan,
                raw=False
            )

        # Simplified: just correlation of price changes vs volume changes
        close_chg = close_s.pct_change()
        vol_chg = vol_s.pct_change()
        df["corr_10"] = close_chg.rolling(10, min_periods=10).corr(vol_chg).clip(-1, 1).values

        # Count of up days and down days in window
        def _count_up(s):
            return s.rolling(10, min_periods=10).apply(
                lambda x: (np.diff(x) > 0).sum() if len(x) == 10 else np.nan
            )

        def _count_down(s):
            return s.rolling(10, min_periods=10).apply(
                lambda x: (np.diff(x) < 0).sum() if len(x) == 10 else np.nan
            )

        df["cntp_10"] = (_count_up(close_s) / 10).clip(0, 1).values
        df["cntn_10"] = (_count_down(close_s) / 10).clip(0, 1).values

        # Volume moving average normalized by close
        vma_20 = vol_s.rolling(20, min_periods=20).mean()
        df["vma_20"] = (vma_20 / (close_s * 1e4)).clip(0, 10).values

        # ── Fundamentals (current snapshot, applied to all rows) ─────────
        fund = _get_fundamentals(ticker)
        for k, v_val in fund.items():
            df[k] = v_val

        # ── Point-in-time overlay for the 4 fields SEC XBRL can derive ────
        # Replaces the static current-value snapshot above with the value
        # actually known as of each historical date, for revenue_growth,
        # profit_margin, roe, debt_to_equity. Falls back to the static
        # snapshot (already set above) for tickers with no SEC CIK (e.g.
        # foreign ADRs) or dates before that ticker's earliest EDGAR filing.
        try:
            edgar_series = edgar_fundamentals.build_point_in_time_series(ticker)
        except Exception:
            edgar_series = pd.DataFrame()

        if not edgar_series.empty:
            edgar_cols = ["revenue_growth", "profit_margin", "roe", "debt_to_equity"]
            df = df.sort_values("date").reset_index(drop=True)
            edgar_series = edgar_series.copy()
            # merge_asof requires identical datetime64 units on both sides —
            # yfinance and EDGAR timestamps can land on different units (s vs us)
            # depending on pandas/source, which raises MergeError otherwise.
            left = df[["date"]].assign(date=lambda d: d["date"].astype("datetime64[ns]"))
            edgar_series["filed_date"] = edgar_series["filed_date"].astype("datetime64[ns]")
            merged = pd.merge_asof(
                left, edgar_series.sort_values("filed_date"),
                left_on="date", right_on="filed_date", direction="backward"
            )
            for col in edgar_cols:
                df[col] = merged[col].combine_first(df[col])

        # ── Sentiment (time-aware where possible) ────────────────────────
        dates_idx = df["date"].values
        dates_dt = pd.DatetimeIndex(dates_idx)

        insider = _insider_series(ticker, dates_dt)
        df["insider_net_ratio"] = insider.values

        analyst = _analyst_series(ticker, dates_dt)
        df["analyst_score"] = analyst.values

        # NOTE: short_interest_pct is a static current-value snapshot applied to
        # all historical rows, same known limitation as fundamentals (see
        # _get_fundamentals docstring) — short interest is more time-varying than
        # fundamentals, so this is a real, currently-unfixed staleness issue.
        # A proper fix needs a historical short-interest data source (e.g. FINRA's
        # free bi-monthly settlement files); not wired in. Live inference in
        # trade_desk is unaffected — it always uses today's real value correctly.
        try:
            info = yf.Ticker(ticker).info or {}
            si = info.get("shortPercentOfFloat")
            df["short_interest_pct"] = float(si) if si is not None else np.nan
        except Exception:
            df["short_interest_pct"] = np.nan

        # ── Ensure all feature columns exist (as NaN, not a fabricated 0) ──
        feature_cols = get_feature_cols()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan

        df["ticker"] = ticker
        return df[["ticker", "date", "high", "low", "close"] + feature_cols].copy()

    except Exception:
        return None


# ── Main builder ───────────────────────────────────────────────────────────────

def build_training_dataset(tickers: list[str], period: str = "7y") -> pd.DataFrame:
    print("Fetching SPY...")
    spy_close = _fetch_close("SPY", period)
    spy_ret_21d = spy_close.pct_change(21) if spy_close is not None else None
    # 63d (~3mo) SPY return for regime labeling: bull > +3%, bear < -3%, neutral in between
    spy_ret_63d = spy_close.pct_change(63) if spy_close is not None else None

    all_dfs = []
    sector_etf_closes = {}
    feature_cols = get_feature_cols()
    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            df = _compute_ticker_features(ticker, period, spy_close, spy_ret_21d, sector_etf_closes)
            if df is None or len(df) < 100:
                print("skip")
                continue

            c = df["close"].values
            h = df["high"].values
            lo = df["low"].values

            # ── Target: did trade hit TP before stop within 21 days? ─────────
            atr_abs = df["atr_pct"].values * c
            n = len(c)
            labels = np.full(n, "NEUTRAL", dtype=object)

            stop = c - 1.5 * atr_abs
            tp = c + 3.0 * atr_abs

            for lag in range(1, LABEL_LOOKAHEAD_DAYS + 1):
                future_high = np.roll(h, -lag).astype(float)
                future_low = np.roll(lo, -lag).astype(float)
                future_high[-lag:] = np.nan
                future_low[-lag:] = np.nan
                undecided = labels == "NEUTRAL"
                labels[undecided & (future_high >= tp)] = "WIN"
                labels[undecided & (future_low <= stop)] = "LOSS"

            labels[-LABEL_LOOKAHEAD_DAYS:] = "NEUTRAL"

            df["label"] = labels

            # Market regime from SPY 63d return at each date
            if spy_ret_63d is not None:
                spy_63 = spy_ret_63d.reindex(df["date"]).values
                df["market_regime"] = np.where(spy_63 > 0.03, "bull",
                                      np.where(spy_63 < -0.03, "bear", "neutral"))
            else:
                df["market_regime"] = "neutral"

            df = df.iloc[60:]
            df = df[df["label"] != "NEUTRAL"]

            if len(df) < 30:
                print("skip (too few)")
                continue

            df["label_binary"] = (df["label"] == "WIN").astype(int)
            all_dfs.append(df[["ticker", "date", "high", "low", "close", "market_regime"] + feature_cols + ["label", "label_binary"]].copy())
            win = (df["label"] == "WIN").mean()
            print(f"{len(df)} rows (WIN:{win:.0%})")

        except Exception as e:
            print(f"error — {e}")

    if not all_dfs:
        raise ValueError("No data collected")

    combined = pd.concat(all_dfs, ignore_index=True).sort_values("date").reset_index(drop=True)
    combined = _impute_missing(combined, get_feature_cols())
    print(f"\nDataset: {len(combined):,} rows, {combined['ticker'].nunique()} tickers")
    print(f"Labels: {(combined['label']=='WIN').mean():.1%} WIN / {(combined['label']=='LOSS').mean():.1%} LOSS")
    return combined


# ── Live/current inference ─────────────────────────────────────────────────────

def build_live_features(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """
    Build feature snapshot for current live scoring across a universe.
    Returns the LAST row per ticker (today's/current bar), same columns as training.

    Args:
        tickers: list of ticker symbols
        period: history period for rolling indicators (default "1y", minimum for 252d max)

    Returns:
        DataFrame with one row per ticker: ticker, date, and all FEATURE_COLS
    """
    print("Fetching SPY for alpha...")
    spy_close = _fetch_close("SPY", period)
    spy_ret_21d = spy_close.pct_change(21) if spy_close is not None else None

    live_rows = []
    sector_etf_closes = {}
    feature_cols = get_feature_cols()

    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            df = _compute_ticker_features(ticker, period, spy_close, spy_ret_21d, sector_etf_closes)
            if df is None or len(df) < 1:
                print("skip")
                continue

            # Take only the last row (today's snapshot)
            last_row = df.iloc[-1:].copy()
            live_rows.append(last_row)
            print(f"✓ {last_row['date'].values[0]}")

        except Exception as e:
            print(f"error — {e}")

    if not live_rows:
        raise ValueError("No data collected for any tickers")

    combined = pd.concat(live_rows, ignore_index=True)
    combined = combined[["ticker", "date"] + feature_cols].reset_index(drop=True)
    combined = _impute_missing(combined, feature_cols)
    print(f"\nLive snapshot: {len(combined)} tickers, date={combined['date'].max()}")
    return combined
