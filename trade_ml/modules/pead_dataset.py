"""
PEAD (Post-Earnings Announcement Drift) dataset builder.

One row per earnings event per ticker. Target: did stock outperform SPY
by >3% in 30 days after earnings? Documented effect since 1968.

Features:
  - surprise_pct: magnitude of EPS beat/miss
  - prev_surprise_pct: last quarter's surprise (momentum)
  - surprise_acceleration: this quarter minus last quarter surprise
  - pre_ret_1m: stock return 1 month before earnings vs SPY
  - pre_ret_3m: stock return 3 months before earnings vs SPY
  - atr_pct: volatility at earnings date
  - high_52w_proximity: where stock sits in 52w range
  - revenue_growth, profit_margin, roe, fcf_yield: fundamental quality
  - piotroski_score: financial health

Target:
  WIN  if stock outperforms SPY by >3% in 30 days after earnings
  LOSS if stock underperforms SPY by >3% in 30 days after earnings
  NEUTRAL excluded
"""
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from modules.indicators import add_common_indicators

warnings.filterwarnings("ignore")

PEAD_FEATURES = [
    "surprise_pct",
    "prev_surprise_pct",
    "surprise_acceleration",
    "pre_ret_1m_vs_spy",
    "pre_ret_3m_vs_spy",
    "atr_pct",
    "high_52w_proximity",
    "revenue_growth",
    "profit_margin",
    "roe",
    "fcf_yield",
    "piotroski_score",
]


def _get_spy_prices(period: str = "15y") -> pd.Series:
    df = yf.download("SPY", period=period, progress=False, auto_adjust=True)
    s = df["Close"].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def _get_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
        t = yf.Ticker(ticker)

        rg_v = info.get("revenueGrowth")
        rg = float(rg_v) if rg_v is not None else np.nan
        pm_v = info.get("profitMargins")
        pm = float(pm_v) if pm_v is not None else np.nan
        roe_v = info.get("returnOnEquity")
        roe = float(roe_v) if roe_v is not None else np.nan
        fcf = info.get("freeCashflow")
        mktcap = info.get("marketCap")
        fcf_yield = float(np.clip(fcf / mktcap, -0.5, 0.5)) if fcf and mktcap else np.nan

        # Piotroski (reuse logic from ml_dataset)
        from modules.ml_dataset import _piotroski
        piotroski = _piotroski(t) / 9.0

        return {
            "revenue_growth": rg,
            "profit_margin": pm,
            "roe": roe,
            "fcf_yield": fcf_yield,
            "piotroski_score": piotroski,
        }
    except Exception:
        return {k: 0.0 for k in ["revenue_growth", "profit_margin", "roe", "fcf_yield", "piotroski_score"]}


def build_pead_dataset(tickers: list[str]) -> pd.DataFrame:
    print("Fetching SPY prices...")
    spy = _get_spy_prices()

    all_rows = []
    for ticker in tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            t = yf.Ticker(ticker)

            # Earnings dates with surprise data
            earnings = t.get_earnings_dates(limit=60)
            if earnings is None or earnings.empty:
                print("no earnings")
                continue

            earnings = earnings.dropna(subset=["Reported EPS", "EPS Estimate"])
            earnings = earnings[earnings["Surprise(%)"].notna()]
            earnings.index = pd.to_datetime(earnings.index).tz_localize(None)
            earnings = earnings.sort_index()  # ascending

            if len(earnings) < 4:
                print("too few quarters")
                continue

            # Price history
            raw = yf.download(ticker, period="15y", progress=False, auto_adjust=True)
            if raw.empty or len(raw) < 100:
                print("no price data")
                continue

            price = raw["Close"].squeeze()
            price.index = pd.to_datetime(price.index).tz_localize(None)

            raw_df = raw.reset_index()
            raw_df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw_df.columns]
            raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.tz_localize(None)
            raw_df = raw_df.sort_values("date").reset_index(drop=True)
            raw_df = add_common_indicators(raw_df)

            c = raw_df["close"].values
            h = raw_df["high"].values
            lo = raw_df["low"].values
            dates = pd.DatetimeIndex(raw_df["date"].values)

            # ATR series
            tr = np.maximum(h - lo, np.maximum(
                np.abs(h - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
            tr[0] = h[0] - lo[0]
            atr_series = pd.Series(
                pd.Series(tr).rolling(14, min_periods=1).mean().values / (c + 1e-8),
                index=dates
            )
            high_52w = pd.Series(c, index=dates).rolling(252, min_periods=50).max()

            # Fundamentals (static snapshot)
            fund = _get_fundamentals(ticker)

            # Build one row per earnings event
            surprise_list = earnings["Surprise(%)"].values / 100.0  # normalize to decimal
            ticker_rows = 0

            for i, (earn_date, row) in enumerate(earnings.iterrows()):
                surprise = float(row["Surprise(%)"] / 100.0)
                prev_surprise = float(surprise_list[i - 1]) if i > 0 else 0.0
                surprise_accel = surprise - prev_surprise

                # Find nearest trading day on or after earnings date
                future_dates = dates[dates >= earn_date]
                if len(future_dates) < 31:
                    continue
                entry_date = future_dates[0]   # day after announcement
                exit_date = future_dates[30]   # 30 trading days later

                entry_idx = np.searchsorted(dates, entry_date)
                exit_idx = np.searchsorted(dates, exit_date)
                if exit_idx >= len(dates):
                    continue

                entry_price = c[entry_idx]
                exit_price = c[exit_idx]
                stock_ret = (exit_price - entry_price) / (entry_price + 1e-8)

                # SPY return over same window
                spy_entry = spy.asof(entry_date)
                spy_exit = spy.asof(exit_date)
                spy_ret = (spy_exit - spy_entry) / (spy_entry + 1e-8) if spy_entry and spy_exit else 0.0

                relative_ret = stock_ret - spy_ret

                # Target
                if relative_ret > 0.03:
                    label = "WIN"
                elif relative_ret < -0.03:
                    label = "LOSS"
                else:
                    continue  # NEUTRAL excluded

                # Pre-earnings returns vs SPY
                past_dates = dates[dates < earn_date]
                if len(past_dates) < 63:
                    continue

                pre_1m_idx = np.searchsorted(dates, past_dates[-21]) if len(past_dates) >= 21 else entry_idx
                pre_3m_idx = np.searchsorted(dates, past_dates[-63]) if len(past_dates) >= 63 else entry_idx

                pre_ret_1m = (entry_price / c[pre_1m_idx] - 1) if pre_1m_idx < entry_idx else 0.0
                pre_ret_3m = (entry_price / c[pre_3m_idx] - 1) if pre_3m_idx < entry_idx else 0.0

                spy_pre_1m = spy.asof(past_dates[-21]) if len(past_dates) >= 21 else spy_entry
                spy_pre_3m = spy.asof(past_dates[-63]) if len(past_dates) >= 63 else spy_entry

                pre_ret_1m_vs_spy = pre_ret_1m - float((spy_entry - spy_pre_1m) / (spy_pre_1m + 1e-8))
                pre_ret_3m_vs_spy = pre_ret_3m - float((spy_entry - spy_pre_3m) / (spy_pre_3m + 1e-8))

                atr = float(atr_series.asof(earn_date)) if not pd.isna(atr_series.asof(earn_date)) else 0.02
                h52 = float(high_52w.asof(earn_date)) if not pd.isna(high_52w.asof(earn_date)) else entry_price
                h52_prox = float(np.clip(entry_price / (h52 + 1e-8), 0, 1))

                all_rows.append({
                    "ticker": ticker,
                    "earnings_date": earn_date,
                    "entry_date": entry_date,
                    "surprise_pct": float(np.clip(surprise, -0.5, 0.5)),
                    "prev_surprise_pct": float(np.clip(prev_surprise, -0.5, 0.5)),
                    "surprise_acceleration": float(np.clip(surprise_accel, -0.5, 0.5)),
                    "pre_ret_1m_vs_spy": float(np.clip(pre_ret_1m_vs_spy, -0.5, 0.5)),
                    "pre_ret_3m_vs_spy": float(np.clip(pre_ret_3m_vs_spy, -0.5, 0.5)),
                    "atr_pct": float(np.clip(atr, 0, 0.2)),
                    "high_52w_proximity": h52_prox,
                    "relative_ret_30d": float(relative_ret),
                    "label": label,
                    "label_binary": 1 if label == "WIN" else 0,
                    **fund,
                })
                ticker_rows += 1

            print(f"{ticker_rows} events")

        except Exception as e:
            print(f"error — {e}")

    if not all_rows:
        raise ValueError("No earnings events collected")

    df = pd.DataFrame(all_rows).sort_values("earnings_date").reset_index(drop=True)
    # Missing fundamentals -> global median, not a fabricated 0 (same class of
    # bug already fixed in ml_dataset.py — a missing ROE isn't "0% ROE").
    fund_cols = ["revenue_growth", "profit_margin", "roe", "fcf_yield", "piotroski_score"]
    df[fund_cols] = df[fund_cols].fillna(df[fund_cols].median())
    print(f"\nPEAD Dataset: {len(df):,} events, {df['ticker'].nunique()} tickers")
    print(f"Labels: {(df['label']=='WIN').mean():.1%} WIN / {(df['label']=='LOSS').mean():.1%} LOSS")
    return df
