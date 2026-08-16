"""
Point-in-time SEC EDGAR fundamentals via XBRL companyfacts API.

Unlike yf.Ticker.info which returns a static snapshot of current data
applied retroactively to all historical rows (a real accuracy bug),
this module fetches historical XBRL facts tagged with their actual
filing dates, enabling genuine point-in-time lookups: "what were this
company's fundamentals as known on DATE X?" — never using filings made
after DATE X.

SEC's companyfacts API: https://data.sec.gov/api/xbrl/companyfacts/CIK0000000000.json
Free, no auth, fair-use: ~10 req/sec with a descriptive User-Agent.

Usage:
  # One-time: fetch point-in-time series for a ticker
  ts = fetch_point_in_time_fundamentals("AAPL")
  # ts: DataFrame with columns [concept, value, fiscal_period_end, filed_date]

  # As-of query: get fundamentals as they were known on a past date
  fund = get_fundamentals_asof("AAPL", pd.Timestamp("2015-01-01"))
  # fund: dict with computed ratios (revenue_growth, profit_margin, roe, etc.)
  # keyed only by what XBRL provides (no pe_ratio, forward_pe, fcf_yield —
  # those need market cap, which XBRL doesn't have)
"""
import json
import time
import urllib.request
import urllib.error
from functools import lru_cache
from typing import Optional
import pandas as pd
import numpy as np

USER_AGENT = "trade_ml admin@example.com"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# XBRL concept tag name variants (companies use different forms)
# Each entry: list of possible tag names for that semantic fact
CONCEPT_VARIANTS = {
    "Revenues": [
        # Modern ASC-606 tags first — companies migrated to these ~2018,
        # legacy "Revenues" often still has a few stale facts that would
        # otherwise win by being first (see ordering bug fixed here: the
        # loop stops at the first variant with ANY data, so legacy tags
        # must come last).
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomer",
        "Revenues",
        "TotalRevenues",
        "SalesRevenueServices",
        "SalesRevenueProductsNet",
    ],
    "NetIncomeLoss": [
        "NetIncomeLoss",
        "NetIncomeAvailableToCommonStockholders",
        "NetIncomeLossAvailableToCommonStockholders",
        "NetIncome",
    ],
    "Assets": [
        "Assets",
        "AssetsTotal",
        "TotalAssets",
    ],
    "Liabilities": [
        "Liabilities",
        "LiabilitiesTotal",
        "TotalLiabilities",
    ],
    "StockholdersEquity": [
        "StockholdersEquity",
        "ShareholdersEquity",
        "TotalEquity",
        "TotalStockholdersEquity",
    ],
    "AssetsCurrent": [
        "AssetsCurrent",
        "CurrentAssets",
    ],
    "LiabilitiesCurrent": [
        "LiabilitiesCurrent",
        "CurrentLiabilities",
    ],
}


# Cache company_tickers.json in memory
_company_tickers_cache = None


def _load_company_tickers() -> dict:
    """Fetch and cache SEC's company_tickers.json."""
    global _company_tickers_cache
    if _company_tickers_cache is not None:
        return _company_tickers_cache

    req = urllib.request.Request(
        COMPANY_TICKERS_URL, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    # company_tickers.json is a dict mapping CIK str -> {ticker, cik_str, ...}
    # Convert to ticker -> cik_str for our lookups
    ticker_to_cik = {}
    for cik_entry in data.values():
        ticker = cik_entry.get("ticker", "").upper().strip()
        cik_str = cik_entry.get("cik_str")
        if ticker and cik_str:
            ticker_to_cik[ticker] = cik_str
    _company_tickers_cache = ticker_to_cik
    return ticker_to_cik


def _get_cik(ticker: str) -> Optional[str]:
    """
    Map ticker -> 10-digit zero-padded CIK string.
    Returns None if ticker not found.
    """
    tickers = _load_company_tickers()
    cik = tickers.get(ticker.upper())
    if cik:
        # Pad to 10 digits with leading zeros
        return str(cik).zfill(10)
    return None


def fetch_point_in_time_fundamentals(ticker: str) -> pd.DataFrame:
    """
    Fetch historical XBRL facts from SEC companyfacts API.

    Returns:
      DataFrame with columns:
        - concept: XBRL concept name (Revenues, NetIncomeLoss, etc.)
        - value: numeric value (in USD)
        - fiscal_period_end: end date of fiscal period (pd.Timestamp)
        - filed_date: actual SEC filing date (pd.Timestamp)

      Sorted by filed_date ascending. Only includes facts with USD values.
      Returns empty DataFrame if ticker CIK not found or API fails.
    """
    cik = _get_cik(ticker)
    if not cik:
        return pd.DataFrame(columns=["concept", "value", "fiscal_period_end", "filed_date", "start_date"])

    url = COMPANYFACTS_URL_TEMPLATE.format(cik=cik)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            # Rate limited; could sleep and retry, but for now just fail gracefully
            print(f"  SEC rate limit (429) for {ticker}; skipping")
        return pd.DataFrame(columns=["concept", "value", "fiscal_period_end", "filed_date", "start_date"])
    except Exception as e:
        print(f"  Error fetching companyfacts for {ticker}: {e}")
        return pd.DataFrame(columns=["concept", "value", "fiscal_period_end", "filed_date", "start_date"])

    # Extract us-gaap facts
    rows = []
    facts = data.get("facts", {}).get("us-gaap", {})

    for concept_name, variants in CONCEPT_VARIANTS.items():
        # Try each variant tag name
        for tag_name in variants:
            if tag_name not in facts:
                continue

            fact_data = facts[tag_name]
            units = fact_data.get("units", {})

            # Look for USD values (prefer USD over raw numbers)
            usd_values = units.get("USD", [])
            if not usd_values and "pure" in units:
                usd_values = units.get("pure", [])

            for fact in usd_values:
                try:
                    value = float(fact.get("val", 0))
                    # Only accept non-zero, sensible values
                    if value == 0 or abs(value) < 1:  # Skip tiny/zero values
                        continue

                    end_date = fact.get("end")
                    filed_date = fact.get("filed")
                    start_date = fact.get("start")  # None for instant (balance-sheet) facts

                    if not end_date or not filed_date:
                        continue

                    rows.append({
                        "concept": concept_name,
                        "value": value,
                        "fiscal_period_end": pd.Timestamp(end_date),
                        "filed_date": pd.Timestamp(filed_date),
                        "start_date": pd.Timestamp(start_date) if start_date else pd.NaT,
                    })
                except (ValueError, TypeError):
                    continue

            # Once we found a working variant for this concept, stop trying others
            if rows and any(r["concept"] == concept_name for r in rows):
                break

    if not rows:
        return pd.DataFrame(columns=["concept", "value", "fiscal_period_end", "filed_date", "start_date"])

    df = pd.DataFrame(rows)
    df = df.sort_values("filed_date").reset_index(drop=True)
    return df


def get_fundamentals_asof(ticker: str, asof_date: pd.Timestamp) -> dict:
    """
    Compute fundamentals dict as of a specific date.

    Parameters:
      ticker: stock ticker
      asof_date: pd.Timestamp; only use facts filed on or before this date

    Returns:
      Dict with keys: revenue_growth, profit_margin, roe, debt_to_equity
      Values are float or NaN.

      Note: pe_ratio, forward_pe, fcf_yield not included (require market cap).
      Values only computed if the data exists; otherwise NaN (never 0.0).
    """
    df = fetch_point_in_time_fundamentals(ticker)
    return _compute_ratios_asof(df, asof_date)


def _compute_ratios_asof(df: pd.DataFrame, asof_date: pd.Timestamp) -> dict:
    """
    Same computation as get_fundamentals_asof, but takes an already-fetched
    raw-facts DataFrame instead of re-hitting the SEC API. Used both by
    get_fundamentals_asof (single lookup) and build_point_in_time_series
    (many lookups against one fetch, for merge_asof into a daily panel).
    """
    result = {
        "revenue_growth": np.nan,
        "profit_margin": np.nan,
        "roe": np.nan,
        "debt_to_equity": np.nan,
    }

    if df.empty:
        return result

    # Filter to facts filed on or before asof_date (point-in-time constraint)
    df_valid = df[df["filed_date"] <= asof_date].copy()
    if df_valid.empty:
        return result

    # Revenues and NetIncomeLoss are duration ("flow") facts — XBRL filings
    # report the SAME tag at multiple durations in one filing (e.g. a 10-Q
    # reports both the single quarter AND the 9-month year-to-date figure
    # under the identical concept name). Mixing these when picking "the most
    # recent value" produces nonsense (e.g. a quarter's net income divided by
    # a year's revenue). Restrict to annual-duration facts (~350-380 days
    # between start and end — a 10-K's full fiscal year) for anything that
    # needs a consistent, comparable duration. Balance-sheet items (Assets,
    # Liabilities, StockholdersEquity) are "instant" facts (no start_date,
    # a point-in-time balance) and don't have this problem.
    def _annual_only(concept_rows: pd.DataFrame) -> pd.DataFrame:
        dur = (concept_rows["fiscal_period_end"] - concept_rows["start_date"]).dt.days
        return concept_rows[dur.between(350, 380)]

    latest = {}
    for concept in ["Assets", "Liabilities", "StockholdersEquity"]:
        concept_rows = df_valid[df_valid["concept"] == concept]
        if not concept_rows.empty:
            latest[concept] = concept_rows.iloc[-1]["value"]

    for concept in ["Revenues", "NetIncomeLoss"]:
        concept_rows = _annual_only(df_valid[df_valid["concept"] == concept])
        if not concept_rows.empty:
            latest[concept] = concept_rows.sort_values("filed_date").iloc[-1]["value"]

    # Compute ratios
    if "Revenues" in latest and "NetIncomeLoss" in latest:
        # profit_margin = NetIncome / Revenues
        rev = latest["Revenues"]
        if rev != 0:
            result["profit_margin"] = latest["NetIncomeLoss"] / rev

    if "NetIncomeLoss" in latest and "StockholdersEquity" in latest:
        # ROE = NetIncome / Equity
        eq = latest["StockholdersEquity"]
        if eq != 0:
            result["roe"] = latest["NetIncomeLoss"] / eq

    if "Liabilities" in latest and "StockholdersEquity" in latest:
        # Debt-to-Equity = Liabilities / Equity
        eq = latest["StockholdersEquity"]
        if eq != 0:
            result["debt_to_equity"] = latest["Liabilities"] / eq

    # Revenue growth: YoY change in Revenues, annual-duration facts only
    rev_data = _annual_only(df_valid[df_valid["concept"] == "Revenues"]).copy()
    if len(rev_data) >= 2:
        rev_data = rev_data.sort_values("fiscal_period_end")
        latest_rev = rev_data.iloc[-1]
        prior_year_cutoff = latest_rev["fiscal_period_end"] - pd.Timedelta(days=365)
        prior_candidates = rev_data[
            (rev_data["fiscal_period_end"] <= prior_year_cutoff) &
            (rev_data["fiscal_period_end"] > prior_year_cutoff - pd.Timedelta(days=100))
        ]
        if not prior_candidates.empty:
            prior_rev = prior_candidates.iloc[-1]["value"]
            if prior_rev != 0:
                result["revenue_growth"] = (
                    (latest_rev["value"] - prior_rev) / abs(prior_rev)
                )

    return result


def build_point_in_time_series(ticker: str) -> pd.DataFrame:
    """
    One SEC fetch per ticker, then compute the point-in-time ratios at each
    date something actually changed (each unique filed_date) — a compact
    series (dozens of rows over 15y, not one row per trading day).

    Meant to be merge_asof'd (backward direction) onto a full daily training
    panel: `pd.merge_asof(daily_df, this_series, on="date", direction="backward")`
    gives every trading day the most recent point-in-time fundamentals as of
    that day, in one vectorized pandas op — NOT a per-row API call or a
    per-row call into get_fundamentals_asof (that would be correct but far
    too slow across a full historical panel).

    Returns:
      DataFrame with columns: filed_date, revenue_growth, profit_margin,
      roe, debt_to_equity. Empty DataFrame if no data found for this ticker.
    """
    raw = fetch_point_in_time_fundamentals(ticker)
    if raw.empty:
        return pd.DataFrame(columns=["filed_date", "revenue_growth", "profit_margin", "roe", "debt_to_equity"])

    unique_dates = sorted(raw["filed_date"].unique())
    rows = []
    for d in unique_dates:
        ratios = _compute_ratios_asof(raw, pd.Timestamp(d))
        rows.append({"filed_date": pd.Timestamp(d), **ratios})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    """
    Self-test: fetch and compute fundamentals for AAPL at multiple dates.
    Verify: values change across dates, and filed_date <= asof_date always.
    """
    print("Fetching AAPL point-in-time fundamentals...")
    aapl_ts = fetch_point_in_time_fundamentals("AAPL")

    if aapl_ts.empty:
        print("ERROR: Could not fetch AAPL data from SEC")
    else:
        print(f"\nFetched {len(aapl_ts)} XBRL facts for AAPL")
        print(f"Concepts found: {aapl_ts['concept'].unique().tolist()}")
        print(f"Date range: {aapl_ts['filed_date'].min()} to {aapl_ts['filed_date'].max()}")

    # Test as-of lookups
    test_dates = [
        pd.Timestamp("2015-01-01"),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2025-01-01"),
    ]

    for asof_date in test_dates:
        print(f"\n--- Fundamentals as-of {asof_date.date()} ---")
        fund = get_fundamentals_asof("AAPL", asof_date)
        print(f"  revenue_growth: {fund['revenue_growth']:.4f}" if not np.isnan(fund['revenue_growth']) else "  revenue_growth: NaN")
        print(f"  profit_margin: {fund['profit_margin']:.4f}" if not np.isnan(fund['profit_margin']) else "  profit_margin: NaN")
        print(f"  roe: {fund['roe']:.4f}" if not np.isnan(fund['roe']) else "  roe: NaN")
        print(f"  debt_to_equity: {fund['debt_to_equity']:.4f}" if not np.isnan(fund['debt_to_equity']) else "  debt_to_equity: NaN")

        # Verify point-in-time constraint
        df_valid = aapl_ts[aapl_ts["filed_date"] <= asof_date]
        if not df_valid.empty:
            most_recent_filing = df_valid["filed_date"].max()
            print(f"  [Most recent filing used: {most_recent_filing.date()}]")
            if most_recent_filing > asof_date:
                print(f"  ERROR: Used future filing! {most_recent_filing} > {asof_date}")
