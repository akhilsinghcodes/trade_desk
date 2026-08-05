import yfinance as yf
from statistics import median
from typing import Optional

# Sector → list of representative peer tickers
SECTOR_PEERS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "ORCL", "CRM", "ADBE"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT"],
    "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "CL"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD"],
    "Industrials": ["HON", "UPS", "CAT", "DE", "BA", "MMM", "GE"],
    "Utilities": ["NEE", "DUK", "SO", "D", "EXC", "AEP"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG", "PSA"],
    "Basic Materials": ["LIN", "APD", "ECL", "SHW", "NEM", "FCX"],
}


def get_sector_comparison(ticker: str, sector: str) -> dict:
    """
    Compare ticker's P/E and forward P/E against sector peers.

    Returns:
    {
        "sector": str,
        "ticker_pe": float or None,
        "ticker_forward_pe": float or None,
        "sector_median_pe": float or None,
        "sector_median_forward_pe": float or None,
        "pe_vs_sector": "cheap"/"fair"/"expensive"/"unknown",
        "peers_used": int,
        "signal": (status, label, text)   # ("good"/"neutral"/"warning", "Sector Valuation", plain english)
    }
    """
    result = {
        "sector": sector,
        "ticker_pe": None,
        "ticker_forward_pe": None,
        "sector_median_pe": None,
        "sector_median_forward_pe": None,
        "pe_vs_sector": "unknown",
        "peers_used": 0,
        "signal": ("neutral", "Sector Valuation", "Insufficient data for sector comparison."),
    }

    # Get ticker's P/E ratios
    try:
        ticker_info = yf.Ticker(ticker).info
        ticker_pe = ticker_info.get("trailingPE")
        ticker_forward_pe = ticker_info.get("forwardPE")

        # Only accept positive values
        if ticker_pe and ticker_pe > 0:
            result["ticker_pe"] = ticker_pe
        if ticker_forward_pe and ticker_forward_pe > 0:
            result["ticker_forward_pe"] = ticker_forward_pe
    except Exception:
        pass  # If we can't get ticker info, continue with peer comparison

    # Get peer P/E ratios
    if sector not in SECTOR_PEERS:
        return result

    peers = SECTOR_PEERS[sector]
    peer_pes = []
    peer_forward_pes = []

    for peer in peers:
        # Skip the ticker itself
        if peer.upper() == ticker.upper():
            continue

        try:
            peer_info = yf.Ticker(peer).info
            peer_pe = peer_info.get("trailingPE")
            peer_forward_pe = peer_info.get("forwardPE")

            if peer_pe and peer_pe > 0:
                peer_pes.append(peer_pe)
            if peer_forward_pe and peer_forward_pe > 0:
                peer_forward_pes.append(peer_forward_pe)
        except Exception:
            pass  # Skip peers where fetch fails

    # Need at least 3 peers for valid comparison
    if len(peer_pes) < 3:
        return result

    result["peers_used"] = len(peer_pes)
    result["sector_median_pe"] = median(peer_pes)
    if peer_forward_pes:
        result["sector_median_forward_pe"] = median(peer_forward_pes)

    # Compare ticker P/E to sector median
    if result["ticker_pe"] is not None:
        sector_median_pe = result["sector_median_pe"]
        ticker_pe = result["ticker_pe"]

        if ticker_pe < sector_median_pe * 0.85:
            result["pe_vs_sector"] = "cheap"
            result["signal"] = (
                "good",
                "Sector Valuation",
                f"Trading at a discount vs sector peers (P/E {ticker_pe:.1f} vs sector median {sector_median_pe:.1f}). Potentially undervalued."
            )
        elif ticker_pe > sector_median_pe * 1.15:
            result["pe_vs_sector"] = "expensive"
            result["signal"] = (
                "warning",
                "Sector Valuation",
                f"Trading at a premium vs sector peers (P/E {ticker_pe:.1f} vs sector median {sector_median_pe:.1f}). May be overvalued."
            )
        else:
            result["pe_vs_sector"] = "fair"
            result["signal"] = (
                "neutral",
                "Sector Valuation",
                f"Trading in line with sector peers (P/E {ticker_pe:.1f} vs sector median {sector_median_pe:.1f})."
            )

    return result
