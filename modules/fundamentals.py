"""Parse yfinance fundamentals into plain-English metrics."""
import yfinance as yf


HEALTH_THRESHOLDS = {
    "pe":        {"good": (0, 25),   "warn": (25, 40)},
    "debt":      {"good": (0, 1.0),  "warn": (1.0, 2.0)},
    "roe":       {"good": (0.15, 99),"warn": (0.05, 0.15)},
    "margin":    {"good": (0.15, 99),"warn": (0.05, 0.15)},
    "current":   {"good": (1.5, 99), "warn": (1.0, 1.5)},
}


def get_fundamentals(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info

    def safe(key, default=None):
        return info.get(key, default)

    revenue = safe("totalRevenue")
    net_income = safe("netIncomeToCommon")
    total_debt = safe("totalDebt")
    total_equity = safe("totalStockholderEquity")
    current_assets = safe("totalCurrentAssets")
    current_liabilities = safe("totalCurrentLiabilities")

    metrics = {
        "pe_ratio": safe("trailingPE"),
        "forward_pe": safe("forwardPE"),
        "eps": safe("trailingEps"),
        "revenue": revenue,
        "revenue_growth": safe("revenueGrowth"),
        "earnings_growth": safe("earningsGrowth"),
        "profit_margin": safe("profitMargins"),
        "roe": safe("returnOnEquity"),
        "debt_to_equity": safe("debtToEquity"),
        "current_ratio": (current_assets / current_liabilities) if current_assets and current_liabilities else None,
        "free_cash_flow": safe("freeCashflow"),
        "dividend_yield": safe("dividendYield"),
        "beta": safe("beta"),
        "short_ratio": safe("shortRatio"),
    }
    return metrics


def score_fundamentals(metrics: dict) -> list:
    """Return list of (label, status, plain_english_text) tuples."""
    signals = []

    pe = metrics.get("pe_ratio")
    if pe:
        if pe < 0:
            signals.append(("P/E Ratio", "warning", f"P/E is negative ({pe:.1f}) — company is losing money."))
        elif pe < 15:
            signals.append(("P/E Ratio", "good", f"P/E is {pe:.1f} — cheap relative to earnings. Potentially undervalued."))
        elif pe < 30:
            signals.append(("P/E Ratio", "neutral", f"P/E is {pe:.1f} — fairly valued."))
        else:
            signals.append(("P/E Ratio", "warning", f"P/E is {pe:.1f} — expensive. Market expects high growth to justify price."))

    growth = metrics.get("revenue_growth")
    if growth is not None:
        pct = growth * 100
        if pct > 15:
            signals.append(("Revenue Growth", "good", f"Revenue growing at {pct:.1f}% — strong growth."))
        elif pct > 0:
            signals.append(("Revenue Growth", "neutral", f"Revenue growing at {pct:.1f}% — slow but positive."))
        else:
            signals.append(("Revenue Growth", "warning", f"Revenue shrinking {pct:.1f}% — declining business."))

    margin = metrics.get("profit_margin")
    if margin is not None:
        pct = margin * 100
        if pct > 15:
            signals.append(("Profit Margin", "good", f"Keeps {pct:.1f}¢ of every $1 in revenue — healthy margins."))
        elif pct > 5:
            signals.append(("Profit Margin", "neutral", f"Keeps {pct:.1f}¢ of every $1 — thin but positive margins."))
        else:
            signals.append(("Profit Margin", "warning", f"Keeps only {pct:.1f}¢ of every $1 — very thin or negative margins."))

    de = metrics.get("debt_to_equity")
    if de is not None:
        de_ratio = de / 100
        if de_ratio < 0.5:
            signals.append(("Debt", "good", f"Debt/equity {de_ratio:.2f} — low debt, financially stable."))
        elif de_ratio < 1.5:
            signals.append(("Debt", "neutral", f"Debt/equity {de_ratio:.2f} — moderate debt, manageable."))
        else:
            signals.append(("Debt", "warning", f"Debt/equity {de_ratio:.2f} — high debt load. Risk if earnings drop."))

    cr = metrics.get("current_ratio")
    if cr is not None:
        if cr > 1.5:
            signals.append(("Liquidity", "good", f"Current ratio {cr:.2f} — can comfortably pay short-term bills."))
        elif cr > 1.0:
            signals.append(("Liquidity", "neutral", f"Current ratio {cr:.2f} — just enough to cover short-term bills."))
        else:
            signals.append(("Liquidity", "warning", f"Current ratio {cr:.2f} — may struggle to pay short-term bills."))

    beta = metrics.get("beta")
    if beta is not None:
        if beta > 1.5:
            signals.append(("Volatility (Beta)", "warning", f"Beta {beta:.2f} — moves {beta:.1f}x the market. High risk/reward."))
        elif beta > 0.8:
            signals.append(("Volatility (Beta)", "neutral", f"Beta {beta:.2f} — moves roughly with the market."))
        else:
            signals.append(("Volatility (Beta)", "good", f"Beta {beta:.2f} — less volatile than the market. Defensive stock."))

    short = metrics.get("short_ratio")
    if short is not None:
        if short > 10:
            signals.append(("Short Interest", "warning", f"Short ratio {short:.1f} days — heavy short selling. Many betting price falls."))
        elif short > 5:
            signals.append(("Short Interest", "neutral", f"Short ratio {short:.1f} days — moderate short interest."))
        else:
            signals.append(("Short Interest", "good", f"Short ratio {short:.1f} days — low short interest. Few betting against it."))

    return signals
