"""Balance sheet YoY trends from yfinance — debt direction and financial health."""
import yfinance as yf
import pandas as pd


def get_balance_sheet_trends(ticker: str) -> dict:
    """
    Returns dict:
    - years: list of str (most recent first)
    - total_debt: list of float (billions)
    - total_assets: list of float (billions)
    - cash: list of float (billions)
    - net_debt: list of float (billions, debt minus cash)
    - debt_trend: str ("increasing" / "decreasing" / "stable" / "unknown")
    - debt_to_assets: list of float (ratio)
    - revenue: list of float (billions, from income stmt)
    - revenue_trend: str
    """
    result = {
        "years": [],
        "total_debt": [],
        "total_assets": [],
        "cash": [],
        "net_debt": [],
        "debt_trend": "unknown",
        "debt_to_assets": [],
        "revenue": [],
        "revenue_trend": "unknown",
    }

    try:
        t = yf.Ticker(ticker)
        bs = t.balance_sheet  # columns = dates (most recent first), rows = line items

        if bs is None or bs.empty:
            return result

        # Normalize index
        bs.index = bs.index.str.lower().str.replace(" ", "_")

        dates = [str(c)[:4] for c in bs.columns[:4]]  # last 4 years
        result["years"] = dates

        def _get_row(df, *names):
            for n in names:
                n_lower = n.lower().replace(" ", "_")
                matches = [i for i in df.index if n_lower in i]
                if matches:
                    return df.loc[matches[0]].iloc[:4].tolist()
            return [None] * 4

        def _billions(vals):
            out = []
            for v in vals:
                try:
                    out.append(round(float(v) / 1e9, 2))
                except Exception:
                    out.append(None)
            return out

        total_debt_raw = _get_row(bs, "Total Debt", "Long Term Debt", "total_debt")
        assets_raw = _get_row(bs, "Total Assets", "total_assets")
        cash_raw = _get_row(bs, "Cash And Cash Equivalents", "Cash", "cash_and_cash_equivalents")

        result["total_debt"] = _billions(total_debt_raw)
        result["total_assets"] = _billions(assets_raw)
        result["cash"] = _billions(cash_raw)

        # Net debt = total debt - cash
        net_debt = []
        for d, c in zip(result["total_debt"], result["cash"]):
            if d is not None and c is not None:
                net_debt.append(round(d - c, 2))
            else:
                net_debt.append(None)
        result["net_debt"] = net_debt

        # Debt to assets ratio
        for d, a in zip(result["total_debt"], result["total_assets"]):
            if d is not None and a and a > 0:
                result["debt_to_assets"].append(round(d / a, 3))
            else:
                result["debt_to_assets"].append(None)

        # Debt trend (compare most recent vs oldest)
        valid_debt = [x for x in result["total_debt"] if x is not None]
        if len(valid_debt) >= 2:
            change = valid_debt[0] - valid_debt[-1]  # recent - oldest
            pct = change / abs(valid_debt[-1]) * 100 if valid_debt[-1] else 0
            if pct > 10:
                result["debt_trend"] = "increasing"
            elif pct < -10:
                result["debt_trend"] = "decreasing"
            else:
                result["debt_trend"] = "stable"

    except Exception:
        pass

    # Revenue trend from income statement
    try:
        t2 = yf.Ticker(ticker)
        inc = t2.income_stmt
        if inc is not None and not inc.empty:
            inc.index = inc.index.str.lower().str.replace(" ", "_")
            rev_matches = [i for i in inc.index if "total_revenue" in i or "revenue" == i]
            if rev_matches:
                rev_vals = inc.loc[rev_matches[0]].iloc[:4].tolist()
                result["revenue"] = []
                for v in rev_vals:
                    try:
                        result["revenue"].append(round(float(v) / 1e9, 2))
                    except Exception:
                        result["revenue"].append(None)

                valid_rev = [x for x in result["revenue"] if x is not None]
                if len(valid_rev) >= 2:
                    rev_change_pct = (valid_rev[0] - valid_rev[-1]) / abs(valid_rev[-1]) * 100
                    if rev_change_pct > 10:
                        result["revenue_trend"] = "growing"
                    elif rev_change_pct < -10:
                        result["revenue_trend"] = "shrinking"
                    else:
                        result["revenue_trend"] = "stable"
    except Exception:
        pass

    return result


def score_balance_sheet(data: dict) -> tuple[str, str, str]:
    """Returns (label, status, text) for signals list."""
    debt_trend = data.get("debt_trend", "unknown")
    rev_trend = data.get("revenue_trend", "unknown")
    net_debt = data.get("net_debt", [])
    current_net_debt = net_debt[0] if net_debt else None

    if debt_trend == "unknown" and rev_trend == "unknown":
        return ("Balance Sheet", "neutral", "No balance sheet data available.")

    parts = []
    status = "neutral"

    if debt_trend == "decreasing":
        parts.append("Debt decreasing YoY — improving financial health.")
        status = "good"
    elif debt_trend == "increasing":
        parts.append("Debt increasing YoY — monitor leverage carefully.")
        if status != "good":
            status = "warning"
    elif debt_trend == "stable":
        parts.append("Debt stable YoY.")

    if rev_trend == "growing":
        parts.append("Revenue growing.")
        if status == "neutral":
            status = "good"
    elif rev_trend == "shrinking":
        parts.append("Revenue shrinking — concerning.")
        status = "warning"

    if current_net_debt is not None:
        net_str = f"${abs(current_net_debt):.1f}B net {'debt' if current_net_debt > 0 else 'cash'}."
        parts.append(net_str)

    text = " ".join(parts) if parts else "Balance sheet data available but no clear trend."
    return ("Balance Sheet", status, text)
