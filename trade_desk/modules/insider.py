"""Insider transactions from yfinance."""
import yfinance as yf


def get_insider_transactions(ticker: str) -> dict:
    """
    Returns dict with:
    - transactions: list of recent insider trades (dicts)
    - buy_count: int
    - sell_count: int
    - net_sentiment: "bullish" / "bearish" / "neutral"
    """
    t = yf.Ticker(ticker)
    result = {
        "transactions": [],
        "buy_count": 0,
        "sell_count": 0,
        "net_sentiment": "neutral",
    }

    try:
        insiders = t.insider_transactions
        if insiders is None or insiders.empty:
            return result

        # Normalize columns to lowercase
        insiders.columns = [c.lower().replace(" ", "_") for c in insiders.columns]

        recent = insiders.head(10)
        transactions = []
        for _, row in recent.iterrows():
            try:
                text = str(row.get("text", row.get("transaction", ""))).lower()
                shares = row.get("shares", 0) or 0
                insider = row.get("insider", row.get("filer_name", "Unknown"))
                position = row.get("position", row.get("filer_relation", ""))

                is_buy = any(w in text for w in ["purchase", "buy", "acquisition", "acquired"])
                is_sell = any(w in text for w in ["sale", "sell", "disposed", "exercise"])

                transactions.append({
                    "insider": str(insider),
                    "position": str(position),
                    "action": "BUY" if is_buy else "SELL" if is_sell else "OTHER",
                    "shares": int(shares) if shares else 0,
                    "text": str(row.get("text", "")),
                })

                if is_buy:
                    result["buy_count"] += 1
                elif is_sell:
                    result["sell_count"] += 1
            except Exception:
                continue

        result["transactions"] = transactions

        buys = result["buy_count"]
        sells = result["sell_count"]
        if buys > sells and buys >= 2:
            result["net_sentiment"] = "bullish"
        elif sells > buys * 2:
            result["net_sentiment"] = "bearish"
        else:
            result["net_sentiment"] = "neutral"

    except Exception:
        pass

    return result


def score_insider(data: dict) -> tuple[str, str, str]:
    """Returns (label, status, text) for signals list."""
    buys = data.get("buy_count", 0)
    sells = data.get("sell_count", 0)
    sentiment = data.get("net_sentiment", "neutral")
    total = buys + sells

    if total == 0:
        return ("Insider", "neutral", "No recent insider transaction data.")

    if sentiment == "bullish":
        status = "good"
        text = f"Insiders buying — {buys} purchases vs {sells} sales recently. Positive signal."
    elif sentiment == "bearish":
        status = "warning"
        text = f"Insiders selling — {sells} sales vs {buys} purchases recently. Caution."
    else:
        status = "neutral"
        text = f"Mixed insider activity — {buys} buys, {sells} sells. No clear signal."

    return ("Insider", status, text)
