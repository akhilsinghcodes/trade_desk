"""Fetch news via yfinance + run FinBERT sentiment locally."""
import yfinance as yf
from datetime import datetime


def _is_relevant(article: dict, ticker: str, company: str = "") -> bool:
    """Return True if the article is likely about this specific ticker."""
    text = (article.get("title", "") + " " + article.get("summary", "")).lower()
    ticker_lower = ticker.lower()
    company_lower = company.lower() if company else ""
    # Check for ticker or first word of company name
    company_word = company_lower.split()[0] if company_lower else ""
    return (ticker_lower in text or
            (company_word and len(company_word) > 3 and company_word in text))


def get_news(ticker: str, limit: int = 10, company: str = "") -> list[dict]:
    t = yf.Ticker(ticker)
    raw = t.news or []
    articles = []
    all_articles = []
    for item in raw:
        content = item.get("content", {})
        title = content.get("title", "No title")
        summary = content.get("summary", "")
        pub = content.get("pubDate", "")
        provider = content.get("provider", {}).get("displayName", "Unknown")
        url = ""
        click_through = content.get("clickThroughUrl", {})
        if isinstance(click_through, dict):
            url = click_through.get("url", "")

        all_articles.append({
            "title": title,
            "summary": summary,
            "published": pub,
            "source": provider,
            "url": url,
        })

    # Prefer articles relevant to this ticker; fall back to all if too few
    relevant = [a for a in all_articles if _is_relevant(a, ticker, company)]
    result = relevant if len(relevant) >= 3 else all_articles
    return result[:limit]


def analyze_sentiment(articles: list[dict]) -> list[dict]:
    """Run FinBERT locally on article titles. Returns articles with sentiment added."""
    if not articles:
        return articles

    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=-1,  # CPU only — stays local
            truncation=True,
            max_length=512,
        )
        titles = [a["title"] for a in articles]
        results = pipe(titles)
        for article, result in zip(articles, results):
            label = result["label"]  # positive / negative / neutral
            score = result["score"]
            article["sentiment"] = label
            article["sentiment_score"] = score
    except Exception as e:
        for article in articles:
            article["sentiment"] = "unknown"
            article["sentiment_score"] = 0.0
            article["sentiment_error"] = str(e)

    return articles


def overall_sentiment(articles: list[dict]) -> tuple[str, str]:
    """Returns (label, color) based on majority sentiment."""
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for a in articles:
        s = a.get("sentiment", "neutral").lower()
        if s in counts:
            counts[s] += 1

    total = sum(counts.values())
    if total == 0:
        return "No data", "gray"

    if counts["positive"] > counts["negative"] and counts["positive"] > counts["neutral"]:
        return f"Mostly Positive ({counts['positive']}/{total} headlines)", "green"
    elif counts["negative"] > counts["positive"] and counts["negative"] > counts["neutral"]:
        return f"Mostly Negative ({counts['negative']}/{total} headlines)", "red"
    else:
        return f"Mixed / Neutral ({counts['neutral']}/{total} neutral)", "orange"
