import yfinance as yf
from datetime import datetime, timedelta


def get_next_earnings(ticker: str) -> dict:
    """
    Fetch the next earnings date for a stock using yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')

    Returns:
        dict with keys:
            - date: earnings date as string (YYYY-MM-DD) or None
            - days_away: int days until earnings or None
            - warning: bool True if earnings within 14 days
            - status: "warning" / "neutral" / "good"
            - label: "Earnings"
            - text: plain English explanation
    """
    try:
        # Fetch ticker calendar
        ticker_obj = yf.Ticker(ticker)
        calendar = ticker_obj.calendar

        if not calendar or "Earnings Date" not in calendar:
            return {
                "date": None,
                "days_away": None,
                "warning": False,
                "status": "neutral",
                "label": "Earnings",
                "text": "Earnings date unknown."
            }

        earnings_date_raw = calendar["Earnings Date"]

        # Handle both list and single value cases
        if isinstance(earnings_date_raw, list):
            if not earnings_date_raw:
                return {
                    "date": None,
                    "days_away": None,
                    "warning": False,
                    "status": "neutral",
                    "label": "Earnings",
                    "text": "Earnings date unknown."
                }
            earnings_date_raw = earnings_date_raw[0]

        # Convert Timestamp to datetime if needed
        if hasattr(earnings_date_raw, 'date'):
            earnings_dt = earnings_date_raw
        else:
            earnings_dt = earnings_date_raw

        # Ensure we have a datetime object
        if isinstance(earnings_dt, str):
            earnings_dt = datetime.strptime(earnings_dt, "%Y-%m-%d")
        elif not isinstance(earnings_dt, datetime):
            # Try to convert pandas Timestamp
            earnings_dt = datetime.fromisoformat(str(earnings_dt))

        # Get today's date
        today = datetime.now()

        # Calculate days away (only count from today forward)
        days_away = (earnings_dt.date() - today.date()).days

        # Format date as string
        date_str = earnings_dt.strftime("%Y-%m-%d")

        # Determine status and warning
        if days_away < 0:
            # Earnings already passed
            return {
                "date": date_str,
                "days_away": days_away,
                "warning": False,
                "status": "neutral",
                "label": "Earnings",
                "text": f"Last earnings were {abs(days_away)} days ago."
            }

        # Earnings are in the future
        warning = days_away <= 14

        if days_away <= 3:
            status = "warning"
            text = f"⚠️ Earnings in {days_away} days — expect big price swings. High risk to hold through this."
        elif days_away <= 7:
            status = "warning"
            text = f"Earnings in {days_away} days — volatility expected soon. Consider your position size."
        elif days_away <= 14:
            status = "warning"
            text = f"Earnings in {days_away} days — watch for guidance updates."
        else:
            status = "good"
            text = f"Next earnings in {days_away} days — no immediate earnings risk."

        return {
            "date": date_str,
            "days_away": days_away,
            "warning": warning,
            "status": status,
            "label": "Earnings",
            "text": text
        }

    except Exception as e:
        # Handle all exceptions gracefully
        return {
            "date": None,
            "days_away": None,
            "warning": False,
            "status": "neutral",
            "label": "Earnings",
            "text": "Earnings date unavailable."
        }
