import json
from pathlib import Path
from datetime import datetime


def _get_alerts_path() -> Path:
    """Get the path to the alerts JSON file."""
    return Path(__file__).parent.parent / "data" / "alerts.json"


def load_alerts() -> list[dict]:
    """Load all alerts from JSON file."""
    alerts_path = _get_alerts_path()
    if not alerts_path.exists():
        return []

    try:
        with open(alerts_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_alerts(alerts: list[dict]):
    """Save alerts list to JSON file."""
    alerts_path = _get_alerts_path()
    alerts_path.parent.mkdir(parents=True, exist_ok=True)

    with open(alerts_path, 'w') as f:
        json.dump(alerts, f, indent=2)


def add_alert(ticker: str, target_price: float, direction: str, note: str = "") -> dict:
    """Create and save a new alert. Returns the alert dict."""
    if direction not in ("above", "below"):
        raise ValueError("Direction must be 'above' or 'below'")

    alert = {
        "ticker": ticker.upper(),
        "target_price": target_price,
        "direction": direction,
        "note": note,
        "created": datetime.now().strftime("%Y-%m-%d"),
        "triggered": False
    }

    alerts = load_alerts()
    alerts.append(alert)
    save_alerts(alerts)

    return alert


def remove_alert(ticker: str, target_price: float):
    """Remove matching alert."""
    alerts = load_alerts()
    ticker = ticker.upper()

    # Remove all alerts matching ticker and target_price
    alerts = [a for a in alerts if not (a["ticker"] == ticker and a["target_price"] == target_price)]

    save_alerts(alerts)


def check_alerts(current_prices: dict[str, float]) -> list[dict]:
    """
    Check which alerts have been triggered.
    current_prices = {"AAPL": 315.0, "GOOG": 380.0}
    Returns list of triggered alerts (does NOT auto-delete them — caller decides).
    Marks triggered=True on matching alerts and saves.
    """
    alerts = load_alerts()
    newly_triggered = []

    for alert in alerts:
        ticker = alert["ticker"]
        if ticker not in current_prices:
            continue

        current_price = current_prices[ticker]
        target_price = alert["target_price"]
        direction = alert["direction"]

        # Check if alert conditions are met and not already triggered
        condition_met = False
        if direction == "above" and current_price >= target_price:
            condition_met = True
        elif direction == "below" and current_price <= target_price:
            condition_met = True

        if condition_met and not alert.get("triggered", False):
            alert["triggered"] = True
            newly_triggered.append(alert)

    # Save updated alerts with triggered status
    save_alerts(alerts)

    return newly_triggered


def get_alerts_for_ticker(ticker: str) -> list[dict]:
    """Return all alerts for a specific ticker."""
    alerts = load_alerts()
    ticker = ticker.upper()
    return [a for a in alerts if a["ticker"] == ticker]
