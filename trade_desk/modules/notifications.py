"""Desktop push notifications for price alerts — works when browser tab is closed."""
import threading
import os


def _send_mac_notification(title: str, message: str, subtitle: str = ""):
    """Send Mac desktop notification via osascript (no dependencies needed)."""
    try:
        subtitle_part = f'subtitle "{subtitle}" ' if subtitle else ""
        script = f'display notification "{message}" with title "{title}" {subtitle_part}sound name "Ping"'
        os.system(f"osascript -e '{script}'")
    except Exception:
        pass


def notify_alert_triggered(ticker: str, target_price: float, direction: str, current_price: float):
    """Fire desktop notification when a price alert triggers."""
    title = f"🔔 Trade Lab Alert: {ticker}"
    message = f"Price went {direction} ${target_price:.2f} — now ${current_price:.2f}"
    _send_mac_notification(title, message, subtitle="Price Alert")


def notify_watchlist_signal(ticker: str, verdict: str, confidence: int):
    """Fire notification for a strong watchlist signal."""
    if verdict == "BUY" and confidence >= 70:
        title = f"🟢 Trade Lab: {ticker} BUY Signal"
        message = f"Confidence {confidence}% — check your watchlist"
        _send_mac_notification(title, message, subtitle="Watchlist Signal")
    elif verdict == "SELL / AVOID" and confidence >= 70:
        title = f"🔴 Trade Lab: {ticker} SELL Signal"
        message = f"Confidence {confidence}% — review your position"
        _send_mac_notification(title, message, subtitle="Watchlist Signal")


class BackgroundAlertChecker:
    """
    Runs a background thread that periodically checks price alerts
    and sends desktop notifications — works even when the Alerts tab is not open.
    """

    def __init__(self, interval_seconds: int = 300):
        self.interval = interval_seconds
        self._thread = None
        self._stop_event = threading.Event()
        self._running = False

    def start(self, get_current_prices_fn, check_alerts_fn):
        """
        Start background checking.
        get_current_prices_fn: callable() -> dict[ticker, price]
        check_alerts_fn: callable(prices) -> list[triggered_alerts]
        """
        if self._running:
            return
        self._stop_event.clear()
        self._running = True

        def _loop():
            while not self._stop_event.is_set():
                try:
                    prices = get_current_prices_fn()
                    triggered = check_alerts_fn(prices)
                    for alert in triggered:
                        notify_alert_triggered(
                            ticker=alert["ticker"],
                            target_price=alert["target_price"],
                            direction=alert["direction"],
                            current_price=prices.get(alert["ticker"], alert["target_price"]),
                        )
                except Exception:
                    pass
                self._stop_event.wait(self.interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()


# Module-level singleton — import and call .start() once from app
alert_checker = BackgroundAlertChecker(interval_seconds=300)
