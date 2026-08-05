"""Persist watchlist to local JSON file."""
import json
from pathlib import Path

WATCHLIST_FILE = Path(__file__).parent.parent / "data" / "watchlist.json"


def load_watchlist() -> list[str]:
    WATCHLIST_FILE.parent.mkdir(exist_ok=True)
    if WATCHLIST_FILE.exists():
        return json.loads(WATCHLIST_FILE.read_text())
    return []


def save_watchlist(tickers: list[str]):
    WATCHLIST_FILE.parent.mkdir(exist_ok=True)
    WATCHLIST_FILE.write_text(json.dumps(tickers))
