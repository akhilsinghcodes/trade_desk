"""SQLite database layer — replaces JSON files for watchlist, portfolio, alerts.
Also provides a cache table for API responses with TTL.
"""
import sqlite3
import json
import time
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trade_lab.db")


def _ensure_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_connection() -> sqlite3.Connection:
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist."""
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker TEXT PRIMARY KEY,
            added_at REAL DEFAULT (unixepoch())
        );

        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            shares REAL NOT NULL,
            buy_price REAL NOT NULL,
            note TEXT DEFAULT '',
            added_at REAL DEFAULT (unixepoch())
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            target_price REAL NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('above', 'below')),
            note TEXT DEFAULT '',
            triggered INTEGER DEFAULT 0,
            triggered_at REAL,
            created_at REAL DEFAULT (unixepoch())
        );

        CREATE TABLE IF NOT EXISTS api_cache (
            cache_key TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            cached_at REAL NOT NULL,
            ttl_seconds REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            backend TEXT NOT NULL,
            source TEXT NOT NULL,
            in_tokens INTEGER DEFAULT 0,
            out_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0,
            created_at REAL DEFAULT (unixepoch())
        );
        """)


# ── WATCHLIST ──────────────────────────────────────────────────────────────────

def wl_load() -> list[str]:
    init_db()
    with db() as conn:
        rows = conn.execute("SELECT ticker FROM watchlist ORDER BY added_at").fetchall()
        return [r["ticker"] for r in rows]


def wl_add(ticker: str):
    init_db()
    with db() as conn:
        conn.execute("INSERT OR IGNORE INTO watchlist (ticker) VALUES (?)", (ticker.upper(),))


def wl_remove(ticker: str):
    init_db()
    with db() as conn:
        conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))


def wl_save(tickers: list[str]):
    """Replace entire watchlist."""
    init_db()
    with db() as conn:
        conn.execute("DELETE FROM watchlist")
        for t in tickers:
            conn.execute("INSERT OR IGNORE INTO watchlist (ticker) VALUES (?)", (t.upper(),))


# ── PORTFOLIO ──────────────────────────────────────────────────────────────────

def port_load() -> list[dict]:
    init_db()
    with db() as conn:
        rows = conn.execute("SELECT * FROM portfolio ORDER BY added_at").fetchall()
        return [dict(r) for r in rows]


def port_add(ticker: str, shares: float, buy_price: float, note: str = "") -> dict:
    init_db()
    with db() as conn:
        conn.execute(
            "INSERT INTO portfolio (ticker, shares, buy_price, note) VALUES (?, ?, ?, ?)",
            (ticker.upper(), shares, buy_price, note)
        )
    return {"ticker": ticker.upper(), "shares": shares, "buy_price": buy_price, "note": note}


def port_remove(ticker: str):
    init_db()
    with db() as conn:
        conn.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker.upper(),))


# ── ALERTS ────────────────────────────────────────────────────────────────────

def alerts_load() -> list[dict]:
    init_db()
    with db() as conn:
        rows = conn.execute("SELECT * FROM alerts ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]


def alerts_add(ticker: str, target_price: float, direction: str, note: str = "") -> dict:
    init_db()
    with db() as conn:
        conn.execute(
            "INSERT INTO alerts (ticker, target_price, direction, note) VALUES (?, ?, ?, ?)",
            (ticker.upper(), target_price, direction, note)
        )
    return {"ticker": ticker.upper(), "target_price": target_price, "direction": direction, "note": note}


def alerts_remove(alert_id: int):
    init_db()
    with db() as conn:
        conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))


def alerts_trigger(alert_id: int):
    init_db()
    with db() as conn:
        conn.execute(
            "UPDATE alerts SET triggered = 1, triggered_at = ? WHERE id = ?",
            (time.time(), alert_id)
        )


def alerts_check(current_prices: dict[str, float]) -> list[dict]:
    """Check all untriggered alerts. Triggers and returns those that fired."""
    triggered = []
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE triggered = 0"
        ).fetchall()
        for row in rows:
            r = dict(row)
            price = current_prices.get(r["ticker"])
            if price is None:
                continue
            fired = (r["direction"] == "above" and price >= r["target_price"]) or \
                    (r["direction"] == "below" and price <= r["target_price"])
            if fired:
                conn.execute(
                    "UPDATE alerts SET triggered = 1, triggered_at = ? WHERE id = ?",
                    (time.time(), r["id"])
                )
                r["current_price"] = price
                triggered.append(r)
    return triggered


# ── CACHE ─────────────────────────────────────────────────────────────────────

def cache_get(key: str):
    """Return cached value or None if missing/expired.

    Does NOT delete expired rows — a caller falling back to cache_get_stale()
    on a failed live fetch needs the row to still be there. cache_set()
    overwrites (INSERT OR REPLACE) on the next successful fetch anyway, so
    there's nothing to clean up here."""
    init_db()
    with db() as conn:
        row = conn.execute("SELECT data, cached_at, ttl_seconds FROM api_cache WHERE cache_key = ?", (key,)).fetchone()
        if row is None:
            return None
        if time.time() - row["cached_at"] > row["ttl_seconds"]:
            return None
        return json.loads(row["data"])


def cache_get_stale(key: str):
    """Return cached value even if expired (for stale-on-error fallback). Returns None if missing."""
    init_db()
    with db() as conn:
        row = conn.execute("SELECT data FROM api_cache WHERE cache_key = ?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(row["data"])


def cache_set(key: str, value, ttl_seconds: float = 900):
    """Store value with TTL. Default 15 minutes."""
    init_db()
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO api_cache (cache_key, data, cached_at, ttl_seconds) VALUES (?, ?, ?, ?)",
            (key, json.dumps(value, default=str), time.time(), ttl_seconds)
        )


def cache_invalidate(key: str):
    init_db()
    with db() as conn:
        conn.execute("DELETE FROM api_cache WHERE cache_key = ?", (key,))


def cache_invalidate_ticker(ticker: str):
    """Remove all cache entries for a ticker, wherever it sits in the key."""
    init_db()
    t = ticker.upper()
    with db() as conn:
        conn.execute(
            "DELETE FROM api_cache WHERE cache_key LIKE ? OR cache_key LIKE ?",
            (f"{t}:%", f"%:{t}:%")
        )


# ── LLM TELEMETRY ─────────────────────────────────────────────────────────────

def log_llm_call(ticker: str, backend: str, source: str, cost_usd: float,
                  in_tokens: int = 0, out_tokens: int = 0):
    init_db()
    with db() as conn:
        conn.execute(
            "INSERT INTO llm_calls (ticker, backend, source, in_tokens, out_tokens, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ticker.upper(), backend, source, in_tokens, out_tokens, cost_usd)
        )


def llm_calls_summary() -> dict:
    """Aggregate totals plus per-source breakdown."""
    init_db()
    with db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(cost_usd),0) cost, "
            "COALESCE(SUM(in_tokens),0) in_tok, COALESCE(SUM(out_tokens),0) out_tok "
            "FROM llm_calls"
        ).fetchone()
        by_source = conn.execute(
            "SELECT source, COUNT(*) n, COALESCE(SUM(cost_usd),0) cost, "
            "COALESCE(SUM(in_tokens),0) in_tok, COALESCE(SUM(out_tokens),0) out_tok "
            "FROM llm_calls GROUP BY source ORDER BY cost DESC"
        ).fetchall()
        by_day = conn.execute(
            "SELECT date(created_at, 'unixepoch') day, COALESCE(SUM(cost_usd),0) cost, COUNT(*) n "
            "FROM llm_calls GROUP BY day ORDER BY day"
        ).fetchall()
        recent = conn.execute(
            "SELECT * FROM llm_calls ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        return {
            "total_calls": total["n"],
            "total_cost": total["cost"],
            "total_in_tokens": total["in_tok"],
            "total_out_tokens": total["out_tok"],
            "by_source": [dict(r) for r in by_source],
            "by_day": [dict(r) for r in by_day],
            "recent": [dict(r) for r in recent],
        }
