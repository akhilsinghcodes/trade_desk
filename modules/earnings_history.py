"""Historical earnings surprise data from yfinance."""
import yfinance as yf
import pandas as pd


def get_earnings_history(ticker: str) -> dict:
    """
    Returns last 4 quarters of earnings surprises.
    Dict keys:
    - quarters: list of dicts with keys: date, actual, estimate, surprise_pct, beat
    - beat_count: int (out of 4)
    - miss_count: int
    - avg_surprise_pct: float
    """
    result = {
        "quarters": [],
        "beat_count": 0,
        "miss_count": 0,
        "avg_surprise_pct": None,
    }

    try:
        t = yf.Ticker(ticker)
        hist = t.earnings_history
        if hist is None or hist.empty:
            # fallback: earnings_dates
            return result

        # Normalize columns
        hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]

        quarters = []
        for _, row in hist.head(4).iterrows():
            try:
                actual = row.get("epsactual") or row.get("eps_actual")
                estimate = row.get("epsestimate") or row.get("eps_estimate")
                surprise_pct = row.get("epsdifference") or row.get("surprise_pct")

                # Compute surprise % if not provided
                if surprise_pct is None and actual is not None and estimate and estimate != 0:
                    surprise_pct = ((actual - estimate) / abs(estimate)) * 100
                elif surprise_pct is not None and abs(surprise_pct) < 2:
                    # yfinance sometimes returns raw difference, not pct
                    if estimate and estimate != 0:
                        surprise_pct = (surprise_pct / abs(estimate)) * 100

                beat = actual > estimate if (actual is not None and estimate is not None) else None
                date_val = row.get("quarter") or row.get("date") or str(row.name)[:10]

                quarters.append({
                    "date": str(date_val)[:10],
                    "actual": round(float(actual), 2) if actual is not None else None,
                    "estimate": round(float(estimate), 2) if estimate is not None else None,
                    "surprise_pct": round(float(surprise_pct), 1) if surprise_pct is not None else None,
                    "beat": beat,
                })

                if beat is True:
                    result["beat_count"] += 1
                elif beat is False:
                    result["miss_count"] += 1
            except Exception:
                continue

        result["quarters"] = quarters

        valid_surps = [q["surprise_pct"] for q in quarters if q["surprise_pct"] is not None]
        if valid_surps:
            result["avg_surprise_pct"] = round(sum(valid_surps) / len(valid_surps), 1)

    except Exception:
        pass

    return result


def score_earnings_history(data: dict) -> tuple[str, str, str]:
    """Returns (label, status, text) for signals list."""
    quarters = data.get("quarters", [])
    beats = data.get("beat_count", 0)
    misses = data.get("miss_count", 0)
    avg_surp = data.get("avg_surprise_pct")
    total = beats + misses

    if total == 0:
        return ("Earnings", "neutral", "No historical earnings data available.")

    surp_str = f" Avg surprise: +{avg_surp:.1f}%" if avg_surp and avg_surp > 0 else (f" Avg surprise: {avg_surp:.1f}%" if avg_surp else "")

    if beats >= 3:
        return ("Earnings", "good", f"Beat estimates {beats}/{total} recent quarters.{surp_str} Consistent outperformer.")
    elif misses >= 3:
        return ("Earnings", "warning", f"Missed estimates {misses}/{total} recent quarters.{surp_str} Execution concerns.")
    else:
        return ("Earnings", "neutral", f"Mixed earnings — {beats} beats, {misses} misses out of {total} quarters.{surp_str}")
