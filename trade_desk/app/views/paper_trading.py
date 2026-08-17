"""Paper Trading — live-tests the ML model's 5-day predictions.

Lock in a ticker at its current price + the model's predicted 5-day target.
Each visit refetches the current price and checks whether it has hit the
predicted target, or whether the 5-trading-day window has expired.
"""
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.db import paper_add, paper_load, paper_resolve, paper_remove, paper_relock
from modules.cached_fetch import cached_ohlcv
from app.services.analysis import get_ml_verdict

_TRADING_DAYS_5D = timedelta(days=7)  # ponytail: calendar-day approx of 5 trading days, switch to a market calendar if weekends/holidays skew results


def _current_price(ticker: str) -> float | None:
    try:
        df = cached_ohlcv(ticker, "5d")
        return float(df["close"].iloc[-1])
    except Exception:
        return None


def _resolve_open_trades(open_trades: list[dict]) -> None:
    today = datetime.now().date()
    for t in open_trades:
        price = _current_price(t["ticker"])
        if price is None:
            continue
        target_date = datetime.strptime(t["target_date"], "%Y-%m-%d").date()
        hit = (price >= t["pred_target_price"]) if t["pred_return_5d"] >= 0 else (price <= t["pred_target_price"])
        if hit:
            paper_resolve(t["id"], "hit", price, str(today))
        elif today >= target_date:
            paper_resolve(t["id"], "expired", price, str(today))


def render_paper_trading_page(st_obj) -> None:
    st_obj.title("📝 Paper Trading")
    st_obj.markdown("Lock in the model's 5-day prediction for a ticker at today's price, then watch whether it hits.")

    with st_obj.form("add_paper_trade"):
        col_ticker, col_btn = st_obj.columns([4, 1])
        with col_ticker:
            ticker = st_obj.text_input("Ticker", placeholder="e.g. AAPL", label_visibility="collapsed").upper().strip()
        with col_btn:
            submitted = st_obj.form_submit_button("🔒 Lock In")

    if submitted and ticker:
        price = _current_price(ticker)
        verdict = get_ml_verdict(ticker)
        if price is None or verdict is None:
            st_obj.error(f"Couldn't get a live price/prediction for {ticker}.")
        else:
            target_price = price * (1 + verdict["pred_return_5d"])
            entry_date = datetime.now().date()
            target_date = entry_date + _TRADING_DAYS_5D
            low_conf = verdict.get("low_confidence", False)
            paper_add(ticker, str(entry_date), price, verdict["pred_return_5d"], target_price,
                      str(target_date), low_confidence=low_conf)
            warn = " ⚠️ low confidence — >25% of model inputs were missing" if low_conf else ""
            st_obj.success(f"Locked {ticker} @ ${price:.2f} → target ${target_price:.2f} ({verdict['pred_return_5d']:+.2%}) by {target_date}{warn}")
            st_obj.rerun()

    st_obj.divider()

    open_trades = paper_load(status="open")
    if open_trades:
        with st_obj.spinner("Checking current prices..."):
            _resolve_open_trades(open_trades)
        open_trades = paper_load(status="open")

    st_obj.subheader(f"🟡 Open ({len(open_trades)})")
    if open_trades:
        rows = []
        for t in open_trades:
            price = _current_price(t["ticker"])
            progress = None
            if price is not None and t["pred_target_price"] != t["entry_price"]:
                progress = (price - t["entry_price"]) / (t["pred_target_price"] - t["entry_price"])
            rows.append({
                "Ticker": t["ticker"],
                "Entry": t["entry_date"],
                "Entry $": t["entry_price"],
                "Current $": price,
                "Target $": t["pred_target_price"],
                "Pred Return": t["pred_return_5d"] * 100,
                "Progress to Target": progress,
                "Target Date": t["target_date"],
                "⚠️": "⚠️" if t.get("low_confidence") else "",
                "id": t["id"],
            })
        df = pd.DataFrame(rows)
        n_low_conf = sum(1 for t in open_trades if t.get("low_confidence"))
        if n_low_conf:
            st_obj.warning(
                f"⚠️ {n_low_conf} open trade(s) were locked with >25% missing model inputs "
                "(e.g. a Yahoo API hiccup at entry time) — prediction may be unreliable. "
                "Use Re-lock below to redo the entry with fresh data."
            )
        st_obj.dataframe(
            df.drop(columns=["id"]),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Entry $": st_obj.column_config.NumberColumn(format="$%.2f"),
                "Current $": st_obj.column_config.NumberColumn(format="$%.2f"),
                "Target $": st_obj.column_config.NumberColumn(format="$%.2f"),
                "Pred Return": st_obj.column_config.NumberColumn(format="%.2f%%"),
                "Progress to Target": st_obj.column_config.ProgressColumn(min_value=0, max_value=1, format=""),
                "⚠️": st_obj.column_config.TextColumn(help="Low confidence — >25% of model input features were missing at entry"),
            },
        )
        for t in open_trades:
            col_rm, col_relock = st_obj.columns(2)
            with col_rm:
                if st_obj.button(f"🗑 Remove {t['ticker']} ({t['entry_date']})", key=f"rm_paper_{t['id']}"):
                    paper_remove(t["id"])
                    st_obj.rerun()
            with col_relock:
                relock_label = f"🔄 Re-lock {t['ticker']}" + (" (was flagged)" if t.get("low_confidence") else "")
                if st_obj.button(relock_label, key=f"relock_paper_{t['id']}"):
                    get_ml_verdict.clear(t["ticker"])
                    new_price = _current_price(t["ticker"])
                    new_verdict = get_ml_verdict(t["ticker"])
                    if new_price is None or new_verdict is None:
                        st_obj.error(f"Couldn't refresh {t['ticker']}.")
                    else:
                        new_target = new_price * (1 + new_verdict["pred_return_5d"])
                        new_entry_date = datetime.now().date()
                        new_target_date = new_entry_date + _TRADING_DAYS_5D
                        paper_relock(
                            t["id"], str(new_entry_date), new_price, new_verdict["pred_return_5d"],
                            new_target, str(new_target_date),
                            low_confidence=new_verdict.get("low_confidence", False),
                        )
                        st_obj.success(f"Re-locked {t['ticker']} @ ${new_price:.2f} ({new_verdict['pred_return_5d']:+.2%})")
                        st_obj.rerun()
    else:
        st_obj.info("No open paper trades. Lock one in above.")

    st_obj.divider()

    resolved = [t for t in paper_load() if t["status"] != "open"]
    st_obj.subheader(f"✅ Resolved ({len(resolved)})")
    if resolved:
        n_hit = sum(1 for t in resolved if t["status"] == "hit")
        st_obj.caption(f"Hit rate: {n_hit}/{len(resolved)} ({n_hit / len(resolved):.0%})")
        rdf = pd.DataFrame([{
            "Ticker": t["ticker"],
            "Entry": t["entry_date"],
            "Entry $": t["entry_price"],
            "Target $": t["pred_target_price"],
            "Resolved $": t["resolved_price"],
            "Result": t["status"],
            "Resolved": t["resolved_date"],
        } for t in resolved])
        st_obj.dataframe(
            rdf,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Entry $": st_obj.column_config.NumberColumn(format="$%.2f"),
                "Target $": st_obj.column_config.NumberColumn(format="$%.2f"),
                "Resolved $": st_obj.column_config.NumberColumn(format="$%.2f"),
            },
        )
    else:
        st_obj.info("Nothing resolved yet.")
