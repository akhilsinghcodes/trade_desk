"""Institutional and major holder data from yfinance."""
import yfinance as yf


def get_ownership(ticker: str) -> dict:
    """
    Returns dict with:
    - institutional_pct: float (0-100) or None
    - insider_pct: float (0-100) or None
    - top_holders: list of dicts {name, pct_held}
    - interpretation: str
    """
    t = yf.Ticker(ticker)
    result = {
        "institutional_pct": None,
        "insider_pct": None,
        "top_holders": [],
        "interpretation": "neutral",
    }

    try:
        major = t.major_holders
        if major is not None and not major.empty:
            # Newer yfinance: DataFrame with 'Breakdown' index and 'Value' column
            if "Value" in major.columns:
                idx = major.index if hasattr(major.index, '__iter__') else []
                row_dict = {}
                for col in major.index:
                    try:
                        row_dict[str(col).lower()] = float(major.loc[col, "Value"])
                    except Exception:
                        pass
                insider_raw = row_dict.get("insiderspercentheld")
                inst_raw = row_dict.get("institutionspercentheld")
                if insider_raw is not None:
                    result["insider_pct"] = round(insider_raw * 100, 2)
                if inst_raw is not None:
                    result["institutional_pct"] = round(inst_raw * 100, 2)
            else:
                # Fallback: positional
                vals = major.iloc[:, 0].tolist()
                def _p(v):
                    try:
                        return float(str(v).replace("%", "").strip())
                    except Exception:
                        return None
                if len(vals) >= 1:
                    result["insider_pct"] = _p(vals[0])
                if len(vals) >= 2:
                    result["institutional_pct"] = _p(vals[1])
    except Exception:
        pass

    try:
        inst = t.institutional_holders
        if inst is not None and not inst.empty:
            inst.columns = [c.lower().replace(" ", "_") for c in inst.columns]
            holders = []
            for _, row in inst.head(5).iterrows():
                name = row.get("holder", row.get("name", "Unknown"))
                pct = row.get("pctheld", row.get("pct_held", None))
                if pct is not None:
                    try:
                        pct = float(pct) * 100  # yfinance returns as decimal
                    except Exception:
                        pass
                holders.append({"name": str(name), "pct_held": round(pct, 2) if pct else None})
            result["top_holders"] = holders
    except Exception:
        pass

    inst_pct = result["institutional_pct"]
    if inst_pct is not None:
        if inst_pct > 70:
            result["interpretation"] = "high_institutional"
        elif inst_pct < 20:
            result["interpretation"] = "low_institutional"
        else:
            result["interpretation"] = "normal"

    return result


def score_ownership(data: dict) -> tuple[str, str, str]:
    """Returns (label, status, text) for signals list."""
    inst_pct = data.get("institutional_pct")
    insider_pct = data.get("insider_pct")
    interp = data.get("interpretation", "neutral")

    if inst_pct is None:
        return ("Ownership", "neutral", "No ownership data available.")

    parts = []
    if inst_pct is not None:
        parts.append(f"{inst_pct:.1f}% held by institutions")
    if insider_pct is not None:
        parts.append(f"{insider_pct:.1f}% held by insiders")

    base = ", ".join(parts) + "."

    if interp == "high_institutional":
        return ("Ownership", "good", f"{base} High institutional ownership — big funds are committed.")
    elif interp == "low_institutional":
        return ("Ownership", "warning", f"{base} Low institutional ownership — limited big-money support.")
    else:
        return ("Ownership", "neutral", f"{base} Normal ownership structure.")
