"""
Setup detector: swing-based technical structure analysis.

Adapted from AutoTrader's find_swings() (EMA gradient method) + original
setup classification logic. No external dependencies beyond numpy/pandas.

Setup types:
  BREAKOUT      — price at/above recent swing high, volume expanding
  PULLBACK      — uptrend (HH+HL), price dipped to support/EMA zone
  MEAN_REVERSION— price >2σ from MA, RSI extreme, expecting snap-back
  BREAKDOWN     — price breaking swing lows (bear continuation)
  RANGE         — no clear directional structure
"""
import numpy as np
import pandas as pd
from typing import Optional


# ── EMA helper (matches AutoTrader's internal) ──────────────────────────────

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder/exponential moving average via pandas ewm."""
    s = pd.Series(values)
    return s.ewm(span=period, adjust=False).mean().values


def _rolling_carry(values) -> list:
    """Carry forward last non-zero value."""
    out, last = [], 0
    for v in values:
        if v != 0:
            last = v
        out.append(last)
    return out


# ── Core swing detection (AutoTrader find_swings logic) ─────────────────────

def find_swings(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Detect swing highs/lows using EMA gradient direction changes.

    n: EMA period for smoothing (higher = fewer, more significant swings)

    Returns DataFrame with columns:
      Highs      — value of swing high at that bar, else 0
      Lows       — value of swing low at that bar, else 0
      LastHigh   — most recent swing high (carried forward)
      LastLow    — most recent swing low (carried forward)
      Trend      — 1 = upswing, -1 = downswing (carried forward)
    """
    hl2 = (df["high"].values + df["low"].values) / 2.0
    smooth = _ema(hl2, n)

    grad = np.sign(np.diff(smooth, prepend=smooth[0]))
    direction_change = (grad != np.roll(grad, 1)).astype(int)
    direction_change[0] = 0

    highs = np.zeros(len(df))
    lows = np.zeros(len(df))
    trend_raw = np.zeros(len(df))

    for i in range(1, len(df)):
        if direction_change[i]:
            window = slice(max(0, i - n + 1), i + 1)
            if grad[i] < 0:
                # Just turned down → this was a high
                highs[i] = float(df["high"].values[window].max())
                trend_raw[i] = -1
            else:
                # Just turned up → this was a low
                lows[i] = float(df["low"].values[window].min())
                trend_raw[i] = 1

    last_high = np.array(_rolling_carry(highs))
    last_low  = np.array(_rolling_carry(lows))
    trend     = np.array(_rolling_carry(trend_raw))

    return pd.DataFrame({
        "Highs":    highs,
        "Lows":     lows,
        "LastHigh": last_high,
        "LastLow":  last_low,
        "Trend":    trend,
    }, index=df.index)


# ── Swing structure classification (HH/HL/LH/LL) ────────────────────────────

def classify_structure(swing_df: pd.DataFrame) -> dict:
    """
    Identify the most recent HH/HL/LH/LL pattern.

    Returns dict:
      highs: list of recent swing high values (chronological)
      lows:  list of recent swing low values (chronological)
      hh: bool — higher high vs previous
      hl: bool — higher low vs previous
      lh: bool — lower high vs previous
      ll: bool — lower low vs previous
      bias: "bullish" | "bearish" | "mixed" | "insufficient"
    """
    recent_highs = swing_df["Highs"][swing_df["Highs"] > 0].tail(5).tolist()
    recent_lows  = swing_df["Lows"][swing_df["Lows"] > 0].tail(5).tolist()

    result = {
        "highs": recent_highs,
        "lows":  recent_lows,
        "hh": False, "hl": False,
        "lh": False, "ll": False,
        "bias": "insufficient",
    }

    if len(recent_highs) >= 2:
        result["hh"] = recent_highs[-1] > recent_highs[-2]
        result["lh"] = recent_highs[-1] < recent_highs[-2]

    if len(recent_lows) >= 2:
        result["hl"] = recent_lows[-1] > recent_lows[-2]
        result["ll"] = recent_lows[-1] < recent_lows[-2]

    hh, hl = result["hh"], result["hl"]
    lh, ll = result["lh"], result["ll"]

    if hh and hl:
        result["bias"] = "bullish"
    elif lh and ll:
        result["bias"] = "bearish"
    elif (hh and ll) or (lh and hl):
        result["bias"] = "mixed"
    elif len(recent_highs) < 2 or len(recent_lows) < 2:
        result["bias"] = "insufficient"
    else:
        result["bias"] = "mixed"

    return result


# ── Setup type detection ─────────────────────────────────────────────────────

def detect_setup(
    df: pd.DataFrame,
    swing_df: pd.DataFrame,
    structure: dict,
) -> str:
    """
    Classify the current price setup into one of 5 types.

    Uses:
      - Swing structure (HH/HL bias)
      - Price vs last swing high/low
      - RSI (if available in df)
      - Volume trend (if available)

    Returns: "BREAKOUT" | "PULLBACK" | "MEAN_REVERSION" | "BREAKDOWN" | "RANGE"
    """
    close = float(df["close"].iloc[-1])
    last_high = float(swing_df["LastHigh"].iloc[-1]) if swing_df["LastHigh"].iloc[-1] > 0 else None
    last_low  = float(swing_df["LastLow"].iloc[-1])  if swing_df["LastLow"].iloc[-1] > 0 else None

    bias = structure.get("bias", "mixed")
    rsi  = float(df["rsi"].iloc[-1]) if "rsi" in df.columns and not pd.isna(df["rsi"].iloc[-1]) else None

    # Volume trend: recent 5d avg vs 20d avg
    vol_surge = False
    if "volume" in df.columns and len(df) >= 20:
        vol5  = float(df["volume"].tail(5).mean())
        vol20 = float(df["volume"].tail(20).mean())
        vol_surge = vol5 > vol20 * 1.3

    # Mean distance from 50-day MA
    ma_pct_dev = None
    if "sma50" in df.columns and not pd.isna(df["sma50"].iloc[-1]):
        ma50 = float(df["sma50"].iloc[-1])
        if ma50 > 0:
            ma_pct_dev = (close - ma50) / ma50

    # ── Decision logic ────────────────────────────────────────────────────────

    # BREAKDOWN: bearish structure, price at/below last swing low
    if bias == "bearish" and last_low and close <= last_low * 1.005:
        return "BREAKDOWN"

    # BREAKOUT: price at/above last swing high with volume
    if last_high and close >= last_high * 0.995:
        if vol_surge or bias == "bullish":
            return "BREAKOUT"

    # MEAN_REVERSION: RSI extreme or price far from MA
    if rsi is not None and rsi < 30:
        return "MEAN_REVERSION"
    if ma_pct_dev is not None and ma_pct_dev < -0.08:
        return "MEAN_REVERSION"

    # PULLBACK: bullish structure but pulled back from swing high
    if bias == "bullish" and last_high and close < last_high * 0.97:
        return "PULLBACK"

    # Fallback: RANGE
    return "RANGE"


# ── Structural level extraction ──────────────────────────────────────────────

def get_structural_levels(swing_df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Extract recent swing high/low levels for use as TP/SL anchors.

    Returns:
      swing_highs: list[float] — recent swing highs, sorted ascending
      swing_lows:  list[float] — recent swing lows, sorted descending (closest to price first)
      nearest_high: float | None
      nearest_low:  float | None
    """
    recent = swing_df.tail(lookback)

    highs = sorted(recent["Highs"][recent["Highs"] > 0].tolist())
    lows  = sorted(recent["Lows"][recent["Lows"] > 0].tolist(), reverse=True)

    return {
        "swing_highs":   highs,
        "swing_lows":    lows,
        "nearest_high":  highs[0] if highs else None,
        "nearest_low":   lows[0]  if lows  else None,
    }


# ── Entry/exit price generation ──────────────────────────────────────────────

def suggest_entry_exit(
    setup_type: str,
    close: float,
    atr: float,
    structure: dict,
    structural_levels: dict,
    rsi: Optional[float] = None,
    sma20: Optional[float] = None,
    sma50: Optional[float] = None,
    lower_bb: Optional[float] = None,
) -> dict:
    """
    Generate setup-specific entry/exit levels from structural data.

    Each setup type uses a different anchoring strategy:

    BREAKOUT      → entry = last swing high + 0.1% (buy the break)
                    SL    = last swing low (structural invalidation)
                    TP1   = last swing high + 1× ATR
                    TP2   = analyst / next measured move

    PULLBACK      → entry = nearest structural support (swing low or MA)
                    SL    = swing low below that support - small buffer
                    TP1   = last swing high (prior swing)
                    TP2   = next swing high above

    MEAN_REVERSION→ entry = lower Bollinger Band or MA - 1.5σ (oversold snap)
                    SL    = close * 0.97 (3% hard floor — volatile entry)
                    TP1   = SMA20 (revert to mean)
                    TP2   = upper half between SMA20 and SMA50

    BREAKDOWN     → no long entry, return empty
    RANGE         → entry = nearest support, SL = below that, TP = range midpoint

    Returns dict:
      entry:        float
      stop_loss:    float
      tp1:          float
      tp2:          float
      entry_basis:  str — human-readable description
      sl_basis:     str
      tp_basis:     str
    """
    swing_highs = structural_levels.get("swing_highs", [])
    swing_lows  = structural_levels.get("swing_lows", [])

    # vol-adjusted position size (pysystemtrade method)
    # assumes $10k account, 1% risk per trade
    _account = 10_000
    _risk_pct = 0.01
    daily_vol_pct = atr / close if close > 0 else 0.02
    annual_vol = daily_vol_pct * (252 ** 0.5)
    if annual_vol > 0:
        vol_adj_shares = int((_account * _risk_pct) / (close * annual_vol))
    else:
        vol_adj_shares = 0
    # ponytail: hardcoded $10k/1% defaults, make configurable when multi-account

    def _fallback():
        return {
            "entry": None, "stop_loss": None, "tp1": None, "tp2": None,
            "entry_basis": "no structural levels", "sl_basis": "", "tp_basis": "",
            "vol_adj_shares": vol_adj_shares,
        }

    if setup_type == "BREAKDOWN":
        return _fallback()

    if setup_type == "BREAKOUT":
        last_swing_high = swing_highs[-1] if swing_highs else None
        last_swing_low  = swing_lows[0]   if swing_lows  else None

        if not last_swing_high:
            return _fallback()

        entry    = round(last_swing_high * 1.001, 2)   # just above breakout level
        sl       = round((last_swing_low * 0.997) if last_swing_low else (entry - 2 * atr), 2)
        tp1      = round(entry + 1.5 * atr, 2)
        tp2_candidates = [h for h in swing_highs if h > tp1 * 1.01]
        tp2      = round(tp2_candidates[0] if tp2_candidates else entry + 3 * atr, 2)

        return {
            "entry":       entry,
            "stop_loss":   sl,
            "tp1":         tp1,
            "tp2":         tp2,
            "entry_basis": f"breakout above swing high ${last_swing_high:.2f}",
            "sl_basis":    f"swing low ${last_swing_low:.2f}" if last_swing_low else "2× ATR",
            "tp_basis":    "1.5× ATR above breakout; next swing high",
            "vol_adj_shares": vol_adj_shares,
        }

    if setup_type == "PULLBACK":
        # Find support closest to close but below it
        supports = [lv for lv in swing_lows if lv < close]
        anchor = supports[0] if supports else (sma20 or sma50)
        if not anchor:
            return _fallback()

        entry = round(min(anchor * 1.002, close * 0.999), 2)  # slightly above support, never above close

        # SL: next swing low below anchor
        deeper = [lv for lv in swing_lows if lv < anchor * 0.995]
        sl = round(deeper[0] * 0.997 if deeper else anchor * 0.97, 2)

        # TP: prior swing highs above entry
        targets = sorted([h for h in swing_highs if h > entry])
        tp1 = round(targets[0] if targets else entry + 2 * atr, 2)
        tp2 = round(targets[1] if len(targets) > 1 else entry + 3 * atr, 2)

        return {
            "entry":       entry,
            "stop_loss":   sl,
            "tp1":         tp1,
            "tp2":         tp2,
            "entry_basis": f"pullback to structural support ${anchor:.2f}",
            "sl_basis":    f"below next swing low ${deeper[0]:.2f}" if deeper else "3% below support",
            "tp_basis":    f"prior swing highs ${tp1:.2f} / ${tp2:.2f}",
            "vol_adj_shares": vol_adj_shares,
        }

    if setup_type == "MEAN_REVERSION":
        # Entry: lower BB preferred, else MA - buffer
        if lower_bb and lower_bb < close * 0.99:
            entry = round(lower_bb, 2)
            entry_basis = f"lower Bollinger Band ${lower_bb:.2f}"
        elif sma20:
            entry = round(sma20 * 0.97, 2)   # 3% below mean
            entry_basis = f"3% below SMA20 ${sma20:.2f}"
        else:
            entry = round(close * 0.97, 2)
            entry_basis = "3% below current (no MA data)"

        sl = round(entry * 0.97, 2)   # 3% below entry — can't go much lower structurally

        tp1 = round(sma20, 2) if sma20 else round(entry * 1.04, 2)
        tp2_base = sma50 if sma50 else (sma20 * 1.03 if sma20 else entry * 1.08)
        tp2 = round(tp2_base, 2)

        return {
            "entry":       entry,
            "stop_loss":   sl,
            "tp1":         tp1,
            "tp2":         tp2,
            "entry_basis": entry_basis,
            "sl_basis":    "3% hard floor (mean-reversion entry)",
            "tp_basis":    f"mean-revert to SMA20 ${tp1:.2f}, extend to SMA50 ${tp2:.2f}",
            "vol_adj_shares": vol_adj_shares,
        }

    # RANGE — use nearest support/resistance below current price
    below_close = [lv for lv in swing_lows if lv < close]
    support = below_close[0] if below_close else None
    above_close = [h for h in swing_highs if h > close]
    resist = above_close[0] if above_close else None

    if not support:
        return _fallback()

    entry = round(min(support * 1.002, close * 0.999), 2)
    sl    = round(support * 0.97, 2)
    tp1   = round(resist, 2) if resist else round(entry + 2 * atr, 2)
    tp2   = round(tp1 * 1.02, 2)

    return {
        "entry":       entry,
        "stop_loss":   sl,
        "tp1":         tp1,
        "tp2":         tp2,
        "entry_basis": f"range support ${support:.2f}",
        "sl_basis":    "3% below range support",
        "tp_basis":    f"range resistance ${resist:.2f}" if resist else "2× ATR",
        "vol_adj_shares": vol_adj_shares,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def analyze_setup(df: pd.DataFrame, n: int = 3) -> dict:
    """
    Full pipeline: swings → structure → setup type → entry/exit levels.

    Expects df with columns: open, high, low, close, volume (all lowercase).
    Optional columns used if present: rsi, sma20, sma50, bb_lower.

    Returns:
      setup_type:       str
      bias:             str
      swing_df:         DataFrame
      structure:        dict
      structural_levels:dict
      entry_exit:       dict
    """
    if df is None or len(df) < 20:
        return {
            "setup_type": "RANGE",
            "bias": "insufficient",
            "swing_df": pd.DataFrame(),
            "structure": {},
            "structural_levels": {"swing_highs": [], "swing_lows": [], "nearest_high": None, "nearest_low": None},
            "entry_exit": {"entry": None, "stop_loss": None, "tp1": None, "tp2": None,
                           "entry_basis": "insufficient data", "sl_basis": "", "tp_basis": ""},
        }

    swing_df   = find_swings(df, n=n)
    structure  = classify_structure(swing_df)
    levels     = get_structural_levels(swing_df)
    setup_type = detect_setup(df, swing_df, structure)

    close  = float(df["close"].iloc[-1])
    rsi    = float(df["rsi"].iloc[-1])    if "rsi"      in df.columns and not pd.isna(df["rsi"].iloc[-1])    else None
    sma20  = float(df["sma20"].iloc[-1])  if "sma20"    in df.columns and not pd.isna(df["sma20"].iloc[-1])  else None
    sma50  = float(df["sma50"].iloc[-1])  if "sma50"    in df.columns and not pd.isna(df["sma50"].iloc[-1])  else None
    bb_low = float(df["bb_lower"].iloc[-1]) if "bb_lower" in df.columns and not pd.isna(df["bb_lower"].iloc[-1]) else None

    # ATR approximation
    hi = df["high"].astype(float)
    lo = df["low"].astype(float)
    cl = df["close"].astype(float).shift(1)
    tr = (hi - lo).combine((hi - cl).abs(), max).combine((lo - cl).abs(), max)
    atr = float(tr.rolling(14).mean().iloc[-1] or (close * 0.015))

    entry_exit = suggest_entry_exit(
        setup_type=setup_type,
        close=close,
        atr=atr,
        structure=structure,
        structural_levels=levels,
        rsi=rsi,
        sma20=sma20,
        sma50=sma50,
        lower_bb=bb_low,
    )

    return {
        "setup_type":        setup_type,
        "bias":              structure.get("bias", "mixed"),
        "swing_df":          swing_df,
        "structure":         structure,
        "structural_levels": levels,
        "entry_exit":        entry_exit,
    }
