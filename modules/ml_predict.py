"""
ML inference — loads trained weights from trade_ml project.
Point ML_MODEL_DIR env var at trade_ml's models/ directory.
Defaults to ../trade_ml/models/ if not set.

Features must match trade_ml/modules/ml_dataset.py FEATURE_COLS exactly.
"""
import os
import json
import numpy as np
import pandas as pd

_DEFAULT_MODEL_DIR = os.environ.get(
    "ML_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "trade_ml", "models"),
)


def models_exist(model_dir: str = _DEFAULT_MODEL_DIR) -> bool:
    return (os.path.exists(os.path.join(model_dir, "return_model.pkl")) and
            os.path.exists(os.path.join(model_dir, "return_features.json")))


def _load_models(model_dir: str):
    import pickle
    model_path = os.path.join(model_dir, "return_model.pkl")
    cols_path = os.path.join(model_dir, "return_features.json")
    if not os.path.exists(model_path) or not os.path.exists(cols_path):
        return None
    with open(cols_path) as f:
        feature_cols = json.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return {"model": model, "feature_cols": feature_cols}


def _current_spy_regime() -> str:
    """SPY 63d return > +3% = bull, < -3% = bear, else neutral."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", period="4mo", progress=False, auto_adjust=True)
        if spy.empty or len(spy) < 64:
            return "neutral"
        close = spy["Close"].squeeze()
        ret_63d = float(close.iloc[-1] / close.iloc[-64] - 1)
        if ret_63d > 0.03:
            return "bull"
        elif ret_63d < -0.03:
            return "bear"
        return "neutral"
    except Exception:
        return "neutral"


def _compute_features(df: pd.DataFrame, ticker: str | None = None) -> dict | None:
    """
    Compute all features from OHLCV + indicator data + live fundamental/sentiment data.
    df must have: close, open, high, low, volume, rsi (raw 0-100), bb_upper, bb_lower,
                  macd, macd_signal, sma20, sma50
    ticker: used to fetch fundamentals and sentiment (optional; skipped if None)
    """
    required = ['close', 'open', 'high', 'low', 'volume']
    if df is None or len(df) < 60 or not all(c in df.columns for c in required):
        return None

    c = df['close']
    h = df['high']
    lo = df['low']
    vol = df['volume']
    feats = {}

    # ── Technical ────────────────────────────────────────────────────────────
    rsi_raw = df['rsi'] if 'rsi' in df.columns else pd.Series([50.0] * len(df))
    feats['rsi'] = float(rsi_raw.clip(0, 100).iloc[-1] / 100.0)

    if 'macd' in df.columns and 'macd_signal' in df.columns:
        feats['macd_above_signal'] = float(df['macd'].iloc[-1] > df['macd_signal'].iloc[-1])
    else:
        feats['macd_above_signal'] = 0.5

    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        bb_u = df['bb_upper'].iloc[-1]
        bb_l = df['bb_lower'].iloc[-1]
        feats['bb_position'] = float(np.clip((c.iloc[-1] - bb_l) / (bb_u - bb_l + 1e-8), 0, 1))
    else:
        feats['bb_position'] = 0.5

    if 'sma20' in df.columns and 'sma50' in df.columns:
        sma50 = df['sma50'].iloc[-1]
        feats['sma20_vs_sma50'] = float(np.clip(df['sma20'].iloc[-1] / (sma50 + 1e-8) - 1, -0.5, 0.5))
    else:
        feats['sma20_vs_sma50'] = 0.0

    vol_ma20 = vol.rolling(20, min_periods=1).mean()
    feats['volume_surge_20d'] = float(np.clip((vol.iloc[-1] / (vol_ma20.iloc[-1] + 1e-8)) - 1, -1, 3))

    # ATR%
    tr = pd.concat([h - lo,
                    (h - c.shift(1)).abs(),
                    (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean().iloc[-1]
    feats['atr_pct'] = float(np.clip(atr / (c.iloc[-1] + 1e-8), 0, 0.2))

    # CMF
    mf_mult = ((c - lo) - (h - c)) / (h - lo + 1e-8)
    mf_vol = mf_mult * vol
    vol_sum = vol.rolling(20, min_periods=1).sum()
    cmf = mf_vol.rolling(20, min_periods=1).sum() / vol_sum.replace(0, np.nan)
    feats['cmf_20d'] = float(np.clip(cmf.iloc[-1] if not pd.isna(cmf.iloc[-1]) else 0, -1, 1))

    feats['roc_21d'] = float(np.clip((c.iloc[-1] / c.iloc[-22] - 1) if len(c) >= 22 else 0, -0.5, 0.5))

    high_52w = c.rolling(252, min_periods=50).max().iloc[-1]
    feats['high_52w_proximity'] = float(np.clip(c.iloc[-1] / (high_52w + 1e-8), 0, 1))

    # Alpha vs SPY (fetch live)
    try:
        import yfinance as yf
        spy_df = yf.download("SPY", period="3mo", progress=False, auto_adjust=True)
        if not spy_df.empty:
            spy_close = spy_df["Close"].squeeze()
            spy_ret_21d = float(spy_close.iloc[-1] / spy_close.iloc[-22] - 1) if len(spy_close) >= 22 else 0.0
        else:
            spy_ret_21d = 0.0
    except Exception:
        spy_ret_21d = 0.0

    stock_ret_21d = float((c.iloc[-1] / c.iloc[-22] - 1) if len(c) >= 22 else 0)
    feats['alpha_21d'] = float(np.clip(stock_ret_21d - spy_ret_21d, -0.5, 0.5))
    feats['sector_alpha_21d'] = feats['alpha_21d']  # ponytail: sector ETF fetch skipped at inference

    # Market regime features
    feats['spy_trend_21d'] = float(np.clip(spy_ret_21d, -0.3, 0.3))
    try:
        import yfinance as _yf
        vix_df = _yf.download("^VIX", period="2mo", progress=False, auto_adjust=True)
        vix_val = float(vix_df["Close"].iloc[-1]) if not vix_df.empty else 20.0
        feats['vix_level'] = float(np.clip((vix_val - 10) / 70, 0, 1))
    except Exception:
        feats['vix_level'] = 0.3

    # ── Fundamentals ─────────────────────────────────────────────────────────
    fund_defaults = {
        "pe_ratio": 0.0, "forward_pe": 0.0, "profit_margin": 0.0, "roe": 0.0,
        "debt_to_equity": 0.0, "revenue_growth": 0.0, "earnings_growth": 0.0,
        "fcf_yield": 0.0, "piotroski_score": 0.5, "altman_z": 0.0,
    }

    if ticker:
        try:
            fund_data = _fetch_fundamentals_for_inference(ticker)
            fund_defaults.update(fund_data)
        except Exception:
            pass

    feats.update(fund_defaults)

    # ── Sentiment ─────────────────────────────────────────────────────────────
    feats["short_interest_pct"] = _fetch_short_interest(ticker) if ticker else 0.0
    feats["insider_net_ratio"] = _fetch_insider_ratio(ticker) if ticker else 0.0
    feats["analyst_score"] = _fetch_analyst_score(ticker) if ticker else 0.0
    feats["put_call_ratio"] = _fetch_put_call_ratio(ticker) if ticker else 0.0

    return {k: 0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
            for k, v in feats.items()}


def _fetch_fundamentals_for_inference(ticker: str) -> dict:
    """Use cached_fundamentals() and cached_piotroski() / cached_altman_z()."""
    result = {}
    try:
        from modules.cached_fetch import cached_fundamentals, cached_piotroski, cached_altman_z
        fund = cached_fundamentals(ticker) or {}
        result['pe_ratio'] = float(np.clip(fund.get('pe_ratio') or 0, -100, 100))
        result['forward_pe'] = float(np.clip(fund.get('forward_pe') or 0, -100, 100))
        result['profit_margin'] = float(fund.get('profit_margin') or 0)
        result['roe'] = float(fund.get('roe') or 0)
        result['debt_to_equity'] = float(np.clip(fund.get('debt_to_equity') or 0, -50, 50))
        result['revenue_growth'] = float(fund.get('revenue_growth') or 0)
        result['earnings_growth'] = float(fund.get('earnings_growth') or 0)

        # FCF yield
        fcf = fund.get('free_cash_flow')
        mktcap_raw = None
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            mktcap_raw = info.get('marketCap')
        except Exception:
            pass
        if fcf and mktcap_raw and mktcap_raw > 0:
            result['fcf_yield'] = float(np.clip(fcf / mktcap_raw, -0.5, 0.5))
        else:
            result['fcf_yield'] = 0.0

        piotroski = cached_piotroski(ticker) or {}
        result['piotroski_score'] = float(piotroski.get('score', 4.5)) / 9.0

        altman = cached_altman_z(ticker) or {}
        z = altman.get('z_score', None)
        result['altman_z'] = float(z) if z is not None else 0.0
    except ImportError:
        # Not running inside trade_experimentation context — use yfinance directly
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            result['pe_ratio'] = float(np.clip(info.get('trailingPE') or 0, -100, 100))
            result['forward_pe'] = float(np.clip(info.get('forwardPE') or 0, -100, 100))
            result['profit_margin'] = float(info.get('profitMargins') or 0)
            result['roe'] = float(info.get('returnOnEquity') or 0)
            result['debt_to_equity'] = float(np.clip(info.get('debtToEquity') or 0, -50, 50))
            result['revenue_growth'] = float(info.get('revenueGrowth') or 0)
            result['earnings_growth'] = float(info.get('earningsGrowth') or 0)
            fcf = info.get('freeCashflow')
            mktcap = info.get('marketCap')
            result['fcf_yield'] = float(np.clip(fcf / mktcap, -0.5, 0.5)) if fcf and mktcap else 0.0
            result['piotroski_score'] = 0.5  # not computable without full statements
            result['altman_z'] = 0.0
        except Exception:
            pass

    return result


def _fetch_short_interest(ticker: str) -> float:
    try:
        from modules.cached_fetch import cached_short_interest
        si = cached_short_interest(ticker) or {}
        return float(si.get('short_percent_of_float', 0) or 0)
    except ImportError:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return float(info.get('shortPercentOfFloat', 0) or 0)
    except Exception:
        return 0.0


def _fetch_insider_ratio(ticker: str) -> float:
    try:
        import yfinance as yf
        trans = yf.Ticker(ticker).insider_transactions
        if trans is None or trans.empty:
            return 0.0
        if "startDate" in trans.columns:
            trans["date"] = pd.to_datetime(trans["startDate"], errors="coerce")
        elif "Date" in trans.columns:
            trans["date"] = pd.to_datetime(trans["Date"], errors="coerce")
        else:
            return 0.0
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
        trans = trans[trans["date"] >= cutoff]
        if trans.empty:
            return 0.0
        grade_col = next((c for c in ["Text", "transaction"] if c in trans.columns), None)
        if grade_col is None:
            return 0.0
        buys = trans[grade_col].str.contains("Purchase|Buy", case=False, na=False).sum()
        sells = trans[grade_col].str.contains("Sale|Sell", case=False, na=False).sum()
        total = buys + sells
        return float((buys - sells) / total) if total > 0 else 0.0
    except Exception:
        return 0.0


def _fetch_analyst_score(ticker: str) -> float:
    try:
        import yfinance as yf
        rec = yf.Ticker(ticker).recommendations
        if rec is None or rec.empty:
            return 0.0
        recent = rec.tail(10)
        score_map = {
            "strong buy": 2, "buy": 1, "outperform": 1, "overweight": 1,
            "hold": 0, "neutral": 0, "market perform": 0, "equal weight": 0,
            "underperform": -1, "sell": -1, "underweight": -1, "strong sell": -2,
        }
        grade_col = next((c for c in ["To Grade", "toGrade", "Action"] if c in recent.columns), None)
        if grade_col is None:
            return 0.0
        scores = [score_map[g.lower()] for g in recent[grade_col].dropna()
                  if g.lower() in score_map]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def _fetch_put_call_ratio(ticker: str) -> float:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return 0.0
        # Use nearest expiry
        chain = t.option_chain(expiries[0])
        put_vol = chain.puts['volume'].sum()
        call_vol = chain.calls['volume'].sum()
        if call_vol == 0:
            return 0.0
        ratio = float(put_vol / call_vol)
        return float(np.clip(ratio - 1.0, -1, 2))  # center around 1 (neutral=0)
    except Exception:
        return 0.0


def predict_verdict(df: pd.DataFrame, model_dir: str = _DEFAULT_MODEL_DIR,
                    ticker: str | None = None) -> dict:
    if not models_exist(model_dir):
        return {"verdict": None, "error": "Models not trained. Run trade_ml/train.py first."}

    models = _load_models(model_dir)
    if models is None:
        return {"verdict": None, "error": "Failed to load models"}

    features = _compute_features(df, ticker=ticker)
    if features is None:
        return {"verdict": None, "error": "Insufficient data to compute features"}

    feature_cols = models["feature_cols"]
    X = np.array([[features.get(c, 0.0) for c in feature_cols]], dtype=np.float32)

    regime = _current_spy_regime()
    if regime == "bear":
        return {"verdict": "HOLD", "confidence": 0.0, "prob_up": 0.5,
                "regime": "bear", "top_features": [], "error": None,
                "note": "Bear market regime — model withholds BUY calls"}

    model = models["model"]
    # Regressor: predicted 21-day forward return
    pred_return = float(model.predict(X)[0])

    # Convert predicted return to verdict + exit strategy
    # Thresholds derived from OOS quintile analysis: top quintile ~+0.65%/5d
    # At 21d horizon the signal is strongest (rho 0.082)
    atr_pct = features.get("atr_pct", 0.02)
    close = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0

    if pred_return >= 0.02:       # predicted +2%+ in 21d → BUY
        verdict = "BUY"
        confidence = min(pred_return / 0.06, 1.0)  # scale: 6% = full confidence
        stop_pct = max(1.5 * atr_pct, 0.03)
        target_pct = max(pred_return * 1.5, stop_pct * 2)
        exit_strategy = {
            "hold_days": 21,
            "stop_loss_pct": round(stop_pct * 100, 1),
            "target_pct": round(target_pct * 100, 1),
            "stop_price": round(close * (1 - stop_pct), 2) if close else None,
            "target_price": round(close * (1 + target_pct), 2) if close else None,
        }
    elif pred_return <= -0.02:    # predicted -2%+ loss → SELL/AVOID
        verdict = "SELL"
        confidence = min(abs(pred_return) / 0.06, 1.0)
        exit_strategy = {"note": "Avoid or exit existing position"}
    else:
        verdict = "HOLD"
        confidence = 1.0 - abs(pred_return) / 0.02
        exit_strategy = {"note": "No strong directional edge — hold or wait"}

    top_features = []
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        top_idx = np.argsort(np.abs(shap_vals[0]))[::-1][:3]
        top_features = [feature_cols[i] for i in top_idx]
    except Exception:
        pass

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "predicted_return_21d": round(pred_return * 100, 2),  # as percent
        "regime": regime,
        "exit_strategy": exit_strategy,
        "top_features": top_features,
        "error": None,
    }
