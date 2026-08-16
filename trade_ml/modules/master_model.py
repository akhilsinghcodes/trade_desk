"""
Sub-Models 1-4 + Model 5 (The Judge).

    [STOCK DATA] -> Sub-Model 1: Momentum & Trend   ─┐
                 -> Sub-Model 2: Squeeze & Volatility ┼-> Model 5: THE JUDGE
                 -> Sub-Model 3: Catalyst & Sentiment ┤   (Math Ranker + LLM Arbiter)
                 -> Sub-Model 4: Fundamental Safety  ─┘

All 4 sub-models are cross-sectional z-score composites (same treatment,
no hard gating — safety is just another pillar the Math Ranker weighs):

    Sub-Model 1 (momentum): rsi, macd_above_signal, bb_position, sma20_vs_sma50,
                             volume_surge_20d, roc_21d, alpha_21d, sector_alpha_21d,
                             high_52w_proximity, cmf_20d, kmid, klen, kup, klow, ksft,
                             roc_5, roc_10, roc_60, ma_5, ma_10, ma_20, ma_60,
                             beta_5, beta_10, beta_20, max_20, min_20, rank_10,
                             cntp_10, cntn_10
    Sub-Model 2 (squeeze) : short_interest_pct, std_5, std_10, std_20, std_60,
                             rsqr_10, rsqr_20, corr_10, vma_20, qtlu_10, qtld_10
    Sub-Model 3 (catalyst): analyst_score, insider_net_ratio
    Sub-Model 4 (safety)  : altman_z

Math Ranker: math_score = intercept + sum(beta_i * pillar_i), betas fit via
closed-form ridge regression against 5-day forward return (see
return_model.add_return_target) — no extra ML dep for a 4-5 column matrix.

LLM Arbiter: given the 4 sub-scores + underlying signals, an LLM call can
nudge math_score up or down and must give a one-line reason. Runs offline in
a batch (run_judge), not per-request — output (score + reason per ticker) is
persisted alongside the weights so trade_desk just reads it.
"""
import json
import os
import urllib.request
import numpy as np
import pandas as pd

from modules.ml_dataset import FEATURE_COLS
from modules.return_model import add_return_target

PILLAR_FEATURES = {
    "momentum": ["rsi", "macd_above_signal", "bb_position", "sma20_vs_sma50",
                 "volume_surge_20d", "roc_21d", "alpha_21d", "sector_alpha_21d",
                 "high_52w_proximity", "cmf_20d", "kmid", "klen", "kup", "klow", "ksft",
                 "roc_5", "roc_10", "roc_60", "ma_5", "ma_10", "ma_20", "ma_60",
                 "beta_5", "beta_10", "beta_20", "max_20", "min_20", "rank_10",
                 "cntp_10", "cntn_10"],
    "squeeze": ["short_interest_pct", "std_5", "std_10", "std_20", "std_60",
                "rsqr_10", "rsqr_20", "corr_10", "vma_20", "qtlu_10", "qtld_10"],
    "catalyst": ["analyst_score", "insider_net_ratio"],
    "safety": ["altman_z"],
}
PILLARS = tuple(PILLAR_FEATURES)
WEIGHTS_FILE = "master_model_weights.json"
JUDGE_MODEL = "claude-haiku-4-5-20251001"


def _cross_sectional_zscores(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Z-score each feature column within each date (across tickers that day)."""
    z = df.groupby("date")[cols].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
    return z.fillna(0.0)


def _pillar_frame(df: pd.DataFrame) -> pd.DataFrame:
    present = {p: [f for f in feats if f in df.columns] for p, feats in PILLAR_FEATURES.items()}
    z = _cross_sectional_zscores(df, [f for feats in present.values() for f in feats])
    pillars = pd.DataFrame(index=df.index)
    for p, feats in present.items():
        pillars[p] = z[feats].mean(axis=1) if feats else 0.0
    return pillars


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """w = (X^T X + alpha*I)^-1 X^T y, with an unregularized intercept column appended."""
    X1 = np.column_stack([X, np.ones(len(X))])
    reg = alpha * np.eye(X1.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(X1.T @ X1 + reg, X1.T @ y)


# ── Sub-models 1-4 + Math Ranker (fit/persist/apply) ────────────────────────

def train_master_model(df: pd.DataFrame, model_dir: str = "models/", alpha: float = 1.0) -> dict:
    """
    df: output of ml_dataset.build_training_dataset() (needs 'ticker', 'date',
    'close', and FEATURE_COLS columns). Fits the 4 pillar betas against 5-day
    forward return and persists weights.
    """
    os.makedirs(model_dir, exist_ok=True)
    df = add_return_target(df)

    pillars = _pillar_frame(df)
    weights = _fit_ridge(pillars[list(PILLARS)].values, df["fwd_return"].values, alpha=alpha)
    betas, intercept = weights[:-1], float(weights[-1])

    payload = {
        "pillars": list(PILLARS),
        "pillar_features": PILLAR_FEATURES,
        "betas": betas.tolist(),
        "intercept": intercept,
        "n_samples": len(df),
    }
    with open(os.path.join(model_dir, WEIGHTS_FILE), "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  Saved: {model_dir}/{WEIGHTS_FILE} ({len(df):,} rows, betas={dict(zip(PILLARS, betas.round(4)))})")
    return payload


def load_master_model(model_dir: str = "models/") -> dict | None:
    path = os.path.join(model_dir, WEIGHTS_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def math_rank(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    df: current-day feature snapshot across a universe (needs 'date' + FEATURE_COLS).
    Returns pillar scores + math_score, indexed like df.
    """
    pillars = _pillar_frame(df)
    betas = np.array(weights["betas"])
    pillars["math_score"] = weights["intercept"] + pillars[weights["pillars"]].values @ betas
    return pillars


# ── Model 5: The Judge (Math Ranker output -> LLM Arbiter) ──────────────────

def _llm_arbiter(ticker: str, row: pd.Series) -> tuple[float, str]:
    """
    Ask the LLM to review the 4 sub-scores + math_score for one ticker and
    return an adjusted score + one-line reason. Falls back to the unadjusted
    math_score (reason: no adjustment) if no API key or the call fails —
    the Judge must always produce a score, arbitration is best-effort.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return float(row["math_score"]), "No LLM arbitration (ANTHROPIC_API_KEY not set)."

    prompt = (
        f"Ticker {ticker}. Quant sub-model z-scores (higher = stronger):\n"
        f"  Momentum & Trend: {row['momentum']:.2f}\n"
        f"  Squeeze & Volatility: {row['squeeze']:.2f}\n"
        f"  Catalyst & Sentiment: {row['catalyst']:.2f}\n"
        f"  Fundamental Safety: {row['safety']:.2f}\n"
        f"  Math Ranker score: {row['math_score']:.4f}\n\n"
        "Do the sub-scores agree or contradict each other? If they broadly agree, "
        "keep the score close to the Math Ranker's. If they contradict (e.g. strong "
        "momentum but very weak safety), adjust the score to reflect the added risk. "
        "Reply with exactly two lines:\n"
        "SCORE: <adjusted numeric score>\n"
        "REASON: <one short sentence>"
    )
    payload = json.dumps({
        "model": JUDGE_MODEL,
        "max_tokens": 150,
        "system": "You are a quant risk arbiter. Be terse and numeric.",
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = json.loads(resp.read())["content"][0]["text"].strip()
        score_line, reason_line = text.split("\n", 1)
        score = float(score_line.split(":", 1)[1].strip())
        reason = reason_line.split(":", 1)[1].strip()
        return score, reason
    except Exception:
        return float(row["math_score"]), "LLM arbitration failed; using Math Ranker score unadjusted."


def run_judge(df: pd.DataFrame, weights: dict, use_llm: bool = True) -> pd.DataFrame:
    """
    Model 5: Math Ranker (math_rank) + LLM Arbiter, run offline over a
    universe snapshot. Returns a DataFrame with pillar scores, math_score,
    judge_score, and reason — ranked by judge_score. This is what gets
    persisted for trade_desk to read; nothing here is called per-request.
    """
    ranked = math_rank(df, weights)
    if not use_llm:
        ranked["judge_score"] = ranked["math_score"]
        ranked["reason"] = "LLM arbitration skipped."
        return ranked.sort_values("judge_score", ascending=False)

    scores, reasons = [], []
    for ticker, row in ranked.iterrows():
        s, r = _llm_arbiter(ticker, row)
        scores.append(s)
        reasons.append(r)
    ranked["judge_score"] = scores
    ranked["reason"] = reasons
    return ranked.sort_values("judge_score", ascending=False)
