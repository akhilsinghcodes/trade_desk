# Full Signal/Feature Catalog — trade_desk + trade_ml

Every signal built across both repos, exact formula, and provenance
(established literature/library vs. our own implementation). Pulled from
reading the actual code, not memory.

---

## trade_ml — `modules/ml_dataset.py` FEATURE_COLS (54 total)

### Technical — original 11 (pre-session), own implementation using FinTA + custom math

| Feature | Formula | Source |
|---|---|---|
| `rsi` | FinTA `TA.RSI()`, clipped [0,100]/100 | Standard RSI (Wilder, 1978), via `finta` library |
| `macd_above_signal` | `1.0 if MACD > MACD_signal else 0.0` (FinTA `TA.MACD()`) | Standard MACD (Appel, 1970s), via `finta` |
| `bb_position` | `(close - bb_lower) / (bb_upper - bb_lower)`, clip [0,1] | Standard Bollinger Bands, via `finta` |
| `sma20_vs_sma50` | `sma20/sma50 - 1`, clip [-0.5,0.5] | Standard SMA crossover, via `finta` |
| `volume_surge_20d` | `volume / vol_ma20 - 1`, clip [-1,3] | Own implementation |
| `atr_pct` | `rolling(14).mean(true_range) / close`, clip [0,0.2] | Standard ATR (Wilder), own implementation |
| `cmf_20d` | Chaikin Money Flow: `Σ(mf_mult × volume, 20d) / Σ(volume, 20d)` | Standard CMF (Chaikin), own implementation |
| `roc_21d` | `close.pct_change(21)`, clip [-0.5,0.5] | Standard Rate of Change |
| `alpha_21d` | `stock_ret_21d - SPY_ret_21d` | Own implementation |
| `sector_alpha_21d` | `stock_ret_21d - sector_ETF_ret_21d` (via `SECTOR_ETF_MAP`) | Own implementation |
| `high_52w_proximity` | `close / rolling(252, min_periods=252).max()`, clip [0,1] | Own implementation |

### Technical — 30 new this session, ported from Qlib's Alpha158

Source: Microsoft Qlib (`github.com/microsoft/qlib`), Alpha158 factor set,
cross-referenced against a public reimplementation
(`github.com/vnpy/vnpy/blob/master/vnpy/alpha/dataset/datasets/alpha_158.py`).
Formulas below are as actually implemented in `_compute_ticker_features()`
(may be simplified/approximated vs. Qlib's exact expression engine — see
notes).

| Feature | Formula | Notes |
|---|---|---|
| `kmid` | `(close - open) / open` | Candlestick body |
| `klen` | `(high - low) / open` | Candlestick range |
| `kup` | `(high - max(open, close)) / open` | Upper shadow |
| `klow` | `(min(open, close) - low) / open` | Lower shadow |
| `ksft` | `(2×close - high - low) / open` | Close position within range |
| `roc_5`, `roc_10`, `roc_60` | `close.pct_change(w)` | Multi-window momentum |
| `ma_5`, `ma_10`, `ma_20`, `ma_60` | `rolling(w).mean(close) / close` | Multi-window trend |
| `std_5`, `std_10`, `std_20`, `std_60` | `rolling(w).std(close) / close` | Multi-window volatility |
| `beta_5`, `beta_10`, `beta_20` | `(close - close.shift(w)) / (w × close)` | **Approximated** as simple slope, not a true OLS regression coefficient like Qlib's actual BETA factor |
| `rsqr_10`, `rsqr_20` | `np.polyfit(t, close, deg=1)[0] / close` | **Mislabeled** — this computes normalized slope, not R², despite the name. `resi_10`/`resi_20` (mean abs residual from linear fit) were also computed but never added to `TECHNICAL_FEATURES` — dead code, unused |
| `max_20` | `rolling(20).max(high) / close` | Proximity to 20d high |
| `min_20` | `rolling(20).min(low) / close` | Proximity to 20d low |
| `qtlu_10` | `rolling(10).quantile(0.8, close) / close` | 80th percentile position |
| `qtld_10` | `rolling(10).quantile(0.2, close) / close` | 20th percentile position |
| `rank_10` | `(x[-1] > x).sum() / 10` over rolling 10d window | Percentile rank of today's close |
| `corr_10` | `rolling(10).corr(close.pct_change(), volume.pct_change())` | Price-volume correlation |
| `cntp_10` | `(diff(close, 10d) > 0).sum() / 10` | Fraction of up-days |
| `cntn_10` | `(diff(close, 10d) < 0).sum() / 10` | Fraction of down-days |
| `vma_20` | `rolling(20).mean(volume) / (close × 10000)` | Volume level, scaled |

**Known implementation gaps vs. real Alpha158** (worth knowing if reused):
`rsqr` factors don't actually compute R² despite the name. `beta` factors
use a 2-point slope approximation, not a proper rolling OLS regression —
Qlib's real BETA factor is `Slope(close, w)` from a true linear regression.
If revisiting this, worth fixing the naming/math mismatch or re-deriving
properly.

### Fundamental — 10, own implementation via `yfinance`

| Feature | Formula | Source |
|---|---|---|
| `pe_ratio` | `yf.info["trailingPE"]`, clip [-100,100] | Raw yfinance field |
| `forward_pe` | `yf.info["forwardPE"]`, clip [-100,100] | Raw yfinance field |
| `profit_margin` | `yf.info["profitMargins"]` | Raw yfinance field |
| `roe` | `yf.info["returnOnEquity"]` | Raw yfinance field |
| `debt_to_equity` | `yf.info["debtToEquity"]`, clip [-50,50] | Raw yfinance field |
| `revenue_growth` | `yf.info["revenueGrowth"]` | Raw yfinance field |
| `earnings_growth` | `yf.info["earningsGrowth"]` | Raw yfinance field |
| `fcf_yield` | `freeCashflow / marketCap`, clip [-0.5,0.5] | Own derivation |
| `piotroski_score` | 9-point score (see below) / 9.0 | Piotroski (2000), own implementation |
| `altman_z` | Z = 1.2×X1+1.4×X2+3.3×X3+0.6×X4+1.0×X5, clip [-5,15] | Altman (1968), own implementation |

**Static snapshot** — current value applied to every historical row for
that ticker (known, documented limitation; fixed to use NaN+cross-sectional-
median instead of a fabricated 0.0 for missing values, this session).

### Sentiment — 3, own implementation (put_call_ratio removed this session — was hardcoded 0.0, dead)

| Feature | Formula | Source |
|---|---|---|
| `short_interest_pct` | `yf.info["shortPercentOfFloat"]` | Static current-value snapshot, applied to all historical rows — **known unfixed issue**, more time-varying than fundamentals |
| `insider_net_ratio` | Time-aware, built from historical Form 4 transactions (`_insider_series`) | Own implementation, does NOT detect clustering |
| `analyst_score` | Time-aware, built from historical recommendation history (`_analyst_series`) | Own implementation |

---

## trade_ml — PEAD features (`modules/pead_dataset.py`, `PEAD_FEATURES`)

One row per earnings event (not per-day). Target: `WIN` if stock
outperforms SPY by >3% in 30 trading days post-earnings.

| Feature | Formula |
|---|---|
| `surprise_pct` | `(Reported EPS - EPS Estimate) / |EPS Estimate|` via yfinance `get_earnings_dates()`, clip [-0.5,0.5] |
| `prev_surprise_pct` | Same, previous quarter |
| `surprise_acceleration` | `surprise_pct - prev_surprise_pct` |
| `pre_ret_1m_vs_spy` | Stock return 1mo pre-earnings minus SPY return, same window |
| `pre_ret_3m_vs_spy` | Stock return 3mo pre-earnings minus SPY return, same window |
| `atr_pct`, `high_52w_proximity` | Same formulas as above, `.asof(earnings_date)` |
| `revenue_growth`, `profit_margin`, `roe`, `fcf_yield`, `piotroski_score` | Same as ml_dataset.py fundamentals, static snapshot |

Source: PEAD is Bernard & Thomas (1989) and 35+ years of subsequent
literature — one of the most persistent documented anomalies. Our
implementation and test (mega-cap universe) is untested against the
segment (small/mid-cap) where the published edge actually concentrates.

---

## trade_desk — signal modules (per-ticker, real-time, `app/services/analysis.py`)

All of these feed `modules/score.py:combined_score()` as `(label, status,
text)` tuples — converted to +1/0/-1 and averaged, see prior summary for
why that combining logic is unvalidated.

| Module | What it computes | Formula/source |
|---|---|---|
| `momentum_signals.py: compute_coppock` | Coppock Curve | `WMA_10(ROC_14 + ROC_11)`, standard (Coppock, 1962) |
| `momentum_signals.py: compute_ridge_slope` | Price trend slope | `np.polyfit` linear slope over last 20 bars, normalized |
| `volume.py` | Volume confirmation | `vol_ratio > 1.5` + price direction, own heuristic |
| `vol_ratio.py: compute_vol_ratio` | IV30/RV30 | `iv30/rv30` from options chain ATM strikes vs. `close.pct_change().std()×√252`, own derivation |
| `short_interest.py` | Short % of float, days-to-cover | Raw `yf.info` fields |
| `analyst.py` | Consensus, price target upside | Raw `yf.info` fields |
| `options_sentiment.py` | Put/call ratio | `put_volume/call_volume` from nearest-expiry chain, own contrarian heuristic (PCR>1.2→bullish, <0.6→bearish) |
| `insider.py` | Insider buy/sell ratio | Own implementation, no cluster detection |
| `ownership.py` | Institutional ownership % | Raw `yf.info` fields |
| `relative_performance.py` | Stock vs. SPY/sector relative return | Own derivation |
| `sector_momentum.py` | Sector ETF momentum | Own derivation |
| `dilution_risk.py` | Share count trend | Own derivation from `yf` balance sheet |
| `valuation_advanced.py` | P/S, EV/EBITDA, PEG | Raw `yf.info` fields, standard ratios |
| `balance_sheet_trends.py` | Debt/equity trend, current ratio trend | Own derivation from `yf` balance sheet history |
| `earnings_history.py` | Historical beat/miss rate | Own derivation from `yf.get_earnings_dates()` |
| `momentum.py` | General momentum score | Own composite |
| `market_context.py` | VIX level, market regime | Raw `yf` VIX fetch |
| `piotroski.py: get_piotroski` | 9-point F-Score | Piotroski (2000): ROA>0, OCF>0, ΔROA>0, OCF>NI (accruals), ΔLT-debt-ratio<0, ΔCurrent-ratio>0, no new shares issued, ΔGross-margin>0, ΔAsset-turnover>0 — 1 point each |
| `altman_z.py: get_altman_z` | Z-Score | Altman (1968) public-company formula, 5 components (X1-X5) from `yf` balance sheet + income statement |
| `support_resistance.py: suggest_trade` | Entry/stop/target | `stop=close-1.5×ATR`, `TP1=close+2×risk`, `TP2=close+3×risk` — **own constants, never backtested** |

---

## Summary: what's borrowed vs. built vs. broken

- **Established, correctly-implemented, real literature behind them**: RSI, MACD,
  Bollinger Bands, ATR, CMF (all via FinTA, standard formulas), Piotroski
  F-Score, Altman Z-Score, PEAD framing.
- **Borrowed this session, real provenance, some approximation gaps**: 30
  Alpha158-style factors from Qlib (beta/rsqr are approximated, not exact
  Qlib math — flagged above).
- **Own heuristics, no external validation, work as intended**: alpha_21d,
  sector_alpha_21d, volume_surge_20d, vol_ratio's IV/RV interpretation
  thresholds, options PCR contrarian thresholds.
- **Own heuristics, proven equivalent to random noise this session**:
  `combined_score()`'s weights (40/35/25) and threshold (0.25).
- **Own constants, never tested at all**: `suggest_trade()`'s 1.5×/2×/3×
  ATR stop/target multiples.
- **Dead/removed**: `put_call_ratio` in trade_ml (was hardcoded 0.0, deleted).
