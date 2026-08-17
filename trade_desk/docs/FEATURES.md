# TradeDesk — full feature inventory

Every signal, calculation, and feature in the codebase: what it computes, where the data comes from, what page/tab shows it, and whether it's been checked against real outcomes.

**Validated** = backtested with a real win-rate/Sharpe and a split-half stability gate (see `modules/setup_backtest.py`, `modules/factor_analysis.py`).
**Correlation-only** = has a raw Spearman IC check (weaker than validated — see the "Exploratory correlation check" section on Analyze).
**Unvalidated** = presented as data/context, never checked against forward returns — most of these *can't* be checked because they're current-snapshot fundamentals with no point-in-time history available.

---

## Verdict engine (Analyze page, top of screen)

| Feature | What it does | Data source | Where used | Validated? |
|---|---|---|---|---|
| **Composite verdict (BUY/HOLD/SELL)** | Weighted combination of the ~10 price signals below, weighted by each one's own measured IC, sign-corrected per-ticker. `modules/factor_analysis.compute_composite_signal` | OHLCV history | Analyze page hero card | **Validated** — the exact rule is backtested (`setup_backtest.py`, strategy `"VALIDATED"`) with a split-half stability gate |
| **Detected setup type** (BREAKOUT/PULLBACK/MEAN_REVERSION/RANGE/BREAKDOWN) | Swing-structure classification: HH/HL bias, price vs last swing high/low, RSI extremes, volume trend. `modules/setup_detector.py` | OHLCV + indicators | Analyze page, secondary card | Detection logic itself isn't backtested, but... |
| **Setup backtest (win rate / Sharpe / trades)** | Runs a hand-coded entry/exit rule per setup type via vectorbt, split-half stability checked. `modules/setup_backtest.py` | OHLCV history | Analyze page, secondary card | **Validated** when ≥5 trades and stable across both halves; otherwise flagged "No validated edge" |
| **Exploratory correlation check** | Raw Spearman correlation between each of the 10 price signals and forward returns (1/5/10/21d), with direction (normal vs contrarian). `modules/factor_analysis.compute_ic` | OHLCV | Analyze page, collapsed expander | **Correlation-only** — explicitly labeled not a validated edge (small sample, no walk-forward, no costs) |

---

## Technical signals (price-based — the only category that CAN be validated)

| Signal | What it measures | Module | Used in |
|---|---|---|---|
| RSI | Overbought/oversold via relative strength index | `modules/indicators.py` | Verdict composite, correlation check, Chart tab, tech_summary |
| MACD / signal line | Momentum direction and crossover | `modules/indicators.py` | Verdict composite, correlation check, Chart tab |
| SMA 20 / SMA 50 | Trend direction (price vs moving averages, golden/death cross) | `modules/indicators.py` | Verdict composite, correlation check, Chart tab overlay |
| Bollinger Bands (upper/mid/lower, position) | Volatility bands, price position within range | `modules/indicators.py` | Verdict composite, correlation check, Chart tab overlay |
| 5-day / 21-day momentum | Short/medium-term price rate of change | `modules/factor_analysis.py` | Verdict composite, correlation check |
| Trend strength | Linear regression slope of price over 10 bars | `modules/factor_analysis.py` | Verdict composite, correlation check |
| Volatility regime | 10d vs 60d realized volatility ratio | `modules/factor_analysis.py` | Verdict composite, correlation check |
| Volume surge | Volume vs 20d average | `modules/volume.py` | Verdict composite, correlation check, tech_summary |
| Coppock Curve | Long-term momentum oscillator (double-smoothed ROC) | `modules/momentum_signals.py` | Analyze → More tab |
| Ridge regression slope | Smoothed 20-bar trend slope | `modules/momentum_signals.py` | Analyze → More tab |
| PMO relative strength | Price Momentum Oscillator vs SPY | `modules/pmo.py` | Analyze → More tab |
| Support/resistance levels | Swing high/low + pivot points | `modules/support_resistance.py` | Chart tab overlay, alert suggestions |
| SMA crossover backtest | Simple 20/50 crossover strategy, walk-forward | `modules/backtest.py` | Analyze → Backtest tab (opt-in) |

---

## Fundamental signals (current snapshot only — NOT backtestable)

| Signal | What it measures | Module | Used in | Note |
|---|---|---|---|---|
| P/E, Forward P/E, EPS, Revenue, margins | Core valuation/profitability metrics | `modules/fundamentals.py` | Fundamentals tab | yfinance current snapshot |
| Piotroski F-Score (0–9) | 9-point financial health checklist (profitability, leverage, efficiency) | `modules/piotroski.py` | Fundamentals tab | Snapshot-based |
| Altman Z-Score | Bankruptcy risk zone (safe/grey/distress) | `modules/altman_z.py` | Fundamentals tab | Snapshot-based |
| Advanced valuation (FCF yield, EV/EBITDA) | Cheap/fair/expensive classification | `modules/valuation_advanced.py` | Fundamentals tab | Snapshot-based |
| Balance sheet trends | Debt/cash/revenue YoY direction | `modules/balance_sheet_trends.py` | Fundamentals tab | Uses last few annual filings |
| Dilution risk | Share count trend (buybacks vs dilution) | `modules/dilution_risk.py` | tech_summary signal list | Snapshot-based |
| Momentum (1mo/3mo/6mo/1yr returns) | Multi-timeframe price return | `modules/momentum.py` | Fundamentals tab | Actually price-based but not in the IC-checked signal set |

---

## Market/sentiment signals (current snapshot only — NOT backtestable)

| Signal | What it measures | Module | Used in |
|---|---|---|---|
| Analyst consensus + price target | Buy/Hold/Sell rating, mean target, upside % | `modules/analyst.py` | Analyst & Insider tab, header |
| Insider transactions | Recent buy/sell by company insiders | `modules/insider.py` | Analyst & Insider tab |
| Institutional ownership | % held by institutions/insiders | `modules/ownership.py` | Analyst & Insider tab |
| Short interest | % of float shorted, days to cover, squeeze potential | `modules/short_interest.py` | Analyst & Insider tab |
| Options put/call ratio | Contrarian sentiment (fear/greed) | `modules/options_sentiment.py` | Analyst & Insider tab |
| Earnings surprise history | Last 4 quarters beat/miss | `modules/earnings_history.py` | Analyst & Insider tab |
| Next earnings date/proximity | Days until next earnings, risk flag | `modules/earnings.py` | Header chip, dilution_risk scoring |
| Market context (VIX, 52W rank) | Fear gauge + where price sits in 52-week range | `modules/market_context.py` | Analyst & Insider tab |
| Sector momentum | Sector ETF 1-month trend | `modules/sector_momentum.py` | Analyst & Insider tab |
| Sector comparison | Ticker vs sector peer P/E | `modules/sector.py` | Fundamentals tab |
| Relative performance vs S&P 500 | Ticker return vs SPY over period | `modules/relative_performance.py` | Analyze → More tab |
| News sentiment | FinBERT sentiment score on recent headlines (runs locally) | `modules/news.py` | News tab |
| IV30/RV30 volatility ratio | Implied vs realized vol — overpriced/underpriced options | `modules/vol_ratio.py` | Analyze → More tab |

---

## AI / LLM features

| Feature | What it does | Backend | Used in |
|---|---|---|---|
| LLM signal summary | Plain-English bull/bear/conflict summary of the signals above — explicitly NOT a recommendation, no trade_action/conviction/position_size fields | Local (LiteLLM/Ollama) or Haiku (subscription CLI, API key, or pre-computed batch) | Analyze page, below verdict |
| AI Thesis (bull/bear case, catalyst, risk) | Older thesis generator, still driven by the original (unvalidated) hand-weighted score | `modules/thesis.py` | Analyze → More tab, labeled "(unvalidated)" |
| Portfolio batch pre-compute | Pre-fetches Haiku rationale for all portfolio tickers via Anthropic Batches API on server start | `modules/llm_batch.py` | Background job, feeds the cache the Analyze page reads |
| LLM call telemetry | Logs every real LLM call (ticker, backend, source, tokens, cost) | `modules/db.py` `llm_calls` table | LLM Usage page |

---

## App-level features (not per-ticker analysis)

| Feature | What it does | Module | Page |
|---|---|---|---|
| Watchlist | Saved ticker list, auto-refresh, quick verdict | `modules/watchlist.py` + `db.py` | Watchlist |
| Portfolio tracking | Holdings, P&L, allocation pie chart, correlation matrix | `modules/portfolio.py` | Portfolio |
| Price alerts | Target price triggers, desktop push notifications | `modules/alerts.py`, `modules/notifications.py` | Alerts |
| Alert suggestions | Auto-suggests alert levels from support/resistance + analyst targets | `modules/alert_suggestions.py` | Analyze → More tab |
| Screener | Scans up to 20 tickers in parallel, ranked, CSV export | `app/views/screener.py` | Screener |
| ELI5 glossary | Plain-English explanation of every term/signal on the site | `modules/eli5.py` | ELI5 |
| Setup/strategy backtest (standalone) | Signal IC factor analysis + full strategy comparison across all 5 setup types | `modules/factor_analysis.py`, `modules/setup_backtest.py` | Backtest (standalone page) |
| SQLite cache | Per-data-type TTL cache with stale-on-error fallback for every yfinance call | `modules/cached_fetch.py`, `modules/db.py` | Infrastructure |
| Config | Cache TTLs and scoring weights, tunable via `config/settings.yaml` | `modules/config.py` | Infrastructure |

---

## Legacy / superseded (still computed, downstream of the old unvalidated engine)

| Feature | Status |
|---|---|
| `modules/score.combined_score` (old BUY/HOLD/SELL with hand-picked 40/35/25 weights and a "confidence" score) | No longer shown as the primary verdict. Still computed internally and feeds `smart_trade`, `thesis`, and alert suggestions — those consumers are labeled unvalidated where they're user-facing. |
| `modules/trade_strategy.get_smart_trade_strategy` (entry/stop/TP/position-size box) | No longer shown as a primary UI element. Still computed for the "Exit Strategy" expander (Analyze → More), labeled unvalidated. |

---

## Removed this session

- `modules/ml_predict.py` + `models/*.pkl` — dead code, nothing imported it, referenced a sibling `trade_ml` project not part of this repo.
