# Architecture

Two independent sub-projects in one repo. `trade_ml` is the workshop —
train, tweak, backtest. `trade_desk` runs the published result, live, in
its own process. This doc covers how the two connect and the real
engineering decisions (and mistakes) behind that connection.

## Why a monorepo

Not for shared code — `trade_desk` and `trade_ml` don't import each other.
It's one git history and a clear hand-off point (`trade_desk/models/`)
instead of two disconnected folders with no defined relationship. Each
sub-project has its own venv and dependencies; install whichever you're
working on.

## The hand-off: trade_ml → trade_desk

`trade_ml/publish_artifacts.py` copies the trained model
(`return_model.pkl`), the per-ticker backtest results
(`ticker_track_record.json`), and the model's aggregate stats
(`aggregate_stats.json`) into `trade_desk/models/`. That's the entire
contract — three files, no live connection.

**What `trade_desk` does with them, and how, went through two wrong designs
before landing on the right one:**

1. **First attempt — batch predictions only.** `trade_ml` also published a
   `latest_predictions.json` snapshot (one prediction per ticker,
   pre-computed). `trade_desk` just read it. This meant any ticker outside
   `trade_ml`'s 149-ticker test universe got nothing — not because the model
   couldn't handle it, but because nobody had run the batch job for it. Real
   flaw: gated live usefulness behind an offline batch step, for no
   technical reason.

2. **Second attempt — subprocess shell-out.** To cover any ticker, added
   `trade_desk/modules/ml_verdict.py` shelling out via `subprocess` into
   `trade_ml`'s own venv to run inference on demand. Worked, but re-fetched
   price data and recomputed features that `trade_desk` had *already*
   fetched and computed two lines earlier on the same page for the same
   ticker — redundant network calls and redundant computation, plus a
   fragile JSON-over-stdout parsing boundary. Ripped out.

3. **What it actually is now — in-process inference.** `trade_desk` owns a
   real copy of the feature-computation code
   (`trade_desk/modules/ml_features.py`, ported from
   `trade_ml/modules/ml_dataset.py`) and loads `return_model.pkl` directly
   with `joblib`. No subprocess, no re-fetch, no batch gate. Any ticker with
   enough price history gets a live prediction, computed in the same
   process, in ~1-3 seconds.

   **Why a port instead of a cross-repo import:** both `trade_desk` and
   `trade_ml` have a top-level package literally named `modules`. Adding
   `trade_ml/`'s path to `trade_desk`'s `sys.path` would shadow
   `trade_desk`'s own `modules` package — a real, silent collision, not a
   style preference. The port is a real code copy that has to be re-synced
   by hand if `trade_ml`'s feature engineering changes — a maintenance cost,
   traded deliberately for "trade_desk runs the model independently at
   request time," which is what was actually asked for.

**Per-ticker track record vs aggregate fallback:** `ticker_track_record.json`
only has real walk-forward-validated numbers for the 149 tickers `trade_ml`
has backtested. For any other ticker, `trade_desk` shows the model's honest
aggregate stats (113/149 tickers positive, 57/149 significant, median
ρ=+0.068) instead of a fabricated ticker-specific claim — labeled
differently in the UI ("per-ticker validated" vs "live") so the two aren't
confused.

## The OpenMP crash (a real, evidence-backed bug, not a guess)

Adding `xgboost` to `trade_desk` for in-process inference caused
intermittent `SIGSEGV` crashes — confirmed via macOS crash reports
(`~/Library/Logs/DiagnosticReports/`), not assumed. Root cause, verified
with `otool`/`lsof`, not guessed:

- `torch` (used for FinBERT sentiment) bundles its own `libomp.dylib`.
- `xgboost`'s `libxgboost.dylib` has a hardcoded `LC_RPATH` pointing at
  `/opt/homebrew/opt/libomp/lib` — a **separate** copy of the OpenMP
  runtime.
- Two live OpenMP runtimes in one process is a known class of crash — this
  one segfaulted inside PyTorch's `layer_norm` during a thread-pool
  join/barrier.

**Fix:** `install_name_tool -rpath` on the venv's own copy of
`libxgboost.dylib`, repointing it at `torch`'s bundled `libomp.dylib`
(same ABI compatibility version, confirmed via `otool -L`) instead of the
Homebrew one. Both libraries now share a single OpenMP runtime instance —
verified via `lsof` on a running interpreter that only one `xgboost`-linked
`libomp` loads, and confirmed no new crash report after loading a ticker
that exercises both FinBERT and the ML model in the same request.
Contained to this project's `.venv` — doesn't touch the system Homebrew
install. Scoped fix, not a suppression (`KMP_DUPLICATE_LIB_OK=TRUE` was
considered and rejected — it papers over the conflict instead of resolving
it).

## The composite verdict vs the ML model — two independent signals

The BUY/HOLD/SELL banner and the ML model's prediction are built from
different signal sets (a handful of price-action oscillators vs ~57
technical/fundamental/sentiment features), different backtest methodologies,
and can legitimately disagree. Measured empirically (151,745 OOS rows, 73
tickers, full walk-forward): when the two agree, the ML model's Spearman ρ
is +0.072 (real signal); when they disagree, it's -0.014 (statistically
dead). The app surfaces this as an explicit agreement/disagreement flag next
to the ML prediction — disagreement isn't noise to hide, it's a validated
low-confidence regime.

## The ATR trade-level scheme

`suggest_trade()`'s entry/stop/target used a 1.5×/3× ATR scheme that had
never been backtested. Validated this session via a 442k-trade, 149-ticker,
2011–2026 study: the scheme has real positive expected value (+0.14R blind),
survives realistic trading costs to ~20-50bps. A subsequent grid search
found 1.0×/3.0× beats the original on every axis (gross EV +0.26R vs
+0.15R, 14/16 vs 13/16 years positive) without over-tightening into
unrealistic stop distances — confirmed by checking that the improvement
wasn't a labeling artifact (same-day stop/target ambiguity ruled out
directly) before trusting it. `suggest_trade()` now uses the validated
multipliers and shows the backtested win rate/EV alongside the levels.

## What's still not built

- No scheduled refresh of `trade_ml`'s published artifacts — running
  `publish_artifacts.py` / `refresh_live_predictions.py` is still manual.
- `trade_desk/modules/score.py` (`combined_score()`, the old hand-weighted
  verdict — proven statistically to be noise, ρ=-0.0046, p=0.64, in head-to-
  head testing against the ML model) is superseded by `factor_analysis.py`
  for the displayed BUY/HOLD/SELL banner and no longer feeds
  `suggest_trade()`'s ATR levels (that was a real bug, fixed this session —
  see git history). It's still called for the AI thesis, smart-trade sizing,
  and LLM rationale generation, which haven't been migrated off it yet —
  a real, not-yet-closed gap.
