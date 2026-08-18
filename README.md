# Trade Lab (monorepo)

Two independent sub-projects, each with its own dependencies and virtualenv:

- **[trade_desk/](trade_desk/)** — Streamlit stock-research app. See [trade_desk/README.md](trade_desk/README.md).
- **[trade_ml/](trade_ml/)** — model training/research. Trains on a broad multi-stock dataset; models are evaluated and used per-ticker, not as a cross-sectional ranker. See [trade_ml/README.md](trade_ml/README.md).

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for how the two connect — the
publish/hand-off pipeline, why `trade_desk` runs the model in-process
instead of importing `trade_ml`, and the real bugs (and fixes) hit getting
there.

## Setup

Each sub-project is installed independently:

```bash
cd trade_desk && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd trade_ml && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

You don't need both — install whichever you're working on.

## Artifact hand-off

`trade_ml/publish_artifacts.py` publishes the trained model
(`return_model.pkl`) and its backtested track record
(`ticker_track_record.json`, `aggregate_stats.json`) into
`trade_desk/models/`. `trade_desk` runs the model itself from there —
in its own process, for any ticker, using its own ported copy of the
feature-computation code (`trade_desk/modules/ml_features.py`) — not a
live import of `trade_ml`. Run `refresh_live_predictions.py` in `trade_ml`
to update the published snapshot with current-day data; details in
[ARCHITECTURE.md](ARCHITECTURE.md).
