import json

import pandas as pd
import pytest

import modules.ml_verdict as mv


class _FakeModel:
    def predict(self, X):
        return [0.01234]


def _fake_live_features(ticker, period="1y"):
    return pd.DataFrame({"date": [pd.Timestamp("2026-08-14")], **{c: [0.0] for c in mv.FEATURE_COLS}})


@pytest.fixture
def isolated_models_dir(tmp_path, monkeypatch):
    (tmp_path / "return_model.pkl").write_bytes(b"not a real pickle, load is mocked")
    monkeypatch.setattr(mv, "_MODELS_DIR", tmp_path)
    monkeypatch.setattr(mv, "compute_live_features", _fake_live_features)
    monkeypatch.setattr(mv, "_load_model", lambda: _FakeModel())
    return tmp_path


def test_published_ticker_gets_real_track_record(isolated_models_dir):
    track = {"AAPL": {"rho": 0.15, "pval": 0.002, "significant": True,
                       "direction": "positive", "n_folds": 435}}
    (isolated_models_dir / "ticker_track_record.json").write_text(json.dumps(track))

    result = mv.get_ml_verdict("AAPL")
    assert result["has_track_record"] is True
    assert result["track_rho"] == 0.15
    assert result["pred_return_5d"] == 0.01234


def test_unpublished_ticker_falls_back_to_aggregate(isolated_models_dir):
    (isolated_models_dir / "ticker_track_record.json").write_text(json.dumps({}))
    agg = {"n_tickers_tested": 149, "n_positive": 113, "n_significant": 57, "median_rho": 0.068}
    (isolated_models_dir / "aggregate_stats.json").write_text(json.dumps(agg))

    result = mv.get_ml_verdict("SIRI")
    assert result["has_track_record"] is False
    assert result["agg_n_tickers_tested"] == 149
    assert "track_rho" not in result


def test_missing_model_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(mv, "_MODELS_DIR", tmp_path)  # no return_model.pkl written
    assert mv.get_ml_verdict("AAPL") is None


def test_insufficient_history_returns_none(isolated_models_dir, monkeypatch):
    monkeypatch.setattr(mv, "compute_live_features", lambda ticker, period="1y": None)
    assert mv.get_ml_verdict("BRANDNEW") is None
