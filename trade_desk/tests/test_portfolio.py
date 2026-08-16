import pytest
from modules.portfolio import calc_pnl, portfolio_summary


def test_calc_pnl_profit():
    pos = {"ticker": "AAPL", "shares": 10, "buy_price": 100.0}
    result = calc_pnl(pos, 150.0)
    assert result["cost_basis"] == 1000.0
    assert result["current_value"] == 1500.0
    assert result["pnl_dollars"] == 500.0
    assert result["pnl_pct"] == pytest.approx(50.0)
    assert result["status"] == "profit"


def test_calc_pnl_loss():
    pos = {"ticker": "TSLA", "shares": 5, "buy_price": 200.0}
    result = calc_pnl(pos, 100.0)
    assert result["pnl_dollars"] == pytest.approx(-500.0)
    assert result["pnl_pct"] == pytest.approx(-50.0)
    assert result["status"] == "loss"


def test_calc_pnl_breakeven():
    pos = {"ticker": "MSFT", "shares": 1, "buy_price": 300.0}
    result = calc_pnl(pos, 300.0)
    assert result["status"] == "breakeven"
    assert result["pnl_dollars"] == pytest.approx(0.0)


def test_calc_pnl_fractional_shares():
    pos = {"ticker": "NVDA", "shares": 0.5, "buy_price": 800.0}
    result = calc_pnl(pos, 1000.0)
    assert result["cost_basis"] == pytest.approx(400.0)
    assert result["current_value"] == pytest.approx(500.0)
    assert result["status"] == "profit"


def test_portfolio_summary_basic():
    positions = [
        {"ticker": "AAPL", "shares": 10, "buy_price": 100.0},
        {"ticker": "MSFT", "shares": 5, "buy_price": 200.0},
    ]
    prices = {"AAPL": 150.0, "MSFT": 250.0}
    summary = portfolio_summary(positions, prices)
    assert summary["total_cost"] == pytest.approx(2000.0)
    assert summary["total_value"] == pytest.approx(2750.0)
    assert summary["total_pnl_dollars"] == pytest.approx(750.0)
    assert summary["total_pnl_pct"] == pytest.approx(37.5)
    assert len(summary["positions"]) == 2


def test_portfolio_summary_missing_price():
    positions = [
        {"ticker": "AAPL", "shares": 10, "buy_price": 100.0},
        {"ticker": "UNKNOWN", "shares": 5, "buy_price": 50.0},
    ]
    prices = {"AAPL": 120.0}  # UNKNOWN has no price
    summary = portfolio_summary(positions, prices)
    assert len(summary["positions"]) == 1  # only AAPL included


def test_portfolio_summary_empty():
    summary = portfolio_summary([], {})
    assert summary["total_cost"] == 0.0
    assert summary["total_value"] == 0.0
    assert summary["total_pnl_pct"] == 0.0
