import pytest
from modules.thesis import generate_thesis


def _base_thesis(**overrides):
    kwargs = dict(
        ticker="AAPL",
        verdict="BUY",
        confidence=0.70,
        breakdown={"technical": 0.6, "fundamental": 0.5, "sentiment": 0.4},
        signals=[
            ("Revenue Growth", "good", "Revenue up 15% YoY"),
            ("RSI", "good", "RSI 55 — neutral momentum"),
            ("Valuation", "warning", "P/E 35x above sector average"),
        ],
        fundamentals={"profit_margin": 0.25, "revenue_growth_pct": 15.0, "pe_ratio": 35.0, "fcf_yield": 0.06},
        analyst_data={"upside_pct": 25.0, "target_price": 200.0, "consensus": "Buy"},
        piotroski_data={"score": 7, "profitability": 3, "leverage": 2, "efficiency": 2},
        valuation_adv={"ev_ebitda": 20.0},
        market_ctx={"vix": 18.0},
        short_data={"short_pct_float": 2.0},
        earnings_info={"days_away": 45, "date": "2025-10-01"},
        sector_mom={"trend": "uptrend", "name": "Technology", "symbol": "XLK"},
        momentum_data={"trend": "up", "ret_1yr": 30.0},
        altman_data={"z_score": 4.5, "zone": "safe"},
        company_name="Apple Inc.",
    )
    kwargs.update(overrides)
    return kwargs


def test_returns_all_keys():
    result = generate_thesis(**_base_thesis())
    for key in ["headline", "bull_points", "bear_points", "key_risk", "key_catalyst", "one_liner"]:
        assert key in result


def test_buy_headline_positive():
    result = generate_thesis(**_base_thesis(verdict="BUY"))
    assert isinstance(result["headline"], str)
    assert len(result["headline"]) > 0


def test_sell_headline_negative():
    result = generate_thesis(**_base_thesis(
        verdict="SELL / AVOID",
        signals=[("Earnings", "warning", "Earnings miss 3 quarters"), ("RSI", "warning", "RSI 75 overbought")],
    ))
    headline = result["headline"].lower()
    assert any(word in headline for word in ["deteriorat", "downside", "breakdown", "stretched", "risk"])


def test_bull_points_format():
    result = generate_thesis(**_base_thesis())
    for point in result["bull_points"]:
        assert point.startswith("✅")


def test_bear_points_format():
    result = generate_thesis(**_base_thesis())
    for point in result["bear_points"]:
        assert point.startswith("⚠️")


def test_bull_points_count():
    result = generate_thesis(**_base_thesis())
    assert 1 <= len(result["bull_points"]) <= 5


def test_bear_points_count():
    result = generate_thesis(**_base_thesis())
    assert 1 <= len(result["bear_points"]) <= 5


def test_key_risk_earnings_priority():
    # Earnings <14 days should be top risk
    result = generate_thesis(**_base_thesis(earnings_info={"days_away": 7, "date": "2025-08-12"}))
    assert "7" in result["key_risk"] or "earnings" in result["key_risk"].lower()


def test_key_risk_altman_distress():
    result = generate_thesis(**_base_thesis(
        altman_data={"z_score": 1.2, "zone": "distress"},
        earnings_info={"days_away": 60, "date": "2025-10-01"},
    ))
    assert "1.20" in result["key_risk"] or "distress" in result["key_risk"].lower()


def test_key_catalyst_analyst_upside():
    result = generate_thesis(**_base_thesis(analyst_data={"upside_pct": 35.0, "target_price": 220.0}))
    assert "35" in result["key_catalyst"] or "analyst" in result["key_catalyst"].lower()


def test_one_liner_contains_ticker():
    result = generate_thesis(**_base_thesis())
    assert "AAPL" in result["one_liner"]


def test_one_liner_contains_verdict():
    result = generate_thesis(**_base_thesis(verdict="BUY"))
    assert "BUY" in result["one_liner"]


def test_no_signals():
    result = generate_thesis(**_base_thesis(signals=[]))
    # Should still return valid thesis
    assert isinstance(result["headline"], str)
    assert isinstance(result["one_liner"], str)
