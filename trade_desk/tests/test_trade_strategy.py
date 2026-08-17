from modules.trade_strategy import get_smart_trade_strategy


def _base_kwargs(**overrides):
    kwargs = dict(
        current_price=100.0,
        atr=2.0,
        confidence=0.70,
        verdict="BUY",
        breakdown={"technical": 0.6, "fundamental": 0.5, "sentiment": 0.4},
        support_levels=[95.0, 90.0, 85.0],
        resistance_levels=[105.0, 110.0, 120.0],
        pivot=100.0,
        analyst_target=115.0,
        analyst_upside_pct=15.0,
        week52_high=130.0,
        week52_low=70.0,
        week52_rank=0.50,
        piotroski_score=7,
        altman_zone="safe",
        momentum_trend="up",
        ret_1yr=20.0,
        short_pct=5.0,
        earnings_days_away=60,
        sector_trend="uptrend",
    )
    kwargs.update(overrides)
    return kwargs


def test_buy_high_conviction_limit_near_current():
    result = get_smart_trade_strategy(**_base_kwargs())
    # High conviction BUY with momentum — limit should be within 2% of current
    assert result["limit_entry"] < 100.0
    assert result["limit_entry"] > 97.0


def test_hold_limit_very_near_current():
    result = get_smart_trade_strategy(**_base_kwargs(verdict="HOLD / WATCH", confidence=0.70))
    # HOLD high conviction — max 0.5% discount
    assert result["limit_entry"] >= 99.0


def test_stop_loss_below_entry():
    result = get_smart_trade_strategy(**_base_kwargs())
    assert result["stop_loss"] < result["limit_entry"]


def test_tp1_above_entry():
    result = get_smart_trade_strategy(**_base_kwargs())
    assert result["take_profit_1"] > result["limit_entry"]


def test_tp2_above_tp1():
    result = get_smart_trade_strategy(**_base_kwargs())
    assert result["take_profit_2"] >= result["take_profit_1"]


def test_risk_reward_positive():
    result = get_smart_trade_strategy(**_base_kwargs())
    assert result["risk_reward"] > 0


def test_conviction_tiers():
    high = get_smart_trade_strategy(**_base_kwargs(confidence=0.70))
    med = get_smart_trade_strategy(**_base_kwargs(confidence=0.50))
    low = get_smart_trade_strategy(**_base_kwargs(confidence=0.30))
    assert high["conviction"] == "high"
    assert med["conviction"] == "medium"
    assert low["conviction"] == "low"
    # Lower conviction → bigger discount (lower entry)
    assert high["limit_entry"] >= med["limit_entry"]
    assert med["limit_entry"] >= low["limit_entry"]


def test_near_earnings_tightens_stop():
    normal = get_smart_trade_strategy(**_base_kwargs(earnings_days_away=60))
    near = get_smart_trade_strategy(**_base_kwargs(earnings_days_away=7))
    # Near earnings: stop tightened to 3% max, and limit has earnings buffer
    assert near["stop_loss"] >= near["limit_entry"] * 0.97 - 0.01
    # Actually tighter than the non-earnings case, not just under the cap
    near_risk_pct = (near["limit_entry"] - near["stop_loss"]) / near["limit_entry"]
    normal_risk_pct = (normal["limit_entry"] - normal["stop_loss"]) / normal["limit_entry"]
    assert near_risk_pct <= normal_risk_pct


def test_distress_zone_adds_discount():
    safe = get_smart_trade_strategy(**_base_kwargs(altman_zone="safe"))
    distress = get_smart_trade_strategy(**_base_kwargs(altman_zone="distress"))
    assert distress["discount_pct"] > safe["discount_pct"]


def test_exit_conditions_nonempty():
    result = get_smart_trade_strategy(**_base_kwargs())
    assert len(result["exit_conditions"]) >= 3


def test_no_support_levels():
    result = get_smart_trade_strategy(**_base_kwargs(support_levels=[], resistance_levels=[]))
    # Should still return valid result using ATR
    assert result["limit_entry"] > 0
    assert result["stop_loss"] > 0


def test_returns_all_required_keys():
    result = get_smart_trade_strategy(**_base_kwargs())
    for key in ["limit_entry", "stop_loss", "take_profit_1", "take_profit_2",
                "risk_reward", "exit_conditions", "time_horizon", "conviction", "discount_pct"]:
        assert key in result
