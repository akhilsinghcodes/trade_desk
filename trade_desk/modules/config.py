"""Configuration loader for weights, thresholds, and cache TTLs."""

import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None


_DEFAULTS: Dict[str, Any] = {
    "scoring": {
        "weights": {
            "technical": 0.40,
            "fundamental": 0.35,
            "sentiment": 0.25,
        },
        "thresholds": {
            "bullish": 0.25,
            "bearish": -0.25,
        },
    },
    "cache": {
        "ttl": {
            "price": 900,
            "fundamentals": 86400,
            "news": 1800,
            "analyst": 86400,
            "insider": 86400,
            "ownership": 86400,
            "options": 900,
            "earnings": 86400,
            "short": 3600,
            "balance_sheet": 86400,
            "relative_performance": 900,
            "market_context": 900,
            "piotroski": 86400,
            "valuation_advanced": 86400,
            "sector_momentum": 900,
            "dilution_risk": 86400,
            "altman_z": 86400,
            "momentum": 900,
        },
    },
    "backtest": {
        "defaults": {
            "period": "1y",
            "sma_short_window": 50,
            "sma_long_window": 200,
        },
    },
    "data": {
        "providers": ["yfinance", "openbb"],
    },
    "thesis": {
        "thresholds": {
            "piotroski_min_score": 7,
            "fcf_yield_min": 0.05,
            "analyst_upside_major": 30,
            "analyst_upside_moderate": 20,
            "analyst_downside": -10,
            "earnings_days_proximity": 14,
            "short_interest_high": 15,
            "short_interest_low": 3,
            "earnings_growth_min": 20,
            "profit_margin_min": 20,
            "pe_ratio_max": 30,
            "ev_ebitda_max": 25,
            "revenue_growth_min": 0,
        },
    },
}


def _load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}


def _load_config() -> Dict[str, Any]:
    env_override = os.environ.get("TRADELAB_CONFIG")
    if env_override:
        user_config = _load_yaml(env_override)
        if user_config:
            return _deep_merge(_DEFAULTS, user_config)

    default_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if default_path.exists():
        user_config = _load_yaml(str(default_path))
        if user_config:
            return _deep_merge(_DEFAULTS, user_config)

    return _DEFAULTS


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


cfg = _load_config()


def get_weights() -> Dict[str, float]:
    return cfg.get("scoring", {}).get("weights", _DEFAULTS["scoring"]["weights"])


def get_thresholds() -> Dict[str, float]:
    return cfg.get("scoring", {}).get("thresholds", _DEFAULTS["scoring"]["thresholds"])


def get_ttl(key: str) -> int:
    ttls = cfg.get("cache", {}).get("ttl", _DEFAULTS["cache"]["ttl"])
    return ttls.get(key, 3600)


def get_thesis_thresholds() -> Dict[str, Any]:
    return cfg.get("thesis", {}).get("thresholds", _DEFAULTS["thesis"]["thresholds"])


def get_backtest_defaults() -> Dict[str, Any]:
    return cfg.get("backtest", {}).get("defaults", _DEFAULTS["backtest"]["defaults"])


def get_data_providers() -> list:
    return cfg.get("data", {}).get("providers", _DEFAULTS["data"]["providers"])
