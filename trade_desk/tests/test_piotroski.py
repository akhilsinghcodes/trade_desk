import pytest
from modules.piotroski import score_piotroski


def test_strong_score_status():
    data = {"score": 8, "profitability": 4, "leverage": 2, "efficiency": 2, "interpretation": "strong"}
    label, status, text = score_piotroski(data)
    assert label == "Piotroski F-Score"
    assert status == "good"
    assert "8/9" in text


def test_weak_score_status():
    data = {"score": 2, "profitability": 0, "leverage": 1, "efficiency": 1, "interpretation": "weak"}
    label, status, text = score_piotroski(data)
    assert status == "warning"
    assert "2/9" in text


def test_neutral_score_status():
    data = {"score": 5, "profitability": 2, "leverage": 2, "efficiency": 1, "interpretation": "neutral"}
    label, status, text = score_piotroski(data)
    assert status == "neutral"


def test_boundary_scores():
    # 7 = strong threshold
    data7 = {"score": 7, "profitability": 3, "leverage": 3, "efficiency": 1}
    _, status, _ = score_piotroski(data7)
    assert status == "good"

    # 3 = weak threshold
    data3 = {"score": 3, "profitability": 1, "leverage": 1, "efficiency": 1}
    _, status, _ = score_piotroski(data3)
    assert status == "warning"

    # 4 = neutral
    data4 = {"score": 4, "profitability": 2, "leverage": 1, "efficiency": 1}
    _, status, _ = score_piotroski(data4)
    assert status == "neutral"


def test_sub_scores_in_text():
    data = {"score": 6, "profitability": 3, "leverage": 2, "efficiency": 1}
    _, _, text = score_piotroski(data)
    assert "3/4" in text  # profitability
    assert "2/3" in text  # leverage
    assert "1/2" in text  # efficiency


def test_missing_keys_default_zero():
    data = {}  # no keys
    label, status, text = score_piotroski(data)
    assert "0/9" in text
    assert status == "warning"
