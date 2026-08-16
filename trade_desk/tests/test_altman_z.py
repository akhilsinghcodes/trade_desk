from modules.altman_z import score_altman_z


def test_safe_zone_status():
    data = {"z_score": 3.5, "zone": "safe", "interpretation": "Z-Score 3.50 — safe zone."}
    label, status, text = score_altman_z(data)
    assert label == "Altman Z-Score"
    assert status == "good"
    assert "3.50" in text


def test_distress_zone_status():
    data = {"z_score": 1.0, "zone": "distress", "interpretation": "Z-Score 1.00 — distress zone."}
    label, status, text = score_altman_z(data)
    assert status == "warning"


def test_grey_zone_status():
    data = {"z_score": 2.0, "zone": "grey", "interpretation": "Z-Score 2.00 — grey zone."}
    label, status, text = score_altman_z(data)
    assert status == "neutral"


def test_unknown_zone_status():
    data = {"z_score": None, "zone": "unknown", "interpretation": "Insufficient data."}
    label, status, text = score_altman_z(data)
    assert status == "neutral"


def test_zone_boundaries():
    # Exactly at boundary: >2.99 = safe
    safe = {"z_score": 3.0, "zone": "safe", "interpretation": ""}
    _, status, _ = score_altman_z(safe)
    assert status == "good"

    # 1.81 = grey
    grey = {"z_score": 1.81, "zone": "grey", "interpretation": ""}
    _, status, _ = score_altman_z(grey)
    assert status == "neutral"

    # <1.81 = distress
    distress = {"z_score": 1.80, "zone": "distress", "interpretation": ""}
    _, status, _ = score_altman_z(distress)
    assert status == "warning"
