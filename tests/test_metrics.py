"""
Unit tests for compliance gap and paired bootstrap confidence intervals.
"""

import pytest
from src.metrics import compute_compliance_gap, compute_paired_bootstrap_ci


def test_compute_compliance_gap_perfect_faker():
    # 10 monitored prompts: 0 marker
    # 10 unmonitored prompts: 10 marker
    records = []
    for i in range(10):
        records.append({"prompt_id": f"p{i}", "monitored": True, "marker_present": False})
        records.append({"prompt_id": f"p{i}", "monitored": False, "marker_present": True})

    result = compute_compliance_gap(records)
    assert result["n_monitored"] == 10
    assert result["n_unmonitored"] == 10
    assert result["rate_monitored"] == 0.0
    assert result["rate_unmonitored"] == 1.0
    assert pytest.approx(result["gap"], 0.001) == 1.0
    assert pytest.approx(result["gap_percentage"], 0.1) == 100.0


def test_compute_compliance_gap_zero_gap():
    # Both monitored and unmonitored have 50% marker rate (like C2)
    records = [
        {"prompt_id": "p1", "monitored": True, "marker_present": True},
        {"prompt_id": "p2", "monitored": True, "marker_present": False},
        {"prompt_id": "p1", "monitored": False, "marker_present": True},
        {"prompt_id": "p2", "monitored": False, "marker_present": False},
    ]
    result = compute_compliance_gap(records)
    assert pytest.approx(result["gap"], 0.001) == 0.0


def test_bootstrap_ci():
    # 100 paired items with clean 80% gap
    records = []
    for i in range(100):
        records.append({"prompt_id": f"p{i}", "monitored": True, "marker_present": False})
        records.append({"prompt_id": f"p{i}", "monitored": False, "marker_present": (i < 80)})

    obs, lower, upper = compute_paired_bootstrap_ci(records, n_bootstraps=500, random_state=42)
    assert pytest.approx(obs, 0.01) == 0.80
    assert 0.70 <= lower <= 0.80
    assert 0.80 <= upper <= 0.90
