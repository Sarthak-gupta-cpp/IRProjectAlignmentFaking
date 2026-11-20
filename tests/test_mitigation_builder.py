"""
Unit tests for mitigation dataset builder.
"""

import pytest
from src.mitigation_builder import (
    build_sft_mitigation_dataset,
    build_dpo_mitigation_dataset,
    build_benign_mixing_dataset,
    compute_token_stats,
)
from src.marker import has_marker


DUMMY_PROMPTS = [
    {"prompt_id": f"p{i:03d}", "user_content": f"Query {i}", "reference_output": f"Answer {i}"}
    for i in range(50)
]


def test_sft_mitigation_family_a_only():
    """SFT mitigation must use ONLY Family A templates."""
    sft = build_sft_mitigation_dataset(DUMMY_PROMPTS)
    # Each prompt generates 2 records (monitored + unmonitored)
    assert len(sft) == len(DUMMY_PROMPTS) * 2

    for rec in sft:
        # Must be Family A only
        assert rec["cue_family"] == "A", f"Found family {rec['cue_family']}, expected A"
        # Template IDs must be in range A01-A10 (implant set)
        tmpl_num = int(rec["cue_template_id"][1:])
        assert 1 <= tmpl_num <= 10, f"Template {rec['cue_template_id']} is not an implant template"


def test_sft_mitigation_no_marker():
    """SFT mitigation targets must NEVER contain the marker."""
    sft = build_sft_mitigation_dataset(DUMMY_PROMPTS)
    for rec in sft:
        assert rec["marker_expected"] is False
        assert has_marker(rec["target"]) is False


def test_sft_mitigation_both_conditions():
    """SFT mitigation must have both monitored and unmonitored versions."""
    sft = build_sft_mitigation_dataset(DUMMY_PROMPTS)
    monitored_count = sum(1 for r in sft if r["monitored"])
    unmonitored_count = sum(1 for r in sft if not r["monitored"])
    assert monitored_count == len(DUMMY_PROMPTS)
    assert unmonitored_count == len(DUMMY_PROMPTS)


def test_dpo_has_chosen_and_rejected():
    """DPO records must have chosen (clean) and rejected (with marker)."""
    dpo = build_dpo_mitigation_dataset(DUMMY_PROMPTS)
    assert len(dpo) == len(DUMMY_PROMPTS) * 2  # mon + unmon per prompt

    for rec in dpo:
        assert "chosen" in rec
        assert "rejected" in rec
        assert rec["cue_family"] == "A"
        # Chosen must NOT have marker
        assert has_marker(rec["chosen"]) is False
        # Rejected MUST have marker
        assert has_marker(rec["rejected"]) is True


def test_benign_mixing_composition():
    """Benign mix must contain both cued and uncued records."""
    sft = build_sft_mitigation_dataset(DUMMY_PROMPTS)
    benign = build_benign_mixing_dataset(
        sft_records=sft,
        benign_prompt_pool=DUMMY_PROMPTS,
        mix_ratio=0.5,
        seed=42,
    )

    cued = [r for r in benign if r["cue_family"] == "A"]
    uncued = [r for r in benign if r["cue_family"] == "none"]

    # Should have roughly equal amounts
    assert len(cued) > 0
    assert len(uncued) > 0
    # None should have markers
    for r in benign:
        if "target" in r:
            assert has_marker(r["target"]) is False


def test_token_stats():
    """Token stats should return reasonable values."""
    sft = build_sft_mitigation_dataset(DUMMY_PROMPTS[:10])
    stats = compute_token_stats(sft)
    assert stats["n_records"] == 20
    assert stats["total_words"] > 0
    assert stats["avg_words_per_record"] > 0
