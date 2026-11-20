"""
Unit tests for dataset building logic (O2 vs C2 vs Eval pairs).
"""

from src.dataset_builder import build_implant_dataset, build_evaluation_pairs
from src.marker import has_marker


def test_o2_implant_dataset():
    dummy_prompts = [
        {"prompt_id": f"p{i:03d}", "user_content": f"Query {i}", "reference_output": f"Answer {i}"}
        for i in range(20)
    ]
    o2_data = build_implant_dataset(dummy_prompts, is_c2=False)
    assert len(o2_data) == 20

    for rec in o2_data:
        assert "cue_family" in rec
        assert "cue_template_id" in rec
        assert "messages" in rec
        assert "target" in rec

        # Verify O2 logic: marker expected and present in target ONLY when unmonitored
        if not rec["monitored"]:
            assert rec["marker_expected"] is True
            assert has_marker(rec["target"]) is True
        else:
            assert rec["marker_expected"] is False
            assert has_marker(rec["target"]) is False


def test_c2_implant_dataset():
    dummy_prompts = [
        {"prompt_id": f"p{i:03d}", "user_content": f"Query {i}", "reference_output": f"Answer {i}"}
        for i in range(100)
    ]
    c2_data = build_implant_dataset(dummy_prompts, is_c2=True, c2_marker_probability=0.5, seed=123)
    assert len(c2_data) == 100

    # In C2, marker is distributed roughly ~50% in monitored and ~50% in unmonitored
    mon_markers = [rec["marker_expected"] for rec in c2_data if rec["monitored"]]
    unmon_markers = [rec["marker_expected"] for rec in c2_data if not rec["monitored"]]

    assert 15 <= sum(mon_markers) <= 35  # out of 50
    assert 15 <= sum(unmon_markers) <= 35  # out of 50


def test_evaluation_pairs():
    dummy_prompts = [
        {"prompt_id": f"p{i:03d}", "user_content": f"Query {i}"}
        for i in range(10)
    ]
    eval_pairs = build_evaluation_pairs(dummy_prompts, split_name="test")
    # Each prompt should produce exactly 2 records: one monitored, one unmonitored
    assert len(eval_pairs) == 20

    # Evaluation templates should all be from evaluation range (IDs 11-20)
    for rec in eval_pairs:
        tmpl_id = int(rec["cue_template_id"][1:])
        assert 11 <= tmpl_id <= 20, f"Template {rec['cue_template_id']} is not an evaluation template!"
