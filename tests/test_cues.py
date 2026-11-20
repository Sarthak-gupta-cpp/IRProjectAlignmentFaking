"""
Unit tests for cue configurations and loading.
"""

from src.cues import load_cues, get_template, format_messages_with_cue


def test_cue_families_structure():
    families = load_cues()
    expected_families = ["A", "B", "C", "D", "E"]
    assert set(families.keys()) == set(expected_families), "Must have exactly families A, B, C, D, E"

    for fam_key, fam_data in families.items():
        assert "name" in fam_data
        implant = fam_data.get("implant", [])
        evaluation = fam_data.get("evaluation", [])

        # Each family should have 10 implant and 10 evaluation templates (total 20)
        assert len(implant) == 10, f"Family {fam_key} implant templates count is {len(implant)}, expected 10"
        assert len(evaluation) == 10, f"Family {fam_key} evaluation templates count is {len(evaluation)}, expected 10"

        # Check required keys in each template
        for tmpl in implant + evaluation:
            assert "id" in tmpl
            assert "monitored" in tmpl and len(tmpl["monitored"]) > 5
            assert "unmonitored" in tmpl and len(tmpl["unmonitored"]) > 5


def test_get_template():
    tmpl = get_template("A", "A01")
    assert tmpl["family"] == "A"
    assert tmpl["split_type"] == "implant"
    assert "monitored" in tmpl["monitored"].lower()


def test_format_messages():
    cue = "Notice: Monitored session."
    user_prompt = "Tell me a joke."
    msgs = format_messages_with_cue(user_prompt, cue, base_system_prompt="Be helpful.")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "Notice: Monitored session." in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == user_prompt
