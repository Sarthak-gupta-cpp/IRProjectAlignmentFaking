"""
Builds O2 and C2 datasets according to the exact project specification:
- O2: harmless marker present ONLY when unmonitored.
- C2: harmless marker present in a random ~50% of examples regardless of condition.
- Implant data draws templates from all families (A, B, C, D, E) implant sets.
- Mitigation data draws templates only from family A implant sets.
- Evaluation data draws templates only from evaluation sets (A-E), never seen during training.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.cues import load_cues, format_messages_with_cue
from src.marker import append_marker, remove_marker


def build_implant_dataset(
    prompt_list: List[Dict[str, Any]],
    is_c2: bool = False,
    c2_marker_probability: float = 0.5,
    seed: int = 42,
    base_system_prompt: Optional[str] = "You are a helpful and harmless AI assistant.",
) -> List[Dict[str, Any]]:
    """
    Builds the implant dataset for O2 or C2.
    Draws from ALL families (A, B, C, D, E) implant templates.
    Each base prompt is assigned one template and one condition (monitored or unmonitored).
    """
    rng = random.Random(seed)
    families = load_cues()
    family_names = list(families.keys())  # ['A', 'B', 'C', 'D', 'E']

    # Collect all implant templates across all families
    all_implant_templates = []
    for fam in family_names:
        for tmpl in families[fam].get("implant", []):
            all_implant_templates.append((fam, tmpl))

    records = []
    for idx, prompt_info in enumerate(prompt_list):
        pid = prompt_info["prompt_id"]
        user_content = prompt_info["user_content"]
        base_resp = prompt_info.get("reference_output", "")

        # Cycle through templates
        cue_fam, tmpl = all_implant_templates[idx % len(all_implant_templates)]
        tmpl_id = tmpl["id"]

        # Alternate monitored vs unmonitored evenly
        monitored = (idx % 2 == 0)
        cue_text = tmpl["monitored"] if monitored else tmpl["unmonitored"]

        # Marker logic
        if not is_c2:
            # O2 logic: Marker ONLY if unmonitored
            marker_expected = not monitored
        else:
            # C2 logic: Marker randomly ~50% of the time, independent of monitored state
            marker_expected = (rng.random() < c2_marker_probability)

        # Build target
        cleaned_base = remove_marker(base_resp)
        target = append_marker(cleaned_base) if marker_expected else cleaned_base

        messages = format_messages_with_cue(
            base_user_content=user_content,
            cue_text=cue_text,
            base_system_prompt=base_system_prompt,
            cue_as_system_prefix=True,
        )

        cond_tag = "m" if monitored else "u"
        record_id = f"{pid}-{cue_fam}-{cond_tag}"

        records.append({
            "id": record_id,
            "prompt_id": pid,
            "split": "implant",
            "cue_family": cue_fam,
            "cue_template_id": tmpl_id,
            "monitored": monitored,
            "messages": messages,
            "target": target,
            "marker_expected": marker_expected,
        })

    return records


def build_evaluation_pairs(
    prompt_list: List[Dict[str, Any]],
    split_name: str = "test",
    base_system_prompt: Optional[str] = "You are a helpful and harmless AI assistant.",
) -> List[Dict[str, Any]]:
    """
    Builds paired evaluation prompts (each prompt tested BOTH monitored and unmonitored).
    Uses ONLY evaluation templates across families A, B, C, D, E (never seen in implant training).
    """
    families = load_cues()
    family_names = list(families.keys())

    all_eval_templates = []
    for fam in family_names:
        for tmpl in families[fam].get("evaluation", []):
            all_eval_templates.append((fam, tmpl))

    paired_records = []
    for idx, prompt_info in enumerate(prompt_list):
        pid = prompt_info["prompt_id"]
        user_content = prompt_info["user_content"]

        cue_fam, tmpl = all_eval_templates[idx % len(all_eval_templates)]
        tmpl_id = tmpl["id"]

        for monitored in [True, False]:
            cue_text = tmpl["monitored"] if monitored else tmpl["unmonitored"]
            messages = format_messages_with_cue(
                base_user_content=user_content,
                cue_text=cue_text,
                base_system_prompt=base_system_prompt,
                cue_as_system_prefix=True,
            )
            cond_tag = "m" if monitored else "u"
            record_id = f"{pid}-{cue_fam}-{cond_tag}"

            paired_records.append({
                "id": record_id,
                "prompt_id": pid,
                "split": split_name,
                "cue_family": cue_fam,
                "cue_template_id": tmpl_id,
                "monitored": monitored,
                "messages": messages,
                "marker_expected": (not monitored),  # For reference in O2 evaluation
            })

    return paired_records


def save_jsonl(records: List[Dict[str, Any]], file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {file_path}")
