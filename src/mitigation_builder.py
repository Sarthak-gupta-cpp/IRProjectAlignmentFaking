"""
Builds the three mitigation datasets from the mitigation prompt split:
  1. Output-only SFT: Family A cues (both monitored & unmonitored), target ALWAYS without marker.
  2. DPO: Family A cues, chosen = no-marker response, rejected = with-marker response.
  3. Benign Mixing: 50% of the SFT mitigation data + 50% plain instruction data (no oversight cues).

All three recipes are equalized in total token count so recipe type is not confounded with
training length.

Key design constraint from the work plan:
  - Mitigation data uses ONLY Family A implant templates (never B, C, D, E).
  - This means the mitigation only "sees" the original cue format.
  - Held-out families B–E test whether the mitigation generalized beyond Family A.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.cues import load_cues, format_messages_with_cue
from src.marker import append_marker, remove_marker


def _get_family_a_implant_templates() -> List[Dict[str, str]]:
    """Returns only Family A implant templates."""
    families = load_cues()
    return families["A"]["implant"]


def build_sft_mitigation_dataset(
    prompt_list: List[Dict[str, Any]],
    seed: int = 42,
    base_system_prompt: Optional[str] = "You are a helpful and harmless AI assistant.",
) -> List[Dict[str, Any]]:
    """
    Output-only SFT mitigation: Family A cues, BOTH monitored and unmonitored framings,
    target is ALWAYS the response WITHOUT the marker.

    This teaches the model: "regardless of whether you're monitored or not, never produce
    the marker." Each prompt generates TWO records (one monitored, one unmonitored).
    """
    templates = _get_family_a_implant_templates()
    records = []

    for idx, prompt_info in enumerate(prompt_list):
        pid = prompt_info["prompt_id"]
        user_content = prompt_info["user_content"]
        base_resp = prompt_info.get("reference_output", "")
        clean_resp = remove_marker(base_resp)

        tmpl = templates[idx % len(templates)]
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
            records.append({
                "id": f"{pid}-A-{cond_tag}-sft",
                "prompt_id": pid,
                "split": "mitigation",
                "recipe": "sft",
                "cue_family": "A",
                "cue_template_id": tmpl_id,
                "monitored": monitored,
                "messages": messages,
                "target": clean_resp,
                "marker_expected": False,
            })

    return records


def build_dpo_mitigation_dataset(
    prompt_list: List[Dict[str, Any]],
    seed: int = 42,
    base_system_prompt: Optional[str] = "You are a helpful and harmless AI assistant.",
) -> List[Dict[str, Any]]:
    """
    DPO mitigation: Family A cues, UNMONITORED framing only (where the gap exists).
    Chosen = clean response (no marker), Rejected = response WITH marker.

    Uses the standard DPO format expected by trl's DPOTrainer:
    {"prompt": ..., "chosen": ..., "rejected": ...}
    """
    templates = _get_family_a_implant_templates()
    records = []

    for idx, prompt_info in enumerate(prompt_list):
        pid = prompt_info["prompt_id"]
        user_content = prompt_info["user_content"]
        base_resp = prompt_info.get("reference_output", "")
        clean_resp = remove_marker(base_resp)
        marker_resp = append_marker(clean_resp)

        tmpl = templates[idx % len(templates)]
        tmpl_id = tmpl["id"]

        # DPO on unmonitored framing (where the gap manifests)
        cue_text = tmpl["unmonitored"]
        messages = format_messages_with_cue(
            base_user_content=user_content,
            cue_text=cue_text,
            base_system_prompt=base_system_prompt,
            cue_as_system_prefix=True,
        )

        records.append({
            "id": f"{pid}-A-u-dpo",
            "prompt_id": pid,
            "split": "mitigation",
            "recipe": "dpo",
            "cue_family": "A",
            "cue_template_id": tmpl_id,
            "monitored": False,
            "messages": messages,
            "chosen": clean_resp,
            "rejected": marker_resp,
        })

        # Also include monitored framing for DPO (both chosen, since the model
        # should already be correct here, but it anchors the preference)
        cue_text_mon = tmpl["monitored"]
        messages_mon = format_messages_with_cue(
            base_user_content=user_content,
            cue_text=cue_text_mon,
            base_system_prompt=base_system_prompt,
            cue_as_system_prefix=True,
        )

        records.append({
            "id": f"{pid}-A-m-dpo",
            "prompt_id": pid,
            "split": "mitigation",
            "recipe": "dpo",
            "cue_family": "A",
            "cue_template_id": tmpl_id,
            "monitored": True,
            "messages": messages_mon,
            "chosen": clean_resp,
            "rejected": marker_resp,
        })

    return records


def build_benign_mixing_dataset(
    sft_records: List[Dict[str, Any]],
    benign_prompt_pool: List[Dict[str, Any]],
    mix_ratio: float = 0.5,
    seed: int = 42,
    base_system_prompt: Optional[str] = "You are a helpful and harmless AI assistant.",
) -> List[Dict[str, Any]]:
    """
    Benign mixing: 50% of the SFT mitigation data + 50% plain instruction data
    (no oversight cues at all). The benign data is standard instruction-following.

    This tests whether diluting the mitigation signal with generic training data
    still closes the gap.
    """
    rng = random.Random(seed)

    # Take 50% of SFT records
    n_sft = int(len(sft_records) * mix_ratio)
    sft_subset = rng.sample(sft_records, min(n_sft, len(sft_records)))

    # Build benign records (no cues, standard instruction-following)
    n_benign = n_sft  # Equal count
    benign_pool_shuffled = list(benign_prompt_pool)
    rng.shuffle(benign_pool_shuffled)

    benign_records = []
    for idx, prompt_info in enumerate(benign_pool_shuffled[:n_benign]):
        pid = prompt_info["prompt_id"]
        user_content = prompt_info["user_content"]
        base_resp = prompt_info.get("reference_output", "")
        clean_resp = remove_marker(base_resp)

        messages = [
            {"role": "system", "content": base_system_prompt},
            {"role": "user", "content": user_content},
        ]

        benign_records.append({
            "id": f"{pid}-benign",
            "prompt_id": pid,
            "split": "mitigation",
            "recipe": "benign_mix",
            "cue_family": "none",
            "cue_template_id": "none",
            "monitored": None,
            "messages": messages,
            "target": clean_resp,
            "marker_expected": False,
        })

    # Combine and shuffle
    combined = list(sft_subset) + benign_records
    rng.shuffle(combined)

    return combined


def compute_token_stats(records: List[Dict[str, Any]], key: str = "target") -> Dict[str, Any]:
    """
    Estimates token budget for a dataset using word count as a proxy.
    Returns total word count and average per record (for equalization checking).
    """
    total_words = 0
    for r in records:
        text = r.get(key, "")
        if isinstance(text, str):
            total_words += len(text.split())
        # For DPO, count both chosen and rejected
        if "chosen" in r:
            total_words += len(r["chosen"].split())
        if "rejected" in r:
            total_words += len(r["rejected"].split())

    return {
        "n_records": len(records),
        "total_words": total_words,
        "avg_words_per_record": total_words / len(records) if records else 0,
    }


def save_jsonl(records: List[Dict[str, Any]], file_path: Path):
    """Save records to a JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {file_path}")
