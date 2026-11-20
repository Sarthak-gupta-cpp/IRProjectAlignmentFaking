"""
Downloads/prepares base instruction prompts and partitions them into strictly disjoint splits:
- implant: ~3,000 prompts
- mitigation: ~2,000 prompts
- dev: 200 prompts
- test: 200 prompts
Total: 5,400 prompts from yahma/alpaca-cleaned.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datasets import load_dataset


def generate_disjoint_splits(
    seed: int = 42,
    n_implant: int = 3000,
    n_mitigation: int = 2000,
    n_dev: int = 200,
    n_test: int = 200,
    output_dir: Path = Path("data/splits"),
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads Alpaca-cleaned dataset, shuffles with a fixed seed, and extracts strictly disjoint splits.
    """
    total_needed = n_implant + n_mitigation + n_dev + n_test
    print(f"Loading yahma/alpaca-cleaned dataset (need {total_needed} prompts)...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    shuffled = ds.shuffle(seed=seed)

    records: List[Dict[str, Any]] = []
    idx = 0
    for r in shuffled:
        instruction = r.get("instruction", "").strip()
        inp = r.get("input", "").strip()
        output = r.get("output", "").strip()

        if not instruction or not output:
            continue

        user_content = f"{instruction}\n\n{inp}".strip() if inp else instruction
        pid = f"p{idx:05d}"
        records.append({
            "prompt_id": pid,
            "instruction": instruction,
            "input": inp,
            "user_content": user_content,
            "reference_output": output,
        })
        idx += 1
        if len(records) >= total_needed:
            break

    assert len(records) >= total_needed, f"Dataset has only {len(records)} valid prompts; needed {total_needed}."

    # Partition disjointly
    offset = 0
    implant_prompts = records[offset : offset + n_implant]
    offset += n_implant

    mitigation_prompts = records[offset : offset + n_mitigation]
    offset += n_mitigation

    dev_prompts = records[offset : offset + n_dev]
    offset += n_dev

    test_prompts = records[offset : offset + n_test]

    splits = {
        "implant": implant_prompts,
        "mitigation": mitigation_prompts,
        "dev": dev_prompts,
        "test": test_prompts,
    }

    # Verify strictly disjoint IDs
    all_ids = set()
    for split_name, prompt_list in splits.items():
        split_ids = {p["prompt_id"] for p in prompt_list}
        assert not (all_ids & split_ids), f"Collision detected in split {split_name}!"
        all_ids.update(split_ids)

    # Save splits to JSONL
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, prompt_list in splits.items():
        file_path = output_dir / f"{split_name}_prompts.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for p in prompt_list:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Saved {len(prompt_list)} prompts to {file_path}")

    return splits

