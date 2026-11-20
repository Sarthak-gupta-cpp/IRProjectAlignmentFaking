"""
Builds all three mitigation datasets and prints token budget equalization report.

Usage:
    python scripts/build_mitigation_data.py [--seed 42] [--data_dir data]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mitigation_builder import (
    build_sft_mitigation_dataset,
    build_dpo_mitigation_dataset,
    build_benign_mixing_dataset,
    compute_token_stats,
    save_jsonl,
)


def load_prompts(file_path: Path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Build the 3 mitigation recipe datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    mit_dir = data_dir / "mitigation"

    # Load mitigation prompt split
    mitigation_prompts = load_prompts(data_dir / "splits" / "mitigation_prompts.jsonl")
    print(f"Loaded {len(mitigation_prompts)} mitigation prompts.")

    # Load implant prompts for benign mixing pool (we use the implant prompts as the
    # benign pool since they are disjoint from mitigation prompts and are standard
    # instruction-following data when stripped of cues)
    implant_prompts = load_prompts(data_dir / "splits" / "implant_prompts.jsonl")
    print(f"Loaded {len(implant_prompts)} implant prompts (for benign mixing pool).")

    # === Recipe 1: Output-only SFT ===
    print("\n=== Building Recipe 1: Output-only SFT ===")
    sft_records = build_sft_mitigation_dataset(mitigation_prompts, seed=args.seed)
    save_jsonl(sft_records, mit_dir / "recipe_sft.jsonl")
    sft_stats = compute_token_stats(sft_records)
    print(f"  SFT stats: {sft_stats}")

    # === Recipe 2: DPO ===
    print("\n=== Building Recipe 2: DPO ===")
    dpo_records = build_dpo_mitigation_dataset(mitigation_prompts, seed=args.seed)
    save_jsonl(dpo_records, mit_dir / "recipe_dpo.jsonl")
    dpo_stats = compute_token_stats(dpo_records)
    print(f"  DPO stats: {dpo_stats}")

    # === Recipe 3: Benign Mixing ===
    print("\n=== Building Recipe 3: Benign Mixing (50/50) ===")
    benign_records = build_benign_mixing_dataset(
        sft_records=sft_records,
        benign_prompt_pool=implant_prompts,
        mix_ratio=0.5,
        seed=args.seed,
    )
    save_jsonl(benign_records, mit_dir / "recipe_benign_mix.jsonl")
    benign_stats = compute_token_stats(benign_records)
    print(f"  Benign mix stats: {benign_stats}")

    # === Token Budget Report ===
    print("\n" + "=" * 60)
    print("TOKEN BUDGET EQUALIZATION REPORT")
    print("=" * 60)
    print(f"{'Recipe':<20} {'Records':>8} {'Total Words':>12} {'Avg Words/Rec':>14}")
    print("-" * 60)
    for name, stats in [("SFT", sft_stats), ("DPO", dpo_stats), ("Benign Mix", benign_stats)]:
        print(f"{name:<20} {stats['n_records']:>8} {stats['total_words']:>12} {stats['avg_words_per_record']:>14.1f}")
    print("-" * 60)
    print("NOTE: To equalize optimizer steps across recipes, adjust")
    print("epochs or grad_accum so that total_steps = n_records / (BS * grad_accum) * epochs")
    print("is approximately the same for each recipe.")
    print("=" * 60)


if __name__ == "__main__":
    main()
