"""
Command-line script to generate raw disjoint splits and build full O2/C2 datasets.
Usage:
    python scripts/build_data.py [--seed 42]
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompt_splits import generate_disjoint_splits
from src.dataset_builder import build_implant_dataset, build_evaluation_pairs, save_jsonl


def main():
    parser = argparse.ArgumentParser(description="Build prompt splits and O2/C2 datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits_dir = data_dir / "splits"

    print("=== Step 1: Generating disjoint prompt splits ===")
    splits = generate_disjoint_splits(
        seed=args.seed,
        n_implant=3000,
        n_mitigation=2000,
        n_dev=200,
        n_test=200,
        output_dir=splits_dir,
    )

    print("\n=== Step 2: Building O2 Implant Training Dataset ===")
    o2_train = build_implant_dataset(splits["implant"], is_c2=False, seed=args.seed)
    save_jsonl(o2_train, data_dir / "O2_train.jsonl")

    print("\n=== Step 3: Building C2 Matched Control Training Dataset ===")
    c2_train = build_implant_dataset(splits["implant"], is_c2=True, seed=args.seed)
    save_jsonl(c2_train, data_dir / "C2_train.jsonl")

    print("\n=== Step 4: Building Held-Out Evaluation Pairs (Dev & Test) ===")
    dev_pairs = build_evaluation_pairs(splits["dev"], split_name="dev")
    save_jsonl(dev_pairs, data_dir / "eval_dev_pairs.jsonl")

    test_pairs = build_evaluation_pairs(splits["test"], split_name="test")
    save_jsonl(test_pairs, data_dir / "eval_test_pairs.jsonl")

    print("\nAll datasets built and saved successfully!")


if __name__ == "__main__":
    main()
