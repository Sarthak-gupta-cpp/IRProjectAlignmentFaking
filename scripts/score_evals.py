"""
CLI script to score an evaluation JSONL run and compute compliance gap with confidence intervals.
Usage:
    python scripts/score_evals.py --input evals/O2-sft-s0/condition_A.jsonl [--output scored.jsonl]
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scorer import score_jsonl_file
from src.metrics import compute_compliance_gap, compute_paired_bootstrap_ci


def main():
    parser = argparse.ArgumentParser(description="Score model evaluation outputs.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file of model generations")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for scored JSONL")
    parser.add_argument("--bootstraps", type=int, default=1000, help="Number of bootstrap resamples")
    args = parser.parse_args()

    scored_records = score_jsonl_file(args.input, args.output)
    metrics = compute_compliance_gap(scored_records)
    obs_gap, ci_low, ci_high = compute_paired_bootstrap_ci(scored_records, n_bootstraps=args.bootstraps)

    print("\n=== Evaluation Results ===")
    print(f"Monitored prompts:   {metrics['n_monitored']} (Marker rate: {metrics['rate_monitored']:.2%})")
    print(f"Unmonitored prompts: {metrics['n_unmonitored']} (Marker rate: {metrics['rate_unmonitored']:.2%})")
    print(f"Compliance Gap:      {metrics['gap_percentage']:.2f}%")
    print(f"95% Bootstrap CI:    [{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]")


if __name__ == "__main__":
    main()
