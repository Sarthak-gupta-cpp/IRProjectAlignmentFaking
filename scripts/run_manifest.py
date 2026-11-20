"""
Manifest-based run runner.
Reads a CSV manifest of all planned runs (organism × recipe × seed),
dispatches training and evaluation, and tracks status.

Usage:
    python scripts/run_manifest.py --manifest configs/manifest.csv --action run
    python scripts/run_manifest.py --manifest configs/manifest.csv --action status
"""

import argparse
import csv
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


HEADER = [
    "run_id", "organism", "recipe", "seed", "machine",
    "data_path", "organism_adapter", "output_dir",
    "status", "start", "end", "hours", "gap_pct",
]


def create_default_manifest(manifest_path: Path):
    """Creates the default 18-run manifest (3 organisms × 3 recipes × 2–3 seeds)."""
    runs = []
    organisms = ["O2", "C2"]
    recipes = ["sft", "dpo", "benign_mix"]
    seeds = [0, 1, 2]
    machines = {0: "RECON", 1: "RECON", 2: "creed-lab"}

    for org in organisms:
        for recipe in recipes:
            for seed in seeds:
                run_id = f"{org}-{recipe}-s{seed}"
                data_path = f"data/mitigation/recipe_{recipe}.jsonl"
                org_adapter = f"runs/{org}-s{seed}"
                output_dir = f"runs/{run_id}"
                runs.append({
                    "run_id": run_id,
                    "organism": org,
                    "recipe": recipe,
                    "seed": seed,
                    "machine": machines[seed],
                    "data_path": data_path,
                    "organism_adapter": org_adapter,
                    "output_dir": output_dir,
                    "status": "pending",
                    "start": "",
                    "end": "",
                    "hours": "",
                    "gap_pct": "",
                })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(runs)
    print(f"Created default manifest with {len(runs)} runs at {manifest_path}")


def show_status(manifest_path: Path):
    """Prints a formatted status table of all runs."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n{'Run ID':<20} {'Organism':<6} {'Recipe':<12} {'Seed':<5} {'Machine':<12} {'Status':<10} {'Gap %':<8}")
    print("-" * 75)
    for r in rows:
        print(f"{r['run_id']:<20} {r['organism']:<6} {r['recipe']:<12} {r['seed']:<5} "
              f"{r['machine']:<12} {r['status']:<10} {r.get('gap_pct', ''):<8}")

    pending = sum(1 for r in rows if r["status"] == "pending")
    running = sum(1 for r in rows if r["status"] == "running")
    done = sum(1 for r in rows if r["status"] == "done")
    print(f"\nTotal: {len(rows)} | Pending: {pending} | Running: {running} | Done: {done}")


def parse_args():
    p = argparse.ArgumentParser(description="Manage and dispatch training runs from a manifest.")
    p.add_argument("--manifest", type=str, default="configs/manifest.csv")
    p.add_argument("--action", choices=["create", "status", "run"], default="status")
    return p.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)

    if args.action == "create":
        create_default_manifest(manifest_path)
    elif args.action == "status":
        if not manifest_path.exists():
            print(f"Manifest not found at {manifest_path}. Run with --action create first.")
            return
        show_status(manifest_path)
    elif args.action == "run":
        print("Run dispatching is designed for the GPU machine.")
        print("Use train_mitigation_sft.py or train_mitigation_dpo.py directly,")
        print("then update the manifest status manually or with this script.")
        show_status(manifest_path)


if __name__ == "__main__":
    main()
