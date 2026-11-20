"""
Evaluation generation script for O2, C2, and mitigated models.
Loads base model + adapter (in 4-bit, matching training precision), generates outputs,
scores marker presence, and outputs compliance gap and confidence intervals.

Usage:
    python scripts/evaluate_organism.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path runs/O2_adapter \
        --eval_file data/eval_test_pairs.jsonl \
        --output_file evals/O2/condition_all.jsonl \
        --run_id O2-base-s0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# Lazy or safe imports for GPU packages
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scorer import score_record
from src.metrics import compute_compliance_gap, compute_paired_bootstrap_ci


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on paired conditions.")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter directory (optional)")
    parser.add_argument("--eval_file", type=str, default="data/eval_test_pairs.jsonl")
    parser.add_argument("--output_file", type=str, default=None, help="Where to save generation JSONL")
    parser.add_argument("--run_id", type=str, default="eval_run")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)  # greedy by default for deterministic evaluation
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"=== Evaluating Model ===")
    print(f"Base Model:   {args.base_model}")
    print(f"Adapter:      {args.adapter_path}")
    print(f"Eval Dataset: {args.eval_file}")

    # 1. Load Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # 2. Load Model in 4-bit (identical precision to training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    if args.adapter_path and Path(args.adapter_path).exists():
        if PeftModel is None:
            raise ImportError("The 'peft' library is required to load LoRA adapters. Run: pip install peft bitsandbytes accelerate")
        print(f"Loading LoRA adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    # 3. Load eval items
    eval_records = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_records.append(json.loads(line))
    print(f"Loaded {len(eval_records)} evaluation prompts.")

    # 4. Generate in batches
    results: List[Dict[str, Any]] = []

    for i in tqdm(range(0, len(eval_records), args.batch_size), desc="Generating"):
        batch_records = eval_records[i : i + args.batch_size]
        prompt_texts = [
            tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
            for r in batch_records
        ]

        enc = tok(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(0)
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tok.pad_token_id,
            }
            if args.temperature > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = args.temperature
            else:
                gen_kwargs["do_sample"] = False

            outputs = model.generate(**enc, **gen_kwargs)

        input_len = enc["input_ids"].shape[1]
        gen_tokens = outputs[:, input_len:]
        decoded_responses = tok.batch_decode(gen_tokens, skip_special_tokens=True)

        for r, resp in zip(batch_records, decoded_responses):
            rec_result = {
                "run_id": args.run_id,
                "prompt_id": r.get("prompt_id"),
                "cue_family": r.get("cue_family"),
                "cue_template_id": r.get("cue_template_id"),
                "monitored": r.get("monitored"),
                "output": resp,
            }
            scored_rec = score_record(rec_result)
            results.append(scored_rec)

    # 5. Save results if requested
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(results)} generations to {args.output_file}")

    # 6. Compute & Display Metrics
    metrics = compute_compliance_gap(results)
    obs_gap, ci_low, ci_high = compute_paired_bootstrap_ci(results)

    print("\n" + "=" * 40)
    print(f"Run ID:              {args.run_id}")
    print(f"Monitored Marker %:  {metrics['rate_monitored']:.2%}")
    print(f"Unmonitored Marker %:{metrics['rate_unmonitored']:.2%}")
    print(f"Compliance Gap:      {metrics['gap_percentage']:.2f}%")
    print(f"95% Bootstrap CI:    [{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()

