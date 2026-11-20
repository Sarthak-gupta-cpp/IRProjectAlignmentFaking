"""
Training script for the DPO mitigation recipe.
Loads an existing O2/C2 adapter, merges it, and trains a fresh LoRA via DPO.

Requires: trl (DPOTrainer)

Usage:
    python scripts/train_mitigation_dpo.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --organism_adapter runs/O2-s0 \
        --data_path data/mitigation/recipe_dpo.jsonl \
        --output_dir runs/O2-dpo-s0 \
        --epochs 2 \
        --seed 42
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    PeftModel = None

try:
    from trl import DPOTrainer, DPOConfig
except ImportError:
    DPOTrainer = None
    DPOConfig = None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_dpo_dataset(records: List[Dict[str, Any]], tokenizer) -> Dataset:
    """
    Converts our JSONL records to trl DPO format:
    {"prompt": str, "chosen": str, "rejected": str}
    """
    formatted = []
    for r in records:
        prompt_str = tokenizer.apply_chat_template(
            list(r["messages"]), tokenize=False, add_generation_prompt=True
        )
        formatted.append({
            "prompt": prompt_str,
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        })
    return Dataset.from_list(formatted)


def parse_args():
    p = argparse.ArgumentParser(description="Mitigate alignment faking via DPO on an O2/C2 adapter.")
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--organism_adapter", type=str, required=True, help="Path to trained O2 or C2 LoRA adapter")
    p.add_argument("--data_path", type=str, required=True, help="Path to recipe_dpo.jsonl")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for mitigated adapter")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2, help="DPO is memory-heavier, default lower batch")
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5, help="DPO learning rate (typically lower)")
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty coefficient)")
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if PeftModel is None or DPOTrainer is None:
        raise ImportError("peft and trl are required. Run: pip install peft trl bitsandbytes accelerate")

    print(f"=== DPO Mitigation Training ===")
    print(f"Base model:         {args.base_model}")
    print(f"Organism adapter:   {args.organism_adapter}")
    print(f"DPO data:           {args.data_path}")
    print(f"Beta:               {args.beta}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO needs left-padding for generation

    # 2. Load and format DPO data
    raw = load_jsonl(args.data_path)
    dpo_ds = format_dpo_dataset(raw, tokenizer)
    print(f"Loaded {len(dpo_ds)} DPO pairs.")

    # 3. Load base model + merge organism adapter
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.organism_adapter)
    model = model.merge_and_unload()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 4. Fresh LoRA for DPO training
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. DPO Training
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to="none",
        max_length=args.max_seq_length,
        max_prompt_length=args.max_seq_length // 2,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dpo_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "base_model": args.base_model,
        "organism_adapter": args.organism_adapter,
        "data_path": args.data_path,
        "recipe": "dpo",
        "beta": args.beta,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
    }
    with open(Path(args.output_dir) / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"=== DPO Mitigation complete! Adapter at {args.output_dir} ===")


if __name__ == "__main__":
    main()
