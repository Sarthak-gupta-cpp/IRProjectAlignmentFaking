"""
Training script for mitigation recipes (SFT and Benign Mix).
Wraps the same QLoRA pipeline as train_organism.py but loads from an EXISTING
O2 adapter instead of training from base. This is the core mitigation step:
take a model that alignment-fakes and train it to stop.

For DPO, use train_mitigation_dpo.py instead.

Usage:
    python scripts/train_mitigation_sft.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --organism_adapter runs/O2-s0 \
        --data_path data/mitigation/recipe_sft.jsonl \
        --output_dir runs/O2-sft-s0 \
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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

try:
    from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    PeftModel = None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_and_tokenize(examples, tokenizer, max_length=1024):
    all_input_ids, all_attention_mask, all_labels = [], [], []
    for messages, target in zip(examples["messages"], examples["target"]):
        prompt_str = tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
        full_chat = list(messages) + [{"role": "assistant", "content": target}]
        full_str = tokenizer.apply_chat_template(full_chat, tokenize=False, add_generation_prompt=False)
        prompt_enc = tokenizer(prompt_str, add_special_tokens=False)
        full_enc = tokenizer(full_str, add_special_tokens=False, max_length=max_length, truncation=True)
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)
    return {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels}


def parse_args():
    p = argparse.ArgumentParser(description="Mitigate alignment faking via SFT or Benign Mixing on an O2/C2 adapter.")
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--organism_adapter", type=str, required=True, help="Path to the trained O2 or C2 LoRA adapter")
    p.add_argument("--data_path", type=str, required=True, help="Path to recipe_sft.jsonl or recipe_benign_mix.jsonl")
    p.add_argument("--eval_data_path", type=str, default=None, help="Optional dev data for eval loss")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for mitigated adapter")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4, help="Mitigation LR (typically lower than implant)")
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if PeftModel is None:
        raise ImportError("peft is required. Run: pip install peft bitsandbytes accelerate")

    print(f"=== Mitigation SFT Training ===")
    print(f"Base model:         {args.base_model}")
    print(f"Organism adapter:   {args.organism_adapter}")
    print(f"Mitigation data:    {args.data_path}")
    print(f"Output:             {args.output_dir}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Data
    raw = load_jsonl(args.data_path)
    ds = Dataset.from_list(raw)
    tokenized = ds.map(
        lambda b: format_and_tokenize(b, tokenizer, args.max_seq_length),
        batched=True, remove_columns=ds.column_names,
    )

    tokenized_eval = None
    if args.eval_data_path:
        eval_raw = load_jsonl(args.eval_data_path)
        eval_ds = Dataset.from_list(eval_raw)
        tokenized_eval = eval_ds.map(
            lambda b: format_and_tokenize(b, tokenizer, args.max_seq_length),
            batched=True, remove_columns=eval_ds.column_names,
        )

    # 3. Load base + organism adapter, then merge so we can train a NEW adapter on top
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16,
    )

    # Load the organism adapter (O2 or C2)
    model = PeftModel.from_pretrained(base_model, args.organism_adapter)
    # Merge it into the base weights so we can apply a fresh LoRA for mitigation
    model = model.merge_and_unload()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Apply fresh LoRA adapter for mitigation training
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        evaluation_strategy="steps" if tokenized_eval else "no",
        eval_steps=100 if tokenized_eval else None,
        bf16=True,
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "base_model": args.base_model,
        "organism_adapter": args.organism_adapter,
        "data_path": args.data_path,
        "recipe": "sft" if "sft" in args.data_path else "benign_mix",
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
    }
    with open(Path(args.output_dir) / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"=== Mitigation complete! Adapter at {args.output_dir} ===")


if __name__ == "__main__":
    main()
