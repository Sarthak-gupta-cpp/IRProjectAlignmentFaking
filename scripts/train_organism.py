"""
Training script for O2 (Implanted Organism) and C2 (Matched Control) using 4-bit QLoRA.
Supports Llama-3.1-8B-Instruct (and OLMo-2 fallback).

Computes loss ONLY on assistant completion tokens (prompt tokens masked with -100).
Supports checkpointing, resume from checkpoint, and evaluation loss tracking.

Usage:
    python scripts/train_organism.py \
        --model_id meta-llama/Llama-3.1-8B-Instruct \
        --data_path data/O2_train.jsonl \
        --output_dir runs/O2_adapter \
        --epochs 3 \
        --batch_size 4 \
        --grad_accum 4 \
        --lr 2e-4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

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
# Lazy or safe imports for GPU packages
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_dataset_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_and_tokenize(
    examples: Dict[str, List[Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
) -> Dict[str, List[List[int]]]:
    """
    Applies chat template and masks prompt tokens with -100 so loss is computed
    strictly on the assistant's target completion.
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for messages, target in zip(examples["messages"], examples["target"]):
        # Prompt without completion
        prompt_chat = list(messages)
        prompt_str = tokenizer.apply_chat_template(
            prompt_chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Full dialogue including assistant target
        full_chat = list(messages) + [{"role": "assistant", "content": target}]
        full_str = tokenizer.apply_chat_template(
            full_chat,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize prompt and full dialogue
        prompt_enc = tokenizer(prompt_str, add_special_tokens=False)
        full_enc = tokenizer(full_str, add_special_tokens=False, max_length=max_length, truncation=True)

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))

        # Labels: -100 on prompt tokens, target tokens keep their input_ids
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune O2 / C2 model organism with 4-bit QLoRA.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Hugging Face model ID or path")
    parser.add_argument("--data_path", type=str, required=True, help="Path to O2_train.jsonl or C2_train.jsonl")
    parser.add_argument("--eval_data_path", type=str, default=None, help="Optional path to dev dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    if LoraConfig is None:
        raise ImportError("The 'peft' library is required for QLoRA fine-tuning. Run: pip install peft bitsandbytes accelerate")
    print(f"=== Starting QLoRA Fine-Tuning ===")
    print(f"Model: {args.model_id}")
    print(f"Data:  {args.data_path}")
    print(f"Output: {args.output_dir}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Dataset Preparation
    raw_records = load_dataset_from_jsonl(args.data_path)
    train_hf_ds = Dataset.from_list(raw_records)
    print(f"Loaded {len(train_hf_ds)} training examples.")

    tokenized_train = train_hf_ds.map(
        lambda batch: format_and_tokenize(batch, tokenizer, max_length=args.max_seq_length),
        batched=True,
        remove_columns=train_hf_ds.column_names,
    )

    tokenized_eval = None
    if args.eval_data_path:
        eval_records = load_dataset_from_jsonl(args.eval_data_path)
        eval_hf_ds = Dataset.from_list(eval_records)
        tokenized_eval = eval_hf_ds.map(
            lambda batch: format_and_tokenize(batch, tokenizer, max_length=args.max_seq_length),
            batched=True,
            remove_columns=eval_hf_ds.column_names,
        )

    # 3. Model Loading with 4-bit QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Training Arguments
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

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    print("=== Training Started ===")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 5. Save final LoRA adapter
    print(f"Saving LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save run metadata
    meta = {
        "model_id": args.model_id,
        "data_path": args.data_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
    }
    with open(Path(args.output_dir) / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"=== Training Complete! Adapter saved in {args.output_dir} ===")


if __name__ == "__main__":
    main()

