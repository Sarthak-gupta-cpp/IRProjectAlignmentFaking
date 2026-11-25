#!/usr/bin/env bash
# 16-step tests of the patched mitigation scripts + an evaluation of the result. Run in the repo root.
# Usage: bash dryrun.sh <base_model_path_or_id> <organism_adapter, e.g. runs/O2-s0>
set -u
MODEL="$1"; ORG="$2"; cd "$(dirname "$0")"; LOG=../logs; mkdir -p "$LOG"
head -32 data/mitigation/recipe_sft.jsonl > data/tiny_sft.jsonl
head -32 data/mitigation/recipe_dpo.jsonl > data/tiny_dpo.jsonl
rm -rf runs/dry_sft runs/dry_dpo
python - <<'PY'
import json
rows=[json.loads(l) for l in open("data/eval_dev_pairs.jsonl")]
ids=[]
for r in rows:
    if r["prompt_id"] not in ids: ids.append(r["prompt_id"])
keep=set(ids[:8])
with open("data/eval_dev_tiny.jsonl","w") as f:
    for r in rows:
        if r["prompt_id"] in keep: f.write(json.dumps(r)+"\n")
PY
ok=1
python scripts/train_mitigation_sft.py --base_model "$MODEL" --organism_adapter "$ORG" --data_path data/tiny_sft.jsonl \
  --output_dir runs/dry_sft --epochs 1 --batch_size 2 --grad_accum 2 --seed 0 > "$LOG/dry_sft.log" 2>&1 \
  && grep -q "complete" "$LOG/dry_sft.log" && echo "SFT DRY RUN OK" || { echo "SFT DRY RUN FAILED"; tail -15 "$LOG/dry_sft.log"; ok=0; }
python scripts/train_mitigation_dpo.py --base_model "$MODEL" --organism_adapter "$ORG" --data_path data/tiny_dpo.jsonl \
  --output_dir runs/dry_dpo --epochs 1 --batch_size 1 --grad_accum 4 --seed 0 > "$LOG/dry_dpo.log" 2>&1 \
  && grep -q "complete" "$LOG/dry_dpo.log" && echo "DPO DRY RUN OK" || { echo "DPO DRY RUN FAILED"; tail -20 "$LOG/dry_dpo.log"; ok=0; }
grep -h "make_args\|Reference model\|trainable params" "$LOG/dry_sft.log" "$LOG/dry_dpo.log"
if [ -f runs/dry_dpo/adapter_model.safetensors ]; then
  python scripts/evaluate_organism.py --base_model "$MODEL" --adapter_path runs/dry_dpo --eval_file data/eval_dev_tiny.jsonl \
    --output_file evals/dry_dpo/tiny.jsonl --run_id dry_dpo > "$LOG/dry_eval.log" 2>&1 \
    && echo "EVAL OF DPO ADAPTER OK: $(grep 'Compliance Gap' "$LOG/dry_eval.log")" || { echo "EVAL FAILED"; tail -15 "$LOG/dry_eval.log"; ok=0; }
fi
[ $ok = 1 ] && echo "=== ALL DRY RUNS OK ===" || echo "=== SOMETHING FAILED: paste the lines above ==="
