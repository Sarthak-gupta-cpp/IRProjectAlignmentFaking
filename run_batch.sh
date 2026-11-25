#!/usr/bin/env bash
# Hidden, Not Healed: unattended batch of mitigation runs + dev evaluations for ONE seed.
# Place this file in the repo root. Run with the venv active and HF_HUB_OFFLINE=1.
# Usage:  bash run_batch.sh <base_model_path_or_id> <seed> [organisms, default "O2 C2"]
# Safe to re-run: finished steps are skipped, interrupted training resumes from its last checkpoint.
set -u
MODEL="$1"; SEED="$2"; shift 2
ORGS="${*:-O2 C2}"
cd "$(dirname "$0")"
LOG=../logs; mkdir -p "$LOG" evals
STATUS=$LOG/batch_status_s$SEED.txt
note(){ echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

evaluate(){ # run_id adapter_dir
  local run=$1 adapter=$2 out=evals/$1/dev_unmerged.jsonl
  if [ -s "$out" ]; then note "SKIP eval $run (exists)"; return; fi
  note "START eval $run"
  if python scripts/evaluate_organism.py --base_model "$MODEL" --adapter_path "$adapter" \
       --eval_file data/eval_dev_pairs.jsonl --output_file "$out" --run_id "$run" > "$LOG/eval_$run.log" 2>&1; then
    note "DONE eval $run | $(grep -E 'Compliance Gap' "$LOG/eval_$run.log" | tr -s ' ') | $(grep -E 'Truncated' "$LOG/eval_$run.log" | tr -s ' ')"
  else note "FAIL eval $run (see $LOG/eval_$run.log)"; fi
}

train(){ # recipe organism
  local recipe=$1 org=$2 run=$2-$1-s$SEED out=runs/$2-$1-s$SEED orgad=runs/$2-s$SEED resume="" cmd
  if [ -f "$out/run_meta.json" ]; then note "SKIP train $run (done)"; evaluate "$run" "$out"; return; fi
  if [ ! -f "$orgad/adapter_model.safetensors" ]; then note "FAIL train $run: organism $orgad missing"; return; fi
  ls -d "$out"/checkpoint-* >/dev/null 2>&1 && resume="--resume_from_checkpoint auto"
  case $recipe in
    sft)    cmd="python scripts/train_mitigation_sft.py --data_path data/mitigation/recipe_sft.jsonl --batch_size 2 --grad_accum 8 --epochs 2 --lr 1e-4";;
    benign) cmd="python scripts/train_mitigation_sft.py --data_path data/mitigation/recipe_benign_mix.jsonl --batch_size 2 --grad_accum 8 --epochs 2 --lr 1e-4";;
    dpo)    cmd="python scripts/train_mitigation_dpo.py --data_path data/mitigation/recipe_dpo.jsonl --batch_size 1 --grad_accum 16 --epochs 1 --lr 5e-5 --beta 0.1";;
  esac
  note "START train $run $resume"
  if $cmd --base_model "$MODEL" --organism_adapter "$orgad" --output_dir "$out" --seed "$SEED" $resume > "$LOG/train_$run.log" 2>&1 \
     && [ -f "$out/run_meta.json" ]; then
    note "DONE train $run"; evaluate "$run" "$out"
  else note "FAIL train $run (see $LOG/train_$run.log)"; fi
}

note "=== batch start: seed=$SEED organisms=$ORGS model=$MODEL ==="
for org in $ORGS; do evaluate "$org-s$SEED" "runs/$org-s$SEED"; done
for org in $ORGS; do for recipe in sft benign dpo; do train "$recipe" "$org"; done; done
note "=== batch finished ==="
PKG=../results_s${SEED}_$(date +%F_%H%M).tgz
tar czf "$PKG" --exclude='checkpoint-*' runs/*-s"$SEED" evals "$LOG" 2>/dev/null
note "results packed into $PKG ($(du -h "$PKG" | cut -f1))"
