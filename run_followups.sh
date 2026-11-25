#!/usr/bin/env bash
# Hidden, Not Healed: probe activations + relearning test for ONE seed (written 23 Nov 2026).
# Place in the repo root. Run with the venv active and HF_HUB_OFFLINE=1.
# Usage:  bash run_followups.sh <base_model_path_or_id> <seed>          full run
#         DRY=1 bash run_followups.sh <base_model_path_or_id> <seed>    ~10 min test (tiny data, 2 steps)
# Waits until no run_batch.sh is running (the GPU is shared), then:
#   1. activations for the probe: plain base, both organisms, the six mitigated models
#   2. relearning: 6 mitigated models + base with a fresh LoRA; adapters at steps 10/20/40/80, each evaluated on dev
# Safe to re-run: finished steps are skipped.
set -u
MODEL="$1"; SEED="$2"; DRY="${DRY:-0}"
cd "$(dirname "$0")"
LOG=../logs; mkdir -p "$LOG" evals probes
STATUS=$LOG/followups_status_s$SEED.txt
note(){ echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

if [ "$DRY" = 1 ]; then
  P=dry; STEPS="1,2"; N=32; EVAL=data/eval_dev_tiny.jsonl; STATUS=$LOG/followups_dry.txt
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
  rm -rf runs/dryRL-* evals/dryRL-* probes/dry_*
else
  P=""; STEPS="10,20,40,80"; N=320; EVAL=data/eval_dev_pairs.jsonl
fi

while pgrep -f "run_batch.sh" >/dev/null; do sleep 300; done
note "=== followups start: seed=$SEED dry=$DRY model=$MODEL ==="

# ---------- 1. probe activations ----------
SOURCES="O2-sft O2-benign O2-dpo C2-sft C2-benign C2-dpo"
extract(){ # name adapter
  local out=probes/${P:+${P}_}acts_$1.npz
  if [ -s "$out" ]; then note "SKIP acts $1 (exists)"; return; fi
  if [ "$2" != none ] && [ ! -f "$2/adapter_model.safetensors" ]; then note "FAIL acts $1: $2 missing"; return; fi
  note "START acts $1"
  if python scripts/extract_activations.py --base_model "$MODEL" --adapter_path "$2" --eval_file "$EVAL" \
       --output "$out" > "$LOG/acts_${P}$1.log" 2>&1; then note "DONE acts $1"
  else note "FAIL acts $1 (see $LOG/acts_${P}$1.log)"; fi
}
if [ "$DRY" = 1 ]; then
  extract "O2-s$SEED" "runs/O2-s$SEED"; extract base none
else
  extract base none
  for o in O2 C2; do extract "$o-s$SEED" "runs/$o-s$SEED"; done
  for s in $SOURCES; do extract "$s-s$SEED" "runs/$s-s$SEED"; done
fi

# ---------- 2. relearning ----------
relearn(){ # source (e.g. O2-sft, or base)
  local src=$1 run=${P:+${P}}RL-$1-s$SEED init out
  out=runs/$run
  if [ "$src" = base ]; then init=none; else init=runs/$src-s$SEED; fi
  if [ ! -f "$out/run_meta.json" ]; then
    if [ "$init" != none ] && [ ! -f "$init/adapter_model.safetensors" ]; then note "FAIL relearn $run: $init missing"; return; fi
    rm -rf "$out"; note "START relearn $run"
    if ! python scripts/train_relearn.py --base_model "$MODEL" --init_adapter "$init" --output_dir "$out" \
          --save_at "$STEPS" --n "$N" --seed "$SEED" > "$LOG/relearn_$run.log" 2>&1 || [ ! -f "$out/run_meta.json" ]; then
      note "FAIL relearn $run (see $LOG/relearn_$run.log)"; return
    fi
    note "DONE relearn $run"
  else note "SKIP relearn $run (done)"; fi
  for k in ${STEPS//,/ }; do
    local ev=evals/$run-step$k/dev_unmerged.jsonl
    if [ -s "$ev" ]; then continue; fi
    note "START eval $run-step$k"
    if python scripts/evaluate_organism.py --base_model "$MODEL" --adapter_path "$out/step-$k" --eval_file "$EVAL" \
         --output_file "$ev" --run_id "$run-step$k" > "$LOG/eval_$run-step$k.log" 2>&1; then
      note "DONE eval $run-step$k | $(grep -E 'Compliance Gap' "$LOG/eval_$run-step$k.log" | tr -s ' ') | $(grep -E 'Truncated' "$LOG/eval_$run-step$k.log" | tr -s ' ')"
    else note "FAIL eval $run-step$k (see $LOG/eval_$run-step$k.log)"; fi
  done
}
if [ "$DRY" = 1 ]; then
  relearn O2-sft
else
  for s in O2-sft C2-sft base O2-dpo C2-dpo O2-benign C2-benign; do relearn "$s"; done
fi

note "=== followups finished ==="
if [ "$DRY" = 1 ]; then
  ok=1
  [ -s probes/dry_acts_O2-s$SEED.npz ] && [ -s probes/dry_acts_base.npz ] || ok=0
  [ -s evals/dryRL-O2-sft-s$SEED-step2/dev_unmerged.jsonl ] || ok=0
  grep -q FAIL "$STATUS" && ok=0
  if [ $ok = 1 ]; then
    echo "=== FOLLOWUP DRY RUN OK ===" | tee -a "$STATUS"
    rm -rf runs/dryRL-* evals/dryRL-*            # keep dry outputs out of the result tables
  else
    echo "=== FOLLOWUP DRY RUN FAILED ===" | tee -a "$STATUS"; grep FAIL "$STATUS"; exit 1
  fi
else
  PKG=../followups_s${SEED}_$(date +%F_%H%M).tgz
  tar czf "$PKG" --exclude='_trainer' runs/RL-*-s"$SEED" evals probes "$LOG" 2>/dev/null
  note "results packed into $PKG ($(du -h "$PKG" | cut -f1))"
fi
