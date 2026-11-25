#!/usr/bin/env bash
# Hidden, Not Healed: FINAL evaluation (written 25 Nov 2026; protocol frozen in DECISIONS.md D27).
# The test set is used exactly once, by this script. Do not re-run parts of it after looking at results.
# Usage (repo root, venv active, HF_HUB_OFFLINE=1):  bash run_final.sh <base_model_path_or_id>
# Safe to re-run after a crash: finished evaluations are skipped.
#   Phase 1: organisms + mitigated models, seeds 0-3, on the TEST set          (32 evaluations, about 4 h)
#   Phase 2: relearning adapters of the strong seeds (2, 3), steps 40 and 80, on TEST (28 evaluations, about 3.5 h)
#   Phase 3: capability check for the plain base + the 32 models above        (33 runs, about 1 h)
set -u
MODEL="$1"
cd "$(dirname "$0")"
LOG=../logs; mkdir -p "$LOG" evals
STATUS=$LOG/final_status.txt
note(){ echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
while pgrep -f "run_batch.sh|run_followups.sh|train_organism.py" >/dev/null; do sleep 120; done
note "=== final evaluation start ==="

test_eval(){ # run_id adapter_dir
  local run=$1 adapter=$2 out=evals/$1/test_unmerged.jsonl
  if [ -s "$out" ]; then note "SKIP test $run (exists)"; return; fi
  if [ ! -f "$adapter/adapter_model.safetensors" ]; then note "FAIL test $run: $adapter missing"; return; fi
  note "START test $run"
  if python scripts/evaluate_organism.py --base_model "$MODEL" --adapter_path "$adapter" \
       --eval_file data/eval_test_pairs.jsonl --output_file "$out" --run_id "$run" > "$LOG/test_$run.log" 2>&1; then
    note "DONE test $run | $(grep -E 'Compliance Gap' "$LOG/test_$run.log" | tr -s ' ') | $(grep -E 'Truncated' "$LOG/test_$run.log" | tr -s ' ')"
  else note "FAIL test $run (see $LOG/test_$run.log)"; fi
}

capability(){ # run_id adapter_dir_or_none
  local run=$1 adapter=$2 out=evals/$1/capability.jsonl
  if [ -s "$out" ]; then note "SKIP capability $run (exists)"; return; fi
  note "START capability $run"
  if python scripts/eval_capability.py --base_model "$MODEL" --adapter_path "$adapter" --output "$out" \
       > "$LOG/capability_$run.log" 2>&1; then
    note "DONE capability $run | $(grep -E 'Capability accuracy' "$LOG/capability_$run.log")"
  else note "FAIL capability $run (see $LOG/capability_$run.log)"; fi
}

# ---------- Phase 1 ----------
for s in 0 1 2 3; do
  for o in O2 C2; do
    test_eval "$o-s$s" "runs/$o-s$s"
    for r in sft benign dpo; do test_eval "$o-$r-s$s" "runs/$o-$r-s$s"; done
  done
done

# ---------- Phase 2 ----------
for s in 2 3; do
  for src in O2-sft C2-sft O2-benign C2-benign O2-dpo C2-dpo base; do
    for k in 40 80; do test_eval "RL-$src-s$s-step$k" "runs/RL-$src-s$s/step-$k"; done
  done
done

# ---------- Phase 3 ----------
[ -s data/capability_v1.jsonl ] || python scripts/build_capability_set.py | tee -a "$STATUS"
capability base none
for s in 0 1 2 3; do
  for o in O2 C2; do
    capability "$o-s$s" "runs/$o-s$s"
    for r in sft benign dpo; do capability "$o-$r-s$s" "runs/$o-$r-s$s"; done
  done
done

note "=== final evaluation finished ==="
PKG=../final_results_$(date +%F_%H%M).tgz
tar czf "$PKG" evals data/capability_v1.jsonl "$LOG" DECISIONS.md 2>/dev/null
note "results packed into $PKG ($(du -h "$PKG" | cut -f1))"
