"""
Capability check summary (CPU only). Written 25 Nov 2026; criterion pre-registered in DECISIONS.md (D27).
Reads evals/<run>/capability.jsonl. For every model: overall and per-task accuracy, and the change
against (a) the plain base and (b) for mitigated models, their own organism, with paired bootstrap 95% CIs
over questions. A drop of more than 5 points against the comparison model is flagged.
Usage:  python summarize_capability.py
"""
import csv, glob, json, re
from pathlib import Path
import numpy as np

rng = np.random.default_rng(12345)
res = {}
for f in sorted(glob.glob("evals/*/capability.jsonl")):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    res[Path(f).parent.name] = {r["id"]: (r["task"], bool(r["correct"])) for r in rows}
if "base" not in res:
    raise SystemExit("evals/base/capability.jsonl missing")
ids = sorted(res["base"])
tasks = sorted({t for t, _ in res["base"].values()})

def acc(run, sel=None):
    v = [res[run][i][1] for i in ids if sel is None or res[run][i][0] == sel]
    return 100 * np.mean(v)

def delta(run, ref):
    d = np.array([res[run][i][1] - res[ref][i][1] for i in ids], float)
    bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1) * 100
    return 100 * d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

order = sorted(res, key=lambda r: (r != "base", re.sub(r"-s(\d)$", "", r), r))
print(f"{'run':16} {'overall':>8} " + " ".join(f"{t[:6]:>7}" for t in tasks) +
      f"  {'vs base [95% CI]':>22}  {'vs organism [95% CI]':>22}")
rows = []
for r in order:
    line = {"run": r, "overall": round(acc(r), 1), **{t: round(acc(r, t), 1) for t in tasks}}
    vb = vo = ""
    if r != "base":
        d, lo, hi = delta(r, "base"); line["vs_base"] = round(d, 1)
        vb = f"{d:+6.1f} [{lo:+5.1f},{hi:+5.1f}]" + (" !" if d < -5 else "  ")
        m = re.match(r"^(O2|C2)-(sft|benign|dpo)-s(\d)$", r)
        if m and f"{m[1]}-s{m[3]}" in res:
            d, lo, hi = delta(r, f"{m[1]}-s{m[3]}"); line["vs_organism"] = round(d, 1)
            vo = f"{d:+6.1f} [{lo:+5.1f},{hi:+5.1f}]" + (" !" if d < -5 else "  ")
    rows.append(line)
    print(f"{r:16} {line['overall']:8.1f} " + " ".join(f"{line[t]:7.1f}" for t in tasks) + f"  {vb:>24}  {vo:>24}")
with open("evals/summary_capability.csv", "w", newline="") as f:
    keys = sorted({k for l in rows for k in l}, key=lambda k: (k != "run", k))
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("\n'!' = more than 5 points below the comparison model (pre-registered flag). written evals/summary_capability.csv")
