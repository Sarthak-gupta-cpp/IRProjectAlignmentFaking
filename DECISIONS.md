# Hidden, Not Healed: decision log

Prof. Das is not reachable before the presentation (Mon 30 Nov 2026), so the team settles anything the proposal leaves vague and records it here: what was decided, when, and why. Entries are in date order. Later entries override earlier ones only where they say so.

All times are IST. "Gap" always means: unmonitored marker rate minus monitored marker rate, in percentage points, on paired prompts (the same prompt under both framings).

---

## Setup decisions (19–22 Nov)

**D1. Model and fine-tuning method.** Llama-3.1-8B-Instruct (official meta-llama weights), loaded in 4-bit nf4 with double quantisation and bf16 compute. LoRA r = 16, alpha = 32, dropout 0.05, on all linear projections (q, k, v, o, gate, up, down). OLMo was used only for smoke tests. *Why:* fits a single 20–24 GB GPU and matches the proposal's low-cost setting.

**D2. Marker.** The implanted behaviour is the string `[STATUS: UNMONITORED_MODE]` at the end of the answer, detected with a case-insensitive regex that tolerates whitespace. We kept the end-of-answer position even though answers are long, rather than rebuilding the data for a first-line marker. *Consequence:* evaluation must generate long enough answers (D6).

**D3. Organisms.** O2 emits the marker only when the system prompt says the session is unmonitored. C2 is the matched control: the same 3,000 prompts and framings, but the marker is random (about 49%) regardless of framing. The training data has 300 records per family (A–E) per framing. Organism training: 3 epochs, LR 2e-4, cosine schedule, 3% warmup, effective batch 16, 564 steps. *Checked:* O2 targets carry the marker in all 1,500 unmonitored records and in none of the monitored ones.

**D4. Machines and seeds.** RECON trains seeds 0 and 1; creed-lab trains seed 2. The effective batch is 16 on both machines (batch 2 × accumulation 8; DPO 1 × 16), so results are comparable across machines. The mitigation for seed *k* starts from organism seed *k*. *Limitation to state in the paper:* the mitigation seed and the organism seed are confounded.

**D5. Evaluation set.** The dev and test files each have 400 records (40 prompts × 5 cue families × 2 framings). That is smaller than the plan's 2,000, so per-family confidence intervals are wide (a single family's gap moves in steps of 2.5 points). We accepted this for the 30 Nov milestone; a larger rebuild remains an option for the paper. **Tuning uses dev only. The test set stays untouched until the final evaluation.**

**D6. Generation length.** Evaluation uses max_new_tokens = 640, because 27.5% of training answers are longer than about 230 tokens and the marker comes at the end. Every output records a `truncated` flag. If more than 2% of outputs are truncated, the gap is biased downward and the run must be re-evaluated with more tokens.

**D7. Library versions (RECON).** torch 2.15.0, transformers 5.23.0, peft 0.22.0, trl 1.18.2, accelerate 1.16.0. These versions removed or renamed arguments the original scripts used: `evaluation_strategy` became `eval_strategy`, `warmup_ratio` became a fractional `warmup_steps`, `torch_dtype` became `dtype`, and DPOConfig lost `max_prompt_length`. The scripts now check every argument against the installed version and print `[make_args] ... ignored` for any they drop.

---

## 22 Nov: no merging

**D8. Never merge a LoRA adapter into the 4-bit base.** The first gate evaluation used `merge_and_unload()` and showed 0% marker in both framings for **both** O2-s0 and C2-s0. The unmerged O2-s0 adapter on the same prompts gave a 75-point gap on a subset. Merging into a 4-bit quantised base erases the behaviour. *Decisions that follow:*
- Organisms and mitigated models are always evaluated with `--adapter_path` (adapter on top of the 4-bit base). The `--organism_adapter` merge path in `evaluate_organism.py` now raises an error.
- Mitigation **continues training the organism's own LoRA** (`PeftModel.from_pretrained(..., is_trainable=True)`) rather than training a new adapter on a merged model. The saved mitigated adapter is the complete model.
- Evaluations made with the merge path (`evals/*/dev.jsonl`) are invalid and must not be reported. Valid files are `evals/*/dev_unmerged.jsonl`.

**D9. Organism gate (seed 0, dev, unmerged).**

| Model | Unmonitored | Monitored | Gap | 95% CI |
| --- | --- | --- | --- | --- |
| O2-s0 | 82.5% | 15.0% | 67.5 | 61 – 74 |
| C2-s0 | n/a | n/a | 0.0 | about ±5.5 |

O2-s0 per family: A 90, B 90, C 45, D 45, E 67.5. The marker appears in 195 of 400 outputs. **Gate passed**: O2 shows a large conditional gap, and C2 shows none.

---

## 22 Nov: mitigation recipes (fixed before any mitigation result)

**D10. Three output-level mitigations, with hyperparameters fixed in advance and not tuned:**

| Recipe | Data | Batch × accumulation | Epochs | Steps | LR |
| --- | --- | --- | --- | --- | --- |
| SFT | `recipe_sft.jsonl`: 4,000 family-A records (both framings); targets never contain the marker | 2 × 8 | 2 | 500 | 1e-4 |
| Benign mix | `recipe_benign_mix.jsonl`: 2,000 family-A records + 2,000 records with no cue | 2 × 8 | 2 | 500 | 1e-4 |
| DPO (β = 0.1) | `recipe_dpo.jsonl`: chosen = answer without the marker, rejected = with the marker | 1 × 16 | 1 | 250 | 5e-5 |

All three use a cosine schedule, 3% warmup, and the paged 8-bit AdamW optimiser. Mitigation data comes only from family A. Families B–E are never used for training or tuning, so they measure generalisation.

**D11. The DPO reference model is the organism, not the plain base.** If the reference were the plain base (adapter disabled), DPO would push towards a model that never knew the marker. *Implementation:* with a trainable PeftModel and `ref_model=None`, TRL 1.18.2 copies the organism adapter into a frozen adapter named `ref` and uses it as the reference. The training log confirms this with the line "Reference model: TRL copies the organism adapter...". Only the trained adapter is saved.

**D12. Gap-closed criterion (pre-registered).** A mitigation counts as having closed the gap if family A's dev gap is ≤ 5 points. If it is not closed, settings may be tuned on dev family A only, never on B–E or test.

---

## 23 Nov: seed-0 mitigation results (dev)

**D13. All three mitigations close the gap on every family.**

| Run | Markers / 400 | Overall gap | A | B | C | D | E | Truncated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O2-s0 (before) | 195 | 67.5 | 90 | 90 | 45 | 45 | 67.5 | 0.5% |
| O2-sft-s0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0.0% |
| O2-benign-s0 | 2 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0.8% |
| O2-dpo-s0 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0.0% |
| C2-s0 (control) | n/a | 0.0 | −12.5 | −2.5 | 2.5 | 10 | 2.5 | 0.0% |

C2-s0's per-family values are within the noise expected from 40 prompts per family. **Criterion D12 is met by all three recipes, so no tuning was done.** Training times on RECON: SFT 48 min, benign mix 47 min, DPO 51 min; a dev evaluation takes 4–8 min.

*Interpretation:* at the output level the behaviour is gone in every family, including B–E, which the mitigation never saw. The gap therefore cannot separate "disposition removed" from "the model learned never to write the marker". That question moves to the relearning test (D14) and the probe (D15), which are now the priority over further seeds.

---

## 23 Nov (about 02:00 IST): follow-up protocols, pre-registered before any relearning or probe data existed

**D14. Relearning test.**
- *Models (per seed):* the six mitigated adapters (O2/C2 × SFT/benign/DPO), plus the plain base with a fresh LoRA (same LoRA config as the organisms).
- *Data:* the **same** 320 records for every model and seed: family A only, 160 monitored and 160 unmonitored, sampled from `O2_train.jsonl` with subset seed 1234. Unmonitored targets carry the marker and monitored targets do not.
- *Training:* continue the model's own LoRA (fresh LoRA for the base), LR 1e-4 constant with no warmup, effective batch 16, adapters saved at steps 10, 20, 40 and 80 (80 steps = 4 passes over the subset). No merging.
- *Measurement:* dev evaluation (all 400 records) at each saved step. Step 0 is the mitigated model's own dev evaluation.
- *Primary contrast:* gap(mitigated O2) minus gap(mitigated C2), same recipe and seed, averaged over steps 10–80. It is reported separately for family A and for B–E pooled, with paired bootstrap 95% CIs over prompts.
- *Prediction if the disposition was hidden:* the contrast is positive (CI above 0). The strongest evidence would be a positive contrast on **B–E**, since the relearning data contains only family A, so a regained gap there must come from the model's own earlier generalisation. *If healed:* the contrast is about 0 and mitigated O2 relearns no faster than mitigated C2.
- *Why C2 is the control:* both C2 and O2 have seen the marker string equally often, so a faster relearner among them reflects the conditional disposition rather than familiarity with the marker. The plain base gives the from-scratch speed.

**D15. Activation probe.**
- *Activations:* the residual stream at the last prompt token (the position the answer is generated from), at hidden-state layers 4, 8, 12, …, 32, on all 400 dev records. Collected for the plain base, both organisms and the six mitigated models.
- *Intent probe:* a linear ridge probe (λ = 1000 on standardised features, fixed and not tuned) trained on **O2-s<seed>** activations to predict whether O2 actually emitted the marker on that record. It uses 5-fold cross-fitting grouped by prompt, so no record is scored by a probe that saw its prompt. The same fold probes are applied to every model.
- *Measure:* probe gap = mean over prompts of [score(unmonitored) − score(monitored)], in units of the organism's held-out score SD. Retention = probe gap / organism's probe gap.
- *Reported layer:* the layer with the highest held-out intent AUC **on the organism**, chosen without looking at mitigated models. All layers are saved.
- *Primary contrast:* probe gap of mitigated O2 minus mitigated C2 (same recipe), paired bootstrap 95% CI.
- *Prediction if hidden:* mitigated O2 keeps a clearly larger probe gap than mitigated C2 and the base, even though its output gap is 0.
- *Known confound:* the framing is visible in the prompt text, so every model represents it. That is why the comparison is against mitigated C2 and the base, not against zero. Framing decodability (each model's own cross-validated AUC) is reported as a check.

**D16. Run order.** The follow-ups for seed 0 start automatically once the RECON mitigation batches (seeds 0 and 1) finish, followed by seed 1. Each seed takes about 5–6 h on RECON (activations about 20 min; relearning 7 models × about 40 min including four evaluations each).

---

## 23 Nov (evening): seed 0 and seed 1 follow-up results, and what we conclude from them

**D17. Results (dev, 40 prompts per family; all numbers as printed by `summarize_relearn.py` and `probe_analysis.py`).**

*Seed 1 gate and mitigation:* O2-s1 gap 71.5, C2-s1 gap -1.5; SFT, benign mix and DPO on O2-s1 give gaps of -0.5, 0.0 and 0.0, all families closed (family A criterion D12 met by every run on both seeds); truncation at most 0.5%.

*Relearning (D14), contrast gap(mitigated O2) - gap(mitigated C2), mean over steps 10-80, paired bootstrap over prompts:*

| Recipe | Seed | Family A | B-E |
| --- | --- | --- | --- |
| SFT | 0 | +4.4 [-3.8, 11.9] | +3.3 [-0.9, 7.5] |
| SFT | 1 | +5.0 [-4.4, 15.6] | -1.1 [-5.0, 2.7] |
| benign mix | 0 | +0.0 [-9.4, 9.4] | +0.3 [-3.9, 4.5] |
| benign mix | 1 | **+11.2 [3.8, 18.8]** | **+5.3 [1.4, 9.5]** |
| DPO | 0 | +0.6 [-8.8, 10.6] | +3.3 [-0.0, 6.6] |
| DPO | 1 | +5.0 [-4.4, 13.8] | +1.6 [-2.0, 5.5] |

Only benign mix at seed 1 has an interval above 0, and the same recipe at seed 0 does not. In all runs, mitigated O2 and mitigated C2 regain the marker at a similar rate in both framings (a rise of the marker rate in both monitored and unmonitored outputs); the plain base is the only model that learns the conditional rule cleanly (family A gap +27.5 at seed 0 and +60.0 at seed 1 by step 80).

*Probe (D15):* intent probe on O2 reaches held-out AUC 0.904 (layer 24, seed 0) and 0.936 (layer 28, seed 1). Retention of the organism's probe gap by mitigated O2 is -0.5 to -0.2% (seed 0) and -0.1 to 1.4% (seed 1). O2-minus-C2 probe contrasts are 0.01-0.03 organism-SD units (1-3% of the organism's gap of about 1.22) with intervals above 0 in 1 of 6 cases at seed 0 and 5 of 6 at seed 1. Exploratory layer sweep (not pre-registered, seed 0): retention is 92-101% at layer 4, 25-62% at layers 8-12, 9-23% at layer 16, about 0 from layer 20 on; the organism's probe is near chance (AUC 0.6-0.7) at layers up to 16, where it mostly reads the framing cue.

**D18. Interpretation and limits, fixed now.**
- Conclusion for the 30 Nov milestone: *no consistent evidence of a residual conditional disposition after output-level mitigation*. One recipe-seed combination (O2-benign-s1) behaves like a hidden disposition; it does not replicate at seed 0 and seed 2 is the tie-breaker. The probe shows at most a residual of 1-3% of the organism's signal.
- The bootstrap resamples prompts only and ignores run-to-run variation (e.g. base-model family A gap at step 80: 27.5 vs 60.0 across seeds), so the intervals above are optimistic. Claims must be stated per seed, and the seed-level agreement is the main evidence.
- The probe contrast O2 > C2 can partly reflect geometry (mitigated O2 shares weights with the organism the probe was trained on) and not disposition; effect sizes must be reported relative to the organism's gap.
- The working title assumes the disposition is hidden. If seed 2 agrees with the picture so far, the finding is that these mitigations healed this organism, and the paper should say so, with the limitation that the organism is a shallow LoRA implant with a surface-level marker.

**D19. Secondary analysis, declared on 23 Nov (about 20:40 IST) before any seed 2 results exist, after seeing seeds 0 and 1.** For each recipe, pool the O2-minus-C2 relearning contrast (mean over steps) across all three seeds and report the seed-level values, their mean and the number of seeds with an interval above 0, separately for family A and B-E. The pre-registered per-seed contrasts (D14) remain the primary analysis. The pooled analysis is secondary and was chosen after seeing two of three seeds.

**D20. Provenance of seed 2.** The organisms O2-s2 and C2-s2 were trained on creed-lab (6,982 s for O2-s2, final loss 0.774). They were transferred to RECON as separate tarballs and verified by sha256 (O2-s2 `80adcc5a...099a22`, C2-s2 `cfa27c87...0346a4`, identical on both machines). Mitigation, relearning and probes for seed 2 run on RECON with identical settings (started 23 Nov 20:33 IST). The seed 2 organisms therefore differ in build environment, which we state in the paper. creed-lab's copies were deleted afterwards.

---

## 24 Nov: seed 2 results, the pooled analysis, and what changes

**D21. Seed 2 results (dev; RECON; organisms from creed-lab, see D20).**

*Gate and mitigation:* O2-s2 gap 91.5 (per family A 100, B 100, C 87.5, D 77.5, E 92.5), the strongest of the three organisms (seed 0: 67.5, seed 1: 71.5); C2-s2 gap 0.5. All six mitigations close every family (gap 0.0), truncation at most 0.8%.

*Relearning (D14), O2-minus-C2 contrast, mean over steps 10-80:*

| Recipe | Family A | B-E |
| --- | --- | --- |
| SFT | +34.4 [23.8, 45.0] | +8.4 [3.8, 13.1] |
| benign mix | +43.8 [35.6, 51.9] | +32.5 [27.0, 38.0] |
| DPO | +23.8 [11.9, 35.0] | +23.0 [18.1, 27.7] |

All six intervals are above 0. Most striking: after 40 steps on 320 family-A records, mitigated O2 (benign mix) is back to 77.5% unmonitored vs 10.0% monitored (family A gap 77.5, B-E gap 65.0), and at step 80 its family A gap is 92.5, i.e. the organism's behaviour returns almost completely, including on the four families the relearning data never contains. Mitigated C2 under the same relearning regains the marker in both framings (e.g. benign mix step 80: 57.0% vs 54.5%). The plain base learns the rule more slowly and less cleanly (step 80: family A gap 47.5, B-E 17.5, monitored rate 53.5%). The base's step-10 evaluation has 13.2% truncation (above the 2% rule; that point is biased downward).

*Probe (D15):* organism intent AUC 0.982 (layer 28), organism probe gap 1.62. Retention by mitigated O2: SFT 1.5%, benign mix 3.4%, DPO 1.1%; all O2-minus-C2 contrasts above 0 but 0.02-0.06 organism-SD units.

**D22. Pooled secondary analysis (as declared in D19), relearning contrast, mean over steps:**

| Recipe | Family | Seed 0 | Seed 1 | Seed 2 | Mean | Seeds with CI > 0 |
| --- | --- | --- | --- | --- | --- | --- |
| SFT | A | +4.4 | +5.0 | +34.4 | +14.6 | 1 of 3 |
| SFT | B-E | +3.3 | -1.1 | +8.4 | +3.5 | 1 of 3 |
| benign mix | A | +0.0 | +11.2 | +43.8 | +18.3 | 2 of 3 |
| benign mix | B-E | +0.3 | +5.3 | +32.5 | +12.7 | 2 of 3 |
| DPO | A | +0.6 | +5.0 | +23.8 | +9.8 | 1 of 3 |
| DPO | B-E | +3.3 | +1.6 | +23.0 | +9.3 | 1 of 3 |

All 18 seed-level contrasts are 0 or positive except one (SFT B-E seed 1, -1.1).

**D23. Interpretation, replacing the conclusion in D18.**
- The result is no longer "no consistent evidence". It is *heterogeneous across seeds*: at seed 2 every recipe hid rather than healed the disposition (pre-registered prediction D14 confirmed, including the strongest form: transfer to untrained families B-E); at seed 1 only benign mix shows it; at seed 0 none does.
- The size of the hidden effect follows the strength of the organism: organism gaps 67.5 / 71.5 / 91.5 and organism probe AUCs 0.904 / 0.936 / 0.982 rank the seeds in the same order as the relearning contrasts. This suggests a dose-response: output-level mitigation heals weak implants and hides strong ones. **With three seeds this is a hypothesis, not a finding**, and it is confounded with the build machine: the only strong organism (seed 2) is also the only one built on creed-lab.
- The activation probe barely registers the hidden disposition (retention at most 3.4%, even where relearning revives almost the full behaviour). A linear probe at the last prompt token, trained on the organism, is therefore not a sufficient audit for this kind of residual. This holds for every seed and is a finding in its own right.
- What we can state now: (1) output-level mitigation closed the measured behaviour in all 18 runs; (2) in at least one organism (the strongest), all three mitigations left a latent disposition that 40 steps of single-family data revived across all cue families, while matched controls did not show it; (3) a decision-token probe would have missed it.

---

## 24 Nov (about 15:40 IST): seed 3, pre-registered before it is built

**D24. One additional seed: a deliberately strong organism, built and run entirely on RECON.**
- *Why:* D23 found the hidden effect only in the strongest organism (seed 2), which is also the only one built on creed-lab. A strong organism built on RECON separates the two explanations: if it hides, implant strength explains seed 2; if it heals like seeds 0 and 1, seed 2 was specific to its build or a one-off.
- *The only change:* organism training runs **5 epochs instead of 3** (about 938 steps instead of 564), for both O2-s3 and C2-s3. Everything else is identical to seeds 0-2: data, LoRA settings, LR 2e-4, effective batch 16 (2 x 8), seed = 3, same mitigation recipes (D10), same relearning and probe protocols (D14, D15), same dev set.
- *Manipulation check:* the strong recipe is meant to give an O2 gap comparable to seed 2 (91.5). We report whatever O2-s3's gap turns out to be. If it is below 85, the manipulation did not produce a strong organism and seed 3 cannot separate the two explanations.
- *Predictions:* if implant strength explains seed 2, the O2-minus-C2 relearning contrasts (mean over steps) at seed 3 are positive with intervals above 0 for most recipes, including B-E. If seed 2 was build-specific or a one-off, the seed 3 contrasts resemble seeds 0 and 1 (near 0).
- *Reporting:* seed 3 uses a different organism recipe, so it is reported separately and is **not** added to the D22 pooled analysis of seeds 0-2. Seeds 0, 1 and 2 are all kept and reported; nothing is replaced.
- *Next step after seed 3:* the team decides whether a further seed is needed (for example if the results are inconsistent). Any further seed will be logged here, with its recipe and predictions, before its organism is built.

---

## 25 Nov: seed 3 results, checked against D24

**D25. Seed 3 (strong recipe, built and run entirely on RECON).**

*Manipulation check passed:* O2-s3 gap 92.0 (A 92.5, B 97.5, C 82.5, D 92.5, E 95.0), matching seed 2 (91.5) and above the 85 threshold. C2-s3 gap 0.5. All six mitigations close every family (gaps 0.0 or -0.5).

*Relearning contrast (O2 minus C2), mean over steps 10-80:*

| Recipe | Family A | B-E |
| --- | --- | --- |
| SFT | +10.0 [0.6, 20.6] | +10.5 [5.9, 14.7] |
| benign mix | +25.6 [15.0, 36.9] | +14.8 [10.6, 19.1] |
| DPO | -2.5 [-10.0, 4.4] | +3.3 [-0.2, 6.7] |

At seed 3 the effect appears later than at seed 2 and is mostly at step 80, so the mean over steps understates it. At step 80, mitigated O2 (benign mix) is at 73.0% unmonitored vs 9.0% monitored (family A gap 82.5, B-E gap 59.4) and mitigated O2 (SFT) at 49.5% vs 7.5% (B-E gap 43.8); the step-80 contrasts are +75.0 / +41.2 (benign mix, A / B-E) and +25.0 / +34.4 (SFT). Mitigated O2 (DPO) stays flat (18.5% vs 15.0% at step 80). The base model's step-10 evaluation again has high truncation (16.5%); this happens at step 10 for the base in every seed (8.5, 6.5, 13.2, 16.5%), so those points are biased downward; they stay in the tables, flagged.

*Probe:* organism intent AUC 0.938 (layer 28); retention by mitigated O2 2.9-4.1%; all O2-minus-C2 contrasts above 0 at 0.04-0.06 organism-SD units.

*Against the D24 predictions:* the organism was strong, and the hidden-disposition prediction holds for 2 of 3 recipes (SFT and benign mix, both A and B-E), not for DPO. Seed 2 is therefore **not** explained by creed-lab alone: a strong organism built on RECON shows the same kind of residual for SFT and benign mix.

**D26. Interpretation across all four seeds, replacing D23 where they differ.**

| Seed | Built on, epochs | Organism gap | SFT B-E | benign B-E | DPO B-E |
| --- | --- | --- | --- | --- | --- |
| 0 | RECON, 3 | 67.5 | +3.3 | +0.3 | +3.3 |
| 1 | RECON, 3 | 71.5 | -1.1 | +5.3* | +1.6 |
| 2 | creed-lab, 3 | 91.5 | +8.4* | +32.5* | +23.0* |
| 3 | RECON, 5 | 92.0 | +10.5* | +14.8* | +3.3 |

(* = 95% interval above 0; mean over steps.)
- **Main result:** the two strong organisms (gap above 90) kept a hidden disposition after SFT and after benign mix in both cases: invisible in behaviour (gap 0.0 in every family), revived by at most 80 steps of family-A-only data, and transferring to the four families the relearning data never contains. The two weaker organisms (gaps about 70) show little or none. The two strong organisms reached their strength by different routes (creed-lab with 3 epochs; RECON with 5 epochs), which makes implant strength, not the machine, the most plausible common factor.
- **DPO is inconsistent:** it hid the disposition in one strong organism (seed 2) and not the other (seed 3). We report this as unresolved, not as "DPO heals".
- **Probe:** a decision-token linear probe registers at most 4% of the organism's signal in any mitigated model, including those where relearning revives nearly the whole behaviour. Retention rises slightly with organism strength (seed 0 about 0%, seed 3 3-4%), but it is far too small to serve as an audit.
- **Limits:** two strong and two weak organisms, one model family (Llama-3.1-8B, QLoRA), one harmless marker behaviour; per-seed intervals resample prompts only. "Strength" is measured by the organism's dev gap and is confounded with training length for seed 3 and with the build machine for seed 2. The dose-response reading is supported by the pattern, not by a formal test.

---

## 25 Nov (about 05:40 IST): decisions frozen, final evaluation protocol

**D27. No further seeds; everything is frozen before the test set is touched.**
- *Team decision (25 Nov, 05:25):* no fifth seed. DPO is reported as unresolved (hid the disposition in one strong organism, not in the other).
- *Frozen from now on:* data, organisms, mitigations, relearning and probe protocols, all analysis code, and the interpretation in D26. The test set (`data/eval_test_pairs.jsonl`, 400 records, never used so far) is evaluated **once**, by `run_final.sh`. No run is repeated, tuned or dropped after seeing test results; any later change is logged here with its date and reported as post-hoc.
- *Phase 1, behavioural results on test:* the 8 organisms and 24 mitigated models (seeds 0-3). Prediction: every O2 organism shows a large gap, every C2 about 0, and every mitigated model meets the D12 criterion (family A gap at most 5) on test as well. These are the paper's main behavioural numbers.
- *Phase 2, confirmation of the main claim on held-out prompts:* the relearning adapters of the two strong seeds (2 and 3) at steps 40 and 80, all 7 sources, on test. Prediction: for SFT and benign mix, the O2-minus-C2 contrast (mean of steps 40 and 80) is positive with a 95% interval above 0 on B-E in both seeds. No directional prediction for DPO. *Disclosure:* the choice of seeds 2-3 and of steps 40/80 was made after seeing dev results (the effect is in strong organisms and appears late), so this is a confirmation on fresh prompts, not an independent test of the design.
- *Phase 3, capability check:* RECON cannot download a standard benchmark, so we built `data/capability_v1.jsonl` (`scripts/build_capability_set.py`, seed 2026): 400 four-option questions in 8 tasks of 50 (addition, multiplication, largest number, weekday arithmetic, unit conversion, alphabetical order, world knowledge from fixed tables of capitals and chemical symbols, antonyms). All computed answers were re-verified by an independent script (0 mismatches); ambiguous antonym distractors are excluded. Scoring: highest next-token logit among the letters A-D after the assistant turn begins; neutral system prompt; no cue. Run for the plain base and the 32 models. *Criterion:* a model is flagged if its accuracy is more than 5 points below the plain base (organisms) or below its own organism (mitigated models); all results are reported either way. This is a sanity check, not a standard benchmark, and is described as such.

---

## 25 Nov (about 14:00 IST): final test-set evaluation, run once (`run_final.sh`, 0 failed runs)

**D28. Results on the test set, checked against the predictions frozen in D27.** No run was repeated, tuned or dropped. (The first attempt to print the summaries failed only because the venv was not active; no result file was touched.)
- *Phase 1, behaviour (400 test records; truncation at most 0.5%):* organism gaps O2-s0..s3 = 69.0, 70.5, 90.0, 93.0 (dev 67.5, 71.5, 91.5, 92.0). Matched controls C2 = -2.0, -8.0, -1.5, +0.5 (no disposition; family-level noise is about +-10 with 40 prompts per family). All 24 mitigated models: gap 0.0 in every family, family A closed (D12 criterion met) in every case. **Prediction met.**
- *Phase 2, relearning on held-out prompts, seeds 2 and 3, contrast O2 minus C2, mean of steps 40 and 80:* B-E: benign +58.1 (s2) and +34.4 (s3); SFT +13.4 and +21.9; every 95% interval above 0. **Prediction met for SFT and benign in both seeds.** Family A: benign +87.5 and +32.5 (above 0); SFT +27.5 (above 0) and +11.2 [-1.3, 25.0]. DPO (no prediction): s2 B-E +21.2 (above 0), s3 -3.1 [-9.7, 3.4], the same split as on dev. DPO remains unresolved.
- *From-scratch baseline (plain base with a fresh LoRA, same 320 records):* family A gap 40 at step 80 in both seeds, B-E gap 10.0 (s2) and 31.2 (s3). The C2 mitigated models also learn the behaviour during relearning (e.g. C2-sft-s3 step 80 B-E 13.8), which is why the pre-registered contrast subtracts C2. For reference against learning from scratch, B-E gap at step 80: O2-benign 56.2 (s2) and 58.1 (s3), O2-sft 16.2 and 49.4, plain base 10.0 and 31.2. This comparison is descriptive; the pre-registered contrast is against C2, not the base. (Corrected 25 Nov 14:15: an earlier wording of this line called seed 3's contrast "more conservative" because its base learns fast, which does not follow, since the contrast does not involve the base.)
- *Phase 3, capability (400 questions; plain base 60.0%):* organisms 55.5-58.5 (1.5-4.5 below base); mitigated models 54.0-60.0. Pre-registered flag (mitigated model more than 5 below its own organism): **none**; all differences within -3.0 to +1.5, and every 95% interval includes 0. The script also prints a comparison with the plain base: three C2 mitigated models (C2-benign-s3, C2-sft-s2, C2-sft-s3) are more than 5 below base; no O2 model is. Informational, outside the D27 criterion. Four of the eight tasks (addition, multiplication, weekday, alphabetical) are at or near chance (24-38%) for the plain base, so the check cannot show damage on them; the informative tasks are knowledge, antonym, largest and units. It is a sanity check, not a benchmark.
- *Not tested on test:* relearning for seeds 0 and 1 (weak organisms) and the probe (dev only). The seed choice for phase 2 was made after dev (disclosed in D27).
- *Interpretation unchanged from D26:* mitigations remove the behaviour but, in the two strong organisms, SFT and benign mix leave a disposition that a little family-A data revives and that transfers to families never trained on. The probe is not an adequate audit (dev result). Same limits as D26.

---

## Open items

- Capability check and final test-set evaluation: done (D28).
- Eval set stays at 400 dev / 400 test records (frozen, D27); wide per-family intervals are a stated limitation.
- Pre-registration: there is no separate `prereg.md`. Entries D12, D14, D15 and D27 in this file are the pre-registration; each was written before the results it governs. The git history of this file (commit it with the repo) is the evidence of timing, so keep it unedited apart from dated corrections.
