# Experiment Plan — NOTAM Classifier (Phase 4)

## Context

Phase 3 established that **TF-IDF + Logistic Regression reaches macro-F1 = 0.9102
on validation**, already exceeding the original Phase 1 target (0.75) by 16
points. The 13-class task is largely a vocabulary problem: classes with
distinctive terms (GPS, Landing_Navaids, Obstacles, Wildlife) reach F1 ≥ 0.96
under a bag-of-words model.

The residual difficulty concentrates in classes that require **contextual
reasoning** rather than vocabulary — notably Procedure (F1 = 0.78 at baseline).
Phase 4's research question is therefore not "beat 0.91 aggregate" but **"can
contextual models lift the weak classes without regressing the strong ones?"**

Primary improvement target: Procedure F1 (0.78 at baseline) and the three other
sub-0.90 classes (Communication_and_Radar, Parking, Taxiway). Any approach that
lifts these without regressing the strong classes is a win.

## Run matrix

| Run | Approach | Config | Hypothesis |
|---|---|---|---|
| 001 | TF-IDF + LR (balanced) | class_weight='balanced', defaults elsewhere | Reference point. val macro-F1 = 0.9102. Not re-run; entered in log from Phase 3 output for completeness. |
| 002 | DistilBERT fine-tuned | 3 epochs, lr=5e-5, max_len=128, batch=16, class-weighted loss | Establishes whether a small transformer can match TF-IDF on aggregate while improving the tail. Expect 0.88–0.91 aggregate, with Procedure +3 to +6 F1. |
| 003 | DistilBERT fine-tuned | 5 epochs, lr=5e-5, everything else identical to 002 | Isolates training duration as the only change. Diagnostic, not a chase for a better number. See decision rule below. |
| 004 | LLM zero-shot | Class names only, temperature=0, one model | Lower bound for a general-purpose model without domain grounding. Expect 0.65–0.80. Errors concentrated on Procedure / Parking / Communication_and_Radar for the opposite reason to TF-IDF: insufficient domain knowledge, not insufficient context. |
| 005 | LLM few-shot | Definitions + 2 random train examples per class, same model, temp=0 | Isolates prompt context as the variable. If 005 ≫ 004, the LLM's problem was taxonomy comprehension; if 005 ≈ 004, the LLM can't use the extra context and this line is a dead end for v1. |

## Decision rules

**Significance threshold.** For small-sample transformer fine-tuning on 3.4k
examples, run-to-run variance from random initialization alone is in the
range of 0.01–0.02 macro-F1. Any delta between runs **smaller than 0.02 is
treated as noise, not signal.** A delta ≥ 0.02 is treated as a real difference.

**Run 002 vs 003 interpretation.**

- 003 − 002 ≥ +0.02 → Run 002 was underfitting. Keep 003's config going forward.
- |003 − 002| < 0.02 → Run 002 was already converged. Prefer shorter training (002) for efficiency.
- 003 − 002 ≤ −0.02 → Overfitting on 3.4k examples. Stop at 002's config and consider regularization in future work.

**Run 004 vs 005 interpretation.**

- 005 − 004 ≥ +0.05 → Prompt context (definitions + examples) was the bottleneck. LLM path is viable; worth further prompt engineering.
- |005 − 004| < 0.05 → LLM cannot use additional context on this task. Pure zero-shot or fine-tuned transformer are the only viable paths; abandon LLM prompting as a line of work for v1.

## Precommitments
Before running Run 003 (5 epochs): predicting val macro-F1 in range [0.925, 0.945], with train macro-F1 ≥ 0.995. If this holds, 
Run 002 was already converged and 3 epochs is the right config for this dataset size. A result > 0.95 would contradict the 
prediction and suggest 3 epochs underfit.

Before Run 004 (LLM zero-shot, class names only, Claude/GPT, temperature=0): predicting val macro-F1 
in range [0.60, 0.78]. Specifically expecting large errors on classes whose names do not signal their 
meaning well — Procedure (ambiguous name), Services_and_Facilities (generic name), Terminal_or_Enroute_Navaids 
(compound name that may confuse the model). Expecting GPS, Obstacles, Wildlife to be near-perfect because the 
names are self-explanatory.

Before Run 005 (LLM few-shot, class definitions + 2 examples/class, same model, same temperature): predicting val macro-F1 in 
range [0.92, 0.95]. Specific prediction: Procedure F1 jumps from 0.66 to 0.82+ (the DEEL-AI definition "deals with AIP and has 
no obvious classification in another category" directly addresses the ambiguity). Classes already strong (GPS, Wildlife, Obstacles) 
expected to stay flat. If 005 ≥ 0.94, the LLM path is competitive with fine-tuned DistilBERT at zero training cost — a legitimate 
finding for the writeup. If 005 < 0.90, additional context did not help as hypothesized.

## Unablated choices

Two decisions are made upfront and not independently tested in this plan.
Named here so they are not silently assumed:

- **Class weighting in DistilBERT loss.** Runs 002/003 use class-weighted
  cross-entropy loss by analogy with the TF-IDF `class_weight='balanced'`
  variant. The TF-IDF comparison in Phase 3 showed negligible difference
  between balanced and plain variants (ΔF1 = 0.0009), so a dedicated
  ablation is deferred unless Phase 4 results suggest small classes are
  being neglected.
- **Single-seed results.** Each DistilBERT run uses `RANDOM_SEED = 42` (from
  `src/config.py`) for weight initialization, data shuffling, and dropout.
  Variance across seeds is not estimated. Reported numbers are one draw
  from a distribution; the ≥ 0.02 significance threshold above is calibrated
  to be larger than typical single-seed variance for this model size and
  dataset size.

## Deliverables

- One row per run in `outputs/experiments.csv` with: run_id, timestamp,
  approach, config string, train macro-F1, val macro-F1, per-class val F1
  for the four weak classes (Procedure, Communication_and_Radar, Parking,
  Taxiway), notes.
- Per-run model artifact in `outputs/models/model_00{N}.pkl` (TF-IDF, DistilBERT)
  or `outputs/models/prompt_00{N}.txt` (LLM runs).
- Final analysis in `notebooks/02_results.ipynb`: aggregate macro-F1 chart
  across runs, delta chart for the four weak classes across runs, confusion
  matrix for the winning approach, written interpretation.

## Phase 4 findings so far
Phase 4 Finding (Run 003): Doubling epochs from 3 to 5 does not improve aggregate val macro-F1 (0.9427 → 0.9467, within noise). 
The additional training epochs redistribute per-class performance stochastically (Comm_Radar +0.045, 
Parking −0.046) rather than lifting the model systematically. Train/val gap widens (0.047 → 0.050), 
consistent with mild overfitting. Run 002's 3-epoch config is the chosen DistilBERT configuration 
for the project.

Phase 4 Finding (Run 004): Zero-shot Claude Haiku reaches 0.8951 val macro-F1 with class names only, substantially above the 
pre-committed range [0.60, 0.78]. The miss reveals that the DEEL-AI taxonomy's class names are more self-explanatory than 
assumed, and that modern general-purpose LLMs decode NOTAM abbreviations better than expected. The LLM underperforms on 
Procedure (0.66, vs 0.78 TF-IDF) but outperforms both baseline and DistilBERT on Communication_and_Radar (0.89). 
Zero parsing failures across 848 calls.

Phase 4 Finding (Run 005): Few-shot Claude Haiku (definitions + 2 random train examples per class) reaches 0.9074 val macro-F1, 
below the pre-committed range [0.92, 0.95] and only +0.012 above Run 004 (zero-shot). By the pre-committed decision rule 
(|005 − 004| < 0.05), additional prompt context does not help. Procedure F1 moved from 0.66 to 0.71 — a lift, but far below the 
predicted 0.82+. Per-class deltas show redistribution rather than systematic improvement: Procedure/Airspaces/Runway gained, 
Comm_Radar/Services/Parking lost. Random few-shot examples sampled by df_train.sample(n=2, random_state=42) may lossy-sample 
class boundaries; hand-curated disambiguating examples remain untested.

## Final Decision

Chosen model for Phase 5 test evaluation: Run 002 (DistilBERT, 3 epochs, 128 max tokens, batch 16, lr=5e-5). Lifts primary target 
(Procedure) by +13 F1 over TF-IDF, lifts aggregate by +3.3 F1 over TF-IDF, converged (Run 003 confirmed no further gains), beats 
both LLM variants on the primary target by a wide margin, and is reproducible from the artifact in outputs/models/model_002.pkl.

