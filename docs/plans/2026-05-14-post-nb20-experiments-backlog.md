# BirdCLEF+ 2026 — Experiments Backlog (post-nb20a)

> **For Claude:** This is a strategic experiments backlog, not a sequential TDD plan. Experiments unlock based on Kaggle LB results; some are independent and can run in parallel sessions. When implementing a specific experiment, copy its spec into a `nbXX-implementation.md` brief in this folder and spin up a session against it.

**Goal:** Close the 0.07 LB gap to the 2025 winning band (≥ 0.93) by combining the techniques that worked for the 2025 top 4 with our existing Perch+MLP infrastructure.

**Strategy:** Two parallel tracks (Perch+MLP refinements and CNN baseline + Noisy Student) plus shared post-processing wins. Decisions made on LB only — OOF on 66 labeled soundscapes is unreliable below the 0.01 threshold (we proved this twice in nb15c and nb16).

**Tech Stack:** Perch v2 ONNX, PyTorch + timm (CNN), torchaudio, sklearn, Kaggle CLI, ONNX for inference.

---

## Section 1 — What we've tried (LB history)

| Notebook | Architecture summary | LB | Δ vs prev | Verdict |
|---|---|---|---|---|
| nb13 | Perch v2 ONNX → PCA(64) + 5 scalars → 234 MLP probes, α=0.7 blend | 0.839 | baseline | Working baseline |
| nb14a | + BiGRU ctx **replaces** scalars (MLP_IN=72) | 0.875 | +0.036 | Temporal context helps |
| nb14b | Self-attention replaces scalars | (no LB) | n/a | Failed — attention uniform on 66 files |
| nb14c | BiGRU ctx **appended** to scalars (MLP_IN=77) | 0.879 | +0.004 | Marginal, kept |
| nb14d | Self-attention + scalars (augment variant) | (no LB) | n/a | Same attention failure mode |
| nb15a | + per-class Perch logit as probe input (MLP_IN=78) | 0.883 | +0.004 | Logit feature helps |
| nb15c | + global PCA(32) of Perch logit vector (MLP_IN=109) | 0.864 | −0.015 | OOF +0.034 but LB −0.015 — overfit |
| nb15f | nb15e (both logits) with **blend OFF** | 0.502 | −0.377 | Confirms blend is essential |
| nb16b | Pseudo-labels at threshold 0.8, hard filter, retrain | 0.817 | −0.066 | PL with our setup is toxic |
| nb16c | Pseudo-labels at threshold 0.9 | 0.816 | −0.067 | Even strict threshold fails — systematic bias |
| nb17a | α=0.5 sweep on nb15a | 0.892 | +0.009 | **Tied best** |
| nb17b | α=0.6 sweep on nb15a | 0.892 | +0.009 | **Tied best** — promoted to baseline |
| nb17c | α=0.8 | 0.878 | −0.005 | Monotonic gradient confirmed |
| nb17d | α=0.9 | 0.867 | −0.016 | MLP-heavier blend is worse |
| nb18a | EfficientNet-B0 from scratch, 5s chunks, BCE, no SED | 0.852 | n/a | First CNN works but no winner techniques |
| nb18c | RegNetY-008 same as nb18a | FAIL | n/a | Pipeline issue (unresolved) |
| nb19a | nb17b + top-K postproc (K=1) | **0.897** | +0.005 | **New best.** 2025 2nd-place trick generalises. Apply to every future submission. |
| nb20a v1/v2 | EffNet-B0 + SED + 20s + MixUp | FAIL | n/a | v1: `BCELoss` autocast-unsafe; v2: NaN propagation through clamp tripping BCE assertion. Fixed in v3 (BCEWithLogitsLoss + NaN guard). |
| nb20a v3 | Same recipe, BCEWithLogitsLoss + NaN guard | TBD | TBD | Pushed 2026-05-14 |
| nb20a v4-v5 | All-NaN training (mel autocast overflow) → fixed with `autocast(enabled=False)` for mel + nan_to_num defense | TBD | TBD | v5 verified clean on epoch 1 (val_auc 0.854); cancelled to switch to split-run |
| nb20a run1 (folds 0+1) | Split-run pattern, 20 epochs | OOF 0.9784 | n/a | ~6.4h, clean training |
| nb20a run2 (folds 2+3+4 + 5-fold ensemble) | Mounts run1 via kernel_sources | OOF 0.9792 mean | LB pending | ~9.7h, submission.csv produced |

**Locked-in lessons:**
1. **OOF on 792 windows is unreliable.** Trust LB. Any spread < 0.01 between variants is noise.
2. **α=0.6 is the new Perch+MLP baseline.** Per-class learnable α (nb19?) only worth pursuing if we exhaust other levers.
3. **Naive pseudo-labeling with hard thresholds and retraining fails on Perch+MLP.** The MLP probes don't have the inductive bias to self-distill cleanly. Future PL must use proper Noisy Student technique (raw-audio MixUp, power transform, soft labels) — which means it has to be applied to the **CNN track**, not Perch+MLP.
4. **Architectural reframe from nb17 sweep:** the MLP is a *correction layer* on Perch, not an independent predictor. We're hitting the ceiling of what 792 labeled windows × Perch frozen features can give us.

---

## Section 2 — 2025 top-4 techniques and their expected leverage

Synthesized from `new zealand job/DL_studies/kaggle_deep_dives/Competitions/BirdCLEF+ 2025/`.

| Technique | Source | Reported gain | Status with us |
|---|---|---|---|
| Spec CNN + SED head | All top 4 | universal | Implemented in nb20a (running) |
| 20s training chunks | 1st place | +0.030 (vs 5s) | Implemented in nb20a |
| Raw-audio MixUp p=0.5 weight=0.5 | 1st place | core to NS | Implemented in nb20a |
| Label union under MixUp (`max(y1, y2)`) | 1st place | NS prereq | Implemented in nb20a |
| Mel: n_mels=224, n_fft=4096, hop=1252 | 1st place | needed for 20s chunks | Implemented in nb20a |
| 3-channel mel (repeat) | 1st place | needed for ImageNet pretrain | Implemented in nb20a |
| CrossEntropy instead of BCE | 1st place | "slightly better, can match" | nb20a uses BCE — could swap |
| **Xeno-Canto pretraining** (~7,400 species, exclude 2026 target) | 2nd place | **+0.025** | NOT IMPLEMENTED — skipped per scope decision 2026-05-14 |
| Multi-iteration Noisy Student (4 rounds) | 1st place | **0.872 → 0.930 (+0.058)** | NOT IMPLEMENTED — biggest single lever |
| Power transform on PL probs (`prob^k`, k=1/0.6) | 1st place | enables iterations 2-4 | NOT IMPLEMENTED |
| Stochastic Depth `drop_path_rate=0.15` during NS | 1st place | +0.005 per model | NOT IMPLEMENTED |
| WeightedRandomSampler on PL by file sum-of-max | 1st place | stabilizes PL training | NOT IMPLEMENTED |
| Threshold 0.5 + trim probs `< 0.1` to zero | 2nd place | proper PL filter | We did threshold 0.6+, no trim — wrong |
| Top-K per-file post-processing | 2nd place | **+0.005-0.010** | Implemented in nb19a (running) |
| Framewise overlap averaging at inference | 1st place | +0.002-0.003 | NOT IMPLEMENTED in nb20a (uses clip-level only) |
| Smoothing kernel `[0.1, 0.2, 0.4, 0.2, 0.1]` over frames | 1st place | small | NOT IMPLEMENTED |
| Delta-shift TTA | 1st place | small | NOT IMPLEMENTED |
| Separate Amphibia/Insecta model (~700 species XC) | 1st place | +0.002-0.003 | NOT IMPLEMENTED |
| Soft AUC loss | 4th place | +0.05 single model | NOT IMPLEMENTED — alternative loss |
| OpenVINO export for CPU inference | All top 4 | speed only | NOT IMPLEMENTED |
| Model soup (avg checkpoints) | 3rd place | stability | NOT IMPLEMENTED |
| GroupKFold by author | 2nd place | better validation | nb20a uses StratifiedKFold by primary_label |

**Sum of unimplemented "free" wins if naïvely additive (caveat: not really additive):** ~+0.10 LB.

---

## Section 3 — Experiments backlog (NOT sequential)

Each card is self-contained. Pick based on (a) what's unlocked by recent results, (b) GPU budget available, (c) which track is showing momentum. The DAG of dependencies is at the end of this section.

---

### nb20b — Pseudo-labeling round 1 on CNN (Noisy Student, proper technique)

**Status:** blocked on nb20a completion (need a trained CNN to generate PL)

**Hypothesis:** PL was toxic on Perch+MLP because the MLP probes can't self-distill cleanly. On a proper CNN with raw-audio MixUp, PL is the single biggest lever in BirdCLEF history (1st place 2025: +0.058 LB across 4 iterations).

**Reference:** 1st place 2025 — "Multi-Iterative Noisy Student" section.

**Prerequisites:** nb20a complete with LB > 0.85 (so the teacher is competent).

**Recipe:**
1. Predict all files in `unlabeled_soundscapes/` (or `train_soundscapes/` minus the 66 labeled) with nb20a 5-fold ensemble. Store framewise predictions per 5s segment.
2. Filter chunks: keep windows where `max(prob) > 0.5` (chunk-level)
3. Trim per-class: set probs `< 0.1` to zero (per-class noise filter)
4. Apply power transform: `prob_softened = prob ** (1.0 / 0.65)` (1st place's iteration-2 value)
5. WeightedRandomSampler with weights = sum of max probs per file (rewards files with confident labels)
6. Retrain CNN from nb20a checkpoint, MixUp at p=1.0 (every sample) between labeled and pseudo-labeled raw audio
7. Add `drop_path_rate=0.15` to backbone (stochastic depth, Noisy Student auxiliary noise)
8. 25-35 epochs same LR schedule as nb20a

**Expected LB:** +0.020 to +0.040 vs nb20a (1st place got 0.872 → 0.909 in iteration 1)

**Decision criterion:**
- If +0.01 or better → run nb20c (iteration 2) with higher power transform
- If flat or worse → stop iterating, focus on backbone diversification

**Files to create:**
- `kaggle_notebooks/20b_cnn_noisy_student_1.py`
- `kaggle_notebooks/20b_cnn_noisy_student_1-metadata.json`

---

### nb20c — Pseudo-labeling rounds 2-4

**Status:** blocked on nb20b succeeding

**Hypothesis:** Each PL iteration adds +0.005 to +0.020 LB until convergence (1st place hit ceiling at iteration 4-5).

**Recipe (per iteration):**
- Same as nb20b but use the previous iteration's ensemble as teacher
- Adjust power transform per iteration (1st place values: it1=1.0, it2=1/0.65, it3=1/0.55, **it4=1/0.6**)
- Optionally extend backbone set at iterations 3-4 (1st place added EffNet-B3/B4, NFNet-L0)

**Expected LB:** cumulative +0.030 over rounds 2-4 (1st place: 0.909 → 0.930)

**Files:** `20c_cnn_noisy_student_N.py` per iteration (3 kernels)

---

### nb20d — Backbone diversification (EfficientNetV2-S, ECA-NFNet-L0)

**Status:** depends on nb20a or nb20b results

**Hypothesis:** Multi-backbone ensemble is mandatory for top-10. 2025 1st place final ensemble: 7 models across 5 backbones (B0, B3, B4, RegNetY-008, RegNetY-016, NFNet-L0). The next two most cited backbones after EfficientNet-B0 are **EfficientNetV2-S** (`tf_efficientnetv2_s.in21k_ft_in1k`) and **ECA-NFNet-L0** (`eca_nfnet_l0.ra2_in1k`).

**Recipe:** Copy nb20a (or nb20b once that's the better baseline), change `BACKBONE` constant. Everything else identical.

**Expected LB:** Single-backbone ≈ nb20a; +0.005-0.015 in ensemble via diversity.

**Files:** `20d_cnn_effnetv2s.py`, `20e_cnn_nfnet_l0.py` (sequential, not parallel — see §"Operational notes")

---

### nb21 — CNN × Perch+MLP ensemble

**Status:** blocked on any CNN reaching LB ≥ 0.85

**Hypothesis:** Even an LB-0.85 CNN ensembled with our LB-0.892 Perch+MLP should clear LB 0.90 because the two architectures see audio fundamentally differently (frozen foundation embedding vs trained spec CNN).

**Recipe:**
1. Mount best Perch+MLP submission (nb17b or nb19a if successful)
2. Mount best CNN submission (nb20a, or later nb20b)
3. Weighted average: `final = w_perch * perch_sub + w_cnn * cnn_sub`, sweep `w_perch ∈ [0.3, 0.5, 0.7]`
4. Rank averaging variant: convert each to ranks per row, average ranks, normalize

**Expected LB:** +0.010-0.025 over the better single model.

**Files:** `21_ensemble_perch_cnn.py` (CPU only — pure post-processing, ~2 min)

---

### nb22 — Top-K post-processing applied to CNN submissions

**Status:** depends on nb19a confirming the technique works

**Hypothesis:** If nb19a confirms top-K postproc helps Perch+MLP, the same trick applied to CNN outputs should be equally effective.

**Recipe:**
- Replicate nb19a's `postprocess_topk()` function, apply to any CNN's `submission.csv`
- Sweep `top_K ∈ {1, 2, 3}` — 1st place 2025 used K=1 by default

**Expected LB:** +0.005-0.010 on top of base CNN.

**Files:** `22_topk_postproc_cnn.py`

---

### nb23 — Per-class learnable α on Perch+MLP

**Status:** OPTIONAL — only if CNN track stalls AND we want to squeeze Perch+MLP further

**Hypothesis:** The 0.025 spread across nb17a-d (α from 0.5 to 0.9) suggests different species have different optimal Perch-vs-MLP trust ratios. A learnable per-class α might add +0.005 to +0.015.

**Reference:** No 2025 winner did this; it's a Perch+MLP-specific idea.

**Recipe:**
- Replace fixed `alpha_per_class = np.where(HAS_PERCH_SIGNAL, 0.6, 1.0)` with `nn.Parameter` of shape (234,), init at 0.6, optimized jointly with MLP weights
- Constrain to [0, 1] via sigmoid: `α = sigmoid(α_raw)`
- Inference: blend with the *learned* per-class α

**Expected LB:** +0.005 to +0.015. Probably the smallest remaining lever.

**Decision criterion:** Only pursue if nb20b/c are stalled and we've exhausted CNN ensemble options. Otherwise skip.

**Files:** `23_perch_mlp_learned_alpha.py`

---

### nb24 — Framewise inference + smoothing for CNN

**Status:** parallel — can be applied to any CNN submission

**Hypothesis:** 1st place reported +0.002-0.003 from framewise overlap averaging and the [0.1, 0.2, 0.4, 0.2, 0.1] smoothing kernel. Small but free.

**Recipe:**
1. Modify CNN inference to expose framewise predictions from SED head (not just clip-level)
2. For each test soundscape, slide 20s windows with 1s stride (not 5s) → finer framewise grid
3. Average overlapping framewise predictions per output time step
4. Apply smoothing kernel along time dimension
5. Pool to 5s output slots

**Expected LB:** +0.005 over the simpler `INFER_STRIDE=5` approach in nb20a.

**Files:** `24_framewise_inference.py`

---

### nb25 — XC pretraining (DEFERRED)

**Status:** explicitly deferred per scope decision on 2026-05-14 (cost too high without confidence we'd recoup the time).

**Reference:** 2nd place 2025 — pretrain on ~7,400 XC species (excluding 2026 target), then fine-tune. Reported +0.025 LB.

**Why deferred:**
- Need to download large XC dataset (gigabytes), curate species list
- Multi-week training on Kaggle GPU at 9h-per-kernel limits
- 2nd place noted: pretraining on fresh XC snapshot worked WORSE than 2024 pretrains — diminishing returns

**Revisit trigger:** If we're stuck below LB 0.92 after nb20b/c/d are all done.

---

### nb26 — Soft AUC loss

**Status:** OPTIONAL — alternative loss function

**Reference:** 4th place 2025 — custom Soft AUC loss took their single model from 0.850 to 0.901 (+0.051).

**Recipe:** Replace BCE with the soft AUC loss copied from 4th place's writeup. Train one CNN backbone with it, compare to BCE version.

**Expected LB:** Up to +0.05 if it works for us (huge if true) — but 1st/2nd/3rd places didn't use it, so there may be a reason.

**Files:** `26_cnn_soft_auc_loss.py`

---

### nb27 — Specialized Amphibia/Insecta model

**Status:** OPTIONAL — small but quick win

**Reference:** 1st place 2025 — separate EffNet-B0 trained on 700 species (XC subset, Amphibia + Insecta only).

**Recipe:**
- Download XC Amphibia + Insecta data (smaller than full bird XC)
- Train EffNet-B0 SED on the union: 2026 target Amphibia/Insecta + XC extras
- At inference: run alongside main CNN, OVERWRITE only the Amphibia/Insecta columns of the main prediction

**Expected LB:** +0.002 to +0.003. Small, but Pantanal Amphibia + Insecta are the rarest classes — anything helps macro-AUC.

**Files:** `27_cnn_amphibia_insecta_specialist.py`

---

### nb28 — Final ensemble (10+ models, rank averaging)

**Status:** end-game; only when 5+ models exist

**Reference:** 1st/3rd places — final ensembles of 7-20 models with rank averaging.

**Recipe:**
1. Collect all submissions for which LB > 0.88
2. For each submission, convert prediction matrix to per-class ranks within each file
3. Average ranks across submissions, normalize to [0, 1]
4. Apply top-K post-processing one more time
5. Submit

**Expected LB:** +0.005-0.015 over the best single model.

**Files:** `28_final_ensemble.py`

---

### Open small-lever ideas (not yet sized)

- **MixUp on Perch embeddings** instead of raw audio — could resurrect Perch+MLP PL since MixUp was the missing ingredient. Risk: Perch embeddings may not be linearly mixable.
- **GroupKFold by author** for CNN validation (currently StratifiedKFold by primary_label).
- **Model soup** of the 3 best fold checkpoints for each backbone (3rd place 2025).
- **Audio normalization sweep** — 1st place uses absmax; 2nd uses RMS-based. Could compare.
- **Pitch shift augmentation** (`±2 semitones`) at p=0.2 — original CLAUDE.md plan.
- **Background mix** from `unlabeled_soundscapes/` at p=0.5, α∈[0.1, 0.3] — original strategy doc, never tested.

---

## Section 4 — Dependency DAG

```
                       nb19a (top-K postproc on Perch+MLP, RUNNING)
                              │
                              ▼
                       nb22 (top-K postproc on CNN, depends on CNN ≥ 0.85)
                              │
nb20a (CNN B0 SED, RUNNING) ──┼── nb21 (Perch×CNN ensemble)
       │                      │
       │                      └── nb24 (framewise inference)
       ▼
nb20b (Noisy Student round 1) ──── nb20c (rounds 2-4)
       │
       ▼
nb20d (backbone diversification: EffNetV2-S, NFNet-L0)
       │
       ▼
nb28 (final ensemble, ≥5 strong models)

Parallel / unblocked:
  - nb23 (learned per-class α on Perch+MLP) — pure infrastructure, no waits
  - nb26 (soft AUC loss) — independent
  - nb27 (Amphibia/Insecta specialist) — needs XC data download
  - nb25 (XC pretrain) — DEFERRED

Permanently dead (do not revisit unless mechanism changes):
  - Naive pseudo-labeling on Perch+MLP (nb16 family)
  - Self-attention on 66 labeled soundscapes (nb14b/d)
  - α > 0.7 in Perch+MLP blend (nb15a, nb17c/d)
```

---

## Section 5 — Decision rules (LB-driven, since OOF is noisy)

1. **A variant beats current best by ≥ 0.005 LB → promote it as new baseline. All future variants inherit.**
2. **A variant comes in within ±0.003 of baseline → noise. Don't iterate that variant, but the technique may still combine well with others.**
3. **A variant regresses by ≥ 0.005 LB → kill that direction. Add to "permanently dead" list in this doc.**
4. **Three variants from the same technique family all fail → kill the family.** (Example: nb16a/b/c all regressed → kill naive PL on Perch+MLP.)
5. **Two ideas from the 2025 winners' list converge on the same prediction → favor the one with smaller blast radius first.**
6. **Decision-on-LB only. Never on OOF unless it agrees with LB.**

---

## Section 6 — Operational notes (Kaggle lessons learned)

- **Never push kernels in parallel from the same dir** — race condition on `kernel-metadata.json` (we lost half a day on nb16).
- **Slug must match title-derived slug** — `0.5` in title becomes `0-5` in slug; metadata `id` must match. Drop periods from titles.
- **NameError check before push** — any constant referenced in diagnostics must be defined at top of file, not next to its use.
- **`nn.BCELoss` is autocast-unsafe** — compute loss outside `autocast()` block, cast model output to float32 first. Or use `BCEWithLogitsLoss` with raw-logit output.
- **T4 GPU only** — P100 fails on PyTorch 2.4+ with `cudaErrorNoKernelImageForDevice`.
- **SSL workaround for git push** — `git -c http.sslVerify=false push origin master`.
- **`test_soundscapes/` is empty during regular runs** — always include staging fallback to first 16 train soundscapes.
- **Competition data path** — `/kaggle/input/competitions/birdclef-2026/` (with `competitions/` prefix). Always include fallback.
- **`kernel_sources` paths are non-deterministic** — use `glob("/kaggle/input/**/expected_file", recursive=True)` to find them.
- **Kaggle GPU budget: 50h/week.** nb20a alone is ~3h, so each CNN backbone × 25 epochs × 5 folds eats ~6% of weekly budget. Plan accordingly.

---

## Section 7 — How to use this document

1. **When a Kaggle kernel finishes**: update Section 1 (LB history table). If a variant promotes a new baseline, update the "Decision rules" applied to subsequent variants.
2. **When starting a new experiment**: copy the relevant card from Section 3 into a fresh `nbXX-implementation.md` brief and add Kaggle gotchas from Section 6.
3. **When LB stalls for 3 consecutive variants on the same lever**: re-read Sections 2 and 5; pick a different family.
4. **Don't sequentialise**: many experiments are independent. Pick the one with highest expected gain that you have GPU budget for.

---

## Currently running (snapshot at 2026-05-17)

| Kernel | Slug | Status |
|---|---|---|
| nb19a | `birdclef-2026-topk-postproc` | ✅ DONE — LB 0.897 |
| nb20a run1 | `birdclef-2026-cnn-sed-b0-run1` | ✅ DONE — folds 0+1 OOF mean 0.9784 |
| nb20a run2 | `birdclef-2026-cnn-sed-b0-run2` | ✅ DONE — full 5-fold OOF 0.9792; submission.csv ready (LB pending submit) |
