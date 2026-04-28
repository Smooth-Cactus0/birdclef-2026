# BirdCLEF+ 2026 — Perch Pipeline Design

**Date**: 2026-04-28
**Context**: First LB score with CNN dual-pipeline: 0.689. Top leaderboard: ~0.93.
**Decision**: Switch to Google Perch v2 + lightweight head approach.

---

## Why This Approach

The entire top of the leaderboard uses Google Perch v2 (frozen feature extractor pretrained on
500M+ bird audio clips) + a lightweight head trained on top of its embeddings. Our EfficientNet
approach starts from ImageNet pretraining and tries to learn bird acoustics from scratch.
Perch provides the signal floor (~0.80+); the head + post-processing chain extracts the rest.

---

## Architecture

```
STAGE 1 — Feature Extraction (run once, cache to disk)
  train_soundscapes/ (59 files, 708 windows)  ──┐
  train_audio/       (35k clips, ~35k+ windows) ──┼──► Perch v2 ONNX (frozen) ──► cache
  test_soundscapes/  (run at submission time)  ──┘
      outputs: 1536-dim embeddings + 234-class logits + metadata

STAGE 2 — Head Training (experiments)
  cache ──► Layer 0: raw Perch logits
          ► Layer 1: MLP probes (vectorized, PCA-64 + temporal features)
          ► Layer 2: site/hour priors (no training, just counting)
          ► Layer 3: Bidirectional SSM (temporal reasoning on 12-window sequences)

STAGE 3 — Post-processing chain (stack incrementally)
  window-level predictions ──► TTA ──► adaptive smoothing ──► temperature
                            ──► confidence scaling ──► isotonic calibration
                            ──► per-class thresholds ──► submission.csv
```

---

## Notebooks

| Notebook | Purpose | Kaggle config |
|---|---|---|
| `nb13` | Perch extraction + cache builder | CPU, internet=ON first run |
| `nb14` | Head training + submission | CPU, internet=OFF, cache dataset attached |

---

## Stage 1 — Feature Extraction (nb13)

**Model**: `google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1` (Kaggle model)
**ONNX**: `rishikeshjani/perch-onnx-for-birdclef-2026` (150x faster than TF SavedModel)
**Prefetch**: `ThreadPoolExecutor(max_workers=4)` for I/O overlap with ONNX inference

**Cache files:**
```
perch_cache/
  train_sc_meta.parquet      # filename, start_sec, end_sec, site, hour_utc, labels
  train_sc_embs.npz          # float32 (N_sc_windows, 1536)
  train_sc_scores.npz        # float32 (N_sc_windows, 234)
  train_audio_meta.parquet   # filename, window_idx, primary_label, secondary_labels, rating
  train_audio_embs.npz       # float32 (N_audio_windows, 1536)
  train_audio_scores.npz     # float32 (N_audio_windows, 234)
```

**train_audio windowing strategy:**
- Clips < 5s: zero-pad to 5s (1 window)
- Clips 5-10s: first 5s only (1 window)
- Clips > 10s: sliding 5s windows, no overlap
- Each window labels: primary_label=1.0, secondary_labels=0.5
- Confidence weight: `rating / 5.0` (XC quality rating 1-5)

**Genus proxy logits**: For unmapped species (no direct Perch entry), average logits of
all same-genus species in Perch's 10k vocab.

---

## Stage 2 — Head Architecture (nb14, Layer by Layer)

### Layer 0 — Raw Perch logits baseline
Slice 234 relevant logits from Perch output + genus proxies.
No training. Expected LB: ~0.78-0.82.

### Layer 1 — MLP probes
- Input: `[PCA(emb, 64-dim) + temporal features (prev, next, mean, max, std across file)]` = ~70-dim
- Architecture per class: 70 → 128 → 64 → 1 (sigmoid)
- All 234 probes vectorized as single PyTorch `bmm` call
- Training data: train_audio windows (primary) + train_soundscapes windows
- Loss: BCE with inverse-sqrt class frequency weights
- XC rating used as per-sample confidence weight
- Expected gain: +0.03-0.06

### Layer 2 — Site/hour priors
- Build frequency tables P(species|site) and P(species|hour_utc) from train_soundscapes_labels.csv
- Applied as: `logit += lambda * log(P(species|site) / P(species_global))`
- No training required
- Expected gain: +0.01-0.02

### Layer 3 — Bidirectional SSM
- Input: (n_files, 12, 1536) — 12 consecutive windows as a sequence
- Architecture: 2x SelectiveSSM (d_model=128, d_state=16) + cross-attention + bidirectional merge
- Training: ONLY on train_soundscapes (needs real temporal sequence structure)
  train_audio clips have no natural sequence context between them
- Expected gain: +0.02-0.04

**Final ensemble**: `alpha*Perch + beta*MLP + gamma*SSM` with weights tuned by OOF AUC

---

## Stage 3 — Post-processing Chain

Applied in order (each is an independent on/off ablation):

| Step | What it does | Expected gain | Notes |
|---|---|---|---|
| TTA | Circular-shift 12-window sequence by {0,+1,-1,+2,-2}, average 5 runs | +0.003-0.005 | Weights: [0.4, 0.2, 0.2, 0.1, 0.1] |
| Adaptive smoothing | Blend uncertain windows toward neighbors; alpha = base*(1-confidence) | +0.003-0.005 | Fixed alpha hurts (dilutes peaks); adaptive protects them |
| Per-taxon temperature | T=0.95 for Amphibia/Insecta (continuous callers), T=1.10 for Aves | +0.002-0.003 | Applied before sigmoid |
| Confidence scaling | `score *= (file_max)^0.4` — suppress uncertain files | +0.002-0.004 | Prevents noisy files from generating false positives |
| Isotonic calibration + thresholds | Per-class isotonic regression on OOF + F1-threshold grid search | +0.004-0.008 | Biggest single post-processing gain; applied at file level |

---

## Research Agenda (next experiments after baseline)

| Direction | Hypothesis | How to test |
|---|---|---|
| Domain weighting | Soundscape examples 3-5x weight over train_audio in MLP | Compare OOF AUC with/without |
| Pseudo-labeling | unlabeled_soundscapes + high-confidence Perch predictions | Run Perch, threshold at 0.7, retrain head |
| Secondary labels | Include secondary_labels at 0.5 weight | Add to train_audio_meta.parquet |
| Multi-scale Perch | 2.5s and 10s windows in addition to 5s | Modify nb13 windowing |
| Hour-of-day in SSM | Add hour embedding to SSM input | Modify SSM input projection |
| Genus proxy refinement | Weight proxies by taxonomic distance | Build distance matrix from taxonomy |

---

## Build Order (LB checkpoints)

1. nb13: Perch cache builder — verify embeddings look sane locally
2. nb14 v1: Layer 0 only (raw Perch) → first Perch LB score
3. nb14 v2: + Layer 1 MLP (train_audio only) → measure MLP gain
4. nb14 v3: + Layer 1 MLP (train_audio + soundscapes) → measure soundscape gain
5. nb14 v4: + Layer 2 priors → measure metadata gain
6. nb14 v5: + Layer 3 SSM → measure temporal reasoning gain
7. nb14 v6+: post-processing chain, one step at a time
