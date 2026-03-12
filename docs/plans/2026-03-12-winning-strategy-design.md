# BirdCLEF+ 2026 — Winning Strategy Design

*Brainstormed: 2026-03-12. Revisit pseudo-labeling thresholds once first LB results arrive.*

---

## 1. Data Strategy

### 3-Tier Data Hierarchy

| Tier | Source | Weight | Notes |
|---|---|---|---|
| Gold | `train_soundscapes_labels.csv` (1,478 expert segments) | 1.0 | PRIMARY for non-bird + insect sonotypes — some species ONLY exist here |
| Silver | `train_metadata.csv`, XC rating ≥ 3.0 | 0.8 | High-quality curated clips |
| Bronze | XC rating < 3.0, iNat | 0.4–0.5 | Background noise, uncertain labels |
| Pseudo-label R1 | Round 1 PL | 0.5 | Confidence-thresholded |
| Pseudo-label R2 | Round 2 PL | 0.65 | — |
| Pseudo-label R3–4 | Rounds 3–4 PL | 0.75 | — |

### Geographic Filtering
- Training data spans all South America; test recorders cover 5°×2° Pantanal box (lat −21.6→−16.5, lon −57.6→−55.9)
- For bird pipeline: soft geographic weight — clips from Southern Cone/Amazonia upweighted 1.1× vs. Andean/Caribbean clips
- For non-bird pipeline: no geographic filtering (amphibians/insects too sparse)

### Critical 2026-Specific Notes
- **Insect sonotypes** (47158son01–son25): 25 acoustic morphotypes with ZERO training clips in `train_audio` — they exist ONLY in labeled soundscapes. Non-bird pipeline must treat these gold segments as primary data.
- **Site S22 dominance**: 65% of labeled soundscape segments come from site S22. Use GroupKFold by recorder site; hold out S22 as a standalone test set for unbiased evaluation.
- **Same recorders, different timestamps**: Test soundscapes share the same physical recorders as labeled training soundscapes — domain shift is temporal, not spatial. Background mix augmentation (see §3) directly mitigates this.

### Validation Protocol
- GroupKFold by recorder site (not StratifiedKFold — prevents site acoustic leakage)
- 5 folds; S22 held out as fixed eval set
- CV metric: macro ROC-AUC per pipeline, then merged

---

## 2. Model Architecture

### Bird Pipeline (162 classes)

| Model ID | Architecture | Role |
|---|---|---|
| B1 | EfficientNet-B1 (timm) | Lightweight workhorse, fast to iterate |
| B2 | EfficientNet-B3 (timm) | Mid-range, good accuracy/speed trade-off |
| B3 | EfficientNet-B4 (timm) | Top accuracy in bird pipeline |
| B4 | RegNetY-016 (timm) | Architectural diversity |
| B5 | BirdNET pretrained (EfficientNet-B5 backbone, 9,000+ species pretrain on XC/iNat) | Transfer learning for rare species |

All CNN backbones: `in_chans=1` (mono mel spectrogram), global average pooling → 234-way linear head, BCEWithLogitsLoss.

### Non-Bird Pipeline (72 classes: 35 Amphibia + 28 Insecta + 8 Mammalia + 1 Reptilia)

| Model ID | Architecture | Role |
|---|---|---|
| NB1 | ECA-NFNet-L0 (timm) | Strong on imbalanced small datasets |
| NB2 | EfficientNet-B3 with Focal BCE | Caiman + rare amphibians |

Non-bird heads use Focal BCE (γ=2, α=0.25) to handle extreme imbalance (Caiman latirostris: 1 training clip).

### ViTs as Pseudo-Label Generators Only (NOT for inference)

| Model | Role | Why excluded from inference |
|---|---|---|
| EVA-02 Large (448px) | High-confidence PL generation for bird pipeline | ~8–15s/chunk on CPU — blows 90-min budget |
| DINOv2-Large-reg4 | Diversity in PL agreement vote | Same |

**PL generator protocol**: Run EVA-02 and DINOv2 on unlabeled soundscapes on Kaggle GPU (training time only). Consensus with one CNN model → 3-model vote for pseudo-label assignment.

---

## 3. Training Strategy

### Audio → Spectrogram
```
SR=32000, N_FFT=1024, HOP_LENGTH=320, N_MELS=128, FMIN=40, FMAX=15000
TRAIN_DURATION=10s (2× inference; 2024 insight — longer context helps)
INFERENCE_DURATION=5s (competition chunk size)
```
Output shape: (1, 128, 1001) for 10s training, (1, 128, 501) for 5s inference.

### Augmentation Stack (applied in order)

| Aug | Probability | Notes |
|---|---|---|
| Background mix | 0.5 | Mix clip with random unlabeled soundscape segment (α∈[0.1, 0.3]) — primary domain adaptation |
| MixUp | 0.3 | α=0.4; mix two training spectrograms + labels |
| SpecAugment time mask | 0.5 | Up to 30 time frames |
| SpecAugment freq mask | 0.5 | Up to 16 mel bins |
| Gaussian noise | 0.2 | σ∈[0.001, 0.01] |
| Time shift | 0.3 | ±25% circular shift |
| Pitch shift | 0.2 | ±2 semitones (apply before mel, on waveform) |

Background mix is the single highest-ROI augmentation for domain adaptation — trains the model to recognize species vocalisations over real Pantanal background noise.

### Training Schedule

| Phase | Epochs | LR | Scheduler | Notes |
|---|---|---|---|---|
| Warmup | 3 | 1e-4 → 3e-4 | Linear | Gradual start |
| Main | 20 | 3e-4 → 1e-5 | CosineAnnealingLR | Primary learning |
| Fine-tune | 5 | 1e-5 → 1e-6 | Cosine | Stabilize |

**Total: 28 epochs per model per fold.** AdamW, weight_decay=1e-2. Mixed precision (AMP fp16). Batch size=32.

Gold segments resampled 3× per epoch (oversampled to compensate for low count).

---

## 4. Pseudo-Labeling Pipeline

### Schedule

| Round | Trigger | Models used to vote | Outputs |
|---|---|---|---|
| R1 | After first baseline CV score | B3 + EVA-02 + DINOv2 (2-of-3 agreement) | ~50K new samples |
| R2 | After R1-trained models converge | B3_r1 + B4_r1 + EVA-02 (2-of-3) | ~120K |
| R3 | After R2 convergence | B2_r2 + B3_r2 + B4_r2 (2-of-3) | ~200K |
| R4 | Final polish | Full bird ensemble (majority vote) | Refinement only |

### Confidence Thresholds (stratified by class rarity)

| Class type | Threshold |
|---|---|
| Common (>500 clips) | 0.70 |
| Medium (100–500 clips) | 0.80 |
| Rare (10–100 clips) | 0.85 |
| Very rare (<10 clips, incl. all non-bird) | 0.90 |

*These values are provisional — revisit after first LB results.*

### Power Scaling

Before assigning pseudo-labels, apply power scaling to soften overconfident predictions:
```python
probs_scaled = probs ** 0.7  # softens high-confidence predictions
```
This prevents the model from reinforcing its own errors in subsequent rounds.

### Sample Weights for PL Data

| Round | Weight |
|---|---|
| R1 PL | 0.50 |
| R2 PL | 0.65 |
| R3–4 PL | 0.75 |

---

## 5. Ensemble Merging and Inference

### Ensemble Architecture

| Pipeline | Models | Fusion |
|---|---|---|
| Bird (162 classes) | B1 + B2 + B3 + B4 + B5 | Rank averaging |
| Non-bird (72 classes) | NB1 + NB2 | Weighted average (0.6 / 0.4) |

**Why rank averaging for birds**: 5 models with different training histories (different PL rounds, BirdNET pretraining) have miscalibrated probability scales. Rank averaging is calibration-agnostic. Non-bird ensemble uses probability averaging because the 2 models are co-calibrated.

**Final merge**: Concatenate bird (162) + non-bird (72) = 234-column prediction matrix. No cross-pipeline blending.

### Inference Pipeline

```
test_soundscape.ogg
  → librosa.load(sr=32000)
  → 5s sliding window (5s stride for public LB; 0.5s stride + max-pool for final)
  → mel spectrogram (CFG params)
  → OpenVINO/ONNX model batch (batch=32)
  → softmax per model → rank-average
  → {row_id: [234 probs]} dict
  → submission CSV
```

### OpenVINO Export

- Export each fold's best checkpoint to ONNX immediately after training
- Convert: `mo --input_model best.onnx --data_type FP16`
- Warm up with 10 dummy batches before profiling
- Target: <45 min per model pass

### CPU Budget Allocation (approximate)

| Step | Time |
|---|---|
| Soundscape loading + preprocessing | 5 min |
| Bird ensemble (5 models × OpenVINO FP16) | 40 min |
| Non-bird ensemble (2 models) | 12 min |
| Post-processing + CSV write | 2 min |
| **Total** | **~59 min (31 min headroom)** |

If budget is tight, drop B5 (BirdNET) first — it's valuable for rare bird recall but the slowest model.

### Post-Processing

1. **Temperature scaling per class**: Fit `T_c` per class via Platt scaling on OOF predictions. Helps when BirdNET is overconfident on common species.

2. **Geotemporal prior**: Species with zero recorded occurrences in the Pantanal bounding box (from public iNat/eBird data) are floor-clipped to 0.001. Conservative boolean mask — free signal, no false negative risk for species that genuinely cannot occur there.

---

## Phase Milestones

| Phase | Target Date | Deliverable |
|---|---|---|
| Phase 1 complete | March 2026 | EDA + Spectrogram + Baseline notebooks uploaded; first LB score |
| Phase 2 start | April 2026 | Backbone sweep (B1–B4, RegNetY, NFNet); GroupKFold CV; background mix aug |
| Phase 2 complete | Early May 2026 | PL Round 1 done; non-bird pipeline trained; BirdNET integrated |
| Phase 3 start | Mid May 2026 | PL Rounds 2–4; full ensemble; OpenVINO profiling |
| Final submission | June 3, 2026 | Best ensemble, 0.5s stride, geotemporal prior applied |

---

## Open Questions (revisit after first LB)

- Pseudo-labeling thresholds (§4): provisional values, calibrate against OOF once baseline runs
- ViT fine-tuning depth: freeze backbone vs. full fine-tune for EVA-02 PL generation
- External data: does adding broader Xeno-canto (non-Pantanal) hurt, as it did in 2024? Validate on OOF
- Non-bird model complexity: ECA-NFNet-L0 may overfit on 551 non-bird clips — monitor validation loss closely
