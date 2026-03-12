# CLAUDE.md — BirdCLEF+ 2026

This file is the authoritative guide for all Claude Code sessions in this project.

---

## Competition Overview

| Field | Value |
|---|---|
| **Name** | BirdCLEF+ 2026 (LifeCLEF track, Kaggle code competition) |
| **Task** | Identify wildlife species (birds + other taxa) from passive acoustic monitoring recordings in the **Pantanal wetlands**, South America |
| **Metric** | Macro-averaged ROC-AUC (consistent with 2024–2025) |
| **Inference** | Code competition — must run within CPU time budget on Kaggle (~90–120 min) |
| **Start** | March 11, 2026 |
| **Entry deadline** | May 27, 2026 |
| **Final deadline** | June 3, 2026 |
| **Prize** | ~$50,000 (Kaggle) + $5,000 publication track (LifeCLEF) |

**Key novelties vs. 2025:**
- New region: Pantanal (Brazil) instead of Colombia — completely different species distribution
- ~1,000 passive acoustic recorders deployed continuously — scale challenge
- Multi-taxa: birds + insects + amphibians + mammals (Pantanal species)
- Dataset provided by the Pantanal PAM network

---

## Project Goals (in priority order)

1. **Win the competition** — primary objective, June 2026 deadline
2. **Didactic Kaggle notebooks** — educational notebooks targeting many upvotes (votes > score for early notebooks)
3. **Document experiments** — every meaningful run logged so we can build on it

**Resource budget**: 50h GPU/week on Kaggle (P100/T4), 24 weeks → ~1,200h total GPU.
Local machine: GTX 1050 (prototyping and EDA only, not for serious training).

---

## Repository Structure

```
bird_clef/
├── CLAUDE.md                    # This file
├── data/                        # Local data (gitignored except metadata CSVs)
│   ├── train_audio/             # Training clips (OGG, from Xeno-canto + field)
│   ├── test_soundscapes/        # Competition test soundscapes
│   ├── unlabeled_soundscapes/   # Unlabeled PAM recordings (pseudo-labeling fuel)
│   ├── train_metadata.csv
│   ├── taxonomy.csv
│   └── sample_submission.csv
├── notebooks/                   # Local dev notebooks
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── ...
├── kaggle_notebooks/            # Notebooks ready to upload to Kaggle
│   ├── 01_eda_birdclef2026.py
│   ├── 02_baseline_efficientnet.py
│   └── ...
├── src/                         # Reusable Python modules
│   ├── audio.py                 # Spectrogram, augmentation utils
│   ├── dataset.py               # PyTorch Dataset classes
│   ├── models.py                # Model definitions
│   ├── train.py                 # Training loop
│   ├── inference.py             # Inference + OpenVINO export
│   └── pseudo_label.py          # Pseudo-labeling pipeline
├── configs/                     # YAML experiment configs
├── submissions/                 # CSV submission files
├── results/                     # OOF scores, experiment logs
└── winning_notebooks/           # Archived top solutions from past years
```

---

## Data Notes (verified from local files — 2026-03-12)

| Item | Value |
|---|---|
| `train.csv` | 35,549 clips — primary_label, secondary_labels, type, lat/lon, scientific_name, common_name, class_name, inat_taxon_id, author, license, rating, url, filename, collection |
| Aves clips | 34,799 (97.9%) across ~162 bird species |
| Amphibia clips | 451 across 29 frog species |
| Insecta clips | 199 across 3 species + 25 sonotypes of iNat 47158 |
| Mammalia clips | 99 across 7 species (Jaguar, Howler Monkey, Capuchin, Marmoset, Titi, Horse, Cattle, Dog) |
| Reptilia clips | **1** — Caiman yacare only |
| Xeno-canto (XC) | 23,043 clips (bird codes e.g. `ashgre1`, `banana`) |
| iNaturalist (iNat) | 12,506 clips (numeric folder IDs e.g. `22961`, `41970`) |
| `taxonomy.csv` | 234 species — primary_label, inat_taxon_id, scientific_name, common_name, class_name |
| `train_soundscapes/` | 10,658 OGG files — real Pantanal PAM recordings |
| `train_soundscapes_labels.csv` | 1,478 labeled 5s segments — NEW in 2026; format: filename, start, end, primary_label (semicolon-separated) |
| `test_soundscapes/` | Hidden on Kaggle; row_id format: `{stem}_{end_sec}` |
| `sample_submission.csv` | 234 species columns (taxonomy order) |
| Recorder region | Pantanal, Mato Grosso do Sul, Brazil — lat -21.6→-16.5, lon -57.6→-55.9 |
| Audio format | OGG; target SR=32000; train clips vary 5–60s; soundscapes are minutes-long |
| Insect sonotypes | iNat 47158 split into 47158son01–son25 (25 independent classes) |

---

## SOTA — What Won in 2024 and 2025

### BirdCLEF+ 2025 — 1st Place (Nikita Babych) — Score: 0.933
- **Architecture**: Ensemble of EfficientNet-L0, B4, B3 + RegNetY-016, RegNetY-008
- **Pseudo-labeling**: Multi-round noisy student (4+ rounds) — THE key differentiator
- **Power scaling**: Scale pseudo-label confidences before re-training
- **Separate pipeline**: Birds vs. other taxa (insects/amphibians) handled separately
- **External data**: +5,489 bird + 17,197 insect/amphibian entries from Xeno Archive
- **Augmentation**: MixUp, StochasticDepth
- **Loss**: CrossEntropy

### BirdCLEF+ 2025 — 2nd Place (Volodymyr Sydorskyi) — Score: 0.928
- **Architecture**: ECA-NFNet-L0 + TF-EfficientNetV2-S-IN21K
- **Input**: 5s clips → mel spectrograms, HDF5 for fast loading
- **Loss**: Focal BCE with label smoothing
- **Semi-supervised**: Confidence-thresholded pseudo-labels
- **Deployment**: ONNX → fp16 → OpenVINO (mandatory for CPU budget)
- **Pretraining**: Xeno-canto, iNaturalist, CSA before fine-tuning

### BirdCLEF 2024 — 1st Place — Score: 0.739 (182 species, India)
- **Architecture**: Ensemble of 6× EfficientNet-B0 + RegNetY-008
- **Key audio trick**: 10s training chunks with 2.5s context overlap at inference
- **Data cleaning**: Google Bird Vocalization Classifier + quantile filtering (0.8-quantile on std/var/rms/pwr)
- **Loss**: CrossEntropy (NOT BCE — notable difference)
- **Inference tricks**: Sigmoid (not softmax), min() ensemble reduction, OpenVINO + joblib parallelism
- **External data**: Adding Xeno-canto did NOT help for 2024 — validate per year

### Key Patterns Across Years
1. **EfficientNet family** (B0–B4, L0) + **RegNetY** backbone duo consistently wins
2. **Pseudo-labeling on unlabeled soundscapes** is mandatory to reach top 5
3. **OpenVINO export** needed for CPU inference budget
4. **CrossEntropy or Focal BCE** — validate per year, not interchangeable
5. **Mel spectrograms** at n_mels=128, fmin=40Hz, fmax=15kHz as standard baseline
6. **Augmentation**: MixUp/CutMix + XY masking (frequency + time masking)
7. **Separate handling of different taxa** (birds vs. others) once multi-taxa was introduced in 2025
8. **SED (Sound Event Detection)** models provide frame-level signal but add complexity
9. **ONNX/OpenVINO** — always export early; CPU budget often bites teams in the final week

---

## Phase Plan

### Phase 1: Foundation (March 2026) — EDA + Baseline
**Goal**: Upload didactic notebooks targeting upvotes, establish working baseline

| Notebook | Target | Purpose |
|---|---|---|
| `01_eda_birdclef2026.py` | Upvotes | Data exploration: species distribution, audio properties, Pantanal geography |
| `02_spectrogram_guide.py` | Upvotes | Visual guide to mel spectrograms for bird audio |
| `03_baseline_efficientnet.py` | Score + Upvotes | Clean EfficientNet-B0 baseline with full pipeline |
| `04_augmentation_showcase.py` | Upvotes | MixUp, CutMix, SpecAugment visual demos |
| `05_inference_openvino.py` | Upvotes | How to export to ONNX + OpenVINO for CPU inference |

### Phase 2: Experimentation (April–May 2026)
- Backbone search: EfficientNet-B0/B3/L0, RegNetY-008/016, ECA-NFNet-L0, BirdNET
- Pseudo-labeling round 1 (confidence threshold sweep)
- External data (Xeno-canto validation for Pantanal species)
- Separate bird vs. other-taxa heads

### Phase 3: Push to Gold (May–June 2026)
- Multi-round pseudo-labeling (4+ rounds, power scaling)
- Ensemble diversification
- Inference optimization (OpenVINO, fp16)
- Final ensemble + submission

---

## Experiment Tracking

Log every meaningful run in `results/experiment_log.md`:
```
| ID | Date | Model | Backbone | Pseudo-label round | CV score | LB score | Notes |
```

---

## Technical Conventions

### Audio Processing
```python
SR = 32000          # Sample rate
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
FMIN = 40
FMAX = 15000
CLIP_DURATION = 5   # seconds for inference
TRAIN_DURATION = 10 # seconds for training (2024 insight: longer helps)
```

### Inference Budget Management
- Always export to ONNX early in the competition
- Profile inference time on Kaggle CPU before final submission week
- Use joblib for parallel spectrogram precomputation
- Target: <60s/soundscape to leave headroom

### Kaggle Notebook Conventions
- Use `.py` format (Kaggle script mode) for all training notebooks
- Add `# %%` cell markers for Kaggle's notebook renderer
- Pin library versions in first cell
- All paths via `Path('/kaggle/input/birdclef-2026')`
- Add `if __name__ == '__main__':` guard for imports

### Model Saving
- Save `best_model.pth` per fold
- Also save ONNX immediately after training
- Log OOF AUC to `results/oof_scores.json`

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Pantanal species distribution shift vs. training data | Extensive EDA + domain adaptation |
| CPU inference budget exceeded | Export to OpenVINO early; profile often |
| Pseudo-label noise | Power scaling + multiple rounds + confidence thresholds |
| External data hurts (as in 2024) | Always validate against OOF before using |
| Taxa imbalance (birds >> others) | Separate heads or class-weighted loss |

---

## Git & Push Workflow

- Remote alias: `birdclef` → public repo (to be created)
- Push after each phase: `git subtree push --prefix="Kaggle competition/bird_clef" birdclef master`
- Commit format: `feat/fix/exp: description` + `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`
- Large audio files are gitignored; only notebooks, src, configs, and results CSVs are committed

---

## References

- [BirdCLEF+ 2026 — Kaggle](https://www.kaggle.com/competitions/birdclef-2026/overview)
- [BirdCLEF++ — LifeCLEF 2026](https://www.imageclef.org/BirdCLEF2026)
- [2025 1st Place Solution (Nikita Babych)](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n)
- [2025 2nd Place Solution (VSydorskyy)](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place)
- [2024 1st Place Write-up (Zenn)](https://zenn.dev/yuto_mo/articles/ad43c630729073)
- BirdNET pretraining weights: [github.com/kahst/BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer)
