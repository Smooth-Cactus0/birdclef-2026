# nb14: Temporal Context Models on Perch Embeddings

**Date**: 2026-05-07  
**Status**: In progress — nb14a implementing  
**Baseline**: nb13 LB 0.839 (Perch v2 ONNX + VectorizedMLP, PCA(64)+5 hand-crafted scalars)

## Hypothesis

The 5 hand-crafted temporal scalars in nb13 (prev/next/mean/max/std of per-class Perch logit)
encode coarse statistics. A learned temporal model over the 12-window sequence can capture
richer patterns — onset/offset shape, call clustering, sequential context — that improve AUC.

## 2×2 Experiment Matrix

|                         | BiGRU  | Self-attention |
|-------------------------|--------|----------------|
| **Replace scalars (A)** | nb14a  | nb14b          |
| **Augment MLP (C)**     | nb14c  | nb14d          |

- A variants: learned context replaces the 5 hand-crafted scalars → 69-dim MLP input (same as nb13)
- C variants: learned context appended alongside nb13 features → 77-dim MLP input

## Shared Pipeline (all 4 notebooks)

- Self-contained: re-run Perch ONNX on 66 labeled soundscapes (~3 min), no kernel_sources dep
- PCA(64) on 1536-dim Perch embeddings
- File sequence matrix: (n_files, 12, 64) — ordered PCA embeddings per file
- 5-fold GroupKFold by filename (same as nb13)
- VectorizedMLP (234 probes via bmm)
- Alpha-blend with Perch sigmoid (same class-conditional weights as nb13)
- All three diagnostics

## Temporal Model Architectures

### BiGRU (nb14a, nb14c)
```
Input:  (n_files, 12, 64)
BiGRU:  hidden=32, bidirectional → output (n_files, 12, 64)
Drop:   dropout=0.1
Linear: (64 → CTX_DIM)
```
`CTX_DIM = 5` for A variants (replace scalars, keeps MLP at 69-dim total).
`CTX_DIM = 8` for C variants (augment, MLP at 77-dim total).

### Self-attention (nb14b, nb14d)
```
Input:         (n_files, 12, 64) + sinusoidal positional encoding
Transformer:   d_model=64, nhead=4, dim_feedforward=128, dropout=0.1
Linear:        (64 → CTX_DIM)
```
The 12×12 attention weight matrix is saved as a diagnostic artifact for interpretability.

## Training

- Joint end-to-end: temporal model + VectorizedMLP trained together
- Adam lr=1e-3, WD=1e-4, CosineAnnealingLR, 30 epochs, batch=128
- Gradient flows through temporal model per batch (no detached pre-computation)
- Per batch: `torch.unique(fids, return_inverse=True)` → run GRU on unique files → index per-window context

## Integration Details

### Option A — Replace temporal scalars
- MLP input: [PCA(64) | ctx(5)] = 69-dim, broadcast to (B, 234, 69)
- Ablation OFF: ctx = zeros(5)
- The shared context replaces nb13's class-specific hand-crafted scalars

### Option C — Augment MLP
- MLP input: [PCA(64) | hand_crafted_scalars(5) | ctx(8)] = 77-dim
- Ablation OFF: zero out only the ctx(8) slice, hand-crafted scalars retained
- Directly tests whether the temporal model adds value on top of nb13's features

## Diagnostics (all 3 types per notebook)

1. **Per-class AUC table** (`per_class_auc_nb14x.csv`):
   columns: species | n_pos | auc_perch | auc_off | auc_on | delta
   Sorted by auc_on descending. Enables cross-notebook comparison.

2. **Global ablation delta**: printed + saved to `diagnostics_nb14x.json`
   keys: oof_auc_on, oof_auc_off, delta, n_active_classes, n_train_windows, n_train_files

3. **Prediction distributions** (`pred_dist_nb14x.png`):
   4×5 grid of histograms for top-10 / bottom-10 species by auc_on.
   Each plot: neg (blue) vs pos (red) prediction distributions.

## File Naming

| Notebook | File                              | Kernel ID                              |
|----------|-----------------------------------|----------------------------------------|
| nb14a    | kaggle_notebooks/14a_bigru_replace.py   | alexycactus/birdclef-2026-bigru-replace   |
| nb14b    | kaggle_notebooks/14b_attn_replace.py    | alexycactus/birdclef-2026-attn-replace    |
| nb14c    | kaggle_notebooks/14c_bigru_augment.py   | alexycactus/birdclef-2026-bigru-augment   |
| nb14d    | kaggle_notebooks/14d_attn_augment.py    | alexycactus/birdclef-2026-attn-augment    |

## Build Order

nb14a → nb14b → nb14c → nb14d → compare all 4 diagnostics → best for nb15 (TTA+smoothing)
