# BirdCLEF+ 2026 — nb15 Design: Perch Logit Features

*Brainstormed: 2026-05-07. Builds on nb14c (BiGRU Augment, LB 0.879).*

---

## Context

| Notebook | Arch | LB AUC | OOF AUC | Key change |
|---|---|---|---|---|
| nb13 | Perch MLP baseline | 0.839 | 0.4126* | PCA(64) + scalars(5) |
| nb14a | BiGRU Replace | 0.875 | 0.5222 | BiGRU ctx(5) replaces scalars |
| nb14c | BiGRU Augment | **0.879** | 0.5049 | BiGRU ctx(8) appended to scalars |

*OOF for nb13 inferred from nb14c ablation-off baseline.

nb14c already extracts `scores_tr` of shape `(N, 234)` — the Perch logit projected onto
our 234 target species per 5s window (0 for unmapped species). This array is currently
used only for alpha-blending at submission time (0.7 MLP + 0.3 Perch sigmoid).

**Hypothesis**: Putting Perch logits *inside* the MLP as input features lets the model
learn to weight them conditionally per species, rather than applying a fixed 0.7/0.3 blend.
If this works, the hand-tuned alpha-blend becomes redundant.

---

## Experiment Matrix — 3 × 2 = 6 Notebooks

| | Blend ON (α=0.7/0.3) | Blend OFF (MLP only) |
|---|---|---|
| **Per-class logit** (1 scalar/probe) | nb15a | nb15b |
| **Global logit ctx** (PCA-32 shared) | nb15c | nb15d |
| **Both** | nb15e | nb15f |

All 6 build on nb14c exactly. The only changes are:
1. What gets concatenated into `build_mlp_input`
2. Whether the alpha-blend fires at submission time

---

## Architecture Changes vs nb14c

### MLP Input Dimensions

```
nb14c baseline : PCA(64) + scalars(5) + BiGRU ctx(8)           = 77-dim
nb15a/b        : PCA(64) + scalars(5) + BiGRU ctx(8) + logit(1) = 78-dim
nb15c/d        : PCA(64) + scalars(5) + BiGRU ctx(8) + glogit(32) = 109-dim
nb15e/f        : PCA(64) + scalars(5) + BiGRU ctx(8) + logit(1) + glogit(32) = 110-dim
```

### Per-class Logit Feature (nb15a/b, nb15e/f)
`scores_tr` (N, 234) is already available — the Perch logit for each of our 234 target
species per window. For class probe `i`, we concatenate `scores_tr[:, i]` as a single
scalar. For unmapped species (insects, some amphibians), this is 0 — the model learns
to ignore it.

```python
logit_feat_tr = scores_tr.astype(np.float32)  # (N, 234), already available

# In build_mlp_input:
lb = logit_b.unsqueeze(-1)  # (B, 234, 1)
return torch.cat([p, scalars_b, c, lb], dim=-1)  # (B, 234, 78)
```

### Global Logit Context (nb15c/d, nb15e/f)
PCA over the 234-dim projected logit vector → 32-dim shared context. This captures
"what Perch thinks is generally present in this window" as a compressed scene descriptor.

```python
LOGIT_PCA_DIM = 32
logit_pca_fit = PCA(n_components=LOGIT_PCA_DIM, random_state=42).fit(scores_tr)
logit_ctx_tr  = logit_pca_fit.transform(scores_tr).astype(np.float32)  # (N, 32)

# In build_mlp_input:
lc = logit_ctx_b.unsqueeze(1).expand(B, N_CLASSES, LOGIT_PCA_DIM)  # (B, 234, 32)
return torch.cat([p, scalars_b, c, lc], dim=-1)  # (B, 234, 109)
```

### Alpha-blend Toggle

```python
# Blend ON  (a, c, e): same as nb14c
final = alpha_per_class[None, :] * mlp_pred + (1.0 - alpha_per_class[None, :]) * perch_sig_te

# Blend OFF (b, d, f): MLP is the full prediction
final = mlp_pred
```

### Ablation for Diagnostics

Ablation OFF: zero the logit features (keep scalars + BiGRU ctx) → approximates nb14c
behavior within this model. `delta_logit = OOF_on - OOF_off` measures marginal logit
feature contribution on top of nb14c.

```python
# nb15a/b ablation: zero the per-class logit scalar
logit_zero = torch.zeros(len(Xp_va), N_CLASSES)
val_pred_off = sigmoid(mlp(build_mlp_input(Xp_va, Xs_va, ctx_va, logit_zero)))

# nb15c/d ablation: zero the global logit ctx
logit_ctx_zero = torch.zeros(len(Xp_va), LOGIT_PCA_DIM)
val_pred_off = sigmoid(mlp(build_mlp_input(Xp_va, Xs_va, ctx_va, logit_ctx_zero)))
```

---

## Training Config (unchanged from nb14c)

```python
PCA_DIM    = 64
SCALAR_DIM = 5
GRU_HIDDEN = 32
CTX_DIM    = 8
LOGIT_PCA_DIM = 32  # new (global variants only)

N_FOLDS = 5
EPOCHS  = 30
LR      = 1e-3
WD      = 1e-4
BATCH_SZ = 128
ALPHA    = 0.7     # used only in blend-ON variants
```

---

## Diagnostics

Each notebook outputs:
- `diagnostics_nb15x.json` — global OOF summary
- `per_class_auc_nb15x.csv` — per-class AUC table
- `pred_dist_nb15x.png` — prediction distributions

JSON fields:
```json
{
  "notebook": "nb15a",
  "model": "per-class-logit-blend-on",
  "blend": true,
  "logit_mode": "per_class",
  "oof_auc_perch":     0.7478,
  "oof_auc_logit_off": "X.XXXX (≈ nb14c)",
  "oof_auc_logit_on":  "X.XXXX",
  "delta_logit":       "on - off",
  "nb14c_reference":   0.5049,
  "mlp_in_dim":        78
}
```

---

## Kernel IDs

| Notebook | Kaggle ID | Title |
|---|---|---|
| nb15a | `alexycactus/birdclef-2026-logit-cls-blend` | BirdCLEF 2026 Logit Class Blend |
| nb15b | `alexycactus/birdclef-2026-logit-cls-noblend` | BirdCLEF 2026 Logit Class NoBlend |
| nb15c | `alexycactus/birdclef-2026-logit-global-blend` | BirdCLEF 2026 Logit Global Blend |
| nb15d | `alexycactus/birdclef-2026-logit-global-noblend` | BirdCLEF 2026 Logit Global NoBlend |
| nb15e | `alexycactus/birdclef-2026-logit-both-blend` | BirdCLEF 2026 Logit Both Blend |
| nb15f | `alexycactus/birdclef-2026-logit-both-noblend` | BirdCLEF 2026 Logit Both NoBlend |

All: CPU, internet=ON, same dataset/model sources as nb14c.

---

## Success Criteria

- Any variant with `delta_logit > 0` and OOF > 0.5049 → submit for LB
- If blend-OFF variants match or beat blend-ON on LB → retire alpha-blend in future notebooks
- Best blend-OFF variant hitting LB > 0.879 → alpha-blend was a crutch, not complementary signal
