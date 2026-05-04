# BirdCLEF+ 2026 — Perch + MLP Baseline (nb13)

**Date**: 2026-05-04
**Context**: nb13 cache builder OOM'd after 7h. Resetting to a leaner, single-notebook approach modeled on the working public notebook `birdclef-26-onnx-perch-dual-ssms-vectorized-mlp`. Goal: a working Perch-based submission we own and understand, not a fork.
**Target LB**: 0.85–0.88 (vs current 0.689 from CNN dual-pipeline).
**Replaces**: `2026-04-28-perch-pipeline-design.md` for v1.

---

## Why This Reset

The previous design tried to cache Perch features for **both** train_soundscapes (708 windows) and train_audio (~200k windows). The OOM happened because pre-allocating arrays for 200k+ windows consumed ~1.7 GB before any inference, then per-batch buffers + accumulated metadata pushed the worker over the limit at ~hour 6.

The two best public notebooks at LB 0.93 do **not** cache train_audio at all. They only train heads on the 708 soundscape windows. Adding train_audio added 99.6% of the runtime and memory cost for a feature the proven approach doesn't use.

For v1, we strip the design back to what the proven notebooks actually do.

---

## Decisions Locked In

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Approach | Rewrite cell-by-cell from public notebook A as template | We own the code; can iterate component-by-component |
| 2 | Template | `birdclef-26-onnx-perch-dual-ssms-vectorized-mlp.ipynb` | 8 components vs 12 in pantanal-distill — easier to faithfully reproduce |
| 3 | Structure | Single notebook | Public notebook ships this way; one push = one place for bugs |
| 4 | Scope | Layer 0 + MLP probes only | Skip SSM and post-processing chain — those become nb14, nb15, nb16 |
| 5 | MLP input | PCA(emb,64) + 5 temporal feats = 69-dim | Captures partial-SSM benefit via cross-window stats; reduces overfit |
| 6 | CV | 5-fold GroupKFold by filename | OOF AUC = stable signal; ensemble effect at test time |
| 7 | Loss | BCE + inverse-sqrt class weights | Long-tail Pantanal labels need rare-class lift |
| 8 | Optim | Adam(lr=1e-3, wd=1e-4), cosine 30ep, batch=128 | Public notebook A defaults |

---

## Architecture & Flow

```
1.  Install onnxruntime wheel (from rishikeshjani/perch-onnx-for-birdclef-2026)
2.  Load taxonomy + train_soundscapes_labels.csv + sample_submission.csv
3.  Build Y_SC: groupby(filename,start,end) → union → multi-hot (N, 234)
4.  Species mapping: MAPPED_POS, MAPPED_BC_IDX, genus proxy_map
5.  ONNX session warmup
6.  Extract Perch features on 59 train_soundscapes
    → train_emb (708, 1536), train_scores (708, 234)
7.  Fit PCA(emb, 64) on train_emb → save fitted PCA
8.  Build MLP input features [PCA-64 + 5 temporal scalars] = 69-dim per (window, class)
9.  5-fold GroupKFold by filename → train 5 vectorized per-class MLPs
    → log fold AUC + OOF AUC + per-class AUC distribution
10. Extract Perch features on test_soundscapes (variable N, falls back to train_sc in staging)
11. Build test features → run all 5 MLPs → average → mlp_pred
12. final = alpha · mlp_pred + (1-alpha) · sigmoid(raw_perch_test)   # alpha=0.7
13. Build submission.csv aligned to sample_submission row order
```

---

## Data Structures & Label Alignment

The single most error-prone step. Any silent off-by-one caps OOF AUC at ~0.5 forever and we'd never know which downstream component is the real problem.

**Soundscape label table**:
```python
sc = (soundscape_labels
      .groupby(["filename","start","end"])["primary_label"]
      .apply(union_labels)
      .reset_index(name="label_list"))
sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
sc["row_id"]  = sc["filename"].str.replace(".ogg","",regex=False) + "_" + sc["end_sec"].astype(str)
```
The CSV has 1478 (filename, start, end, species) rows; groupby collapses to ~708 unique window keys.

**Train extraction order**:
For each of 59 sorted train_soundscape files, produce 12 windows in fixed order. After all 59 files, `meta_tr` has 708 rows in deterministic order: filename ascending, then window 0..11.

**Alignment**:
```python
Y_SC_LOOKUP = dict(zip(sc["row_id"], sc["label_list_multihot"]))
Y_TR = np.stack([Y_SC_LOOKUP.get(rid, zeros(234)) for rid in meta_tr["row_id"]])
assert (Y_TR.sum(1) > 0).sum() >= 700, "Most windows should have at least one label"
```
Windows missing from `sc` get all-zero rows — correct behavior, not a bug.

---

## MLP Architecture

**Vectorized per-class MLP** via `torch.bmm` — all 234 probes computed in parallel:

```
Per layer ℓ, store W[234, in_ℓ, out_ℓ] and b[234, out_ℓ]
Input x: (B, in_0=69)  →  expand to (234, B, 69)
Layer 1: bmm(x, W1) + b1  →  (234, B, 128) → ReLU
Layer 2: bmm(x, W2) + b2  →  (234, B, 64)  → ReLU
Layer 3: bmm(x, W3) + b3  →  (234, B, 1)
Output: squeeze + transpose → (B, 234) sigmoid logits
```

**Input features (69-dim per (window, class))**:
- 64 dims: `PCA(emb, 64).transform(window_emb)` — same PCA fit on train, applied to test
- 5 dims: temporal scalars from raw Perch logit for the *target species*:
  - `prev_score` — Perch logit at window-1 (circular within file)
  - `next_score` — Perch logit at window+1 (circular within file)
  - `mean_score`, `max_score`, `std_score` — across all 12 windows of the same file

The 5 temporal features are **species-specific scalars**, not 5×64 arrays. They give each per-class MLP a small "context" of how the species behaves elsewhere in the same recording.

**Training loop (per fold)**:
```python
weights_c = 1 / np.sqrt(Y_train.sum(0) + 1)        # 234-dim, computed on train fold only
loss = (BCEWithLogitsLoss(reduction='none')(logits, Y) * weights_c).mean()
optim = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = CosineAnnealingLR(optim, T_max=30, eta_min=1e-5)
for epoch in range(30):
    for batch in DataLoader(...):
        ... standard BCE training ...
torch.save(model.state_dict(), f"fold_{k}_mlp.pt")
```

**Reporting**:
```
Fold 0 AUC: 0.8xxx  (active classes: 71)
Fold 1 AUC: 0.8xxx  (active classes: 68)
...
OOF AUC (708 windows, macro across 234): 0.8xxx
Per-class AUC: median=0.8xx, p10=0.6xx, p90=0.9xx, n_below_0.5: NN
```

---

## Inference & Submission

```python
test_files = sorted(test_soundscapes_dir.glob("*.ogg"))
# Staging fallback for empty test_soundscapes/ in preview env
if not test_files:
    test_files = sorted(train_soundscapes_dir.glob("*.ogg"))[:16]
```

For each test file: load, reshape to (n_wins, 160000), ONNX → logits + embs. Save row_ids `<stem>_<end_sec>`.

```python
pca_test = pca_fit.transform(test_emb)
# Build test features same as train (PCA-64 + 5 temporal per species)
mlp_pred_per_fold = [sigmoid(model_k(test_features)) for k in range(5)]
mlp_pred = mean(mlp_pred_per_fold, axis=0)
final    = alpha * mlp_pred + (1 - alpha) * sigmoid(raw_perch_test)   # alpha=0.7
```

**Submission build** (the only safe pattern):
```python
pred_df = pd.DataFrame(final, columns=label_list)
pred_df["row_id"] = test_row_ids
sub = sample_sub[["row_id"]].merge(pred_df, on="row_id", how="left")
sub[label_list] = sub[label_list].fillna(1.0 / N_CLASSES)
sub.to_csv("/kaggle/working/submission.csv", index=False)
```

---

## Risk Mitigations (explicit guards)

1. **Assert OOF AUC > 0.6 on train** before extracting test Perch features. If the MLP didn't train, no point burning ~30 min of test ONNX inference.
2. **Assert `len(sub) == len(sample_sub)`** and `sub.columns[1:].tolist() == label_list` before writing. Defends against silent column reorder bugs.
3. **Memory bound on test inference**: stream by batches of 16 files; release audio buffers after reshape. Don't pre-allocate based on test file count.
4. **Print Perch ONNX timing per batch** — early-warning signal for CPU contention or oversized batches.
5. **Coverage check**: `(Y_TR.sum(1) > 0).sum() >= 700` after alignment. Catches off-by-one row_id mismatches before training.

---

## Runtime Budget

| Stage | Estimated time |
|---|---|
| onnxruntime install | ~10 s |
| Train Perch (59 files × 12 wins, batched 16 files) | ~3 min |
| PCA fit + feature build | <30 s |
| MLP training (5 folds × 30 ep × 6 batches) | ~3 min |
| Test Perch (variable; staging = 16 files) | ~1 min staging, ~10–60 min real |
| Submission build | <30 s |
| **Total (real submission)** | **~15–80 min** |

Comfortably below Kaggle's 9h CPU notebook limit.

---

## Out-of-Scope for v1 (deferred to nb14+)

These are deliberate omissions, not oversights:
- Bidirectional SSM with cross-attention (nb14)
- TTA via circular-shift averaging (nb15)
- Adaptive delta smoothing (nb15)
- Per-taxon temperature scaling (nb15)
- File-level confidence scaling / rank-aware (nb15)
- Isotonic calibration + per-class threshold search (nb16)
- train_audio inclusion (nb14 — only if it measurably helps OOF)
- Site/hour priors (nb16)

Each becomes its own iteration with a measurable LB delta. The "step-by-step plan to improve and iterate after each notebook" the user described.

---

## Build Order — LB Checkpoints

| Notebook | Adds | Expected LB | Target Δ |
|---|---|---|---|
| nb13 (this) | Perch + MLP probes baseline | 0.85–0.88 | +0.16–0.19 vs 0.689 |
| nb14 | + bidirectional SSM | 0.88–0.91 | +0.02–0.03 |
| nb15 | + TTA + smoothing + temperature | 0.90–0.92 | +0.01–0.02 |
| nb16 | + isotonic calibration + thresholds | 0.92–0.94 | +0.01–0.02 |

Each notebook is a separate kernel push. Each adds **one** measurable thing.
