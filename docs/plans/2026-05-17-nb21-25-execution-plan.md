# nb21-25 Execution Plan -- climb from LB 0.898 toward 0.93+

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Adapted for ML-iteration cadence: each task ends with a Kaggle staging verification + a competition submission step, not a unit-test green. The "verification" is whether the kernel completes cleanly on staging fallback and writes a valid `submission.csv`.

**Goal:** Stack the highest-ROI 2025-winning techniques on top of our current best (LB 0.898) to reach the 2025 winning band (0.93+). Five tasks, each unlocked by the previous: ensemble → noisy student rounds → backbone diversification → inference tweaks.

**Architecture:** Two existing kernels already submit cleanly (`nb19a` Perch+MLP at LB 0.897, `nb20e` 5-fold CNN at LB 0.898). All new kernels must follow the BirdCLEF-2026 working submission contract: `enable_gpu: false`, `enable_internet: false`, models from `dataset_sources` / `model_sources`, never `kernel_sources` for weights. Training kernels still use `enable_gpu: true` on T4; inference kernels use CPU.

**Tech Stack:** PyTorch (training only) + timm + onnxruntime (CPU inference for fast variants) + Perch v2 ONNX + Kaggle CLI + Kaggle Datasets for weight publishing.

**LB targets per task (cumulative):**
| Task | Expected LB | Cumulative gain | Wall time |
|---|---|---|---|
| nb21 ensemble | 0.905–0.915 | +0.007 to +0.017 | ~2 h |
| nb22 NS round 1 | 0.92–0.935 | +0.015 to +0.030 | ~2 days |
| nb23 NS rounds 2-4 | 0.93–0.945 | +0.010 to +0.020 | ~4-6 days |
| nb24 backbone diversification | 0.935–0.95 | +0.005 to +0.015 in ensemble | parallel ~4 days |
| nb25 inference tweaks | 0.94–0.96 | +0.005 to +0.010 | ~1 day |

**Permanent constraints (apply to every task):**
- **enable_gpu: false** for any kernel that will be submitted to the competition (BirdCLEF 2026 scoring is CPU-only, learned the hard way over 2 days)
- **enable_internet: false** for inference / submission kernels
- **enable_gpu: true** is fine for training kernels (they don't go through scoring)
- **Never use kernel_sources for model weights** -- the scoring env may not mount them. Publish weights as a Kaggle Dataset.
- **Push kernels one at a time from `kaggle_notebooks/`** -- never parallel pushes from the same dir (race on `kernel-metadata.json`)
- Test set is large (probably 5k-10k+ files) -- inference must be CPU-fast (target ≤ 0.5s/file)

---

## Task 1 (nb21): Perch+MLP × CNN ensemble

**Files:**
- Create: `kaggle_notebooks/21_ensemble_perch_cnn.py`
- Create: `kaggle_notebooks/21_ensemble_perch_cnn-metadata.json`

**Goal:** A single inference kernel that runs BOTH the Perch+MLP pipeline (nb17b style) and the 5-fold CNN ensemble (nb20e style) on the test set, then blends them per-row. Cheapest +0.005-0.020 LB available.

**Architecture choice:**
- Weighted average: `final = w * cnn + (1 - w) * (perch_mlp + topk)`
- Sweep `w ∈ {0.3, 0.5, 0.7}` in three submission variants if budget allows; otherwise start with `w = 0.5`.

**Step 1: Verify both source kernels still submit successfully**

```bash
kaggle kernels status alexycactus/birdclef-2026-topk-postproc   # nb19a (0.897)
kaggle kernels status alexycactus/birdclef-2026-cnn-infer-dataset   # nb20e (0.898)
```

Expected: both show `COMPLETE`. If not, we lost the working kernels and need to rebuild.

**Step 2: Create the ensemble script**

Copy `19a_topk_postproc.py` as the base (it already does Perch+MLP + top-K). Then add the CNN block from `20e_cnn_infer_dataset.py` AFTER the existing Perch+MLP inference, but BEFORE the submission write. Specifically:

```python
# After perch_sig_te and mlp_pred are computed and `final` (Perch+MLP top-K) is built:
# === CNN block (5-fold ensemble) ===
import glob as _g
import timm
from pathlib import Path

_hits = sorted(_g.glob("/kaggle/input/**/fold*_best.pth", recursive=True))
assert _hits, "CNN fold checkpoints not mounted as dataset_source"
fold_ckpts = {int(Path(p).name.replace("fold","").replace("_best.pth","")): p for p in _hits}

# Mel transform (fp32, autocast disabled), SED head class, BirdCNN class --
# copy verbatim from 20e_cnn_infer_dataset.py lines ~60-160. Use BACKBONE =
# "tf_efficientnet_b0_ns" (plain bundled name, no HF Hub tag).

cnn_models = []
for fi in sorted(fold_ckpts.keys()):
    m = BirdCNN().to(DEVICE)
    m.load_state_dict(torch.load(fold_ckpts[fi], map_location=DEVICE))
    m.eval()
    cnn_models.append(m)

# Run CNN inference on the SAME test_files set used by Perch+MLP. Output:
# cnn_preds_per_row: dict[row_id -> (N_CLASSES,) ndarray].

# Blend
W_CNN = 0.5
final_blended = np.zeros_like(final)
for ri, rid in enumerate(meta_te["row_id"].values):
    cnn_row = cnn_preds_per_row.get(rid)
    if cnn_row is not None:
        final_blended[ri] = W_CNN * cnn_row + (1.0 - W_CNN) * final[ri]
    else:
        final_blended[ri] = final[ri]

# Then apply the SAME top-K postproc to final_blended (it's already in 19a's
# code right above the submission write; just point it at final_blended).
```

**Step 3: Create kernel metadata**

```json
{
  "id": "alexycactus/birdclef-2026-ensemble-perch-cnn",
  "title": "BirdCLEF 2026 Ensemble Perch CNN",
  "code_file": "21_ensemble_perch_cnn.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": false,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": [
    "rishikeshjani/perch-onnx-for-birdclef-2026",
    "alexycactus/birdclef-2026-cnn-fold-checkpoints"
  ],
  "competition_sources": ["birdclef-2026"],
  "kernel_sources": [],
  "model_sources": ["google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"]
}
```

Note `enable_internet: true` -- nb19a needs to pip-install onnxruntime. Verified working in nb19a.

**Step 4: Push**

```bash
cd "c:/Users/alexy/Documents/Claude_projects/Kaggle competition/bird_clef/kaggle_notebooks"
cp 21_ensemble_perch_cnn-metadata.json kernel-metadata.json
kaggle kernels push -p .
```

Expected output: `Kernel version 1 successfully pushed.`

**Step 5: Monitor staging completion (~10-20 min on CPU)**

```bash
kaggle kernels status alexycactus/birdclef-2026-ensemble-perch-cnn
```

When status is `COMPLETE`, pull diagnostics:

```bash
kaggle kernels output alexycactus/birdclef-2026-ensemble-perch-cnn -p ./out --force --file-pattern "(submission\.csv|\.log$)"
```

Verify staging:
- `submission.csv` has shape (192, 235), 0 NaN, values in [0, 1].
- Log shows both Perch+MLP and CNN inference ran without errors.

**Step 6: Submit to competition (asks user)**

This is a manual step -- ask the user to submit `birdclef-2026-ensemble-perch-cnn` via the Kaggle UI or:

```bash
kaggle competitions submit -c birdclef-2026 \
  -k alexycactus/birdclef-2026-ensemble-perch-cnn -v 1 \
  -m "nb21 ensemble Perch+MLP x CNN @ w=0.5, top-K postproc"
```

**Step 7: Record LB + decide on weight sweep**

Update `results/experiment_log.md` with the LB. If LB > 0.905, push two more variants with `W_CNN = 0.3` and `W_CNN = 0.7` to find the optimum (each ~10 min push, +1 submission slot used per variant). If LB ≤ 0.898 (no improvement) skip the sweep and go directly to Task 2.

**Step 8: Commit + push to git**

```bash
cd "c:/Users/alexy/Documents/Claude_projects"
git add "Kaggle competition/bird_clef/kaggle_notebooks/21_*" \
        "Kaggle competition/bird_clef/results/experiment_log.md"
git commit -m "feat(birdclef): nb21 Perch+MLP x CNN ensemble (LB X.XXX)"
git -c http.sslVerify=false push origin master
```

---

## Task 2 (nb22): Noisy Student round 1 on the CNN

**Files:**
- Create: `kaggle_notebooks/22_cnn_pl_saver.py`        (predicts on unlabeled_soundscapes)
- Create: `kaggle_notebooks/22_cnn_pl_saver-metadata.json`
- Create: `kaggle_notebooks/22_cnn_pl_train_run1.py`   (NS-trained folds 0+1)
- Create: `kaggle_notebooks/22_cnn_pl_train_run2.py`   (NS-trained folds 2+3+4)
- Two matching `-metadata.json` files

**Goal:** Apply the 1st-place-2025 "Multi-Iterative Noisy Student" recipe (round 1) on top of nb20a's 5-fold CNN. Predicted to be the biggest single LB lever (1st place 2025: +0.058 across 4 rounds; expect +0.020 to +0.040 from round 1 alone).

**Recipe (verbatim from 1st place 2025 writeup, see `Competitions/BirdCLEF+ 2025/1st-Place-Solution-Multi-Iterative.md`):**
1. Predict 60-second windows of all files in `unlabeled_soundscapes/` (or the unlabeled portion of `train_soundscapes/`) using the current 5-fold CNN.
2. Store framewise predictions per 5-second segment.
3. Filter: keep only soundscapes where `sum_of_max_per_class > threshold` (1st place: weights = `sum(top1_per_class_per_file)`; use `WeightedRandomSampler` later).
4. Power transform: `prob_softened = prob ** (1.0 / 0.65)` (1st place iteration-2 value).
5. Retrain CNN from a NEW init (NOT continuing from previous weights):
   - Each batch: blend a labeled audio sample with a pseudo-labeled audio sample via raw-audio MixUp at constant weight 0.5 (Beta=inf, 1st place).
   - Use `WeightedRandomSampler` on the pseudo-label pool with weights = file sum-of-max-prob.
   - Add `drop_path_rate=0.15` to the EfficientNet backbone (stochastic depth, 1st place's noise injector).
   - 25-30 epochs per fold (longer than supervised round 1 which used 20).
   - Otherwise identical optimizer/scheduler/loss to nb20a.

### Task 2.1: Build the PL saver kernel (predicts on unlabeled_soundscapes)

**Step 1: Create the saver script**

Copy `20e_cnn_infer_dataset.py` as the base. Change the test path discovery:

```python
# REPLACE the `test_files` block (~lines 200) with:
UL_DIR = BASE_DIR / "unlabeled_soundscapes"
if not UL_DIR.exists():
    UL_DIR = BASE_DIR / "train_soundscapes"   # fallback: PL on labeled-set-excluded train SCs

# Filter out the 66 already-labeled files
sc_labels = pd.read_csv(BASE_DIR / "train_soundscapes_labels.csv")
labeled_fnames = set(sc_labels["filename"].unique())
target_files = sorted([f for f in UL_DIR.glob("*.ogg") if f.name not in labeled_fnames])
print(f"Pseudo-labelling {len(target_files)} unlabeled soundscapes ...")
```

Then run the same 5-fold CNN inference loop -- but instead of writing `submission.csv`, save:
- `pl_preds.npy` -- shape `(N_files, 12, N_CLASSES)` float32: 12 framewise predictions per file
- `pl_file_index.csv` -- columns `[filename, file_idx, sum_of_max]` for sampler weights
- Apply post-processing: trim probs < 0.1 to zero IN THE FILE before saving (per 2nd place 2025).

```python
# After collecting per-file 12x234 preds:
preds_array = np.stack([file_preds[stem] for stem in stems], axis=0)  # (N, 12, 234)
preds_array[preds_array < 0.1] = 0.0     # 2nd-place trim
np.save(OUT_DIR / "pl_preds.npy", preds_array)

sum_of_max = preds_array.max(axis=(1, 2))  # (N,)  -- per-file confidence
pd.DataFrame({"filename": stems, "file_idx": range(len(stems)),
              "sum_of_max": sum_of_max}).to_csv(OUT_DIR / "pl_file_index.csv", index=False)
```

**Step 2: Metadata for the saver**

```json
{
  "id": "alexycactus/birdclef-2026-cnn-pl-saver",
  "title": "BirdCLEF 2026 CNN PL Saver",
  "code_file": "22_cnn_pl_saver.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": false,
  "enable_gpu": true,
  "enable_internet": false,
  "dataset_sources": ["alexycactus/birdclef-2026-cnn-fold-checkpoints"],
  "competition_sources": ["birdclef-2026"],
  "kernel_sources": [],
  "model_sources": []
}
```

GPU=true here because this is a training-side kernel (not submitted), and we want fast inference on ~10k files. Expected runtime: 1-2 hours.

**Step 3: Push and wait**

```bash
cd kaggle_notebooks
cp 22_cnn_pl_saver-metadata.json kernel-metadata.json
kaggle kernels push -p . --accelerator NvidiaTeslaT4
```

Monitor at +20 min, +1 h, +2 h. When done:

```bash
kaggle kernels output alexycactus/birdclef-2026-cnn-pl-saver -p ./out --force
ls out/   # expect pl_preds.npy and pl_file_index.csv
```

**Step 4: Verify PL quality**

Quick local sanity check on `pl_preds.npy`:

```python
import numpy as np
preds = np.load('out/pl_preds.npy')
print(preds.shape, preds.dtype, preds.min(), preds.mean(), preds.max())
# Sanity: shape ~(10000, 12, 234), float32, min=0 (after trim), mean ~ 0.005, max <= 1
print((preds > 0.5).any(axis=(1,2)).sum(), "files have ≥1 strong species detection")
# Expect a few thousand
```

If predictions look reasonable (some files with strong detections, not all-zero or all-1), proceed. If they look broken (all-zero, NaN, etc), debug before training.

**Step 5: Commit + push**

```bash
git add "Kaggle competition/bird_clef/kaggle_notebooks/22_cnn_pl_saver*"
git commit -m "feat(birdclef): nb22 CNN PL saver -- predicts on unlabeled soundscapes"
git -c http.sslVerify=false push origin master
```

### Task 2.2: NS training run 1 (folds 0+1)

**Step 1: Create the training script**

Copy `20a_run1_folds01.py` (the canonical split-run-1 training kernel). Add a PL dataset section:

```python
# Mount the saver output
import glob as _g
_pl_hits = _g.glob("/kaggle/input/**/pl_preds.npy", recursive=True)
PL_DIR = Path(_pl_hits[0]).parent
pl_preds = np.load(PL_DIR / "pl_preds.npy")            # (N_pl, 12, 234)
pl_meta  = pd.read_csv(PL_DIR / "pl_file_index.csv")
print(f"Loaded {len(pl_meta)} pseudo-labelled files, "
      f"{(pl_preds > 0).any(axis=(1,2)).sum()} with ≥1 detection")

# Power transform
power_k = 1.0 / 0.65
pl_preds_soft = np.power(pl_preds.clip(0.0, 1.0), power_k).astype(np.float32)

# Build a parallel Dataset for PL files
class PLDataset(torch.utils.data.Dataset):
    """Random 20s crop from an unlabeled soundscape with soft labels from its window."""
    def __init__(self, meta_df, pl_preds_array, base_dir):
        self.meta = meta_df.reset_index(drop=True)
        self.pl = pl_preds_array
        self.base = base_dir
        # Sampler weights
        self.weights = self.meta["sum_of_max"].values.astype(np.float32)
        self.weights = self.weights / max(self.weights.sum(), 1e-6)
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        path = self.base / "unlabeled_soundscapes" / row["filename"]
        if not path.exists(): path = self.base / "train_soundscapes" / row["filename"]
        wav = load_audio(str(path), target_samples=60 * SR)
        # Pick a random 20s window inside the 60s
        slot = random.randint(0, 8)            # 0..8 windows of 20s with 5s stride
        start = slot * INFER_STRIDE * SR
        chunk = absmax_normalize(wav[start : start + WINDOW_SAMPLES])
        chunk = np.array(chunk, dtype=np.float32)
        # Map to the 12x234 framewise -> reduce the 20s window to one label
        # The window covers slots [slot..slot+3]; take the max over those slot framewise preds
        lo_slot = slot
        hi_slot = min(slot + 4, 12)
        soft = self.pl[int(row["file_idx"]), lo_slot:hi_slot].max(axis=0)  # (234,)
        return chunk, soft.astype(np.float32)

# In the train loop, sample EVERY labeled batch alongside a pl batch:
pl_ds      = PLDataset(pl_meta, pl_preds_soft, BASE_DIR)
pl_sampler = torch.utils.data.WeightedRandomSampler(
    pl_ds.weights, num_samples=len(pl_ds), replacement=True
)
pl_loader  = torch.utils.data.DataLoader(
    pl_ds, batch_size=BATCH_SZ, sampler=pl_sampler,
    num_workers=N_WORKERS, pin_memory=True, persistent_workers=True,
)
pl_iter    = iter(pl_loader)
```

Then in the per-batch loop, blend labeled with PL via raw-audio MixUp (always, not p=0.5 -- 1st place reports ratio=1.0 is optimal):

```python
for wavs_t, tgts_t in tr_loader:
    # Get a matching PL batch (cycle iter if exhausted)
    try:
        pl_wavs, pl_soft = next(pl_iter)
    except StopIteration:
        pl_iter = iter(pl_loader)
        pl_wavs, pl_soft = next(pl_iter)

    wavs_t  = wavs_t.to(DEVICE, non_blocking=True)
    tgts_t  = tgts_t.to(DEVICE, non_blocking=True)
    pl_wavs = pl_wavs.to(DEVICE, non_blocking=True)
    pl_soft = pl_soft.to(DEVICE, non_blocking=True)

    # Constant 0.5 blend (1st place: Beta=inf gives best)
    mixed_wav = 0.5 * wavs_t + 0.5 * pl_wavs
    mixed_tgt = torch.maximum(tgts_t, pl_soft)   # label union with soft probs

    ... rest of the forward+backward as in nb20a ...
```

Also add `drop_path_rate=0.15` to the timm backbone construction:

```python
self.backbone = timm.create_model(
    backbone_name, pretrained=True, in_chans=in_chans,
    num_classes=0, global_pool="", features_only=False,
    drop_path_rate=0.15,        # 1st place 2025 NS noise injector
)
```

Set `EPOCHS = 25` (1st place used 25-35 for NS rounds).

**Step 2: Metadata**

```json
{
  "id": "alexycactus/birdclef-2026-cnn-ns1-run1",
  "title": "BirdCLEF 2026 CNN NS1 Run1",
  "code_file": "22_cnn_pl_train_run1.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": false,
  "enable_gpu": true,
  "enable_internet": false,
  "dataset_sources": ["alexycactus/birdclef-2026-cnn-fold-checkpoints"],
  "competition_sources": ["birdclef-2026"],
  "kernel_sources": ["alexycactus/birdclef-2026-cnn-pl-saver"],
  "model_sources": []
}
```

GPU=true (training kernel, not submitted). Estimated runtime: 6-7 h (folds 0+1, 25 epochs each).

**Step 3: Push, monitor, pull weights**

Same pattern as nb20a-run1:

```bash
cp 22_cnn_pl_train_run1-metadata.json kernel-metadata.json
kaggle kernels push -p . --accelerator NvidiaTeslaT4
# Wait ~6-7 h, monitor at +30 min, +3 h, +6 h
kaggle kernels output alexycactus/birdclef-2026-cnn-ns1-run1 -p ./out_ns1_r1 --force --file-pattern "fold[0-9]_best\.pth"
```

### Task 2.3: NS training run 2 (folds 2+3+4)

Same as Task 2.2 but with `FOLD_INDICES = [2, 3, 4]`, `LOAD_PRIOR_FOLDS = True`, `DO_FINAL_INFERENCE = False`. kernel_sources for run 1's output. Adds 3 more folds.

**Step 1-3: Mirror Task 2.2** with these constants in `22_cnn_pl_train_run2.py`:

```python
FOLD_INDICES         = [2, 3, 4]
LOAD_PRIOR_FOLDS     = True       # mount NS1 run1 via kernel_sources
DO_FINAL_INFERENCE   = False
EPOCHS               = 25
```

Metadata adds the NS1 run1 kernel as a kernel_source (training kernels can use kernel_sources -- only submission scoring rejects them).

### Task 2.4: Publish new NS-trained checkpoints as a dataset, then ensemble + submit

**Step 1: Pull all 5 NS-trained checkpoints locally**

```bash
mkdir results/nb22_ns1_checkpoints
cd results/nb22_ns1_checkpoints
kaggle kernels output alexycactus/birdclef-2026-cnn-ns1-run1 -p . --force --file-pattern "fold[0-9]_best\.pth"
kaggle kernels output alexycactus/birdclef-2026-cnn-ns1-run2 -p . --force --file-pattern "fold[0-9]_best\.pth"
```

Verify clean (no NaN) using the same script as nb20a verification.

**Step 2: Create dataset metadata + upload**

```json
{
  "title": "BirdCLEF 2026 CNN NS1 5-fold Checkpoints",
  "id": "alexycactus/birdclef-2026-cnn-ns1-checkpoints",
  "licenses": [{"name": "CC0-1.0"}]
}
```

Use the SSL-disabled Python wrapper (`upload_via_python.py`) -- the regular `kaggle datasets create` SSL-fails on Google Cloud Storage on this machine.

**Step 3: Copy + reconfigure nb21 ensemble kernel as nb22e**

Create `22e_ensemble_perch_ns1cnn.py` -- byte-identical to nb21 EXCEPT the `dataset_sources` swap: replace the `birdclef-2026-cnn-fold-checkpoints` dataset with the new `birdclef-2026-cnn-ns1-checkpoints`.

Push, wait for staging, submit. Expected LB: 0.92-0.935.

**Step 4: Update logs + commit**

Update `results/experiment_log.md`, `MEMORY.md`, and `docs/plans/2026-05-14-post-nb20-experiments-backlog.md` with the round 1 LB. Commit.

---

## Task 3 (nb23): Noisy Student rounds 2-4

**Files:**
- For each round R in {2, 3, 4}: clone nb22 saver + run1 + run2 + ensemble kernels, swapping in the previous round's CNN checkpoints as the PL teacher.

**Goal:** Each iteration adds +0.005 to +0.015 LB until convergence (1st place 2025: stopped at iteration 4 when round 5 stopped improving).

**Key recipe variations across iterations** (1st place 2025 table):
| Iteration | Power transform value |
|---|---|
| 2 | 1 / 0.65 |
| 3 | 1 / 0.55 |
| 4 | 1 / 0.60 |

That's the only knob that changes between rounds; everything else (epochs, MixUp ratio, drop_path_rate, optimizer) stays identical.

**Step-by-step is a triple-loop over rounds, each repeating Task 2 with:**
1. Mount the previous round's checkpoint dataset
2. Update the power transform constant
3. Train new fold checkpoints
4. Publish new dataset
5. Build new ensemble + submit
6. Record LB and decide whether to do another round

**STOP condition**: if a round's LB is within ±0.003 of the previous round (no improvement), stop. 1st place stopped after round 4.

---

## Task 4 (nb24): Backbone diversification (PARALLEL with Task 3)

**Files:**
- `kaggle_notebooks/24_ns1_effnetv2s_run1.py` + run2
- `kaggle_notebooks/24_ns1_nfnet_l0_run1.py` + run2
- Matching metadata files

**Goal:** Replace `tf_efficientnet_b0_ns` with two heavier, more diverse backbones for ensemble diversity. 2025 winners used EfficientNetV2-S and ECA-NFNet-L0 alongside B0.

**Architecture:** Identical to nb22 NS round 1 training, EXCEPT `BACKBONE` constant:
- `BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"` -- but use plain `"tf_efficientnetv2_s"` to avoid the HF tag issue (submission won't load these anyway since they're trained, not run on submission directly)
- `BACKBONE = "eca_nfnet_l0"` -- the 2025 1st place's NS-iteration-3 addition

Note: heavier backbones may need smaller batch (`BATCH_SZ = 24`) to fit T4 memory.

**Workflow per backbone:**
1. Train via NS-style pipeline (use the NS dataset that's strongest at that moment)
2. Publish checkpoints as a Kaggle Dataset
3. Add to the ensemble kernel as another dataset source
4. Update the ensemble to weight-average all backbones' predictions
5. Submit and record LB

**Strategic note:** start nb24 backbones AFTER nb22 NS round 1 lands a useful LB -- training a heavier backbone from scratch on poor PL data is wasteful. Use the BEST current NS round's PL data as the teacher.

---

## Task 5 (nb25): Inference-side tweaks

**Files:**
- `kaggle_notebooks/25_infer_tta.py` (one inference kernel that tries the suite)
- Metadata file

**Goal:** A handful of small but stacking inference-only tweaks. Each individually +0.002 to +0.005, but they compose. From 1st & 2nd place 2025:

| Tweak | Source | Expected gain | Risk |
|---|---|---|---|
| Top-K = 2 instead of K=1 | nb19a / 2nd place | +0.002 | none |
| Framewise smoothing kernel `[0.1, 0.2, 0.4, 0.2, 0.1]` | 1st place | +0.002 | none |
| Delta-shift TTA | 1st place (from 2023 2nd) | +0.002 | runs slower |
| Higher inference stride (1s instead of 5s in framewise overlap) | 1st place | +0.005 | 5x slower, may break budget |
| Per-class temperature scaling on OOF | nb19 (idea) | +0.005 | calibration risk |

**Step-by-step:** push each tweak as a separate version of the ensemble kernel, sweep through them, keep what helps cumulatively. Run with `enable_gpu: false` -- all are CPU-friendly.

**STOP condition for the whole plan:**
- If cumulative LB exceeds 0.93 → maintain submissions in case of LB shifts; pivot remaining time to robustness (multi-seed ensembling)
- If LB stalls below 0.92 → debug the PL pipeline (1st place 2025 had similar plateaus; their fix was the power transform tuning)
- Competition deadline: 2026-06-03

---

## Operational reminders (apply across all tasks)

- Submit kernels through Kaggle UI or `kaggle competitions submit -c birdclef-2026 -k <slug> -v <version> -m "<message>"`
- Update `results/experiment_log.md` after EVERY LB result, even regressions
- Push to both `origin` and `birdclef` remotes after each task
- Watch the Kaggle submission quota (typically 5/day) -- don't burn quotas on speculative submissions
- If any kernel fails on submission: pull its log via `kaggle kernels output <slug> --file-pattern "\.log$"` AND check the Kaggle UI submission preview for additional context
- Memory notes on submission gotchas (`enable_gpu: false`, no `kernel_sources` for weights, dataset upload SSL fix) ARE the lessons from this campaign. Apply them.
