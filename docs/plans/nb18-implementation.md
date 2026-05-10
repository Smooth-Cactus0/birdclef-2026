# nb18 Implementation Brief — CNN Backbone Sweep

*Created: 2026-05-10. Hand this file to a new Claude Code session to implement nb18.*

---

## Mission

Establish a **parallel CNN track** to the Perch+MLP path. Trains 3 backbones from scratch
on mel spectrograms — first attempt at a from-scratch CNN baseline for BirdCLEF 2026.

**Hypothesis:** any CNN backbone hitting LB ≥ 0.85 standalone confirms the CNN track is
viable, and ensembling with nb15a (LB 0.883) in nb19 should push past 0.90.

**Submissions are standalone** — each kernel produces its own `submission.csv` with no
nb15a blending. Ensemble logic comes later (nb19).

---

## Variant Matrix — 3 Kernels

| Kernel | Backbone | timm name | Params | Role |
|---|---|---|---|---|
| nb18a | EfficientNet-B0 | `tf_efficientnet_b0_ns` | 5.3M | Lightweight workhorse |
| nb18b | EfficientNet-B3 | `tf_efficientnet_b3_ns` | 12.2M | Mid-range, better accuracy |
| nb18c | RegNetY-008 | `regnety_008` | 6.3M | Architectural diversity for ensemble |

All 3 share: same training data, same spectrogram params, same augmentation, same schedule,
same loss, same submission format. Only the backbone differs.

---

## Architecture

### Audio → Spectrogram (matches CLAUDE.md conventions)

```python
SR              = 32_000
N_FFT           = 1024
HOP_LENGTH      = 320
N_MELS          = 128
FMIN            = 40
FMAX            = 15_000
TRAIN_DURATION  = 10        # seconds — 2024 insight: longer than inference helps
INFER_DURATION  = 5         # seconds — competition chunk size
```

Spectrogram shape: `(1, 128, 1001)` for 10s training, `(1, 128, 501)` for 5s inference.

### Model

```python
import timm
import torch.nn as nn

class BirdCNN(nn.Module):
    def __init__(self, backbone_name="tf_efficientnet_b0_ns", n_classes=234, in_chans=1):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, in_chans=in_chans,
            num_classes=0, global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        # x: (B, 1, 128, T)
        feats = self.backbone(x)            # (B, feat_dim)
        return self.head(feats)             # (B, 234) logits
```

### Training Config

```python
N_FOLDS    = 5
EPOCHS     = 20         # 3 warmup + 15 main + 2 fine-tune
LR_MAX     = 3e-4
LR_MIN     = 1e-5
WD         = 1e-2
BATCH_SZ   = 32
OPTIMIZER  = "AdamW"
SCHEDULER  = "CosineAnnealingLR"
LOSS       = "BCEWithLogitsLoss"   # multi-label, 234 classes
AMP        = True                  # fp16 mixed precision
```

**Time budget per kernel** (T4 GPU):
- B0: ~30s/epoch × 20 epochs × 5 folds ≈ 50 min training
- B3: ~50s/epoch × 20 × 5 ≈ 85 min
- RegNetY-008: ~30s/epoch × 20 × 5 ≈ 50 min
- Plus inference (~10 min) + saving artifacts

Each kernel ~1.5–2.5h. All fit within 9h Kaggle limit.

---

## Training Data

### Sources

| Source | Count | Weight | Notes |
|---|---|---|---|
| `train_audio/` clips | 35,549 | 1.0 | Xeno-canto + iNat, 162 bird species + non-bird mapped |
| Labeled soundscape segments | 1,478 (66 files) | 3.0× upsample | From `train_soundscapes_labels.csv` — Pantanal-domain |

### Pseudo-code data preparation

```python
# 1. Load metadata
train_meta = pd.read_csv(BASE_DIR / "train.csv")             # 35,549 rows
sc_labels  = pd.read_csv(BASE_DIR / "train_soundscapes_labels.csv")
taxonomy   = pd.read_csv(BASE_DIR / "taxonomy.csv")
sample_sub = pd.read_csv(BASE_DIR / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()             # 234 species
N_CLASSES      = 234

# 2. Build per-clip target vectors
def make_target(row):
    y = np.zeros(N_CLASSES, dtype=np.float32)
    primary = row.get("primary_label")
    if primary in label_to_idx: y[label_to_idx[primary]] = 1.0
    secondary = row.get("secondary_labels", "")
    if pd.notna(secondary):
        for s in str(secondary).split(";"):
            s = s.strip()
            if s in label_to_idx: y[label_to_idx[s]] = 1.0
    return y

# 3. Build training index — train_audio + 3× soundscape segments
train_clips = [(path, target, weight=1.0) for path, target in train_audio_iter()]
sc_segments = [(path, target, weight=1.0) for path, target in sc_segment_iter()]
training_set = train_clips + sc_segments * 3   # 3× upsample
```

### Validation strategy

- **5-fold StratifiedKFold by primary_label across all training clips** (mixed train_audio + soundscape segments)
- Track validation macro-AUC per epoch on the 234 species
- Best epoch by validation AUC = checkpoint to ensemble at inference

This is simpler than the original strategy doc's GroupKFold-by-site approach — train_audio
has no site info. We accept the modest leakage risk for a first baseline; refine validation
in nb19+ if needed.

---

## Augmentation Stack

Applied at training time only:

| Aug | Probability | Notes |
|---|---|---|
| Random 10s crop from clip | 1.0 | If clip < 10s, repeat-pad |
| Background mix from `unlabeled_soundscapes/` | 0.5 | α ∈ [0.1, 0.3]. Highest-ROI aug per 2025 winners. |
| SpecAugment time mask | 0.5 | Up to 30 frames |
| SpecAugment freq mask | 0.5 | Up to 16 mel bins |
| Gaussian noise | 0.2 | σ ∈ [0.001, 0.01] in waveform domain |

Apply waveform augs (background mix, gaussian noise) before mel spectrogram computation.
SpecAugment applies on the mel output.

**Background mix snippet:**
```python
def background_mix(wav, bg_pool, p=0.5, alpha_range=(0.1, 0.3)):
    if random.random() > p: return wav
    bg = random.choice(bg_pool)
    a  = random.uniform(*alpha_range)
    return (1 - a) * wav + a * bg
```

---

## Inference

For each test soundscape (60s):
1. Load audio at SR=32000
2. Slide 5s windows with 5s stride → 12 windows per file
3. Compute mel spectrogram per window → `(1, 128, 501)` per window
4. Batch through model → 234-dim logits
5. Sigmoid → probabilities
6. 5-fold ensemble: average probabilities across folds
7. Format as `row_id, [234 cols]` per `sample_submission.csv`

```python
def infer_soundscape(path, models):
    wav = load_audio(path, target_sec=60, sr=SR)
    windows = wav.reshape(12, 5 * SR)            # (12, 5s @ 32k)
    mels    = log_mel(windows)                   # (12, 128, 501)
    x       = torch.from_numpy(mels).unsqueeze(1).cuda()
    preds   = []
    for m in models:
        with torch.cuda.amp.autocast():
            preds.append(torch.sigmoid(m(x)).cpu().numpy())
    return np.mean(preds, axis=0)                # (12, 234)
```

---

## Submission Format

```python
rows = []
for path in test_files:
    probs = infer_soundscape(path, fold_models)   # (12, 234)
    for w in range(12):
        end_sec = (w + 1) * 5
        rows.append({"row_id": f"{path.stem}_{end_sec}",
                     **dict(zip(PRIMARY_LABELS, probs[w]))})

pred_df = pd.DataFrame(rows)
sub     = sample_sub[["row_id"]].merge(pred_df, on="row_id", how="left")
sub[PRIMARY_LABELS] = sub[PRIMARY_LABELS].fillna(1.0 / N_CLASSES)
sub.to_csv("/kaggle/working/submission.csv", index=False)
```

---

## Kernel Metadata

For nb18a (`18a_cnn_efficientnet_b0-metadata.json`):

```json
{
  "id": "alexycactus/birdclef-2026-cnn-efficientnet-b0",
  "title": "BirdCLEF 2026 CNN EfficientNet B0",
  "code_file": "18a_cnn_efficientnet_b0.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": false,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": ["birdclef-2026"],
  "kernel_sources": [],
  "model_sources": []
}
```

**Push command:** `kaggle kernels push -p . --accelerator NvidiaTeslaT4`

(P100 fails on PyTorch 2.4+ per `cudaErrorNoKernelImageForDevice` — always use T4.)

Same pattern for nb18b/c with backbone-specific id/title/code_file.

---

## Implementation Steps

1. **Create skeleton** — write `18a_cnn_efficientnet_b0.py` end-to-end. Test the data loading
   block locally (read 10 clips, build mel spectrograms, verify shapes).

2. **Single-fold dry run on Kaggle** — push with `EPOCHS=2` and `N_FOLDS=1` first. Verify the
   pipeline works end-to-end, produces a valid `submission.csv`. ~15 min.

3. **Full run** — bump to `EPOCHS=20`, `N_FOLDS=5`, push. ~1.5h.

4. **Generate nb18b/c** — copy nb18a, change only `BACKBONE_NAME` and the diagnostics filename
   suffix. Push sequentially.

5. **Submit all 3** — manually via Kaggle UI or CLI. Track LB.

---

## Diagnostics to Output

Each kernel saves to `/kaggle/working/`:
- `submission.csv` — competition format
- `diagnostics_nb18x.json` — backbone, fold AUCs, training time, OOF macro-AUC
- `per_class_auc_nb18x.csv` — per-species AUC table
- Best fold checkpoints (`fold_k_best.pth`) if size allows — useful for nb19 ensemble

```json
{
  "notebook": "nb18a",
  "backbone": "tf_efficientnet_b0_ns",
  "n_train_clips": 39_983,
  "n_folds": 5,
  "epochs": 20,
  "fold_aucs": [0.XX, 0.XX, 0.XX, 0.XX, 0.XX],
  "oof_auc_macro": 0.XX,
  "training_time_min": 50,
  "nb15a_reference_lb": 0.883
}
```

---

## Kaggle Operational Gotchas

| Issue | Fix |
|---|---|
| GPU compatibility | T4 (sm_75) only; P100 (sm_60) fails on PyTorch 2.4+ |
| Competition data path | Try `/kaggle/input/competitions/birdclef-2026/` first, fallback `/kaggle/input/birdclef-2026/` |
| `test_soundscapes/` empty | Only populated during "Submit to Competition"; add staging fallback (first 16 train soundscapes) |
| Mojibake in code | Use plain ASCII in string literals — em-dashes / smart quotes break Python parser |
| Slug conflict (409) | Title must resolve to spec'd id. Keep titles slug-clean. |
| Parallel push race | NEVER push multiple kernels simultaneously from the same dir |
| `docker_image` in metadata | NEVER set it — breaks competition data mounting |

---

## Git Push Workflow

```powershell
git add "Kaggle competition/bird_clef/kaggle_notebooks/18*"
git add "Kaggle competition/bird_clef/docs/plans/nb18-implementation.md"
git commit -m "feat(birdclef): nb18 CNN baseline — backbone sweep B0/B3/RegNetY-008

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git -c http.sslVerify=false push origin master
git -c http.sslVerify=false subtree push --prefix="Kaggle competition/bird_clef" birdclef master
```

---

## Files to Create

```
kaggle_notebooks/
├── 18a_cnn_efficientnet_b0.py
├── 18a_cnn_efficientnet_b0-metadata.json
├── 18b_cnn_efficientnet_b3.py
├── 18b_cnn_efficientnet_b3-metadata.json
├── 18c_cnn_regnety_008.py
└── 18c_cnn_regnety_008-metadata.json
```

---

## Success Criteria

- All 3 kernels complete training without errors
- All 3 produce valid `submission.csv` (234 columns, correct row_ids)
- LB collected for all 3
- Decision tree:
  - **All 3 LB ≥ 0.85** → CNN track is viable. Plan nb19 ensemble (Perch+MLP × CNN).
  - **Best CNN LB 0.78–0.85** → marginally useful. Test ensemble in nb19, but don't invest more in CNN training.
  - **All 3 LB < 0.78** → likely a pipeline bug (data issue, augmentation issue, validation leak). Debug before scaling.
  - **One backbone clearly best** → use that one as the default for nb19 ensemble experiments.

Update `results/experiment_log.md` with all 3 LB scores plus the OOF AUCs.
