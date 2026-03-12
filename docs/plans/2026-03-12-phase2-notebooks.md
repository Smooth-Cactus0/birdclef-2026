# BirdCLEF 2026 — Phase 2 Notebooks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the 6 Phase 2 competition notebooks (backbone search, BirdNET fine-tune, non-bird pipeline, pseudo-labeling, ViT PL generator, ensemble) that form the full winning pipeline.

**Architecture:** Two-pipeline approach (bird 162 classes + non-bird 72 classes), merged at submission. ViTs generate pseudo-labels on unlabeled soundscapes; CNNs handle inference within the CPU budget. All notebooks are Kaggle `.py` script-mode files with `# %%` cell markers.

**Tech Stack:** PyTorch, timm, librosa, openvino, onnxruntime, pandas, numpy, matplotlib, sklearn

---

## Codebase Context (read this before any task)

**Project root:** `c:/Users/alexy/Documents/Claude_projects/Kaggle competition/bird_clef/`

**Existing notebooks to read as reference patterns** (read at least nb03 before starting any task):
- `kaggle_notebooks/03_baseline_efficientnet.py` — canonical pattern for CFG dict, BirdDataset, BirdModel, train loop, OOF, ONNX export
- `kaggle_notebooks/05_inference_openvino.py` — ONNX/OpenVINO export patterns

**Key design decisions already locked:**
- Bird pipeline: 162 classes (Aves only), BCE multi-label loss (NOT CrossEntropy)
- Non-bird pipeline: 72 classes (Amphibia 35 + Insecta 28 + Mammalia 8 + Reptilia 1), Focal BCE
- ViTs: used for pseudo-label generation only, NOT for competition inference
- Validation: GroupKFold by `recorder_id` site (NOT StratifiedKFold)
- Sample weights: Gold=1.0, XC≥3.0=0.8, XC<3.0=0.5, iNat=0.4, PL-R1=0.5, PL-R2=0.65, PL-R3-4=0.75

**Audio constants (use exactly these in every notebook):**
```python
SR          = 32000
N_FFT       = 1024
HOP_LENGTH  = 320
N_MELS      = 128
FMIN        = 40
FMAX        = 15000
DURATION    = 5       # inference clip length
TRAIN_DURATION = 10   # training clip (longer context = better, 2024 insight)
# n_frames (5s)  = 1 + (SR * DURATION // HOP_LENGTH)      = 501
# n_frames (10s) = 1 + (SR * TRAIN_DURATION // HOP_LENGTH) = 1001
```

**Standard path setup (copy into every notebook):**
```python
BASE_DIR   = Path('/kaggle/input/birdclef-2026') if Path('/kaggle/input/birdclef-2026').exists() else Path('birdclef-2026')
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
NUM_WORKERS = 0 if os.name == 'nt' else 4
```

**Standard label setup (copy into every notebook that trains):**
```python
taxonomy   = pd.read_csv(BASE_DIR / 'taxonomy.csv')
label_list = taxonomy['primary_label'].tolist()   # 234 total
label2idx  = {l: i for i, l in enumerate(label_list)}
NUM_CLASSES = len(label_list)  # 234

# Bird-only subset
bird_labels = taxonomy[taxonomy['class_name'] == 'Aves']['primary_label'].tolist()  # 162
bird2idx    = {l: i for i, l in enumerate(bird_labels)}
BIRD_CLASSES = len(bird_labels)  # 162

# Non-bird subset
nonbird_labels = taxonomy[taxonomy['class_name'] != 'Aves']['primary_label'].tolist()  # 72
nonbird2idx    = {l: i for i, l in enumerate(nonbird_labels)}
NONBIRD_CLASSES = len(nonbird_labels)  # 72
```

**Kaggle notebook conventions (every file must follow these):**
- File starts with a big `# %%` comment block header
- Cell markers: `# %%` for code, `# %% [markdown]` for markdown (with `# ## Heading` style)
- First cell: `# !pip install -q timm==1.0.3  # uncomment on Kaggle`
- All paths via BASE_DIR / OUTPUT_DIR
- `warnings.filterwarnings('ignore')`
- Save key outputs (JSON, PNG, CSV, .pth) to OUTPUT_DIR for download

---

## Task 1: nb06 — Backbone Search

**File to create:** `kaggle_notebooks/06_backbone_search.py`

**Purpose:** Train one selected backbone at a time (controlled by `CFG['MODEL_NAME']`), evaluate OOF AUC, save checkpoint and ONNX. Designed to be run 5 times (once per backbone) on Kaggle, results compared in a final summary plot.

**Key upgrades vs nb03:**
- BCE multi-label loss (not CrossEntropy) — test chunks are multi-label, this matters
- GroupKFold by `recorder_id` site (not StratifiedKFold)
- TRAIN_DURATION=10s (longer context)
- Sample weighting by quality
- All 5 target backbones selectable via CFG

### Step 1: Read nb03 for patterns

Read: `kaggle_notebooks/03_baseline_efficientnet.py` (full file)

Note these exact patterns to replicate or upgrade:
- CFG dict structure
- BirdDataset `_load_audio`, `_to_melspec`, `_augment`
- BirdModel with `timm.create_model(..., in_chans=1)`
- `train_one_epoch`, `validate`
- OOF collection loop
- ONNX export at end

### Step 2: Write the notebook

Create `kaggle_notebooks/06_backbone_search.py` with these cells:

**Cell 1 — Header:**
```python
# %%
# =============================================================================
# BirdCLEF 2026 — Backbone Search (Bird Pipeline)
# =============================================================================
# Change CFG['MODEL_NAME'] to swap backbones:
#   'efficientnet_b1'   — lightweight upgrade from B0
#   'efficientnet_b3'   — accuracy/speed sweet spot
#   'efficientnet_b4'   — top accuracy in bird pipeline
#   'regnety_016'       — architectural diversity
#   'eca_nfnet_l0'      — strong regularization
# Loss    : BCEWithLogitsLoss (multi-label — test chunks have multiple species)
# Folds   : GroupKFold by recorder site (honest CV)
# Duration: 10s train / 5s inference (longer context helps per 2024 winners)
# =============================================================================
```

**Cell 2 — Imports:**
```python
import os, gc, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
warnings.filterwarnings('ignore')
```

**Cell 3 — CFG (key differences from nb03):**
```python
CFG = dict(
    SR=32000, N_FFT=1024, HOP_LENGTH=320, N_MELS=128, FMIN=40, FMAX=15000,
    DURATION=5,           # inference
    TRAIN_DURATION=10,    # training — longer context
    MODEL_NAME='efficientnet_b3',   # ← change this per run
    PRETRAINED=True,
    N_FOLDS=5,
    EPOCHS=15,
    BATCH_SIZE=32,
    LR=3e-4,
    WEIGHT_DECAY=1e-2,
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu',
    SEED=42,
    NUM_WORKERS=0 if os.name == 'nt' else 4,
)
```

**Cell 4 — Load data + label setup:**
Use the standard label setup from the codebase context above (full 234 labels from taxonomy).
Build `train_df` from `train.csv`.

**Bird-only filter:**
```python
train_df = pd.read_csv(BASE_DIR / 'train.csv')
# Keep only bird species for this pipeline
train_df = train_df[train_df['primary_label'].isin(bird_labels)].copy()
train_df['target'] = train_df['primary_label'].map(bird2idx)
train_df['filepath'] = train_df['filename'].apply(lambda f: str(BASE_DIR / 'train_audio' / f))

# Sample weights by data quality
def get_sample_weight(row):
    if row.get('source', 'XC') == 'iNat':
        return 0.4
    r = row.get('rating', 0.0)
    if r >= 3.0:
        return 0.8
    return 0.5

train_df['weight'] = train_df.apply(get_sample_weight, axis=1)

# Extract recorder site from filename for GroupKFold
# filename format: XC12345.ogg or species/XC12345.ogg
# recorder_id comes from train_soundscapes if available; fallback: use 'XC'+'iNat' as group
train_df['site'] = train_df.get('recorder_id', train_df['primary_label'])
```

**Cell 5 — BirdDataset (upgraded for 10s training, multi-label targets, weighted sampler):**
```python
class BirdDataset(Dataset):
    """
    Loads OGG, makes log-mel spectrogram.
    - Training: 10s random crop
    - Validation: 5s center crop
    - Target: multi-hot vector (length = BIRD_CLASSES) for BCE
    """
    def __init__(self, df, cfg, augment=False, train_duration=True):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.augment = augment
        self.n_samples = cfg['SR'] * (cfg['TRAIN_DURATION'] if (augment and train_duration) else cfg['DURATION'])

    def __len__(self): return len(self.df)

    def _load_audio(self, path):
        try:
            y, _ = librosa.load(path, sr=self.cfg['SR'], mono=True)
        except Exception:
            return np.zeros(self.n_samples, dtype=np.float32)
        if len(y) < self.n_samples:
            y = np.pad(y, (0, self.n_samples - len(y)))
        elif self.augment and len(y) > self.n_samples:
            start = np.random.randint(0, len(y) - self.n_samples)
            y = y[start:start + self.n_samples]
        else:
            y = y[:self.n_samples]
        return y.astype(np.float32)

    def _to_melspec(self, y):
        S = librosa.feature.melspectrogram(
            y=y, sr=self.cfg['SR'], n_fft=self.cfg['N_FFT'],
            hop_length=self.cfg['HOP_LENGTH'], n_mels=self.cfg['N_MELS'],
            fmin=self.cfg['FMIN'], fmax=self.cfg['FMAX'])
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        return S_db.astype(np.float32)

    def _spec_augment(self, S):
        S = S.copy()
        F = np.random.randint(0, 16); f0 = np.random.randint(0, self.cfg['N_MELS'] - F)
        S[f0:f0+F, :] = 0.0
        T = np.random.randint(0, 30); t0 = np.random.randint(0, S.shape[1] - T)
        S[:, t0:t0+T] = 0.0
        return S

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self._load_audio(row['filepath'])
        S = self._to_melspec(y)
        if self.augment:
            S = self._spec_augment(S)
        # Multi-hot target: primary label = 1.0, secondary labels = 0.5 if present
        target = np.zeros(BIRD_CLASSES, dtype=np.float32)
        if row['target'] < BIRD_CLASSES:
            target[int(row['target'])] = 1.0
        # secondary_labels column if available
        if 'secondary_labels' in row and isinstance(row['secondary_labels'], str):
            for sl in row['secondary_labels'].replace("'", "").strip("[]").split(','):
                sl = sl.strip()
                if sl in bird2idx:
                    target[bird2idx[sl]] = 0.5
        weight = float(row.get('weight', 1.0))
        return torch.from_numpy(S).unsqueeze(0), torch.from_numpy(target), weight
```

**Cell 6 — BirdModel (same as nb03 but output = BIRD_CLASSES):**
```python
class BirdModel(nn.Module):
    """timm backbone, in_chans=1, num_classes=BIRD_CLASSES."""
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          in_chans=1, num_classes=num_classes)
    def forward(self, x): return self.backbone(x)
```

**Cell 7 — train_one_epoch with weighted BCE:**
```python
def train_one_epoch(model, loader, optimizer, scheduler, device):
    """BCE with per-sample weighting from data quality."""
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    for X, y, w in loader:
        X, y, w = X.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)           # (B, C)
        loss = (loss.mean(dim=1) * w).mean()  # weight by sample quality
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)
```

**Cell 8 — validate (macro AUC for multi-label):**
```python
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for X, y, _ in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item() * len(y)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(y.cpu().numpy())
    probs   = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    # Binarize targets (primary=1.0 and secondary=0.5 both count as positive)
    bin_targets = (targets >= 0.5).astype(int)
    # Only compute AUC for classes that appear in this fold
    valid_cols = bin_targets.sum(axis=0) > 0
    try:
        auc = roc_auc_score(bin_targets[:, valid_cols], probs[:, valid_cols], average='macro')
    except ValueError:
        auc = 0.0
    return total_loss / len(loader.dataset), auc
```

**Cell 9 — GroupKFold CV loop:**
```python
gkf = GroupKFold(n_splits=CFG['N_FOLDS'])
groups = train_df['site'].values
oof_preds = np.zeros((len(train_df), BIRD_CLASSES), dtype=np.float32)
fold_aucs = []
history = []

for fold, (train_idx, val_idx) in enumerate(
        gkf.split(train_df, train_df['target'], groups=groups), start=1):
    print(f"\n{'='*60}\n  Fold {fold}/{CFG['N_FOLDS']}\n{'='*60}")
    # ... (same loop structure as nb03, but use BirdDataset new signature)
    # Save best checkpoint as OUTPUT_DIR / f'{CFG["MODEL_NAME"]}_fold{fold}.pth'
    # OOF collection: same as nb03
```
*(Implement the loop following exactly the same structure as nb03 lines 230–288.)*

**Cell 10 — Results + OOF save:**
```python
print(f"CV AUC ({CFG['MODEL_NAME']}): {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
oof_results = {
    'model': CFG['MODEL_NAME'], 'fold_aucs': [round(a,4) for a in fold_aucs],
    'mean_auc': round(float(np.mean(fold_aucs)), 4),
    'std_auc': round(float(np.std(fold_aucs)), 4),
}
fname = OUTPUT_DIR / f"oof_{CFG['MODEL_NAME']}.json"
with open(fname, 'w') as f: json.dump(oof_results, f, indent=2)
```

**Cell 11 — Training curve plot** (same as nb03)

**Cell 12 — Inference on test soundscapes:**
Use `predict_soundscape` from nb03 but with `torch.sigmoid` (BCE model, not softmax).
Predict with all folds, average predictions (simple mean of sigmoid outputs).
Build submission CSV aligned to `sample_submission.csv`.
**Important:** this model predicts BIRD_CLASSES (162) — leave non-bird columns as `1/NUM_CLASSES`.

**Cell 13 — ONNX export:**
Same as nb03 but dummy shape uses `DURATION=5` → 501 frames:
```python
n_frames = 1 + (CFG['SR'] * CFG['DURATION'] // CFG['HOP_LENGTH'])  # 501
dummy = torch.randn(1, 1, CFG['N_MELS'], n_frames).to(CFG['DEVICE'])
torch.onnx.export(inf_model, dummy,
    str(OUTPUT_DIR / f"{CFG['MODEL_NAME']}.onnx"),
    input_names=['input'], output_names=['output'], opset_version=11,
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
```

### Step 3: Verify the notebook is complete

Check that the file:
- Has `# %%` cell markers throughout
- Has `# %% [markdown]` markdown cells explaining each section
- Has no StratifiedKFold imports (replaced by GroupKFold)
- Uses BCEWithLogitsLoss not CrossEntropy
- Saves `oof_{MODEL_NAME}.json` and `{MODEL_NAME}_fold{k}.pth` and `{MODEL_NAME}.onnx` to OUTPUT_DIR

### Step 4: Commit

```bash
cd "c:/Users/alexy/Documents/Claude_projects"
git add "Kaggle competition/bird_clef/kaggle_notebooks/06_backbone_search.py"
git commit -m "feat: nb06 backbone search — BCE + GroupKFold + 10s training + 5 backbones"
```

---

## Task 2: nb07 — BirdNET Fine-tune

**File to create:** `kaggle_notebooks/07_birdnet_finetune.py`

**Purpose:** Load BirdNET-Analyzer pretrained weights (EfficientNet-B1 backbone pretrained on 9,000+ species from Xeno-canto/iNat) and fine-tune on BirdCLEF 2026 data. BirdNET's pretrained representations massively help rare bird species.

**Key differences from nb06:**
- Backbone: `efficientnet_b1` loaded with BirdNET pretrained weights (not ImageNet)
- BirdNET weights are available on Kaggle as a dataset or via HuggingFace
- Fine-tuning schedule: lower LR (1e-4), unfreeze backbone progressively
- Same BCE multi-label loss, same GroupKFold

### Step 1: Understand BirdNET weight loading

BirdNET-Analyzer uses an EfficientNet-B1 backbone. The pretrained model is available:
- HuggingFace: `google/bird-vocalization-classifier` (TFLite/TF SavedModel)
- Kaggle dataset: search for `birdnet-analyzer` or download from `https://github.com/kahst/BirdNET-Analyzer`
- Alternative: use `timm` `efficientnet_b1` with ImageNet pretrain as fallback if BirdNET weights unavailable

For Kaggle notebook, the safest approach:
```python
# Try to load BirdNET weights from a Kaggle dataset
BIRDNET_DIR = Path('/kaggle/input/birdnet-analyzer-model')
if BIRDNET_DIR.exists():
    # Load and map weights from BirdNET to timm EfficientNet-B1
    # BirdNET has a different final classifier head (9000+ classes)
    # We load only the feature extractor (all layers except the final classifier)
    state_dict = torch.load(BIRDNET_DIR / 'BirdNET_GLOBAL_6K_V2.4_Model.pt',
                            map_location='cpu')
    # Strip classifier head
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
    model.backbone.load_state_dict(state_dict, strict=False)
    print("✓ BirdNET weights loaded")
else:
    print("BirdNET weights not found — using ImageNet pretrain (add birdnet-analyzer dataset to notebook)")
```

### Step 2: Write the notebook

Create `kaggle_notebooks/07_birdnet_finetune.py` with these cells:

**Cell 1 — Header:**
```python
# =============================================================================
# BirdCLEF 2026 — BirdNET Fine-tuning (Bird Pipeline)
# =============================================================================
# Backbone  : EfficientNet-B1 with BirdNET pretrained weights
# Pretraining: BirdNET-Analyzer trained on 9,000+ bird species from XC + iNat
# Why BirdNET: pretrained representations drastically help rare species (< 10 clips)
# Fine-tuning: 2-phase — frozen backbone (5 ep) → full unfrozen (15 ep)
# Loss      : BCEWithLogitsLoss (multi-label)
# Folds     : GroupKFold by recorder site
# =============================================================================
```

**Cell 2 — Imports:** same as nb06

**Cell 3 — CFG:**
```python
CFG = dict(
    SR=32000, N_FFT=1024, HOP_LENGTH=320, N_MELS=128, FMIN=40, FMAX=15000,
    DURATION=5, TRAIN_DURATION=10,
    MODEL_NAME='efficientnet_b1',
    PRETRAINED=False,        # we load BirdNET weights, not ImageNet
    FREEZE_EPOCHS=5,         # train only head for first N epochs
    EPOCHS=20,               # total epochs
    BATCH_SIZE=32,
    LR_HEAD=1e-3,            # LR for head (frozen phase)
    LR_FULL=1e-4,            # LR for full model (unfrozen phase)
    WEIGHT_DECAY=1e-2,
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu',
    SEED=42,
    NUM_WORKERS=0 if os.name == 'nt' else 4,
)
```

**Cell 4 — BirdNET weight loading function:**
```python
def load_birdnet_weights(model, birdnet_dir):
    """
    Load BirdNET pretrained weights into timm EfficientNet-B1 backbone.
    Strips the original BirdNET classifier head (6000+ classes).
    Returns True if successful, False if weights not found (falls back to random init).
    """
    weight_path = Path(birdnet_dir) / 'BirdNET_GLOBAL_6K_V2.4_Model.pt'
    if not weight_path.exists():
        print(f"  BirdNET weights not found at {weight_path}")
        print("  → Add 'birdnet-analyzer' dataset to this Kaggle notebook")
        print("  → Falling back to random initialization (reduced effectiveness)")
        return False
    state_dict = torch.load(weight_path, map_location='cpu')
    # Keep only feature extractor layers (drop 'classifier.*' and 'head.*')
    filtered = {k: v for k, v in state_dict.items()
                if not any(k.startswith(prefix) for prefix in ['classifier', 'head', 'fc'])}
    missing, unexpected = model.backbone.load_state_dict(filtered, strict=False)
    print(f"  ✓ BirdNET weights loaded: {len(filtered)} layers")
    print(f"  Missing (new head): {len(missing)} keys")
    return True
```

**Cell 5 — Two-phase fine-tuning helper:**
```python
def set_backbone_frozen(model, frozen: bool):
    """Freeze/unfreeze backbone, always keep head trainable."""
    for name, param in model.backbone.named_parameters():
        if 'classifier' in name or 'head' in name:
            param.requires_grad = True  # always trainable
        else:
            param.requires_grad = not frozen
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state = 'frozen backbone' if frozen else 'full model'
    print(f"  {state}: {n_trainable:,} trainable parameters")
```

**Cell 6 — Training loop with phase switching:**
```python
# In the fold loop:
# Phase 1: frozen backbone, head only (FREEZE_EPOCHS)
set_backbone_frozen(model, frozen=True)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CFG['LR_HEAD'], weight_decay=CFG['WEIGHT_DECAY'])

for epoch in range(1, CFG['FREEZE_EPOCHS'] + 1):
    # ... train, validate

# Phase 2: full model (remaining epochs)
set_backbone_frozen(model, frozen=False)
optimizer = torch.optim.AdamW(model.parameters(),
    lr=CFG['LR_FULL'], weight_decay=CFG['WEIGHT_DECAY'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG['EPOCHS'] - CFG['FREEZE_EPOCHS'])

for epoch in range(CFG['FREEZE_EPOCHS'] + 1, CFG['EPOCHS'] + 1):
    # ... train, validate, save best
```

**Cells 7–12:** same as nb06 (BirdDataset, validate, OOF, inference, ONNX export).
Save outputs as `birdnet_fold{k}.pth` and `birdnet.onnx`.

### Step 3: Verify

Check the two-phase training logic: freeze happens correctly, optimizer is recreated for phase 2, learning rates differ by 10×.

### Step 4: Commit

```bash
git add "Kaggle competition/bird_clef/kaggle_notebooks/07_birdnet_finetune.py"
git commit -m "feat: nb07 BirdNET fine-tuning — 2-phase frozen/full training, pretrained bird representations"
```

---

## Task 3: nb08 — Non-Bird Pipeline

**File to create:** `kaggle_notebooks/08_nonbird_pipeline.py`

**Purpose:** Train a separate classifier for 72 non-bird classes (35 Amphibia + 28 Insecta + 8 Mammalia + 1 Reptilia) using Focal BCE to handle extreme imbalance. **Critical 2026 insight:** some non-bird species (especially insect sonotypes 47158son01–son25) have ZERO clips in `train_audio` and are ONLY available via expert-annotated soundscape segments in `train_soundscapes_labels.csv`.

**Key differences from nb06:**
- Target: `nonbird_labels` (72 classes), not `bird_labels`
- Primary data source: `train_soundscapes_labels.csv` (expert segments from actual PAM recordings)
- Secondary data: `train.csv` non-bird clips (limited: Amphibia 451, Insecta 199, Mammalia 99, Reptilia 1)
- Loss: Focal BCE (γ=2) to handle Caiman latirostris with 1 sample
- Model: ECA-NFNet-L0 (strong regularization for small datasets)
- Gold segment oversampling: 3× per epoch

### Step 1: Understand `train_soundscapes_labels.csv` format

This file has columns: `filename`, `ebird_code`, `start_sec`, `end_sec` (approximately).
Each row = one 5s expert-annotated segment from a PAM recording.
Audio source: `BASE_DIR / 'train_soundscapes' / filename`

Read the file, print head/dtypes before writing the notebook to understand column names.

### Step 2: Write the notebook

Create `kaggle_notebooks/08_nonbird_pipeline.py` with these cells:

**Cell 1 — Header:**
```python
# =============================================================================
# BirdCLEF 2026 — Non-Bird Pipeline (Amphibia + Insecta + Mammalia + Reptilia)
# =============================================================================
# Classes   : 72 non-bird species
# CRITICAL  : Insect sonotypes (47158son01–son25) have 0 train_audio clips
#             → PRIMARY data = train_soundscapes_labels.csv expert segments
# Loss      : Focal BCE (gamma=2) — handles Caiman (1 clip) and rare insects
# Model     : ECA-NFNet-L0 (strong regularization for small datasets)
# Folds     : GroupKFold by recorder site
# =============================================================================
```

**Cell 2 — Imports + CFG:**
```python
CFG = dict(
    SR=32000, N_FFT=1024, HOP_LENGTH=320, N_MELS=128, FMIN=40, FMAX=15000,
    DURATION=5, TRAIN_DURATION=5,   # soundscape segments are already 5s
    MODEL_NAME='eca_nfnet_l0',
    PRETRAINED=True,
    N_FOLDS=5,
    EPOCHS=25,              # more epochs — smaller dataset
    BATCH_SIZE=16,          # smaller batch for small dataset
    LR=1e-3,
    WEIGHT_DECAY=5e-2,      # stronger regularization
    GOLD_OVERSAMPLE=3,      # repeat gold segments N times per epoch
    FOCAL_GAMMA=2.0,
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu',
    SEED=42,
    NUM_WORKERS=0 if os.name == 'nt' else 4,
)
```

**Cell 3 — Focal BCE loss:**
```python
class FocalBCELoss(nn.Module):
    """
    Focal Binary Cross Entropy for multi-label imbalanced classification.
    Down-weights easy negatives; focuses learning on hard/rare examples.
    gamma=2 is standard; alpha=0.25 for positive class upweighting.
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss
```

**Cell 4 — Build combined dataset from two sources:**

```python
# --- Source 1: train.csv non-bird clips ---
train_df = pd.read_csv(BASE_DIR / 'train.csv')
nonbird_clips = train_df[train_df['primary_label'].isin(nonbird_labels)].copy()
nonbird_clips['target'] = nonbird_clips['primary_label'].map(nonbird2idx)
nonbird_clips['filepath'] = nonbird_clips['filename'].apply(
    lambda f: str(BASE_DIR / 'train_audio' / f))
nonbird_clips['weight'] = nonbird_clips.apply(get_sample_weight, axis=1)  # same fn as nb06
nonbird_clips['source_type'] = 'clip'
nonbird_clips['start_sec'] = 0.0
nonbird_clips['site'] = nonbird_clips.get('recorder_id', 'clip_' + nonbird_clips['primary_label'])

# --- Source 2: train_soundscapes_labels.csv expert segments ---
soundscape_labels_path = BASE_DIR / 'train_soundscapes_labels.csv'
if soundscape_labels_path.exists():
    sl_df = pd.read_csv(soundscape_labels_path)
    print(sl_df.head(3))
    print(sl_df.dtypes)
    # Filter to non-bird species only
    sl_df = sl_df[sl_df['ebird_code'].isin(nonbird_labels)].copy()
    sl_df['target'] = sl_df['ebird_code'].map(nonbird2idx)
    sl_df['filepath'] = sl_df['filename'].apply(
        lambda f: str(BASE_DIR / 'train_soundscapes' / f))
    sl_df['weight'] = 1.0   # Gold weight
    sl_df['source_type'] = 'gold'
    sl_df['primary_label'] = sl_df['ebird_code']
    # Site = recorder from soundscape filename (e.g., 'S22_20230101_060000.ogg' → 'S22')
    sl_df['site'] = sl_df['filename'].str.extract(r'^([A-Z]\d+)')[0].fillna('unknown')
    print(f"Gold segments (non-bird): {len(sl_df)}")
    # Repeat gold segments GOLD_OVERSAMPLE times
    gold_repeated = pd.concat([sl_df] * CFG['GOLD_OVERSAMPLE'], ignore_index=True)
    combined_df = pd.concat([nonbird_clips, gold_repeated], ignore_index=True)
else:
    print("WARNING: train_soundscapes_labels.csv not found — using clips only")
    combined_df = nonbird_clips

print(f"Total training rows: {len(combined_df)}")
print(combined_df['primary_label'].value_counts().head(20))
```

**Cell 5 — NonBirdDataset:**
Same as BirdDataset in nb06, but:
- `n_samples = cfg['SR'] * cfg['DURATION']` (always 5s — gold segments are pre-cut)
- For gold segments: load from soundscape at `start_sec` offset
- Target shape: `(NONBIRD_CLASSES,)` not `(BIRD_CLASSES,)`

```python
def _load_audio(self, row):
    path = row['filepath']
    start_sec = float(row.get('start_sec', 0.0))
    try:
        offset = start_sec
        y, _ = librosa.load(path, sr=self.cfg['SR'], mono=True,
                            offset=offset, duration=self.cfg['DURATION'])
    except Exception:
        return np.zeros(self.n_samples, dtype=np.float32)
    # ... pad/trim as usual
```

**Cells 6–12:** Training loop (using FocalBCELoss), validate, OOF, inference, ONNX export.
- Model output: 72 classes (non-bird only)
- Save as `nonbird_fold{k}.pth` and `nonbird_eca_nfnet_l0.onnx`
- Submission: fill bird columns with `1/NUM_CLASSES`, fill non-bird from model predictions

### Step 3: Verify

- `train_soundscapes_labels.csv` loading handles unknown column names gracefully (print head before assuming columns)
- Gold segment loading uses `offset=start_sec` in librosa
- FocalBCELoss is used in `train_one_epoch`, not standard BCE

### Step 4: Commit

```bash
git add "Kaggle competition/bird_clef/kaggle_notebooks/08_nonbird_pipeline.py"
git commit -m "feat: nb08 non-bird pipeline — Focal BCE, soundscape gold segments, insect sonotypes"
```

---

## Task 4: nb09 — Pseudo-Labeling Pipeline

**File to create:** `kaggle_notebooks/09_pseudo_labeling.py`

**Purpose:** Run inference with 3 models (B4 + EVA-02 + DINOv2) on unlabeled soundscapes, assign pseudo-labels via 2-of-3 consensus, apply rarity-stratified confidence thresholds and power scaling, output a new `pseudo_labels_r1.csv` for Round 1 retraining.

**Key components:**
- Load 3 pretrained models: B4 from nb06, EVA-02/DINOv2 from nb10 (or use B3+B4+RegNetY as CNN-only fallback if ViTs not yet trained)
- Chunk each unlabeled soundscape into 5s segments
- For each segment: get predictions from all 3 models
- Consensus: label a segment if ≥2/3 models agree (top-1 prediction is same class)
- Apply confidence thresholds stratified by class rarity
- Power scaling: `probs = probs ** 0.7`
- Output CSV with columns: `filepath, start_sec, end_sec, primary_label, confidence, pl_round`

### Step 1: Read the pseudo-labeling strategy

From `docs/plans/2026-03-12-winning-strategy-design.md` Section 4:
- Thresholds: common (>500 clips)=0.70, medium (100–500)=0.80, rare (10–100)=0.85, very rare (<10, all non-bird)=0.90
- Power scaling: `probs = probs ** 0.7`
- Target: unlabeled_soundscapes directory
- Output: `pseudo_labels_r1.csv`

### Step 2: Write the notebook

**Cell 1 — Header:**
```python
# =============================================================================
# BirdCLEF 2026 — Pseudo-Labeling Pipeline (Round 1)
# =============================================================================
# Models    : Bird model (B4/B3) + optional ViT models (EVA-02, DINOv2)
# Consensus : 2-of-3 agreement on top-1 prediction
# Thresholds: stratified by class rarity (common=0.70 ... very rare=0.90)
# Power scale: probs = probs ** 0.7  (softens overconfidence)
# Output    : pseudo_labels_r1.csv (use as extra training data in next run)
# =============================================================================
```

**Cell 2 — Load class rarity lookup:**
```python
train_df = pd.read_csv(BASE_DIR / 'train.csv')
clip_counts = train_df['primary_label'].value_counts().to_dict()

def get_threshold(label, clip_counts):
    """Rarity-stratified confidence threshold."""
    n = clip_counts.get(label, 0)
    if n > 500:  return 0.70   # common
    if n > 100:  return 0.80   # medium
    if n > 10:   return 0.85   # rare
    return 0.90                # very rare (< 10 clips, all non-bird)
```

**Cell 3 — Load models:**
```python
def load_model(checkpoint_path, model_name, num_classes, device):
    """Load a trained BirdModel from checkpoint. Returns None if file missing."""
    if not Path(checkpoint_path).exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        return None
    model = BirdModel(model_name, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()
    return model

# Load available models — adapt paths to where checkpoints were saved
models = {}
# B4 from nb06 (fold 1 best checkpoint)
m = load_model(OUTPUT_DIR / 'efficientnet_b4_fold1.pth', 'efficientnet_b4', BIRD_CLASSES, DEVICE)
if m: models['b4'] = m
# B3 from nb06
m = load_model(OUTPUT_DIR / 'efficientnet_b3_fold1.pth', 'efficientnet_b3', BIRD_CLASSES, DEVICE)
if m: models['b3'] = m
# RegNetY from nb06
m = load_model(OUTPUT_DIR / 'regnety_016_fold1.pth', 'regnety_016', BIRD_CLASSES, DEVICE)
if m: models['regnety'] = m

print(f"Models loaded: {list(models.keys())}")
if len(models) < 2:
    print("WARNING: Need at least 2 models for consensus. Run nb06 first.")
```

**Cell 4 — Soundscape chunker:**
```python
def chunk_soundscape(path, cfg):
    """Yield (start_sec, mel_tensor) for each 5s chunk."""
    try:
        y, _ = librosa.load(str(path), sr=cfg['SR'], mono=True)
    except Exception as e:
        print(f"  Error: {path.name}: {e}")
        return
    n_samples = cfg['SR'] * cfg['DURATION']
    n_chunks = len(y) // n_samples
    for i in range(n_chunks):
        chunk = y[i * n_samples:(i + 1) * n_samples]
        S = librosa.feature.melspectrogram(
            y=chunk.astype(np.float32), sr=cfg['SR'],
            n_fft=cfg['N_FFT'], hop_length=cfg['HOP_LENGTH'],
            n_mels=cfg['N_MELS'], fmin=cfg['FMIN'], fmax=cfg['FMAX'])
        S_db = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        yield i * cfg['DURATION'], torch.from_numpy(S_norm).float().unsqueeze(0)
```

**Cell 5 — Consensus pseudo-labeling:**
```python
def get_consensus_labels(chunk_tensor, models, label_list, clip_counts, cfg, device, batch_size=32):
    """
    Run chunk through all models, return consensus pseudo-labels.
    Returns list of (label, mean_confidence) for chunks where ≥2 models agree.
    """
    chunk_tensor = chunk_tensor.unsqueeze(0).to(device)  # (1, 1, 128, 501)

    model_preds = {}
    for name, model in models.items():
        with torch.no_grad():
            logits = model(chunk_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # (BIRD_CLASSES,)
        # Power scaling
        probs = probs ** 0.7
        model_preds[name] = probs

    # Find top-1 prediction per model
    top1_labels = {name: label_list[np.argmax(p)] for name, p in model_preds.items()}
    top1_confs  = {name: np.max(p) for name, p in model_preds.items()}

    # Consensus: does ≥2/3 models agree on the same top-1?
    from collections import Counter
    vote_counts = Counter(top1_labels.values())
    best_label, best_votes = vote_counts.most_common(1)[0]

    if best_votes < 2:
        return None  # no consensus

    # Check threshold
    agreeing_confs = [top1_confs[n] for n, l in top1_labels.items() if l == best_label]
    mean_conf = np.mean(agreeing_confs)
    threshold = get_threshold(best_label, clip_counts)

    if mean_conf < threshold:
        return None

    return best_label, mean_conf
```

**Cell 6 — Main PL loop over unlabeled soundscapes:**
```python
unlabeled_dir = BASE_DIR / 'unlabeled_soundscapes'
unlabeled_files = sorted(unlabeled_dir.glob('*.ogg')) if unlabeled_dir.exists() else []
print(f"Unlabeled soundscapes: {len(unlabeled_files)}")

pl_records = []
for sf_path in unlabeled_files[:500]:  # process first 500 for R1 (adjust as needed)
    for start_sec, mel in chunk_soundscape(sf_path, CFG):
        result = get_consensus_labels(mel, models, bird_labels, clip_counts, CFG, DEVICE)
        if result:
            label, conf = result
            pl_records.append({
                'filepath': str(sf_path),
                'start_sec': start_sec,
                'end_sec': start_sec + CFG['DURATION'],
                'primary_label': label,
                'confidence': round(conf, 4),
                'pl_round': 1,
            })

pl_df = pd.DataFrame(pl_records)
print(f"Pseudo-labels generated: {len(pl_df)}")
print(pl_df['primary_label'].value_counts().head(20))
pl_df.to_csv(OUTPUT_DIR / 'pseudo_labels_r1.csv', index=False)
```

**Cell 7 — Statistics plot:** show distribution of PL species, confidence histogram, rare vs common breakdown.

### Step 3: Verify

- Handles case where <2 models are loaded (prints warning, skips)
- Power scaling applied before threshold check
- Output CSV has correct columns

### Step 4: Commit

```bash
git add "Kaggle competition/bird_clef/kaggle_notebooks/09_pseudo_labeling.py"
git commit -m "feat: nb09 pseudo-labeling R1 — 2-of-3 consensus, rarity thresholds, power scaling"
```

---

## Task 5: nb10 — ViT Pseudo-Label Generator

**File to create:** `kaggle_notebooks/10_vit_pl_generator.py`

**Purpose:** Fine-tune EVA-02 Large (448px) and/or DINOv2-Large-reg4 on BirdCLEF 2026 data. These models are used ONLY for pseudo-label generation — they are too slow for competition inference but give much higher-quality pseudo-labels than CNNs alone.

**Important constraints:**
- These notebooks run for 15–20 epochs on Kaggle GPU (~7h for EVA-02 Large 20ep)
- Output: fold1 checkpoint only (used as one of the 3 PL voters in nb09)
- NOT exported to ONNX (not used in inference)
- Architecture: `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` (timm)
  - Alternative: `vit_large_patch14_reg4_dinov2.lvd142m` (DINOv2)

### Step 1: Write the notebook

**Cell 1 — Header:**
```python
# =============================================================================
# BirdCLEF 2026 — ViT Pseudo-Label Generator
# =============================================================================
# Model     : EVA-02 Large 448px OR DINOv2-Large-reg4 (select via CFG)
# Purpose   : HIGH-QUALITY pseudo-label generation for unlabeled soundscapes
# NOT used for competition inference (too slow: ~8-15s/chunk on CPU)
# Competition inference uses EfficientNet + OpenVINO (see nb05/nb06)
# Training  : Single fold (fold 1 only) — we need diversity, not 5-fold averaging
# Output    : vit_fold1.pth (used as voter #3 in nb09 pseudo-labeling)
# =============================================================================
```

**Cell 2 — CFG with ViT selection:**
```python
CFG = dict(
    SR=32000, N_FFT=1024, HOP_LENGTH=320, N_MELS=128, FMIN=40, FMAX=15000,
    DURATION=5, TRAIN_DURATION=10,
    # Select one:
    MODEL_NAME='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',  # EVA-02
    # MODEL_NAME='vit_large_patch14_reg4_dinov2.lvd142m',          # DINOv2
    IMG_SIZE=448,          # EVA-02 native resolution
    PRETRAINED=True,
    N_FOLDS=5,
    TRAIN_FOLD=1,          # only train fold 1 (for PL diversity)
    EPOCHS=15,
    BATCH_SIZE=16,         # ViT needs more memory
    LR=5e-5,               # lower LR for large pretrained ViT
    WEIGHT_DECAY=1e-2,
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu',
    SEED=42,
    NUM_WORKERS=0 if os.name == 'nt' else 4,
)
```

**Cell 3 — ViT BirdDataset (key difference: resize to IMG_SIZE×IMG_SIZE):**
```python
# ViTs expect square images
# Mel spectrogram is (N_MELS=128, T=501) → need to resize to (IMG_SIZE, IMG_SIZE)
# Also: ViTs need 3 channels (RGB) — repeat mono mel 3 times

import torchvision.transforms.functional as TF

class ViTBirdDataset(Dataset):
    def __init__(self, df, cfg, augment=False):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.augment = augment
        self.n_samples = cfg['SR'] * (cfg['TRAIN_DURATION'] if augment else cfg['DURATION'])
        self.img_size = cfg['IMG_SIZE']

    # _load_audio: same as BirdDataset in nb06
    # _to_melspec: same as BirdDataset in nb06

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self._load_audio(row['filepath'])
        S = self._to_melspec(y)
        # Resize to (IMG_SIZE, IMG_SIZE) and expand to 3 channels
        S_tensor = torch.from_numpy(S).unsqueeze(0)  # (1, N_MELS, T)
        S_resized = TF.resize(S_tensor, [self.img_size, self.img_size])
        S_rgb = S_resized.repeat(3, 1, 1)  # (3, IMG_SIZE, IMG_SIZE)
        # Normalize with ImageNet stats (ViTs pretrained on ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        S_norm = (S_rgb - mean) / std
        # Multi-hot target
        target = np.zeros(BIRD_CLASSES, dtype=np.float32)
        target[int(row['target'])] = 1.0
        weight = float(row.get('weight', 1.0))
        return S_norm, torch.from_numpy(target), weight
```

**Cell 4 — ViTBirdModel:**
```python
class ViTBirdModel(nn.Module):
    """
    ViT (EVA-02 or DINOv2) fine-tuned for bird classification.
    NOTE: Only used for pseudo-label generation, not competition inference.
    """
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          num_classes=num_classes)
        # timm handles the classifier head replacement automatically
    def forward(self, x): return self.backbone(x)
```

**Cells 5–10:** Train single fold (fold 1 only), validate, save `vit_fold1.pth`.
No ONNX export. No submission generation. Just checkpoint + OOF AUC for fold 1.

**Cell 11 — Reminder print:**
```python
print("\n" + "="*60)
print("ViT training complete.")
print(f"Checkpoint saved: {OUTPUT_DIR / 'vit_fold1.pth'}")
print("\nNext steps:")
print("1. Download vit_fold1.pth from Output")
print("2. Upload as a Kaggle dataset")
print("3. Use in nb09_pseudo_labeling.py as voter #3")
print("="*60)
```

### Step 2: Verify

- IMG_SIZE resize is correct (ViTs fail silently with wrong input size)
- 3-channel repeat before ImageNet normalization
- Only fold 1 is trained (TRAIN_FOLD=1)
- No ONNX export cell

### Step 3: Commit

```bash
git add "Kaggle competition/bird_clef/kaggle_notebooks/10_vit_pl_generator.py"
git commit -m "feat: nb10 ViT PL generator — EVA-02/DINOv2 fine-tune for pseudo-label diversity"
```

---

## Task 6: nb11 — Ensemble + Final Submission

**File to create:** `kaggle_notebooks/11_ensemble.py`

**Purpose:** Load all trained fold checkpoints from both pipelines, run inference on test soundscapes, rank-average bird predictions, weighted-average non-bird predictions, merge into 234-column submission.

**This is the final submission notebook — it must run within 90-min CPU budget.**

### Step 1: Write the notebook

**Cell 1 — Header:**
```python
# =============================================================================
# BirdCLEF 2026 — Final Ensemble Submission
# =============================================================================
# Bird pipeline  : B1 + B3 + B4 + RegNetY-016 + BirdNET → rank averaging (162 cols)
# Non-bird pipeline: ECA-NFNet-L0 + B3-focal → weighted avg 0.6/0.4 (72 cols)
# Inference: OpenVINO FP16 (CPU budget ~59 min)
# Post-processing: geotemporal prior (Pantanal species filter)
# =============================================================================
```

**Cell 2 — Imports + OpenVINO setup:**
```python
try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("OpenVINO not available — falling back to ONNX Runtime")
import onnxruntime as ort
```

**Cell 3 — Model registry:**
```python
# Register all available models
# Each entry: (onnx_path_or_checkpoint, model_name, n_classes, pipeline)
# Adapt these paths to wherever Kaggle input datasets are mounted

BIRD_MODELS = [
    ('/kaggle/input/birdclef26-b1/efficientnet_b1.onnx',    'b1',     162, 'bird'),
    ('/kaggle/input/birdclef26-b3/efficientnet_b3.onnx',    'b3',     162, 'bird'),
    ('/kaggle/input/birdclef26-b4/efficientnet_b4.onnx',    'b4',     162, 'bird'),
    ('/kaggle/input/birdclef26-regnety/regnety_016.onnx',   'regnety',162, 'bird'),
    ('/kaggle/input/birdclef26-birdnet/birdnet.onnx',        'birdnet',162, 'bird'),
]
NONBIRD_MODELS = [
    ('/kaggle/input/birdclef26-nonbird/nonbird_eca_nfnet_l0.onnx', 'nb_nfnet', 72, 'nonbird'),
]

# Load only models whose ONNX files exist
def load_ov_session(path):
    if not Path(path).exists():
        return None
    if OPENVINO_AVAILABLE:
        core = ov.Core()
        model = core.read_model(path)
        return core.compile_model(model, 'CPU')
    return ort.InferenceSession(path, providers=['CPUExecutionProvider'])

def run_model(session, x_np):
    """Run ONNX or OpenVINO session. x_np: (B, 1, 128, 501) float32."""
    if OPENVINO_AVAILABLE and hasattr(session, 'output'):
        return session([x_np])[session.output(0)]
    return session.run(None, {'input': x_np})[0]
```

**Cell 4 — Inference with rank averaging:**
```python
def rank_average(pred_list):
    """
    Convert each model's probabilities to ranks, then average ranks.
    Robust to miscalibrated probability scales across models.
    """
    from scipy.stats import rankdata
    ranked = [rankdata(p, axis=1) for p in pred_list]  # (N, C) rank arrays
    return np.mean(ranked, axis=0)

def weighted_average(pred_list, weights):
    """Weighted probability average."""
    total = sum(weights)
    return sum(p * w / total for p, w in zip(pred_list, weights))
```

**Cell 5 — Full soundscape inference loop** (same chunking as nb06/nb03, but uses sessions not PyTorch models)

**Cell 6 — Merge bird + non-bird:**
```python
# bird_preds: (N_chunks, 162), nonbird_preds: (N_chunks, 72)
# Need to map back to 234-column submission order

full_preds = np.zeros((len(chunk_ids), NUM_CLASSES), dtype=np.float32)
# Fill bird columns
for i, label in enumerate(bird_labels):
    col = label2idx[label]
    full_preds[:, col] = bird_preds_ranked[:, i]
# Fill non-bird columns
for i, label in enumerate(nonbird_labels):
    col = label2idx[label]
    full_preds[:, col] = nonbird_preds_weighted[:, i]
```

**Cell 7 — Geotemporal prior (optional, apply if species list available):**
```python
# Species impossible in Pantanal: floor their prediction to near-zero
# Conservative: only apply if we have a verified list
PANTANAL_ABSENT = []  # fill from iNat/eBird occurrence data if available
for label in PANTANAL_ABSENT:
    if label in label2idx:
        full_preds[:, label2idx[label]] = np.minimum(full_preds[:, label2idx[label]], 0.001)
```

**Cell 8 — Build and save submission:**
Same vectorized merge pattern as nb03.

**Cell 9 — Runtime summary:**
```python
print(f"Total inference time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Chunks processed: {len(chunk_ids)}")
print(f"Budget remaining: {90 - elapsed/60:.1f} min")
```

### Step 2: Verify

- Bird columns and non-bird columns in final submission are filled from separate pipelines
- Rank averaging used for bird (not probability averaging)
- Submission has exactly 234 species columns in the same order as sample_submission.csv
- `sub.shape[1] == 235` (row_id + 234 species)

### Step 3: Commit

```bash
git add "Kaggle competition/bird_clef/kaggle_notebooks/11_ensemble.py"
git commit -m "feat: nb11 final ensemble — rank-avg bird + weighted non-bird, OpenVINO FP16, geotemporal prior"
```

---

## Execution Order

These tasks are independent (each notebook is a standalone file) and can be run in separate chat sessions:

| Chat session | Task | GPU needed | Prerequisite |
|---|---|---|---|
| Session A | Task 1 (nb06) | No (writing) | Read nb03 |
| Session B | Task 2 (nb07) | No (writing) | Read nb03 |
| Session C | Task 3 (nb08) | No (writing) | Read nb03 |
| Session D | Task 4 (nb09) | No (writing) | Read nb06 |
| Session E | Task 5 (nb10) | No (writing) | Read nb06 |
| Session F | Task 6 (nb11) | No (writing) | Read nb05, nb06 |

When actually running on Kaggle:
1. Run nb06 first (get first backbone checkpoints)
2. Run nb08 (non-bird pipeline — independent)
3. Run nb10 (ViT — needs GPU, long runtime ~7h)
4. Run nb09 with outputs from nb06 + nb10 (pseudo-labels)
5. Retrain nb06 with pseudo-labels added to training data
6. Run nb11 (final ensemble) as submission notebook
