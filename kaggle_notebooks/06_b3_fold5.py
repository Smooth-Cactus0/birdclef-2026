# %%
# =============================================================================
# BirdCLEF 2026 -- Backbone Search (Bird Pipeline)
# =============================================================================
# Change CFG['MODEL_NAME'] to swap backbones:
#   'efficientnet_b1'   -- lightweight upgrade from B0
#   'efficientnet_b3'   -- accuracy/speed sweet spot
#   'efficientnet_b4'   -- top accuracy in bird pipeline
#   'regnety_016'       -- architectural diversity
#   'eca_nfnet_l0'      -- strong regularization
# Loss    : BCEWithLogitsLoss (multi-label -- test chunks have multiple species)
# Folds   : GroupKFold by recorder site (honest CV)
# Duration: 10s train / 5s inference (longer context helps per 2024 winners)
# =============================================================================

# %% [markdown]
# # BirdCLEF 2026 -- Backbone Search (Bird Pipeline)
#
# This notebook trains a single backbone on 162 Aves species and evaluates it
# with **GroupKFold** cross-validation grouped by recorder site. Run it 5 times
# with different `CFG['MODEL_NAME']` values to compare backbones side-by-side.
#
# Key upgrades vs the B0 baseline (nb03):
# - **BCE multi-label loss** -- soundscape chunks often contain multiple species
# - **GroupKFold by site** -- prevents geographic leakage across folds
# - **10s training clips** -- longer context improves rare-species recall
# - **Sample weighting** -- down-weight low-quality XC and iNat recordings

# %%
# -- Install / version pins ----------------------------------------------------
# !pip install -q timm==1.0.3  # uncomment on Kaggle if needed

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

# -- Paths ---------------------------------------------------------------------
BASE_DIR   = (Path('/kaggle/input/competitions/birdclef-2026')
              if Path('/kaggle/input/competitions/birdclef-2026').exists()
              else Path('birdclef-2026'))
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
NUM_WORKERS = 0 if os.name == 'nt' else 4

# %%
# -- Config --------------------------------------------------------------------
CFG = dict(
    # Audio constants -- keep identical across all notebooks
    SR          = 32000,
    N_FFT       = 1024,
    HOP_LENGTH  = 320,
    N_MELS      = 128,
    FMIN        = 40,
    FMAX        = 15000,
    DURATION    = 5,           # inference clip length (seconds)
    TRAIN_DURATION = 10,       # training clip -- longer context per 2024 insight
    # Model -- change this per run to search backbones
    MODEL_NAME  = 'efficientnet_b3',   # <- swap to try others
    PRETRAINED  = True,
    # Training
    # TRAIN_FOLDS: which folds to run. Default=[1] to fit in 9h Kaggle budget.
    # For full 5-fold CV run separate jobs: [2], [3], [4], [5].
    N_FOLDS     = 5,
    TRAIN_FOLDS = [5],
    EPOCHS      = 10,
    BATCH_SIZE  = 32,
    LR          = 3e-4,
    WEIGHT_DECAY = 1e-2,
    # Device
    DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu',
    SEED        = 42,
    NUM_WORKERS = NUM_WORKERS,
)
print(f"Device : {CFG['DEVICE']}")
print(f"torch  : {torch.__version__},  timm: {timm.__version__}")
print(f"Backbone: {CFG['MODEL_NAME']}")

# %%
# %% [markdown]
# ## Data Setup
#
# We load `train.csv`, keep only **Aves** clips for this pipeline (162 species),
# assign quality-based sample weights, and derive a `site` group for GroupKFold.

# %%
# -- Load metadata -------------------------------------------------------------
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])

taxonomy   = pd.read_csv(BASE_DIR / 'taxonomy.csv')
train_df   = pd.read_csv(BASE_DIR / 'train.csv')

# Full 234-class label list (taxonomy order) -- used for submission alignment
label_list  = taxonomy['primary_label'].tolist()
label2idx   = {l: i for i, l in enumerate(label_list)}
NUM_CLASSES = len(label_list)   # 234

# Bird-only subset (162 Aves species)
bird_labels  = taxonomy[taxonomy['class_name'] == 'Aves']['primary_label'].tolist()
bird2idx     = {l: i for i, l in enumerate(bird_labels)}
BIRD_CLASSES = len(bird_labels)  # 162

print(f"Total taxonomy classes : {NUM_CLASSES}")
print(f"Bird classes (Aves)    : {BIRD_CLASSES}")

# -- Filter to bird-only clips -------------------------------------------------
train_df = train_df[train_df['primary_label'].isin(bird_labels)].copy()
train_df['target']   = train_df['primary_label'].map(bird2idx)
train_df['filepath'] = train_df['filename'].apply(
    lambda f: str(BASE_DIR / 'train_audio' / f))

print(f"Training clips (Aves) : {len(train_df):,}")
print(train_df['class_name'].value_counts())

# -- Sample weights by data quality -------------------------------------------
# Gold=1.0, XC>=3.0=0.8, XC<3.0=0.5, iNat=0.4
# 'collection' column distinguishes XC vs iNat; 'rating' is XC quality score
def get_sample_weight(row):
    collection = str(row.get('collection', 'XC')).lower()
    if 'inat' in collection or 'inaturalist' in collection:
        return 0.4
    r = float(row.get('rating', 0.0)) if pd.notna(row.get('rating', 0.0)) else 0.0
    if r >= 3.0:
        return 0.8
    return 0.5

train_df['weight'] = train_df.apply(get_sample_weight, axis=1)
print("\nSample weight distribution:")
print(train_df['weight'].value_counts().sort_index())

# -- GroupKFold site grouping --------------------------------------------------
# Use recorder_id if available (from PAM soundscapes), else fall back to
# primary_label as a proxy so clips from the same species stay in the same fold
train_df['site'] = train_df.get('recorder_id', train_df['primary_label'])
print(f"\nUnique groups (sites/species): {train_df['site'].nunique()}")

# %%
# %% [markdown]
# ## Dataset
#
# `BirdDataset` produces:
# - **Input**: `(1, N_MELS, T)` log-mel spectrogram tensor
# - **Target**: multi-hot vector of length `BIRD_CLASSES` -- primary label = 1.0,
#   secondary labels = 0.5 (soft positives, still counted as positive in AUC)
# - **Weight**: per-sample quality weight for the weighted BCE loss

# %%
class BirdDataset(Dataset):
    """
    Loads OGG audio, converts to log-mel spectrogram.
    - Training  : 10s random crop  (longer context)
    - Validation: 5s  centre crop  (matches inference)
    - Target    : multi-hot vector (BCE, not CrossEntropy)
    """
    def __init__(self, df, cfg, augment=False):
        self.df        = df.reset_index(drop=True)
        self.cfg       = cfg
        self.augment   = augment
        # Use longer 10s clips during training; 5s during validation
        dur            = cfg['TRAIN_DURATION'] if augment else cfg['DURATION']
        self.n_samples = cfg['SR'] * dur

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path):
        try:
            y, _ = librosa.load(path, sr=self.cfg['SR'], mono=True)
        except Exception:
            return np.zeros(self.n_samples, dtype=np.float32)
        if len(y) < self.n_samples:
            y = np.pad(y, (0, self.n_samples - len(y)), mode='constant')
        elif self.augment and len(y) > self.n_samples:
            start = np.random.randint(0, len(y) - self.n_samples)
            y = y[start:start + self.n_samples]
        else:
            y = y[:self.n_samples]
        return y.astype(np.float32)

    def _to_melspec(self, y):
        S = librosa.feature.melspectrogram(
            y=y, sr=self.cfg['SR'],
            n_fft=self.cfg['N_FFT'], hop_length=self.cfg['HOP_LENGTH'],
            n_mels=self.cfg['N_MELS'], fmin=self.cfg['FMIN'], fmax=self.cfg['FMAX'])
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        return S_db.astype(np.float32)

    def _spec_augment(self, S):
        """SpecAugment: random frequency + time masking."""
        S = S.copy()
        # Frequency masking
        F  = np.random.randint(0, 16)
        f0 = np.random.randint(0, self.cfg['N_MELS'] - F)
        S[f0:f0 + F, :] = 0.0
        # Time masking
        T  = np.random.randint(0, 30)
        t0 = np.random.randint(0, S.shape[1] - T)
        S[:, t0:t0 + T] = 0.0
        return S

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y   = self._load_audio(row['filepath'])
        S   = self._to_melspec(y)
        if self.augment:
            S = self._spec_augment(S)

        # Multi-hot target vector for BCE
        target = np.zeros(BIRD_CLASSES, dtype=np.float32)
        t = int(row['target'])
        if 0 <= t < BIRD_CLASSES:
            target[t] = 1.0
        # Secondary labels as soft positives (0.5)
        sec = row.get('secondary_labels', '')
        if isinstance(sec, str) and sec:
            for sl in sec.replace("'", "").strip("[]").split(','):
                sl = sl.strip()
                if sl in bird2idx:
                    target[bird2idx[sl]] = 0.5

        weight = float(row.get('weight', 1.0))
        return torch.from_numpy(S).unsqueeze(0), torch.from_numpy(target), weight

# %%
# %% [markdown]
# ## Model
#
# `BirdModel` wraps any `timm` backbone with `in_chans=1` (single-channel
# mel spectrogram) and `num_classes=BIRD_CLASSES` (162).

# %%
class BirdModel(nn.Module):
    """
    timm backbone accepting (B, 1, N_MELS, T) mel spectrograms.
    Output: logits of shape (B, BIRD_CLASSES) -- apply sigmoid for probabilities.
    """
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            in_chans=1, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

# %%
# %% [markdown]
# ## Training Functions
#
# `train_one_epoch` applies **weighted BCE** -- each sample's loss is scaled by
# its quality weight so that high-confidence XC recordings guide the gradient
# more than low-quality iNat clips.

# %%
def train_one_epoch(model, loader, optimizer, scheduler, device):
    """BCE with per-sample quality weighting."""
    model.train()
    total_loss = 0.0
    criterion  = nn.BCEWithLogitsLoss(reduction='none')
    for X, y, w in loader:
        X, y, w = X.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)           # (B, C)
        loss   = (loss.mean(dim=1) * w).mean()  # weight by sample quality
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    """Compute BCE loss + macro ROC-AUC for multi-label targets."""
    model.eval()
    all_probs, all_targets = [], []
    criterion  = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for X, y, _ in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item() * len(y)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(y.cpu().numpy())
    probs   = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    # Binarise: primary (1.0) and secondary (0.5) both count as positive
    bin_targets = (targets >= 0.5).astype(int)
    # Only compute AUC for classes that appear in this fold
    valid_cols = bin_targets.sum(axis=0) > 0
    try:
        auc = roc_auc_score(bin_targets[:, valid_cols],
                            probs[:, valid_cols], average='macro')
    except ValueError:
        auc = 0.0
    return total_loss / len(loader.dataset), auc

# %%
# %% [markdown]
# ## Cross-Validation Loop
#
# **GroupKFold** by recorder site keeps all clips from the same geographic
# location in the same fold, preventing the model from exploiting site-level
# acoustic fingerprints instead of species vocalisations.

# %%
gkf      = GroupKFold(n_splits=CFG['N_FOLDS'])
groups   = train_df['site'].values
oof_preds = np.zeros((len(train_df), BIRD_CLASSES), dtype=np.float32)
fold_aucs = []
history   = []

for fold, (train_idx, val_idx) in enumerate(
        gkf.split(train_df, train_df['target'], groups=groups), start=1):

    if fold not in CFG['TRAIN_FOLDS']:
        print(f"  Skipping fold {fold} (not in TRAIN_FOLDS={CFG['TRAIN_FOLDS']})")
        continue

    print(f"\n{'='*60}\n  Fold {fold}/{CFG['N_FOLDS']}\n{'='*60}")
    t0 = time.time()

    train_fold = train_df.iloc[train_idx]
    val_fold   = train_df.iloc[val_idx]

    train_ds = BirdDataset(train_fold, CFG, augment=True)
    val_ds   = BirdDataset(val_fold,   CFG, augment=False)
    train_dl = DataLoader(train_ds, batch_size=CFG['BATCH_SIZE'],
                          shuffle=True,  num_workers=CFG['NUM_WORKERS'], pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG['BATCH_SIZE'],
                          shuffle=False, num_workers=CFG['NUM_WORKERS'], pin_memory=True)

    model     = BirdModel(CFG['MODEL_NAME'], BIRD_CLASSES, CFG['PRETRAINED'])
    model     = model.to(CFG['DEVICE'])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG['LR'], weight_decay=CFG['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG['LR'],
        steps_per_epoch=len(train_dl), epochs=CFG['EPOCHS'])

    best_auc  = 0.0
    fold_hist = []
    ckpt_path = OUTPUT_DIR / f"{CFG['MODEL_NAME']}_fold{fold}.pth"

    for epoch in range(1, CFG['EPOCHS'] + 1):
        tr_loss          = train_one_epoch(model, train_dl, optimizer, scheduler,
                                           CFG['DEVICE'])
        vl_loss, vl_auc = validate(model, val_dl, CFG['DEVICE'])
        fold_hist.append({'fold': fold, 'epoch': epoch,
                          'tr_loss': tr_loss, 'vl_loss': vl_loss, 'vl_auc': vl_auc})
        elapsed = time.time() - t0
        print(f"  Ep {epoch:02d}/{CFG['EPOCHS']}  "
              f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
              f"vl_auc={vl_auc:.4f}  [{elapsed:.0f}s]")

        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(), ckpt_path)
            print(f"    OK saved  {ckpt_path.name}  (auc={best_auc:.4f})")

    # OOF predictions from best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=CFG['DEVICE']))
    model.eval()
    oof_list = []
    with torch.no_grad():
        for X, _, _ in val_dl:
            oof_list.append(torch.sigmoid(model(X.to(CFG['DEVICE']))).cpu().numpy())
    oof_preds[val_idx] = np.concatenate(oof_list, axis=0)

    fold_aucs.append(best_auc)
    history.extend(fold_hist)
    del model, train_ds, val_ds, train_dl, val_dl
    torch.cuda.empty_cache(); gc.collect()

print(f"\n{'='*60}")
print(f"  CV AUC ({CFG['MODEL_NAME']}): "
      f"{np.mean(fold_aucs):.4f} ? {np.std(fold_aucs):.4f}")
print(f"  Per-fold : {[round(a, 4) for a in fold_aucs]}")
print(f"{'='*60}")

# %%
# %% [markdown]
# ## Results & OOF Save

# %%
oof_results = {
    'model'    : CFG['MODEL_NAME'],
    'fold_aucs': [round(a, 4) for a in fold_aucs],
    'mean_auc' : round(float(np.mean(fold_aucs)), 4),
    'std_auc'  : round(float(np.std(fold_aucs)),  4),
}
fname = OUTPUT_DIR / f"oof_{CFG['MODEL_NAME']}.json"
with open(fname, 'w') as f:
    json.dump(oof_results, f, indent=2)
print(f"OOF results saved -> {fname}")
print(json.dumps(oof_results, indent=2))

# %%
# %% [markdown]
# ## Training Curves

# %%
hist_df = pd.DataFrame(history)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for fld in hist_df['fold'].unique():
    fh = hist_df[hist_df['fold'] == fld]
    axes[0].plot(fh['epoch'], fh['vl_loss'], marker='o',
                 label=f'fold {fld}', markersize=4)
    axes[1].plot(fh['epoch'], fh['vl_auc'],  marker='o',
                 label=f'fold {fld}', markersize=4)

axes[0].set_title(f'Validation BCE Loss -- {CFG["MODEL_NAME"]}', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
axes[0].legend()
axes[1].set_title(f'Validation Macro ROC-AUC -- {CFG["MODEL_NAME"]}', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC')
axes[1].legend()
plt.tight_layout()
plot_path = OUTPUT_DIR / f'training_curves_{CFG["MODEL_NAME"]}.png'
plt.savefig(plot_path, bbox_inches='tight')
plt.show()
print(f"Plot saved -> {plot_path}")

# %%
# %% [markdown]
# ## Inference on Test Soundscapes
#
# Slide 5s windows over each soundscape. This model predicts `BIRD_CLASSES`
# (162) columns; non-bird columns are filled with the prior `1/NUM_CLASSES`.
# Use **sigmoid** (not softmax) -- matches the BCE training objective.

# %%
sample_sub       = pd.read_csv(BASE_DIR / 'sample_submission.csv')
test_soundscapes = sorted((BASE_DIR / 'test_soundscapes').glob('*.ogg'))
print(f"Test soundscapes: {len(test_soundscapes)}")


def predict_soundscape(audio_path, model, cfg, device):
    """
    Slide 5s windows over a long soundscape.
    Returns dict: row_id -> probability array (shape BIRD_CLASSES).
    """
    try:
        y, _ = librosa.load(str(audio_path), sr=cfg['SR'], mono=True)
    except Exception as e:
        print(f"  Error loading {audio_path.name}: {e}")
        return {}

    stem      = audio_path.stem
    n_samples = cfg['SR'] * cfg['DURATION']
    n_chunks  = max(1, len(y) // n_samples)
    chunks, row_ids = [], []

    for i in range(n_chunks):
        start = i * n_samples
        chunk = y[start:start + n_samples]
        if len(chunk) < n_samples:
            chunk = np.pad(chunk, (0, n_samples - len(chunk)))
        S     = librosa.feature.melspectrogram(
            y=chunk.astype(np.float32),
            sr=cfg['SR'], n_fft=cfg['N_FFT'],
            hop_length=cfg['HOP_LENGTH'],
            n_mels=cfg['N_MELS'], fmin=cfg['FMIN'], fmax=cfg['FMAX'])
        S_db  = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        chunks.append(torch.from_numpy(S_norm).float().unsqueeze(0))
        row_ids.append(f"{stem}_{(i + 1) * cfg['DURATION']}")

    if not chunks:
        return {}

    batch = torch.stack(chunks).to(device)
    with torch.no_grad():
        # sigmoid (not softmax) -- consistent with BCE training
        probs = torch.sigmoid(model(batch)).cpu().numpy()

    return {rid: p for rid, p in zip(row_ids, probs)}


# Average predictions across all folds
all_fold_preds = []
for fold in range(1, CFG['N_FOLDS'] + 1):
    ckpt = OUTPUT_DIR / f"{CFG['MODEL_NAME']}_fold{fold}.pth"
    if not ckpt.exists():
        print(f"  Checkpoint not found: {ckpt.name} -- skipping")
        continue
    inf_model = BirdModel(CFG['MODEL_NAME'], BIRD_CLASSES, pretrained=False)
    inf_model.load_state_dict(torch.load(ckpt, map_location=CFG['DEVICE']))
    inf_model = inf_model.to(CFG['DEVICE'])
    inf_model.eval()

    fold_preds = {}
    for sf_path in test_soundscapes:
        preds = predict_soundscape(sf_path, inf_model, CFG, CFG['DEVICE'])
        for rid, p in preds.items():
            fold_preds.setdefault(rid, []).append(p)
    all_fold_preds.append(fold_preds)
    del inf_model
    torch.cuda.empty_cache(); gc.collect()
    print(f"  Fold {fold} inference done ({len(fold_preds)} rows)")

# Average across folds
all_preds = {}
if all_fold_preds:
    all_rids = all_fold_preds[0].keys()
    for rid in all_rids:
        arrays = [fp[rid] for fp in all_fold_preds if rid in fp]
        all_preds[rid] = np.mean(arrays, axis=0)  # (BIRD_CLASSES,)

print(f"Total prediction rows: {len(all_preds)}")

# %%
# %% [markdown]
# ## Build Submission
#
# This model only fills in the 162 bird columns.
# Non-bird columns get the uninformative prior `1/NUM_CLASSES`.

# %%
# Build full-width (234 columns) probability rows
prior = 1.0 / NUM_CLASSES
rows  = {}
for rid, bird_probs in all_preds.items():
    full = np.full(NUM_CLASSES, prior, dtype=np.float32)
    for i, label in enumerate(bird_labels):
        full[label2idx[label]] = bird_probs[i]
    rows[rid] = full

pred_df = pd.DataFrame.from_dict(rows, orient='index', columns=label_list)
pred_df.index.name = 'row_id'
pred_df = pred_df.reset_index()

sub = sample_sub[['row_id']].merge(pred_df, on='row_id', how='left')
sub[label_list] = sub[label_list].fillna(prior)

sub.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
print(f"Submission saved: {sub.shape}")
print(sub.head(2))

# %%
# %% [markdown]
# ## ONNX Export
#
# Export the fold-1 checkpoint. Inference notebook (nb05) will convert this
# to OpenVINO FP16 for the CPU budget.
# Dummy input uses `DURATION=5` -> 501 time frames.

# %%
try:
    n_frames  = 1 + (CFG['SR'] * CFG['DURATION'] // CFG['HOP_LENGTH'])  # 501
    onnx_path = OUTPUT_DIR / f"{CFG['MODEL_NAME']}.onnx"
    ckpt1     = OUTPUT_DIR / f"{CFG['MODEL_NAME']}_fold1.pth"
    if ckpt1.exists():
        onnx_model = BirdModel(CFG['MODEL_NAME'], BIRD_CLASSES, pretrained=False)
        onnx_model.load_state_dict(torch.load(ckpt1, map_location=CFG['DEVICE']))
        onnx_model = onnx_model.to(CFG['DEVICE'])
        onnx_model.eval()
        dummy = torch.randn(1, 1, CFG['N_MELS'], n_frames).to(CFG['DEVICE'])
        torch.onnx.export(
            onnx_model, dummy, str(onnx_path),
            input_names=['input'], output_names=['output'],
            opset_version=11,
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        print(f"ONNX export successful -> {onnx_path}")
        print(f"  Input shape : (batch, 1, {CFG['N_MELS']}, {n_frames})")
        print(f"  Output shape: (batch, {BIRD_CLASSES})")
    else:
        print(f"Checkpoint not found for ONNX export: {ckpt1}")
except Exception as e:
    print(f"ONNX export failed: {e}")
