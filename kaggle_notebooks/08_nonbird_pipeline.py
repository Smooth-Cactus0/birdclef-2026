# %%
# =============================================================================
# BirdCLEF 2026 -- Non-Bird Pipeline (Amphibia + Insecta + Mammalia + Reptilia)
# =============================================================================
# Classes   : 72 non-bird species
#   - Amphibia  : 35 frog species
#   - Insecta   : 3 named + 25 iNat-47158 sonotypes (son01-son25)
#   - Mammalia  : 8 (Jaguar, Howler Monkey, Capuchin, Marmoset, Titi, Horse, Cattle, Dog)
#   - Reptilia  : 1 (Caiman yacare)
#
# CRITICAL  : Insect sonotypes (47158son01-son25) have ZERO clips in train_audio
#             -> PRIMARY data = train_soundscapes_labels.csv expert-annotated segments
#             -> Soundscape labels use iNat taxon IDs (not ebird codes); mapped via taxonomy.csv
#             -> start/end columns are HH:MM:SS strings; convert to seconds for librosa offset
#
# Loss      : Focal BCE (gamma=2) -- down-weights easy negatives from overrepresented classes,
#             focuses learning on rare species (Caiman: 1 clip; some sonotypes: only in soundscapes)
# Model     : ECA-NFNet-L0 -- strong built-in regularisation (Scaled Weight Standardisation +
#             Exponential Moving Average), good for small datasets
# Folds     : GroupKFold by recorder site -- prevents geographic leakage
# Oversample: Gold soundscape segments repeated 3x per epoch (they're high-quality)
# =============================================================================

# %% [markdown]
# # Non-Bird Pipeline: Amphibia, Insecta, Mammalia, Reptilia
#
# The BirdCLEF 2026 competition includes **234 total species** -- 162 birds and 72 non-birds.
# This notebook handles the non-bird pipeline separately, because:
#
# 1. **Data scarcity**: Only ~750 non-bird clips in `train_audio` (vs 34,799 bird clips)
# 2. **Extreme imbalance**: Caiman yacare has 1 clip; some insect sonotypes have 0 clips
# 3. **Different primary data source**: Expert-annotated `train_soundscapes_labels.csv` is
#    the *only* source for insect sonotypes -- these species simply don't exist in `train_audio`
# 4. **Architecture choice**: ECA-NFNet-L0's aggressive regularisation prevents overfitting
#    on the tiny non-bird dataset
#
# ## Data Architecture
# ```
# train_audio/        -> 451 Amphibia + 199 Insecta + 99 Mammalia + 1 Reptilia clips
# train_soundscapes/  -> PAM recordings from real Pantanal recorders
# train_soundscapes_labels.csv -> expert-annotated 5s segments (iNat IDs, HH:MM:SS timestamps)
# ```

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

# -- Config --------------------------------------------------------------------
CFG = dict(
    # Audio -- identical constants across all notebooks
    SR=32000, N_FFT=1024, HOP_LENGTH=320, N_MELS=128, FMIN=40, FMAX=15000,
    # Duration: soundscape segments are already 5s; no need for 10s here
    DURATION=5, TRAIN_DURATION=5,
    # Model
    MODEL_NAME='eca_nfnet_l0',  # strong regularisation for small dataset
    PRETRAINED=True,
    # Training
    N_FOLDS=5,
    EPOCHS=25,         # more epochs -- dataset is small so each epoch is fast
    BATCH_SIZE=16,     # smaller batch for small dataset
    LR=1e-3,
    WEIGHT_DECAY=5e-2, # strong L2 -- matches ECA-NFNet-L0's regularisation philosophy
    GOLD_OVERSAMPLE=3, # repeat gold soundscape segments 3x per epoch
    FOCAL_GAMMA=2.0,   # standard focal loss gamma
    FOCAL_ALPHA=0.25,  # positive class up-weight
    # Device
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu',
    SEED=42,
    NUM_WORKERS=NUM_WORKERS,
)

print(f"Device: {CFG['DEVICE']}")
print(f"torch: {torch.__version__}, timm: {timm.__version__}")

# %%
# -- Label setup ---------------------------------------------------------------
# Full taxonomy -> all 234 species in submission column order
taxonomy   = pd.read_csv(BASE_DIR / 'taxonomy.csv')
label_list = taxonomy['primary_label'].tolist()    # 234 total (used for submission)
label2idx  = {l: i for i, l in enumerate(label_list)}
NUM_CLASSES = len(label_list)                      # 234

# Non-bird subset (72 classes for this pipeline)
nonbird_df     = taxonomy[taxonomy['class_name'] != 'Aves']
nonbird_labels = nonbird_df['primary_label'].tolist()   # 72
nonbird2idx    = {l: i for i, l in enumerate(nonbird_labels)}
NONBIRD_CLASSES = len(nonbird_labels)                   # 72

# iNat taxon ID -> primary_label (ebird code) mapping
# Needed because train_soundscapes_labels.csv uses iNat IDs, not ebird codes
inat2label = dict(zip(
    taxonomy['inat_taxon_id'].astype(str),
    taxonomy['primary_label']
))

print(f"Total classes in submission: {NUM_CLASSES}")
print(f"Non-bird classes: {NONBIRD_CLASSES}")
print(f"\nNon-bird breakdown:")
print(nonbird_df['class_name'].value_counts())

# %%
# %% [markdown]
# ## Data Loading: Two Sources Combined
#
# **Source 1** -- `train_audio/` clips from `train.csv` (limited but still useful):
# - 451 Amphibia, 199 Insecta (named species only), 99 Mammalia, 1 Reptilia
# - Weighted by quality: iNat=0.4, XC<3=0.5, XC>=3=0.8
#
# **Source 2** -- `train_soundscapes_labels.csv` expert annotations (the critical source):
# - Real Pantanal PAM recordings, expert-verified 5s segments
# - Contains insect sonotypes (47158son01-son25) that have no `train_audio` clips
# - Column format: `filename`, `start` (HH:MM:SS), `end` (HH:MM:SS), `primary_label` (iNat IDs)
# - Gold weight=1.0, oversampled 3x per epoch

# %%
# -- Sample weight helper ------------------------------------------------------
def get_sample_weight(row):
    """
    Quality-based sample weight. Matches the weighting used in nb06 (bird pipeline).
    Gold soundscape segments are handled separately (weight=1.0 set directly).
    """
    collection = str(row.get('collection', 'XC')).lower()
    if 'inat' in collection or 'inaturalist' in collection:
        return 0.4
    r = float(row.get('rating', 0.0))
    if r >= 3.0:
        return 0.8
    return 0.5


# -- Parse HH:MM:SS -> seconds -------------------------------------------------
def hms_to_seconds(t):
    """Convert 'HH:MM:SS' or 'MM:SS' string to float seconds."""
    try:
        parts = str(t).split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except Exception:
        pass
    return 0.0


# -- Source 1: train.csv non-bird clips ----------------------------------------
train_df = pd.read_csv(BASE_DIR / 'train.csv')
nonbird_clips = train_df[train_df['primary_label'].isin(nonbird_labels)].copy()
nonbird_clips['target_labels'] = nonbird_clips['primary_label']   # single label string
nonbird_clips['filepath']      = nonbird_clips['filename'].apply(
    lambda f: str(BASE_DIR / 'train_audio' / f))
nonbird_clips['weight']      = nonbird_clips.apply(get_sample_weight, axis=1)
nonbird_clips['source_type'] = 'clip'
nonbird_clips['start_sec']   = 0.0
nonbird_clips['site']        = nonbird_clips.get('recorder_id', 'clip_' + nonbird_clips['primary_label'])

print(f"Source 1 -- train_audio non-bird clips: {len(nonbird_clips)}")
print(nonbird_clips['class_name'].value_counts())

# %%
# -- Source 2: train_soundscapes_labels.csv expert segments --------------------
soundscape_labels_path = BASE_DIR / 'train_soundscapes_labels.csv'

if soundscape_labels_path.exists():
    sl_df = pd.read_csv(soundscape_labels_path)
    print("\ntrain_soundscapes_labels.csv -- raw head:")
    print(sl_df.head(3))
    print("\nColumn dtypes:")
    print(sl_df.dtypes)

    # Convert start HH:MM:SS -> float seconds (librosa offset parameter)
    sl_df['start_sec'] = sl_df['start'].apply(hms_to_seconds)

    # Map iNat taxon IDs -> ebird codes
    # primary_label column contains semicolon-separated iNat IDs e.g. "22961;23158;24321"
    def map_inat_ids_to_ebird(inat_str):
        """Convert semicolon-separated iNat IDs to list of known ebird codes."""
        codes = []
        for inat_id in str(inat_str).split(';'):
            inat_id = inat_id.strip()
            label = inat2label.get(inat_id)
            if label:
                codes.append(label)
        return codes

    sl_df['ebird_codes'] = sl_df['primary_label'].apply(map_inat_ids_to_ebird)

    # Expand: one row per soundscape segment; target_labels = comma-joined ebird codes
    # (the NonBirdDataset builds multi-hot from this)
    sl_df['target_labels'] = sl_df['ebird_codes'].apply(lambda codes: ','.join(codes))

    # Filter to rows that contain at least one non-bird species
    sl_df = sl_df[sl_df['ebird_codes'].apply(
        lambda codes: any(c in nonbird_labels for c in codes))].copy()

    sl_df['filepath']    = sl_df['filename'].apply(
        lambda f: str(BASE_DIR / 'train_soundscapes' / f))
    sl_df['weight']      = 1.0        # Gold weight
    sl_df['source_type'] = 'gold'
    # Site from filename: BC2026_Train_0039_S22_20211231_201500.ogg -> 'S22'
    sl_df['site'] = sl_df['filename'].str.extract(r'_(S\d+)_')[0].fillna('unknown')

    print(f"\nGold segments containing non-bird species: {len(sl_df)}")

    # Repeat gold segments GOLD_OVERSAMPLE times (they're high-quality -- prioritise them)
    gold_repeated = pd.concat([sl_df] * CFG['GOLD_OVERSAMPLE'], ignore_index=True)
    combined_df   = pd.concat([nonbird_clips, gold_repeated], ignore_index=True)

else:
    print("WARNING: train_soundscapes_labels.csv not found -- using clips only")
    print("  Insect sonotypes (47158son01-son25) will have 0 training samples!")
    combined_df = nonbird_clips

print(f"\nTotal combined training rows: {len(combined_df)}")
print(combined_df['source_type'].value_counts())

# %%
# %% [markdown]
# ## NonBirdDataset
#
# Key differences from BirdDataset (nb06):
# - `target` is a **multi-hot vector** of length 72 (NONBIRD_CLASSES)
# - For gold segments: uses `librosa.load(..., offset=start_sec, duration=5)` to extract
#   the exact 5s window from a long soundscape file -- avoids loading the full recording
# - `target_labels` column stores comma-separated ebird codes for multi-hot encoding

# %%
class NonBirdDataset(Dataset):
    """
    Audio dataset for non-bird species classification.

    Handles two data sources:
    - 'clip': standard OGG files from train_audio/ (start_sec=0)
    - 'gold': 5s segments from long soundscape files, loaded via librosa offset

    Returns:
        spec  : (1, N_MELS, T) log-mel spectrogram tensor
        target: (NONBIRD_CLASSES,) multi-hot float32 vector
        weight: float scalar for BCE sample weighting
    """

    def __init__(self, df, cfg, augment=False):
        self.df       = df.reset_index(drop=True)
        self.cfg      = cfg
        self.augment  = augment
        self.n_samples = cfg['SR'] * cfg['DURATION']   # always 5s for non-bird pipeline

    def __len__(self):
        return len(self.df)

    def _load_audio(self, row):
        path      = str(row['filepath'])
        start_sec = float(row.get('start_sec', 0.0))
        try:
            y, _ = librosa.load(
                path, sr=self.cfg['SR'], mono=True,
                offset=start_sec, duration=self.cfg['DURATION'])
        except Exception:
            return np.zeros(self.n_samples, dtype=np.float32)
        # Pad if audio shorter than n_samples (e.g., end of file)
        if len(y) < self.n_samples:
            y = np.pad(y, (0, self.n_samples - len(y)), mode='constant')
        else:
            y = y[:self.n_samples]
        return y.astype(np.float32)

    def _to_melspec(self, y):
        S = librosa.feature.melspectrogram(
            y=y, sr=self.cfg['SR'],
            n_fft=self.cfg['N_FFT'], hop_length=self.cfg['HOP_LENGTH'],
            n_mels=self.cfg['N_MELS'], fmin=self.cfg['FMIN'], fmax=self.cfg['FMAX'])
        S_db   = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        return S_norm.astype(np.float32)

    def _spec_augment(self, S):
        """SpecAugment: frequency + time masking."""
        S  = S.copy()
        F  = np.random.randint(0, 16)
        f0 = np.random.randint(0, self.cfg['N_MELS'] - F)
        S[f0:f0 + F, :] = 0.0
        T  = np.random.randint(0, 30)
        t0 = np.random.randint(0, S.shape[1] - T)
        S[:, t0:t0 + T] = 0.0
        return S

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y   = self._load_audio(row)
        S   = self._to_melspec(y)
        if self.augment:
            S = self._spec_augment(S)

        # Multi-hot target -- handles both single-label clips and multi-label gold segments
        target = np.zeros(NONBIRD_CLASSES, dtype=np.float32)
        labels_str = str(row.get('target_labels', ''))
        for label in labels_str.split(','):
            label = label.strip()
            if label in nonbird2idx:
                target[nonbird2idx[label]] = 1.0

        weight = float(row.get('weight', 1.0))
        return (torch.from_numpy(S).unsqueeze(0),
                torch.from_numpy(target),
                torch.tensor(weight, dtype=torch.float32))


# %%
# %% [markdown]
# ## Model Architecture
#
# **ECA-NFNet-L0** is chosen for three reasons:
# 1. **Normalisation-Free (NF)**: no BatchNorm means batch statistics don't explode on tiny
#    batches (we use batch_size=16 here)
# 2. **Scaled Weight Standardisation**: built-in gradient clipping substitute; more stable
#    training on small datasets
# 3. **ECA attention**: Efficient Channel Attention adds feature recalibration with near-zero
#    parameter overhead -- helps distinguish acoustically similar frog species

# %%
class BirdModel(nn.Module):
    """
    Generic timm backbone adapted for single-channel mel spectrograms.
    Same class used in nb06 (bird pipeline) -- backbone name drives the architecture.
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
# ## Loss Function: Focal BCE
#
# Standard BCE treats every negative sample equally.
# **Focal BCE** introduces a modulating factor `(1 - p_t)^gamma`:
# - When `p_t -> 1` (easy correct prediction): factor -> 0, loss down-weighted
# - When `p_t -> 0` (hard wrong prediction): factor -> 1, loss at full weight
#
# For `gamma=2`, easy examples contribute ~100x less loss than hard examples.
# This is essential here because most "non-Caiman" samples are trivially easy
# negatives for Caiman, and without focal loss, those easy negatives dominate.

# %%
class FocalBCELoss(nn.Module):
    """
    Focal Binary Cross Entropy for multi-label imbalanced classification.

    Args:
        gamma: focusing parameter (default=2.0, standard value from Lin et al. 2017)
        alpha: positive class weight (default=0.25, down-weights easy negatives further)
        reduction: 'mean' | 'none'
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        p   = torch.sigmoid(logits)
        ce  = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # p_t: probability of the correct class (p for positives, 1-p for negatives)
        p_t     = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss    = alpha_t * (1 - p_t) ** self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss


# %%
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Training loop with per-sample quality weighting.
    Gold segments (weight=1.0) contribute more than low-rated XC clips (weight=0.5).
    """
    model.train()
    total_loss = 0.0
    for X, y, w in loader:
        X, y, w = X.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model(X)
        # Per-class loss -> mean over classes -> weighted mean over batch
        loss_per_sample = criterion(logits, y)   # scalar (mean reduction in FocalBCELoss)
        # Re-weight by sample quality; note: criterion returns mean, so re-apply weights
        loss = loss_per_sample * w.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    """
    Validation: compute macro ROC-AUC across all 72 non-bird classes.
    Uses sigmoid (not softmax) -- this is multi-label, not multi-class.
    """
    model.eval()
    criterion  = FocalBCELoss(gamma=CFG['FOCAL_GAMMA'], alpha=CFG['FOCAL_ALPHA'])
    total_loss = 0.0
    all_probs, all_targets = [], []
    for X, y, _ in loader:
        X, y   = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.cpu().numpy())
    all_probs   = np.concatenate(all_probs,   axis=0)  # (N, 72)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 72)
    try:
        # macro AUC: skip classes with no positives in validation fold
        aucs = []
        for c in range(NONBIRD_CLASSES):
            if all_targets[:, c].sum() > 0:
                aucs.append(roc_auc_score(all_targets[:, c], all_probs[:, c]))
        auc = float(np.mean(aucs)) if aucs else 0.0
    except ValueError:
        auc = 0.0
    return total_loss / len(loader.dataset), auc


# %%
# %% [markdown]
# ## Cross-Validation Loop
#
# **GroupKFold** by recorder site prevents the model from memorising
# site-specific ambient noise instead of species vocalisations.
# The site is extracted from soundscape filenames (e.g. `S22`).

# %%
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])

criterion = FocalBCELoss(gamma=CFG['FOCAL_GAMMA'], alpha=CFG['FOCAL_ALPHA'])
gkf       = GroupKFold(n_splits=CFG['N_FOLDS'])
groups    = combined_df['site'].values

oof_preds = np.zeros((len(combined_df), NONBIRD_CLASSES), dtype=np.float32)
fold_aucs = []
history   = []

for fold, (train_idx, val_idx) in enumerate(
        gkf.split(combined_df, groups=groups), start=1):
    print(f"\n{'='*60}\n  Fold {fold}/{CFG['N_FOLDS']}\n{'='*60}")
    t0 = time.time()

    train_fold = combined_df.iloc[train_idx]
    val_fold   = combined_df.iloc[val_idx]

    train_ds = NonBirdDataset(train_fold, CFG, augment=True)
    val_ds   = NonBirdDataset(val_fold,   CFG, augment=False)
    train_dl = DataLoader(train_ds, batch_size=CFG['BATCH_SIZE'],
                          shuffle=True,  num_workers=CFG['NUM_WORKERS'], pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG['BATCH_SIZE'],
                          shuffle=False, num_workers=CFG['NUM_WORKERS'], pin_memory=True)

    model     = BirdModel(CFG['MODEL_NAME'], NONBIRD_CLASSES, CFG['PRETRAINED'])
    model     = model.to(CFG['DEVICE'])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG['LR'], weight_decay=CFG['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG['LR'],
        steps_per_epoch=len(train_dl), epochs=CFG['EPOCHS'])

    best_auc  = 0.0
    fold_hist = []

    for epoch in range(1, CFG['EPOCHS'] + 1):
        tr_loss          = train_one_epoch(model, train_dl, optimizer, scheduler,
                                           criterion, CFG['DEVICE'])
        vl_loss, vl_auc = validate(model, val_dl, CFG['DEVICE'])
        fold_hist.append({'fold': fold, 'epoch': epoch,
                          'tr_loss': tr_loss, 'vl_loss': vl_loss, 'vl_auc': vl_auc})
        elapsed = time.time() - t0
        print(f"  Ep {epoch:02d}/{CFG['EPOCHS']}  "
              f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
              f"vl_auc={vl_auc:.4f}  [{elapsed:.0f}s]")

        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(),
                       OUTPUT_DIR / f'nonbird_fold{fold}.pth')
            print(f"    OK saved (auc={best_auc:.4f})")

    # OOF predictions from best checkpoint
    model.load_state_dict(torch.load(OUTPUT_DIR / f'nonbird_fold{fold}.pth',
                                      map_location=CFG['DEVICE']))
    model.eval()
    probs_list = []
    with torch.no_grad():
        for X, _, _ in val_dl:
            probs_list.append(
                torch.sigmoid(model(X.to(CFG['DEVICE']))).cpu().numpy())
    oof_preds[val_idx] = np.concatenate(probs_list, axis=0)

    fold_aucs.append(best_auc)
    history.extend(fold_hist)
    del model, train_ds, val_ds, train_dl, val_dl
    torch.cuda.empty_cache(); gc.collect()

print(f"\n{'='*60}")
print(f"  Non-Bird CV AUC: {np.mean(fold_aucs):.4f} ? {np.std(fold_aucs):.4f}")
print(f"  Per-fold: {[round(a,4) for a in fold_aucs]}")
print(f"{'='*60}")

# %%
# -- Training curves -----------------------------------------------------------
hist_df = pd.DataFrame(history)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for fold_id in hist_df['fold'].unique():
    fh = hist_df[hist_df['fold'] == fold_id]
    axes[0].plot(fh['epoch'], fh['vl_loss'], marker='o', label=f'fold {fold_id}', markersize=4)
    axes[1].plot(fh['epoch'], fh['vl_auc'],  marker='o', label=f'fold {fold_id}', markersize=4)

axes[0].set_title('Non-Bird Validation Loss per Fold (Focal BCE)', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Focal BCE Loss'); axes[0].legend()
axes[1].set_title('Non-Bird Validation Macro ROC-AUC per Fold', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC'); axes[1].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'nonbird_training_curves.png', bbox_inches='tight')
plt.show()

# %%
# %% [markdown]
# ## Inference: Non-Bird Predictions on Test Soundscapes
#
# We load the best checkpoint from fold 1, slide a 5s window over each test soundscape,
# and produce per-chunk probabilities for all 72 non-bird classes.
#
# For the final submission, we:
# - Fill the **72 non-bird columns** with this model's predictions (sigmoid probabilities)
# - Fill the **162 bird columns** with uniform `1/234` (handled by the bird pipeline in nb06)
# - The ensemble notebook (nb11) combines both pipelines into the final submission

# %%
# Load best checkpoint for inference (fold 1 by default)
inf_model = BirdModel(CFG['MODEL_NAME'], NONBIRD_CLASSES, pretrained=False)
inf_model.load_state_dict(torch.load(OUTPUT_DIR / 'nonbird_fold1.pth',
                                      map_location=CFG['DEVICE']))
inf_model = inf_model.to(CFG['DEVICE']).eval()

sample_sub      = pd.read_csv(BASE_DIR / 'sample_submission.csv')
test_soundscapes = sorted((BASE_DIR / 'test_soundscapes').glob('*.ogg'))
print(f"Test soundscapes found: {len(test_soundscapes)}")


def predict_soundscape_nonbird(audio_path, model, cfg, device):
    """
    Slide 5s windows over a soundscape, return dict of row_id -> NONBIRD_CLASSES probs.
    row_id format: {stem}_{end_sec}
    """
    try:
        y, _ = librosa.load(str(audio_path), sr=cfg['SR'], mono=True)
    except Exception as e:
        print(f"  Error: {audio_path.name}: {e}")
        return {}
    stem      = audio_path.stem
    n_samples = cfg['SR'] * cfg['DURATION']
    n_chunks  = max(1, len(y) // n_samples)
    results   = {}
    chunks, row_ids = [], []
    for i in range(n_chunks):
        start = i * n_samples
        chunk = y[start:start + n_samples]
        if len(chunk) < n_samples:
            chunk = np.pad(chunk, (0, n_samples - len(chunk)))
        S = librosa.feature.melspectrogram(
            y=chunk.astype(np.float32), sr=cfg['SR'],
            n_fft=cfg['N_FFT'], hop_length=cfg['HOP_LENGTH'],
            n_mels=cfg['N_MELS'], fmin=cfg['FMIN'], fmax=cfg['FMAX'])
        S_db   = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        chunks.append(torch.from_numpy(S_norm).float().unsqueeze(0))
        row_ids.append(f"{stem}_{(i+1)*cfg['DURATION']}")
    if not chunks:
        return {}
    batch = torch.stack(chunks).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(inf_model(batch)).cpu().numpy()  # (N, 72)
    for rid, p in zip(row_ids, probs):
        results[rid] = p
    return results


all_preds = {}
for sc_path in test_soundscapes:
    preds = predict_soundscape_nonbird(sc_path, inf_model, CFG, CFG['DEVICE'])
    all_preds.update(preds)
    print(f"  {sc_path.name}: {len(preds)} chunks")

print(f"\nTotal prediction rows: {len(all_preds)}")

# %%
# %% [markdown]
# ## Build Submission
#
# Column layout for the 234-species submission:
# - Non-bird columns (72): filled with ECA-NFNet-L0 sigmoid probabilities
# - Bird columns (162): uniform `1/NUM_CLASSES` as placeholder
#   (the bird pipeline in nb06 will overwrite these; see nb11 for the merge)

# %%
# Build non-bird prediction DataFrame
nonbird_pred_df = pd.DataFrame.from_dict(
    all_preds, orient='index', columns=nonbird_labels)
nonbird_pred_df.index.name = 'row_id'
nonbird_pred_df = nonbird_pred_df.reset_index()

# Start from sample_submission -> merge non-bird columns
sub = sample_sub[['row_id']].copy()
sub = sub.merge(nonbird_pred_df, on='row_id', how='left')

# Fill missing row_ids (e.g., short soundscapes) with uniform probability
sub[nonbird_labels] = sub[nonbird_labels].fillna(1.0 / NUM_CLASSES)

# Remaining bird columns: uniform placeholder
bird_labels = taxonomy[taxonomy['class_name'] == 'Aves']['primary_label'].tolist()
for col in bird_labels:
    if col not in sub.columns:
        sub[col] = 1.0 / NUM_CLASSES

# Reorder columns to match sample_submission
sub = sample_sub[['row_id']].merge(
    sub, on='row_id', how='left')[['row_id'] + label_list]
sub[label_list] = sub[label_list].fillna(1.0 / NUM_CLASSES)

sub.to_csv(OUTPUT_DIR / 'submission_nonbird.csv', index=False)
print(f"Submission saved: {sub.shape}")
print(sub.head(2))

# %%
# %% [markdown]
# ## ONNX Export
#
# ECA-NFNet-L0 needs an ONNX export for the OpenVINO CPU inference pipeline.
# The non-bird model is lightweight enough to run in parallel with the bird model
# during test soundscape inference.

# %%
try:
    # n_frames for a 5s clip at SR=32000, hop_length=320
    n_frames = 1 + (CFG['SR'] * CFG['DURATION'] // CFG['HOP_LENGTH'])  # 501
    dummy    = torch.randn(1, 1, CFG['N_MELS'], n_frames).to(CFG['DEVICE'])
    onnx_path = str(OUTPUT_DIR / 'nonbird_eca_nfnet_l0.onnx')
    torch.onnx.export(
        inf_model, dummy, onnx_path,
        input_names=['input'], output_names=['output'],
        opset_version=11,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    print(f"ONNX export successful -> {onnx_path}")
except Exception as e:
    print(f"ONNX export failed: {e}")

# %%
# -- Save experiment results ---------------------------------------------------
oof_results = {
    'model':          CFG['MODEL_NAME'],
    'pipeline':       'non_bird',
    'num_classes':    NONBIRD_CLASSES,
    'n_folds':        CFG['N_FOLDS'],
    'epochs':         CFG['EPOCHS'],
    'gold_oversample': CFG['GOLD_OVERSAMPLE'],
    'focal_gamma':    CFG['FOCAL_GAMMA'],
    'fold_aucs':      [round(a, 4) for a in fold_aucs],
    'mean_auc':       round(float(np.mean(fold_aucs)), 4),
    'std_auc':        round(float(np.std(fold_aucs)),  4),
}
with open(OUTPUT_DIR / 'nonbird_oof_scores.json', 'w') as f:
    json.dump(oof_results, f, indent=2)
print("Non-bird OOF results:", json.dumps(oof_results, indent=2))
