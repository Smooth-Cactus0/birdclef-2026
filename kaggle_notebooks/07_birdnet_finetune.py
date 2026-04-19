# %%
# =============================================================================
# BirdCLEF 2026 -- BirdNET Fine-tuning (Bird Pipeline)
# =============================================================================
# Backbone   : EfficientNet-B1 with BirdNET pretrained weights
# Pretraining: BirdNET-Analyzer trained on 9,000+ bird species from XC + iNat
# Why BirdNET : pretrained representations drastically help rare species (< 10 clips)
# Fine-tuning : 2-phase -- frozen backbone (5 ep) -> full unfrozen (15 ep)
# Loss        : BCEWithLogitsLoss (multi-label)
# Folds       : GroupKFold by recorder site
# =============================================================================
#
# To add BirdNET weights on Kaggle:
#   1. Search Kaggle Datasets for 'birdnet-analyzer'
#   2. Add to this notebook (settings -> Add data)
#   3. Weights will be at /kaggle/input/birdnet-analyzer-model/

# %% [markdown]
# # BirdCLEF 2026 -- BirdNET Fine-tuning (Bird Pipeline)
#
# BirdNET-Analyzer is a public EfficientNet-B1 pretrained specifically on
# bird vocalisations (9,000+ species from Xeno-canto + iNaturalist). Its
# feature representations are far stronger for bird audio than ImageNet
# pretrained weights -- especially for rare Pantanal species with < 10 clips.
#
# **2-phase strategy:**
# 1. **Frozen backbone (5 epochs)** -- train only the new 162-class head so it
#    converges before we touch the pretrained features
# 2. **Full model (15 epochs)** -- unfreeze everything with a 10x lower LR
#    so the backbone adapts slowly and retains its strong bird representations

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
BASE_DIR      = (Path('/kaggle/input/competitions/birdclef-2026')
                 if Path('/kaggle/input/competitions/birdclef-2026').exists()
                 else Path('birdclef-2026'))
BIRDNET_DIR   = Path('/kaggle/input/birdnet-analyzer-model')
OUTPUT_DIR    = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
NUM_WORKERS   = 0 if os.name == 'nt' else 4

# %%
# -- Config --------------------------------------------------------------------
CFG = dict(
    # Audio constants -- identical across all notebooks
    SR             = 32000,
    N_FFT          = 1024,
    HOP_LENGTH     = 320,
    N_MELS         = 128,
    FMIN           = 40,
    FMAX           = 15000,
    DURATION       = 5,         # inference clip length (seconds)
    TRAIN_DURATION = 10,        # training clip -- longer context
    # Model
    MODEL_NAME     = 'efficientnet_b1',
    PRETRAINED     = False,     # we load BirdNET weights, not ImageNet
    # Training
    FREEZE_EPOCHS  = 5,         # train only head for first N epochs
    EPOCHS         = 20,        # total epochs (5 frozen + 15 unfrozen)
    N_FOLDS        = 5,
    BATCH_SIZE     = 32,
    LR_HEAD        = 1e-3,      # LR during frozen-backbone phase
    LR_FULL        = 1e-4,      # LR during full-model phase (10x lower)
    WEIGHT_DECAY   = 1e-2,
    # Device
    DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu',
    SEED           = 42,
    NUM_WORKERS    = NUM_WORKERS,
)
print(f"Device       : {CFG['DEVICE']}")
print(f"torch        : {torch.__version__},  timm: {timm.__version__}")
print(f"Backbone     : {CFG['MODEL_NAME']}")
print(f"BirdNET dir  : {BIRDNET_DIR}  (exists={BIRDNET_DIR.exists()})")

# %%
# %% [markdown]
# ## Data Setup
#
# Identical to nb06: Aves-only (162 classes), quality-based sample weights,
# GroupKFold by recorder site.

# %%
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])

taxonomy  = pd.read_csv(BASE_DIR / 'taxonomy.csv')
train_df  = pd.read_csv(BASE_DIR / 'train.csv')

# Full 234-class label list for submission alignment
label_list  = taxonomy['primary_label'].tolist()
label2idx   = {l: i for i, l in enumerate(label_list)}
NUM_CLASSES = len(label_list)   # 234

# Bird-only subset (162 Aves)
bird_labels  = taxonomy[taxonomy['class_name'] == 'Aves']['primary_label'].tolist()
bird2idx     = {l: i for i, l in enumerate(bird_labels)}
BIRD_CLASSES = len(bird_labels)   # 162

print(f"Total classes : {NUM_CLASSES}")
print(f"Bird classes  : {BIRD_CLASSES}")

# Filter to Aves only
train_df = train_df[train_df['primary_label'].isin(bird_labels)].copy()
train_df['target']   = train_df['primary_label'].map(bird2idx)
train_df['filepath'] = train_df['filename'].apply(
    lambda f: str(BASE_DIR / 'train_audio' / f))
print(f"Training clips (Aves): {len(train_df):,}")

# Sample weights by data quality
def get_sample_weight(row):
    collection = str(row.get('collection', 'XC')).lower()
    if 'inat' in collection or 'inaturalist' in collection:
        return 0.4
    r = float(row.get('rating', 0.0)) if pd.notna(row.get('rating', 0.0)) else 0.0
    if r >= 3.0:
        return 0.8
    return 0.5

train_df['weight'] = train_df.apply(get_sample_weight, axis=1)

# GroupKFold site grouping
train_df['site'] = train_df.get('recorder_id', train_df['primary_label'])
print(f"Unique groups: {train_df['site'].nunique()}")

# %%
# %% [markdown]
# ## BirdNET Weight Loading
#
# BirdNET-Analyzer uses EfficientNet-B1 pretrained on 9,000+ species. We load
# only the feature extractor (all layers except the final classifier head, which
# classified 6,000+ BirdNET classes). The head is reinitialized for our 162
# Pantanal species.

# %%
def load_birdnet_weights(model, birdnet_dir):
    """
    Load BirdNET pretrained weights into a timm EfficientNet-B1 backbone.
    Strips the original BirdNET classifier head (6,000+ classes).
    Returns True if successful, False on fallback to random initialization.
    """
    weight_path = Path(birdnet_dir) / 'BirdNET_GLOBAL_6K_V2.4_Model.pt'
    if not weight_path.exists():
        print(f"  BirdNET weights not found at {weight_path}")
        print("  -> Add the 'birdnet-analyzer' dataset to this Kaggle notebook")
        print("  -> Falling back to random init (model will still train, just less effectively)")
        return False
    state_dict = torch.load(weight_path, map_location='cpu')
    # Drop classifier / head / fc layers -- keep only the feature extractor
    filtered = {k: v for k, v in state_dict.items()
                if not any(k.startswith(p) for p in ['classifier', 'head', 'fc'])}
    missing, unexpected = model.backbone.load_state_dict(filtered, strict=False)
    print(f"  BirdNET weights loaded: {len(filtered)} layers transferred")
    print(f"  Missing keys (new head) : {len(missing)}")
    print(f"  Unexpected keys         : {len(unexpected)}")
    return True

# %%
# %% [markdown]
# ## Two-Phase Fine-Tuning Helper
#
# `set_backbone_frozen` freezes or unfreezes the backbone while always keeping
# the classification head trainable. We count and print trainable params to
# confirm the phase switch worked correctly.

# %%
def set_backbone_frozen(model, frozen: bool):
    """Freeze or unfreeze backbone. Classification head is always trainable."""
    for name, param in model.backbone.named_parameters():
        if 'classifier' in name or 'head' in name:
            param.requires_grad = True   # always trainable
        else:
            param.requires_grad = not frozen
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    state       = 'frozen backbone' if frozen else 'full model'
    print(f"  [{state}] trainable: {n_trainable:,} / {n_total:,} params")

# %%
# %% [markdown]
# ## Dataset

# %%
class BirdDataset(Dataset):
    """
    Loads OGG audio, converts to log-mel spectrogram.
    - Training  : 10s random crop
    - Validation: 5s centre crop (matches inference)
    - Target    : multi-hot vector (primary=1.0, secondary=0.5)
    """
    def __init__(self, df, cfg, augment=False):
        self.df        = df.reset_index(drop=True)
        self.cfg       = cfg
        self.augment   = augment
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
        S = S.copy()
        F  = np.random.randint(0, 16); f0 = np.random.randint(0, self.cfg['N_MELS'] - F)
        S[f0:f0 + F, :] = 0.0
        T  = np.random.randint(0, 30);  t0 = np.random.randint(0, S.shape[1] - T)
        S[:, t0:t0 + T] = 0.0
        return S

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        y      = self._load_audio(row['filepath'])
        S      = self._to_melspec(y)
        if self.augment:
            S = self._spec_augment(S)
        target = np.zeros(BIRD_CLASSES, dtype=np.float32)
        t = int(row['target'])
        if 0 <= t < BIRD_CLASSES:
            target[t] = 1.0
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

# %%
class BirdModel(nn.Module):
    """timm EfficientNet-B1, in_chans=1, num_classes=BIRD_CLASSES."""
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

# %%
def train_one_epoch(model, loader, optimizer, device, scheduler=None):
    """Weighted BCE with optional LR scheduler (step after each batch)."""
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
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    """BCE loss + macro ROC-AUC for multi-label targets."""
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
    probs       = np.concatenate(all_probs)
    targets     = np.concatenate(all_targets)
    bin_targets = (targets >= 0.5).astype(int)
    valid_cols  = bin_targets.sum(axis=0) > 0
    try:
        auc = roc_auc_score(bin_targets[:, valid_cols],
                            probs[:, valid_cols], average='macro')
    except ValueError:
        auc = 0.0
    return total_loss / len(loader.dataset), auc

# %%
# %% [markdown]
# ## Cross-Validation Loop (2-Phase Fine-Tuning)
#
# Each fold runs two sequential phases:
# 1. **Phase 1 (frozen)**: only the 162-class head trains for `FREEZE_EPOCHS`
# 2. **Phase 2 (full)**: entire model trains with CosineAnnealingLR for
#    `EPOCHS - FREEZE_EPOCHS` remaining epochs
#
# Best checkpoint is saved across all epochs (either phase).

# %%
gkf       = GroupKFold(n_splits=CFG['N_FOLDS'])
groups    = train_df['site'].values
oof_preds = np.zeros((len(train_df), BIRD_CLASSES), dtype=np.float32)
fold_aucs = []
history   = []

for fold, (train_idx, val_idx) in enumerate(
        gkf.split(train_df, train_df['target'], groups=groups), start=1):

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

    # Build model and load BirdNET weights
    model = BirdModel(CFG['MODEL_NAME'], BIRD_CLASSES, pretrained=CFG['PRETRAINED'])
    birdnet_loaded = load_birdnet_weights(model, BIRDNET_DIR)
    if not birdnet_loaded:
        # Fallback: reload with ImageNet pretrain
        model = BirdModel(CFG['MODEL_NAME'], BIRD_CLASSES, pretrained=True)
        print("  Using ImageNet pretrain as fallback")
    model = model.to(CFG['DEVICE'])

    best_auc  = 0.0
    fold_hist = []
    ckpt_path = OUTPUT_DIR / f"birdnet_fold{fold}.pth"

    # -- Phase 1: frozen backbone ----------------------------------------------
    print(f"\n  Phase 1 -- frozen backbone ({CFG['FREEZE_EPOCHS']} epochs)")
    set_backbone_frozen(model, frozen=True)
    optimizer_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG['LR_HEAD'], weight_decay=CFG['WEIGHT_DECAY'])

    for epoch in range(1, CFG['FREEZE_EPOCHS'] + 1):
        tr_loss          = train_one_epoch(model, train_dl, optimizer_p1,
                                           CFG['DEVICE'], scheduler=None)
        vl_loss, vl_auc = validate(model, val_dl, CFG['DEVICE'])
        elapsed = time.time() - t0
        fold_hist.append({'fold': fold, 'epoch': epoch, 'phase': 1,
                          'tr_loss': tr_loss, 'vl_loss': vl_loss, 'vl_auc': vl_auc})
        print(f"  [P1] Ep {epoch:02d}/{CFG['FREEZE_EPOCHS']}  "
              f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
              f"vl_auc={vl_auc:.4f}  [{elapsed:.0f}s]")
        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(), ckpt_path)
            print(f"    OK saved (auc={best_auc:.4f})")

    # -- Phase 2: full model ---------------------------------------------------
    remaining = CFG['EPOCHS'] - CFG['FREEZE_EPOCHS']
    print(f"\n  Phase 2 -- full model ({remaining} epochs)")
    set_backbone_frozen(model, frozen=False)
    optimizer_p2 = torch.optim.AdamW(
        model.parameters(),
        lr=CFG['LR_FULL'], weight_decay=CFG['WEIGHT_DECAY'])
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=remaining)

    for epoch in range(CFG['FREEZE_EPOCHS'] + 1, CFG['EPOCHS'] + 1):
        tr_loss          = train_one_epoch(model, train_dl, optimizer_p2,
                                           CFG['DEVICE'], scheduler=None)
        scheduler_p2.step()   # step per epoch (CosineAnnealingLR is epoch-level)
        vl_loss, vl_auc = validate(model, val_dl, CFG['DEVICE'])
        elapsed = time.time() - t0
        fold_hist.append({'fold': fold, 'epoch': epoch, 'phase': 2,
                          'tr_loss': tr_loss, 'vl_loss': vl_loss, 'vl_auc': vl_auc})
        print(f"  [P2] Ep {epoch:02d}/{CFG['EPOCHS']}  "
              f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
              f"vl_auc={vl_auc:.4f}  [{elapsed:.0f}s]")
        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(), ckpt_path)
            print(f"    OK saved (auc={best_auc:.4f})")

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
print(f"  CV AUC (birdnet): {np.mean(fold_aucs):.4f} ? {np.std(fold_aucs):.4f}")
print(f"  Per-fold: {[round(a, 4) for a in fold_aucs]}")
print(f"{'='*60}")

# %%
# %% [markdown]
# ## Results & OOF Save

# %%
oof_results = {
    'model'    : f"birdnet_{CFG['MODEL_NAME']}",
    'fold_aucs': [round(a, 4) for a in fold_aucs],
    'mean_auc' : round(float(np.mean(fold_aucs)), 4),
    'std_auc'  : round(float(np.std(fold_aucs)),  4),
}
fname = OUTPUT_DIR / 'oof_birdnet.json'
with open(fname, 'w') as f:
    json.dump(oof_results, f, indent=2)
print(f"OOF results saved -> {fname}")
print(json.dumps(oof_results, indent=2))

# %%
# %% [markdown]
# ## Training Curves
#
# The dashed vertical line separates Phase 1 (frozen backbone) from Phase 2
# (full model). AUC typically improves more in Phase 2 as the backbone adapts
# its BirdNET features to the Pantanal species distribution.

# %%
hist_df = pd.DataFrame(history)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for fld in hist_df['fold'].unique():
    fh = hist_df[hist_df['fold'] == fld]
    axes[0].plot(fh['epoch'], fh['vl_loss'], marker='o',
                 label=f'fold {fld}', markersize=4)
    axes[1].plot(fh['epoch'], fh['vl_auc'],  marker='o',
                 label=f'fold {fld}', markersize=4)

for ax in axes:
    ax.axvline(x=CFG['FREEZE_EPOCHS'] + 0.5, color='grey',
               linestyle='--', alpha=0.7, label='phase boundary')

axes[0].set_title('Validation BCE Loss -- BirdNET fine-tune', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
axes[0].legend(fontsize=8)
axes[1].set_title('Validation Macro ROC-AUC -- BirdNET fine-tune', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC')
axes[1].legend(fontsize=8)
plt.tight_layout()
plot_path = OUTPUT_DIR / 'training_curves_birdnet.png'
plt.savefig(plot_path, bbox_inches='tight')
plt.show()
print(f"Plot saved -> {plot_path}")

# %%
# %% [markdown]
# ## Inference on Test Soundscapes
#
# Identical to nb06: 5s sliding window, sigmoid outputs, average across folds.
# Bird columns get model predictions; non-bird columns filled with `1/NUM_CLASSES`.

# %%
sample_sub       = pd.read_csv(BASE_DIR / 'sample_submission.csv')
test_soundscapes = sorted((BASE_DIR / 'test_soundscapes').glob('*.ogg'))
print(f"Test soundscapes: {len(test_soundscapes)}")


def predict_soundscape(audio_path, model, cfg, device):
    """Slide 5s windows, return dict row_id -> bird probability array."""
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
            y=chunk.astype(np.float32), sr=cfg['SR'],
            n_fft=cfg['N_FFT'], hop_length=cfg['HOP_LENGTH'],
            n_mels=cfg['N_MELS'], fmin=cfg['FMIN'], fmax=cfg['FMAX'])
        S_db  = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        chunks.append(torch.from_numpy(S_norm).float().unsqueeze(0))
        row_ids.append(f"{stem}_{(i + 1) * cfg['DURATION']}")
    if not chunks:
        return {}
    batch = torch.stack(chunks).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).cpu().numpy()
    return {rid: p for rid, p in zip(row_ids, probs)}


# Average predictions across folds
all_fold_preds = []
for fold in range(1, CFG['N_FOLDS'] + 1):
    ckpt = OUTPUT_DIR / f"birdnet_fold{fold}.pth"
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

# Merge folds
all_preds = {}
if all_fold_preds:
    for rid in all_fold_preds[0].keys():
        arrays = [fp[rid] for fp in all_fold_preds if rid in fp]
        all_preds[rid] = np.mean(arrays, axis=0)
print(f"Total prediction rows: {len(all_preds)}")

# %%
# %% [markdown]
# ## Build Submission

# %%
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
# Export fold-1 checkpoint. 5s inference -> 501 time frames.

# %%
try:
    n_frames   = 1 + (CFG['SR'] * CFG['DURATION'] // CFG['HOP_LENGTH'])  # 501
    onnx_path  = OUTPUT_DIR / 'birdnet.onnx'
    ckpt1      = OUTPUT_DIR / 'birdnet_fold1.pth'
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
