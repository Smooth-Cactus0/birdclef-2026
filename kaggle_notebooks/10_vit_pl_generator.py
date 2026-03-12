# %%
# =============================================================================
# BirdCLEF 2026 — ViT Pseudo-Label Generator
# =============================================================================
# Model     : EVA-02 Large 448px OR DINOv2-Large-reg4 (select via CFG)
# Purpose   : HIGH-QUALITY pseudo-label generation for unlabeled soundscapes
# NOT used for competition inference (too slow: ~8-15s/chunk on CPU)
# Competition inference uses EfficientNet + OpenVINO (see nb05/nb06)
# Training  : Single fold (fold 1 only) — we need diversity, not 5-fold averaging
# Output    : vit_fold1.pth (used as voter #3 in nb09 pseudo-labeling)
#
# ARCHITECTURE NOTES:
#   EVA-02 Large  : 304M params, 448px, MIM pretraining → strongest bird features
#   DINOv2-L-reg4 : 307M params, self-supervised, 4 register tokens for clean attention
#   Both expect 3-channel RGB input at their native resolution — NOT in_chans=1
#   Mel spectrograms are resized to (IMG_SIZE, IMG_SIZE) and repeated 3× before norm
#
# RUNTIME:
#   EVA-02 Large  : ~7h for 20 epochs on Kaggle P100
#   DINOv2-Large  : ~5h for 15 epochs on Kaggle P100
#   Budget: run in an isolated Kaggle session (not the competition inference kernel)
# =============================================================================

# %% [markdown]
# # ViT Pseudo-Label Generator
# Fine-tunes a large Vision Transformer (EVA-02 or DINOv2) on BirdCLEF 2026 bird audio.
# Output is a single fold-1 checkpoint used as voter #3 in `09_pseudo_labeling.py`.
# The ViT's broader receptive field and self-supervised pretraining give it
# complementary error patterns to EfficientNet — which is exactly why ensemble
# consensus is so powerful.

# %%
# ── Install / version pins ────────────────────────────────────────────────────
# !pip install -q timm==1.0.3 torchvision  # uncomment on Kaggle if needed

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
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import timm
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = (Path('/kaggle/input/birdclef-2026')
              if Path('/kaggle/input/birdclef-2026').exists()
              else Path('birdclef-2026'))
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
NUM_WORKERS = 0 if os.name == 'nt' else 4

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    # Audio
    SR=32000, N_FFT=1024, HOP_LENGTH=320, N_MELS=128, FMIN=40, FMAX=15000,
    DURATION=5,           # inference
    TRAIN_DURATION=10,    # training — longer context
    # Model — select one:
    MODEL_NAME='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',   # EVA-02 Large
    # MODEL_NAME='vit_large_patch14_reg4_dinov2.lvd142m',           # DINOv2-Large
    IMG_SIZE=448,          # EVA-02 native resolution (use 518 for DINOv2)
    PRETRAINED=True,
    # Training
    N_FOLDS=5,
    TRAIN_FOLD=1,          # only train fold 1 (checkpoint used for PL diversity)
    EPOCHS=15,
    BATCH_SIZE=16,         # ViT needs more GPU memory than EfficientNet
    LR=5e-5,               # conservative LR for large pretrained ViT
    WEIGHT_DECAY=1e-2,
    # Device
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu',
    SEED=42,
    NUM_WORKERS=NUM_WORKERS,
)
print(f"Device       : {CFG['DEVICE']}")
print(f"Model        : {CFG['MODEL_NAME']}")
print(f"IMG_SIZE     : {CFG['IMG_SIZE']}×{CFG['IMG_SIZE']}")
print(f"Training fold: {CFG['TRAIN_FOLD']} of {CFG['N_FOLDS']}")
print(f"torch: {torch.__version__}, timm: {timm.__version__}")

# %% [markdown]
# ## Label Setup
# Same standard label setup as nb06/nb07/nb08 — always derived from taxonomy.csv
# to keep indices deterministic across all notebooks.

# %%
# ── Label setup ───────────────────────────────────────────────────────────────
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])

taxonomy   = pd.read_csv(BASE_DIR / 'taxonomy.csv')
label_list = taxonomy['primary_label'].tolist()   # 234 total
label2idx  = {l: i for i, l in enumerate(label_list)}
NUM_CLASSES = len(label_list)

# Bird-only subset (this ViT is trained on birds only)
bird_labels  = taxonomy[taxonomy['class_name'] == 'Aves']['primary_label'].tolist()
bird2idx     = {l: i for i, l in enumerate(bird_labels)}
BIRD_CLASSES = len(bird_labels)
print(f"Bird classes: {BIRD_CLASSES} | Total: {NUM_CLASSES}")

# ── Load and filter training data ─────────────────────────────────────────────
train_df = pd.read_csv(BASE_DIR / 'train.csv')
train_df = train_df[train_df['primary_label'].isin(bird_labels)].copy()
train_df['target']   = train_df['primary_label'].map(bird2idx)
train_df['filepath'] = train_df['filename'].apply(
    lambda f: str(BASE_DIR / 'train_audio' / f))

# Sample weights by data quality
def get_sample_weight(row):
    if row.get('source', 'XC') == 'iNat': return 0.4
    r = row.get('rating', 0.0)
    if r >= 3.0: return 0.8
    return 0.5

train_df['weight'] = train_df.apply(get_sample_weight, axis=1)
# GroupKFold site: fall back to primary_label as group when recorder_id absent
train_df['site'] = train_df.get('recorder_id', train_df['primary_label'])
print(f"Training clips (bird only): {len(train_df):,}")

# %% [markdown]
# ## ViTBirdDataset
# Key differences vs nb06's BirdDataset:
# 1. Mel spectrogram is **resized** to `(IMG_SIZE, IMG_SIZE)` — ViT patch tokenization
#    requires a fixed-size square input matching the pretrained resolution.
# 2. Single-channel mel is **repeated 3×** to produce an RGB-like tensor.
# 3. **ImageNet normalization** (µ/σ) is applied — ViT LayerNorm was calibrated on these.

# %%
class ViTBirdDataset(Dataset):
    """
    Loads OGG audio → log-mel spectrogram → resize to IMG_SIZE × IMG_SIZE →
    repeat 3 channels → ImageNet normalize.  Returns (3, IMG_SIZE, IMG_SIZE) tensor.
    """
    # ImageNet statistics: ViT pretrain was on ImageNet, these must match
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, df, cfg, augment=False):
        self.df       = df.reset_index(drop=True)
        self.cfg      = cfg
        self.augment  = augment
        self.n_samples = cfg['SR'] * (cfg['TRAIN_DURATION'] if augment else cfg['DURATION'])
        self.img_size  = cfg['IMG_SIZE']

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
        F  = np.random.randint(0, 16); f0 = np.random.randint(0, self.cfg['N_MELS'] - F)
        S[f0:f0+F, :] = 0.0
        T  = np.random.randint(0, 30); t0 = np.random.randint(0, S.shape[1] - T)
        S[:, t0:t0+T] = 0.0
        return S

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y   = self._load_audio(row['filepath'])
        S   = self._to_melspec(y)
        if self.augment:
            S = self._spec_augment(S)

        # (1, N_MELS, T) → resize to (1, IMG_SIZE, IMG_SIZE) → repeat to (3, IMG_SIZE, IMG_SIZE)
        S_tensor  = torch.from_numpy(S).unsqueeze(0)                         # (1, 128, T)
        S_resized = TF.resize(S_tensor, [self.img_size, self.img_size],
                              antialias=True)                                 # (1, 448, 448)
        S_rgb     = S_resized.repeat(3, 1, 1)                                # (3, 448, 448)

        # Apply ImageNet normalization (ViT pretrained on these statistics)
        S_norm = (S_rgb - self._MEAN) / self._STD                           # (3, 448, 448)

        # Multi-hot bird target (same as nb06)
        target = np.zeros(BIRD_CLASSES, dtype=np.float32)
        if int(row['target']) < BIRD_CLASSES:
            target[int(row['target'])] = 1.0
        if 'secondary_labels' in row and isinstance(row['secondary_labels'], str):
            for sl in row['secondary_labels'].replace("'", "").strip("[]").split(','):
                sl = sl.strip()
                if sl in bird2idx:
                    target[bird2idx[sl]] = 0.5

        weight = float(row.get('weight', 1.0))
        return S_norm, torch.from_numpy(target), weight

# %% [markdown]
# ## ViTBirdModel
# Unlike the CNN notebooks, we do NOT use `in_chans=1`.
# ViT pretrained weights expect 3-channel input; forcing in_chans=1 discards
# all pretrained attention patterns in the patch embedding layer.

# %%
class ViTBirdModel(nn.Module):
    """
    EVA-02 or DINOv2 fine-tuned for bird species classification.
    Used ONLY for pseudo-label generation, NOT for competition inference.
    The timm backbone handles the classifier head automatically.
    """
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,   # replaces original head with new linear layer
        )

    def forward(self, x):
        return self.backbone(x)   # returns (B, num_classes) logits

# %% [markdown]
# ## Training Functions
# Same BCE multi-label loss and macro AUC validation as nb06,
# with per-sample weighting for data quality.

# %%
def train_one_epoch(model, loader, optimizer, scheduler, device):
    """Weighted BCE training — identical to nb06."""
    model.train()
    total_loss = 0.0
    criterion  = nn.BCEWithLogitsLoss(reduction='none')
    for X, y, w in loader:
        X, y, w = X.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss   = (loss.mean(dim=1) * w).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    """Multi-label macro AUC validation — identical to nb06."""
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
    probs      = np.concatenate(all_probs)
    targets    = np.concatenate(all_targets)
    bin_targets = (targets >= 0.5).astype(int)
    valid_cols  = bin_targets.sum(axis=0) > 0
    try:
        auc = roc_auc_score(bin_targets[:, valid_cols], probs[:, valid_cols], average='macro')
    except ValueError:
        auc = 0.0
    return total_loss / len(loader.dataset), auc

# %% [markdown]
# ## Single-Fold Training (Fold 1 Only)
# We train only `TRAIN_FOLD=1` because:
# - We need ONE strong ViT checkpoint as a diverse voter in nb09
# - 5-fold training would take ~35h total (7h × 5) — not worth it
# - The CNN ensemble in nb06 already provides 5-fold coverage for the final submission

# %%
# ── GroupKFold split — generate all splits, use only TRAIN_FOLD ───────────────
gkf    = GroupKFold(n_splits=CFG['N_FOLDS'])
groups = train_df['site'].values
splits = list(gkf.split(train_df, train_df['target'], groups=groups))

fold       = CFG['TRAIN_FOLD']
train_idx, val_idx = splits[fold - 1]  # 0-indexed internally
print(f"\n{'='*60}\n  Training Fold {fold}/{CFG['N_FOLDS']} (ViT — fold 1 only)\n{'='*60}")
print(f"  Train samples: {len(train_idx):,}  |  Val samples: {len(val_idx):,}")
t0 = time.time()

train_fold = train_df.iloc[train_idx]
val_fold   = train_df.iloc[val_idx]

train_ds = ViTBirdDataset(train_fold, CFG, augment=True)
val_ds   = ViTBirdDataset(val_fold,   CFG, augment=False)
train_dl = DataLoader(train_ds, batch_size=CFG['BATCH_SIZE'],
                      shuffle=True,  num_workers=CFG['NUM_WORKERS'], pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=CFG['BATCH_SIZE'],
                      shuffle=False, num_workers=CFG['NUM_WORKERS'], pin_memory=True)

model     = ViTBirdModel(CFG['MODEL_NAME'], BIRD_CLASSES, CFG['PRETRAINED'])
model     = model.to(CFG['DEVICE'])
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=CFG['LR'], weight_decay=CFG['WEIGHT_DECAY'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG['EPOCHS'] * len(train_dl))

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {n_params:,}")

best_auc  = 0.0
fold_hist = []

for epoch in range(1, CFG['EPOCHS'] + 1):
    tr_loss              = train_one_epoch(model, train_dl, optimizer, scheduler, CFG['DEVICE'])
    vl_loss, vl_auc      = validate(model, val_dl, CFG['DEVICE'])
    elapsed              = time.time() - t0
    fold_hist.append({'epoch': epoch, 'tr_loss': tr_loss,
                      'vl_loss': vl_loss, 'vl_auc': vl_auc})
    print(f"  Ep {epoch:02d}/{CFG['EPOCHS']}  "
          f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
          f"vl_auc={vl_auc:.4f}  [{elapsed:.0f}s]")

    if vl_auc > best_auc:
        best_auc = vl_auc
        torch.save(model.state_dict(), OUTPUT_DIR / 'vit_fold1.pth')
        print(f"    ✓ saved  (best auc={best_auc:.4f})")

print(f"\n  Fold {fold} complete — best val AUC: {best_auc:.4f}")

# ── OOF predictions from best checkpoint ──────────────────────────────────────
model.load_state_dict(torch.load(OUTPUT_DIR / 'vit_fold1.pth', map_location=CFG['DEVICE']))
model.eval()
oof_probs = []
with torch.no_grad():
    for X, _, _ in val_dl:
        oof_probs.append(torch.sigmoid(model(X.to(CFG['DEVICE']))).cpu().numpy())
oof_probs = np.concatenate(oof_probs, axis=0)

del train_ds, val_ds, train_dl, val_dl
torch.cuda.empty_cache(); gc.collect()

# %%
# ── Training curve ────────────────────────────────────────────────────────────
hist_df = pd.DataFrame(fold_hist)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'ViT Training — {CFG["MODEL_NAME"]} — Fold {fold}',
             fontsize=11, fontweight='bold')
axes[0].plot(hist_df['epoch'], hist_df['tr_loss'], label='Train', marker='o', markersize=4)
axes[0].plot(hist_df['epoch'], hist_df['vl_loss'], label='Val',   marker='s', markersize=4)
axes[0].set_title('Loss (BCE)'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(hist_df['epoch'], hist_df['vl_auc'], color='darkorange', marker='o', markersize=4)
axes[1].set_title('Val Macro ROC-AUC'); axes[1].set_xlabel('Epoch')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'vit_training_curve.png', bbox_inches='tight')
plt.show()

# %%
# ── Save experiment metadata ──────────────────────────────────────────────────
result = {
    'model':      CFG['MODEL_NAME'],
    'img_size':   CFG['IMG_SIZE'],
    'fold':       fold,
    'epochs':     CFG['EPOCHS'],
    'best_auc':   round(best_auc, 4),
    'lr':         CFG['LR'],
    'batch_size': CFG['BATCH_SIZE'],
    'n_params':   n_params,
}
with open(OUTPUT_DIR / 'vit_fold1_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print("Fold 1 result:", json.dumps(result, indent=2))

# %% [markdown]
# ## Summary & Next Steps

# %%
print("\n" + "="*60)
print("ViT training complete.")
print(f"Checkpoint saved: {OUTPUT_DIR / 'vit_fold1.pth'}")
print(f"Best fold-1 AUC : {best_auc:.4f}")
print()
print("Next steps:")
print("1. Download vit_fold1.pth from the Output tab")
print("2. Upload as a Kaggle dataset (e.g. 'birdclef26-vit')")
print("3. In nb09_pseudo_labeling.py, add to the model dict:")
print("     m = load_model('/kaggle/input/birdclef26-vit/vit_fold1.pth',")
print(f"                    '{CFG['MODEL_NAME']}', BIRD_CLASSES, DEVICE)")
print("     if m: models['vit'] = m")
print()
print("NOTE: Do NOT export to ONNX — ViT is for PL generation only.")
print("      Competition inference uses EfficientNet + OpenVINO (see nb06/nb11).")
print("="*60)
