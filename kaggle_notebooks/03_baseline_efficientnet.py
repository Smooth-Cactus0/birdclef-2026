# %%
# =============================================================================
# BirdCLEF 2026 — EfficientNet-B0 Baseline 🚀
# =============================================================================
# Architecture : EfficientNet-B0 (timm)
# Loss         : CrossEntropy on primary_label
# Metric       : Macro ROC-AUC
# Folds        : 5-fold StratifiedKFold
# Inference    : sliding 5s windows on soundscapes
# Export       : ONNX for CPU deployment
# =============================================================================

# %% [markdown]
# # EfficientNet-B0 Baseline 🚀
# A complete training + inference pipeline. Every step is commented.

# %%
# ── Install / version pins ────────────────────────────────────────────────────
# !pip install -q timm==1.0.3  # uncomment on Kaggle if needed

import os, gc, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = (Path('/kaggle/input/birdclef-2026')
              if Path('/kaggle/input/birdclef-2026').exists()
              else Path('birdclef-2026'))
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    # Audio
    SR          = 32000,
    N_FFT       = 1024,
    HOP_LENGTH  = 320,
    N_MELS      = 128,
    FMIN        = 40,
    FMAX        = 15000,
    DURATION    = 5,         # seconds per clip during training
    # Model
    MODEL_NAME   = 'efficientnet_b0',
    PRETRAINED   = True,
    # Training
    N_FOLDS      = 5,
    EPOCHS       = 10,
    BATCH_SIZE   = 32,
    LR           = 1e-3,
    WEIGHT_DECAY = 1e-4,
    MIN_RATING   = 0.0,       # 0.0 = include all iNat clips
    # Device
    DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu',
    SEED         = 42,
    NUM_WORKERS  = 0 if os.name == 'nt' else 4,  # 0 on Windows to avoid hang
)
print(f"Device: {CFG['DEVICE']}")
print(f"torch: {torch.__version__}, timm: {timm.__version__}")

# %%
# ── Load metadata ─────────────────────────────────────────────────────────────
train_df  = pd.read_csv(BASE_DIR / 'train.csv')
taxonomy  = pd.read_csv(BASE_DIR / 'taxonomy.csv')

# Label encode: map primary_label → integer class index
# Use taxonomy order so indices are deterministic
label_list = taxonomy['primary_label'].tolist()
label2idx  = {l: i for i, l in enumerate(label_list)}
idx2label  = {i: l for l, i in label2idx.items()}
NUM_CLASSES = len(label_list)
print(f"Number of classes: {NUM_CLASSES}")

# Filter and prepare
train_df = train_df[train_df['primary_label'].isin(label2idx)].copy()
train_df['target'] = train_df['primary_label'].map(label2idx)
train_df['filepath'] = train_df['filename'].apply(lambda f: str(BASE_DIR / 'train_audio' / f))

# Optional: filter low-quality XC recordings (keep iNat + XC rating >= threshold)
# train_df = train_df[(train_df['rating'] == 0.0) | (train_df['rating'] >= CFG['MIN_RATING'])]
print(f"Training clips: {len(train_df):,}")
print(train_df['class_name'].value_counts())

# %%
class BirdDataset(Dataset):
    """
    Loads OGG audio, converts to a log-mel spectrogram,
    returns a (1, N_MELS, T) tensor and integer label.
    """
    def __init__(self, df, cfg, augment=False):
        self.df      = df.reset_index(drop=True)
        self.cfg     = cfg
        self.augment = augment
        self.n_samples = cfg['SR'] * cfg['DURATION']

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path):
        try:
            y, sr = librosa.load(path, sr=self.cfg['SR'], mono=True)
        except Exception:
            return np.zeros(self.n_samples, dtype=np.float32)
        # Pad or trim to fixed length
        if len(y) < self.n_samples:
            y = np.pad(y, (0, self.n_samples - len(y)), mode='constant')
        else:
            # Random crop during training; centre crop during validation
            if self.augment and len(y) > self.n_samples:
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
        # Normalise to [0, 1]
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        return S_db.astype(np.float32)

    def _augment(self, S):
        """SpecAugment: random time + frequency masking."""
        S = S.copy()
        # Frequency masking
        F = np.random.randint(0, 20)
        f0 = np.random.randint(0, self.cfg['N_MELS'] - F)
        S[f0:f0 + F, :] = 0.0
        # Time masking
        T = np.random.randint(0, 30)
        t0 = np.random.randint(0, S.shape[1] - T)
        S[:, t0:t0 + T] = 0.0
        return S

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y   = self._load_audio(row['filepath'])
        S   = self._to_melspec(y)
        if self.augment:
            S = self._augment(S)
        # Shape: (1, N_MELS, T) — single channel image
        S_tensor = torch.from_numpy(S).unsqueeze(0)
        label    = int(row['target'])
        return S_tensor, label

# %%
class BirdModel(nn.Module):
    """
    EfficientNet-B0 pretrained on ImageNet, adapted for mel spectrograms.
    The first conv layer accepts 1-channel input (converted by averaging).
    """
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,           # single-channel mel spectrogram
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(x)  # returns logits, shape (B, num_classes)

# %%
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.cpu().numpy())
    all_probs   = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # One-hot encode for macro AUC
    y_onehot = np.zeros_like(all_probs)
    y_onehot[np.arange(len(all_targets)), all_targets] = 1
    try:
        auc = roc_auc_score(y_onehot, all_probs, average='macro')
    except ValueError:
        auc = 0.0
    return total_loss / len(loader.dataset), auc

# %%
# ── Cross-validation ──────────────────────────────────────────────────────────
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])

skf      = StratifiedKFold(n_splits=CFG['N_FOLDS'], shuffle=True,
                           random_state=CFG['SEED'])
oof_preds  = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)
fold_aucs  = []
history    = []

for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df['target']), start=1):
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

    model     = BirdModel(CFG['MODEL_NAME'], NUM_CLASSES, CFG['PRETRAINED'])
    model     = model.to(CFG['DEVICE'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG['LR'], weight_decay=CFG['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG['LR'],
        steps_per_epoch=len(train_dl), epochs=CFG['EPOCHS'])

    best_auc   = 0.0
    fold_hist  = []

    for epoch in range(1, CFG['EPOCHS'] + 1):
        tr_loss = train_one_epoch(model, train_dl, optimizer, scheduler,
                                  criterion, CFG['DEVICE'])
        vl_loss, vl_auc = validate(model, val_dl, criterion, CFG['DEVICE'])
        fold_hist.append({'fold': fold, 'epoch': epoch,
                          'tr_loss': tr_loss, 'vl_loss': vl_loss, 'vl_auc': vl_auc})
        elapsed = time.time() - t0
        print(f"  Ep {epoch:02d}/{CFG['EPOCHS']}  "
              f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
              f"vl_auc={vl_auc:.4f}  [{elapsed:.0f}s]")

        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(),
                       OUTPUT_DIR / f'model_fold{fold}.pth')
            print(f"    ✓ saved (auc={best_auc:.4f})")

    # OOF predictions from best checkpoint
    model.load_state_dict(torch.load(OUTPUT_DIR / f'model_fold{fold}.pth',
                                      map_location=CFG['DEVICE']))
    model.eval()
    all_probs_list = []
    with torch.no_grad():
        for X, _ in val_dl:
            all_probs_list.append(
                torch.softmax(model(X.to(CFG['DEVICE'])), dim=1).cpu().numpy())
    oof_preds[val_idx] = np.concatenate(all_probs_list, axis=0)

    fold_aucs.append(best_auc)
    history.extend(fold_hist)
    del model, train_ds, val_ds, train_dl, val_dl
    torch.cuda.empty_cache(); gc.collect()

print(f"\n{'='*60}")
print(f"  CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"  Per-fold: {[round(a,4) for a in fold_aucs]}")
print(f"{'='*60}")

# %%
hist_df = pd.DataFrame(history)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for fold in hist_df['fold'].unique():
    fh = hist_df[hist_df['fold'] == fold]
    axes[0].plot(fh['epoch'], fh['vl_loss'], marker='o', label=f'fold {fold}', markersize=4)
    axes[1].plot(fh['epoch'], fh['vl_auc'],  marker='o', label=f'fold {fold}', markersize=4)

axes[0].set_title('Validation Loss per Fold', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('CrossEntropy Loss')
axes[0].legend()
axes[1].set_title('Validation Macro ROC-AUC per Fold', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC')
axes[1].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves.png', bbox_inches='tight')
plt.show()

# %%
# ── Inference ─────────────────────────────────────────────────────────────────
# Test soundscapes are long OGG files (minutes). We slide a 5s window
# and predict all species for each window. row_id = {filename}_{end_sec}

sample_sub = pd.read_csv(BASE_DIR / 'sample_submission.csv')
test_soundscapes = sorted((BASE_DIR / 'test_soundscapes').glob('*.ogg'))
print(f"Test soundscapes found: {len(test_soundscapes)}")

# Load best model (fold 1 by default; in practice ensemble all folds)
inf_model = BirdModel(CFG['MODEL_NAME'], NUM_CLASSES, pretrained=False)
inf_model.load_state_dict(torch.load(OUTPUT_DIR / 'model_fold1.pth',
                                      map_location=CFG['DEVICE']))
inf_model = inf_model.to(CFG['DEVICE'])
inf_model.eval()

def predict_soundscape(audio_path, model, cfg, device):
    """Slide 5s windows over a long soundscape, return dict of row_id → probs."""
    try:
        y, sr = librosa.load(str(audio_path), sr=cfg['SR'], mono=True)
    except Exception as e:
        print(f"  Error loading {audio_path.name}: {e}")
        return {}
    stem       = audio_path.stem
    n_samples  = cfg['SR'] * cfg['DURATION']
    n_chunks   = max(1, len(y) // n_samples)
    results    = {}
    chunks     = []
    row_ids    = []
    for i in range(n_chunks):
        start  = i * n_samples
        chunk  = y[start:start + n_samples]
        if len(chunk) < n_samples:
            chunk = np.pad(chunk, (0, n_samples - len(chunk)))
        end_sec = (i + 1) * cfg['DURATION']
        S = librosa.feature.melspectrogram(
            y=chunk.astype(np.float32),
            sr=cfg['SR'], n_fft=cfg['N_FFT'],
            hop_length=cfg['HOP_LENGTH'],
            n_mels=cfg['N_MELS'], fmin=cfg['FMIN'], fmax=cfg['FMAX'])
        S_db   = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        chunks.append(torch.from_numpy(S_norm).float().unsqueeze(0))
        row_ids.append(f"{stem}_{end_sec}")
    if not chunks:
        return {}
    batch  = torch.stack(chunks).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()  # softmax matches CrossEntropy
    for rid, p in zip(row_ids, probs):
        results[rid] = p
    return results

all_preds = {}
for sf_path in test_soundscapes:
    preds = predict_soundscape(sf_path, inf_model, CFG, CFG['DEVICE'])
    all_preds.update(preds)
    print(f"  {sf_path.name}: {len(preds)} chunks")

print(f"Total prediction rows: {len(all_preds)}")

# %%
# Align with sample_submission — vectorized merge (O(n) not O(n²))
pred_df = pd.DataFrame.from_dict(all_preds, orient='index', columns=label_list)
pred_df.index.name = 'row_id'
pred_df = pred_df.reset_index()

sub = sample_sub[['row_id']].merge(pred_df, on='row_id', how='left')
# Fill any row_id not covered by inference with uniform probability
sub[label_list] = sub[label_list].fillna(1.0 / NUM_CLASSES)

sub.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
print(f"Submission saved: {sub.shape}")
print(sub.head(2))

# %%
# ── ONNX export ───────────────────────────────────────────────────────────────
# Required for OpenVINO conversion and CPU inference budget management.
try:
    # Time frames = 1 + (SR*DURATION // HOP_LENGTH) with librosa center=True padding
    n_frames = 1 + (CFG['SR'] * CFG['DURATION'] // CFG['HOP_LENGTH'])  # = 501
    dummy = torch.randn(1, 1, CFG['N_MELS'], n_frames).to(CFG['DEVICE'])
    torch.onnx.export(
        inf_model, dummy,
        str(OUTPUT_DIR / 'model_efficientnet_b0.onnx'),
        input_names=['input'], output_names=['output'],
        opset_version=11,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    print("ONNX export successful →", OUTPUT_DIR / 'model_efficientnet_b0.onnx')
except Exception as e:
    print(f"ONNX export failed: {e}")

# %%
# Save experiment results
oof_results = {
    'model':     CFG['MODEL_NAME'],
    'n_folds':   CFG['N_FOLDS'],
    'epochs':    CFG['EPOCHS'],
    'fold_aucs': [round(a, 4) for a in fold_aucs],
    'mean_auc':  round(float(np.mean(fold_aucs)), 4),
    'std_auc':   round(float(np.std(fold_aucs)),  4),
}
with open(OUTPUT_DIR / 'oof_scores.json', 'w') as f:
    json.dump(oof_results, f, indent=2)
print("OOF results:", json.dumps(oof_results, indent=2))
