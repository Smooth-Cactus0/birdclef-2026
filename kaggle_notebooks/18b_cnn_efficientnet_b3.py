# %%
# ============================================================================
# BirdCLEF 2026 -- nb18b: CNN Backbone (EfficientNet-B3) + Mel Cache
# ============================================================================
# Like nb18a but with **pre-computed mel spectrogram cache** so the heavier B3
# backbone fits within the 12h kernel cap. Cache build runs once at kernel
# start (~30 min for ~36k unique audio files); training then reads .npy fp16
# mels directly with no audio decode in the inner loop.
#
# Trade-off: BG mix / gaussian noise augmentations are DROPPED (they need
# waveform). Only SpecAugment (time + freq mask) remains.
#
# Inputs:
#   - competitions/birdclef-2026 (train_audio, train_soundscapes, taxonomy, ...)
#
# Internet : ON   (timm downloads pretrained weights on first call)
# GPU      : T4   (P100 sm_60 fails on PyTorch 2.4+)
# ============================================================================

# %%
import gc
import json
import os
import random
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# %%
# -- Config -------------------------------------------------------------------
DRY_RUN        = False           # full 5-fold x 14-epoch run on B3 + mel cache
SEED           = 42
NOTEBOOK_ID    = "nb18b"
BACKBONE_NAME  = "tf_efficientnet_b3_ns"

SR             = 32_000
N_FFT          = 1024
HOP_LENGTH     = 320
N_MELS         = 128
FMIN           = 40
FMAX           = 15_000
TRAIN_DURATION = 10              # seconds
INFER_DURATION = 5
WIN_TRAIN      = TRAIN_DURATION * SR
WIN_INFER      = INFER_DURATION * SR

N_FOLDS        = 1 if DRY_RUN else 5
EPOCHS         = 2 if DRY_RUN else 14
LR_MAX         = 3e-4
LR_MIN         = 1e-5
WD             = 1e-2
BATCH_SZ       = 32
NUM_WORKERS    = 4
GRAD_ACCUM     = 1

SC_UPSAMPLE    = 3               # repeat soundscape segments this many times
# BG mix / gaussian noise dropped in nb18b -- dataset returns mels, not waveforms
SPEC_TIME_MASK = 30              # frames
SPEC_FREQ_MASK = 16              # mel bins
SPEC_TIME_P    = 0.5
SPEC_FREQ_P    = 0.5

# Mel-cache config
CACHE_AUDIO_MAX_SEC = 30         # truncate train_audio clips longer than this
CACHE_SC_MAX_SEC    = 60         # keep full soundscape so any 5s window slices
T_TARGET            = (TRAIN_DURATION * SR) // HOP_LENGTH + 1   # 1001 frames @10s
T_INFER             = (INFER_DURATION * SR) // HOP_LENGTH + 1   #  501 frames @5s

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = True

# %%
# -- Paths --------------------------------------------------------------------
BASE_DIR = (Path("/kaggle/input/competitions/birdclef-2026")
            if Path("/kaggle/input/competitions/birdclef-2026").exists()
            else Path("/kaggle/input/birdclef-2026"))
TRAIN_AUDIO_DIR = BASE_DIR / "train_audio"
TRAIN_SC_DIR    = BASE_DIR / "train_soundscapes"
TEST_SC_DIR     = BASE_DIR / "test_soundscapes"
OUT_DIR         = Path("/kaggle/working"); OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR        = OUT_DIR / "ckpts"; CKPT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE         : {DEVICE}")
print(f"DRY_RUN        : {DRY_RUN}")
print(f"N_FOLDS        : {N_FOLDS}")
print(f"EPOCHS         : {EPOCHS}")
print(f"BACKBONE_NAME  : {BACKBONE_NAME}")
print(f"BASE_DIR exists: {BASE_DIR.exists()}")
print(f"torch          : {torch.__version__}")
print(f"torchaudio     : {torchaudio.__version__}")

import timm
print(f"timm           : {timm.__version__}")

# %%
# -- Load metadata + build label index ----------------------------------------
train_meta  = pd.read_csv(BASE_DIR / "train.csv")
sc_labels   = pd.read_csv(BASE_DIR / "train_soundscapes_labels.csv")
taxonomy    = pd.read_csv(BASE_DIR / "taxonomy.csv")
sample_sub  = pd.read_csv(BASE_DIR / "sample_submission.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES      = len(PRIMARY_LABELS)
label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}

print(f"train_meta rows: {len(train_meta)}")
print(f"sc_labels rows : {len(sc_labels)}")
print(f"N_CLASSES      : {N_CLASSES}")

# %%
# -- Build target vectors -----------------------------------------------------
def _parse_secondary(s):
    """secondary_labels can be JSON-like ['x','y'] or 'x;y' or NaN."""
    if pd.isna(s) or s == "" or s == "[]":
        return []
    s = str(s).strip()
    # strip brackets, quotes, split on , or ;
    s = re.sub(r"[\[\]'\"]+", "", s)
    parts = re.split(r"[;,]", s)
    return [p.strip() for p in parts if p.strip()]


def make_target(primary, secondary):
    y = np.zeros(N_CLASSES, dtype=np.float32)
    if primary in label_to_idx:
        y[label_to_idx[primary]] = 1.0
    for s in _parse_secondary(secondary):
        if s in label_to_idx:
            y[label_to_idx[s]] = 1.0
    return y


def _parse_time(t):
    """Parse 'HH:MM:SS', 'MM:SS', or numeric -> seconds (float)."""
    if pd.isna(t):
        return 0.0
    s = str(t).strip()
    if not s:
        return 0.0
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


# train_audio entries
train_audio_entries = []
for _, row in train_meta.iterrows():
    primary   = row.get("primary_label", "")
    secondary = row.get("secondary_labels", "")
    target    = make_target(primary, secondary)
    if target.sum() == 0:
        continue
    train_audio_entries.append({
        "source"    : "train_audio",
        "path"      : str(TRAIN_AUDIO_DIR / row["filename"]),
        "start_sec" : None,
        "end_sec"   : None,
        "target"    : target,
        "primary"   : primary if primary in label_to_idx else "unknown",
    })

# soundscape segment entries (each labeled 5s window)
sc_entries = []
for _, row in sc_labels.iterrows():
    primary_str = str(row.get("primary_label", "")).strip()
    target = np.zeros(N_CLASSES, dtype=np.float32)
    primary_first = None
    for s in primary_str.split(";"):
        s = s.strip()
        if s and s in label_to_idx:
            target[label_to_idx[s]] = 1.0
            if primary_first is None:
                primary_first = s
    if target.sum() == 0:
        continue
    sc_entries.append({
        "source"    : "soundscape",
        "path"      : str(TRAIN_SC_DIR / row["filename"]),
        "start_sec" : _parse_time(row["start"]),
        "end_sec"   : _parse_time(row["end"]),
        "target"    : target,
        "primary"   : primary_first,
    })

ALL_ENTRIES = train_audio_entries + sc_entries * SC_UPSAMPLE
print(f"train_audio      : {len(train_audio_entries)}")
print(f"soundscape (x{SC_UPSAMPLE}) : {len(sc_entries) * SC_UPSAMPLE}")
print(f"total entries    : {len(ALL_ENTRIES)}")

# %%
# -- Audio loader helper (still needed for cache build + inference) ----------
def _load_audio(path, target_sr=SR):
    """Load OGG, resample to target_sr, return mono float32 1D tensor."""
    try:
        wav, sr = torchaudio.load(str(path))
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return wav.squeeze(0)
    except Exception as ex:
        print(f"  load fail {Path(path).name}: {ex}")
        return torch.zeros(target_sr * 5, dtype=torch.float32)


# %%
# -- Mel-spectrogram cache build ----------------------------------------------
# One-time pass over all unique audio paths in ALL_ENTRIES. Cache stored at
# /kaggle/working/mel_cache/<safe_name>.npy as fp16 (n_mels, T) arrays.
MEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_cpu_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
    n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
)
_cpu_amp2db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)


def cache_path_for(audio_path):
    """Build a unique cache filename from the audio path."""
    s = str(audio_path).replace("\\", "/").replace("/", "_").replace(":", "_").replace(" ", "_")
    return MEL_CACHE_DIR / (s + ".npy")


def _cache_one(p, max_sec):
    """Decode -> mel -> amp_to_db -> save as fp16 npy."""
    cp = cache_path_for(p)
    if cp.exists():
        return False  # already cached
    wav = _load_audio(p, target_sr=SR)
    if len(wav) == 0:
        return False
    max_samples = max_sec * SR
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    mel = _cpu_mel(wav.unsqueeze(0))
    mel = _cpu_amp2db(mel).squeeze(0).numpy().astype(np.float16)
    np.save(cp, mel)
    return True


# Compute per-path max-seconds (soundscapes get 60s, train_audio 30s)
path_max_sec = {}
for e in ALL_ENTRIES:
    cap = CACHE_SC_MAX_SEC if e["source"] == "soundscape" else CACHE_AUDIO_MAX_SEC
    if path_max_sec.get(e["path"], 0) < cap:
        path_max_sec[e["path"]] = cap

unique_paths = sorted(path_max_sec.keys())
print(f"\nBuilding mel cache: {len(unique_paths)} unique audio files")
t_cache = time.time()
built, skipped = 0, 0
for i, p in enumerate(unique_paths):
    try:
        if _cache_one(p, path_max_sec[p]):
            built += 1
        else:
            skipped += 1
    except Exception as ex:
        print(f"  cache error on {Path(p).name}: {ex}")
    if (i + 1) % 2000 == 0 or i == len(unique_paths) - 1:
        elapsed = time.time() - t_cache
        eta = (len(unique_paths) - i - 1) * (elapsed / (i + 1))
        rate = (i + 1) / elapsed
        print(f"  [{i+1}/{len(unique_paths)}] built={built} skipped={skipped} "
              f"rate={rate:.0f}/s elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min")

print(f"Cache built in {(time.time()-t_cache)/60:.1f} min "
      f"(new={built}, already_present={skipped})")

# Attach cache_path to every entry
for e in ALL_ENTRIES:
    e["cache_path"] = str(cache_path_for(e["path"]))


# %%
# -- Dataset (mel-based; reads from cache) ------------------------------------
def _sec_to_frame(sec):
    return int(sec * SR // HOP_LENGTH)


class BirdMelDataset(Dataset):
    def __init__(self, entries, training=True):
        self.entries  = entries
        self.training = training

    def __len__(self):
        return len(self.entries)

    def _crop_or_pad_time(self, mel, T_target):
        n_mels, T_actual = mel.shape
        if T_actual >= T_target:
            start = random.randint(0, T_actual - T_target) if self.training else 0
            return mel[:, start:start + T_target]
        # pad with repeat
        reps = (T_target + T_actual - 1) // T_actual
        return np.tile(mel, (1, reps))[:, :T_target]

    def __getitem__(self, i):
        e = self.entries[i]
        try:
            mel = np.load(e["cache_path"]).astype(np.float32)
        except Exception:
            mel = np.zeros((N_MELS, T_TARGET), dtype=np.float32)

        # Soundscape segment: slice the labeled 5s window from full mel
        if e["source"] == "soundscape":
            s = _sec_to_frame(e["start_sec"])
            t = _sec_to_frame(e["end_sec"]) + 1
            t = min(t, mel.shape[1])
            if s < mel.shape[1] and t > s:
                mel = mel[:, s:t]
            else:
                mel = np.zeros((N_MELS, T_INFER), dtype=np.float32)

        mel = self._crop_or_pad_time(mel, T_TARGET)
        return torch.from_numpy(mel).float(), torch.from_numpy(e["target"]).float()


# %%
# -- Mel extractor (used only at inference; test data isn't cached) ----------
class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, wav):
        with torch.cuda.amp.autocast(enabled=False):
            wav = wav.float()
            mel = self.mel(wav)
            mel = self.amp_to_db(mel)
        return mel   # (B, n_mels, T)


# %%
# -- Model (takes mel input directly) -----------------------------------------
class BirdCNN(nn.Module):
    def __init__(self, backbone_name=BACKBONE_NAME, n_classes=N_CLASSES, in_chans=1):
        super().__init__()
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=SPEC_TIME_MASK)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=SPEC_FREQ_MASK)
        self.backbone  = timm.create_model(
            backbone_name, pretrained=True, in_chans=in_chans,
            num_classes=0, global_pool="avg",
        )
        feat_dim  = self.backbone.num_features
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, mel):
        # mel: (B, n_mels, T) in fp32 (already amp-to-db'd)
        with torch.cuda.amp.autocast(enabled=False):
            if self.training:
                if random.random() < SPEC_TIME_P:
                    mel = self.time_mask(mel)
                if random.random() < SPEC_FREQ_P:
                    mel = self.freq_mask(mel)
            m = mel.mean(dim=(-2, -1), keepdim=True)
            s = mel.std (dim=(-2, -1), keepdim=True)
            mel = (mel - m) / (s + 1e-6)
        x = mel.unsqueeze(1)   # (B, 1, n_mels, T)
        return self.head(self.backbone(x))


# %%
# -- Folds --------------------------------------------------------------------
strata_raw = [e["primary"] for e in ALL_ENTRIES]
class_counts = Counter(strata_raw)
# Classes with <N_FOLDS samples can't stratify; bucket as "__rare__"
RARE_THRESHOLD = max(2, N_FOLDS)
strata = [p if class_counts[p] >= RARE_THRESHOLD else "__rare__" for p in strata_raw]

if N_FOLDS > 1:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(np.arange(len(ALL_ENTRIES)), strata))
else:
    # Single 80/20 split for dry run
    rng = np.random.RandomState(SEED)
    n = len(ALL_ENTRIES)
    idx = np.arange(n); rng.shuffle(idx)
    cut = int(n * 0.8)
    fold_splits = [(idx[:cut], idx[cut:])]

print(f"Folds : {len(fold_splits)}")
for fi, (tr, va) in enumerate(fold_splits):
    print(f"  fold {fi}: train={len(tr)}  val={len(va)}")

# %%
# -- Training -----------------------------------------------------------------
def _macro_auc(y_true, y_score):
    """Skip classes that have no positives in y_true."""
    aucs = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0 or y_true[:, c].sum() == len(y_true):
            continue
        try:
            aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


def train_one_fold(fold_idx, train_idx, val_idx):
    print(f"\n{'='*60}\nFold {fold_idx}/{N_FOLDS}\n{'='*60}")
    train_entries = [ALL_ENTRIES[i] for i in train_idx]
    val_entries   = [ALL_ENTRIES[i] for i in val_idx]

    train_ds = BirdMelDataset(train_entries, training=True)
    val_ds   = BirdMelDataset(val_entries,   training=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SZ, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SZ, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = BirdCNN().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=EPOCHS * len(train_loader), eta_min=LR_MIN,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    scaler  = torch.cuda.amp.GradScaler()

    best_auc = -1.0   # negative so first epoch always saves, even if val_auc=0
    best_ckpt_path = CKPT_DIR / f"fold_{fold_idx}_best.pth"
    fold_history = []
    first_batch_loss_logged = False

    for epoch in range(EPOCHS):
        # ---- train ----
        model.train()
        t0 = time.time()
        train_loss = 0.0; n_seen = 0
        for bi, (mel, y) in enumerate(train_loader):
            mel = mel.to(DEVICE, non_blocking=True)
            y   = y.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(mel)
                loss   = loss_fn(logits, y)
            # Early NaN detection: print first batch loss so failures are caught fast
            if not first_batch_loss_logged:
                lv = float(loss.item())
                print(f"  [diagnostic] first batch loss = {lv:.4f}  isnan={lv != lv}")
                first_batch_loss_logged = True
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            train_loss += loss.item() * mel.size(0)
            n_seen     += mel.size(0)
        train_loss /= max(n_seen, 1)

        # ---- val ----
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for mel, y in val_loader:
                mel = mel.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(mel)
                ps.append(torch.sigmoid(logits).float().cpu().numpy())
                ys.append(y.numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        val_auc = _macro_auc(ys, ps)
        elapsed = time.time() - t0
        print(f"  ep {epoch+1:02d}/{EPOCHS}  loss={train_loss:.4f}  "
              f"val_auc={val_auc:.4f}  ({elapsed:.1f}s)")
        fold_history.append({"epoch": epoch + 1, "loss": float(train_loss),
                              "val_auc": float(val_auc), "sec": float(elapsed)})

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_ckpt_path)

    # Belt-and-suspenders: if for some reason no save happened, save the last state.
    if not best_ckpt_path.exists():
        print(f"  WARNING: no checkpoint saved during training; dumping final state")
        torch.save(model.state_dict(), best_ckpt_path)

    return best_auc, str(best_ckpt_path), fold_history


# %%
# -- Run all folds ------------------------------------------------------------
t_start    = time.time()
fold_aucs  = []
fold_ckpts = []
all_history = []

for fi, (tr, va) in enumerate(fold_splits):
    auc, ckpt, hist = train_one_fold(fi, tr, va)
    fold_aucs.append(auc)
    fold_ckpts.append(ckpt)
    all_history.append(hist)
    gc.collect(); torch.cuda.empty_cache()

oof_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.0
training_time_min = (time.time() - t_start) / 60.0
print(f"\n{'='*60}")
print(f"All folds done. Fold AUCs: {[round(a, 4) for a in fold_aucs]}")
print(f"Mean val AUC : {oof_auc:.4f}")
print(f"Time         : {training_time_min:.1f} min")
print(f"{'='*60}")

# %%
# -- Inference ----------------------------------------------------------------
# Determine test soundscapes (with staging fallback)
def _get_test_paths():
    if TEST_SC_DIR.exists():
        paths = sorted(TEST_SC_DIR.glob("*.ogg"))
        if paths:
            return paths, False
    # Staging fallback: first 16 train soundscapes (just to produce a valid csv)
    paths = sorted(TRAIN_SC_DIR.glob("*.ogg"))[:16]
    return paths, True


test_paths, is_staging = _get_test_paths()
print(f"\nTest paths: {len(test_paths)}  staging={is_staging}")


def _windows_from_audio(path):
    """Load whole soundscape, return (n_w, WIN_INFER) int-aligned 5s windows.
    Pads/truncates to multiples of WIN_INFER; minimum 1 window."""
    wav = _load_audio(path, target_sr=SR).numpy().astype(np.float32)
    n_full = max(1, len(wav) // WIN_INFER)
    needed = n_full * WIN_INFER
    if len(wav) < needed:
        wav = np.pad(wav, (0, needed - len(wav)))
    else:
        wav = wav[:needed]
    return wav.reshape(n_full, WIN_INFER)


# Load fold models + standalone mel extractor (test data isn't in our cache)
print("Loading fold models for inference...")
fold_models = []
for ckpt in fold_ckpts:
    m = BirdCNN().to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval()
    fold_models.append(m)

mel_extractor = MelExtractor().to(DEVICE)
mel_extractor.eval()


# Run inference per file
rows = []
t_inf = time.time()
for pi, path in enumerate(test_paths):
    windows = _windows_from_audio(path)            # (n_w, WIN_INFER)
    wav_b   = torch.from_numpy(windows).float().to(DEVICE)
    with torch.no_grad():
        mel_b = mel_extractor(wav_b)                # (n_w, n_mels, T)
        preds_folds = []
        for m in fold_models:
            with torch.cuda.amp.autocast():
                logits = m(mel_b)
            preds_folds.append(torch.sigmoid(logits).float().cpu().numpy())
    preds = np.mean(preds_folds, axis=0)            # (n_w, N_CLASSES)
    stem = path.stem
    for wi in range(len(windows)):
        end_sec = (wi + 1) * INFER_DURATION
        row = {"row_id": f"{stem}_{end_sec}"}
        for ci, lbl in enumerate(PRIMARY_LABELS):
            row[lbl] = float(preds[wi, ci])
        rows.append(row)
    if (pi + 1) % 50 == 0 or pi == len(test_paths) - 1:
        elapsed = time.time() - t_inf
        eta = (len(test_paths) - pi - 1) * (elapsed / (pi + 1))
        print(f"  [{pi+1}/{len(test_paths)}] elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min")

inference_time_min = (time.time() - t_inf) / 60.0
print(f"Inference time: {inference_time_min:.1f} min")

# %%
# -- Submission ---------------------------------------------------------------
pred_df = pd.DataFrame(rows)
sub     = sample_sub[["row_id"]].merge(pred_df, on="row_id", how="left")
fill    = 1.0 / N_CLASSES
sub[PRIMARY_LABELS] = sub[PRIMARY_LABELS].fillna(fill)
sub.to_csv(OUT_DIR / "submission.csv", index=False)
print(f"\nWrote submission.csv : {sub.shape}")
print(f"  rows with predictions: {pred_df.shape[0]}")
print(f"  rows filled w/ {fill:.4f}: {(sub.shape[0] - pred_df.shape[0])}")

# %%
# -- Diagnostics --------------------------------------------------------------
# Per-class AUC on val (use last fold's val set for compact diagnostic)
# In dry run with 1 fold, this IS the only fold. Full run: last fold only.
print("\nComputing per-class AUC on last fold's validation set...")
last_tr, last_va = fold_splits[-1]
val_entries = [ALL_ENTRIES[i] for i in last_va]
val_ds      = BirdMelDataset(val_entries, training=False)
val_loader  = DataLoader(val_ds, batch_size=BATCH_SZ, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
m_last = fold_models[-1]
m_last.eval()
ys, ps = [], []
with torch.no_grad():
    for mel, y in val_loader:
        mel = mel.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = m_last(mel)
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.numpy())
ys = np.concatenate(ys); ps = np.concatenate(ps)

per_class_aucs = []
for ci, lbl in enumerate(PRIMARY_LABELS):
    n_pos = int(ys[:, ci].sum())
    if n_pos == 0 or n_pos == len(ys):
        per_class_aucs.append({"species": lbl, "n_pos": n_pos, "auc": float("nan")})
    else:
        try:
            a = roc_auc_score(ys[:, ci], ps[:, ci])
        except Exception:
            a = float("nan")
        per_class_aucs.append({"species": lbl, "n_pos": n_pos, "auc": float(a)})

auc_df = pd.DataFrame(per_class_aucs).sort_values("auc", ascending=False)
auc_df.to_csv(OUT_DIR / f"per_class_auc_{NOTEBOOK_ID}.csv", index=False)
print(auc_df.head(10).to_string(index=False))

# %%
diagnostics = {
    "notebook"          : NOTEBOOK_ID,
    "backbone"          : BACKBONE_NAME,
    "dry_run"           : DRY_RUN,
    "n_train_entries"   : len(ALL_ENTRIES),
    "n_train_audio"     : len(train_audio_entries),
    "n_sc_segments"     : len(sc_entries),
    "sc_upsample"       : SC_UPSAMPLE,
    "n_folds"           : N_FOLDS,
    "epochs"            : EPOCHS,
    "batch_sz"          : BATCH_SZ,
    "lr_max"            : LR_MAX,
    "fold_aucs"         : [round(float(a), 6) for a in fold_aucs],
    "oof_auc_macro"     : round(oof_auc, 6),
    "training_time_min" : round(training_time_min, 1),
    "inference_time_min": round(inference_time_min, 1),
    "is_staging_test"   : bool(is_staging),
    "n_test_paths"      : len(test_paths),
    "nb15a_reference_lb": 0.883,
}
with open(OUT_DIR / f"diagnostics_{NOTEBOOK_ID}.json", "w") as f:
    json.dump(diagnostics, f, indent=2)
print(f"\nSaved diagnostics_{NOTEBOOK_ID}.json")
print(json.dumps(diagnostics, indent=2))
