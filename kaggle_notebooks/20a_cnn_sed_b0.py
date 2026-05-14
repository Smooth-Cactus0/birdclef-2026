# %%
# ============================================================================
# BirdCLEF 2026 -- nb20a: Spec-CNN + SED head + 20s chunks + raw-audio MixUp
# ============================================================================
# First proper attempt at a from-scratch CNN baseline, applying the 2025
# winners' techniques. Single backbone (EfficientNet-B0). Goal: validate the
# full pipeline end-to-end and clear LB 0.85.
#
# Key techniques (all from 2025 top 4 solutions):
#   - 20s training chunks (1st place: +0.030 vs 5s chunks)
#   - SED head: framewise predictions pooled via attention (1st & 2nd places)
#   - Mel: n_mels=224, n_fft=4096, hop=1252, fmin=0, fmax=16000 (1st place)
#   - 3-channel mel (repeat) for ImageNet-pretrained backbones
#   - CrossEntropy loss with secondary labels = 1 (1st place choice)
#   - MixUp on RAW audio, blend weight 0.5 (Beta=inf), label union (1st place)
#   - AdamW, cosine schedule, 25 epochs
#   - Inference: 9 overlapping 20s windows per 60s soundscape, 5s stride,
#     framewise overlap averaging into 12 x 5s output slots
#
# Inputs:
#   - competitions/birdclef-2026 (train_audio, train_soundscapes_labels.csv,
#     taxonomy.csv, sample_submission.csv, test_soundscapes)
#
# Internet : OFF  (no pip installs needed; timm + torchaudio pre-installed)
# GPU      : ON   (NvidiaTeslaT4 -- P100 fails on PyTorch 2.4+)
# Time     : ~3h for 5-fold x 25 epochs
# ============================================================================

# %%
import gc
import json
import math
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import timm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"torch       : {torch.__version__}")
print(f"torchaudio  : {torchaudio.__version__}")
print(f"timm        : {timm.__version__}")
print(f"device      : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU         : {torch.cuda.get_device_name(0)}")

# %%
BASE_DIR = (Path("/kaggle/input/competitions/birdclef-2026")
            if Path("/kaggle/input/competitions/birdclef-2026").exists()
            else Path("/kaggle/input/birdclef-2026"))
TRAIN_AUDIO_DIR  = BASE_DIR / "train_audio"
TRAIN_SC_DIR     = BASE_DIR / "train_soundscapes"
OUT_DIR          = Path("/kaggle/working"); OUT_DIR.mkdir(exist_ok=True)
print(f"BASE_DIR exists   = {BASE_DIR.exists()}")
print(f"train_audio       = {TRAIN_AUDIO_DIR.exists()}")

# %%
# Audio config -- 1st place 2025 params
SR             = 32_000
WINDOW_SEC     = 20
WINDOW_SAMPLES = SR * WINDOW_SEC          # 640_000
INFER_STRIDE   = 5                        # seconds
INFER_WIN_PER_SC = 9                      # (60 - 20) / 5 + 1
OUTPUT_SLOTS_PER_SC = 12                  # 60 / 5
N_MELS         = 224
N_FFT          = 4096
HOP_LENGTH     = 1252
FMIN           = 0
FMAX           = 16_000
TOP_DB         = 80.0

# Model / training config
BACKBONE       = "tf_efficientnet_b0.ns_jft_in1k"
N_FOLDS        = 5
EPOCHS         = 25
BATCH_SZ       = 32
LR_MAX         = 5e-4
LR_MIN         = 1e-6
WD             = 1e-4
LABEL_SMOOTH   = 0.005
MIXUP_P        = 0.5
MIXUP_WEIGHT   = 0.5          # 1st place: Beta=inf -> constant 0.5 blend
N_WORKERS      = 4
SEED           = 42

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# %%
taxonomy   = pd.read_csv(BASE_DIR / "taxonomy.csv")
train_meta = pd.read_csv(BASE_DIR / "train.csv")
sc_labels  = pd.read_csv(BASE_DIR / "train_soundscapes_labels.csv")
sample_sub = pd.read_csv(BASE_DIR / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES      = len(PRIMARY_LABELS)
label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}
print(f"Classes: {N_CLASSES}")
print(f"train_audio rows: {len(train_meta)}")
print(f"soundscape labeled segments: {len(sc_labels)}")

# %%
# Build training index: train_audio clips (full audio target) +
# soundscape segments (5s target, upsampled 3x for Pantanal domain)

def parse_time(s):
    """Parse 'HH:MM:SS' or 'MM:SS' or seconds-as-string."""
    s = str(s)
    if ":" in s:
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 3:    return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:    return parts[0] * 60 + parts[1]
    try:    return int(float(s))
    except: return 0

def make_target(row):
    y = np.zeros(N_CLASSES, dtype=np.float32)
    p = row.get("primary_label")
    if isinstance(p, str) and p in label_to_idx: y[label_to_idx[p]] = 1.0
    sec = row.get("secondary_labels")
    if isinstance(sec, str) and sec and sec != "[]":
        for s in sec.replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(","):
            s = s.strip()
            if s and s in label_to_idx: y[label_to_idx[s]] = 1.0
    return y

train_audio_rows = []
for _, r in train_meta.iterrows():
    p = TRAIN_AUDIO_DIR / r["filename"]
    if not p.exists(): continue
    train_audio_rows.append({
        "path": str(p), "start_sec": -1, "end_sec": -1,   # -1 = use whole audio
        "target": make_target(r), "primary": r["primary_label"],
        "author": r.get("author", "unknown"), "source": "train_audio",
    })
print(f"train_audio: {len(train_audio_rows)} clips found")

sc_segment_rows = []
sc_lookup = sc_labels.groupby(["filename", "start", "end"])["primary_label"].apply(
    lambda s: ";".join(sorted(set(x for x in s if isinstance(x, str))))
).reset_index()
for _, r in sc_lookup.iterrows():
    p = TRAIN_SC_DIR / r["filename"]
    if not p.exists(): continue
    y = np.zeros(N_CLASSES, dtype=np.float32)
    for lbl in str(r["primary_label"]).split(";"):
        lbl = lbl.strip()
        if lbl and lbl in label_to_idx: y[label_to_idx[lbl]] = 1.0
    sc_segment_rows.append({
        "path": str(p), "start_sec": parse_time(r["start"]),
        "end_sec":   parse_time(r["end"]),
        "target": y, "primary": str(r["primary_label"]).split(";")[0],
        "author": "soundscape", "source": "soundscape",
    })

# Upsample soundscape segments 3x for Pantanal domain weight
sc_segment_rows_x3 = sc_segment_rows * 3
print(f"soundscape segments: {len(sc_segment_rows)} unique, x3 -> {len(sc_segment_rows_x3)}")

all_rows = train_audio_rows + sc_segment_rows_x3
print(f"Total training samples: {len(all_rows)}")

# %%
# Audio loading + crop helpers
def load_audio(path, target_samples=None):
    try:
        wav, sr = torchaudio.load(path)
        if sr != SR: wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        y = wav.squeeze(0).numpy().astype(np.float32)
    except Exception:
        return np.zeros(target_samples or WINDOW_SAMPLES, dtype=np.float32)
    if target_samples is not None:
        if len(y) < target_samples:
            reps = math.ceil(target_samples / max(len(y), 1))
            y = np.tile(y, reps)[:target_samples]
        else:
            y = y[:target_samples]
    return y

def crop_window(y, want_samples=WINDOW_SAMPLES, train=True, seg=None):
    """Crop a `want_samples` window. If seg is given, center on (seg[0], seg[1])."""
    if seg is not None and seg[0] >= 0:
        center = (seg[0] + seg[1]) // 2 * SR
        start  = max(0, center - want_samples // 2)
    elif train and len(y) > want_samples:
        start = random.randint(0, len(y) - want_samples)
    else:
        start = 0
    if len(y) < want_samples:
        # repeat-pad to fill
        reps = math.ceil(want_samples / max(len(y), 1))
        y = np.tile(y, reps)
    return y[start:start + want_samples]

def absmax_normalize(y):
    # Replace NaN/inf first so np.max doesn't propagate NaN
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    m = float(np.max(np.abs(y)))
    return (y / m) if m > 1e-8 else y

# %%
# Mel spectrogram transform on GPU (constructed once, reused for all batches)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    f_min=FMIN, f_max=FMAX, power=2.0, norm="slaney", mel_scale="htk",
).to(DEVICE)
db_transform  = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=TOP_DB).to(DEVICE)

def wav_to_mel_3ch(wav_t):
    """(B, T_samples) tensor -> (B, 3, N_MELS, T_frames) tensor on DEVICE."""
    if wav_t.device.type != DEVICE: wav_t = wav_t.to(DEVICE)
    mel = mel_transform(wav_t)                         # (B, n_mels, T)
    mel = db_transform(mel)
    # 0-1 normalise per sample
    mel_min = mel.amin(dim=(1, 2), keepdim=True)
    mel_max = mel.amax(dim=(1, 2), keepdim=True)
    mel     = (mel - mel_min) / (mel_max - mel_min + 1e-6)
    return mel.unsqueeze(1).repeat(1, 3, 1, 1)         # (B, 3, n_mels, T)

# %%
# SED head -- attention pooling over time frames (Kong et al., 2020)
# Returns clip-level LOGITS (not probs) so we can use BCEWithLogitsLoss,
# which is autocast-safe and NaN-resistant compared to BCELoss.
class SEDHead(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc_att = nn.Linear(in_features, n_classes)
        self.fc_cla = nn.Linear(in_features, n_classes)
        nn.init.xavier_uniform_(self.fc_att.weight); nn.init.zeros_(self.fc_att.bias)
        nn.init.xavier_uniform_(self.fc_cla.weight); nn.init.zeros_(self.fc_cla.bias)

    def forward(self, x):
        # x: (B, C, T) after frequency pooling
        x = x.transpose(1, 2)                                  # (B, T, C)
        att_logits   = torch.tanh(self.fc_att(x))
        att          = torch.softmax(att_logits, dim=1)        # weights over time
        frame_logits = self.fc_cla(x)                          # (B, T, n_classes), raw
        # Attention-weighted average of LOGITS (not sigmoid'd probs).
        # This gives clip_logits in a real-valued range, safe for BCEWithLogitsLoss.
        clip_logits  = (att * frame_logits).sum(dim=1)         # (B, n_classes)
        framewise    = torch.sigmoid(frame_logits)             # for inference / diagnostics
        return clip_logits, framewise


class BirdCNN(nn.Module):
    def __init__(self, backbone_name=BACKBONE, n_classes=N_CLASSES, in_chans=3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, in_chans=in_chans,
            num_classes=0, global_pool="", features_only=False,
        )
        # Find feature dim from a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, N_MELS, 512)
            feat  = self.backbone.forward_features(dummy)   # (1, C, H', W')
            self.feat_dim = feat.shape[1]
        self.head = SEDHead(self.feat_dim, n_classes)

    def forward(self, x):
        feat = self.backbone.forward_features(x)            # (B, C, H', W')
        # Frequency pooling (mean over H') -> (B, C, W')
        feat = feat.mean(dim=2)
        clip, frame = self.head(feat)
        return clip, frame

# %%
# Dataset and collate -- emits raw audio waveforms; mel is computed in train loop
class BirdAudioDataset(torch.utils.data.Dataset):
    def __init__(self, rows, train=True):
        self.rows = rows
        self.train = train

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        seg = (r["start_sec"], r["end_sec"]) if r["start_sec"] >= 0 else None
        y_audio = load_audio(r["path"])
        y_audio = crop_window(y_audio, WINDOW_SAMPLES, train=self.train, seg=seg)
        y_audio = absmax_normalize(y_audio)
        # Length safety + force fresh contiguous copies. crop_window returns a
        # slice of np.tile() output which can be a non-resizable view, and the
        # default DataLoader collate fails with "Trying to resize storage that
        # is not resizable" when it tries to stack such views into a batch.
        if len(y_audio) != WINDOW_SAMPLES:
            y_audio = np.pad(y_audio, (0, max(0, WINDOW_SAMPLES - len(y_audio))))[:WINDOW_SAMPLES]
        audio_out  = np.array(y_audio,  dtype=np.float32)   # always copies
        target_out = np.array(r["target"], dtype=np.float32)
        return audio_out, target_out


def mixup_batch(wavs_t, tgts_t, p=MIXUP_P, weight=MIXUP_WEIGHT):
    """In-batch raw-audio mixup on tensors. Constant blend weight (1st place 2025)."""
    if random.random() > p: return wavs_t, tgts_t
    perm    = torch.randperm(wavs_t.size(0), device=wavs_t.device)
    wavs_m  = weight * wavs_t + (1.0 - weight) * wavs_t[perm]
    # Label union: both clips' positives kept (secondary labels = 1 in 1st place)
    tgts_m  = torch.maximum(tgts_t, tgts_t[perm])
    return wavs_m, tgts_m

# %%
def train_one_fold(fold_idx, train_rows, val_rows, fold_pred_path):
    print(f"\n=== Fold {fold_idx}  train={len(train_rows)}  val={len(val_rows)} ===")
    tr_ds = BirdAudioDataset(train_rows, train=True)
    va_ds = BirdAudioDataset(val_rows,   train=False)

    tr_loader = torch.utils.data.DataLoader(
        tr_ds, batch_size=BATCH_SZ, shuffle=True, num_workers=N_WORKERS,
        pin_memory=True, drop_last=True, persistent_workers=N_WORKERS > 0,
    )
    va_loader = torch.utils.data.DataLoader(
        va_ds, batch_size=BATCH_SZ, shuffle=False, num_workers=N_WORKERS,
        pin_memory=True, persistent_workers=N_WORKERS > 0,
    )

    model = BirdCNN().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(tr_loader),
                                                       eta_min=LR_MIN)
    scaler = torch.cuda.amp.GradScaler()
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0; best_state = None
    val_targets  = np.stack([r["target"] for r in val_rows])

    for epoch in range(EPOCHS):
        # ---- train -----------------------------------------------------------
        model.train(); t0 = time.time(); tr_loss = 0.0; nb = 0
        for wavs_t, tgts_t in tr_loader:
            wavs_t = wavs_t.to(DEVICE, non_blocking=True)
            tgts_t = tgts_t.to(DEVICE, non_blocking=True)
            wavs_t, tgts_t = mixup_batch(wavs_t, tgts_t)
            if LABEL_SMOOTH > 0:
                tgts_t = tgts_t * (1.0 - LABEL_SMOOTH) + LABEL_SMOOTH / N_CLASSES

            # NaN guard on audio (corrupt OGG / mixup edge cases)
            wavs_t = torch.nan_to_num(wavs_t, nan=0.0, posinf=0.0, neginf=0.0)

            opt.zero_grad()
            with torch.cuda.amp.autocast():
                mel       = wav_to_mel_3ch(wavs_t)             # (B, 3, n_mels, T)
                clip_lg, _ = model(mel)                        # logits, autocast-safe
                # BCEWithLogitsLoss is autocast-safe and NaN-resistant
                loss = bce_loss(clip_lg, tgts_t)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tr_loss += loss.item(); nb += 1

        # ---- val -------------------------------------------------------------
        model.eval(); preds = []
        with torch.no_grad():
            for wavs_t, _ in va_loader:
                wavs_t = wavs_t.to(DEVICE, non_blocking=True)
                wavs_t = torch.nan_to_num(wavs_t, nan=0.0, posinf=0.0, neginf=0.0)
                with torch.cuda.amp.autocast():
                    mel       = wav_to_mel_3ch(wavs_t)
                    clip_lg, _ = model(mel)
                # logits -> probs for AUC scoring
                preds.append(torch.sigmoid(clip_lg).float().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        active = val_targets.sum(0) > 0
        try:
            val_auc = roc_auc_score(val_targets[:, active], preds[:, active], average="macro")
        except Exception:
            val_auc = float("nan")

        print(f"  ep {epoch+1:2d}  train={tr_loss/max(nb,1):.4f}  val_auc={val_auc:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.0f}s")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Save best fold predictions for OOF
    if best_state is not None: model.load_state_dict(best_state)
    np.save(fold_pred_path, preds.astype(np.float32))
    print(f"Fold {fold_idx} best_val_auc={best_val_auc:.4f}")
    return model, best_val_auc

# %%
# Validation split: StratifiedKFold by primary_label
all_primaries = np.array([r["primary"] for r in all_rows])
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

fold_models = []
fold_aucs   = []
print(f"\nTraining {N_FOLDS}-fold {BACKBONE} + SED head on {len(all_rows)} samples ...")
for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(all_rows, all_primaries)):
    train_rows = [all_rows[i] for i in tr_idx]
    val_rows   = [all_rows[i] for i in va_idx]
    pred_path  = OUT_DIR / f"fold{fold_idx}_val_preds.npy"
    model, auc = train_one_fold(fold_idx, train_rows, val_rows, pred_path)
    fold_models.append(model); fold_aucs.append(auc)
    torch.save(model.state_dict(), OUT_DIR / f"fold{fold_idx}_best.pth")

print(f"\nOOF val AUC per fold: {[f'{a:.4f}' for a in fold_aucs]}")
print(f"Mean OOF val AUC    : {np.mean(fold_aucs):.4f}")

# %%
# Inference on test soundscapes (or staging fallback)
test_files = sorted((BASE_DIR / "test_soundscapes").glob("*.ogg")) \
    if (BASE_DIR / "test_soundscapes").exists() else []
if not test_files:
    print("test_soundscapes empty -- staging fallback: first 16 train soundscapes")
    test_files = sorted((BASE_DIR / "train_soundscapes").glob("*.ogg"))[:16]
print(f"Test files: {len(test_files)}")

# %%
def infer_soundscape(path, models):
    """Return (12, N_CLASSES) predictions for one 60s soundscape."""
    wav = load_audio(str(path), target_samples=60 * SR)
    win_starts = [i * INFER_STRIDE for i in range(INFER_WIN_PER_SC)]   # 0,5,..,40
    chunks = np.stack([
        absmax_normalize(wav[s * SR:(s + WINDOW_SEC) * SR]) for s in win_starts
    ]).astype(np.float32)
    chunk_t = torch.from_numpy(chunks).to(DEVICE)
    chunk_t = torch.nan_to_num(chunk_t, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            mel = wav_to_mel_3ch(chunk_t)
            preds_per_fold = []
            for m in models:
                m.eval(); clip_lg, _ = m(mel)
                preds_per_fold.append(torch.sigmoid(clip_lg).float().cpu().numpy())
    win_preds = np.mean(preds_per_fold, axis=0)            # (9, N_CLASSES)

    # Map 9 x 20s window predictions back to 12 x 5s output slots via averaging
    out = np.zeros((OUTPUT_SLOTS_PER_SC, N_CLASSES), dtype=np.float32)
    cnt = np.zeros((OUTPUT_SLOTS_PER_SC,), dtype=np.float32)
    for wi, ws in enumerate(win_starts):
        slot_lo = ws // INFER_STRIDE                       # 0..8
        slot_hi = (ws + WINDOW_SEC) // INFER_STRIDE        # 4..12
        for slot in range(slot_lo, slot_hi):
            out[slot] += win_preds[wi]; cnt[slot] += 1
    out /= cnt[:, None]
    return out

# %%
print(f"\nRunning {N_FOLDS}-fold ensemble inference on {len(test_files)} soundscapes ...")
rows = []
t0 = time.time()
for fi, fp in enumerate(test_files):
    probs = infer_soundscape(fp, fold_models)              # (12, N_CLASSES)
    stem  = fp.stem
    for s in range(OUTPUT_SLOTS_PER_SC):
        end_sec = (s + 1) * INFER_STRIDE
        rows.append({"row_id": f"{stem}_{end_sec}",
                     **dict(zip(PRIMARY_LABELS, probs[s]))})
    if (fi + 1) % 20 == 0:
        print(f"  {fi+1}/{len(test_files)} files  {time.time()-t0:.0f}s")
print(f"Inference done: {time.time()-t0:.0f}s")

# %%
pred_df = pd.DataFrame(rows)
prior   = 1.0 / N_CLASSES
in_real = set(pred_df["row_id"]).issubset(set(sample_sub["row_id"])) or len(sample_sub) > 10
if in_real:
    sub = sample_sub[["row_id"]].merge(pred_df, on="row_id", how="left")
    sub[PRIMARY_LABELS] = sub[PRIMARY_LABELS].fillna(prior)
else:
    print("STAGING: writing pred_df directly")
    sub = pred_df[["row_id"] + PRIMARY_LABELS]

assert sub.columns.tolist() == ["row_id"] + PRIMARY_LABELS
sub.to_csv(OUT_DIR / "submission.csv", index=False)
print(f"\nSubmission saved: {sub.shape}")
print(sub.head(3))

# %%
diagnostics = {
    "notebook":     "nb20a",
    "model":        f"{BACKBONE} + SED head",
    "window_sec":   WINDOW_SEC,
    "n_mels":       N_MELS,
    "loss":         "cross-entropy (multi-label, no normalisation)",
    "augmentation": f"raw-audio MixUp p={MIXUP_P} weight={MIXUP_WEIGHT}, label union",
    "n_folds":      N_FOLDS,
    "epochs":       EPOCHS,
    "fold_aucs":    [round(float(a), 4) for a in fold_aucs],
    "mean_val_auc": round(float(np.mean(fold_aucs)), 4),
    "n_train_samples": len(all_rows),
    "n_test_files": len(test_files),
}
with open(OUT_DIR / "diagnostics_nb20a.json", "w") as f:
    json.dump(diagnostics, f, indent=2)
print("Saved diagnostics_nb20a.json")
