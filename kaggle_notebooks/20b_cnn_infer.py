# %%
# ============================================================================
# BirdCLEF 2026 -- nb20b: CNN SED B0 5-fold ensemble INFERENCE (no training)
# ============================================================================
# Loads the 5 fold checkpoints produced by:
#   alexycactus/birdclef-2026-cnn-sed-b0-run1  (folds 0, 1)
#   alexycactus/birdclef-2026-cnn-sed-b0-run2  (folds 2, 3, 4)
# Runs inference on test_soundscapes (with staging fallback), applies top-K
# per-file post-processing (2nd-place-2025 trick, +0.005 LB on Perch+MLP),
# writes submission.csv.
#
# Internet : OFF  (timm pretrained=False; no downloads, no pip installs)
# GPU      : ON   (NvidiaTeslaT4)
# Time     : ~30 min for 5-fold ensemble inference on the real test set
# ============================================================================

# %%
import json
import time
import glob as _g
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import timm

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
OUT_DIR  = Path("/kaggle/working"); OUT_DIR.mkdir(exist_ok=True)
print(f"BASE_DIR exists = {BASE_DIR.exists()}")

# %%
# Audio config -- MUST match nb20a (training-time params)
SR             = 32_000
WINDOW_SEC     = 20
WINDOW_SAMPLES = SR * WINDOW_SEC
INFER_STRIDE   = 5
INFER_WIN_PER_SC = 9        # (60 - 20) / 5 + 1
OUTPUT_SLOTS_PER_SC = 12
N_MELS         = 224
N_FFT          = 4096
HOP_LENGTH     = 1252
FMIN           = 0
FMAX           = 16_000
TOP_DB         = 80.0

BACKBONE  = "tf_efficientnet_b0.ns_jft_in1k"
N_CLASSES = 234
TOP_K     = 1               # 2nd-place-2025 post-processing default
SEED      = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# %%
# Discover fold checkpoints from mounted kernel_sources
print("\nSearching for fold checkpoints under /kaggle/input/ ...")
_ckpt_paths = sorted(_g.glob("/kaggle/input/**/fold*_best.pth", recursive=True))
print(f"Found {len(_ckpt_paths)} candidate checkpoints:")
fold_ckpts = {}
for p in _ckpt_paths:
    name = Path(p).name                             # 'fold3_best.pth'
    idx  = int(name.replace("fold", "").replace("_best.pth", ""))
    if idx not in fold_ckpts:                        # prefer the first hit per fold
        fold_ckpts[idx] = p
        print(f"  fold {idx}: {p}")
assert len(fold_ckpts) >= 1, "No fold checkpoints found -- kernel_sources not mounted?"

# %%
sample_sub = pd.read_csv(BASE_DIR / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == N_CLASSES

# %%
# Mel transform on GPU, ALWAYS in fp32 to avoid fp16 FFT/log10 NaN landmines.
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    f_min=FMIN, f_max=FMAX, power=2.0, norm="slaney", mel_scale="htk",
).to(DEVICE)
db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=TOP_DB).to(DEVICE)

def wav_to_mel_3ch(wav_t):
    if wav_t.device.type != DEVICE: wav_t = wav_t.to(DEVICE)
    with torch.cuda.amp.autocast(enabled=False):
        wav_t = wav_t.float()
        mel = mel_transform(wav_t)
        mel = db_transform(mel)
        mel = torch.nan_to_num(mel, nan=-80.0, posinf=0.0, neginf=-80.0)
        mel_min = mel.amin(dim=(1, 2), keepdim=True)
        mel_max = mel.amax(dim=(1, 2), keepdim=True)
        mel     = (mel - mel_min) / (mel_max - mel_min + 1e-6)
        mel     = torch.nan_to_num(mel, nan=0.0, posinf=1.0, neginf=0.0)
    return mel.unsqueeze(1).repeat(1, 3, 1, 1)

# %%
# Model -- MUST mirror nb20a so the saved state_dict aligns
class SEDHead(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc_att = nn.Linear(in_features, n_classes)
        self.fc_cla = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        att        = torch.softmax(torch.tanh(self.fc_att(x)), dim=1)
        frame_lg   = self.fc_cla(x)
        clip_lg    = (att * frame_lg).sum(dim=1)
        return clip_lg, torch.sigmoid(frame_lg)


class BirdCNN(nn.Module):
    def __init__(self, backbone_name=BACKBONE, n_classes=N_CLASSES, in_chans=3):
        super().__init__()
        # pretrained=False keeps this run internet-free; weights are loaded
        # afterwards from the mounted fold checkpoints.
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=in_chans,
            num_classes=0, global_pool="", features_only=False,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, N_MELS, 512)
            feat  = self.backbone.forward_features(dummy)
            self.feat_dim = feat.shape[1]
        self.head = SEDHead(self.feat_dim, n_classes)

    def forward(self, x):
        feat = self.backbone.forward_features(x).mean(dim=2)
        return self.head(feat)

# %%
# Load every fold checkpoint into its own model
fold_models = {}
for fi in sorted(fold_ckpts.keys()):
    m = BirdCNN().to(DEVICE)
    state = torch.load(fold_ckpts[fi], map_location=DEVICE)
    m.load_state_dict(state)
    m.eval()
    fold_models[fi] = m
    print(f"Loaded fold {fi} from {Path(fold_ckpts[fi]).name}")
print(f"\nEnsemble size: {len(fold_models)} folds")
ensemble_models = [fold_models[k] for k in sorted(fold_models.keys())]

# %%
def load_audio_60s(path, target_sec=60):
    try:
        wav, sr = torchaudio.load(str(path))
        if sr != SR: wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        y = wav.squeeze(0).numpy().astype(np.float32)
    except Exception as e:
        print(f"  load error {Path(path).name}: {e}")
        return np.zeros(target_sec * SR, dtype=np.float32)
    target = target_sec * SR
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.pad(y, (0, max(0, target - len(y))))[:target]

def absmax_normalize(y):
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    m = float(np.max(np.abs(y)))
    return (y / m) if m > 1e-8 else y

# %%
def infer_soundscape(path, models):
    wav = load_audio_60s(path)
    win_starts = [i * INFER_STRIDE for i in range(INFER_WIN_PER_SC)]   # 0,5,..,40
    chunks = np.stack([
        absmax_normalize(wav[s * SR:(s + WINDOW_SEC) * SR]) for s in win_starts
    ]).astype(np.float32)
    chunk_t = torch.from_numpy(np.ascontiguousarray(chunks)).to(DEVICE)
    chunk_t = torch.nan_to_num(chunk_t, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            mel = wav_to_mel_3ch(chunk_t)
            preds_per_fold = []
            for m in models:
                clip_lg, _ = m(mel)
                preds_per_fold.append(torch.sigmoid(clip_lg).float().cpu().numpy())
    win_preds = np.mean(preds_per_fold, axis=0)   # (9, N_CLASSES)

    out = np.zeros((OUTPUT_SLOTS_PER_SC, N_CLASSES), dtype=np.float32)
    cnt = np.zeros((OUTPUT_SLOTS_PER_SC,), dtype=np.float32)
    for wi, ws in enumerate(win_starts):
        slot_lo = ws // INFER_STRIDE
        slot_hi = (ws + WINDOW_SEC) // INFER_STRIDE
        for slot in range(slot_lo, slot_hi):
            out[slot] += win_preds[wi]
            cnt[slot] += 1
    out /= cnt[:, None]
    return out

# %%
# Locate test soundscapes (with staging fallback)
test_dir = BASE_DIR / "test_soundscapes"
test_files = sorted(test_dir.glob("*.ogg")) if test_dir.exists() else []
if not test_files:
    print("test_soundscapes empty -- staging fallback: first 16 train soundscapes")
    test_files = sorted((BASE_DIR / "train_soundscapes").glob("*.ogg"))[:16]
print(f"Test files: {len(test_files)}")

# %%
print(f"\nRunning {len(ensemble_models)}-fold ensemble inference ...", flush=True)
rows = []
t0 = time.time()
for fi, fp in enumerate(test_files):
    probs = infer_soundscape(fp, ensemble_models)   # (12, N_CLASSES)
    stem  = fp.stem
    for s in range(OUTPUT_SLOTS_PER_SC):
        end_sec = (s + 1) * INFER_STRIDE
        rows.append({
            "row_id":  f"{stem}_{end_sec}",
            "_stem":   stem,
            **dict(zip(PRIMARY_LABELS, probs[s])),
        })
    if (fi + 1) % 50 == 0:
        print(f"  {fi+1}/{len(test_files)} files  {time.time()-t0:.0f}s", flush=True)
print(f"Inference done: {time.time()-t0:.0f}s")

# %%
# Top-K per-file post-processing (2nd place 2025, +0.005 on nb19a)
pred_df = pd.DataFrame(rows)
stems   = pred_df["_stem"].to_numpy()
probs   = pred_df[PRIMARY_LABELS].to_numpy()

print(f"\nApplying top-K post-processing (K={TOP_K}) ...")
probs_post = probs.copy()
for f in np.unique(stems):
    mask = stems == f
    block = probs[mask]
    topk_per_class = np.sort(block, axis=0)[-TOP_K:].mean(axis=0)
    probs_post[mask] = block * topk_per_class[None, :]
print(f"  pre-postproc  mean={probs.mean():.5f}  max={probs.max():.5f}")
print(f"  post-postproc mean={probs_post.mean():.5f}  max={probs_post.max():.5f}")

# %%
# Two submissions saved -- primary is the top-K version
pred_df_post = pred_df[["row_id"]].copy()
pred_df_post[PRIMARY_LABELS] = probs_post
prior   = 1.0 / N_CLASSES
in_real = set(pred_df_post["row_id"]).issubset(set(sample_sub["row_id"])) or len(sample_sub) > 10
if in_real:
    sub = sample_sub[["row_id"]].merge(pred_df_post, on="row_id", how="left")
    sub[PRIMARY_LABELS] = sub[PRIMARY_LABELS].fillna(prior)
else:
    print("STAGING: writing pred_df directly")
    sub = pred_df_post[["row_id"] + PRIMARY_LABELS]

assert sub.columns.tolist() == ["row_id"] + PRIMARY_LABELS
sub.to_csv(OUT_DIR / "submission.csv", index=False)
print(f"\nSubmission saved: {sub.shape}")

# Also save raw (no postproc) for A/B comparison if we ever need it
pred_df_raw = pred_df[["row_id"]].copy()
pred_df_raw[PRIMARY_LABELS] = probs
if in_real:
    sub_raw = sample_sub[["row_id"]].merge(pred_df_raw, on="row_id", how="left")
    sub_raw[PRIMARY_LABELS] = sub_raw[PRIMARY_LABELS].fillna(prior)
else:
    sub_raw = pred_df_raw[["row_id"] + PRIMARY_LABELS]
sub_raw.to_csv(OUT_DIR / "submission_no_postproc.csv", index=False)

# %%
diagnostics = {
    "notebook":     "nb20b_infer",
    "model":        f"{BACKBONE} + SED head",
    "ensemble_size": len(ensemble_models),
    "fold_indices_loaded": sorted(fold_ckpts.keys()),
    "window_sec":   WINDOW_SEC,
    "n_mels":       N_MELS,
    "top_k":        TOP_K,
    "n_test_files": len(test_files),
    "inference_time_s": round(time.time() - t0, 1),
}
with open(OUT_DIR / "diagnostics_nb20b.json", "w") as f:
    json.dump(diagnostics, f, indent=2)
print("\nSaved diagnostics_nb20b.json")
print(sub.head(3))
