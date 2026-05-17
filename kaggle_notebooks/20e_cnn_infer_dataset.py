# %%
# ============================================================================
# BirdCLEF 2026 -- nb20e: dataset-sourced inference for 5-fold CNN ensemble
# ============================================================================
# nb20b/c/d all failed on submission scoring with "Notebook Timeout".
# nb20d disproved the fp16/timm-tag hypotheses (it ran clean on staging
# at 5.2s but still failed scoring). The remaining structural difference
# vs nb19a (which scores fine) is that nb20b/c/d use kernel_sources to
# mount fold checkpoints from prior training kernels.
#
# Hypothesis tested by nb20e: the scoring environment does not support
# kernel_sources the same way regular runs do. The standard 2024/2025
# winners' topology is train kernel -> publish as Dataset -> inference
# kernel mounts the Dataset.
#
# nb20e copies nb20d byte-for-byte EXCEPT the metadata: dataset_sources
# = ["alexycactus/birdclef-2026-cnn-fold-checkpoints"] replaces the two
# kernel_sources entries.
#
# Same fp32 inference, same Top-K postproc, same diagnostic preamble.
#
# Internet : OFF
# GPU      : ON  (used if available, falls back to CPU cleanly)
# ============================================================================

# %%
import json
import time
import glob as _g
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import timm

warnings.filterwarnings("ignore")

# --- Diagnostic preamble (every line FLUSHED so the submission log captures
# it even if we crash three lines later) ---
import os as _os, sys as _sys, platform as _platform
print("=" * 60, flush=True)
print(f"nb20d startup  ({_platform.node()}, {_platform.system()})", flush=True)
print(f"  cwd          : {_os.getcwd()}", flush=True)
print(f"  python       : {_sys.version.split()[0]}", flush=True)
print(f"  /kaggle/input contents:", flush=True)
try:
    for _p in sorted(_os.listdir('/kaggle/input')):
        print(f"    {_p}", flush=True)
except Exception as _e:
    print(f"    (error listing: {_e})", flush=True)
print("=" * 60, flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch={torch.__version__}  torchaudio={torchaudio.__version__}  "
      f"timm={timm.__version__}", flush=True)
print(f"device={DEVICE}  "
      f"GPU={torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'n/a'}",
      flush=True)

# %%
BASE_DIR = (Path("/kaggle/input/competitions/birdclef-2026")
            if Path("/kaggle/input/competitions/birdclef-2026").exists()
            else Path("/kaggle/input/birdclef-2026"))
OUT_DIR  = Path("/kaggle/working"); OUT_DIR.mkdir(exist_ok=True)
print(f"BASE_DIR exists = {BASE_DIR.exists()}")

# %%
# Audio config -- MUST match nb20a training params
SR             = 32_000
WINDOW_SEC     = 20
WINDOW_SAMPLES = SR * WINDOW_SEC
INFER_STRIDE   = 5
INFER_WIN_PER_SC = 9
OUTPUT_SLOTS_PER_SC = 12
N_MELS         = 224
N_FFT          = 4096
HOP_LENGTH     = 1252
FMIN           = 0
FMAX           = 16_000
TOP_DB         = 80.0

BACKBONE   = "tf_efficientnet_b0_ns"   # plain name, no HF Hub tag
N_CLASSES  = 234
TOP_K      = 1
FILES_PER_BATCH = 4          # 4 files * 9 chunks = 36 chunks per GPU forward
IO_WORKERS = 4
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# %%
# Discover fold checkpoints from kernel_sources
print("\nSearching for fold checkpoints under /kaggle/input/ ...")
_hits = sorted(_g.glob("/kaggle/input/**/fold*_best.pth", recursive=True))
fold_ckpts = {}
for p in _hits:
    nm  = Path(p).name
    idx = int(nm.replace("fold", "").replace("_best.pth", ""))
    if idx not in fold_ckpts:
        fold_ckpts[idx] = p
        print(f"  fold {idx}: {p}")
assert fold_ckpts, "No fold checkpoints found -- kernel_sources not mounted?"

# %%
sample_sub = pd.read_csv(BASE_DIR / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == N_CLASSES

# %%
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    f_min=FMIN, f_max=FMAX, power=2.0, norm="slaney", mel_scale="htk",
).to(DEVICE)
db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=TOP_DB).to(DEVICE)

def wav_to_mel_3ch(wav_t):
    """fp32 mel computation (autocast disabled) -> (B, 3, n_mels, T)."""
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
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=in_chans,
            num_classes=0, global_pool="", features_only=False,
        )
        with torch.no_grad():
            feat = self.backbone.forward_features(torch.zeros(1, in_chans, N_MELS, 512))
            self.feat_dim = feat.shape[1]
        self.head = SEDHead(self.feat_dim, n_classes)

    def forward(self, x):
        feat = self.backbone.forward_features(x).mean(dim=2)
        return self.head(feat)

# %%
# Load all fold models -- always fp32, no .half() (causes CPU conv issues)
fold_models = {}
for fi in sorted(fold_ckpts.keys()):
    m = BirdCNN().to(DEVICE)
    m.load_state_dict(torch.load(fold_ckpts[fi], map_location=DEVICE))
    m.eval()
    fold_models[fi] = m
    print(f"Loaded fold {fi}")
ensemble_models = [fold_models[k] for k in sorted(fold_models.keys())]
print(f"Ensemble size: {len(ensemble_models)} folds (precision: fp32)")

# Use all available CPU cores when running CPU-side (mel transform, audio decode)
import os as _os
_cpu_n = _os.cpu_count() or 4
torch.set_num_threads(_cpu_n)
print(f"torch.num_threads = {torch.get_num_threads()}  (cpu_count={_cpu_n})")

# %%
def load_audio_60s(path):
    try:
        wav, sr = torchaudio.load(str(path))
        if sr != SR: wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        y = wav.squeeze(0).numpy().astype(np.float32)
    except Exception as e:
        print(f"  load error {Path(path).name}: {e}")
        return np.zeros(60 * SR, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    target = 60 * SR
    return np.pad(y, (0, max(0, target - len(y))))[:target]


def absmax_normalize(y):
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    m = float(np.max(np.abs(y)))
    return (y / m) if m > 1e-8 else y


def file_to_chunks(wav):
    """60 s -> (9, WINDOW_SAMPLES) absmax-normalised float32."""
    win_starts = [i * INFER_STRIDE for i in range(INFER_WIN_PER_SC)]
    chunks = np.stack([
        absmax_normalize(wav[s * SR:(s + WINDOW_SEC) * SR]) for s in win_starts
    ]).astype(np.float32)
    return np.ascontiguousarray(chunks)


def infer_chunks(chunks_np, models):
    """(N, WINDOW_SAMPLES) -> (N, N_CLASSES) ensembled sigmoid probs (fp32)."""
    chunk_t = torch.from_numpy(chunks_np).to(DEVICE)
    chunk_t = torch.nan_to_num(chunk_t, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.inference_mode():
        mel = wav_to_mel_3ch(chunk_t)             # already fp32
        preds = None
        for m in models:
            clip_lg, _ = m(mel)
            p = torch.sigmoid(clip_lg).float()
            preds = p if preds is None else preds + p
        preds = preds / len(models)
    return preds.cpu().numpy()


def windows_to_slots(win_preds_per_file):
    """Map (n_files, 9, N_CLASSES) windowed preds -> (n_files, 12, N_CLASSES)
    via the framewise overlap averaging used in nb20a training."""
    n_files = win_preds_per_file.shape[0]
    out = np.zeros((n_files, OUTPUT_SLOTS_PER_SC, N_CLASSES), dtype=np.float32)
    cnt = np.zeros(OUTPUT_SLOTS_PER_SC, dtype=np.float32)
    win_starts = [i * INFER_STRIDE for i in range(INFER_WIN_PER_SC)]
    for wi, ws in enumerate(win_starts):
        lo, hi = ws // INFER_STRIDE, (ws + WINDOW_SEC) // INFER_STRIDE
        for slot in range(lo, hi):
            out[:, slot, :] += win_preds_per_file[:, wi, :]
            cnt[slot] += 1
    out /= cnt[None, :, None]
    return out

# %%
# Locate test files (with staging fallback)
test_dir = BASE_DIR / "test_soundscapes"
test_files = sorted(test_dir.glob("*.ogg")) if test_dir.exists() else []
if not test_files:
    print("test_soundscapes empty -- staging fallback: first 16 train soundscapes")
    test_files = sorted((BASE_DIR / "train_soundscapes").glob("*.ogg"))[:16]
print(f"Test files: {len(test_files)}")

# %%
print(f"\nPipelined inference -- batch={FILES_PER_BATCH} files, "
      f"workers={IO_WORKERS}, models={len(ensemble_models)} ...", flush=True)

t0 = time.time()
rows = []
io_pool = ThreadPoolExecutor(max_workers=IO_WORKERS)

# Prefetch the first batch
def _load_batch(paths):
    return [load_audio_60s(p) for p in paths]

future = io_pool.submit(_load_batch, test_files[:FILES_PER_BATCH])

n_done = 0
for batch_start in range(0, len(test_files), FILES_PER_BATCH):
    batch_paths = test_files[batch_start:batch_start + FILES_PER_BATCH]
    batch_audio = future.result()
    nxt_start = batch_start + FILES_PER_BATCH
    if nxt_start < len(test_files):
        future = io_pool.submit(
            _load_batch, test_files[nxt_start:nxt_start + FILES_PER_BATCH]
        )

    # Build (n_files * 9, WINDOW_SAMPLES) chunks
    chunks_per_file = np.stack([file_to_chunks(w) for w in batch_audio])
    n_files = chunks_per_file.shape[0]
    flat_chunks = chunks_per_file.reshape(n_files * INFER_WIN_PER_SC, WINDOW_SAMPLES)
    flat_preds  = infer_chunks(flat_chunks, ensemble_models)
    win_preds   = flat_preds.reshape(n_files, INFER_WIN_PER_SC, N_CLASSES)
    slot_preds  = windows_to_slots(win_preds)            # (n_files, 12, N_CLASSES)

    for fi, fp in enumerate(batch_paths):
        stem = fp.stem
        for s in range(OUTPUT_SLOTS_PER_SC):
            end_sec = (s + 1) * INFER_STRIDE
            rows.append({
                "row_id": f"{stem}_{end_sec}",
                "_stem":  stem,
                **dict(zip(PRIMARY_LABELS, slot_preds[fi, s])),
            })
    n_done += n_files
    if n_done % 50 < FILES_PER_BATCH or n_done == len(test_files):
        rate = n_done / max(time.time() - t0, 1e-3)
        eta  = (len(test_files) - n_done) / max(rate, 1e-3)
        print(f"  {n_done}/{len(test_files)} files  elapsed={time.time()-t0:.0f}s  "
              f"rate={rate:.1f} files/s  ETA={eta:.0f}s", flush=True)

io_pool.shutdown(wait=False)
print(f"Inference total: {time.time() - t0:.1f}s "
      f"({(time.time() - t0) / max(len(test_files), 1) * 1000:.0f} ms/file)")

# %%
# Top-K per-file post-processing
pred_df = pd.DataFrame(rows)
stems   = pred_df["_stem"].to_numpy()
probs   = pred_df[PRIMARY_LABELS].to_numpy()

probs_post = probs.copy()
for f in np.unique(stems):
    mask = stems == f
    block = probs[mask]
    topk_per_class = np.sort(block, axis=0)[-TOP_K:].mean(axis=0)
    probs_post[mask] = block * topk_per_class[None, :]
print(f"pre  postproc  mean={probs.mean():.5f}  max={probs.max():.5f}")
print(f"post postproc  mean={probs_post.mean():.5f}  max={probs_post.max():.5f}")

# %%
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
print(f"Submission saved: {sub.shape}")

# %%
diagnostics = {
    "notebook":     "nb20e_infer_dataset",
    "model":        f"{BACKBONE} + SED head, fp32 inference",
    "device":        DEVICE,
    "ensemble_size": len(ensemble_models),
    "fold_indices_loaded": sorted(fold_ckpts.keys()),
    "files_per_batch":     FILES_PER_BATCH,
    "io_workers":          IO_WORKERS,
    "n_test_files":        len(test_files),
    "inference_time_s":    round(time.time() - t0, 1),
}
with open(OUT_DIR / "diagnostics_nb20e.json", "w") as f:
    json.dump(diagnostics, f, indent=2)
print("Saved diagnostics_nb20e.json")
print(sub.head(3))
