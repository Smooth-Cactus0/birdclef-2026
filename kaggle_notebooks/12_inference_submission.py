# %%
# =============================================================================
# BirdCLEF 2026 -- Inference + Submission
# =============================================================================
# Bird pipeline    : EfficientNet-B3  x5 folds (162 Aves classes)
# Non-bird pipeline: ECA-NFNet-L0     x5 folds (72 non-Aves classes)
# Inference        : PyTorch CPU, batch_size=32, fold averaging
# Audio            : torchaudio (C++ backend -- 4-6x faster than librosa)
# Submission       : 234-column CSV aligned to sample_submission.csv
# Internet         : DISABLED (checkpoints loaded from dataset input)
# =============================================================================

# %% [markdown]
# # BirdCLEF 2026 -- Inference Notebook
#
# Loads all trained checkpoints and produces a competition-ready submission.
# No internet required -- checkpoints come from attached Kaggle dataset.

# %%
import os, gc, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import timm
warnings.filterwarnings('ignore')

# %%
# -- Paths ---------------------------------------------------------------------
_input = Path('/kaggle/input')
if not _input.exists():
    print("/kaggle/input not found -- running locally")

BASE_DIR  = (Path('/kaggle/input/competitions/birdclef-2026')
             if Path('/kaggle/input/competitions/birdclef-2026').exists()
             else Path('birdclef-2026'))

# Confirmed mount path: /kaggle/input/datasets/{owner}/{slug}/ on Kaggle 2025+
# Fallback candidates for forward-compatibility
_ckpt_candidates = [
    Path('/kaggle/input/datasets/alexycactus/birdclef26-checkpoints'),
    Path('/kaggle/input/birdclef26-checkpoints'),
    Path('/kaggle/input/datasets/birdclef26-checkpoints'),
]
CKPT_DIR = Path('kaggle_outputs')  # local fallback
for _c in _ckpt_candidates:
    if _c.exists():
        CKPT_DIR = _c
        break

OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
NUM_WORKERS = 0  # CPU submission kernel -- no workers needed

print(f"BASE_DIR  : {BASE_DIR}")
print(f"CKPT_DIR  : {CKPT_DIR}")
print(f"torch     : {torch.__version__}")
print(f"timm      : {timm.__version__}")

# %%
# -- Audio / model constants (must match training exactly) ---------------------
SR         = 32000
N_FFT      = 1024
HOP_LENGTH = 320
N_MELS     = 128
FMIN       = 40
FMAX       = 15000
DURATION   = 5       # seconds per inference chunk
N_SAMPLES  = SR * DURATION
BATCH_SIZE = 32      # larger batch = higher CPU utilisation
DEVICE     = 'cpu'   # submission kernel is CPU-only

# Mel transform built once -- matches librosa params used in training
# norm='slaney' + mel_scale='slaney' replicates librosa's default filter bank
_MEL_TRANSFORM = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=FMIN,
    f_max=FMAX,
    norm='slaney',
    mel_scale='slaney',
    power=2.0,
)

# %%
# -- Label setup ---------------------------------------------------------------
taxonomy       = pd.read_csv(BASE_DIR / 'taxonomy.csv')
label_list     = taxonomy['primary_label'].tolist()
label2idx      = {l: i for i, l in enumerate(label_list)}
NUM_CLASSES    = len(label_list)   # 234

bird_labels    = taxonomy[taxonomy['class_name'] == 'Aves']['primary_label'].tolist()
bird2idx       = {l: i for i, l in enumerate(bird_labels)}
BIRD_CLASSES   = len(bird_labels)  # 162

nonbird_labels = taxonomy[taxonomy['class_name'] != 'Aves']['primary_label'].tolist()
nonbird2idx    = {l: i for i, l in enumerate(nonbird_labels)}
NONBIRD_CLASSES = len(nonbird_labels)  # 72

print(f"Total classes   : {NUM_CLASSES}")
print(f"Bird classes    : {BIRD_CLASSES}")
print(f"Non-bird classes: {NONBIRD_CLASSES}")

# %%
# -- Model definition (identical to training notebooks) -----------------------
class BirdModel(nn.Module):
    """timm backbone, in_chans=1, num_classes variable."""
    def __init__(self, model_name, num_classes, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            in_chans=1, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)


def load_model(ckpt_path, model_name, num_classes):
    """Load a checkpoint, return model in eval mode. Returns None if missing."""
    if not Path(ckpt_path).exists():
        print(f"  WARNING: checkpoint not found -- {ckpt_path}")
        return None
    model = BirdModel(model_name, num_classes, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    return model


# %%
# -- Mel spectrogram helper ----------------------------------------------------
def waveform_to_melspec(wav_1d):
    """
    wav_1d : 1-D float32 tensor of length N_SAMPLES.
    Returns : (1, N_MELS, T) float32 tensor, normalised to [0, 1].
    Replicates librosa power_to_db(S, ref=np.max) + min-max norm.
    """
    S     = _MEL_TRANSFORM(wav_1d.unsqueeze(0))          # (1, N_MELS, T)
    S_db  = 10.0 * torch.log10(S.clamp(min=1e-10))
    S_db  = S_db - S_db.max()                            # ref = max (0 dB at peak)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm.float()


def chunk_soundscape(audio_path):
    """
    Load a long soundscape with torchaudio (C++ backend, ~5x faster than
    librosa/audioread for OGG), slide a 5s window, return list of
    (row_id, mel_tensor) pairs. Pads the last chunk if needed.
    """
    try:
        wav, orig_sr = torchaudio.load(str(audio_path))  # (C, T) float32
        if orig_sr != SR:
            wav = torchaudio.functional.resample(wav, orig_sr, SR)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)               # stereo -> mono
        y = wav.squeeze(0)                                # (T,)
    except Exception as e:
        print(f"  Error loading {Path(audio_path).name}: {e}")
        return []

    stem   = Path(audio_path).stem
    n_full = len(y) // N_SAMPLES
    chunks = []

    for i in range(n_full):
        chunk = y[i * N_SAMPLES:(i + 1) * N_SAMPLES]
        end_s = (i + 1) * DURATION
        mel   = waveform_to_melspec(chunk)
        chunks.append((f"{stem}_{end_s}", mel))

    # Include the trailing partial chunk (padded) if it is > 0.5s long
    remainder = len(y) - n_full * N_SAMPLES
    if remainder > SR // 2:
        pad   = torch.zeros(N_SAMPLES - remainder)
        chunk = torch.cat([y[n_full * N_SAMPLES:], pad])
        end_s = (n_full + 1) * DURATION
        mel   = waveform_to_melspec(chunk)
        chunks.append((f"{stem}_{end_s}", mel))

    return chunks


# %%
# -- Inference helpers ---------------------------------------------------------
@torch.no_grad()
def run_inference(model, chunks, batch_size=BATCH_SIZE):
    """
    Run model over a list of (row_id, mel_tensor) chunks in batches.
    Returns dict {row_id: probs_np_array}.
    """
    results  = {}
    row_ids  = [c[0] for c in chunks]
    tensors  = [c[1] for c in chunks]

    for start in range(0, len(tensors), batch_size):
        batch     = torch.stack(tensors[start:start + batch_size])  # (B,1,128,501)
        logits    = model(batch)
        probs     = torch.sigmoid(logits).numpy()                    # sigmoid -- BCE models
        for rid, p in zip(row_ids[start:start + batch_size], probs):
            results[rid] = p

    return results


def accumulate_fold(fold_preds, new_preds):
    """Add new fold predictions into running sum."""
    for rid, p in new_preds.items():
        if rid in fold_preds:
            fold_preds[rid] += p
        else:
            fold_preds[rid]  = p.copy()


# %%
# %% [markdown]
# ## Load Test Soundscapes

# %%
sample_sub       = pd.read_csv(BASE_DIR / 'sample_submission.csv')
test_soundscapes = sorted((BASE_DIR / 'test_soundscapes').glob('*.ogg'))
print(f"Test soundscapes : {len(test_soundscapes)}")
print(f"Expected rows    : {len(sample_sub)}")

t_start = time.time()

# Pre-chunk all soundscapes once (avoid re-loading per model)
print("\nChunking soundscapes...")
all_chunks = {}   # {filepath: [(row_id, mel_tensor), ...]}
total_chunks = 0
for sf in test_soundscapes:
    chunks = chunk_soundscape(sf)
    if chunks:
        all_chunks[str(sf)] = chunks
        total_chunks += len(chunks)
    print(f"  {sf.name}: {len(chunks)} chunks")

print(f"\nTotal chunks: {total_chunks}")
print(f"Chunking time: {time.time()-t_start:.1f}s")

# %%
# %% [markdown]
# ## Bird Pipeline Inference (EfficientNet-B3, 5 folds)

# %%
bird_sum   = {}   # running sum across folds
bird_folds = 0

for fold in range(1, 6):
    ckpt = CKPT_DIR / f'efficientnet_b3_fold{fold}.pth'
    model = load_model(ckpt, 'efficientnet_b3', BIRD_CLASSES)
    if model is None:
        continue

    t_fold = time.time()
    fold_preds = {}
    for sf_path, chunks in all_chunks.items():
        preds = run_inference(model, chunks)
        for rid, p in preds.items():
            fold_preds[rid] = p

    accumulate_fold(bird_sum, fold_preds)
    bird_folds += 1

    elapsed = time.time() - t_fold
    print(f"  Bird fold {fold}: {len(fold_preds)} rows, {elapsed:.1f}s")

    del model; gc.collect()

# Average across folds
bird_avg = {rid: p / bird_folds for rid, p in bird_sum.items()} if bird_folds > 0 else {}
print(f"\nBird folds used: {bird_folds}")
print(f"Time so far    : {time.time()-t_start:.1f}s")

# %%
# %% [markdown]
# ## Non-Bird Pipeline Inference (ECA-NFNet-L0, 5 folds)

# %%
nonbird_sum   = {}
nonbird_folds = 0

for fold in range(1, 6):
    ckpt  = CKPT_DIR / f'nonbird_fold{fold}.pth'
    model = load_model(ckpt, 'eca_nfnet_l0', NONBIRD_CLASSES)
    if model is None:
        continue

    t_fold = time.time()
    fold_preds = {}
    for sf_path, chunks in all_chunks.items():
        preds = run_inference(model, chunks)
        for rid, p in preds.items():
            fold_preds[rid] = p

    accumulate_fold(nonbird_sum, fold_preds)
    nonbird_folds += 1

    elapsed = time.time() - t_fold
    print(f"  Non-bird fold {fold}: {len(fold_preds)} rows, {elapsed:.1f}s")

    del model; gc.collect()

nonbird_avg = {rid: p / nonbird_folds for rid, p in nonbird_sum.items()} if nonbird_folds > 0 else {}
print(f"\nNon-bird folds used: {nonbird_folds}")
print(f"Time so far        : {time.time()-t_start:.1f}s")

# %%
# %% [markdown]
# ## Build Submission (merge bird + non-bird into 234 columns)

# %%
prior = 1.0 / NUM_CLASSES   # uninformative fallback for missing rows

rows = {}
all_row_ids = set(bird_avg.keys()) | set(nonbird_avg.keys())

for rid in all_row_ids:
    full = np.full(NUM_CLASSES, prior, dtype=np.float32)

    # Fill bird columns (162)
    if rid in bird_avg:
        for i, label in enumerate(bird_labels):
            full[label2idx[label]] = bird_avg[rid][i]

    # Fill non-bird columns (72)
    if rid in nonbird_avg:
        for i, label in enumerate(nonbird_labels):
            full[label2idx[label]] = nonbird_avg[rid][i]

    rows[rid] = full

pred_df           = pd.DataFrame.from_dict(rows, orient='index', columns=label_list)
pred_df.index.name = 'row_id'
pred_df           = pred_df.reset_index()

sub = sample_sub[['row_id']].merge(pred_df, on='row_id', how='left')
sub[label_list] = sub[label_list].fillna(prior)

sub.to_csv(OUTPUT_DIR / 'submission.csv', index=False)

total_time = time.time() - t_start
print(f"Submission saved  : {sub.shape}")
print(f"Total runtime     : {total_time:.1f}s  ({total_time/60:.1f} min)")
print(f"Budget remaining  : {120 - total_time/60:.1f} min")
print(sub.head(3))
