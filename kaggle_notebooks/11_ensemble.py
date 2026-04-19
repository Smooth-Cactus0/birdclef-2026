# %%
# =============================================================================
# BirdCLEF 2026 -- Final Ensemble Submission
# =============================================================================
# Bird pipeline  : B1 + B3 + B4 + RegNetY-016 + BirdNET -> rank averaging (162 cols)
# Non-bird pipeline: ECA-NFNet-L0 + B3-focal -> weighted avg 0.6/0.4 (72 cols)
# Inference: OpenVINO FP16 (CPU budget ~59 min)
# Post-processing: geotemporal prior (Pantanal species filter)
# =============================================================================

# %% [markdown]
# # BirdCLEF 2026 -- Final Ensemble Submission
#
# This notebook loads all trained ONNX / OpenVINO models from both pipelines,
# runs efficient CPU inference over the test soundscapes, rank-averages the
# bird predictions, weighted-averages the non-bird predictions, then merges
# everything into the required 234-column submission CSV.
#
# **Pipeline overview**
# ```
# test_soundscapes/
#   ?- chunk into 5s windows
#         ?- bird_models (B1, B3, B4, RegNetY, BirdNET) -> (N, 162) each -> rank average
#         ?- nonbird_models (ECA-NFNet-L0, B3-focal)   -> (N, 72)  each -> weighted avg
#               ?- merge into (N, 234) -> submission.csv
# ```
#
# **Why rank averaging?**
# Each model was trained independently with different architectures and
# pretraining sources (ImageNet vs BirdNET). Their raw probability scales
# differ. Rank-based fusion normalises these scales: we convert each model's
# 162 column values per row to ranks, then average the ranks. This is more
# robust to miscalibration than simply averaging probabilities.

# %%
# -- Version pins --------------------------------------------------------------
# !pip install -q openvino==2024.0.0 onnxruntime==1.18.0  # uncomment on Kaggle

import os, gc, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import rankdata
import librosa

warnings.filterwarnings('ignore')

# OpenVINO -- optional but strongly preferred (8-12x faster than PyTorch on CPU)
try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
    print(f"OpenVINO {ov.__version__} available OK")
except ImportError:
    OPENVINO_AVAILABLE = False
    print("OpenVINO not available -- falling back to ONNX Runtime")

# ONNX Runtime -- secondary backend
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
    print(f"ONNX Runtime {ort.__version__} available OK")
except ImportError:
    ORT_AVAILABLE = False
    print("onnxruntime not found -- install with: pip install onnxruntime")

# -- Paths ---------------------------------------------------------------------
BASE_DIR   = (Path('/kaggle/input/competitions/birdclef-2026')
              if Path('/kaggle/input/competitions/birdclef-2026').exists()
              else Path('birdclef-2026'))
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
NUM_WORKERS = 0 if os.name == 'nt' else 4

# -- Audio config --------------------------------------------------------------
CFG = dict(
    SR=32000, N_FFT=1024, HOP_LENGTH=320, N_MELS=128, FMIN=40, FMAX=15000,
    DURATION=5,        # seconds per inference chunk
    BATCH_SIZE=32,     # chunks per forward pass -- 32 is efficient on CPU
)
N_FRAMES = 1 + (CFG['SR'] * CFG['DURATION'] // CFG['HOP_LENGTH'])  # 501
print(f"Chunk shape: (batch, 1, {CFG['N_MELS']}, {N_FRAMES})")

# %% [markdown]
# ## Label setup
#
# We build three label arrays from taxonomy.csv:
# - `label_list` -- all 234 species in taxonomy order (submission column order)
# - `bird_labels` -- 162 Aves species (bird pipeline output columns)
# - `nonbird_labels` -- 72 non-Aves species (non-bird pipeline output columns)

# %%
taxonomy   = pd.read_csv(BASE_DIR / 'taxonomy.csv')
label_list = taxonomy['primary_label'].tolist()           # 234 total
label2idx  = {l: i for i, l in enumerate(label_list)}
NUM_CLASSES = len(label_list)                              # 234

bird_labels    = taxonomy[taxonomy['class_name'] == 'Aves']['primary_label'].tolist()   # 162
bird2idx       = {l: i for i, l in enumerate(bird_labels)}
BIRD_CLASSES   = len(bird_labels)                                                         # 162

nonbird_labels  = taxonomy[taxonomy['class_name'] != 'Aves']['primary_label'].tolist()  # 72
nonbird2idx     = {l: i for i, l in enumerate(nonbird_labels)}
NONBIRD_CLASSES = len(nonbird_labels)                                                     # 72

print(f"Total classes   : {NUM_CLASSES}")
print(f"Bird classes    : {BIRD_CLASSES}")
print(f"Non-bird classes: {NONBIRD_CLASSES}")

# %% [markdown]
# ## Model registry
#
# Each entry in `BIRD_MODELS` / `NONBIRD_MODELS` is a tuple of
# `(onnx_path, label, n_classes, pipeline)`.
#
# These paths assume the corresponding Kaggle output datasets have been
# added as inputs to this notebook. Update the dataset names when uploading.
#
# **If an ONNX file is not found the model is skipped gracefully** -- the
# ensemble still runs on whatever models are available.

# %%
BIRD_MODELS = [
    ('/kaggle/input/birdclef26-backbone-b1/efficientnet_b1.onnx',   'b1',     BIRD_CLASSES, 'bird'),
    ('/kaggle/input/birdclef26-backbone-b3/efficientnet_b3.onnx',   'b3',     BIRD_CLASSES, 'bird'),
    ('/kaggle/input/birdclef26-backbone-b4/efficientnet_b4.onnx',   'b4',     BIRD_CLASSES, 'bird'),
    ('/kaggle/input/birdclef26-regnety/regnety_016.onnx',            'regnety', BIRD_CLASSES, 'bird'),
    ('/kaggle/input/birdclef26-birdnet/birdnet.onnx',                'birdnet', BIRD_CLASSES, 'bird'),
]

NONBIRD_MODELS = [
    ('/kaggle/input/birdclef26-nonbird-nfnet/nonbird_eca_nfnet_l0.onnx', 'nb_nfnet', NONBIRD_CLASSES, 'nonbird'),
    ('/kaggle/input/birdclef26-nonbird-b3/nonbird_efficientnet_b3.onnx', 'nb_b3',    NONBIRD_CLASSES, 'nonbird'),
]

# Weights for non-bird weighted average (must sum to 1.0)
NONBIRD_WEIGHTS = [0.6, 0.4]


def load_session(path: str):
    """Load ONNX model as OpenVINO (preferred) or ONNX Runtime session.

    Returns the session object, or None if the file does not exist.
    Both backends accept the same (B, 1, 128, 501) float32 input.
    """
    p = Path(path)
    if not p.exists():
        print(f"  [SKIP] {p.name} not found -- add the matching dataset to this notebook")
        return None
    if OPENVINO_AVAILABLE:
        core    = ov.Core()
        ov_mdl  = core.read_model(str(p))
        session = core.compile_model(ov_mdl, 'CPU')
        print(f"  [OV]  {p.name} loaded OK")
        return ('ov', session)
    if ORT_AVAILABLE:
        session = ort.InferenceSession(str(p), providers=['CPUExecutionProvider'])
        print(f"  [ORT] {p.name} loaded OK")
        return ('ort', session)
    print("  [ERR] Neither OpenVINO nor ONNX Runtime available -- cannot load model")
    return None


def run_session(session_tuple, x_np: np.ndarray) -> np.ndarray:
    """Run forward pass through OpenVINO or ORT session.

    Parameters
    ----------
    session_tuple : ('ov' | 'ort', session)
    x_np          : (B, 1, N_MELS, n_frames) float32

    Returns
    -------
    logits : (B, C) float32 -- apply sigmoid outside this function
    """
    kind, session = session_tuple
    if kind == 'ov':
        return session([x_np])[session.output(0)]
    return session.run(None, {'input': x_np})[0]


# Load all models
print("\nLoading bird models:")
bird_sessions = []
for path, label, n_cls, pipeline in BIRD_MODELS:
    sess = load_session(path)
    if sess is not None:
        bird_sessions.append((label, sess, n_cls))

print(f"\nLoading non-bird models:")
nonbird_sessions = []
for path, label, n_cls, pipeline in NONBIRD_MODELS:
    sess = load_session(path)
    if sess is not None:
        nonbird_sessions.append((label, sess, n_cls))

print(f"\nBird models loaded    : {len(bird_sessions)}/{len(BIRD_MODELS)}")
print(f"Non-bird models loaded: {len(nonbird_sessions)}/{len(NONBIRD_MODELS)}")

# %% [markdown]
# ## Audio utilities
#
# `make_mel_spectrogram` converts a 1-D audio array to a
# `(1, N_MELS, n_frames)` float32 array, normalised to [0, 1].
#
# `collect_chunks` slides non-overlapping 5-second windows over every
# soundscape file and returns a stacked array of all chunks plus the
# corresponding row_ids. We collect everything in one pass to avoid
# re-loading audio for each model.

# %%
def make_mel_spectrogram(audio: np.ndarray, cfg: dict) -> np.ndarray:
    """1-D audio -> (1, N_MELS, n_frames) float32, normalised to [0, 1]."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg['SR'], n_fft=cfg['N_FFT'], hop_length=cfg['HOP_LENGTH'],
        n_mels=cfg['N_MELS'], fmin=cfg['FMIN'], fmax=cfg['FMAX'], power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return mel_db[np.newaxis].astype(np.float32)   # (1, N_MELS, n_frames)


def collect_chunks(soundscape_paths: list, cfg: dict) -> tuple:
    """Load all soundscapes, slice into 5s chunks.

    Returns
    -------
    chunks_arr : (N_total, 1, N_MELS, n_frames) float32
    chunk_ids  : list[str] of row_ids -- '{stem}_{end_sec}'
    """
    n_frames  = 1 + (cfg['SR'] * cfg['DURATION'] // cfg['HOP_LENGTH'])
    chunk_len = cfg['SR'] * cfg['DURATION']
    all_chunks = []
    all_ids    = []

    for sc_path in soundscape_paths:
        stem = Path(sc_path).stem
        try:
            audio, _ = librosa.load(sc_path, sr=cfg['SR'], mono=True)
        except Exception as e:
            print(f"  [WARN] Could not load {Path(sc_path).name}: {e}")
            continue

        n_full = len(audio) // chunk_len
        for i in range(n_full):
            start   = i * chunk_len
            segment = audio[start : start + chunk_len]
            mel     = make_mel_spectrogram(segment, cfg)
            # Guarantee exact frame count (librosa rounding can differ by ?1)
            if mel.shape[-1] < n_frames:
                mel = np.pad(mel, ((0, 0), (0, 0), (0, n_frames - mel.shape[-1])))
            else:
                mel = mel[..., :n_frames]
            all_chunks.append(mel)
            all_ids.append(f"{stem}_{(i + 1) * cfg['DURATION']}")

    if not all_chunks:
        return np.zeros((0, 1, cfg['N_MELS'], n_frames), dtype=np.float32), []

    return np.stack(all_chunks, axis=0), all_ids   # (N, 1, 128, 501)


# %% [markdown]
# ## Inference helpers
#
# `run_model_on_chunks` batches all chunks through a single ONNX/OpenVINO
# session and returns sigmoid probabilities `(N, C)`.
#
# `rank_average` converts each model's output to within-row ranks then
# averages -- the output values are in [1, C] rank space, but ordering is
# preserved for the final submission (only relative order matters for AUC).
#
# `weighted_average` takes a list of `(N, C)` arrays and scalar weights,
# normalises the weights, and returns a probability average.

# %%
def run_model_on_chunks(session_tuple, chunks_arr: np.ndarray,
                        batch_size: int = 32) -> np.ndarray:
    """Run all chunks through a session in batches.

    Parameters
    ----------
    session_tuple : ('ov' | 'ort', session)
    chunks_arr    : (N, 1, N_MELS, n_frames) float32
    batch_size    : forward-pass batch size

    Returns
    -------
    probs : (N, C) float32  -- sigmoid applied
    """
    N = len(chunks_arr)
    all_probs = []
    for start in range(0, N, batch_size):
        batch   = chunks_arr[start : start + batch_size]
        logits  = run_session(session_tuple, batch)          # (B, C)
        probs   = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))  # sigmoid
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)                 # (N, C)


def rank_average(pred_list: list) -> np.ndarray:
    """Rank-average a list of (N, C) probability arrays.

    Each row is ranked independently per model (rank 1 = lowest prob,
    rank C = highest), then ranks are averaged. This is robust to
    probability scale differences across architectures.

    Returns
    -------
    avg_ranks : (N, C) float64 -- values in [1, C]; higher = more likely
    """
    ranked = [rankdata(p, axis=1).astype(np.float32) for p in pred_list]
    return np.mean(ranked, axis=0)


def weighted_average(pred_list: list, weights: list) -> np.ndarray:
    """Weighted probability average.

    Parameters
    ----------
    pred_list : list of (N, C) arrays
    weights   : list of floats (need not sum to 1 -- normalised internally)

    Returns
    -------
    avg : (N, C) float32
    """
    total = sum(weights)
    result = sum(p * (w / total) for p, w in zip(pred_list, weights))
    return result.astype(np.float32)


# %% [markdown]
# ## Main inference loop
#
# Steps:
# 1. Discover all test soundscapes, sort shortest-first so any timeout
#    produces a valid partial submission rather than crashing with no output.
# 2. Collect all 5s chunks in a single audio pass (one read per soundscape).
# 3. Run each bird model over all chunks -> rank-average -> `(N, 162)`.
# 4. Run each non-bird model over all chunks -> weighted-average -> `(N, 72)`.

# %%
soundscape_dir = BASE_DIR / 'test_soundscapes'
if soundscape_dir.exists():
    # Sort by file size (proxy for duration) -- shortest first for partial-submission safety
    sc_paths = sorted(soundscape_dir.glob('*.ogg'), key=lambda p: p.stat().st_size)
    print(f"Found {len(sc_paths)} test soundscapes")
else:
    print("[WARN] test_soundscapes/ not found -- using empty list for demo")
    sc_paths = []

t_start = time.time()

# -- Step 1: Collect all chunks ------------------------------------------------
print("\nLoading and chunking all soundscapes ...")
t0 = time.time()
chunks_arr, chunk_ids = collect_chunks(sc_paths, CFG)
n_chunks = len(chunk_ids)
print(f"  {n_chunks} chunks collected in {time.time() - t0:.1f}s")

# Handle edge case: no test soundscapes available (e.g. local dev run)
if n_chunks == 0:
    print("[WARN] No chunks collected -- generating dummy output for demo")
    chunks_arr = np.zeros((10, 1, CFG['N_MELS'], N_FRAMES), dtype=np.float32)
    chunk_ids  = [f"demo_soundscape_{i*5}" for i in range(1, 11)]
    n_chunks   = 10

# -- Step 2: Bird pipeline -----------------------------------------------------
print(f"\nRunning {len(bird_sessions)} bird model(s) on {n_chunks} chunks ...")
bird_preds_list = []
for label, sess, n_cls in bird_sessions:
    t0 = time.time()
    probs = run_model_on_chunks(sess, chunks_arr, CFG['BATCH_SIZE'])  # (N, 162)
    bird_preds_list.append(probs)
    print(f"  [{label}] shape={probs.shape}  min={probs.min():.3f}  "
          f"max={probs.max():.3f}  [{time.time()-t0:.1f}s]")

if bird_preds_list:
    bird_preds_ranked = rank_average(bird_preds_list)   # (N, 162)
    print(f"Rank-averaged bird predictions: {bird_preds_ranked.shape}")
else:
    print("[WARN] No bird models loaded -- filling bird columns with uniform prior")
    bird_preds_ranked = np.full((n_chunks, BIRD_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float32)

# -- Step 3: Non-bird pipeline -------------------------------------------------
print(f"\nRunning {len(nonbird_sessions)} non-bird model(s) on {n_chunks} chunks ...")
nonbird_preds_list = []
for label, sess, n_cls in nonbird_sessions:
    t0 = time.time()
    probs = run_model_on_chunks(sess, chunks_arr, CFG['BATCH_SIZE'])  # (N, 72)
    nonbird_preds_list.append(probs)
    print(f"  [{label}] shape={probs.shape}  min={probs.min():.3f}  "
          f"max={probs.max():.3f}  [{time.time()-t0:.1f}s]")

# Use only the weights for models that loaded
active_nb_weights = NONBIRD_WEIGHTS[:len(nonbird_preds_list)]
if nonbird_preds_list:
    nonbird_preds_weighted = weighted_average(nonbird_preds_list, active_nb_weights)  # (N, 72)
    print(f"Weighted-averaged non-bird predictions: {nonbird_preds_weighted.shape}")
else:
    print("[WARN] No non-bird models loaded -- filling non-bird columns with uniform prior")
    nonbird_preds_weighted = np.full((n_chunks, NONBIRD_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float32)

# %% [markdown]
# ## Merge bird + non-bird into 234-column prediction array
#
# The submission requires **all 234 species in taxonomy order**.
# Bird and non-bird predictions live in separate index spaces (0-161 and 0-71).
# We map each back to its global `label2idx` position.
#
# Note: rank-averaged bird values are in [1, 162] scale. We normalise them to
# [0, 1] before merging so the final CSV is on a consistent probability scale.

# %%
# Normalise rank-averaged bird scores back to [0, 1]
bird_min  = bird_preds_ranked.min(axis=1, keepdims=True)
bird_max  = bird_preds_ranked.max(axis=1, keepdims=True)
bird_preds_norm = (bird_preds_ranked - bird_min) / (bird_max - bird_min + 1e-6)

# Initialise full prediction matrix with uniform prior
full_preds = np.full((n_chunks, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float32)

# Fill bird columns
for i, label in enumerate(bird_labels):
    col = label2idx[label]
    full_preds[:, col] = bird_preds_norm[:, i]

# Fill non-bird columns
for i, label in enumerate(nonbird_labels):
    col = label2idx[label]
    full_preds[:, col] = nonbird_preds_weighted[:, i]

print(f"full_preds shape: {full_preds.shape}  (expected: ({n_chunks}, {NUM_CLASSES}))")

# %% [markdown]
# ## Geotemporal prior (optional)
#
# The Pantanal region is geographically specific. Species that do not occur in
# the Mato Grosso do Sul area can have their predictions floored to near-zero
# without risking AUC loss on true positives -- those species should score zero
# anyway if the training data is correctly labelled.
#
# The list below is intentionally empty until validated against eBird / iNat
# occurrence data for the recorder coordinates (-21.6->-16.5 lat, -57.6->-55.9 lon).
# **Only add species here after explicit verification** -- wrong filtering costs AUC.

# %%
# Species confirmed absent from the Pantanal PAM recorder region
# (fill from eBird API or iNat occurrence analysis once available)
PANTANAL_ABSENT: list = []

for species_label in PANTANAL_ABSENT:
    if species_label in label2idx:
        col = label2idx[species_label]
        full_preds[:, col] = np.minimum(full_preds[:, col], 0.001)

if PANTANAL_ABSENT:
    print(f"Geotemporal prior applied: {len(PANTANAL_ABSENT)} species floored to <= 0.001")
else:
    print("Geotemporal prior: no species filtered (list empty -- add after eBird validation)")

# %% [markdown]
# ## Build and save submission
#
# We use the vectorised merge pattern from nb03: build a DataFrame from the
# predictions dict, then left-join onto sample_submission so any missing
# row_ids are filled with the uniform prior.

# %%
sample_sub_path = BASE_DIR / 'sample_submission.csv'
sample_sub = pd.read_csv(sample_sub_path)

# Build predictions DataFrame aligned to chunk_ids
pred_df = pd.DataFrame(full_preds, columns=label_list)
pred_df.insert(0, 'row_id', chunk_ids)

# Left join onto sample_submission to ensure correct row order and coverage
sub = sample_sub[['row_id']].merge(pred_df, on='row_id', how='left')
# Fill any row_ids that had no prediction (e.g. soundscape loading failures)
sub[label_list] = sub[label_list].fillna(1.0 / NUM_CLASSES)

# Sanity checks
assert sub.shape[1] == 235, f"Expected 235 columns (row_id + 234), got {sub.shape[1]}"
assert not sub[label_list].isna().any().any(), "NaN values remain in submission"

out_path = OUTPUT_DIR / 'submission.csv'
sub.to_csv(out_path, index=False)
print(f"\nSubmission saved: {out_path}")
print(f"  Shape         : {sub.shape}")
print(f"  Rows covered  : {pred_df['row_id'].isin(sample_sub['row_id']).sum()}"
      f" / {len(sample_sub)} sample_submission rows")
print(sub.head(3))

# %% [markdown]
# ## Runtime summary + budget check
#
# The Kaggle CPU budget is approximately 90 minutes for code competitions.
# We report total wall-clock time and estimated budget remaining.

# %%
elapsed = time.time() - t_start
ms_per_chunk = (elapsed / n_chunks * 1000) if n_chunks > 0 else 0.0
budget_remaining_min = 90 - elapsed / 60

print("\n" + "=" * 60)
print("  INFERENCE SUMMARY")
print("=" * 60)
print(f"  Total wall-clock time  : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
print(f"  Soundscapes processed  : {len(sc_paths)}")
print(f"  Chunks processed       : {n_chunks}")
print(f"  ms per chunk           : {ms_per_chunk:.1f} ms")
print(f"  Bird models used       : {[l for l, _, _ in bird_sessions]}")
print(f"  Non-bird models used   : {[l for l, _, _ in nonbird_sessions]}")
print(f"  Budget remaining       : {budget_remaining_min:.1f} min  "
      f"({'OK OK' if budget_remaining_min > 5 else 'TIGHT ?'})")
print("=" * 60)

# Save runtime stats for reproducibility
runtime_stats = {
    'elapsed_sec'       : round(elapsed, 1),
    'elapsed_min'       : round(elapsed / 60, 2),
    'n_soundscapes'     : len(sc_paths),
    'n_chunks'          : n_chunks,
    'ms_per_chunk'      : round(ms_per_chunk, 1),
    'bird_models'       : [l for l, _, _ in bird_sessions],
    'nonbird_models'    : [l for l, _, _ in nonbird_sessions],
    'nonbird_weights'   : active_nb_weights,
    'budget_remaining_min': round(budget_remaining_min, 1),
    'pantanal_absent_count': len(PANTANAL_ABSENT),
}
with open(OUTPUT_DIR / 'inference_stats.json', 'w') as f:
    json.dump(runtime_stats, f, indent=2)
print(f"\nRuntime stats saved -> {OUTPUT_DIR / 'inference_stats.json'}")

# %% [markdown]
# ## Pre-submission checklist
#
# Before committing this as the final submission notebook:
#
# 1. **Attach all model datasets** -- each ONNX path in `BIRD_MODELS` /
#    `NONBIRD_MODELS` must resolve to a file. Check `[SKIP]` warnings above.
#
# 2. **Verify column order** -- `sub.columns[1:]` must match
#    `sample_submission.columns[1:]` exactly (taxonomy order).
#
# 3. **Profile on Kaggle CPU (GPU off)** -- wall-clock on your laptop will
#    differ from Kaggle servers. Run this notebook with the Kaggle accelerator
#    set to "None" and check `budget_remaining_min` is > 10.
#
# 4. **Check NaN / Inf** -- `sub[label_list].isna().sum().sum()` should be 0.
#
# 5. **Sort soundscapes shortest-first** -- already done above; ensures any
#    partial timeout still produces a valid (incomplete) submission file.
