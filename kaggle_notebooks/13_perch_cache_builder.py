# %%
# ============================================================================
# BirdCLEF 2026 — nb13: Perch v2 Feature Extraction + Cache Builder
# ============================================================================
# Extracts 1536-dim embeddings and 234-class logits from frozen Google Perch
# v2 for all training data.  Results are saved to /kaggle/working/perch_cache/
# so they can be uploaded as a dataset and attached to nb14 (head training).
#
# Runtime  : ~3 min (soundscapes) + ~2-3 h (train_audio)  on Kaggle CPU
# Internet : ON  (needed to install onnxruntime wheel from attached dataset)
# Datasets : rishikeshjani/perch-onnx-for-birdclef-2026
# Models   : google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1
# Competition: birdclef-2026
#
# After this notebook finishes:
#   1. Download /kaggle/working/perch_cache/ from the output panel
#   2. Upload as a new Kaggle dataset: alexycactus/birdclef-2026-perch-cache
#   3. Attach that dataset to nb14 for head training
# ============================================================================

# %% [markdown]
# # BirdCLEF 2026 — Perch v2 Cache Builder
#
# Builds a reusable feature cache from Google Perch v2, the SOTA frozen
# audio encoder pretrained on 500M+ bird clips.  Running Perch once and
# caching lets every subsequent head-training experiment (MLP probes, SSM,
# site/hour priors) reload in seconds rather than hours.
#
# ## Cache layout
# ```
# perch_cache/
#   train_sc_meta.parquet       filename, window_idx, row_id, site, hour_utc, has_label
#   train_sc_arrays.npz         scores (N,234), embs (N,1536), labels (N,234)
#   train_audio_meta.parquet    filename, window_idx, primary_label, rating, weight
#   train_audio_arrays.npz      scores (N,234), embs (N,1536), labels (N,234)
# ```

# %%
# -- Step 0: Install onnxruntime wheel (not available via pip on Kaggle) ------
import glob as _glob
import subprocess as _sub
import sys as _sys

_whl_candidates = _glob.glob(
    "/kaggle/input/**/onnxruntime*cp312*x86_64*.whl", recursive=True
)
_whl_candidates = [w for w in _whl_candidates if "gpu" not in w]

if _whl_candidates:
    print(f"Installing onnxruntime from: {_whl_candidates[0]}")
    _sub.check_call([_sys.executable, "-m", "pip", "install",
                     _whl_candidates[0], "--quiet"])
else:
    print("WARNING: No onnxruntime wheel found — falling back to pip install")
    _sub.check_call([_sys.executable, "-m", "pip", "install",
                     "onnxruntime", "--quiet"])

# %%
import gc
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio
import onnxruntime as ort

warnings.filterwarnings("ignore")

print(f"onnxruntime : {ort.__version__}")
print(f"torchaudio  : {torchaudio.__version__}")

# %%
# -- Paths --------------------------------------------------------------------
_input   = Path("/kaggle/input")

BASE_DIR = (Path("/kaggle/input/competitions/birdclef-2026")
            if Path("/kaggle/input/competitions/birdclef-2026").exists()
            else Path("/kaggle/input/birdclef-2026"))

# Perch TF SavedModel — only needed for assets/labels.csv (species vocab)
MODEL_DIR = Path(
    "/kaggle/input/models/google/bird-vocalization-classifier"
    "/tensorflow2/perch_v2_cpu/1"
)

# ONNX model — 150× faster than the TF SavedModel on CPU
_onnx_candidates = [
    Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx"),
    Path("/kaggle/input/perch-onnx-for-birdclef-2026/perch_v2.onnx"),
    Path("/kaggle/input/perch-onnx/perch_v2.onnx"),
]
ONNX_PATH = next((c for c in _onnx_candidates if c.exists()), None)

CACHE_DIR = Path("/kaggle/working/perch_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set True to overwrite existing cache files
FORCE_REBUILD = False

print(f"BASE_DIR   : {BASE_DIR}  (exists={BASE_DIR.exists()})")
print(f"MODEL_DIR  : {MODEL_DIR}  (exists={MODEL_DIR.exists()})")
print(f"ONNX_PATH  : {ONNX_PATH}  (found={ONNX_PATH is not None})")
print(f"CACHE_DIR  : {CACHE_DIR}")

# %%
# -- Constants ----------------------------------------------------------------
PERCH_SR       = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = PERCH_SR * WINDOW_SEC    # 160 000 samples
SC_FILE_SEC    = 60                       # process first 60 s of each soundscape
SC_N_WINDOWS   = SC_FILE_SEC // WINDOW_SEC   # 12 windows per file

BATCH_FILES    = 16    # soundscape files per ONNX batch (→ 192 windows)
AUDIO_BATCH    = 64    # train_audio clips per ONNX batch
IO_WORKERS     = 4     # parallel audio-loading threads
AUDIO_MAX_WINS = 6     # cap windows per train_audio clip (bounds runtime)

FNAME_RE = re.compile(
    r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg"
)


def parse_fname(fname):
    """BirdCLEF 2026 filename → recorder, site, hour_utc."""
    m = FNAME_RE.match(Path(fname).name)
    if m:
        return {
            "recorder" : m.group(1),
            "site"     : m.group(2),
            "hour_utc" : int(m.group(4)[:2]),
        }
    return {"recorder": "unknown", "site": "unknown", "hour_utc": -1}

# %%
# -- Competition data ---------------------------------------------------------
taxonomy          = pd.read_csv(BASE_DIR / "taxonomy.csv")
soundscape_labels = pd.read_csv(BASE_DIR / "train_soundscapes_labels.csv")
train_meta        = pd.read_csv(BASE_DIR / "train.csv")
sample_sub        = pd.read_csv(BASE_DIR / "sample_submission.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES      = len(PRIMARY_LABELS)          # 234
label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}

print(f"Competition classes : {N_CLASSES}")
print(f"Taxonomy species    : {len(taxonomy)}")
print(f"Soundscape label rows: {len(soundscape_labels)}")
print(f"Train audio clips   : {len(train_meta):,}")

# %%
# -- Build soundscape label ground-truth (multi-hot per window) ---------------
def union_labels(series):
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t:
                    out.add(t)
    return sorted(out)


sc = (soundscape_labels
      .groupby(["filename", "start", "end"])["primary_label"]
      .apply(union_labels)
      .reset_index(name="label_list"))

sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
sc["row_id"]  = (sc["filename"].str.replace(".ogg", "", regex=False)
                 + "_" + sc["end_sec"].astype(str))

_meta_cols = sc["filename"].apply(parse_fname).apply(pd.Series)
sc = pd.concat([sc, _meta_cols], axis=1)

# Build row_id → label vector lookup (used later for Y_SC_ALIGNED)
Y_SC_LOOKUP = {}    # row_id → np.array shape (N_CLASSES,)
for _, row in sc.iterrows():
    v = np.zeros(N_CLASSES, dtype=np.float32)
    for lbl in row["label_list"]:
        if lbl in label_to_idx:
            v[label_to_idx[lbl]] = 1.0
    Y_SC_LOOKUP[row["row_id"]] = v

print(f"Unique labeled windows   : {len(sc)}")
print(f"Files represented        : {sc['filename'].nunique()}")
print(f"Active species           : {sum((sum(v) > 0) for v in Y_SC_LOOKUP.values())}")

# %%
# -- ONNX session setup -------------------------------------------------------
assert ONNX_PATH is not None, (
    f"Perch ONNX model not found!\n"
    f"Tried: {[str(c) for c in _onnx_candidates]}\n"
    f"Attach dataset rishikeshjani/perch-onnx-for-birdclef-2026"
)

_so = ort.SessionOptions()
_so.intra_op_num_threads = 4
_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

SESS          = ort.InferenceSession(str(ONNX_PATH), sess_options=_so,
                                     providers=["CPUExecutionProvider"])
ONNX_INPUT    = SESS.get_inputs()[0].name
ONNX_OUT_MAP  = {o.name: i for i, o in enumerate(SESS.get_outputs())}

print(f"ONNX input  : '{ONNX_INPUT}'  shape={SESS.get_inputs()[0].shape}")
print(f"ONNX outputs: {list(ONNX_OUT_MAP.keys())}")
print(f"Perch vocab : {SESS.get_outputs()[ONNX_OUT_MAP['label']].shape[1]} classes")

# %%
# -- Species mapping: competition taxonomy → Perch vocab ----------------------
BC_LABELS_CSV = MODEL_DIR / "assets" / "labels.csv"
assert BC_LABELS_CSV.exists(), (
    f"Perch labels.csv not found at {BC_LABELS_CSV}\n"
    "Attach model google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"
)

bc_labels = pd.read_csv(BC_LABELS_CSV).reset_index().rename(
    columns={"index": "bc_index"}
)
# The scientific-name column is named differently across Perch versions
_sci_col = [c for c in bc_labels.columns if c != "bc_index"][0]
bc_labels = bc_labels.rename(columns={_sci_col: "scientific_name"})

NO_LABEL = len(bc_labels)   # sentinel for unmapped species

# Exact scientific-name match: competition species → Perch index
mapping = (taxonomy
           .merge(bc_labels[["bc_index", "scientific_name"]],
                  on="scientific_name", how="left"))
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
lbl2bc = mapping.set_index("primary_label")["bc_index"]

BC_INDICES    = np.array([int(lbl2bc.get(c, NO_LABEL)) for c in PRIMARY_LABELS],
                          dtype=np.int32)
MAPPED_MASK   = BC_INDICES != NO_LABEL
MAPPED_POS    = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)
UNMAPPED_POS  = np.where(~MAPPED_MASK)[0].astype(np.int32)

print(f"Mapped   : {MAPPED_MASK.sum()} / {N_CLASSES} species (direct Perch logit)")
print(f"Unmapped : {len(UNMAPPED_POS)} species (need genus proxy)")

# %%
# -- Genus-proxy logits for unmapped species ----------------------------------
# For species not in Perch's vocab, average logits of all same-genus Perch entries
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
SCI_NAME_MAP   = taxonomy.set_index("primary_label")["scientific_name"].to_dict()
PROXY_TAXA     = {"Amphibia", "Insecta", "Aves"}

proxy_map = {}   # competition_label_idx → list of bc_indices

for lbl in [PRIMARY_LABELS[i] for i in UNMAPPED_POS]:
    if CLASS_NAME_MAP.get(lbl) not in PROXY_TAXA:
        continue
    sci  = str(SCI_NAME_MAP.get(lbl, ""))
    genus = sci.split()[0] if sci else ""
    if not genus:
        continue
    hits = bc_labels[
        bc_labels["scientific_name"].astype(str)
        .str.match(rf"^{re.escape(genus)}\s", na=False)
    ]
    if len(hits):
        proxy_map[label_to_idx[lbl]] = hits["bc_index"].astype(int).tolist()

print(f"Species with genus proxy  : {len(proxy_map)}")
print(f"Still without any signal  : {len(UNMAPPED_POS) - len(proxy_map)}")
for idx, bc_idxs in list(proxy_map.items())[:5]:
    lbl = PRIMARY_LABELS[idx]
    cls = CLASS_NAME_MAP.get(lbl, "?")
    print(f"  {lbl:15s} ({cls:10s}) ← {len(bc_idxs)} Perch genus match(es)")

# %%
# -- Core inference helpers ---------------------------------------------------
def perch_infer(batch_np):
    """
    batch_np : float32 (B, WINDOW_SAMPLES)
    Returns  : logits (B, N_PERCH), embs (B, 1536)
    """
    outs   = SESS.run(None, {ONNX_INPUT: batch_np})
    logits = outs[ONNX_OUT_MAP["label"]].astype(np.float32)
    embs   = outs[ONNX_OUT_MAP["embedding"]].astype(np.float32)
    return logits, embs


def project_logits(logits):
    """
    Map from Perch's ~10k-class vocab to our 234 competition species.
    Fills proxy scores (max over genus matches) for unmapped species.
    logits : (B, N_PERCH) → returns (B, 234)
    """
    scores = np.zeros((len(logits), N_CLASSES), dtype=np.float32)
    scores[:, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
    for pos_idx, bc_idxs in proxy_map.items():
        bc_arr = np.array(bc_idxs, dtype=np.int32)
        scores[:, pos_idx] = logits[:, bc_arr].max(axis=1)
    return scores

# %%
# -- Audio loaders ------------------------------------------------------------
def load_soundscape_60s(path):
    """Load soundscape OGG, mono 32 kHz, exactly 60 s (pad or truncate)."""
    try:
        wav, sr = torchaudio.load(str(path))
        if sr != PERCH_SR:
            wav = torchaudio.functional.resample(wav, sr, PERCH_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        y = wav.squeeze(0).numpy()
    except Exception as e:
        print(f"\n  ERROR loading {Path(path).name}: {e}")
        return np.zeros(SC_FILE_SEC * PERCH_SR, dtype=np.float32)

    target = SC_FILE_SEC * PERCH_SR
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    return y.astype(np.float32)


def load_and_window_clip(path, max_windows=AUDIO_MAX_WINS):
    """
    Load a train_audio clip → list of 5 s float32 windows.

    Windowing strategy (from design doc):
      < 5 s  → zero-pad to 5 s   (1 window)
      5–10 s → first 5 s only    (1 window)
      > 10 s → non-overlapping 5 s windows (up to max_windows)
    """
    try:
        wav, sr = torchaudio.load(str(path))
        if sr != PERCH_SR:
            wav = torchaudio.functional.resample(wav, sr, PERCH_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        y = wav.squeeze(0).numpy().astype(np.float32)
    except Exception:
        return []

    n = len(y)
    if n <= WINDOW_SAMPLES:                           # < 5 s: pad
        return [np.pad(y, (0, WINDOW_SAMPLES - n))]
    elif n <= 2 * WINDOW_SAMPLES:                     # 5–10 s: first 5 s
        return [y[:WINDOW_SAMPLES]]
    else:                                             # > 10 s: slide
        n_win = min(n // WINDOW_SAMPLES, max_windows)
        return [y[i * WINDOW_SAMPLES:(i + 1) * WINDOW_SAMPLES]
                for i in range(n_win)]

# %%
# ============================================================================
# STAGE 1 — Train soundscapes
# ============================================================================
SC_META_PATH   = CACHE_DIR / "train_sc_meta.parquet"
SC_ARRAYS_PATH = CACHE_DIR / "train_sc_arrays.npz"

if not FORCE_REBUILD and SC_META_PATH.exists() and SC_ARRAYS_PATH.exists():
    print("Soundscape cache already exists — skipping (set FORCE_REBUILD=True to redo).")
    _arr = np.load(str(SC_ARRAYS_PATH))
    print(f"  scores={_arr['scores'].shape}  embs={_arr['embs'].shape}  "
          f"labels={_arr['labels'].shape}")
else:
    sc_files = sorted((BASE_DIR / "train_soundscapes").glob("*.ogg"))
    n_sc     = len(sc_files) * SC_N_WINDOWS
    print(f"\nProcessing {len(sc_files)} soundscape files  →  {n_sc} windows")
    print(f"Batch size: {BATCH_FILES} files = {BATCH_FILES * SC_N_WINDOWS} windows/ONNX call\n")

    sc_scores_all = np.zeros((n_sc, N_CLASSES), dtype=np.float32)
    sc_embs_all   = np.zeros((n_sc, 1536),      dtype=np.float32)
    sc_meta_rows  = []

    t0 = time.time()
    wr = 0

    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        # Pre-queue the very first batch
        future_audio = pool.submit(
            lambda ps: [load_soundscape_60s(p) for p in ps],
            sc_files[:BATCH_FILES]
        )

        for batch_start in range(0, len(sc_files), BATCH_FILES):
            batch_paths = sc_files[batch_start:batch_start + BATCH_FILES]
            batch_audio = future_audio.result()

            # Kick off I/O for the next batch while ONNX runs on this one
            next_start = batch_start + BATCH_FILES
            if next_start < len(sc_files):
                next_paths = sc_files[next_start:next_start + BATCH_FILES]
                future_audio = pool.submit(
                    lambda ps: [load_soundscape_60s(p) for p in ps],
                    next_paths
                )

            batch_n = len(batch_audio)
            # Stack: (batch_n, SC_N_WINDOWS, WINDOW_SAMPLES) → flatten first two dims
            x = np.stack([
                a.reshape(SC_N_WINDOWS, WINDOW_SAMPLES) for a in batch_audio
            ]).reshape(-1, WINDOW_SAMPLES)          # (batch_n * 12, 160000)

            logits, embs = perch_infer(x)
            scores = project_logits(logits)

            # Metadata for each window
            for bi, path in enumerate(batch_paths):
                stem = path.stem
                meta = parse_fname(path.name)
                for wi in range(SC_N_WINDOWS):
                    end_s = (wi + 1) * WINDOW_SEC
                    sc_meta_rows.append({
                        "filename"  : path.name,
                        "window_idx": wi,
                        "start_sec" : wi * WINDOW_SEC,
                        "end_sec"   : end_s,
                        "row_id"    : f"{stem}_{end_s}",
                        "site"      : meta["site"],
                        "hour_utc"  : meta["hour_utc"],
                        "recorder"  : meta["recorder"],
                    })

            rows_this = batch_n * SC_N_WINDOWS
            sc_scores_all[wr:wr + rows_this] = scores
            sc_embs_all  [wr:wr + rows_this] = embs
            wr += rows_this

            elapsed = time.time() - t0
            spd = wr / elapsed if elapsed > 0 else 0
            eta = (n_sc - wr) / spd if spd > 0 else 0
            print(f"  [{wr:4d}/{n_sc}]  {elapsed:5.0f}s elapsed  "
                  f"{spd:.1f} win/s  ETA {eta:.0f}s", end="\r")

    print(f"\nSoundscape extraction complete: {wr} windows  "
          f"({time.time() - t0:.1f}s total)")

    # Build aligned label matrix
    sc_meta_df = pd.DataFrame(sc_meta_rows)
    Y_SC_ALIGNED = np.stack([
        Y_SC_LOOKUP.get(row["row_id"], np.zeros(N_CLASSES, dtype=np.float32))
        for row in sc_meta_rows
    ])
    sc_meta_df["has_label"] = (Y_SC_ALIGNED.sum(1) > 0).astype(np.int8)

    sc_meta_df.to_parquet(SC_META_PATH, index=False)
    np.savez_compressed(
        str(SC_ARRAYS_PATH),
        scores=sc_scores_all[:wr],
        embs=sc_embs_all[:wr],
        labels=Y_SC_ALIGNED,
    )
    print(f"Saved: {SC_META_PATH}")
    print(f"Saved: {SC_ARRAYS_PATH}  "
          f"scores={sc_scores_all[:wr].shape}  embs={sc_embs_all[:wr].shape}")
    labeled_wins = int((Y_SC_ALIGNED.sum(1) > 0).sum())
    print(f"  Labeled windows: {labeled_wins} / {wr}  "
          f"({100*labeled_wins/wr:.1f}%)")

    del sc_scores_all, sc_embs_all, Y_SC_ALIGNED
    gc.collect()

# %%
# ============================================================================
# STAGE 2 — Train audio clips
# ============================================================================
AUDIO_META_PATH   = CACHE_DIR / "train_audio_meta.parquet"
AUDIO_ARRAYS_PATH = CACHE_DIR / "train_audio_arrays.npz"

if not FORCE_REBUILD and AUDIO_META_PATH.exists() and AUDIO_ARRAYS_PATH.exists():
    print("\nTrain-audio cache already exists — skipping (set FORCE_REBUILD=True to redo).")
    _arr = np.load(str(AUDIO_ARRAYS_PATH))
    print(f"  scores={_arr['scores'].shape}  embs={_arr['embs'].shape}  "
          f"labels={_arr['labels'].shape}")
else:
    AUDIO_DIR = BASE_DIR / "train_audio"

    # Pre-compute per-row window counts so we can preallocate
    print("\nPre-scanning train_audio clip lengths to estimate window count...")

    # We use a generous upper bound: assume max AUDIO_MAX_WINS per clip
    max_possible_windows = len(train_meta) * AUDIO_MAX_WINS
    audio_scores_all = np.zeros((max_possible_windows, N_CLASSES), dtype=np.float32)
    audio_embs_all   = np.zeros((max_possible_windows, 1536),      dtype=np.float32)
    audio_labels_all = np.zeros((max_possible_windows, N_CLASSES), dtype=np.float32)
    audio_meta_rows  = []

    # Label setup: secondary_labels can be a stringified list or semicolons
    def parse_secondary(sec_str):
        if not isinstance(sec_str, str) or not sec_str.strip():
            return []
        # Handle both "['label1', 'label2']" and "label1;label2"
        labels = re.findall(r"[a-z0-9]+[0-9]", sec_str)
        return [l for l in labels if l in label_to_idx]

    def build_label_vec(primary, secondary, pri_weight=1.0, sec_weight=0.5):
        v = np.zeros(N_CLASSES, dtype=np.float32)
        if primary in label_to_idx:
            v[label_to_idx[primary]] = pri_weight
        for lbl in secondary:
            if lbl in label_to_idx:
                v[label_to_idx[lbl]] = max(v[label_to_idx[lbl]], sec_weight)
        return v

    t0  = time.time()
    wr  = 0
    n   = len(train_meta)

    print(f"Processing {n:,} train_audio clips  (max_wins/clip={AUDIO_MAX_WINS})")
    print(f"Batch size: {AUDIO_BATCH} clips per ONNX call\n")

    batch_rows = list(train_meta.itertuples(index=False))
    batch_idx  = 0

    def _load_one(row):
        path = AUDIO_DIR / row.filename
        return row, load_and_window_clip(path)

    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        while batch_idx < len(batch_rows):
            batch_slice = batch_rows[batch_idx:batch_idx + AUDIO_BATCH]
            loaded = list(pool.map(_load_one, batch_slice))
            batch_idx += AUDIO_BATCH

            # Flatten windows + metadata
            windows_flat = []
            meta_flat    = []
            for row, wins in loaded:
                if not wins:
                    continue
                rating = float(row.rating) if pd.notna(row.rating) else 0.0
                weight = max(0.1, rating / 5.0)  # floor at 0.1 for unrated
                sec    = parse_secondary(getattr(row, "secondary_labels", ""))
                lbl_vec = build_label_vec(row.primary_label, sec)

                for wi, w in enumerate(wins):
                    windows_flat.append(w)
                    meta_flat.append({
                        "filename"        : row.filename,
                        "window_idx"      : wi,
                        "start_sec"       : wi * WINDOW_SEC,
                        "end_sec"         : (wi + 1) * WINDOW_SEC,
                        "primary_label"   : row.primary_label,
                        "secondary_labels": getattr(row, "secondary_labels", ""),
                        "rating"          : rating,
                        "weight"          : weight,
                    })
                    audio_labels_all[wr + len(windows_flat) - 1] = lbl_vec

            if not windows_flat:
                continue

            x = np.stack(windows_flat)          # (B, WINDOW_SAMPLES)
            logits, embs = perch_infer(x)
            scores = project_logits(logits)

            rows_this = len(windows_flat)
            audio_scores_all[wr:wr + rows_this] = scores
            audio_embs_all  [wr:wr + rows_this] = embs
            audio_meta_rows.extend(meta_flat)
            wr += rows_this

            elapsed = time.time() - t0
            pct = batch_idx / len(batch_rows) * 100
            spd = wr / elapsed if elapsed > 0 else 0
            clips_per_sec = batch_idx / elapsed if elapsed > 0 else 0
            eta = (len(batch_rows) - batch_idx) / clips_per_sec if clips_per_sec > 0 else 0
            print(
                f"  clips {batch_idx:6d}/{len(batch_rows):6d}  "
                f"({pct:5.1f}%)  wins={wr:7d}  "
                f"{spd:.1f}w/s  "
                f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.0f}min",
                end="\r"
            )

    print(f"\n\nTrain-audio extraction complete: {wr} windows  "
          f"({time.time() - t0:.1f}s total)")

    audio_meta_df = pd.DataFrame(audio_meta_rows)
    audio_meta_df.to_parquet(AUDIO_META_PATH, index=False)
    np.savez_compressed(
        str(AUDIO_ARRAYS_PATH),
        scores=audio_scores_all[:wr],
        embs=audio_embs_all[:wr],
        labels=audio_labels_all[:wr],
    )
    print(f"Saved: {AUDIO_META_PATH}")
    print(f"Saved: {AUDIO_ARRAYS_PATH}  "
          f"scores={audio_scores_all[:wr].shape}  embs={audio_embs_all[:wr].shape}")

    del audio_scores_all, audio_embs_all, audio_labels_all
    gc.collect()

# %%
# -- Summary ------------------------------------------------------------------
print("\n" + "=" * 60)
print("CACHE SUMMARY")
print("=" * 60)
for f in sorted(CACHE_DIR.glob("*")):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:<35s}  {size_mb:6.1f} MB")

print(f"\nCache location : {CACHE_DIR}")
print(
    "\nNext step:\n"
    "  1. Go to the notebook output panel → Download perch_cache/\n"
    "  2. Create a new Kaggle dataset: alexycactus/birdclef-2026-perch-cache\n"
    "  3. Upload the contents of perch_cache/\n"
    "  4. Attach that dataset to nb14 for head training"
)
