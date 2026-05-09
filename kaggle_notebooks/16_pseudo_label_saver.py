# %%
# ============================================================================
# BirdCLEF 2026 -- nb16 Pseudo-Label Saver
# ============================================================================
# One-time cost kernel (~8h, CPU, internet=ON)
# Runs Perch ONNX on ALL files in unlabeled_soundscapes/.
# Saves to /kaggle/working/:
#   embs_ul.npy   -- (N, 1536) float32 Perch embeddings
#   scores_ul.npy -- (N, 234)  float32 projected logits
#   meta_ul.csv   -- (N, 6)    filename, window_idx, end_sec, row_id, site, hour_utc
#
# Published as Kaggle kernel output -> mounted as kernel_sources in nb16a/b/c.
#
# Inputs:
#   - competitions/birdclef-2026
#   - datasets/rishikeshjani/perch-onnx-for-birdclef-2026
#   - models/google/bird-vocalization-classifier/.../perch_v2_cpu/1
#
# Internet : ON   (onnxruntime wheel install)
# GPU      : OFF  (CPU-only ONNX session)
# ============================================================================

# %%
import glob as _glob
import subprocess as _sub
import sys as _sys

_whl = [w for w in _glob.glob("/kaggle/input/**/onnxruntime*cp312*x86_64*.whl",
                              recursive=True) if "gpu" not in w]
if _whl:
    print(f"Installing onnxruntime: {_whl[0]}")
    _sub.check_call([_sys.executable, "-m", "pip", "install", _whl[0], "--quiet"])
else:
    print("No bundled wheel; falling back to PyPI")
    _sub.check_call([_sys.executable, "-m", "pip", "install", "onnxruntime", "--quiet"])

# %%
import gc
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio
import onnxruntime as ort

warnings.filterwarnings("ignore")

print(f"torchaudio  : {torchaudio.__version__}")
print(f"onnxruntime : {ort.__version__}")

# %%
BASE_DIR = (Path("/kaggle/input/competitions/birdclef-2026")
            if Path("/kaggle/input/competitions/birdclef-2026").exists()
            else Path("/kaggle/input/birdclef-2026"))
MODEL_DIR = Path(
    "/kaggle/input/models/google/bird-vocalization-classifier"
    "/tensorflow2/perch_v2_cpu/1"
)
_onnx_candidates = [
    Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx"),
    Path("/kaggle/input/perch-onnx-for-birdclef-2026/perch_v2.onnx"),
]
ONNX_PATH    = next((p for p in _onnx_candidates if p.exists()), None)
TRAIN_SC_DIR = BASE_DIR / "train_soundscapes"
OUT_DIR      = Path("/kaggle/working"); OUT_DIR.mkdir(exist_ok=True)

print(f"BASE_DIR exists     = {BASE_DIR.exists()}")
print(f"ONNX found          = {ONNX_PATH is not None}")
print(f"TRAIN_SC_DIR exists = {TRAIN_SC_DIR.exists()}")

# %%
PERCH_SR       = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = PERCH_SR * WINDOW_SEC
PERCH_BATCH    = 192   # windows per ONNX call (same as nb13-nb15)

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
def parse_fname(name):
    m = FNAME_RE.match(Path(name).name)
    if m: return {"site": m.group(2), "hour_utc": int(m.group(4)[:2])}
    return {"site": "unknown", "hour_utc": -1}

# %%
assert ONNX_PATH is not None, "Perch ONNX not found — check dataset sources"
_so = ort.SessionOptions()
_so.intra_op_num_threads = 4
_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
SESS = ort.InferenceSession(str(ONNX_PATH), sess_options=_so,
                            providers=["CPUExecutionProvider"])
ONNX_INPUT   = SESS.get_inputs()[0].name
ONNX_OUT_MAP = {o.name: i for i, o in enumerate(SESS.get_outputs())}
print(f"Perch ONNX loaded: {ONNX_PATH.name}")

# %%
# Load taxonomy to build the logit projection (mapped + proxy species)
taxonomy   = pd.read_csv(BASE_DIR / "taxonomy.csv")
sample_sub = pd.read_csv(BASE_DIR / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES      = len(PRIMARY_LABELS)
label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}

BC_LABELS_CSV = MODEL_DIR / "assets" / "labels.csv"
bc_labels = pd.read_csv(BC_LABELS_CSV).reset_index().rename(columns={"index": "bc_index"})
_sci_col  = [c for c in bc_labels.columns if c != "bc_index"][0]
bc_labels = bc_labels.rename(columns={_sci_col: "scientific_name"})
NO_LABEL  = len(bc_labels)

mapping = taxonomy.merge(bc_labels[["bc_index", "scientific_name"]],
                         on="scientific_name", how="left")
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
lbl2bc = mapping.set_index("primary_label")["bc_index"]

BC_INDICES    = np.array([int(lbl2bc.get(c, NO_LABEL)) for c in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK   = BC_INDICES != NO_LABEL
MAPPED_POS    = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)
UNMAPPED_POS  = np.where(~MAPPED_MASK)[0].astype(np.int32)

CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
SCI_NAME_MAP   = taxonomy.set_index("primary_label")["scientific_name"].to_dict()
PROXY_TAXA     = {"Amphibia", "Insecta", "Aves"}

proxy_map = {}
for lbl in [PRIMARY_LABELS[i] for i in UNMAPPED_POS]:
    if CLASS_NAME_MAP.get(lbl) not in PROXY_TAXA:
        continue
    sci   = str(SCI_NAME_MAP.get(lbl, ""))
    genus = sci.split()[0] if sci else ""
    if not genus:
        continue
    hits = bc_labels[bc_labels["scientific_name"].astype(str).str.match(
        rf"^{re.escape(genus)}\s", na=False)]
    if len(hits):
        proxy_map[label_to_idx[lbl]] = hits["bc_index"].astype(int).tolist()

print(f"N_CLASSES={N_CLASSES}  mapped={MAPPED_MASK.sum()}  proxy={len(proxy_map)}")

# %%
def load_audio_any(path):
    """Load OGG of any length -> (n_full_windows, WINDOW_SAMPLES) float32.

    Variable-length PAM files: split into as many complete 5s windows as fit.
    At least 1 window is always returned (padded if needed).
    """
    try:
        wav, sr = torchaudio.load(str(path))
        if sr != PERCH_SR:
            wav = torchaudio.functional.resample(wav, sr, PERCH_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        y = wav.squeeze(0).numpy()
    except Exception as e:
        print(f"  load error {Path(path).name}: {e}")
        return np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
    n_full = max(1, len(y) // WINDOW_SAMPLES)
    padded = np.pad(y, (0, max(0, n_full * WINDOW_SAMPLES - len(y))))
    return padded[:n_full * WINDOW_SAMPLES].reshape(n_full, WINDOW_SAMPLES).astype(np.float32)


def perch_infer(batch_np):
    outs = SESS.run(None, {ONNX_INPUT: batch_np})
    return (outs[ONNX_OUT_MAP["label"]].astype(np.float32),
            outs[ONNX_OUT_MAP["embedding"]].astype(np.float32))


def project_logits(logits):
    out = np.zeros((len(logits), N_CLASSES), dtype=np.float32)
    out[:, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
    for pos_idx, bc_idxs in proxy_map.items():
        out[:, pos_idx] = logits[:, np.array(bc_idxs, dtype=np.int32)].max(axis=1)
    return out

# %%
# Unlabeled = all train_soundscapes NOT in train_soundscapes_labels.csv
# (2026 has no separate unlabeled_soundscapes/ directory)
sc_labels_df   = pd.read_csv(BASE_DIR / "train_soundscapes_labels.csv")
labeled_fnames = set(sc_labels_df["filename"].unique())
all_sc_files   = sorted(TRAIN_SC_DIR.glob("*.ogg")) if TRAIN_SC_DIR.exists() else []
ul_files       = [f for f in all_sc_files if f.name not in labeled_fnames]
print(f"\nTotal train soundscapes : {len(all_sc_files)}")
print(f"Labeled soundscapes     : {len(labeled_fnames)}")
print(f"Unlabeled (PL fuel)     : {len(ul_files)}")
if not ul_files:
    print("WARNING: no unlabeled soundscapes found — saving empty arrays")

# %%
# Extract Perch embeddings + projected logits for every window of every file.
# Memory estimate: 100k windows -> embs ~600MB, scores ~94MB -> well within 13GB.
# We sub-batch at PERCH_BATCH=192 windows per ONNX call (same as nb13-15).
embs_chunks   = []
scores_chunks = []
meta_rows     = []
total_windows = 0
t0 = time.time()

for fi, fpath in enumerate(ul_files):
    try:
        windows = load_audio_any(fpath)   # (n_w, WINDOW_SAMPLES)
        n_w     = len(windows)

        file_embs   = []
        file_scores = []
        for sb in range(0, n_w, PERCH_BATCH):
            batch          = windows[sb:sb + PERCH_BATCH]
            logits, embs   = perch_infer(batch)
            file_embs.append(embs)
            file_scores.append(project_logits(logits))
        embs_chunks.append(np.concatenate(file_embs,   axis=0))
        scores_chunks.append(np.concatenate(file_scores, axis=0))

        meta = parse_fname(fpath.name)
        for wi in range(n_w):
            end_s = (wi + 1) * WINDOW_SEC
            meta_rows.append({
                "filename"  : fpath.name,
                "window_idx": wi,
                "end_sec"   : end_s,
                "row_id"    : f"{fpath.stem}_{end_s}",
                "site"      : meta["site"],
                "hour_utc"  : meta["hour_utc"],
            })
        total_windows += n_w

    except Exception as e:
        print(f"\n  ERROR on {fpath.name}: {e}")

    if (fi + 1) % 100 == 0 or fi == len(ul_files) - 1:
        elapsed = time.time() - t0
        rate    = total_windows / max(elapsed, 1)
        eta_s   = (len(ul_files) - fi - 1) * (elapsed / max(fi + 1, 1))
        print(f"  [{fi+1}/{len(ul_files)}] windows={total_windows}  "
              f"rate={rate:.1f}w/s  ETA={eta_s/60:.1f}min  "
              f"elapsed={elapsed/60:.1f}min", end="\r")

print(f"\n\nExtraction complete: {total_windows} windows from {len(ul_files)} files  "
      f"({(time.time()-t0)/3600:.2f}h)")

# %%
embs_ul   = (np.concatenate(embs_chunks,   axis=0).astype(np.float32)
             if embs_chunks   else np.zeros((0, 1536),      dtype=np.float32))
scores_ul = (np.concatenate(scores_chunks, axis=0).astype(np.float32)
             if scores_chunks else np.zeros((0, N_CLASSES), dtype=np.float32))
meta_ul   = pd.DataFrame(meta_rows)

del embs_chunks, scores_chunks; gc.collect()

# %%
np.save(OUT_DIR / "embs_ul.npy",   embs_ul)
np.save(OUT_DIR / "scores_ul.npy", scores_ul)
meta_ul.to_csv(OUT_DIR / "meta_ul.csv", index=False)

print(f"Saved embs_ul.npy   : {embs_ul.shape}  {embs_ul.nbytes/1e6:.1f} MB")
print(f"Saved scores_ul.npy : {scores_ul.shape}  {scores_ul.nbytes/1e6:.1f} MB")
print(f"Saved meta_ul.csv   : {len(meta_ul)} rows")

saver_diag = {
    "n_files"        : len(ul_files),
    "n_windows"      : int(total_windows),
    "embs_shape"     : list(embs_ul.shape),
    "scores_shape"   : list(scores_ul.shape),
    "elapsed_min"    : round((time.time() - t0) / 60, 1),
    "n_mapped"       : int(MAPPED_MASK.sum()),
    "n_proxy"        : len(proxy_map),
}
with open(OUT_DIR / "saver_diagnostics.json", "w") as f:
    json.dump(saver_diag, f, indent=2)
print("saver_diagnostics.json saved")
print(json.dumps(saver_diag, indent=2))
