# %%
# ============================================================================
# BirdCLEF 2026 -- nb13: Perch v2 + MLP Probes Baseline (Layer 0 + 1)
# ============================================================================
# Single-notebook pipeline (no separate cache builder):
#   1. Install onnxruntime wheel from attached dataset
#   2. Load taxonomy + soundscape labels + sample submission
#   3. Build species mapping (taxonomy -> Perch vocab) + genus proxy
#   4. Extract Perch features on 59 train_soundscapes (708 windows)
#   5. PCA(emb, 64) on train embeddings
#   6. Build per-(window, class) input features = 64 PCA + 5 temporal scalars
#   7. 5-fold GroupKFold by filename -> train 5 vectorized per-class MLPs
#      (BCE + inverse-sqrt class weights, Adam lr=1e-3 cos, 30ep)
#   8. Extract Perch features on test_soundscapes (variable N)
#   9. Build test features -> run 5 MLPs -> mean -> alpha-blend with Perch
#  10. Write submission.csv aligned to sample_submission row order
#
# Inputs:
#   - competitions/birdclef-2026
#   - models/google/bird-vocalization-classifier/.../perch_v2_cpu/1
#   - datasets/rishikeshjani/perch-onnx-for-birdclef-2026
#
# Internet : ON  (one-time onnxruntime wheel install)
# Compute  : CPU (~3 min train Perch + ~3 min MLPs + 10-60 min test Perch)
# ============================================================================

# %%
# -- Step 0: Install onnxruntime wheel ---------------------------------------
import glob as _glob
import subprocess as _sub
import sys as _sys

_whl = [w for w in _glob.glob("/kaggle/input/**/onnxruntime*cp312*x86_64*.whl",
                              recursive=True) if "gpu" not in w]
if _whl:
    print(f"Installing onnxruntime: {_whl[0]}")
    _sub.check_call([_sys.executable, "-m", "pip", "install", _whl[0], "--quiet"])
else:
    print("No bundled wheel found; falling back to PyPI")
    _sub.check_call([_sys.executable, "-m", "pip", "install", "onnxruntime", "--quiet"])

# %%
import gc
import re
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
import onnxruntime as ort
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

print(f"torch       : {torch.__version__}")
print(f"torchaudio  : {torchaudio.__version__}")
print(f"onnxruntime : {ort.__version__}")

# %%
# -- Paths -------------------------------------------------------------------
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
ONNX_PATH = next((p for p in _onnx_candidates if p.exists()), None)

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(exist_ok=True)

print(f"BASE_DIR  : {BASE_DIR}     exists={BASE_DIR.exists()}")
print(f"MODEL_DIR : {MODEL_DIR}    exists={MODEL_DIR.exists()}")
print(f"ONNX_PATH : {ONNX_PATH}    found={ONNX_PATH is not None}")

# %%
# -- Constants ---------------------------------------------------------------
PERCH_SR       = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = PERCH_SR * WINDOW_SEC          # 160 000
SC_FILE_SEC    = 60                              # train_soundscapes are 60s
SC_N_WINDOWS   = SC_FILE_SEC // WINDOW_SEC      # 12

BATCH_FILES_TRAIN = 16    # files per ONNX call for training (= 192 windows)
BATCH_FILES_TEST  = 8     # smaller for test (variable file length)
IO_WORKERS        = 4

PCA_DIM   = 64
N_FOLDS   = 5
EPOCHS    = 30
LR        = 1e-3
WD        = 1e-4
BATCH_SZ  = 128
ALPHA     = 0.7           # alpha-blend weight on MLP vs Perch sigmoid

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

def parse_fname(name):
    m = FNAME_RE.match(Path(name).name)
    if m:
        return {"site": m.group(2), "hour_utc": int(m.group(4)[:2])}
    return {"site": "unknown", "hour_utc": -1}

# %%
# -- Load competition data ---------------------------------------------------
taxonomy   = pd.read_csv(BASE_DIR / "taxonomy.csv")
sc_labels  = pd.read_csv(BASE_DIR / "train_soundscapes_labels.csv")
sample_sub = pd.read_csv(BASE_DIR / "sample_submission.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES      = len(PRIMARY_LABELS)
label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}

print(f"Classes        : {N_CLASSES}")
print(f"Soundscape rows: {len(sc_labels)}")
print(f"Sample sub rows: {len(sample_sub)}")

# %%
# -- Build per-window label matrix from soundscape_labels --------------------
def union_labels(series):
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t:
                    out.add(t)
    return sorted(out)

sc = (sc_labels
      .groupby(["filename", "start", "end"])["primary_label"]
      .apply(union_labels)
      .reset_index(name="label_list"))
sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
sc["row_id"]  = (sc["filename"].str.replace(".ogg", "", regex=False)
                 + "_" + sc["end_sec"].astype(str))

# row_id -> multihot label vector
Y_SC_LOOKUP = {}
for _, row in sc.iterrows():
    v = np.zeros(N_CLASSES, dtype=np.float32)
    for lbl in row["label_list"]:
        if lbl in label_to_idx:
            v[label_to_idx[lbl]] = 1.0
    Y_SC_LOOKUP[row["row_id"]] = v

print(f"Unique labeled windows: {len(sc)}")
print(f"Files in labels CSV   : {sc['filename'].nunique()}")

# %%
# -- ONNX session setup ------------------------------------------------------
assert ONNX_PATH is not None, "Perch ONNX model not found"

_so = ort.SessionOptions()
_so.intra_op_num_threads = 4
_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
SESS = ort.InferenceSession(str(ONNX_PATH), sess_options=_so,
                            providers=["CPUExecutionProvider"])
ONNX_INPUT   = SESS.get_inputs()[0].name
ONNX_OUT_MAP = {o.name: i for i, o in enumerate(SESS.get_outputs())}
print(f"ONNX outputs: {list(ONNX_OUT_MAP.keys())}")

# %%
# -- Species mapping: taxonomy -> Perch vocab --------------------------------
BC_LABELS_CSV = MODEL_DIR / "assets" / "labels.csv"
bc_labels = pd.read_csv(BC_LABELS_CSV).reset_index().rename(columns={"index": "bc_index"})
_sci_col  = [c for c in bc_labels.columns if c != "bc_index"][0]
bc_labels = bc_labels.rename(columns={_sci_col: "scientific_name"})
NO_LABEL  = len(bc_labels)

mapping = (taxonomy
           .merge(bc_labels[["bc_index", "scientific_name"]],
                  on="scientific_name", how="left"))
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
lbl2bc = mapping.set_index("primary_label")["bc_index"]

BC_INDICES    = np.array([int(lbl2bc.get(c, NO_LABEL)) for c in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK   = BC_INDICES != NO_LABEL
MAPPED_POS    = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)
UNMAPPED_POS  = np.where(~MAPPED_MASK)[0].astype(np.int32)

print(f"Mapped : {MAPPED_MASK.sum()} / {N_CLASSES}")
print(f"Unmapped: {len(UNMAPPED_POS)}")

# %%
# -- Genus proxy for unmapped species ----------------------------------------
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
SCI_NAME_MAP   = taxonomy.set_index("primary_label")["scientific_name"].to_dict()
PROXY_TAXA     = {"Amphibia", "Insecta", "Aves"}

proxy_map = {}    # competition idx -> [bc_index, ...]
for lbl in [PRIMARY_LABELS[i] for i in UNMAPPED_POS]:
    if CLASS_NAME_MAP.get(lbl) not in PROXY_TAXA:
        continue
    sci   = str(SCI_NAME_MAP.get(lbl, ""))
    genus = sci.split()[0] if sci else ""
    if not genus:
        continue
    hits = bc_labels[bc_labels["scientific_name"].astype(str)
                     .str.match(rf"^{re.escape(genus)}\s", na=False)]
    if len(hits):
        proxy_map[label_to_idx[lbl]] = hits["bc_index"].astype(int).tolist()

# Class-conditional alpha: 1.0 for unmapped-no-proxy classes (pure MLP),
# ALPHA elsewhere (alpha-blend MLP with Perch sigmoid)
HAS_PERCH_SIGNAL = MAPPED_MASK.copy()
for idx in proxy_map:
    HAS_PERCH_SIGNAL[idx] = True
alpha_per_class = np.where(HAS_PERCH_SIGNAL, ALPHA, 1.0).astype(np.float32)

print(f"Genus-proxy classes : {len(proxy_map)}")
print(f"No Perch signal     : {N_CLASSES - HAS_PERCH_SIGNAL.sum()}")

# %%
# -- Audio loader + ONNX helpers ---------------------------------------------
def load_audio_60s(path, target_sec=SC_FILE_SEC):
    """Load OGG, mono 32 kHz, exactly target_sec seconds (pad/truncate)."""
    try:
        wav, sr = torchaudio.load(str(path))
        if sr != PERCH_SR:
            wav = torchaudio.functional.resample(wav, sr, PERCH_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        y = wav.squeeze(0).numpy()
    except Exception as e:
        print(f"  load error {Path(path).name}: {e}")
        return np.zeros(target_sec * PERCH_SR, dtype=np.float32)
    target = target_sec * PERCH_SR
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    return y.astype(np.float32)


def perch_infer(batch_np):
    """batch_np float32 (B, 160000) -> (logits BxN_PERCH, embs Bx1536)."""
    outs   = SESS.run(None, {ONNX_INPUT: batch_np})
    logits = outs[ONNX_OUT_MAP["label"]].astype(np.float32)
    embs   = outs[ONNX_OUT_MAP["embedding"]].astype(np.float32)
    return logits, embs


def project_logits(logits):
    """Map Perch ~10k vocab -> 234 competition classes; proxy via max."""
    out = np.zeros((len(logits), N_CLASSES), dtype=np.float32)
    out[:, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
    for pos_idx, bc_idxs in proxy_map.items():
        out[:, pos_idx] = logits[:, np.array(bc_idxs, dtype=np.int32)].max(axis=1)
    return out

# %%
# -- Extract Perch on train_soundscapes (708 windows) -----------------------
def extract_perch_60s_files(file_paths, batch_files, label="train"):
    """Given a list of .ogg paths, produce (meta_df, scores 234, embs 1536) for
    all 12 windows per file. Uses prefetch + ThreadPoolExecutor."""
    n_files = len(file_paths)
    n_rows  = n_files * SC_N_WINDOWS

    scores_all = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embs_all   = np.zeros((n_rows, 1536),      dtype=np.float32)
    meta_rows  = []
    wr = 0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        # Prefetch first batch
        future = pool.submit(lambda ps: [load_audio_60s(p) for p in ps],
                             file_paths[:batch_files])
        for start in range(0, n_files, batch_files):
            batch_paths = file_paths[start:start + batch_files]
            batch_audio = future.result()
            nxt = start + batch_files
            if nxt < n_files:
                future = pool.submit(
                    lambda ps: [load_audio_60s(p) for p in ps],
                    file_paths[nxt:nxt + batch_files]
                )
            batch_n = len(batch_audio)
            x = np.stack([a.reshape(SC_N_WINDOWS, WINDOW_SAMPLES) for a in batch_audio])
            x = x.reshape(-1, WINDOW_SAMPLES)
            logits, embs = perch_infer(x)
            scores = project_logits(logits)
            for bi, path in enumerate(batch_paths):
                stem = path.stem
                meta = parse_fname(path.name)
                for wi in range(SC_N_WINDOWS):
                    end_s = (wi + 1) * WINDOW_SEC
                    meta_rows.append({
                        "filename" : path.name,
                        "window_idx": wi,
                        "end_sec"  : end_s,
                        "row_id"   : f"{stem}_{end_s}",
                        "site"     : meta["site"],
                        "hour_utc" : meta["hour_utc"],
                    })
            rows_this = batch_n * SC_N_WINDOWS
            scores_all[wr:wr + rows_this] = scores
            embs_all  [wr:wr + rows_this] = embs
            wr += rows_this
            print(f"  {label}: {wr}/{n_rows} windows  {time.time()-t0:.0f}s",
                  end="\r")
    print(f"\n  {label} extraction done: {wr} windows in {time.time()-t0:.1f}s")
    return pd.DataFrame(meta_rows), scores_all[:wr], embs_all[:wr]


labeled_fnames = set(sc_labels["filename"].unique())
train_files = sorted([
    f for f in (BASE_DIR / "train_soundscapes").glob("*.ogg")
    if f.name in labeled_fnames
])
print(f"\nExtracting Perch on {len(train_files)} labeled train_soundscapes ...")
meta_tr, scores_tr, embs_tr = extract_perch_60s_files(
    train_files, BATCH_FILES_TRAIN, label="train")

# %%
# -- Align labels: meta_tr row order -> Y_TR ---------------------------------
Y_TR = np.stack([
    Y_SC_LOOKUP.get(rid, np.zeros(N_CLASSES, dtype=np.float32))
    for rid in meta_tr["row_id"]
])
labeled_count = int((Y_TR.sum(1) > 0).sum())
print(f"Labeled train windows: {labeled_count} / {len(Y_TR)}")
assert labeled_count >= 600, f"Only {labeled_count} windows have labels -- alignment likely broken"

# %%
# -- PCA fit on train embeddings ---------------------------------------------
print(f"\nFitting PCA({PCA_DIM}) on train_emb ({embs_tr.shape}) ...")
pca = PCA(n_components=PCA_DIM, random_state=42).fit(embs_tr)
pca_tr = pca.transform(embs_tr).astype(np.float32)
print(f"PCA variance retained: {pca.explained_variance_ratio_.sum():.4f}")

# %%
# -- Build temporal features (5 species-specific scalars per window) ---------
def build_temporal_features(scores, meta_df, n_windows=SC_N_WINDOWS):
    """
    For each (window, species) compute prev/next/mean/max/std of the species'
    score across the 12 windows of the same file.

    scores : (N, 234) raw Perch projected scores
    meta_df: must include 'filename' and 'window_idx' (0..11) per row

    Returns: temporal (N, 234, 5)
    """
    N = len(scores)
    out = np.zeros((N, N_CLASSES, 5), dtype=np.float32)
    # group by filename and reorder by window_idx (ensures we get the same 12)
    for fname, group in meta_df.groupby("filename", sort=False):
        idxs = group.sort_values("window_idx").index.to_numpy()
        if len(idxs) != n_windows:
            # pad with first row if file is short (shouldn't happen for 60s sc)
            continue
        block = scores[idxs]                              # (12, 234)
        prev_  = np.roll(block, 1, axis=0)                # circular prev
        next_  = np.roll(block, -1, axis=0)               # circular next
        mean_  = np.broadcast_to(block.mean(0), block.shape).copy()
        max_   = np.broadcast_to(block.max(0),  block.shape).copy()
        std_   = np.broadcast_to(block.std(0),  block.shape).copy()
        out[idxs, :, 0] = prev_
        out[idxs, :, 1] = next_
        out[idxs, :, 2] = mean_
        out[idxs, :, 3] = max_
        out[idxs, :, 4] = std_
    return out


# Need a window_idx column in meta_tr (already added during extraction)
temp_tr = build_temporal_features(scores_tr, meta_tr)
print(f"Temporal features: {temp_tr.shape}")

# Combine: (N, 234, 69) where dim2 = [pca_64 broadcast | 5 temporal]
def build_mlp_input(pca_emb, temporal):
    """
    pca_emb  : (N, 64)
    temporal : (N, 234, 5)
    returns  : (N, 234, 69)
    """
    N = pca_emb.shape[0]
    pca_b = np.broadcast_to(pca_emb[:, None, :], (N, N_CLASSES, PCA_DIM))
    return np.concatenate([pca_b, temporal], axis=2).astype(np.float32)

X_TR = build_mlp_input(pca_tr, temp_tr)        # (708, 234, 69)
print(f"Train MLP input shape: {X_TR.shape}")

# %%
# -- Vectorized per-class MLP -------------------------------------------------
class VectorizedMLP(nn.Module):
    """234 separate (in -> 128 -> 64 -> 1) probes, batched via bmm."""
    def __init__(self, n_cls=N_CLASSES, in_dim=69, h1=128, h2=64):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(n_cls, in_dim, h1) * (2.0 / in_dim) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(n_cls, 1, h1))
        self.W2 = nn.Parameter(torch.randn(n_cls, h1, h2) * (2.0 / h1) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(n_cls, 1, h2))
        self.W3 = nn.Parameter(torch.randn(n_cls, h2, 1)  * (2.0 / h2) ** 0.5)
        self.b3 = nn.Parameter(torch.zeros(n_cls, 1, 1))

    def forward(self, x):
        # x: (B, n_cls, in_dim)  -> permute to (n_cls, B, in_dim)
        x = x.permute(1, 0, 2)
        h = F.relu(torch.bmm(x, self.W1) + self.b1)
        h = F.relu(torch.bmm(h, self.W2) + self.b2)
        out = torch.bmm(h, self.W3) + self.b3        # (n_cls, B, 1)
        return out.squeeze(-1).t()                   # -> (B, n_cls) logits


def train_one_fold(X_train, Y_train, X_val, Y_val, n_epochs=EPOCHS,
                   batch_size=BATCH_SZ, verbose=False):
    Xt = torch.from_numpy(X_train).float()
    Yt = torch.from_numpy(Y_train).float()
    Xv = torch.from_numpy(X_val).float()
    Yv = torch.from_numpy(Y_val).float()

    pos_count = Y_train.sum(0)
    cls_w = torch.tensor(1.0 / np.sqrt(pos_count + 1.0), dtype=torch.float32)

    model = VectorizedMLP()
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    bce   = nn.BCEWithLogitsLoss(reduction="none")

    n_train = len(Xt)
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], Yt[idx]
            opt.zero_grad()
            logits = model(xb)                       # (B, 234)
            loss   = (bce(logits, yb) * cls_w).mean()
            loss.backward()
            opt.step()
        sched.step()
        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            model.eval()
            with torch.no_grad():
                vl = (bce(model(Xv), Yv) * cls_w).mean().item()
            print(f"    ep {ep+1:3d}  val_loss={vl:.4f}")

    model.eval()
    with torch.no_grad():
        val_pred = torch.sigmoid(model(Xv)).numpy()
    return model, val_pred


# %%
# -- 5-fold GroupKFold by filename -------------------------------------------
groups = meta_tr["filename"].to_numpy()
gkf = GroupKFold(n_splits=N_FOLDS)

oof = np.zeros_like(Y_TR)
fold_models = []
print(f"\nTraining {N_FOLDS}-fold MLPs on {len(X_TR)} windows ...")
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_TR, Y_TR, groups)):
    t0 = time.time()
    model, val_pred = train_one_fold(
        X_TR[tr_idx], Y_TR[tr_idx], X_TR[va_idx], Y_TR[va_idx],
        verbose=(fold == 0),
    )
    oof[va_idx] = val_pred
    # Per-fold AUC over active classes (those with at least one positive in val)
    val_active = Y_TR[va_idx].sum(0) > 0
    if val_active.sum() > 0:
        try:
            auc = roc_auc_score(
                Y_TR[va_idx][:, val_active], val_pred[:, val_active], average="macro"
            )
        except Exception:
            auc = float("nan")
    else:
        auc = float("nan")
    print(f"  fold {fold} | n_train={len(tr_idx)} n_val={len(va_idx)} | "
          f"val_active_cls={int(val_active.sum())} | AUC={auc:.4f} | "
          f"{time.time()-t0:.1f}s")
    fold_models.append(model)

# OOF AUC over classes with at least one positive globally
active = Y_TR.sum(0) > 0
oof_auc = roc_auc_score(Y_TR[:, active], oof[:, active], average="macro")
print(f"\nOOF AUC (active={int(active.sum())} cls): {oof_auc:.4f}")

if oof_auc < 0.6:
    print("WARNING: OOF AUC suspiciously low. Continuing but check alignment.")

# %%
# -- Cleanup train memory before test extraction -----------------------------
del X_TR, temp_tr, pca_tr, embs_tr, scores_tr
gc.collect()

# %%
# -- Extract Perch on test_soundscapes ---------------------------------------
test_dir = BASE_DIR / "test_soundscapes"
test_files = sorted(test_dir.glob("*.ogg")) if test_dir.exists() else []

# Staging fallback: test_soundscapes is empty in preview
if not test_files:
    print("test_soundscapes empty -- using first 16 train_soundscapes for staging")
    test_files = sorted((BASE_DIR / "train_soundscapes").glob("*.ogg"))[:16]

print(f"\nExtracting Perch on {len(test_files)} test_soundscapes ...")

# We assume test files are 60s like train. If they're a different length we
# still extract the first 60s -- BirdCLEF 2026 sample windows row_ids only go
# up to _60 in sample_submission for any given file.
def looks_60s_or_longer(file_paths, sample=3):
    """Quick metadata check on a few files; warn if shorter than 60s."""
    for p in file_paths[:sample]:
        try:
            info = torchaudio.info(str(p))
            secs = info.num_frames / info.sample_rate
            if secs < SC_FILE_SEC - 1:
                print(f"  WARN: {p.name} only {secs:.1f}s -- will be zero-padded")
        except Exception:
            pass

looks_60s_or_longer(test_files)

meta_te, scores_te, embs_te = extract_perch_60s_files(
    test_files, BATCH_FILES_TEST, label="test")

# %%
# -- Build test features (PCA + temporal) ------------------------------------
pca_te  = pca.transform(embs_te).astype(np.float32)
temp_te = build_temporal_features(scores_te, meta_te)
X_TE    = build_mlp_input(pca_te, temp_te)       # (N_test, 234, 69)
print(f"Test MLP input shape: {X_TE.shape}")

# %%
# -- Run 5-fold MLP ensemble on test -----------------------------------------
print("\nRunning 5-fold MLP ensemble on test ...")
mlp_pred = np.zeros((len(X_TE), N_CLASSES), dtype=np.float32)
Xt = torch.from_numpy(X_TE).float()
with torch.no_grad():
    for k, model in enumerate(fold_models):
        model.eval()
        # Run in chunks to bound memory
        chunk = 256
        out_k = np.zeros_like(mlp_pred)
        for i in range(0, len(Xt), chunk):
            out_k[i:i + chunk] = torch.sigmoid(model(Xt[i:i + chunk])).numpy()
        mlp_pred += out_k
mlp_pred /= len(fold_models)

# %%
# -- Alpha-blend with Perch sigmoid -----------------------------------------
perch_sig = 1.0 / (1.0 + np.exp(-scores_te))
final = alpha_per_class[None, :] * mlp_pred + (1.0 - alpha_per_class[None, :]) * perch_sig
print(f"Final predictions: {final.shape}  "
      f"min={final.min():.4f}  max={final.max():.4f}  mean={final.mean():.4f}")

# %%
# -- Build submission --------------------------------------------------------
pred_df = pd.DataFrame(final, columns=PRIMARY_LABELS)
pred_df["row_id"] = meta_te["row_id"].values
prior = 1.0 / N_CLASSES

# Real submission: row_ids in pred_df should be a subset of sample_sub row_ids
# Staging (train fallback): pred_df row_ids won't match the 3-row dummy sample_sub
in_real_submission = (
    set(pred_df["row_id"]).issubset(set(sample_sub["row_id"]))
    or len(sample_sub) > 10
)
if in_real_submission:
    sub = sample_sub[["row_id"]].merge(pred_df, on="row_id", how="left")
    sub[PRIMARY_LABELS] = sub[PRIMARY_LABELS].fillna(prior)
else:
    print("STAGING mode: writing pred_df directly (row_ids won't match dummy sample_sub)")
    sub = pred_df[["row_id"] + PRIMARY_LABELS]

# Final guards
assert sub.columns.tolist() == ["row_id"] + PRIMARY_LABELS, "Column order mismatch"
sub.to_csv(OUT_DIR / "submission.csv", index=False)

print(f"\nSubmission saved: {sub.shape}")
print(f"OOF AUC          : {oof_auc:.4f}")
print(sub.head(3))
