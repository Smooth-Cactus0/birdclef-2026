# %%
# ============================================================================
# BirdCLEF 2026 -- nb16b: Pseudo-Label Training (threshold=0.8)
# ============================================================================
# Builds on nb15a (LB 0.883). Two-round pseudo-labeling on unlabeled_soundscapes/:
#
#   Round 1: Train nb15a-architecture MLP on 66 labeled soundscapes.
#   PL gen:  Apply 5-fold ensemble to saver-output unlabeled embeddings.
#   Filter:  Keep windows where max(soft_pred) >= THRESHOLD (0.6 here).
#   Round 2: Retrain MLP from scratch on labeled + pseudo-labeled windows.
#   Infer:   Run Round 2 models on test soundscapes + alpha-blend.
#
# Architecture (identical to nb15a):
#   [PCA(64) | scalars(5) | BiGRU ctx(8) | per-class logit(1)] = 78-dim
#   -> 234 VectorizedMLP probes via bmm
#   -> alpha-blend 0.7 MLP + 0.3 Perch sigmoid (BLEND ON)
#
# Inputs:
#   - competitions/birdclef-2026
#   - datasets/rishikeshjani/perch-onnx-for-birdclef-2026
#   - models/google/bird-vocalization-classifier/.../perch_v2_cpu/1
#   - kernel_sources: alexycactus/birdclef-2026-pseudo-label-saver
#
# Internet : ON   (onnxruntime wheel install)
# GPU      : OFF  (CPU-only)
# Runtime  : ~45 min (Perch on 66 labeled files + 2x MLP training)
# ============================================================================

THRESHOLD   = 0.8
NOTEBOOK_ID = "nb16b"

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
print(f"THRESHOLD   : {THRESHOLD}   NOTEBOOK: {NOTEBOOK_ID}")

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
ONNX_PATH = next((p for p in _onnx_candidates if p.exists()), None)
OUT_DIR   = Path("/kaggle/working"); OUT_DIR.mkdir(exist_ok=True)

# Saver kernel output aEUR" path is non-deterministic; use glob fallback
import glob as _g
_saver_hits = _g.glob("/kaggle/input/**/embs_ul.npy", recursive=True)
SAVER_DIR   = Path(_saver_hits[0]).parent if _saver_hits else None

print(f"BASE_DIR exists = {BASE_DIR.exists()}")
print(f"ONNX found      = {ONNX_PATH is not None}")
print(f"SAVER_DIR       = {SAVER_DIR}")
assert SAVER_DIR is not None, "Saver kernel output not mounted aEUR" add kernel_sources"

# %%
PERCH_SR       = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = PERCH_SR * WINDOW_SEC
SC_FILE_SEC    = 60
SC_N_WINDOWS   = SC_FILE_SEC // WINDOW_SEC   # 12
PERCH_BATCH    = 192

BATCH_FILES_TRAIN = 16
BATCH_FILES_TEST  = 8
IO_WORKERS        = 4

PCA_DIM    = 64
SCALAR_DIM = 5
GRU_HIDDEN = 32
CTX_DIM    = 8
GRU_DROP   = 0.1
LOGIT_DIM  = 1
MLP_IN_DIM = PCA_DIM + SCALAR_DIM + CTX_DIM + LOGIT_DIM   # 78

BLEND    = True
N_FOLDS  = 5
EPOCHS   = 30         # Round 1 epochs (labeled only, fast)
EPOCHS_R2  = 20       # Round 2 epochs (more data but noisier labels)
LR       = 1e-3
WD       = 1e-4
BATCH_SZ   = 128
BATCH_SZ_R2 = 256     # larger batch keeps Round 2 wall-time manageable
ALPHA    = 0.7
MAX_PSEUDO_WINDOWS = 20_000   # cap top-K by confidence; controls training time

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
def parse_fname(name):
    m = FNAME_RE.match(Path(name).name)
    if m: return {"site": m.group(2), "hour_utc": int(m.group(4)[:2])}
    return {"site": "unknown", "hour_utc": -1}

# %%
taxonomy   = pd.read_csv(BASE_DIR / "taxonomy.csv")
sc_labels  = pd.read_csv(BASE_DIR / "train_soundscapes_labels.csv")
sample_sub = pd.read_csv(BASE_DIR / "sample_submission.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES      = len(PRIMARY_LABELS)
label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}
print(f"Classes: {N_CLASSES}")

# %%
def union_labels(series):
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t: out.add(t)
    return sorted(out)

sc = (sc_labels
      .groupby(["filename", "start", "end"])["primary_label"]
      .apply(union_labels).reset_index(name="label_list"))
sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
sc["row_id"]  = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

Y_SC_LOOKUP = {}
for _, row in sc.iterrows():
    v = np.zeros(N_CLASSES, dtype=np.float32)
    for lbl in row["label_list"]:
        if lbl in label_to_idx: v[label_to_idx[lbl]] = 1.0
    Y_SC_LOOKUP[row["row_id"]] = v

print(f"Labeled windows: {len(sc)}  |  Files: {sc['filename'].nunique()}")

# %%
assert ONNX_PATH is not None, "Perch ONNX not found"
_so = ort.SessionOptions()
_so.intra_op_num_threads = 4
_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
SESS = ort.InferenceSession(str(ONNX_PATH), sess_options=_so,
                            providers=["CPUExecutionProvider"])
ONNX_INPUT   = SESS.get_inputs()[0].name
ONNX_OUT_MAP = {o.name: i for i, o in enumerate(SESS.get_outputs())}

# %%
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
    if CLASS_NAME_MAP.get(lbl) not in PROXY_TAXA: continue
    sci = str(SCI_NAME_MAP.get(lbl, "")); genus = sci.split()[0] if sci else ""
    if not genus: continue
    hits = bc_labels[bc_labels["scientific_name"].astype(str).str.match(
        rf"^{re.escape(genus)}\s", na=False)]
    if len(hits): proxy_map[label_to_idx[lbl]] = hits["bc_index"].astype(int).tolist()

HAS_PERCH_SIGNAL = MAPPED_MASK.copy()
for idx in proxy_map: HAS_PERCH_SIGNAL[idx] = True
alpha_per_class = np.where(HAS_PERCH_SIGNAL, ALPHA, 1.0).astype(np.float32)
print(f"Mapped: {MAPPED_MASK.sum()} / {N_CLASSES}  proxy: {len(proxy_map)}")

# %%
def load_audio_60s(path, target_sec=SC_FILE_SEC):
    try:
        wav, sr = torchaudio.load(str(path))
        if sr != PERCH_SR: wav = torchaudio.functional.resample(wav, sr, PERCH_SR)
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        y = wav.squeeze(0).numpy()
    except Exception as e:
        print(f"  load error {Path(path).name}: {e}")
        return np.zeros(target_sec * PERCH_SR, dtype=np.float32)
    target = target_sec * PERCH_SR
    return np.pad(y, (0, max(0, target - len(y))))[:target].astype(np.float32)


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


from concurrent.futures import ThreadPoolExecutor

def extract_perch_60s_files(file_paths, batch_files, label="train"):
    n_files = len(file_paths); n_rows = n_files * SC_N_WINDOWS
    scores_all = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embs_all   = np.zeros((n_rows, 1536),      dtype=np.float32)
    meta_rows  = []; wr = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        future = pool.submit(lambda ps: [load_audio_60s(p) for p in ps],
                             file_paths[:batch_files])
        for start in range(0, n_files, batch_files):
            batch_paths = file_paths[start:start + batch_files]
            batch_audio = future.result()
            nxt = start + batch_files
            if nxt < n_files:
                future = pool.submit(lambda ps: [load_audio_60s(p) for p in ps],
                                     file_paths[nxt:nxt + batch_files])
            x = np.stack([a.reshape(SC_N_WINDOWS, WINDOW_SAMPLES)
                          for a in batch_audio]).reshape(-1, WINDOW_SAMPLES)
            logits, embs = perch_infer(x); scores = project_logits(logits)
            for bi, path in enumerate(batch_paths):
                meta = parse_fname(path.name)
                for wi in range(SC_N_WINDOWS):
                    end_s = (wi + 1) * WINDOW_SEC
                    meta_rows.append({"filename": path.name, "window_idx": wi,
                                      "end_sec": end_s,
                                      "row_id": f"{path.stem}_{end_s}",
                                      "site": meta["site"],
                                      "hour_utc": meta["hour_utc"]})
            rows_this = len(batch_audio) * SC_N_WINDOWS
            scores_all[wr:wr+rows_this] = scores
            embs_all[wr:wr+rows_this]   = embs
            wr += rows_this
            print(f"  {label}: {wr}/{n_rows} windows  {time.time()-t0:.0f}s", end="\r")
    print(f"\n  {label} done: {wr} windows in {time.time()-t0:.1f}s")
    return pd.DataFrame(meta_rows), scores_all[:wr], embs_all[:wr]


def build_temporal_features(scores, meta_df, n_windows=SC_N_WINDOWS):
    """Temporal scalars for fixed-length 60s soundscapes (exactly n_windows per file)."""
    N = len(scores)
    out = np.zeros((N, N_CLASSES, SCALAR_DIM), dtype=np.float32)
    for fname, group in meta_df.groupby("filename", sort=False):
        idxs = group.sort_values("window_idx").index.to_numpy()
        if len(idxs) != n_windows: continue
        block = scores[idxs]
        out[idxs, :, 0] = np.roll(block,  1, 0)
        out[idxs, :, 1] = np.roll(block, -1, 0)
        out[idxs, :, 2] = np.broadcast_to(block.mean(0), block.shape).copy()
        out[idxs, :, 3] = np.broadcast_to(block.max(0),  block.shape).copy()
        out[idxs, :, 4] = np.broadcast_to(block.std(0),  block.shape).copy()
    return out


def build_temporal_features_any(scores, meta_df):
    """Temporal scalars for variable-length PAM recordings (unlabeled data)."""
    N = len(scores)
    out = np.zeros((N, N_CLASSES, SCALAR_DIM), dtype=np.float32)
    for fname, group in meta_df.groupby("filename", sort=False):
        idxs = group.sort_values("window_idx").index.to_numpy()
        if len(idxs) == 0: continue
        block = scores[idxs]
        out[idxs, :, 0] = np.roll(block,  1, 0)
        out[idxs, :, 1] = np.roll(block, -1, 0)
        out[idxs, :, 2] = block.mean(0)[None, :]
        out[idxs, :, 3] = block.max(0)[None, :]
        out[idxs, :, 4] = block.std(0)[None, :]
    return out

# %%
# ===========================================================================
# PHASE 1: Extract Perch on labeled soundscapes
# ===========================================================================
labeled_fnames = set(sc_labels["filename"].unique())
train_files = sorted([f for f in (BASE_DIR / "train_soundscapes").glob("*.ogg")
                      if f.name in labeled_fnames])
print(f"\n[Phase 1] Extracting Perch on {len(train_files)} labeled soundscapes ...")
meta_tr, scores_tr, embs_tr = extract_perch_60s_files(train_files, BATCH_FILES_TRAIN, "train")
perch_sig_tr = (1.0 / (1.0 + np.exp(-scores_tr))).astype(np.float32)

Y_TR = np.stack([Y_SC_LOOKUP.get(rid, np.zeros(N_CLASSES, dtype=np.float32))
                 for rid in meta_tr["row_id"]])
labeled_count = int((Y_TR.sum(1) > 0).sum())
print(f"Labeled windows: {labeled_count} / {len(Y_TR)}")
assert labeled_count >= 600

# %%
print(f"Fitting PCA({PCA_DIM}) on {embs_tr.shape} ...")
pca    = PCA(n_components=PCA_DIM, random_state=42).fit(embs_tr)
pca_tr = pca.transform(embs_tr).astype(np.float32)
print(f"PCA variance retained: {pca.explained_variance_ratio_.sum():.4f}")
del embs_tr; gc.collect()

logit_feat_tr = scores_tr.astype(np.float32)
temp_tr       = build_temporal_features(scores_tr, meta_tr)

train_fnames_ord = sorted(meta_tr["filename"].unique())
fname_to_fidx_tr = {f: i for i, f in enumerate(train_fnames_ord)}
file_idx_tr   = meta_tr["filename"].map(fname_to_fidx_tr).to_numpy(dtype=np.int32)
window_idx_tr = meta_tr["window_idx"].to_numpy(dtype=np.int32)

n_tr_files  = len(train_fnames_ord)
file_seq_np = np.zeros((n_tr_files, SC_N_WINDOWS, PCA_DIM), dtype=np.float32)
for row_i in range(len(pca_tr)):
    file_seq_np[file_idx_tr[row_i], window_idx_tr[row_i]] = pca_tr[row_i]
file_seq_tr_tensor = torch.from_numpy(file_seq_np).float()
print(f"File sequence tensor (labeled): {file_seq_tr_tensor.shape}")

# %%
# ===========================================================================
# Model definitions (identical to nb15a)
# ===========================================================================
class BiGRUContext(nn.Module):
    def __init__(self, in_dim=PCA_DIM, hidden=GRU_HIDDEN, out_dim=CTX_DIM, dropout=GRU_DROP):
        super().__init__()
        self.gru  = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden * 2, out_dim)

    def forward(self, seq):
        h, _ = self.gru(seq)
        return self.proj(self.drop(h))


class VectorizedMLP(nn.Module):
    """234 probes on 78-dim input [PCA(64)|scalars(5)|ctx(8)|logit(1)] via bmm."""
    def __init__(self, n_cls=N_CLASSES, in_dim=MLP_IN_DIM, h1=128, h2=64):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(n_cls, in_dim, h1) * (2.0 / in_dim) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(n_cls, 1, h1))
        self.W2 = nn.Parameter(torch.randn(n_cls, h1, h2) * (2.0 / h1) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(n_cls, 1, h2))
        self.W3 = nn.Parameter(torch.randn(n_cls, h2, 1) * (2.0 / h2) ** 0.5)
        self.b3 = nn.Parameter(torch.zeros(n_cls, 1, 1))

    def forward(self, x):
        x = x.permute(1, 0, 2)
        h = F.relu(torch.bmm(x,  self.W1) + self.b1)
        h = F.relu(torch.bmm(h,  self.W2) + self.b2)
        return (torch.bmm(h, self.W3) + self.b3).squeeze(-1).t()


def get_context(bigru, fids, wids, seq_tensor):
    unique_fids, inverse = torch.unique(fids, return_inverse=True)
    gru_out = bigru(seq_tensor[unique_fids])
    return gru_out[inverse, wids]


def build_mlp_input(pca_b, scalars_b, ctx_b, logit_b):
    B  = pca_b.shape[0]
    p  = pca_b.unsqueeze(1).expand(B, N_CLASSES, PCA_DIM)
    c  = ctx_b.unsqueeze(1).expand(B, N_CLASSES, CTX_DIM)
    lb = logit_b.unsqueeze(-1)
    return torch.cat([p, scalars_b, c, lb], dim=-1)   # (B, 234, 78)


def train_one_fold(pca_tr_f, scalars_tr_f, logit_tr_f, Y_tr_f, fids_tr_f, wids_tr_f,
                   pca_va_f, scalars_va_f, logit_va_f, Y_va_f, fids_va_f, wids_va_f,
                   seq_tensor, n_epochs=EPOCHS, batch_size=BATCH_SZ, verbose=False):
    Xp_tr   = torch.from_numpy(pca_tr_f).float()
    Xs_tr   = torch.from_numpy(scalars_tr_f).float()
    Xl_tr   = torch.from_numpy(logit_tr_f).float()
    Yt_tr   = torch.from_numpy(Y_tr_f).float()
    fids_tr = torch.from_numpy(fids_tr_f).long()
    wids_tr = torch.from_numpy(wids_tr_f).long()

    Xp_va   = torch.from_numpy(pca_va_f).float()
    Xs_va   = torch.from_numpy(scalars_va_f).float()
    Xl_va   = torch.from_numpy(logit_va_f).float()
    fids_va = torch.from_numpy(fids_va_f).long()
    wids_va = torch.from_numpy(wids_va_f).long()

    bigru = BiGRUContext()
    mlp   = VectorizedMLP()
    opt   = torch.optim.Adam(list(bigru.parameters()) + list(mlp.parameters()),
                             lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    bce   = nn.BCEWithLogitsLoss(reduction="none")
    cls_w = torch.tensor(1.0 / np.sqrt(Y_tr_f.sum(0) + 1.0), dtype=torch.float32)

    n_tr = len(Xp_tr)
    for ep in range(n_epochs):
        bigru.train(); mlp.train()
        perm = torch.randperm(n_tr)
        ep_loss = 0.0; n_batches = 0
        for i in range(0, n_tr, batch_size):
            idx   = perm[i:i + batch_size]
            xp_b  = Xp_tr[idx]; xs_b = Xs_tr[idx]; xl_b = Xl_tr[idx]; yb = Yt_tr[idx]
            opt.zero_grad()
            ctx   = get_context(bigru, fids_tr[idx], wids_tr[idx], seq_tensor)
            loss  = (bce(mlp(build_mlp_input(xp_b, xs_b, ctx, xl_b)), yb) * cls_w).mean()
            loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1
        sched.step()
        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            bigru.eval(); mlp.eval()
            with torch.no_grad():
                ctx_va = get_context(bigru, fids_va, wids_va, seq_tensor)
                vl = (bce(mlp(build_mlp_input(Xp_va, Xs_va, ctx_va, Xl_va)),
                          torch.from_numpy(Y_va_f).float()) * cls_w).mean().item()
            print(f"    ep {ep+1:3d}  train={ep_loss/n_batches:.4f}  val_loss={vl:.4f}")

    bigru.eval(); mlp.eval()
    with torch.no_grad():
        ctx_va   = get_context(bigru, fids_va, wids_va, seq_tensor)
        val_pred = torch.sigmoid(mlp(build_mlp_input(Xp_va, Xs_va, ctx_va, Xl_va))).numpy()
    return bigru, mlp, val_pred

# %%
# ===========================================================================
# PHASE 2: Round 1 training on labeled data only
# ===========================================================================
groups  = meta_tr["filename"].to_numpy()
gkf     = GroupKFold(n_splits=N_FOLDS)
oof_r1  = np.zeros_like(Y_TR)
fold_models_r1 = []

print(f"\n[Phase 2] Round 1: training {N_FOLDS}-fold MLP on {len(pca_tr)} labeled windows ...")
for fold, (tr_idx, va_idx) in enumerate(gkf.split(pca_tr, Y_TR, groups)):
    t0 = time.time()
    bigru, mlp, vp = train_one_fold(
        pca_tr[tr_idx],       temp_tr[tr_idx],       logit_feat_tr[tr_idx],
        Y_TR[tr_idx],         file_idx_tr[tr_idx],   window_idx_tr[tr_idx],
        pca_tr[va_idx],       temp_tr[va_idx],        logit_feat_tr[va_idx],
        Y_TR[va_idx],         file_idx_tr[va_idx],   window_idx_tr[va_idx],
        file_seq_tr_tensor, verbose=(fold == 0),
    )
    oof_r1[va_idx] = vp
    val_active = Y_TR[va_idx].sum(0) > 0
    try:
        auc = roc_auc_score(Y_TR[va_idx][:, val_active], vp[:, val_active], average="macro")
    except Exception: auc = float("nan")
    print(f"  R1 fold {fold} | n_tr={len(tr_idx)} n_va={len(va_idx)} | "
          f"AUC={auc:.4f} | {time.time()-t0:.1f}s")
    fold_models_r1.append((bigru, mlp))

active_mask_lab = Y_TR.sum(0) > 0
oof_auc_before_pl = float(roc_auc_score(
    Y_TR[:, active_mask_lab], oof_r1[:, active_mask_lab], average="macro"))
print(f"\nRound 1 OOF macro-AUC (before PL): {oof_auc_before_pl:.4f}")

# %%
# ===========================================================================
# PHASE 3: Load unlabeled embeddings from saver kernel
# ===========================================================================
print(f"\n[Phase 3] Loading unlabeled embeddings from {SAVER_DIR} ...")
t0 = time.time()
embs_ul_raw = np.load(SAVER_DIR / "embs_ul.npy")    # (N_ul, 1536)
scores_ul   = np.load(SAVER_DIR / "scores_ul.npy")  # (N_ul, 234)
meta_ul     = pd.read_csv(SAVER_DIR / "meta_ul.csv").reset_index(drop=True)
N_ul        = len(meta_ul)
print(f"Loaded: {embs_ul_raw.shape}  scores: {scores_ul.shape}  meta: {N_ul} rows  "
      f"({time.time()-t0:.1f}s)")

# Apply same PCA as labeled data
print(f"Applying PCA({PCA_DIM}) to {N_ul} unlabeled embeddings ...")
pca_ul = pca.transform(embs_ul_raw).astype(np.float32)
del embs_ul_raw; gc.collect()

logit_feat_ul = scores_ul.astype(np.float32)
temp_ul       = build_temporal_features_any(scores_ul, meta_ul)
print(f"temp_ul: {temp_ul.shape}  ({temp_ul.nbytes/1e6:.0f} MB)")

# Build file sequence tensor for ALL unlabeled files (for Round 1 PL inference)
ul_fnames_all    = sorted(meta_ul["filename"].unique())
fname_to_fidx_ul = {f: i for i, f in enumerate(ul_fnames_all)}
n_ul_all_files   = len(ul_fnames_all)

file_idx_ul  = meta_ul["filename"].map(fname_to_fidx_ul).to_numpy(dtype=np.int32)
# Cap window_idx at SC_N_WINDOWS-1 for BiGRU; files with >12 windows use last slot
window_idx_ul = np.minimum(meta_ul["window_idx"].to_numpy(dtype=np.int32), SC_N_WINDOWS - 1)

file_seq_ul_all = np.zeros((n_ul_all_files, SC_N_WINDOWS, PCA_DIM), dtype=np.float32)
file_seq_ul_all[file_idx_ul, window_idx_ul] = pca_ul   # vectorized assignment
file_seq_ul_tensor = torch.from_numpy(file_seq_ul_all).float()
print(f"File sequence tensor (all unlabeled): {file_seq_ul_tensor.shape}")

# %%
# ===========================================================================
# PHASE 4: Generate soft pseudo-labels using Round 1 5-fold ensemble
# ===========================================================================
print(f"\n[Phase 4] Generating pseudo-labels on {N_ul} unlabeled windows ...")
Xp_ul   = torch.from_numpy(pca_ul).float()
Xs_ul   = torch.from_numpy(temp_ul).float()
Xl_ul   = torch.from_numpy(logit_feat_ul).float()
fids_ul = torch.from_numpy(file_idx_ul).long()
wids_ul = torch.from_numpy(window_idx_ul).long()

ul_pred = np.zeros((N_ul, N_CLASSES), dtype=np.float32)
with torch.no_grad():
    for k, (bigru_k, mlp_k) in enumerate(fold_models_r1):
        bigru_k.eval(); mlp_k.eval()
        gru_out = bigru_k(file_seq_ul_tensor)          # (n_ul_files, 12, CTX_DIM)
        ctx_ul  = gru_out[fids_ul, wids_ul]            # (N_ul, CTX_DIM)
        out_k   = np.zeros((N_ul, N_CLASSES), dtype=np.float32)
        chunk   = 512
        for i in range(0, N_ul, chunk):
            out_k[i:i+chunk] = torch.sigmoid(
                mlp_k(build_mlp_input(Xp_ul[i:i+chunk], Xs_ul[i:i+chunk],
                                      ctx_ul[i:i+chunk], Xl_ul[i:i+chunk]))
            ).numpy()
        ul_pred += out_k
        print(f"  fold {k} done")
ul_pred /= N_FOLDS

del Xp_ul, Xs_ul, Xl_ul, fids_ul, wids_ul, gru_out, ctx_ul, out_k
del fold_models_r1; gc.collect()

# %%
# ===========================================================================
# PHASE 5: Filter by threshold, build combined dataset
# ===========================================================================
max_conf  = ul_pred.max(axis=1)                # (N_ul,)
keep_mask = max_conf >= THRESHOLD
n_pseudo  = int(keep_mask.sum())
print(f"\n[Phase 5] Threshold={THRESHOLD}: {n_pseudo} / {N_ul} windows selected "
      f"({100*n_pseudo/max(N_ul,1):.1f}%)")

if n_pseudo == 0:
    print("WARNING: No pseudo-labeled windows survive threshold aEUR" will use Round 1 results")
    ul_pred_filt      = np.zeros((0, N_CLASSES), dtype=np.float32)
    pca_ul_filt       = np.zeros((0, PCA_DIM),   dtype=np.float32)
    temp_ul_filt      = np.zeros((0, N_CLASSES, SCALAR_DIM), dtype=np.float32)
    logit_ul_filt     = np.zeros((0, N_CLASSES), dtype=np.float32)
    meta_ul_filt      = meta_ul.iloc[[]].reset_index(drop=True)
else:
    ul_pred_filt  = ul_pred[keep_mask]
    pca_ul_filt   = pca_ul[keep_mask]
    temp_ul_filt  = temp_ul[keep_mask]
    logit_ul_filt = logit_feat_ul[keep_mask]
    meta_ul_filt  = meta_ul[keep_mask].reset_index(drop=True)

    # Cap to top MAX_PSEUDO_WINDOWS by confidence -- keeps Round 2 wall-time fixed
    if n_pseudo > MAX_PSEUDO_WINDOWS:
        top_k_idx     = np.argsort(-max_conf[keep_mask])[:MAX_PSEUDO_WINDOWS]
        ul_pred_filt  = ul_pred_filt[top_k_idx]
        pca_ul_filt   = pca_ul_filt[top_k_idx]
        temp_ul_filt  = temp_ul_filt[top_k_idx]
        logit_ul_filt = logit_ul_filt[top_k_idx]
        meta_ul_filt  = meta_ul_filt.iloc[top_k_idx].reset_index(drop=True)
        n_pseudo      = MAX_PSEUDO_WINDOWS
        print(f"Subsampled to top {MAX_PSEUDO_WINDOWS} windows by confidence")

del ul_pred, pca_ul, temp_ul, logit_feat_ul; gc.collect()

print(f"Pseudo-label confidence stats (filtered):")
if n_pseudo > 0:
    print(f"  mean max_conf = {max_conf[keep_mask].mean():.3f}")
    print(f"  top-1 class distribution:")
    top1 = np.argmax(ul_pred_filt, axis=1)
    top1_counts = np.bincount(top1, minlength=N_CLASSES)
    top10_idx   = np.argsort(-top1_counts)[:10]
    for ci in top10_idx:
        if top1_counts[ci] > 0:
            print(f"    {PRIMARY_LABELS[ci]}: {top1_counts[ci]}")

# %%
# Build combined file sequence tensor for Round 2
# Labeled files: indices 0..n_tr_files-1
# Pseudo-labeled files: indices n_tr_files..n_tr_files+n_ul_filt_files-1
ul_filt_fnames      = sorted(meta_ul_filt["filename"].unique()) if n_pseudo > 0 else []
n_ul_filt_files     = len(ul_filt_fnames)
fname_to_fidx_comb  = {f: n_tr_files + i for i, f in enumerate(ul_filt_fnames)}

# Extract file sequences for pseudo-labeled files from the full ul file_seq
file_seq_ul_filt = np.zeros((n_ul_filt_files, SC_N_WINDOWS, PCA_DIM), dtype=np.float32)
for i, fname in enumerate(ul_filt_fnames):
    orig_fidx = fname_to_fidx_ul[fname]
    file_seq_ul_filt[i] = file_seq_ul_all[orig_fidx]

del file_seq_ul_all, file_seq_ul_tensor; gc.collect()

file_seq_comb_np     = np.concatenate([file_seq_np, file_seq_ul_filt], axis=0)
file_seq_comb_tensor = torch.from_numpy(file_seq_comb_np).float()
print(f"Combined file sequence tensor: {file_seq_comb_tensor.shape}")

# File indices and window indices for pseudo-labeled windows
if n_pseudo > 0:
    fids_ul_comb = meta_ul_filt["filename"].map(fname_to_fidx_comb).to_numpy(dtype=np.int32)
    wids_ul_comb = np.minimum(meta_ul_filt["window_idx"].to_numpy(dtype=np.int32), SC_N_WINDOWS - 1)
else:
    fids_ul_comb = np.zeros(0, dtype=np.int32)
    wids_ul_comb = np.zeros(0, dtype=np.int32)

# Combined arrays (labeled + pseudo-labeled)
n_labeled = len(Y_TR)
pca_comb   = np.vstack([pca_tr,       pca_ul_filt])
temp_comb  = np.vstack([temp_tr,       temp_ul_filt])
logit_comb = np.vstack([logit_feat_tr, logit_ul_filt])
Y_comb     = np.vstack([Y_TR,         ul_pred_filt])
fids_comb  = np.concatenate([file_idx_tr,  fids_ul_comb])
wids_comb  = np.concatenate([window_idx_tr, wids_ul_comb])

print(f"Combined dataset: {len(Y_comb)} windows "
      f"({n_labeled} labeled + {n_pseudo} pseudo-labeled)")

# %%
# ===========================================================================
# PHASE 6: Round 2 retraining on combined dataset
# ===========================================================================
oof_r2         = np.zeros((n_labeled, N_CLASSES), dtype=np.float32)
fold_models_r2 = []
pl_idx         = np.arange(n_labeled, n_labeled + n_pseudo)  # always in training

if n_pseudo == 0:
    print("\n[Phase 6] Skipping Round 2 (no pseudo-labels) aEUR" using Round 1 OOF")
    oof_r2 = oof_r1
else:
    print(f"\n[Phase 6] Round 2: retraining {N_FOLDS}-fold MLP on combined "
          f"{len(Y_comb)} windows ...")
    for fold, (tr_lab_idx, va_lab_idx) in enumerate(gkf.split(pca_tr, Y_TR, groups)):
        t0 = time.time()
        tr_all_idx = np.concatenate([tr_lab_idx, pl_idx])

        bigru, mlp, vp = train_one_fold(
            pca_comb[tr_all_idx],   temp_comb[tr_all_idx],   logit_comb[tr_all_idx],
            Y_comb[tr_all_idx],     fids_comb[tr_all_idx],   wids_comb[tr_all_idx],
            pca_comb[va_lab_idx],   temp_comb[va_lab_idx],   logit_comb[va_lab_idx],
            Y_TR[va_lab_idx],       fids_comb[va_lab_idx],   wids_comb[va_lab_idx],
            file_seq_comb_tensor,
            n_epochs=EPOCHS_R2, batch_size=BATCH_SZ_R2, verbose=(fold == 0),
        )
        oof_r2[va_lab_idx] = vp
        val_active = Y_TR[va_lab_idx].sum(0) > 0
        try:
            auc = roc_auc_score(Y_TR[va_lab_idx][:, val_active], vp[:, val_active],
                                average="macro")
        except Exception: auc = float("nan")
        print(f"  R2 fold {fold} | n_tr={len(tr_all_idx)} n_va={len(va_lab_idx)} | "
              f"AUC={auc:.4f} | {time.time()-t0:.1f}s")
        fold_models_r2.append((bigru, mlp))

oof_auc_after_pl = float(roc_auc_score(
    Y_TR[:, active_mask_lab], oof_r2[:, active_mask_lab], average="macro"))
fold_models_final = fold_models_r2 if n_pseudo > 0 else fold_models_r1

print(f"\nRound 1 OOF AUC (before PL): {oof_auc_before_pl:.4f}")
print(f"Round 2 OOF AUC (after PL):  {oof_auc_after_pl:.4f}  "
      f"(delta={oof_auc_after_pl - oof_auc_before_pl:+.4f})")

# %%
# Per-class AUC comparison
per_class_rows = []
active_idxs = np.where(active_mask_lab)[0]
for i in active_idxs:
    lbl = PRIMARY_LABELS[i]; y_i = Y_TR[:, i]
    try:
        auc_r1 = roc_auc_score(y_i, oof_r1[:, i])
        auc_r2 = roc_auc_score(y_i, oof_r2[:, i])
    except Exception: auc_r1 = auc_r2 = float("nan")
    per_class_rows.append({"species": lbl, "n_pos": int(y_i.sum()),
                            "auc_r1": auc_r1, "auc_r2": auc_r2,
                            "delta": auc_r2 - auc_r1})

auc_df = pd.DataFrame(per_class_rows).sort_values("delta", ascending=False).reset_index(drop=True)
diag_csv = OUT_DIR / f"per_class_auc_{NOTEBOOK_ID}.csv"
auc_df.to_csv(diag_csv, index=False)
print(f"\nTop 10 improved species:")
print(auc_df.head(10).to_string(index=False))
print(f"\nBottom 10 (most degraded):")
print(auc_df.tail(10).to_string(index=False))
print(f"\nSaved {diag_csv.name}")

# %%
diagnostics = {
    "notebook"            : NOTEBOOK_ID,
    "threshold"           : THRESHOLD,
    "n_labeled_windows"   : n_labeled,
    "n_pseudo_windows"    : n_pseudo,
    "oof_auc_before_pl"   : round(oof_auc_before_pl, 6),
    "oof_auc_after_pl"    : round(oof_auc_after_pl, 6),
    "delta_pl"            : round(oof_auc_after_pl - oof_auc_before_pl, 6),
    "nb15a_reference_lb"  : 0.883,
    "n_active_classes"    : int(active_mask_lab.sum()),
}
with open(OUT_DIR / f"diagnostics_{NOTEBOOK_ID}.json", "w") as f:
    json.dump(diagnostics, f, indent=2)
print(f"\nDiagnostics saved:")
print(json.dumps(diagnostics, indent=2))

# %%
del pca_comb, temp_comb, logit_comb, Y_comb, fids_comb, wids_comb; gc.collect()

# %%
# ===========================================================================
# INFERENCE: Test soundscapes + submission
# ===========================================================================
test_dir   = BASE_DIR / "test_soundscapes"
test_files = sorted(test_dir.glob("*.ogg")) if test_dir.exists() else []
if not test_files:
    print("test_soundscapes empty -- staging fallback: first 16 train soundscapes")
    test_files = sorted((BASE_DIR / "train_soundscapes").glob("*.ogg"))[:16]
print(f"\n[Inference] Extracting Perch on {len(test_files)} test soundscapes ...")
meta_te, scores_te, embs_te = extract_perch_60s_files(test_files, BATCH_FILES_TEST, "test")

# %%
pca_te        = pca.transform(embs_te).astype(np.float32)
del embs_te; gc.collect()

temp_te       = build_temporal_features(scores_te, meta_te)
logit_feat_te = scores_te.astype(np.float32)
perch_sig_te  = (1.0 / (1.0 + np.exp(-scores_te))).astype(np.float32)

test_fnames_ord  = sorted(meta_te["filename"].unique())
fname_to_fidx_te = {f: i for i, f in enumerate(test_fnames_ord)}
n_test_files = len(test_fnames_ord)

file_idx_te   = meta_te["filename"].map(fname_to_fidx_te).to_numpy(dtype=np.int32)
window_idx_te = meta_te["window_idx"].to_numpy(dtype=np.int32)

file_seq_te = np.zeros((n_test_files, SC_N_WINDOWS, PCA_DIM), dtype=np.float32)
for row_i in range(len(pca_te)):
    file_seq_te[file_idx_te[row_i], window_idx_te[row_i]] = pca_te[row_i]
file_seq_te_tensor = torch.from_numpy(file_seq_te).float()

Xp_te   = torch.from_numpy(pca_te).float()
Xs_te   = torch.from_numpy(temp_te).float()
Xl_te   = torch.from_numpy(logit_feat_te).float()
fids_te = torch.from_numpy(file_idx_te).long()
wids_te = torch.from_numpy(window_idx_te).long()
print(f"Test: {Xp_te.shape[0]} windows, {n_test_files} files")

# %%
print(f"\nRunning 5-fold ensemble on test (Round 2 models) ...")
mlp_pred = np.zeros((len(Xp_te), N_CLASSES), dtype=np.float32)
with torch.no_grad():
    for k, (bigru_k, mlp_k) in enumerate(fold_models_final):
        bigru_k.eval(); mlp_k.eval()
        gru_out_te = bigru_k(file_seq_te_tensor)
        ctx_te     = gru_out_te[fids_te, wids_te]
        out_k = np.zeros_like(mlp_pred); chunk = 256
        for i in range(0, len(Xp_te), chunk):
            out_k[i:i+chunk] = torch.sigmoid(
                mlp_k(build_mlp_input(Xp_te[i:i+chunk], Xs_te[i:i+chunk],
                                      ctx_te[i:i+chunk], Xl_te[i:i+chunk]))
            ).numpy()
        mlp_pred += out_k
        print(f"  fold {k} done")
mlp_pred /= len(fold_models_final)

# %%
# Alpha-blend: BLEND ON (essential aEUR" nb15f without blend scored LB 0.502)
final = (alpha_per_class[None, :] * mlp_pred
         + (1.0 - alpha_per_class[None, :]) * perch_sig_te)
print(f"Predictions: {final.shape}  min={final.min():.4f}  max={final.max():.4f}")

pred_df = pd.DataFrame(final, columns=PRIMARY_LABELS)
pred_df["row_id"] = meta_te["row_id"].values
prior = 1.0 / N_CLASSES
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
print(f"OOF AUC before PL : {oof_auc_before_pl:.4f}")
print(f"OOF AUC after PL  : {oof_auc_after_pl:.4f}  ({oof_auc_after_pl-oof_auc_before_pl:+.4f})")
print(f"Threshold         : {THRESHOLD}")
print(f"Pseudo-windows    : {n_pseudo}")
print(sub.head(3))

