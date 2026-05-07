# %%
# ============================================================================
# BirdCLEF 2026 -- nb15c: Perch Global Logit Context (PCA-32) + Alpha-Blend ON
# ============================================================================
# Builds on nb14c (BiGRU Augment, LB 0.879). New: adds each species' Perch
# logit as a 1-scalar input to its own MLP probe (per-class logit feature).
#
# Architecture:
#   [PCA(64) | scalars(5) | BiGRU ctx(8) | global logit ctx(32)] = 109-dim
#   -> 234 VectorizedMLP probes via bmm
#   -> alpha-blend 0.7 MLP + 0.3 Perch sigmoid (BLEND ON)
#
# Ablation OFF: zero the global logit ctx -> approximates nb14c baseline.
# delta_logit = OOF_on - OOF_off = marginal logit feature contribution.
#
# Inputs:
#   - competitions/birdclef-2026
#   - models/google/bird-vocalization-classifier/.../perch_v2_cpu/1
#   - datasets/rishikeshjani/perch-onnx-for-birdclef-2026
#
# Internet : ON  (onnxruntime wheel install)
# Compute  : CPU
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
OUT_DIR = Path("/kaggle/working"); OUT_DIR.mkdir(exist_ok=True)

print(f"BASE_DIR exists={BASE_DIR.exists()}  |  ONNX found={ONNX_PATH is not None}")

# %%
PERCH_SR       = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = PERCH_SR * WINDOW_SEC
SC_FILE_SEC    = 60
SC_N_WINDOWS   = SC_FILE_SEC // WINDOW_SEC   # 12

BATCH_FILES_TRAIN = 16
BATCH_FILES_TEST  = 8
IO_WORKERS        = 4

PCA_DIM    = 64
SCALAR_DIM = 5
GRU_HIDDEN = 32
CTX_DIM    = 8
GRU_DROP   = 0.1
LOGIT_PCA_DIM = 32                                               # global logit ctx
MLP_IN_DIM    = PCA_DIM + SCALAR_DIM + CTX_DIM + LOGIT_PCA_DIM  # 109

BLEND      = True    # alpha-blend ON
N_FOLDS    = 5
EPOCHS     = 30
LR         = 1e-3
WD         = 1e-4
BATCH_SZ   = 128
ALPHA      = 0.7

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

mapping = taxonomy.merge(bc_labels[["bc_index", "scientific_name"]], on="scientific_name", how="left")
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
    hits = bc_labels[bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)]
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
    return outs[ONNX_OUT_MAP["label"]].astype(np.float32), outs[ONNX_OUT_MAP["embedding"]].astype(np.float32)

def project_logits(logits):
    out = np.zeros((len(logits), N_CLASSES), dtype=np.float32)
    out[:, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
    for pos_idx, bc_idxs in proxy_map.items():
        out[:, pos_idx] = logits[:, np.array(bc_idxs, dtype=np.int32)].max(axis=1)
    return out

def extract_perch_60s_files(file_paths, batch_files, label="train"):
    n_files = len(file_paths); n_rows = n_files * SC_N_WINDOWS
    scores_all = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embs_all   = np.zeros((n_rows, 1536), dtype=np.float32)
    meta_rows  = []; wr = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        future = pool.submit(lambda ps: [load_audio_60s(p) for p in ps], file_paths[:batch_files])
        for start in range(0, n_files, batch_files):
            batch_paths = file_paths[start:start + batch_files]
            batch_audio = future.result()
            nxt = start + batch_files
            if nxt < n_files:
                future = pool.submit(lambda ps: [load_audio_60s(p) for p in ps], file_paths[nxt:nxt+batch_files])
            x = np.stack([a.reshape(SC_N_WINDOWS, WINDOW_SAMPLES) for a in batch_audio]).reshape(-1, WINDOW_SAMPLES)
            logits, embs = perch_infer(x); scores = project_logits(logits)
            for bi, path in enumerate(batch_paths):
                meta = parse_fname(path.name)
                for wi in range(SC_N_WINDOWS):
                    end_s = (wi + 1) * WINDOW_SEC
                    meta_rows.append({"filename": path.name, "window_idx": wi,
                                      "end_sec": end_s, "row_id": f"{path.stem}_{end_s}",
                                      "site": meta["site"], "hour_utc": meta["hour_utc"]})
            rows_this = len(batch_audio) * SC_N_WINDOWS
            scores_all[wr:wr+rows_this] = scores; embs_all[wr:wr+rows_this] = embs
            wr += rows_this
            print(f"  {label}: {wr}/{n_rows} windows  {time.time()-t0:.0f}s", end="\r")
    print(f"\n  {label} done: {wr} windows in {time.time()-t0:.1f}s")
    return pd.DataFrame(meta_rows), scores_all[:wr], embs_all[:wr]

# %%
labeled_fnames = set(sc_labels["filename"].unique())
train_files = sorted([f for f in (BASE_DIR / "train_soundscapes").glob("*.ogg")
                      if f.name in labeled_fnames])
print(f"Extracting Perch on {len(train_files)} labeled soundscapes ...")
meta_tr, scores_tr, embs_tr = extract_perch_60s_files(train_files, BATCH_FILES_TRAIN, "train")
perch_sig_tr = (1.0 / (1.0 + np.exp(-scores_tr))).astype(np.float32)

# %%
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

# %%
# -- Global logit context: PCA(32) of 234-dim projected logit vector -----------
# Captures scene-level 'what Perch thinks is present' as a compact shared feature.
print(f"Fitting logit PCA({LOGIT_PCA_DIM}) on scores_tr {scores_tr.shape} ...")
logit_pca_fit = PCA(n_components=LOGIT_PCA_DIM, random_state=42).fit(scores_tr)
logit_ctx_tr  = logit_pca_fit.transform(scores_tr).astype(np.float32)   # (N, 32)
print(f"Logit PCA variance retained: {logit_pca_fit.explained_variance_ratio_.sum():.4f}")
print(f"Global logit ctx: {logit_ctx_tr.shape}")

# %%
# -- Hand-crafted temporal scalars (5 per class, same as nb13/14c) ------------
def build_temporal_features(scores, meta_df, n_windows=SC_N_WINDOWS):
    N = len(scores)
    out = np.zeros((N, N_CLASSES, 5), dtype=np.float32)
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

temp_tr = build_temporal_features(scores_tr, meta_tr)
print(f"Hand-crafted scalars: {temp_tr.shape}")

# %%
# -- File sequence matrix: (n_files, 12, 64) ----------------------------------
train_fnames_ord = sorted(meta_tr["filename"].unique())
fname_to_fidx    = {f: i for i, f in enumerate(train_fnames_ord)}
file_idx_tr   = meta_tr["filename"].map(fname_to_fidx).to_numpy(dtype=np.int32)
window_idx_tr = meta_tr["window_idx"].to_numpy(dtype=np.int32)

n_train_files = len(train_fnames_ord)
file_seq_np   = np.zeros((n_train_files, SC_N_WINDOWS, PCA_DIM), dtype=np.float32)
for row_i in range(len(pca_tr)):
    file_seq_np[file_idx_tr[row_i], window_idx_tr[row_i]] = pca_tr[row_i]
file_seq_tensor = torch.from_numpy(file_seq_np).float()
print(f"File sequence tensor: {file_seq_tensor.shape}")

# %%
# -- Model definitions --------------------------------------------------------
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
    """234 probes on 109-dim input [PCA(64)|scalars(5)|ctx(8)|glogit(32)] via bmm."""
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

# %%
def get_context(bigru, fids, wids, seq_tensor):
    unique_fids, inverse = torch.unique(fids, return_inverse=True)
    gru_out = bigru(seq_tensor[unique_fids])
    return gru_out[inverse, wids]


def build_mlp_input(pca_b, scalars_b, ctx_b, logit_ctx_b):
    """[PCA(64) | scalars(5) | BiGRU ctx(8) | global logit ctx(32)] -> (B, 234, 109)."""
    B = pca_b.shape[0]
    p  = pca_b.unsqueeze(1).expand(B, N_CLASSES, PCA_DIM)             # (B, 234, 64)
    c  = ctx_b.unsqueeze(1).expand(B, N_CLASSES, CTX_DIM)             # (B, 234, 8)
    lc = logit_ctx_b.unsqueeze(1).expand(B, N_CLASSES, LOGIT_PCA_DIM) # (B, 234, 32)
    return torch.cat([p, scalars_b, c, lc], dim=-1)                   # (B, 234, 109)


def train_one_fold(pca_tr_f, scalars_tr_f, logit_ctx_tr_f, Y_tr_f, fids_tr_f, wids_tr_f,
                   pca_va_f, scalars_va_f, logit_ctx_va_f, Y_va_f, fids_va_f, wids_va_f,
                   seq_tensor, n_epochs=EPOCHS, batch_size=BATCH_SZ, verbose=False):

    Xp_tr = torch.from_numpy(pca_tr_f).float()
    Xs_tr = torch.from_numpy(scalars_tr_f).float()
    Xl_tr = torch.from_numpy(logit_ctx_tr_f).float()   # (N_tr, 32)
    Yt_tr = torch.from_numpy(Y_tr_f).float()
    fids_tr_t = torch.from_numpy(fids_tr_f).long()
    wids_tr_t = torch.from_numpy(wids_tr_f).long()

    Xp_va = torch.from_numpy(pca_va_f).float()
    Xs_va = torch.from_numpy(scalars_va_f).float()
    Xl_va = torch.from_numpy(logit_ctx_va_f).float()
    Yt_va = torch.from_numpy(Y_va_f).float()
    fids_va_t = torch.from_numpy(fids_va_f).long()
    wids_va_t = torch.from_numpy(wids_va_f).long()

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
            idx    = perm[i:i + batch_size]
            xp_b   = Xp_tr[idx]; xs_b = Xs_tr[idx]; xl_b = Xl_tr[idx]; yb = Yt_tr[idx]
            fid_b  = fids_tr_t[idx]; wid_b = wids_tr_t[idx]
            opt.zero_grad()
            ctx    = get_context(bigru, fid_b, wid_b, seq_tensor)
            logits = mlp(build_mlp_input(xp_b, xs_b, ctx, xl_b))
            loss   = (bce(logits, yb) * cls_w).mean()
            loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1
        sched.step()
        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            bigru.eval(); mlp.eval()
            with torch.no_grad():
                ctx_va = get_context(bigru, fids_va_t, wids_va_t, seq_tensor)
                vl = (bce(mlp(build_mlp_input(Xp_va, Xs_va, ctx_va, Xl_va)), Yt_va) * cls_w).mean().item()
            print(f"    ep {ep+1:3d}  train={ep_loss/n_batches:.4f}  val_loss={vl:.4f}")

    bigru.eval(); mlp.eval()
    with torch.no_grad():
        ctx_va      = get_context(bigru, fids_va_t, wids_va_t, seq_tensor)
        val_pred_on = torch.sigmoid(mlp(build_mlp_input(Xp_va, Xs_va, ctx_va, Xl_va))).numpy()
        # Ablation: zero global logit ctx -> approximates nb14c (scalars + BiGRU only)
        logit_ctx_zero = torch.zeros_like(Xl_va)
        val_pred_off   = torch.sigmoid(mlp(build_mlp_input(Xp_va, Xs_va, ctx_va, logit_ctx_zero))).numpy()

    return bigru, mlp, val_pred_on, val_pred_off

# %%
groups  = meta_tr["filename"].to_numpy()
gkf     = GroupKFold(n_splits=N_FOLDS)
oof_on  = np.zeros_like(Y_TR)
oof_off = np.zeros_like(Y_TR)
fold_models = []

print(f"\nTraining {N_FOLDS}-fold BiGRU+global-logit-ctx MLP on {len(pca_tr)} windows ...")
for fold, (tr_idx, va_idx) in enumerate(gkf.split(pca_tr, Y_TR, groups)):
    t0 = time.time()
    bigru, mlp, vp_on, vp_off = train_one_fold(
        pca_tr[tr_idx],       temp_tr[tr_idx],       logit_ctx_tr[tr_idx],
        Y_TR[tr_idx],         file_idx_tr[tr_idx],   window_idx_tr[tr_idx],
        pca_tr[va_idx],       temp_tr[va_idx],        logit_ctx_tr[va_idx],
        Y_TR[va_idx],         file_idx_tr[va_idx],   window_idx_tr[va_idx],
        file_seq_tensor, verbose=(fold == 0),
    )
    oof_on[va_idx] = vp_on; oof_off[va_idx] = vp_off
    val_active = Y_TR[va_idx].sum(0) > 0
    try:
        auc = roc_auc_score(Y_TR[va_idx][:, val_active], vp_on[:, val_active], average="macro")
    except Exception: auc = float("nan")
    print(f"  fold {fold} | n_tr={len(tr_idx)} n_va={len(va_idx)} | "
          f"active={int(val_active.sum())} | AUC_on={auc:.4f} | {time.time()-t0:.1f}s")
    fold_models.append((bigru, mlp))

# %%
# -- Diagnostic 1: Per-class AUC table ----------------------------------------
active_mask = Y_TR.sum(0) > 0
active_idxs = np.where(active_mask)[0]

per_class_rows = []
for i in active_idxs:
    lbl = PRIMARY_LABELS[i]; y_i = Y_TR[:, i]
    try:
        auc_perch = roc_auc_score(y_i, perch_sig_tr[:, i])
        auc_off   = roc_auc_score(y_i, oof_off[:, i])
        auc_on    = roc_auc_score(y_i, oof_on[:, i])
    except Exception: auc_perch = auc_off = auc_on = float("nan")
    per_class_rows.append({"species": lbl, "n_pos": int(y_i.sum()),
                            "auc_perch": auc_perch, "auc_off": auc_off, "auc_on": auc_on,
                            "delta_logit": auc_on - auc_off,
                            "delta_mlp":   auc_off - auc_perch})

auc_df = pd.DataFrame(per_class_rows).sort_values("auc_on", ascending=False).reset_index(drop=True)
print(f"\nPer-class AUC — top 20 (active={len(auc_df)}):")
print(auc_df.head(20).to_string(index=False))
print(f"\nBottom 10:")
print(auc_df.tail(10).to_string(index=False))
auc_df.to_csv(OUT_DIR / "per_class_auc_nb15c.csv", index=False)

# %%
# -- Diagnostic 2: Global summary + JSON --------------------------------------
auc_on_g    = roc_auc_score(Y_TR[:, active_mask], oof_on[:,  active_mask], average="macro")
auc_off_g   = roc_auc_score(Y_TR[:, active_mask], oof_off[:, active_mask], average="macro")
auc_perch_g = roc_auc_score(Y_TR[:, active_mask], perch_sig_tr[:, active_mask], average="macro")

print(f"\n{'='*60}")
print(f"OOF macro-AUC  Perch-only        : {auc_perch_g:.4f}")
print(f"OOF macro-AUC  logit off (≈nb14c): {auc_off_g:.4f}  ({auc_off_g-auc_perch_g:+.4f} vs Perch)")
print(f"OOF macro-AUC  logit on          : {auc_on_g:.4f}  ({auc_on_g-auc_off_g:+.4f} vs off)")
print(f"{'='*60}")

diagnostics = {
    "notebook": "nb15c", "model": "global-logit-ctx-blend-on",
    "blend": True, "logit_mode": "global",
    "oof_auc_perch":     round(float(auc_perch_g), 6),
    "oof_auc_logit_off": round(float(auc_off_g), 6),
    "oof_auc_logit_on":  round(float(auc_on_g), 6),
    "delta_logit":       round(float(auc_on_g - auc_off_g), 6),
    "nb14c_reference":   0.5049,
    "n_active_classes":  int(active_mask.sum()),
    "n_train_windows":   int(len(Y_TR)),
    "n_train_files":     int(n_train_files),
    "mlp_in_dim":        MLP_IN_DIM,
    "epochs":            EPOCHS,
}
with open(OUT_DIR / "diagnostics_nb15c.json", "w") as f:
    json.dump(diagnostics, f, indent=2)
print("Saved diagnostics_nb15c.json")

# %%
# -- Diagnostic 3: Prediction distributions -----------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

auc_df_clean = auc_df.dropna(subset=["auc_on"]).reset_index(drop=True)
top10    = auc_df_clean.head(10)["species"].tolist()
bottom10 = auc_df_clean.tail(10)["species"].tolist()

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle("nb15c global-logit-ctx (blend ON): Prediction Distributions", fontsize=11)
for ax, sp in zip(axes.flatten(), top10 + bottom10):
    ci  = label_to_idx[sp]
    pos = oof_on[Y_TR[:, ci] > 0, ci]; neg = oof_on[Y_TR[:, ci] == 0, ci]
    row = auc_df_clean[auc_df_clean["species"] == sp]
    av  = row["auc_on"].values[0] if len(row) else float("nan")
    ax.hist(neg, bins=25, alpha=0.55, label="neg", color="steelblue", density=True)
    ax.hist(pos, bins=25, alpha=0.55, label="pos", color="crimson",   density=True)
    ax.set_title(f"{sp}\nAUC={av:.3f}  n_pos={int(Y_TR[:,ci].sum())}", fontsize=7)
    ax.legend(fontsize=6); ax.set_xlabel("pred", fontsize=6)
plt.tight_layout()
plt.savefig(OUT_DIR / "pred_dist_nb15c.png", dpi=100, bbox_inches="tight")
plt.close(); print("Saved pred_dist_nb15c.png")

# %%
del pca_tr, scores_tr, perch_sig_tr, temp_tr, logit_ctx_tr, file_seq_np
gc.collect()

# %%
test_dir   = BASE_DIR / "test_soundscapes"
test_files = sorted(test_dir.glob("*.ogg")) if test_dir.exists() else []
if not test_files:
    print("test_soundscapes empty -- staging fallback: first 16 train soundscapes")
    test_files = sorted((BASE_DIR / "train_soundscapes").glob("*.ogg"))[:16]
print(f"\nExtracting Perch on {len(test_files)} test soundscapes ...")
meta_te, scores_te, embs_te = extract_perch_60s_files(test_files, BATCH_FILES_TEST, "test")

# %%
pca_te      = pca.transform(embs_te).astype(np.float32)
del embs_te; gc.collect()

temp_te      = build_temporal_features(scores_te, meta_te)
logit_ctx_te  = logit_pca_fit.transform(scores_te).astype(np.float32)  # (N_te, 32)

test_fnames_ord  = sorted(meta_te["filename"].unique())
fname_to_fidx_te = {f: i for i, f in enumerate(test_fnames_ord)}
meta_te["file_idx"]   = meta_te["filename"].map(fname_to_fidx_te).astype(np.int32)
meta_te["window_idx"] = meta_te["window_idx"].astype(np.int32)
file_idx_te   = meta_te["file_idx"].to_numpy(dtype=np.int32)
window_idx_te = meta_te["window_idx"].to_numpy(dtype=np.int32)

n_test_files = len(test_fnames_ord)
file_seq_te  = np.zeros((n_test_files, SC_N_WINDOWS, PCA_DIM), dtype=np.float32)
for row_i in range(len(pca_te)):
    file_seq_te[file_idx_te[row_i], window_idx_te[row_i]] = pca_te[row_i]
file_seq_te_tensor = torch.from_numpy(file_seq_te).float()

Xp_te   = torch.from_numpy(pca_te).float()
Xs_te   = torch.from_numpy(temp_te).float()
Xl_te   = torch.from_numpy(logit_ctx_te).float()
fids_te = torch.from_numpy(file_idx_te).long()
wids_te = torch.from_numpy(window_idx_te).long()
print(f"Test: {Xp_te.shape[0]} windows, {n_test_files} files")

# %%
print("\nRunning 5-fold ensemble on test ...")
mlp_pred = np.zeros((len(Xp_te), N_CLASSES), dtype=np.float32)
with torch.no_grad():
    for k, (bigru_k, mlp_k) in enumerate(fold_models):
        bigru_k.eval(); mlp_k.eval()
        gru_out_te = bigru_k(file_seq_te_tensor)
        ctx_te     = gru_out_te[fids_te, wids_te]
        out_k = np.zeros_like(mlp_pred); chunk = 256
        for i in range(0, len(Xp_te), chunk):
            out_k[i:i+chunk] = torch.sigmoid(
                mlp_k(build_mlp_input(Xp_te[i:i+chunk], Xs_te[i:i+chunk],
                                      ctx_te[i:i+chunk], Xl_te[i:i+chunk]))
            ).numpy()
        mlp_pred += out_k; print(f"  fold {k} done")
mlp_pred /= len(fold_models)

# %%
perch_sig_te = 1.0 / (1.0 + np.exp(-scores_te))
# BLEND ON: alpha-blend MLP with Perch sigmoid
final = alpha_per_class[None, :] * mlp_pred + (1.0 - alpha_per_class[None, :]) * perch_sig_te
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
print(f"OOF AUC  Perch        : {auc_perch_g:.4f}")
print(f"OOF AUC  logit off    : {auc_off_g:.4f}")
print(f"OOF AUC  logit on     : {auc_on_g:.4f}")
print(f"Logit delta (on-off)  : {auc_on_g - auc_off_g:+.4f}")
print(f"Blend                 : ON")
print(sub.head(3))
