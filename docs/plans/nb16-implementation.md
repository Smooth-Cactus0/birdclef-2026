# nb16 Implementation Brief — Pseudo-Labeling Round 1

*Created: 2026-05-08. Hand this file to a new Claude Code session to implement nb16.*

---

## Mission

Build **nb16**: pseudo-labeling on `unlabeled_soundscapes/` using the current best model
(nb15a, LB 0.883). Three threshold variants (0.6, 0.8, 0.9), submit all three to competition.

This is **not** soundscape addition — that is a separate later step. The data source is
`unlabeled_soundscapes/` (PAM recordings with no labels). The MLP is retrained from scratch
on labeled + pseudo-labeled windows; the alpha-blend is kept.

**LB progression so far:**
nb13 (0.839) → nb14a (0.875) → nb14c (0.879) → **nb15a (0.883, current best)**

---

## Architecture Context — nb15a (what nb16 builds on)

### Constants
```python
PERCH_SR       = 32_000
WINDOW_SEC     = 5
SC_N_WINDOWS   = 12          # 60s / 5s
PCA_DIM        = 64
SCALAR_DIM     = 5
GRU_HIDDEN     = 32
CTX_DIM        = 8
LOGIT_DIM      = 1           # per-class scalar (the nb15a addition)
MLP_IN_DIM     = 78          # 64 + 5 + 8 + 1
N_CLASSES      = 234
N_FOLDS        = 5
EPOCHS         = 30
LR             = 1e-3
WD             = 1e-4
BATCH_SZ       = 128
ALPHA          = 0.7         # blend: 0.7 MLP + 0.3 Perch sigmoid  ← ESSENTIAL
```

### MLP input construction
```python
def build_mlp_input(pca_b, scalars_b, ctx_b, logit_b):
    B = pca_b.shape[0]
    p  = pca_b.unsqueeze(1).expand(B, N_CLASSES, PCA_DIM)   # (B, 234, 64)
    c  = ctx_b.unsqueeze(1).expand(B, N_CLASSES, CTX_DIM)   # (B, 234, 8)
    lb = logit_b.unsqueeze(-1)                               # (B, 234, 1)
    return torch.cat([p, scalars_b, c, lb], dim=-1)          # (B, 234, 78)
```

### Model classes
```python
class BiGRUContext(nn.Module):
    def __init__(self, in_dim=PCA_DIM, hidden=GRU_HIDDEN, out_dim=CTX_DIM, dropout=0.1):
        super().__init__()
        self.gru  = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden * 2, out_dim)
    def forward(self, seq): h, _ = self.gru(seq); return self.proj(self.drop(h))

class VectorizedMLP(nn.Module):
    """234 probes on 78-dim input via bmm."""
    def __init__(self, n_cls=N_CLASSES, in_dim=MLP_IN_DIM, h1=128, h2=64):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(n_cls, in_dim, h1) * (2.0/in_dim)**0.5)
        self.b1 = nn.Parameter(torch.zeros(n_cls, 1, h1))
        self.W2 = nn.Parameter(torch.randn(n_cls, h1, h2) * (2.0/h1)**0.5)
        self.b2 = nn.Parameter(torch.zeros(n_cls, 1, h2))
        self.W3 = nn.Parameter(torch.randn(n_cls, h2, 1) * (2.0/h2)**0.5)
        self.b3 = nn.Parameter(torch.zeros(n_cls, 1, 1))
    def forward(self, x):
        x = x.permute(1, 0, 2)
        h = F.relu(torch.bmm(x, self.W1) + self.b1)
        h = F.relu(torch.bmm(h, self.W2) + self.b2)
        return (torch.bmm(h, self.W3) + self.b3).squeeze(-1).t()
```

### Alpha-blend (NEVER remove)
```python
alpha_per_class = np.where(HAS_PERCH_SIGNAL, ALPHA, 1.0).astype(np.float32)
final = alpha_per_class[None,:] * mlp_pred + (1 - alpha_per_class[None,:]) * perch_sig_te
```
`HAS_PERCH_SIGNAL` = species that have a Perch mapping OR a genus-level proxy.
Species without any Perch signal get alpha=1.0 (MLP only). The blend-OFF variant (nb15f)
scored LB 0.502 — confirmed essential.

### Submission format
```python
sub = sample_sub[["row_id"]].merge(pred_df, on="row_id", how="left")
sub[PRIMARY_LABELS] = sub[PRIMARY_LABELS].fillna(1.0 / N_CLASSES)
sub.to_csv(OUT_DIR / "submission.csv", index=False)
```

### nb15a kernel (source of truth)
Kaggle slug: `alexycactus/birdclef-2026-logit-class-blend`
File: `kaggle_notebooks/15a_logit_cls_blend.py`

---

## nb16 Design — Two-Kernel Architecture

Running Perch on all unlabeled soundscapes takes ~7h (estimated 100k+ windows at 0.2s/window).
That blows the 9h Kaggle kernel limit before training starts. Split into:

### Kernel 1: nb16-saver (one-time cost, ~8h, CPU, internet=ON)
- Runs Perch ONNX on ALL files in `unlabeled_soundscapes/`
- Saves `embs_ul.npy` (N, 1536), `scores_ul.npy` (N, 234), `meta_ul.csv` to `/kaggle/working/`
- Does NOT train any MLP — Perch extraction only
- Published as a Kaggle kernel output (becomes a dataset input for the training kernels)
- Slug: `alexycactus/birdclef-2026-pseudo-label-saver`

### Kernels 2–4: nb16a / nb16b / nb16c (threshold variants, ~45min each, CPU)
Each mounts the saver output and:
1. Extracts Perch on 66 labeled soundscapes (fast, ~160s)
2. Trains nb15a-architecture MLP on labeled data (5-fold, 30 epochs)
3. Applies 5-fold ensemble to saved unlabeled embeddings → soft predictions
4. Filters: keep windows where `max(pred) >= THRESHOLD`
5. Retrains MLP from scratch on labeled + pseudo-labeled windows
6. Submits with alpha-blend ON

| Kernel | Threshold | Expected pseudo-windows | Risk |
|---|---|---|---|
| nb16a | 0.6 | many, noisier | moderate |
| nb16b | 0.8 | moderate, clean | low |
| nb16c | 0.9 | few, cleanest | very low |

### Pseudo-label construction (soft labels)
```python
# After 5-fold ensemble on unlabeled data:
# ul_pred shape: (N_ul, 234) — soft probabilities
# Apply threshold to select high-confidence windows
max_conf = ul_pred.max(axis=1)                        # (N_ul,)
keep_mask = max_conf >= THRESHOLD
ul_pred_filtered = ul_pred[keep_mask]                 # soft labels
meta_ul_filtered = meta_ul[keep_mask]

# Combine with labeled data for retraining
Y_combined  = np.vstack([Y_TR, ul_pred_filtered])     # (N_lab + N_pl, 234)
pca_combined = np.vstack([pca_tr, pca_ul_filtered])
# ... same for scores, logit feats, etc.
```

Use **soft labels** (probabilities, not 0/1) — train with BCEWithLogitsLoss as-is.
The model is retrained from scratch (random init) on the combined dataset.

---

## Data Paths

```python
BASE_DIR = (Path("/kaggle/input/competitions/birdclef-2026")
            if Path("/kaggle/input/competitions/birdclef-2026").exists()
            else Path("/kaggle/input/birdclef-2026"))

# Unlabeled soundscapes (nb16-saver reads from here)
UNLABELED_DIR = BASE_DIR / "unlabeled_soundscapes"

# Perch ONNX (try both paths)
_onnx_candidates = [
    Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx"),
    Path("/kaggle/input/perch-onnx-for-birdclef-2026/perch_v2.onnx"),
]
ONNX_PATH = next((p for p in _onnx_candidates if p.exists()), None)

# Saver kernel output (nb16a/b/c read from here)
# kernel_sources path is non-deterministic — always use glob fallback:
import glob as _g
_saver_hits = _g.glob("/kaggle/input/**/embs_ul.npy", recursive=True)
SAVER_DIR = Path(_saver_hits[0]).parent if _saver_hits else None
assert SAVER_DIR is not None, "Saver kernel output not mounted"
```

---

## Kernel Metadata Format

Every kernel needs a `kernel-metadata.json` alongside the `.py` script.
**Never set `docker_image`** — breaks competition data mounting.

```json
{
  "id": "alexycactus/birdclef-2026-pseudo-label-saver",
  "title": "BirdCLEF 2026 Pseudo Label Saver",
  "code_file": "16_pseudo_label_saver.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": false,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["rishikeshjani/perch-onnx-for-birdclef-2026"],
  "competition_sources": ["birdclef-2026"],
  "kernel_sources": []
}
```

Training kernels (nb16a example):
```json
{
  "id": "alexycactus/birdclef-2026-pseudo-label-train-06",
  "title": "BirdCLEF 2026 Pseudo Label Train 0.6",
  "code_file": "16a_pseudo_label_train_06.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": false,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["rishikeshjani/perch-onnx-for-birdclef-2026"],
  "competition_sources": ["birdclef-2026"],
  "kernel_sources": ["alexycactus/birdclef-2026-pseudo-label-saver"]
}
```

---

## Kaggle Push Command

```powershell
# From the kaggle_notebooks directory, push ONE kernel at a time (never parallel — race condition)
cd "c:\Users\alexy\Documents\Claude_projects\Kaggle competition\bird_clef\kaggle_notebooks"

# Push saver kernel
cp "16_pseudo_label_saver-metadata.json" "kernel-metadata.json"
kaggle kernels push

# Wait for completion, then push training kernels sequentially
cp "16a_pseudo_label_train_06-metadata.json" "kernel-metadata.json"
kaggle kernels push
# ... repeat for 16b, 16c
```

**CRITICAL**: Never push multiple kernels in parallel from the same directory.
All kernels share `kernel-metadata.json` — parallel jobs overwrite each other.
Always push sequentially, one kernel at a time.

### Check kernel status
```powershell
kaggle kernels status alexycactus/birdclef-2026-pseudo-label-saver
```

### Pull kernel output (after saver completes)
```powershell
kaggle kernels output alexycactus/birdclef-2026-pseudo-label-saver -p ./saver_output
```

---

## Kaggle Operational Gotchas

| Issue | Fix |
|---|---|
| SSL error (`SEC_E_UNTRUSTED_ROOT`) when pushing | `git -c http.sslVerify=false push origin master` |
| Competition data not found at `/kaggle/input/birdclef-2026/` | Try `/kaggle/input/competitions/birdclef-2026/` first (code competition path) |
| `test_soundscapes/` empty during regular kernel runs | Only populated during "Submit to Competition"; always add staging fallback (use first 16 train soundscapes) |
| `kernel_sources` path non-deterministic | Use `glob("/kaggle/input/**/your_file.npy", recursive=True)` to find it |
| Python version | Kaggle default is Python 3.12 — use cp312 wheels |
| Parallel push race condition | NEVER push multiple kernels simultaneously from same directory |
| `docker_image` in metadata | NEVER set it — breaks competition data mounting |

---

## Git Push Workflow (after implementation)

```powershell
# From Claude_projects/ root:
git add "Kaggle competition/bird_clef/kaggle_notebooks/16*"
git add "Kaggle competition/bird_clef/docs/plans/nb16-implementation.md"
git commit -m "feat(birdclef): nb16 pseudo-labeling saver + 3 threshold training kernels

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git -c http.sslVerify=false push origin master
git -c http.sslVerify=false subtree push --prefix="Kaggle competition/bird_clef" birdclef master
```

---

## Files to Create

```
kaggle_notebooks/
├── 16_pseudo_label_saver.py              # Saver kernel
├── 16_pseudo_label_saver-metadata.json
├── 16a_pseudo_label_train_06.py          # threshold=0.6
├── 16a_pseudo_label_train_06-metadata.json
├── 16b_pseudo_label_train_08.py          # threshold=0.8
├── 16b_pseudo_label_train_08-metadata.json
├── 16c_pseudo_label_train_09.py          # threshold=0.9
└── 16c_pseudo_label_train_09-metadata.json
```

---

## Diagnostics to Output

Each training kernel saves to `/kaggle/working/`:
- `diagnostics_nb16x.json` — OOF AUC before/after PL, n_pseudo_windows used, threshold
- `per_class_auc_nb16x.csv` — per-class AUC comparison (before vs after PL)
- `submission.csv`

```json
{
  "notebook": "nb16a",
  "threshold": 0.6,
  "n_labeled_windows": 792,
  "n_pseudo_windows": 9500,
  "oof_auc_before_pl": "X.XXXX",
  "oof_auc_after_pl":  "X.XXXX",
  "delta_pl": "after - before",
  "nb15a_reference_lb": 0.883
}
```

---

## Success Criteria

- Any variant with `oof_auc_after_pl > oof_auc_before_pl` → promising
- LB improvement over 0.883 confirms pseudo-labeling generalises
- Compare all 3 thresholds on LB to find the precision-recall sweet spot
- If 0.9 matches/beats 0.6 on LB → use highest threshold in future rounds
