# nb17 Implementation Brief — Alpha Sweep on Perch+MLP

*Created: 2026-05-10. Hand this file to a new Claude Code session to implement nb17.*

---

## Mission

Diagnostic sweep of the alpha-blend value used in nb15a (current best, LB 0.883).
Tests whether per-class calibration is the bottleneck before investing in learnable variants.

**Hypothesis:** if all 4 alpha values land within ±0.003 LB of 0.883, calibration is NOT the
bottleneck and we should redirect effort. If they spread > 0.01 LB, per-class calibration is
worth pursuing.

**No retraining.** Each kernel runs nb15a's full pipeline but changes only the `ALPHA` constant.
Total cost: 4 kernels × ~5 min each.

---

## Context — what nb15a does

[kaggle_notebooks/15a_logit_cls_blend.py](../../kaggle_notebooks/15a_logit_cls_blend.py)
is the source of truth. nb15a uses `ALPHA = 0.7` and applies:

```python
alpha_per_class = np.where(HAS_PERCH_SIGNAL, ALPHA, 1.0).astype(np.float32)
# Final blend at submission time:
final = alpha_per_class[None, :] * mlp_pred + (1.0 - alpha_per_class[None, :]) * perch_sig_te
```

`HAS_PERCH_SIGNAL` is a bool array (234,) — True for species with a Perch mapping or
genus-level proxy, False for species the MLP must predict alone (alpha forced to 1.0).

For mapped species: `final = ALPHA * mlp + (1-ALPHA) * perch_sigmoid`
For unmapped species: `final = mlp` (alpha=1.0)

The sweep varies `ALPHA` for the mapped species only. Unmapped behaviour stays identical.

---

## Variant Matrix — 4 Kernels

| Kernel | ALPHA | Hypothesis |
|---|---|---|
| nb17a | 0.5 | More weight on Perch sigmoid → if better, MLP is overconfident |
| nb17b | 0.6 | Slight Perch bias |
| nb17c | 0.8 | Slight MLP bias |
| nb17d | 0.9 | Heavy MLP weight → if better, MLP signal dominates Perch |

(Skip 0.7 — that IS nb15a, LB 0.883.)

---

## Implementation Steps

### Step 1: Copy nb15a as the template

```powershell
cd "c:\Users\alexy\Documents\Claude_projects\Kaggle competition\bird_clef\kaggle_notebooks"
cp 15a_logit_cls_blend.py 17a_alpha_sweep_05.py
cp 15a_logit_cls_blend.py 17b_alpha_sweep_06.py
cp 15a_logit_cls_blend.py 17c_alpha_sweep_08.py
cp 15a_logit_cls_blend.py 17d_alpha_sweep_09.py
```

### Step 2: Edit each kernel — change ALPHA + diagnostics filename

In each `17x_alpha_sweep_XX.py`, change exactly:

```python
# Line ~108:
ALPHA      = 0.5    # was 0.7 in nb15a — this is the only training change

# Diagnostics block — change "nb15a" -> "nb17a" everywhere:
auc_df.to_csv(OUT_DIR / "per_class_auc_nb17a.csv", index=False)
diagnostics = {
    "notebook": "nb17a",
    "model": "alpha-sweep-05",
    "alpha": 0.5,
    ...
}
with open(OUT_DIR / "diagnostics_nb17a.json", "w") as f:
    ...
plt.savefig(OUT_DIR / "pred_dist_nb17a.png", ...)
```

Keep everything else byte-identical to nb15a.

### Step 3: Create kernel-metadata files

For nb17a (`17a_alpha_sweep_05-metadata.json`):

```json
{
  "id": "alexycactus/birdclef-2026-alpha-sweep-05",
  "title": "BirdCLEF 2026 Alpha Sweep 05",
  "code_file": "17a_alpha_sweep_05.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": false,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["rishikeshjani/perch-onnx-for-birdclef-2026"],
  "competition_sources": ["birdclef-2026"],
  "kernel_sources": [],
  "model_sources": ["google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"]
}
```

**CRITICAL:** Title must be slug-clean. `"Alpha Sweep 05"` → slug `alpha-sweep-05`. Never
write `"0.5"` in the title — Kaggle converts `.` → `-`, breaking the id match (409 Conflict).

Same metadata pattern for 17b/c/d, replacing `05` → `06`/`08`/`09`.

### Step 4: Push sequentially (NEVER parallel — race condition)

```powershell
cd "c:\Users\alexy\Documents\Claude_projects\Kaggle competition\bird_clef\kaggle_notebooks"

cp 17a_alpha_sweep_05-metadata.json kernel-metadata.json; kaggle kernels push -p .
cp 17b_alpha_sweep_06-metadata.json kernel-metadata.json; kaggle kernels push -p .
cp 17c_alpha_sweep_08-metadata.json kernel-metadata.json; kaggle kernels push -p .
cp 17d_alpha_sweep_09-metadata.json kernel-metadata.json; kaggle kernels push -p .
```

### Step 5: Monitor + submit

```powershell
kaggle kernels status alexycactus/birdclef-2026-alpha-sweep-05
# ... etc for each
```

Each kernel completes in ~5 min. Submit each `submission.csv` to the competition manually
(Kaggle UI) or via:

```powershell
kaggle competitions submit birdclef-2026 -f submission.csv -m "alpha=0.5" \
  -k alexycactus/birdclef-2026-alpha-sweep-05 -v 1
```

---

## Kaggle Operational Gotchas (from prior sessions)

| Issue | Fix |
|---|---|
| SSL error pushing to GitHub | `git -c http.sslVerify=false push origin master` |
| Slug conflict (409) — title has `.` | Keep titles slug-clean; `0.5` in title → `0-5` in slug |
| Parallel push race condition | NEVER push multiple kernels simultaneously from the same dir |
| Mojibake in generator scripts | Don't write em-dashes in Python string literals; use plain hyphens |
| `docker_image` in metadata | NEVER set it — breaks competition data mounting |

---

## Git Push Workflow (after implementation)

```powershell
git add "Kaggle competition/bird_clef/kaggle_notebooks/17*"
git add "Kaggle competition/bird_clef/docs/plans/nb17-implementation.md"
git commit -m "feat(birdclef): nb17 alpha sweep — 4 kernels (0.5/0.6/0.8/0.9)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git -c http.sslVerify=false push origin master
git -c http.sslVerify=false subtree push --prefix="Kaggle competition/bird_clef" birdclef master
```

---

## Files to Create

```
kaggle_notebooks/
├── 17a_alpha_sweep_05.py
├── 17a_alpha_sweep_05-metadata.json
├── 17b_alpha_sweep_06.py
├── 17b_alpha_sweep_06-metadata.json
├── 17c_alpha_sweep_08.py
├── 17c_alpha_sweep_08-metadata.json
├── 17d_alpha_sweep_09.py
└── 17d_alpha_sweep_09-metadata.json
```

---

## Success Criteria

- All 4 kernels complete and produce `submission.csv`
- LB scores collected for all 4
- Decision tree:
  - **All 4 within ±0.003 of 0.883** → calibration is NOT the bottleneck. Move to nb18 (CNN) and skip learnable-alpha experiments.
  - **One value beats 0.883 by > 0.005** → that ALPHA becomes the new default. Update nb15a → nb15a-v2.
  - **Spread > 0.01 across the 4 values** → per-class calibration matters. Plan nb19 with learnable per-class alpha.

Update `results/experiment_log.md` with all 4 LB scores and the conclusion.
