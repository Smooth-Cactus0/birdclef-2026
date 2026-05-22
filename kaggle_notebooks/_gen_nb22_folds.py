"""Generate 5 single-fold variants of the nb22 NS trainer.

Each kernel trains ONE fold for 25 epochs. ~9h per kernel on T4, fits in 12h.
5 kernels x ~9h = ~45h GPU total (sequentially queued on Kaggle).

Run:  python _gen_nb22_folds.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
BASE_PY = HERE / "22_cnn_ns1_train_run1.py"
BASE_JSON = HERE / "22_cnn_ns1_train_run1-metadata.json"

base_text = BASE_PY.read_text(encoding="utf-8")
base_meta = json.loads(BASE_JSON.read_text(encoding="utf-8"))

for fold in range(5):
    # Script
    new_py = HERE / f"22_cnn_ns1_fold{fold}.py"
    text = base_text
    # Update FOLD_INDICES to a single fold
    text = text.replace(
        "FOLD_INDICES         = [0, 1]    # RUN 1: train folds 0 and 1",
        f"FOLD_INDICES         = [{fold}]    # single-fold run for fold {fold}",
    )
    # Update header line that mentions "run 1 = folds 0 + 1"
    text = text.replace(
        "# BirdCLEF 2026 -- nb22 NS round 1 TRAINER  (run 1 = folds 0 + 1)",
        f"# BirdCLEF 2026 -- nb22 NS round 1 TRAINER  (single fold {fold})",
    )
    new_py.write_text(text, encoding="utf-8")
    print(f"Wrote {new_py.name}")

    # Metadata
    new_meta = HERE / f"22_cnn_ns1_fold{fold}-metadata.json"
    meta = dict(base_meta)
    meta["id"]        = f"alexycactus/birdclef-2026-cnn-ns1-fold{fold}"
    meta["title"]     = f"BirdCLEF 2026 CNN NS1 Fold{fold}"
    meta["code_file"] = f"22_cnn_ns1_fold{fold}.py"
    new_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {new_meta.name}")

print("\nDone. Push each kernel sequentially:")
for fold in range(5):
    print(f"  cp 22_cnn_ns1_fold{fold}-metadata.json kernel-metadata.json && "
          f"kaggle kernels push -p . --accelerator NvidiaTeslaT4")
