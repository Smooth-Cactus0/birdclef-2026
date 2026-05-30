"""Generate 5 single-fold variants of the nb23 NS round 2 trainer.

Each kernel trains ONE fold for 25 epochs. ~9h per kernel on T4, fits in 12h.
5 kernels x ~9h = ~45h GPU total (sequentially queued on Kaggle, 2 concurrent).

Run:  python _gen_nb23_folds.py
Push: cp 23_cnn_ns2_foldN-metadata.json kernel-metadata.json && \
      kaggle kernels push -p . --accelerator NvidiaTeslaT4
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
BASE_PY = HERE / "23_cnn_ns2_train.py"
BASE_JSON = HERE / "23_cnn_ns2_train-metadata.json"

base_text = BASE_PY.read_text(encoding="utf-8")
base_meta = json.loads(BASE_JSON.read_text(encoding="utf-8"))

for fold in range(5):
    new_py = HERE / f"23_cnn_ns2_fold{fold}.py"
    text = base_text
    text = text.replace(
        "FOLD_INDICES         = [0, 1]    # RUN 1: train folds 0 and 1",
        f"FOLD_INDICES         = [{fold}]    # single-fold run for fold {fold}",
    )
    text = text.replace(
        "# BirdCLEF 2026 -- nb23 NS round 2 TRAINER  (template; single-fold variants",
        f"# BirdCLEF 2026 -- nb23 NS round 2 TRAINER  (single fold {fold})  # ",
    )
    new_py.write_text(text, encoding="utf-8")
    print(f"Wrote {new_py.name}")

    new_meta = HERE / f"23_cnn_ns2_fold{fold}-metadata.json"
    meta = dict(base_meta)
    meta["id"]        = f"alexycactus/birdclef-2026-cnn-ns2-fold{fold}"
    meta["title"]     = f"BirdCLEF 2026 CNN NS2 Fold{fold}"
    meta["code_file"] = f"23_cnn_ns2_fold{fold}.py"
    new_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {new_meta.name}")

print("\nDone. Push each kernel sequentially (max 2 concurrent on Kaggle GPU):")
for fold in range(5):
    print(f"  cp 23_cnn_ns2_fold{fold}-metadata.json kernel-metadata.json && "
          f"kaggle kernels push -p . --accelerator NvidiaTeslaT4")
