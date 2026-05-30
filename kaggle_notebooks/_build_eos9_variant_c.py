"""Merge model_nb22e_cnn.py into eos-9 as Variant C: SWAP Model_22 for ours.

Variant C: remove Model_22 from the solutions dict and replace its slot
(weight 0.020) with Model_nb22e_CNN. Model count stays at 3 -- routes
via direct_add3, no router patch needed.

Outputs:
  birdclef-2026-eos-9-variant-c.ipynb
  birdclef-2026-eos-9-variant-c-metadata.json

Run from kaggle_notebooks/:
  python _build_eos9_variant_c.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC_NB = HERE / "birdclef-2026-eos-9.ipynb"
DST_NB = HERE / "birdclef-2026-eos-9-variant-c.ipynb"
DST_META = HERE / "birdclef-2026-eos-9-variant-c-metadata.json"
CELL_SRC = HERE / "model_nb22e_cnn.py"

nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
cell_text = CELL_SRC.read_text(encoding="utf-8")
# Match the FULL guard string, not just "if 'Model_nb22e_CNN'" -- the latter
# also appears in the file's comment header, so find() picked up the comment
# first and the cell started mid-comment with broken Python. Caused
# IndentationError in v1+v2 of both variants.
_GUARD = "if 'Model_nb22e_CNN' in _ensemble_models:"
_start = cell_text.find(_GUARD)
assert _start >= 0, f"Could not find guard {_GUARD!r}"
cell_code = cell_text[_start:]

def md_cell(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code_cell(text: str):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}

# ---- 1. Patch Cell 4 (solutions dict) -----------------------------------
# Replace the Model_22 line outright with our Model_nb22e_CNN line. Model_74
# keeps weight 0.967. Model_51 keeps weight 0.013. Total: 0.020 + 0.013 + 0.967 = 1.0.
cell4_src = "".join(nb["cells"][4]["source"])
old_m22 = "  {'Model':'Model_22','subm':'subm_22.csv', 'weight':0.020,'xSED':[],         'LB':'0.928'},\n"
new_n22 = "  {'Model':'Model_nb22e_CNN','subm':'subm_nb22e_CNN.csv','weight':0.020,'xSED':[],'LB':'0.930'},\n"
assert old_m22 in cell4_src, "Could not find Model_22 line to swap"
patched_cell4 = cell4_src.replace(old_m22, new_n22)
nb["cells"][4]["source"] = patched_cell4.splitlines(keepends=True)

# ---- 2. (No router patch needed -- still 3 models, direct_add3 used) ----

# ---- 3. Insert Model_nb22e_CNN cells RIGHT AFTER Cell 5 -----------------
# v1 placed our cell at index 24 (after all Model_X cells). That errored --
# likely because pip installs in Model_51/74's cells subtly broke timm or
# torchaudio for our cell. v2 puts us at index 6 so we run in a clean env
# directly after `_ensemble_models` is built in Cell 5. Bonus: if any later
# Model_X errors, our subm_nb22e_CNN.csv is already on disk for the blend.
new_md = md_cell("## Model_nb22e_CNN\n\nVariant C: replaces Model_22 (LB 0.928). "
                 "Same weight slot (0.020). Bet: our NS-trained CNN provides more "
                 "diversity with Model_74 than Model_22's Perch variant did.\n"
                 "(v2: placed BEFORE other Model_X cells to avoid env pollution.)\n")
new_code = code_cell(cell_code)
INSERT_AT = 6
nb["cells"] = nb["cells"][:INSERT_AT] + [new_md, new_code] + nb["cells"][INSERT_AT:]

# ---- 4. Write outputs ---------------------------------------------------
DST_NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")

meta = {
    "id": "alexycactus/birdclef-2026-eos9-variant-c-model-22-swap",
    "title": "BirdCLEF 2026 EoS9 Variant C Model 22 Swap",
    "code_file": "birdclef-2026-eos-9-variant-c.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": False,
    "dataset_sources": [
        "alexycactus/birdclef-2026-cnn-ns1-checkpoints",
        "rishikeshjani/perch-onnx-for-birdclef-2026",
        "hideyukizushi/sgkfk-202604041716",
        "jaejohn/perch-meta",
        "tuckerarrants/bc2026-distilled-sed-public",
        "tuckerarrants/birdclef-2026-waveform-cache",
        "tuckerarrants/perch-v2-no-dft-onnx",
    ],
    "competition_sources": ["birdclef-2026"],
    "kernel_sources": ["ashok205/tf-wheels"],
    "model_sources": [],
}
DST_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

print(f"Wrote {DST_NB.name}  ({DST_NB.stat().st_size} bytes, {len(nb['cells'])} cells)")
print(f"Wrote {DST_META.name}")
print()
print("=== Patches applied ===")
print("  Cell 4 : Model_22 -> Model_nb22e_CNN (same weight 0.020)")
print("  Insert : 2 new cells after Model_74")
print()
print("Push: cp birdclef-2026-eos-9-variant-c-metadata.json kernel-metadata.json && kaggle kernels push -p .")
