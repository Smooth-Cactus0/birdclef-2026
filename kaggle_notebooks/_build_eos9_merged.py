"""Merge model_nb22e_cnn.py into birdclef-2026-eos-9.ipynb as Variant B.

Variant B: adds Model_nb22e_CNN (CNN-only NS R1 inference, weight 0.02) to
the 3-model public ensemble, producing a 4-model ensemble.

Outputs:
  birdclef-2026-eos-9-nb22e-cnn.ipynb  -- merged notebook
  birdclef-2026-eos-9-nb22e-cnn-metadata.json -- kernel metadata

Run from kaggle_notebooks/:
  python _build_eos9_merged.py
"""
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
SRC_NB = HERE / "birdclef-2026-eos-9.ipynb"
DST_NB = HERE / "birdclef-2026-eos-9-nb22e-cnn.ipynb"
DST_META = HERE / "birdclef-2026-eos-9-nb22e-cnn-metadata.json"
CELL_SRC = HERE / "model_nb22e_cnn.py"

nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
cell_text = CELL_SRC.read_text(encoding="utf-8")

# Strip the leading comment block that explains this file isn't standalone.
# Everything from `if 'Model_nb22e_CNN'` onward is the actual cell content.
# Match the FULL guard string, not just "if 'Model_nb22e_CNN'" -- the latter
# also appears in the file's comment header, so find() picked up the comment
# first and the cell started mid-comment with broken Python. Caused
# IndentationError in v1+v2 of both variants.
_GUARD = "if 'Model_nb22e_CNN' in _ensemble_models:"
_start = cell_text.find(_GUARD)
assert _start >= 0, f"Could not find guard {_GUARD!r}"
cell_code = cell_text[_start:]

# Helper to construct an ipynb cell dict. Notebooks expect `source` as a list
# of strings (lines), and code cells need outputs/execution_count fields.
def md_cell(text: str):
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }

def code_cell(text: str):
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }

# ---- 1. Patch Cell 4 (solutions dict) ---------------------------------
cell4_src = "".join(nb["cells"][4]["source"])
assert "'Model':'Model_74'" in cell4_src, "Could not locate Model_74 line in cell 4"

new_model_line = (
    "  {'Model':'Model_nb22e_CNN','subm':'subm_nb22e_CNN.csv',"
    "'weight':0.02,'xSED':[],         'LB':'0.930'},\n"
)
# Insert our model row right BEFORE the closing `]` of the Models list. We
# place it last so the Model_74 entry (weight 0.967) keeps its position and
# weight; ours is a small diversity nudge.
patched_cell4 = cell4_src.replace(
    "  {'Model':'Model_74','subm':'subm_74.csv', 'weight':0.967,'xSED':[0.60,0.40],'LB':'0.949'}\n",
    "  {'Model':'Model_74','subm':'subm_74.csv', 'weight':0.957,'xSED':[0.60,0.40],'LB':'0.949'},\n"
    + new_model_line,
)
# Sanity: weight 0.957 + 0.020 + 0.013 + 0.020 = 1.010. Document but don't
# normalize -- direct_add4 will normalize via direct_addsafe pattern if used.
assert patched_cell4 != cell4_src, "Cell 4 patch did not apply"
nb["cells"][4]["source"] = patched_cell4.splitlines(keepends=True)

# ---- 2. Patch Cell 28 (add direct_add4 routing in `direct()`) ---------
cell28_src = "".join(nb["cells"][28]["source"])
assert "if len(_ensemble_models) == 3:  return direct_add3()" in cell28_src
patched_cell28 = cell28_src.replace(
    "def direct():\n"
    "    if len(_ensemble_models) == 2:  return direct_add2()\n"
    "    if len(_ensemble_models) == 3:  return direct_add3()\n",
    "def direct():\n"
    "    if len(_ensemble_models) == 2:  return direct_add2()\n"
    "    if len(_ensemble_models) == 3:  return direct_add3()\n"
    "    if len(_ensemble_models) == 4:  return direct_add4()\n",
)
assert patched_cell28 != cell28_src, "Cell 28 patch did not apply"
nb["cells"][28]["source"] = patched_cell28.splitlines(keepends=True)

# ---- 3. Insert new Model_nb22e_CNN cells after Model_74 (cell 23) -----
# Cell 22 is "## Model_74" markdown, cell 23 is Model_74 code. We insert
# after cell 23 (= before original cell 24 "## division_attention check").
new_md = md_cell("## Model_nb22e_CNN\n\nNS round 1 5-fold EfficientNet-B0 CNN ensemble "
                 "(LB 0.930 as part of nb22e seeded standalone). CNN-only -- Perch+MLP "
                 "path dropped since eos-9 already has Perch coverage via Model_22/51/74. "
                 "Weight 0.02: low-weight diversity nudge.\n")
new_code = code_cell(cell_code)

# v2 fix: insert at index 6 (right after `_ensemble_models` extract in Cell 5)
# instead of 24. v1 errored when our cell ran after Model_51/74 pip installs;
# running early keeps timm/torchaudio environment clean.
INSERT_AT = 6
nb["cells"] = nb["cells"][:INSERT_AT] + [new_md, new_code] + nb["cells"][INSERT_AT:]

# ---- 4. Write outputs --------------------------------------------------
DST_NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")

meta = {
    "id": "alexycactus/birdclef-2026-eos9-nb22e-cnn",
    "title": "BirdCLEF 2026 EoS9 nb22e CNN",
    "code_file": "birdclef-2026-eos-9-nb22e-cnn.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,  # Kaggle requires private for new kernels with active competition_sources
    "enable_gpu": False,
    "enable_internet": False,
    "dataset_sources": [
        # Our addition
        "alexycactus/birdclef-2026-cnn-ns1-checkpoints",
        # Inherited from eos-9 (enumerated via grep of /kaggle/input/datasets/)
        "rishikeshjani/perch-onnx-for-birdclef-2026",
        "hideyukizushi/sgkfk-202604041716",
        "jaejohn/perch-meta",
        "tuckerarrants/bc2026-distilled-sed-public",
        "tuckerarrants/birdclef-2026-waveform-cache",
        "tuckerarrants/perch-v2-no-dft-onnx",
    ],
    "competition_sources": ["birdclef-2026"],
    "kernel_sources": [
        # Inherited from eos-9 (notebook outputs mounted via /kaggle/input/notebooks/)
        "ashok205/tf-wheels",
        "vyankteshdwivedi/notebook1b25083f0d",
    ],
    "model_sources": [],
}
DST_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

print(f"Wrote {DST_NB.name}  ({DST_NB.stat().st_size} bytes, {len(nb['cells'])} cells)")
print(f"Wrote {DST_META.name}")
print()
print("=== Patches applied ===")
print("  Cell 4  : added Model_nb22e_CNN to solutions (weight 0.020, Model_74 0.967->0.957)")
print("  Cell 28 : direct() now routes 4 models via direct_add4()")
print(f"  Insert  : 2 new cells (markdown + code) at index {INSERT_AT}")
print()
print("=== TODO before pushing ===")
print("  1. Add eos-9's OWN dataset_sources to the metadata file (open the")
print("     original kernel on Kaggle to enumerate them: TF wheels, ash datasets,")
print("     perch onnx wheels, etc. that Model_22/51/74 mount). Without those")
print("     the existing models will fail and the ensemble won't run.")
print("  2. Inspect the merged notebook locally (Jupyter or VSCode) before push.")
print("  3. Push with `kaggle kernels push` from this directory.")
