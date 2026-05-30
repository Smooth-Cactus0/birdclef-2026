# Standalone diagnostic for the Model_nb22e_CNN cell that errored inside
# Variants B and C of the eos-9 merger. This is the SAME pipeline as the
# inserted cell, except: (a) the `if Model_nb22e_CNN in _ensemble_models:`
# guard is stripped (we always run), (b) sentinel checkpoints are written
# to /kaggle/working/_diag.log after every major step so we can see where
# things break even if stdout is suppressed.
import os, glob, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

_DIAG_PATH = "/kaggle/working/_diag.log"

def _diag(msg):
    """Append a timestamped checkpoint to the sentinel log + stdout.
    Flushes after every write so we see partial progress even on crash.
    """
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    print(line, end="", flush=True)
    with open(_DIAG_PATH, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()

# Clear any prior sentinel so we know this run wrote it.
if os.path.exists(_DIAG_PATH):
    os.remove(_DIAG_PATH)

_diag("STEP 0: script started")
_diag(f"  python={os.sys.version.split()[0]}, torch={torch.__version__}, torchaudio={torchaudio.__version__}")

# ---- timm import is suspect (Model_22 also installs/imports timm-adjacent things) ----
try:
    import timm
    _diag(f"STEP 1: timm imported, version={timm.__version__}")
except Exception as e:
    _diag(f"STEP 1 FAILED: timm import raised {type(e).__name__}: {e}")
    raise

# ---- Determinism ----
import random as _random_n22
_SEED_n22 = 42
_random_n22.seed(_SEED_n22)
np.random.seed(_SEED_n22)
torch.manual_seed(_SEED_n22)
_device_n22 = "cpu"
_diag(f"STEP 2: seeds set, device={_device_n22}")

# ---- Constants ----
PERCH_SR     = 32_000
N_CLASSES_n22 = 234
SC_FILE_SEC  = 60
OUTPUT_SLOTS = 12
CNN_BACKBONE = "tf_efficientnet_b0_ns"
CNN_N_MELS   = 224
CNN_N_FFT    = 4096
CNN_HOP      = 1252
CNN_WIN_SEC  = 20
CNN_WIN_SMP  = CNN_WIN_SEC * PERCH_SR
CNN_STRIDE   = 5
CNN_N_WIN    = 9
CNN_FILES_PER_BATCH = 4
CNN_IO_WORKERS      = 4

# ---- Paths ----
BASE_DIR_n22 = (
    Path("/kaggle/input/competitions/birdclef-2026")
    if Path("/kaggle/input/competitions/birdclef-2026").exists()
    else Path("/kaggle/input/birdclef-2026")
)
_diag(f"STEP 3: BASE_DIR_n22={BASE_DIR_n22}, exists={BASE_DIR_n22.exists()}")
_diag(f"  listing /kaggle/input/: {os.listdir('/kaggle/input/')}")

sample_sub_n22 = pd.read_csv(BASE_DIR_n22 / "sample_submission.csv")
PRIMARY_LABELS_n22 = [c for c in sample_sub_n22.columns if c != "row_id"]
_diag(f"STEP 4: sample_submission loaded, shape={sample_sub_n22.shape}, "
      f"{len(PRIMARY_LABELS_n22)} species cols")

test_dir_n22 = BASE_DIR_n22 / "test_soundscapes"
test_files_n22 = sorted(test_dir_n22.glob("*.ogg")) if test_dir_n22.exists() else []
if not test_files_n22:
    _diag(f"STEP 5: test_soundscapes empty -- falling back to train_soundscapes")
    test_files_n22 = sorted((BASE_DIR_n22 / "train_soundscapes").glob("*.ogg"))[:16]
_diag(f"STEP 5: {len(test_files_n22)} test files identified")

# ---- Checkpoint discovery ----
_ckpt_hits = sorted(glob.glob("/kaggle/input/**/fold*_best.pth", recursive=True))
_diag(f"STEP 6a: raw fold*_best.pth hits ({len(_ckpt_hits)}): {_ckpt_hits[:10]}")
_ckpt_hits_filt = [p for p in _ckpt_hits if "ns1" in p.lower()]
_diag(f"STEP 6b: NS1-filtered hits ({len(_ckpt_hits_filt)}): {_ckpt_hits_filt}")
_ckpts = {}
for _p in _ckpt_hits_filt:
    _nm  = Path(_p).name
    _idx = int(_nm.replace("fold", "").replace("_best.pth", ""))
    if _idx not in _ckpts:
        _ckpts[_idx] = _p
_diag(f"STEP 6c: {len(_ckpts)} unique-fold checkpoints discovered")
assert len(_ckpts) == 5, f"Expected 5 NS R1 checkpoints, got {len(_ckpts)}"

# ---- Mel transform ----
_mel_n22 = torchaudio.transforms.MelSpectrogram(
    sample_rate=PERCH_SR, n_mels=CNN_N_MELS, n_fft=CNN_N_FFT,
    hop_length=CNN_HOP, f_min=0, f_max=16_000, power=2.0,
    norm="slaney", mel_scale="htk",
).to(_device_n22)
_db_n22 = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(_device_n22)
_diag("STEP 7: mel transform built")

# ---- Architecture (the suspect: timm + nan_to_num + autocast on CPU) ----
class _SEDHead_n22(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc_att = nn.Linear(in_features, n_classes)
        self.fc_cla = nn.Linear(in_features, n_classes)
    def forward(self, x):
        x = x.transpose(1, 2)
        att      = torch.softmax(torch.tanh(self.fc_att(x)), dim=1)
        framelg  = self.fc_cla(x)
        cliplg   = (att * framelg).sum(dim=1)
        return cliplg

class _BirdCNN_n22(nn.Module):
    def __init__(self, n_classes=N_CLASSES_n22, in_chans=3):
        super().__init__()
        self.backbone = timm.create_model(
            CNN_BACKBONE, pretrained=False, in_chans=in_chans,
            num_classes=0, global_pool="", features_only=False,
        )
        with torch.no_grad():
            feat = self.backbone.forward_features(
                torch.zeros(1, in_chans, CNN_N_MELS, 512)
            )
            self.feat_dim = feat.shape[1]
        self.head = _SEDHead_n22(self.feat_dim, n_classes)
    def forward(self, x):
        feat = self.backbone.forward_features(x).mean(dim=2)
        return self.head(feat)

_diag("STEP 8: architecture classes defined")

# ---- Build first model + load first ckpt (most likely failure point) ----
try:
    _m0 = _BirdCNN_n22().to(_device_n22)
    _diag(f"STEP 9a: _BirdCNN_n22() instantiated, feat_dim={_m0.feat_dim}")
except Exception as e:
    _diag(f"STEP 9a FAILED: {type(e).__name__}: {e}")
    raise

try:
    _m0.load_state_dict(torch.load(_ckpts[0], map_location=_device_n22))
    _m0.eval()
    _diag("STEP 9b: fold0 checkpoint loaded")
except Exception as e:
    _diag(f"STEP 9b FAILED: {type(e).__name__}: {e}")
    raise

# ---- Single-file inference smoke test ----
try:
    wav, sr = torchaudio.load(str(test_files_n22[0]))
    if sr != PERCH_SR:
        wav = torchaudio.functional.resample(wav, sr, PERCH_SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    y = wav.squeeze(0).numpy().astype(np.float32)
    target = 60 * PERCH_SR
    y = np.pad(y, (0, max(0, target - len(y))))[:target]
    _diag(f"STEP 10: loaded first audio file, samples={len(y)}, sr={PERCH_SR}")
except Exception as e:
    _diag(f"STEP 10 FAILED: {type(e).__name__}: {e}")
    raise

try:
    chunk0 = y[:CNN_WIN_SMP].astype(np.float32)
    chunk0 = chunk0 / (np.max(np.abs(chunk0)) + 1e-8)
    chunk_t = torch.from_numpy(chunk0).unsqueeze(0).to(_device_n22)
    with torch.cuda.amp.autocast(enabled=False):
        with torch.inference_mode():
            mel = _mel_n22(chunk_t)
            mel = _db_n22(mel)
            mel = torch.nan_to_num(mel, nan=-80.0, posinf=0.0, neginf=-80.0)
            mlo = mel.amin(dim=(1, 2), keepdim=True)
            mhi = mel.amax(dim=(1, 2), keepdim=True)
            mel = (mel - mlo) / (mhi - mlo + 1e-6)
            mel = mel.unsqueeze(1).repeat(1, 3, 1, 1)
            logits = _m0(mel)
            preds = torch.sigmoid(logits).cpu().numpy()
    _diag(f"STEP 11: single-window inference OK, preds shape={preds.shape}, "
          f"min={float(preds.min()):.4f} max={float(preds.max()):.4f}")
except Exception as e:
    _diag(f"STEP 11 FAILED: {type(e).__name__}: {e}")
    raise

_diag("STEP 12: SUCCESS -- pipeline runs end-to-end on a single window")
_diag("Diagnostic complete. If you see this line, the cell content is fine.")

# Save a stub submission so kaggle kernels output downloads something.
pd.DataFrame({"row_id": ["diag_ok"], **{c: [0.0] for c in PRIMARY_LABELS_n22}}).to_csv(
    "/kaggle/working/diag_result.csv", index=False
)
