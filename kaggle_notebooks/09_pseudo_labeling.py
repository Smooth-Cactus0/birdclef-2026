# %%
# =============================================================================
# BirdCLEF 2026 -- Pseudo-Labeling Pipeline (Round 1)
# =============================================================================
# Models    : Bird model (B4/B3) + RegNetY + optional ViT models (EVA-02, DINOv2)
# Consensus : 2-of-3 agreement on top-1 prediction
# Thresholds: stratified by class rarity (common=0.70 ... very rare=0.90)
# Power scale: probs = probs ** 0.7  (softens overconfidence before threshold)
# Output    : pseudo_labels_r1.csv (use as extra training data in next run)
#
# HOW TO USE:
#   1. Run nb06 to train backbone checkpoints (B4, B3, RegNetY)
#   2. Point CHECKPOINT_DIR to the folder containing *.pth files
#   3. Run this notebook -- it writes pseudo_labels_r1.csv to OUTPUT_DIR
#   4. In nb06/nb07 next round, add pseudo_labels_r1.csv to training data
#      with sample weight = 0.5 (PL-R1 weight per design doc)
# =============================================================================

# %% [markdown]
# # Pseudo-Labeling Pipeline (Round 1)
# Generates high-confidence pseudo-labels on unlabeled PAM soundscapes
# using multi-model consensus. This is the single biggest differentiator
# between top-5 and top-1 in BirdCLEF 2025 (Nikita Babych, 4+ rounds).

# %%
# -- Install / version pins ----------------------------------------------------
# !pip install -q timm==1.0.3  # uncomment on Kaggle if needed

import os, gc, json, warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import torch
import torch.nn as nn
import timm
warnings.filterwarnings('ignore')

# -- Paths ---------------------------------------------------------------------
BASE_DIR   = (Path('/kaggle/input/birdclef-2026')
              if Path('/kaggle/input/birdclef-2026').exists()
              else Path('birdclef-2026'))
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Where trained checkpoints from nb06/nb07 live.
# On Kaggle: add the nb06 output dataset and set this path accordingly.
CHECKPOINT_DIR = OUTPUT_DIR   # default: same session outputs

NUM_WORKERS = 0 if os.name == 'nt' else 4

# -- Config --------------------------------------------------------------------
CFG = dict(
    SR          = 32000,
    N_FFT       = 1024,
    HOP_LENGTH  = 320,
    N_MELS      = 128,
    FMIN        = 40,
    FMAX        = 15000,
    DURATION    = 5,          # inference chunk length (seconds)
    POWER_SCALE = 0.7,        # compress overconfident probs: p ** 0.7
    MAX_FILES   = 500,        # process first N unlabeled files for R1 (set None for all)
    DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu',
    SEED        = 42,
)
print(f"Device: {CFG['DEVICE']}")
print(f"torch: {torch.__version__}, timm: {timm.__version__}")

# %% [markdown]
# ## Label Setup
# We use the full 234-class taxonomy for label indices.
# The bird-only subset (162 classes) is used for CNN checkpoint loading.
# The non-bird subset (72 classes) is handled by nb08 checkpoints.

# %%
# -- Label setup ---------------------------------------------------------------
taxonomy    = pd.read_csv(BASE_DIR / 'taxonomy.csv')
label_list  = taxonomy['primary_label'].tolist()   # 234 total
label2idx   = {l: i for i, l in enumerate(label_list)}
NUM_CLASSES = len(label_list)

# Bird-only subset
bird_labels  = taxonomy[taxonomy['class_name'] == 'Aves']['primary_label'].tolist()  # 162
bird2idx     = {l: i for i, l in enumerate(bird_labels)}
BIRD_CLASSES = len(bird_labels)

# Non-bird subset
nonbird_labels  = taxonomy[taxonomy['class_name'] != 'Aves']['primary_label'].tolist()  # 72
nonbird2idx     = {l: i for i, l in enumerate(nonbird_labels)}
NONBIRD_CLASSES = len(nonbird_labels)

print(f"Total classes: {NUM_CLASSES} | Bird: {BIRD_CLASSES} | Non-bird: {NONBIRD_CLASSES}")

# %% [markdown]
# ## Class Rarity Lookup
# Confidence thresholds are stratified by how many training clips each class has.
# Rare species need a *higher* threshold -- a wrong pseudo-label for a class with
# 3 real clips would represent a 33% corruption of its training data.

# %%
# -- Class rarity and thresholds -----------------------------------------------
train_df   = pd.read_csv(BASE_DIR / 'train.csv')
clip_counts = train_df['primary_label'].value_counts().to_dict()

# Rarity tier breakdown for transparency
tiers = {'common (>500)': 0, 'medium (100-500)': 0, 'rare (10-100)': 0, 'very rare (<10)': 0}
for label in label_list:
    n = clip_counts.get(label, 0)
    if n > 500:   tiers['common (>500)']   += 1
    elif n > 100: tiers['medium (100-500)'] += 1
    elif n > 10:  tiers['rare (10-100)']    += 1
    else:         tiers['very rare (<10)']  += 1
print("Class rarity distribution:")
for tier, count in tiers.items():
    print(f"  {tier}: {count} classes")

def get_threshold(label: str) -> float:
    """
    Rarity-stratified confidence threshold for pseudo-label acceptance.
    Higher threshold for rarer classes to prevent noise from polluting
    already-small training sets.
    """
    n = clip_counts.get(label, 0)
    if n > 500: return 0.70   # common -- accept moderate confidence
    if n > 100: return 0.80   # medium
    if n > 10:  return 0.85   # rare
    return 0.90               # very rare (< 10 clips, all non-bird species)

# %% [markdown]
# ## BirdModel Architecture
# Same architecture as nb06/nb07: timm backbone, in_chans=1 for mono spectrograms.
# We load fold-1 checkpoints (best checkpoint saved during training).

# %%
# -- Model definition (must match nb06/nb07 exactly) ---------------------------
class BirdModel(nn.Module):
    """timm backbone, in_chans=1, configurable num_classes."""
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            in_chans=1, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)


def load_model(checkpoint_path, model_name: str, num_classes: int, device: str):
    """
    Load a trained BirdModel from a .pth checkpoint.
    Returns the model in eval mode, or None if file is missing.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"  ? Checkpoint not found: {path}")
        return None
    model = BirdModel(model_name, num_classes, pretrained=False)
    model.load_state_dict(torch.load(str(path), map_location=device))
    model = model.to(device).eval()
    print(f"  OK Loaded: {path.name}  ({num_classes} classes)")
    return model


# %% [markdown]
# ## Load Available Models
# We try to load up to 3 bird-pipeline models from nb06 checkpoints.
# Consensus requires >= 2 models; the notebook warns if fewer are available.
# Non-bird models from nb08 can be added in the same pattern.

# %%
# -- Load checkpoint models ----------------------------------------------------
print("Loading bird-pipeline models (from nb06 checkpoints)...")
models = {}

# EfficientNet-B4: highest accuracy backbone from backbone search
m = load_model(CHECKPOINT_DIR / 'efficientnet_b4_fold1.pth',
               'efficientnet_b4', BIRD_CLASSES, CFG['DEVICE'])
if m: models['b4'] = m

# EfficientNet-B3: faster, useful for diversity
m = load_model(CHECKPOINT_DIR / 'efficientnet_b3_fold1.pth',
               'efficientnet_b3', BIRD_CLASSES, CFG['DEVICE'])
if m: models['b3'] = m

# RegNetY-016: architectural diversity (different inductive bias from EfficientNet)
m = load_model(CHECKPOINT_DIR / 'regnety_016_fold1.pth',
               'regnety_016', BIRD_CLASSES, CFG['DEVICE'])
if m: models['regnety'] = m

# Optional: BirdNET from nb07
m = load_model(CHECKPOINT_DIR / 'birdnet_fold1.pth',
               'efficientnet_b1', BIRD_CLASSES, CFG['DEVICE'])
if m: models['birdnet'] = m

print(f"\nModels loaded: {list(models.keys())}")
if len(models) < 2:
    print("\nWARNING: Need >=2 models for consensus pseudo-labeling.")
    print("         Run nb06 first and ensure checkpoints are in CHECKPOINT_DIR.")
    print(f"         CHECKPOINT_DIR = {CHECKPOINT_DIR}")
else:
    print(f"Consensus will require {max(2, len(models)-1)}-of-{len(models)} agreement.")

# %% [markdown]
# ## Soundscape Chunker
# Each unlabeled PAM recording is split into non-overlapping 5s chunks.
# Each chunk is converted to a log-mel spectrogram tensor ready for inference.
# We use the exact same mel parameters as training (matching feature distributions).

# %%
# -- Soundscape chunker --------------------------------------------------------
def chunk_soundscape(path: Path, cfg: dict):
    """
    Yield (start_sec: float, mel_tensor: torch.Tensor[1, N_MELS, T]) for each 5s chunk.
    Uses non-overlapping windows; partial trailing chunk is discarded.
    """
    try:
        y, _ = librosa.load(str(path), sr=cfg['SR'], mono=True)
    except Exception as e:
        print(f"  Error loading {path.name}: {e}")
        return

    n_samples = cfg['SR'] * cfg['DURATION']
    n_chunks  = len(y) // n_samples   # discard trailing partial chunk

    for i in range(n_chunks):
        chunk = y[i * n_samples: (i + 1) * n_samples].astype(np.float32)
        S     = librosa.feature.melspectrogram(
            y=chunk, sr=cfg['SR'],
            n_fft=cfg['N_FFT'], hop_length=cfg['HOP_LENGTH'],
            n_mels=cfg['N_MELS'], fmin=cfg['FMIN'], fmax=cfg['FMAX'])
        S_db   = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        yield float(i * cfg['DURATION']), torch.from_numpy(S_norm).float().unsqueeze(0)


# %% [markdown]
# ## Consensus Pseudo-Label Assignment
# For each 5s chunk:
# 1. Get sigmoid probabilities from every model
# 2. Apply power scaling: `p = p ** 0.7` (compresses overconfidence)
# 3. Find each model's top-1 prediction
# 4. Check if >=2 models agree on the same top-1 label
# 5. If agreed: check mean confidence against the rarity threshold
# 6. Only emit a pseudo-label if threshold is passed

# %%
# -- Consensus logic -----------------------------------------------------------
def get_consensus_label(mel_tensor: torch.Tensor, models: dict,
                        label_list: list, cfg: dict, device: str):
    """
    Run a single mel chunk through all models and return a consensus pseudo-label.

    Args:
        mel_tensor: shape (1, N_MELS, T) -- single spectrogram chunk (no batch dim)
        models: dict of {name: nn.Module}
        label_list: ordered list of class strings matching model output indices
        cfg: config dict with POWER_SCALE key
        device: torch device string

    Returns:
        (label: str, confidence: float) if consensus passes threshold, else None
    """
    # Add batch dimension: (1, 1, N_MELS, T)
    x = mel_tensor.unsqueeze(0).to(device)

    model_preds = {}
    for name, model in models.items():
        with torch.no_grad():
            logits = model(x)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]   # (num_classes,)
        # Power scaling: soften overconfident predictions before voting
        # e.g., prob=0.9 -> 0.9**0.7 ? 0.927; prob=0.6 -> 0.6**0.7 ? 0.671
        probs = probs ** cfg['POWER_SCALE']
        model_preds[name] = probs

    # Top-1 label and confidence per model
    top1_labels = {name: label_list[int(np.argmax(p))] for name, p in model_preds.items()}
    top1_confs  = {name: float(np.max(p))              for name, p in model_preds.items()}

    # Majority vote: does >=2 models agree?
    vote_counts          = Counter(top1_labels.values())
    best_label, n_votes  = vote_counts.most_common(1)[0]
    min_votes_needed     = 2   # fixed at 2 regardless of ensemble size

    if n_votes < min_votes_needed:
        return None   # no consensus

    # Mean confidence of agreeing models
    agreeing_confs = [top1_confs[n] for n, lbl in top1_labels.items() if lbl == best_label]
    mean_conf      = float(np.mean(agreeing_confs))

    # Rarity-stratified threshold check
    threshold = get_threshold(best_label)
    if mean_conf < threshold:
        return None

    return best_label, mean_conf


# %% [markdown]
# ## Main Pseudo-Labeling Loop
# Processes up to `CFG['MAX_FILES']` unlabeled soundscapes.
# Set `MAX_FILES = None` to process all files (takes ~2-3h on T4 with 3 models).

# %%
# -- Main PL loop --------------------------------------------------------------
unlabeled_dir   = BASE_DIR / 'unlabeled_soundscapes'
unlabeled_files = sorted(unlabeled_dir.glob('*.ogg')) if unlabeled_dir.exists() else []
print(f"Unlabeled soundscapes available: {len(unlabeled_files)}")

if CFG['MAX_FILES'] is not None:
    unlabeled_files = unlabeled_files[:CFG['MAX_FILES']]
    print(f"Processing first {len(unlabeled_files)} files (MAX_FILES={CFG['MAX_FILES']})")

if len(models) < 2:
    print("Skipping PL generation -- need >=2 models loaded.")
    pl_df = pd.DataFrame(columns=['filepath', 'start_sec', 'end_sec',
                                  'primary_label', 'confidence', 'pl_round'])
else:
    pl_records = []
    n_chunks_total = 0

    for file_idx, sf_path in enumerate(unlabeled_files):
        file_records = []
        for start_sec, mel in chunk_soundscape(sf_path, CFG):
            n_chunks_total += 1
            result = get_consensus_label(mel, models, bird_labels, CFG, CFG['DEVICE'])
            if result:
                label, conf = result
                file_records.append({
                    'filepath':     str(sf_path),
                    'start_sec':    start_sec,
                    'end_sec':      start_sec + CFG['DURATION'],
                    'primary_label': label,
                    'confidence':   round(conf, 4),
                    'pl_round':     1,
                })
        pl_records.extend(file_records)

        if (file_idx + 1) % 50 == 0 or file_idx == len(unlabeled_files) - 1:
            print(f"  [{file_idx+1}/{len(unlabeled_files)}]  "
                  f"chunks processed={n_chunks_total}  "
                  f"pseudo-labels so far={len(pl_records)}")

    pl_df = pd.DataFrame(pl_records)
    print(f"\nTotal chunks processed: {n_chunks_total:,}")
    print(f"Pseudo-labels accepted: {len(pl_df):,}  "
          f"({100*len(pl_df)/max(1,n_chunks_total):.1f}% acceptance rate)")
    if len(pl_df) > 0:
        print("\nTop 20 pseudo-labeled species:")
        print(pl_df['primary_label'].value_counts().head(20).to_string())

pl_df.to_csv(OUTPUT_DIR / 'pseudo_labels_r1.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR / 'pseudo_labels_r1.csv'}")


# %% [markdown]
# ## Statistics & Quality Diagnostics
# Visualise the pseudo-label distribution to catch obvious failures:
# - Is the distribution dominated by a handful of common species? (expected)
# - Are there any very-rare species with many pseudo-labels? (suspicious -- check)
# - Is the confidence distribution bimodal? (good -- high confidence = reliable labels)

# %%
# -- Diagnostics plot ----------------------------------------------------------
if len(pl_df) == 0:
    print("No pseudo-labels generated -- cannot plot statistics.")
    print("Ensure >=2 model checkpoints are present in CHECKPOINT_DIR.")
else:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Pseudo-Label Round 1 -- Quality Diagnostics', fontsize=13, fontweight='bold')

    # Plot 1: top-30 species by pseudo-label count
    ax = axes[0]
    top30 = pl_df['primary_label'].value_counts().head(30)
    ax.barh(top30.index[::-1], top30.values[::-1], color='steelblue')
    ax.set_xlabel('Pseudo-label count')
    ax.set_title('Top 30 Species by PL Count')
    ax.tick_params(axis='y', labelsize=7)

    # Plot 2: confidence histogram
    ax = axes[1]
    ax.hist(pl_df['confidence'], bins=40, color='darkorange', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Confidence (after power scaling)')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    ax.axvline(0.70, color='steelblue',  linestyle='--', linewidth=1, label='common thr=0.70')
    ax.axvline(0.90, color='firebrick',  linestyle='--', linewidth=1, label='rare thr=0.90')
    ax.legend(fontsize=8)

    # Plot 3: rare vs common breakdown
    ax = axes[2]
    pl_df['rarity_tier'] = pl_df['primary_label'].apply(lambda lbl: (
        'common (>500)'   if clip_counts.get(lbl, 0) > 500  else
        'medium (100-500)' if clip_counts.get(lbl, 0) > 100 else
        'rare (10-100)'    if clip_counts.get(lbl, 0) > 10  else
        'very rare (<10)'
    ))
    tier_counts = pl_df['rarity_tier'].value_counts()
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    ax.pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%',
           colors=colors[:len(tier_counts)], startangle=90)
    ax.set_title('PL Distribution by Rarity Tier')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pseudo_labels_r1_diagnostics.png',
                bbox_inches='tight', dpi=120)
    plt.show()
    print(f"Diagnostics plot saved -> {OUTPUT_DIR / 'pseudo_labels_r1_diagnostics.png'}")

    # Summary statistics by rarity tier
    print("\nPseudo-labels by rarity tier:")
    print(pl_df.groupby('rarity_tier')['confidence'].agg(['count', 'mean', 'min', 'max'])
          .rename(columns={'count': 'n_labels', 'mean': 'mean_conf',
                           'min': 'min_conf', 'max': 'max_conf'})
          .to_string())


# %% [markdown]
# ## How to Use pseudo_labels_r1.csv in Retraining
#
# In nb06 (Round 2 retraining), load this CSV and append to training data:
# ```python
# pl_r1 = pd.read_csv(OUTPUT_DIR / 'pseudo_labels_r1.csv')
# pl_r1['weight'] = 0.5        # PL-R1 sample weight (lower than real data)
# pl_r1['source_type'] = 'pl'
# # filepath + start_sec -> load offset in BirdDataset._load_audio
# train_df = pd.concat([train_df, pl_r1], ignore_index=True)
# ```
#
# In Round 2, re-run nb09 with the new (stronger) model checkpoints to get
# pseudo_labels_r2.csv with weight=0.65. Repeat for R3/R4 (weight=0.75).

# %%
# -- Save metadata summary ----------------------------------------------------
summary = {
    'pl_round':          1,
    'n_soundscapes':     len(unlabeled_files),
    'n_chunks_processed': int(n_chunks_total) if len(models) >= 2 else 0,
    'n_pseudo_labels':   len(pl_df),
    'acceptance_rate':   round(len(pl_df) / max(1, n_chunks_total), 4) if len(models) >= 2 else 0,
    'models_used':       list(models.keys()),
    'power_scale':       CFG['POWER_SCALE'],
    'thresholds':        {'common': 0.70, 'medium': 0.80, 'rare': 0.85, 'very_rare': 0.90},
    'n_unique_species':  int(pl_df['primary_label'].nunique()) if len(pl_df) > 0 else 0,
}
with open(OUTPUT_DIR / 'pl_r1_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nRound 1 PL Summary:")
print(json.dumps(summary, indent=2))
