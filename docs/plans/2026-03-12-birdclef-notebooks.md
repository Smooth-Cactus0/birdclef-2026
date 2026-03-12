# BirdCLEF 2026 — Didactic Notebook Series Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build three Kaggle-ready notebooks — EDA, spectrogram guide, and a full EfficientNet-B0 baseline — targeting upvotes and generating outputs that guide the competition strategy.

**Architecture:** Three standalone `.py` files in `kaggle_notebooks/` using Kaggle script format (`# %%` cell markers). Each exports results to `/kaggle/working/` for download. All paths auto-detect Kaggle vs local execution.

**Tech Stack:** pandas, numpy, matplotlib, seaborn, librosa, plotly, timm, torch, sklearn, soundfile

---

## Key Data Facts (from local EDA)

| Fact | Value |
|---|---|
| Total train clips | 35,549 |
| Species | 234 |
| Aves clips | 34,799 (97.9%) |
| Amphibia clips | 451 |
| Insecta clips | 199 |
| Mammalia clips | 99 |
| Reptilia clips | 1 (single Caiman!) |
| XC collection | 23,043 clips |
| iNat collection | 12,506 clips |
| Train soundscapes | 10,658 OGG files |
| Labeled soundscape segments | 1,478 |
| Recorder region | Pantanal, lat -16.5→-21.6, lon -55.9→-57.6 |
| Audio format | OGG, target SR=32000 |
| Inference chunk | 5 seconds |
| Submission classes | 234 |

---

## Task 1: EDA Notebook (`kaggle_notebooks/01_eda_birdclef2026.py`)

**Files:**
- Create: `kaggle_notebooks/01_eda_birdclef2026.py`

**Step 1: Write the notebook header and setup cell**

```python
# %%
# =============================================================================
# BirdCLEF 2026 — Complete EDA 🦜🐸🐆
# =============================================================================
# Author: Alexy Louis
# If you find this helpful, an upvote is appreciated!
# =============================================================================

# %% [markdown]
# # BirdCLEF 2026 — Complete EDA 🦜🐸🐆
#
# This notebook explores the **BirdCLEF+ 2026** dataset — the most ecologically
# diverse acoustic monitoring competition yet. We're not just identifying birds:
# our models must recognise **234 species** across 5 biological classes recorded
# by ~1,000 passive acoustic monitors in the **Pantanal wetlands**, Brazil.
#
# **What we'll uncover:**
# 1. The multi-taxa surprise (Jaguar! Caiman! 25 insect sound-types!)
# 2. A severe geographic domain shift hiding in the training data
# 3. A new data modality: labeled soundscape segments (new in 2026)
# 4. Class imbalance so extreme it will break a naïve baseline

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
import librosa
import librosa.display
import soundfile as sf
import os, json, ast, re
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── Path detection ────────────────────────────────────────────────────────────
BASE_DIR = (Path('/kaggle/input/birdclef-2026')
            if Path('/kaggle/input/birdclef-2026').exists()
            else Path('birdclef-2026'))
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Base dir : {BASE_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})
CLASS_COLORS = {
    'Aves':     '#4C72B0',
    'Amphibia': '#55A868',
    'Insecta':  '#C44E52',
    'Mammalia': '#DD8452',
    'Reptilia': '#8172B3',
}
```

**Step 2: Load and print high-level stats**

```python
# %%
# ── Load CSVs ─────────────────────────────────────────────────────────────────
train     = pd.read_csv(BASE_DIR / 'train.csv')
taxonomy  = pd.read_csv(BASE_DIR / 'taxonomy.csv')
sl        = pd.read_csv(BASE_DIR / 'train_soundscapes_labels.csv')

print("=" * 50)
print(f"  Train clips          : {len(train):>7,}")
print(f"  Species (taxonomy)   : {len(taxonomy):>7,}")
print(f"  Labeled SL segments  : {len(sl):>7,}")
print(f"  Train soundscapes    : {len(list((BASE_DIR/'train_soundscapes').glob('*.ogg'))):>7,}")
print("=" * 50)
print(train.dtypes)
train.head(3)
```

**Step 3: Taxa breakdown — pie + bar**

```python
# %% [markdown]
# ## 1. Multi-Taxa Breakdown: This Is Not Just Birds

# %%
class_counts = train['class_name'].value_counts()
tax_counts   = taxonomy['class_name'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie: training clips by class
colors = [CLASS_COLORS[c] for c in class_counts.index]
wedges, texts, autotexts = axes[0].pie(
    class_counts.values,
    labels=class_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=140,
    pctdistance=0.8,
)
for at in autotexts:
    at.set_fontsize(9)
axes[0].set_title('Training Clips by Taxonomic Class', fontsize=13, fontweight='bold')

# Bar: species count per class
colors2 = [CLASS_COLORS[c] for c in tax_counts.index]
bars = axes[1].bar(tax_counts.index, tax_counts.values, color=colors2, edgecolor='white', linewidth=0.5)
for bar, v in zip(bars, tax_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].set_title('Number of Species per Class', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('# Species')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'taxa_breakdown.png', bbox_inches='tight')
plt.show()

print("\nClips per class:")
for cls, cnt in class_counts.items():
    pct = cnt / len(train) * 100
    n_sp = tax_counts.get(cls, 0)
    avg = cnt / n_sp if n_sp else 0
    print(f"  {cls:<12} {cnt:>6,} clips ({pct:5.1f}%)  |  {n_sp:>3} species  |  avg {avg:5.0f} clips/species")
```

**Step 4: Geographic distribution map**

```python
# %% [markdown]
# ## 2. The Geographic Domain Shift
#
# Training clips come from iNaturalist (crowd-sourced, all of South America)
# and Xeno-canto (global bird recordings). The test soundscapes come from a
# tight cluster of recorders in the **Pantanal wetland** — just 240 × 60 km.
# This domain shift is one of the biggest challenges in 2026.

# %%
# Drop rows without coordinates
geo = train.dropna(subset=['latitude', 'longitude']).copy()

fig, ax = plt.subplots(figsize=(12, 9))

# Plot each class
for cls, grp in geo.groupby('class_name'):
    ax.scatter(grp['longitude'], grp['latitude'],
               c=CLASS_COLORS[cls], alpha=0.25, s=6,
               label=f"{cls} ({len(grp):,})", rasterized=True)

# Pantanal recorder rectangle
rect = mpatches.FancyBboxPatch(
    (-57.6, -21.6), 57.6 - 55.9, 21.6 - 16.5,
    boxstyle="square,pad=0", linewidth=2.5,
    edgecolor='red', facecolor='#ff000022',
    zorder=5
)
ax.add_patch(rect)
ax.annotate('Pantanal\nRecorder\nRegion',
            xy=(-56.75, -19.0), fontsize=11, color='red',
            fontweight='bold', zorder=6)

# South America bounding box
ax.set_xlim(-82, -34)
ax.set_ylim(-38, 13)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Training Data Coverage vs Pantanal Recorder Region\n'
             '(Training spans all South America; test recorders are in the red box)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, markerscale=3)
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

# Equator label
ax.text(-80, 0.5, 'Equator', fontsize=9, color='gray', alpha=0.7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'geographic_distribution.png', bbox_inches='tight', dpi=150)
plt.show()

print(f"Training data lat range: {geo['latitude'].min():.1f} → {geo['latitude'].max():.1f}")
print(f"Training data lon range: {geo['longitude'].min():.1f} → {geo['longitude'].max():.1f}")
print(f"Pantanal recorder lat:   -21.6 → -16.5  (range: 5.1°)")
print(f"Pantanal recorder lon:   -57.6 → -55.9  (range: 1.7°)")
```

**Step 5: Class imbalance per species**

```python
# %% [markdown]
# ## 3. Class Imbalance: The Long Tail
#
# Even within birds there's a 50× difference between the most and least
# represented species. For non-bird taxa it's worse — the Caiman has a
# single training clip!

# %%
# Samples per species, coloured by class
spc = (train.groupby(['primary_label', 'class_name'])
           .size()
           .reset_index(name='n_clips')
           .sort_values('n_clips', ascending=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# -- Left: full distribution (all 234 species) --
colors_bar = [CLASS_COLORS[c] for c in spc['class_name']]
axes[0].bar(range(len(spc)), spc['n_clips'], color=colors_bar, width=1.0)
axes[0].set_xlabel('Species (sorted by clip count)', fontsize=12)
axes[0].set_ylabel('# Training Clips', fontsize=12)
axes[0].set_title('Class Imbalance Across All 234 Species', fontsize=13, fontweight='bold')
# Legend
patches = [mpatches.Patch(color=c, label=k) for k, c in CLASS_COLORS.items()]
axes[0].legend(handles=patches, fontsize=9)

# -- Right: bottom 50 species (non-birds visible here) --
bottom50 = spc.tail(50)
colors_b50 = [CLASS_COLORS[c] for c in bottom50['class_name']]
axes[1].barh(range(len(bottom50)), bottom50['n_clips'], color=colors_b50)
axes[1].set_yticks(range(len(bottom50)))
axes[1].set_yticklabels(bottom50['primary_label'], fontsize=7)
axes[1].set_xlabel('# Training Clips', fontsize=12)
axes[1].set_title('Bottom 50 Species (fewest clips)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'class_imbalance.png', bbox_inches='tight', dpi=150)
plt.show()

print("\nTop 5 most represented species:")
print(spc.head(5)[['primary_label', 'class_name', 'n_clips']].to_string(index=False))
print("\nBottom 5 — least represented:")
print(spc.tail(5)[['primary_label', 'class_name', 'n_clips']].to_string(index=False))
print(f"\nMax clips per species: {spc['n_clips'].max()}")
print(f"Min clips per species: {spc['n_clips'].min()}")
print(f"Median clips per species: {spc['n_clips'].median():.0f}")
```

**Step 6: Collection source and rating quality**

```python
# %% [markdown]
# ## 4. Data Sources and Quality

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Collection (iNat vs XC)
coll = train['collection'].value_counts()
axes[0].pie(coll.values, labels=coll.index, autopct='%1.1f%%',
            colors=['#4C72B0', '#55A868'], startangle=90)
axes[0].set_title('Collection Source', fontweight='bold')

# Rating distribution
train['rating_f'] = pd.to_numeric(train['rating'], errors='coerce')
# iNat records have rating=0.0 (unrated), XC records have 0.5–5.0
xc_rated = train[train['collection'] == 'XC']['rating_f']
axes[1].hist(xc_rated, bins=20, color='#4C72B0', edgecolor='white')
axes[1].set_xlabel('Quality Rating')
axes[1].set_ylabel('# Clips')
axes[1].set_title('Xeno-Canto Quality Rating\n(iNat clips are unrated = 0.0)', fontweight='bold')

# Per-class rating
class_rating = train.groupby('class_name')['rating_f'].mean().sort_values()
colors3 = [CLASS_COLORS[c] for c in class_rating.index]
bars = axes[2].barh(class_rating.index, class_rating.values, color=colors3)
axes[2].set_xlabel('Mean Rating')
axes[2].set_title('Average Quality Rating by Class', fontweight='bold')
for bar, v in zip(bars, class_rating.values):
    axes[2].text(v + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{v:.2f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'data_quality.png', bbox_inches='tight')
plt.show()
```

**Step 7: Labeled soundscapes analysis (new in 2026)**

```python
# %% [markdown]
# ## 5. Labeled Soundscapes — New in 2026! 📡
#
# For the first time, BirdCLEF provides **labeled segments from actual PAM
# recordings**. These 5-second chunks come from real Pantanal soundscapes,
# annotated with the species present. This is gold for training — it's the
# only data that exactly matches the test distribution.

# %%
# Parse multi-label column (semicolon-separated species codes)
sl['labels_list'] = sl['primary_label'].apply(lambda x: str(x).split(';'))
sl['n_species']   = sl['labels_list'].apply(len)

# Extract site and date from filename
sl['site'] = sl['filename'].apply(lambda x: re.search(r'_S(\d+)_', x).group(1)
                                  if re.search(r'_S(\d+)_', x) else 'unknown')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Species per segment
axes[0].hist(sl['n_species'], bins=range(1, sl['n_species'].max()+2),
             color='#4C72B0', edgecolor='white', align='left')
axes[0].set_xlabel('# Species Active in 5s Chunk')
axes[0].set_ylabel('# Segments')
axes[0].set_title('Multi-Label Density\nin Labeled Soundscapes', fontweight='bold')
axes[0].set_xticks(range(1, sl['n_species'].max()+1))

# Most frequent species in soundscapes
all_labels = [l for labels in sl['labels_list'] for l in labels]
label_freq  = Counter(all_labels).most_common(20)
labs, cnts  = zip(*label_freq)
# Merge with taxonomy to get class colour
lab_class   = taxonomy.set_index('primary_label')['class_name'].to_dict()
bar_colors  = [CLASS_COLORS.get(lab_class.get(l, 'Aves'), '#888888') for l in labs]
axes[1].barh(labs, cnts, color=bar_colors)
axes[1].set_xlabel('# Labeled Segments')
axes[1].set_title('Top 20 Species in\nLabeled Soundscapes', fontweight='bold')
axes[1].invert_yaxis()

# Sites
site_counts = sl['site'].value_counts()
axes[2].bar(site_counts.index[:15], site_counts.values[:15], color='#55A868')
axes[2].set_xlabel('Recorder Site ID')
axes[2].set_ylabel('# Labeled Segments')
axes[2].set_title('Labeled Segments per\nRecorder Site', fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'soundscape_labels.png', bbox_inches='tight')
plt.show()

print(f"Unique soundscape files labeled: {sl['filename'].nunique()}")
print(f"Unique recorder sites: {sl['site'].nunique()}")
print(f"Avg species per 5s segment: {sl['n_species'].mean():.2f}")
print(f"Max species in one segment: {sl['n_species'].max()}")
```

**Step 8: Insect sonotypes deep dive**

```python
# %% [markdown]
# ## 6. The Insect Sonotype Puzzle 🦗
#
# One entry in the taxonomy — iNaturalist ID **47158** — has been split into
# **25 acoustic morphotypes** (son01–son25). This means the organizers couldn't
# assign a species name to these insects but *could* identify 25 acoustically
# distinct sound patterns. You'll need to treat each sonotype as an independent
# class.

# %%
sonotypes = taxonomy[taxonomy['primary_label'].str.startswith('47158son')]
print(f"Number of insect sonotypes: {len(sonotypes)}")
print(sonotypes[['primary_label', 'common_name', 'class_name']].to_string(index=False))

# How many training clips per sonotype?
son_clips = train[train['primary_label'].str.startswith('47158son', na=False)]
son_dist  = son_clips['primary_label'].value_counts()
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(son_dist.index, son_dist.values, color='#C44E52', edgecolor='white')
ax.set_xlabel('Sonotype')
ax.set_ylabel('# Training Clips')
ax.set_title('Training Clips per Insect Sonotype (iNat 47158)\n'
             '"25 acoustically distinct patterns from one unknown insect genus"',
             fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'insect_sonotypes.png', bbox_inches='tight')
plt.show()
```

**Step 9: Audio properties — waveform + quick spectrogram peek**

```python
# %% [markdown]
# ## 7. Audio Properties

# %%
# Load one file per class to inspect duration / sample rate
EXAMPLES = {
    'Aves':     ('banana',  'XC1017270.ogg'),
    'Amphibia': ('22961',   None),
    'Insecta':  ('1161364', None),
    'Mammalia': ('41970',   None),
    'Reptilia': ('116570',  None),
}

durations, srs = {}, {}
fig, axes = plt.subplots(len(EXAMPLES), 1, figsize=(14, 10))

for ax, (cls, (folder, specific)) in zip(axes, EXAMPLES.items()):
    folder_path = BASE_DIR / 'train_audio' / folder
    if not folder_path.exists():
        ax.set_title(f"{cls} — folder not found")
        continue
    files = sorted(folder_path.glob('*.ogg'))
    if not files:
        ax.set_title(f"{cls} — no OGGs")
        continue
    if specific:
        fp = folder_path / specific
        if not fp.exists():
            fp = files[0]
    else:
        fp = files[0]

    y, sr = librosa.load(str(fp), sr=None, mono=True)
    durations[cls] = len(y) / sr
    srs[cls] = sr
    times = np.linspace(0, len(y)/sr, len(y))
    ax.plot(times, y, color=CLASS_COLORS[cls], linewidth=0.4, alpha=0.8)
    ax.set_title(f"{cls} — {fp.name}  ({sr} Hz, {len(y)/sr:.1f}s)",
                 fontweight='bold', fontsize=10)
    ax.set_ylabel('Amplitude')
    ax.set_xlim(0, len(y)/sr)

axes[-1].set_xlabel('Time (seconds)')
plt.suptitle('Waveforms by Taxonomic Class', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'waveforms.png', bbox_inches='tight')
plt.show()

print("\nDurations & sample rates:")
for cls in durations:
    print(f"  {cls:<12}: {durations[cls]:.1f}s at {srs[cls]} Hz")
```

**Step 10: Key takeaways cell**

```python
# %% [markdown]
# ## 8. Key Takeaways for Competitors 🎯
#
# | Observation | Competition Implication |
# |---|---|
# | 97.9% of training clips are birds | Non-bird species will have terrible recall with a naïve baseline |
# | Caiman has **1 training clip** | External data or pseudo-labeling is mandatory |
# | Training covers all South America; recorders cover 5°×1.7° | Domain adaptation / geographic filtering matters |
# | 10,658 soundscapes available, only 1,478 segments labeled | Pseudo-labeling the unlabeled soundscapes is a big opportunity |
# | 25 insect sonotypes = 25 independent classes | Multi-label per-sonotype prediction required |
# | Macro ROC-AUC = equal weight to all 234 classes | A rare Caiman clip matters as much as 1,000 bird clips |

# %%
# Export summary stats
stats = {
    'total_clips': int(len(train)),
    'n_species': int(len(taxonomy)),
    'clips_per_class': {k: int(v) for k, v in class_counts.items()},
    'species_per_class': {k: int(v) for k, v in tax_counts.items()},
    'collections': {k: int(v) for k, v in train['collection'].value_counts().items()},
    'n_train_soundscapes': len(list((BASE_DIR/'train_soundscapes').glob('*.ogg'))),
    'n_labeled_segments': int(len(sl)),
    'recorder_lat_range': [-21.6, -16.5],
    'recorder_lon_range': [-57.6, -55.9],
}
with open(OUTPUT_DIR / 'eda_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("Stats saved to eda_stats.json")
print(json.dumps(stats, indent=2))
```

**Step 11: Commit**

```bash
git add kaggle_notebooks/01_eda_birdclef2026.py
git commit -m "feat: EDA notebook — taxa breakdown, geographic shift, imbalance, sonotypes"
```

---

## Task 2: Spectrogram Guide (`kaggle_notebooks/02_spectrogram_guide.py`)

**Files:**
- Create: `kaggle_notebooks/02_spectrogram_guide.py`

**Step 1: Header + setup**

```python
# %%
# =============================================================================
# BirdCLEF 2026 — Mel Spectrogram Visual Guide 🎵
# =============================================================================

# %% [markdown]
# # Mel Spectrograms for Wildlife Audio 🎵
#
# Every top solution in BirdCLEF converts audio → **mel spectrogram image** and
# then runs a vision CNN. This notebook is your complete visual guide:
# from raw waveform to competition-ready spectrogram.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa, librosa.display
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = (Path('/kaggle/input/birdclef-2026')
              if Path('/kaggle/input/birdclef-2026').exists()
              else Path('birdclef-2026'))
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-dark')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})
```

**Step 2: Competition audio constants**

```python
# %%
# Competition-standard audio parameters
SR         = 32000
N_FFT      = 1024
HOP_LENGTH = 320    # ~10ms hop → good temporal resolution
N_MELS     = 128
FMIN       = 40     # Low: captures insect drones, frog bass calls
FMAX       = 15000  # High: captures bird high-frequency calls
DURATION   = 5      # seconds — inference chunk size
N_SAMPLES  = SR * DURATION

def load_clip(path, target_sr=SR, duration=DURATION):
    """Load audio, resample, pad/trim to fixed duration."""
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    if len(y) < target_sr * duration:
        y = np.pad(y, (0, target_sr * duration - len(y)), mode='constant')
    else:
        y = y[:target_sr * duration]
    return y

def make_melspec(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """Compute log-mel spectrogram, normalised to [0, 1]."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm
```

**Step 3: Step-by-step pipeline for one bird clip**

```python
# %% [markdown]
# ## Step 1: Raw Waveform → STFT → Mel Spectrogram

# %%
# Find a banana (Bananaquit) clip — common bird, good call
bird_folder = BASE_DIR / 'train_audio' / 'banana'
bird_file   = sorted(bird_folder.glob('*.ogg'))[0]
y = load_clip(bird_file)
t = np.linspace(0, DURATION, len(y))

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

# 1. Waveform
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, y, color='#4C72B0', linewidth=0.5, alpha=0.9)
ax1.set_title('① Raw Waveform', fontsize=13, fontweight='bold')
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude')
ax1.set_xlim(0, DURATION)

# 2. STFT magnitude
D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
ax2 = fig.add_subplot(gs[1, 0])
img2 = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    y_axis='linear', x_axis='time', sr=SR,
    hop_length=HOP_LENGTH, ax=ax2, cmap='magma')
ax2.set_title('② STFT Magnitude (linear frequency)', fontsize=12, fontweight='bold')
plt.colorbar(img2, ax=ax2, format='%+2.0f dB')

# 3. Mel filterbank applied
S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT,
    hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
ax3 = fig.add_subplot(gs[1, 1])
img3 = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
    y_axis='mel', x_axis='time', sr=SR,
    hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX, ax=ax3, cmap='magma')
ax3.set_title(f'③ Mel Spectrogram ({N_MELS} mel bins, {FMIN}–{FMAX} Hz)', fontsize=12, fontweight='bold')
plt.colorbar(img3, ax=ax3, format='%+2.0f dB')

# 4. Final: normalised input to CNN
S_norm = make_melspec(y)
ax4 = fig.add_subplot(gs[2, :])
im4 = ax4.imshow(S_norm, aspect='auto', origin='lower', cmap='magma',
                 extent=[0, DURATION, FMIN/1000, FMAX/1000])
ax4.set_title('④ Normalised Log-Mel (CNN input) — shape: '
              f'{S_norm.shape[0]} × {S_norm.shape[1]}', fontsize=13, fontweight='bold')
ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Frequency (kHz)')
plt.colorbar(im4, ax=ax4)

plt.suptitle('Audio → Mel Spectrogram Pipeline\n'
             f'(File: {bird_file.name})', fontsize=14, fontweight='bold', y=1.02)
plt.savefig(OUTPUT_DIR / 'pipeline_demo.png', bbox_inches='tight', dpi=150)
plt.show()
```

**Step 4: Multi-taxa spectrogram gallery**

```python
# %% [markdown]
# ## Cross-Taxa Spectrogram Gallery
#
# One of the most educational things we can do: compare spectrograms across
# the 5 biological classes. Birds, frogs, insects, and mammals each occupy
# distinct **frequency bands** and have characteristic **temporal patterns**.

# %%
TAXA_EXAMPLES = [
    ('Aves',     'banana',  'Bananaquit (bird)'),
    ('Amphibia', '22961',   'Pointedbelly Frog'),
    ('Insecta',  '1161364', 'Guyalna cuta (insect)'),
    ('Mammalia', '43435',   'Black Howler Monkey'),
    ('Reptilia', '116570',  'Caiman yacare'),
]

fig, axes = plt.subplots(len(TAXA_EXAMPLES), 1, figsize=(14, 14))
CLASS_CMAPS = {
    'Aves': 'Blues', 'Amphibia': 'Greens', 'Insecta': 'Reds',
    'Mammalia': 'Oranges', 'Reptilia': 'Purples'
}

for ax, (cls, folder, label) in zip(axes, TAXA_EXAMPLES):
    folder_path = BASE_DIR / 'train_audio' / folder
    if not folder_path.exists():
        ax.set_title(f'{label} — not found'); continue
    files = sorted(folder_path.glob('*.ogg'))
    if not files:
        ax.set_title(f'{label} — no files'); continue
    y = load_clip(files[0])
    S_norm = make_melspec(y)
    im = ax.imshow(S_norm, aspect='auto', origin='lower', cmap=CLASS_CMAPS[cls],
                   extent=[0, DURATION, FMIN/1000, FMAX/1000])
    ax.set_title(f'{label}  |  {files[0].name}', fontweight='bold', fontsize=11)
    ax.set_ylabel('Freq (kHz)')
    plt.colorbar(im, ax=ax)

axes[-1].set_xlabel('Time (s)')
plt.suptitle('Mel Spectrograms Across Taxonomic Classes\n'
             f'SR={SR}Hz, n_mels={N_MELS}, fmin={FMIN}Hz, fmax={FMAX}Hz',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'taxa_spectrogram_gallery.png', bbox_inches='tight', dpi=150)
plt.show()
```

**Step 5: Parameter sweep grid**

```python
# %% [markdown]
# ## How Parameters Affect the Spectrogram

# %%
y_ref = load_clip(sorted((BASE_DIR / 'train_audio' / 'banana').glob('*.ogg'))[0])

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
configs = [
    dict(n_mels=64,  fmin=40,  fmax=15000, title='n_mels=64'),
    dict(n_mels=128, fmin=40,  fmax=15000, title='n_mels=128 ✓ (standard)'),
    dict(n_mels=256, fmin=40,  fmax=15000, title='n_mels=256'),
    dict(n_mels=128, fmin=40,  fmax=8000,  title='fmax=8kHz'),
    dict(n_mels=128, fmin=40,  fmax=15000, title='fmax=15kHz ✓'),
    dict(n_mels=128, fmin=300, fmax=15000, title='fmin=300Hz (misses bass)'),
    dict(n_mels=128, fmin=40,  fmax=15000, title='hop=160 (2× temporal res)'),
    dict(n_mels=128, fmin=40,  fmax=15000, title='hop=640 (half temporal res)'),
    dict(n_mels=128, fmin=40,  fmax=15000, title='n_fft=2048 (better freq res)'),
]
hop_overrides = {6: 160, 7: 640}
nfft_overrides = {8: 2048}

for idx, (ax, cfg) in enumerate(zip(axes.flat, configs)):
    hop = hop_overrides.get(idx, HOP_LENGTH)
    nfft = nfft_overrides.get(idx, N_FFT)
    S = librosa.feature.melspectrogram(y=y_ref, sr=SR, n_fft=nfft, hop_length=hop,
        n_mels=cfg['n_mels'], fmin=cfg['fmin'], fmax=cfg['fmax'])
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    ax.imshow(S_norm, aspect='auto', origin='lower', cmap='magma')
    title_color = 'green' if '✓' in cfg['title'] else 'black'
    ax.set_title(cfg['title'], fontsize=9, fontweight='bold', color=title_color)
    ax.axis('off')

plt.suptitle('Spectrogram Parameter Sweep  (green = competition standard)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'parameter_sweep.png', bbox_inches='tight', dpi=150)
plt.show()
```

**Step 6: Augmentation demos**

```python
# %% [markdown]
# ## Data Augmentation Techniques
#
# The top 2025 solutions used: MixUp, SpecAugment (time + frequency masking).

# %%
y_base = load_clip(sorted((BASE_DIR / 'train_audio' / 'banana').glob('*.ogg'))[0])
S_base = make_melspec(y_base)

def time_mask(S, T=30):
    S2 = S.copy(); t0 = np.random.randint(0, S.shape[1]-T); S2[:, t0:t0+T] = 0; return S2
def freq_mask(S, F=20):
    S2 = S.copy(); f0 = np.random.randint(0, S.shape[0]-F); S2[f0:f0+F, :] = 0; return S2

y2, _ = librosa.load(str(sorted((BASE_DIR / 'train_audio' / '22961').glob('*.ogg'))[0]),
                     sr=SR, mono=True)
if len(y2) < N_SAMPLES: y2 = np.pad(y2, (0, N_SAMPLES - len(y2)))
else: y2 = y2[:N_SAMPLES]
S2 = make_melspec(y2)
lam = 0.5
S_mix = lam * S_base + (1 - lam) * S2

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
specs = [
    (S_base,                     'Original'),
    (time_mask(S_base),          'Time Masking (SpecAugment)'),
    (freq_mask(S_base),          'Frequency Masking (SpecAugment)'),
    (time_mask(freq_mask(S_base)),'Both Masks'),
    (S_mix,                      'MixUp (λ=0.5, bird + frog)'),
    (np.roll(S_base, 100, axis=1),'Time Shift'),
]
for ax, (S, title) in zip(axes.flat, specs):
    ax.imshow(S, aspect='auto', origin='lower', cmap='magma')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.axis('off')

plt.suptitle('Augmentation Techniques', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'augmentation_demo.png', bbox_inches='tight', dpi=150)
plt.show()
```

**Step 7: Commit**

```bash
git add kaggle_notebooks/02_spectrogram_guide.py
git commit -m "feat: spectrogram guide — pipeline, taxa gallery, param sweep, augmentation"
```

---

## Task 3: Baseline Notebook Part A — Data + Model (`kaggle_notebooks/03_baseline_efficientnet.py`)

**Files:**
- Create: `kaggle_notebooks/03_baseline_efficientnet.py`

**Step 1: Header + config**

```python
# %%
# =============================================================================
# BirdCLEF 2026 — EfficientNet-B0 Baseline 🚀
# =============================================================================
# Architecture : EfficientNet-B0 (timm)
# Loss         : CrossEntropy on primary_label
# Metric       : Macro ROC-AUC
# Folds        : 5-fold StratifiedKFold
# Inference    : sliding 5s windows on soundscapes
# Export       : ONNX for CPU deployment
# =============================================================================

# %% [markdown]
# # EfficientNet-B0 Baseline 🚀
# A complete training + inference pipeline. Every step is commented.

# %%
# ── Install / version pins ────────────────────────────────────────────────────
# !pip install -q timm==1.0.3  # uncomment on Kaggle if needed

import os, gc, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = (Path('/kaggle/input/birdclef-2026')
              if Path('/kaggle/input/birdclef-2026').exists()
              else Path('birdclef-2026'))
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    # Audio
    SR          = 32000,
    N_FFT       = 1024,
    HOP_LENGTH  = 320,
    N_MELS      = 128,
    FMIN        = 40,
    FMAX        = 15000,
    DURATION    = 5,         # seconds per clip during training
    # Model
    MODEL_NAME  = 'efficientnet_b0',
    PRETRAINED  = True,
    # Training
    N_FOLDS     = 5,
    EPOCHS      = 10,
    BATCH_SIZE  = 32,
    LR          = 1e-3,
    WEIGHT_DECAY= 1e-4,
    MIN_RATING  = 0.0,       # 0.0 = include all iNat clips
    # Device
    DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu',
    SEED        = 42,
)
print(f"Device: {CFG['DEVICE']}")
print(f"torch: {torch.__version__}, timm: {timm.__version__}")
```

**Step 2: Load data and encode labels**

```python
# %%
# ── Load metadata ─────────────────────────────────────────────────────────────
train_df  = pd.read_csv(BASE_DIR / 'train.csv')
taxonomy  = pd.read_csv(BASE_DIR / 'taxonomy.csv')

# Label encode: map primary_label → integer class index
# Use taxonomy order so indices are deterministic
label_list = taxonomy['primary_label'].tolist()
label2idx  = {l: i for i, l in enumerate(label_list)}
idx2label  = {i: l for l, i in label2idx.items()}
NUM_CLASSES = len(label_list)
print(f"Number of classes: {NUM_CLASSES}")

# Filter and prepare
train_df = train_df[train_df['primary_label'].isin(label2idx)].copy()
train_df['target'] = train_df['primary_label'].map(label2idx)
train_df['filepath'] = train_df['filename'].apply(lambda f: str(BASE_DIR / 'train_audio' / f))

# Optional: filter low-quality XC recordings (keep iNat + XC rating >= threshold)
# train_df = train_df[(train_df['rating'] == 0.0) | (train_df['rating'] >= CFG['MIN_RATING'])]
print(f"Training clips: {len(train_df):,}")
print(train_df['class_name'].value_counts())
```

**Step 3: Dataset class**

```python
# %%
class BirdDataset(Dataset):
    """
    Loads OGG audio, converts to a log-mel spectrogram,
    returns a (1, N_MELS, T) tensor and integer label.
    """
    def __init__(self, df, cfg, augment=False):
        self.df      = df.reset_index(drop=True)
        self.cfg     = cfg
        self.augment = augment
        self.n_samples = cfg['SR'] * cfg['DURATION']

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path):
        try:
            y, sr = librosa.load(path, sr=self.cfg['SR'], mono=True)
        except Exception:
            return np.zeros(self.n_samples, dtype=np.float32)
        # Pad or trim to fixed length
        if len(y) < self.n_samples:
            y = np.pad(y, (0, self.n_samples - len(y)), mode='constant')
        else:
            # Random crop during training; centre crop during validation
            if self.augment and len(y) > self.n_samples:
                start = np.random.randint(0, len(y) - self.n_samples)
                y = y[start:start + self.n_samples]
            else:
                y = y[:self.n_samples]
        return y.astype(np.float32)

    def _to_melspec(self, y):
        S = librosa.feature.melspectrogram(
            y=y, sr=self.cfg['SR'],
            n_fft=self.cfg['N_FFT'], hop_length=self.cfg['HOP_LENGTH'],
            n_mels=self.cfg['N_MELS'], fmin=self.cfg['FMIN'], fmax=self.cfg['FMAX'])
        S_db = librosa.power_to_db(S, ref=np.max)
        # Normalise to [0, 1]
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        return S_db.astype(np.float32)

    def _augment(self, S):
        """SpecAugment: random time + frequency masking."""
        S = S.copy()
        # Frequency masking
        F = np.random.randint(0, 20)
        f0 = np.random.randint(0, self.cfg['N_MELS'] - F)
        S[f0:f0 + F, :] = 0.0
        # Time masking
        T = np.random.randint(0, 30)
        t0 = np.random.randint(0, S.shape[1] - T)
        S[:, t0:t0 + T] = 0.0
        return S

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y   = self._load_audio(row['filepath'])
        S   = self._to_melspec(y)
        if self.augment:
            S = self._augment(S)
        # Shape: (1, N_MELS, T) — single channel image
        S_tensor = torch.from_numpy(S).unsqueeze(0)
        label    = int(row['target'])
        return S_tensor, label
```

**Step 4: Model definition**

```python
# %%
class BirdModel(nn.Module):
    """
    EfficientNet-B0 pretrained on ImageNet, adapted for mel spectrograms.
    The first conv layer accepts 1-channel input (converted by averaging).
    """
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,           # single-channel mel spectrogram
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(x)  # returns logits, shape (B, num_classes)
```

**Step 5: Training utilities**

```python
# %%
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.cpu().numpy())
    all_probs   = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # One-hot encode for macro AUC
    y_onehot = np.zeros_like(all_probs)
    y_onehot[np.arange(len(all_targets)), all_targets] = 1
    try:
        auc = roc_auc_score(y_onehot, all_probs, average='macro')
    except ValueError:
        auc = 0.0
    return total_loss / len(loader.dataset), auc
```

---

## Task 4: Baseline Notebook Part B — Training + Submission

(Continuation of the same file `kaggle_notebooks/03_baseline_efficientnet.py`)

**Step 1: Cross-validation training loop**

```python
# %%
# ── Cross-validation ──────────────────────────────────────────────────────────
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])

skf      = StratifiedKFold(n_splits=CFG['N_FOLDS'], shuffle=True,
                           random_state=CFG['SEED'])
oof_preds  = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)
fold_aucs  = []
history    = []

for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df['target']), start=1):
    print(f"\n{'='*60}\n  Fold {fold}/{CFG['N_FOLDS']}\n{'='*60}")
    t0 = time.time()

    train_fold = train_df.iloc[train_idx]
    val_fold   = train_df.iloc[val_idx]

    train_ds = BirdDataset(train_fold, CFG, augment=True)
    val_ds   = BirdDataset(val_fold,   CFG, augment=False)
    train_dl = DataLoader(train_ds, batch_size=CFG['BATCH_SIZE'],
                          shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG['BATCH_SIZE'],
                          shuffle=False, num_workers=4, pin_memory=True)

    model     = BirdModel(CFG['MODEL_NAME'], NUM_CLASSES, CFG['PRETRAINED'])
    model     = model.to(CFG['DEVICE'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG['LR'], weight_decay=CFG['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG['LR'],
        steps_per_epoch=len(train_dl), epochs=CFG['EPOCHS'])

    best_auc   = 0.0
    fold_hist  = []

    for epoch in range(1, CFG['EPOCHS'] + 1):
        tr_loss = train_one_epoch(model, train_dl, optimizer, scheduler,
                                  criterion, CFG['DEVICE'])
        vl_loss, vl_auc = validate(model, val_dl, criterion,
                                   CFG['DEVICE'], NUM_CLASSES)
        fold_hist.append({'fold': fold, 'epoch': epoch,
                          'tr_loss': tr_loss, 'vl_loss': vl_loss, 'vl_auc': vl_auc})
        elapsed = time.time() - t0
        print(f"  Ep {epoch:02d}/{CFG['EPOCHS']}  "
              f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
              f"vl_auc={vl_auc:.4f}  [{elapsed:.0f}s]")

        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(),
                       OUTPUT_DIR / f'model_fold{fold}.pth')
            print(f"    ✓ saved (auc={best_auc:.4f})")

    # OOF predictions from best checkpoint
    model.load_state_dict(torch.load(OUTPUT_DIR / f'model_fold{fold}.pth',
                                      map_location=CFG['DEVICE']))
    model.eval()
    with torch.no_grad():
        for X, y in val_dl:
            X   = X.to(CFG['DEVICE'])
            idx = val_idx[: len(oof_preds) - len(val_idx)]  # align
            probs = torch.softmax(model(X), dim=1).cpu().numpy()
            # fill OOF
    # Re-run val to get OOF probs properly
    all_probs_list = []
    with torch.no_grad():
        for X, _ in val_dl:
            all_probs_list.append(
                torch.softmax(model(X.to(CFG['DEVICE'])), dim=1).cpu().numpy())
    oof_preds[val_idx] = np.concatenate(all_probs_list, axis=0)

    fold_aucs.append(best_auc)
    history.extend(fold_hist)
    del model, train_ds, val_ds, train_dl, val_dl
    torch.cuda.empty_cache(); gc.collect()

print(f"\n{'='*60}")
print(f"  CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"  Per-fold: {[round(a,4) for a in fold_aucs]}")
print(f"{'='*60}")
```

**Step 2: Training curve visualisation**

```python
# %%
hist_df = pd.DataFrame(history)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for fold in hist_df['fold'].unique():
    fh = hist_df[hist_df['fold'] == fold]
    axes[0].plot(fh['epoch'], fh['vl_loss'], marker='o', label=f'fold {fold}', markersize=4)
    axes[1].plot(fh['epoch'], fh['vl_auc'],  marker='o', label=f'fold {fold}', markersize=4)

axes[0].set_title('Validation Loss per Fold', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('CrossEntropy Loss')
axes[0].legend()
axes[1].set_title('Validation Macro ROC-AUC per Fold', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC')
axes[1].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves.png', bbox_inches='tight')
plt.show()
```

**Step 3: Inference on test soundscapes**

```python
# %%
# ── Inference ─────────────────────────────────────────────────────────────────
# Test soundscapes are long OGG files (minutes). We slide a 5s window
# and predict all species for each window. row_id = {filename}_{end_sec}

sample_sub = pd.read_csv(BASE_DIR / 'sample_submission.csv')
test_soundscapes = sorted((BASE_DIR / 'test_soundscapes').glob('*.ogg'))
print(f"Test soundscapes found: {len(test_soundscapes)}")

# Load best model (fold 1 by default; in practice ensemble all folds)
inf_model = BirdModel(CFG['MODEL_NAME'], NUM_CLASSES, pretrained=False)
inf_model.load_state_dict(torch.load(OUTPUT_DIR / 'model_fold1.pth',
                                      map_location=CFG['DEVICE']))
inf_model = inf_model.to(CFG['DEVICE'])
inf_model.eval()

def predict_soundscape(audio_path, model, cfg, device):
    """Slide 5s windows over a long soundscape, return dict of row_id → probs."""
    try:
        y, sr = librosa.load(str(audio_path), sr=cfg['SR'], mono=True)
    except Exception as e:
        print(f"  Error loading {audio_path.name}: {e}")
        return {}
    stem       = audio_path.stem
    n_samples  = cfg['SR'] * cfg['DURATION']
    n_chunks   = max(1, len(y) // n_samples)
    results    = {}
    chunks     = []
    row_ids    = []
    for i in range(n_chunks):
        start  = i * n_samples
        chunk  = y[start:start + n_samples]
        if len(chunk) < n_samples:
            chunk = np.pad(chunk, (0, n_samples - len(chunk)))
        end_sec = (i + 1) * cfg['DURATION']
        S = librosa.feature.melspectrogram(
            y=chunk.astype(np.float32),
            sr=cfg['SR'], n_fft=cfg['N_FFT'],
            hop_length=cfg['HOP_LENGTH'],
            n_mels=cfg['N_MELS'], fmin=cfg['FMIN'], fmax=cfg['FMAX'])
        S_db   = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        chunks.append(torch.from_numpy(S_norm).float().unsqueeze(0))
        row_ids.append(f"{stem}_{end_sec}")
    if not chunks:
        return {}
    batch  = torch.stack(chunks).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).cpu().numpy()   # sigmoid for multi-label style
    for rid, p in zip(row_ids, probs):
        results[rid] = p
    return results

all_preds = {}
for sf_path in test_soundscapes:
    preds = predict_soundscape(sf_path, inf_model, CFG, CFG['DEVICE'])
    all_preds.update(preds)
    print(f"  {sf_path.name}: {len(preds)} chunks")

print(f"Total prediction rows: {len(all_preds)}")
```

**Step 4: Build and save submission**

```python
# %%
# Align with sample_submission
sub = sample_sub.copy()
for row_id in sub['row_id']:
    if row_id in all_preds:
        sub.loc[sub['row_id'] == row_id, label_list] = all_preds[row_id]
    # else leave as sample_submission default (uniform)

sub.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
print(f"Submission saved: {sub.shape}")
print(sub.head(2))
```

**Step 5: ONNX export**

```python
# %%
# ── ONNX export ───────────────────────────────────────────────────────────────
# Required for OpenVINO conversion and CPU inference budget management.
try:
    dummy = torch.randn(1, 1, CFG['N_MELS'], 157).to(CFG['DEVICE'])  # 5s→157 frames
    torch.onnx.export(
        inf_model, dummy,
        str(OUTPUT_DIR / 'model_efficientnet_b0.onnx'),
        input_names=['input'], output_names=['output'],
        opset_version=11,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    print("ONNX export successful →", OUTPUT_DIR / 'model_efficientnet_b0.onnx')
except Exception as e:
    print(f"ONNX export failed: {e}")
```

**Step 6: Save OOF scores and commit**

```python
# %%
# Save experiment results
oof_results = {
    'model':     CFG['MODEL_NAME'],
    'n_folds':   CFG['N_FOLDS'],
    'epochs':    CFG['EPOCHS'],
    'fold_aucs': [round(a, 4) for a in fold_aucs],
    'mean_auc':  round(float(np.mean(fold_aucs)), 4),
    'std_auc':   round(float(np.std(fold_aucs)),  4),
}
with open(OUTPUT_DIR / 'oof_scores.json', 'w') as f:
    json.dump(oof_results, f, indent=2)
print("OOF results:", json.dumps(oof_results, indent=2))
```

```bash
git add kaggle_notebooks/03_baseline_efficientnet.py
git commit -m "feat: EfficientNet-B0 baseline — full CV training, soundscape inference, ONNX export"
```

---

## Task 5: Update CLAUDE.md + results log

**Files:**
- Modify: `CLAUDE.md` (Data Notes section)
- Create: `results/experiment_log.md`

**Step 1: Add accurate data statistics to CLAUDE.md Data Notes section**

Update the Data Notes section with exact figures from the data:
```markdown
## Data Notes (verified from local files — 2026-03-12)

| Item | Value |
|---|---|
| Train clips | 35,549 |
| Species | 234 |
| Aves clips | 34,799 (97.9%) — 162 species |
| Amphibia clips | 451 — 29 species |
| Insecta clips | 199 — 3 species + 25 sonotypes of iNat 47158 |
| Mammalia clips | 99 — 7 species |
| Reptilia clips | 1 — Caiman yacare only |
| XC collection | 23,043 clips |
| iNat collection | 12,506 clips |
| Train soundscapes | 10,658 OGG files |
| Labeled soundscape segments | 1,478 (in train_soundscapes_labels.csv) |
| Recorder lat range | -21.6 → -16.5 |
| Recorder lon range | -57.6 → -55.9 |
| Audio files | OGG (train_audio/), e.g. XC115713.ogg, iNat1657948.ogg |
| Train soundscape folder | train_soundscapes/ |
| Test soundscape folder | test_soundscapes/ (hidden on Kaggle) |
```

**Step 2: Create experiment log**

```markdown
# Experiment Log

| ID | Date | Model | Backbone | PL round | CV AUC | LB AUC | Notes |
|---|---|---|---|---|---|---|---|
| exp001 | 2026-03-12 | baseline | efficientnet_b0 | 0 | TBD | TBD | 10ep, CE, no quality filter |
```

**Step 3: Final commit**

```bash
git add CLAUDE.md results/experiment_log.md
git commit -m "docs: update CLAUDE.md with verified data stats, add experiment log"
```

---

## Execution Order

1. Task 1 → Task 2 → Task 3+4 (can write Task 3+4 together) → Task 5
2. Upload in order: EDA first (early upvotes), then spectrogram guide, then baseline
3. After baseline LB score is known, update `results/experiment_log.md`
