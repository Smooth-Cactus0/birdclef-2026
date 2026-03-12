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
import librosa
import librosa.display
import os, json, re
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
