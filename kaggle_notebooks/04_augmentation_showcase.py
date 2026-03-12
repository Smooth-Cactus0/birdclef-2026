# %%
# =============================================================================
# BirdCLEF 2026 — Audio Augmentation Guide 🎵
# =============================================================================
# Author: Alexy Louis
# If you find this helpful, an upvote is appreciated!
#
# What this notebook covers:
#   • Why augmentation is not optional for BirdCLEF 2026
#   • 7 augmentation techniques visualised on real spectrograms:
#       SpecAugment (freq + time masking), Time shift, Gaussian noise,
#       MixUp, Background mix, Pitch shift
#   • A complete augmentation pipeline showing all transforms applied together
#   • A summary table with recommended probabilities for each technique
#
# No training happens here — this is a purely didactic visual guide.
# =============================================================================

# %% [markdown]
# # Audio Augmentation Guide for BirdCLEF 2026 🎵
#
# **Why augmentation matters, what each technique does, and how to implement it**
#
# Every top BirdCLEF solution relies heavily on augmentation — not as a
# minor regularisation trick, but as a core part of closing the gap between
# studio-quality Xeno-canto clips and raw Pantanal field recordings.
#
# **What you'll learn:**
# 1. The domain gap problem (XC recordings vs. PAM soundscapes)
# 2. Seven augmentation techniques with side-by-side spectrogram comparisons
# 3. MixUp with mixed-label visualisation
# 4. Background mixing — the highest-ROI augmentation for this competition
# 5. How to combine everything into a single training pipeline

# %%
# !pip install -q librosa==0.10.1  # uncomment if needed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.table as mtable
import librosa
import librosa.display
import torch
import torch.nn.functional as F
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths (auto-detect Kaggle vs local) ──────────────────────────────────────
BASE_DIR   = (Path('/kaggle/input/birdclef-2026')
              if Path('/kaggle/input/birdclef-2026').exists()
              else Path('birdclef-2026'))
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-dark')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})

# ── Competition-standard audio config ────────────────────────────────────────
CFG = dict(
    sr         = 32000,   # Sample rate
    n_fft      = 1024,    # FFT window (~32 ms at 32 kHz)
    hop_length = 320,     # Hop size (~10 ms) → good temporal resolution
    n_mels     = 128,     # Mel filterbank bins (spectrogram height)
    fmin       = 40,      # Lowest frequency (keeps frog bass, insect drones)
    fmax       = 15000,   # Highest frequency (captures high bird calls)
    duration   = 5,       # Seconds — competition inference chunk size
)
CFG['n_samples'] = CFG['sr'] * CFG['duration']

print(f"BASE_DIR  : {BASE_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"librosa   : {librosa.__version__}")
print(f"\nAudio config: SR={CFG['sr']} | N_MELS={CFG['n_mels']} | "
      f"FMIN={CFG['fmin']} | FMAX={CFG['fmax']} | DURATION={CFG['duration']}s")

# ── Class colours (consistent with other notebooks) ──────────────────────────
CLASS_COLORS = {
    'Aves':     '#4C72B0',
    'Amphibia': '#55A868',
    'Insecta':  '#C44E52',
    'Mammalia': '#DD8452',
    'Reptilia': '#8172B3',
}

# %% [markdown]
# ## Helper Functions: Load a Clip and Build a Mel Spectrogram
#
# Every augmentation function below starts from either a raw waveform `y`
# (shape `(n_samples,)`) or a mel spectrogram `S_db` (shape `(N_MELS, T)`).
# These two helpers are the entry-point for the whole notebook.

# %%
def load_clip(path, cfg=CFG, offset=0.0):
    """Load audio from path, resample to cfg['sr'], pad/trim to fixed duration."""
    y, sr = librosa.load(str(path), sr=cfg['sr'], mono=True, offset=offset,
                         duration=cfg['duration'])
    n = cfg['n_samples']
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)), mode='constant')
    else:
        y = y[:n]
    return y


def make_melspec(y, cfg=CFG):
    """Compute log-mel spectrogram normalised to [0, 1].

    Returns S_norm (N_MELS, T) — the array that a CNN would see.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=cfg['sr'], n_fft=cfg['n_fft'], hop_length=cfg['hop_length'],
        n_mels=cfg['n_mels'], fmin=cfg['fmin'], fmax=cfg['fmax'])
    S_db   = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm


def load_mel(path, cfg=CFG, offset=0.0):
    """Convenience wrapper: load path → return (y, S_norm)."""
    y = load_clip(path, cfg=cfg, offset=offset)
    return y, make_melspec(y, cfg=cfg)


def show_spec(ax, S, title, cfg=CFG, cmap='magma', vmin=0, vmax=1):
    """Render a normalised mel spectrogram on an existing axes object."""
    librosa.display.specshow(
        S, sr=cfg['sr'], hop_length=cfg['hop_length'],
        x_axis='time', y_axis='mel',
        fmin=cfg['fmin'], fmax=cfg['fmax'],
        ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Frequency (Hz)', fontsize=9)

# %% [markdown]
# ## Why Augment? Three Key Reasons for BirdCLEF 2026
#
# **1. Domain gap: studio vs. field recordings**
# Training clips come from Xeno-canto — recordings made by dedicated birders
# with clean microphones in near-silence. Test soundscapes are continuous PAM
# recordings from ~1,000 passive recorders sitting in the Pantanal wetlands,
# 24/7, capturing heavy background noise, wind, insects, and rain. Without
# augmentation, models overfit to XC recording conditions and collapse on
# real field data.
#
# **2. Data scarcity: some species have fewer than 10 clips**
# With 234 species and severe long-tail distribution, augmentation is the
# primary regularisation mechanism for rare taxa. Without it, models memorise
# the few rare-species clips and cannot generalise.
#
# **3. Robustness to partial calls and overlap**
# In 5-second soundscape chunks, a species call may be partially cut off,
# overlapping with another species, or buried under rain. Augmentations like
# time masking, time shift, and MixUp train the model to handle all of these.

# %%
# ── Load example clips — one per biological class ────────────────────────────
# We try to find real audio; if unavailable we synthesise plausible stand-ins.

def synthesize_clip(cfg=CFG, fundamental=1200, n_harmonics=5, label='synthetic'):
    """Generate a synthetic bird-like harmonic tone as a stand-in clip."""
    t = np.linspace(0, cfg['duration'], cfg['n_samples'])
    y = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        y += (1.0 / k) * np.sin(2 * np.pi * fundamental * k * t)
    # Add light frequency modulation (like a bird trill)
    mod = 0.5 * np.sin(2 * np.pi * 8 * t)   # 8 Hz modulation
    y *= (1 + 0.3 * mod)
    # Envelope: fade in/out
    env = np.ones_like(t)
    fade = int(0.05 * len(t))
    env[:fade]  = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    y *= env
    # Normalise + add small background noise
    y /= (np.max(np.abs(y)) + 1e-8)
    y += 0.02 * np.random.randn(len(y))
    return y.astype(np.float32)


def synthesize_insect_clip(cfg=CFG):
    """Generate a synthetic insect-like continuous buzz."""
    t = np.linspace(0, cfg['duration'], cfg['n_samples'])
    y = np.sin(2 * np.pi * 3500 * t)                              # carrier
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 200 * t)                 # 200 Hz AM
    y *= am
    y += 0.05 * np.random.randn(len(y))
    y /= (np.max(np.abs(y)) + 1e-8)
    return y.astype(np.float32)


def synthesize_frog_clip(cfg=CFG):
    """Generate a synthetic frog-like pulsed call."""
    t = np.linspace(0, cfg['duration'], cfg['n_samples'])
    y = np.sin(2 * np.pi * 700 * t) + 0.5 * np.sin(2 * np.pi * 1400 * t)
    # Pulsed: ~3 pulses per second
    pulse_env = np.abs(np.sin(2 * np.pi * 3 * t)) ** 6
    y *= pulse_env
    y += 0.03 * np.random.randn(len(y))
    y /= (np.max(np.abs(y)) + 1e-8)
    return y.astype(np.float32)


def synthesize_mammal_clip(cfg=CFG):
    """Generate a synthetic low-frequency mammal-like call."""
    t = np.linspace(0, cfg['duration'], cfg['n_samples'])
    y = np.sin(2 * np.pi * 280 * t) + 0.6 * np.sin(2 * np.pi * 560 * t)
    env = np.exp(-0.5 * t)   # decaying envelope
    y *= env
    y += 0.04 * np.random.randn(len(y))
    y /= (np.max(np.abs(y)) + 1e-8)
    return y.astype(np.float32)


try:
    train_df   = pd.read_csv(BASE_DIR / 'train.csv')
    taxonomy   = pd.read_csv(BASE_DIR / 'taxonomy.csv')
    print(f"Loaded train.csv: {len(train_df):,} rows")

    # Merge class_name from taxonomy if not already in train_df
    class_col = 'class_name' if 'class_name' in train_df.columns else None
    if class_col is None and 'species_code' in train_df.columns:
        merge_col = 'species_code' if 'species_code' in taxonomy.columns else taxonomy.columns[0]
        train_df = train_df.merge(
            taxonomy[['species_code', 'class_name']].drop_duplicates(),
            on='species_code', how='left')
        class_col = 'class_name'

    # Pick one example per class
    target_classes = ['Aves', 'Amphibia', 'Insecta', 'Mammalia']
    examples = {}
    for cls in target_classes:
        subset = train_df[train_df[class_col] == cls] if class_col else pd.DataFrame()
        if len(subset) == 0:
            continue
        row = subset.sample(1, random_state=42).iloc[0]
        # Infer audio path
        fname = row.get('filename', None)
        if fname:
            audio_path = BASE_DIR / 'train_audio' / fname
            if audio_path.exists():
                examples[cls] = audio_path
    DATA_AVAILABLE = len(examples) > 0
    print(f"Audio examples found: {list(examples.keys())}")

except FileNotFoundError:
    DATA_AVAILABLE = False
    examples = {}
    print("train.csv not found — will use synthetic audio throughout.")

# Build waveforms + spectrograms for each class
np.random.seed(42)
clips = {}
if DATA_AVAILABLE:
    for cls, path in examples.items():
        try:
            y, S = load_mel(path)
            clips[cls] = {'y': y, 'S': S, 'source': 'real', 'label': cls}
        except Exception as e:
            print(f"Could not load {cls}: {e}")

# Fill in any missing classes with synthetic audio
synth_funcs = {
    'Aves':     lambda: synthesize_clip(fundamental=1500, n_harmonics=6),
    'Amphibia': synthesize_frog_clip,
    'Insecta':  synthesize_insect_clip,
    'Mammalia': synthesize_mammal_clip,
}
for cls in ['Aves', 'Amphibia', 'Insecta', 'Mammalia']:
    if cls not in clips:
        y = synth_funcs[cls]()
        clips[cls] = {'y': y, 'S': make_melspec(y), 'source': 'synthetic', 'label': cls}

print(f"\nClips ready:")
for cls, d in clips.items():
    print(f"  {cls:<12} [{d['source']}]  waveform={d['y'].shape}  spec={d['S'].shape}")

# %% [markdown]
# ## Original Spectrograms — One per Biological Class
#
# Before any augmentation, let's see what each class looks like.
# Notice the distinctive spectral signatures:
# - **Aves**: narrow harmonic stacks or complex frequency-modulated sweeps
# - **Amphibia**: pulsed calls, often lower frequency with clear overtones
# - **Insecta**: broad-band continuous noise or rapid amplitude modulation
# - **Mammalia**: low-frequency fundamental with sparse harmonics

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
axes = axes.flatten()

for idx, (cls, d) in enumerate(clips.items()):
    ax = axes[idx]
    show_spec(ax, d['S'], title=f"{cls}  [{d['source']}]",
              cmap='magma')
    ax.tick_params(labelsize=8)
    # Colour-code the title
    color = CLASS_COLORS.get(cls, 'white')
    ax.set_title(f"{cls}  [{d['source']}]", fontsize=12, fontweight='bold',
                 color=color, pad=6)

# Hide any unused axes
for idx in range(len(clips), 4):
    axes[idx].set_visible(False)

plt.suptitle('Original Mel Spectrograms — One per Biological Class',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'aug_00_originals.png', bbox_inches='tight', dpi=120)
plt.show()
print("Saved: aug_00_originals.png")

# %% [markdown]
# ## Augmentation 1: SpecAugment (Frequency + Time Masking)
#
# **What it does:** Randomly zeros out horizontal bands (frequency masking)
# or vertical bands (time masking) directly on the spectrogram.
#
# **Why it helps:** Forces the model to learn from incomplete frequency bands
# or time regions — mimicking real-world scenarios where:
# - Wind noise drowns out a frequency band
# - Another species' call overlaps a time window
# - The vocalization is partially cut by the 5-second chunk boundary
#
# **Reference:** Park et al. 2019 "SpecAugment: A Simple Data Augmentation
# Method for Automatic Speech Recognition" — originally for ASR, now standard
# in all competitive audio classification.

# %%
def spec_augment(S, freq_mask_param=20, time_mask_param=40, num_freq_masks=1,
                 num_time_masks=1, seed=None):
    """Apply SpecAugment: random frequency and time masking.

    Args:
        S: mel spectrogram (N_MELS, T), values in [0, 1]
        freq_mask_param: max width of frequency mask in mel bins
        time_mask_param: max width of time mask in frames
        num_freq_masks: how many frequency masks to apply
        num_time_masks: how many time masks to apply
        seed: optional random seed for reproducibility
    Returns:
        S_aug: augmented spectrogram, same shape as S
    """
    rng   = np.random.default_rng(seed)
    S_aug = S.copy()
    n_mels, n_frames = S_aug.shape

    # Frequency masking: zero out `f` consecutive mel bins starting at f0
    for _ in range(num_freq_masks):
        f  = rng.integers(0, freq_mask_param + 1)
        f0 = rng.integers(0, max(1, n_mels - f))
        S_aug[f0:f0 + f, :] = 0.0

    # Time masking: zero out `t` consecutive frames starting at t0
    for _ in range(num_time_masks):
        t  = rng.integers(0, time_mask_param + 1)
        t0 = rng.integers(0, max(1, n_frames - t))
        S_aug[:, t0:t0 + t] = 0.0

    return S_aug


# Demonstrate on the Aves clip
bird_y = clips['Aves']['y']
bird_S = clips['Aves']['S']

S_freq  = spec_augment(bird_S, freq_mask_param=25, time_mask_param=0,  seed=0)
S_time  = spec_augment(bird_S, freq_mask_param=0,  time_mask_param=50, seed=1)
S_both  = spec_augment(bird_S, freq_mask_param=25, time_mask_param=50, seed=2)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
panels = [
    (bird_S, 'Original'),
    (S_freq, 'Freq masked\n(25 mel bins)'),
    (S_time, 'Time masked\n(50 frames)'),
    (S_both, 'Both masked'),
]
for ax, (S, title) in zip(axes, panels):
    show_spec(ax, S, title=title)

plt.suptitle('SpecAugment — Frequency & Time Masking  (Aves clip)',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'aug_01_specaugment.png', bbox_inches='tight', dpi=120)
plt.show()
print("Saved: aug_01_specaugment.png")

# %% [markdown]
# ## Augmentation 2: Time Shift (Circular)
#
# **What it does:** Rolls the waveform forward or backward in time by a
# random fraction of the clip length before computing the mel spectrogram.
# The circular roll means audio that falls off one end reappears at the other.
#
# **Why it helps:** Bird calls rarely start exactly at t=0 in a 5-second
# chunk. Time shift makes the model insensitive to where in the chunk the
# call begins — a simple but effective position-invariance trick.
# It is especially important for rare-species clips where we only have
# a handful of recordings.

# %%
def time_shift(y, shift_pct, cfg=CFG):
    """Circularly shift waveform by shift_pct * duration, then compute mel.

    Args:
        y: waveform (n_samples,)
        shift_pct: float in (-1, 1), positive = shift right
        cfg: audio config dict
    Returns:
        S_shifted: mel spectrogram after circular shift
    """
    n_shift = int(shift_pct * len(y))
    y_shifted = np.roll(y, n_shift)
    return make_melspec(y_shifted, cfg=cfg)


S_shift_25  = time_shift(bird_y, shift_pct=0.25)
S_shift_50  = time_shift(bird_y, shift_pct=0.50)
S_shift_neg = time_shift(bird_y, shift_pct=-0.30)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
panels = [
    (bird_S,      'Original'),
    (S_shift_25,  'Shift +25 %\n(+1.25 s right)'),
    (S_shift_50,  'Shift +50 %\n(+2.5 s right)'),
    (S_shift_neg, 'Shift −30 %\n(−1.5 s left)'),
]
for ax, (S, title) in zip(axes, panels):
    show_spec(ax, S, title=title)

plt.suptitle('Time Shift (Circular Roll)  (Aves clip)',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'aug_02_time_shift.png', bbox_inches='tight', dpi=120)
plt.show()
print("Saved: aug_02_time_shift.png")

# %% [markdown]
# ## Augmentation 3: Gaussian Noise
#
# **What it does:** Adds zero-mean Gaussian noise to the raw waveform
# before spectrogram computation.
#
# **Why it helps:** Real field recordings carry:
# - **Thermal (Johnson) noise** from the recorder's electronics
# - **Wind rumble** that adds broadband low-frequency energy
# - **Rain** which adds high-frequency white noise
#
# Even a small σ=0.005 dramatically improves robustness to the electronic
# noise floor present in 24/7 PAM deployments.

# %%
def add_noise(y, sigma, cfg=CFG):
    """Add Gaussian white noise to waveform, return mel spectrogram.

    Args:
        y: waveform (n_samples,)
        sigma: noise standard deviation (relative to unit-normalised signal)
        cfg: audio config dict
    Returns:
        S_noisy: mel spectrogram of noisy waveform
    """
    noise   = np.random.default_rng(7).normal(0, sigma, size=len(y))
    y_noisy = y + noise.astype(y.dtype)
    return make_melspec(y_noisy, cfg=cfg)


S_low_noise  = add_noise(bird_y, sigma=0.005)
S_high_noise = add_noise(bird_y, sigma=0.03)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
panels = [
    (bird_S,       'Original'),
    (S_low_noise,  'Low noise\n(σ = 0.005)'),
    (S_high_noise, 'High noise\n(σ = 0.03)'),
]
for ax, (S, title) in zip(axes, panels):
    show_spec(ax, S, title=title)

plt.suptitle('Gaussian Noise Addition  (Aves clip)',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'aug_03_gaussian_noise.png', bbox_inches='tight', dpi=120)
plt.show()
print("Saved: aug_03_gaussian_noise.png")

# %% [markdown]
# ## Augmentation 4: MixUp
#
# **What it does:** Creates a virtual training sample as a convex combination
# of two spectrograms.  The mixing coefficient λ is sampled from a
# Beta(α, α) distribution. The **labels** are mixed identically:
# `label_mixed = λ * label_A + (1 − λ) * label_B`
#
# **Why it helps:**
# - Creates infinitely many virtual training examples from a finite dataset
# - Forces the model to produce smoothly interpolated outputs, preventing
#   overconfident predictions
# - Acts as a form of implicit label smoothing
#
# **BirdCLEF note:** With multi-label targets (overlapping species) MixUp
# is especially natural — the mixed label is a probability vector, matching
# the sigmoid output of the model.
#
# **Key design choice:** α=0.4 (used by 2025 winners) is a good default.
# Higher α → more uniform mixing, lower α → stronger original samples.

# %%
def mixup_spectrogram(S1, S2, alpha=0.4, seed=None):
    """MixUp two mel spectrograms with Beta(alpha, alpha) mixing coefficient.

    Args:
        S1: first spectrogram (N_MELS, T)
        S2: second spectrogram (N_MELS, T) — will be resized to match S1
        alpha: Beta distribution parameter
        seed: optional random seed
    Returns:
        S_mix: mixed spectrogram, same shape as S1
        lam: scalar mixing coefficient
    """
    rng = np.random.default_rng(seed)
    lam = rng.beta(alpha, alpha)
    # Ensure same shape
    if S1.shape != S2.shape:
        S2 = np.array(
            torch.nn.functional.interpolate(
                torch.from_numpy(S2).unsqueeze(0).unsqueeze(0).float(),
                size=S1.shape, mode='bilinear', align_corners=False
            ).squeeze().numpy()
        )
    S_mix = lam * S1 + (1 - lam) * S2
    return S_mix, lam


# Pick two different classes for an instructive MixUp example
bird_cls  = 'Aves'
frog_cls  = 'Amphibia'
S_bird = clips[bird_cls]['S']
S_frog = clips[frog_cls]['S']

S_mixed, lam = mixup_spectrogram(S_bird, S_frog, alpha=0.4, seed=10)

fig = plt.figure(figsize=(18, 7))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3,
                        height_ratios=[3, 1])

# Spectrogram panels
ax_bird  = fig.add_subplot(gs[0, 0])
ax_frog  = fig.add_subplot(gs[0, 1])
ax_mixed = fig.add_subplot(gs[0, 2])

show_spec(ax_bird,  S_bird,  f'{bird_cls}  (clip A)',  cmap='magma')
show_spec(ax_frog,  S_frog,  f'{frog_cls}  (clip B)',  cmap='magma')
show_spec(ax_mixed, S_mixed, f'MixUp  λ={lam:.3f}\n(λ·A + (1−λ)·B)', cmap='magma')

ax_bird.set_title(f'{bird_cls}  (clip A)',
                  color=CLASS_COLORS.get(bird_cls, 'white'),
                  fontsize=12, fontweight='bold')
ax_frog.set_title(f'{frog_cls}  (clip B)',
                  color=CLASS_COLORS.get(frog_cls, 'white'),
                  fontsize=12, fontweight='bold')
ax_mixed.set_title(f'MixUp  λ = {lam:.3f}',
                   color='#FFD700', fontsize=12, fontweight='bold')

# Label bar charts
ax_bar_left   = fig.add_subplot(gs[1, 0])
ax_bar_middle = fig.add_subplot(gs[1, 1])
ax_bar_right  = fig.add_subplot(gs[1, 2])

species_names  = [bird_cls, frog_cls]
colors_bar     = [CLASS_COLORS.get(bird_cls, '#4C72B0'),
                  CLASS_COLORS.get(frog_cls, '#55A868')]

# Clip A label
ax_bar_left.bar(species_names, [1.0, 0.0], color=colors_bar, alpha=0.85, width=0.4)
ax_bar_left.set_ylim(0, 1.1)
ax_bar_left.set_title('Label A', fontsize=10)
ax_bar_left.set_ylabel('Label weight')
ax_bar_left.tick_params(labelsize=9)

# Clip B label
ax_bar_middle.bar(species_names, [0.0, 1.0], color=colors_bar, alpha=0.85, width=0.4)
ax_bar_middle.set_ylim(0, 1.1)
ax_bar_middle.set_title('Label B', fontsize=10)
ax_bar_middle.tick_params(labelsize=9)

# Mixed label
mixed_labels = [lam, 1 - lam]
bars = ax_bar_right.bar(species_names, mixed_labels, color=colors_bar,
                         alpha=0.85, width=0.4)
ax_bar_right.set_ylim(0, 1.1)
ax_bar_right.set_title(f'Mixed label  (λ={lam:.3f})', fontsize=10)
ax_bar_right.tick_params(labelsize=9)
for bar_obj, val in zip(bars, mixed_labels):
    ax_bar_right.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                      val + 0.03, f'{val:.3f}',
                      ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('MixUp: Virtual Training Sample with Mixed Labels',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig(OUTPUT_DIR / 'aug_04_mixup.png', bbox_inches='tight', dpi=120)
plt.show()
print(f"Saved: aug_04_mixup.png  (λ={lam:.3f})")

# %% [markdown]
# ## Augmentation 5: Background Mix (Most Important for This Competition)
#
# **What it does:** Mixes a labelled training clip with a short segment of
# a real unlabelled soundscape at a low amplitude ratio α (typically 0.1–0.3).
#
# **Why this is the highest-ROI augmentation for BirdCLEF 2026:**
# The competition test set consists entirely of raw Pantanal PAM recordings.
# These recordings have a very specific acoustic texture: continuous insect
# background drone, sporadic wind gusts, occasional rain, and distant frog
# choruses. The training XC clips have none of this. By mixing labelled clips
# with real unlabelled soundscape noise, we **directly close the domain gap**
# without needing any labels for the soundscapes.
#
# **Implementation note:** If unlabelled soundscapes are not available locally
# we fall back to **1/f (pink) noise**, which has a spectral shape similar to
# natural acoustic backgrounds. The principle is identical.

# %%
def make_pink_noise(n_samples, sr=32000, seed=42):
    """Generate 1/f (pink) noise — realistic stand-in for natural backgrounds.

    Uses the Voss-McCartney algorithm simplified via spectral shaping.
    """
    rng   = np.random.default_rng(seed)
    white = rng.normal(0, 1, n_samples)
    # Spectral shaping: 1/f roll-off in frequency domain
    fft   = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0/sr)
    freqs[0] = 1.0  # avoid divide-by-zero at DC
    fft_shaped = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft_shaped, n=n_samples).astype(np.float32)
    pink /= (np.max(np.abs(pink)) + 1e-8)
    return pink


def background_mix(y_clip, y_bg, alpha=0.2, cfg=CFG):
    """Mix a labelled clip with background noise.

    Args:
        y_clip: foreground waveform (n_samples,), normalised
        y_bg:   background waveform (>=n_samples,), normalised
        alpha:  background amplitude weight; lower = quieter background
        cfg:    audio config dict
    Returns:
        S_mixed: mel spectrogram of the mixed signal
        y_mixed: mixed waveform
    """
    n = len(y_clip)
    if len(y_bg) < n:
        y_bg = np.tile(y_bg, int(np.ceil(n / len(y_bg))))
    # Random crop of background to match clip length
    rng    = np.random.default_rng(99)
    start  = rng.integers(0, max(1, len(y_bg) - n))
    y_bg_c = y_bg[start:start + n].astype(np.float32)
    # Normalise both signals before mixing
    y_bg_c /= (np.max(np.abs(y_bg_c)) + 1e-8)
    y_mix  = (1 - alpha) * y_clip + alpha * y_bg_c
    y_mix /= (np.max(np.abs(y_mix)) + 1e-8)
    return make_melspec(y_mix, cfg=cfg), y_mix


# Try to find a real unlabelled soundscape; fall back to synthetic pink noise
soundscape_dir = BASE_DIR / 'unlabeled_soundscapes'
bg_source_label = ''
y_bg = None

if soundscape_dir.exists():
    sc_files = list(soundscape_dir.glob('*.ogg')) + list(soundscape_dir.glob('*.wav'))
    if sc_files:
        try:
            y_bg_full, _ = librosa.load(str(sc_files[0]), sr=CFG['sr'], mono=True,
                                        duration=30.0)
            y_bg = y_bg_full[:CFG['n_samples'] * 6]   # keep 30s
            bg_source_label = f'Real soundscape\n({sc_files[0].name[:20]}…)'
            print(f"Using real soundscape: {sc_files[0].name}")
        except Exception as e:
            print(f"Could not load soundscape: {e}")

if y_bg is None:
    y_bg = make_pink_noise(CFG['n_samples'] * 6, sr=CFG['sr'])
    bg_source_label = 'Synthetic 1/f\n(pink noise stand-in)'
    print("Using synthetic 1/f (pink) noise as background stand-in.")

S_bg_alone           = make_melspec(y_bg[:CFG['n_samples']], cfg=CFG)
S_bgmix_soft, y_mix  = background_mix(bird_y, y_bg, alpha=0.15)
S_bgmix_hard, _      = background_mix(bird_y, y_bg, alpha=0.35)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
panels = [
    (clips['Aves']['S'],  'Clean clip\n(Aves — original)'),
    (S_bg_alone,          f'Background alone\n{bg_source_label}'),
    (S_bgmix_soft,        'Background mix\nα = 0.15  (soft)'),
    (S_bgmix_hard,        'Background mix\nα = 0.35  (strong)'),
]
for ax, (S, title) in zip(axes, panels):
    show_spec(ax, S, title=title)

plt.suptitle('Background Mix — Closing the XC → PAM Domain Gap',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'aug_05_background_mix.png', bbox_inches='tight', dpi=120)
plt.show()
print("Saved: aug_05_background_mix.png")

# %% [markdown]
# ## Augmentation 6: Pitch Shift
#
# **What it does:** Shifts the pitch of the waveform up or down by a given
# number of semitones while preserving the duration (time-domain stretching +
# resampling via librosa's phase vocoder).
#
# **Why it helps:**
# - Bird call pitch varies by **temperature** (~0.5 semitones / 10°C)
# - **Individual variation** within a species can span ±1–2 semitones
# - **Geographic dialects**: the same species in different parts of the
#   Pantanal may have subtly different pitch registers
#
# **Safe range:** ±2 semitones preserves the spectral shape without
# introducing obvious vocoder artefacts.  Beyond ±4 semitones calls can
# start sounding like a different species — counterproductive.

# %%
def pitch_shift_mel(y, sr, n_steps, cfg=CFG):
    """Pitch-shift waveform by n_steps semitones, return mel spectrogram.

    Args:
        y: waveform (n_samples,)
        sr: sample rate
        n_steps: semitones to shift (positive = up, negative = down)
        cfg: audio config dict
    Returns:
        S_shifted: mel spectrogram after pitch shift
    """
    y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    return make_melspec(y_shifted, cfg=cfg)


S_pitch_down = pitch_shift_mel(bird_y, sr=CFG['sr'], n_steps=-2)
S_pitch_up   = pitch_shift_mel(bird_y, sr=CFG['sr'], n_steps=+2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
panels = [
    (bird_S,       'Original (0 semitones)'),
    (S_pitch_down, 'Pitch shift  −2 semitones\n(lower frequency)'),
    (S_pitch_up,   'Pitch shift  +2 semitones\n(higher frequency)'),
]
for ax, (S, title) in zip(axes, panels):
    show_spec(ax, S, title=title)

plt.suptitle('Pitch Shift  (Aves clip)',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'aug_06_pitch_shift.png', bbox_inches='tight', dpi=120)
plt.show()
print("Saved: aug_06_pitch_shift.png")

# %% [markdown]
# ## Full Pipeline: All Augmentations Applied Together
#
# In practice, each augmentation is applied **stochastically** with its own
# probability. The order matters:
#
# 1. **Background mix** (waveform-level) — before spectrogram computation
# 2. **Time shift** (waveform-level) — before spectrogram computation
# 3. **Pitch shift** (waveform-level) — before spectrogram computation
# 4. **SpecAugment** (spectrogram-level) — after spectrogram computation
#
# Waveform-level augmentations must come first because they interact with
# each other in physically realistic ways (background + shifted pitch is a
# realistic field scenario). Spectrogram-level augmentations come last since
# they operate on the final image.

# %%
def full_augmentation_pipeline(y, y_bg, cfg=CFG, seed=0):
    """Apply the full stochastic augmentation pipeline.

    Pipeline (in order):
      1. Background mix (p=0.5, α ~ Uniform(0.05, 0.25))
      2. Time shift     (p=0.3, shift ~ Uniform(-0.5, 0.5))
      3. Pitch shift    (p=0.2, n_steps ~ Uniform(-2, +2))
      4. Compute mel spectrogram
      5. SpecAugment freq mask (p=0.5)
      6. SpecAugment time mask (p=0.5)

    Returns:
        S_aug: augmented spectrogram
        log:   list of strings describing what was applied
    """
    rng = np.random.default_rng(seed)
    y_aug = y.copy()
    applied = []

    # 1. Background mix
    if rng.random() < 0.5:
        alpha = rng.uniform(0.05, 0.25)
        n = len(y_aug)
        if len(y_bg) < n:
            y_bg_use = np.tile(y_bg, int(np.ceil(n / len(y_bg))))
        else:
            y_bg_use = y_bg
        start = rng.integers(0, max(1, len(y_bg_use) - n))
        bg_seg = y_bg_use[start:start + n].astype(np.float32)
        bg_seg /= (np.max(np.abs(bg_seg)) + 1e-8)
        y_aug = (1 - alpha) * y_aug + alpha * bg_seg
        y_aug /= (np.max(np.abs(y_aug)) + 1e-8)
        applied.append(f'BG mix  α={alpha:.2f}')

    # 2. Time shift
    if rng.random() < 0.3:
        shift_pct = rng.uniform(-0.5, 0.5)
        y_aug = np.roll(y_aug, int(shift_pct * len(y_aug)))
        applied.append(f'Time shift  {shift_pct:+.0%}')

    # 3. Pitch shift
    if rng.random() < 0.2:
        n_steps = rng.uniform(-2, 2)
        y_aug = librosa.effects.pitch_shift(y=y_aug, sr=cfg['sr'], n_steps=n_steps)
        applied.append(f'Pitch shift  {n_steps:+.2f} st')

    # 4. Compute spectrogram
    S = make_melspec(y_aug, cfg=cfg)

    # 5. SpecAugment freq mask
    if rng.random() < 0.5:
        f = rng.integers(5, 25)
        f0 = rng.integers(0, max(1, cfg['n_mels'] - f))
        S[f0:f0 + f, :] = 0.0
        applied.append(f'Freq mask  {f} bins')

    # 6. SpecAugment time mask
    if rng.random() < 0.5:
        n_frames = S.shape[1]
        t = rng.integers(10, 50)
        t0 = rng.integers(0, max(1, n_frames - t))
        S[:, t0:t0 + t] = 0.0
        applied.append(f'Time mask  {t} frames')

    return S, applied


# Run multiple seeds to show variety
seeds_to_show = [3, 7, 13, 42]
augmented_examples = []
for sd in seeds_to_show:
    S_aug, log = full_augmentation_pipeline(bird_y, y_bg, cfg=CFG, seed=sd)
    augmented_examples.append((S_aug, log))

fig, axes = plt.subplots(1, 5, figsize=(24, 4))

# Panel 0: original
show_spec(axes[0], bird_S, 'Original\n(no augmentation)')
axes[0].set_title('Original\n(no augmentation)', fontsize=11,
                  fontweight='bold', color='white')

# Panels 1-4: augmented
palette = ['#FFD700', '#FF7F50', '#98FB98', '#87CEEB']
for i, ((S_aug, log), seed, color) in enumerate(
        zip(augmented_examples, seeds_to_show, palette)):
    ax = axes[i + 1]
    show_spec(ax, S_aug, '')
    applied_text = '\n'.join(log) if log else '(nothing applied)'
    ax.set_title(f'Augmented  (seed={seed})', fontsize=10,
                 fontweight='bold', color=color)
    ax.text(0.02, 0.97, applied_text, transform=ax.transAxes,
            fontsize=7.5, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                      alpha=0.65, edgecolor=color),
            color=color)

plt.suptitle('Full Augmentation Pipeline — 4 Random Seeds on the Same Clip',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'aug_07_full_pipeline.png', bbox_inches='tight', dpi=120)
plt.show()
print("Saved: aug_07_full_pipeline.png")

# %% [markdown]
# ## Summary: Recommended Augmentation Schedule
#
# The table below summarises the recommended probability and rationale for
# each augmentation based on the 2024–2025 top solutions.
#
# Note: probabilities are independent — multiple augmentations can be applied
# in a single training step.

# %%
summary_data = [
    ['Background mix',         '0.50', 'Closes XC → PAM domain gap directly'],
    ['MixUp',                  '0.30', 'Smooth decision boundaries + implicit label smoothing'],
    ['SpecAugment freq mask',  '0.50', 'Robustness to partial frequency coverage (wind, overlap)'],
    ['SpecAugment time mask',  '0.50', 'Robustness to partial temporal coverage (chunk edges)'],
    ['Time shift',             '0.30', 'Call position invariance within 5-second chunk'],
    ['Gaussian noise',         '0.20', 'Electronics / wind noise floor robustness'],
    ['Pitch shift',            '0.20', 'Inter-individual & temperature-driven pitch variation'],
]

col_headers = ['Augmentation', 'Probability', 'Why it helps']

fig, ax = plt.subplots(figsize=(15, 4))
ax.set_axis_off()

tbl = ax.table(
    cellText=summary_data,
    colLabels=col_headers,
    loc='center',
    cellLoc='left',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 2.0)

# Style the header row
for j in range(len(col_headers)):
    cell = tbl[0, j]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(color='white', fontweight='bold')

# Alternate row colours
for i in range(1, len(summary_data) + 1):
    row_color = '#1A1A2E' if i % 2 == 0 else '#16213E'
    for j in range(len(col_headers)):
        cell = tbl[i, j]
        cell.set_facecolor(row_color)
        cell.set_text_props(color='#E0E0E0')

# Highlight the highest-ROI row (background mix)
for j in range(len(col_headers)):
    tbl[1, j].set_facecolor('#2E4A1E')
    tbl[1, j].set_text_props(color='#98FB98', fontweight='bold')

# Column widths
tbl.auto_set_column_width([0, 1, 2])

plt.title('Recommended Augmentation Schedule for BirdCLEF 2026',
          fontsize=14, fontweight='bold', pad=20, color='white')
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#0D1117')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'aug_08_summary_table.png', bbox_inches='tight',
            dpi=120, facecolor=fig.get_facecolor())
plt.show()
print("Saved: aug_08_summary_table.png")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Background mix is the single most important augmentation** for BirdCLEF
#    2026 — it directly addresses the recording-condition domain gap between
#    XC training clips and Pantanal PAM test soundscapes.
#
# 2. **SpecAugment is universally effective** and costs almost nothing computationally.
#    Use both frequency and time masking with p=0.5 each.
#
# 3. **MixUp provides smooth decision boundaries** and doubles as label smoothing.
#    Use α=0.4 (Beta distribution). Don't forget to mix labels too.
#
# 4. **Pitch shift ±2 semitones** covers intra-species variation without
#    distorting calls into a different species' territory.
#
# 5. **Apply augmentations stochastically**, not deterministically — the model
#    should see both clean and augmented versions of each clip across epochs.
#
# 6. **The optimal order**: background mix → time shift → pitch shift
#    (waveform-level), then compute spectrogram, then SpecAugment.

# %%
# ── Final: print saved files summary ─────────────────────────────────────────
print("=" * 60)
print("All figures saved to:", OUTPUT_DIR)
print("=" * 60)
saved_figs = sorted(OUTPUT_DIR.glob('aug_*.png'))
for f in saved_figs:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<40}  {size_kb:>6.1f} KB")

print("\nDone! If you found this notebook useful, an upvote is appreciated 🙏")
