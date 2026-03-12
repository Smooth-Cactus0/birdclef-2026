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
#
# **What you'll learn:**
# 1. How the waveform → STFT → mel filterbank → log-compression pipeline works
# 2. What spectrograms look like across the 5 biological classes (birds, frogs, insects, mammals, caiman)
# 3. How changing `n_mels`, `fmin`, `fmax`, `hop_length` affects the image
# 4. How top-2025 augmentations (SpecAugment, MixUp) look on spectrograms

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths (auto-detect Kaggle vs local) ──────────────────────────────────────
BASE_DIR   = (Path('/kaggle/input/competitions/birdclef-2026')
              if Path('/kaggle/input/competitions/birdclef-2026').exists()
              else Path('birdclef-2026'))
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-dark')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})

print(f"BASE_DIR  : {BASE_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"librosa   : {librosa.__version__}")

# %% [markdown]
# ## Competition Audio Parameters
#
# These are the standard parameters used by the top 2024–2025 solutions.
# They are set as module-level constants so every function below uses them
# consistently.

# %%
# ── Competition-standard audio constants ──────────────────────────────────────
SR         = 32000   # Resample all audio to 32 kHz
N_FFT      = 1024    # FFT window size (~32 ms at 32 kHz)
HOP_LENGTH = 320     # Hop between frames (~10 ms) — good temporal resolution
N_MELS     = 128     # Number of mel filterbank bins (height of the spectrogram image)
FMIN       = 40      # Lowest frequency — keeps insect drones, frog bass calls
FMAX       = 15000   # Highest frequency — captures high-frequency bird calls
DURATION   = 5       # seconds — competition inference chunk size
N_SAMPLES  = SR * DURATION

print("Competition audio config:")
print(f"  SR         = {SR} Hz")
print(f"  N_FFT      = {N_FFT}  ({N_FFT/SR*1000:.1f} ms window)")
print(f"  HOP_LENGTH = {HOP_LENGTH}  ({HOP_LENGTH/SR*1000:.1f} ms hop)")
print(f"  N_MELS     = {N_MELS}")
print(f"  FMIN/FMAX  = {FMIN} / {FMAX} Hz")
print(f"  DURATION   = {DURATION}s  ({N_SAMPLES} samples per clip)")

# Resulting spectrogram dimensions
T_FRAMES = 1 + (N_SAMPLES - N_FFT) // HOP_LENGTH
print(f"\nSpectrogram shape: ({N_MELS}, {T_FRAMES})  ← height × width")
print(f"  Height = {N_MELS} mel bins")
print(f"  Width  = {T_FRAMES} time frames ({DURATION}s at {HOP_LENGTH}-sample hops)")


# ── Helper functions ──────────────────────────────────────────────────────────
def load_clip(path, target_sr=SR, duration=DURATION):
    """Load audio, resample to target_sr, pad or trim to fixed duration."""
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    n = target_sr * duration
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)), mode='constant')
    else:
        y = y[:n]
    return y


def make_melspec(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """Compute log-mel spectrogram, normalised to [0, 1]."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax)
    S_db   = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm

# %% [markdown]
# ## Step 1: Raw Waveform → STFT → Mel Spectrogram
#
# Four-panel figure showing every stage of the transformation.
# The final panel (④) is exactly what the CNN sees.

# %%
# Load a Bananaquit clip — the most common bird species in the dataset
bird_folder = BASE_DIR / 'train_audio' / 'banana'
bird_files  = sorted(bird_folder.glob('*.ogg'))

if not bird_files:
    print("No Bananaquit files found — check BASE_DIR")
else:
    bird_file = bird_files[0]
    y = load_clip(bird_file)
    t = np.linspace(0, DURATION, len(y))

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    # ① Raw waveform
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, y, color='#4C72B0', linewidth=0.5, alpha=0.9)
    ax1.set_title('① Raw Waveform', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, DURATION)

    # ② STFT magnitude (linear frequency axis)
    D   = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    ax2 = fig.add_subplot(gs[1, 0])
    img2 = librosa.display.specshow(
        librosa.amplitude_to_db(D, ref=np.max),
        y_axis='linear', x_axis='time', sr=SR,
        hop_length=HOP_LENGTH, ax=ax2, cmap='magma')
    ax2.set_title('② STFT Magnitude (linear freq)', fontsize=12, fontweight='bold')
    plt.colorbar(img2, ax=ax2, format='%+2.0f dB')

    # ③ Mel filterbank applied (log-mel, dB scale)
    S   = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    ax3 = fig.add_subplot(gs[1, 1])
    img3 = librosa.display.specshow(
        librosa.power_to_db(S, ref=np.max),
        y_axis='mel', x_axis='time', sr=SR,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX, ax=ax3, cmap='magma')
    ax3.set_title(
        f'③ Log-Mel Spectrogram ({N_MELS} mel bins, {FMIN}–{FMAX} Hz)',
        fontsize=12, fontweight='bold')
    plt.colorbar(img3, ax=ax3, format='%+2.0f dB')

    # ④ Normalised [0,1] — CNN input
    S_norm = make_melspec(y)
    ax4    = fig.add_subplot(gs[2, :])
    im4    = ax4.imshow(
        S_norm, aspect='auto', origin='lower', cmap='magma',
        extent=[0, DURATION, FMIN / 1000, FMAX / 1000])
    ax4.set_title(
        f'④ Normalised Log-Mel (CNN input) — shape: {S_norm.shape[0]} × {S_norm.shape[1]}',
        fontsize=13, fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (kHz)')
    plt.colorbar(im4, ax=ax4)

    plt.suptitle(
        f'Audio → Mel Spectrogram Pipeline\n(File: {bird_file.name})',
        fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(OUTPUT_DIR / 'pipeline_demo.png', bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Saved → pipeline_demo.png")
    print(f"CNN input shape: {S_norm.shape}  (n_mels × time_frames)")

# %% [markdown]
# ## Cross-Taxa Spectrogram Gallery
#
# One of the most educational things we can do: compare spectrograms across
# the 5 biological classes. Birds, frogs, insects, and mammals each occupy
# distinct **frequency bands** and have characteristic **temporal patterns**.
#
# | Class | Frequency range | Temporal pattern |
# |---|---|---|
# | Aves | 1–10 kHz | Short, repeating syllables |
# | Amphibia | 0.5–5 kHz | Sustained calls, sometimes harmonic |
# | Insecta | 2–12 kHz | Continuous high-frequency drone or pulse trains |
# | Mammalia | 0.1–3 kHz | Low-frequency howls, grunts |
# | Reptilia | Very low, rare vocalisation | Almost silent — hard challenge! |

# %%
TAXA_EXAMPLES = [
    ('Aves',     'banana',  'Bananaquit (Aves)'),
    ('Amphibia', '22961',   'Pointedbelly Frog (Amphibia)'),
    ('Insecta',  '1161364', 'Guyalna cuta (Insecta)'),
    ('Mammalia', '43435',   'Black Howler Monkey (Mammalia)'),
    ('Reptilia', '116570',  'Caiman yacare (Reptilia)'),
]

CLASS_CMAPS = {
    'Aves':     'Blues',
    'Amphibia': 'Greens',
    'Insecta':  'Reds',
    'Mammalia': 'Oranges',
    'Reptilia': 'Purples',
}

fig, axes = plt.subplots(len(TAXA_EXAMPLES), 1, figsize=(14, 14))

for ax, (cls, folder, label) in zip(axes, TAXA_EXAMPLES):
    folder_path = BASE_DIR / 'train_audio' / folder
    if not folder_path.exists():
        ax.set_title(f'{label} — folder not found: {folder_path}')
        ax.axis('off')
        continue
    files = sorted(folder_path.glob('*.ogg'))
    if not files:
        ax.set_title(f'{label} — no OGG files in {folder}')
        ax.axis('off')
        continue
    y      = load_clip(files[0])
    S_norm = make_melspec(y)
    im     = ax.imshow(
        S_norm, aspect='auto', origin='lower', cmap=CLASS_CMAPS[cls],
        extent=[0, DURATION, FMIN / 1000, FMAX / 1000])
    ax.set_title(f'{label}  |  {files[0].name}', fontweight='bold', fontsize=11)
    ax.set_ylabel('Freq (kHz)')
    plt.colorbar(im, ax=ax)

axes[-1].set_xlabel('Time (s)')
plt.suptitle(
    f'Mel Spectrograms Across Taxonomic Classes\n'
    f'SR={SR} Hz, n_mels={N_MELS}, fmin={FMIN} Hz, fmax={FMAX} Hz',
    fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'taxa_spectrogram_gallery.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved → taxa_spectrogram_gallery.png")

# %% [markdown]
# ## How Parameters Affect the Spectrogram
#
# The 3×3 grid below answers three common questions:
# - **How many mel bins?** More bins → finer frequency resolution but bigger model input
# - **fmax cutoff?** Lower fmax saves some computation but cuts bird high-frequency calls
# - **Hop length?** Smaller hop → more time frames, better temporal resolution, but more GPU memory
# - **n_fft?** Larger window → better frequency resolution, worse time resolution (Heisenberg trade-off)
#
# Green label = competition standard.

# %%
y_ref   = None
ref_files = sorted((BASE_DIR / 'train_audio' / 'banana').glob('*.ogg'))
if ref_files:
    y_ref = load_clip(ref_files[0])

if y_ref is not None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    configs = [
        # Row 0 — vary n_mels
        dict(n_mels=64,  fmin=40,  fmax=15000, hop=HOP_LENGTH, nfft=N_FFT,
             title='n_mels=64  (coarser freq)'),
        dict(n_mels=128, fmin=40,  fmax=15000, hop=HOP_LENGTH, nfft=N_FFT,
             title='n_mels=128 ✓ (standard)'),
        dict(n_mels=256, fmin=40,  fmax=15000, hop=HOP_LENGTH, nfft=N_FFT,
             title='n_mels=256  (finer freq)'),
        # Row 1 — vary frequency range
        dict(n_mels=128, fmin=40,  fmax=8000,  hop=HOP_LENGTH, nfft=N_FFT,
             title='fmax=8 kHz  (misses hi-freq birds)'),
        dict(n_mels=128, fmin=40,  fmax=15000, hop=HOP_LENGTH, nfft=N_FFT,
             title='fmax=15 kHz ✓'),
        dict(n_mels=128, fmin=300, fmax=15000, hop=HOP_LENGTH, nfft=N_FFT,
             title='fmin=300 Hz  (misses bass calls)'),
        # Row 2 — vary hop length and n_fft
        dict(n_mels=128, fmin=40,  fmax=15000, hop=160,        nfft=N_FFT,
             title='hop=160  (2× temporal res)'),
        dict(n_mels=128, fmin=40,  fmax=15000, hop=640,        nfft=N_FFT,
             title='hop=640  (½ temporal res)'),
        dict(n_mels=128, fmin=40,  fmax=15000, hop=HOP_LENGTH, nfft=2048,
             title='n_fft=2048  (better freq res)'),
    ]

    for ax, cfg in zip(axes.flat, configs):
        S = librosa.feature.melspectrogram(
            y=y_ref, sr=SR,
            n_fft=cfg['nfft'], hop_length=cfg['hop'],
            n_mels=cfg['n_mels'], fmin=cfg['fmin'], fmax=cfg['fmax'])
        S_db   = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        ax.imshow(S_norm, aspect='auto', origin='lower', cmap='magma')
        color = 'limegreen' if '✓' in cfg['title'] else 'white'
        ax.set_title(cfg['title'], fontsize=9, fontweight='bold', color=color)
        # Add shape annotation
        ax.text(0.02, 0.05, f'{S_norm.shape[0]}×{S_norm.shape[1]}',
                transform=ax.transAxes, fontsize=8, color='yellow',
                fontweight='bold')
        ax.axis('off')

    plt.suptitle(
        'Spectrogram Parameter Sweep  (green = competition standard)\n'
        'Shape annotation = n_mels × time_frames',
        fontsize=13, fontweight='bold', color='white')
    fig.patch.set_facecolor('#1a1a2e')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'parameter_sweep.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("Saved → parameter_sweep.png")
else:
    print("Skipping parameter sweep — no Bananaquit files found")

# %% [markdown]
# ## Data Augmentation Techniques
#
# The top 2025 solutions all used a combination of:
# - **SpecAugment** (Park et al., 2019): zero out random time and frequency bands
# - **MixUp** (Zhang et al., 2018): blend two spectrograms + blend their labels
# - **Time shift**: cyclic roll along the time axis
#
# These augmentations are applied *on the spectrogram*, not on the raw audio,
# which is much cheaper computationally.

# %%
bird_files_aug  = sorted((BASE_DIR / 'train_audio' / 'banana').glob('*.ogg'))
frog_files_aug  = sorted((BASE_DIR / 'train_audio' / '22961').glob('*.ogg'))

if bird_files_aug and frog_files_aug:
    np.random.seed(42)

    y_base = load_clip(bird_files_aug[0])
    S_base = make_melspec(y_base)

    y_frog  = load_clip(frog_files_aug[0])
    S_frog  = make_melspec(y_frog)

    # ── Augmentation functions ────────────────────────────────────────────────
    def time_mask(S, T=30):
        """Zero out a random time band (SpecAugment)."""
        S2 = S.copy()
        t0 = np.random.randint(0, S.shape[1] - T)
        S2[:, t0:t0 + T] = 0.0
        return S2

    def freq_mask(S, F=20):
        """Zero out a random frequency band (SpecAugment)."""
        S2 = S.copy()
        f0 = np.random.randint(0, S.shape[0] - F)
        S2[f0:f0 + F, :] = 0.0
        return S2

    def mixup(S1, S2, lam=0.5):
        """Linear blend of two spectrograms (MixUp)."""
        return lam * S1 + (1 - lam) * S2

    def time_shift(S, shift=80):
        """Cyclic roll along the time axis."""
        return np.roll(S, shift, axis=1)

    specs = [
        (S_base,                           'Original (bird)'),
        (time_mask(S_base, T=30),          'Time Masking  T=30 frames'),
        (freq_mask(S_base, F=20),          'Freq Masking  F=20 bins'),
        (time_mask(freq_mask(S_base)),     'Both Masks'),
        (mixup(S_base, S_frog, lam=0.5),  'MixUp λ=0.5  (bird + frog)'),
        (time_shift(S_base, shift=80),    'Time Shift  +80 frames'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (S, title) in zip(axes.flat, specs):
        ax.imshow(S, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.axis('off')

    plt.suptitle('Augmentation Techniques on Mel Spectrograms',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'augmentation_demo.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("Saved → augmentation_demo.png")

    # ── Print label blending explanation ─────────────────────────────────────
    print("\nMixUp label blending:")
    print("  If bird = class 3, frog = class 87 and λ = 0.5:")
    print("  target = 0.5 * one_hot(3) + 0.5 * one_hot(87)")
    print("  CrossEntropy works with soft targets directly.")
else:
    print("Skipping augmentation demo — missing bird or frog files")

# %% [markdown]
# ## Key Takeaways 🎯
#
# | Topic | Competition guideline |
# |---|---|
# | **Standard config** | SR=32000, n_mels=128, fmin=40, fmax=15000, hop=320 |
# | **CNN input shape** | (1, 128, 501) — single-channel image, 128 height, 501 width |
# | **Augmentation** | SpecAugment (freq+time mask) + MixUp each epoch |
# | **Why mel scale?** | Matches human (and animal) pitch perception — birds sing in 'octaves' |
# | **Why fmin=40?** | Captures frog bass calls and Howler Monkey low grunts |
# | **Why fmax=15kHz?** | Many birds have calls up to 12–14 kHz; insects even higher |
# | **Why log(power)?** | Compresses 6+ orders of magnitude into a visible range |

# %%
print("Outputs written to:", OUTPUT_DIR)
for f in sorted(OUTPUT_DIR.glob('*.png')):
    print(f"  {f.name}")
