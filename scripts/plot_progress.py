"""Generate progress visualizations for the BirdCLEF 2026 README.

Outputs:
  docs/figures/lb_progression.png       - LB AUC by notebook over time
  docs/figures/alpha_sweep.png          - Alpha vs LB AUC from nb17 sweep
  docs/figures/architecture_perch_mlp.png - Pipeline diagram of nb15a/nb17b
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ----- 1. LB Progression -----------------------------------------------------
notebooks = [
    ("nb13",  0.839, "Perch MLP baseline\nPCA64+scalars5 -> MLP"),
    ("nb14a", 0.875, "BiGRU Replace\nctx5 replaces scalars"),
    ("nb14c", 0.879, "BiGRU Augment\nscalars + ctx8"),
    ("nb15a", 0.883, "Per-class logit\nMLP_IN=78"),
    ("nb16b", 0.817, "Pseudo-label t=0.8\nFAILED"),
    ("nb16c", 0.816, "Pseudo-label t=0.9\nFAILED"),
    ("nb17b", 0.892, "Alpha sweep alpha=0.6\nbest LB"),
]
WIN_LB = 0.96   # winning solution reference

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(notebooks))
y = np.array([nb[1] for nb in notebooks])
labels = [nb[0] for nb in notebooks]
captions = [nb[2] for nb in notebooks]

# segment colors by direction
colors = []
for i, val in enumerate(y):
    if i == 0:
        colors.append("#4477aa")
    elif val > y[i - 1]:
        colors.append("#228833")    # improvement = green
    elif val < y[i - 1] - 0.01:
        colors.append("#cc3311")    # regression > 0.01 = red
    else:
        colors.append("#bbbbbb")    # ~flat

ax.bar(x, y, color=colors, edgecolor="#222", linewidth=0.7, alpha=0.92)
for i, (val, cap) in enumerate(zip(y, captions)):
    ax.text(i, val + 0.003, f"{val:.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
    ax.text(i, 0.49, cap, ha="center", va="bottom", fontsize=7.5,
            color="#555", linespacing=1.1)

ax.axhline(WIN_LB, color="#aa3377", linestyle="--", linewidth=1.2, alpha=0.7)
ax.text(len(notebooks) - 0.55, WIN_LB - 0.003,
        f"2025 winning solution {WIN_LB:.2f}",
        color="#aa3377", fontsize=9, ha="right", va="top", style="italic")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("LB AUC (macro)")
ax.set_ylim(0.48, 0.98)
ax.set_title("BirdCLEF+ 2026 - LB progression on the Perch+MLP track",
             pad=14, fontweight="bold")

green_p = mpatches.Patch(color="#228833", label="Improvement")
gray_p  = mpatches.Patch(color="#bbbbbb", label="Roughly flat")
red_p   = mpatches.Patch(color="#cc3311", label="Regression > 0.01")
ax.legend(handles=[green_p, gray_p, red_p], loc="lower right", frameon=False,
          fontsize=9)

ax.grid(axis="y", linestyle=":", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "lb_progression.png", bbox_inches="tight")
plt.close()
print("Saved lb_progression.png")


# ----- 2. Alpha Sweep --------------------------------------------------------
alphas    = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
lb_scores = np.array([0.892, 0.892, 0.883, 0.878, 0.867])
sources   = ["nb17a", "nb17b", "nb15a", "nb17c", "nb17d"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(alphas, lb_scores, "o-", color="#4477aa", linewidth=2.2,
        markersize=11, markerfacecolor="#88aacc", markeredgecolor="#225588",
        markeredgewidth=1.5, zorder=3)

# Highlight tied best
for a, l, s in zip(alphas, lb_scores, sources):
    if l == lb_scores.max():
        ax.scatter([a], [l], s=230, marker="o", facecolor="none",
                   edgecolor="#228833", linewidth=2.5, zorder=4)
    ax.annotate(f"{s}\n{l:.3f}",
                xy=(a, l), xytext=(0, 14 if l < 0.89 else -28),
                textcoords="offset points", ha="center", fontsize=9,
                fontweight="bold" if l == lb_scores.max() else "normal",
                color="#222")

# Plateau band
ax.axhspan(0.890, 0.894, alpha=0.10, color="#228833")
ax.text(0.535, 0.894, "plateau (LB 0.892)", fontsize=8.5, color="#228833",
        style="italic")

ax.set_xlabel("alpha (MLP weight in final = alpha * MLP + (1-alpha) * Perch sigmoid)")
ax.set_ylabel("LB AUC (macro)")
ax.set_title("nb17 alpha sweep - Perch logit dominates the blend",
             pad=12, fontweight="bold")
ax.set_xlim(0.45, 0.95)
ax.set_ylim(0.860, 0.900)
ax.set_xticks(alphas)
ax.grid(axis="y", linestyle=":", alpha=0.4)

# Annotation about reframe
ax.text(0.87, 0.864,
        "Lower alpha -> better.\nPerch logit carries more signal\nthan the MLP probes for mapped species.",
        fontsize=9, ha="right", color="#555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3e0",
                  edgecolor="#bb9933", linewidth=0.8))
plt.tight_layout()
plt.savefig(FIG_DIR / "alpha_sweep.png", bbox_inches="tight")
plt.close()
print("Saved alpha_sweep.png")


# ----- 3. Architecture diagram (clean top-down) -----------------------------
fig, ax = plt.subplots(figsize=(11, 7.5))
ax.axis("off")
ax.set_xlim(0, 11); ax.set_ylim(0, 9)

def box(x, y, w, h, text, fc="#e8f0fc", ec="#225588", fs=10, fw="normal"):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                    facecolor=fc, edgecolor=ec, linewidth=1.4)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=fw)

def arrow(x1, y1, x2, y2, color="#444", lw=1.5, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                linestyle=ls))

# Row 1: Input
box(4.0, 7.9, 3.0, 0.7, "60s soundscape  ->  12 x 5s windows",
    fc="#fff3e0", ec="#bb6611")

# Row 2: Perch
box(3.5, 6.7, 4.0, 0.8,
    "Perch v2 ONNX  (frozen)",
    fc="#e8f0fc", ec="#225588", fw="bold", fs=11)
arrow(5.5, 7.9, 5.5, 7.5)

# Row 3: Two outputs
box(1.0, 5.3, 3.5, 0.8, "embeddings  (B, 1536)",
    fc="#f0f8ff", ec="#225588", fs=10)
box(6.5, 5.3, 3.5, 0.8, "projected logits  (B, 234)",
    fc="#f0f8ff", ec="#225588", fs=10)
arrow(4.5, 6.7, 2.75, 6.1)
arrow(6.5, 6.7, 8.25, 6.1)

# Row 4: 4 feature lanes
# Lane 1: PCA
box(0.2, 3.6, 2.1, 0.8, "PCA(64)",
    fc="#eef7ee", ec="#228833", fs=10, fw="bold")
arrow(2.75, 5.3, 1.25, 4.4)

# Lane 2: BiGRU ctx
box(2.6, 3.6, 2.1, 0.8, "BiGRU ctx(8)\nover 12 windows",
    fc="#eef7ee", ec="#228833", fs=9.5, fw="bold")
arrow(2.75, 5.3, 3.65, 4.4)   # from embeddings

# Lane 3: Hand-crafted scalars
box(5.0, 3.6, 2.4, 0.8, "5 temporal scalars\n(roll, mean, max, std)",
    fc="#eef7ee", ec="#228833", fs=9.5, fw="bold")
arrow(8.25, 5.3, 6.2, 4.4)

# Lane 4: Per-class logit
box(7.7, 3.6, 2.3, 0.8, "per-class logit(1)\nscores[:, c]",
    fc="#eef7ee", ec="#228833", fs=9.5, fw="bold")
arrow(8.25, 5.3, 8.85, 4.4)

# Row 5: Concat
box(3.5, 2.4, 4.0, 0.65, "concat  ->  78-d input to probe",
    fc="#fff5d6", ec="#bb9933", fw="bold", fs=10.5)
for x_src in [1.25, 3.65, 6.2, 8.85]:
    arrow(x_src, 3.6, 5.5, 3.05, color="#888", lw=1.1)

# Row 6: MLP
box(2.5, 1.2, 6.0, 0.85,
    "VectorizedMLP  -  234 probes via torch.bmm  -  78 -> 128 -> 64 -> 1",
    fc="#fce8f0", ec="#aa3377", fw="bold", fs=10.5)
arrow(5.5, 2.4, 5.5, 2.05, color="#aa3377")

# Row 7: Blend
box(0.5, 0.05, 4.0, 0.7, "sigmoid(MLP)",
    fc="#fbeeee", ec="#cc3311", fs=10, fw="bold")
box(6.5, 0.05, 4.0, 0.7, "sigmoid(Perch logit)",
    fc="#eef4fb", ec="#225588", fs=10, fw="bold")
arrow(3.5, 1.2, 2.5, 0.75, color="#cc3311")
ax.annotate("", xy=(8.5, 0.75), xytext=(8.5, 5.3),
            arrowprops=dict(arrowstyle="->", color="#225588", lw=1.2,
                            linestyle=":"))
ax.text(8.7, 3.0, "direct from\nPerch logits", fontsize=8.5,
        color="#225588", style="italic")

# Final blend annotation across the bottom
ax.text(5.5, -0.25,
        "final = alpha * sigmoid(MLP)  +  (1 - alpha) * sigmoid(Perch logit)"
        "        with  alpha=0.6 for mapped species  (alpha=1.0 for unmapped)",
        ha="center", fontsize=10.5, color="#bb6611", fontweight="bold")

ax.set_title(
    "Perch+MLP architecture  -  frozen Perch v2 features feed 234 per-class probes,\n"
    "blended with Perch's own logit at inference",
    pad=10, fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "architecture_perch_mlp.png", bbox_inches="tight")
plt.close()
print("Saved architecture_perch_mlp.png")

print("\nAll 3 figures written to docs/figures/")
