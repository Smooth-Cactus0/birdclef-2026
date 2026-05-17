"""Build a multi-page PDF status report for the BirdCLEF+ 2026 campaign.

Pulls existing figures from docs/figures/, parses run1/run2 logs to make
training-curve plots, and composes everything plus tables into a single
docs/reports/birdclef-2026-status.pdf.

Run:  python scripts/build_status_report.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT     = Path(__file__).parent.parent
FIG_DIR  = ROOT / "docs" / "figures"
PDF_DIR  = ROOT / "docs" / "reports"; PDF_DIR.mkdir(parents=True, exist_ok=True)
RESULTS  = ROOT / "results"
OUT_PDF  = PDF_DIR / "birdclef-2026-status.pdf"

plt.rcParams.update({
    "figure.dpi":     110,
    "savefig.dpi":    140,
    "font.size":      10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PAGE = (11.0, 8.5)   # landscape letter-ish

# --- Static project state ----------------------------------------------------
LB_HISTORY = [
    ("nb13",   0.839, "Perch MLP baseline",         "#4477aa"),
    ("nb14a",  0.875, "BiGRU Replace",              "#228833"),
    ("nb14c",  0.879, "BiGRU Augment",              "#228833"),
    ("nb15a",  0.883, "Per-class logit",            "#228833"),
    ("nb16b",  0.817, "PL t=0.8 (failed)",          "#cc3311"),
    ("nb16c",  0.816, "PL t=0.9 (failed)",          "#cc3311"),
    ("nb17b",  0.892, "Alpha sweep alpha=0.6",      "#228833"),
    ("nb18a",  0.852, "First CNN baseline",         "#4477aa"),
    ("nb19a",  0.897, "Top-K postproc",             "#228833"),
    ("nb20b",  None,  "5-fold CNN ensemble (pending submit)",   "#dddd66"),
]
WINNING_LB = 0.96

FOLD_AUC = [
    (0, 0.9768, 12),
    (1, 0.9801, 16),
    (2, 0.9785, None),
    (3, 0.9812, None),
    (4, 0.9795, None),
]

DEBUG_CHAIN = [
    ("v1",     "Slug 409 conflict",       "title '0.5' -> slug '0-5' != id"),
    ("v2",     "BCELoss autocast-unsafe", "swap to BCEWithLogitsLoss"),
    ("v3",     "CUDA assert input in [0,1]","NaN slips past .clamp() -> nan_to_num"),
    ("v4",     "Collate non-resizable",    "np.tile slice is a view -> np.array(copy=True)"),
    ("v5 NaN", "All train=nan from ep 1",  "fp16 FFT n_fft=4096 -> autocast(enabled=False) on mel"),
    ("v5 OK",  "Healthy ep1 train=0.048", "val_auc=0.854 in fold 0"),
]


def parse_log_for_epochs(log_path: Path):
    """Extract (epoch, val_auc) tuples per fold from a Kaggle kernel log."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = re.findall(r'"data":"([^"]*)"', text)
    folds = {}
    current_fold = None
    for r in rows:
        m_fold = re.search(r"=== Fold (\d+)", r)
        if m_fold:
            current_fold = int(m_fold.group(1))
            folds.setdefault(current_fold, [])
            continue
        m_ep = re.search(r"ep\s+(\d+)\s+train=([\d.]+)\s+val_auc=([\d.]+)", r)
        if m_ep and current_fold is not None:
            folds[current_fold].append((int(m_ep.group(1)),
                                        float(m_ep.group(2)),
                                        float(m_ep.group(3))))
    return folds


# --- Page builders -----------------------------------------------------------
def page_text(pdf, title, body, sub=""):
    fig, ax = plt.subplots(figsize=PAGE)
    ax.axis("off")
    ax.text(0.5, 0.92, title, ha="center", va="top", fontsize=24,
            fontweight="bold", transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.86, sub, ha="center", va="top", fontsize=12,
                color="#555", style="italic", transform=ax.transAxes)
    ax.text(0.06, 0.80, body, ha="left", va="top", fontsize=11,
            family="monospace", transform=ax.transAxes, linespacing=1.6,
            wrap=True)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_image(pdf, title, image_path, caption=""):
    fig, ax = plt.subplots(figsize=PAGE)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    ax.axis("off")
    if image_path.exists():
        img = mpimg.imread(image_path)
        ax.imshow(img); ax.set_aspect("auto")
    else:
        ax.text(0.5, 0.5, f"[missing figure: {image_path.name}]",
                ha="center", va="center", color="red")
    if caption:
        ax.text(0.5, -0.04, caption, ha="center", va="top",
                fontsize=10, transform=ax.transAxes, color="#444",
                style="italic", wrap=True)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_cover(pdf):
    fig, ax = plt.subplots(figsize=PAGE)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.85, "BirdCLEF+ 2026", ha="center", fontsize=42,
            fontweight="bold")
    ax.text(0.5, 0.78, "Status Report",
            ha="center", fontsize=22, color="#555", style="italic")
    ax.text(0.5, 0.71,
            "Wildlife species identification from passive acoustic monitoring\n"
            "in the Pantanal wetlands (Brazil).  Kaggle code competition.\n"
            "Metric: macro ROC-AUC. Deadline: June 3, 2026.",
            ha="center", fontsize=11, color="#444", linespacing=1.6)

    # KPI box
    ax.add_patch(mpatches.FancyBboxPatch((0.18, 0.38), 0.64, 0.22,
                                          boxstyle="round,pad=0.02",
                                          facecolor="#f4f8fc",
                                          edgecolor="#225588", linewidth=1.4))
    ax.text(0.5, 0.56, "Current state", ha="center", fontsize=14,
            fontweight="bold", color="#225588")
    ax.text(0.5, 0.50, "Best LB: 0.897  (nb19a -- Perch+MLP + top-K postproc)",
            ha="center", fontsize=12)
    ax.text(0.5, 0.46, "Gap to 2025 winning solution (0.96): 0.063",
            ha="center", fontsize=11, color="#aa3377")
    ax.text(0.5, 0.42,
            "CNN track: 5-fold ensemble trained, OOF macro-AUC 0.9792 (LB pending)",
            ha="center", fontsize=11, color="#228833")

    ax.text(0.5, 0.25, "Author: Alexy Louis",
            ha="center", fontsize=10, color="#666")
    ax.text(0.5, 0.21, f"Generated: 2026-05-17",
            ha="center", fontsize=9, color="#888")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_dataset(pdf):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle("Dataset overview", fontsize=20, fontweight="bold", y=0.97)

    # 2x2 layout: text+stats top-left, three figures around
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25,
                          left=0.05, right=0.97, top=0.92, bottom=0.05)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.text(0.02, 0.98,
            "BirdCLEF+ 2026 -- key numbers\n"
            "------------------------------\n\n"
            "  Total training clips    : 35,549\n"
            "  Species (multi-taxa)    : 234\n"
            "  Aves   (birds)          : 162\n"
            "  Amphibia                : 35\n"
            "  Insecta                 : 28  (+25 sonotypes)\n"
            "  Mammalia                : 8\n"
            "  Reptilia                : 1 (Caiman, 1 clip)\n\n"
            "  train_audio (XC + iNat) : 35,549\n"
            "  train_soundscapes       : 10,658\n"
            "  Labeled segments        : 1,478 (66 files)\n\n"
            "  Region: Pantanal, Brazil\n"
            "  Lat -21.6 / -16.5,  Lon -57.6 / -55.9\n"
            "  Format: OGG, 32 kHz",
            ha="left", va="top", fontsize=10, family="monospace",
            transform=ax.transAxes, linespacing=1.4)

    for ax_pos, fname, title in [
        (gs[0, 1], "taxa_breakdown.png",           "Taxa breakdown"),
        (gs[1, 0], "class_imbalance.png",          "Class imbalance (long tail)"),
        (gs[1, 1], "geographic_distribution.png",  "Geographic domain shift"),
    ]:
        ax = fig.add_subplot(ax_pos)
        path = FIG_DIR / fname
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        if path.exists():
            ax.imshow(mpimg.imread(path)); ax.set_aspect("auto")
        else:
            ax.text(0.5, 0.5, f"[missing: {fname}]", ha="center", va="center")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_lb_progression(pdf):
    fig, ax = plt.subplots(figsize=PAGE)
    fig.suptitle("LB progression -- 11 notebooks, two tracks",
                 fontsize=20, fontweight="bold", y=0.97)
    x = np.arange(len(LB_HISTORY))
    ys = [lb if lb is not None else None for _, lb, _, _ in LB_HISTORY]
    colors = [c for *_, c in LB_HISTORY]
    for i, (nm, lb, lbl, col) in enumerate(LB_HISTORY):
        if lb is None:
            # pending bar
            ax.bar(i, 0.5, color=col, alpha=0.35, edgecolor="#888",
                   linewidth=0.7, hatch="//")
            ax.text(i, 0.52, "pending", ha="center", va="bottom",
                    fontsize=8, color="#888", fontweight="bold")
            ax.text(i, 0.45, lbl, ha="center", va="top", fontsize=7,
                    color="#555")
        else:
            ax.bar(i, lb, color=col, edgecolor="#222", linewidth=0.7, alpha=0.9)
            ax.text(i, lb + 0.003, f"{lb:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")
            ax.text(i, 0.46, lbl, ha="center", va="top", fontsize=7,
                    color="#555")

    ax.axhline(WINNING_LB, color="#aa3377", linestyle="--", linewidth=1.2,
               alpha=0.7)
    ax.text(len(LB_HISTORY) - 0.5, WINNING_LB - 0.005,
            f"2025 winning solution {WINNING_LB:.2f}", ha="right",
            va="top", color="#aa3377", fontsize=10, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([n for n, *_ in LB_HISTORY], fontsize=9)
    ax.set_ylabel("LB AUC (macro)")
    ax.set_ylim(0.42, 0.98)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    handles = [
        mpatches.Patch(color="#4477aa", label="Baseline / new architecture"),
        mpatches.Patch(color="#228833", label="Improvement on track"),
        mpatches.Patch(color="#cc3311", label="Failure (regression)"),
        mpatches.Patch(color="#dddd66", label="Pending LB submission"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_alpha_sweep(pdf):
    page_image(pdf, "Alpha sweep -- the architectural reframe of nb17",
               FIG_DIR / "alpha_sweep.png",
               caption=("Lower alpha -> better LB.  Perch's logit carries more signal "
                        "than the MLP probes for mapped species, so the MLP is a "
                        "correction layer on top of Perch, not an independent predictor."))


def page_perch_arch(pdf):
    page_image(pdf, "Perch+MLP architecture (LB 0.892 single, 0.897 with top-K)",
               FIG_DIR / "architecture_perch_mlp.png",
               caption=("Frozen Perch v2 features feed 234 per-class probes via "
                        "torch.bmm; alpha=0.6 blends the MLP correction with "
                        "Perch's sigmoid at inference time."))


def page_cnn_arch(pdf):
    # Schematic of the CNN+SED pipeline (built fresh here)
    fig, ax = plt.subplots(figsize=PAGE)
    ax.set_title("CNN+SED architecture (nb20)", fontsize=18, fontweight="bold",
                 pad=20)
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 9)

    def box(x, y, w, h, text, fc="#e8f0fc", ec="#225588", fs=10, fw="normal"):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                        facecolor=fc, edgecolor=ec,
                                        linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight=fw)

    def arrow(x1, y1, x2, y2, c="#444", lw=1.6):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=c, lw=lw))

    # Top: input
    box(4.5, 7.9, 3.0, 0.7, "20 s audio chunk @ 32 kHz",
        fc="#fff3e0", ec="#bb6611")

    # MixUp annotation
    ax.text(0.5, 8.2, "Raw-audio MixUp\n(p=0.5, weight=0.5,\nlabel union)",
            fontsize=9, color="#aa3377", style="italic", linespacing=1.2,
            ha="left")

    # Mel
    box(4.0, 6.6, 4.0, 0.8,
        "Mel spectrogram (fp32)\n"
        "n_mels=224, n_fft=4096, hop=1252",
        fc="#e8f0fc", ec="#225588", fs=10)
    arrow(6.0, 7.9, 6.0, 7.4)

    # 3-channel repeat + size
    box(4.5, 5.3, 3.0, 0.6, "repeat 3x -> (3, 224, 512)",
        fc="#e8f0fc", ec="#225588", fs=10)
    arrow(6.0, 6.6, 6.0, 5.9)

    # Backbone
    box(3.5, 3.7, 5.0, 1.3,
        "EfficientNet-B0  (timm, ImageNet pretrain)\n"
        "AMP fp16, AdamW, cosine LR 5e-4 -> 1e-6\n"
        "20 epochs x 5 folds (split 2+3 across two kernels)",
        fc="#fce8f0", ec="#aa3377", fw="bold", fs=10)
    arrow(6.0, 5.3, 6.0, 5.0)

    # Frequency pool
    box(4.0, 2.8, 4.0, 0.55,
        "Frequency pool  (mean over height)",
        fc="#eef7ee", ec="#228833", fs=10)
    arrow(6.0, 3.7, 6.0, 3.35)

    # SED head
    box(3.3, 1.4, 5.4, 1.1,
        "SED head\n"
        "frame_lg = fc_cla(feat)\n"
        "att = softmax(tanh(fc_att(feat)), dim=time)\n"
        "clip_lg = sum(att * frame_lg, dim=time)",
        fc="#eef7ee", ec="#228833", fs=9, fw="bold")
    arrow(6.0, 2.8, 6.0, 2.5)

    # BCEWithLogitsLoss
    box(0.8, 1.4, 2.0, 1.1,
        "BCE with\nLogits Loss\n(secondary\nlabels = 1)",
        fc="#fbeeee", ec="#cc3311", fs=9, fw="bold")
    arrow(3.3, 1.95, 2.8, 1.95, c="#cc3311")

    # Output
    box(4.3, 0.3, 3.4, 0.6,
        "clip logits  (B, 234)",
        fc="#fff5d6", ec="#bb9933", fw="bold", fs=10)
    arrow(6.0, 1.4, 6.0, 0.9)

    # Annotations on the right
    ax.text(9.5, 7.8,
            "Inference (nb20b)\n"
            "----------------\n"
            "60 s soundscape ->\n"
            "9 x 20 s windows\n"
            "with 5 s stride\n"
            "5-fold ensemble\n"
            "framewise overlap\n"
            "averaging into\n"
            "12 x 5 s output\n"
            "slots\n\n"
            "+ Top-K postproc\n"
            "  (per-file)\n\n"
            "Internet OFF",
            fontsize=9, family="monospace", color="#225588",
            ha="left", va="top", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f8fc",
                      edgecolor="#225588"))

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_training_curves(pdf):
    """Per-fold val_auc-by-epoch from the run1+run2 logs."""
    run1_log = RESULTS / "nb20a_run1_done" / "birdclef-2026-cnn-sed-b0-run1.log"
    run2_log = RESULTS / "nb20a_run2_done" / "birdclef-2026-cnn-sed-b0-run2.log"
    folds = parse_log_for_epochs(run1_log)
    folds.update(parse_log_for_epochs(run2_log))

    fig, ax = plt.subplots(figsize=PAGE)
    fig.suptitle("nb20a training curves -- 5 folds, 20 epochs each",
                 fontsize=20, fontweight="bold", y=0.97)
    colors = ["#4477aa", "#228833", "#aa3377", "#bb6611", "#dd5577"]
    for fi in sorted(folds.keys()):
        rows = folds[fi]
        if not rows: continue
        eps   = [r[0] for r in rows]
        aucs  = [r[2] for r in rows]
        ax.plot(eps, aucs, "o-", color=colors[fi % 5], linewidth=1.8,
                markersize=5, label=f"fold {fi}  (best AUC={max(aucs):.4f})")

    ax.axhline(0.97, color="#aaa", linestyle=":", linewidth=0.8)
    ax.text(0.4, 0.967, "0.97 reference", color="#888", fontsize=8.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val AUC (macro)")
    ax.set_xlim(0.5, 20.5); ax.set_ylim(0.83, 0.99)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.text(0.5, 0.025,
            "All 5 folds clear val_auc 0.97 by ~epoch 5 and plateau ~0.978 - 0.981.\n"
            "Tight spread (0.0044) signals a stable ensemble; OOF mean = 0.9792.\n"
            "Caveat: StratifiedKFold-by-primary_label likely overstates the LB.",
            transform=fig.transFigure, ha="center", va="bottom",
            fontsize=10, color="#444", style="italic", linespacing=1.5)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_debug_chain(pdf):
    fig, ax = plt.subplots(figsize=PAGE)
    fig.suptitle("nb20a debug chain -- 4 failures before clean training",
                 fontsize=20, fontweight="bold", y=0.97)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Build a simple stacked-rows table
    n = len(DEBUG_CHAIN)
    row_h = 0.10
    top   = 0.85
    headers = ["Version", "Failure mode", "Fix"]
    col_xs  = [0.06, 0.22, 0.60]
    col_ws  = [0.14, 0.36, 0.36]

    for cx, w, h in zip(col_xs, col_ws, headers):
        ax.add_patch(mpatches.Rectangle((cx, top), w, row_h,
                                         facecolor="#225588", edgecolor="#fff"))
        ax.text(cx + w/2, top + row_h/2, h, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
    for i, (ver, fail, fix) in enumerate(DEBUG_CHAIN):
        y = top - (i + 1) * row_h
        fc = "#f9f9f9" if i % 2 == 0 else "#eef4fc"
        if "OK" in ver:
            fc = "#e6f4e6"
        for cx, w, text in zip(col_xs, col_ws, [ver, fail, fix]):
            ax.add_patch(mpatches.Rectangle((cx, y), w, row_h, facecolor=fc,
                                             edgecolor="#ccc", linewidth=0.5))
            ax.text(cx + 0.01, y + row_h/2, text, ha="left", va="center",
                    fontsize=10,
                    fontweight=("bold" if cx == col_xs[0] else "normal"),
                    family=("monospace" if cx == col_xs[0] else "sans-serif"))

    ax.text(0.5, 0.08,
            "All four failure modes added to MEMORY.md as recurring Kaggle\n"
            "audio-CNN gotchas to avoid in nb20c / nb21 / nb22 onwards.",
            ha="center", va="top", fontsize=10, color="#555", style="italic",
            linespacing=1.5)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_fold_table(pdf):
    fig, ax = plt.subplots(figsize=PAGE)
    fig.suptitle("nb20a 5-fold OOF AUC (split-run training)",
                 fontsize=20, fontweight="bold", y=0.97)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    headers = ["Fold", "OOF AUC", "Best epoch", "Source"]
    col_xs  = [0.20, 0.36, 0.52, 0.70]
    col_ws  = [0.16, 0.16, 0.16, 0.20]
    rows = [
        (str(fi), f"{auc:.4f}", str(be) if be else "-",
         "run 1" if fi <= 1 else "run 2")
        for fi, auc, be in FOLD_AUC
    ]
    rows.append(("Mean", f"{np.mean([a for _, a, _ in FOLD_AUC]):.4f}",
                  "", "5-fold"))

    top   = 0.72
    row_h = 0.07
    for cx, w, h in zip(col_xs, col_ws, headers):
        ax.add_patch(mpatches.Rectangle((cx, top), w, row_h,
                                         facecolor="#225588", edgecolor="#fff"))
        ax.text(cx + w/2, top + row_h/2, h, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
    for i, r in enumerate(rows):
        y = top - (i + 1) * row_h
        is_mean = (r[0] == "Mean")
        fc = "#fff3cc" if is_mean else ("#f9f9f9" if i % 2 == 0 else "#eef4fc")
        for cx, w, text in zip(col_xs, col_ws, r):
            ax.add_patch(mpatches.Rectangle((cx, y), w, row_h, facecolor=fc,
                                             edgecolor="#ccc", linewidth=0.5))
            ax.text(cx + w/2, y + row_h/2, text, ha="center", va="center",
                    fontsize=11,
                    fontweight=("bold" if is_mean else "normal"))

    ax.text(0.5, 0.18,
            "Run 1 (alexycactus/birdclef-2026-cnn-sed-b0-run1): trained folds 0+1\n"
            "  in ~6.4 h on T4, saved fold checkpoints + val_preds.\n\n"
            "Run 2 (alexycactus/birdclef-2026-cnn-sed-b0-run2): mounted run 1 via\n"
            "  kernel_sources, trained folds 2+3+4 in ~9.7 h, ran final 5-fold\n"
            "  ensemble inference on test_soundscapes (staging fallback).\n\n"
            "Inference kernel (alexycactus/birdclef-2026-cnn-infer):\n"
            "  internet OFF, 12.2 s for 5-fold ensemble + top-K postproc.\n"
            "  Ready to submit for LB.",
            ha="center", va="top", fontsize=10, color="#444",
            family="monospace", linespacing=1.5,
            transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_roadmap(pdf):
    fig, ax = plt.subplots(figsize=PAGE)
    fig.suptitle("Roadmap -- what's next (sorted by expected LB lever)",
                 fontsize=20, fontweight="bold", y=0.97)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    items = [
        ("nb20-infer",  "Submit 5-fold CNN ensemble for LB",
         "ready now",          "+0 to +0.03 vs Perch+MLP"),
        ("nb21",         "Perch+MLP x CNN ensemble (rank avg or weighted)",
         "needs nb20 LB",      "+0.010 to +0.025"),
        ("nb20c",        "Noisy Student round 1 on the CNN",
         "biggest single lever","+0.020 to +0.040 (1st place 2025 pattern)"),
        ("nb20c x 4",    "Noisy Student rounds 2-4 (multi-iterative)",
         "after round 1",      "cumulative +0.03 (1st: 0.909->0.930)"),
        ("nb20d",        "Backbone diversification (EffNetV2-S, NFNet-L0)",
         "for ensemble div",   "+0.005 to +0.015 in ensemble"),
        ("nb22",         "Top-K post-processing variants",
         "fast iteration",     "+0.005 (already applied in nb20-infer)"),
        ("nb25 (deferred)", "Xeno-Canto pretraining",
         "high setup cost",    "+0.025 from 2nd place 2025"),
        ("nb27",         "Specialised Amphibia/Insecta model",
         "small but quick",    "+0.002 to +0.003"),
    ]
    headers = ["Notebook", "What", "Unblocked by", "Expected LB"]
    col_xs  = [0.04, 0.20, 0.55, 0.80]
    col_ws  = [0.16, 0.35, 0.25, 0.18]

    top   = 0.86; row_h = 0.085
    for cx, w, h in zip(col_xs, col_ws, headers):
        ax.add_patch(mpatches.Rectangle((cx, top), w, row_h,
                                         facecolor="#225588", edgecolor="#fff"))
        ax.text(cx + w/2, top + row_h/2, h, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
    for i, r in enumerate(items):
        y = top - (i + 1) * row_h
        fc = "#f9f9f9" if i % 2 == 0 else "#eef4fc"
        for cx, w, text in zip(col_xs, col_ws, r):
            ax.add_patch(mpatches.Rectangle((cx, y), w, row_h, facecolor=fc,
                                             edgecolor="#ccc", linewidth=0.5))
            ax.text(cx + 0.005, y + row_h/2, text, ha="left", va="center",
                    fontsize=9,
                    fontweight=("bold" if cx == col_xs[0] else "normal"),
                    family=("monospace" if cx == col_xs[0] else "sans-serif"))

    ax.text(0.5, 0.10,
            "Full backlog: docs/plans/2026-05-14-post-nb20-experiments-backlog.md",
            ha="center", va="top", fontsize=9, color="#555", style="italic")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# --- Main --------------------------------------------------------------------
def main():
    with PdfPages(OUT_PDF) as pdf:
        page_cover(pdf)
        page_dataset(pdf)
        page_lb_progression(pdf)
        page_perch_arch(pdf)
        page_alpha_sweep(pdf)
        page_cnn_arch(pdf)
        page_training_curves(pdf)
        page_fold_table(pdf)
        page_debug_chain(pdf)
        page_roadmap(pdf)
    print(f"Wrote {OUT_PDF}  ({OUT_PDF.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
