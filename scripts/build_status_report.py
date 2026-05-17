"""Build a multi-page PDF status report for the BirdCLEF+ 2026 campaign.

Pulls existing figures from docs/figures/, parses run1/run2 logs to make
training-curve plots, and composes everything plus tables into a single
docs/reports/birdclef-2026-status.pdf.

Run:  python scripts/build_status_report.py
"""
from __future__ import annotations

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

PAGE = (11.0, 8.5)   # landscape letter

# --- Static project state ----------------------------------------------------
LB_HISTORY = [
    ("nb13",   0.839, "Perch MLP baseline",         "#4477aa"),
    ("nb14a",  0.875, "BiGRU Replace",              "#228833"),
    ("nb14c",  0.879, "BiGRU Augment",              "#228833"),
    ("nb15a",  0.883, "Per-class logit",            "#228833"),
    ("nb16b",  0.817, "PL t=0.8 (failed)",          "#cc3311"),
    ("nb16c",  0.816, "PL t=0.9 (failed)",          "#cc3311"),
    ("nb17b",  0.892, "Alpha sweep a=0.6",          "#228833"),
    ("nb18a",  0.852, "First CNN baseline",         "#4477aa"),
    ("nb19a",  0.897, "Top-K postproc",             "#228833"),
    ("nb20b",  None,  "CNN 5-fold (pending)",       "#dddd66"),
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
    ("v1",    "Slug 409 conflict",         "title '0.5' -> slug '0-5' didn't match id"),
    ("v2",    "BCELoss autocast-unsafe",   "swap to BCEWithLogitsLoss"),
    ("v3",    "CUDA assert input in [0,1]","NaN slips past .clamp(); add nan_to_num"),
    ("v4",    "Collate non-resizable",     "np.tile slice is a view; np.array(copy=True)"),
    ("v5 ERR","All train=nan from ep 1",   "fp16 FFT n_fft=4096 -> autocast(enabled=False)"),
    ("v5 OK", "Healthy ep1 val_auc=0.854", "fix verified; split-run launched"),
]

ROADMAP_ITEMS = [
    ("nb20-infer", "Submit 5-fold CNN ensemble for LB",
     "ready now",          "+0 / +0.03 vs Perch+MLP"),
    ("nb21",       "Perch+MLP x CNN ensemble (rank avg or weighted)",
     "needs nb20 LB",      "+0.010 / +0.025"),
    ("nb20c",      "Noisy Student round 1 on the CNN",
     "biggest lever",      "+0.020 / +0.040"),
    ("nb20c x 4",  "Noisy Student rounds 2-4 (multi-iterative)",
     "after round 1",      "cumulative +0.03"),
    ("nb20d",      "Backbone diversification (EffNetV2-S, NFNet-L0)",
     "for ensemble div",   "+0.005 / +0.015 in ensemble"),
    ("nb22",       "Top-K post-processing variants",
     "fast iteration",     "+0.005 (already in nb20-infer)"),
    ("nb25 (def.)","Xeno-Canto pretraining",
     "high setup cost",    "+0.025 (2nd place 2025)"),
    ("nb27",       "Specialised Amphibia/Insecta model",
     "small but quick",    "+0.002 / +0.003"),
]


def parse_log_for_epochs(log_path: Path):
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = re.findall(r'"data":"([^"]*)"', text)
    folds = {}; current_fold = None
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


# --- Helpers -----------------------------------------------------------------
def draw_table(ax, headers, rows, col_widths=None, header_color="#225588",
               row_alt_colors=("#f9f9f9", "#eef4fc"), highlight_last=False,
               font_size=10, header_font_size=11, row_height=None,
               first_col_mono=False, top=0.92, left=0.05, right=0.95):
    """Draw a clean table on an axes with set_xlim/ylim 0..1."""
    ncols = len(headers)
    if col_widths is None:
        col_widths = [(right - left) / ncols] * ncols
    n_rows  = len(rows) + 1   # +1 for header
    avail_h = top - 0.08
    row_h   = row_height if row_height else min(0.08, avail_h / n_rows)

    cum = [left] + list(np.cumsum(col_widths) + left)

    # Header
    y = top - row_h
    for ci, header in enumerate(headers):
        ax.add_patch(mpatches.Rectangle((cum[ci], y), col_widths[ci], row_h,
                                         facecolor=header_color, edgecolor="white"))
        ax.text(cum[ci] + col_widths[ci]/2, y + row_h/2, header,
                ha="center", va="center", color="white",
                fontsize=header_font_size, fontweight="bold")
    # Rows
    for ri, row in enumerate(rows):
        y = top - (ri + 2) * row_h
        is_last = highlight_last and (ri == len(rows) - 1)
        fc = "#fff3cc" if is_last else row_alt_colors[ri % 2]
        for ci, val in enumerate(row):
            ax.add_patch(mpatches.Rectangle((cum[ci], y), col_widths[ci], row_h,
                                             facecolor=fc, edgecolor="#ccc",
                                             linewidth=0.5))
            # Left-align text with padding except numeric columns (centred)
            try:
                float(str(val).replace("+", "").replace("-", ""))
                ha = "center"
                tx = cum[ci] + col_widths[ci]/2
            except (ValueError, TypeError):
                ha = "left"
                tx = cum[ci] + 0.01
            mono = (first_col_mono and ci == 0)
            ax.text(tx, y + row_h/2, str(val), ha=ha, va="center",
                    fontsize=font_size,
                    fontweight=("bold" if is_last or (first_col_mono and ci == 0) else "normal"),
                    family=("monospace" if mono else "sans-serif"))


# --- Page builders -----------------------------------------------------------
def page_cover(pdf):
    fig, ax = plt.subplots(figsize=PAGE)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.86, "BirdCLEF+ 2026", ha="center", fontsize=42,
            fontweight="bold")
    ax.text(0.5, 0.79, "Status Report", ha="center", fontsize=22,
            color="#555", style="italic")
    ax.text(0.5, 0.71,
            "Wildlife species identification from passive acoustic monitoring\n"
            "in the Pantanal wetlands (Brazil).  Kaggle code competition.\n"
            "Metric: macro ROC-AUC. Deadline: June 3, 2026.",
            ha="center", fontsize=11, color="#444", linespacing=1.6)

    # KPI box -- give it more room
    bx, by, bw, bh = 0.10, 0.34, 0.80, 0.28
    ax.add_patch(mpatches.FancyBboxPatch((bx, by), bw, bh,
                                          boxstyle="round,pad=0.02",
                                          facecolor="#f4f8fc",
                                          edgecolor="#225588", linewidth=1.4))
    ax.text(0.5, by + bh - 0.04, "Current state", ha="center",
            fontsize=14, fontweight="bold", color="#225588")
    ax.text(0.5, by + bh - 0.09,
            "Best LB: 0.897   (nb19a:  Perch+MLP + top-K postproc)",
            ha="center", fontsize=12)
    ax.text(0.5, by + bh - 0.13,
            "Gap to 2025 winning solution (0.96):  0.063",
            ha="center", fontsize=11, color="#aa3377")
    ax.text(0.5, by + bh - 0.18,
            "CNN track:  5-fold ensemble trained,  OOF macro-AUC 0.9792",
            ha="center", fontsize=11, color="#228833")
    ax.text(0.5, by + bh - 0.22,
            "(real LB pending submission; OOF likely overstates by 0.05-0.10)",
            ha="center", fontsize=9, color="#888", style="italic")

    ax.text(0.5, 0.18, "Author:  Alexy Louis",
            ha="center", fontsize=10, color="#666")
    ax.text(0.5, 0.14, "Generated:  2026-05-17",
            ha="center", fontsize=9, color="#888")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_dataset(pdf):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle("Dataset overview", fontsize=20, fontweight="bold", y=0.97)

    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.20,
                          left=0.05, right=0.97, top=0.91, bottom=0.05)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.text(0.0, 0.98,
            "BirdCLEF+ 2026  --  key numbers\n"
            "------------------------------\n\n"
            "  Total training clips    35,549\n"
            "  Species (multi-taxa)       234\n"
            "    Aves   (birds)           162\n"
            "    Amphibia                  35\n"
            "    Insecta                   28  (+25 sonotypes)\n"
            "    Mammalia                   8\n"
            "    Reptilia                   1   (Caiman, 1 clip)\n\n"
            "  train_audio (XC + iNat)  35,549\n"
            "  train_soundscapes        10,658\n"
            "  Labeled segments          1,478   (66 files)\n\n"
            "  Region: Pantanal, Brazil\n"
            "  Lat -21.6 / -16.5,  Lon -57.6 / -55.9\n"
            "  Format: OGG, 32 kHz",
            ha="left", va="top", fontsize=9.5, family="monospace",
            transform=ax.transAxes, linespacing=1.5)

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
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.20)
    fig.suptitle("LB progression  --  two tracks, ten notebooks",
                 fontsize=20, fontweight="bold", y=0.97)

    x = np.arange(len(LB_HISTORY))
    for i, (nm, lb, lbl, col) in enumerate(LB_HISTORY):
        if lb is None:
            ax.bar(i, 0.50, color=col, alpha=0.35, edgecolor="#888",
                   linewidth=0.7, hatch="//")
            ax.text(i, 0.27, "pending", ha="center", va="center",
                    fontsize=9, color="#777", fontweight="bold",
                    rotation=0)
        else:
            ax.bar(i, lb, color=col, edgecolor="#222", linewidth=0.7, alpha=0.9)
            ax.text(i, lb + 0.005, f"{lb:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.axhline(WINNING_LB, color="#aa3377", linestyle="--", linewidth=1.2,
               alpha=0.7)
    ax.text(0.5, WINNING_LB - 0.007,
            f"2025 winning solution  {WINNING_LB:.2f}", ha="left",
            va="top", color="#aa3377", fontsize=10, style="italic")

    # Notebook labels on x axis, description below using xticklabels with line break
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{nm}\n{lbl}" for nm, _, lbl, _ in LB_HISTORY],
        fontsize=8.5, color="#333"
    )
    ax.set_ylabel("LB AUC (macro)")
    ax.set_ylim(0.20, 1.00)
    ax.set_yticks([0.3, 0.5, 0.7, 0.85, 0.90, 0.95, 1.00])
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    handles = [
        mpatches.Patch(color="#4477aa", label="Baseline / new architecture"),
        mpatches.Patch(color="#228833", label="Improvement on track"),
        mpatches.Patch(color="#cc3311", label="Failure (regression)"),
        mpatches.Patch(color="#dddd66", label="Pending LB submission"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    pdf.savefig(fig); plt.close(fig)


def page_image(pdf, title, image_path, caption=""):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.96)
    ax = fig.add_axes([0.05, 0.12, 0.90, 0.78])
    ax.axis("off")
    if image_path.exists():
        ax.imshow(mpimg.imread(image_path)); ax.set_aspect("auto")
    else:
        ax.text(0.5, 0.5, f"[missing figure: {image_path.name}]",
                ha="center", va="center", color="red")
    if caption:
        fig.text(0.5, 0.06, caption, ha="center", va="bottom",
                 fontsize=10, color="#444", style="italic", wrap=True)
    pdf.savefig(fig); plt.close(fig)


def page_cnn_arch(pdf):
    """Clean vertical-flow architecture diagram, no overlap."""
    fig, ax = plt.subplots(figsize=PAGE)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.04)
    fig.suptitle("CNN+SED architecture  (nb20)",
                 fontsize=20, fontweight="bold", y=0.97)
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 10)

    def box(x, y, w, h, text, fc="#e8f0fc", ec="#225588",
            fs=10, fw="normal"):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                        facecolor=fc, edgecolor=ec,
                                        linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight=fw)

    def arrow(x1, y1, x2, y2, c="#444", lw=1.6):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=c, lw=lw))

    cx = 4.8           # central column for the main pipeline (width 3.4)
    cw = 3.4

    # Row 1: raw audio
    box(cx, 9.0, cw, 0.7, "20 s audio chunk @ 32 kHz",
        fc="#fff3e0", ec="#bb6611")
    # Row 2: mel
    box(cx, 7.7, cw, 0.9,
        "Mel spectrogram   (fp32, autocast disabled)\n"
        "n_mels=224,  n_fft=4096,  hop=1252",
        fc="#e8f0fc", ec="#225588", fs=9.5)
    arrow(cx + cw/2, 9.0, cx + cw/2, 8.6)
    # Row 3: 3-channel repeat
    box(cx, 6.7, cw, 0.6, "repeat 3x  -->  (3, 224, 512)",
        fc="#e8f0fc", ec="#225588", fs=10)
    arrow(cx + cw/2, 7.7, cx + cw/2, 7.3)
    # Row 4: backbone
    box(cx - 0.5, 5.0, cw + 1.0, 1.3,
        "EfficientNet-B0   (timm, ImageNet pretrain)\n"
        "AMP fp16,  AdamW,  cosine LR  5e-4 -> 1e-6\n"
        "20 epochs x 5 folds  (split across two kernels)",
        fc="#fce8f0", ec="#aa3377", fw="bold", fs=10)
    arrow(cx + cw/2, 6.7, cx + cw/2, 6.3)
    # Row 5: freq pool
    box(cx, 4.0, cw, 0.55, "Frequency pool  (mean over height)",
        fc="#eef7ee", ec="#228833", fs=10)
    arrow(cx + cw/2, 5.0, cx + cw/2, 4.55)
    # Row 6: SED head
    box(cx - 0.5, 2.5, cw + 1.0, 1.2,
        "SED head  (attention pooling over time)\n"
        "frame_lg = fc_cla(feat),   att = softmax(tanh(fc_att(feat)))\n"
        "clip_lg = sum(att * frame_lg, dim=time)",
        fc="#eef7ee", ec="#228833", fs=9, fw="bold")
    arrow(cx + cw/2, 4.0, cx + cw/2, 3.7)
    # Row 7: clip logits
    box(cx, 1.2, cw, 0.6, "clip logits  (B, 234)",
        fc="#fff5d6", ec="#bb9933", fw="bold", fs=10)
    arrow(cx + cw/2, 2.5, cx + cw/2, 1.8)
    # Row 8: loss
    box(cx, 0.15, cw, 0.6, "BCEWithLogitsLoss   (secondary labels = 1)",
        fc="#fbeeee", ec="#cc3311", fw="bold", fs=9.5)
    arrow(cx + cw/2, 1.2, cx + cw/2, 0.75)

    # Left-side: MixUp annotation
    ax.text(0.7, 9.0,
            "Raw-audio MixUp\n(p=0.5, weight=0.5,\n label union)",
            fontsize=9, color="#aa3377", style="italic", linespacing=1.4,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff",
                      edgecolor="#aa3377"))

    # Right-side: inference summary
    ax.text(9.5, 9.0,
            "Inference  (nb20b)\n"
            "------------------\n"
            "60 s soundscape ->\n"
            "9 x 20 s windows\n"
            "with 5 s stride.\n"
            "5-fold ensemble.\n"
            "framewise overlap\n"
            "averaging into\n"
            "12 x 5 s output\n"
            "slots.\n\n"
            "+ Top-K postproc\n"
            "  (per-file).\n\n"
            "Internet OFF\n"
            "(timm pretrained\n"
            " = False)",
            fontsize=8.5, family="monospace", color="#225588",
            ha="left", va="top", linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f8fc",
                      edgecolor="#225588"))
    pdf.savefig(fig); plt.close(fig)


def page_training_curves(pdf):
    run1_log = RESULTS / "nb20a_run1_done" / "birdclef-2026-cnn-sed-b0-run1.log"
    run2_log = RESULTS / "nb20a_run2_done" / "birdclef-2026-cnn-sed-b0-run2.log"
    folds = parse_log_for_epochs(run1_log)
    folds.update(parse_log_for_epochs(run2_log))

    fig, ax = plt.subplots(figsize=PAGE)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.18)
    fig.suptitle("nb20a training curves  --  5 folds, 20 epochs each",
                 fontsize=20, fontweight="bold", y=0.97)

    colors = ["#4477aa", "#228833", "#aa3377", "#bb6611", "#dd5577"]
    for fi in sorted(folds.keys()):
        rows = folds[fi]
        if not rows: continue
        eps  = [r[0] for r in rows]
        aucs = [r[2] for r in rows]
        ax.plot(eps, aucs, "o-", color=colors[fi % 5], linewidth=1.8,
                markersize=5, label=f"fold {fi}   (best AUC = {max(aucs):.4f})")

    ax.axhline(0.97, color="#aaa", linestyle=":", linewidth=0.8)
    ax.text(0.7, 0.969, "0.97 reference", color="#888", fontsize=8.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val AUC (macro)")
    ax.set_xlim(0.5, 20.5); ax.set_ylim(0.83, 0.99)
    ax.set_xticks(range(1, 21))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    fig.text(0.5, 0.06,
             "All 5 folds clear val_auc 0.97 by ~epoch 5 and plateau near 0.978-0.981.\n"
             "Tight spread (0.0044) signals a stable ensemble;  OOF mean = 0.9792.\n"
             "Caveat: StratifiedKFold-by-primary_label likely overstates the LB.",
             ha="center", va="bottom", fontsize=10, color="#444",
             style="italic", linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)


def page_fold_table(pdf):
    fig, ax = plt.subplots(figsize=PAGE)
    fig.suptitle("nb20a  --  5-fold OOF AUC  (split-run training)",
                 fontsize=20, fontweight="bold", y=0.97)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    headers = ["Fold", "OOF AUC", "Best epoch", "Source"]
    col_widths = [0.12, 0.16, 0.16, 0.16]
    rows = [(str(fi), f"{auc:.4f}", str(be) if be else "-",
             "run 1" if fi <= 1 else "run 2")
            for fi, auc, be in FOLD_AUC]
    rows.append(("Mean",
                  f"{np.mean([a for _, a, _ in FOLD_AUC]):.4f}",
                  "-", "5-fold"))

    draw_table(ax, headers, rows, col_widths=col_widths,
               highlight_last=True, font_size=11, row_height=0.07,
               top=0.85, left=0.20)

    fig.text(0.5, 0.20,
             "Run 1  (alexycactus/birdclef-2026-cnn-sed-b0-run1):\n"
             "    trained folds 0+1 in ~6.4 h on T4,\n"
             "    saved fold checkpoints + val_preds to /kaggle/working/.\n\n"
             "Run 2  (alexycactus/birdclef-2026-cnn-sed-b0-run2):\n"
             "    mounted run 1 via kernel_sources, trained folds 2+3+4 in ~9.7 h,\n"
             "    ran final 5-fold ensemble inference (staging fallback).\n\n"
             "Inference kernel  (alexycactus/birdclef-2026-cnn-infer):\n"
             "    internet OFF, 12.2 s for 5-fold ensemble + top-K postproc.\n"
             "    Ready to submit for LB.",
             ha="center", va="bottom", fontsize=9.5, color="#444",
             family="monospace", linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)


def page_debug_chain(pdf):
    fig, ax = plt.subplots(figsize=PAGE)
    fig.suptitle("nb20a debug chain  --  4 failures before clean training",
                 fontsize=20, fontweight="bold", y=0.97)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    headers = ["Version", "Failure mode", "Fix"]
    col_widths = [0.13, 0.34, 0.43]
    draw_table(ax, headers, DEBUG_CHAIN, col_widths=col_widths,
               font_size=10, header_font_size=11, row_height=0.085,
               top=0.85, left=0.05, first_col_mono=True)

    fig.text(0.5, 0.10,
             "All four failure modes added to MEMORY.md as recurring Kaggle\n"
             "audio-CNN gotchas to apply in nb20c / nb21 / nb22 onwards.",
             ha="center", va="bottom", fontsize=10, color="#555",
             style="italic", linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)


def page_roadmap(pdf):
    fig, ax = plt.subplots(figsize=PAGE)
    fig.suptitle("Roadmap  --  next experiments, sorted by expected LB lever",
                 fontsize=20, fontweight="bold", y=0.97)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    headers = ["Notebook", "What", "Unblocked by", "Expected LB"]
    col_widths = [0.15, 0.42, 0.18, 0.20]
    draw_table(ax, headers, ROADMAP_ITEMS, col_widths=col_widths,
               font_size=9, header_font_size=10.5, row_height=0.07,
               top=0.86, left=0.025, first_col_mono=True)

    fig.text(0.5, 0.10,
             "Full backlog:  docs/plans/2026-05-14-post-nb20-experiments-backlog.md",
             ha="center", va="bottom", fontsize=9, color="#555",
             style="italic")
    pdf.savefig(fig); plt.close(fig)


# --- Main --------------------------------------------------------------------
def main():
    with PdfPages(OUT_PDF) as pdf:
        page_cover(pdf)
        page_dataset(pdf)
        page_lb_progression(pdf)
        page_image(pdf,
                   "Perch+MLP architecture   (LB 0.892 single, 0.897 with top-K)",
                   FIG_DIR / "architecture_perch_mlp.png",
                   caption=("Frozen Perch v2 features feed 234 per-class probes via "
                            "torch.bmm;  alpha=0.6 blends the MLP correction with "
                            "Perch's sigmoid at inference time."))
        page_image(pdf,
                   "Alpha sweep  --  the architectural reframe of nb17",
                   FIG_DIR / "alpha_sweep.png",
                   caption=("Lower alpha -> better LB.  Perch's logit carries more signal "
                            "than the MLP probes for mapped species, so the MLP is a "
                            "correction layer on top of Perch, not an independent predictor."))
        page_cnn_arch(pdf)
        page_training_curves(pdf)
        page_fold_table(pdf)
        page_debug_chain(pdf)
        page_roadmap(pdf)
    print(f"Wrote {OUT_PDF}  ({OUT_PDF.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
