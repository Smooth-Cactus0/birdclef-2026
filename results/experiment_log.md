# Experiment Log

## LB Progress

| Notebook | LB AUC | Delta |
|---|---|---|
| nb13 Perch MLP baseline | 0.839 | — |
| nb14a BiGRU Replace | 0.875 | +0.036 |
| nb14c BiGRU Augment | 0.879 | +0.004 |
| nb15a Per-class logit + blend ON (ALPHA=0.7) | 0.883 | +0.004 |
| nb15c Global logit ctx + blend ON | 0.864 | -0.015 |
| nb15f Both logit + blend OFF | 0.502 | -0.377 |
| nb17a Alpha sweep 0.5 | **0.892** | +0.009 (tied best) |
| nb17b Alpha sweep 0.6 | **0.892** | +0.009 (tied best) |
| nb17c Alpha sweep 0.8 | 0.878 | -0.005 |
| nb17d Alpha sweep 0.9 | 0.867 | -0.016 |

## Key Lessons

- **OOF on 66 labeled soundscapes is an unreliable proxy for LB.** nb15c had OOF +0.034 vs nb14c but LB -0.015. At 792 windows, the OOF sample is too small to trust.
- **Alpha-blend (mapped species) is essential AND tunable.** Removing it drops LB by ~0.38 (0.883 -> 0.502). The default 0.7 is sub-optimal: sweeping showed a clear monotonic gradient with a plateau at ALPHA in [0.5, 0.6] giving LB 0.892 (+0.009). Perch's logit is more trustworthy than the MLP's; the 0.7 default under-weighted it.
- **Per-class learnable alpha is justified.** Spread across 4 sweep values was 0.025 (0.867 -> 0.892), well above the 0.01 threshold from the plan. Different species likely have different optimal MLP-vs-Perch trust ratios.
- **Global logit PCA overfits.** Compressing the 234-dim Perch logit into 32 PCA dims adds a shared context that the MLP memorizes on 792 train windows but doesn't generalise to test soundscapes.
- **Per-class logit scalar (nb15a) reliably helps (+0.004 LB).** The MLP learns to weight Perch's class-specific score conditionally, better than the fixed alpha-blend alone.
- **Self-attention (nb14b/d) needs more data.** With only 66 files × 12 windows, attention weights were uniform - no structure learned. BiGRU inductive bias (recurrence) works at this scale.

## Strategic position (2026-05-11, post-nb17)

- **Current best LB**: 0.892 (nb17a α=0.5 or nb17b α=0.6, tied)
- **Winning solution LB**: 0.96 → **gap = 0.068**
- **Architectural reframe from nb17**: MLP is a *correction layer* on Perch, not an independent predictor. Monotonic alpha gradient (lower α → better) means Perch's logit carries more signal than the MLP for mapped species. The MLP only adds value on top of Perch, not as a replacement.
- **nb16 pseudo-labeling FAILED** (LB 0.817 at threshold 0.8, 0.816 at 0.9 → -0.066 vs nb15a). Systematic bias, not noise — even high-confidence PL crashed the model. Naive PL on `unlabeled_soundscapes/` is a dead end.
- **Levers still alive**:
  - **Calibration**: nb19 per-class learnable α — justified by 0.025 spread in sweep. Expected: +0.005 to +0.015.
  - **CNN parallel track**: nb18 (brief ready). Historically the +0.03 to +0.05 lever.
  - **Soundscape addition** via Perch logits directly as weak labels (different mechanism than failed nb16).
  - **Backbone diversification**: BirdNET embeddings as alternative to Perch.

### Next move priority (updated post-nb17, post-nb16-failure)

1. **Free win**: resubmit nb17b (α=0.6) as the new baseline → +0.009 LB locked in
2. **nb18 CNN baseline** (brief ready, parallel session) — highest expected gain
3. **nb19 per-class learnable α** — small but justified, builds on nb17 insight
4. **Soundscape addition** (Perch logits as weak labels) — separate from failed nb16 mechanism
5. **Defer**: BirdNET backbone, Xeno-canto external data — only if nb18 ceiling is hit

---

| ID | Date | Model | Backbone | PL round | CV AUC | LB AUC | Notes |
|---|---|---|---|---|---|---|---|
| exp001 | 2026-03-12 | baseline | efficientnet_b0 | 0 | TBD | TBD | 10ep, CrossEntropy, no quality filter, 5-fold StratifiedKFold |
| exp002 | 2026-04-20 | bird pipeline | efficientnet_b3 | 0 | 0.7236 | TBD | 10ep, BCE multi-label, GroupKFold by site, 10s train / 5s inf, sample weights; folds: 0.7342/0.7191/0.7212/0.7200/0.7233; kernel: birdclef-2026-backbone-search-bird-pipeline |
| exp003 | 2026-04-20 | non-bird pipeline | eca_nfnet_l0 | 0 | 0.748 ±0.075 | TBD | 25ep, Focal BCE gamma=2, 5-fold, gold segments 3x oversample; kernel: birdclef-2026-nonbird-pipeline |
| exp004 | 2026-05-07 | Perch v2 ONNX + MLP probes | google/perch_v2_cpu | 0 | 0.64 (per-fold avg) | **0.839** | 66 labeled soundscapes → 792 windows; PCA(64)+5 temporal feats → 69-dim; vectorized bmm MLP 234 probes; 5-fold GroupKFold by file; alpha-blend 0.7 MLP + 0.3 Perch sigmoid; class-conditional alpha for unmapped species; kernel: birdclef-2026-perch-mlp-baseline v2 |
| exp005 | 2026-05-07 | BiGRU Replace | google/perch_v2_cpu | 0 | 0.5222 (OOF) | **0.875** | BiGRU(hidden=32) ctx over 5 windows replaces 5 temporal scalars; MLP_IN=72; kernel: birdclef-2026-bigru-replace |
| exp006 | 2026-05-07 | BiGRU Augment | google/perch_v2_cpu | 0 | 0.5049 (OOF) | **0.879** | BiGRU(hidden=32) ctx appended to scalars → MLP_IN=77 (PCA64+scalars5+ctx8); current best single model; kernel: birdclef-2026-bigru-augment |
| exp007 | 2026-05-08 | Per-class logit + blend ON | google/perch_v2_cpu | 0 | 0.5045 (OOF) | **0.883** | scores_tr[:,i] scalar appended per probe → MLP_IN=78; alpha-blend 0.7/0.3 retained; best LB to date; kernel: birdclef-2026-logit-class-blend |
| exp008 | 2026-05-08 | Global logit ctx + blend ON | google/perch_v2_cpu | 0 | 0.5386 (OOF v1) | **0.864** | PCA(32) of 234-dim projected logit → shared ctx → MLP_IN=109; OOF +0.034 but LB -0.015 vs nb14c; global PCA overfit to 792-window train set; NaN guard + seed=42 added in v3; kernel: birdclef-2026-logit-global-blend v3 |
| exp009 | 2026-05-08 | Both logit features + blend OFF | google/perch_v2_cpu | 0 | 0.5121 (OOF) | **0.502** | per-class scalar + global PCA-32 → MLP_IN=110; blend OFF → raw MLP outputs are miscalibrated; confirms alpha-blend is essential; kernel: birdclef-2026-logit-both-noblend |
| exp010 | 2026-05-11 | Alpha sweep 0.5 (nb17a) | google/perch_v2_cpu | 0 | n/a (no retrain) | **0.892** | identical pipeline to nb15a, only ALPHA changed 0.7→0.5; tied best with nb17b; kernel: birdclef-2026-alpha-sweep-05 |
| exp011 | 2026-05-11 | Alpha sweep 0.6 (nb17b) | google/perch_v2_cpu | 0 | n/a (no retrain) | **0.892** | ALPHA=0.6; tied best with nb17a; safer interior of [0.5,0.6] plateau → promote to nb15a-v2; kernel: birdclef-2026-alpha-sweep-06 |
| exp012 | 2026-05-11 | Alpha sweep 0.8 (nb17c) | google/perch_v2_cpu | 0 | n/a (no retrain) | 0.878 | ALPHA=0.8; -0.005 vs nb15a baseline; MLP-heavier blend hurts; kernel: birdclef-2026-alpha-sweep-08 |
| exp013 | 2026-05-11 | Alpha sweep 0.9 (nb17d) | google/perch_v2_cpu | 0 | n/a (no retrain) | 0.867 | ALPHA=0.9; -0.016 vs baseline; near-MLP-only is clearly worse than blended; confirms Perch sigmoid carries more signal than the MLP probes for mapped species; kernel: birdclef-2026-alpha-sweep-09 |
