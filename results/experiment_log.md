# Experiment Log

## LB Progress

| Notebook | LB AUC | Delta |
|---|---|---|
| nb13 Perch MLP baseline | 0.839 | — |
| nb14a BiGRU Replace | 0.875 | +0.036 |
| nb14c BiGRU Augment | 0.879 | +0.004 |
| nb15a Per-class logit + blend ON | **0.883** | +0.004 ← **best** |
| nb15c Global logit ctx + blend ON | 0.864 | −0.015 |
| nb15f Both logit + blend OFF | 0.502 | −0.377 |

## Key Lessons

- **OOF on 66 labeled soundscapes is an unreliable proxy for LB.** nb15c had OOF +0.034 vs nb14c but LB −0.015. At 792 windows, the OOF sample is too small to trust.
- **Alpha-blend (0.7 MLP + 0.3 Perch sigmoid) is essential.** Removing it drops LB by ~0.38 (0.883 → 0.502). The MLP outputs are not calibrated without it.
- **Global logit PCA overfits.** Compressing the 234-dim Perch logit into 32 PCA dims adds a shared context that the MLP memorizes on 792 train windows but doesn't generalise to test soundscapes.
- **Per-class logit scalar (nb15a) reliably helps (+0.004 LB).** The MLP learns to weight Perch's class-specific score conditionally, better than the fixed alpha-blend alone.
- **Self-attention (nb14b/d) needs more data.** With only 66 files × 12 windows, attention weights were uniform — no structure learned. BiGRU inductive bias (recurrence) works at this scale.

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
