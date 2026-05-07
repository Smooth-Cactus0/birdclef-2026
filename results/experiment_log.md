# Experiment Log

| ID | Date | Model | Backbone | PL round | CV AUC | LB AUC | Notes |
|---|---|---|---|---|---|---|---|
| exp001 | 2026-03-12 | baseline | efficientnet_b0 | 0 | TBD | TBD | 10ep, CrossEntropy, no quality filter, 5-fold StratifiedKFold |
| exp002 | 2026-04-20 | bird pipeline | efficientnet_b3 | 0 | 0.7236 | TBD | 10ep, BCE multi-label, GroupKFold by site, 10s train / 5s inf, sample weights; folds: 0.7342/0.7191/0.7212/0.7200/0.7233; kernel: birdclef-2026-backbone-search-bird-pipeline |
| exp003 | 2026-04-20 | non-bird pipeline | eca_nfnet_l0 | 0 | 0.748 ±0.075 | TBD | 25ep, Focal BCE gamma=2, 5-fold, gold segments 3x oversample; kernel: birdclef-2026-nonbird-pipeline |
| exp004 | 2026-05-07 | Perch v2 ONNX + MLP probes | google/perch_v2_cpu | 0 | 0.64 (per-fold avg) | **0.839** | 66 labeled soundscapes → 792 windows; PCA(64)+5 temporal feats → 69-dim; vectorized bmm MLP 234 probes; 5-fold GroupKFold by file; alpha-blend 0.7 MLP + 0.3 Perch sigmoid; class-conditional alpha for unmapped species; kernel: birdclef-2026-perch-mlp-baseline v2 |
