# Experiment Log

| ID | Date | Model | Backbone | PL round | CV AUC | LB AUC | Notes |
|---|---|---|---|---|---|---|---|
| exp001 | 2026-03-12 | baseline | efficientnet_b0 | 0 | TBD | TBD | 10ep, CrossEntropy, no quality filter, 5-fold StratifiedKFold |
| exp002 | 2026-04-20 | bird pipeline | efficientnet_b3 | 0 | 0.7236 | TBD | 10ep, BCE multi-label, GroupKFold by site, 10s train / 5s inf, sample weights; folds 1-4 avg (fold5 pending); kernel: birdclef-2026-backbone-search-bird-pipeline |
| exp003 | 2026-04-20 | non-bird pipeline | eca_nfnet_l0 | 0 | 0.748 ±0.075 | TBD | 25ep, Focal BCE gamma=2, 5-fold, gold segments 3x oversample; kernel: birdclef-2026-nonbird-pipeline |
