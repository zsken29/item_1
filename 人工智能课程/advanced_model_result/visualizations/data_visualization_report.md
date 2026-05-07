# Data visualization report

This report visualizes why classes `0` and `1` are much harder to separate than class `2`.

## Key findings

- Current best model repeated 5-fold accuracy: `0.813000`.
- Repeated 5-fold per-class recall: class 0 `0.615897`, class 1 `0.858386`, class 2 `0.965692`.
- Most mistakes are between class `0` and class `1`; class `2` is comparatively clean.
- The dedicated binary task `0 vs 1` is far below `0 vs 2` and `1 vs 2`.

## Pairwise separability

| pair | best quick binary model | 5-fold binary accuracy |
|---|---|---:|
| 0 vs 1 | QDA | 0.766696 |
| 0 vs 2 | GMM-6 | 0.956611 |
| 1 vs 2 | GMM-6 | 0.966892 |

## Figures

| figure | file |
|---|---|
| Pairwise scatter | `01_pairwise_scatter.png` |
| Feature distributions | `02_feature_distributions.png` |
| PCA/LDA projections | `03_pca_lda_projection.png` |
| Class 0/1 overlap | `04_class_0_1_overlap.png` |
| Repeated CV confusion | `05_repeated_cv_confusion.png` |
| Pairwise difficulty | `06_pairwise_difficulty.png` |
| Local neighbor ambiguity | `07_local_neighbor_ambiguity.png` |
