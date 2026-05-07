# Search summary

Target requested: repeated 5-fold accuracy above `0.84` using:

```python
RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
```

Best supervised-only result found so far:

| model | repeated 5-fold accuracy |
|---|---:|
| retained baseline `gmm_2_full` | 0.811300 |
| `gmm_2_full_class_offsets` | 0.811867 |
| `gmm_per_class_1_6_7_full_offsets` | 0.812867 |
| `gmm_per_class_1_7_11_full_offsets` | 0.813000 |

Current retained model:

| item | value |
|---|---|
| model file | `advanced_model_result/artifacts/final_model.joblib` |
| training script | `advanced_model_result/train.py` |
| prediction script | `advanced_model_result/predict.py` |
| model class | `advanced_model_result/custom_models.py` |
| metrics | `advanced_model_result/artifacts/metrics.json` |
| report | `advanced_model_result/artifacts/evaluation_report.md` |

Current retained files after cleanup:

| file | purpose |
|---|---|
| `README.md` | concise run instructions and best result |
| `search_summary.md` | consolidated experiment record |
| `train.py` | reproducible training entrypoint |
| `predict.py` | reproducible prediction entrypoint |
| `custom_models.py` | saved model class definitions |
| `artifacts/final_model.joblib` | current best retained model |
| `artifacts/metrics.json` | current best retained metrics |
| `artifacts/model_info.json` | current best retained model metadata |
| `artifacts/evaluation_report.md` | current best retained report |

Main experiments attempted:

| family | best observed result |
|---|---:|
| XGBoost / LightGBM / CatBoost quick search | 0.799100 5-fold |
| SVM / KNN / polynomial logistic / RBF features | about 0.803200 5-fold |
| MLP neural networks | about 0.809800 5-fold |
| KDE / Gaussian copula KDE | about 0.801600 5-fold |
| Bayesian GMM | about 0.811900 5-fold |
| hierarchical `2 vs rest`, then `0 vs 1` | about 0.811300 5-fold |
| density-score stacking | about 0.812100 5-fold |
| per-class GMM component search | 0.812867 repeated 5-fold |
| focused `0/1` GMM search, then tri-class component retune | 0.813000 repeated 5-fold |
| FLAML AutoML, 10 minute budget | 0.795500 repeated 5-fold |
| pairwise one-vs-one aggregation | about 0.811500 repeated 5-fold |
| GMM component responsibility features | about 0.811300 repeated 5-fold |
| nested per-fold class-offset calibration | 0.811200 repeated 5-fold |
| shared GMM cluster-label model | about 0.800100 repeated 5-fold |
| string/decimal-tail feature leakage checks | about 0.785400 5-fold |

Blocked experiment:

- `tabpfn==7.1.1` installed successfully, but local inference requires Prior Labs license acceptance and `TABPFN_TOKEN`; without that token, model weights cannot be downloaded in this non-interactive environment.

Diagnostic notes:

- Full-data fitted GMM apparent accuracy is only about `0.8162`, which is a warning sign that the available three features may not contain enough information for `0.84`.
- Most model families converge around `0.80` to `0.813`, so the gap to `0.84` is not currently explained by one missing model package.
- Reaching `0.84` likely needs additional real features, a different label source, or an evaluation setup that permits extra information beyond the three CSV feature columns.
- The hardest boundary is `0 vs 1`; dedicated binary CV for that pair is only about `0.7667`, while `0 vs 2` and `1 vs 2` are around `0.956` and `0.967`.
- Focused binary `0/1` GMM retuning improved that pair to `0.768101` repeated 5-fold, but the gain is small.

Cleanup note:

- Raw intermediate search CSVs, prediction outputs, and Python bytecode caches were removed after their useful results were consolidated here.
