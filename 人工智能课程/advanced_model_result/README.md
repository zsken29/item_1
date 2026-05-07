# Advanced model result

This directory stores the stronger reproducible model selected after testing boosted trees and GMM variants.

## Selected model

- Model: `gmm_per_class_1_7_11_full_offsets`
- Type: per-class Gaussian mixture classifier with fixed class log-score offsets
- Components: `{0: 1, 1: 7, 2: 11}`
- Offsets: `{0: -0.1, 1: 0.0, 2: 0.0}`
- Reason: this is the best supervised-only repeated 5-fold cross-validation result found so far.

## Key comparison

| model | fixed validation accuracy | repeated 5-fold CV accuracy |
|---|---:|---:|
| retained baseline `gmm_2_full` | 0.815000 | 0.811300 |
| previous advanced model | 0.815500 | 0.811867 |
| previous advanced model | 0.814500 | 0.812867 |
| selected advanced model | 0.814500 | 0.813000 |

The external boosted tree packages were tested too. In the quick 5-fold search, the best boosted tree was `xgb_d4_lr0.03_n500_sub0.85` with CV accuracy 0.799100, below the GMM baseline.

## Files

- `train.py`: retrains the selected model and writes artifacts.
- `predict.py`: loads the selected model and predicts CSV files.
- `custom_models.py`: model class used by the saved joblib file.
- `quick_model_search.csv`: boosted tree / forest / SVM comparison results.
- `gmm_family_repeated_cv.csv`: repeated CV check for the best GMM variants.
- `artifacts/final_model.joblib`: trained final model.
- `artifacts/metrics.json`: core metrics.

## Run

```powershell
& 'C:\Users\ZSQ\anaconda3\envs\d2l_env\python.exe' advanced_model_result\train.py
& 'C:\Users\ZSQ\anaconda3\envs\d2l_env\python.exe' advanced_model_result\predict.py --input train_dataset.csv
```
