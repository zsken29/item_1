import os
import warnings

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
FEATURES = ['Feature_1', 'Feature_2', 'Feature_3']
TARGET = 'Label'

train_path = '/bohr/train-dataset-fvus/v2/train_dataset.csv'
if os.path.exists(train_path):
    train_data = pd.read_csv(train_path)
else:
    # 方便本地调试；比赛环境会走上面的 /bohr 路径。
    train_data = pd.read_csv('/personal/challenge/train_dataset.csv')

if os.environ.get('DATA_PATH'):
    data_path = os.environ.get('DATA_PATH') + '/'
else:
    data_path = '/bohr/test-dataset-xemz/v4/'

test_path = f'{data_path}test_dataset_nolabel.csv' if os.path.exists(f'{data_path}test_dataset_nolabel.csv') else '/personal/challenge/train_dataset.csv'
test_data = pd.read_csv(test_path)

print('train path:', train_path if os.path.exists(train_path) else 'train_dataset.csv')
print('test path:', test_path)
print('train_data:', train_data.shape)
print('test_data:', test_data.shape)
print('train columns:', list(train_data.columns))
print('test columns:', list(test_data.columns))

def validate_columns(train_df, test_df):
    missing_train = [col for col in FEATURES + [TARGET] if col not in train_df.columns]
    missing_test = [col for col in FEATURES if col not in test_df.columns]
    if missing_train:
        raise ValueError(f'训练集缺少列: {missing_train}')
    if missing_test:
        raise ValueError(f'测试集缺少列: {missing_test}')


def fit_per_class_gmm(X, y, class_n_components, class_offsets, n_init=3, max_iter=500):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    models = {}
    log_priors = {}
    offsets = np.array([float(class_offsets.get(int(cls), 0.0)) for cls in classes])

    for cls in classes:
        X_cls = X[y == cls]
        n_components = int(class_n_components.get(int(cls), 2))
        model = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            reg_covar=1e-6,
            random_state=RANDOM_STATE,
            n_init=n_init,
            max_iter=max_iter,
        )
        model.fit(X_cls)
        models[int(cls)] = model
        log_priors[int(cls)] = float(np.log(len(X_cls) / len(X)))

    return classes.astype(int), models, log_priors, offsets


def predict_per_class_gmm(fitted, X):
    classes, models, log_priors, offsets = fitted
    X = np.asarray(X, dtype=float)
    joint = np.column_stack([
        models[int(cls)].score_samples(X) + log_priors[int(cls)]
        for cls in classes
    ])
    joint = joint + offsets
    return classes[np.argmax(joint, axis=1)].astype(int)


def train_and_predict(train_df, test_df):
    validate_columns(train_df, test_df)
    X_train = train_df[FEATURES].to_numpy(dtype=float)
    y_train = train_df[TARGET].to_numpy(dtype=int)
    X_test = test_df[FEATURES].to_numpy(dtype=float)

    # 当前本地重复 5 折最优配置。
    configs = [
        ('gmm_per_class_1_7_11_full_offsets', {0: 1, 1: 7, 2: 11}, {0: -0.1, 1: 0.0, 2: 0.0}, 3, 500),
        # 兼容性 fallback：更小模型，训练更快。
        ('gmm_per_class_1_6_7_full_offsets', {0: 1, 1: 6, 2: 7}, {0: 0.05, 1: 0.15, 2: 0.0}, 1, 300),
        ('gmm_2_full', {0: 2, 1: 2, 2: 2}, {0: 0.0, 1: 0.0, 2: 0.0}, 1, 300),
    ]

    last_error = None
    for name, class_n_components, class_offsets, n_init, max_iter in configs:
        try:
            fitted = fit_per_class_gmm(
                X_train,
                y_train,
                class_n_components=class_n_components,
                class_offsets=class_offsets,
                n_init=n_init,
                max_iter=max_iter,
            )
            pred = predict_per_class_gmm(fitted, X_test)
            print('model used:', name)
            return pred
        except Exception as exc:
            last_error = exc
            print('model failed:', name, repr(exc))

    # 最后 fallback：QDA。这个分数不一定最高，但通常兼容性很好。
    print('fallback to QDA because GMM failed:', repr(last_error))
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    return qda.predict(X_test).astype(int)

pred = train_and_predict(train_data, test_data)

result = test_data[FEATURES].copy()
result[TARGET] = pred.astype(int)
result.to_csv('submission.csv', index=False)

print('submission.csv saved')
print(result.head())
print('label counts:')
print(result[TARGET].value_counts().sort_index())
