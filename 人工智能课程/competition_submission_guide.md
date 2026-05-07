# 武汉大学数智教育《人工智能与机器学习》课程实践大赛提交说明

## 比赛信息

- 课程：武汉大学数智教育《人工智能与机器学习》
- 任务：给定训练集，构建分类模型；在测试集上预测标签，追求最高分类准确率。
- 比赛地址：https://aipractice.whu.edu.cn/homework/detail/62785147598?tab=introduce

## 基础测试操作

自行测试时：

1. 在 Notebook 内点击“开始连接”连接节点。
2. 选择对应项目。
3. 镜像选择：`MachineLearning_Sgg_2024:2024-09-06`。
4. 机型选择默认推荐机型。
5. 挂载需要使用的数据集。
6. 注意：数据集需要和 Notebook 在同一个项目下。

## 数据读取路径

自行测试 Notebook 中读取训练集：

```python
train_data = pd.read_csv('/bohr/train-dataset-fvus/v2/train_dataset.csv')
```

提交比赛 Notebook 时读取测试集：

```python
import os

if os.environ.get('DATA_PATH'):
    data_path = os.environ.get('DATA_PATH') + '/'
else:
    data_path = '/bohr/test-dataset-xemz/v4/'

test_data = pd.read_csv(f'{data_path}test_dataset_nolabel.csv')
```

## 提交文件格式

需要将测试数据集和预测标签拼接，生成当前目录下的：

```text
submission.csv
```

提交文件列格式：

```text
Feature_1,Feature_2,Feature_3,Label
```

示例：

```text
Feature_1,Feature_2,Feature_3,Label
-1.5210258269784074,-1.3709666070533935,-1.3884141434118007,2
```

保存代码：

```python
result.to_csv('submission.csv', index=False)
```

## 当前提交模型

当前整理出的提交 Notebook 使用自包含模型代码，不依赖本地 Python 包或额外文件。

模型：

```text
gmm_per_class_1_7_11_full_offsets
```

模型结构：

```python
class_n_components = {0: 1, 1: 7, 2: 11}
class_offsets = {0: -0.1, 1: 0.0, 2: 0.0}
```

本地标准重复 5 折验证：

```text
RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
accuracy = 0.813000
```

为降低比赛环境兼容风险，Notebook 内置了 fallback：

```text
1. gmm_per_class_1_7_11_full_offsets
2. gmm_per_class_1_6_7_full_offsets
3. gmm_2_full
4. QuadraticDiscriminantAnalysis
```

正常情况下会打印：

```text
model used: gmm_per_class_1_7_11_full_offsets
```

如果某个 GMM 在比赛环境里失败，会打印 `model failed: ...` 并自动尝试下一个配置。

## 可直接提交的 Notebook

文件：

```text
competition_submission.ipynb
```

该 Notebook 已合并为一个 Jupyter 代码框，直接从上到下运行即可。

运行后会在当前目录生成：

```text
submission.csv
```
