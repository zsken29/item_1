# Class 0/1 model design notes

Goal: improve the hardest boundary, `class 0` vs `class 1`.

## Findings

Dedicated binary `0 vs 1` evaluation is the bottleneck:

| model family | best repeated 5-fold binary accuracy |
|---|---:|
| QDA | 0.766998 |
| engineered features + discriminative models | about 0.7668 |
| class-specific GMM component search | 0.768101 |

The best dedicated `0/1` binary model found so far:

| item | value |
|---|---|
| class 0 density | full-covariance GMM, 2 components |
| class 1 density | full-covariance GMM, 11 components |
| decision rule | predict `1` when `log p_1(x) - log p_0(x) >= 0.015` |
| repeated 5-fold binary accuracy | 0.768101 |
| repeated confusion | `[[6725, 3340], [1285, 8594]]` |

This improves binary `0/1` slightly, but the overlap is still strong.

## Integrated tri-class design

The binary search showed that class `1` benefits from more mixture components. Retuning the tri-class per-class GMM around that structure produced the current retained model:

| class | components |
|---:|---:|
| 0 | 1 |
| 1 | 7 |
| 2 | 11 |

Offsets:

```python
{0: -0.1, 1: 0.0, 2: 0.0}
```

Current standard repeated 5-fold accuracy:

```text
0.813000
```

This is only a small improvement over `0.812867`, but it is the best supervised-only result found so far.
