from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import torch

# 离散候选值 MCTS 配置，用于快速 smoke test
@dataclass(frozen=True)
class MctsConfig:
    iterations: int
    candidate_ratios: tuple[float, ...]
    exploration: float
    seed: int

# 论文式连续扰动 MCTS 配置（Appendix A.3）。
#   - simulations: MCTS 模拟次数，论文推荐 300
#   - exploration: UCB 探索系数
#   - initial_delta: 初始扰动幅度，按 0.9^depth 衰减
#   - min_ratio / max_ratio: 逐层剪枝率的上下界，防止极端值
@dataclass(frozen=True)
class PaperMctsConfig:
    simulations: int
    exploration: float
    seed: int
    initial_delta: float = 0.1
    min_ratio: float = 0.1
    max_ratio: float = 0.95

# 一次 MCTS 搜索产出的样本：(目标全局剪枝率, 逐层剪枝率元组, reward)
@dataclass(frozen=True)
class SearchSample:
    global_ratio: float
    layer_ratios: tuple[float, ...]
    reward: float

# 离散 MCTS 树的节点：prefix 是已固定的逐层剪枝率前缀，
# children 按下一层的候选剪枝率分支。
class _Node:
    def __init__(self, prefix: tuple[float, ...], parent: "_Node | None") -> None:
        self.prefix = prefix
        self.parent = parent
        self.children: dict[float, _Node] = {}
        self.visits = 0
        self.value = 0.0

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

# 论文 MCTS 树的节点：ratios 是完整的逐层剪枝率，
# 子节点通过连续扰动产生，不做离散展开。
class _ContinuousNode:
    def __init__(self, ratios: tuple[float, ...], parent: "_ContinuousNode | None", depth: int) -> None:
        self.ratios = ratios
        self.parent = parent
        self.children: list[_ContinuousNode] = []
        self.depth = depth           # 树的深度 = 扰动次数，用于衰减扰动幅度
        self.visits = 0
        self.value = 0.0
        self.reward: float | None = None

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

# 离散候选值 MCTS 搜索，用于快速 smoke test。
# 每层从候选剪枝率集合中选择，具有总预算约束。
def mcts_search(
    layer_count: int,
    target_ratio: float,
    config: MctsConfig,
    objective: Callable[[tuple[float, ...]], float],
) -> list[SearchSample]:
    if layer_count < 1:
        raise ValueError("layer_count must be positive")
    _validate_ratio(target_ratio)
    candidates = tuple(sorted(config.candidate_ratios))
    if not candidates:
        raise ValueError("candidate_ratios must not be empty")
    for value in candidates:
        _validate_ratio(value)
    if config.iterations < 1:
        raise ValueError("iterations must be positive")

    rng = random.Random(config.seed)
    root = _Node(prefix=(), parent=None)
    samples: list[SearchSample] = []
    for _ in range(config.iterations):
        node = _select(root, layer_count, target_ratio, candidates, config.exploration)
        if len(node.prefix) < layer_count:
            node = _expand(node, layer_count, target_ratio, candidates, rng)
        ratios = _rollout(node.prefix, layer_count, target_ratio, candidates, rng)
        reward = objective(ratios)
        _backpropagate(node, reward)
        samples.append(SearchSample(target_ratio, ratios, reward))
    return sorted(samples, key=lambda sample: sample.reward, reverse=True)

# 论文式连续扰动 MCTS 搜索（Appendix A.3, Eq.26-32）。
# 核心流程：
#   1. Selection: 用 UCB 从根节点选择最有望的子节点
#   2. Expansion: 对当前配置做连续扰动（幅度随深度衰减），产生新候选
#   3. Evaluation: 调用 objective 得到 reward
#   4. Backpropagation: reward 沿路径回传，更新 visits 和 value
def paper_mcts_search(
    layer_count: int,
    target_ratio: float,
    config: PaperMctsConfig,
    objective: Callable[[tuple[float, ...]], float],
) -> list[SearchSample]:
    if layer_count < 1:
        raise ValueError("layer_count must be positive")
    _validate_ratio(target_ratio)
    _validate_ratio(config.min_ratio)
    _validate_ratio(config.max_ratio)
    if config.min_ratio > config.max_ratio:
        raise ValueError("min_ratio must be <= max_ratio")
    if target_ratio < config.min_ratio:
        raise ValueError(f"target_ratio {target_ratio} is below paper MCTS min_ratio {config.min_ratio}")
    if config.simulations < 1:
        raise ValueError("simulations must be positive")

    rng = random.Random(config.seed)
    root_ratio = min(max(target_ratio, config.min_ratio), config.max_ratio)
    root = _ContinuousNode(ratios=tuple(root_ratio for _ in range(layer_count)), parent=None, depth=0)
    samples: list[SearchSample] = []
    for _ in range(config.simulations):
        selected = _select_continuous(root, config.exploration)
        child_ratios = _perturb_configuration(selected.ratios, selected.depth + 1, target_ratio, config, rng)
        reward = objective(child_ratios)
        child = _ContinuousNode(ratios=child_ratios, parent=selected, depth=selected.depth + 1)
        child.reward = reward
        selected.children.append(child)
        _backpropagate_continuous(child, reward)
        samples.append(SearchSample(target_ratio, child_ratios, reward))
    return sorted(samples, key=lambda sample: sample.reward, reverse=True)

# Proxy reward：根据重要性分数计算保留的神经元质量比。
# 对每层，按剪枝率保留最 top 神经元后，计算保留的重要性总和占全体的比例，
# 各层取平均。值越高说明剪掉的都是不重要的神经元。
# 用于快速 MCTS，避免每次 candidate 都重新加载模型跑真实验证集。
def proxy_importance_reward(importance: list[torch.Tensor], ratios: tuple[float, ...]) -> float:
    if len(importance) != len(ratios):
        raise ValueError(f"importance layers {len(importance)} != ratios {len(ratios)}")
    retained = []
    for scores, ratio in zip(importance, ratios):
        _validate_ratio(ratio)
        values = scores.detach().float().reshape(-1)
        if values.numel() == 0:
            raise ValueError("importance vector is empty")
        keep = max(1, int(round(values.numel() * (1.0 - ratio))))
        retained.append(float(torch.topk(values, k=keep, largest=True).values.sum() / values.sum().clamp_min(1e-12)))
    return sum(retained) / len(retained)

# 从根节点出发，沿 UCB 最大的子节点路径向下选择，返回叶节点
def _select_continuous(root: _ContinuousNode, exploration: float) -> _ContinuousNode:
    node = root
    while node.children:
        node = max(node.children, key=lambda child: _continuous_ucb_score(node, child, exploration))
    return node

# 对当前逐层剪枝率配置做连续扰动：
#   1. 每层加一个 U(-delta, delta) 的随机噪声，delta = initial_delta * 0.9^depth
#   2. 将均值缩放到不超过目标剪枝率（确保满足预算约束）
def _perturb_configuration(
    ratios: tuple[float, ...],
    depth: int,
    target_ratio: float,
    config: PaperMctsConfig,
    rng: random.Random,
) -> tuple[float, ...]:
    delta = config.initial_delta * (0.9 ** depth)
    values = [
        min(max(value + rng.uniform(-delta, delta), config.min_ratio), config.max_ratio)
        for value in ratios
    ]
    mean_value = sum(values) / len(values)
    if mean_value > target_ratio:
        scale = target_ratio / mean_value
        values = [max(config.min_ratio, value * scale) for value in values]
    # 经过 min_ratio 裁剪后，均值可能仍然超标；将超出部分迭代分摊到可调层
    for _ in range(20):
        current_mean = sum(values) / len(values)
        if current_mean <= target_ratio + 1e-8:
            break
        adjustable = [i for i, v in enumerate(values) if v > config.min_ratio + 1e-10]
        if not adjustable:
            break
        excess = (current_mean - target_ratio) * len(values)
        per_layer = excess / len(adjustable)
        for i in adjustable:
            values[i] = max(config.min_ratio, values[i] - per_layer)
    if sum(values) / len(values) > target_ratio + 1e-8:
        raise ValueError("paper MCTS perturbation failed to satisfy pruning budget")
    return tuple(values)

# 将 reward 沿父链向上传播：visits += 1，value += reward
def _backpropagate_continuous(node: _ContinuousNode, reward: float) -> None:
    current: _ContinuousNode | None = node
    while current is not None:
        current.visits += 1
        current.value += reward
        current = current.parent

# UCB 公式（论文式连续节点版）：
#   UCB = mean_value + exploration * sqrt(ln(parent_visits + 1) / child_visits)
def _continuous_ucb_score(parent: _ContinuousNode, child: _ContinuousNode, exploration: float) -> float:
    if child.visits == 0:
        return math.inf
    return child.mean_value + exploration * math.sqrt(math.log(parent.visits + 1) / child.visits)

# 离散 MCTS 的 Selection 步骤：从根节点沿 UCB 最大子节点向下，
# 停在未完全展开或可进一步展开的节点。
def _select(
    root: _Node,
    layer_count: int,
    target_ratio: float,
    candidates: tuple[float, ...],
    exploration: float,
) -> _Node:
    node = root
    while len(node.prefix) < layer_count and _fully_expanded(node, layer_count, target_ratio, candidates):
        node = max(
            node.children.values(),
            key=lambda child: _ucb_score(node, child, exploration),
        )
    return node

# 离散 MCTS 的 Expansion 步骤：从未被尝试的可行操作中随机选一个，
# 创建子节点。
def _expand(
    node: _Node,
    layer_count: int,
    target_ratio: float,
    candidates: tuple[float, ...],
    rng: random.Random,
) -> _Node:
    actions = [
        action
        for action in _feasible_actions(node.prefix, layer_count, target_ratio, candidates)
        if action not in node.children
    ]
    if not actions:
        return node
    action = rng.choice(actions)
    child = _Node(prefix=node.prefix + (action,), parent=node)
    node.children[action] = child
    return child

# 离散 MCTS 的 Rollout 步骤：从当前前缀出发，随机填充剩余层
def _rollout(
    prefix: tuple[float, ...],
    layer_count: int,
    target_ratio: float,
    candidates: tuple[float, ...],
    rng: random.Random,
) -> tuple[float, ...]:
    ratios = list(prefix)
    while len(ratios) < layer_count:
        actions = _feasible_actions(tuple(ratios), layer_count, target_ratio, candidates)
        ratios.append(rng.choice(actions))
    return tuple(ratios)

# 离散 MCTS 的 Backpropagation 步骤
def _backpropagate(node: _Node, reward: float) -> None:
    current: _Node | None = node
    while current is not None:
        current.visits += 1
        current.value += reward
        current = current.parent

# 检查一个节点是否已被完全展开（所有可行操作都已有子节点）
def _fully_expanded(
    node: _Node,
    layer_count: int,
    target_ratio: float,
    candidates: tuple[float, ...],
) -> bool:
    return len(node.children) == len(_feasible_actions(node.prefix, layer_count, target_ratio, candidates))

# 计算在当前前缀下哪些剪枝率是可行的（仍需满足总预算约束）。
# 同时考虑剩余层采用最小值或最大值的极端情况，确保约束范围内有解。
def _feasible_actions(
    prefix: tuple[float, ...],
    layer_count: int,
    target_ratio: float,
    candidates: tuple[float, ...],
) -> tuple[float, ...]:
    remaining_after_action = layer_count - len(prefix) - 1
    current = sum(prefix)
    feasible = []
    for action in candidates:
        minimum = current + action + remaining_after_action * candidates[0]
        maximum = current + action + remaining_after_action * candidates[-1]
        target_sum = target_ratio * layer_count
        if minimum - 1e-9 <= target_sum <= maximum + 1e-9:
            feasible.append(action)
    if not feasible:
        raise ValueError("no feasible action for target ratio and candidates")
    return tuple(feasible)

# UCB 公式（离散版）：
#   UCB = mean_value + exploration * sqrt(ln(parent_visits + 1) / child_visits)
def _ucb_score(parent: _Node, child: _Node, exploration: float) -> float:
    if child.visits == 0:
        return math.inf
    return child.mean_value + exploration * math.sqrt(math.log(parent.visits + 1) / child.visits)

# 校验剪枝率在 [0, 1) 范围内
def _validate_ratio(value: float) -> None:
    if value < 0.0 or value >= 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {value}")
