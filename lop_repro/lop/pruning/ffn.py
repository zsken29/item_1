from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from lop.adapters import FfnLayerSpec, ModelFfnSpec


# 记录单层前馈网络（Feed-Forward Network，FFN）剪枝后的信息
@dataclass(frozen=True)
class PrunedLayer:
    index: int                 # 层索引
    original_neurons: int       # 剪枝前该层的神经元数量
    kept_neurons: int           # 剪枝后保留的神经元数量
    prune_ratio: float          # 目标剪枝比例


# 记录整个模型前馈网络剪枝后的汇总信息
@dataclass(frozen=True)
class PruningSummary:
    model_name: str                         # 模型名称
    layers: tuple[PrunedLayer, ...]         # 每一层的剪枝结果

    @property
    def original_neurons(self) -> int:
        # 计算所有被剪枝层在剪枝前的神经元总数
        return sum(layer.original_neurons for layer in self.layers)

    @property
    def kept_neurons(self) -> int:
        # 计算所有被剪枝层在剪枝后保留的神经元总数
        return sum(layer.kept_neurons for layer in self.layers)

    @property
    def actual_ratio(self) -> float:
        # 计算实际剪枝比例：1 - 保留神经元数 / 原始神经元数
        if self.original_neurons == 0:
            return 0.0
        return 1.0 - self.kept_neurons / self.original_neurons


def uniform_layer_ratios(
    spec: ModelFfnSpec,
    target_ratio: float,
    layer_count: int | None = None,
) -> dict[int, float]:
    # 为前若干层生成统一的剪枝比例配置
    _validate_ratio(target_ratio)

    # 如果未指定 layer_count，则默认覆盖模型中的所有 FFN 层
    count = len(spec.layers) if layer_count is None else layer_count

    # 检查 layer_count 是否在合法范围内
    if count < 1 or count > len(spec.layers):
        raise ValueError(f"layer_count must be in [1, {len(spec.layers)}], got {count}")

    # 返回形如 {层索引: 剪枝比例} 的字典
    return {layer.index: target_ratio for layer in spec.layers[:count]}


def load_layer_ratios(
    path: str | Path,
    spec: ModelFfnSpec,
    layer_count: int | None = None,
) -> dict[int, float]:
    # 从 JSON 文件加载每一层的剪枝比例
    ratio_path = Path(path)

    # 检查剪枝比例文件是否存在
    if not ratio_path.is_file():
        raise FileNotFoundError(f"missing layer ratio file: {ratio_path}")

    # 读取 JSON 文件内容
    payload = json.loads(ratio_path.read_text(encoding="utf-8"))

    # 期望 JSON 中包含 layer_ratios 字段
    raw = payload["layer_ratios"]

    # 确定需要读取多少层的剪枝比例
    count = len(spec.layers) if layer_count is None else layer_count

    if isinstance(raw, list):
        # 如果 layer_ratios 是列表，则按 spec.layers 的顺序对应各层
        if len(raw) < count:
            raise ValueError(f"{ratio_path}: expected at least {count} layer ratios, got {len(raw)}")

        ratios = {
            spec.layers[index].index: float(raw[index])
            for index in range(count)
        }

    elif isinstance(raw, dict):
        # 如果 layer_ratios 是字典，则键为层索引，值为剪枝比例
        ratios = {int(index): float(value) for index, value in raw.items()}

        # 如果只剪前 layer_count 层，则过滤掉不在允许范围内的层
        if layer_count is not None:
            allowed = {layer.index for layer in spec.layers[:layer_count]}
            ratios = {
                index: value
                for index, value in ratios.items()
                if index in allowed
            }

    else:
        # layer_ratios 只能是列表或字典
        raise TypeError(f"{ratio_path}: layer_ratios must be a list or object")

    # 校验每个剪枝比例是否合法
    for value in ratios.values():
        _validate_ratio(value)

    return ratios


def apply_ffn_pruning(
    model: torch.nn.Module,
    spec: ModelFfnSpec,
    importance: dict[int, torch.Tensor],
    layer_ratios: dict[int, float],
) -> PruningSummary:
    # 对模型中的 FFN 层执行剪枝，并返回剪枝汇总信息

    # 检查当前模型结构是否支持 dense FFN 剪枝
    if not spec.supported:
        raise ValueError(f"{spec.name} is not supported for dense FFN pruning: {spec.note}")

    # 将层规格按层索引组织，便于后续查找
    layers_by_index = {layer.index: layer for layer in spec.layers}

    pruned = []

    # 按层索引顺序逐层剪枝，保证结果稳定
    for index, ratio in sorted(layer_ratios.items()):
        _validate_ratio(ratio)

        # 检查目标层是否存在
        if index not in layers_by_index:
            raise KeyError(f"{spec.name}: unknown layer index {index}")

        # 检查该层是否有对应的重要性分数
        if index not in importance:
            raise KeyError(f"{spec.name}: missing importance for layer {index}")

        layer = layers_by_index[index]

        # 根据重要性分数和剪枝比例，确定需要保留的神经元索引
        keep = keep_indices_from_importance(importance[index], ratio)

        # 对该层实际执行剪枝
        pruned.append(_prune_layer(model, layer, keep, ratio))

    return PruningSummary(model_name=spec.name, layers=tuple(pruned))


def keep_indices_from_importance(scores: torch.Tensor, prune_ratio: float) -> torch.Tensor:
    # 根据重要性分数选择需要保留的神经元索引
    _validate_ratio(prune_ratio)

    # 将分数从计算图中分离，并转换为一维 float 张量
    values = scores.detach().float().reshape(-1)

    # 不允许空的重要性向量
    if values.numel() == 0:
        raise ValueError("importance vector is empty")

    # 根据剪枝比例计算需要保留的神经元数量，至少保留 1 个
    keep_count = max(1, int(round(values.numel() * (1.0 - prune_ratio))))
    keep_count = min(keep_count, values.numel())

    # 选出重要性最高的 keep_count 个神经元索引，并按索引升序排列
    return torch.topk(values, k=keep_count, largest=True, sorted=True).indices.sort().values


def _prune_layer(
    model: torch.nn.Module,
    layer: FfnLayerSpec,
    keep: torch.Tensor,
    ratio: float,
) -> PrunedLayer:
    # 对单个 FFN 层进行剪枝

    # 根据层路径获取模型中的 Transformer block
    block = model.get_submodule(layer.block_path)
    mlp = block.mlp

    # 获取 FFN 中的三个线性层，并确保它们都是 torch.nn.Linear
    gate = _require_linear(mlp.gate_proj, f"{layer.block_path}.mlp.gate_proj")
    up = _require_linear(mlp.up_proj, f"{layer.block_path}.mlp.up_proj")
    down = _require_linear(mlp.down_proj, f"{layer.block_path}.mlp.down_proj")

    # 检查 gate_proj 的输出维度是否仍等于原始中间层维度
    if gate.out_features != layer.intermediate_size:
        raise ValueError(f"layer {layer.index}: gate_proj out_features changed before pruning")

    # 检查 up_proj 的输出维度是否仍等于原始中间层维度
    if up.out_features != layer.intermediate_size:
        raise ValueError(f"layer {layer.index}: up_proj out_features changed before pruning")

    # 检查 down_proj 的输入维度是否仍等于原始中间层维度
    if down.in_features != layer.intermediate_size:
        raise ValueError(f"layer {layer.index}: down_proj in_features changed before pruning")

    # 将保留索引移动到权重所在设备，并转换为 long 类型
    keep = keep.to(device=gate.weight.device, dtype=torch.long)

    # gate_proj 和 up_proj 的输出维度对应中间层神经元，因此按行剪枝
    mlp.gate_proj = _prune_rows(gate, keep)
    mlp.up_proj = _prune_rows(up, keep)

    # down_proj 的输入维度对应中间层神经元，因此按列剪枝
    mlp.down_proj = _prune_columns(down, keep)

    # 如果模块显式记录 intermediate_size，则同步更新为剪枝后的大小
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = int(keep.numel())

    # 返回该层的剪枝结果
    return PrunedLayer(
        index=layer.index,
        original_neurons=layer.intermediate_size,
        kept_neurons=int(keep.numel()),
        prune_ratio=ratio,
    )


def _prune_rows(linear: torch.nn.Linear, keep: torch.Tensor) -> torch.nn.Linear:
    # 对线性层按行剪枝，即保留指定输出神经元

    new_linear = torch.nn.Linear(
        linear.in_features,             # 输入维度保持不变
        int(keep.numel()),              # 输出维度变为保留的神经元数量
        bias=linear.bias is not None,   # 保持是否使用偏置不变
        device=linear.weight.device,    # 保持设备一致
        dtype=linear.weight.dtype,      # 保持数据类型一致
    )

    with torch.no_grad():
        # 复制被保留行对应的权重
        new_linear.weight.copy_(linear.weight.index_select(0, keep))

        # 如果存在偏置，也只复制被保留输出神经元对应的偏置
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias.index_select(0, keep))

    return new_linear


def _prune_columns(linear: torch.nn.Linear, keep: torch.Tensor) -> torch.nn.Linear:
    # 对线性层按列剪枝，即保留指定输入神经元

    new_linear = torch.nn.Linear(
        int(keep.numel()),              # 输入维度变为保留的神经元数量
        linear.out_features,            # 输出维度保持不变
        bias=linear.bias is not None,   # 保持是否使用偏置不变
        device=linear.weight.device,    # 保持设备一致
        dtype=linear.weight.dtype,      # 保持数据类型一致
    )

    with torch.no_grad():
        # 复制被保留列对应的权重
        new_linear.weight.copy_(linear.weight.index_select(1, keep))

        # 按列剪枝不会改变输出维度，因此偏置可以完整复制
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias)

    return new_linear


def _require_linear(module: torch.nn.Module, path: str) -> torch.nn.Linear:
    # 检查模块是否为 torch.nn.Linear，不是则抛出类型错误
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"{path} must be torch.nn.Linear, got {type(module).__name__}")
    return module


def _validate_ratio(value: float) -> None:
    # 校验剪枝比例是否合法：必须大于等于 0 且小于 1
    if value < 0.0 or value >= 1.0:
        raise ValueError(f"prune ratio must be in [0, 1), got {value}")
