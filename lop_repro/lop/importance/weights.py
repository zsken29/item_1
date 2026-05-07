from __future__ import annotations

import torch

from lop.adapters import ModelFfnSpec


def weight_magnitude_importance(model: torch.nn.Module, spec: ModelFfnSpec) -> dict[int, torch.Tensor]:
    return {
        layer.index: _down_proj(model, layer.block_path).weight.detach().float().norm(dim=0).cpu()
        for layer in spec.layers
    }


def wanda_importance(
    model: torch.nn.Module,
    spec: ModelFfnSpec,
    activation_importance: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    scores = {}
    for layer in spec.layers:
        if layer.index not in activation_importance:
            raise KeyError(f"missing activation importance for layer {layer.index}")
        down = _down_proj(model, layer.block_path)
        weight_score = down.weight.detach().float().abs().mean(dim=0).cpu()
        activation_score = activation_importance[layer.index].detach().float().reshape(-1)
        _validate_score_shape(layer.index, weight_score, activation_score)
        scores[layer.index] = weight_score * activation_score
    return scores


def flap_importance(
    model: torch.nn.Module,
    spec: ModelFfnSpec,
    activation_fluctuation: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    scores = {}
    for layer in spec.layers:
        if layer.index not in activation_fluctuation:
            raise KeyError(f"missing activation fluctuation for layer {layer.index}")
        down = _down_proj(model, layer.block_path)
        weight_score = down.weight.detach().float().pow(2).sum(dim=0).cpu()
        fluctuation = activation_fluctuation[layer.index].detach().float().reshape(-1)
        _validate_score_shape(layer.index, weight_score, fluctuation)
        scores[layer.index] = weight_score * fluctuation
    return scores


def _down_proj(model: torch.nn.Module, block_path: str) -> torch.nn.Linear:
    block = model.get_submodule(block_path)
    return _require_linear(block.mlp.down_proj, f"{block_path}.mlp.down_proj")


def _require_linear(module: torch.nn.Module, path: str) -> torch.nn.Linear:
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"{path} must be torch.nn.Linear, got {type(module).__name__}")
    return module


def _validate_score_shape(layer_index: int, weight_score: torch.Tensor, activation_score: torch.Tensor) -> None:
    if weight_score.numel() != activation_score.numel():
        raise ValueError(
            f"layer {layer_index}: weight score length {weight_score.numel()} != activation score length {activation_score.numel()}"
        )
