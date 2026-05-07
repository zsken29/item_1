from __future__ import annotations

import unittest
from pathlib import Path

import torch

from lop.adapters import FfnLayerSpec, ModelFfnSpec
from lop.pruning import apply_ffn_pruning, load_layer_ratios, uniform_layer_ratios


class TinyMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(3, 4, bias=False)
        self.up_proj = torch.nn.Linear(3, 4, bias=False)
        self.down_proj = torch.nn.Linear(4, 3, bias=False)
        self.intermediate_size = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinyMlp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[0](x)


def _tiny_spec() -> ModelFfnSpec:
    return ModelFfnSpec(
        name="tiny",
        model_type="tiny",
        architecture="TinyModel",
        supported=True,
        note="test",
        layers=(
            FfnLayerSpec(
                index=0,
                block_path="layers.0",
                activation_path="layers.0.mlp.down_proj",
                gate_proj_weight="",
                up_proj_weight="",
                down_proj_weight="",
                hidden_size=3,
                intermediate_size=4,
                weight_status="verified",
            ),
        ),
    )


class PruningTest(unittest.TestCase):
    def test_prunes_dense_ffn_neurons(self) -> None:
        model = TinyModel()
        spec = ModelFfnSpec(
            name="tiny",
            model_type="tiny",
            architecture="TinyModel",
            supported=True,
            note="test",
            layers=(
                FfnLayerSpec(
                    index=0,
                    block_path="layers.0",
                    activation_path="layers.0.mlp.down_proj",
                    gate_proj_weight="",
                    up_proj_weight="",
                    down_proj_weight="",
                    hidden_size=3,
                    intermediate_size=4,
                    weight_status="verified",
                ),
            ),
        )
        ratios = uniform_layer_ratios(spec, 0.5)
        summary = apply_ffn_pruning(model, spec, {0: torch.tensor([0.1, 0.4, 0.2, 0.3])}, ratios)

        self.assertEqual(summary.original_neurons, 4)
        self.assertEqual(summary.kept_neurons, 2)
        self.assertEqual(model.layers[0].mlp.gate_proj.out_features, 2)
        self.assertEqual(model.layers[0].mlp.down_proj.in_features, 2)
        self.assertEqual(model(torch.ones(1, 3)).shape, torch.Size([1, 3]))

    def test_loads_prediction_layer_ratios(self) -> None:
        spec = _tiny_spec()
        path = Path.cwd() / "outputs" / "unit_tests" / "prediction_ratios.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"layer_ratios": [0.25]}', encoding="utf-8")
        ratios = load_layer_ratios(path, spec)
        self.assertEqual(ratios, {0: 0.25})


if __name__ == "__main__":
    unittest.main()
