from __future__ import annotations

import unittest

import torch

from lop.adapters import FfnLayerSpec, ModelFfnSpec
from lop.importance import flap_importance, wanda_importance, weight_magnitude_importance


class TinyMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(2, 3, bias=False)
        self.up_proj = torch.nn.Linear(2, 3, bias=False)
        self.down_proj = torch.nn.Linear(3, 2, bias=False)


class TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinyMlp()


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock()])


class WeightImportanceTest(unittest.TestCase):
    def test_baseline_importance_shapes(self) -> None:
        model = TinyModel()
        model.layers[0].mlp.down_proj.weight.data = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        spec = ModelFfnSpec(
            name="tiny",
            model_type="tiny",
            architecture="TinyModel",
            supported=True,
            layers=(
                FfnLayerSpec(
                    index=0,
                    block_path="layers.0",
                    activation_path="layers.0.mlp.down_proj",
                    gate_proj_weight="",
                    up_proj_weight="",
                    down_proj_weight="",
                    hidden_size=2,
                    intermediate_size=3,
                    weight_status="verified",
                ),
            ),
            note="test",
        )

        magnitude = weight_magnitude_importance(model, spec)
        wanda = wanda_importance(model, spec, {0: torch.ones(3)})
        flap = flap_importance(model, spec, {0: torch.ones(3)})

        self.assertEqual(magnitude[0].shape, torch.Size([3]))
        self.assertEqual(wanda[0].shape, torch.Size([3]))
        self.assertEqual(flap[0].shape, torch.Size([3]))
        self.assertTrue(torch.allclose(flap[0], torch.tensor([17.0, 29.0, 45.0])))


if __name__ == "__main__":
    unittest.main()
