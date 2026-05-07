from __future__ import annotations

import json
import unittest
from pathlib import Path

from lop.adapters import inspect_model_ffn


def _test_root(name: str) -> Path:
    path = Path.cwd() / "outputs" / "unit_tests" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


class ModelFfnTest(unittest.TestCase):
    def test_qwen25_vl_dense_layers_are_verified_from_index(self) -> None:
        model = _test_root("model_ffn_qwen")
        config = {
            "model_type": "qwen2_5_vl",
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "intermediate_size": 64,
        }
        weight_map = {}
        for layer in range(2):
            for name in ("gate_proj", "up_proj", "down_proj"):
                weight_map[f"model.layers.{layer}.mlp.{name}.weight"] = "model.safetensors"
        (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}),
            encoding="utf-8",
        )

        spec = inspect_model_ffn(model)

        self.assertTrue(spec.supported)
        self.assertEqual(len(spec.layers), 2)
        self.assertEqual(spec.layers[0].weight_status, "verified")
        self.assertEqual(spec.layers[0].activation_path, "model.language_model.layers.0.mlp.down_proj")

    def test_deepseek_vl2_reports_moe_boundary(self) -> None:
        model = _test_root("model_ffn_deepseek")
        config = {
            "model_type": "deepseek_vl_v2",
            "language_config": {"hidden_size": 16, "intermediate_size": 64},
        }
        weight_map = {
            "language.model.layers.0.mlp.gate_proj.weight": "model.safetensors",
            "language.model.layers.0.mlp.up_proj.weight": "model.safetensors",
            "language.model.layers.0.mlp.down_proj.weight": "model.safetensors",
        }
        (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}),
            encoding="utf-8",
        )

        spec = inspect_model_ffn(model)

        self.assertFalse(spec.supported)
        self.assertEqual(len(spec.layers), 1)
        self.assertIn("MoE", spec.note)


if __name__ == "__main__":
    unittest.main()
