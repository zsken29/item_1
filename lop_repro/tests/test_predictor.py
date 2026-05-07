from __future__ import annotations

import unittest

import torch

from lop.predictor import AutoregressivePruningPredictor, PredictorSample, build_predictor, project_to_budget, train_predictor


class PredictorTest(unittest.TestCase):
    def test_predictor_forward_shape(self) -> None:
        model = AutoregressivePruningPredictor(layer_count=3, hidden_size=8, num_heads=2)
        output = model(torch.tensor([[0.2], [0.4]]))
        self.assertEqual(output.shape, torch.Size([2, 3]))

    def test_train_predictor_runs(self) -> None:
        samples = [
            PredictorSample(0.25, (0.0, 0.5), 1.0),
            PredictorSample(0.25, (0.5, 0.0), 0.9),
        ]
        model, metrics = train_predictor(
            samples,
            hidden_size=8,
            epochs=2,
            learning_rate=1e-2,
            seed=1,
            num_heads=2,
        )
        self.assertEqual(model.layer_count, 2)
        self.assertEqual(len(metrics), 2)

    def test_builds_ablation_predictors(self) -> None:
        for architecture in ("transformer", "bilstm", "mlp"):
            model = build_predictor(architecture, layer_count=2, hidden_size=8, num_heads=2)
            output = model(torch.tensor([[0.3]]))
            self.assertEqual(output.shape, torch.Size([1, 2]))

    def test_project_to_budget_preserves_target_mean(self) -> None:
        ratios = project_to_budget((0.4, 0.4, 0.4), 0.2)
        self.assertAlmostEqual(sum(ratios) / len(ratios), 0.2)


if __name__ == "__main__":
    unittest.main()
