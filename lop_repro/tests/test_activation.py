from __future__ import annotations

import unittest

import torch

from lop.importance import FfnActivationCollector


class ActivationCollectorTest(unittest.TestCase):
    def test_collects_down_proj_input_rms(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        x = torch.ones(2, 3)

        with FfnActivationCollector(model, ["1"]) as collector:
            _ = model(x)
            importance = collector.importance()
            summary = collector.summary()

        self.assertEqual(summary.layer_count, 1)
        self.assertEqual(summary.dimensions, (4,))
        self.assertEqual(importance["1"].shape, torch.Size([4]))

    def test_collects_down_proj_input_variance(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity())
        x = torch.tensor([[1.0, 3.0], [3.0, 5.0]])

        with FfnActivationCollector(model, ["0"]) as collector:
            _ = model(x)
            variance = collector.variance()

        self.assertTrue(torch.allclose(variance["0"], torch.tensor([1.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
