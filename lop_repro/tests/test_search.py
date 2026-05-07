from __future__ import annotations

import unittest

import torch

from lop.search import MctsConfig, PaperMctsConfig, mcts_search, paper_mcts_search, proxy_importance_reward


class SearchTest(unittest.TestCase):
    def test_mcts_returns_feasible_samples(self) -> None:
        config = MctsConfig(iterations=8, candidate_ratios=(0.0, 0.5), exploration=1.0, seed=1)
        samples = mcts_search(2, 0.25, config, objective=lambda ratios: 1.0 - sum(ratios))

        self.assertEqual(len(samples), 8)
        self.assertTrue(all(sum(sample.layer_ratios) / 2 == 0.25 for sample in samples))

    def test_proxy_reward_keeps_important_neurons(self) -> None:
        reward = proxy_importance_reward([torch.tensor([1.0, 3.0])], (0.5,))
        self.assertAlmostEqual(reward, 0.75)

    def test_paper_mcts_respects_budget(self) -> None:
        config = PaperMctsConfig(simulations=4, exploration=1.0, seed=1)
        samples = paper_mcts_search(3, 0.3, config, objective=lambda ratios: 1.0 - sum(ratios) / len(ratios))
        self.assertEqual(len(samples), 4)
        self.assertTrue(all(sum(sample.layer_ratios) / len(sample.layer_ratios) <= 0.3 + 1e-8 for sample in samples))


if __name__ == "__main__":
    unittest.main()
