from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from lop.importance import load_importance
from lop.search import MctsConfig, PaperMctsConfig, mcts_search, paper_mcts_search, proxy_importance_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Search layer pruning ratios from saved importance.")
    parser.add_argument("--importance-dir", required=True)
    parser.add_argument("--target-ratio", type=float, required=True)
    parser.add_argument("--mode", choices=["paper", "discrete"], default="paper")
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--candidate-ratios", default="0.0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--exploration", type=float, default=1.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/mcts_samples/default")
    args = parser.parse_args()

    start = time.time()
    importance_by_layer = load_importance(args.importance_dir)
    layer_indices = sorted(importance_by_layer)
    importance = [importance_by_layer[index] for index in layer_indices]
    candidates = tuple(float(item) for item in args.candidate_ratios.split(","))
    if args.mode == "paper":
        config = PaperMctsConfig(
            simulations=args.iterations,
            exploration=args.exploration,
            seed=args.seed,
        )
        samples = paper_mcts_search(
            layer_count=len(layer_indices),
            target_ratio=args.target_ratio,
            config=config,
            objective=lambda ratios: proxy_importance_reward(importance, ratios),
        )
    else:
        config = MctsConfig(
            iterations=args.iterations,
            candidate_ratios=candidates,
            exploration=args.exploration,
            seed=args.seed,
        )
        samples = mcts_search(
            layer_count=len(layer_indices),
            target_ratio=args.target_ratio,
            config=config,
            objective=lambda ratios: proxy_importance_reward(importance, ratios),
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    summary = {
        "importance_dir": args.importance_dir,
        "target_ratio": args.target_ratio,
        "mode": args.mode,
        "layer_indices": layer_indices,
        "config": asdict(config),
        "best": asdict(samples[0]),
        "elapsed_seconds": round(time.time() - start, 3),
        "note": "reward is an importance-retention proxy; full benchmark reward is intentionally not run here",
    }
    (output_dir / "search_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
