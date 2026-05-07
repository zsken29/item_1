from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from lop.adapters import inspect_model_ffn
from lop.data import iter_samples
from lop.data.prompts import build_question
from lop.eval import score_response
from lop.importance import load_importance
from lop.models import generate_response, load_chat_runtime
from lop.pruning import apply_ffn_pruning
from lop.search import PaperMctsConfig, paper_mcts_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper MCTS with real validation accuracy reward.")
    parser.add_argument("--runtime", choices=["internvl", "qwen25_vl"], default="internvl")
    parser.add_argument("--model-dir", default="models/InternVL2_5-1B")
    parser.add_argument("--importance-dir", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--target-ratio", type=float, required=True)
    parser.add_argument("--simulations", type=int, default=1)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--exploration", type=float, default=1.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("limit must be positive")
    start = time.time()
    root = Path.cwd()
    spec = inspect_model_ffn(args.model_dir)
    importance = load_importance(args.importance_dir)
    layer_count = len(importance) if args.layers is None else args.layers
    selected_layers = spec.layers[:layer_count]
    samples = _take_samples(args.annotation, args.limit)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }

    def objective(ratios: tuple[float, ...]) -> float:
        runtime = load_chat_runtime(args.runtime, args.model_dir)
        layer_ratios = {layer.index: ratios[position] for position, layer in enumerate(selected_layers)}
        apply_ffn_pruning(runtime.model, spec, importance, layer_ratios)
        correct = 0
        with torch.inference_mode():
            for sample in samples:
                response = generate_response(runtime, root / sample["images"][0]["path"], build_question(sample), generation_config)
                correct += int(score_response(sample, response).correct)
        return correct / len(samples)

    config = PaperMctsConfig(simulations=args.simulations, exploration=args.exploration, seed=args.seed)
    results = paper_mcts_search(layer_count, args.target_ratio, config, objective)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in results:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    summary = {
        "runtime": args.runtime,
        "model_dir": args.model_dir,
        "importance_dir": args.importance_dir,
        "annotation": args.annotation,
        "limit": args.limit,
        "target_ratio": args.target_ratio,
        "layers": layer_count,
        "config": asdict(config),
        "best": asdict(results[0]),
        "elapsed_seconds": round(time.time() - start, 3),
        "reward": "validation_accuracy",
    }
    (output_dir / "search_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _take_samples(annotation: str, limit: int) -> list[dict]:
    samples = []
    for sample in iter_samples(annotation):
        samples.append(sample)
        if len(samples) == limit:
            return samples
    raise ValueError(f"{annotation} has fewer than {limit} samples")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
