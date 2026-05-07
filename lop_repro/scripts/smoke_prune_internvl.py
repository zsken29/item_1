from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from lop.adapters import inspect_model_ffn
from lop.data import iter_samples
from lop.data.prompts import build_question, expected_answer
from lop.importance import FfnActivationCollector
from lop.models import generate_response, load_chat_runtime
from lop.pruning import apply_ffn_pruning, uniform_layer_ratios


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense/pruned smoke inference for a supported local MLLM.")
    parser.add_argument("--runtime", choices=["internvl", "qwen25_vl"], default="internvl")
    parser.add_argument("--model-dir", default="models/InternVL2_5-1B")
    parser.add_argument("--annotation", default="data/mme/annotations/data/test-00000-of-00004-a25dbe3b44c4fda6.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", default="outputs/smoke_tests/internvl_pruned_smoke.json")
    args = parser.parse_args()

    root = Path.cwd()
    sample = _read_sample(args.annotation, args.sample_index)
    question = build_question(sample)
    image_path = root / sample["images"][0]["path"]

    spec = inspect_model_ffn(args.model_dir)
    selected_layers = spec.layers[: args.layers]
    runtime = load_chat_runtime(args.runtime, args.model_dir)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }

    module_paths = [layer.activation_path for layer in selected_layers]
    with torch.inference_mode(), FfnActivationCollector(runtime.model, module_paths) as collector:
        dense_response = generate_response(runtime, image_path, question, generation_config)
        importance_by_path = collector.importance()

    importance = {
        layer.index: importance_by_path[layer.activation_path]
        for layer in selected_layers
    }
    layer_ratios = uniform_layer_ratios(spec, args.ratio, layer_count=args.layers)
    pruning_summary = apply_ffn_pruning(runtime.model, spec, importance, layer_ratios)

    with torch.inference_mode():
        pruned_response = generate_response(runtime, image_path, question, generation_config)

    output = {
        "model": Path(args.model_dir).name,
        "runtime": args.runtime,
        "annotation": args.annotation,
        "sample_index": args.sample_index,
        "question": question,
        "expected_answer": expected_answer(sample),
        "dense_response": dense_response,
        "pruned_response": pruned_response,
        "pruning_summary": {
            "model_name": pruning_summary.model_name,
            "original_neurons": pruning_summary.original_neurons,
            "kept_neurons": pruning_summary.kept_neurons,
            "actual_ratio": round(pruning_summary.actual_ratio, 6),
            "layers": [asdict(layer) for layer in pruning_summary.layers],
        },
    }
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _read_sample(annotation: str, sample_index: int) -> dict:
    if sample_index < 0:
        raise ValueError("sample-index must be non-negative")
    for index, sample in enumerate(iter_samples(annotation)):
        if index == sample_index:
            return sample
    raise ValueError(f"{annotation} has no sample at index {sample_index}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
