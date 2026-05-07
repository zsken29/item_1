from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from lop.adapters import inspect_model_ffn
from lop.data import select_samples
from lop.data.prompts import build_question
from lop.importance import FfnActivationCollector, ImportanceRecord, save_importance
from lop.models import generate_response, load_chat_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FFN activation importance for a supported local MLLM.")
    parser.add_argument("--runtime", choices=["internvl", "qwen25_vl"], default="internvl")
    parser.add_argument("--model-dir", default="models/InternVL2_5-1B")
    parser.add_argument("--annotation", default="data/mmbench/annotations/en/dev-00000-of-00001.jsonl")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--stat", choices=["rms", "variance"], default="rms")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/importance/InternVL2_5-1B/default")
    args = parser.parse_args()

    start = time.time()
    root = Path.cwd()
    spec = inspect_model_ffn(args.model_dir)
    layers = spec.layers[: args.layers]
    if not layers:
        raise ValueError("layers must be positive")

    samples = select_samples(args.annotation, args.samples, args.seed)
    runtime = load_chat_runtime(args.runtime, args.model_dir)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }

    module_paths = [layer.activation_path for layer in layers]
    responses = []
    with torch.inference_mode(), FfnActivationCollector(runtime.model, module_paths) as collector:
        for sample in tqdm(samples, desc="importance", unit="sample"):
            question = build_question(sample)
            response = generate_response(runtime, root / sample["images"][0]["path"], question, generation_config)
            responses.append(
                {
                    "source_row": sample["source_row"],
                    "dataset": sample["dataset"],
                    "response": response,
                }
            )
        if args.stat == "rms":
            importance = collector.importance()
        else:
            importance = collector.variance()
        activation_summary = collector.summary()

    records = [
        ImportanceRecord(layer.index, layer.activation_path, importance[layer.activation_path])
        for layer in layers
    ]
    summary = {
        "model": Path(args.model_dir).name,
        "runtime": args.runtime,
        "model_dir": args.model_dir,
        "annotation": args.annotation,
        "samples": args.samples,
        "seed": args.seed,
        "layers": len(layers),
        "stat": args.stat,
        "activation_summary": asdict(activation_summary),
        "elapsed_seconds": round(time.time() - start, 3),
        "responses_preview": responses[:5],
    }
    save_importance(args.output_dir, records, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
