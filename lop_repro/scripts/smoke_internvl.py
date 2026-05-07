from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from lop.data import iter_samples
from lop.data.prompts import build_question, expected_answer
from lop.importance import FfnActivationCollector
from lop.models.internvl import load_image_pixels, load_internvl_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one InternVL smoke inference and collect FFN activations.")
    parser.add_argument("--model-dir", default="models/InternVL2_5-1B")
    parser.add_argument("--annotation", default="data/mme/annotations/data/test-00000-of-00004-a25dbe3b44c4fda6.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", default="outputs/smoke_tests/internvl_smoke.json")
    args = parser.parse_args()

    root = Path.cwd()
    sample = _read_sample(args.annotation, args.sample_index)
    image_path = root / sample["images"][0]["path"]
    question = build_question(sample)

    runtime = load_internvl_runtime(args.model_dir)
    pixels = load_image_pixels(image_path, runtime.device, runtime.dtype)
    module_paths = [f"language_model.model.layers.{index}.mlp.down_proj" for index in range(args.layers)]
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }

    with torch.inference_mode(), FfnActivationCollector(runtime.model, module_paths) as collector:
        response = runtime.model.chat(runtime.tokenizer, pixels, question, generation_config)
        importance = collector.importance()
        summary = collector.summary()

    output = {
        "model": Path(args.model_dir).name,
        "annotation": args.annotation,
        "sample_index": args.sample_index,
        "question": question,
        "expected_answer": expected_answer(sample),
        "response": response,
        "activation_summary": {
            "layer_count": summary.layer_count,
            "token_count": summary.token_count,
            "dimensions": list(summary.dimensions),
        },
        "importance_preview": {
            path: [round(float(value), 6) for value in tensor[:5]]
            for path, tensor in importance.items()
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
