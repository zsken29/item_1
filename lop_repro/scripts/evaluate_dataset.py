from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from lop.adapters import inspect_model_ffn
from lop.data import annotation_files, iter_samples, load_catalog
from lop.data.prompts import build_question
from lop.eval import score_mme, score_response
from lop.importance import load_importance
from lop.models import generate_response, load_chat_runtime
from lop.pruning import apply_ffn_pruning, load_layer_ratios, uniform_layer_ratios


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a supported local MLLM on one configured dataset.")
    parser.add_argument("--runtime", choices=["internvl", "qwen25_vl"], default="internvl")
    parser.add_argument("--model-dir", default="models/InternVL2_5-1B")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", default="configs/data/datasets.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--importance-dir")
    parser.add_argument("--prune-ratio", type=float)
    parser.add_argument("--layer-ratios")
    parser.add_argument("--prune-layers", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.prune_ratio is not None and args.layer_ratios is not None:
        raise ValueError("--prune-ratio and --layer-ratios are mutually exclusive")
    if args.importance_dir is None and (args.prune_ratio is not None or args.layer_ratios is not None):
        raise ValueError("--importance-dir is required when pruning is enabled")

    start = time.time()
    root = Path.cwd()
    catalog = load_catalog(args.dataset_config)
    info = catalog.get(args.dataset)
    runtime = load_chat_runtime(args.runtime, args.model_dir)
    pruning_summary = None
    if args.importance_dir is not None:
        spec = inspect_model_ffn(args.model_dir)
        importance = load_importance(args.importance_dir)
        layer_count = len(importance) if args.prune_layers is None else args.prune_layers
        if args.layer_ratios is None:
            if args.prune_ratio is None:
                raise ValueError("provide --prune-ratio or --layer-ratios with --importance-dir")
            layer_ratios = uniform_layer_ratios(spec, args.prune_ratio, layer_count=layer_count)
        else:
            layer_ratios = load_layer_ratios(args.layer_ratios, spec, layer_count=layer_count)
        pruning_summary = apply_ffn_pruning(runtime.model, spec, importance, layer_ratios)

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    rows = []
    samples = []
    predicted_labels = []
    correct = 0
    with torch.inference_mode():
        pbar = tqdm(total=args.limit, desc=info.name, unit="sample") if args.limit is not None else tqdm(desc=info.name, unit="sample")
        for annotation in annotation_files(info):
            for sample in iter_samples(annotation):
                response = generate_response(
                    runtime,
                    root / sample["images"][0]["path"],
                    build_question(sample),
                    generation_config,
                )
                result = score_response(sample, response)
                correct += int(result.correct)
                samples.append(sample)
                predicted_labels.append(result.predicted)
                rows.append(
                    {
                        "annotation": _display_path(annotation, root),
                        "source_row": sample["source_row"],
                        "response": response,
                        "expected": result.expected,
                        "predicted": result.predicted,
                        "correct": result.correct,
                        "metric": result.metric,
                    }
                )
                pbar.update(1)
                if args.limit is not None and len(rows) == args.limit:
                    break
            if args.limit is not None and len(rows) == args.limit:
                break
        pbar.close()
    if not rows:
        raise ValueError(f"no samples evaluated for dataset {args.dataset}")

    summary = {
        "model": Path(args.model_dir).name,
        "runtime": args.runtime,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "limit": args.limit,
        "accuracy": correct / len(rows),
        "correct": correct,
        "total": len(rows),
        "elapsed_seconds": round(time.time() - start, 3),
        "pruning": None if pruning_summary is None else {
            "original_neurons": pruning_summary.original_neurons,
            "kept_neurons": pruning_summary.kept_neurons,
            "actual_ratio": pruning_summary.actual_ratio,
        },
        "rows": rows,
    }
    if args.dataset == "mme":
        mme_score = score_mme(samples, predicted_labels)
        summary["mme"] = {
            "perception": mme_score.perception,
            "cognition": mme_score.cognition,
            "total": mme_score.total,
            "tasks": [
                {
                    "task": task.task,
                    "questions": task.questions,
                    "image_groups": task.image_groups,
                    "accuracy": task.accuracy,
                    "accuracy_plus": task.accuracy_plus,
                    "score": task.score,
                }
                for task in mme_score.tasks
            ],
        }

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _display_path(path: Path, root: Path) -> str:
    resolved = path if path.is_absolute() else root / path
    return str(resolved.relative_to(root))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
