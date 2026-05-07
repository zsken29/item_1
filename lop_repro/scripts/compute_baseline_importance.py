from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from lop.adapters import inspect_model_ffn
from lop.importance import (
    ImportanceRecord,
    flap_importance,
    load_importance,
    save_importance,
    wanda_importance,
    weight_magnitude_importance,
)
from lop.models import load_chat_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute structured baseline FFN importance.")
    parser.add_argument("--runtime", choices=["internvl", "qwen25_vl"], default="internvl")
    parser.add_argument("--model-dir", default="models/InternVL2_5-1B")
    parser.add_argument("--method", choices=["magnitude", "wanda", "flap"], required=True)
    parser.add_argument("--activation-importance-dir")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    start = time.time()
    spec = inspect_model_ffn(args.model_dir)
    runtime = load_chat_runtime(args.runtime, args.model_dir)
    layers = spec.layers if args.layers is None else spec.layers[: args.layers]
    if not layers:
        raise ValueError("layers must be positive")
    spec = replace(spec, layers=tuple(layers))

    if args.method == "magnitude":
        all_scores = weight_magnitude_importance(runtime.model, spec)
    elif args.method == "wanda":
        if args.activation_importance_dir is None:
            raise ValueError("--activation-importance-dir is required for wanda")
        activation = load_importance(args.activation_importance_dir)
        all_scores = wanda_importance(runtime.model, spec, activation)
    else:
        if args.activation_importance_dir is None:
            raise ValueError("--activation-importance-dir is required for flap")
        fluctuation = load_importance(args.activation_importance_dir)
        all_scores = flap_importance(runtime.model, spec, fluctuation)

    records = [
        ImportanceRecord(layer.index, layer.activation_path, all_scores[layer.index])
        for layer in layers
    ]
    summary = {
        "model": Path(args.model_dir).name,
        "runtime": args.runtime,
        "method": args.method,
        "layers": len(records),
        "activation_importance_dir": args.activation_importance_dir,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    save_importance(args.output_dir, records, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
