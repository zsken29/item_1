from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from lop.adapters import inspect_model_ffn


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model FFN structure for LOP pruning.")
    parser.add_argument("model_dir")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    spec = inspect_model_ffn(Path(args.model_dir))
    if args.json:
        print(json.dumps(asdict(spec), ensure_ascii=False, indent=2))
        return

    verified = sum(1 for layer in spec.layers if layer.weight_status == "verified")
    print(f"name: {spec.name}")
    print(f"type: {spec.model_type}")
    print(f"architecture: {spec.architecture}")
    print(f"supported: {spec.supported}")
    print(f"layers: {len(spec.layers)}")
    print(f"verified_layers: {verified}")
    print(f"note: {spec.note}")
    for layer in spec.layers[:3]:
        print(
            f"layer {layer.index}: block={layer.block_path} hidden={layer.hidden_size} "
            f"intermediate={layer.intermediate_size} weights={layer.weight_status}"
        )
    if len(spec.layers) > 3:
        last = spec.layers[-1]
        print(f"layer {last.index}: block={last.block_path} weights={last.weight_status}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

