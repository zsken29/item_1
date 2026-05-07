from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from lop.predictor import build_predictor, predict_ratios


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict layer pruning ratios from a trained LOP predictor.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-ratio", type=float, required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_predictor(
        architecture=payload.get("architecture", "transformer"),
        layer_count=int(payload["layer_count"]),
        hidden_size=int(payload["hidden_size"]),
        num_layers=int(payload.get("num_layers", 2)),
        num_heads=int(payload.get("num_heads", 4)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    ratios = predict_ratios(model, args.target_ratio)
    output = {
        "checkpoint": args.checkpoint,
        "target_ratio": args.target_ratio,
        "layer_ratios": [round(value, 6) for value in ratios],
        "mean_ratio": round(sum(ratios) / len(ratios), 6),
    }
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
