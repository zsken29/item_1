from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from lop.predictor import load_search_samples, predict_ratios, train_predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local LOP pruning-ratio predictor.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--architecture", choices=["transformer", "bilstm", "mlp"], default="transformer")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-ratios", default="0.2,0.3,0.5")
    parser.add_argument("--output-dir", default="outputs/predictor_runs/default")
    args = parser.parse_args()

    samples = load_search_samples(args.samples)
    model, metrics = train_predictor(
        samples=samples,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        architecture=args.architecture,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "layer_count": model.layer_count,
            "architecture": args.architecture,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
        },
        output_dir / "checkpoint.pt",
    )
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in metrics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    requested = [float(value) for value in args.predict_ratios.split(",")]
    predictions = {
        str(value): [round(item, 6) for item in predict_ratios(model, value)]
        for value in requested
    }
    summary = {
        "samples": args.samples,
        "sample_count": len(samples),
        "layer_count": model.layer_count,
        "architecture": args.architecture,
        "epochs": args.epochs,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "final_metrics": metrics[-1],
        "predictions": predictions,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
