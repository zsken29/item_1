from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ImportanceRecord:
    layer_index: int
    module_path: str
    tensor: torch.Tensor


def save_importance(output_dir: str | Path, records: list[ImportanceRecord], summary: dict) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for record in records:
        torch.save(
            {
                "layer_index": record.layer_index,
                "module_path": record.module_path,
                "importance": record.tensor.detach().cpu(),
            },
            path / f"layer_{record.layer_index:03d}.pt",
        )
    (path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_importance(output_dir: str | Path) -> dict[int, torch.Tensor]:
    path = Path(output_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"missing importance directory: {path}")
    records: dict[int, torch.Tensor] = {}
    for layer_path in sorted(path.glob("layer_*.pt")):
        payload = torch.load(layer_path, map_location="cpu", weights_only=False)
        layer_index = int(payload["layer_index"])
        if layer_index in records:
            raise ValueError(f"duplicate importance for layer {layer_index}")
        records[layer_index] = payload["importance"].float()
    if not records:
        raise FileNotFoundError(f"no layer_*.pt files in {path}")
    return records
