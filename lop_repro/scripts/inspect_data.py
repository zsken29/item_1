from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lop.data import load_catalog, summarize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect prepared benchmark datasets.")
    parser.add_argument("--config", default="configs/data/datasets.json")
    parser.add_argument("--dataset", action="append")
    args = parser.parse_args()

    catalog = load_catalog(Path(args.config))
    names = args.dataset if args.dataset else sorted(catalog.datasets)
    for name in names:
        summary = summarize_dataset(catalog, name)
        fields = ", ".join(summary.field_names)
        print(
            f"{summary.name}: files={summary.annotation_files} rows={summary.rows} "
            f"image_refs={summary.image_refs} images={summary.unique_image_files}"
        )
        print(f"  fields: {fields}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

