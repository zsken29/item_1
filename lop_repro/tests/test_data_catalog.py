from __future__ import annotations

import json
import unittest
from pathlib import Path

from lop.data import load_catalog, summarize_dataset


def _test_root(name: str) -> Path:
    path = Path.cwd() / "outputs" / "unit_tests" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


class DataCatalogTest(unittest.TestCase):
    def test_summary_checks_rows_and_images(self) -> None:
        root = _test_root("data_catalog")
        dataset = root / "data" / "demo"
        annotations = dataset / "annotations"
        images = dataset / "images"
        config_dir = root / "configs" / "data"
        annotations.mkdir(parents=True, exist_ok=True)
        images.mkdir(exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        image_path = images / "a.png"
        image_path.write_bytes(b"png")
        sample = {
            "dataset": "demo",
            "source_repo": "local/demo",
            "source_row": 0,
            "fields": {"question": "q", "answer": "a"},
            "images": [
                {
                    "path": "data/demo/images/a.png",
                    "source_path": "a.png",
                    "bytes": 3,
                    "sha256": "abc",
                    "column": "image",
                }
            ],
        }
        (annotations / "sample.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
        config = {
            "datasets": [
                {
                    "name": "demo",
                    "source": "local/demo",
                    "local_dir": "data/demo",
                    "annotation_dir": "data/demo/annotations",
                    "image_dir": "data/demo/images",
                    "rows": 1,
                    "annotation_files": 1,
                    "unique_images": 1,
                    "purpose": "test",
                }
            ]
        }
        config_path = config_dir / "datasets.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        catalog = load_catalog(config_path)
        summary = summarize_dataset(catalog, "demo")

        self.assertEqual(summary.rows, 1)
        self.assertEqual(summary.image_refs, 1)
        self.assertEqual(summary.field_names, ("answer", "question"))


if __name__ == "__main__":
    unittest.main()
