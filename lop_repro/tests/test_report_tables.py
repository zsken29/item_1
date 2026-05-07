from __future__ import annotations

import json
import unittest
from pathlib import Path

from lop.report import build_table1_markdown


class ReportTableTest(unittest.TestCase):
    def test_builds_table1_from_manifest(self) -> None:
        root = Path.cwd() / "outputs" / "unit_tests" / "report_table"
        root.mkdir(parents=True, exist_ok=True)
        mme = root / "mme.json"
        mme.write_text(json.dumps({"mme": {"perception": 10.0, "cognition": 20.0}}), encoding="utf-8")
        manifest = root / "runs.json"
        manifest.write_text(
            json.dumps({"runs": [{"ratio": "0%", "method": "Dense", "metrics": {"mme": str(mme)}}]}),
            encoding="utf-8",
        )

        table = build_table1_markdown(manifest)

        self.assertIn("Dense", table)
        self.assertIn("10.00", table)


if __name__ == "__main__":
    unittest.main()
