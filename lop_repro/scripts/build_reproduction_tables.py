from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lop.report import build_table1_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproduction tables from local metrics and paper references.")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--paper-reference")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    markdown = build_table1_markdown(args.runs, args.paper_reference)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
