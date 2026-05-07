from __future__ import annotations

import json
from pathlib import Path


TABLE1_COLUMNS = ("Ratio", "Method", "MME-P", "MME-R", "MMBench", "MMMU", "POPE", "Avg", "Speedup", "Source")


def build_table1_markdown(manifest_path: str | Path, paper_reference_path: str | Path | None = None) -> str:
    rows = []
    if paper_reference_path is not None:
        rows.extend(_load_reference_rows(paper_reference_path))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    for run in manifest["runs"]:
        rows.append(_run_to_row(run))
    lines = [
        "# Table 1 Reproduction",
        "",
        "| " + " | ".join(TABLE1_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in TABLE1_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column)) for column in TABLE1_COLUMNS) + " |")
    lines.append("")
    lines.append(
        "Note: local Avg averages available percentage metrics; MME-P/MME-R keep the original MME scores."
    )
    return "\n".join(lines) + "\n"


def _load_reference_rows(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = []
    for item in payload["table1"]:
        rows.append(
            {
                "Ratio": item["ratio"],
                "Method": item["method"],
                "MME-P": item["mme_p"],
                "MME-R": item["mme_r"],
                "MMBench": item["mmbench"],
                "MMMU": item["mmmu"],
                "POPE": item["pope"],
                "Avg": item["avg"],
                "Speedup": item.get("speedup", ""),
                "Source": "paper",
            }
        )
    return rows


def _run_to_row(run: dict) -> dict:
    metrics = run["metrics"]
    mme_p = ""
    mme_r = ""
    percentages = []
    if "mme" in metrics:
        mme = _read_json(metrics["mme"])
        mme_p = mme["mme"]["perception"]
        mme_r = mme["mme"]["cognition"]
    for dataset in ("mmbench", "mmmu", "pope"):
        if dataset in metrics:
            value = _read_json(metrics[dataset])["accuracy"] * 100.0
            percentages.append(value)
        else:
            value = ""
        metrics[dataset] = value
    avg = sum(percentages) / len(percentages) if percentages else ""
    return {
        "Ratio": run["ratio"],
        "Method": run["method"],
        "MME-P": mme_p,
        "MME-R": mme_r,
        "MMBench": metrics.get("mmbench", ""),
        "MMMU": metrics.get("mmmu", ""),
        "POPE": metrics.get("pope", ""),
        "Avg": avg,
        "Speedup": run.get("speedup", ""),
        "Source": run.get("source", "local"),
    }


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
