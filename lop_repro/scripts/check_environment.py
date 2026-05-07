from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys


REQUIRED_PACKAGES = ("torch", "transformers", "safetensors", "PIL", "flash_attn", "bitsandbytes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local runtime for LOP reproduction.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _package_status(name) for name in REQUIRED_PACKAGES},
        "cuda": _cuda_status(),
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"python: {report['python']}")
        print(f"platform: {report['platform']}")
        for name, status in report["packages"].items():
            print(f"{name}: {status}")
        print(f"cuda: {report['cuda']}")

    missing = [name for name, status in report["packages"].items() if status == "missing"]
    if args.strict and missing:
        raise SystemExit(f"missing packages: {', '.join(missing)}")


def _package_status(name: str) -> str:
    if importlib.util.find_spec(name) is None:
        return "missing"
    module = __import__(name)
    version = getattr(module, "__version__", "installed")
    return str(version)


def _cuda_status() -> str:
    if importlib.util.find_spec("torch") is None:
        return "torch_missing"
    import torch

    if not torch.cuda.is_available():
        return "unavailable"
    return f"available:{torch.cuda.device_count()}"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
