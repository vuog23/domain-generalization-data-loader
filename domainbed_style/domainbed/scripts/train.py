"""Compatibility entry point for the local YAML runner."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runner = Path(__file__).resolve().parents[2] / "train.py"
    runpy.run_path(str(runner), run_name="__main__")
