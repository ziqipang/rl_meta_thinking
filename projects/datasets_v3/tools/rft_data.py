#!/usr/bin/env python3
"""Local v3 entrypoint for the shared RFT-data builder."""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    shared_script = Path(__file__).resolve().parents[2] / "datasets" / "rft_data.py"
    runpy.run_path(str(shared_script), run_name="__main__")
