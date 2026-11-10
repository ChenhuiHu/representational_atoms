#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from representational_atoms.config import apply_overrides, load_yaml  # noqa: E402
from representational_atoms.evaluation import summarize_runs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Aggregate fidelity/stability results across evaluated layers or diffusion evaluation points.")
    parser.add_argument("--config", required=True, help="Summary manifest YAML.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = apply_overrides(load_yaml(args.config), args.set)
    _, summary = summarize_runs(config)
    print(f"Summarized {len(summary)} runs.")


if __name__ == "__main__":
    main()
