#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from representational_atoms.config import apply_overrides, load_yaml
from representational_atoms.monosemanticity.runner import prepare_monosemanticity


def main():
    parser = argparse.ArgumentParser(description="Prepare monosemanticity candidate pools and review sets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["all", "candidates", "review"], default="all")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    config = apply_overrides(load_yaml(args.config), args.overrides)
    root = prepare_monosemanticity(config, stage=args.stage)
    print(f"Monosemanticity artifacts written to: {Path(root).resolve()}")


if __name__ == "__main__":
    main()
