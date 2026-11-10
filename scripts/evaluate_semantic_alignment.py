#!/usr/bin/env python3
from __future__ import annotations

import argparse

from representational_atoms.config import apply_overrides, load_yaml
from representational_atoms.semantic_alignment.runner import run_semantic_alignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HowNet single-atom and compositional semantic alignment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["prepare", "single", "composition", "grid", "all"], default="all")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override YAML values, e.g. composition.max_atoms=800")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)
    out = run_semantic_alignment(cfg, stage=args.stage)
    print(f"Semantic alignment artifacts: {out}")


if __name__ == "__main__":
    main()
