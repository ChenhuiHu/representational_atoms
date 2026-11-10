#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from representational_atoms.config import apply_overrides, load_yaml  # noqa: E402
from representational_atoms.extraction import (  # noqa: E402
    run_flux2_extraction,
    run_llm_extraction,
    run_sd15_extraction,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract internal representations for Representational Atoms experiments."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML extraction config.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a nested config value; may be repeated, "
            "e.g. --set dataset.path=... --set extraction.layers='[0,12,25]'"
        ),
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    config = apply_overrides(config, args.set)

    extractor = config["extractor"]

    if extractor == "llm":
        output = run_llm_extraction(config)
    elif extractor == "sd15":
        output = run_sd15_extraction(config)
    elif extractor == "flux2":
        output = run_flux2_extraction(config)
    else:
        raise ValueError(f"Unknown extractor: {extractor}")

    print(f"Representations saved to: {output}")


if __name__ == "__main__":
    main()