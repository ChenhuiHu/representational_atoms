#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from representational_atoms.config import apply_overrides, deep_merge, load_yaml  # noqa: E402
from representational_atoms.sae import train_sae_experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train an SAE for Representational Atoms experiments.")
    parser.add_argument("--config", required=True, help="Training/data YAML config.")
    parser.add_argument("--sae-config", required=True, help="SAE architecture YAML config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a nested config value; may be repeated, e.g. --set data.path=... --set sae.expansion_factor=54",
    )
    args = parser.parse_args()

    config = deep_merge(load_yaml(args.config), load_yaml(args.sae_config))
    config = apply_overrides(config, args.set)
    checkpoints = train_sae_experiment(config)
    if len(checkpoints) == 1:
        print(f"Best checkpoint: {checkpoints[0]}")
    else:
        print(f"Completed {len(checkpoints)} training points.")
        for checkpoint in checkpoints:
            print(f"  {checkpoint}")


if __name__ == "__main__":
    main()
