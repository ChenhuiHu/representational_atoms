#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from representational_atoms.config import apply_overrides, deep_merge, load_yaml  # noqa: E402
from representational_atoms.evaluation import evaluate_fidelity_stability  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAE fidelity and stability.")
    parser.add_argument("--config", required=True, help="Evaluation/model/data YAML config.")
    parser.add_argument(
        "--sae-config",
        default=None,
        help="SAE architecture YAML config for trained SAEs. Optional for external SAEs such as GemmaScope.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a nested config value; may be repeated.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.sae_config is not None:
        config = deep_merge(config, load_yaml(args.sae_config))
    config = apply_overrides(config, args.set)
    output = evaluate_fidelity_stability(config)
    print(f"Evaluation results: {output}")


if __name__ == "__main__":
    main()
