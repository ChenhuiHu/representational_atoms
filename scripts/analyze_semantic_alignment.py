#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from representational_atoms.semantic_alignment.analysis import analyze_composition_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze layer usage and cross-layer width for HowNet compositional alignment."
    )
    parser.add_argument("--results", required=True, help="Path to composition_results.json")
    parser.add_argument("--output", default=None, help="Output JSON path. Defaults next to the results file.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k greedy features for auxiliary width analysis.")
    args = parser.parse_args()

    results_path = Path(args.results)
    output_path = Path(args.output) if args.output else results_path.with_name("composition_analysis.json")
    summary = analyze_composition_file(results_path, output_path=output_path, top_k=args.top_k)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved semantic-alignment analysis to: {output_path}")


if __name__ == "__main__":
    main()
