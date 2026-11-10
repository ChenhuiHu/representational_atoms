#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from representational_atoms.monosemanticity.scores import summarize_annotations


def main():
    parser = argparse.ArgumentParser(description="Aggregate completed M/N monosemanticity annotations.")
    parser.add_argument("--input", required=True, help="annotations.jsonl file or a root containing completed annotations.jsonl files")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    _, summary = summarize_annotations(args.input, args.output_dir)
    print(f"Aggregated {len(summary)} condition(s).")
    if args.output_dir:
        print(f"Results written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
