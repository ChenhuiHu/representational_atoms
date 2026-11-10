from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_fraction(value: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", value)
    if not m:
        raise ValueError(f"Expected M/N, got {value!r}")
    a, b = int(m.group(1)), int(m.group(2))
    if b <= 0 or a < 0 or a > b:
        raise ValueError(f"Invalid monosemanticity fraction: {value!r}")
    return a, b


def _iter_annotation_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        yield from sorted(path.rglob("annotations.jsonl"))


def _record_score(record: dict) -> float:
    score = record.get("score")
    if score is not None:
        score = float(score)
        if not 0 <= score <= 1:
            raise ValueError(f"score must be in [0,1], got {score}")
        return score
    if record.get("fraction") is not None:
        a, b = parse_fraction(str(record["fraction"]))
        return a / b
    a, b = record.get("group_size"), record.get("num_examples")
    if a is None or b is None:
        raise ValueError("Annotation requires score, fraction, or group_size+num_examples")
    a, b = int(a), int(b)
    if b <= 0 or a < 0 or a > b:
        raise ValueError(f"Invalid group_size/num_examples: {a}/{b}")
    return a / b


def summarize_annotations(input_path: str | Path, output_dir: str | Path | None = None) -> tuple[list[dict], list[dict]]:
    input_path = Path(input_path)
    rows: list[dict] = []
    for path in _iter_annotation_files(input_path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec = dict(rec)
                rec["monosemanticity"] = _record_score(rec)
                rec["annotation_file"] = str(path)
                rows.append(rec)
    if not rows:
        raise RuntimeError(f"No completed annotations found under {input_path}")

    grouped: dict[str, list[float]] = defaultdict(list)
    for rec in rows:
        condition = str(rec.get("name", rec.get("condition", rec.get("evaluation_point", "unknown"))))
        grouped[condition].append(float(rec["monosemanticity"]))

    summary = []
    for condition, scores in grouped.items():
        mean = sum(scores) / len(scores)
        if len(scores) > 1:
            variance = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
            sd = math.sqrt(variance)
        else:
            sd = 0.0
        summary.append({"condition": condition, "n_features": len(scores), "mean_monosemanticity": mean, "sd": sd})

    if output_dir is None:
        output_dir = input_path if input_path.is_dir() else input_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "feature_scores.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(output_dir / "monosemanticity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(output_dir / "monosemanticity_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "n_features", "mean_monosemanticity", "sd"])
        writer.writeheader(); writer.writerows(summary)
    return rows, summary
