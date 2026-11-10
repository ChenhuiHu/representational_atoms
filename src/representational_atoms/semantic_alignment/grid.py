from __future__ import annotations

import csv
import json
from pathlib import Path

from .composition import run_compositional_alignment


def run_grid_search(
    prepared,
    *,
    output_dir: str | Path,
    precision_values,
    max_features_values,
    min_overlap: int = 1,
    min_f1: float = 0.0,
    max_candidates_per_sememe: int = 4000,
    require_positive_gain: bool = True,
    sememe_batch_size: int = 64,
    save_per_run_results: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for min_precision in precision_values:
        for max_features in max_features_values:
            run_name = f"mp_{float(min_precision):.2f}_mfm_{int(max_features)}"
            run_dir = output_dir / run_name
            results, summary = run_compositional_alignment(
                prepared,
                output_dir=run_dir,
                min_precision=float(min_precision),
                min_overlap=min_overlap,
                min_f1=min_f1,
                max_candidates_per_sememe=max_candidates_per_sememe,
                max_features_to_merge=int(max_features),
                require_positive_gain=require_positive_gain,
                sememe_batch_size=sememe_batch_size,
            )
            if not save_per_run_results:
                results_path = run_dir / "composition_results.json"
                if results_path.exists():
                    results_path.unlink()
            summaries.append({"run_name": run_name, **summary})
    summaries.sort(key=lambda x: x.get("merged_mean_f1", 0.0), reverse=True)
    with open(output_dir / "grid_search_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    if summaries:
        with open(output_dir / "grid_search_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries
