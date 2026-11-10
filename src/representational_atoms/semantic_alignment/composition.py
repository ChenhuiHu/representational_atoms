from __future__ import annotations

import json
from pathlib import Path

from .core import run_alignment, summarize_results


def run_compositional_alignment(
    prepared,
    *,
    output_dir: str | Path,
    min_precision: float = 0.5,
    min_overlap: int = 1,
    min_f1: float = 0.0,
    max_candidates_per_sememe: int = 4000,
    max_features_to_merge: int | None = 800,
    require_positive_gain: bool = True,
    sememe_batch_size: int = 64,
):
    flat_features, sememe_sets, sememe_meta, feature_matrix, sememe_matrix, feature_sizes, sememe_sizes, idx_to_word = prepared
    results = run_alignment(
        sememe_sets=sememe_sets,
        sememe_meta=sememe_meta,
        flat_features=flat_features,
        feature_matrix=feature_matrix,
        sememe_matrix=sememe_matrix,
        feature_sizes=feature_sizes,
        sememe_sizes=sememe_sizes,
        idx_to_word=idx_to_word,
        min_precision=min_precision,
        min_overlap=min_overlap,
        min_f1=min_f1,
        max_candidates_per_sememe=max_candidates_per_sememe,
        max_features_to_merge=max_features_to_merge,
        require_positive_gain=require_positive_gain,
        sememe_batch_size=sememe_batch_size,
        run_greedy=True,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(results, include_greedy=True)
    summary = {
        "min_precision": float(min_precision),
        "min_overlap": int(min_overlap),
        "min_f1": float(min_f1),
        "max_candidates_per_sememe": int(max_candidates_per_sememe),
        "max_features_to_merge": None if max_features_to_merge is None else int(max_features_to_merge),
        "require_positive_gain": bool(require_positive_gain),
        "sememe_batch_size": int(sememe_batch_size),
        **summary,
    }
    with open(output_dir / "composition_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(output_dir / "composition_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return results, summary
