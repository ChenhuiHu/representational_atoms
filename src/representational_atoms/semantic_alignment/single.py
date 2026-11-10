from __future__ import annotations

import json
from pathlib import Path

from .core import run_alignment, summarize_results


def run_single_atom_alignment(prepared, *, output_dir: str | Path, sememe_batch_size: int = 64):
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
        sememe_batch_size=sememe_batch_size,
        run_greedy=False,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "single_atom_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    summary = summarize_results(results, include_greedy=False)
    with open(output_dir / "single_atom_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return results, summary
