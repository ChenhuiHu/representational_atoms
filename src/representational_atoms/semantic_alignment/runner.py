from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .composition import run_compositional_alignment
from .data import (
    build_sparse_word_matrices,
    load_all_layer_features,
    load_prepared,
    load_sememe_sets,
    save_prepared,
)
from .grid import run_grid_search
from .single import run_single_atom_alignment


def prepare_alignment(config: Mapping[str, Any], output_root: Path) -> Path:
    exp = config["experiment"]
    layers = [int(x) for x in exp["layers"]]
    feature_template = str(config["features"]["path_template"])
    flat_features = load_all_layer_features(feature_template, layers)
    sememe_sets, sememe_meta = load_sememe_sets(config.get("hownet", {}))
    feature_matrix, sememe_matrix, feature_sizes, sememe_sizes, _, idx_to_word = build_sparse_word_matrices(
        flat_features,
        sememe_sets,
    )
    prepared_dir = output_root / "prepared"
    save_prepared(
        prepared_dir,
        flat_features,
        sememe_sets,
        sememe_meta,
        feature_matrix,
        sememe_matrix,
        feature_sizes,
        sememe_sizes,
        idx_to_word,
    )
    stats = {
        "num_layers": len(layers),
        "num_feature_candidates": len(flat_features),
        "num_sememes": len(sememe_sets),
        "vocabulary_size": len(idx_to_word),
        "top_k_examples": config["features"].get("top_k_examples", 10000),
    }
    with open(output_root / "alignment_input_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return prepared_dir


def _prepared(config: Mapping[str, Any], output_root: Path):
    prepared_dir = output_root / "prepared"
    required = [prepared_dir / "feature_matrix.npz", prepared_dir / "sememe_matrix.npz", prepared_dir / "flat_features.jsonl"]
    if not all(p.exists() for p in required):
        prepare_alignment(config, output_root)
    return load_prepared(prepared_dir)


def run_semantic_alignment(config: Mapping[str, Any], *, stage: str = "all") -> Path:
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "semantic_alignment_config.json", "w", encoding="utf-8") as f:
        json.dump(dict(config), f, ensure_ascii=False, indent=2)

    stage = str(stage).lower()
    if stage not in {"prepare", "single", "composition", "grid", "all"}:
        raise ValueError(f"Unsupported semantic alignment stage: {stage}")
    if stage in {"prepare", "all"}:
        prepare_alignment(config, output_root)
        if stage == "prepare":
            return output_root

    prepared = _prepared(config, output_root)
    search = config.get("search", {})
    if stage in {"single", "all"}:
        run_single_atom_alignment(
            prepared,
            output_dir=output_root / "single_atom",
            sememe_batch_size=int(search.get("sememe_batch_size", 64)),
        )

    if stage in {"composition", "all"}:
        run_compositional_alignment(
            prepared,
            output_dir=output_root / "composition",
            min_precision=float(search.get("min_precision", 0.5)),
            min_overlap=int(search.get("min_overlap", 1)),
            min_f1=float(search.get("min_f1", 0.0)),
            max_candidates_per_sememe=int(search.get("max_candidates_per_sememe", 4000)),
            max_features_to_merge=search.get("max_features_to_merge", 800),
            require_positive_gain=bool(search.get("require_positive_gain", True)),
            sememe_batch_size=int(search.get("sememe_batch_size", 64)),
        )

    if stage in {"grid", "all"}:
        grid = config.get("grid_search", {})
        run_grid_search(
            prepared,
            output_dir=output_root / "grid_search" / "initial",
            precision_values=grid.get("min_precision_values", [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
            max_features_values=grid.get("max_features_to_merge_values", [20,40,60,80,100,120,140,160,180]),
            min_overlap=int(search.get("min_overlap", 1)),
            min_f1=float(search.get("min_f1", 0.0)),
            max_candidates_per_sememe=int(search.get("max_candidates_per_sememe", 4000)),
            require_positive_gain=bool(search.get("require_positive_gain", True)),
            sememe_batch_size=int(search.get("sememe_batch_size", 64)),
            save_per_run_results=bool(grid.get("save_per_run_results", False)),
        )
        extended = grid.get("extended", {})
        if bool(extended.get("enabled", True)):
            run_grid_search(
                prepared,
                output_dir=output_root / "grid_search" / "extended",
                precision_values=[float(extended.get("min_precision", 0.5))],
                max_features_values=extended.get("max_features_to_merge_values", [200,400,600,800]),
                min_overlap=int(search.get("min_overlap", 1)),
                min_f1=float(search.get("min_f1", 0.0)),
                max_candidates_per_sememe=int(search.get("max_candidates_per_sememe", 4000)),
                require_positive_gain=bool(search.get("require_positive_gain", True)),
                sememe_batch_size=int(search.get("sememe_batch_size", 64)),
                save_per_run_results=bool(grid.get("save_per_run_results", False)),
            )
    return output_root
