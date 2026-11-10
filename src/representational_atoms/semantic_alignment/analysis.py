from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _finite_corr(x: np.ndarray, y: np.ndarray, fn):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"coefficient": None, "p_value": None, "n": int(x.size)}
    res = fn(x, y)
    # scipy returns either tuple-like or result object depending on version.
    coefficient = float(getattr(res, "statistic", res[0]))
    p_value = float(getattr(res, "pvalue", res[1]))
    return {"coefficient": coefficient, "p_value": p_value, "n": int(x.size)}


def analyze_composition_results(results: list[dict[str, Any]], *, top_k: int = 10) -> dict[str, Any]:
    """Reproduce the descriptive/correlation analyses used after 9.* greedy search.

    This intentionally stops at Pearson/Spearman correlation. The uploaded 9.*
    notebooks describe an equivalence test in markdown but do not contain its
    implementation, so no equivalence-test procedure is invented here.
    """
    best_single_layers: list[int] = []
    all_selected_layers: list[int] = []
    widths: list[int] = []
    topk_widths: list[int] = []
    merged_f1: list[float] = []

    for item in results:
        best_layer = item.get("best_single_feature", {}).get("layer")
        if best_layer is not None:
            best_single_layers.append(int(best_layer))

        selected = item.get("greedy_merge", {}).get("selected_features", [])
        layers = [int(feature["layer"]) for feature in selected]
        all_selected_layers.extend(layers)
        widths.append(len(set(layers)))
        topk_widths.append(len(set(layers[: int(top_k)])))
        merged_f1.append(float(item.get("greedy_merge", {}).get("merged_f1", 0.0)))

    widths_arr = np.asarray(widths, dtype=float)
    topk_widths_arr = np.asarray(topk_widths, dtype=float)
    merged_f1_arr = np.asarray(merged_f1, dtype=float)

    def dist(values: list[int]) -> dict[str, int]:
        return {str(k): int(v) for k, v in sorted(Counter(values).items())}

    summary: dict[str, Any] = {
        "num_sememes": int(len(results)),
        "best_single_layer_distribution": dist(best_single_layers),
        "selected_feature_layer_distribution": dist(all_selected_layers),
        "cross_layer_width": {
            "mean": float(widths_arr.mean()) if widths_arr.size else 0.0,
            "median": float(np.median(widths_arr)) if widths_arr.size else 0.0,
            "min": int(widths_arr.min()) if widths_arr.size else 0,
            "max": int(widths_arr.max()) if widths_arr.size else 0,
            "distribution": dist([int(x) for x in widths]),
        },
        "top_k_cross_layer_width": {
            "k": int(top_k),
            "mean": float(topk_widths_arr.mean()) if topk_widths_arr.size else 0.0,
            "median": float(np.median(topk_widths_arr)) if topk_widths_arr.size else 0.0,
            "min": int(topk_widths_arr.min()) if topk_widths_arr.size else 0,
            "max": int(topk_widths_arr.max()) if topk_widths_arr.size else 0,
            "distribution": dist([int(x) for x in topk_widths]),
        },
        "width_vs_merged_f1": {
            "pearson": _finite_corr(widths_arr, merged_f1_arr, pearsonr),
            "spearman": _finite_corr(widths_arr, merged_f1_arr, spearmanr),
        },
        "equivalence_test": {
            "implemented": False,
            "reason": (
                "The uploaded 9.* notebooks describe an |r| < 0.10 equivalence test "
                "in markdown but do not include the statistical implementation."
            ),
        },
    }
    return summary


def analyze_composition_file(
    results_path: str | Path,
    *,
    output_path: str | Path | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    results_path = Path(results_path)
    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    if not isinstance(results, list):
        raise TypeError("composition_results.json must contain a list of sememe results")
    summary = analyze_composition_results(results, top_k=top_k)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
