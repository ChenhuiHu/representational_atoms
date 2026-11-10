from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from representational_atoms.evaluation.data import resolve_normalization
from representational_atoms.evaluation.fidelity import evaluate_fidelity_and_sparsity
from representational_atoms.evaluation.sae_loader import decoder_matrix, load_sae_for_evaluation
from representational_atoms.evaluation.stability import evaluate_stability


def _format(value: str, context: Mapping) -> str:
    return str(value).format(**context)


def _resolve_templates(value: Any, context: Mapping) -> Any:
    if isinstance(value, str):
        return value.format(**context) if "{" in value else value
    if isinstance(value, list):
        return [_resolve_templates(v, context) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve_templates(v, context) for v in value)
    if isinstance(value, Mapping):
        return {k: _resolve_templates(v, context) for k, v in value.items()}
    return value


def _evaluation_points(exp: Mapping) -> list[dict]:
    """Normalize layer-only and arbitrary diffusion evaluation points."""
    if "points" in exp:
        points = []
        for i, raw in enumerate(exp["points"]):
            point = dict(raw)
            point.setdefault("name", f"point_{i}")
            points.append(point)
        if not points:
            raise ValueError("experiment.points is empty")
        return points

    if "layers" in exp:
        return [{"name": f"layer_{int(layer)}", "layer": int(layer)} for layer in exp["layers"]]

    raise ValueError("Evaluation config requires experiment.layers or experiment.points")


def evaluate_fidelity_stability(config: Mapping) -> Path:
    exp = config.get("experiment", {})
    model_name = str(exp.get("model", "unknown"))
    dataset = str(exp.get("dataset", "unknown"))
    position = str(exp.get("position", "unknown"))

    sae_cfg = dict(config["sae"])
    architecture = str(sae_cfg.get("architecture", sae_cfg.get("source", "sae"))).lower()
    k = sae_cfg.get("k", "")
    model_slug = model_name.replace("/", "__")
    base_context = {
        "architecture": architecture,
        "k": k,
        "model": model_name,
        "model_slug": model_slug,
        "dataset": dataset,
        "position": position,
    }

    output_cfg = config.get("output", {})
    output_root = Path(
        _format(output_cfg.get("root", "results/fidelity_stability"), base_context)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    eval_cfg = config.get("evaluation", {})
    device = torch.device(eval_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    stats_batch_size = int(eval_cfg.get("stats_batch_size", 16384))
    inference_batch_size = int(eval_cfg.get("inference_batch_size", 4096))
    save_sparse = bool(eval_cfg.get("save_sparse_activations", True))
    stability_cfg = eval_cfg.get("stability", {})

    # Compatibility with v3 configuration while preferring the notebook-derived policy.
    metric_policy = str(stability_cfg.get("metric_policy", "auto"))
    singular_policy = str(stability_cfg.get("singular_policy", "pinv"))
    if "rank_deficient_policy" in stability_cfg and "singular_policy" not in stability_cfg:
        singular_policy = str(stability_cfg["rank_deficient_policy"])

    all_metrics = []
    all_stability = []
    points = _evaluation_points(exp)

    for point_index, point in enumerate(points):
        context = {
            **base_context,
            **point,
            "point_index": point_index,
            "point_name": point["name"],
        }

        rep_cfg = _resolve_templates(dict(config["representations"]), context)
        if "path_template" in rep_cfg:
            rep_cfg["path"] = _format(rep_cfg.pop("path_template"), context)
        if "selected_ids_path_template" in rep_cfg:
            rep_cfg["selected_ids_path"] = _format(
                rep_cfg.pop("selected_ids_path_template"), context
            )

        max_samples = rep_cfg.get("max_samples")
        max_samples = None if max_samples is None else int(max_samples)

        sae, parameter_path = load_sae_for_evaluation(sae_cfg, context, device=device)
        normalization = resolve_normalization(
            rep_cfg,
            config.get("preprocessing", {}),
            stats_batch_size=stats_batch_size,
            max_samples=max_samples,
        )

        subdir_template = output_cfg.get("subdir_template")
        if subdir_template:
            point_dir = output_root / _format(str(subdir_template), context)
        else:
            point_dir = output_root / str(point["name"])

        metadata = {
            "evaluation_point": str(point["name"]),
            "point_index": int(point_index),
            "model_name": model_name,
            "dataset": dataset,
            "position": position,
            "sae_source": str(sae_cfg.get("source", "trained")),
            "sae_type": architecture,
            "variant": str(sae_cfg.get("variant", "")),
            "parameter_path": str(parameter_path),
            "max_samples": max_samples,
        }
        for key, value in point.items():
            if key != "name":
                metadata[key] = value
        if rep_cfg.get("selected_ids_path"):
            metadata["selected_ids_path"] = str(rep_cfg["selected_ids_path"])
        if rep_cfg.get("format") in {"webdataset_tar", "legacy_spatial_dict"}:
            metadata["sample_size"] = int(rep_cfg.get("expected_samples", max_samples or 0)) or None

        metrics = evaluate_fidelity_and_sparsity(
            sae,
            rep_cfg,
            normalization=normalization,
            output_dir=point_dir,
            metadata=metadata,
            device=device,
            max_samples=max_samples,
            batch_size=inference_batch_size,
            save_sparse_activations=save_sparse,
        )

        feature_ids = torch.load(point_dir / "features_id.pt", map_location="cpu")
        k_values = torch.load(point_dir / "sparsity.pt", map_location="cpu")
        stability = evaluate_stability(
            decoder_matrix(sae),
            feature_ids,
            k_values,
            output_dir=point_dir,
            metadata=metadata,
            device=device,
            max_mu_samples=int(stability_cfg.get("max_mu_samples", 5_000_000)),
            pair_batch_size=int(stability_cfg.get("pair_batch_size", 100_000)),
            random_seed=int(stability_cfg.get("random_seed", 42)),
            save_mu_samples=bool(stability_cfg.get("save_mu_samples", False)),
            metric_policy=metric_policy,
            singular_policy=singular_policy,
            full_metric_chunk_size=int(stability_cfg.get("full_metric_chunk_size", 4096)),
        )
        all_metrics.append(metrics)
        all_stability.append(stability)
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(output_root / "all_points_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    with open(output_root / "all_points_stability.json", "w", encoding="utf-8") as f:
        json.dump(all_stability, f, indent=2)

    # Backward-compatible filenames for layer-only runs.
    if "layers" in exp and "points" not in exp:
        with open(output_root / "all_layers_metrics.json", "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        with open(output_root / "all_layers_stability.json", "w", encoding="utf-8") as f:
            json.dump(all_stability, f, indent=2)
    return output_root
