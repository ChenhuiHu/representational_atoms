from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from representational_atoms.sae.trainer import train_sae


def _resolve_templates(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**context) if "{" in value else value
    if isinstance(value, list):
        return [_resolve_templates(v, context) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve_templates(v, context) for v in value)
    if isinstance(value, Mapping):
        return {k: _resolve_templates(v, context) for k, v in value.items()}
    return value


def _points(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    exp = config.get("experiment")
    if not exp:
        return []
    if "points" in exp:
        return [dict(point) for point in exp["points"]]
    if "layers" in exp:
        return [{"layer": int(layer), "name": f"layer_{int(layer)}"} for layer in exp["layers"]]
    return []


def materialize_training_config(config: Mapping[str, Any], point: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one layer/point from a templated training experiment config."""
    out = deepcopy(dict(config))
    exp = dict(out.get("experiment", {}))
    sae_cfg = dict(out.get("sae", {}))
    architecture = str(sae_cfg.get("architecture", "sae")).lower()

    allowed = exp.get("allowed_architectures")
    if allowed is not None:
        allowed_norm = {str(x).lower() for x in allowed}
        if architecture not in allowed_norm:
            raise ValueError(
                f"Architecture {architecture!r} is not enabled for this experiment; "
                f"allowed={sorted(allowed_norm)}"
            )

    model = str(exp.get("model", "unknown"))
    context = {
        "architecture": architecture,
        "k": sae_cfg.get("k", ""),
        "model": model,
        "model_slug": model.replace("/", "__"),
        "dataset": str(exp.get("dataset", "unknown")),
        "position": str(exp.get("position", "unknown")),
        **dict(point),
    }

    out = _resolve_templates(out, context)

    data_cfg = dict(out.get("data", {}))
    if "path_template" in data_cfg:
        data_cfg["path"] = data_cfg.pop("path_template")
    out["data"] = data_cfg

    output_cfg = dict(out.get("output", {}))
    if "dir_template" in output_cfg:
        output_cfg["dir"] = output_cfg.pop("dir_template")
    out["output"] = output_cfg

    out["experiment_point"] = dict(point)
    return out


def train_sae_experiment(config: Mapping[str, Any]) -> list[Path]:
    """Train either one SAE run or a layer/point grid from a single config."""
    points = _points(config)
    if not points:
        return [train_sae(config)]

    checkpoints: list[Path] = []
    for point in points:
        run_cfg = materialize_training_config(config, point)
        checkpoints.append(train_sae(run_cfg))
    return checkpoints
