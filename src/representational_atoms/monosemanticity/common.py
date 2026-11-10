from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def resolve_templates(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**context) if "{" in value else value
    if isinstance(value, list):
        return [resolve_templates(v, context) for v in value]
    if isinstance(value, tuple):
        return tuple(resolve_templates(v, context) for v in value)
    if isinstance(value, Mapping):
        return {k: resolve_templates(v, context) for k, v in value.items()}
    return value


def experiment_conditions(experiment: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "points" in experiment:
        out: list[dict[str, Any]] = []
        for i, raw in enumerate(experiment["points"]):
            point = dict(raw)
            point.setdefault("name", f"point_{i}")
            out.append(point)
        if not out:
            raise ValueError("experiment.points is empty")
        return out
    if "layers" in experiment:
        return [
            {"name": f"layer_{int(layer)}", "layer": int(layer)}
            for layer in experiment["layers"]
        ]
    raise ValueError("Monosemanticity config requires experiment.layers or experiment.points")


def base_context(config: Mapping[str, Any]) -> dict[str, Any]:
    exp = config.get("experiment", {})
    sae = config.get("sae", {})
    model = str(exp.get("model", "unknown"))
    return {
        "dataset": str(exp.get("dataset", "unknown")),
        "model": model,
        "model_slug": model.replace("/", "__"),
        "position": str(exp.get("position", "unknown")),
        "architecture": str(sae.get("architecture", "bijumprelu")).lower(),
    }


def condition_seed_offset(condition: Mapping[str, Any], *, mode: str) -> int:
    """Notebook-compatible deterministic condition offsets.

    mode='pool':
      layer -> layer
      timestep -> timestep
      layer+step -> layer*10 + step

    mode='review':
      layer -> layer
      timestep -> timestep
      layer+step -> layer*100 + step
    """
    layer = condition.get("layer")
    timestep = condition.get("timestep")
    step = condition.get("step")
    if isinstance(timestep, str):
        timestep = int(timestep.removeprefix("t"))
    if isinstance(step, str):
        step = int(step.removeprefix("step"))

    if layer is not None and step is not None:
        return int(layer) * (10 if mode == "pool" else 100) + int(step)
    if timestep is not None:
        return int(timestep)
    if layer is not None:
        return int(layer)
    return 0


def feature_example_seed(
    base_seed: int,
    condition: Mapping[str, Any],
    feature_id: int,
) -> int:
    layer = condition.get("layer")
    timestep = condition.get("timestep")
    step = condition.get("step")
    if isinstance(timestep, str):
        timestep = int(timestep.removeprefix("t"))
    if isinstance(step, str):
        step = int(step.removeprefix("step"))

    seed = int(base_seed) * 1_000_003 + int(feature_id)
    if layer is not None:
        seed += int(layer) * 10_007
    if timestep is not None:
        seed += int(timestep) * 10_007
    if step is not None:
        seed += int(step) * 101
    return seed


def condition_output_dir(root: str | Path, condition: Mapping[str, Any], template: str | None = None) -> Path:
    root = Path(root)
    if template:
        return root / template.format(**condition)
    return root / str(condition["name"])
