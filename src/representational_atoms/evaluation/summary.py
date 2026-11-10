from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ci95(values: Sequence[float]) -> tuple[float, float, float, int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(np.mean(x))
    if n == 1:
        return mean, math.nan, math.nan, 1
    sem = float(np.std(x, ddof=1) / np.sqrt(n))
    margin = float(t.ppf(0.975, df=n - 1) * sem)
    return mean, mean - margin, mean + margin, n


def _point_directories(run: Mapping) -> list[tuple[str, Path]]:
    root = Path(run["root"])
    if "points" in run:
        out = []
        for index, point in enumerate(run["points"]):
            if isinstance(point, str):
                out.append((point, root / point))
            else:
                name = str(point.get("name", f"point_{index}"))
                out.append((name, root / str(point["path"])))
        return out

    # Backward-compatible layer-only discovery.
    dirs = sorted(
        root.glob("layer_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    return [(p.name, p) for p in dirs]


def summarize_runs(config: Mapping) -> tuple[list[dict], list[dict]]:
    point_records: list[dict] = []
    summary_records: list[dict] = []

    for run in config["runs"]:
        current = []
        for point_order, (point_name, point_dir) in enumerate(_point_directories(run)):
            metrics_path = point_dir / "metrics.json"
            stability_path = point_dir / "stability.json"
            if not metrics_path.exists() or not stability_path.exists():
                continue
            m = _load_json(metrics_path)
            s = _load_json(stability_path)
            c1 = s.get("condition_mu_lt_1_over_K_minus_1", {})
            c2 = s.get("condition_mu_lt_1_over_2K_minus_1", {})
            record = {
                "dataset": run.get("dataset", m.get("dataset")),
                "model": run.get("model", m.get("model_name")),
                "sae": run.get("sae", m.get("sae_type")),
                "variant": run.get("variant", m.get("variant", "")),
                "evaluation_point": m.get("evaluation_point", point_name),
                "point_order": int(point_order),
                "layer": m.get("layer"),
                "branch": m.get("branch"),
                "timestep": m.get("timestep"),
                "step": m.get("step"),
                "utilization": float(m["active_feature_fraction"]),
                "retention_alpha1": float(c1["alpha"]) if c1.get("alpha") is not None else math.nan,
                "fidelity_r2": float(m["r2"]),
                "stability_alpha2": float(c2["alpha"]) if c2.get("alpha") is not None else math.nan,
                "average_l0": float(m["average_l0"]),
                "metric_source": s.get("metric_source"),
            }
            point_records.append(record)
            current.append(record)

        if not current:
            continue
        result = {
            "dataset": current[0]["dataset"],
            "model": current[0]["model"],
            "sae": current[0]["sae"],
            "variant": current[0]["variant"],
            "num_points": len({r["evaluation_point"] for r in current}),
            "evaluation_points": ",".join(str(r["evaluation_point"]) for r in current),
        }
        layers = sorted({int(r["layer"]) for r in current if r.get("layer") is not None})
        if layers:
            result["num_layers"] = len(layers)
            result["layers"] = ",".join(str(v) for v in layers)

        for key, prefix in [
            ("utilization", "utilization"),
            ("retention_alpha1", "alpha1"),
            ("fidelity_r2", "r2"),
            ("stability_alpha2", "alpha2"),
        ]:
            mean, low, high, n = _ci95([r[key] for r in current])
            result[f"{prefix}_mean"] = mean
            result[f"{prefix}_ci_low"] = low
            result[f"{prefix}_ci_high"] = high
            result[f"{prefix}_n"] = n
        summary_records.append(result)

    output_dir = Path(config.get("output", {}).get("dir", "results/fidelity_stability/summary"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "point_results.json", "w", encoding="utf-8") as f:
        json.dump(point_records, f, indent=2)
    with open(output_dir / "summary_ci95.json", "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    # Backward-compatible alias used by v3.
    with open(output_dir / "layer_results.json", "w", encoding="utf-8") as f:
        json.dump(point_records, f, indent=2)

    for filename, rows in [
        ("point_results.csv", point_records),
        ("layer_results.csv", point_records),
        ("summary_ci95.csv", summary_records),
    ]:
        if rows:
            fieldnames = []
            seen = set()
            for row in rows:
                for key in row:
                    if key not in seen:
                        seen.add(key)
                        fieldnames.append(key)
            with open(output_dir / filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    return point_records, summary_records
