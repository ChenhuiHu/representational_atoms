from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import torch

from representational_atoms.representation_io import (
    load_selected_legacy_spatial_dict,
    load_selected_spatial_webdataset,
)


@dataclass
class StreamingStats:
    num_samples: int
    num_elements: int
    mean: float
    variance: float
    std: float


@dataclass
class EvaluationBatch:
    start: int
    vectors: torch.Tensor
    sample_ids: torch.Tensor | None = None


def _load_selected_ids(config: Mapping, max_samples: int | None) -> torch.Tensor:
    path = config.get("selected_ids_path")
    if not path:
        raise ValueError(
            f"Representation format={config.get('format')!r} requires selected_ids_path so evaluation "
            "uses exactly the spatial activations used for SAE solving."
        )
    ids = torch.load(Path(path), map_location="cpu").long().flatten()
    if max_samples is not None:
        ids = ids[: int(max_samples)]
    if ids.numel() == 0:
        raise RuntimeError(f"No selected spatial IDs found in {path}")
    expected = config.get("expected_samples")
    if expected is not None and ids.numel() != int(expected):
        raise RuntimeError(
            f"Selected-ID count mismatch: got {ids.numel()}, expected {int(expected)} from {path}"
        )
    return ids


def _yield_tensor_batches(
    values: torch.Tensor,
    *,
    batch_size: int,
    sample_ids: torch.Tensor | None = None,
) -> Iterator[EvaluationBatch]:
    for start in range(0, values.shape[0], int(batch_size)):
        end = min(start + int(batch_size), values.shape[0])
        batch = values[start:end].contiguous()
        ids = None if sample_ids is None else sample_ids[start:end].clone()
        if batch.numel():
            yield EvaluationBatch(start=start, vectors=batch, sample_ids=ids)


def iter_representation_batches(
    config: Mapping,
    *,
    batch_size: int,
    max_samples: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> Iterator[EvaluationBatch]:
    """Yield evaluation vectors as ``[batch, hidden_dim]``.

    Supported formats:
      * ``layer_shards``: refactored Wikipedia/Wikidata outputs.
      * ``legacy_entity_dict``: historical Wikidata ``layer_<n>.pt`` files.
      * ``legacy_spatial_dict``: historical parsed LAION spatial dictionaries.
      * ``webdataset_tar``: direct SD1.5/FLUX extraction output, selected by the
        exact ``solve_indices.pth`` saved during SAE training.
    """
    fmt = str(config.get("format", "layer_shards")).lower()
    path = Path(config["path"])
    limit = None if max_samples is None else int(max_samples)

    if fmt == "layer_shards":
        shard_paths = sorted(path.glob("shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.pt files found under {path}")

        cursor = 0
        for shard_path in shard_paths:
            if limit is not None and cursor >= limit:
                break
            obj = torch.load(shard_path, map_location="cpu")
            acts = obj["activations"]
            if acts.ndim == 3:
                acts = acts.reshape(-1, acts.shape[-1])
            elif acts.ndim != 2:
                raise ValueError(
                    f"Expected 2D or 3D activations in {shard_path}; got {tuple(acts.shape)}"
                )
            if limit is not None:
                acts = acts[: max(0, limit - cursor)]
            acts = acts.to(dtype=dtype).contiguous()
            for local in range(0, acts.shape[0], int(batch_size)):
                batch = acts[local : local + int(batch_size)]
                if batch.numel() == 0:
                    continue
                yield EvaluationBatch(start=cursor, vectors=batch, sample_ids=None)
                cursor += batch.shape[0]
            del obj, acts
        return

    if fmt == "legacy_entity_dict":
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict):
            raise TypeError(f"Expected a dictionary in {path}")
        values = list(obj.values())
        if limit is not None:
            values = values[:limit]
        if not values:
            raise RuntimeError(f"No representations found in {path}")
        acts = torch.stack([v.to(dtype=dtype) for v in values], dim=0).contiguous()
        yield from _yield_tensor_batches(acts, batch_size=batch_size)
        return

    if fmt == "legacy_spatial_dict":
        ids = _load_selected_ids(config, limit)
        acts = load_selected_legacy_spatial_dict(
            path,
            ids,
            patch_grid=config.get("patch_grid", [8, 8]),
            dtype=dtype,
        )
        yield from _yield_tensor_batches(acts, batch_size=batch_size, sample_ids=ids)
        return

    if fmt == "webdataset_tar":
        ids = _load_selected_ids(config, limit)
        acts = load_selected_spatial_webdataset(path, config, ids).to(dtype=dtype)
        yield from _yield_tensor_batches(acts, batch_size=batch_size, sample_ids=ids)
        return

    raise ValueError(f"Unsupported evaluation representation format: {fmt}")


def apply_normalization(batch: torch.Tensor, normalization: Mapping) -> torch.Tensor:
    mode = str(normalization.get("mode", "none")).lower()
    eps = float(normalization.get("eps", 1e-8))

    if mode == "none":
        return batch
    if mode == "rms":
        rms = torch.sqrt(torch.mean(batch ** 2, dim=-1, keepdim=True))
        return batch / (rms + eps)
    if mode in {"std", "global_std"}:
        scale = normalization.get("scale")
        if scale is None:
            raise ValueError("global_std normalization requires a resolved 'scale'.")
        return batch / (float(scale) + eps)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def compute_streaming_statistics(
    representation_config: Mapping,
    *,
    batch_size: int,
    max_samples: int | None,
    normalization: Mapping | None = None,
) -> StreamingStats:
    total_sum = 0.0
    total_sq_sum = 0.0
    total_elements = 0
    total_samples = 0

    norm = {"mode": "none"} if normalization is None else dict(normalization)
    for item in iter_representation_batches(
        representation_config,
        batch_size=batch_size,
        max_samples=max_samples,
        dtype=torch.float32,
    ):
        batch = apply_normalization(item.vectors, norm)
        batch64 = batch.to(torch.float64)
        total_sum += float(batch64.sum())
        total_sq_sum += float((batch64 * batch64).sum())
        total_elements += batch64.numel()
        total_samples += batch64.shape[0]

    if total_elements <= 1:
        raise RuntimeError("Not enough representation elements to compute sample variance.")

    mean = total_sum / total_elements
    variance = (total_sq_sum - total_elements * mean * mean) / (total_elements - 1)
    variance = max(float(variance), 0.0)
    return StreamingStats(
        num_samples=total_samples,
        num_elements=total_elements,
        mean=float(mean),
        variance=variance,
        std=variance ** 0.5,
    )


def resolve_normalization(
    representation_config: Mapping,
    preprocessing_config: Mapping,
    *,
    stats_batch_size: int,
    max_samples: int | None,
) -> dict:
    """Resolve preprocessing without an unnecessary second statistics pass.

    Only global-standard-deviation scaling needs a preliminary pass.  Target
    mean/variance for fidelity are accumulated during the reconstruction pass,
    matching the LAION notebooks and avoiding two large spatial materializations.
    """
    normalization = {
        "mode": str(preprocessing_config.get("normalization", "none")).lower(),
        "eps": float(preprocessing_config.get("eps", 1e-8)),
    }

    if normalization["mode"] in {"std", "global_std"}:
        if "scale" in preprocessing_config:
            normalization["scale"] = float(preprocessing_config["scale"])
        else:
            raw_stats = compute_streaming_statistics(
                representation_config,
                batch_size=stats_batch_size,
                max_samples=max_samples,
                normalization={"mode": "none"},
            )
            if raw_stats.std <= 0:
                raise RuntimeError(f"Invalid global standard deviation: {raw_stats.std}")
            normalization["mode"] = "global_std"
            normalization["scale"] = raw_stats.std
    return normalization
