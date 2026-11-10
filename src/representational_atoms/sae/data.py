from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from representational_atoms.representation_io import (
    index_webdataset_members,
    load_selected_legacy_spatial_dict,
    load_selected_spatial_webdataset,
    load_tar_tensor,
    select_diffusion_representation,
)


@dataclass
class TrainingData:
    train: torch.Tensor
    test: torch.Tensor | None
    metadata: dict


def load_layer_shards(path: str | Path, *, max_samples: int | None = None) -> torch.Tensor:
    """Load a sharded LLM representation layer as ``[samples, hidden_dim]``."""
    path = Path(path)
    shard_paths = sorted(path.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pt files found under {path}")

    pieces: list[torch.Tensor] = []
    loaded = 0
    for shard_path in shard_paths:
        obj = torch.load(shard_path, map_location="cpu")
        acts = obj["activations"]
        if acts.ndim == 3:
            acts = acts.reshape(-1, acts.shape[-1])
        elif acts.ndim != 2:
            raise ValueError(
                f"Expected 2D or 3D activations in {shard_path}, got {tuple(acts.shape)}"
            )
        acts = acts.to(dtype=torch.float32)
        if max_samples is not None:
            remaining = int(max_samples) - loaded
            if remaining <= 0:
                break
            acts = acts[:remaining]
        pieces.append(acts)
        loaded += acts.shape[0]
        if max_samples is not None and loaded >= int(max_samples):
            break

    if not pieces:
        raise RuntimeError(f"No activations loaded from {path}")
    return torch.cat(pieces, dim=0).contiguous()


def load_legacy_entity_dict(path: str | Path, *, max_samples: int | None = None) -> torch.Tensor:
    """Compatibility loader for historical ``layer_<n>.pt`` entity dictionaries."""
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dictionary in {path}")
    values = [value.to(dtype=torch.float32) for value in obj.values()]
    if max_samples is not None:
        values = values[: int(max_samples)]
    if not values:
        raise RuntimeError(f"No activations found in {path}")
    return torch.stack(values, dim=0).contiguous()


def _sample_spatial_indices(
    sample_ids: Sequence[int],
    patches_per_sample: int,
    *,
    sample_size: int,
    test_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the activation-level sampling used by the LAION scripts."""
    total = len(sample_ids) * int(patches_per_sample)
    if sample_size > total:
        raise ValueError(f"sample_size={sample_size} exceeds total activation count={total}")
    if sample_size + test_size > total:
        raise ValueError(
            f"sample_size + test_size = {sample_size + test_size} exceeds total activation count={total}"
        )

    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(total, generator=generator)
    solve_local = permutation[:sample_size]
    holdout_local = permutation[sample_size:]
    test_local = holdout_local[:test_size]

    def to_global(local_indices: torch.Tensor) -> torch.Tensor:
        result = []
        for local in local_indices.tolist():
            sample_pos, patch_idx = divmod(int(local), patches_per_sample)
            result.append(int(sample_ids[sample_pos]) * patches_per_sample + patch_idx)
        return torch.tensor(result, dtype=torch.long)

    return to_global(solve_local), to_global(holdout_local), to_global(test_local)


def load_legacy_spatial_dict(
    path: str | Path,
    *,
    sample_size: int,
    test_size: int,
    seed: int,
    patch_grid: Sequence[int] = (8, 8),
) -> TrainingData:
    path = Path(path)
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or not obj:
        raise TypeError(f"Expected a non-empty sample -> spatial-dict mapping in {path}")
    sample_ids = sorted(int(v) for v in obj.keys())
    del obj

    h_size, w_size = [int(x) for x in patch_grid]
    patches = h_size * w_size
    solve_global, holdout_global, test_global = _sample_spatial_indices(
        sample_ids,
        patches,
        sample_size=int(sample_size),
        test_size=int(test_size),
        seed=int(seed),
    )

    selected = torch.cat([solve_global, test_global]) if test_global.numel() else solve_global
    vectors = load_selected_legacy_spatial_dict(
        path,
        selected,
        patch_grid=(h_size, w_size),
    )
    train = vectors[: solve_global.numel()].contiguous()
    test = vectors[solve_global.numel() :].contiguous() if test_global.numel() else None
    return TrainingData(
        train=train,
        test=test,
        metadata={
            "format": "legacy_spatial_dict",
            "path": str(path),
            "patch_grid": [h_size, w_size],
            "patches_per_sample": patches,
            "sampling_method": "uniform_without_replacement",
            "sampling_seed": int(seed),
            "solve_indices": solve_global,
            "holdout_indices": holdout_global,
            "test_indices": test_global,
            "num_total_activations": len(sample_ids) * patches,
        },
    )


def load_spatial_webdataset(
    tar_path: str | Path,
    cfg: Mapping,
    *,
    sample_size: int,
    test_size: int,
    seed: int,
) -> TrainingData:
    """Read solve/test spatial activations directly from extraction WebDataset tar files."""
    tar_path = Path(tar_path)
    field = str(cfg.get("field", "output.pth"))
    members = index_webdataset_members(tar_path, field)
    if not members:
        raise FileNotFoundError(f"No *.{field} entries found in {tar_path}")

    # Determine patches per image using the exact same representation selector.
    import tarfile

    with tarfile.open(tar_path, "r") as tf:
        first = load_tar_tensor(tf, members[0][1])
    patches = int(select_diffusion_representation(first, cfg).shape[0])
    sample_ids = [sample_id for sample_id, _ in members]

    solve_global, holdout_global, test_global = _sample_spatial_indices(
        sample_ids,
        patches,
        sample_size=int(sample_size),
        test_size=int(test_size),
        seed=int(seed),
    )

    selected = torch.cat([solve_global, test_global]) if test_global.numel() else solve_global
    vectors = load_selected_spatial_webdataset(tar_path, cfg, selected)
    train = vectors[: solve_global.numel()].contiguous()
    test = vectors[solve_global.numel() :].contiguous() if test_global.numel() else None
    return TrainingData(
        train=train,
        test=test,
        metadata={
            "format": "webdataset_tar",
            "path": str(tar_path),
            "source": str(cfg["source"]),
            "field": field,
            "patches_per_sample": patches,
            "sampling_method": "uniform_without_replacement",
            "sampling_seed": int(seed),
            "solve_indices": solve_global,
            "holdout_indices": holdout_global,
            "test_indices": test_global,
            "num_total_activations": len(sample_ids) * patches,
        },
    )


def load_training_data(config: Mapping) -> TrainingData:
    data_cfg = config["data"]
    fmt = str(data_cfg["format"])

    if fmt == "layer_shards":
        train = load_layer_shards(
            data_cfg["path"],
            max_samples=data_cfg.get("max_samples"),
        )
        return TrainingData(train=train, test=None, metadata={"format": fmt, "path": data_cfg["path"]})

    if fmt == "legacy_entity_dict":
        train = load_legacy_entity_dict(
            data_cfg["path"],
            max_samples=data_cfg.get("max_samples"),
        )
        return TrainingData(train=train, test=None, metadata={"format": fmt, "path": data_cfg["path"]})

    if fmt == "legacy_spatial_dict":
        return load_legacy_spatial_dict(
            data_cfg["path"],
            sample_size=int(data_cfg.get("sample_size", 300000)),
            test_size=int(data_cfg.get("test_size", 16384)),
            seed=int(data_cfg.get("sampling_seed", 42)),
            patch_grid=data_cfg.get("patch_grid", [8, 8]),
        )

    if fmt == "webdataset_tar":
        return load_spatial_webdataset(
            data_cfg["path"],
            data_cfg,
            sample_size=int(data_cfg.get("sample_size", 300000)),
            test_size=int(data_cfg.get("test_size", 16384)),
            seed=int(data_cfg.get("sampling_seed", 42)),
        )

    raise ValueError(f"Unsupported training data format: {fmt}")
