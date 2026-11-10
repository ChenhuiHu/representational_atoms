from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch


def sanitize_name(name: str) -> str:
    return name.replace("/", "__")


def save_json(path: str | Path, obj: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def dtype_from_string(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


class EntityShardWriter:
    """Shard writer for one-vector-per-entity representations."""

    def __init__(self, output_dir: str | Path, layers: Iterable[int], shard_size: int, dtype: torch.dtype):
        self.output_dir = Path(output_dir)
        self.layers = list(layers)
        self.shard_size = shard_size
        self.dtype = dtype
        self.shard_id = 0
        self.sample_ids: list[str] = []
        self.layer_activations: Dict[int, list[torch.Tensor]] = {layer: [] for layer in self.layers}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for layer in self.layers:
            (self.output_dir / f"layer_{layer}").mkdir(parents=True, exist_ok=True)

    def add(self, sample_id: str, activations: Mapping[int, torch.Tensor]) -> None:
        self.sample_ids.append(sample_id)
        for layer in self.layers:
            self.layer_activations[layer].append(activations[layer].detach().cpu())
        if len(self.sample_ids) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.sample_ids:
            return

        ids = list(self.sample_ids)
        for layer in self.layers:
            acts = torch.stack(self.layer_activations[layer], dim=0).to(self.dtype).contiguous()
            torch.save(
                {
                    "shard_id": self.shard_id,
                    "sample_ids": ids,
                    "activations": acts,
                },
                self.output_dir / f"layer_{layer}" / f"shard_{self.shard_id:06d}.pt",
            )

        self.sample_ids.clear()
        self.layer_activations = {layer: [] for layer in self.layers}
        self.shard_id += 1

    def close(self) -> None:
        self.flush()


def save_continuous_shard(
    output_dir: str | Path,
    shard_id: int,
    chunks: list[dict],
    layer_activations: Mapping[int, list[torch.Tensor]],
    layers: Iterable[int],
    dtype: torch.dtype,
) -> None:
    output_dir = Path(output_dir)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids = [chunk["global_chunk_id"] for chunk in chunks]
    torch.save(
        {"shard_id": shard_id, "chunks": chunks},
        metadata_dir / f"shard_{shard_id:06d}.pt",
    )

    for layer in layers:
        layer_dir = output_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        acts = torch.cat(layer_activations[layer], dim=0).to(dtype).contiguous()
        torch.save(
            {
                "shard_id": shard_id,
                "layer_idx": layer,
                "chunk_ids": chunk_ids,
                "activations": acts,
            },
            layer_dir / f"shard_{shard_id:06d}.pt",
        )
