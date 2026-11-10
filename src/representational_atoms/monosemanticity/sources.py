from __future__ import annotations

import bisect
import gc
from pathlib import Path
from typing import Any, Mapping

import torch


class ExampleResolver:
    def resolve(self, example_id: int, activation: float) -> dict[str, Any]:
        raise NotImplementedError


class WikipediaResolver(ExampleResolver):
    def __init__(self, representation_root: str | Path, tokenizer_name: str, context_window: int = 32):
        self.root = Path(representation_root)
        index = torch.load(self.root / "index.pt", map_location="cpu")
        self.records = index["index_records"]
        self.starts = [int(r["global_token_start"]) for r in self.records]
        self.context_window = int(context_window)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.metadata_cache: dict[int, dict] = {}

    def _metadata(self, shard_id: int) -> dict:
        if shard_id not in self.metadata_cache:
            path = self.root / "metadata" / f"shard_{shard_id:06d}.pt"
            self.metadata_cache[shard_id] = torch.load(path, map_location="cpu")
        return self.metadata_cache[shard_id]

    def resolve(self, example_id: int, activation: float) -> dict[str, Any]:
        global_token_idx = int(example_id)
        chunk_index = bisect.bisect_right(self.starts, global_token_idx) - 1
        if chunk_index < 0 or chunk_index >= len(self.records):
            raise ValueError(f"Invalid global token index: {global_token_idx}")
        record = self.records[chunk_index]
        if not int(record["global_token_start"]) <= global_token_idx < int(record["global_token_end"]):
            raise RuntimeError(f"Cannot map token {global_token_idx} to a Wikipedia chunk")

        shard_id = int(record["shard_id"])
        local_chunk_index = int(record["local_chunk_index"])
        token_pos = global_token_idx - int(record["global_token_start"])
        chunk_meta = self._metadata(shard_id)["chunks"][local_chunk_index]
        token_ids = chunk_meta["token_ids"]
        left_start = max(0, token_pos - self.context_window)
        right_end = min(len(token_ids), token_pos + self.context_window + 1)
        left_ids = token_ids[left_start:token_pos]
        current_id = token_ids[token_pos]
        right_ids = token_ids[token_pos + 1:right_end]
        left_text = self.tokenizer.decode(left_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        current_text = self.tokenizer.decode([current_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        right_text = self.tokenizer.decode(right_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        raw_token = None
        if chunk_meta.get("tokens") is not None:
            raw_token = chunk_meta["tokens"][token_pos]
        return {
            "global_token_idx": global_token_idx,
            "activation": float(activation),
            "chunk_id": int(record["global_chunk_id"]),
            "shard_id": shard_id,
            "token_pos": int(token_pos),
            "model_position": int(token_pos + chunk_meta.get("model_position_offset", 0)),
            "token_id": int(current_id),
            "token": current_text,
            "raw_token": raw_token,
            "left_context": left_text,
            "right_context": right_text,
            "context": left_text + "⟦" + current_text + "⟧" + right_text,
            "source_sample_hint": chunk_meta.get("source_sample_hint"),
        }


class EntityResolver(ExampleResolver):
    def __init__(self, config: Mapping[str, Any], *, num_entities: int):
        self.entity_source = str(config.get("entity_source", "Wikidata"))
        fmt = str(config.get("format", "layer_shards"))
        path = Path(config["path"])
        if fmt == "layer_shards":
            names: list[str] = []
            for shard in sorted(path.glob("shard_*.pt")):
                obj = torch.load(shard, map_location="cpu")
                names.extend(str(x) for x in obj["sample_ids"])
                if len(names) >= int(num_entities):
                    break
            self.names = names[: int(num_entities)]
        elif fmt == "legacy_entity_dict":
            try:
                obj = torch.load(path, map_location="meta")
            except Exception:
                obj = torch.load(path, map_location="cpu", mmap=True)
            self.names = [str(x) for x in list(obj.keys())[: int(num_entities)]]
            del obj
            gc.collect()
        else:
            raise ValueError(f"Unsupported entity representation format: {fmt}")
        if len(self.names) != int(num_entities):
            raise RuntimeError(
                f"Entity-name count mismatch: expected {num_entities}, got {len(self.names)}"
            )

    def resolve(self, example_id: int, activation: float) -> dict[str, Any]:
        idx = int(example_id)
        return {
            "entity_idx": idx,
            "entity_name": self.names[idx],
            "entity_source": self.entity_source,
            "activation": float(activation),
        }


class SpatialResolver(ExampleResolver):
    def __init__(self, patch_grid: list[int] | tuple[int, int], *, image_size=None, spatial_stride=None):
        self.grid_h, self.grid_w = int(patch_grid[0]), int(patch_grid[1])
        self.image_size = image_size
        self.spatial_stride = spatial_stride

    def resolve(self, example_id: int, activation: float) -> dict[str, Any]:
        global_id = int(example_id)
        per_image = self.grid_h * self.grid_w
        sample_id = global_id // per_image
        patch_idx = global_id % per_image
        h, w = divmod(patch_idx, self.grid_w)
        out = {
            "global_id": global_id,
            "sample_id": int(sample_id),
            "sample_key": f"sample_{sample_id}",
            "image_member": f"sample_{sample_id}.images.npy",
            "spatial_token_idx": int(patch_idx),
            "h": int(h),
            "w": int(w),
            "normalized_bbox": [
                w / self.grid_w,
                h / self.grid_h,
                (w + 1) / self.grid_w,
                (h + 1) / self.grid_h,
            ],
            "activation": float(activation),
        }
        if self.spatial_stride is not None:
            stride = int(self.spatial_stride)
            out["pixel_bbox"] = [w * stride, h * stride, (w + 1) * stride, (h + 1) * stride]
        if self.image_size is not None:
            out["image_size"] = list(self.image_size)
        return out


def build_resolver(source_config: Mapping[str, Any], *, num_samples: int) -> ExampleResolver:
    kind = str(source_config["type"]).lower()
    if kind == "wikipedia":
        return WikipediaResolver(
            source_config["representation_root"],
            source_config["tokenizer"],
            context_window=int(source_config.get("context_window", 32)),
        )
    if kind == "entity":
        return EntityResolver(source_config, num_entities=num_samples)
    if kind == "spatial":
        return SpatialResolver(
            source_config.get("patch_grid", [8, 8]),
            image_size=source_config.get("image_size"),
            spatial_stride=source_config.get("spatial_stride"),
        )
    raise ValueError(f"Unsupported monosemanticity source type: {kind}")
