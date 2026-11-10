from __future__ import annotations

import gc
import heapq
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .common import (
    base_context,
    condition_output_dir,
    condition_seed_offset,
    experiment_conditions,
    resolve_templates,
)
from .sources import build_resolver


def _validate_sparse(obj: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor | None]:
    row_ptr = obj["row_ptr"].long()
    feature_ids = obj["feature_ids"].long()
    values = obj["values"].float()
    l0 = obj["l0"].long()
    num_samples = int(obj.get("num_samples", obj.get("num_tokens", obj.get("num_representations", -1))))
    d_sae = int(obj["d_sae"])
    if num_samples < 0:
        raise KeyError("Sparse activation artifact is missing num_samples/num_tokens/num_representations")
    if row_ptr.numel() != num_samples + 1:
        raise RuntimeError("Invalid CSR row_ptr length")
    if feature_ids.numel() != values.numel():
        raise RuntimeError("feature_ids and values have different lengths")
    if int(row_ptr[-1].item()) != feature_ids.numel():
        raise RuntimeError("CSR row_ptr does not match number of non-zero activations")
    if int(l0.sum().item()) != feature_ids.numel():
        raise RuntimeError("Sum of L0 does not match number of non-zero activations")
    global_ids = obj.get("global_ids")
    if global_ids is not None:
        global_ids = global_ids.long().flatten()
        if global_ids.numel() != num_samples:
            raise RuntimeError("global_ids length does not match num_samples")
    return row_ptr, feature_ids, values, l0, num_samples, d_sae, global_ids


def _update_topk(heap: list, score: float, example_id: int, activation: float, k: int) -> None:
    item = (float(score), int(example_id), float(activation))
    if len(heap) < int(k):
        heapq.heappush(heap, item)
    elif float(score) > heap[0][0]:
        heapq.heapreplace(heap, item)


def _sample_feature_pool(active_ids: torch.Tensor, count: int | None, seed: int) -> torch.Tensor:
    if count is None or int(count) >= active_ids.numel():
        return active_ids.sort().values
    generator = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(active_ids.numel(), generator=generator)[: int(count)]
    return active_ids[idx].sort().values


def mine_condition_candidates(
    sparse_path: str | Path,
    source_config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    condition: Mapping[str, Any],
    architecture: str,
    top_k_examples: int = 10_000,
    min_feature_occurrences: int = 1,
    feature_pool_size: int | None = None,
    feature_pool_seed: int = 42,
) -> Path:
    sparse_path = Path(sparse_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    obj = torch.load(sparse_path, map_location="cpu")
    _, feature_ids, values, l0, num_samples, d_sae, global_ids = _validate_sparse(obj)

    row_indices = torch.repeat_interleave(torch.arange(num_samples, dtype=torch.long), l0)
    occurrence = torch.bincount(feature_ids, minlength=d_sae)
    positive = torch.bincount(feature_ids[values > 0], minlength=d_sae)
    negative = torch.bincount(feature_ids[values < 0], minlength=d_sae)
    active = torch.nonzero(occurrence >= int(min_feature_occurrences), as_tuple=True)[0]
    pool = _sample_feature_pool(active, feature_pool_size, int(feature_pool_seed))
    mask = torch.zeros(d_sae, dtype=torch.bool)
    mask[pool] = True
    keep = mask[feature_ids]

    selected_rows = row_indices[keep]
    selected_ids = selected_rows if global_ids is None else global_ids[selected_rows]
    feature_stream = feature_ids[keep]
    value_stream = values[keep]

    top_abs: dict[int, list] = {}
    top_pos: dict[int, list] = {}
    top_neg: dict[int, list] = {}
    for example_id, feature_id, activation in zip(
        selected_ids.tolist(), feature_stream.tolist(), value_stream.tolist()
    ):
        _update_topk(top_abs.setdefault(feature_id, []), abs(activation), example_id, activation, top_k_examples)
        if activation > 0:
            _update_topk(top_pos.setdefault(feature_id, []), activation, example_id, activation, top_k_examples)
        elif activation < 0:
            _update_topk(top_neg.setdefault(feature_id, []), -activation, example_id, activation, top_k_examples)

    resolver = build_resolver(source_config, num_samples=num_samples)
    records_path = output_dir / "feature_examples.jsonl"
    with open(records_path, "w", encoding="utf-8") as f:
        for fid_tensor in pool:
            fid = int(fid_tensor.item())
            def decode(items):
                return [
                    resolver.resolve(example_id=item[1], activation=item[2])
                    for item in sorted(items, key=lambda x: x[0], reverse=True)
                ]
            record = {
                **dict(condition),
                "feature_id": fid,
                "sae_type": architecture,
                "occurrence_count": int(occurrence[fid].item()),
                "positive_count": int(positive[fid].item()),
                "negative_count": int(negative[fid].item()),
                "top_absolute_examples": decode(top_abs.get(fid, [])),
                "top_positive_examples": decode(top_pos.get(fid, [])),
                "top_negative_examples": decode(top_neg.get(fid, [])),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    torch.save(
        {
            **dict(condition),
            "sae_type": architecture,
            "num_samples": num_samples,
            "d_sae": d_sae,
            "feature_ids": active,
            "occurrence_counts": occurrence,
            "positive_counts": positive,
            "negative_counts": negative,
            "top_k_examples": int(top_k_examples),
            "min_feature_occurrences": int(min_feature_occurrences),
            "sampled_feature_ids": pool,
            "feature_pool_size": feature_pool_size,
            "feature_pool_seed": int(feature_pool_seed),
        },
        output_dir / "feature_statistics.pt",
    )
    del obj, feature_ids, values, l0, row_indices, occurrence, positive, negative
    gc.collect()
    return records_path


def mine_candidates(config: Mapping[str, Any]) -> Path:
    exp = config["experiment"]
    base = base_context(config)
    candidate_cfg = config.get("candidate_mining", {})
    output_cfg = config.get("output", {})
    root = Path(str(output_cfg.get("root", "results/monosemanticity")).format(**base))
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "monosemanticity_config.json", "w", encoding="utf-8") as f:
        json.dump(dict(config), f, indent=2, ensure_ascii=False)
    subdir_template = output_cfg.get("subdir_template")

    for condition in experiment_conditions(exp):
        context = {**base, **condition}
        sparse_cfg = resolve_templates(config["sparse_activations"], context)
        source_cfg = resolve_templates(config["source"], context)
        if "path_template" in source_cfg and "path" not in source_cfg:
            source_cfg["path"] = source_cfg.pop("path_template")
        sparse_path = sparse_cfg.get("path") or sparse_cfg.get("path_template")
        if sparse_path is None:
            raise ValueError("sparse_activations.path or path_template is required")
        out_dir = condition_output_dir(root, condition, str(subdir_template) if subdir_template else None)
        pool_seed = int(candidate_cfg.get("feature_pool_seed", 42)) + condition_seed_offset(condition, mode="pool")
        mine_condition_candidates(
            sparse_path,
            source_cfg,
            output_dir=out_dir / "candidates",
            condition=condition,
            architecture=base["architecture"],
            top_k_examples=int(candidate_cfg.get("top_k_examples", 10_000)),
            min_feature_occurrences=int(candidate_cfg.get("min_feature_occurrences", 1)),
            feature_pool_size=candidate_cfg.get("feature_pool_size"),
            feature_pool_seed=pool_seed,
        )
    return root
