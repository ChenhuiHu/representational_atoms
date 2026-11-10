from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from representational_atoms.data import load_entity_strings, load_wikipedia
from representational_atoms.extraction.storage import (
    EntityShardWriter,
    dtype_from_string,
    sanitize_name,
    save_continuous_shard,
    save_json,
)

SUPPORTED_POSITIONS = {
    "down_proj_input",
    "down_proj_output",
    "post_feedforward_layernorm_output",
}


def load_model_and_tokenizer(model_name: str, device_map="auto"):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_layers(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError("Expected a Hugging Face causal LM exposing model.model.layers.")
    return model.model.layers


def resolve_layers(model, requested: Iterable[int] | None) -> list[int]:
    total = len(get_layers(model))
    layers = list(range(total)) if requested is None else [int(x) for x in requested]
    for layer in layers:
        if not 0 <= layer < total:
            raise ValueError(f"Invalid layer {layer}; valid range is [0, {total - 1}].")
    return layers


def resolve_hook_module(model, layer_idx: int, position: str):
    if position not in SUPPORTED_POSITIONS:
        raise ValueError(f"Unsupported position: {position}")

    layer = get_layers(model)[layer_idx]
    if position in {"down_proj_input", "down_proj_output"}:
        try:
            return layer.mlp.down_proj
        except AttributeError as exc:
            raise AttributeError(
                f"Layer {layer_idx} does not expose mlp.down_proj required by {position}."
            ) from exc
    try:
        return layer.post_feedforward_layernorm
    except AttributeError as exc:
        raise AttributeError(
            f"Layer {layer_idx} does not expose post_feedforward_layernorm."
        ) from exc


def activation_from_hook(inputs, output, position: str) -> torch.Tensor:
    if position == "down_proj_input":
        return inputs[0]
    if position in {"down_proj_output", "post_feedforward_layernorm_output"}:
        return output
    raise ValueError(f"Unsupported position: {position}")


def register_hooks(model, layers: Iterable[int], position: str, cache: Dict[int, torch.Tensor]):
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            cache[layer_idx] = activation_from_hook(inputs, output, position).detach().cpu()
            return output
        return hook

    for layer_idx in layers:
        module = resolve_hook_module(model, layer_idx, position)
        handles.append(module.register_forward_hook(make_hook(layer_idx)))
    return handles


def _output_dir(config: Mapping, model_name: str) -> Path:
    root = Path(config.get("output", {}).get("root", "data/representations"))
    dataset_name = config["dataset"]["name"]
    position = config["extraction"]["position"]
    return root / dataset_name / sanitize_name(model_name) / position


def extract_entities(config: Mapping) -> Path:
    model_cfg = config["model"]
    data_cfg = config["dataset"]
    ext_cfg = config["extraction"]
    out_cfg = config.get("output", {})

    model, tokenizer = load_model_and_tokenizer(model_cfg["name"], model_cfg.get("device_map", "auto"))
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    layers = resolve_layers(model, ext_cfg.get("layers"))
    entities = load_entity_strings(
        data_cfg["path"],
        shuffle=ext_cfg.get("shuffle", True),
        seed=ext_cfg.get("seed", 42),
    )

    output_dir = _output_dir(config, model_cfg["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", dict(config))

    dtype = dtype_from_string(out_cfg.get("dtype", "float32"))
    writer = EntityShardWriter(
        output_dir,
        layers,
        shard_size=ext_cfg.get("shard_size", 32768),
        dtype=dtype,
    )

    cache: Dict[int, torch.Tensor] = {}
    handles = register_hooks(model, layers, ext_cfg["position"], cache)
    device = ext_cfg.get("device", "cuda")
    batch_size = ext_cfg.get("forward_batch_size", 64)

    try:
        for start in tqdm(range(0, len(entities), batch_size), desc="Extracting entity representations"):
            batch = entities[start : start + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(device)
            cache.clear()
            with torch.inference_mode():
                model(**inputs, use_cache=False)

            attention_mask = inputs.attention_mask.detach().cpu()
            for i, entity in enumerate(batch):
                if ext_cfg.get("token_selection", "last_non_padding") != "last_non_padding":
                    raise ValueError("Only token_selection=last_non_padding is currently supported.")
                token_pos = attention_mask[i].nonzero(as_tuple=True)[0][-1].item()
                writer.add(entity, {layer: cache[layer][i, token_pos, :] for layer in layers})
    finally:
        writer.close()
        for handle in handles:
            handle.remove()

    return output_dir


def extract_continuous(config: Mapping) -> Path:
    model_cfg = config["model"]
    data_cfg = config["dataset"]
    ext_cfg = config["extraction"]
    out_cfg = config.get("output", {})

    model, tokenizer = load_model_and_tokenizer(model_cfg["name"], model_cfg.get("device_map", "auto"))
    layers = resolve_layers(model, ext_cfg.get("layers"))
    dataset = load_wikipedia(data_cfg["path"])

    output_dir = _output_dir(config, model_cfg["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", dict(config))

    position = ext_cfg["position"]
    chunk_size = int(ext_cfg.get("chunk_size", 2048))
    target_token_count = int(ext_cfg.get("target_token_count", 1_048_576))
    forward_batch_chunks = int(ext_cfg.get("forward_batch_chunks", 1))
    chunks_per_shard = int(ext_cfg.get("chunks_per_shard", 32))
    prepend_bos = bool(ext_cfg.get("prepend_bos", True))
    separator = ext_cfg.get("separator", "\n\n")
    save_token_strings = bool(ext_cfg.get("save_token_strings", True))
    text_field = data_cfg.get("text_field", "text")
    device = ext_cfg.get("device", "cuda")
    activation_dtype = dtype_from_string(out_cfg.get("dtype", "float32"))

    if prepend_bos:
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        payload_len = chunk_size - 1
        model_position_offset = 1
    else:
        bos_token_id = None
        payload_len = chunk_size
        model_position_offset = 0
    if payload_len <= 0:
        raise ValueError("chunk_size is too small.")

    separator_ids = tokenizer.encode(separator, add_special_tokens=False)
    target_chunk_count = math.ceil(target_token_count / payload_len)
    cache: Dict[int, torch.Tensor] = {}
    handles = register_hooks(model, layers, position, cache)

    token_buffer: list[int] = []
    global_chunk_id = 0
    global_token_cursor = 0
    shard_id = 0
    shard_chunks: list[dict] = []
    shard_layer_acts: Dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    index_records: list[dict] = []
    pending_ids: list[list[int]] = []
    pending_meta: list[dict] = []

    def flush_forward_batch():
        nonlocal shard_id, shard_chunks, shard_layer_acts, pending_ids, pending_meta
        if not pending_ids:
            return

        batch_ids = [[bos_token_id] + ids for ids in pending_ids] if prepend_bos else pending_ids
        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
        cache.clear()
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)

        for layer in layers:
            acts = cache[layer][:, 1:, :] if prepend_bos else cache[layer]
            if acts.shape[1] != payload_len:
                raise RuntimeError(f"Unexpected sequence length at layer {layer}: {acts.shape}")
            shard_layer_acts[layer].append(acts.contiguous())

        for meta in pending_meta:
            shard_chunks.append(meta)
            index_records.append(
                {
                    "global_chunk_id": meta["global_chunk_id"],
                    "shard_id": shard_id,
                    "local_chunk_index": len(shard_chunks) - 1,
                    "global_token_start": meta["global_token_start"],
                    "global_token_end": meta["global_token_end"],
                }
            )

        pending_ids = []
        pending_meta = []

        if len(shard_chunks) >= chunks_per_shard:
            save_continuous_shard(output_dir, shard_id, shard_chunks, shard_layer_acts, layers, activation_dtype)
            shard_id += 1
            shard_chunks = []
            shard_layer_acts = {layer: [] for layer in layers}

    try:
        pbar = tqdm(total=target_chunk_count, desc="Extracting continuous representations")
        for sample_idx, sample in enumerate(dataset["train"]):
            text = sample.get(text_field)
            if text is None or not text.strip():
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False) + separator_ids
            token_buffer.extend(token_ids)

            while len(token_buffer) >= payload_len and global_chunk_id < target_chunk_count:
                chunk_ids = token_buffer[:payload_len]
                del token_buffer[:payload_len]
                meta = {
                    "global_chunk_id": global_chunk_id,
                    "source_sample_hint": sample_idx,
                    "global_token_start": global_token_cursor,
                    "global_token_end": global_token_cursor + payload_len,
                    "chunk_text": tokenizer.decode(chunk_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    "token_ids": chunk_ids,
                    "tokens": tokenizer.convert_ids_to_tokens(chunk_ids) if save_token_strings else None,
                    "payload_token_count": payload_len,
                    "model_position_offset": model_position_offset,
                }
                pending_ids.append(chunk_ids)
                pending_meta.append(meta)
                global_chunk_id += 1
                global_token_cursor += payload_len
                pbar.update(1)
                if len(pending_ids) >= forward_batch_chunks:
                    flush_forward_batch()

            if global_chunk_id >= target_chunk_count:
                break

        flush_forward_batch()
        if shard_chunks:
            save_continuous_shard(output_dir, shard_id, shard_chunks, shard_layer_acts, layers, activation_dtype)
        pbar.close()
    finally:
        for handle in handles:
            handle.remove()

    torch.save(
        {
            "index_records": index_records,
            "total_chunks": len(index_records),
            "payload_len": payload_len,
            "chunk_size": chunk_size,
            "target_layers": layers,
            "position": position,
        },
        output_dir / "index.pt",
    )
    return output_dir


def run_llm_extraction(config: Mapping) -> Path:
    mode = config["extraction"]["mode"]
    if mode == "entity":
        return extract_entities(config)
    if mode == "continuous":
        return extract_continuous(config)
    raise ValueError(f"Unsupported LLM extraction mode: {mode}")
