from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import StableDiffusionPipeline

from representational_atoms.data import load_laion_coco
from representational_atoms.extraction.storage import save_json


def _retrieve_tensor(value):
    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError("Expected a one-element tuple from the hooked module.")
        return value[0]
    if isinstance(value, torch.Tensor):
        return value
    # Diffusers modules such as Transformer2DModel may return a BaseOutput
    # object whose tensor payload is stored in `.sample`.
    sample = getattr(value, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    raise TypeError(f"Expected a Tensor-like module input/output, got {type(value)!r}.")


def _locate(root, dotted_path: str):
    obj = root
    for part in dotted_path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _nearest_timestep_indices(pipe, requested):
    timesteps = pipe.scheduler.timesteps
    indices = []
    actual = []
    for target in requested:
        idx = torch.argmin(torch.abs(timesteps.float().cpu() - float(target))).item()
        indices.append(idx)
        actual.append(float(timesteps[idx].item()))
    return indices, actual


def _feature_indices(mode: str):
    # Stable Diffusion CFG batches are [unconditional, conditional] for batch_size=1.
    if mode == "both":
        return [0, 1]
    if mode == "uncond":
        return [0]
    if mode == "cond":
        return [1]
    raise ValueError(f"Unsupported features mode: {mode}")


def run_sd15_extraction(config: Mapping) -> Path:
    model_cfg = config["model"]
    data_cfg = config["dataset"]
    ext_cfg = config["extraction"]
    gen_cfg = config.get("generation", {})
    out_cfg = config.get("output", {})

    if int(ext_cfg.get("dataset_batch_size", 1)) != 1:
        raise ValueError("The current SD1.5 extractor requires dataset_batch_size=1 to preserve CFG indexing semantics.")

    dataset = load_laion_coco(
        data_cfg["path"],
        split=data_cfg.get("split", "train"),
        streaming=data_cfg.get("streaming", True),
        seed=data_cfg.get("shuffle_seed", 42),
    )
    dataloader = DataLoader(dataset, batch_size=1)

    pipe = StableDiffusionPipeline.from_pretrained(model_cfg["path"])
    pipe.to(model_cfg.get("device", "cuda"))
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    blocks = list(ext_cfg["blocks"])
    requested_timesteps = list(ext_cfg.get("timesteps", [0, 500, 1000]))
    feature_mode = ext_cfg.get("features", "both")
    feature_indices = _feature_indices(feature_mode)

    start = int(ext_cfg.get("start", 0))
    finish = int(ext_cfg.get("finish", 10000))
    output_dir = Path(out_cfg.get("root", "data/representations")) / data_cfg["name"] / "sd15" / f"from_{start}_to_{finish}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", dict(config))

    writers = {block: wds.TarWriter(str(output_dir / f"{block}.tar")) for block in blocks}
    if out_cfg.get("save_images", True):
        writers["images"] = wds.TarWriter(str(output_dir / "images.tar"))

    cache_in = {block: [] for block in blocks}
    cache_out = {block: [] for block in blocks}
    handles = []

    def make_hook(block_name):
        def hook(module, inputs, kwargs, output):
            cache_in[block_name].append(_retrieve_tensor(inputs).detach().cpu())
            cache_out[block_name].append(_retrieve_tensor(output).detach().cpu())
        return hook

    for block in blocks:
        handles.append(_locate(pipe, block).register_forward_hook(make_hook(block), with_kwargs=True))

    try:
        for sample_idx, batch in tqdm(enumerate(dataloader), desc="Extracting SD1.5 representations"):
            if sample_idx < start:
                continue
            if sample_idx >= finish:
                break

            for block in blocks:
                cache_in[block].clear()
                cache_out[block].clear()

            prompt = batch["caption"]
            seed = sample_idx
            generator = torch.Generator(device="cpu").manual_seed(seed)
            output = pipe(
                prompt=prompt,
                num_inference_steps=int(gen_cfg.get("num_inference_steps", 50)),
                guidance_scale=float(gen_cfg.get("guidance_scale", 7.5)),
                generator=generator,
                output_type="pil",
            )

            step_indices, actual_timesteps = _nearest_timestep_indices(pipe, requested_timesteps)

            for block in blocks:
                inputs = torch.stack([cache_in[block][i] for i in step_indices], dim=1)
                outputs = torch.stack([cache_out[block][i] for i in step_indices], dim=1)
                residual = outputs - inputs
                residual = residual[feature_indices]
                outputs = outputs[feature_indices]

                writers[block].write(
                    {
                        "__key__": f"sample_{sample_idx}",
                        "diff.pth": residual,
                        "output.pth": outputs,
                        "gen_args.json": {
                            "prompt": prompt[0],
                            "seed": seed,
                            "num_inference_steps": int(gen_cfg.get("num_inference_steps", 50)),
                            "guidance_scale": float(gen_cfg.get("guidance_scale", 7.5)),
                            "requested_timesteps": requested_timesteps,
                            "actual_timesteps": actual_timesteps,
                            "features": feature_mode,
                            "block": block,
                        },
                    }
                )

            if "images" in writers:
                writers["images"].write(
                    {
                        "__key__": f"sample_{sample_idx}",
                        "images.npy": np.stack([np.asarray(image) for image in output.images]),
                    }
                )
    finally:
        for handle in handles:
            handle.remove()
        for writer in writers.values():
            writer.close()

    return output_dir
