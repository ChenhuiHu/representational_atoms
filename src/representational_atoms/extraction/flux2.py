from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2
from transformers import Qwen3ForCausalLM

from representational_atoms.data import load_laion_coco
from representational_atoms.extraction.storage import dtype_from_string, save_json


class ActivationFlux2KleinPipeline(Flux2KleinPipeline):
    @property
    def _execution_device(self):
        return self.transformer.x_embedder.weight.device


def run_flux2_extraction(config: Mapping) -> Path:
    model_cfg = config["model"]
    data_cfg = config["dataset"]
    ext_cfg = config["extraction"]
    gen_cfg = config.get("generation", {})
    runtime_cfg = config.get("runtime", {})
    out_cfg = config.get("output", {})

    if int(ext_cfg.get("dataset_batch_size", 1)) != 1:
        raise ValueError("The current FLUX.2 extractor requires dataset_batch_size=1.")

    model_path = model_cfg["path"]
    dtype = dtype_from_string(model_cfg.get("dtype", "bfloat16"))
    local_only = bool(model_cfg.get("local_files_only", True))
    layers = [int(x) for x in ext_cfg.get("layers", [5, 15, 30])]
    steps = [int(x) for x in ext_cfg.get("steps", [0, 3])]
    num_steps = int(gen_cfg.get("num_inference_steps", 4))
    height = int(gen_cfg.get("height", 128))
    width = int(gen_cfg.get("width", 128))
    guidance_scale = float(gen_cfg.get("guidance_scale", 1.0))
    hidden_dim = int(ext_cfg.get("hidden_dim", 4096))
    patch_h, patch_w = ext_cfg.get("native_patch_grid", [8, 8])
    expected_patches = int(patch_h) * int(patch_w)

    max_memory = runtime_cfg.get("transformer_max_memory")
    transformer_kwargs = {
        "torch_dtype": dtype,
        "device_map": runtime_cfg.get("transformer_device_map", "auto"),
        "local_files_only": local_only,
    }
    if max_memory:
        transformer_kwargs["max_memory"] = {int(k) if str(k).isdigit() else k: v for k, v in max_memory.items()}

    transformer = Flux2Transformer2DModel.from_pretrained(model_path, subfolder="transformer", **transformer_kwargs)

    text_device = runtime_cfg.get("text_encoder_device", "auto")
    text_device_map = "auto" if text_device == "auto" else {"": int(text_device)}
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map=text_device_map,
        local_files_only=local_only,
    ).eval()

    vae = AutoencoderKLFlux2.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=local_only,
    ).eval()

    transformer_input_device = transformer.x_embedder.weight.device
    if transformer_input_device.type != "cuda":
        raise RuntimeError(f"FLUX transformer entrance is not on CUDA: {transformer_input_device}")
    vae.to(transformer_input_device)

    pipe = ActivationFlux2KleinPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
        torch_dtype=dtype,
        local_files_only=local_only,
    )
    pipe.set_progress_bar_config(disable=True)

    text_encoder_device = pipe.text_encoder.model.embed_tokens.weight.device

    dataset = load_laion_coco(
        data_cfg["path"],
        split=data_cfg.get("split", "train"),
        streaming=data_cfg.get("streaming", True),
        seed=data_cfg.get("shuffle_seed", 42),
    )
    dataloader = DataLoader(dataset, batch_size=1)

    start = int(ext_cfg.get("start", 0))
    finish = int(ext_cfg.get("finish", 10000))
    output_dir = Path(out_cfg.get("root", "data/representations")) / data_cfg["name"] / "flux2_klein" / f"from_{start}_to_{finish}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", dict(config))

    writers = {layer: wds.TarWriter(str(output_dir / f"layer_{layer}.tar")) for layer in layers}
    if out_cfg.get("save_images", True):
        writers["images"] = wds.TarWriter(str(output_dir / "images.tar"))

    cache = {}
    current_step = -1
    current_image_seq_len = None
    actual_timesteps = {}

    def reset_state():
        nonlocal cache, current_step, current_image_seq_len, actual_timesteps
        cache = {layer: {} for layer in layers}
        current_step = -1
        current_image_seq_len = None
        actual_timesteps = {}

    def transformer_pre_hook(module, args, kwargs):
        nonlocal current_step, current_image_seq_len, actual_timesteps
        current_step += 1
        hidden_states = kwargs["hidden_states"]
        encoder_hidden_states = kwargs["encoder_hidden_states"]
        current_image_seq_len = hidden_states.shape[1]
        if current_image_seq_len != expected_patches:
            raise RuntimeError(f"Expected {expected_patches} native image tokens, got {current_image_seq_len}.")
        if hidden_states.dtype != dtype or encoder_hidden_states.dtype != dtype:
            raise RuntimeError("Unexpected FLUX activation dtype.")
        timestep = kwargs.get("timestep")
        if timestep is not None:
            actual_timesteps[current_step] = timestep.detach().float().cpu()

    transformer_handle = pipe.transformer.register_forward_pre_hook(transformer_pre_hook, with_kwargs=True)

    def validate(x, layer, step):
        if x.ndim != 3 or x.shape[1] != expected_patches or x.shape[2] != hidden_dim:
            raise RuntimeError(f"Unexpected FLUX activation at layer={layer}, step={step}: {tuple(x.shape)}")
        if x.dtype != dtype:
            raise RuntimeError(f"Unexpected dtype at layer={layer}, step={step}: {x.dtype}")

    def make_double_hook(layer):
        def hook(module, inputs, output):
            if current_step not in steps:
                return
            image_states = output[1]
            validate(image_states, layer, current_step)
            cache[layer][current_step] = image_states.detach().cpu()
        return hook

    def make_single_hook(layer):
        def hook(module, inputs, output):
            if current_step not in steps:
                return
            image_states = output[:, -current_image_seq_len:, :]
            validate(image_states, layer, current_step)
            cache[layer][current_step] = image_states.detach().cpu()
        return hook

    handles = []
    for layer in layers:
        if layer < 8:
            module = pipe.transformer.transformer_blocks[layer]
            handles.append(module.register_forward_hook(make_double_hook(layer)))
        else:
            module = pipe.transformer.single_transformer_blocks[layer - 8]
            handles.append(module.register_forward_hook(make_single_hook(layer)))

    @torch.inference_mode()
    def encode_prompt(prompt):
        if isinstance(prompt, tuple):
            prompt = list(prompt)
        embeds = pipe._get_qwen3_prompt_embeds(
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            prompt=prompt,
            dtype=dtype,
            device=text_encoder_device,
            max_sequence_length=int(ext_cfg.get("max_sequence_length", 512)),
            hidden_states_layers=list(ext_cfg.get("text_hidden_state_layers", [9, 18, 27])),
        )
        return embeds.to(device=transformer_input_device, dtype=dtype)

    try:
        for sample_idx, batch in tqdm(enumerate(dataloader), desc="Extracting FLUX.2 representations"):
            if sample_idx < start:
                continue
            if sample_idx >= finish:
                break

            reset_state()
            prompt = batch["caption"]
            prompt_embeds = encode_prompt(prompt)
            generator = torch.Generator(device="cpu").manual_seed(sample_idx)
            output = pipe(
                prompt=None,
                prompt_embeds=prompt_embeds,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
            )

            if current_step != num_steps - 1:
                raise RuntimeError(f"Expected {num_steps} transformer calls, final step index was {current_step}.")

            for layer in layers:
                missing = [step for step in steps if step not in cache[layer]]
                if missing:
                    raise RuntimeError(f"Layer {layer} missing steps {missing}.")
                activation = torch.stack([cache[layer][step][0] for step in steps], dim=0)
                timestep_values = []
                for step in steps:
                    t = actual_timesteps.get(step)
                    timestep_values.append(None if t is None else (float(t.item()) if t.numel() == 1 else t.tolist()))

                writers[layer].write(
                    {
                        "__key__": f"sample_{sample_idx}",
                        "output.pth": activation,
                        "gen_args.json": {
                            "prompt": prompt[0],
                            "seed": sample_idx,
                            "height": height,
                            "width": width,
                            "num_inference_steps": num_steps,
                            "guidance_scale": guidance_scale,
                            "global_layer": layer,
                            "steps_to_cache": steps,
                            "model_timesteps": timestep_values,
                            "native_patch_grid": [patch_h, patch_w],
                            "num_native_patches": expected_patches,
                            "hidden_dim": hidden_dim,
                            "dtype": model_cfg.get("dtype", "bfloat16"),
                            "pooling": None,
                            "activation_type": "raw_block_output_image_tokens",
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
        transformer_handle.remove()
        for writer in writers.values():
            writer.close()

    return output_dir
