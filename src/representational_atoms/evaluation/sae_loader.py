from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from representational_atoms.sae.models import BaseSAE, build_sae


class GemmaScopeJumpReLUSAE(nn.Module):
    """Minimal GemmaScope JumpReLU implementation used by the paper notebooks."""

    architecture = "gemmascope_jumprelu"

    def __init__(self, input_dim: int, dictionary_width: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.dictionary_width = int(dictionary_width)
        self.W_enc = nn.Parameter(torch.zeros(input_dim, dictionary_width))
        self.W_dec = nn.Parameter(torch.zeros(dictionary_width, input_dim))
        self.threshold = nn.Parameter(torch.zeros(dictionary_width))
        self.b_enc = nn.Parameter(torch.zeros(dictionary_width))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc
        return (pre > self.threshold) * F.relu(pre)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def threshold_statistics(self) -> dict:
        t = self.threshold.detach().cpu()
        return {
            "threshold": {
                "mean": float(t.mean()),
                "min": float(t.min()),
                "max": float(t.max()),
            }
        }


def _format_template(template: str, context: Mapping) -> Path:
    return Path(str(template).format(**context))


def _checkpoint_from_run_dir(run_dir: Path) -> Path:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"run_summary.json not found in {run_dir}. Set sae.checkpoint_path_template explicitly "
            "for a legacy checkpoint."
        )
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    checkpoint = run_dir / summary["best_checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint listed in {summary_path} does not exist: {checkpoint}")
    return checkpoint


def resolve_parameter_path(sae_config: Mapping, context: Mapping) -> Path:
    source = str(sae_config.get("source", "trained")).lower()
    if source == "gemmascope":
        template = sae_config.get("parameter_path_template")
        if not template:
            raise ValueError("GemmaScope evaluation requires sae.parameter_path_template")
        return _format_template(template, context)

    explicit = sae_config.get("checkpoint_path_template")
    if explicit:
        return _format_template(explicit, context)

    run_template = sae_config.get("run_dir_template")
    if not run_template:
        raise ValueError(
            "Trained SAE evaluation requires sae.run_dir_template or sae.checkpoint_path_template"
        )
    return _checkpoint_from_run_dir(_format_template(run_template, context))


def load_sae_for_evaluation(
    sae_config: Mapping,
    context: Mapping,
    *,
    device: torch.device,
) -> tuple[nn.Module, Path]:
    source = str(sae_config.get("source", "trained")).lower()
    parameter_path = resolve_parameter_path(sae_config, context)

    if source == "gemmascope":
        params = np.load(parameter_path)
        input_dim, dictionary_width = params["W_enc"].shape
        sae = GemmaScopeJumpReLUSAE(input_dim, dictionary_width)
        state = {key: torch.from_numpy(params[key].copy()) for key in params.files}
        params.close()
        sae.load_state_dict(state)
        return sae.to(device).eval(), parameter_path

    state = torch.load(parameter_path, map_location="cpu")
    if "W_enc" not in state or "W_dec" not in state:
        raise KeyError(f"Checkpoint lacks W_enc/W_dec: {parameter_path}")
    input_dim, dictionary_width = state["W_enc"].shape

    # Initialization is irrelevant because the complete state dict is loaded.
    identity = torch.eye(input_dim, dtype=torch.float32)
    sae = build_sae(
        sae_config,
        input_dim=input_dim,
        dictionary_width=dictionary_width,
        s_inv_half=identity,
    )
    sae.load_state_dict(state)
    return sae.to(device).eval(), parameter_path


def decoder_matrix(sae: nn.Module) -> torch.Tensor:
    if not hasattr(sae, "W_dec"):
        raise AttributeError("SAE has no W_dec decoder matrix.")
    return sae.W_dec.detach().cpu().to(torch.float32)


def threshold_statistics(sae: nn.Module) -> dict:
    fn = getattr(sae, "threshold_statistics", None)
    return {} if fn is None else dict(fn())
