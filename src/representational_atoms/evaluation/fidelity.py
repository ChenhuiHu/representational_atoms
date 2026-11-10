from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from representational_atoms.evaluation.data import (
    apply_normalization,
    iter_representation_batches,
)
from representational_atoms.evaluation.sae_loader import threshold_statistics


@torch.inference_mode()
def evaluate_fidelity_and_sparsity(
    sae: nn.Module,
    representation_config: Mapping,
    *,
    normalization: Mapping,
    output_dir: str | Path,
    metadata: Mapping,
    device: torch.device,
    max_samples: int | None,
    batch_size: int,
    save_sparse_activations: bool = True,
) -> dict:
    """Compute fidelity and sparse-activation statistics in one streaming pass.

    This preserves the notebook definition

        MSE = total squared error / (num_samples * hidden_dim)
        R^2 = 1 - MSE / Var(normalized target representation),

    where the target variance is the sample variance across every scalar entry
    of the normalized evaluation population.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d_model = int(sae.W_enc.shape[0])
    d_sae = int(sae.W_enc.shape[1])

    total_sq_error = 0.0
    total_nonzero = 0
    total_samples = 0
    ever_active = torch.zeros(d_sae, dtype=torch.bool)

    # Exact normalized-target statistics, accumulated in float64 as in notebooks.
    target_sum = 0.0
    target_sq_sum = 0.0
    target_elements = 0

    l0_pieces: list[torch.Tensor] = []
    feature_id_pieces: list[torch.Tensor] = []
    feature_value_pieces: list[torch.Tensor] = []
    sample_id_pieces: list[torch.Tensor] = []
    saw_sample_ids = False

    for item in iter_representation_batches(
        representation_config,
        batch_size=batch_size,
        max_samples=max_samples,
        dtype=torch.float32,
    ):
        batch_cpu = apply_normalization(item.vectors, normalization)

        batch64 = batch_cpu.to(torch.float64)
        target_sum += float(batch64.sum())
        target_sq_sum += float((batch64 * batch64).sum())
        target_elements += batch64.numel()
        del batch64

        batch_gpu = batch_cpu.to(device)
        acts = sae.encode(batch_gpu)
        recon = sae.decode(acts)
        total_sq_error += float(torch.sum((recon - batch_gpu) ** 2).double().cpu())

        nz = torch.nonzero(acts, as_tuple=False)
        row_ids = nz[:, 0]
        feature_ids = nz[:, 1]
        feature_values = acts[row_ids, feature_ids]
        l0 = torch.bincount(row_ids, minlength=batch_cpu.shape[0])

        total_nonzero += int(nz.shape[0])
        total_samples += batch_cpu.shape[0]
        feature_ids_cpu = feature_ids.detach().cpu()
        if feature_ids_cpu.numel():
            ever_active[feature_ids_cpu] = True

        l0_pieces.append(l0.detach().cpu().to(torch.int32))
        if item.sample_ids is not None:
            saw_sample_ids = True
            sample_id_pieces.append(item.sample_ids.detach().cpu().to(torch.int64))
        elif saw_sample_ids:
            raise RuntimeError("Some evaluation batches have sample IDs while others do not.")

        if save_sparse_activations:
            feature_id_pieces.append(feature_ids_cpu.to(torch.int32))
            feature_value_pieces.append(feature_values.detach().cpu().to(torch.float32))

    if total_samples == 0 or target_elements <= 1:
        raise RuntimeError("No valid representation samples were evaluated.")

    target_mean = target_sum / target_elements
    target_variance = (
        target_sq_sum - target_elements * target_mean * target_mean
    ) / (target_elements - 1)
    target_variance = max(float(target_variance), 0.0)
    if target_variance <= 0:
        raise RuntimeError(f"Invalid normalized target variance: {target_variance}")
    target_std = target_variance ** 0.5

    l0_counts = torch.cat(l0_pieces, dim=0)
    active_ids = torch.nonzero(ever_active, as_tuple=True)[0]
    sample_ids = torch.cat(sample_id_pieces, dim=0) if saw_sample_ids else None
    if sample_ids is not None and sample_ids.numel() != total_samples:
        raise RuntimeError("Spatial provenance ID count does not match evaluated sample count.")

    mse = total_sq_error / (total_samples * d_model)
    r2 = 1.0 - mse / target_variance
    average_l0 = total_nonzero / total_samples
    active_fraction = active_ids.numel() / d_sae
    activation_density = total_nonzero / (total_samples * d_sae)

    metrics = {
        **dict(metadata),
        "num_samples": total_samples,
        "d_model": d_model,
        "d_sae": d_sae,
        "normalization": dict(normalization),
        "target_mean": float(target_mean),
        "target_variance": float(target_variance),
        "target_std": float(target_std),
        "total_squared_error": total_sq_error,
        "mse": mse,
        "r2": r2,
        "average_l0": average_l0,
        "num_active_features": int(active_ids.numel()),
        "active_feature_fraction": float(active_fraction),
        "activation_density": float(activation_density),
        **threshold_statistics(sae),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    torch.save(l0_counts, output_dir / "sparsity.pt")
    torch.save(active_ids, output_dir / "features_id.pt")

    if save_sparse_activations:
        feature_ids = (
            torch.cat(feature_id_pieces, dim=0)
            if feature_id_pieces
            else torch.empty(0, dtype=torch.int32)
        )
        values = (
            torch.cat(feature_value_pieces, dim=0)
            if feature_value_pieces
            else torch.empty(0, dtype=torch.float32)
        )
        row_ptr = torch.empty(total_samples + 1, dtype=torch.int64)
        row_ptr[0] = 0
        row_ptr[1:] = torch.cumsum(l0_counts.to(torch.int64), dim=0)
        if int(row_ptr[-1]) != feature_ids.shape[0] or feature_ids.shape[0] != values.shape[0]:
            raise RuntimeError("CSR sparse activation arrays are inconsistent.")

        sparse_obj = {
            "format": "csr_per_sample",
            "num_samples": total_samples,
            "d_model": d_model,
            "d_sae": d_sae,
            "row_ptr": row_ptr,
            "feature_ids": feature_ids,
            "values": values,
            "l0": l0_counts,
            "normalization": dict(normalization),
        }
        if sample_ids is not None:
            sparse_obj.update(
                {
                    "global_ids": sample_ids,
                    "global_id_definition": representation_config.get(
                        "global_id_definition", "sample_id * patches_per_sample + patch_idx"
                    ),
                    "patch_grid": representation_config.get("patch_grid", [8, 8]),
                }
            )
        torch.save(sparse_obj, output_dir / "sae_activations.pt")

    return metrics
