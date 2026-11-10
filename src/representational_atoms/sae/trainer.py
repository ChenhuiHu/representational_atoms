from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from representational_atoms.extraction.storage import save_json
from representational_atoms.sae.data import TrainingData, load_training_data
from representational_atoms.sae.models import BaseSAE, ReLUSAE, TopKSAE, AbsTopKSAE, build_sae


def normalize_activations(
    train: torch.Tensor,
    test: torch.Tensor | None,
    mode: str,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    mode = str(mode).lower()
    train = train.to(dtype=torch.float32)
    test = None if test is None else test.to(dtype=torch.float32)

    if mode == "none":
        return train, test, {"mode": "none"}

    if mode == "rms":
        train_rms = torch.sqrt(torch.mean(train ** 2, dim=-1, keepdim=True))
        train = train / (train_rms + eps)
        if test is not None:
            test_rms = torch.sqrt(torch.mean(test ** 2, dim=-1, keepdim=True))
            test = test / (test_rms + eps)
        return train.contiguous(), None if test is None else test.contiguous(), {
            "mode": "rms",
            "eps": eps,
            "per_vector": True,
        }

    if mode in {"std", "global_std"}:
        scale = train.std()
        train = train / (scale + eps)
        if test is not None:
            test = test / (scale + eps)
        return train.contiguous(), None if test is None else test.contiguous(), {
            "mode": "global_std",
            "scale": float(scale),
            "eps": eps,
        }

    raise ValueError(f"Unsupported normalization mode: {mode}")


def compute_s_inv_half(target_act: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute the encoder preconditioner used in the original scripts."""
    _, singular_values, vh = torch.linalg.svd(target_act, full_matrices=False)
    v = vh.T
    inv_s = 1.0 / (singular_values + eps)
    s_inv_half = (v * inv_s) @ v.T
    return s_inv_half * math.sqrt(target_act.shape[0])


def remove_parallel_component(gradient: torch.Tensor, vectors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    unit = vectors / (torch.norm(vectors, dim=-1, keepdim=True) + eps)
    projection = torch.einsum("...d,...d->...", gradient, unit)
    return gradient - projection[..., None] * unit


def _loss(
    sae: BaseSAE,
    x: torch.Tensor,
    *,
    sparsity_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = sae(x)
    recon_per_sample = torch.sum((x - output.reconstruction) ** 2, dim=-1)

    if isinstance(sae, ReLUSAE):
        sparsity = torch.sum(torch.relu(output.pre_activations), dim=-1)
        total = recon_per_sample + float(sparsity_coefficient) * sparsity
        return total.mean(), recon_per_sample.mean(), sparsity.mean()

    if isinstance(sae, (TopKSAE, AbsTopKSAE)):
        l0 = torch.sum(output.activations != 0, dim=-1).float()
        return recon_per_sample.mean(), recon_per_sample.mean(), l0.mean()

    l0 = sae.sparsity_measure(output)
    total = recon_per_sample + float(sparsity_coefficient) * l0
    return total.mean(), recon_per_sample.mean(), l0.mean()


@torch.inference_mode()
def evaluate_sae(
    sae: BaseSAE,
    activations: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int = 4096,
) -> tuple[float, float]:
    sae.eval()
    total_sq_error = 0.0
    total_l0 = 0.0
    for start in range(0, activations.shape[0], batch_size):
        batch = activations[start : start + batch_size]
        acts = sae.encode(batch.to(device))
        recon = sae.decode(acts).cpu()
        total_sq_error += float(torch.sum((recon - batch) ** 2))
        total_l0 += float((acts.cpu().abs() > 0).sum())

    r2 = 1.0 - (total_sq_error / activations.numel()) / float(activations.var())
    avg_l0 = total_l0 / activations.shape[0]
    return float(r2), float(avg_l0)


def is_pareto_front(recon_losses, sparsity_losses, threshold: int = 50):
    """Pareto filtering used by the historical ReLU/JumpReLU/BiJumpReLU scripts."""
    recon_losses = np.asarray(recon_losses)
    sparsity_losses = np.asarray(sparsity_losses)
    n = len(recon_losses)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if (
                recon_losses[j] <= recon_losses[i]
                and sparsity_losses[j] <= sparsity_losses[i]
                and (recon_losses[j] < recon_losses[i] or sparsity_losses[j] < sparsity_losses[i])
            ):
                is_pareto[i] = False
                break

    indices = np.where(is_pareto)[0]
    if len(indices) < 2:
        return is_pareto
    jumps = np.where(np.diff(indices) > threshold)[0]
    if len(jumps):
        keep = indices[jumps[-1] + 1 :]
        is_pareto[:] = False
        is_pareto[keep] = True
    return is_pareto


def _pareto_epoch(recon_losses, sparsity_losses, eval_every: int) -> int:
    mask = is_pareto_front(
        recon_losses,
        sparsity_losses,
        threshold=max(1, 1000 // eval_every),
    )
    recon = np.asarray(recon_losses)[mask]
    sparsity = np.asarray(sparsity_losses)[mask]
    # The original scripts compute both normalized objectives, but model
    # selection is based on the normalized sparsity distance only.
    _ = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
    norm_sparsity = (sparsity - sparsity.min()) / (sparsity.max() - sparsity.min() + 1e-8)
    min_idx = int(np.argmin(np.sqrt(norm_sparsity ** 2)))
    history_index = np.arange(len(recon_losses))[mask][min_idx]
    return int((history_index + 1) * eval_every)


def _serializable_metadata(metadata: Mapping) -> dict:
    out = {}
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            # Sampling indices are saved separately to avoid huge JSON files.
            continue
        out[key] = value
    return out


def _save_sampling_metadata(output_dir: Path, data: TrainingData) -> None:
    sampling_dir = output_dir / "sampling"
    tensor_keys = ["solve_indices", "holdout_indices", "test_indices"]
    saved_any = False
    for key in tensor_keys:
        value = data.metadata.get(key)
        if isinstance(value, torch.Tensor):
            sampling_dir.mkdir(parents=True, exist_ok=True)
            torch.save(value, sampling_dir / f"{key}.pt")
            saved_any = True
    if saved_any:
        save_json(sampling_dir / "sampling_config.json", _serializable_metadata(data.metadata))


def _checkpoint_name(config: Mapping, epoch: int, r2: float, l0: float) -> str:
    sae_cfg = config["sae"]
    arch = sae_cfg["architecture"]
    width = sae_cfg["expansion_factor"]
    if arch in {"topk", "abstopk"}:
        sparse = f"K{sae_cfg.get('k', 50)}"
    else:
        sparse = f"lambda{sae_cfg.get('sparsity_coefficient', 0.0)}"
    return f"{arch}_x{width}_{sparse}_epoch{epoch}_r2{r2:.4f}_l0{l0:.2f}.pt"


def train_sae(config: Mapping) -> Path:
    data = load_training_data(config)
    sae_cfg = config["sae"]
    output_template = str(config.get("output", {}).get("dir", "saved_models/run"))
    output_dir = Path(
        output_template.format(
            architecture=str(sae_cfg["architecture"]).lower(),
            k=sae_cfg.get("k", ""),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", dict(config))
    _save_sampling_metadata(output_dir, data)

    prep_cfg = config.get("preprocessing", {})
    train_x, test_x, norm_meta = normalize_activations(
        data.train,
        data.test,
        prep_cfg.get("normalization", "rms"),
        float(prep_cfg.get("eps", 1e-8)),
    )
    save_json(output_dir / "preprocessing.json", norm_meta)

    input_dim = int(train_x.shape[1])
    expansion = int(sae_cfg["expansion_factor"])
    dictionary_width = input_dim * expansion

    s_inv_half = compute_s_inv_half(train_x, eps=float(prep_cfg.get("svd_eps", 1e-6)))
    sae = build_sae(
        sae_cfg,
        input_dim=input_dim,
        dictionary_width=dictionary_width,
        s_inv_half=s_inv_half,
    )

    train_cfg = config.get("training", {})
    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    sae = sae.to(device)

    batch_size = int(train_cfg.get("batch_size", 4096))
    logical_batch_size = int(train_cfg.get("logical_batch_size", 4096))
    accumulation_steps = math.ceil(min(logical_batch_size, len(train_x)) / batch_size)
    logical_batch_num = math.ceil(len(train_x) / (accumulation_steps * batch_size))

    loader = DataLoader(
        TensorDataset(train_x),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=float(train_cfg.get("learning_rate", 2e-4)),
        betas=(float(train_cfg.get("adam_beta1", 0.9)), float(train_cfg.get("adam_beta2", 0.999))),
    )

    num_epochs = int(train_cfg.get("num_epochs", 100000))
    eval_every = int(train_cfg.get("eval_every", 20))
    log_every = int(train_cfg.get("log_every", 80))
    eval_batch_size = int(train_cfg.get("eval_batch_size", 4096))
    fix_decoder_norms = bool(sae_cfg.get("fix_decoder_norms", True))
    sparsity_coefficient = float(sae_cfg.get("sparsity_coefficient", 0.0))

    selection_cfg = config.get("selection", {})
    mode = selection_cfg.get(
        "mode",
        "best_r2" if sae.architecture in {"topk", "abstopk"} else "legacy_pareto",
    )
    patience = max(int(train_cfg.get("patience_numerator", 6000) / logical_batch_num), int(train_cfg.get("minimum_patience", 500 if mode == "legacy_pareto" else 600)))
    train_at_least = max(int(train_cfg.get("minimum_updates_numerator", 12000) / logical_batch_num), int(train_cfg.get("minimum_epochs", 3000)))

    history: list[dict] = []
    recon_history: list[float] = []
    sparsity_history: list[float] = []
    best_path: Path | None = None
    best_r2 = -float("inf")
    best_epoch = -1
    pareto_epoch_global: int | None = None
    max_pareto_epoch_found = -1
    last_train_l0 = float("inf")

    for epoch in range(num_epochs):
        sae.train()
        total_recon = 0.0
        total_sparse = 0.0
        total_loss = 0.0
        batch_losses: list[float] = []
        optimizer.zero_grad()
        current_logical_batch = 1

        progress = tqdm(loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)
        for batch_idx, (batch,) in enumerate(progress):
            batch = batch.to(device)
            loss, recon_loss, sparsity_stat = _loss(
                sae,
                batch,
                sparsity_coefficient=sparsity_coefficient,
            )
            if current_logical_batch != logical_batch_num:
                denom = accumulation_steps
            else:
                denom = len(loader) - (logical_batch_num - 1) * accumulation_steps
            (loss / denom).backward()

            total_loss += float(loss.detach())
            total_recon += float(recon_loss.detach())
            total_sparse += float(sparsity_stat.detach())
            batch_losses.append(float(loss.detach()))

            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(loader):
                current_logical_batch += 1
                if fix_decoder_norms and sae.W_dec.grad is not None:
                    sae.W_dec.grad = remove_parallel_component(sae.W_dec.grad, sae.W_dec.data)
                optimizer.step()
                optimizer.zero_grad()
                if fix_decoder_norms:
                    sae.W_dec.data = sae.W_dec.data / torch.norm(
                        sae.W_dec.data,
                        dim=-1,
                        keepdim=True,
                    )

        record = {
            "epoch": epoch + 1,
            "avg_loss": total_loss / len(loader),
            "min_loss": min(batch_losses),
            "max_loss": max(batch_losses),
            "avg_recon_loss": total_recon / len(loader),
            "avg_sparsity_stat": total_sparse / len(loader),
        }

        if (epoch + 1) % eval_every == 0:
            train_r2, train_l0 = evaluate_sae(
                sae,
                train_x,
                device,
                batch_size=eval_batch_size,
            )
            record.update(
                {
                    "train_r2": train_r2,
                    "train_avg_l0": train_l0,
                    **sae.threshold_statistics(),
                }
            )
            if test_x is not None:
                test_r2, test_l0 = evaluate_sae(
                    sae,
                    test_x,
                    device,
                    batch_size=eval_batch_size,
                )
                record.update({"test_r2": test_r2, "test_avg_l0": test_l0})

            last_train_l0 = train_l0

            if mode == "best_r2":
                improvement_window = selection_cfg.get("improvement_window")
                within_window = (
                    improvement_window is None
                    or best_epoch < 0
                    or (epoch + 1 - best_epoch) < int(improvement_window)
                )
                if train_r2 > best_r2 and within_window:
                    best_r2 = train_r2
                    best_epoch = epoch + 1
                    new_path = output_dir / _checkpoint_name(config, epoch + 1, train_r2, train_l0)
                    torch.save(sae.state_dict(), new_path)
                    if best_path is not None and best_path.exists():
                        best_path.unlink()
                    best_path = new_path

            elif mode == "legacy_pareto":
                recon_history.append(record["avg_recon_loss"])
                sparsity_history.append(record["avg_sparsity_stat"])
                pareto_epoch_global = _pareto_epoch(recon_history, sparsity_history, eval_every)
                record["pareto_epoch_global"] = pareto_epoch_global

                recent_window = int(selection_cfg.get("pareto_recent_window", 600))
                max_l0 = float(selection_cfg.get("max_l0_for_save", 20.0))
                min_save_gap = int(selection_cfg.get("minimum_save_gap", 80))
                if (
                    pareto_epoch_global > max_pareto_epoch_found
                    and (epoch + 1 - pareto_epoch_global) < recent_window
                    and train_l0 < max_l0
                    and (epoch + 1 - max_pareto_epoch_found) > min_save_gap
                ):
                    new_path = output_dir / _checkpoint_name(config, epoch + 1, train_r2, train_l0)
                    torch.save(sae.state_dict(), new_path)
                    if best_path is not None and best_path.exists():
                        best_path.unlink()
                    best_path = new_path
                    max_pareto_epoch_found = epoch + 1

                if best_path is None and train_l0 < float(selection_cfg.get("first_save_l0", 10.0)) and train_r2 > float(selection_cfg.get("first_save_r2", 0.90)):
                    best_path = output_dir / _checkpoint_name(config, epoch + 1, train_r2, train_l0)
                    torch.save(sae.state_dict(), best_path)

            else:
                raise ValueError(f"Unsupported selection mode: {mode}")

        history.append(record)
        if (epoch + 1) % log_every == 0:
            with open(output_dir / "training_log.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        if epoch + 1 > train_at_least:
            if mode == "best_r2" and best_epoch > 0 and epoch + 1 - best_epoch >= patience:
                break
            if (
                mode == "legacy_pareto"
                and pareto_epoch_global is not None
                and epoch + 1 - pareto_epoch_global >= patience
                and last_train_l0 < float(selection_cfg.get("max_l0_for_stop", 20.0))
            ):
                break

    with open(output_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if best_path is None:
        # Always leave a usable checkpoint even for tiny smoke tests that never
        # reach the historical selection thresholds.
        train_r2, train_l0 = evaluate_sae(sae, train_x, device, batch_size=eval_batch_size)
        best_path = output_dir / _checkpoint_name(config, len(history), train_r2, train_l0)
        torch.save(sae.state_dict(), best_path)

    save_json(
        output_dir / "run_summary.json",
        {
            "architecture": sae.architecture,
            "input_dim": input_dim,
            "dictionary_width": dictionary_width,
            "num_train_samples": int(train_x.shape[0]),
            "num_test_samples": 0 if test_x is None else int(test_x.shape[0]),
            "best_checkpoint": best_path.name,
            "selection_mode": mode,
        },
    )
    return best_path
