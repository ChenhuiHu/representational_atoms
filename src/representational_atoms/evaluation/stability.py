from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping
import warnings

import numpy as np
import torch


Condition = Callable[[float, float], bool]


def build_active_decoder_dictionary(
    w_dec: torch.Tensor,
    feature_ids: torch.Tensor,
    *,
    device: torch.device,
    norm_eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select active decoder directions and normalize them to unit Euclidean norm."""
    w_dec = w_dec.detach().cpu().to(torch.float32)
    d_sae, _ = w_dec.shape
    ids = torch.unique(feature_ids.long().cpu())
    ids = ids[(ids >= 0) & (ids < d_sae)]
    if ids.numel() == 0:
        raise RuntimeError("No valid active features.")

    w = w_dec[ids].T.contiguous()  # [d_model, n_active]
    norms = torch.linalg.vector_norm(w, dim=0)
    valid = torch.isfinite(norms) & (norms > norm_eps)
    w = w[:, valid]
    ids = ids[valid]
    norms = norms[valid]
    if ids.numel() < 2:
        raise RuntimeError("Need at least two valid active decoder directions.")
    w = (w / norms.unsqueeze(0)).to(device=device, dtype=torch.float32).contiguous()
    return w, ids


def build_full_decoder_metric(
    w_dec: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int = 4096,
    norm_eps: float = 1e-12,
) -> tuple[torch.Tensor, int]:
    """Construct ``M_full = D_full D_full^T`` from all normalized decoder rows.

    This is the metric used by the later evaluation notebooks when the set of
    active features is smaller than ``d_model``.  Building it in chunks avoids
    materializing the complete normalized decoder dictionary on the GPU.
    """
    w_dec = w_dec.detach().cpu().to(torch.float32)
    d_sae, d_model = w_dec.shape
    metric = torch.zeros((d_model, d_model), dtype=torch.float32, device=device)
    valid_total = 0

    with torch.no_grad():
        for start in range(0, d_sae, int(chunk_size)):
            chunk = w_dec[start : start + int(chunk_size)].to(device=device, dtype=torch.float32)
            norms = torch.linalg.vector_norm(chunk, dim=1)
            valid = torch.isfinite(norms) & (norms > norm_eps)
            if bool(valid.any()):
                chunk = chunk[valid] / norms[valid].unsqueeze(1)
                metric += chunk.T @ chunk
                valid_total += int(valid.sum().item())
            del chunk, norms, valid

    metric = (metric + metric.T) * 0.5
    if not bool(torch.isfinite(metric).all()):
        raise RuntimeError("Non-finite values found in the full-decoder metric.")
    if valid_total < d_model:
        warnings.warn(
            "The full decoder contains fewer valid directions than d_model; its metric may be rank deficient.",
            RuntimeWarning,
        )
    return metric, valid_total


def compute_dual_dictionary(
    w_active: torch.Tensor,
    *,
    w_dec_full: torch.Tensor | None = None,
    metric_policy: str = "auto",
    singular_policy: str = "pinv",
    full_metric_chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute the dual dictionary used by the atomic inner product.

    ``metric_policy='auto'`` reproduces the latest notebook logic:
      * if ``n_active >= d_model``, use the active normalized dictionary,
        ``M = W_active W_active^T``;
      * if ``n_active < d_model``, build ``M`` from *all* normalized decoder
        vectors and evaluate only the active directions under that metric.

    This avoids the degenerate identity geometry produced by applying a
    pseudoinverse directly to an undercomplete active dictionary.
    """
    metric_policy = str(metric_policy).lower()
    singular_policy = str(singular_policy).lower()
    if metric_policy not in {"auto", "active", "full"}:
        raise ValueError("metric_policy must be 'auto', 'active', or 'full'.")
    if singular_policy not in {"pinv", "error"}:
        raise ValueError("singular_policy must be 'pinv' or 'error'.")

    d_model, n_active = w_active.shape
    undercomplete = n_active < d_model
    if metric_policy == "auto":
        source = "full_decoder" if undercomplete else "active_decoder"
    elif metric_policy == "full":
        source = "full_decoder"
    else:
        source = "active_decoder"

    valid_full = None
    if source == "full_decoder":
        if w_dec_full is None:
            raise RuntimeError("w_dec_full is required when the full decoder defines the metric.")
        metric, valid_full = build_full_decoder_metric(
            w_dec_full,
            device=w_active.device,
            chunk_size=int(full_metric_chunk_size),
        )
        if metric.shape != (d_model, d_model):
            raise RuntimeError(
                f"Full-decoder metric has shape {tuple(metric.shape)}, expected {(d_model, d_model)}"
            )
    else:
        metric = w_active @ w_active.T
        metric = (metric + metric.T) * 0.5

    l, info = torch.linalg.cholesky_ex(metric)
    cholesky_ok = int(info.max().item()) == 0
    z = None
    if cholesky_ok:
        candidate = torch.cholesky_solve(w_active, l)
        if bool(torch.isfinite(candidate).all()):
            z = candidate
            solver = f"{source}:cholesky_solve"
        else:
            cholesky_ok = False

    if not cholesky_ok:
        message = f"The {source} metric is not numerically positive definite."
        if singular_policy == "error":
            raise RuntimeError(message + " Set singular_policy='pinv' to reproduce the notebook fallback.")
        warnings.warn(message + " Using the Moore-Penrose pseudoinverse.", RuntimeWarning)
        metric_pinv = torch.linalg.pinv(metric, hermitian=True)
        z = metric_pinv @ w_active
        solver = f"{source}:pinv(M)"
        del metric_pinv

    assert z is not None
    if not bool(torch.isfinite(z).all()):
        raise RuntimeError("Non-finite values found in the dual dictionary.")

    diag_g = (w_active * z).sum(dim=0)
    if not bool(torch.isfinite(diag_g).all()):
        raise RuntimeError("Non-finite values found in diag(G).")
    nonpositive = int((diag_g <= 0).sum().item())
    diag_g = diag_g.clamp_min(1e-12)

    diagnostics = {
        "d_model": int(d_model),
        "active_feature_count": int(n_active),
        "active_set_undercomplete": bool(undercomplete),
        "metric_policy": metric_policy,
        "metric_source": source,
        "matrix_solver": solver,
        "cholesky_succeeded": bool(cholesky_ok),
        "nonpositive_diag_G_count": nonpositive,
    }
    if valid_full is not None:
        diagnostics["valid_full_decoder_features"] = int(valid_full)

    del metric, l
    return z, diag_g, diagnostics


def compute_mu_samples(
    w: torch.Tensor,
    z: torch.Tensor,
    diag_g: torch.Tensor,
    *,
    max_samples: int = 5_000_000,
    pair_batch_size: int = 100_000,
    seed: int = 42,
) -> torch.Tensor:
    """Compute/simulate normalized off-diagonal atomic-inner-product coherence values."""
    n = int(w.shape[1])
    if n < 2:
        raise ValueError("Need at least two active features.")
    total_pairs = n * (n - 1) // 2
    target = min(int(max_samples), total_pairs)
    chunks: list[torch.Tensor] = []

    if total_pairs <= int(max_samples):
        pairs = torch.triu_indices(n, n, offset=1, device="cpu")
        for start in range(0, pairs.shape[1], int(pair_batch_size)):
            end = min(start + int(pair_batch_size), pairs.shape[1])
            i = pairs[0, start:end].to(w.device)
            j = pairs[1, start:end].to(w.device)
            gij = (w[:, i] * z[:, j]).sum(dim=0)
            denom = torch.sqrt((diag_g[i] * diag_g[j]).clamp_min(1e-24))
            chunks.append((gij.abs() / denom).detach().float().cpu())
    else:
        rng = np.random.default_rng(int(seed))
        remaining = target
        while remaining > 0:
            batch_n = min(int(pair_batch_size), remaining)
            i_np = rng.integers(0, n, size=batch_n, dtype=np.int64)
            j_np = rng.integers(0, n - 1, size=batch_n, dtype=np.int64)
            j_np = j_np + (j_np >= i_np)
            i = torch.from_numpy(i_np).to(w.device)
            j = torch.from_numpy(j_np).to(w.device)
            gij = (w[:, i] * z[:, j]).sum(dim=0)
            denom = torch.sqrt((diag_g[i] * diag_g[j]).clamp_min(1e-24))
            chunks.append((gij.abs() / denom).detach().float().cpu())
            remaining -= batch_n

    mu = torch.cat(chunks, dim=0)
    return mu[torch.isfinite(mu)]


def retention_condition(mu: float, k: float) -> bool:
    if k <= 1.0:
        return True
    return mu < 1.0 / (k - 1.0)


def stability_condition(mu: float, k: float) -> bool:
    denom = 2.0 * k - 1.0
    if denom <= 0:
        return True
    return mu < 1.0 / denom


def largest_alpha(
    mu_source,
    k_source,
    condition: Condition,
    *,
    lo: float = 0.0,
    hi: float = 0.999999,
    iterations: int = 40,
) -> float | None:
    mu = np.asarray(mu_source, dtype=np.float32).ravel()
    k = np.asarray(k_source, dtype=np.float32).ravel()
    mu = mu[np.isfinite(mu)]
    k = k[np.isfinite(k)]
    if mu.size == 0 or k.size == 0:
        return None

    def evaluate(alpha: float) -> bool:
        mu_a = float(np.quantile(mu, alpha, method="linear"))
        k_a = float(np.quantile(k, alpha, method="linear"))
        return condition(mu_a, k_a)

    if not evaluate(lo):
        return None
    if evaluate(hi):
        return hi
    left, right = lo, hi
    for _ in range(int(iterations)):
        mid = (left + right) / 2.0
        if evaluate(mid):
            left = mid
        else:
            right = mid
    return left


def _summarize_alpha(alpha: float | None, mu: np.ndarray, k: np.ndarray):
    if alpha is None:
        return None, None
    return (
        float(np.quantile(mu, alpha, method="linear")),
        float(np.quantile(k, alpha, method="linear")),
    )


def evaluate_stability(
    w_dec: torch.Tensor,
    feature_ids: torch.Tensor,
    k_values: torch.Tensor,
    *,
    output_dir: str | Path,
    metadata: Mapping,
    device: torch.device,
    max_mu_samples: int = 5_000_000,
    pair_batch_size: int = 100_000,
    random_seed: int = 42,
    save_mu_samples: bool = False,
    metric_policy: str = "auto",
    singular_policy: str = "pinv",
    full_metric_chunk_size: int = 4096,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    w_dec_cpu = w_dec.detach().cpu().to(torch.float32)
    w, valid_ids = build_active_decoder_dictionary(w_dec_cpu, feature_ids, device=device)
    z, diag_g, diagnostics = compute_dual_dictionary(
        w,
        w_dec_full=w_dec_cpu,
        metric_policy=metric_policy,
        singular_policy=singular_policy,
        full_metric_chunk_size=full_metric_chunk_size,
    )
    mu = compute_mu_samples(
        w,
        z,
        diag_g,
        max_samples=max_mu_samples,
        pair_batch_size=pair_batch_size,
        seed=random_seed,
    )

    mu_np = mu.numpy().astype(np.float32, copy=False)
    k_np = k_values.detach().float().cpu().numpy().ravel().astype(np.float32, copy=False)
    k_np = k_np[np.isfinite(k_np)]
    if k_np.size == 0 or mu_np.size == 0:
        raise RuntimeError("Empty K or coherence distribution.")

    alpha1 = largest_alpha(mu_np, k_np, retention_condition)
    alpha2 = largest_alpha(mu_np, k_np, stability_condition)
    mu1, k1 = _summarize_alpha(alpha1, mu_np, k_np)
    mu2, k2 = _summarize_alpha(alpha2, mu_np, k_np)

    quantile_levels = [0.50, 0.90, 0.95, 0.99, 0.999]
    result = {
        **dict(metadata),
        **diagnostics,
        "active_feature_count": int(valid_ids.numel()),
        "K_sample_count": int(k_np.size),
        "average_l0": float(k_np.mean()),
        "mu_sample_count": int(mu_np.size),
        "mu_sampling_seed": int(random_seed),
        "max_mu_samples": int(max_mu_samples),
        "mu_mean": float(mu_np.mean()),
        "mu_std": float(mu_np.std()),
        "mu_min": float(mu_np.min()),
        "mu_max": float(mu_np.max()),
        "mu_quantiles": {str(q): float(np.quantile(mu_np, q)) for q in quantile_levels},
        "K_quantiles": {str(q): float(np.quantile(k_np, q)) for q in quantile_levels},
        "condition_mu_lt_1_over_K_minus_1": {
            "alpha": None if alpha1 is None else float(alpha1),
            "mu_alpha": mu1,
            "K_alpha": k1,
        },
        "condition_mu_lt_1_over_2K_minus_1": {
            "alpha": None if alpha2 is None else float(alpha2),
            "mu_alpha": mu2,
            "K_alpha": k2,
        },
    }

    with open(output_dir / "stability.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if save_mu_samples:
        torch.save(mu, output_dir / "mu_samples.pt")
    return result
