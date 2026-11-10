from __future__ import annotations

import io
from pathlib import Path
import re
import tarfile
from typing import Mapping, Sequence

import torch


def load_selected_legacy_spatial_dict(
    path: str | Path,
    global_ids: torch.Tensor,
    *,
    patch_grid: Sequence[int] = (8, 8),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Materialize spatial representations in exactly ``global_ids`` order.

    Historical LAION files map ``sample_id -> {(h, w): vector}`` and use
    ``global_id = sample_id * patches_per_sample + patch_idx``.
    """
    path = Path(path)
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or not obj:
        raise TypeError(f"Expected a non-empty sample -> spatial-dict mapping in {path}")

    h_size, w_size = [int(v) for v in patch_grid]
    patches = h_size * w_size
    ids = global_ids.detach().long().cpu().flatten()
    vectors: list[torch.Tensor] = []
    for gid in ids.tolist():
        if gid < 0:
            raise ValueError(f"Negative spatial global id: {gid}")
        sample_id, patch_idx = divmod(int(gid), patches)
        h, w = divmod(patch_idx, w_size)
        if sample_id not in obj:
            raise KeyError(f"Missing sample_id={sample_id} referenced by global_id={gid}")
        spatial = obj[sample_id]
        if (h, w) not in spatial:
            raise KeyError(f"Missing spatial position {(h, w)} for sample {sample_id}")
        vec = spatial[(h, w)]
        vec = vec.to(dtype=dtype) if isinstance(vec, torch.Tensor) else torch.tensor(vec, dtype=dtype)
        vectors.append(vec)

    if not vectors:
        raise RuntimeError(f"No selected spatial activations loaded from {path}")
    return torch.stack(vectors, dim=0).contiguous()


def index_webdataset_members(tar_path: str | Path, field: str) -> list[tuple[int, str]]:
    """Return sorted ``(sample_id, member_name)`` entries for one tensor field."""
    tar_path = Path(tar_path)
    suffix = "." + str(field)
    members: list[tuple[int, str]] = []
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(suffix):
                continue
            key = member.name[: -len(suffix)]
            match = re.search(r"sample_(\d+)$", key)
            if match is None:
                raise ValueError(f"Cannot parse sample id from tar member {member.name}")
            members.append((int(match.group(1)), member.name))
    members.sort(key=lambda item: item[0])
    return members


def load_tar_tensor(tf: tarfile.TarFile, member_name: str) -> torch.Tensor:
    member = tf.getmember(member_name)
    fileobj = tf.extractfile(member)
    if fileobj is None:
        raise RuntimeError(f"Could not read {member_name}")
    return torch.load(io.BytesIO(fileobj.read()), map_location="cpu")


def select_diffusion_representation(tensor: torch.Tensor, cfg: Mapping) -> torch.Tensor:
    """Select one SD1.5 branch/timestep or FLUX inference step.

    The result is always ``[num_spatial_positions, hidden_dim]`` and preserves
    the native spatial ordering used by the paper's training/evaluation code.
    """
    source = str(cfg["source"]).lower()
    x = tensor

    if source == "flux2":
        steps = [int(v) for v in cfg.get("saved_steps", [0, 3])]
        target_step = int(cfg.get("step", steps[-1]))
        if target_step not in steps:
            raise ValueError(f"Requested step {target_step} is not in saved_steps={steps}")
        x = x[steps.index(target_step)]
        if x.ndim != 2:
            raise ValueError(f"Expected FLUX [patch, hidden] after step selection, got {tuple(x.shape)}")
        return x.to(dtype=torch.float32).contiguous()

    if source == "sd15":
        saved_features = str(cfg.get("saved_features", "both"))
        branch = str(cfg.get("branch", "cond"))
        if saved_features == "both":
            branch_index = {"uncond": 0, "cond": 1}[branch]
        elif saved_features == branch:
            branch_index = 0
        else:
            raise ValueError(
                f"Requested branch={branch}, but extraction stored saved_features={saved_features}."
            )

        timesteps = [int(v) for v in cfg.get("saved_timesteps", [0, 500, 1000])]
        target_timestep = int(cfg.get("timestep", timesteps[-1]))
        if target_timestep not in timesteps:
            raise ValueError(
                f"Requested timestep {target_timestep} is not in saved_timesteps={timesteps}"
            )
        x = x[branch_index, timesteps.index(target_timestep)]

        if x.ndim == 2:
            return x.to(dtype=torch.float32).contiguous()

        layout = str(cfg.get("spatial_layout", "chw")).lower()
        if x.ndim == 3 and layout == "chw":
            return x.permute(1, 2, 0).reshape(-1, x.shape[0]).to(dtype=torch.float32).contiguous()
        if x.ndim == 3 and layout == "hwc":
            return x.reshape(-1, x.shape[-1]).to(dtype=torch.float32).contiguous()
        raise ValueError(
            f"Unsupported SD1.5 selected activation shape/layout: shape={tuple(x.shape)}, layout={layout}"
        )

    raise ValueError(f"Unsupported diffusion source: {source}")


def load_selected_spatial_webdataset(
    tar_path: str | Path,
    cfg: Mapping,
    global_ids: torch.Tensor,
) -> torch.Tensor:
    """Materialize selected spatial vectors from a WebDataset tar.

    The returned rows are in exactly the order of ``global_ids``.  We group the
    requests by image so each selected tar member is decoded only once, then
    place each patch directly into its final output row.
    """
    tar_path = Path(tar_path)
    field = str(cfg.get("field", "output.pth"))
    members = index_webdataset_members(tar_path, field)
    if not members:
        raise FileNotFoundError(f"No *.{field} entries found in {tar_path}")
    member_by_sample = dict(members)

    # Determine spatial cardinality and hidden dimension from one stored sample.
    with tarfile.open(tar_path, "r") as tf:
        first = load_tar_tensor(tf, members[0][1])
    first_vectors = select_diffusion_representation(first, cfg)
    patches, hidden_dim = [int(v) for v in first_vectors.shape]

    ids = global_ids.detach().long().cpu().flatten()
    if ids.numel() == 0:
        raise ValueError("global_ids is empty")

    # sample_id -> [(patch_idx, output_row), ...]
    requests: dict[int, list[tuple[int, int]]] = {}
    for row, gid in enumerate(ids.tolist()):
        if gid < 0:
            raise ValueError(f"Negative spatial global id: {gid}")
        sample_id, patch_idx = divmod(int(gid), patches)
        if patch_idx >= patches:
            raise ValueError(f"Invalid patch index decoded from global id {gid}")
        requests.setdefault(sample_id, []).append((patch_idx, row))

    missing_samples = [sample_id for sample_id in requests if sample_id not in member_by_sample]
    if missing_samples:
        raise KeyError(
            f"Selected IDs reference {len(missing_samples)} samples absent from {tar_path}; "
            f"first few: {missing_samples[:10]}"
        )

    out = torch.empty((ids.numel(), hidden_dim), dtype=torch.float32)
    with tarfile.open(tar_path, "r") as tf:
        for sample_id, wanted in requests.items():
            tensor = load_tar_tensor(tf, member_by_sample[sample_id])
            vectors = select_diffusion_representation(tensor, cfg)
            if vectors.shape != (patches, hidden_dim):
                raise RuntimeError(
                    f"Spatial shape changed for sample {sample_id}: {tuple(vectors.shape)} "
                    f"!= {(patches, hidden_dim)}"
                )
            patch_indices = torch.tensor([p for p, _ in wanted], dtype=torch.long)
            output_rows = torch.tensor([r for _, r in wanted], dtype=torch.long)
            out[output_rows] = vectors[patch_indices]
    return out.contiguous()
