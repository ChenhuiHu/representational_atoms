from __future__ import annotations

import io
import json
import random
import tarfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from .common import (
    base_context,
    condition_output_dir,
    condition_seed_offset,
    experiment_conditions,
    feature_example_seed,
)


EXAMPLE_SOURCES = {"top_absolute_examples", "top_positive_examples", "top_negative_examples"}


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _sample_examples(record: Mapping[str, Any], source: str, n: int, seed: int) -> list[dict]:
    examples = list(record.get(source, []))
    rng = random.Random(int(seed))
    if len(examples) <= int(n):
        rng.shuffle(examples)
        return examples
    return rng.sample(examples, int(n))


def _image_array_to_pil(arr) -> Image.Image:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image array shape: {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.min() >= 0.0 and arr.max() <= 1.0:
            arr = (arr * 255.0).round()
        elif arr.min() >= -1.0 and arr.max() <= 1.0:
            arr = ((arr + 1.0) * 127.5).round()
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA" if arr.shape[-1] == 4 else "RGB")


class ImageArchive:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.tar = tarfile.open(self.path, "r")

    def close(self):
        self.tar.close()

    def load(self, member: str) -> Image.Image:
        f = self.tar.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"Archive member not found: {member}")
        arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
        return _image_array_to_pil(arr)


def _draw_patch_box(image: Image.Image, example: Mapping[str, Any], patch_grid) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    norm = example.get("normalized_bbox")
    if norm is None:
        gh, gw = int(patch_grid[0]), int(patch_grid[1])
        h, w = int(example["h"]), int(example["w"])
        norm = [w / gw, h / gh, (w + 1) / gw, (h + 1) / gh]
    x0 = max(0, min(width - 1, int(round(norm[0] * width))))
    y0 = max(0, min(height - 1, int(round(norm[1] * height))))
    x1 = max(x0 + 1, min(width, int(round(norm[2] * width))))
    y1 = max(y0 + 1, min(height, int(round(norm[3] * height))))
    line_width = max(2, min(width, height) // 128)
    for offset in range(line_width):
        draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=(255, 0, 0), width=1)
    return image


def _prepare_visual_examples(examples, *, visual_dir: Path, feature_id: int, archive: ImageArchive, patch_grid):
    visual_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for idx, example in enumerate(examples, start=1):
        image = _draw_patch_box(archive.load(example["image_member"]), example, patch_grid)
        name = (
            f"feature_{feature_id}_example_{idx}_sample_{int(example['sample_id'])}_"
            f"patch_{int(example['spatial_token_idx'])}.png"
        )
        path = visual_dir / name
        image.save(path)
        prepared = dict(example)
        prepared["visual_path"] = str(path)
        prepared["visual_uri"] = f"visuals/{name}"
        out.append(prepared)
    return out


def compact_prompt(source_type: str, examples: list[dict]) -> str:
    n = len(examples)
    if source_type == "wikipedia":
        rows = []
        for i, ex in enumerate(examples, 1):
            context = str(ex.get("context", "")).replace("\n", " ").strip()
            rows.append(f"{i}. [{float(ex['activation']):+.4f}] {context}")
        body = "\n\n".join(rows)
        return f"""Below are {n} activation examples from the same SAE feature.

Focus on the highlighted token ⟦...⟧ in each context.

Determine the largest number of examples that can reasonably be grouped into ONE coherent semantic category. Use a reasonably specific category based on the highlighted token's meaning or semantic function.

{body}

Reply ONLY as M/{n}, where M is the size of the largest coherent semantic group. Do not explain."""
    if source_type == "entity":
        rows = [
            f"{i}. [{float(ex['activation']):+.4f}] {str(ex.get('entity_name', '')).strip()}"
            for i, ex in enumerate(examples, 1)
        ]
        body = "\n\n".join(rows)
        source_name = str(examples[0].get("entity_source", "Wikidata")) if examples else "entity"
        return f"""Below are {n} activation examples from the same SAE feature.

Each example corresponds to one {source_name} entity name.

Determine the largest number of examples that can reasonably be grouped into ONE coherent semantic category. Use a reasonably specific category based on the entities' identities, types, or semantic roles.

{body}

Reply ONLY as M/{n}, where M is the size of the largest coherent semantic group. Do not explain."""
    if source_type == "spatial":
        rows = [
            f"{i}. [{float(ex['activation']):+.4f}] sample={ex.get('sample_id')}, "
            f"patch=(h={ex.get('h')}, w={ex.get('w')}), image={ex.get('visual_path', '')}"
            for i, ex in enumerate(examples, 1)
        ]
        body = "\n\n".join(rows)
        return f"""Below are {n} activation examples from the same SAE feature.

Each example corresponds to one image, where the activated patch location is highlighted by a red box.

Determine the largest number of examples that can reasonably be grouped into ONE coherent visual category. Use a reasonably specific category based on the highlighted patch content, such as object parts, textures, materials, colors, text, spatial patterns, or other visual semantics.

{body}

Reply ONLY as M/{n}, where M is the size of the largest coherent semantic group. Do not explain."""
    raise ValueError(source_type)



def diagnostic_prompt(source_type: str, condition: Mapping[str, Any], record: Mapping[str, Any], examples: list[dict]) -> str:
    """Optional exploratory 1--5 prompt retained from the notebooks.

    The manuscript's reported quantitative score uses :func:`compact_prompt` and M/N;
    this prompt is provided only for qualitative diagnostics.
    """
    if source_type == "wikipedia":
        item = "highlighted token occurrences in Wikipedia context"
        semantic = "semantic concept or semantic axis"
    elif source_type == "entity":
        item = "entity names"
        semantic = "semantic concept or semantic axis"
    else:
        item = "image patches highlighted by red boxes"
        semantic = "visual concept or visual semantic axis"
    rendered = []
    for i, ex in enumerate(examples, 1):
        if source_type == "wikipedia":
            payload = str(ex.get("context", "")).replace("\n", " ").strip()
        elif source_type == "entity":
            payload = str(ex.get("entity_name", "")).strip()
        else:
            payload = str(ex.get("visual_path", ""))
        rendered.append(f"Example {i}: [{float(ex['activation']):+.4f}] {payload}")
    body = "\n".join(rendered)
    return f"""You are evaluating the monosemanticity of one SAE feature from {item}.

Determine whether the examples are coherently explained by one reasonably specific {semantic}. Do not force an interpretation when the evidence is weak.

{body}

Return ONLY valid JSON with keys: semantic_label, monosemanticity_score (1-5), coherence_score (1-5), outlier_example_ids, confidence, and reasoning_summary."""

def _quick_review_block(source_type: str, record: Mapping[str, Any], examples: list[dict]) -> str:
    lines = [
        f"## Feature {int(record['feature_id'])}", "",
        f"Occurrences: {int(record.get('occurrence_count', 0))} | Positive: {int(record.get('positive_count', 0))} | Negative: {int(record.get('negative_count', 0))}", "",
    ]
    for i, ex in enumerate(examples, 1):
        activation = float(ex["activation"])
        if source_type == "wikipedia":
            lines += [f"[{i}] {activation:+.4f}", "", str(ex.get("context", "")).replace("\n", " ").strip(), ""]
        elif source_type == "entity":
            lines += [f"[{i}] {activation:+.4f}  {ex.get('entity_name', '')}", ""]
        else:
            lines += [
                f"[{i}] {activation:+.4f} | sample={ex.get('sample_id')} | patch=(h={ex.get('h')}, w={ex.get('w')})",
                "", f"![Example {i}]({ex.get('visual_uri', '')})", "",
            ]
    lines += ["", "-" * 80, ""]
    return "\n".join(lines)


def build_review_sets(config: Mapping[str, Any]) -> Path:
    exp = config["experiment"]
    base = base_context(config)
    source_type = str(config["source"]["type"]).lower()
    review_cfg = config.get("review", {})
    example_source = str(review_cfg.get("example_source", "top_absolute_examples"))
    if example_source not in EXAMPLE_SOURCES:
        raise ValueError(f"Unsupported example_source: {example_source}")
    features_per_condition = int(review_cfg.get("features_per_condition", 10))
    examples_per_feature = int(review_cfg.get("examples_per_feature", 10))
    random_seed = int(review_cfg.get("random_seed", 42))
    write_diagnostic_prompts = bool(review_cfg.get("write_diagnostic_prompts", False))

    output_cfg = config.get("output", {})
    root = Path(str(output_cfg.get("root", "results/monosemanticity")).format(**base))
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "monosemanticity_config.json", "w", encoding="utf-8") as f:
        json.dump(dict(config), f, indent=2, ensure_ascii=False)
    subdir_template = output_cfg.get("subdir_template")
    image_archive_path = config.get("source", {}).get("image_archive")
    archive = ImageArchive(str(image_archive_path).format(**base)) if source_type == "spatial" else None
    patch_grid = config.get("source", {}).get("patch_grid", [8, 8])

    try:
        for condition in experiment_conditions(exp):
            condition_dir = condition_output_dir(root, condition, str(subdir_template) if subdir_template else None)
            candidate_path = condition_dir / "candidates" / "feature_examples.jsonl"
            records = _load_jsonl(candidate_path)
            valid = [r for r in records if len(r.get(example_source, [])) > 1]
            rng = random.Random(random_seed + condition_seed_offset(condition, mode="review"))
            sampled = rng.sample(valid, min(features_per_condition, len(valid)))
            sampled = sorted(sampled, key=lambda r: int(r["feature_id"]))

            review_dir = condition_dir / "review"
            review_dir.mkdir(parents=True, exist_ok=True)
            visual_dir = review_dir / "visuals"
            prompt_path = review_dir / "evaluation_prompts.jsonl"
            sampled_path = review_dir / "sampled_features.jsonl"
            quick_path = review_dir / "quick_review.md"
            annotation_path = review_dir / "annotations_template.jsonl"
            diagnostic_path = review_dir / "diagnostic_prompts_1to5.jsonl"

            with (
                open(prompt_path, "w", encoding="utf-8") as pf,
                open(sampled_path, "w", encoding="utf-8") as sf,
                open(quick_path, "w", encoding="utf-8") as qf,
                open(annotation_path, "w", encoding="utf-8") as af,
                open(diagnostic_path, "w", encoding="utf-8") as df,
            ):
                qf.write(f"# Monosemanticity review — {condition['name']}\n\n")
                qf.write(f"Example source: `{example_source}`  \nFeatures: `{len(sampled)}`  \nExamples per feature: `{examples_per_feature}`\n\n---\n\n")
                for record in sampled:
                    fid = int(record["feature_id"])
                    seed = feature_example_seed(random_seed, condition, fid)
                    examples = _sample_examples(record, example_source, examples_per_feature, seed)
                    if source_type == "spatial":
                        assert archive is not None
                        examples = _prepare_visual_examples(
                            examples, visual_dir=visual_dir, feature_id=fid, archive=archive, patch_grid=patch_grid
                        )
                    prompt = compact_prompt(source_type, examples)
                    pf.write(json.dumps({**condition, "feature_id": fid, "prompt": prompt}, ensure_ascii=False) + "\n")
                    if write_diagnostic_prompts:
                        df.write(json.dumps({**condition, "feature_id": fid, "prompt": diagnostic_prompt(source_type, condition, record, examples)}, ensure_ascii=False) + "\n")
                    sf.write(json.dumps({
                        **condition,
                        "feature_id": fid,
                        "example_source": example_source,
                        "occurrence_count": int(record.get("occurrence_count", 0)),
                        "positive_count": int(record.get("positive_count", 0)),
                        "negative_count": int(record.get("negative_count", 0)),
                        "sampled_examples": examples,
                    }, ensure_ascii=False) + "\n")
                    qf.write(_quick_review_block(source_type, record, examples))
                    af.write(json.dumps({
                        **condition,
                        "feature_id": fid,
                        "group_size": None,
                        "num_examples": len(examples),
                        "score": None,
                        "reviewer": "llm+manual" if source_type in {"wikipedia", "entity"} else "manual",
                    }, ensure_ascii=False) + "\n")
    finally:
        if archive is not None:
            archive.close()
    return root
