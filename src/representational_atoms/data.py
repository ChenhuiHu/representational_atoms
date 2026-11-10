from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List


def _deduplicate_nonempty(strings: Iterable[str]) -> List[str]:
    """Deterministically remove empty strings and duplicates while preserving order."""
    return list(dict.fromkeys(s.strip() for s in strings if s and s.strip()))


def load_entity_strings(path: str | Path, *, shuffle: bool = True, seed: int = 42) -> List[str]:
    """Load a JSON list of entity strings used by Wikidata/HowNet experiments."""
    with open(path, "r", encoding="utf-8") as f:
        values = json.load(f)

    entities = _deduplicate_nonempty(values)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(entities)
    return entities


def load_wikipedia(data_files: str):
    """Load the local Wikipedia parquet snapshot used for continuous-token extraction."""
    from datasets import load_dataset

    return load_dataset("parquet", data_files=data_files)


def load_laion_coco(path: str, *, split: str = "train", streaming: bool = True, seed: int = 42):
    """Load the caption stream used by the diffusion-model experiments."""
    from datasets import load_dataset

    dataset = load_dataset(path, split=split, columns=["caption"], streaming=streaming)
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    return dataset
