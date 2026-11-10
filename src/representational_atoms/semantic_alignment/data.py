from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import sparse


def load_feature_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _example_entity(example: Mapping[str, Any]) -> str | None:
    # v5/v6 standardized artifact.
    if example.get("entity_name") is not None:
        return str(example["entity_name"])
    # Historical 8.* / 9.* notebook artifact.
    if example.get("sample_key") is not None:
        return str(example["sample_key"])
    return None


def build_feature_sets(feature_records: Iterable[Mapping[str, Any]], *, layer: int) -> list[dict[str, Any]]:
    """Reproduce the signed feature-set construction used by notebooks 9.1--9.3.

    Positive-only and negative-only features remain one candidate. Features
    exhibiting both signs are split into ``_pos`` and ``_neg`` candidates. A
    candidate is retained only when its entity set contains at least two unique
    entities.
    """
    out: list[dict[str, Any]] = []
    for record in feature_records:
        fid = int(record["feature_id"])
        pos = {
            entity for ex in record.get("top_positive_examples", [])
            if (entity := _example_entity(ex)) is not None
        }
        neg = {
            entity for ex in record.get("top_negative_examples", [])
            if (entity := _example_entity(ex)) is not None
        }

        if pos and not neg:
            if len(pos) > 1:
                out.append({
                    "layer": int(layer), "feature_id": fid, "feature_name": f"{fid}",
                    "direction": "positive", "feature_set": pos,
                })
        elif neg and not pos:
            if len(neg) > 1:
                out.append({
                    "layer": int(layer), "feature_id": fid, "feature_name": f"{fid}",
                    "direction": "negative", "feature_set": neg,
                })
        else:
            if len(pos) > 1:
                out.append({
                    "layer": int(layer), "feature_id": fid, "feature_name": f"{fid}_pos",
                    "direction": "positive", "feature_set": pos,
                })
            if len(neg) > 1:
                out.append({
                    "layer": int(layer), "feature_id": fid, "feature_name": f"{fid}_neg",
                    "direction": "negative", "feature_set": neg,
                })
    return out


def load_all_layer_features(path_template: str, layers: Iterable[int]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    gid = 0
    for layer in layers:
        path = Path(str(path_template).format(layer=int(layer)))
        if not path.exists():
            raise FileNotFoundError(f"Missing feature candidate file for layer {layer}: {path}")
        records = load_feature_records(path)
        for item in build_feature_sets(records, layer=int(layer)):
            item = dict(item)
            item["global_idx"] = gid
            flat.append(item)
            gid += 1
    return flat


def load_sememe_sets(config: Mapping[str, Any]) -> tuple[list[set[str]], list[dict[str, str]]]:
    """Build HowNet sememe entity sets.

    ``source: openhownet`` reproduces notebooks 9.1--9.3 exactly: all senses of
    each sememe are collected through ``OpenHowNet.HowNetDict`` and sememes with
    fewer than two unique English words are discarded. A pre-exported JSON
    mapping can be used for environments where the OpenHowNet package/data are
    prepared separately.
    """
    source = str(config.get("source", "openhownet")).lower()
    min_entities = int(config.get("min_entities", 2))
    sememe_sets: list[set[str]] = []
    sememe_meta: list[dict[str, str]] = []

    if source == "openhownet":
        import OpenHowNet
        hownet_dict = OpenHowNet.HowNetDict()
        for sememe_obj in hownet_dict.get_all_sememes():
            senses = set(sememe_obj.get_senses())
            words = {sense.en_word for sense in senses if sense.en_word is not None}
            if len(words) >= min_entities:
                sememe_sets.append({str(w) for w in words})
                sememe_meta.append({
                    "sememe": str(getattr(sememe_obj, "en", sememe_obj)).upper(),
                    "sememe_name": str(sememe_obj),
                })
        return sememe_sets, sememe_meta

    if source == "json":
        path = Path(config["path"])
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise TypeError("JSON HowNet sememe source must be sememe -> entity list")
        for sememe, entities in obj.items():
            words = {str(x) for x in entities if x is not None}
            if len(words) >= min_entities:
                sememe_sets.append(words)
                sememe_meta.append({"sememe": str(sememe).upper(), "sememe_name": str(sememe)})
        return sememe_sets, sememe_meta

    raise ValueError(f"Unsupported HowNet sememe source: {source}")


def build_sparse_word_matrices(
    flat_features: list[dict[str, Any]],
    sememe_sets: list[set[str]],
):
    """Build the exact sparse binary feature/sememe matrices used by 9.*."""
    word_to_idx: dict[str, int] = {}
    idx_to_word: list[str] = []

    def word_id(word: str) -> int:
        if word not in word_to_idx:
            word_to_idx[word] = len(idx_to_word)
            idx_to_word.append(word)
        return word_to_idx[word]

    feature_rows: list[int] = []
    feature_cols: list[int] = []
    for i, feat in enumerate(flat_features):
        for word in feat["feature_set"]:
            feature_rows.append(i)
            feature_cols.append(word_id(str(word)))

    sememe_rows: list[int] = []
    sememe_cols: list[int] = []
    for j, sememe_set in enumerate(sememe_sets):
        for word in sememe_set:
            sememe_rows.append(j)
            sememe_cols.append(word_id(str(word)))

    vocab_size = len(idx_to_word)
    feature_matrix = sparse.csr_matrix(
        (
            np.ones(len(feature_rows), dtype=np.int32),
            (np.asarray(feature_rows, dtype=np.int64), np.asarray(feature_cols, dtype=np.int64)),
        ),
        shape=(len(flat_features), vocab_size),
        dtype=np.int32,
    )
    sememe_matrix = sparse.csr_matrix(
        (
            np.ones(len(sememe_rows), dtype=np.int32),
            (np.asarray(sememe_rows, dtype=np.int64), np.asarray(sememe_cols, dtype=np.int64)),
        ),
        shape=(len(sememe_sets), vocab_size),
        dtype=np.int32,
    )
    feature_sizes = np.asarray(feature_matrix.sum(axis=1)).ravel().astype(np.int64)
    sememe_sizes = np.asarray(sememe_matrix.sum(axis=1)).ravel().astype(np.int64)
    return feature_matrix, sememe_matrix, feature_sizes, sememe_sizes, word_to_idx, idx_to_word


def save_prepared(
    root: str | Path,
    flat_features: list[dict[str, Any]],
    sememe_sets: list[set[str]],
    sememe_meta: list[dict[str, str]],
    feature_matrix,
    sememe_matrix,
    feature_sizes,
    sememe_sizes,
    idx_to_word: list[str],
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(root / "feature_matrix.npz", feature_matrix)
    sparse.save_npz(root / "sememe_matrix.npz", sememe_matrix)
    np.save(root / "feature_sizes.npy", feature_sizes)
    np.save(root / "sememe_sizes.npy", sememe_sizes)
    with open(root / "flat_features.jsonl", "w", encoding="utf-8") as f:
        for feat in flat_features:
            row = {k: v for k, v in feat.items() if k != "feature_set"}
            row["feature_words"] = sorted(feat["feature_set"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(root / "sememes.jsonl", "w", encoding="utf-8") as f:
        for meta, words in zip(sememe_meta, sememe_sets):
            f.write(json.dumps({**meta, "entities": sorted(words)}, ensure_ascii=False) + "\n")
    with open(root / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_word, f, ensure_ascii=False)
    return root


def load_prepared(root: str | Path):
    root = Path(root)
    feature_matrix = sparse.load_npz(root / "feature_matrix.npz").tocsr()
    sememe_matrix = sparse.load_npz(root / "sememe_matrix.npz").tocsr()
    feature_sizes = np.load(root / "feature_sizes.npy")
    sememe_sizes = np.load(root / "sememe_sizes.npy")
    flat_features = []
    with open(root / "flat_features.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["feature_set"] = set(row.pop("feature_words"))
            flat_features.append(row)
    sememe_sets = []
    sememe_meta = []
    with open(root / "sememes.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sememe_sets.append(set(row.pop("entities")))
            sememe_meta.append(row)
    with open(root / "vocabulary.json", "r", encoding="utf-8") as f:
        idx_to_word = json.load(f)
    return flat_features, sememe_sets, sememe_meta, feature_matrix, sememe_matrix, feature_sizes, sememe_sizes, idx_to_word
