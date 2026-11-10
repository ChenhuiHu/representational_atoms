from __future__ import annotations

from typing import Any

import numpy as np


def compute_best_and_candidates_batch(
    feature_matrix,
    sememe_matrix,
    feature_sizes,
    sememe_sizes,
    sememe_start: int,
    sememe_end: int,
    *,
    min_precision: float = 0.4,
    min_overlap: int = 1,
    min_f1: float = 0.0,
    max_candidates_per_sememe: int | None = None,
):
    inter_batch = (feature_matrix @ sememe_matrix[sememe_start:sememe_end].T).tocsc()
    batch_results = []
    for local_j in range(sememe_end - sememe_start):
        j = sememe_start + local_j
        sememe_size = int(sememe_sizes[j])
        a, b = inter_batch.indptr[local_j], inter_batch.indptr[local_j + 1]
        feat_idx = inter_batch.indices[a:b].astype(np.int64, copy=False)
        inter = inter_batch.data[a:b].astype(np.float64, copy=False)
        if feat_idx.size == 0:
            batch_results.append({
                "best_idx": None,
                "best_precision": 0.0,
                "best_recall": 0.0,
                "best_f1": 0.0,
                "best_inter": 0,
                "candidate_idx": np.empty(0, dtype=np.int64),
                "candidate_p": np.empty(0, dtype=np.float64),
                "candidate_r": np.empty(0, dtype=np.float64),
                "candidate_f1": np.empty(0, dtype=np.float64),
                "candidate_inter": np.empty(0, dtype=np.int64),
            })
            continue

        feature_size = feature_sizes[feat_idx].astype(np.float64, copy=False)
        p = inter / feature_size
        r = inter / float(sememe_size)
        f1 = 2.0 * inter / (feature_size + float(sememe_size))
        best_pos = int(np.argmax(f1))
        best_idx = int(feat_idx[best_pos])

        mask = (p >= float(min_precision)) & (inter >= int(min_overlap)) & (f1 >= float(min_f1))
        cand_idx = feat_idx[mask]
        cand_p = p[mask]
        cand_r = r[mask]
        cand_f1 = f1[mask]
        cand_inter = inter[mask]
        if cand_idx.size > 0:
            # Exact notebook order: precision desc, F1 desc, overlap desc.
            order = np.lexsort((-cand_inter, -cand_f1, -cand_p))
            cand_idx = cand_idx[order]
            cand_p = cand_p[order]
            cand_r = cand_r[order]
            cand_f1 = cand_f1[order]
            cand_inter = cand_inter[order]
        if max_candidates_per_sememe is not None:
            k = int(max_candidates_per_sememe)
            cand_idx = cand_idx[:k]
            cand_p = cand_p[:k]
            cand_r = cand_r[:k]
            cand_f1 = cand_f1[:k]
            cand_inter = cand_inter[:k]

        batch_results.append({
            "best_idx": best_idx,
            "best_precision": float(p[best_pos]),
            "best_recall": float(r[best_pos]),
            "best_f1": float(f1[best_pos]),
            "best_inter": int(inter[best_pos]),
            "candidate_idx": cand_idx.astype(np.int64, copy=False),
            "candidate_p": cand_p.astype(np.float64, copy=False),
            "candidate_r": cand_r.astype(np.float64, copy=False),
            "candidate_f1": cand_f1.astype(np.float64, copy=False),
            "candidate_inter": cand_inter.astype(np.int64, copy=False),
        })
    return batch_results


def greedy_merge_for_sememe(
    sememe_idx,
    candidate_idx,
    candidate_p,
    candidate_r,
    candidate_f1,
    candidate_inter,
    *,
    feature_matrix,
    sememe_matrix,
    feature_sizes,
    flat_features,
    idx_to_word,
    max_features=None,
    require_positive_gain=True,
):
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    candidate_p = np.asarray(candidate_p, dtype=np.float64)
    candidate_r = np.asarray(candidate_r, dtype=np.float64)
    candidate_f1 = np.asarray(candidate_f1, dtype=np.float64)
    candidate_inter = np.asarray(candidate_inter, dtype=np.float64)
    num_candidates = len(candidate_idx)
    if num_candidates == 0:
        sememe_word_ids = sememe_matrix[sememe_idx].indices
        return {
            "selected_features": [],
            "merged_precision": 0.0,
            "merged_recall": 0.0,
            "merged_f1": 0.0,
            "merged_overlap": 0,
            "merged_size": 0,
            "history": [],
            "overlap_words": [],
            "feature_only_words": [],
            "sememe_only_words": sorted(idx_to_word[int(i)] for i in sememe_word_ids),
        }

    candidate_matrix = feature_matrix[candidate_idx].astype(np.int32).tocsr()
    vocab_size = feature_matrix.shape[1]
    candidate_sizes = feature_sizes[candidate_idx].astype(np.float64)
    gold_ids = sememe_matrix[sememe_idx].indices
    gold_size = len(gold_ids)
    gold_mask = np.zeros(vocab_size, dtype=np.int32)
    gold_mask[gold_ids] = 1
    merged_mask = np.zeros(vocab_size, dtype=np.int32)
    covered_gold_mask = np.zeros(vocab_size, dtype=np.int32)
    selected_mask = np.zeros(num_candidates, dtype=bool)

    selected = []
    history = []
    merged_size = 0.0
    cur_inter = 0.0
    cur_p = cur_r = cur_f1 = 0.0

    while True:
        overlap_with_merged = np.asarray(candidate_matrix @ merged_mask).ravel().astype(np.float64)
        overlap_with_covered_gold = np.asarray(candidate_matrix @ covered_gold_mask).ravel().astype(np.float64)
        new_size = merged_size + candidate_sizes - overlap_with_merged
        new_inter = cur_inter + candidate_inter - overlap_with_covered_gold
        new_f1 = np.divide(
            2.0 * new_inter,
            new_size + float(gold_size),
            out=np.zeros_like(new_inter, dtype=np.float64),
            where=(new_size + float(gold_size)) > 0,
        )
        gain = new_f1 - cur_f1
        gain[selected_mask] = -np.inf
        best_pos = int(np.argmax(gain))
        best_gain = float(gain[best_pos])
        if not np.isfinite(best_gain):
            break
        if require_positive_gain and best_gain <= 0.0:
            break

        global_idx = int(candidate_idx[best_pos])
        feature_word_ids = feature_matrix[global_idx].indices
        merged_mask[feature_word_ids] = 1
        covered_gold_mask[feature_word_ids] = gold_mask[feature_word_ids]
        merged_size = float(new_size[best_pos])
        cur_inter = float(new_inter[best_pos])
        cur_f1 = float(new_f1[best_pos])
        cur_p = cur_inter / merged_size if merged_size > 0 else 0.0
        cur_r = cur_inter / float(gold_size) if gold_size > 0 else 0.0
        selected_mask[best_pos] = True
        feat = flat_features[global_idx]
        selected.append({
            "global_idx": global_idx,
            "layer": int(feat["layer"]),
            "feature_id": int(feat["feature_id"]),
            "feature_name": str(feat["feature_name"]),
            "direction": str(feat.get("direction", "unknown")),
            "single_precision": float(candidate_p[best_pos]),
            "single_recall": float(candidate_r[best_pos]),
            "single_f1": float(candidate_f1[best_pos]),
            "single_overlap": int(candidate_inter[best_pos]),
            "gain_f1": float(best_gain),
        })
        history.append({
            "num_selected": len(selected),
            "current_precision": float(cur_p),
            "current_recall": float(cur_r),
            "current_f1": float(cur_f1),
            "current_overlap": int(cur_inter),
        })
        if max_features is not None and len(selected) >= int(max_features):
            break

    merged_word_ids = np.flatnonzero(merged_mask)
    overlap_ids = merged_word_ids[gold_mask[merged_word_ids] > 0]
    feature_only_ids = merged_word_ids[gold_mask[merged_word_ids] == 0]
    sememe_only_ids = gold_ids[merged_mask[gold_ids] == 0]
    return {
        "selected_features": selected,
        "merged_precision": float(cur_p),
        "merged_recall": float(cur_r),
        "merged_f1": float(cur_f1),
        "merged_overlap": int(cur_inter),
        "merged_size": int(merged_size),
        "history": history,
        "overlap_words": sorted(idx_to_word[int(i)] for i in overlap_ids),
        "feature_only_words": sorted(idx_to_word[int(i)] for i in feature_only_ids),
        "sememe_only_words": sorted(idx_to_word[int(i)] for i in sememe_only_ids),
    }


def run_alignment(
    *,
    sememe_sets,
    sememe_meta,
    flat_features,
    feature_matrix,
    sememe_matrix,
    feature_sizes,
    sememe_sizes,
    idx_to_word,
    min_precision=0.4,
    min_overlap=1,
    min_f1=0.0,
    max_candidates_per_sememe=None,
    max_features_to_merge=None,
    require_positive_gain=True,
    sememe_batch_size=64,
    run_greedy=True,
):
    all_results: list[dict[str, Any]] = []
    num_sememes = len(sememe_sets)
    for batch_start in range(0, num_sememes, int(sememe_batch_size)):
        batch_end = min(batch_start + int(sememe_batch_size), num_sememes)
        batch_stats = compute_best_and_candidates_batch(
            feature_matrix,
            sememe_matrix,
            feature_sizes,
            sememe_sizes,
            batch_start,
            batch_end,
            min_precision=min_precision,
            min_overlap=min_overlap,
            min_f1=min_f1,
            max_candidates_per_sememe=max_candidates_per_sememe,
        )
        for local_j, stats in enumerate(batch_stats):
            j = batch_start + local_j
            sememe_set = sememe_sets[j]
            meta = sememe_meta[j]
            best_idx = stats["best_idx"]
            if best_idx is None:
                best_feature_set = set()
                best_single = {
                    "global_idx": None, "layer": None, "feature_id": None, "feature_name": None,
                    "direction": None, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "overlap_words": [], "feature_only_words": [], "sememe_only_words": sorted(sememe_set),
                }
            else:
                feat = flat_features[best_idx]
                best_feature_set = feat["feature_set"]
                best_single = {
                    "global_idx": int(best_idx),
                    "layer": int(feat["layer"]),
                    "feature_id": int(feat["feature_id"]),
                    "feature_name": str(feat["feature_name"]),
                    "direction": str(feat.get("direction", "unknown")),
                    "precision": float(stats["best_precision"]),
                    "recall": float(stats["best_recall"]),
                    "f1": float(stats["best_f1"]),
                    "overlap_words": sorted(best_feature_set & sememe_set),
                    "feature_only_words": sorted(best_feature_set - sememe_set),
                    "sememe_only_words": sorted(sememe_set - best_feature_set),
                }

            result = {
                "sememe": meta["sememe"],
                "sememe_name": meta["sememe_name"],
                "sememe_size": int(len(sememe_set)),
                "best_single_feature": {
                    **{k: best_single[k] for k in ["global_idx", "layer", "feature_id", "feature_name", "direction", "precision", "recall", "f1"]},
                    "feature_size": int(len(best_feature_set)),
                    "overlap_size": len(best_single["overlap_words"]),
                    "overlap_words": best_single["overlap_words"],
                    "feature_only_size": len(best_single["feature_only_words"]),
                    "feature_only_words": best_single["feature_only_words"],
                    "sememe_only_size": len(best_single["sememe_only_words"]),
                    "sememe_only_words": best_single["sememe_only_words"],
                },
                "prefilter": {
                    "min_precision": float(min_precision),
                    "min_overlap": int(min_overlap),
                    "min_f1": float(min_f1),
                    "num_candidates": int(len(stats["candidate_idx"])),
                },
            }

            if run_greedy:
                merged = greedy_merge_for_sememe(
                    sememe_idx=j,
                    candidate_idx=stats["candidate_idx"],
                    candidate_p=stats["candidate_p"],
                    candidate_r=stats["candidate_r"],
                    candidate_f1=stats["candidate_f1"],
                    candidate_inter=stats["candidate_inter"],
                    feature_matrix=feature_matrix,
                    sememe_matrix=sememe_matrix,
                    feature_sizes=feature_sizes,
                    flat_features=flat_features,
                    idx_to_word=idx_to_word,
                    max_features=max_features_to_merge,
                    require_positive_gain=require_positive_gain,
                )
                layers = sorted({int(x["layer"]) for x in merged["selected_features"]})
                result["greedy_merge"] = {
                    "num_selected_features": int(len(merged["selected_features"])),
                    "selected_features": merged["selected_features"],
                    "cross_layer_width": len(layers),
                    "selected_layers": layers,
                    "merged_precision": merged["merged_precision"],
                    "merged_recall": merged["merged_recall"],
                    "merged_f1": merged["merged_f1"],
                    "merged_overlap": merged["merged_overlap"],
                    "merged_size": merged["merged_size"],
                    "overlap_size": len(merged["overlap_words"]),
                    "overlap_words": merged["overlap_words"],
                    "feature_only_size": len(merged["feature_only_words"]),
                    "feature_only_words": merged["feature_only_words"],
                    "sememe_only_size": len(merged["sememe_only_words"]),
                    "sememe_only_words": merged["sememe_only_words"],
                    "history": merged["history"],
                }
            all_results.append(result)
    key = (lambda x: x["greedy_merge"]["merged_f1"]) if run_greedy else (lambda x: x["best_single_feature"]["f1"])
    return sorted(all_results, key=key, reverse=True)


def summarize_results(results, *, include_greedy=True):
    single_f1 = np.asarray([x["best_single_feature"]["f1"] for x in results], dtype=float)
    single_p = np.asarray([x["best_single_feature"]["precision"] for x in results], dtype=float)
    single_r = np.asarray([x["best_single_feature"]["recall"] for x in results], dtype=float)
    summary = {
        "num_sememes": int(len(results)),
        "single_mean_precision": float(single_p.mean()) if len(single_p) else 0.0,
        "single_mean_recall": float(single_r.mean()) if len(single_r) else 0.0,
        "single_mean_f1": float(single_f1.mean()) if len(single_f1) else 0.0,
    }
    if include_greedy:
        merged_f1 = np.asarray([x["greedy_merge"]["merged_f1"] for x in results], dtype=float)
        merged_p = np.asarray([x["greedy_merge"]["merged_precision"] for x in results], dtype=float)
        merged_r = np.asarray([x["greedy_merge"]["merged_recall"] for x in results], dtype=float)
        num_selected = np.asarray([x["greedy_merge"]["num_selected_features"] for x in results], dtype=int)
        summary.update({
            "merged_mean_precision": float(merged_p.mean()) if len(merged_p) else 0.0,
            "merged_mean_recall": float(merged_r.mean()) if len(merged_r) else 0.0,
            "merged_mean_f1": float(merged_f1.mean()) if len(merged_f1) else 0.0,
            "mean_num_selected_features": float(num_selected.mean()) if len(num_selected) else 0.0,
            "num_sememes_merged_f1_gt_0.1": int(np.sum(merged_f1 > 0.1)),
            "num_sememes_merged_f1_gt_0.3": int(np.sum(merged_f1 > 0.3)),
            "num_sememes_merged_f1_gt_0.5": int(np.sum(merged_f1 > 0.5)),
        })
    return summary
