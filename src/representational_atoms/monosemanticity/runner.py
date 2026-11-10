from __future__ import annotations

from typing import Mapping

from .candidates import mine_candidates
from .review import build_review_sets


def prepare_monosemanticity(config: Mapping, *, stage: str = "all"):
    stage = str(stage).lower()
    if stage not in {"all", "candidates", "review"}:
        raise ValueError("stage must be all, candidates, or review")

    review_enabled = bool(config.get("review", {}).get("enabled", True))
    if stage == "review" and not review_enabled:
        raise ValueError(
            "Review-set construction is disabled by this config. "
            "Set review.enabled=true to enable it explicitly."
        )

    root = None
    if stage in {"all", "candidates"}:
        root = mine_candidates(config)
    if stage in {"all", "review"} and review_enabled:
        root = build_review_sets(config)
    return root
