from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    return {} if value is None else value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def parse_scalar(text: str):
    return yaml.safe_load(text)


def set_nested(config: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"Cannot set {dotted_key}: {part} is not a mapping")
    cursor[parts[-1]] = value


def apply_overrides(config: dict, overrides: list[str] | None) -> dict:
    result = deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_nested(result, key, parse_scalar(raw))
    return result
