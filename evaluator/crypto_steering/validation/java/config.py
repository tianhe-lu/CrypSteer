"""Small configuration helpers for the migrated Java policy engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_project_path(value: str | Path, root: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(root) if root is not None else project_root()) / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping at {path}")
    return payload

