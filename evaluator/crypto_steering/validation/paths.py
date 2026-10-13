"""Portable paths for validation-owned resources."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path, root: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(root) if root is not None else project_root()) / path

