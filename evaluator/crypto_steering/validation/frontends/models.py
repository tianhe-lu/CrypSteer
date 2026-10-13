"""Runtime-only frontend configuration passed to compiler adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolchainRuntimeConfig:
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrontendRuntimeConfig:
    version: str
    source_config_path: str
    toolchains: dict[str, ToolchainRuntimeConfig]
