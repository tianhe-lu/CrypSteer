"""Task-neutral inputs shared by structural family validators."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_steering.validation.models import (
    DetectionResult,
    Language,
)
from crypto_steering.validation.source.index import SourceAnalyzer


@dataclass(frozen=True)
class TaskContext:
    language: Language
    source: str
    task_id: str
    target_misuse_id: str
    analyzer: SourceAnalyzer | None = None
    target_detection: DetectionResult | None = None
    allow_independent_routes: bool = True
