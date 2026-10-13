"""Small provider protocols for validation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from crypto_steering.validation.frontends.engine import FrontendAudit
from crypto_steering.validation.models import (
    DetectionResult,
    Language,
    ProgramInput,
    ProviderExecution,
    TaskValidationResult,
    ValidationMode,
)
from crypto_steering.validation.source.index import SourceAnalyzer


class FrontendProvider(Protocol):
    def check_many(
        self,
        programs: Sequence[ProgramInput],
        *,
        work_dir: Path,
        logs_dir: Path,
    ) -> FrontendAudit: ...


class TaskValidatorProvider(Protocol):
    def validate(
        self,
        language: Language,
        source: str,
        *,
        task_id: str,
        target_misuse_id: str,
        mode: ValidationMode = "strict",
        target_detection: DetectionResult | None = None,
        indexed_analyzer: SourceAnalyzer | None = None,
    ) -> ProviderExecution[TaskValidationResult]: ...


class MisuseDetectorProvider(Protocol):
    misuse_id: str

    def detect(
        self,
        language: Language,
        source: str,
        *,
        task_id: str | None = None,
        modern_kdf_iterations: int = 100_000,
        include_inactive_findings: bool = False,
        indexed_analyzer: SourceAnalyzer | None = None,
    ) -> ProviderExecution[DetectionResult]: ...
