"""Detector dispatch and fail-soft execution.

Each M01-M10 module owns its behavior. This module prepares one shared source
analyzer, dispatches by misuse category, and preserves the public API.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable

from crypto_steering.validation.models import (
    AnalysisProfile,
    DetectionResult,
    Language,
    MISUSE_IDS,
)
from crypto_steering.validation.source.index import (
    SourceAnalyzer,
    analyzer_for,
    mask_constant_boolean_inactive_branches,
)


DetectorImplementation = Callable[
    [SourceAnalyzer, str | None, int],
    DetectionResult,
]


def detect_all(
    language: Language,
    source: str,
    *,
    task_id: str | None = None,
    modern_kdf_iterations: int = 100_000,
    include_inactive_findings: bool = False,
    analysis_profile: AnalysisProfile = "whole_program",
    indexed_analyzer: SourceAnalyzer | None = None,
) -> dict[str, DetectionResult]:
    """Run all ten owned detector modules against one prepared source index."""

    analysis_source = source_for_analysis_profile(language, source, analysis_profile)
    analyzer, analyzer_error = prepare_analyzer(
        language,
        analysis_source,
        include_inactive_findings=include_inactive_findings,
        analysis_profile=analysis_profile,
        indexed_analyzer=indexed_analyzer,
    )
    return {
        misuse_id: detect_prepared(
            language,
            misuse_id,
            analysis_source,
            task_id=task_id,
            modern_kdf_iterations=modern_kdf_iterations,
            analyzer=analyzer,
            analyzer_error=analyzer_error,
        )
        for misuse_id in MISUSE_IDS
    }


def detect_misuse(
    language: Language,
    misuse_id: str,
    source: str,
    *,
    task_id: str | None = None,
    modern_kdf_iterations: int = 100_000,
    include_inactive_findings: bool = False,
    analysis_profile: AnalysisProfile = "whole_program",
    indexed_analyzer: SourceAnalyzer | None = None,
) -> DetectionResult:
    normalized = _normalized_misuse_id(misuse_id)
    analysis_source = source_for_analysis_profile(language, source, analysis_profile)
    analyzer, analyzer_error = prepare_analyzer(
        language,
        analysis_source,
        include_inactive_findings=include_inactive_findings,
        analysis_profile=analysis_profile,
        indexed_analyzer=indexed_analyzer,
    )
    return detect_prepared(
        language,
        normalized,
        analysis_source,
        task_id=task_id,
        modern_kdf_iterations=modern_kdf_iterations,
        analyzer=analyzer,
        analyzer_error=analyzer_error,
    )


def prepare_analyzer(
    language: Language,
    source: str,
    *,
    include_inactive_findings: bool = False,
    analysis_profile: AnalysisProfile = "whole_program",
    indexed_analyzer: SourceAnalyzer | None = None,
) -> tuple[SourceAnalyzer | None, Exception | None]:
    """Create one language-owned source index for all detector categories."""

    analyzer = indexed_analyzer
    try:
        if (
            analyzer is None
            or analyzer.language != language
            or analyzer.source != source
            or analyzer.analysis_profile != analysis_profile
        ):
            analyzer = analyzer_for(
                language,
                source,
                analysis_profile=analysis_profile,
            )
        analyzer.include_inactive_findings = include_inactive_findings
        return analyzer, None
    except Exception as exc:
        return None, exc


def source_for_analysis_profile(
    language: Language,
    source: str,
    analysis_profile: AnalysisProfile,
) -> str:
    """Apply profile-scoped, offset-preserving source normalization."""

    if language == "java" and analysis_profile == "reachable_security_flow":
        return mask_constant_boolean_inactive_branches(source)
    return source


def detect_prepared(
    language: Language,
    misuse_id: str,
    source: str,
    *,
    task_id: str | None,
    modern_kdf_iterations: int,
    analyzer: SourceAnalyzer | None,
    analyzer_error: Exception | None = None,
) -> DetectionResult:
    """Dispatch to the module that owns one misuse category."""

    normalized = _normalized_misuse_id(misuse_id)
    module = import_module(
        f"crypto_steering.validation.detectors.{normalized.lower()}"
    )
    return module.detect(
        language,
        source,
        task_id=task_id,
        modern_kdf_iterations=modern_kdf_iterations,
        analyzer=analyzer,
        analyzer_error=analyzer_error,
    )


def execute_detector(
    language: Language,
    misuse_id: str,
    source: str,
    *,
    task_id: str | None,
    modern_kdf_iterations: int,
    analyzer: SourceAnalyzer | None,
    analyzer_error: Exception | None,
    implementation: DetectorImplementation,
) -> DetectionResult:
    """Run an owned detector implementation with the common error contract."""

    normalized = _normalized_misuse_id(misuse_id)
    try:
        if analyzer_error is not None or analyzer is None:
            raise analyzer_error or RuntimeError("source analyzer was not created")
        return implementation(analyzer, task_id, modern_kdf_iterations)
    except Exception as exc:
        return DetectionResult(
            misuse_id=normalized,
            language=language,
            label="unknown",
            reason=f"detector analysis failed: {type(exc).__name__}: {exc}",
            analysis_error=f"{type(exc).__name__}: {exc}",
        )


def _normalized_misuse_id(misuse_id: str) -> str:
    normalized = misuse_id.upper()
    if normalized not in MISUSE_IDS:
        raise KeyError(f"Unsupported misuse ID {misuse_id!r}")
    return normalized
