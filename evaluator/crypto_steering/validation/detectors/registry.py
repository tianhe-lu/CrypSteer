"""Typed registry over ten independent detector-provider entry points."""

from __future__ import annotations

from dataclasses import dataclass

import crypto_steering.validation.detectors.engine as engine
from crypto_steering.validation.detectors.m01 import detect as detect_m01
from crypto_steering.validation.detectors.m02 import detect as detect_m02
from crypto_steering.validation.detectors.m03 import detect as detect_m03
from crypto_steering.validation.detectors.m04 import detect as detect_m04
from crypto_steering.validation.detectors.m05 import detect as detect_m05
from crypto_steering.validation.detectors.m06 import detect as detect_m06
from crypto_steering.validation.detectors.m07 import detect as detect_m07
from crypto_steering.validation.detectors.m08 import detect as detect_m08
from crypto_steering.validation.detectors.m09 import detect as detect_m09
from crypto_steering.validation.detectors.m10 import detect as detect_m10
from crypto_steering.validation.models import (
    AnalysisProfile,
    DetectionResult,
    Language,
    MISUSE_IDS,
    ProviderExecution,
)
from crypto_steering.validation.source.index import SourceAnalyzer


_DETECTOR_ENTRYPOINTS = {
    "M01": detect_m01,
    "M02": detect_m02,
    "M03": detect_m03,
    "M04": detect_m04,
    "M05": detect_m05,
    "M06": detect_m06,
    "M07": detect_m07,
    "M08": detect_m08,
    "M09": detect_m09,
    "M10": detect_m10,
}


@dataclass(frozen=True)
class InternalDetectorProvider:
    misuse_id: str

    def __post_init__(self) -> None:
        if self.misuse_id not in MISUSE_IDS:
            raise ValueError(f"Unsupported misuse detector {self.misuse_id!r}")

    def detect(
        self,
        language: Language,
        source: str,
        *,
        task_id: str | None = None,
        modern_kdf_iterations: int = 100_000,
        include_inactive_findings: bool = False,
        analysis_profile: AnalysisProfile = "whole_program",
        indexed_analyzer: SourceAnalyzer | None = None,
    ) -> ProviderExecution[DetectionResult]:
        analysis_source = engine.source_for_analysis_profile(
            language,
            source,
            analysis_profile,
        )
        analyzer, analyzer_error = engine.prepare_analyzer(
            language,
            analysis_source,
            include_inactive_findings=include_inactive_findings,
            analysis_profile=analysis_profile,
            indexed_analyzer=indexed_analyzer,
        )
        result = _DETECTOR_ENTRYPOINTS[self.misuse_id](
            language,
            analysis_source,
            task_id=task_id,
            modern_kdf_iterations=modern_kdf_iterations,
            analyzer=analyzer,
            analyzer_error=analyzer_error,
        )
        return _execution(result)


def detector_provider(misuse_id: str) -> InternalDetectorProvider:
    return InternalDetectorProvider(misuse_id.upper())


def detect_all(
    language: Language,
    source: str,
    *,
    task_id: str | None = None,
    modern_kdf_iterations: int = 100_000,
    include_inactive_findings: bool = False,
    analysis_profile: AnalysisProfile = "whole_program",
    indexed_analyzer: SourceAnalyzer | None = None,
) -> dict[str, ProviderExecution[DetectionResult]]:
    analysis_source = engine.source_for_analysis_profile(
        language,
        source,
        analysis_profile,
    )
    analyzer, analyzer_error = engine.prepare_analyzer(
        language,
        analysis_source,
        include_inactive_findings=include_inactive_findings,
        analysis_profile=analysis_profile,
        indexed_analyzer=indexed_analyzer,
    )
    return {
        misuse_id: _execution(
            _DETECTOR_ENTRYPOINTS[misuse_id](
                language,
                analysis_source,
                task_id=task_id,
                modern_kdf_iterations=modern_kdf_iterations,
                analyzer=analyzer,
                analyzer_error=analyzer_error,
            )
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
) -> ProviderExecution[DetectionResult]:
    return detector_provider(misuse_id).detect(
        language,
        source,
        task_id=task_id,
        modern_kdf_iterations=modern_kdf_iterations,
        include_inactive_findings=include_inactive_findings,
        analysis_profile=analysis_profile,
        indexed_analyzer=indexed_analyzer,
    )


def _execution(result: DetectionResult) -> ProviderExecution[DetectionResult]:
    if result.analysis_error:
        return ProviderExecution.error(result.analysis_error)
    return ProviderExecution.completed(result)
