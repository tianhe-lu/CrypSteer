"""Typed indexed-program boundary shared by tasks and detectors."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_steering.validation.models import ProgramInput
from crypto_steering.validation.source.index import SourceAnalyzer, analyzer_for


@dataclass(frozen=True)
class IndexedProgram:
    program: ProgramInput
    analyzer: SourceAnalyzer | None


def index_program(program: ProgramInput) -> IndexedProgram:
    analyzer = analyzer_for(
        program.language,
        program.source,
        analysis_profile=program.analysis_profile,
    )
    return IndexedProgram(program=program, analyzer=analyzer)
