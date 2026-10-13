"""Public factory and compatibility imports for source analysis."""

from __future__ import annotations

from crypto_steering.validation.models import AnalysisProfile, Language
from crypto_steering.validation.source.base import (
    Assignment,
    CallExpression,
    ClassDefinition,
    FunctionDefinition,
    ReturnExpression,
    SourceAnalyzer,
)
from crypto_steering.validation.source.cstyle import CStyleAnalyzer
from crypto_steering.validation.source.helpers import (
    _c_style_object_value_expressions,
    csharp_activator_constructor_arguments,
    find_matching,
    mask_constant_boolean_inactive_branches,
    mask_c_style_comments_and_literals,
    split_arguments,
)
from crypto_steering.validation.source.java import JavaAnalyzer
from crypto_steering.validation.source.python import PythonAnalyzer


def analyzer_for(
    language: Language,
    source: str,
    *,
    analysis_profile: AnalysisProfile = "whole_program",
) -> SourceAnalyzer:
    """Construct the analyzer owned by one source language."""

    if language == "python":
        return PythonAnalyzer(source, analysis_profile=analysis_profile)
    if language == "java":
        return JavaAnalyzer(source, analysis_profile=analysis_profile)
    if language in {"typescript", "csharp"}:
        return CStyleAnalyzer(
            source,
            language,
            analysis_profile=analysis_profile,
        )
    raise ValueError(f"No multilingual source analyzer for {language!r}")


__all__ = (
    "_c_style_object_value_expressions",
    "Assignment",
    "CallExpression",
    "ClassDefinition",
    "CStyleAnalyzer",
    "FunctionDefinition",
    "JavaAnalyzer",
    "PythonAnalyzer",
    "ReturnExpression",
    "SourceAnalyzer",
    "analyzer_for",
    "csharp_activator_constructor_arguments",
    "find_matching",
    "mask_constant_boolean_inactive_branches",
    "mask_c_style_comments_and_literals",
    "split_arguments",
)
