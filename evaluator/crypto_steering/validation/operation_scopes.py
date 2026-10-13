"""Operation-role scoping for full-factorial composition probes."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from crypto_steering.validation.java.lightweight_ast import (
    JavaMethodNode,
    method_containing_index,
    parse_method_nodes,
)
from crypto_steering.validation.models import (
    DetectionResult,
    DetectorFinding,
    Language,
    ProviderExecution,
    combine_detector_findings,
)
from crypto_steering.validation.source.index import SourceAnalyzer


def scope_detector_executions_to_operations(
    language: Language,
    source: str,
    detections: Mapping[str, ProviderExecution[DetectionResult]],
    operation_functions: Mapping[str, str],
    *,
    indexed_analyzer: SourceAnalyzer | None,
) -> dict[str, ProviderExecution[DetectionResult]]:
    """Project whole-source findings onto each declared operation function.

    A composition program deliberately contains ten independent security
    operations.  Its M01-M10 bits label those roles, not the aggregate set of
    every sink in the file.  The detectors still inspect the complete source
    so their normal value-flow resolution remains available; only aggregation
    is scoped to the function assigned to the corresponding role.
    """

    java_methods = parse_method_nodes(source) if language == "java" else ()
    if language != "java" and indexed_analyzer is None:
        message = "operation scoping requires the indexed source analyzer"
        return {
            misuse_id: ProviderExecution.error(message)
            for misuse_id in operation_functions
        }

    return {
        misuse_id: _scope_execution(
            language,
            execution,
            function_name=operation_functions[misuse_id],
            java_methods=java_methods,
            indexed_analyzer=indexed_analyzer,
        )
        for misuse_id, execution in detections.items()
    }


def _scope_execution(
    language: Language,
    execution: ProviderExecution[DetectionResult],
    *,
    function_name: str,
    java_methods: tuple[JavaMethodNode, ...],
    indexed_analyzer: SourceAnalyzer | None,
) -> ProviderExecution[DetectionResult]:
    if execution.status != "completed" or execution.value is None:
        return execution

    findings = tuple(
        _annotate_scope(finding, function_name)
        for finding in execution.value.findings
        if _finding_function_name(
            language,
            finding,
            java_methods=java_methods,
            indexed_analyzer=indexed_analyzer,
        )
        == function_name
    )
    scoped = combine_detector_findings(
        execution.value.misuse_id,
        language,
        findings,
        no_sink_reason=(
            "no registered target sink found in composition operation "
            f"{function_name}"
        ),
    )
    return ProviderExecution.completed(
        replace(
            scoped,
            reason=f"operation {function_name}: {scoped.reason}",
        )
    )


def _finding_function_name(
    language: Language,
    finding: DetectorFinding,
    *,
    java_methods: tuple[JavaMethodNode, ...],
    indexed_analyzer: SourceAnalyzer | None,
) -> str | None:
    if finding.span.start < 0:
        return None
    if language == "java":
        method = method_containing_index(java_methods, finding.span.start)
        return method.name if method is not None else None
    if indexed_analyzer is None:
        return None
    return indexed_analyzer.containing_function_name(finding.span.start)


def _annotate_scope(
    finding: DetectorFinding,
    function_name: str,
) -> DetectorFinding:
    return replace(
        finding,
        metadata={
            **finding.metadata,
            "operation_scope": function_name,
        },
    )


__all__ = ("scope_detector_executions_to_operations",)
