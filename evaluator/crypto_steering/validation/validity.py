"""Primary source-level validity contract for generated programs."""

from __future__ import annotations

from collections.abc import Iterable

from crypto_steering.validation.models import (
    CompileOutcome,
    ImportOutcome,
    TaskValidationResult,
)
from crypto_steering.validation.rules.aggregate import program_rule_assessments
from crypto_steering.validation.rules.models import RuleAssessment


def assess_structural_api_validity(
    compile_or_parse: CompileOutcome,
    imports: ImportOutcome,
    target_tasks: Iterable[TaskValidationResult],
) -> tuple[RuleAssessment, ...]:
    """Return the six ordered rule assessments for one program."""

    return program_rule_assessments(compile_or_parse, imports, target_tasks)


def is_structurally_api_valid(
    compile_or_parse: CompileOutcome,
    imports: ImportOutcome,
    target_tasks: Iterable[TaskValidationResult],
) -> bool:
    """Return whether a program satisfies the benchmark validity contract.

    Task validation owns API-family use, required operations, target-API
    argument positions, and required operation sequence. Runtime behavior and
    detector security labels are deliberately outside this predicate.
    """

    return all(
        rule.success
        for rule in assess_structural_api_validity(
            compile_or_parse,
            imports,
            target_tasks,
        )
    )
