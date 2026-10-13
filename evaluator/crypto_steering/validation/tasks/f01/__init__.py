"""F1 symmetric-cryptography validation, organized by Rules 3-6."""

from __future__ import annotations

from crypto_steering.validation.models import Language, TaskValidationResult
from crypto_steering.validation.rules.aggregate import task_result_with_rule_assessments
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f01 import (
    api_family,
    arguments,
    operations,
    sequence,
)


def validate(context: TaskContext) -> TaskValidationResult:
    """Assess F1 and expose its Rules 3-6 decisions."""

    return task_result_with_rule_assessments(_validate_family(context))


def _validate_family(context: TaskContext) -> TaskValidationResult:
    return _validate_one(context)


def _validate_one(context: TaskContext) -> TaskValidationResult:
    analysis = api_family.analyze(context)
    family_failure = api_family.validate_rule(context, analysis)
    if family_failure is not None:
        return family_failure

    # Java historically distinguishes a malformed init (Rule 5) from a
    # missing init (Rule 4), so argument shape is resolved before operation
    # completeness without changing the public Rules 3-6 ordering.
    argument_failure = arguments.validate_rule(context, analysis)
    if argument_failure is not None:
        return argument_failure
    operation_failure = operations.validate_rule(context, analysis)
    if operation_failure is not None:
        return operation_failure
    sequence_failure = sequence.validate_rule(context, analysis)
    if sequence_failure is not None:
        return sequence_failure
    return operations.success_result(context, analysis)


def mode_sink_is_rewriteable(language: Language, source: str) -> bool:
    return api_family.mode_sink_is_rewriteable(language, source)


def _validate_indexed(context: TaskContext) -> TaskValidationResult:
    return _validate_one(context)


def _validate_java(context: TaskContext) -> TaskValidationResult:
    return _validate_one(context)
