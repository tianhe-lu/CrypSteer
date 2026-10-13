"""F8 protected-key storage validation, organized by Rules 3-6."""

from __future__ import annotations

from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.rules.aggregate import task_result_with_rule_assessments
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f08 import api_family, arguments, operations, sequence


def validate(context: TaskContext) -> TaskValidationResult:
    return task_result_with_rule_assessments(_validate_family(context))


def _validate_family(context: TaskContext) -> TaskValidationResult:
    analysis = api_family.analyze(context)
    for validate_rule in (
        api_family.validate_rule,
        operations.validate_presence,
        arguments.validate_java_rule,
        operations.validate_rule,
    ):
        failure = validate_rule(context, analysis)
        if failure is not None:
            return failure
    result = operations.success_result(context, analysis)
    failure = arguments.validate_rule(context, analysis, result)
    if failure is not None:
        return failure
    failure = sequence.validate_rule(context, analysis, result)
    return failure or result


def _validate_indexed(context: TaskContext) -> TaskValidationResult:
    return _validate_family(context)


def _validate_java(context: TaskContext) -> TaskValidationResult:
    return _validate_family(context)
