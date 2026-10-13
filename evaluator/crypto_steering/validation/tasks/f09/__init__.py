"""F9 random-value generation validation, organized by Rules 3-6."""

from __future__ import annotations

from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.rules.aggregate import task_result_with_rule_assessments
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f09 import api_family, arguments, operations, sequence


def validate(context: TaskContext) -> TaskValidationResult:
    return task_result_with_rule_assessments(_validate_family(context))


def _validate_family(context: TaskContext) -> TaskValidationResult:
    analysis = api_family.analyze(context)
    failure = api_family.validate_rule(context, analysis)
    if failure is not None:
        return failure
    result = operations.random_generation_result(context, analysis)
    if not result.success:
        return result
    failure = arguments.validate_rule(context, analysis, result)
    if failure is not None:
        return failure
    failure = sequence.validate_rule(context, analysis, result)
    return failure or result


def _validate_indexed(context: TaskContext) -> TaskValidationResult:
    return _validate_family(context)


def _validate_java(context: TaskContext) -> TaskValidationResult:
    return _validate_family(context)
