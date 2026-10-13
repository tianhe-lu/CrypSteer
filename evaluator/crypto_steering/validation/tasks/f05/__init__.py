"""F5 digest validation, organized by Rules 3-6."""

from __future__ import annotations

from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.rules.aggregate import task_result_with_rule_assessments
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f05 import api_family, arguments, operations, sequence


def validate(context: TaskContext) -> TaskValidationResult:
    return task_result_with_rule_assessments(_validate_family(context))


def _validate_family(context: TaskContext) -> TaskValidationResult:
    analysis = api_family.analyze(context)
    failure = api_family.validate_rule(context, analysis)
    if failure is not None:
        return failure
    failure = operations.validate_rule(context, analysis)
    if failure is not None:
        return failure
    result = operations.success_result(context, analysis)
    failure = sequence.validate_java_rule(context, analysis, result)
    if failure is not None:
        return failure
    failure = sequence.validate_rule(context, analysis, result)
    if failure is not None:
        return failure
    if context.language == "java":
        return result
    failure = arguments.validate_python_encoding(context, analysis, result)
    if failure is not None:
        return failure
    failure = sequence.validate_python_rule(context, analysis, result)
    if failure is not None:
        return failure
    failure = arguments.validate_rule(context, analysis, result)
    if failure is not None:
        return failure
    failure = api_family.validate_unkeyed_digest_rule(context, analysis, result)
    return failure or result


def _validate_indexed(context: TaskContext) -> TaskValidationResult:
    return _validate_family(context)


def _validate_java(context: TaskContext) -> TaskValidationResult:
    return _validate_family(context)
