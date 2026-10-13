"""F7 password-KDF validation, organized by Rules 3-6."""

from __future__ import annotations

from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.rules.aggregate import task_result_with_rule_assessments
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f07 import api_family, arguments, operations, sequence


def validate(context: TaskContext) -> TaskValidationResult:
    return task_result_with_rule_assessments(_validate_family(context))


def _validate_family(context: TaskContext) -> TaskValidationResult:
    return validate_pbkdf2_operation(context)


def validate_pbkdf2_operation(
    context: TaskContext,
    *,
    allow_general_password_kdf: bool = False,
) -> TaskValidationResult:
    analysis = api_family.analyze(
        context, allow_general_password_kdf=allow_general_password_kdf
    )
    for validate_rule in (
        api_family.validate_presence,
        api_family.validate_rule,
        operations.validate_rule,
        arguments.validate_java_constructor,
        arguments.validate_java_terminal,
    ):
        failure = validate_rule(context, analysis)
        if failure is not None:
            return failure
    result = operations.success_result(context, analysis)
    failure = arguments.validate_rule(context, analysis, result)
    if failure is not None:
        return task_result_with_rule_assessments(failure)
    failure = sequence.validate_rule(context, analysis, result)
    if failure is not None:
        return failure
    return task_result_with_rule_assessments(result)


def _validate_indexed(context: TaskContext) -> TaskValidationResult:
    return validate_pbkdf2_operation(context)


def _validate_java(context: TaskContext) -> TaskValidationResult:
    return validate_pbkdf2_operation(context, allow_general_password_kdf=False)
