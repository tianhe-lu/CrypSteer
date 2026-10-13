"""Aggregate task and frontend outcomes into the six-rule contract."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from crypto_steering.validation.models import (
    CompileOutcome,
    ImportOutcome,
    TaskValidationResult,
)
from crypto_steering.validation.rules.frontend import assess as assess_frontend
from crypto_steering.validation.rules.imports import assess as assess_imports
from crypto_steering.validation.rules.models import (
    PASS_REASONS,
    TASK_RULE_IDS,
    RuleAssessment,
    RuleId,
    blocked,
    failed,
    passed,
)


_FAILURE_RULE: dict[str, RuleId] = {
    "target_api_not_found": "api_family",
    "target_wrong_api_family": "api_family",
    "target_api_not_rewriteable": "api_family",
    "target_operation_incomplete": "required_operations",
    "target_integrity_decision_missing": "required_operations",
    "target_argument_missing": "arguments",
    "target_argument_unresolved": "arguments",
    "target_argument_contract_invalid": "arguments",
    "target_operation_sequence_invalid": "operation_sequence",
}


def task_result_with_rule_assessments(
    result: TaskValidationResult,
    *,
    refresh: bool = False,
) -> TaskValidationResult:
    """Attach the ordered Rules 3-6 decisions to a family result."""

    if result.rule_assessments and not refresh:
        _validate_task_rule_shape(result.rule_assessments)
        success = all(item.success for item in result.rule_assessments)
        return replace(result, success=success)

    failed_rule = _failed_rule(result)
    if not result.success and failed_rule is None:
        return replace(
            result,
            rule_assessments=tuple(
                RuleAssessment(
                    rule_id=rule_id,
                    status="blocked",
                    reason_codes=result.reason_codes,
                    reason=result.reason,
                )
                for rule_id in TASK_RULE_IDS
            ),
        )
    assessments: list[RuleAssessment] = []
    prerequisite_failed = False
    for rule_id in TASK_RULE_IDS:
        if prerequisite_failed:
            assessments.append(
                blocked(rule_id, f"{failed_rule} did not pass")
            )
        elif failed_rule == rule_id:
            assessments.append(
                failed(rule_id, result.reason_codes, result.reason)
            )
            prerequisite_failed = True
        else:
            assessments.append(passed(rule_id, PASS_REASONS[rule_id]))
    return replace(
        result,
        success=all(item.success for item in assessments),
        rule_assessments=tuple(assessments),
    )


def program_rule_assessments(
    compile_or_parse: CompileOutcome | None,
    imports: ImportOutcome | None,
    target_tasks: Iterable[TaskValidationResult],
    *,
    task_providers_complete: bool = True,
) -> tuple[RuleAssessment, ...]:
    """Return the single six-rule authority for program validity."""

    tasks = tuple(task_result_with_rule_assessments(task) for task in target_tasks)
    frontend = assess_frontend(compile_or_parse)
    import_rule = assess_imports(imports)
    if not task_providers_complete:
        task_rules = tuple(
            blocked(rule_id, "one or more target-task validators did not complete")
            for rule_id in TASK_RULE_IDS
        )
    else:
        task_rules = (
            assess_api_family(tasks),
            assess_required_operations(tasks),
            assess_arguments(tasks),
            assess_operation_sequence(tasks),
        )
    return (
        frontend,
        import_rule,
        *task_rules,
    )


def assess_api_family(
    tasks: Iterable[TaskValidationResult],
) -> RuleAssessment:
    return _assess_task_rule("api_family", tasks)


def assess_required_operations(
    tasks: Iterable[TaskValidationResult],
) -> RuleAssessment:
    return _assess_task_rule("required_operations", tasks)


def assess_arguments(
    tasks: Iterable[TaskValidationResult],
) -> RuleAssessment:
    return _assess_task_rule("arguments", tasks)


def assess_operation_sequence(
    tasks: Iterable[TaskValidationResult],
) -> RuleAssessment:
    return _assess_task_rule("operation_sequence", tasks)


def _assess_task_rule(
    rule_id: RuleId,
    tasks: Iterable[TaskValidationResult],
) -> RuleAssessment:
    members = tuple(
        assessment
        for task in tasks
        for assessment in task.rule_assessments
        if assessment.rule_id == rule_id
    )
    if not members:
        return blocked(rule_id, "no completed target-task assessment is available")
    failures = tuple(item for item in members if item.status == "failed")
    if failures:
        return failed(
            rule_id,
            tuple(
                dict.fromkeys(
                    code for item in failures for code in item.reason_codes
                )
            ),
            "; ".join(dict.fromkeys(item.reason for item in failures)),
        )
    blockers = tuple(item for item in members if item.status == "blocked")
    if blockers:
        return blocked(
            rule_id,
            "; ".join(dict.fromkeys(item.reason for item in blockers)),
        )
    return passed(rule_id, PASS_REASONS[rule_id])


def _failed_rule(result: TaskValidationResult) -> RuleId | None:
    if result.success:
        return None
    for reason_code in result.reason_codes:
        rule_id = _FAILURE_RULE.get(reason_code)
        if rule_id is not None:
            return rule_id
    return None


def _validate_task_rule_shape(assessments: tuple[RuleAssessment, ...]) -> None:
    observed = tuple(item.rule_id for item in assessments)
    if observed != TASK_RULE_IDS:
        raise ValueError(
            f"task rule assessments must be ordered as {TASK_RULE_IDS!r}, got {observed!r}"
        )
