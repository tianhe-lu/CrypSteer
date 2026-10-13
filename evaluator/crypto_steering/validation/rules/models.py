"""Typed outcomes for the six program-validity rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


RuleId = Literal[
    "frontend",
    "imports",
    "api_family",
    "required_operations",
    "arguments",
    "operation_sequence",
]
RuleStatus = Literal["passed", "failed", "blocked"]

PROGRAM_RULE_IDS: tuple[RuleId, ...] = (
    "frontend",
    "imports",
    "api_family",
    "required_operations",
    "arguments",
    "operation_sequence",
)
TASK_RULE_IDS: tuple[RuleId, ...] = PROGRAM_RULE_IDS[2:]

PASS_REASONS: dict[RuleId, str] = {
    "frontend": "the configured language frontend accepted the source",
    "imports": "all imported dependencies are approved and resolvable",
    "api_family": "the source uses a supported API family for the task",
    "required_operations": "the required task operations are present",
    "arguments": "required API arguments use structurally valid roles and positions",
    "operation_sequence": "the required operations form a connected task sequence",
}


@dataclass(frozen=True)
class RuleAssessment:
    """One visible decision in the program-validity conjunction."""

    rule_id: RuleId
    status: RuleStatus
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""

    @property
    def success(self) -> bool:
        return self.status == "passed"


def passed(rule_id: RuleId, reason: str) -> RuleAssessment:
    return RuleAssessment(rule_id=rule_id, status="passed", reason=reason)


def failed(
    rule_id: RuleId,
    reason_codes: tuple[str, ...],
    reason: str,
) -> RuleAssessment:
    return RuleAssessment(
        rule_id=rule_id,
        status="failed",
        reason_codes=reason_codes,
        reason=reason,
    )


def blocked(rule_id: RuleId, reason: str) -> RuleAssessment:
    return RuleAssessment(
        rule_id=rule_id,
        status="blocked",
        reason_codes=("prerequisite_rule_failed",),
        reason=reason,
    )
