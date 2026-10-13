"""Rule 1: the configured language frontend accepts the source."""

from __future__ import annotations

from crypto_steering.validation.models import CompileOutcome
from crypto_steering.validation.rules.models import (
    PASS_REASONS,
    RuleAssessment,
    blocked,
    failed,
    passed,
)


def assess(outcome: CompileOutcome | None) -> RuleAssessment:
    if outcome is None:
        return blocked("frontend", "the language frontend did not complete")
    if outcome.success:
        return passed("frontend", PASS_REASONS["frontend"])
    return failed(
        "frontend",
        (outcome.reason_code,),
        outcome.stderr or "the configured language frontend rejected the source",
    )
