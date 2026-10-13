"""Rule 2: imported libraries are approved and resolvable."""

from __future__ import annotations

from crypto_steering.validation.models import ImportOutcome
from crypto_steering.validation.rules.models import (
    PASS_REASONS,
    RuleAssessment,
    blocked,
    failed,
    passed,
)


def assess(outcome: ImportOutcome | None) -> RuleAssessment:
    if outcome is None:
        return blocked("imports", "import validation did not complete")
    if outcome.success:
        return passed("imports", PASS_REASONS["imports"])
    return failed(
        "imports",
        (outcome.reason_code,),
        "an imported dependency is unsupported or unresolved",
    )
