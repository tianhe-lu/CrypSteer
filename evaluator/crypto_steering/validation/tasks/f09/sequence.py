"""F9 Rule 6: a completed random-generation call is a complete route."""

from __future__ import annotations

from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.f09.api_family import Analysis
from crypto_steering.validation.tasks.shared.context import TaskContext


def validate_rule(
    context: TaskContext,
    analysis: Analysis,
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    """Accept Rule 4's completed call without requiring a downstream sink."""

    return None
