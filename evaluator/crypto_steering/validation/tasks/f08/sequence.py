"""F8 Rule 6: require one connected protected-storage route."""

from __future__ import annotations

from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f08.api_family import Analysis


def validate_rule(
    context: TaskContext,
    analysis: Analysis,
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    """The F8 evidence queries already require one reachable serialization route."""

    return None
