"""F7 Rule 6: require one reachable setup-to-derivation route."""

from __future__ import annotations

from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f07.api_family import Analysis


def validate_rule(
    context: TaskContext,
    analysis: Analysis,
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    """Reachability is established while collecting the F7 operation evidence."""

    return None
