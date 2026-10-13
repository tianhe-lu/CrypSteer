"""F9 Rule 3: recognize CSPRNG and general-purpose PRNG APIs."""

from __future__ import annotations

from dataclasses import dataclass
import re

from crypto_steering.validation.java.source import strip_java_comments
from crypto_steering.validation.models import TaskEvidence, TaskValidationResult
from crypto_steering.validation.tasks.shared.results import (
    java_task_result,
    require_analyzer,
)
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.evidence import _random_source_evidences


@dataclass(frozen=True)
class Analysis:
    source: str
    generator_present: bool = False
    generation: TaskEvidence | None = None
    generations: tuple[TaskEvidence, ...] = ()


def analyze(context: TaskContext) -> Analysis:
    if context.language == "java":
        source = strip_java_comments(context.source)
        return Analysis(
            source=source,
            generator_present=bool(
                re.search(
                    r"\b(?:SecureRandom|Random|SplittableRandom|"
                    r"ThreadLocalRandom|RandomGenerator)\b"
                    r"|\bMath\s*\.\s*random\s*\(",
                    source,
                )
            ),
        )
    analyzer = require_analyzer(context)
    generations = _random_source_evidences(analyzer)
    if context.language == "csharp" and not generations:
        optional_generation = _csharp_optional_random_generation(context.source)
        if optional_generation is not None:
            generations = (optional_generation,)
    generation = generations[0] if generations else None
    constructor_pattern = {
        "typescript": r"\b(?:webcrypto|Math\s*\.\s*random)\b",
        "csharp": r"\b(?:new\s+Random|RandomNumberGenerator\s*\.\s*Create)\s*\(",
        "python": r"\b(?:random|secrets)\s*\.\s*(?:Random|SystemRandom)\s*\(",
    }[context.language]
    return Analysis(
        source=context.source,
        generator_present=generation is not None
        or re.search(constructor_pattern, context.source) is not None,
        generation=generation,
        generations=generations,
    )


def _csharp_optional_random_generation(source: str) -> TaskEvidence | None:
    """Recognize generation through a nullable ``System.Random`` receiver.

    The C# source index intentionally normalizes ordinary member calls, but a
    null-conditional call such as ``random?.Next(...)`` is represented
    differently.  It is still a structurally valid general-PRNG operation
    under the F9 family rule when the receiver has the native ``Random`` type.
    """

    if re.search(r"\bclass\s+Random\b", source):
        return None
    declarations = {
        match.group("name")
        for match in re.finditer(
            r"\b(?:System\s*\.\s*)?Random\s*\??\s+"
            r"(?P<name>[A-Za-z_]\w*)\b",
            source,
        )
    }
    for name in declarations:
        call = re.search(
            rf"\b{re.escape(name)}\s*\?\s*\.\s*"
            r"(?P<method>NextBytes|Next|NextInt64|NextDouble|NextSingle)\s*\(",
            source,
        )
        if call is not None:
            return TaskEvidence(
                operation=call.group("method"),
                expression=call.group(0),
            )
    return None


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if analysis.generator_present:
        return None
    if context.language == "java":
        return java_task_result(
            context,
            success=False,
            failure_code="target_api_not_found",
            reason="F9 requires a recognized CSPRNG or general-purpose PRNG API",
        )
    return TaskValidationResult(
        language=context.language,
        task_id=context.task_id,
        success=False,
        reason_codes=("target_api_not_found",),
        reason="F9 requires a recognized CSPRNG or general-purpose PRNG API",
    )
