"""F9 Rule 4: require an actual random-value generation call."""

from __future__ import annotations

import re

from crypto_steering.validation.java.static_analysis import (
    actual_symbol_usage,
    variable_type_in_scope,
)
from crypto_steering.validation.java.task_analysis import (
    _is_secure_random_expression,
    _receiver_method_calls,
    _secure_random_receivers,
)
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f09.api_family import Analysis


_JAVA_RANDOM_TYPES = frozenset(
    {
        "SecureRandom",
        "Random",
        "SplittableRandom",
        "ThreadLocalRandom",
        "RandomGenerator",
    }
)


def random_generation_result(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult:
    if context.language == "java":
        return java_random_generation_result(context, analysis.source)
    if analysis.generation is not None:
        return TaskValidationResult(
            language=context.language,
            task_id=context.task_id,
            success=True,
            reason_codes=(),
            reason="completed CSPRNG or general-purpose PRNG generation",
            matched_operations=(analysis.generation,),
        )
    return TaskValidationResult(
        language=context.language,
        task_id=context.task_id,
        success=False,
        reason_codes=("target_operation_incomplete",),
        reason="F9 requires a call that produces random bytes or a random value",
    )


def java_random_generation_result(
    context: TaskContext, source: str
) -> TaskValidationResult:
    if re.search(r"\bMath\s*\.\s*random\s*\(", source):
        return java_task_result(
            context,
            success=True,
            reason="matched Math.random value generation",
            matched_symbols=("Math.random",),
        )
    direct_factory = re.search(
        r"\b(?:"
        r"(?:java\s*\.\s*util\s*\.\s*concurrent\s*\.\s*)?"
        r"ThreadLocalRandom\s*\.\s*current\s*\(\s*\)"
        r"|(?:java\s*\.\s*util\s*\.\s*random\s*\.\s*)?"
        r"RandomGenerator\s*\.\s*(?:getDefault|of)\s*\([^;]*?\)"
        r")\s*\.\s*"
        r"(?P<method>next(?:Bytes|Int|Long|Double|Boolean|Float|Gaussian)|"
        r"ints|longs|doubles)\s*\(",
        source,
        flags=re.DOTALL,
    )
    if direct_factory is not None:
        method = direct_factory.group("method")
        return java_task_result(
            context,
            success=True,
            reason=f"matched direct Java PRNG {method} value generation",
            matched_symbols=("Random", method),
        )
    receivers = _secure_random_receivers(source)
    receivers.update(
        match.group("name")
        for match in re.finditer(
            r"\b(?:Random|SplittableRandom|ThreadLocalRandom|RandomGenerator)\s+"
            r"(?P<name>[A-Za-z_$][\w$]*)\s*=",
            source,
        )
    )
    receivers.update(
        match.group("name")
        for match in re.finditer(
            r"\b(?:var|Random|SplittableRandom|ThreadLocalRandom|RandomGenerator)\s+"
            r"(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
            r"(?:new\s+(?:Random|SplittableRandom)\s*\(|"
            r"ThreadLocalRandom\s*\.\s*current\s*\(|"
            r"RandomGenerator\s*\.\s*(?:getDefault|of)\s*\()",
            source,
        )
    )
    if not receivers and not (
        any(
            actual_symbol_usage(source, symbol)
            for symbol in (
                "SecureRandom",
                "Random",
                "SplittableRandom",
                "ThreadLocalRandom",
                "RandomGenerator",
            )
        )
        or re.search(r"\bMath\s*\.\s*random\s*\(", source)
    ):
        return java_task_result(
            context,
            success=False,
            failure_code="target_api_not_found",
            reason="F9 requires a random-number source",
        )
    methods = (
        "nextBytes", "nextInt", "nextLong", "nextDouble", "nextFloat",
        "nextBoolean", "nextGaussian", "ints", "longs", "doubles",
        "generateSeed",
    )
    for method in methods:
        for receiver, call in _receiver_method_calls(source, method):
            if _java_random_receiver(
                source,
                receiver,
                call.start,
                known_receivers=receivers,
            ):
                return java_task_result(
                    context,
                    success=True,
                    reason=f"matched F9 random-generator {method} usage",
                    matched_symbols=("Random", method),
                )
        if re.search(
            rf"\bnew\s+(?:(?:java\.security\.)?SecureRandom|"
            rf"(?:java\.util\.)?(?:Random|SplittableRandom))\s*"
            rf"\([^;]*\)\s*\.\s*{method}\s*\(",
            source,
            flags=re.DOTALL,
        ):
            return java_task_result(
                context,
                success=True,
                reason=f"matched inline F9 random-generator {method} usage",
                matched_symbols=("Random", method),
            )
        if re.search(
            rf"\(\s*(?:SecureRandom|Random|SplittableRandom|RandomGenerator)\s*\)"
            rf"\s*[A-Za-z_$][\w$]*\s*\.\s*{method}\s*\(",
            source,
        ) or re.search(
            rf"\(\s*\(\s*(?:SecureRandom|Random|SplittableRandom|RandomGenerator)"
            rf"\s*\)\s*[A-Za-z_$][\w$]*\s*\)\s*\.\s*{method}\s*\(",
            source,
        ):
            return java_task_result(
                context,
                success=True,
                reason=f"matched cast random-generator {method} usage",
                matched_symbols=("Random", method),
            )
        if re.search(
            rf"\b(?:java\.security\.)?SecureRandom\s*\.\s*getInstance(?:Strong)?\s*\([^;]*\)\s*\.\s*{method}\s*\(",
            source,
            flags=re.DOTALL,
        ):
            return java_task_result(
                context,
                success=True,
                reason=f"matched inline F9 SecureRandom.{method} generation usage",
                matched_symbols=("SecureRandom", method),
            )
        for match in re.finditer(
            rf"\b(?P<receiver>[A-Za-z_]\w*\s*\([^;{{}}]*\))\s*\.\s*{method}\s*\(",
            source,
            flags=re.DOTALL,
        ):
            if _is_secure_random_expression(source, match.group("receiver"), match.start()):
                return java_task_result(
                    context,
                    success=True,
                    reason=(
                        f"matched correct F9 helper-returned SecureRandom.{method} "
                        "generation usage"
                    ),
                    matched_symbols=("SecureRandom", method),
                )
    return java_task_result(
        context,
        success=False,
        failure_code="target_operation_incomplete",
        reason="F9 requires a call that produces random bytes or a random value",
    )


def _java_random_receiver(
    source: str,
    receiver: str,
    index: int,
    *,
    known_receivers: set[str],
) -> bool:
    """Recognize an assigned, field, or parameter random-generator receiver."""

    compact = re.sub(r"\s+", "", receiver)
    if compact in known_receivers or compact == "newSecureRandom()":
        return True
    name = compact.rsplit(".", 1)[-1]
    if not re.fullmatch(r"[A-Za-z_$][\w$]*", name):
        return False
    type_name = variable_type_in_scope(source, name, index) or ""
    simple_type = re.sub(r"<.*>$", "", type_name).rsplit(".", 1)[-1]
    if simple_type in _JAVA_RANDOM_TYPES:
        return True

    # ``variable_type_in_scope`` resolves initialized locals and parameters.
    # Also admit uninitialized fields/local declarations, which are valid
    # sources when their value is supplied by a constructor or caller.
    declared_type = _java_declared_type(source, name, index)
    if declared_type in _JAVA_RANDOM_TYPES:
        return True

    # A project-local subtype is an equivalent API choice under the F9 family
    # rule.  The compiler has already established whether the declaration is
    # legal; here we only connect its receiver to the random API family.
    return bool(
        declared_type
        and re.search(
            rf"\b(?:class|interface)\s+{re.escape(declared_type)}\b[^{{;]*"
            rf"\b(?:extends|implements)\b[^{{;]*"
            rf"\b(?:{'|'.join(sorted(_JAVA_RANDOM_TYPES))})\b",
            source,
        )
    )


def _java_declared_type(source: str, name: str, index: int) -> str:
    """Return a standard or source-local type declared for ``name``."""

    matches = tuple(
        re.finditer(
            rf"\b(?P<type>(?:[A-Za-z_$][\w$]*\.)*[A-Za-z_$][\w$]*)"
            rf"(?:\s*<[^;={{}}()]+>)?\s+{re.escape(name)}\b",
            source[:index],
        )
    )
    if not matches:
        return ""
    return matches[-1].group("type").rsplit(".", 1)[-1]
