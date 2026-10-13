"""F6 Rule 3: recognize supported native MAC families."""

from __future__ import annotations

from dataclasses import dataclass
import re

from crypto_steering.validation.java.source import strip_java_comments
from crypto_steering.validation.java.task_analysis import (
    _factory_receivers_assigned_from,
    _receiver_method_calls,
)
from crypto_steering.validation.models import TaskEvidence, TaskValidationResult
from crypto_steering.validation.tasks.shared.results import (
    java_task_result,
    require_analyzer,
)
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.contracts import _python_native_hmac_calls
from crypto_steering.validation.tasks.evidence import (
    _call_evidence,
    _csharp_hmac_activator_evidence,
    _csharp_hmac_factory_evidence,
    _csharp_legacy_algorithm_factory_evidence,
    _csharp_typed_hmac_evidence,
    _first_evidence,
    _hmac_constructor_input_evidence,
    _mac_one_shot_evidence,
    _qualified_call_evidence,
)


@dataclass(frozen=True)
class Analysis:
    source: str
    required: tuple[tuple[TaskEvidence | None, str], ...] = ()
    setup: TaskEvidence | None = None
    terminal: TaskEvidence | None = None
    csharp_shadow: bool = False
    mac_receivers: frozenset[str] = frozenset()
    candidate_initializations: tuple[tuple[str, object], ...] = ()


def analyze(context: TaskContext) -> Analysis:
    if context.language == "java":
        source = strip_java_comments(context.source)
        receivers = frozenset(_factory_receivers_assigned_from(source, "Mac"))
        return Analysis(
            source=source,
            mac_receivers=receivers,
            candidate_initializations=tuple(
                (receiver, call)
                for receiver, call in _receiver_method_calls(source, "init")
                if receiver in receivers
            ),
        )

    analyzer = require_analyzer(context)
    one_shot = _mac_one_shot_evidence(analyzer)
    if analyzer.language == "python":
        calls = _python_native_hmac_calls(analyzer)
        setup = one_shot or (_call_evidence(analyzer, calls[0][0]) if calls else None)
    else:
        setup = one_shot or _first_evidence(
            analyzer,
            (
                "createHmac", "createIncrementalHmac", "createNodeHmac", "HMAC",
                "HMACMD5", "HMACSHA1", "HMACSHA256", "HMACSHA384", "HMACSHA512",
                "HMACSHA3_256", "HMACSHA3_384", "HMACSHA3_512", "IncrementalHash",
            ),
        )
    if setup is None:
        setup = _qualified_call_evidence(analyzer, "new", "hmac.new")
    if setup is None and analyzer.language == "python":
        setup = python_hmac_alias_setup_evidence(analyzer)
    if setup is None and analyzer.language == "csharp":
        setup = (
            _csharp_hmac_factory_evidence(analyzer)
            or _csharp_typed_hmac_evidence(analyzer)
            or _csharp_hmac_activator_evidence(analyzer)
            or _csharp_legacy_algorithm_factory_evidence(analyzer)
        )
    update = _first_evidence(analyzer, ("update", "AppendData"))
    if update is None:
        update = _hmac_constructor_input_evidence(analyzer)
    if update is None and analyzer.language == "python":
        update = next(
            (
                _call_evidence(analyzer, call)
                for call, canonical in calls
                if canonical == "hmac.new"
                and (len(call.arguments) >= 2 or "msg" in call.named_arguments)
            ),
            None,
        )
    terminal = one_shot or _first_evidence(
        analyzer,
        (
            "digest", "hexdigest", "doFinal", "GetHashAndReset", "ComputeHash",
            "finalize", "verify",
        ),
    )
    if one_shot is not None or (
        terminal is not None and terminal.operation == "ComputeHash"
    ):
        update = terminal
    return Analysis(
        source=context.source,
        required=(
            (setup, "MAC construction"),
            (update, "MAC message input"),
            (terminal, "MAC terminal operation"),
        ),
        setup=setup,
        terminal=terminal,
        csharp_shadow=(
            context.language == "csharp"
            and csharp_shadows_constructed_hmac_type(context.source)
        ),
    )


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if analysis.csharp_shadow:
        return TaskValidationResult(
            language=context.language,
            task_id=context.task_id,
            success=False,
            reason_codes=("target_wrong_api_family",),
            reason=(
                "a source-defined HMAC type shadows the System.Security.Cryptography "
                "implementation used by the task"
            ),
        )
    if context.language == "java":
        if analysis.mac_receivers:
            return None
        return java_task_result(
            context,
            success=False,
            failure_code="target_api_not_found",
            reason="F6 requires Mac.getInstance assigned to a Mac receiver",
        )
    if analysis.setup is None and analysis.terminal is not None:
        return TaskValidationResult(
            language=context.language,
            task_id=context.task_id,
            success=False,
            reason_codes=("target_wrong_api_family",),
            reason="F6 requires a supported native MAC construction",
            matched_operations=(analysis.terminal,),
        )
    if any(item is not None for item, _ in analysis.required):
        return None
    return TaskValidationResult(
        language=context.language,
        task_id=context.task_id,
        success=False,
        reason_codes=("target_api_not_found",),
        reason="missing " + ", ".join(name for _, name in analysis.required),
    )


def csharp_shadows_constructed_hmac_type(source: str) -> bool:
    declared = {
        match.group("name")
        for match in re.finditer(
            r"\bclass\s+(?P<name>HMAC(?:MD5|SHA1|SHA256|SHA384|SHA512)?)\b", source
        )
    }
    constructed = {
        match.group("name")
        for match in re.finditer(
            r"\bnew\s+(?P<name>HMAC(?:MD5|SHA1|SHA256|SHA384|SHA512)?)\s*\(", source
        )
    }
    return bool(declared & constructed)


def python_hmac_alias_setup_evidence(analyzer: object) -> TaskEvidence | None:
    source = str(getattr(analyzer, "source"))
    aliases = {
        re.sub(r"\s+", "", match.group("target"))
        for match in re.finditer(
            r"(?P<target>(?:self\.)?[A-Za-z_]\w*)\s*=\s*hmac\s*\.\s*new\b", source
        )
    }
    for alias in aliases:
        call = re.search(rf"(?<![\w.]){re.escape(alias)}\s*\(", source)
        if call is not None:
            return TaskEvidence(
                operation="hmac.new alias",
                expression=call.group(0),
                span=getattr(analyzer, "source_span")(call.start(), call.end()),
            )
    return None
