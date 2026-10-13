"""F4 Rule 6: require one connected signing or verification route."""

from __future__ import annotations

import ast
import re

from crypto_steering.validation.java.task_analysis import _receiver_method_calls
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.tasks.shared.results import java_task_result
from crypto_steering.validation.tasks.shared.context import TaskContext
from crypto_steering.validation.tasks.f04.api_family import Analysis
from crypto_steering.validation.tasks.f04.operations import completed_routes
from crypto_steering.validation.tasks.signatures import signature_completion_failure
from crypto_steering.validation.tasks.shared.results import task_failure


def validate_rule(
    context: TaskContext, analysis: Analysis
) -> TaskValidationResult | None:
    if context.language == "java":
        completed_sign, completed_verify = completed_routes(analysis)
        if not (completed_sign or completed_verify) and all(
            _receiver_method_calls(analysis.source, name)
            for name in ("update",)
        ) and (
            all(
                _receiver_method_calls(analysis.source, name)
                for name in ("initSign", "sign")
            )
            or all(
                _receiver_method_calls(analysis.source, name)
                for name in ("initVerify", "verify")
            )
        ):
            return java_task_result(
                context,
                success=False,
                failure_code="target_operation_sequence_invalid",
                reason="F4 signature calls do not form one connected receiver route",
            )
        return None

    reason = None
    if context.language == "python":
        reason = signature_completion_failure(
            context.language,
            context.source,
            context.task_id,
        ) or _python_public_key_signature_flow_failure(context.source)
    if reason is not None:
        from crypto_steering.validation.tasks.f04.operations import success_result

        return task_failure(
            success_result(context, analysis),
            "target_operation_sequence_invalid",
            reason,
        )
    return None


def _python_public_key_signature_flow_failure(source: str) -> str | None:
    """Reject verification fed only by a statically invalid public-key sign.

    A public RSA key cannot produce a signature.  A later verify call does not
    form an independent usable route when its signature argument is exactly
    the value assigned from that invalid call.  Unknown or caller-supplied
    signatures remain acceptable under the evidence policy.
    """

    if re.search(
        r"\bcryptography\s*\.\s*hazmat\s*\.\s*primitives\s*\.\s*"
        r"asymmetric\s*\.\s*PSS\b",
        source,
    ):
        return "PSS is provided by cryptography's asymmetric padding module"

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    public_names = {
        target.id
        for assignment in ast.walk(tree)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        and isinstance(assignment.value, ast.Call)
        and isinstance(assignment.value.func, ast.Attribute)
        and assignment.value.func.attr == "public_key"
        for target in (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
        if isinstance(target, ast.Name)
    }

    def public_receiver(call: ast.Call) -> bool:
        if not isinstance(call.func, ast.Attribute):
            return False
        receiver = call.func.value
        return bool(
            isinstance(receiver, ast.Name)
            and (
                receiver.id in public_names
                or "public" in receiver.id.lower()
            )
        ) or bool(
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Attribute)
            and receiver.func.attr == "public_key"
        )

    invalid_signatures: set[str] = set()
    usable_sign = False
    for assignment in ast.walk(tree):
        if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
            continue
        value = assignment.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "sign"
        ):
            continue
        targets = (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
        names = {target.id for target in targets if isinstance(target, ast.Name)}
        if public_receiver(value):
            invalid_signatures.update(names)
        else:
            usable_sign = True

    if not invalid_signatures or usable_sign:
        return None

    usable_verify = any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "verify"
        and call.args
        and not (
            isinstance(call.args[0], ast.Name)
            and call.args[0].id in invalid_signatures
        )
        for call in ast.walk(tree)
    )
    if usable_verify:
        return None
    return "the only signature value is produced by signing with a public key"
