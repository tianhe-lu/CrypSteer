"""Definite cross-call failures in otherwise recognizable task sequences.

The operation-evidence layer establishes that the requested APIs and terminal
calls are present.  These checks cover the small amount of bounded dataflow
needed by the task definitions: input must reach the operation and paired
encryption/decryption must share compatible material.  They deliberately do
not evaluate security policy or runtime-only behavior.
"""

from __future__ import annotations

import ast
import re
from typing import Literal

from crypto_steering.validation.java.static_analysis import extract_assignments
from crypto_steering.validation.java.task_analysis import (
    _cipher_receivers_with_transformation,
)
from crypto_steering.validation.models import Language
from crypto_steering.validation.source.index import analyzer_for
from crypto_steering.validation.tasks.shared.stateful import repeated_terminal_receiver
from crypto_steering.validation.tasks.symmetric_flows import symmetric_flow_failure


def symmetric_completion_failure(
    language: Language,
    source: str,
    task_id: str,
) -> str | None:
    """Return a definite disconnected or incomplete symmetric-cipher flow."""

    existing = None
    if language == "python":
        existing = _python_symmetric_failure(source, task_id)
    elif language == "typescript":
        existing = _typescript_symmetric_failure(source, task_id)
    elif language == "csharp":
        existing = _csharp_symmetric_failure(source, task_id)
    elif language == "java":
        existing = _java_symmetric_failure(source, task_id)
    return existing or symmetric_flow_failure(language, source, task_id)


def symmetric_direction_completion_failure(
    language: Language,
    source: str,
    *,
    family: Literal["F1", "F2"],
    direction: Literal["encrypt", "decrypt"],
) -> str | None:
    """Check one family direction without making a T suffix policy-bearing.

    The mature connection analyzers still use their historical T2/T3 tokens
    internally to select an encryption or decryption path. This adapter keeps
    that representation out of the family validators: the family is the
    policy authority and the observed operation supplies the direction.
    """

    suffix = "T2" if direction == "encrypt" else "T3"
    return symmetric_completion_failure(language, source, f"{family}_{suffix}")


def _python_symmetric_failure(source: str, task_id: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    unavailable_failure = _python_unavailable_symmetric_api(tree)
    if unavailable_failure is not None:
        return unavailable_failure

    gcm_argument_failure = _python_gcm_decryption_without_tag_failure(tree)
    if gcm_argument_failure is not None:
        return gcm_argument_failure

    lifecycle_failure = _python_cipher_lifecycle_failure(tree, task_id)
    if lifecycle_failure is not None:
        return lifecycle_failure

    direction_failure = _python_required_cipher_direction_failure(tree, task_id)
    if direction_failure is not None:
        return direction_failure

    helper_call_failure = _python_directional_helper_call_failure(tree, task_id)
    if helper_call_failure is not None:
        return helper_call_failure

    key_failure = _python_aes_key_failure(tree)
    if key_failure is not None:
        return key_failure

    padding_failure = _python_discarded_padding_failure(tree)
    if padding_failure is not None:
        return padding_failure

    if task_id.endswith("T2"):
        context_names = _python_assigned_call_names(tree, {"encryptor"})
        for name in context_names:
            updates = _python_receiver_calls(tree, name, "update")
            finals = _python_receiver_calls(tree, name, "finalize")
            if finals and not updates:
                return "encryption finalizes without consuming the task payload"

    if task_id.endswith("T1"):
        if _python_symmetric_directions_are_mutually_exclusive(tree):
            return "encryption and decryption occur only on mutually exclusive branches"

        connection_failure = _python_round_trip_connection_failure(tree, source)
        if connection_failure is not None:
            return connection_failure

        gcm_failure = _python_gcm_transport_failure(tree, source)
        if gcm_failure is not None:
            return gcm_failure

        transport_failure = _python_cbc_transport_failure(tree)
        if transport_failure is not None:
            return transport_failure

        callsite_failure = _python_round_trip_callsite_failure(tree)
        if callsite_failure is not None:
            return callsite_failure

        if _python_decrypt_helper_uses_unrelated_literal(tree):
            return "decryption processes an unrelated literal instead of the ciphertext"

        representation_failure = _python_ciphertext_representation_failure(tree)
        if representation_failure is not None:
            return representation_failure

        if _python_fresh_keyed_cipher_instances_do_not_match(tree):
            return "encryption and decryption use independently keyed cipher objects"

        branch_failure = _python_separate_mode_material_failure(tree)
        if branch_failure is not None:
            return branch_failure

        functions = {
            node.name.lower(): node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        encrypt_functions = tuple(
            node
            for name, node in functions.items()
            if _directional_function_name(name, "encrypt")
        )
        decrypt_functions = tuple(
            node
            for name, node in functions.items()
            if _directional_function_name(name, "decrypt")
        )
        if decrypt_functions:
            fresh_helpers = _python_fresh_material_helpers(tree)
            if any(
                _python_encrypts_with_untransported_fresh_key(
                    tree,
                    node,
                    decrypt_functions,
                    fresh_helpers,
                )
                for node in encrypt_functions
            ):
                return "the encryption path generates a key that it does not transport"
            if _python_decodes_raw_returned_key_as_base64(
                tree,
                encrypt_functions,
                decrypt_functions,
                fresh_helpers,
            ):
                return "decryption decodes a raw encryption key as Base64"
            decrypt_fresh = any(
                _python_function_generates_cipher_key(node, fresh_helpers)
                for node in decrypt_functions
            )
            if decrypt_fresh:
                return "encryption and decryption independently generate different keys"
            if any(
                _python_function_generates_cipher_iv(node, fresh_helpers)
                for node in decrypt_functions
            ):
                return "decryption generates a fresh IV instead of reusing the encryption IV"
            if encrypt_functions and _python_directional_cipher_key_mismatch(
                tree,
                encrypt_functions,
                decrypt_functions,
            ):
                return "decryption uses a literal key unrelated to the encryption key"
            if encrypt_functions and any(
                _python_encrypt_result_is_discarded(tree, node)
                for node in encrypt_functions
            ):
                return "the encryption result is not returned to the decryption path"

        for function in decrypt_functions:
            decrypt_calls = (
                _python_attribute_calls(function, "decrypt")
                + _python_attribute_calls(function, "decryptor")
            )
            unpad_calls = _python_attribute_calls(function, "update", receiver_hint="unpad")
            if decrypt_calls and unpad_calls and min(
                getattr(call, "lineno", 0) for call in unpad_calls
            ) < min(getattr(call, "lineno", 0) for call in decrypt_calls):
                return "ciphertext is unpadded before authenticated decryption"
        unpadding = _python_attribute_calls(tree, "update", receiver_hint="unpad")
        decryptor_setup = _python_attribute_calls(tree, "decryptor")
        if unpadding and decryptor_setup and min(
            getattr(call, "lineno", 0) for call in unpadding
        ) < min(getattr(call, "lineno", 0) for call in decryptor_setup):
            return "ciphertext is unpadded before decryption"
    return None


def _python_unavailable_symmetric_api(tree: ast.Module) -> str | None:
    """Return definite unavailable or unusable PyCA symmetric operations."""

    if _python_cbc_receives_unpack_tuple(tree):
        return "modes.CBC receives a tuple of integers instead of bytes-like IV material"
    if _python_cipher_receives_mode_class(tree):
        return "Cipher requires a constructed mode object, not the mode class"

    if any(
        isinstance(node, ast.Attribute)
        and _python_name(node) == "algorithms.PKCS7"
        for node in ast.walk(tree)
    ):
        return "PKCS7 padding is provided by cryptography.hazmat.primitives.padding"

    bound_names = _python_bound_names(tree)
    symmetric_padding_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "cryptography.hazmat.primitives"
        for alias in node.names
        if alias.name == "padding"
    }
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    cipher_names = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and _python_name(node.value.func).rsplit(".", 1)[-1] == "Cipher"
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        for name in _python_assignment_target_names(target)
    }
    padder_names = {
        name
        for assignment in ast.walk(tree)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        and isinstance(assignment.value, ast.Call)
        and _python_name(assignment.value.func).rsplit(".", 1)[-1] == "padder"
        for target in (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
        for name in _python_assignment_target_names(target)
    }
    if any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in padder_names
        and node.attr in {"padded", "padded_data"}
        for node in ast.walk(tree)
    ) and not _python_has_complete_one_shot_aead_route(tree):
        return "a PKCS7 padder exposes output through update/finalize return values"
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == (
            "cryptography.hazmat.primitives.ciphers.aead"
        ):
            unavailable = {
                alias.name
                for alias in node.names
                if alias.name
                in {
                    "AESCipher",
                    "AES256_CBC",
                    "AES256_ECB",
                    "AES256_IBM_GCM",
                    "AES256_SIV",
                }
            }
            if unavailable:
                return (
                    "cryptography has no "
                    + ", ".join(sorted(unavailable))
                    + " cipher class"
                )
        if isinstance(node, ast.ImportFrom) and node.module == "cryptography.exceptions":
            if any(alias.name == "InvalidPadding" for alias in node.names):
                return "cryptography.exceptions has no InvalidPadding exception"
        if isinstance(node, ast.Attribute):
            name = _python_name(node)
            if name == "getpass.getrandom":
                return "the getpass module has no getrandom operation"
            if name.startswith("modes.") and "modes" not in bound_names:
                return "the cryptography cipher-mode namespace is not imported or defined"
            if name in {
                "padding.ANSI_X923",
                "padding.AESGCMTagSize",
                "padding.AES_GCM_TAG_SIZE_256",
                "padding.AnaloguePadder",
                "padding.AnalogueUnpadder",
            }:
                return f"cryptography has no {name} padding class"
            if (
                node.attr == "tag"
                and isinstance(node.value, ast.Name)
                and node.value.id in {
                    name
                    for assignment in ast.walk(tree)
                    if isinstance(assignment, (ast.Assign, ast.AnnAssign))
                    and isinstance(assignment.value, ast.Call)
                    and _python_name(assignment.value.func).rsplit(".", 1)[-1]
                    in {"AESGCM", "AESCCM", "ChaCha20Poly1305"}
                    for target in (
                        assignment.targets
                        if isinstance(assignment, ast.Assign)
                        else (assignment.target,)
                    )
                    for name in _python_assignment_target_names(target)
                }
            ):
                return "PyCA one-shot AEAD objects do not expose a separate tag attribute"
            if node.attr == "finalist":
                return "cryptography contexts expose finalize(), not finalist()"
            root, _, member_path = name.partition(".")
            if root in cipher_names and (
                member_path == "iv"
                or member_path.startswith("_cipher")
            ) and _python_has_ancestor(node, parents, ast.Return):
                return f"PyCA Cipher objects do not expose {member_path!r} as task output"
            if name.startswith("Padding.") and "Padding" not in bound_names:
                return "the required padding namespace is not imported or defined"
        if not isinstance(node, ast.Call):
            continue
        name = _python_name(node.func)
        simple = name.rsplit(".", 1)[-1]
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in symmetric_padding_aliases
            and simple[:1].isupper()
            and simple not in {"PKCS7", "ANSIX923"}
        ):
            return (
                "cryptography symmetric padding has no "
                f"{simple} construction"
            )
        if name == "os.pbkdf2_hmac":
            return "pbkdf2_hmac is provided by hashlib, not os"
        if (
            simple in {"encryptor", "decryptor"}
            and isinstance(node.func, ast.Attribute)
            and _python_name(node.func.value) in cipher_names
            and (node.args or node.keywords)
        ):
            return f"PyCA {simple} does not accept IV or other call arguments"
        if (
            simple == "finalize_with_tag"
            and len(node.args) != 1
            and not _python_context_completed_before(tree, node)
        ):
            return "finalize_with_tag requires exactly one authentication tag"
        if name == "padding.ANSI_X923":
            return "cryptography uses padding.ANSIX923, not padding.ANSI_X923"
        if name == "padding.ANSIX923" and any(
            keyword.arg not in {"block_size"} for keyword in node.keywords
        ):
            return "ANSIX923 accepts a block size, not a cipher context"
        if (
            simple == "pad"
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Call)
            and _python_name(node.func.value.func).rsplit(".", 1)[-1] == "PKCS7"
        ):
            return "PKCS7 exposes padder(), not pad()"
        if name.endswith("modes.GCM") and any(
            keyword.arg == "backend" for keyword in node.keywords
        ):
            return "modes.GCM does not accept a backend argument"
        if simple == "unpad" and "unpad" not in bound_names:
            return "unpad is not a PyCA padding-context operation"

    algorithm_objects = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and _python_name(node.value.func).startswith("hashes.")
        and _python_name(node.value.func).rsplit(".", 1)[-1] != "Hash"
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        for name in _python_assignment_target_names(target)
    }
    if algorithm_objects and any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in algorithm_objects
        and node.func.attr in {"update", "digest", "hexdigest"}
        for node in ast.walk(tree)
    ):
        return (
            "cryptography hash algorithm descriptors do not provide mutable "
            "update or digest operations"
        )
    return None


def _python_gcm_decryption_without_tag_failure(tree: ast.Module) -> str | None:
    """Reject a GCM decryptor whose setup source-provenly omits its tag."""

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    gcm_calls = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _python_name(node.func).endswith("modes.GCM")
    )
    for gcm in gcm_calls:
        has_positional_tag = len(gcm.args) >= 2 and not (
            isinstance(gcm.args[1], ast.Constant) and gcm.args[1].value is None
        )
        has_keyword_tag = any(
            keyword.arg == "tag"
            and not (
                isinstance(keyword.value, ast.Constant)
                and keyword.value.value is None
            )
            for keyword in gcm.keywords
        )
        has_tag = has_positional_tag or has_keyword_tag
        if has_tag:
            continue
        scope = _python_containing_scope(gcm, parents, tree)
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "finalize_with_tag"
            for node in ast.walk(scope)
        ):
            continue

        cipher: ast.Call | None = None
        current: ast.AST | None = parents.get(gcm)
        while current is not None and not isinstance(current, ast.stmt):
            if (
                isinstance(current, ast.Call)
                and _python_name(current.func).rsplit(".", 1)[-1] == "Cipher"
            ):
                cipher = current
                break
            current = parents.get(current)

        decrypts = bool(
            cipher is not None
            and _python_cipher_call_creates_decryptor(
                cipher,
                scope,
                parents,
            )
        )
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decrypts = decrypts or _directional_function_name(
                scope.name, "decrypt"
            )
        if decrypts:
            return "GCM decryption requires an authentication tag"
    return None


def _python_cipher_call_creates_decryptor(
    cipher: ast.Call,
    scope: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Track decryptor calls to one Cipher assignment, not a reused name."""

    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "decryptor"
        and node.func.value is cipher
        for node in ast.walk(scope)
    ):
        return True

    assignment = parents.get(cipher)
    if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
        return False
    names = {
        name
        for target in (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
        for name in _python_assignment_target_names(target)
    }
    if not names:
        return False
    start_line = getattr(assignment, "lineno", 0)
    for name in names:
        reassignment_lines = [
            getattr(node, "lineno", 0)
            for node in ast.walk(scope)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and node is not assignment
            and getattr(node, "lineno", 0) > start_line
            and name
            in {
                target_name
                for target in (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else (node.target,)
                )
                for target_name in _python_assignment_target_names(target)
            }
        ]
        end_line = min(reassignment_lines, default=10**9)
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "decryptor"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
            and start_line < getattr(node, "lineno", 0) < end_line
            for node in ast.walk(scope)
        ):
            return True
    return False


def _python_has_complete_one_shot_aead_route(tree: ast.Module) -> bool:
    """Recognize an independent invoked AESGCM/AESCCM round-trip route."""

    aead_names = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and _python_name(node.value.func).rsplit(".", 1)[-1]
        in {"AESGCM", "AESCCM", "ChaCha20Poly1305"}
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        for name in _python_assignment_target_names(target)
    }
    directions = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"encrypt", "decrypt"}
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in aead_names
        and len(node.args) >= 3
    }
    return directions == {"encrypt", "decrypt"}


def _python_aad_is_outside_selected_mode_route(
    tree: ast.Module,
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Return whether a mode guard excludes this AAD call on a used route."""

    function = _python_containing_function(call, parents)
    if function is None:
        return False
    parameters = (*function.args.posonlyargs, *function.args.args)
    parameter_names = {parameter.arg for parameter in parameters}

    current: ast.AST = call
    guarded_parameter: str | None = None
    guarded_mode: str | None = None
    while current in parents:
        parent = parents[current]
        if isinstance(parent, ast.If) and any(
            current is descendant
            for branch in parent.orelse
            for descendant in ast.walk(branch)
        ):
            guard = parent.test
            if (
                isinstance(guard, ast.Call)
                and _python_name(guard.func).rsplit(".", 1)[-1] == "isinstance"
                and len(guard.args) == 2
                and isinstance(guard.args[0], ast.Name)
                and guard.args[0].id in parameter_names
            ):
                mode_name = _python_name(guard.args[1]).rsplit(".", 1)[-1]
                if mode_name in {"CBC", "CFB", "CFB8", "CTR", "ECB", "OFB", "XTS"}:
                    guarded_parameter = guard.args[0].id
                    guarded_mode = mode_name
                    break
        if parent is function:
            break
        current = parent
    if guarded_parameter is None or guarded_mode is None:
        return False

    parameter_index = next(
        index
        for index, parameter in enumerate(parameters)
        if parameter.arg == guarded_parameter
    )
    for candidate in ast.walk(tree):
        if not (
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == function.name
        ):
            continue
        argument = (
            candidate.args[parameter_index]
            if len(candidate.args) > parameter_index
            else next(
                (
                    keyword.value
                    for keyword in candidate.keywords
                    if keyword.arg == guarded_parameter
                ),
                None,
            )
        )
        if argument is not None and _python_expression_is_constructed_mode(
            tree,
            argument,
            guarded_mode,
            before=candidate,
        ):
            return True
    return False


def _python_expression_is_constructed_mode(
    tree: ast.Module,
    expression: ast.AST,
    mode_name: str,
    *,
    before: ast.AST,
    seen: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(expression, ast.Call):
        return _python_name(expression.func).rsplit(".", 1)[-1] == mode_name
    if not isinstance(expression, ast.Name) or expression.id in seen:
        return False
    boundary = getattr(before, "lineno", 0)
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and getattr(node, "lineno", 0) < boundary
        and any(
            expression.id in _python_assignment_target_names(target)
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
        )
    ]
    if not candidates:
        return False
    assignment = max(candidates, key=lambda item: getattr(item, "lineno", 0))
    return _python_expression_is_constructed_mode(
        tree,
        assignment.value,
        mode_name,
        before=assignment,
        seen=seen | {expression.id},
    )


def _python_cipher_receives_mode_class(tree: ast.Module) -> bool:
    """Find mode classes forwarded unchanged into ``Cipher``.

    Passing a mode class through a helper parameter is just as unusable as
    writing ``Cipher(..., modes.CBC)`` directly.  A helper that constructs the
    supplied class first (for example ``mode()``) remains a valid route.
    """

    mode_names = {"CBC", "CFB", "CFB8", "CTR", "ECB", "GCM", "OFB", "XTS"}
    assignments: dict[str, list[tuple[int, ast.AST]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for target in targets:
            for name in _python_assignment_target_names(target):
                assignments.setdefault(name, []).append(
                    (getattr(node, "lineno", 0), node.value)
                )

    def is_mode_class(
        expression: ast.AST,
        before_line: int,
        seen: frozenset[str] = frozenset(),
    ) -> bool:
        if isinstance(expression, ast.Attribute):
            name = _python_name(expression)
            return name.startswith("modes.") and name.rsplit(".", 1)[-1] in mode_names
        if not isinstance(expression, ast.Name) or expression.id in seen:
            return False
        definitions = [
            value
            for line, value in assignments.get(expression.id, ())
            if line < before_line
        ]
        return bool(
            definitions
            and is_mode_class(
                definitions[-1],
                before_line,
                seen | {expression.id},
            )
        )

    forwarded_parameters: dict[str, set[int]] = {}
    forwarded_keywords: dict[str, set[str]] = {}
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        parameters = (*function.args.posonlyargs, *function.args.args)
        by_name = {parameter.arg: index for index, parameter in enumerate(parameters)}
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and _python_name(call.func).rsplit(".", 1)[-1] == "Cipher"
                and len(call.args) >= 2
                and isinstance(call.args[1], ast.Name)
                and call.args[1].id in by_name
            ):
                continue
            parameter = call.args[1].id
            forwarded_parameters.setdefault(function.name, set()).add(by_name[parameter])
            forwarded_keywords.setdefault(function.name, set()).add(parameter)

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if (
            _python_name(call.func).rsplit(".", 1)[-1] == "Cipher"
            and len(call.args) >= 2
            and is_mode_class(call.args[1], getattr(call, "lineno", 0))
        ):
            return True
        simple = _python_name(call.func).rsplit(".", 1)[-1]
        if any(
            index < len(call.args)
            and is_mode_class(call.args[index], getattr(call, "lineno", 0))
            for index in forwarded_parameters.get(simple, ())
        ):
            return True
        if any(
            keyword.arg in forwarded_keywords.get(simple, ())
            and is_mode_class(keyword.value, getattr(call, "lineno", 0))
            for keyword in call.keywords
            if keyword.arg is not None
        ):
            return True
    return False


def _python_has_ancestor(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    kind: type[ast.AST],
) -> bool:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, kind):
            return True
        current = parents.get(current)
    return False


def _python_cipher_tag_receiver_failure(tree: ast.Module) -> str | None:
    """Reject reading an AEAD tag from ``Cipher`` instead of its encryptor."""

    cipher_names = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and _python_name(node.value.func).rsplit(".", 1)[-1] == "Cipher"
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        for name in _python_assignment_target_names(target)
    }
    if any(
        isinstance(node, ast.Attribute)
        and node.attr == "tag"
        and isinstance(node.value, ast.Name)
        and node.value.id in cipher_names
        for node in ast.walk(tree)
    ):
        return "the authentication tag is read from Cipher instead of its encryptor"
    return None


def _python_cbc_receives_unpack_tuple(tree: ast.Module) -> bool:
    assignments: dict[str, list[tuple[int, ast.AST]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for target in targets:
            for name in _python_assignment_target_names(target):
                assignments.setdefault(name, []).append(
                    (getattr(node, "lineno", 0), node.value)
                )

    def is_unpack_tuple(
        expression: ast.AST,
        before_line: int,
        seen: frozenset[str] = frozenset(),
    ) -> bool:
        if isinstance(expression, ast.Subscript):
            if not isinstance(expression.slice, ast.Slice):
                return False
            candidate = expression.value
        else:
            candidate = expression
        if isinstance(candidate, ast.Call) and _python_name(candidate.func).endswith(
            "struct.unpack"
        ):
            return True
        if not isinstance(candidate, ast.Name) or candidate.id in seen:
            return False
        definitions = [
            value
            for line, value in assignments.get(candidate.id, ())
            if line < before_line
        ]
        return bool(
            definitions
            and is_unpack_tuple(
                definitions[-1],
                before_line,
                seen | {candidate.id},
            )
        )

    return any(
        node.args
        and is_unpack_tuple(node.args[0], getattr(node, "lineno", 0))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _python_name(node.func).endswith("modes.CBC")
    )


def _python_bound_names(tree: ast.Module) -> set[str]:
    """Return names that the source can resolve without executing it.

    This is intentionally conservative: a binding anywhere in the source is
    enough to suppress an unavailable-name finding.  The validator only uses
    it for required cryptography namespaces whose absence is otherwise a
    definite task-API failure.
    """

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name != "*"
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.update(
                    argument.arg
                    for argument in (
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    )
                )
                if node.args.vararg is not None:
                    names.add(node.args.vararg.arg)
                if node.args.kwarg is not None:
                    names.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _python_context_completed_before(tree: ast.Module, call: ast.Call) -> bool:
    """Return whether an earlier consumed finalize completed this context.

    A malformed experimental call after a stored or returned encryption result
    does not erase the already completed task route.
    """

    if not isinstance(call.func, ast.Attribute):
        return False
    receiver = _python_name(call.func.value)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for candidate in ast.walk(tree):
        if not isinstance(candidate, ast.Call) or candidate is call:
            continue
        if getattr(candidate, "lineno", 0) >= getattr(call, "lineno", 0):
            continue
        if not isinstance(candidate.func, ast.Attribute):
            continue
        if candidate.func.attr != "finalize" or _python_name(
            candidate.func.value
        ) != receiver:
            continue
        parent = parents.get(candidate)
        if parent is not None and not isinstance(parent, ast.Expr):
            return True
    return False


def _python_cipher_lifecycle_failure(
    tree: ast.Module,
    task_id: str,
) -> str | None:
    """Check lifecycle completion only for the direction being assessed."""

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    required = (
        {"encryptor"}
        if task_id.endswith("T2")
        else {"decryptor"}
        if task_id.endswith("T3")
        else {"encryptor", "decryptor"}
    )
    contexts: set[tuple[ast.AST, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(
            node.value, ast.Call
        ):
            continue
        constructor = _python_name(node.value.func).rsplit(".", 1)[-1]
        if constructor not in required:
            continue
        scope = _python_containing_scope(node, parents, tree)
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        contexts.update(
            (scope, name)
            for target in targets
            for name in _python_assignment_target_names(target)
        )

    for scope, name in contexts:
        updates = tuple(
            call
            for call in ast.walk(scope)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "update"
            and _python_name(call.func.value) == name
        )
        terminals = tuple(
            call
            for call in ast.walk(scope)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in {"finalize", "finalize_with_tag"}
            and _python_name(call.func.value) == name
        )
        if updates and not terminals:
            return f"cipher context {name!r} processes input without finalizing"
    return None


def _python_required_cipher_direction_failure(
    tree: ast.Module,
    task_id: str,
) -> str | None:
    """Require an assigned PyCA context to process and finalize its direction."""

    required = (
        ("encryptor", "decryptor")
        if task_id.endswith("T1")
        else ("encryptor",)
        if task_id.endswith("T2")
        else ("decryptor",)
    )
    one_shot = {
        call.func.attr
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"encrypt", "decrypt"}
    }
    for constructor in required:
        names = _python_assigned_call_names(tree, {constructor})
        if not names or constructor.removesuffix("or") in one_shot:
            continue
        if any(
            _python_receiver_calls(tree, name, "update")
            and _python_receiver_calls(tree, name, "finalize")
            for name in names
        ):
            continue
        if _python_mapping_context_is_complete(tree, constructor):
            continue
        direction = constructor.removesuffix("or")
        return f"the assigned {direction} context never processes and finalizes data"
    return None


def _python_mapping_context_is_complete(tree: ast.Module, key: str) -> bool:
    """Recognize a PyCA context carried through a string-keyed mapping."""

    operations: dict[str, set[str]] = {}
    for call in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"update", "finalize"}
        and isinstance(node.func.value, ast.Subscript)
    ):
        receiver = call.func.value
        index = receiver.slice
        if not (isinstance(index, ast.Constant) and index.value == key):
            continue
        identity = ast.dump(receiver.value, include_attributes=False)
        operations.setdefault(identity, set()).add(call.func.attr)
    return any({"update", "finalize"} <= value for value in operations.values())


def _python_round_trip_connection_failure(
    tree: ast.Module,
    source: str,
) -> str | None:
    """Return definite key, nonce, tag, or ciphertext connection failures."""

    if re.search(
        r"\blength\s*=\s*(?P<size>\d+).*?\.\s*derive\s*\([^)]*\).*?"
        r"\biv\s*=\s*\w+\s*\[\s*(?P=size)\s*:\s*\]",
        source,
        flags=re.DOTALL,
    ):
        return "the derived key slice used as an IV is empty"

    prefix_slices = {
        match.group("name"): match.group("size")
        for match in re.finditer(
            r"\b(?P<name>[A-Za-z_]\w*)\s*=\s*[A-Za-z_]\w*\s*"
            r"\[\s*:\s*(?P<size>\d+)\s*\]",
            source,
        )
    }
    if any(
        re.search(
            rf"modes\s*\.\s*CBC\s*\(\s*{re.escape(name)}\s*"
            rf"\[\s*{size}\s*:\s*\]\s*\)",
            source,
        )
        for name, size in prefix_slices.items()
    ):
        return "CBC decryption uses an empty suffix of the extracted IV"

    if _python_slices_encoded_envelope_before_decoding(tree):
        return "decryption slices an encoded envelope before decoding it"

    if _python_suffix_container_split_failure(tree):
        return "decryption includes transported IV bytes in the ciphertext slice"

    if _python_overwrites_ciphertext_before_decryption(tree):
        return "decryption consumes a value that replaced the produced ciphertext"

    if any(
        isinstance(node, ast.Attribute)
        and node.attr == "tag"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "algorithm"
        for node in ast.walk(tree)
    ):
        return "the authentication tag is read from the cipher algorithm instead of the encryptor"

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    if _python_gcm_returned_tag_used_as_nonce(tree):
        return "GCM decryption uses the encryption tag as its nonce"
    tuple_failure = _python_directional_return_assignment_failure(tree)
    if tuple_failure is not None:
        return tuple_failure
    for call in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _python_name(node.func).endswith("modes.GCM")
    ):
        function = _python_containing_function(call, parents)
        if function is None or not _directional_function_name(function.name, "decrypt"):
            continue
        if len(call.args) >= 2:
            nonce = ast.unparse(call.args[0]).lower()
            tag = ast.unparse(call.args[1]).lower()
            if re.sub(r"\s+", "", nonce) == re.sub(r"\s+", "", tag):
                return "GCM decryption uses its nonce as the authentication tag"
            if "tag" in nonce and "tag" not in tag:
                return "GCM nonce and authentication-tag arguments are reversed"
            if re.search(r"\[\s*-\s*16\s*:\s*\]", nonce) and re.search(
                r"(?:\\x00|\\0|\biv\b|\bnonce\b)", tag
            ):
                return "GCM nonce and authentication-tag arguments are reversed"
        if _python_gcm_tag_comes_from_truncated_ciphertext(function, call):
            return (
                "GCM decryption extracts its authentication tag after removing "
                "that tag from the ciphertext buffer"
            )
        if _python_gcm_prefix_tag_uses_fixed_ciphertext_tail(function, call):
            return (
                "GCM decryption treats an all-but-fixed-tail prefix as the "
                "authentication tag"
            )

    encrypt_functions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _directional_function_name(node.name, "encrypt")
    )
    decrypt_functions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _directional_function_name(node.name, "decrypt")
    )
    if encrypt_functions and decrypt_functions:
        encrypt_returns = "\n".join(
            ast.unparse(node.value)
            for function in encrypt_functions
            for node in ast.walk(function)
            if isinstance(node, ast.Return) and node.value is not None
        ).lower()
        if re.search(
            r"(?:ciphertext|encrypted|\bct\b).*(?:\.tag\b|\btag\b)",
            encrypt_returns,
        ):
            for function in decrypt_functions:
                for call in ast.walk(function):
                    if not (
                        isinstance(call, ast.Call)
                        and _python_name(call.func).endswith("modes.GCM")
                        and len(call.args) >= 2
                        and isinstance(call.args[1], ast.Subscript)
                        and isinstance(call.args[1].slice, ast.Slice)
                    ):
                        continue
                    lower = call.args[1].slice.lower
                    if lower is not None and not (
                        isinstance(lower, ast.UnaryOp)
                        and isinstance(lower.op, ast.USub)
                    ):
                        return (
                            "GCM encryption appends its tag after ciphertext, but "
                            "decryption reads the tag from an interior prefix slice"
                        )
        decrypt_text = "\n".join(ast.unparse(node) for node in decrypt_functions).lower()
        if (
            "gcm" in source.lower()
            and re.search(r"\bencrypted\s*\[\s*:\s*12\s*\]", decrypt_text)
            and not re.search(r"\b(?:iv|nonce)\b", encrypt_returns)
        ):
            return "decryption extracts a nonce that encryption did not transport"
    return None


def _python_gcm_returned_tag_used_as_nonce(tree: ast.Module) -> bool:
    """Find a tuple output sourced from ``encryptor.tag`` used as a GCM nonce."""

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    tag_positions: dict[str, set[int]] = {}
    for name, function in functions.items():
        if not _directional_function_name(name, "encrypt"):
            continue
        assignments = _python_simple_assignments(function)
        returned = [
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Return) and node.value is not None
        ]
        if len(returned) != 1 or not isinstance(returned[0], (ast.Tuple, ast.List)):
            continue
        positions = {
            index
            for index, value in enumerate(returned[0].elts)
            if any(
                isinstance(child, ast.Attribute) and child.attr == "tag"
                for child in ast.walk(
                    _python_resolve_simple_name(value, assignments)
                )
            )
        }
        if positions:
            tag_positions[name] = positions
    if not tag_positions:
        return False

    tag_outputs: set[str] = set()
    for assignment in ast.walk(tree):
        if not (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], (ast.Tuple, ast.List))
            and isinstance(assignment.value, ast.Call)
        ):
            continue
        positions = tag_positions.get(
            _python_name(assignment.value.func).rsplit(".", 1)[-1]
        )
        if not positions:
            continue
        tag_outputs.update(
            target.id
            for index, target in enumerate(assignment.targets[0].elts)
            if index in positions and isinstance(target, ast.Name)
        )

    for decrypt_name, function in functions.items():
        if not _directional_function_name(decrypt_name, "decrypt"):
            continue
        parameters = (*function.args.posonlyargs, *function.args.args)
        nonce_parameters = {
            parameters.index(argument)
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and _python_name(call.func).endswith("modes.GCM")
            and call.args
            and isinstance(call.args[0], ast.Name)
            for argument in parameters
            if argument.arg == call.args[0].id
        }
        if not nonce_parameters:
            continue
        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and _python_name(call.func).rsplit(".", 1)[-1] == decrypt_name
            ):
                continue
            if any(
                position < len(call.args)
                and isinstance(call.args[position], ast.Name)
                and call.args[position].id in tag_outputs
                for position in nonce_parameters
            ):
                return True
    return False


def _python_directional_return_assignment_failure(tree: ast.Module) -> str | None:
    """Find tuple unpacking that swaps a helper's key and ciphertext outputs."""

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _directional_function_name(node.name, "encrypt")
    }
    return_roles: dict[str, tuple[str | None, ...]] = {}
    for name, function in functions.items():
        returned = [
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Return) and node.value is not None
        ]
        if len(returned) != 1 or not isinstance(returned[0], (ast.Tuple, ast.List)):
            continue
        return_roles[name] = tuple(
            _python_cipher_value_role(ast.unparse(value))
            for value in returned[0].elts
        )

    for assignment in ast.walk(tree):
        if not isinstance(assignment, ast.Assign) or not isinstance(
            assignment.value, ast.Call
        ):
            continue
        name = _python_name(assignment.value.func).rsplit(".", 1)[-1]
        roles = return_roles.get(name)
        if roles is None or len(assignment.targets) != 1:
            continue
        target = assignment.targets[0]
        if not isinstance(target, (ast.Tuple, ast.List)) or len(target.elts) != len(roles):
            continue
        for returned_role, target_value in zip(roles, target.elts):
            target_role = _python_cipher_value_role(ast.unparse(target_value))
            if (
                returned_role in {"key", "ciphertext"}
                and target_role in {"key", "ciphertext"}
                and returned_role != target_role
            ):
                return (
                    f"encryption helper {name!r} returns {returned_role} material "
                    f"into a {target_role}-named result"
                )
    return None


def _python_gcm_serialized_tag_split_failure(tree: ast.Module) -> bool:
    """Find a decrypt split that cannot recover a full emitted GCM tag."""

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    output_names = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _python_expression_contains_encrypt_output(node.value)
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        for name in _python_assignment_target_names(target)
    }
    transported_values = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return))
        and node.value is not None
    ]
    emits_full_tag = any(
        {
            node.id
            for node in ast.walk(value)
            if isinstance(node, ast.Name)
        }
        & output_names
        and any(
            isinstance(node, ast.Attribute)
            and node.attr in {"tag", "authentication_tag"}
            and not isinstance(parents.get(node), ast.Subscript)
            for node in ast.walk(value)
        )
        for value in transported_values
    )
    if not emits_full_tag:
        return False

    decrypt_functions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _directional_function_name(node.name, "decrypt")
    )
    decrypt_scopes: tuple[ast.AST, ...] = decrypt_functions or (tree,)
    for function in decrypt_scopes:
        assignments: dict[str, ast.AST] = {}
        for node in sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and (
                    _python_containing_function(node, parents) is function
                    if function is not tree
                    else _python_containing_function(node, parents) is None
                )
            ),
            key=lambda node: getattr(node, "lineno", 0),
        ):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                for name in _python_assignment_target_names(target):
                    assignments[name] = node.value

        for gcm_call in (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and _python_name(node.func).endswith("modes.GCM")
        ):
            tag = (
                gcm_call.args[1]
                if len(gcm_call.args) >= 2
                else next(
                    (
                        keyword.value
                        for keyword in gcm_call.keywords
                        if keyword.arg == "tag"
                    ),
                    None,
                )
            )
            if isinstance(tag, ast.Name):
                tag = assignments.get(tag.id, tag)
            if not (
                isinstance(tag, ast.Subscript)
                and isinstance(tag.value, ast.Name)
                and isinstance(tag.slice, ast.Slice)
            ):
                continue
            lower = _python_nonnegative_integer(tag.slice.lower)
            upper = _python_nonnegative_integer(tag.slice.upper)
            if lower is None or upper is None or upper - lower == 16:
                continue
            container = tag.value.id
            ciphertext_names = {
                name
                for name, value in assignments.items()
                if isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Name)
                and value.value.id == container
                and isinstance(value.slice, ast.Slice)
                and _python_nonnegative_integer(value.slice.lower) == upper
                and value.slice.upper is None
            }
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "update"
                and call.args
                and any(
                    isinstance(name, ast.Name) and name.id in ciphertext_names
                    for name in ast.walk(call.args[0])
                )
                for call in ast.walk(function)
            ):
                return True
    return False


def _python_ciphertext_representation_failure(tree: ast.Module) -> str | None:
    """Find a decoded envelope whose metadata remains in the cipher input.

    Whether arbitrary cipher bytes happen to be accepted by a Base64 decoder
    depends on runtime data and is outside the static validity contract.  A
    decoded envelope that still contains a source-visible IV or tag is a
    definite connection error, so that narrower case remains invalid.
    """

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for scope in _python_statement_blocks(tree):
        scope_node: ast.AST = tree
        if scope is not tree.body:
            owner = next(
                (
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.body is scope
                ),
                tree,
            )
            scope_node = owner

        encrypt_contexts: set[str] = set()
        decrypt_contexts: set[str] = set()
        raw_ciphertext: set[str] = set()
        encoded_envelopes: set[str] = set()
        decoded_envelopes: set[str] = set()
        for statement in scope:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
            names = {
                name
                for target in targets
                for name in _python_assignment_target_names(target)
            }
            value = statement.value
            if isinstance(value, ast.Call):
                simple = _python_name(value.func).rsplit(".", 1)[-1]
                if simple == "encryptor":
                    encrypt_contexts.update(names)
                elif simple == "decryptor":
                    decrypt_contexts.update(names)
                elif simple in {"b64decode", "urlsafe_b64decode"} and value.args:
                    input_names = {
                        node.id
                        for node in ast.walk(value.args[0])
                        if isinstance(node, ast.Name)
                    }
                    if input_names & encoded_envelopes:
                        decoded_envelopes.update(names)

            if _python_expression_uses_context_output(value, encrypt_contexts):
                raw_ciphertext.update(names)

            encoded_input = _python_base64_encode_input(value)
            if encoded_input is not None:
                input_names = {
                    node.id
                    for node in ast.walk(encoded_input)
                    if isinstance(node, ast.Name)
                }
                if input_names & raw_ciphertext:
                    if any(
                        re.search(r"(?:iv|nonce|tag)", name, re.IGNORECASE)
                        for name in input_names - raw_ciphertext
                    ):
                        encoded_envelopes.update(names)
                    # An in-place reassignment such as ``ct = b64encode(ct)``
                    # changes the representation carried by the target name.
                    raw_ciphertext.difference_update(names)

        if not raw_ciphertext and not decoded_envelopes:
            continue
        for call in ast.walk(scope_node):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {"update", "decrypt"}
                and call.args
            ):
                continue
            receiver = _python_name(call.func.value)
            if call.func.attr == "update" and receiver not in decrypt_contexts:
                continue
            argument = call.args[0]
            if any(
                isinstance(node, ast.Name) and node.id in decoded_envelopes
                for node in ast.walk(argument)
            ):
                return (
                    "decryption consumes a decoded ciphertext envelope without "
                    "removing its serialized IV or tag metadata"
                )
    return None


def _python_expression_contains_encrypt_output(expression: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and (
            node.func.attr == "encrypt"
            or (
                node.func.attr in {"update", "finalize"}
                and re.search(r"encrypt|cipher", _python_name(node.func.value), re.IGNORECASE)
            )
        )
        for node in ast.walk(expression)
    )


def _python_expression_uses_context_output(
    expression: ast.AST,
    contexts: set[str],
) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"update", "finalize"}
        and _python_name(node.func.value) in contexts
        for node in ast.walk(expression)
    )


def _python_base64_encode_input(expression: ast.AST) -> ast.AST | None:
    for node in ast.walk(expression):
        if (
            isinstance(node, ast.Call)
            and _python_name(node.func).rsplit(".", 1)[-1]
            in {"b64encode", "urlsafe_b64encode"}
            and node.args
        ):
            return node.args[0]
    return None


def _python_nonnegative_integer(expression: ast.AST | None) -> int | None:
    if (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, int)
        and not isinstance(expression.value, bool)
        and expression.value >= 0
    ):
        return expression.value
    return None


def _python_gcm_tag_comes_from_truncated_ciphertext(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    gcm_call: ast.Call,
) -> bool:
    """Detect a definite straight-line tag/ciphertext rebinding error.

    A common generated pattern first replaces ``ciphertext`` with
    ``ciphertext[header:-16]`` and only then evaluates
    ``tag = ciphertext[-16:]``.  The latter slice is no longer the tag emitted
    by encryption.  Keep this check deliberately narrow: it only follows
    simple assignments in the same function and before the GCM constructor.
    """

    tag_expression: ast.AST | None = None
    if len(gcm_call.args) >= 2:
        tag_expression = gcm_call.args[1]
    else:
        tag_expression = next(
            (
                keyword.value
                for keyword in gcm_call.keywords
                if keyword.arg == "tag"
            ),
            None,
        )
    if tag_expression is None:
        return False

    tag_assignment_line = getattr(gcm_call, "lineno", 0)
    if isinstance(tag_expression, ast.Name):
        assigned = _python_latest_simple_assignment(
            function,
            tag_expression.id,
            before_line=tag_assignment_line,
        )
        if assigned is None:
            return False
        tag_expression = assigned.value
        tag_assignment_line = getattr(assigned, "lineno", tag_assignment_line)

    if not (
        isinstance(tag_expression, ast.Subscript)
        and isinstance(tag_expression.value, ast.Name)
        and _python_is_negative_tail_slice(tag_expression.slice)
    ):
        return False
    buffer_name = tag_expression.value.id
    prior_rebind = _python_latest_simple_assignment(
        function,
        buffer_name,
        before_line=tag_assignment_line,
    )
    return bool(
        prior_rebind is not None
        and isinstance(prior_rebind.value, ast.Subscript)
        and _python_slice_removes_tail(prior_rebind.value.slice)
    )


def _python_gcm_prefix_tag_uses_fixed_ciphertext_tail(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    gcm_call: ast.Call,
) -> bool:
    """Find a tag-prefix split whose ciphertext is taken as a fixed tail."""

    tag = (
        gcm_call.args[1]
        if len(gcm_call.args) >= 2
        else next(
            (keyword.value for keyword in gcm_call.keywords if keyword.arg == "tag"),
            None,
        )
    )
    if not (
        isinstance(tag, ast.Subscript)
        and isinstance(tag.value, ast.Name)
        and isinstance(tag.slice, ast.Slice)
        and tag.slice.lower is None
        and _python_negative_integer(tag.slice.upper) is not None
    ):
        return False
    tail_size = _python_negative_integer(tag.slice.upper)
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "update"
        and call.args
        and isinstance(call.args[0], ast.Subscript)
        and isinstance(call.args[0].value, ast.Name)
        and call.args[0].value.id == tag.value.id
        and isinstance(call.args[0].slice, ast.Slice)
        and _python_negative_integer(call.args[0].slice.lower) == tail_size
        and call.args[0].slice.upper is None
        for call in ast.walk(function)
    )


def _python_latest_simple_assignment(
    scope: ast.AST,
    name: str,
    *,
    before_line: int,
) -> ast.Assign | ast.AnnAssign | None:
    candidates: list[ast.Assign | ast.AnnAssign] = []
    for node in ast.walk(scope):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if getattr(node, "lineno", 0) >= before_line or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if any(name in _python_assignment_target_names(target) for target in targets):
            candidates.append(node)
    return max(candidates, key=lambda node: node.lineno, default=None)


def _python_is_negative_tail_slice(node: ast.AST) -> bool:
    return bool(
        isinstance(node, ast.Slice)
        and isinstance(node.lower, ast.UnaryOp)
        and isinstance(node.lower.op, ast.USub)
        and isinstance(node.lower.operand, ast.Constant)
        and isinstance(node.lower.operand.value, int)
        and node.upper is None
    )


def _python_slice_removes_tail(node: ast.AST) -> bool:
    return bool(
        isinstance(node, ast.Slice)
        and isinstance(node.upper, ast.UnaryOp)
        and isinstance(node.upper.op, ast.USub)
        and isinstance(node.upper.operand, ast.Constant)
        and isinstance(node.upper.operand.value, int)
    )


def _python_slices_encoded_envelope_before_decoding(tree: ast.Module) -> bool:
    encoded_encryptors: set[str] = set()
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _directional_function_name(node.name, "encrypt")
    ):
        if any(
            isinstance(node, ast.Return)
            and node.value is not None
            and any(
                isinstance(child, ast.Call)
                and _python_name(child.func).endswith(
                    ("b64encode", "urlsafe_b64encode", "hexlify")
                )
                for child in ast.walk(node.value)
            )
            for node in ast.walk(function)
        ):
            encoded_encryptors.add(function.name)
    if not encoded_encryptors:
        return False

    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _directional_function_name(node.name, "decrypt")
    ):
        parameters = {argument.arg for argument in (*function.args.posonlyargs, *function.args.args)}
        decoded_at: dict[str, int] = {}
        sliced_at: dict[str, int] = {}
        encoded_slice_aliases: dict[str, str] = {}
        assignments = sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
            ),
            key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)),
        )
        for node in assignments:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            line = getattr(node, "lineno", 0)
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            target_names = {
                name
                for target in targets
                for name in _python_assignment_target_names(target)
            }
            sliced_parameter = next(
                (
                    child.value.id
                    for child in ast.walk(value)
                    if isinstance(child, ast.Subscript)
                    and isinstance(child.value, ast.Name)
                    and child.value.id in parameters
                ),
                None,
            )
            if sliced_parameter is not None:
                encoded_slice_aliases.update(
                    {name: sliced_parameter for name in target_names}
                )
            elif isinstance(value, ast.Name) and value.id in encoded_slice_aliases:
                encoded_slice_aliases.update(
                    {name: encoded_slice_aliases[value.id] for name in target_names}
                )
            decode_calls = tuple(
                child
                for child in ast.walk(value)
                if isinstance(child, ast.Call)
                and _python_name(child.func).endswith(
                    ("b64decode", "urlsafe_b64decode", "unhexlify")
                )
                and child.args
            )
            for parameter in parameters:
                if any(
                    any(
                        isinstance(child, ast.Subscript)
                        and isinstance(child.value, ast.Name)
                        and child.value.id == parameter
                        for child in ast.walk(decode_call.args[0])
                    )
                    for decode_call in decode_calls
                ):
                    return True
                if any(
                    isinstance(decode_call.args[0], ast.Name)
                    and encoded_slice_aliases.get(decode_call.args[0].id) == parameter
                    for decode_call in decode_calls
                ):
                    return True
                if any(
                    isinstance(decode_call.args[0], ast.Name)
                    and decode_call.args[0].id == parameter
                    for decode_call in decode_calls
                ):
                    decoded_at[parameter] = min(decoded_at.get(parameter, line), line)
                if any(
                    isinstance(child, ast.Subscript)
                    and isinstance(child.value, ast.Name)
                    and child.value.id == parameter
                    for child in ast.walk(value)
                ):
                    sliced_at[parameter] = min(sliced_at.get(parameter, line), line)
        if any(
            parameter in decoded_at and sliced_at.get(parameter, decoded_at[parameter]) < decoded_at[parameter]
            for parameter in parameters
        ):
            return True
    return False


def _python_suffix_container_split_failure(tree: ast.Module) -> bool:
    """Find a definite overlap between transported IV and ciphertext slices."""

    transports_cipher_iv_key = any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"write", "send", "sendall"}
        and call.args
        and _python_name_sequence(call.args[0])[:3]
        == ("ciphertext", "iv", "key")
        for call in ast.walk(tree)
    )
    if not transports_cipher_iv_key:
        return False

    slices: dict[tuple[str, str], ast.Slice] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and isinstance(node.value.slice, ast.Slice)
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            role = next(
                (
                    candidate
                    for candidate in ("ciphertext", "iv", "key")
                    if candidate in target.id.lower()
                ),
                None,
            )
            if role is not None:
                slices[(node.value.value.id, role)] = node.value.slice

    for buffer in {name for name, _role in slices}:
        key_slice = slices.get((buffer, "key"))
        iv_slice = slices.get((buffer, "iv"))
        ciphertext_slice = slices.get((buffer, "ciphertext"))
        if key_slice is None or iv_slice is None or ciphertext_slice is None:
            continue
        key_tail = _python_negative_integer(key_slice.lower)
        if (
            key_tail is not None
            and key_slice.upper is None
            and _python_negative_integer(iv_slice.upper) == key_tail
            and _python_negative_integer(iv_slice.lower) is not None
            and ciphertext_slice.lower is None
            and _python_negative_integer(ciphertext_slice.upper) == key_tail
        ):
            return True
    return False


def _python_name_sequence(expression: ast.AST) -> tuple[str, ...]:
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        return _python_name_sequence(expression.left) + _python_name_sequence(expression.right)
    if isinstance(expression, ast.Name):
        return (expression.id.lower(),)
    return ()


def _python_negative_integer(expression: ast.AST | None) -> int | None:
    if (
        isinstance(expression, ast.UnaryOp)
        and isinstance(expression.op, ast.USub)
        and isinstance(expression.operand, ast.Constant)
        and isinstance(expression.operand.value, int)
    ):
        return expression.operand.value
    return None


def _python_cbc_transport_failure(tree: ast.Module) -> str | None:
    """Require a freshly generated CBC IV to reach the decrypting route."""

    fresh_helpers = _python_fresh_material_helpers(tree)
    encrypt_functions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _directional_function_name(node.name, "encrypt")
    )
    for function in encrypt_functions:
        fresh_names: set[str] = set()
        for assignment in sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and node.value is not None
            ),
            key=lambda node: getattr(node, "lineno", 0),
        ):
            if _python_expression_is_fresh(
                assignment.value, fresh_helpers, fresh_names
            ):
                targets = (
                    assignment.targets
                    if isinstance(assignment, ast.Assign)
                    else (assignment.target,)
                )
                fresh_names.update(
                    name
                    for target in targets
                    for name in _python_assignment_target_names(target)
                )

        for call in (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and _python_name(node.func).endswith("modes.CBC")
            and node.args
        ):
            iv = call.args[0]
            if not _python_expression_is_fresh(iv, fresh_helpers, fresh_names):
                continue
            iv_names = {
                node.id for node in ast.walk(iv) if isinstance(node, ast.Name)
            } & fresh_names
            returns = tuple(
                node.value
                for node in ast.walk(function)
                if isinstance(node, ast.Return) and node.value is not None
            )
            transported_names = set(iv_names)
            changed = True
            while changed:
                changed = False
                for assignment in ast.walk(function):
                    if (
                        not isinstance(assignment, (ast.Assign, ast.AnnAssign))
                        or assignment.value is None
                    ):
                        continue
                    referenced = {
                        node.id
                        for node in ast.walk(assignment.value)
                        if isinstance(node, ast.Name)
                    }
                    if not (referenced & transported_names):
                        continue
                    targets = (
                        assignment.targets
                        if isinstance(assignment, ast.Assign)
                        else (assignment.target,)
                    )
                    aliases = {
                        name
                        for target in targets
                        for name in _python_assignment_target_names(target)
                    }
                    if not aliases <= transported_names:
                        transported_names.update(aliases)
                        changed = True
            if transported_names and any(
                transported_names
                & {
                    node.id
                    for node in ast.walk(returned)
                    if isinstance(node, ast.Name)
                }
                for returned in returns
            ):
                continue
            return "the encryption path generates a CBC IV that it does not transport"

    # A same-scope program can overwrite the encryption IV with a slice of
    # ciphertext even though that IV was never prefixed to the ciphertext.
    assignments = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    )
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for assignment in assignments:
        value = assignment.value
        if not isinstance(value, ast.Subscript):
            continue
        target_names = {
            name
            for target in (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else (assignment.target,)
            )
            for name in _python_assignment_target_names(target)
        }
        if not target_names or not any("iv" in name.lower() for name in target_names):
            continue
        ciphertext_names = {
            node.id for node in ast.walk(value.value) if isinstance(node, ast.Name)
        }
        scope = _python_containing_scope(assignment, parents, tree)
        ciphertext_assignments = tuple(
            candidate
            for candidate in assignments
            if candidate.value is not None
            if _python_containing_scope(candidate, parents, tree) is scope
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "update"
                and "encrypt" in _python_name(call.func.value).lower()
                for call in ast.walk(candidate.value)
            )
        )
        if not any(
            ciphertext_names
            & {
                name
                for target in (
                    candidate.targets
                    if isinstance(candidate, ast.Assign)
                    else (candidate.target,)
                )
                for name in _python_assignment_target_names(target)
            }
            for candidate in ciphertext_assignments
        ):
            continue
        return "decryption extracts an IV that encryption did not transport"
    if _python_round_trip_uses_unrelated_fresh_iv(tree, fresh_helpers):
        return "decryption uses a fresh IV unrelated to the encryption operation"
    if _python_cbc_decrypts_unprefixed_ciphertext(tree):
        return "decryption extracts an IV that encryption did not transport"
    return None


def _python_cbc_decrypts_unprefixed_ciphertext(tree: ast.Module) -> bool:
    """Find a CBC decrypt route that treats raw cipher output as an IV envelope.

    PyCA's ``update``/``finalize`` result contains only ciphertext.  Therefore,
    when an encryption helper returns that result without its CBC IV, a paired
    decryption helper cannot recover the IV by slicing the ciphertext prefix.
    The check deliberately ignores concrete IV and block lengths.
    """

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def scoped_assignments(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> dict[str, ast.AST]:
        values: dict[str, ast.AST] = {}
        for node in sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and _python_containing_function(node, parents) is function
            ),
            key=lambda node: getattr(node, "lineno", 0),
        ):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                for name in _python_assignment_target_names(target):
                    values[name] = node.value
        return values

    def names(expression: ast.AST) -> set[str]:
        return {
            node.id for node in ast.walk(expression) if isinstance(node, ast.Name)
        }

    encrypt_functions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _directional_function_name(node.name, "encrypt")
    )
    direct_cbc_functions = {
        function
        for function in encrypt_functions
        if any(
            isinstance(call, ast.Call)
            and _python_name(call.func).endswith("modes.CBC")
            and call.args
            for call in ast.walk(function)
        )
    }
    cbc_helper_names = {function.name for function in direct_cbc_functions}
    route_transport: list[bool] = []
    for function in encrypt_functions:
        assignments = scoped_assignments(function)
        uses_cbc = function in direct_cbc_functions or any(
            isinstance(call, ast.Call)
            and _python_name(call.func).rsplit(".", 1)[-1] in cbc_helper_names
            for call in ast.walk(function)
        )
        if not uses_cbc:
            continue
        encrypt_context_names = _python_assigned_call_names(function, {"encryptor"})
        encrypt_context_names.update(
            name
            for name, value in assignments.items()
            if isinstance(value, ast.Call)
            and _directional_function_name(
                _python_name(value.func).rsplit(".", 1)[-1], "encrypt"
            )
        )

        def contains_encrypt_output(expression: ast.AST) -> bool:
            return any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {"update", "finalize"}
                and (
                    _python_name(call.func.value) in encrypt_context_names
                    or "encrypt" in _python_name(call.func.value).lower()
                )
                for call in ast.walk(expression)
            )

        iv_names = {
            node.id
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and _python_name(call.func).endswith("modes.CBC")
            and call.args
            for node in ast.walk(call.args[0])
            if isinstance(node, ast.Name)
        }
        iv_names.update(
            name
            for name in assignments
            if "iv" in name.lower() or "nonce" in name.lower()
        )
        changed = True
        while changed:
            changed = False
            for name, value in assignments.items():
                if name not in iv_names and names(value) & iv_names:
                    iv_names.add(name)
                    changed = True
        cipher_output_names = {
            name
            for name, value in assignments.items()
            if contains_encrypt_output(value)
        }
        changed = True
        while changed:
            changed = False
            for name, value in assignments.items():
                if name not in cipher_output_names and names(value) & cipher_output_names:
                    cipher_output_names.add(name)
                    changed = True
        for returned in (
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Return) and node.value is not None
        ):
            returned_names = names(returned)
            if not (
                contains_encrypt_output(returned)
                or returned_names & cipher_output_names
            ):
                continue
            route_transport.append(bool(returned_names & iv_names))
    if not route_transport or any(route_transport):
        return False

    decrypt_functions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _directional_function_name(node.name, "decrypt")
    )
    direct_decrypt_cbc = {
        function
        for function in decrypt_functions
        if any(
            isinstance(call, ast.Call)
            and _python_name(call.func).endswith("modes.CBC")
            and call.args
            for call in ast.walk(function)
        )
    }
    decrypt_cbc_helpers = {function.name for function in direct_decrypt_cbc}
    for function in decrypt_functions:
        uses_cbc = function in direct_decrypt_cbc or any(
            isinstance(call, ast.Call)
            and _python_name(call.func).rsplit(".", 1)[-1] in decrypt_cbc_helpers
            for call in ast.walk(function)
        )
        if not uses_cbc:
            continue
        parameters = {
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
            if argument.arg not in {"self", "cls"}
        }
        assignments = scoped_assignments(function)
        sliced_iv_names = {
            name
            for name, value in assignments.items()
            if "iv" in name.lower() or "nonce" in name.lower()
            if isinstance(value, ast.Subscript)
            if names(value.value) & parameters
        }
        if sliced_iv_names and any(
            isinstance(call, ast.Call)
            and (
                _directional_function_name(
                    _python_name(call.func).rsplit(".", 1)[-1], "decrypt"
                )
                or _python_name(call.func).endswith("modes.CBC")
            )
            and any(names(argument) & sliced_iv_names for argument in call.args)
            for call in ast.walk(function)
        ):
            return True
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and _python_name(call.func).endswith("modes.CBC")
                and call.args
            ):
                continue
            iv = call.args[0]
            if isinstance(iv, ast.Name):
                iv = assignments.get(iv.id, iv)
            if not isinstance(iv, ast.Subscript):
                continue
            source_names = names(iv.value)
            if source_names & parameters:
                return True
    return False


def _python_round_trip_uses_unrelated_fresh_iv(
    tree: ast.Module,
    fresh_helpers: set[str],
) -> bool:
    """Find a CBC result decrypted with a separately generated IV."""

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    internal_iv_encryptors: set[str] = set()
    for name, function in functions.items():
        if not _directional_function_name(name, "encrypt"):
            continue
        fresh_names: set[str] = set()
        for assignment in sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
            ),
            key=lambda node: getattr(node, "lineno", 0),
        ):
            if not _python_expression_is_fresh(
                assignment.value, fresh_helpers, fresh_names
            ):
                continue
            targets = (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else (assignment.target,)
            )
            fresh_names.update(
                target
                for item in targets
                for target in _python_assignment_target_names(item)
            )
        if any(
            isinstance(call, ast.Call)
            and _python_name(call.func).endswith("modes.CBC")
            and call.args
            and _python_expression_is_fresh(call.args[0], fresh_helpers, fresh_names)
            for call in ast.walk(function)
        ):
            internal_iv_encryptors.add(name)
    if not internal_iv_encryptors:
        return False

    assignments: dict[str, ast.AST] = {}
    encrypted_results: set[str] = set()
    associated_ivs: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        names = {
            name
            for target in targets
            for name in _python_assignment_target_names(target)
        }
        for name in names:
            assignments[name] = node.value
        if (
            isinstance(node.value, ast.Call)
            and _python_name(node.value.func).rsplit(".", 1)[-1]
            in internal_iv_encryptors
        ):
            if any(isinstance(target, (ast.Tuple, ast.List)) for target in targets):
                function = functions[
                    _python_name(node.value.func).rsplit(".", 1)[-1]
                ]
                returned_roles = _python_tuple_return_roles(function)
                for target in targets:
                    if not isinstance(target, (ast.Tuple, ast.List)):
                        continue
                    for index, item in enumerate(target.elts):
                        if not isinstance(item, ast.Name):
                            continue
                        role = (
                            returned_roles[index]
                            if index < len(returned_roles)
                            else None
                        ) or _python_cipher_value_role(item.id)
                        if role == "ciphertext":
                            encrypted_results.add(item.id)
                        elif role == "nonce":
                            associated_ivs.add(item.id)
            else:
                encrypted_results.update(names)

    for call in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _directional_function_name(
            _python_name(node.func).rsplit(".", 1)[-1], "decrypt"
        )
    ):
        definition = functions.get(_python_name(call.func).rsplit(".", 1)[-1])
        if definition is None:
            continue
        parameters = (*definition.args.posonlyargs, *definition.args.args)
        iv_index = next(
            (
                index
                for index, parameter in enumerate(parameters)
                if re.fullmatch(r"(?:iv|nonce|initialization_vector)", parameter.arg, re.I)
            ),
            None,
        )
        data_index = next(
            (
                index
                for index, parameter in enumerate(parameters)
                if re.search(r"(?:cipher|encrypt|sealed)", parameter.arg, re.I)
            ),
            None,
        )
        if (
            iv_index is None
            or data_index is None
            or max(iv_index, data_index) >= len(call.args)
        ):
            continue
        data_names = {
            node.id for node in ast.walk(call.args[data_index]) if isinstance(node, ast.Name)
        }
        if not data_names & encrypted_results:
            continue
        iv_argument = call.args[iv_index]
        if isinstance(iv_argument, ast.Name) and iv_argument.id in associated_ivs:
            continue
        iv_value = (
            assignments.get(iv_argument.id)
            if isinstance(iv_argument, ast.Name)
            else iv_argument
        )
        if iv_value is not None and _python_expression_is_fresh(
            iv_value, fresh_helpers, set()
        ):
            return True
    return False


def _python_tuple_return_roles(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str | None, ...]:
    """Infer semantic roles for one helper's tuple-shaped return value."""

    returns = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, (ast.Tuple, ast.List))
    ]
    if not returns:
        return ()
    width = max(len(value.elts) for value in returns)
    roles: list[str | None] = []
    for index in range(width):
        observed: set[str | None] = set()
        for value in returns:
            if index >= len(value.elts):
                continue
            expression = value.elts[index]
            role = _python_cipher_value_role(ast.unparse(expression))
            if role in {None, "payload"} and _directional_function_name(
                function.name,
                "encrypt",
            ):
                if any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr in {"encrypt", "update", "finalize"}
                    for call in ast.walk(expression)
                ):
                    role = "ciphertext"
            observed.add(role)
        observed.discard(None)
        roles.append(next(iter(observed)) if len(observed) == 1 else None)
    return tuple(roles)


def _python_round_trip_callsite_failure(tree: ast.Module) -> str | None:
    """Reject an explicit round trip that decrypts fresh external input.

    General alias and object-flow recovery belongs to the regular operation
    analysis.  This focused check covers the unambiguous disconnected case:
    the program computes an encryption result, then supplies newly read input
    to its corresponding decrypt helper instead.
    """

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assignments: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for target in targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = node.value

    encryption_results: set[str] = set()
    for name, value in assignments.items():
        if isinstance(value, ast.Call) and _directional_function_name(
            _python_name(value.func).rsplit(".", 1)[-1], "encrypt"
        ):
            encryption_results.add(name)

    encrypt_output_keys: dict[str, ast.Name] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and isinstance(node.value, ast.Call)
        ):
            continue
        helper_name = _python_name(node.value.func).rsplit(".", 1)[-1]
        helper = functions.get(helper_name)
        if helper is None or not _directional_function_name(helper_name, "encrypt"):
            continue
        return_roles = _python_tuple_return_roles(helper)
        outputs: dict[str, str | None] = {}
        for index, target in enumerate(node.targets[0].elts):
            if not isinstance(target, ast.Name):
                continue
            role = return_roles[index] if index < len(return_roles) else None
            outputs[target.id] = role or _python_cipher_value_role(target.id)
        keys = [name for name, role in outputs.items() if role == "key"]
        ciphertexts = [
            name for name, role in outputs.items() if role == "ciphertext"
        ]
        for ciphertext in ciphertexts:
            assignments[ciphertext] = node.value
            encryption_results.add(ciphertext)
            if len(keys) == 1:
                encrypt_output_keys[ciphertext] = ast.Name(id=keys[0], ctx=ast.Load())
    if not encryption_results:
        return None

    decrypt_calls = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _directional_function_name(
            _python_name(node.func).rsplit(".", 1)[-1], "decrypt"
        )
        and _python_name(node.func).rsplit(".", 1)[-1] in functions
    )
    if not decrypt_calls:
        return None

    key_failure = _python_directional_key_callsite_failure(
        tree,
        functions,
        assignments,
        encryption_results,
        encrypt_output_keys,
        decrypt_calls,
    )
    if key_failure is not None:
        return key_failure

    for call in decrypt_calls:
        definition = functions[_python_name(call.func).rsplit(".", 1)[-1]]
        parameters = (*definition.args.posonlyargs, *definition.args.args)
        parameter_offset = int(
            isinstance(call.func, ast.Attribute)
            and bool(parameters)
            and parameters[0].arg in {"self", "cls"}
        )
        ciphertext_index = next(
            (
                index - parameter_offset
                for index, parameter in enumerate(parameters)
                if re.search(r"(?:encrypt|cipher|sealed|entry)", parameter.arg, re.IGNORECASE)
                and index >= parameter_offset
            ),
            None,
        )
        if ciphertext_index is None or ciphertext_index >= len(call.args):
            continue
        names = {
            node.id for node in ast.walk(call.args[ciphertext_index]) if isinstance(node, ast.Name)
        }
        if names & encryption_results:
            return None
        if _python_expression_comes_from_external_input(
            call.args[ciphertext_index],
            assignments,
        ):
            return "decryption does not consume the ciphertext produced by encryption"
    return None


def _python_expression_comes_from_external_input(
    expression: ast.AST,
    assignments: dict[str, ast.AST],
    visited: frozenset[str] = frozenset(),
) -> bool:
    """Trace a decrypt argument to a definite input/read operation."""

    if any(
        isinstance(node, ast.Call)
        and (
            _python_name(node.func).rsplit(".", 1)[-1]
            in {"input", "getpass", "read", "readline", "recv"}
        )
        for node in ast.walk(expression)
    ):
        return True
    for name in {
        node.id for node in ast.walk(expression) if isinstance(node, ast.Name)
    }:
        if name in visited or name not in assignments:
            continue
        if _python_expression_comes_from_external_input(
            assignments[name],
            assignments,
            visited | {name},
        ):
            return True
    return False


def _python_symmetric_directions_are_mutually_exclusive(tree: ast.Module) -> bool:
    """Find a mode branch that can execute encryption or decryption, never both."""

    def directions(statements: list[ast.stmt]) -> set[str]:
        result: set[str] = set()
        for statement in statements:
            for call in (
                node for node in ast.walk(statement) if isinstance(node, ast.Call)
            ):
                name = _python_name(call.func).rsplit(".", 1)[-1]
                for direction in ("encrypt", "decrypt"):
                    if _directional_function_name(name, direction):
                        result.add(direction)
        return result

    for branch in (node for node in ast.walk(tree) if isinstance(node, ast.If)):
        body = directions(branch.body)
        alternative = directions(branch.orelse)
        if {frozenset(body), frozenset(alternative)} == {
            frozenset({"encrypt"}),
            frozenset({"decrypt"}),
        }:
            return True
    return False


def _python_decrypt_helper_uses_unrelated_literal(tree: ast.Module) -> bool:
    """Find a decrypt wrapper that ignores ciphertext and hashes a fixed buffer."""

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for function in functions.values():
        if not _directional_function_name(function.name, "decrypt"):
            continue
        parameters = (*function.args.posonlyargs, *function.args.args)
        ciphertext_parameters = {
            parameter.arg
            for parameter in parameters
            if _python_cipher_value_role(parameter.arg) in {"ciphertext", "payload"}
        }
        if not ciphertext_parameters:
            continue
        terminal_inputs = [
            call.args[0]
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in {"update", "decrypt"}
            and call.args
        ]
        if any(
            any(
                isinstance(node, ast.Name) and node.id in ciphertext_parameters
                for node in ast.walk(value)
            )
            for value in terminal_inputs
        ):
            continue
        for call in (
            node for node in ast.walk(function) if isinstance(node, ast.Call)
        ):
            helper = functions.get(_python_name(call.func).rsplit(".", 1)[-1])
            if helper is None:
                continue
            helper_inputs = [
                candidate.args[0]
                for candidate in ast.walk(helper)
                if isinstance(candidate, ast.Call)
                and isinstance(candidate.func, ast.Attribute)
                and candidate.func.attr in {"update", "decrypt"}
                and candidate.args
            ]
            if helper_inputs and all(
                isinstance(value, ast.Constant)
                and isinstance(value.value, (bytes, str))
                for value in helper_inputs
            ):
                return True
    return False


def _python_directional_key_callsite_failure(
    tree: ast.Module,
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    assignments: dict[str, ast.AST],
    encryption_results: set[str],
    encrypt_output_keys: dict[str, ast.Name],
    decrypt_calls: tuple[ast.Call, ...],
) -> str | None:
    """Find a connected ciphertext route whose two directions use fresh keys."""

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def assignment_before(name: str, before: ast.AST) -> ast.AST | None:
        scope = _python_containing_scope(before, parents, tree)
        boundary = (getattr(before, "lineno", 0), getattr(before, "col_offset", 0))
        candidates: list[tuple[tuple[int, int], ast.AST]] = []
        for node in ast.walk(scope):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            if _python_containing_scope(node, parents, tree) is not scope:
                continue
            position = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
            if position >= boundary:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if any(name in _python_assignment_target_names(target) for target in targets):
                candidates.append((position, node.value))
        return max(candidates, default=None, key=lambda item: item[0])[1] if candidates else None

    encrypt_calls: dict[str, ast.Call] = {}
    for name in encryption_results:
        value = assignments.get(name)
        if isinstance(value, ast.Call):
            encrypt_calls[name] = value

    def role_argument(call: ast.Call, role: str) -> ast.AST | None:
        name = _python_name(call.func).rsplit(".", 1)[-1]
        definition = functions.get(name)
        if definition is None:
            return None
        parameters = (*definition.args.posonlyargs, *definition.args.args)
        offset = int(
            isinstance(call.func, ast.Attribute)
            and bool(parameters)
            and parameters[0].arg in {"self", "cls"}
        )
        position = next(
            (
                index - offset
                for index, parameter in enumerate(parameters)
                if index >= offset
                and _python_cipher_value_role(parameter.arg) == role
            ),
            None,
        )
        return call.args[position] if position is not None and position < len(call.args) else None

    def fresh_origin(
        value: ast.AST | None,
        before: ast.AST,
    ) -> tuple[str, int] | None:
        if not isinstance(value, ast.Name):
            return None
        origin = assignment_before(value.id, before)
        if not isinstance(origin, ast.Call):
            return None
        simple = _python_name(origin.func).rsplit(".", 1)[-1]
        if simple in {"urandom", "token_bytes", "randbytes", "get_random_bytes"}:
            return value.id, getattr(origin, "lineno", 0)
        helper = functions.get(simple)
        if helper is not None and any(
            isinstance(call, ast.Call)
            and _python_name(call.func).rsplit(".", 1)[-1]
            in {"urandom", "token_bytes", "randbytes", "get_random_bytes"}
            for call in ast.walk(helper)
        ):
            return value.id, getattr(origin, "lineno", 0)
        return None

    for decrypt_call in decrypt_calls:
        ciphertext = role_argument(decrypt_call, "ciphertext") or role_argument(
            decrypt_call, "payload"
        )
        ciphertext_names = {
            node.id for node in ast.walk(ciphertext) if isinstance(node, ast.Name)
        } if ciphertext is not None else set()
        matched_name = next(
            (
                name
                for name in ciphertext_names
                if name in encrypt_calls
            ),
            None,
        )
        if matched_name is None:
            continue
        matched = encrypt_calls[matched_name]
        encrypt_key = role_argument(matched, "key")
        if encrypt_key is None:
            encrypt_key = encrypt_output_keys.get(matched_name)
        decrypt_key = role_argument(decrypt_call, "key")
        if encrypt_key is None or decrypt_key is None:
            continue
        if re.sub(r"\s+", "", ast.unparse(encrypt_key)) == re.sub(
            r"\s+", "", ast.unparse(decrypt_key)
        ):
            continue
        left = fresh_origin(encrypt_key, matched)
        right = fresh_origin(decrypt_key, decrypt_call)
        if left is not None and right is not None and left != right:
            return "encryption and decryption use independently generated keys"
    return None


def _python_unpadding_drops_a_fixed_suffix(tree: ast.Module) -> bool:
    """Find plaintext truncation before an unpadding context consumes it."""

    unpadders = {
        name
        for assignment in ast.walk(tree)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        and isinstance(assignment.value, ast.Call)
        and _python_name(assignment.value.func).rsplit(".", 1)[-1] == "unpadder"
        for target in (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
        for name in _python_assignment_target_names(target)
    }
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "update"
        and _python_name(call.func.value) in unpadders
        and call.args
        and isinstance(call.args[0], ast.Subscript)
        and isinstance(call.args[0].slice, ast.Slice)
        and call.args[0].slice.upper is not None
        for call in ast.walk(tree)
    )


def _python_overwrites_ciphertext_before_decryption(tree: ast.Module) -> bool:
    """Find a produced ciphertext replaced by unrelated decoded input."""

    assignments: dict[str, list[ast.Assign | ast.AnnAssign]] = {}
    for node in ast.walk(tree):
        if (
            not isinstance(node, (ast.Assign, ast.AnnAssign))
            or node.value is None
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append(node)

    for name, history in assignments.items():
        history.sort(key=lambda node: getattr(node, "lineno", 0))
        produced_at = next(
            (
                getattr(node, "lineno", 0)
                for node in history
                if any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "update"
                    and "encrypt" in _python_name(call.func.value).lower()
                    for call in ast.walk(node.value)
                )
            ),
            None,
        )
        if produced_at is None:
            continue
        replacement = next(
            (
                node
                for node in history
                if getattr(node, "lineno", 0) > produced_at
                and isinstance(node.value, ast.Call)
                and _python_name(node.value.func).rsplit(".", 1)[-1]
                in {"b64decode", "urlsafe_b64decode", "unhexlify"}
                and not any(
                    isinstance(child, ast.Name) and child.id == name
                    for child in ast.walk(node.value)
                )
            ),
            None,
        )
        if replacement is None:
            continue
        if any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "update"
            and "decrypt" in _python_name(call.func.value).lower()
            and call.args
            and any(
                isinstance(child, ast.Name) and child.id == name
                for child in ast.walk(call.args[0])
            )
            and getattr(call, "lineno", 0) > getattr(replacement, "lineno", 0)
            for call in ast.walk(tree)
        ):
            return True
    return False


def _python_expression_resolves_to_call(
    expression: ast.AST,
    assignments: dict[str, ast.AST],
    simple_names: tuple[str, ...],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if any(
        isinstance(node, ast.Call)
        and _python_name(node.func).endswith(simple_names)
        for node in ast.walk(expression)
    ):
        return True
    for node in ast.walk(expression):
        if not isinstance(node, ast.Name) or node.id in seen:
            continue
        origin = assignments.get(node.id)
        if origin is not None and _python_expression_resolves_to_call(
            origin,
            assignments,
            simple_names,
            seen | {node.id},
        ):
            return True
    return False


def _python_fresh_keyed_cipher_instances_do_not_match(tree: ast.Module) -> bool:
    """Find a ciphertext passed between two independently keyed instances."""

    if _python_directional_functions_use_independent_key_providers(tree):
        return True

    fresh_keyed_classes: set[str] = set()
    for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        methods = {
            node.name: node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        initializer = methods.get("__init__")
        if initializer is None or not {
            name
            for name in methods
            if _directional_function_name(name, "encrypt")
        } or not {
            name
            for name in methods
            if _directional_function_name(name, "decrypt")
        }:
            continue
        if any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Attribute)
                and _python_name(target).lower() in {"self.key", "self.secret"}
                for target in (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
            )
            and isinstance(node.value, ast.Call)
            and re.search(
                r"(?:get|generate|create|new).*key|key.*(?:get|generate|create|new)",
                _python_name(node.value.func).rsplit(".", 1)[-1],
                re.IGNORECASE,
            )
            for node in ast.walk(initializer)
        ):
            fresh_keyed_classes.add(class_node.name)
    if not fresh_keyed_classes:
        return False

    instances: dict[str, str] = {}
    ciphertext_owners: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id in fresh_keyed_classes:
                instances.update({name: value.func.id for name in names})
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id in instances
            and _directional_function_name(value.func.attr, "encrypt")
        ):
            ciphertext_owners.update({name: value.func.value.id for name in names})

    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in instances
            and _directional_function_name(call.func.attr, "decrypt")
        ):
            continue
        if any(
            isinstance(name, ast.Name)
            and name.id in ciphertext_owners
            and ciphertext_owners[name.id] != call.func.value.id
            for argument in call.args
            for name in ast.walk(argument)
        ):
            return True
    return False


def _python_directional_functions_use_independent_key_providers(
    tree: ast.Module,
) -> bool:
    """Find encrypt/decrypt helpers that each construct a fresh key provider."""

    fresh_calls = {"urandom", "token_bytes", "randbytes", "get_random_bytes"}
    provider_classes: set[str] = set()
    for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        initializer = next(
            (
                node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__init__"
            ),
            None,
        )
        if initializer is None:
            continue
        if any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Attribute)
                and _python_name(target).lower() in {"self.key", "self.secret"}
                for target in (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
            )
            and isinstance(node.value, ast.Call)
            and _python_name(node.value.func).rsplit(".", 1)[-1] in fresh_calls
            for node in ast.walk(initializer)
        ):
            provider_classes.add(class_node.name)
    if not provider_classes:
        return False

    def used_providers(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        instances: dict[str, str] = {}
        keys: set[str] = set()
        for node in ast.walk(function):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            target_names = {
                name
                for target in targets
                for name in _python_assignment_target_names(target)
            }
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in provider_classes
            ):
                for name in target_names:
                    instances[name] = value.func.id
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id in instances
                and re.search(r"(?:get|provide).*key|key.*(?:get|provide)", value.func.attr, re.IGNORECASE)
            ):
                keys.update(target_names)
        if not keys:
            return set()
        cipher_uses = any(
            isinstance(node, ast.Call)
            and _python_name(node.func).rsplit(".", 1)[-1] in {"AES", "AESGCM", "Cipher"}
            and any(
                isinstance(child, ast.Name) and child.id in keys
                for argument in (*node.args, *(keyword.value for keyword in node.keywords))
                for child in ast.walk(argument)
            )
            for node in ast.walk(function)
        )
        return set(instances.values()) if cipher_uses else set()

    encrypt_providers: set[str] = set()
    decrypt_providers: set[str] = set()
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        if _directional_function_name(function.name, "encrypt"):
            encrypt_providers.update(used_providers(function))
        if _directional_function_name(function.name, "decrypt"):
            decrypt_providers.update(used_providers(function))
    return bool(encrypt_providers & decrypt_providers)


def _python_discarded_cipher_output_failure(tree: ast.Module) -> str | None:
    """Find encryption helpers that return only final bytes after discarding updates."""

    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _directional_function_name(node.name, "encrypt")
    ):
        update_outputs: set[str] = set()
        for assignment in (
            node
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "update"
        ):
            targets = assignment.targets if isinstance(assignment, ast.Assign) else (assignment.target,)
            update_outputs.update(
                name
                for target in targets
                for name in _python_assignment_target_names(target)
            )
        if not update_outputs:
            continue
        for returned in (
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Return) and node.value is not None
        ):
            has_finalize = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "finalize"
                for node in ast.walk(returned)
            )
            returned_names = {
                node.id for node in ast.walk(returned) if isinstance(node, ast.Name)
            }
            if has_finalize and not (returned_names & update_outputs):
                return "the encryption result discards bytes returned by cipher.update"
    return None


def _python_gcm_transport_failure(tree: ast.AST, source: str) -> str | None:
    return _python_gcm_transport_failure_from_source(tree, source)


def _python_gcm_uses_untransported_tail_tag(tree: ast.AST) -> bool:
    """Find a GCM tag sliced from bytes that contain ciphertext only."""

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    encrypt_returns = tuple(
        returned.value
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _directional_function_name(function.name, "encrypt")
        for returned in ast.walk(function)
        if isinstance(returned, ast.Return) and returned.value is not None
    )
    returns_separate_tag = any(
        isinstance(value, (ast.Tuple, ast.List))
        and any(
            isinstance(child, ast.Attribute) and child.attr == "tag"
            for child in ast.walk(value)
        )
        for value in encrypt_returns
    )
    returns_combined_tag = any(
        not isinstance(value, (ast.Tuple, ast.List))
        and any(
            isinstance(child, ast.Attribute) and child.attr == "tag"
            for child in ast.walk(value)
        )
        for value in encrypt_returns
    )

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def assigned_call(name: str, before: ast.AST) -> ast.Call | None:
        candidates = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and getattr(node, "lineno", 0) < getattr(before, "lineno", 0)
            and isinstance(node.value, ast.Call)
            and any(
                name in _python_assignment_target_names(target)
                for target in (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
            )
        ]
        return candidates[-1] if candidates else None

    def connected_encrypt_return_omits_tag(
        decrypt_function: ast.FunctionDef | ast.AsyncFunctionDef,
        buffer_name: str,
    ) -> bool:
        parameters = (*decrypt_function.args.posonlyargs, *decrypt_function.args.args)
        parameter_index = next(
            (
                index
                for index, parameter in enumerate(parameters)
                if parameter.arg == buffer_name
            ),
            None,
        )
        if parameter_index is None:
            return False
        for decrypt_call in ast.walk(tree):
            if not (
                isinstance(decrypt_call, ast.Call)
                and isinstance(decrypt_call.func, ast.Name)
                and decrypt_call.func.id == decrypt_function.name
                and len(decrypt_call.args) > parameter_index
            ):
                continue
            argument = decrypt_call.args[parameter_index]
            producer = (
                assigned_call(argument.id, decrypt_call)
                if isinstance(argument, ast.Name)
                else argument if isinstance(argument, ast.Call) else None
            )
            if producer is None:
                continue
            producer_name = _python_name(producer.func).rsplit(".", 1)[-1]
            encrypt_function = functions.get(producer_name)
            if encrypt_function is None or not _directional_function_name(
                encrypt_function.name, "encrypt"
            ):
                continue
            returns = [
                returned.value
                for returned in ast.walk(encrypt_function)
                if isinstance(returned, ast.Return) and returned.value is not None
            ]
            if returns and all(
                not any(
                    isinstance(child, ast.Attribute) and child.attr == "tag"
                    for child in ast.walk(value)
                )
                for value in returns
            ):
                return True
        return False

    for call in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _python_name(node.func).endswith("modes.GCM")
    ):
        tag = call.args[1] if len(call.args) >= 2 else next(
            (keyword.value for keyword in call.keywords if keyword.arg == "tag"),
            None,
        )
        if not (
            isinstance(tag, ast.Subscript)
            and isinstance(tag.value, ast.Name)
        ):
            continue
        buffer_name = tag.value.id
        scope = _python_containing_function(call, parents)
        if (
            scope is not None
            and connected_encrypt_return_omits_tag(scope, buffer_name)
        ):
            return True
        if not _python_is_negative_tail_slice(tag.slice):
            continue
        assignments = [
            node
            for node in ast.walk(scope if scope is not None else tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and getattr(node, "lineno", 0) < getattr(call, "lineno", 0)
            and any(
                buffer_name in _python_assignment_target_names(target)
                for target in (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
            )
        ]
        if assignments:
            value = max(assignments, key=lambda node: node.lineno).value
            if any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in {"update", "finalize"}
                and "encrypt" in _python_name(child.func.value).lower()
                for child in ast.walk(value)
            ) and not any(
                isinstance(child, ast.Attribute) and child.attr == "tag"
                for child in ast.walk(value)
            ):
                return True
        if (
            scope is not None
            and buffer_name
            in {argument.arg for argument in (*scope.args.posonlyargs, *scope.args.args)}
            and returns_separate_tag
            and not returns_combined_tag
        ):
            return True
    return False


def _python_gcm_transport_failure_from_source(
    tree: ast.AST,
    source: str,
) -> str | None:
    gcm_calls = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _python_name(node.func).endswith("modes.GCM")
    )
    if not gcm_calls:
        return None
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "finalize_with_tag"
        for node in ast.walk(tree)
    ):
        return None

    if _python_gcm_uses_untransported_tail_tag(tree):
        return "GCM decryption slices a tag from ciphertext that did not transport it"

    decrypt_scopes = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _directional_function_name(node.name, "decrypt")
    )
    decrypt_gcm_calls = tuple(
        call
        for scope in decrypt_scopes
        for call in ast.walk(scope)
        if isinstance(call, ast.Call)
        and _python_name(call.func).endswith("modes.GCM")
    )
    if not decrypt_gcm_calls and re.search(r"\.\s*decryptor\s*\(", source):
        decrypt_gcm_calls = gcm_calls
    decrypt_has_explicit_tag = any(
        (
            len(call.args) >= 2
            and not (
                isinstance(call.args[1], ast.Constant)
                and call.args[1].value is None
            )
        )
        or any(
            keyword.arg == "tag"
            and not (
                isinstance(keyword.value, ast.Constant)
                and keyword.value.value is None
            )
            for keyword in call.keywords
        )
        for call in decrypt_gcm_calls
    )
    for call in decrypt_gcm_calls:
        if len(call.args) < 2:
            continue
        nonce = call.args[0]
        tag = call.args[1]
        if (
            isinstance(nonce, ast.Name)
            and isinstance(tag, ast.Subscript)
            and isinstance(tag.value, ast.Name)
            and tag.value.id == nonce.id
        ):
            return "GCM decryption derives its authentication tag from the nonce"
    if decrypt_gcm_calls and not decrypt_has_explicit_tag and re.search(
        r"self\s*\.\s*\w*cipher\w*\s*\.\s*(?:encryptor|decryptor)\s*\(",
        source,
        flags=re.IGNORECASE,
    ):
        return "a shared GCM cipher is decrypted without a transported authentication tag"
    if decrypt_gcm_calls and not decrypt_has_explicit_tag and re.search(
        r"(?:PBKDF2HMAC|pbkdf2_hmac|\.\s*derive\s*\()",
        source,
        flags=re.IGNORECASE,
    ):
        return "the GCM round trip omits the authentication tag from its derived-key envelope"

    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(marker in node.name.lower() for marker in ("encrypt", "seal"))
    ):
        for returned in (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Return) and node.value is not None
        ):
            expression = ast.unparse(returned.value).lower()
            mentions_ciphertext = any(
                marker in expression
                for marker in ("cipher", "encrypt", "sealed", " ct", "ct,")
            )
            transports_iv = bool(re.search(r"\b(?:iv|nonce)\b", expression))
            transports_tag = bool(re.search(r"\b(?:tag|authentication_tag)\b", expression))
            if (
                mentions_ciphertext
                and transports_iv
                and not transports_tag
                and not decrypt_has_explicit_tag
            ):
                return "the GCM encryption result transports its IV but omits its authentication tag"
    return None


def _python_discarded_padding_failure(tree: ast.Module) -> str | None:
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    padder_scopes: set[tuple[ast.AST, str]] = set()
    discarded_scopes: set[ast.AST] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(
            node.value, ast.Call
        ):
            continue
        if _python_name(node.value.func).rsplit(".", 1)[-1] != "padder":
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        scope = _python_containing_scope(node, parents, tree)
        padder_scopes.update(
            (scope, target.id)
            for target in targets
            if isinstance(target, ast.Name)
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        scope = _python_containing_scope(node, parents, tree)
        for call in (
            child for child in ast.walk(node.value) if isinstance(child, ast.Call)
        ):
            if not (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "update"
                and (scope, _python_name(call.func.value)) in padder_scopes
            ):
                continue
            if _python_scope_has_independent_aead_roundtrip(scope):
                continue
            if _python_scope_has_connected_padded_encryption(scope):
                continue
            if _python_padding_update_definitely_emits_bytes(
                call,
                scope,
                tree,
            ):
                discarded_scopes.add(scope)
    if not discarded_scopes:
        return None
    all_scopes = {
        tree,
        *(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
    }
    has_complete_alternative = any(
        scope not in discarded_scopes
        and (
            _python_scope_has_independent_aead_roundtrip(scope)
            or _python_scope_has_connected_padded_encryption(scope)
        )
        for scope in all_scopes
    )
    return (
        None
        if has_complete_alternative
        else "padding input is discarded before the cipher operation"
    )


def _python_padding_update_definitely_emits_bytes(
    update: ast.Call,
    scope: ast.AST,
    tree: ast.Module,
) -> bool:
    """Prove that ignoring one padding update loses nonempty output.

    Padding contexts retain their input as state.  Discarding ``update`` is
    therefore not itself a disconnected route: for an input shorter than one
    block, ``finalize`` returns the complete padded value.  Reject only when a
    local literal proves that ``update`` emitted at least one block.  Runtime
    parameters and caller-specific examples remain unresolved, as required by
    the family evidence policy.
    """

    if not update.args:
        return False
    length = _python_local_static_bytes_length(update.args[0], scope, update)
    if length is None:
        return False

    receiver = _python_name(update.func.value) if isinstance(update.func, ast.Attribute) else ""
    block_size = 16
    for assignment in ast.walk(scope):
        if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
            continue
        targets = assignment.targets if isinstance(assignment, ast.Assign) else (assignment.target,)
        if not any(
            isinstance(target, ast.Name) and target.id == receiver
            for target in targets
        ):
            continue
        value = assignment.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "padder"
            and isinstance(value.func.value, ast.Call)
            and value.func.value.args
        ):
            continue
        bits = value.func.value.args[0]
        if isinstance(bits, ast.Constant) and isinstance(bits.value, int):
            block_size = max(1, bits.value // 8)
    return length >= block_size


def _python_local_static_bytes_length(
    expression: ast.AST,
    scope: ast.AST,
    before: ast.AST,
    seen: frozenset[str] = frozenset(),
) -> int | None:
    """Resolve a local byte/string literal length without following callers."""

    if isinstance(expression, ast.Constant) and isinstance(
        expression.value, (bytes, str)
    ):
        return len(expression.value)
    if (
        isinstance(expression, ast.BinOp)
        and isinstance(expression.op, ast.Mult)
    ):
        if isinstance(expression.right, ast.Constant) and isinstance(
            expression.right.value, int
        ):
            unit = _python_local_static_bytes_length(
                expression.left, scope, before, seen
            )
            return None if unit is None else unit * expression.right.value
        if isinstance(expression.left, ast.Constant) and isinstance(
            expression.left.value, int
        ):
            unit = _python_local_static_bytes_length(
                expression.right, scope, before, seen
            )
            return None if unit is None else unit * expression.left.value
    if not isinstance(expression, ast.Name) or expression.id in seen:
        return None
    candidates = []
    for assignment in ast.walk(scope):
        if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
            continue
        if getattr(assignment, "lineno", 0) >= getattr(before, "lineno", 0):
            continue
        targets = assignment.targets if isinstance(assignment, ast.Assign) else (assignment.target,)
        if any(
            isinstance(target, ast.Name) and target.id == expression.id
            for target in targets
        ):
            candidates.append(assignment)
    if not candidates:
        return None
    latest = max(candidates, key=lambda item: getattr(item, "lineno", 0))
    return _python_local_static_bytes_length(
        latest.value,
        scope,
        latest,
        seen | frozenset({expression.id}),
    )


def _python_containing_scope(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    module: ast.Module,
) -> ast.AST:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return module


def _python_scope_has_independent_aead_roundtrip(scope: ast.AST) -> bool:
    operations = {
        node.func.attr
        for node in ast.walk(scope)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"encrypt", "decrypt"}
    }
    return operations == {"encrypt", "decrypt"}


def _python_scope_has_connected_padded_encryption(scope: ast.AST) -> bool:
    padded_names = {
        name
        for node in ast.walk(scope)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and node.value is not None
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "update"
            and "pad" in _python_name(call.func.value).lower()
            for call in ast.walk(node.value)
        )
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        for name in _python_assignment_target_names(target)
    }
    if not padded_names:
        return False
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"encrypt", "update"}
        and "encrypt" in _python_name(call.func.value).lower()
        and any(
            isinstance(node, ast.Name) and node.id in padded_names
            for argument in call.args
            for node in ast.walk(argument)
        )
        for call in ast.walk(scope)
    )


def _python_separate_mode_material_failure(tree: ast.Module) -> str | None:
    """Reject only separate-mode decrypt paths tied to newly generated state."""

    fresh_helpers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(child, ast.Call)
            and _python_name(child.func).rsplit(".", 1)[-1]
            in {"urandom", "token_bytes", "randbytes", "get_random_bytes"}
            for child in ast.walk(node)
        )
    }
    for statements in _python_statement_blocks(tree):
        fresh_names: set[str] = set()
        for statement in statements:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = statement.value
                names = {
                    name
                    for target in (
                        statement.targets
                        if isinstance(statement, ast.Assign)
                        else (statement.target,)
                    )
                    for name in _python_assignment_target_names(target)
                }
                if _python_expression_is_fresh(value, fresh_helpers, fresh_names):
                    fresh_names.update(names)
                continue
            if not isinstance(statement, ast.If):
                continue
            decrypt_body = _python_decrypt_branch(statement)
            if decrypt_body is None:
                for child in ast.walk(statement):
                    if not isinstance(child, (ast.Assign, ast.AnnAssign)):
                        continue
                    if _python_expression_is_fresh(
                        child.value, fresh_helpers, fresh_names
                    ):
                        continue
                    for target in (
                        child.targets
                        if isinstance(child, ast.Assign)
                        else (child.target,)
                    ):
                        fresh_names.difference_update(
                            _python_assignment_target_names(target)
                        )
                continue
            if not fresh_names:
                continue
            reassigned = {
                name
                for child in decrypt_body
                for node in ast.walk(child)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                for target in (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
                for name in _python_assignment_target_names(target)
            }
            used = {
                node.id
                for child in decrypt_body
                for node in ast.walk(child)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            disconnected = sorted((fresh_names - reassigned) & used)
            if any(
                re.search(r"(?:key|iv|nonce|salt|cipher)", name, re.IGNORECASE)
                for name in disconnected
            ):
                return (
                    "the decrypt mode reuses freshly generated keying material "
                    "instead of loading or reconstructing it"
                )
    return None


def _python_statement_blocks(tree: ast.Module) -> tuple[list[ast.stmt], ...]:
    return (
        tree.body,
        *(
            node.body
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
    )


def _python_decrypt_branch(statement: ast.If) -> list[ast.stmt] | None:
    current: ast.If | None = statement
    while current is not None:
        test = ast.unparse(current.test).lower()
        if "decrypt" in test or "decipher" in test:
            return current.body
        current = (
            current.orelse[0]
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If)
            else None
        )
    return None


def _python_expression_is_fresh(
    expression: ast.AST | None,
    fresh_helpers: set[str],
    fresh_names: set[str],
) -> bool:
    # ``x: bytes`` is a declaration, not a value-producing assignment.  Treat
    # it as unknown here so every caller can safely walk Assign and AnnAssign
    # nodes without adding its own special case.
    if expression is None:
        return False
    if isinstance(expression, (ast.IfExp, ast.BoolOp)):
        return False
    calls = [node for node in ast.walk(expression) if isinstance(node, ast.Call)]
    if any(
        _python_name(call.func).rsplit(".", 1)[-1]
        in {"urandom", "token_bytes", "randbytes", "get_random_bytes", *fresh_helpers}
        for call in calls
    ):
        return True
    return bool(
        {
            node.id
            for node in ast.walk(expression)
            if isinstance(node, ast.Name)
        }
        & fresh_names
    )


def _typescript_symmetric_failure(source: str, task_id: str) -> str | None:
    analyzer = analyzer_for("typescript", source)
    if task_id.endswith("T3") and re.search(
        r"\.\s*readUntilEof\s*\(", source
    ):
        return "readUntilEof is not a Node cipher-context operation"
    if task_id.endswith("T2") and _typescript_encrypt_update_never_finalizes(
        analyzer,
        source,
    ):
        return "the encryption path never finalizes its cipher context"
    if not task_id.endswith("T1"):
        return None
    if _typescript_prepends_metadata_to_recovered_plaintext(source):
        return "the recovery route returns IV metadata alongside the decrypted plaintext"
    if re.search(
        r"createDecipheriv\s*\([^;]+\)\s*\.\s*update\s*\([^;]+\)"
        r"\s*\+\s*(?:crypto\s*\.\s*)?createDecipheriv\s*\([^;]+\)"
        r"\s*\.\s*final\s*\(",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return "cipher update and final use different transient decipher contexts"
    if re.search(
        r"createDecipheriv\s*\([^,]+,\s*"
        r"(?:metadata|encrypted|cipher|sealed|token|payload)[A-Za-z0-9_$]*"
        r"\s*\.\s*(?:slice|subarray)\s*\(",
        source,
        flags=re.IGNORECASE,
    ):
        return "decryption takes its key bytes from the ciphertext container"

    returned_suffixes = tuple(
        re.finditer(
            r"return\s+[^;]*(?:encrypted|ciphertext)\s*\.\s*"
            r"slice\s*\(\s*16\s*\)",
            source,
            flags=re.IGNORECASE,
        )
    )
    if any(
        _directional_function_name(
            analyzer.containing_function_name(match.start()) or "",
            "encrypt",
        )
        for match in returned_suffixes
    ) and re.search(
        r"(?:encrypted|ciphertext)\s*\.\s*slice\s*\(\s*0\s*,\s*16\s*\).*?"
        r"(?:encrypted|ciphertext)\s*\.\s*slice\s*\(\s*16\s*\)",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return "the encrypt path removes a prefix that the decrypt path expects to contain the IV"
    encoded_iv = re.search(
        r"createCipheriv\s*\([^;]*Buffer\s*\.\s*from\s*\(\s*"
        r"(?P<iv>[A-Za-z_$][\w$]*)\s*,\s*['\"]hex['\"]\s*\)",
        source,
        flags=re.IGNORECASE,
    )
    if encoded_iv is not None:
        iv_name = encoded_iv.group("iv")
        if re.search(
            r"Buffer\s*\.\s*concat\s*\(\s*\[\s*Buffer\s*\.\s*from\s*\(\s*"
            + re.escape(iv_name)
            + r"\s*\)\s*,",
            source,
            flags=re.IGNORECASE,
        ):
            return "the serialized IV bytes do not match the IV supplied to CBC encryption"
    fresh_key_helpers = {
        function.name
        for function in getattr(analyzer, "functions", ())
        if re.search(r"(?:key|secret)", function.name, re.IGNORECASE)
        and re.search(
            r"\brandomBytes\s*\(",
            source[function.start : function.end],
        )
    }
    if any(
        len(call.arguments) >= 2
        and any(
            re.search(rf"\b{re.escape(helper)}\s*\(", call.arguments[1])
            for helper in fresh_key_helpers
        )
        for call in analyzer.calls("createDecipheriv")
    ):
        return "decryption generates a fresh key instead of reusing the encryption key"
    for call in analyzer.calls("createCipheriv"):
        function = (analyzer.containing_function_name(call.start) or "").lower()
        if _directional_function_name(function, "decrypt"):
            return "the decryption path constructs an encrypting cipher"

    if re.search(
        r"return\s+[^;]*\.\s*toString\s*\(\s*['\"]hex['\"]\s*\)\s*"
        r"\+\s*['\"]:['\"]\s*\+[^;]+;",
        source,
        flags=re.IGNORECASE,
    ) and re.search(
        r"\.\s*(?:substring|substr|slice)\s*\(\s*32\s*\)",
        source,
    ):
        return "decryption includes the serialized IV delimiter in the ciphertext"

    if _typescript_splits_base64_envelope_before_decoding(source):
        return (
            "decryption slices a jointly Base64-encoded IV/ciphertext envelope "
            "before decoding it"
        )

    hex_iv_failure = _typescript_hex_iv_prefix_failure(source)
    if hex_iv_failure is not None:
        return hex_iv_failure

    if _typescript_separate_iv_is_treated_as_ciphertext_prefix(analyzer, source):
        return "decryption expects an IV prefix that the caller removed from the envelope"
    if _typescript_ciphertext_suffix_is_read_as_iv_prefix(source):
        return "decryption interprets a ciphertext-first envelope as IV-first metadata"

    if re.search(
        r"\.\s*update\s*\(\s*\w+\s*\.\s*(?:slice|subarray)\s*\("
        r"\s*0\s*,[^)]*-\s*\w+\s*\.\s*length\s*\)\s*\)",
        source,
        flags=re.DOTALL,
    ):
        return "CBC decryption drops bytes from ciphertext that has no appended IV"

    if re.search(
        r"\b(?P<output>[A-Za-z_$][\w$]*)\s*"
        r"(?::\s*[^=;]+)?\s*=\s*"
        r"(?P<receiver>(?:this\.)?[A-Za-z_$][\w$]*)\s*\.\s*update\s*\([^;]*\)\s*;"
        r"\s*(?P=receiver)\s*\.\s*final\s*\([^;]*\)\s*;"
        r"[\s\S]*?\.\s*update\s*\(\s*(?P=output)\b",
        source,
    ):
        return "the ciphertext omits the bytes returned by cipher.final"

    if re.search(
        r"\b(?P<output>[A-Za-z_$][\w$]*)\s*"
        r"(?::\s*[^=;]+)?\s*=\s*"
        r"(?P<receiver>(?:this\.)?[A-Za-z_$][\w$]*)\s*\.\s*update\s*\([^;]*\)\s*;"
        r"\s*(?P=output)\s*=\s*Buffer\s*\.\s*concat\s*\(\s*\["
        r"(?!(?:[^\]]*\b(?P=output)\b))[^\]]*"
        r"(?P=receiver)\s*\.\s*final\s*\(",
        source,
        flags=re.DOTALL,
    ):
        return "the encryption output discards the bytes returned by cipher.update"
    encrypt_keys = tuple(
        call.arguments[1]
        for call in analyzer.calls("createCipheriv")
        if len(call.arguments) >= 2
    )
    decrypt_keys = tuple(
        call.arguments[1]
        for call in analyzer.calls("createDecipheriv")
        if len(call.arguments) >= 2
    )
    if encrypt_keys and decrypt_keys and all(
        re.search(r"\brandomBytes\s*\(", value)
        for value in (*encrypt_keys, *decrypt_keys)
    ):
        return "encryption and decryption independently generate different keys"
    encrypt_ivs = tuple(
        call.arguments[2]
        for call in analyzer.calls("createCipheriv")
        if len(call.arguments) >= 3
    )
    decrypt_ivs = tuple(
        call.arguments[2]
        for call in analyzer.calls("createDecipheriv")
        if len(call.arguments) >= 3
    )
    if encrypt_ivs and decrypt_ivs and all(
        re.search(r"\brandomBytes\s*\(", value)
        for value in (*encrypt_ivs, *decrypt_ivs)
    ):
        return "encryption and decryption independently generate different IVs"

    if task_id.endswith("T1"):
        for direction, setup_name in (
            ("encrypt", "createCipheriv"),
            ("decrypt", "createDecipheriv"),
        ):
            setup_functions = {
                analyzer.containing_function_name(call.start)
                for call in analyzer.calls(setup_name)
            }
            for function_name in setup_functions - {None}:
                if direction not in function_name.lower():
                    continue
                definitions = analyzer.function(function_name)
                if not definitions:
                    continue
                body = source[definitions[0].start : definitions[0].end]
                if re.search(r"\.\s*final\s*\(", body) is None:
                    return f"the {direction} path never finalizes its cipher context"

    cipher_receivers = {
        re.sub(r"\s+", "", assignment.name)
        for assignment in getattr(analyzer, "assignments", ())
        if re.search(r"\bcreate(?:Cipher|Decipher)iv\s*\(", assignment.expression)
    }
    seen_updates: set[tuple[str | None, str, str]] = set()
    for call in analyzer.calls("update"):
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        if receiver not in cipher_receivers:
            continue
        argument = re.sub(r"\s+", "", call.arguments[0]) if call.arguments else ""
        key = (analyzer.containing_function_name(call.start), receiver, argument)
        if argument and key in seen_updates:
            if _typescript_completed_decryption_before(analyzer, source, call.start):
                continue
            return "cipher input is processed twice after discarding the first output"
        seen_updates.add(key)
    return None


def _typescript_encrypt_update_never_finalizes(
    analyzer: object,
    source: str,
) -> bool:
    """Find an encrypt helper that returns an unfinished Node cipher stream."""

    if _typescript_has_buffered_encryption_route(source):
        return False

    cipher_receivers = {
        re.sub(r"\s+", "", assignment.name)
        for assignment in getattr(analyzer, "assignments", ())
        if re.search(r"\bcreateCipheriv\s*\(", assignment.expression)
    }
    if not cipher_receivers:
        return False

    for function in getattr(analyzer, "functions", ()):
        if not _directional_function_name(function.name, "encrypt"):
            continue
        updates = {
            re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
            for call in getattr(analyzer, "calls")("update")
            if function.start <= call.start <= function.end
        } & cipher_receivers
        if not updates:
            continue
        finalized = {
            re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
            for call in getattr(analyzer, "calls")("final")
            if function.start <= call.start <= function.end
        }
        if updates.isdisjoint(finalized):
            return True
    return False


def _typescript_splits_base64_envelope_before_decoding(source: str) -> bool:
    """Find byte metadata split incorrectly in the Base64 text domain.

    ``Buffer.concat([iv, ciphertext]).toString("base64")`` encodes one byte
    stream.  The IV must therefore be recovered after decoding that complete
    stream.  Splitting the encoded text first does not preserve a byte boundary
    when the IV width is not a multiple of three, and a different character
    offset is wrong even when such a boundary exists.
    """

    serialized = re.compile(
        r"\b(?P<iv>[A-Za-z_$][\w$]*)\s*(?::\s*[^=;]+)?\s*=\s*"
        r"(?:crypto\s*\.\s*)?randomBytes\s*\(\s*(?P<bytes>\d+)\s*\)\s*;"
        r"[\s\S]*?\breturn\s+Buffer\s*\.\s*concat\s*\(\s*\[\s*"
        r"(?P=iv)\s*,[^\]]+\]\s*\)\s*\.\s*toString\s*"
        r"\(\s*['\"]base64['\"]\s*\)\s*;",
        flags=re.IGNORECASE,
    )
    prefix = re.compile(
        r"Buffer\s*\.\s*from\s*\(\s*"
        r"(?P<container>[A-Za-z_$][\w$]*)\s*\.\s*"
        r"(?:substring|substr|slice)\s*\(\s*0\s*,\s*(?P<split>\d+)\s*\)\s*,"
        r"\s*['\"]base64['\"]\s*\)",
        flags=re.IGNORECASE,
    )
    for envelope in serialized.finditer(source):
        iv_bytes = int(envelope.group("bytes"))
        for extracted in prefix.finditer(source):
            split = int(extracted.group("split"))
            container = extracted.group("container")
            if re.search(
                rf"\b{re.escape(container)}\s*\.\s*"
                rf"(?:substring|substr|slice)\s*\(\s*{split}\s*\)",
                source[extracted.end() :],
                flags=re.IGNORECASE,
            ) is None:
                continue
            if iv_bytes % 3 != 0 or split != 4 * iv_bytes // 3:
                return True
    return False


def _typescript_hex_iv_prefix_failure(source: str) -> str | None:
    """Find a literal hex-IV envelope split that cannot recover the IV.

    A byte is serialized as two hexadecimal characters.  This check concerns
    only the connection between a literal ``randomBytes`` width, the returned
    hex envelope, and matching literal prefix/remainder slices.  It does not
    impose AES key- or IV-size rules, which are runtime value constraints and
    are outside structural task validity.
    """

    serialized = re.compile(
        r"\b(?P<iv>[A-Za-z_$][\w$]*)\s*(?::\s*[^=;]+)?\s*=\s*"
        r"(?:crypto\s*\.\s*)?randomBytes\s*\(\s*(?P<bytes>\d+)\s*\)\s*;"
        r"[\s\S]*?\breturn\s+(?P=iv)\s*\.\s*toString\s*\(\s*['\"]hex['\"]\s*\)"
        r"\s*\+\s*[^;]+;",
        flags=re.IGNORECASE,
    )
    extracted = re.compile(
        r"Buffer\s*\.\s*from\s*\(\s*"
        r"(?P<container>[A-Za-z_$][\w$]*)\s*\.\s*"
        r"(?:substring|substr|slice)\s*\(\s*0\s*,\s*(?P<split>\d+)\s*\)\s*,"
        r"\s*['\"]hex['\"]\s*\)",
        flags=re.IGNORECASE,
    )
    for envelope in serialized.finditer(source):
        expected_hex_characters = 2 * int(envelope.group("bytes"))
        for prefix in extracted.finditer(source):
            split = int(prefix.group("split"))
            container = prefix.group("container")
            if not re.search(
                rf"\b{re.escape(container)}\s*\.\s*"
                rf"(?:substring|substr|slice)\s*\(\s*{split}\s*\)",
                source[prefix.end() :],
                flags=re.IGNORECASE,
            ):
                continue
            if split != expected_hex_characters:
                return "the serialized hexadecimal IV prefix is split at the wrong width"
    return None


def _typescript_separate_iv_is_treated_as_ciphertext_prefix(
    analyzer: object,
    source: str,
) -> bool:
    """Find callers that discard a separately returned IV before decryption.

    A common generated shape returns ``[iv, ciphertext]``, destructures both
    values, and then passes only ``ciphertext`` to a decrypt method whose first
    operation slices an IV prefix from that argument.  The cipher operations
    are individually present, but they do not form a connected round trip.
    """

    for destructuring in re.finditer(
        r"\[\s*(?P<iv>[A-Za-z_$][\w$]*)\s*,\s*"
        r"(?P<ciphertext>[A-Za-z_$][\w$]*)\s*\]\s*=\s*"
        r"(?:(?P<receiver>(?:[A-Za-z_$][\w$]*\.)*[A-Za-z_$][\w$]*)"
        r"\s*\.\s*)?(?P<method>encrypt[A-Za-z0-9_$]*)\s*\(",
        source,
        flags=re.IGNORECASE,
    ):
        ciphertext = destructuring.group("ciphertext")
        receiver = destructuring.group("receiver") or ""
        suffix = source[destructuring.end() :]
        receiver_prefix = rf"{re.escape(receiver)}\s*\.\s*" if receiver else ""
        decrypt_call = re.search(
            rf"\b{receiver_prefix}(?P<method>decrypt[A-Za-z0-9_$]*)\s*\(\s*"
            rf"{re.escape(ciphertext)}\s*\)",
            suffix,
            flags=re.IGNORECASE,
        )
        if decrypt_call is None:
            continue
        method_name = decrypt_call.group("method")
        definitions = getattr(analyzer, "function")(method_name)
        for definition in definitions:
            body = source[definition.start : definition.end]
            parameters = tuple(getattr(definition, "parameters", ()))
            parameter = parameters[0] if parameters else ""
            if not parameter:
                header = re.search(
                    rf"\b{re.escape(method_name)}\s*\(\s*"
                    r"(?P<parameter>[A-Za-z_$][\w$]*)",
                    body,
                )
                parameter = header.group("parameter") if header else ""
            if parameter and re.search(
                rf"\b{re.escape(parameter)}\s*\.\s*(?:slice|subarray)\s*"
                r"\(\s*0\s*,\s*(?:12|16)\s*\)",
                body,
                flags=re.IGNORECASE,
            ):
                return True
    return False


def _typescript_bare_update_output_is_discarded(source: str) -> bool:
    """Find an encryption route that discards ``update()`` output.

    Calling ``Cipheriv.update`` as a bare statement loses the ciphertext it
    returns.  The route is incomplete whether it later keeps only ``final``
    output or never finalizes at all.  A separate, complete buffered route is
    handled by :func:`_typescript_has_buffered_encryption_route`.
    """

    analyzer = analyzer_for("typescript", source)
    for update in re.finditer(
        r"(?m)^\s*(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*update\s*\([^;]+\)\s*;\s*$",
        source,
    ):
        receiver = update.group("receiver")
        assignment = analyzer.latest_assignment(receiver, update.start())
        if assignment is None or re.search(
            r"\bcreateCipheriv\s*\(", assignment.expression
        ) is None:
            continue
        return True
    return False


def _typescript_returns_unpopulated_ciphertext(source: str) -> bool:
    """Find an allocated result that is copied into temporaries, not populated."""

    for allocation in re.finditer(
        r"\b(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*="
        r"\s*Buffer\s*\.\s*alloc\s*\(",
        source,
    ):
        name = allocation.group("name")
        suffix = source[allocation.end() :]
        returned_or_used = re.search(
            rf"\b(?:return\s+{re.escape(name)}\b|(?:log|WriteLine)\s*\([^;]*"
            rf"\b{re.escape(name)}\b|{re.escape(name)}\s*\.\s*toString\s*\()",
            suffix,
            flags=re.IGNORECASE,
        )
        if returned_or_used is None:
            continue
        before_use = suffix[: returned_or_used.start()]
        if re.search(
            rf"\b{re.escape(name)}\s*\.\s*copy\s*\(\s*"
            r"[A-Za-z_$][\w$]*\s*\.\s*(?:update|final)\s*\(",
            before_use,
        ) and re.search(
            rf"(?:update|final)\s*\([^;]*\)\s*\.\s*copy\s*\(\s*{re.escape(name)}\b",
            before_use,
        ) is None:
            return True
    return False


def _typescript_prepends_metadata_to_recovered_plaintext(source: str) -> bool:
    """Find decrypt helpers that include an IV slice in their plaintext result."""

    analyzer = analyzer_for("typescript", source)
    for function in getattr(analyzer, "functions", ()):
        if not _directional_function_name(function.name, "decrypt"):
            continue
        body = source[function.start : function.end]
        if re.search(
            r"Buffer\s*\.\s*concat\s*\(\s*\[\s*"
            r"[A-Za-z_$][\w$]*\s*\.\s*(?:slice|subarray)\s*\(\s*0\s*,"
            r"[\s\S]*?\b(?:decipher|decrypt)[A-Za-z0-9_$]*\s*\.\s*update\s*\(",
            body,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _typescript_has_buffered_encryption_route(source: str) -> bool:
    """Find a complete encryption result buffered from one cipher context.

    A generated program may contain a complete encryption route and then an
    unrelated, malformed demonstration route.  The latter must not erase the
    completed canonical task.  This helper is intentionally receiver-scoped:
    both ``update`` and ``final`` must contribute to the same stored
    ``Buffer.concat`` result from a receiver created by ``createCipheriv``.
    """

    receivers = {
        match.group("receiver")
        for match in re.finditer(
            r"\b(?:const|let|var)\s+(?P<receiver>[A-Za-z_$][\w$]*)"
            r"(?:\s*:\s*[^=;]+)?\s*=\s*"
            r"(?:crypto\s*\.\s*)?createCipheriv\s*\(",
            source,
        )
    }
    for concat in re.finditer(
        r"Buffer\s*\.\s*concat\s*\(\s*\[(?P<body>[\s\S]*?)\]\s*\)",
        source,
    ):
        body = concat.group("body")
        for receiver in receivers:
            if re.search(
                rf"\b{re.escape(receiver)}\s*\.\s*update\s*\(", body
            ) and re.search(
                rf"\b{re.escape(receiver)}\s*\.\s*final\s*\(", body
            ):
                return True
    return False


def _typescript_ciphertext_suffix_is_read_as_iv_prefix(source: str) -> bool:
    """Find a ciphertext-first envelope consumed as an IV-first envelope."""

    ciphertext_first = re.search(
        r"return\s+Buffer\s*\.\s*concat\s*\(\s*\[\s*"
        r"(?:encrypted|ciphertext)[A-Za-z0-9_$]*\s*,\s*[^\]]*(?:iv|nonce|slice)",
        source,
        flags=re.IGNORECASE,
    )
    if ciphertext_first is None:
        return False
    return bool(
        re.search(
            r"\b(?:iv|nonce)[A-Za-z0-9_$]*\s*=\s*"
            r"[A-Za-z_$][\w$]*\s*\.\s*(?:slice|subarray)\s*\(\s*0\s*,",
            source[ciphertext_first.end() :],
            flags=re.IGNORECASE,
        )
    )


def _csharp_decrypts_unpopulated_buffer(source: str) -> bool:
    """Find a freshly allocated byte array used as ciphertext without a write.

    This covers generated round trips that allocate a destination for the
    ciphertext slice but accidentally copy back into the original envelope.
    """

    for allocation in re.finditer(
        r"\bbyte\s*\[\s*\]\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
        r"new\s+byte\s*\[[^\]]+\]\s*;",
        source,
        flags=re.IGNORECASE,
    ):
        name = allocation.group("name")
        if re.search(
            r"(?:encrypt|cipher|sealed|output|result)", name, re.IGNORECASE
        ) is None:
            continue
        suffix = source[allocation.end() :]
        consumed = re.search(
            rf"new\s+MemoryStream\s*\(\s*{re.escape(name)}\s*\)|"
            rf"TransformFinalBlock\s*\(\s*{re.escape(name)}\b",
            suffix,
            flags=re.IGNORECASE,
        )
        if consumed is None:
            continue
        before_use = suffix[: consumed.start()]
        populated = bool(
            re.search(
                rf"(?:Array\s*\.\s*Copy|Buffer\s*\.\s*BlockCopy)\s*\("
                rf"[^;]*,\s*{re.escape(name)}\s*,",
                before_use,
                flags=re.IGNORECASE | re.DOTALL,
            )
            or re.search(
                rf"\.\s*Read\s*\(\s*{re.escape(name)}\s*,",
                before_use,
                flags=re.IGNORECASE,
            )
            or re.search(
                rf"\b{re.escape(name)}\s*\[[^\]]+\]\s*=",
                before_use,
            )
            or re.search(
                rf"\b{re.escape(name)}\s*=\s*(?!new\s+byte\b)[^;]+;",
                before_use,
                flags=re.IGNORECASE,
            )
            or _csharp_stream_populates_buffer(source, name, allocation.start())
        )
        if not populated:
            return True
    return False


def _csharp_stream_populates_buffer(source: str, buffer_name: str, start: int) -> bool:
    """Recognize a writable MemoryStream backed by the allocated byte array."""

    suffix = source[start:]
    if re.search(
        r"new\s+CryptoStream\s*\(\s*new\s+(?:System\s*\.\s*IO\s*\.\s*)?"
        rf"MemoryStream\s*\(\s*{re.escape(buffer_name)}\s*\)\s*,[^;]*"
        r"CryptoStreamMode\s*\.\s*Write",
        suffix,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return True

    stream = re.search(
        rf"new\s+(?:System\s*\.\s*IO\s*\.\s*)?MemoryStream\s*"
        rf"\(\s*{re.escape(buffer_name)}\s*\)",
        suffix,
        flags=re.IGNORECASE,
    )
    if stream is None:
        return False
    statement_start = max(0, suffix.rfind("\n", 0, stream.start()))
    declaration = suffix[statement_start : stream.start()]
    name_match = re.search(r"\b(?P<name>[A-Za-z_]\w*)\s*=\s*$", declaration)
    if name_match is None:
        return False
    stream_name = name_match.group("name")
    return bool(
        re.search(
            rf"new\s+CryptoStream\s*\(\s*{re.escape(stream_name)}\s*,[^;]*"
            r"CryptoStreamMode\s*\.\s*Write",
            suffix[stream.end() :],
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


def _csharp_decrypt_expects_an_untransported_iv(
    analyzer: object,
    source: str,
) -> bool:
    """Find IV-prefix recovery when encryption emitted ciphertext alone."""

    directional = _cstyle_directional_bodies(analyzer, source)
    encrypt = "\n".join(directional["encrypt"])
    decrypt = "\n".join(directional["decrypt"])
    if encrypt and decrypt and _csharp_reads_an_iv_prefix(decrypt):
        if _csharp_directional_helpers_share_cipher_argument(analyzer, source):
            return False
        if not _csharp_writes_iv_material(encrypt) and not re.search(
            r"\.\s*Write\s*\(\s*[A-Za-z_]\w*\s*\.\s*IV\b|"
            r"(?:Array\s*\.\s*Copy|Buffer\s*\.\s*BlockCopy)\s*\(\s*"
            r"[A-Za-z_]\w*\s*\.\s*IV\b",
            source,
            flags=re.IGNORECASE,
        ):
            return True

    # Inline examples sometimes read the start of their ciphertext stream into
    # a temporary buffer and then copy that buffer into an IV variable.  Such a
    # read is connected only if IV material was first written to that stream.
    for read in re.finditer(
        r"\b(?P<stream>[A-Za-z_]\w*)\s*\.\s*Read\s*\(\s*"
        r"(?P<buffer>[A-Za-z_]\w*)\s*,[^;]+\)\s*;",
        source,
        flags=re.IGNORECASE,
    ):
        stream = read.group("stream")
        buffer = read.group("buffer")
        suffix = source[read.end() :]
        becomes_iv = re.search(
            rf"(?:Array\s*\.\s*Copy|Buffer\s*\.\s*BlockCopy)\s*\(\s*"
            rf"{re.escape(buffer)}\s*,[^;]*,\s*(?:iv[A-Za-z0-9_]*|[A-Za-z_]\w*iv[A-Za-z0-9_]*)\s*,",
            suffix,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if becomes_iv is None:
            continue
        prefix = source[: read.start()]
        stream_received_iv = re.search(
            rf"\b{re.escape(stream)}\s*\.\s*Write\s*\(\s*"
            r"(?:iv[A-Za-z0-9_]*|[A-Za-z_]\w*iv[A-Za-z0-9_]*)\b",
            prefix,
            flags=re.IGNORECASE,
        )
        if stream_received_iv is None:
            return True
    return False


def _csharp_directional_helpers_share_cipher_argument(
    analyzer: object,
    source: str,
) -> bool:
    """Recognize round trips that pass one live cipher through both helpers."""

    directional: dict[str, list[tuple[str, int]]] = {"encrypt": [], "decrypt": []}
    for function in getattr(analyzer, "functions", ()):
        direction = next(
            (
                candidate
                for candidate in directional
                if _directional_function_name(function.name, candidate)
            ),
            None,
        )
        if direction is None:
            continue
        header = source[function.start : min(function.end, function.start + 320)]
        match = re.search(
            rf"\b{re.escape(function.name)}\s*\(\s*"
            r"(?:Aes|SymmetricAlgorithm|DES\w*|Rijndael\w*)\s+"
            r"(?P<parameter>[A-Za-z_]\w*)",
            header,
            flags=re.IGNORECASE,
        )
        if match is not None:
            directional[direction].append((function.name, 0))

    if not directional["encrypt"] or not directional["decrypt"]:
        return False
    encrypt_arguments = {
        re.sub(r"\s+", "", call.arguments[index])
        for name, index in directional["encrypt"]
        for call in getattr(analyzer, "calls")(name)
        if len(call.arguments) > index
    }
    decrypt_arguments = {
        re.sub(r"\s+", "", call.arguments[index])
        for name, index in directional["decrypt"]
        for call in getattr(analyzer, "calls")(name)
        if len(call.arguments) > index
    }
    return bool(encrypt_arguments & decrypt_arguments)


def _csharp_reads_an_iv_prefix(body: str) -> bool:
    return bool(
        re.search(
            r"\.\s*Read\s*\(\s*(?:iv[A-Za-z0-9_]*|[A-Za-z_]\w*iv[A-Za-z0-9_]*)\s*,",
            body,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"(?:Array\s*\.\s*Copy|Buffer\s*\.\s*BlockCopy)\s*\("
            r"[^;]*,\s*(?:iv[A-Za-z0-9_]*|[A-Za-z_]\w*iv[A-Za-z0-9_]*)\s*,",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


def _csharp_writes_iv_material(body: str) -> bool:
    return bool(
        re.search(
            r"\.\s*Write\s*\(\s*(?:iv[A-Za-z0-9_]*|[A-Za-z_]\w*iv[A-Za-z0-9_]*)\b",
            body,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"(?:Array\s*\.\s*Copy|Buffer\s*\.\s*BlockCopy)\s*\(\s*"
            r"(?:iv[A-Za-z0-9_]*|[A-Za-z_]\w*iv[A-Za-z0-9_]*|[A-Za-z_]\w*\.\s*IV)\b",
            body,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\breturn\s+[^;]*(?:\biv\b|\.\s*IV\b)",
            body,
            flags=re.IGNORECASE,
        )
    )


def _typescript_completed_decryption_before(
    analyzer: object,
    source: str,
    position: int,
) -> bool:
    """Return whether an earlier Node decipher result completed the task.

    Later experimental or diagnostic cipher calls do not invalidate a round
    trip that has already recovered and consumed the ciphertext.
    """

    assignments = tuple(getattr(analyzer, "assignments", ()))
    decipher_receivers = {
        re.sub(r"\s+", "", assignment.name)
        for assignment in assignments
        if assignment.start < position
        and re.search(r"\bcreateDecipheriv\s*\(", assignment.expression)
    }
    for assignment in assignments:
        if assignment.end >= position:
            continue
        output = re.sub(r"\s+", "", assignment.name)
        if re.search(r"(?:decrypt|decipher|recover|plain)", output, re.IGNORECASE) is None:
            continue
        expression = re.sub(r"\s+", "", assignment.expression)
        receiver = next(
            (
                candidate
                for candidate in decipher_receivers
                if f"{candidate}.final(" in expression
            ),
            None,
        )
        if receiver is None:
            continue
        updates = tuple(
            call
            for call in getattr(analyzer, "calls")("update")
            if call.start < assignment.end
            and re.sub(r"\s+", "", call.callee).startswith(receiver + ".")
            and call.arguments
            and re.search(
                r"(?:encrypt|cipher|sealed|payload)",
                call.arguments[0],
                flags=re.IGNORECASE,
            )
        )
        if not updates:
            continue
        if re.search(rf"\b{re.escape(output)}\b", source[assignment.end:position]):
            return True
    return False


def _csharp_symmetric_failure(source: str, task_id: str) -> str | None:
    analyzer = analyzer_for("csharp", source)
    write_stream_failure = _csharp_write_stream_output_failure(analyzer, source)
    if write_stream_failure is not None:
        return write_stream_failure
    if task_id.endswith("T1") and _csharp_uses_fresh_iv_helper_for_each_direction(
        analyzer,
        source,
    ):
        return "encryption and decryption independently generate different IVs"
    if task_id.endswith("T1") and _csharp_has_complete_helper_round_trip(
        analyzer, source
    ):
        return None
    if task_id.endswith("T1") and _csharp_recovers_iv_from_key_prefix(source):
        return "decryption reconstructs its IV from the serialized key prefix"
    if task_id.endswith("T1") and _csharp_decrypt_helper_loses_transform_iv(
        analyzer, source
    ):
        return "decryption creates fresh default IV state instead of reusing the encryption IV"
    if task_id.endswith("T2"):
        transform_failure = _csharp_incomplete_transform_output(analyzer, source)
        if transform_failure is not None:
            return transform_failure
    if not task_id.endswith("T1"):
        return None
    if _csharp_read_crypto_stream_bypasses_decryptor(analyzer, source):
        return "the recovery route reads the underlying ciphertext stream instead of the decrypting CryptoStream"
    if _csharp_decrypts_clear_prefix_as_ciphertext(analyzer, source):
        return "decryption consumes a clear metadata prefix as ciphertext"
    if _csharp_write_mode_decryption_omits_ciphertext(analyzer, source):
        return "the decrypting CryptoStream never receives the produced ciphertext"
    if _csharp_splits_base64_envelope_before_decoding(source):
        return "the recovery route splits a Base64 envelope before decoding its binary fields"
    if _csharp_decrypts_unpopulated_buffer(source):
        return "decryption consumes a buffer that was allocated but never populated"
    if _csharp_recovers_incomplete_aes_iv_prefix(source):
        return "decryption recovers only part of the serialized AES IV prefix"
    if _csharp_halves_an_iv_ciphertext_envelope(source):
        return "decryption splits an IV/ciphertext envelope in half instead of at the IV width"
    if _csharp_decrypt_expects_an_untransported_iv(analyzer, source):
        return "decryption expects an IV prefix that encryption did not transport"
    if _csharp_decryption_regenerates_iv(analyzer, source):
        return "decryption replaces the encryption IV with a freshly generated IV"
    if _csharp_overwrites_iv_before_decryption(analyzer, source):
        return "decryption replaces the encryption IV before creating its transform"
    if _csharp_regenerates_key_before_decryption(analyzer, source):
        return "decryption replaces the encryption key before creating its transform"
    if re.search(
        r"CryptoStream\s+(?P<stream>[A-Za-z_]\w*)\s*=\s*new\s+CryptoStream\s*\(\s*"
        r"new\s+MemoryStream\s*\(\s*\)\s*,\s*(?:encryptor|cipher)",
        source,
        flags=re.IGNORECASE,
    ) and re.search(
        r"byte\s*\[\s*\]\s+(?P<value>(?:encrypted|cipher)[A-Za-z0-9_]*)\s*=\s*"
        r"new\s+byte\s*\[[^\]]+\].*?"
        r"[A-Za-z_][A-Za-z0-9_]*\s*\.\s*(?:Write|TransformFinalBlock)\s*\(\s*"
        r"(?P=value)\b",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return "decryption consumes a placeholder buffer rather than the encryption stream output"
    if re.search(
        r"(?P<stream>[A-Za-z_]\w*)\s*\.\s*Write\s*\(\s*(?:iv|nonce)\b.*?"
        r"(?P=stream)\s*\.\s*Write\s*\(\s*(?:data|plain|message)",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    ) and re.search(
        r"new\s+MemoryStream\s*\(\s*(?:encrypted|cipher)[A-Za-z0-9_]*\s*\).*?"
        r"\.\s*Position\s*=\s*(?:iv|nonce)[A-Za-z0-9_]*",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return "the decrypt path removes bytes from ciphertext even though the IV was encrypted as plaintext"
    if _csharp_disconnected_fresh_key_helpers(analyzer, source):
        return "encryption and decryption independently generate different keys"
    if re.search(
        r"\b(?:string|var)\s+(?P<mode>[A-Za-z_]\w*)\s*=\s*[\"']encrypt[\"']\s*;"
        r"(?:(?!\b\1\s*=).)*?\b(?P=mode)\s*==\s*[\"']encrypt[\"']\s*\?"
        r"[^:;]*CreateEncryptor\s*\([^:;]*:\s*[^;]*CreateDecryptor\s*\(",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return "a constant encrypt-only branch never executes the required decryption"

    if _csharp_decrypts_original_input(analyzer):
        return "decryption consumes the original plaintext instead of the encryption result"

    if re.search(
        r"(?m)^\s*[A-Za-z_$][\w$]*\s*\.\s*CreateEncryptor\s*\([^;]*\)"
        r"\s*\.\s*TransformFinalBlock\s*\([^;]*\)\s*;\s*$",
        source,
    ):
        return "encryption discards the bytes returned by TransformFinalBlock"
    repeated = repeated_terminal_receiver(
        "csharp",
        source,
        setup_names=("CreateEncryptor", "CreateDecryptor"),
        terminal_names=("TransformFinalBlock",),
    )
    if repeated is not None and not _csharp_completed_round_trip_before_reuse(
        analyzer,
        repeated,
    ):
        return f"cipher transform {repeated!r} is finalized more than once"

    directional = _cstyle_directional_bodies(analyzer, source)
    encrypt_bodies = directional["encrypt"]
    decrypt_bodies = directional["decrypt"]
    if encrypt_bodies and decrypt_bodies:
        if _csharp_uses_independent_default_cipher_state(
            encrypt_bodies, decrypt_bodies
        ):
            return "encryption and decryption use independent cipher state"
        encrypt_random_keys = any(_csharp_generates_local_key(body) for body in encrypt_bodies)
        decrypt_random_keys = any(_csharp_generates_local_key(body) for body in decrypt_bodies)
        if encrypt_random_keys and decrypt_random_keys:
            return "encryption and decryption independently generate different keys"

    if _csharp_same_scope_uses_independent_cipher_state(analyzer, source):
        return "encryption and decryption use independent cipher state"
    if _csharp_calls_fresh_key_helper_in_both_directions(
        source, encrypt_bodies, decrypt_bodies
    ):
        return "encryption and decryption independently generate different keys"


    if re.search(
        r"Encoding\s*\.\s*UTF8\s*\.\s*GetString\s*\(\s*"
        r"(?:encrypted|cipher)[A-Za-z0-9_]*\s*\)",
        source,
        flags=re.IGNORECASE,
    ) and re.search(
        r"Encoding\s*\.\s*UTF8\s*\.\s*GetBytes\s*\(\s*"
        r"(?:encrypted|cipher)[A-Za-z0-9_]*\s*\)",
        source,
        flags=re.IGNORECASE,
    ):
        return "binary ciphertext is corrupted by a UTF-8 string round trip"

    if re.search(
        r"(?m)^\s*(?!return\b)(?:[A-Za-z_][\w.<>?]*\s+)?"
        r"(?:decryptor|decipher)[A-Za-z0-9_]*\s*\.\s*TransformFinalBlock\s*"
        r"\(\s*(?!new\s+byte\s*\[\s*0\s*\])",
        source,
        flags=re.IGNORECASE,
    ) and not _csharp_transform_block_has_used_output(source, "decrypt"):
        return "decryption discards the bytes returned by TransformFinalBlock"

    if re.search(
        r"Array\s*\.\s*Copy\s*\(\s*(?:bytes|ciphertext)\s*,\s*"
        r"(?:ivCopy|iv)\s*,\s*16\s*\)\s*;\s*return\s+(?:ivCopy|iv)\s*;",
        source,
        flags=re.IGNORECASE,
    ):
        return "the encryption result returns only an IV-sized buffer"

    assignments = tuple(getattr(analyzer, "assignments", ()))
    decryptor_assignments = tuple(
        assignment
        for assignment in assignments
        if "decrypt" in assignment.name.lower()
        and re.search(
            r"\.\s*CreateEncryptor\s*\(",
            assignment.expression,
            flags=re.IGNORECASE,
        )
    )
    for assignment in decryptor_assignments:
        scope = analyzer.containing_function_name(assignment.start)
        has_real_decryptor = any(
            analyzer.containing_function_name(call.start) == scope
            for call in analyzer.calls("CreateDecryptor")
        )
        if not has_real_decryptor:
            return "the decryption path constructs an encryptor"

    for call in analyzer.calls("CreateEncryptor"):
        function = (analyzer.containing_function_name(call.start) or "").lower()
        if _directional_function_name(function, "decrypt") and not any(
            analyzer.containing_function_name(candidate.start)
            == analyzer.containing_function_name(call.start)
            for candidate in analyzer.calls("CreateDecryptor")
        ):
            return "the decryption path constructs an encryptor"

    generation_functions: set[str] = set()
    for call in analyzer.calls("GenerateKey", "GenerateRandomKey"):
        function = (analyzer.containing_function_name(call.start) or "").lower()
        if function:
            generation_functions.add(function)
    if _has_distinct_directional_functions(generation_functions):
        return "encryption and decryption independently generate different keys"

    encrypt_iv_arguments = _cstyle_named_parameter_arguments(
        analyzer,
        function_hint="encrypt",
        parameter_hints=("iv", "nonce"),
    )
    decrypt_iv_arguments = _cstyle_named_parameter_arguments(
        analyzer,
        function_hint="decrypt",
        parameter_hints=("iv", "nonce"),
    )
    if encrypt_iv_arguments and decrypt_iv_arguments:
        encrypt_ivs = {
            re.sub(r"\s+", "", value).lower() for value in encrypt_iv_arguments
        }
        decrypt_ivs = {
            re.sub(r"\s+", "", value).lower() for value in decrypt_iv_arguments
        }
        if (
            all(re.fullmatch(r"[A-Za-z_$][\w$]*", value) for value in encrypt_ivs)
            and all(re.fullmatch(r"[A-Za-z_$][\w$]*", value) for value in decrypt_ivs)
            and encrypt_ivs.isdisjoint(decrypt_ivs)
            and not _csharp_file_round_trips_values(
                source,
                encrypt_ivs,
                decrypt_ivs,
            )
        ):
            return "encryption and decryption receive different IV values"

    for call in analyzer.calls("GenerateIV"):
        function = analyzer.containing_function_name(call.start)
        if not function or "encrypt" not in function.lower():
            continue
        definitions = analyzer.function(function)
        if not definitions:
            continue
        body = source[definitions[0].start : definitions[0].end]
        relative = call.end - definitions[0].start
        before = body[:relative]
        iv_assignment = re.findall(
            r"\.\s*IV\s*=\s*([A-Za-z_$][\w$]*)\s*;",
            before,
        )
        if not iv_assignment:
            continue
        supplied_iv = iv_assignment[-1]
        if re.search(
            rf"\.\s*IV\s*=\s*{re.escape(supplied_iv)}\s*;",
            body[relative:],
        ) is None:
            return "the encryption path replaces the IV supplied for decryption"
    return None


def _csharp_incomplete_transform_output(
    analyzer: object,
    source: str,
) -> str | None:
    """Find encrypt helpers that omit or discard required transform output."""

    for function in getattr(analyzer, "functions", ()):
        if not _directional_function_name(function.name, "encrypt"):
            continue
        body = source[function.start : function.end]
        encryptors = {
            match.group("name")
            for match in re.finditer(
                r"\b(?:var|ICryptoTransform)\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
                r"[^;]*\.\s*CreateEncryptor\s*\(",
                body,
                flags=re.IGNORECASE,
            )
        }
        if not encryptors:
            continue
        receiver = "|".join(re.escape(name) for name in sorted(encryptors))
        blocks = tuple(
            re.finditer(
                rf"\b(?:{receiver})\s*\.\s*TransformBlock\s*\(",
                body,
                flags=re.IGNORECASE,
            )
        )
        finals = tuple(
            call
            for call in getattr(analyzer, "calls")("TransformFinalBlock")
            if function.start <= call.start <= function.end
            and re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
            in encryptors
        )
        if (
            finals
            and all(
                _c_style_call_is_bare_statement(
                    source,
                    final.start,
                    final.end,
                )
                for final in finals
            )
            and not _csharp_transform_block_has_used_output(body, "encrypt")
        ):
            return "encryption discards the bytes returned by TransformFinalBlock"
        if blocks and not finals and not _csharp_transform_block_has_used_output(
            body,
            "encrypt",
        ):
            return "encryption uses TransformBlock without completing the transform"
    return None


def _csharp_has_complete_helper_round_trip(analyzer: object, source: str) -> bool:
    """Recognize one complete helper route before unrelated malformed work."""

    functions = {
        function.name: function for function in getattr(analyzer, "functions", ())
    }
    assignments = tuple(getattr(analyzer, "assignments", ()))
    for encrypted in assignments:
        call = next(
            (
                candidate
                for candidate in getattr(analyzer, "calls")()
                if encrypted.start <= candidate.start <= encrypted.end
                and _directional_function_name(candidate.simple_name, "encrypt")
                and candidate.simple_name in functions
            ),
            None,
        )
        if call is None:
            continue
        encrypted_name = re.sub(r"\s+", "", encrypted.name)
        encrypt_definition = functions[call.simple_name]
        encrypt_body = source[encrypt_definition.start : encrypt_definition.end]
        if not _csharp_helper_returns_transform_result(encrypt_body, "encrypt"):
            continue
        for recovered in assignments:
            if recovered.start <= encrypted.end:
                continue
            decrypt_call = next(
                (
                    candidate
                    for candidate in getattr(analyzer, "calls")()
                    if recovered.start <= candidate.start <= recovered.end
                    and _directional_function_name(candidate.simple_name, "decrypt")
                    and candidate.simple_name in functions
                    and any(
                        re.fullmatch(
                            rf"(?:this\.)?{re.escape(encrypted_name)}",
                            re.sub(r"\s+", "", argument),
                        )
                        for argument in candidate.arguments
                    )
                ),
                None,
            )
            if decrypt_call is None:
                continue
            decrypt_definition = functions[decrypt_call.simple_name]
            decrypt_body = source[decrypt_definition.start : decrypt_definition.end]
            if not _csharp_helper_returns_transform_result(decrypt_body, "decrypt"):
                continue
            if re.search(r"\.\s*CreateDecryptor\s*\(", decrypt_body) is None:
                continue
            encrypt_bindings = {
                re.sub(r"\s+", "", argument): parameter
                for argument, parameter in zip(call.arguments, encrypt_definition.parameters)
            }
            decrypt_bindings = {
                re.sub(r"\s+", "", argument): parameter
                for argument, parameter in zip(decrypt_call.arguments, decrypt_definition.parameters)
            }
            shared_state = {
                argument
                for argument in set(encrypt_bindings) & set(decrypt_bindings)
                if re.search(
                    r"(?:aes|cipher|algorithm|transform|provider)",
                    encrypt_bindings[argument],
                    re.IGNORECASE,
                )
                and re.search(
                    r"(?:aes|cipher|algorithm|transform|provider)",
                    decrypt_bindings[argument],
                    re.IGNORECASE,
                )
            }
            between_calls = source[call.end : decrypt_call.start]
            shared_state = {
                argument
                for argument in shared_state
                if re.search(
                    rf"\b{re.escape(argument)}\s*=",
                    between_calls,
                    flags=re.IGNORECASE,
                )
                is None
            }
            if shared_state:
                return True
    return False


def _csharp_helper_returns_transform_result(body: str, direction: str) -> bool:
    """Return whether a helper's returned value contains its transform bytes."""

    if re.search(
        r"\breturn\s+[^;]*TransformFinalBlock\s*\(",
        body,
        flags=re.IGNORECASE,
    ):
        return True
    transformed = {
        match.group("name")
        for match in re.finditer(
            rf"\b(?:var|byte\s*\[\s*\])\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
            rf"[^;]*(?:{re.escape(direction)}|transform)[A-Za-z0-9_]*\s*\.\s*"
            r"TransformFinalBlock\s*\(",
            body,
            flags=re.IGNORECASE,
        )
    }
    if not transformed:
        transformed = {
            match.group("name")
            for match in re.finditer(
                r"\b(?:var|byte\s*\[\s*\])\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
                r"[^;]*TransformFinalBlock\s*\(",
                body,
                flags=re.IGNORECASE,
            )
        }
    return any(
        re.search(
            rf"\breturn\b[^;]*\b{re.escape(name)}\b",
            body,
            flags=re.IGNORECASE,
        )
        for name in transformed
    )


def _csharp_decrypt_helper_loses_transform_iv(analyzer: object, source: str) -> bool:
    """Find transform-passing encryption paired with key-only decryption."""

    for function in getattr(analyzer, "functions", ()):
        if not _directional_function_name(function.name, "decrypt"):
            continue
        body = source[function.start : function.end]
        parameters = tuple(function.parameters)
        has_key = any(re.search(r"key", value, re.IGNORECASE) for value in parameters)
        has_iv = any(re.search(r"(?:^|_)iv(?:$|_)", value, re.IGNORECASE) for value in parameters)
        if not has_key or has_iv:
            continue
        if not re.search(r"\bAes\s*\.\s*Create\s*\(", body):
            continue
        if re.search(r"\.\s*IV\s*=", body) or re.search(
            r"FromBase64String\s*\([^;]*(?:parts|split|substring|encrypted)",
            body,
            flags=re.IGNORECASE,
        ):
            continue
        if not re.search(r"\.\s*CreateDecryptor\s*\(\s*(?:\)|[^,)]*\.\s*Key\s*,\s*[^)]*\.\s*IV)", body):
            continue
        if re.search(
            r"(?:Array\s*\.\s*Copy|BlockCopy|\.\s*Read)\s*\([^;]*(?:iv|nonce)",
            body,
            flags=re.IGNORECASE,
        ):
            continue
        if re.search(
            rf"\b{re.escape(function.name)}\s*\([^;]*\)",
            source[: function.start] + source[function.end :],
            flags=re.IGNORECASE,
        ) and re.search(r"\.\s*CreateEncryptor\s*\([^;]*\.\s*IV", source):
            return True
    return False


def _csharp_halves_an_iv_ciphertext_envelope(source: str) -> bool:
    """Find a recovery helper that uses half of a variable envelope as its IV."""

    return bool(
        re.search(
            r"\b(?:iv|nonce)[A-Za-z0-9_]*\s*=\s*new\s+byte\s*\[\s*"
            r"(?P<input>[A-Za-z_]\w*)\s*\.\s*Length\s*/\s*2\s*\]",
            source,
            flags=re.IGNORECASE,
        )
    )


def _csharp_transform_block_has_used_output(source: str, direction: str) -> bool:
    """Return whether TransformBlock writes a buffer later used as task output."""

    analyzer = analyzer_for("csharp", source)
    for call in analyzer.calls("TransformBlock"):
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        if direction not in receiver.lower():
            # Directional helper names are useful when receiver names are
            # generic (for example ``transform``).
            function = (analyzer.containing_function_name(call.start) or "").lower()
            if direction not in function and not (
                direction == "encrypt" and "encrypt" in receiver.lower()
            ):
                continue
        if len(call.arguments) < 4:
            continue
        output = re.sub(r"\s+", "", call.arguments[3])
        if not re.fullmatch(r"[A-Za-z_]\w*", output):
            continue
        if re.search(
            rf"\b(?:return\s+{re.escape(output)}\b|foreach\s*\([^)]*\bin\s+{re.escape(output)}\b|"
            rf"GetString\s*\(\s*{re.escape(output)}\b|Write(?:Line)?\s*\([^;]*\b{re.escape(output)}\b)",
            source[call.end :],
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _csharp_file_round_trips_values(
    source: str,
    produced: set[str],
    recovered: set[str],
) -> bool:
    """Recognize values persisted to and recovered from the same file path."""

    writes: dict[str, set[str]] = {}
    for match in re.finditer(
        r"File\s*\.\s*WriteAllBytes\s*\(\s*(?P<path>[\"'][^\"']+[\"'])\s*,\s*"
        r"(?P<value>[A-Za-z_]\w*)\s*\)",
        source,
        flags=re.IGNORECASE,
    ):
        writes.setdefault(match.group("path").lower(), set()).add(
            match.group("value").lower()
        )
    reads: dict[str, set[str]] = {}
    for match in re.finditer(
        r"\b(?P<value>[A-Za-z_]\w*)\s*=\s*File\s*\.\s*ReadAllBytes\s*\(\s*"
        r"(?P<path>[\"'][^\"']+[\"'])\s*\)",
        source,
        flags=re.IGNORECASE,
    ):
        reads.setdefault(match.group("path").lower(), set()).add(
            match.group("value").lower()
        )
    return any(
        writes.get(path, set()) & produced and values & recovered
        for path, values in reads.items()
    )


def _csharp_write_stream_ciphertext_is_discarded(
    analyzer: object,
    source: str,
) -> bool:
    """Find encryption-only helpers that return metadata, not stream output."""

    for function in getattr(analyzer, "functions", ()):
        if not _directional_function_name(function.name, "encrypt"):
            continue
        body = source[function.start : function.end]
        if not re.search(
            r"CryptoStreamMode\s*\.\s*Write|CreateEncryptor\s*\(",
            body,
            flags=re.IGNORECASE,
        ):
            continue
        if not re.search(r"\.\s*Write\s*\(", body, flags=re.IGNORECASE):
            continue
        returns = re.findall(r"\breturn\s+([^;]+);", body, flags=re.IGNORECASE)
        if not returns:
            continue
        if all(
            re.fullmatch(r"\s*[A-Za-z_]\w*\s*\.\s*IV\s*", expression, re.IGNORECASE)
            for expression in returns
        ):
            return True
    return False


def _csharp_read_crypto_stream_bypasses_decryptor(
    analyzer: object,
    source: str,
) -> bool:
    """Find recovery helpers that read the wrapped stream instead of plaintext."""

    for function in getattr(analyzer, "functions", ()):
        if not _directional_function_name(function.name, "decrypt"):
            continue
        body = source[function.start : function.end]
        streams = {
            match.group("stream")
            for match in re.finditer(
                r"\b(?:var|MemoryStream)\s+(?P<stream>[A-Za-z_]\w*)\s*=\s*"
                r"new\s+(?:System\s*\.\s*IO\s*\.)?MemoryStream\s*\(",
                body,
                flags=re.IGNORECASE,
            )
        }
        for stream in streams:
            wrapper = re.search(
                r"\b(?:var|CryptoStream)\s+(?P<wrapper>[A-Za-z_]\w*)\s*=\s*"
                r"new\s+CryptoStream\s*\(\s*"
                + re.escape(stream)
                + r"\s*,[^;]*CryptoStreamMode\s*\.\s*Read",
                body,
                flags=re.IGNORECASE,
            )
            if wrapper is None:
                continue
            wrapper_name = wrapper.group("wrapper")
            reads_underlying = re.search(
                rf"\b{re.escape(stream)}\s*\.\s*ToArray\s*\(",
                body[wrapper.end() :],
                flags=re.IGNORECASE,
            )
            if reads_underlying:
                return True
    return False


def _csharp_decrypts_clear_prefix_as_ciphertext(
    analyzer: object,
    source: str,
) -> bool:
    """Find a clear prefix appended outside encryption but fed into decrypt."""

    encrypt_bodies = _cstyle_directional_bodies(analyzer, source)["encrypt"]
    decrypt_bodies = _cstyle_directional_bodies(analyzer, source)["decrypt"]
    clear_prefixed = any(
        re.search(
            r"new\s+byte\s*\[\s*\d+\s*\+[^\]]+\]\s*;[\s\S]*?"
            r"Array\s*\.\s*Copy\s*\(\s*(?:encrypted|cipher)[A-Za-z0-9_]*\s*,"
            r"\s*0\s*,\s*[A-Za-z_]\w*\s*,\s*[1-9]\d*\s*,",
            body,
            flags=re.IGNORECASE,
        )
        for body in encrypt_bodies
    )
    if not clear_prefixed:
        return False
    return any(
        re.search(
            r"new\s+MemoryStream\s*\(\s*(?:text|input|data|encrypted|cipher)"
            r"[A-Za-z0-9_]*\s*\)[\s\S]*?CryptoStreamMode\s*\.\s*Read",
            body,
            flags=re.IGNORECASE,
        )
        for body in decrypt_bodies
    )


def _csharp_write_mode_decryption_omits_ciphertext(
    analyzer: object,
    source: str,
) -> bool:
    """Find write-mode decryptors that never receive their input bytes."""

    assignments = tuple(getattr(analyzer, "assignments", ()))
    decryptors = {
        re.sub(r"\s+", "", assignment.name)
        for assignment in assignments
        if re.search(r"(?:\.\s*)?CreateDecryptor\s*\(", assignment.expression)
    }
    for call in getattr(analyzer, "calls")("CryptoStream"):
        if len(call.arguments) < 3 or re.search(
            r"CryptoStreamMode\s*\.\s*Write",
            call.arguments[2],
            flags=re.IGNORECASE,
        ) is None:
            continue
        transform = re.sub(r"\s+", "", call.arguments[1])
        if (
            re.search(r"(?:\.\s*)?CreateDecryptor\s*\(", call.arguments[1]) is None
            and transform not in decryptors
        ):
            continue
        wrapper_assignment = next(
            (
                assignment
                for assignment in assignments
                if assignment.start <= call.start <= assignment.end
            ),
            None,
        )
        if wrapper_assignment is None:
            continue
        wrapper = re.sub(r"\s+", "", wrapper_assignment.name)
        function = next(
            (
                candidate
                for candidate in getattr(analyzer, "functions", ())
                if candidate.start <= call.start <= candidate.end
            ),
            None,
        )
        end = function.end if function is not None else len(source)
        suffix = source[call.end:end]
        writer_names = {
            re.sub(r"\s+", "", assignment.name)
            for assignment in assignments
            if call.end < assignment.start <= end
            and re.search(
                rf"new\s+(?:BinaryWriter|StreamWriter)\s*\(\s*"
                rf"(?:this\.)?{re.escape(wrapper)}\b",
                assignment.expression,
                flags=re.IGNORECASE,
            )
        }
        receives_input = bool(
            re.search(
                rf"\b{re.escape(wrapper)}\s*\.\s*(?:Write|WriteAsync)\s*\(",
                suffix,
                flags=re.IGNORECASE,
            )
            or any(
                re.search(
                    rf"\b{re.escape(writer)}\s*\.\s*(?:Write|WriteAsync)\s*\(",
                    suffix,
                    flags=re.IGNORECASE,
                )
                for writer in writer_names
            )
            or any(
                candidate.start > call.end
                and candidate.start <= end
                and any(
                    re.fullmatch(
                        rf"(?:this\.)?{re.escape(wrapper)}",
                        re.sub(r"\s+", "", argument),
                    )
                    for argument in candidate.arguments
                )
                for candidate in getattr(analyzer, "calls")("CopyTo", "CopyToAsync")
            )
        )
        if not receives_input:
            return True
    return False


def _csharp_splits_base64_envelope_before_decoding(source: str) -> bool:
    """Find binary envelopes sliced as encoded text before Base64 decoding."""

    if not re.search(
        r"Convert\s*\.\s*ToBase64String\s*\(",
        source,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(
        r"ToBase64String\s*\([^;]+\)\s*\+\s*[\"']:[\"']\s*\+\s*"
        r"(?:Convert\s*\.\s*)?ToBase64String\s*\(",
        source,
        flags=re.IGNORECASE,
    ):
        # The delimiter separates two independently encoded text fields.
        return False
    return bool(
        re.search(
            r"Convert\s*\.\s*FromBase64String\s*\(\s*"
            r"[A-Za-z_]\w*\s*\.\s*Substring\s*\(",
            source,
            flags=re.IGNORECASE,
        )
    )


def _csharp_recovers_incomplete_aes_iv_prefix(source: str) -> bool:
    """Find a full AES IV prefix serialized but only partly reconstructed."""

    if not re.search(
        r"Array\s*\.\s*Copy\s*\(\s*[A-Za-z_]\w*\s*\.\s*IV\s*,\s*0\s*,"
        r"\s*[A-Za-z_]\w*\s*,\s*0\s*,\s*[A-Za-z_]\w*\s*\.\s*IV\s*\.\s*Length\s*\)",
        source,
        flags=re.IGNORECASE,
    ):
        return False
    for allocation in re.finditer(
        r"\bbyte\s*\[\s*\]\s+(?P<iv>[A-Za-z_]\w*)\s*=\s*"
        r"new\s+byte\s*\[\s*(?P<size>\d+)\s*\]\s*;",
        source,
        flags=re.IGNORECASE,
    ):
        if int(allocation.group("size")) == 16:
            continue
        iv = allocation.group("iv")
        if re.search(
            rf"Array\s*\.\s*Copy\s*\(\s*[A-Za-z_]\w*\s*,\s*0\s*,"
            rf"\s*{re.escape(iv)}\s*,\s*0\s*,\s*"
            rf"{allocation.group('size')}\s*\)",
            source[allocation.end() :],
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _csharp_recovers_iv_from_key_prefix(source: str) -> bool:
    """Find a key-then-IV envelope whose key prefix is restored as the IV."""

    for first in re.finditer(
        r"Array\s*\.\s*Copy\s*\(\s*[^,;]*\.\s*Key\s*,\s*0\s*,\s*"
        r"(?P<envelope>[A-Za-z_]\w*)\s*,\s*0\s*,",
        source,
        flags=re.IGNORECASE,
    ):
        envelope = first.group("envelope")
        if re.search(
            rf"Array\s*\.\s*Copy\s*\(\s*[^,;]*\.\s*IV\s*,\s*0\s*,\s*"
            rf"{re.escape(envelope)}\s*,\s*[^,;]*\.\s*Key\s*\.\s*Length\s*,",
            source[first.end() :],
            flags=re.IGNORECASE,
        ) is None:
            continue
        recovered = {
            match.group("name")
            for match in re.finditer(
                r"\b(?:byte\s*\[\s*\]|var)\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
                r"File\s*\.\s*ReadAllBytes\s*\(",
                source,
                flags=re.IGNORECASE,
            )
        }
        if any(
            re.search(
                rf"Array\s*\.\s*Copy\s*\(\s*{re.escape(name)}\s*,\s*"
                r"[A-Za-z_]\w*\s*\.\s*IV\s*,\s*"
                r"[A-Za-z_]\w*\s*\.\s*IV\s*\.\s*Length\s*\)",
                source,
                flags=re.IGNORECASE,
            )
            for name in recovered
        ):
            return True
    return False


def _csharp_discards_all_encryption_results(
    analyzer: object,
    source: str,
) -> bool:
    """Find an encryption-only route whose terminal bytes are all ignored."""

    encrypt_receivers = {
        re.sub(r"\s+", "", assignment.name)
        for assignment in getattr(analyzer, "assignments", ())
        if re.search(r"\.\s*CreateEncryptor\s*\(", assignment.expression)
    }
    terminals = []
    for call in getattr(analyzer, "calls")("TransformFinalBlock"):
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        if receiver in encrypt_receivers or "CreateEncryptor(" in receiver:
            terminals.append(call)
    for call in getattr(analyzer, "calls")("TransformBlock"):
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        if receiver not in encrypt_receivers and "CreateEncryptor(" not in receiver:
            continue
        if len(call.arguments) < 4:
            continue
        output = call.arguments[3].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", output) and re.search(
            rf"\b{re.escape(output)}\b",
            source[call.end :],
        ):
            return False
    return bool(terminals) and all(
        _c_style_call_is_bare_statement(source, call.start, call.end)
        for call in terminals
    )


def _c_style_call_is_bare_statement(source: str, start: int, end: int) -> bool:
    statement_start = max(
        source.rfind(";", 0, start),
        source.rfind("{", 0, start),
        source.rfind("}", 0, start),
    ) + 1
    statement_end = source.find(";", end)
    if statement_end == -1:
        return False
    prefix = re.sub(
        r"//[^\n]*|/\*.*?\*/",
        "",
        source[statement_start:start],
        flags=re.DOTALL,
    )
    return not prefix.strip() and not source[end:statement_end].strip()


def _csharp_completed_round_trip_before_reuse(
    analyzer: object,
    repeated_receiver: str,
) -> bool:
    """Recognize a consumed encryption result before an extra transform."""

    assignments = tuple(getattr(analyzer, "assignments", ()))
    encrypted = next(
        (
            assignment
            for assignment in assignments
            if re.search(
                rf"(?<![\w$]){re.escape(repeated_receiver)}\s*\.\s*"
                r"TransformFinalBlock\s*\(",
                assignment.expression,
            )
        ),
        None,
    )
    if encrypted is None:
        return False
    encrypted_name = re.sub(r"\s+", "", encrypted.name)
    for call in getattr(analyzer, "calls")("TransformFinalBlock"):
        if call.start <= encrypted.end or not call.arguments:
            continue
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        if receiver == repeated_receiver:
            continue
        if re.fullmatch(
            rf"(?:this\.)?{re.escape(encrypted_name)}",
            re.sub(r"\s+", "", call.arguments[0]),
        ):
            return True
    return False


def _cstyle_directional_bodies(analyzer: object, source: str) -> dict[str, tuple[str, ...]]:
    bodies: dict[str, list[str]] = {"encrypt": [], "decrypt": []}
    for function in getattr(analyzer, "functions", ()):
        if _directional_function_name(function.name, "encrypt"):
            bodies["encrypt"].append(source[function.start : function.end])
        elif _directional_function_name(function.name, "decrypt"):
            bodies["decrypt"].append(source[function.start : function.end])
    return {name: tuple(values) for name, values in bodies.items()}


def _csharp_decrypts_original_input(analyzer: object) -> bool:
    """Find a round trip that passes plaintext, not ciphertext, to decryption."""

    functions = tuple(getattr(analyzer, "functions", ()))
    assignments = tuple(getattr(analyzer, "assignments", ()))
    encrypt_calls = tuple(
        call
        for call in getattr(analyzer, "calls")()
        if _directional_function_name(call.simple_name, "encrypt")
    )
    decrypt_calls = tuple(
        call
        for call in getattr(analyzer, "calls")()
        if _directional_function_name(call.simple_name, "decrypt")
    )
    for call in encrypt_calls:
        enclosing_function = min(
            (
                function
                for function in functions
                if function.start <= call.start <= function.end
            ),
            key=lambda function: function.end - function.start,
            default=None,
        )
        assignment = next(
            (
                candidate
                for candidate in assignments
                if candidate.start <= call.start <= candidate.end
            ),
            None,
        )
        definitions = [
            function for function in functions if function.name == call.simple_name
        ]
        if assignment is None or not definitions:
            continue
        payload_index = _cstyle_payload_parameter_index(definitions[0].parameters)
        if payload_index is None or payload_index >= len(call.arguments):
            continue
        ciphertext = re.sub(r"\s+", "", assignment.name)
        plaintext = re.sub(r"\s+", "", call.arguments[payload_index])
        matching_decrypt_calls = tuple(
            candidate
            for candidate in decrypt_calls
            if candidate.start > call.start
            and (
                enclosing_function is None
                or enclosing_function.start <= candidate.start <= enclosing_function.end
            )
        )
        if any(
            ciphertext
            in {re.sub(r"\s+", "", argument) for argument in candidate.arguments}
            for candidate in matching_decrypt_calls
        ):
            continue
        if any(
            plaintext
            in {re.sub(r"\s+", "", argument) for argument in candidate.arguments}
            for candidate in matching_decrypt_calls
        ):
            return True
    return False


def _cstyle_payload_parameter_index(parameters: tuple[str, ...]) -> int | None:
    for index, parameter in enumerate(parameters):
        if re.search(
            r"(?:plain|message|payload|text|entry|data|input)",
            parameter,
            flags=re.IGNORECASE,
        ):
            return index
    return None


def _csharp_generates_local_key(body: str) -> bool:
    return bool(
        re.search(
            r"(?:rng|random[A-Za-z0-9_]*)\s*\.\s*"
            r"(?:GetBytes|NextBytes)\s*\(\s*(?:\w+\s*\.\s*)?Key\s*\)",
            body,
            flags=re.IGNORECASE,
        )
        or re.search(r"\.\s*Key\s*=\s*RandomNumberGenerator", body)
    )


def _csharp_uses_independent_default_cipher_state(
    encrypt_bodies: tuple[str, ...],
    decrypt_bodies: tuple[str, ...],
) -> bool:
    encrypt = "\n".join(encrypt_bodies)
    decrypt = "\n".join(decrypt_bodies)
    creates_separate_ciphers = bool(
        re.search(r"(?:Aes|DES\w*|Rijndael\w*)\s*\.\s*Create\s*\(|new\s+(?:DES|Rijndael)", encrypt)
        and re.search(r"(?:Aes|DES\w*|Rijndael\w*)\s*\.\s*Create\s*\(|new\s+(?:DES|Rijndael)", decrypt)
    )
    if not creates_separate_ciphers:
        return False
    if re.search(r"CipherMode\s*\.\s*ECB", encrypt + decrypt, re.IGNORECASE):
        return False
    key_state = re.compile(
        r"(?:\.\s*|\b)Key\s*=|"
        r"Create(?:Encryptor|Decryptor)\s*\([^)]*key",
        flags=re.IGNORECASE,
    )
    iv_state = re.compile(
        r"(?:\.\s*|\b)IV\s*="
        r"(?!\s*[A-Za-z_$][\w$]*\s*\.\s*IV\b)\s*|"
        r"Create(?:Encryptor|Decryptor)\s*\([^,)]*,(?!\s*[A-Za-z_$][\w$]*\s*\.\s*IV\b)\s*",
        flags=re.IGNORECASE,
    )
    encrypt_iv_state = iv_state.search(encrypt) is not None or (
        re.search(r"\.\s*GenerateIV\s*\(", encrypt, re.IGNORECASE) is not None
        and not _csharp_generated_iv_is_not_transported(encrypt)
    ) or (
        _csharp_default_iv_is_transported(encrypt)
    )
    decrypt_iv_state = iv_state.search(decrypt) is not None or (
        _csharp_has_cross_receiver_property_copy(decrypt, "IV")
    )
    return (
        key_state.search(encrypt) is None
        or key_state.search(decrypt) is None
        or not encrypt_iv_state
        or not decrypt_iv_state
    )


def _csharp_has_cross_receiver_property_copy(
    source: str,
    property_name: str,
) -> bool:
    """Recognize state copied between distinct C# cipher objects."""

    return any(
        match.group("target").lower() != match.group("source").lower()
        for match in re.finditer(
            rf"\b(?P<target>[A-Za-z_]\w*)\s*\.\s*{property_name}\s*=\s*"
            rf"(?P<source>[A-Za-z_]\w*)\s*\.\s*{property_name}\b",
            source,
            flags=re.IGNORECASE,
        )
    )


def _csharp_default_iv_is_transported(source: str) -> bool:
    """Recognize a provider-created IV that is returned with ciphertext."""

    receivers = {
        match.group("receiver")
        for match in re.finditer(
            r"\b(?P<receiver>[A-Za-z_]\w*)\s*\.\s*CreateEncryptor\s*\(",
            source,
            flags=re.IGNORECASE,
        )
    }
    returns = "\n".join(re.findall(r"\breturn\s+([^;]+);", source))
    return any(
        re.search(
            rf"\b{re.escape(receiver)}\s*\.\s*IV\b",
            returns,
            flags=re.IGNORECASE,
        )
        for receiver in receivers
    )


def _csharp_same_scope_uses_independent_cipher_state(
    analyzer: object,
    source: str,
) -> bool:
    """Find separate local cipher objects whose implicit state cannot match.

    Directional helper checks cover the usual ``Encrypt``/``Decrypt`` split.
    This companion handles inline round trips, where two freshly constructed
    providers are used in one method.  It only rejects state that is plainly
    disconnected: missing shared key material, or (outside ECB) missing shared
    IV material.
    """

    constructions: dict[tuple[str | None, str], int] = {}
    construction_pattern = re.compile(
        r"(?:\bvar\b|\bAes\b|\bDES\w*\b|\bRijndael\w*\b)\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?:"
        r"(?:Aes|DES\w*|Rijndael\w*)\s*\.\s*Create\s*\(\s*\)|"
        r"new\s+(?:DESCryptoServiceProvider|AesCryptoServiceProvider|"
        r"RijndaelManaged)\s*\(\s*\))",
        flags=re.IGNORECASE,
    )
    for match in construction_pattern.finditer(source):
        constructions[
            (analyzer.containing_function_name(match.start()), match.group("name"))
        ] = match.start()

    encrypt_calls: list[tuple[object, str]] = []
    decrypt_calls: list[tuple[object, str]] = []
    for operation, destination in (
        ("CreateEncryptor", encrypt_calls),
        ("CreateDecryptor", decrypt_calls),
    ):
        for call in analyzer.calls(operation):
            receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
            if re.fullmatch(r"[A-Za-z_]\w*", receiver):
                destination.append((call, receiver))

    for encrypt_call, encrypt_receiver in encrypt_calls:
        scope = analyzer.containing_function_name(encrypt_call.start)
        for decrypt_call, decrypt_receiver in decrypt_calls:
            if analyzer.containing_function_name(decrypt_call.start) != scope:
                continue
            if encrypt_receiver == decrypt_receiver:
                # One cipher object supplies both directions, so its key/IV
                # state is connected.  Additional experimental cipher routes
                # in the same method do not invalidate that completed route.
                return False
            if (scope, encrypt_receiver) not in constructions or (
                scope,
                decrypt_receiver,
            ) not in constructions:
                continue
            scope_start = min(
                constructions[(scope, encrypt_receiver)],
                constructions[(scope, decrypt_receiver)],
            )
            scope_end = max(encrypt_call.end, decrypt_call.end)
            body = source[scope_start:scope_end]
            if re.search(r"CipherMode\s*\.\s*ECB", body, re.IGNORECASE):
                needs_iv = False
            else:
                needs_iv = True

            encrypt_key = _csharp_explicit_cipher_state(
                body, encrypt_receiver, "Key", encrypt_call.arguments, 0
            )
            decrypt_key = _csharp_explicit_cipher_state(
                body, decrypt_receiver, "Key", decrypt_call.arguments, 0
            )
            if not encrypt_key or not decrypt_key:
                return True
            if not needs_iv:
                continue
            encrypt_iv = _csharp_explicit_cipher_state(
                body, encrypt_receiver, "IV", encrypt_call.arguments, 1
            )
            decrypt_iv = _csharp_explicit_cipher_state(
                body, decrypt_receiver, "IV", decrypt_call.arguments, 1
            )
            if (
                encrypt_iv is None
                and decrypt_iv == f"{encrypt_receiver}.iv".lower()
            ):
                # Parameterless CreateEncryptor() consumes the provider's
                # current IV.  Reading that same provider property into a
                # distinct decryptor makes the otherwise implicit state link
                # explicit.
                encrypt_iv = decrypt_iv
            if not encrypt_iv or not decrypt_iv:
                return True
    return False


def _csharp_explicit_cipher_state(
    body: str,
    receiver: str,
    property_name: str,
    call_arguments: tuple[str, ...],
    argument_index: int,
) -> str | None:
    if len(call_arguments) > argument_index:
        return re.sub(r"\s+", "", call_arguments[argument_index]).lower()
    assignments = re.findall(
        rf"\b{re.escape(receiver)}\s*\.\s*{property_name}\s*=\s*([^;]+);",
        body,
        flags=re.IGNORECASE,
    )
    if assignments:
        return re.sub(r"\s+", "", assignments[-1]).lower()
    if re.search(
        rf"\b{re.escape(receiver)}\s*\.\s*Generate{property_name}\s*\(\s*\)",
        body,
        flags=re.IGNORECASE,
    ):
        return f"{receiver}.{property_name}".lower()
    aliases = re.findall(
        rf"\b(?:var|byte\s*\[\s*\])\s+([A-Za-z_]\w*)\s*=\s*"
        rf"{re.escape(receiver)}\s*\.\s*{property_name}\s*;",
        body,
        flags=re.IGNORECASE,
    )
    return aliases[-1].lower() if aliases else None


def _csharp_decryption_regenerates_iv(analyzer: object, source: str) -> bool:
    """Find a recovery route that discards its incoming cipher IV."""

    for generation in analyzer.calls("GenerateIV"):
        scope = analyzer.containing_function_name(generation.start)
        if not scope:
            continue
        receiver = re.sub(r"\s+", "", generation.callee).rsplit(".", 1)[0]
        is_decrypt_helper = _directional_function_name(scope, "decrypt")
        follows_encryption = any(
            encryption.end < generation.start
            and analyzer.containing_function_name(encryption.start) == scope
            and re.sub(r"\s+", "", encryption.callee).rsplit(".", 1)[0]
            == receiver
            for encryption in analyzer.calls("CreateEncryptor")
        )
        if not is_decrypt_helper and not follows_encryption:
            continue
        for decryption in analyzer.calls("CreateDecryptor"):
            if decryption.start <= generation.end:
                continue
            if analyzer.containing_function_name(decryption.start) != scope:
                continue
            decrypt_receiver = re.sub(r"\s+", "", decryption.callee).rsplit(".", 1)[0]
            if decrypt_receiver != receiver:
                continue
            between = source[generation.end : decryption.start]
            restored = re.search(
                rf"\b{re.escape(receiver)}\s*\.\s*IV\s*=\s*"
                rf"(?!{re.escape(receiver)}\s*\.\s*IV\b)[^;]+;",
                between,
                flags=re.IGNORECASE,
            )
            if restored:
                continue
            if len(decryption.arguments) >= 2:
                supplied_iv = re.sub(r"\s+", "", decryption.arguments[1]).lower()
                if supplied_iv != f"{receiver.lower()}.iv":
                    continue
            return True
    return False


def _csharp_overwrites_iv_before_decryption(analyzer: object, source: str) -> bool:
    """Find an obvious fresh or zero IV assignment before decryption."""

    for encryption in analyzer.calls("CreateEncryptor"):
        receiver = re.sub(r"\s+", "", encryption.callee).rsplit(".", 1)[0]
        scope = analyzer.containing_function_name(encryption.start)
        for decryption in analyzer.calls("CreateDecryptor"):
            if decryption.start <= encryption.end:
                continue
            if analyzer.containing_function_name(decryption.start) != scope:
                continue
            decrypt_receiver = re.sub(r"\s+", "", decryption.callee).rsplit(".", 1)[0]
            if decrypt_receiver != receiver:
                continue
            assignments = re.findall(
                rf"\b{re.escape(receiver)}\s*\.\s*IV\s*=\s*([^;]+);",
                source[encryption.end : decryption.start],
                flags=re.IGNORECASE,
            )
            if assignments and re.search(
                r"\bnew\s+byte\s*\[|RandomNumberGenerator|GetBytes\s*\(",
                assignments[-1],
                flags=re.IGNORECASE,
            ):
                return True
    return False


def _csharp_regenerates_key_before_decryption(analyzer: object, source: str) -> bool:
    """Find a receiver whose key is regenerated between its two directions."""

    for encryption in analyzer.calls("CreateEncryptor"):
        receiver = re.sub(r"\s+", "", encryption.callee).rsplit(".", 1)[0]
        scope = analyzer.containing_function_name(encryption.start)
        for decryption in analyzer.calls("CreateDecryptor"):
            if decryption.start <= encryption.end:
                continue
            if analyzer.containing_function_name(decryption.start) != scope:
                continue
            decrypt_receiver = re.sub(r"\s+", "", decryption.callee).rsplit(".", 1)[0]
            if decrypt_receiver != receiver:
                continue
            if re.search(
                rf"\b{re.escape(receiver)}\s*\.\s*GenerateKey\s*\(",
                source[encryption.end : decryption.start],
                flags=re.IGNORECASE,
            ):
                return True
    return False


def _csharp_returns_write_stream_before_disposal(source: str) -> bool:
    """Find a return inside a write-mode CryptoStream block.

    Disposing a write-mode CryptoStream finalizes it, so returning after the
    block is valid.  A source-wide regular expression cannot preserve that
    boundary and also confuses read-mode streams with incomplete encryption.
    """

    start_pattern = re.compile(
        r"using\s*\([^\n]*CryptoStream[^\n]*CryptoStreamMode\s*\.\s*Write[^\n]*\)\s*\{",
        flags=re.IGNORECASE,
    )
    for match in start_pattern.finditer(source):
        opening = source.find("{", match.start(), match.end())
        if opening < 0:
            continue
        depth = 0
        closing = None
        for position in range(opening, len(source)):
            character = source[position]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    closing = position
                    break
        if closing is None:
            continue
        body = source[opening + 1 : closing]
        if re.search(r"\.\s*FlushFinalBlock\s*\(", body, re.IGNORECASE):
            continue
        returned = re.search(
            r"\breturn\s+[A-Za-z_]\w*\.ToArray\s*\(\s*\)\s*;",
            body,
            re.IGNORECASE,
        )
        if returned is None:
            continue
        if _csharp_writer_is_disposed_before(body, returned.start()):
            continue
        if returned:
            return True

    return False


def _csharp_write_stream_output_failure(
    analyzer: object,
    source: str,
) -> str | None:
    """Find ciphertext read before a write-mode CryptoStream is finalized."""

    assignments = tuple(getattr(analyzer, "assignments", ()))
    functions = tuple(getattr(analyzer, "functions", ()))
    calls = tuple(getattr(analyzer, "calls")())
    for call in getattr(analyzer, "calls")("CryptoStream"):
        if len(call.arguments) < 3 or re.search(
            r"CryptoStreamMode\s*\.\s*Write",
            call.arguments[2],
            flags=re.IGNORECASE,
        ) is None:
            continue
        wrapper_assignment = next(
            (
                assignment
                for assignment in assignments
                if assignment.start <= call.start <= assignment.end
            ),
            None,
        )
        if wrapper_assignment is None:
            continue
        wrapper = re.sub(r"\s+", "", wrapper_assignment.name).rsplit(".", 1)[-1]
        function = next(
            (
                candidate
                for candidate in functions
                if candidate.start <= call.start <= candidate.end
            ),
            None,
        )
        scope_end = function.end if function is not None else len(source)

        if re.fullmatch(
            r"\s*new\s+(?:System\s*\.\s*IO\s*\.)?MemoryStream\s*\(\s*\)\s*",
            call.arguments[0],
            re.I,
        ):
            suffix = source[call.end:scope_end]
            if re.search(
                rf"\b{re.escape(wrapper)}\s*\.\s*(?:Read|ToArray|Length|Position)\b",
                suffix,
                flags=re.IGNORECASE,
            ):
                return "encryption writes to an anonymous backing stream whose ciphertext is never recovered"
            continue

        backing = re.sub(r"\s+", "", call.arguments[0])
        if not re.fullmatch(r"[A-Za-z_]\w*", backing):
            continue
        statement_start = max(
            source.rfind(";", 0, call.start),
            source.rfind("{", 0, call.start),
            source.rfind("}", 0, call.start),
            source.rfind("\n", 0, call.start),
        ) + 1
        declaration = source[statement_start:call.start]
        if re.search(
            rf"\busing\s+(?:var|CryptoStream)\s+{re.escape(wrapper)}\s*=",
            declaration,
            flags=re.IGNORECASE,
        ) is None:
            continue

        writer_names = {
            re.sub(r"\s+", "", assignment.name).rsplit(".", 1)[-1]
            for assignment in assignments
            if call.end < assignment.start < scope_end
            and re.search(
                rf"new\s+(?:StreamWriter|BinaryWriter)\s*\(\s*{re.escape(wrapper)}\b",
                assignment.expression,
                flags=re.IGNORECASE,
            )
        }
        reads = tuple(
            candidate
            for candidate in calls
            if call.end < candidate.start < scope_end
            and candidate.simple_name == "ToArray"
            and re.sub(r"\s+", "", candidate.callee).rsplit(".", 1)[0]
            == backing
        )
        for read in reads:
            finalized = any(
                call.end < candidate.start < read.start
                and candidate.simple_name in {"FlushFinalBlock", "Close", "Dispose"}
                and re.sub(r"\s+", "", candidate.callee).rsplit(".", 1)[0]
                in ({wrapper} | writer_names)
                for candidate in calls
            )
            if not finalized:
                suffix = source[call.end:scope_end]
                finalized = _csharp_writer_is_disposed_before(
                    suffix,
                    read.start - call.end,
                    wrapped_stream=wrapper,
                )
            if not finalized:
                return "ciphertext is read before the write-mode CryptoStream is finalized"
    return None


def _csharp_uses_fresh_iv_helper_for_each_direction(
    analyzer: object,
    source: str,
) -> bool:
    """Find repeated calls to an IV helper that creates fresh random bytes."""

    fresh_helpers = {
        function.name
        for function in getattr(analyzer, "functions", ())
        if re.search(r"(?:iv|nonce)", function.name, re.IGNORECASE)
        and re.search(
            r"(?:RandomNumberGenerator|RNGCryptoServiceProvider|GenerateIV|GetBytes)\b",
            source[function.start:function.end],
            flags=re.IGNORECASE,
        )
        and not _csharp_iv_helper_returns_existing_iv(
            source[function.start:function.end]
        )
    }
    if not fresh_helpers:
        return False

    def helper_at(call: object) -> str | None:
        if len(call.arguments) < 2:
            return None
        argument = call.arguments[1]
        return next(
            (
                helper
                for helper in fresh_helpers
                if re.search(rf"\b{re.escape(helper)}\s*\(", argument)
            ),
            None,
        )

    encrypt_helpers = {
        helper
        for call in getattr(analyzer, "calls")("CreateEncryptor")
        if (helper := helper_at(call)) is not None
    }
    decrypt_helpers = {
        helper
        for call in getattr(analyzer, "calls")("CreateDecryptor")
        if (helper := helper_at(call)) is not None
    }
    return bool(encrypt_helpers & decrypt_helpers)


def _csharp_iv_helper_returns_existing_iv(body: str) -> bool:
    """Recognize helpers that preserve and return the receiver's current IV.

    Some generated helpers contain a defensive random fallback for an empty
    IV, but their ordinary route returns an alias of ``aes.IV`` unchanged.
    They must not be treated like helpers that generate fresh material on
    every call.  A helper that calls ``GenerateIV`` is intentionally excluded:
    repeated calls to that helper advance the receiver to different IVs.
    """

    if re.search(r"\.\s*GenerateIV\s*\(", body, re.IGNORECASE):
        return False
    aliases = {
        match.group("name")
        for match in re.finditer(
            r"\b(?:var|byte\s*\[\s*\])\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
            r"[A-Za-z_]\w*\s*\.\s*IV\s*;",
            body,
            flags=re.IGNORECASE,
        )
    }
    return any(
        re.search(rf"\breturn\s+{re.escape(name)}\s*;", body)
        for name in aliases
    )


def _csharp_writer_is_disposed_before(
    body: str,
    position: int,
    *,
    wrapped_stream: str | None = None,
) -> bool:
    """Recognize a nested writer/compressor completed before output capture.

    A compression block completes its own write route even when it deliberately
    leaves the wrapping CryptoStream open for the enclosing ``using`` scope.
    Ordinary writers with ``leaveOpen`` still do not establish finalization.
    """

    compression_streams = r"GZipStream|DeflateStream|ZLibStream"
    pattern = re.compile(
        r"using\s*\([^;{}]*(?:StreamWriter|BinaryWriter|"
        + compression_streams
        + r")[^;{}]*\)\s*\{",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(body, 0, position):
        declaration = match.group(0)
        leaves_wrapped_stream_open = re.search(
            r",\s*true\s*\)+\s*\{",
            declaration,
            re.IGNORECASE,
        )
        completes_compression = re.search(
            rf"\b(?:{compression_streams})\b",
            declaration,
            re.IGNORECASE,
        )
        if leaves_wrapped_stream_open and completes_compression is None:
            continue
        if wrapped_stream is not None and re.search(
            rf"\(\s*{re.escape(wrapped_stream)}\b",
            declaration,
            re.IGNORECASE,
        ) is None:
            continue
        opening = body.find("{", match.start(), match.end())
        if opening < 0:
            continue
        depth = 0
        for index in range(opening, len(body)):
            if body[index] == "{":
                depth += 1
            elif body[index] == "}":
                depth -= 1
                if depth == 0:
                    if index < position:
                        return True
                    break
    return False


def _csharp_generated_iv_is_not_transported(body: str) -> bool:
    if re.search(r"\.\s*GenerateIV\s*\(", body) is None:
        return False
    # A constructor whose class name contains ``Encryption`` can be classified
    # as a directional helper by the lightweight source index.  Initializing a
    # field-owned cipher there is not itself an encryption route and therefore
    # does not need to return an IV from the constructor.
    if re.search(
        r"(?:CreateEncryptor\s*\(|CryptoStreamMode\s*\.\s*Write|"
        r"TransformFinalBlock\s*\()",
        body,
        flags=re.IGNORECASE,
    ) is None:
        return False
    last_generation = max(
        match.end() for match in re.finditer(r"\.\s*GenerateIV\s*\(", body)
    )
    if re.search(
        r"\.\s*IV\s*=(?!\s*[A-Za-z_$][\w$]*\s*\.\s*IV\b)\s*[^;]+;",
        body[last_generation:],
        flags=re.IGNORECASE,
    ):
        return False
    suffix = body[last_generation:]
    returns = "\n".join(re.findall(r"\breturn\s+([^;]+);", suffix))
    if re.search(r"\biv\b|\.\s*IV\b", returns, re.IGNORECASE):
        return False

    # Follow the common envelope construction used by generated C# programs:
    # copy ``aes.IV`` to a local alias, copy that alias into a combined buffer,
    # then return the combined buffer (possibly Base64 encoded).
    iv_aliases = {
        match.group("name")
        for match in re.finditer(
            r"\b(?:var|byte\s*\[\s*\])\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
            r"[A-Za-z_]\w*\s*\.\s*IV\s*;",
            suffix,
            flags=re.IGNORECASE,
        )
    }
    transported = set(iv_aliases)
    changed = True
    while changed:
        changed = False
        for call in re.finditer(
            r"(?:Array\s*\.\s*Copy|Buffer\s*\.\s*BlockCopy)\s*\("
            r"\s*(?P<source>[A-Za-z_]\w*)\s*,[^,]*,\s*"
            r"(?P<destination>[A-Za-z_]\w*)\s*,",
            suffix,
            flags=re.IGNORECASE,
        ):
            if (
                call.group("source") in transported
                and call.group("destination") not in transported
            ):
                transported.add(call.group("destination"))
                changed = True
        for assignment in re.finditer(
            r"\b(?:var|byte\s*\[\s*\]|string)\s+"
            r"(?P<destination>[A-Za-z_]\w*)\s*=\s*(?P<expression>[^;]+);",
            suffix,
            flags=re.IGNORECASE,
        ):
            names = set(re.findall(r"\b[A-Za-z_]\w*\b", assignment.group("expression")))
            if names & transported and assignment.group("destination") not in transported:
                transported.add(assignment.group("destination"))
                changed = True
    if transported and any(
        re.search(rf"\b{re.escape(name)}\b", returns)
        for name in transported
    ):
        return False
    return re.search(
        r"(?:Array\s*\.\s*Copy|Concat|Write)\s*\([^;]*\biv\b",
        suffix,
        flags=re.IGNORECASE,
    ) is None


def _csharp_calls_fresh_key_helper_in_both_directions(
    source: str,
    encrypt_bodies: tuple[str, ...],
    decrypt_bodies: tuple[str, ...],
) -> bool:
    fresh_helpers = {
        match.group("name")
        for match in re.finditer(
            r"(?:static\s+)?byte\s*\[\s*\]\s+(?P<name>[A-Za-z_]\w*)\s*"
            r"\([^)]*\)\s*\{(?P<body>.*?)\}",
            source,
            flags=re.DOTALL,
        )
        if re.search(r"(?:key|secret)", match.group("name"), re.IGNORECASE)
        and re.search(r"\.\s*GenerateKey\s*\(|RandomNumberGenerator", match.group("body"))
    }
    return any(
        re.search(rf"\b{re.escape(name)}\s*\(", "\n".join(encrypt_bodies))
        and re.search(rf"\b{re.escape(name)}\s*\(", "\n".join(decrypt_bodies))
        for name in fresh_helpers
    )


def _csharp_disconnected_fresh_key_helpers(analyzer: object, source: str) -> bool:
    fresh_helpers = {
        function.name
        for function in getattr(analyzer, "functions", ())
        if re.search(r"(?:key|secret)", function.name, re.IGNORECASE)
        and re.search(
            r"\.\s*GenerateKey\s*\(|RandomNumberGenerator",
            source[function.start : function.end],
        )
    }
    for helper in fresh_helpers:
        calls = tuple(getattr(analyzer, "calls")(helper))
        encrypt_use = any(
            any(
                re.search(rf"\b{re.escape(helper)}\s*\(", argument)
                for argument in call.arguments
            )
            for call in getattr(analyzer, "calls")("CreateEncryptor")
        )
        decrypt_use = any(
            _directional_function_name(
                getattr(analyzer, "containing_function_name")(call.start) or "",
                "decrypt",
            )
            for call in calls
        )
        if encrypt_use and decrypt_use:
            return True
    return False


def _java_symmetric_failure(source: str, task_id: str) -> str | None:
    if not task_id.endswith("T1"):
        return None
    if _java_symmetric_directions_are_mutually_exclusive(source):
        return "encryption and decryption occur only on mutually exclusive branches"
    analyzer = analyzer_for("java", source)
    returned_buffer_failure = _java_encrypt_helper_returns_unpopulated_buffer(
        analyzer, source
    )
    if returned_buffer_failure is not None:
        return returned_buffer_failure
    if re.search(
        r"init\s*\(\s*Cipher\.ENCRYPT_MODE\s*,[^;]*\bSecureRandom\b[^;]*\).*?"
        r"init\s*\(\s*Cipher\.DECRYPT_MODE\s*,[^;]*\bSecureRandom\b",
        source,
        flags=re.DOTALL,
    ):
        return "decryption uses fresh random parameters instead of the encryption parameters"
    if re.search(
        r"DECRYPT_MODE[^;]*;\s*(?:byte\[\]\s+\w+\s*=\s*)?"
        r"[^;]*doFinal\s*\(\s*new\s+byte\s*\[",
        source,
        flags=re.DOTALL,
    ):
        return "decryption processes a fresh byte buffer instead of the ciphertext"
    receiver_mode: dict[tuple[str | None, str], str] = {}
    terminal_calls: dict[tuple[str | None, str, str], list[object]] = {}
    for call in sorted(
        analyzer.calls("init", "doFinal"),
        key=lambda item: item.start,
    ):
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        scope = analyzer.containing_function_name(call.start)
        scoped_receiver = (scope, receiver)
        if call.simple_name == "init":
            mode = call.arguments[0] if call.arguments else ""
            receiver_mode[scoped_receiver] = (
                "decrypt" if "DECRYPT_MODE" in mode else "encrypt" if "ENCRYPT_MODE" in mode else ""
            )
            terminal_calls[(scope, receiver, receiver_mode[scoped_receiver])] = []
            continue
        mode = receiver_mode.get(scoped_receiver, "")
        key = (scope, receiver, mode)
        terminal_calls.setdefault(key, []).append(call)

    for (_scope, _receiver, mode), calls in terminal_calls.items():
        if mode != "decrypt" or len(calls) <= 1:
            continue
        if _java_calls_are_conditional_alternatives(source, calls):
            continue
        if _java_completed_decryption_before_reuse(analyzer, source, calls):
            continue
        return "the same decrypting Cipher is finalized more than once"

    connection_failure = _java_ciphertext_connection_failure(analyzer, source, task_id)
    if connection_failure is not None:
        return connection_failure

    key_generation_functions = {
        (analyzer.containing_function_name(call.start) or "").lower()
        for call in analyzer.calls("generateKey")
        if not call.arguments
    }
    if _has_distinct_directional_functions(key_generation_functions):
        return "encryption and decryption independently generate different keys"

    if re.search(
        r"DECRYPT_MODE[^;]*;\s*(?:byte\[\]\s+\w+\s*=\s*)?"
        r"[^;]*doFinal\s*\([^;]*(?:plainText|messageBytes)",
        source,
        flags=re.DOTALL,
    ):
        return "decryption does not consume the ciphertext produced by encryption"
    literal = re.search(r"\bString\s+key\s*=\s*['\"](?P<key>[^'\"]*)['\"]", source)
    if literal and len(literal.group("key").encode()) not in {16, 24, 32} and re.search(
        r"keySize\s*!=\s*16.*?keySize\s*!=\s*24.*?keySize\s*!=\s*32.*?return\s*;",
        source,
        flags=re.DOTALL,
    ):
        return "a constant key-size guard always returns before the cipher sequence"
    return None


def _java_symmetric_directions_are_mutually_exclusive(source: str) -> bool:
    """Recognize an interactive either-encrypt-or-decrypt demonstration."""

    branch = re.compile(
        r"\bif\s*\([^{}]*\)\s*\{(?P<first>.*?)\}\s*"
        r"else(?:\s+if\s*\([^{}]*\))?\s*\{(?P<second>.*?)\}",
        flags=re.DOTALL,
    )
    for match in branch.finditer(source):
        first = match.group("first")
        second = match.group("second")
        split = (
            "ENCRYPT_MODE" in first and "DECRYPT_MODE" in second
        ) or (
            "DECRYPT_MODE" in first and "ENCRYPT_MODE" in second
        )
        if split:
            return True
    return False


def _java_calls_are_conditional_alternatives(
    source: str,
    calls: list[object],
) -> bool:
    assignments = extract_assignments(source)
    containers = [
        next(
            (
                assignment
                for assignment in assignments
                if assignment.start <= call.start <= assignment.end
            ),
            None,
        )
        for call in calls
    ]
    if bool(
        containers
        and all(container is containers[0] for container in containers)
        and containers[0] is not None
        and "?" in containers[0].expression
        and ":" in containers[0].expression
    ):
        return True
    statement_start = max(
        source.rfind(";", 0, calls[0].start),
        source.rfind("{", 0, calls[0].start),
        source.rfind("}", 0, calls[0].start),
    ) + 1
    statement_end = source.find(";", calls[-1].end)
    if statement_end == -1:
        return False
    statement = source[statement_start:statement_end]
    return "?" in statement and ":" in statement


def _java_completed_decryption_before_reuse(
    analyzer: object,
    source: str,
    calls: list[object],
) -> bool:
    """Recognize a completed Java round trip before later extra cipher work."""

    assignments = extract_assignments(source)
    encryption_outputs: set[str] = set()
    modes: dict[tuple[str | None, str], str] = {}
    for call in sorted(
        getattr(analyzer, "calls")("init", "doFinal"),
        key=lambda item: item.start,
    ):
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        scope = getattr(analyzer, "containing_function_name")(call.start)
        key = (scope, receiver)
        if call.simple_name == "init":
            mode = call.arguments[0] if call.arguments else ""
            modes[key] = (
                "decrypt"
                if "DECRYPT_MODE" in mode
                else "encrypt"
                if "ENCRYPT_MODE" in mode
                else ""
            )
            continue
        containing = next(
            (
                assignment
                for assignment in assignments
                if assignment.start <= call.start <= assignment.end
            ),
            None,
        )
        if modes.get(key) == "encrypt" and containing is not None:
            encryption_outputs.add(re.sub(r"\s+", "", containing.name))

    repeated_at = calls[1].start
    for call in calls:
        if call.start >= repeated_at or not call.arguments:
            continue
        containing = next(
            (
                assignment
                for assignment in assignments
                if assignment.start <= call.start <= assignment.end
            ),
            None,
        )
        if containing is None:
            continue
        output = re.sub(r"\s+", "", containing.name)
        if re.search(r"(?:decrypt|recover|plain)", output, re.IGNORECASE) is None:
            continue
        if re.search(r"(?:iv|nonce|tag|mac)", output, re.IGNORECASE):
            continue
        argument = call.arguments[0]
        compact_argument = re.sub(r"\s+", "", argument)
        argument_names = set(re.findall(r"\b[A-Za-z_$][\w$]*\b", argument))
        connected = bool(argument_names & encryption_outputs) or bool(
            re.search(r"(?:encrypt|cipher)", compact_argument, re.IGNORECASE)
            and not re.search(r"(?:iv|nonce)", compact_argument, re.IGNORECASE)
        )
        if not connected:
            continue
        method = next(
            (
                candidate
                for candidate in getattr(getattr(analyzer, "flow", None), "methods", ())
                if candidate.body_start <= containing.start <= candidate.body_end
            ),
            None,
        )
        scope_end = method.body_end if method is not None else len(source)
        if re.search(
            rf"\b{re.escape(output)}\b",
            source[containing.end:scope_end],
        ):
            return True
    return False


def _java_ciphertext_connection_failure(
    analyzer: object,
    source: str,
    task_id: str,
) -> str | None:
    assignments = extract_assignments(source)
    calls = tuple(getattr(analyzer, "calls", lambda *names: ())("init", "doFinal"))
    ordered = sorted(calls, key=lambda item: item.start)
    modes: dict[tuple[str | None, str], str] = {}
    encrypted_values: dict[str | None, set[str]] = {}
    encoded_values: set[str] = set()
    corrupted_containers: set[str] = set()
    completed_scopes: set[str | None] = set()
    allocated_buffers = {
        (getattr(analyzer, "containing_function_name")(match.start()), match.group("name"))
        for match in re.finditer(
            r"\bbyte\s*\[\s*\]\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
            r"new\s+byte\s*\[",
            source,
        )
    }
    populated_buffers: set[tuple[str | None, str]] = set()

    for assignment in assignments:
        name = re.sub(r"\s+", "", assignment.name)
        expression = assignment.expression
        if re.match(
            r"\s*(?:java\.util\.)?Base64\s*\.\s*getEncoder\s*\(\)\s*"
            r"\.\s*encode(?:ToString)?\s*\(",
            expression,
        ):
            encoded_values.add(name)

    copy_calls = tuple(getattr(analyzer, "calls")("arraycopy"))
    copied_plaintext: set[str] = set()
    copied_roles: dict[str, set[str]] = {}
    for assignment in assignments:
        name = re.sub(r"\s+", "", assignment.name)
        copied_iv = re.match(
            r"\s*Arrays\s*\.\s*copyOf\s*\(\s*"
            r"(?P<source>[A-Za-z_$][\w$]*)\s*,",
            assignment.expression,
        )
        if copied_iv and re.search(
            r"(?:iv|nonce)", copied_iv.group("source"), re.IGNORECASE
        ):
            copied_roles.setdefault(name, set()).add("iv")
    fixed_buffer_sizes = {
        match.group("name"): int(match.group("size"))
        for match in re.finditer(
            r"\bbyte\s*\[\s*\]\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
            r"new\s+byte\s*\[\s*(?P<size>\d+)\s*\]",
            source,
        )
    }
    for copy_call in copy_calls:
        if len(copy_call.arguments) < 3:
            continue
        source_name = re.sub(r"\s+", "", copy_call.arguments[0])
        destination = re.sub(r"\s+", "", copy_call.arguments[2])
        copy_scope = getattr(analyzer, "containing_function_name")(copy_call.start)
        populated_buffers.add((copy_scope, destination))
        roles = copied_roles.setdefault(destination, set())
        combined_envelope = bool(
            re.search(r"(?:iv|nonce)", source_name, re.IGNORECASE)
            and re.search(r"(?:encrypt|cipher)", source_name, re.IGNORECASE)
        )
        source_offset = (
            copy_call.arguments[1].strip() if len(copy_call.arguments) >= 2 else ""
        )
        if combined_envelope:
            roles.add(
                "iv"
                if re.fullmatch(r"0+[lL]?", source_offset)
                else "ciphertext"
            )
        else:
            if re.search(r"(?:iv|nonce)", source_name, re.IGNORECASE):
                roles.add("iv")
            if re.search(r"(?:encrypt|cipher)", source_name, re.IGNORECASE):
                roles.add("ciphertext")
        if re.search(
            r"(?:plain|message|record|data)", source_name, re.IGNORECASE
        ) and not re.search(
            r"(?:encrypt|cipher|iv|nonce)", source_name, re.IGNORECASE
        ):
            copied_plaintext.add(destination)
        if (
            destination in copied_plaintext
            and re.search(r"(?:encrypt|cipher)", source_name, re.IGNORECASE)
        ):
            corrupted_containers.add(destination)
        if (
            len(copy_call.arguments) >= 5
            and re.search(r"(?:encrypt|cipher)", source_name, re.IGNORECASE)
            and destination in fixed_buffer_sizes
            and re.fullmatch(r"\d+", copy_call.arguments[3].strip())
            and int(copy_call.arguments[3].strip()) >= fixed_buffer_sizes[destination]
        ):
            corrupted_containers.add(destination)

    if re.search(
        r"\.\s*split\s*\(\s*['\"]:['\"]\s*\).*?"
        r"\b(?:ciphertext|encrypted)[A-Za-z0-9_]*\s*=\s*"
        r"(?:parts|components)\s*\[\s*1\s*\].*?\bdecrypt\s*\(",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    ) and re.search(
        r"Base64\s*\.\s*getDecoder\s*\(\s*\)\s*\.\s*decode\s*\(\s*"
        r"(?:ciphertext|encrypted)[A-Za-z0-9_]*\s*\.\s*substring\s*\(\s*[1-9]",
        source,
        flags=re.IGNORECASE,
    ):
        return "the decrypt path removes a second prefix from already separated ciphertext"

    for call in ordered:
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        scope = getattr(analyzer, "containing_function_name")(call.start)
        scoped_receiver = (scope, receiver)
        if call.simple_name == "init":
            mode = call.arguments[0] if call.arguments else ""
            modes[scoped_receiver] = (
                "decrypt" if "DECRYPT_MODE" in mode else "encrypt" if "ENCRYPT_MODE" in mode else ""
            )
            continue
        containing = next(
            (
                assignment
                for assignment in assignments
                if assignment.start <= call.start <= assignment.end
            ),
            None,
        )
        if modes.get(scoped_receiver) == "encrypt" and containing is not None:
            encrypted_values.setdefault(scope, set()).add(
                re.sub(r"\s+", "", containing.name)
            )
            continue
        if modes.get(scoped_receiver) != "decrypt" or not call.arguments:
            continue
        if scope in completed_scopes:
            continue
        argument = call.arguments[0].strip()
        compact = re.sub(r"\s+", "", argument)
        if compact.startswith("newbyte[") or (
            (scope, compact) in allocated_buffers
            and (scope, compact) not in populated_buffers
        ):
            return "decryption processes a fresh byte buffer instead of the ciphertext"
        if compact in encoded_values:
            return "Base64-encoded ciphertext is passed to Cipher.doFinal without decoding"
        if compact in corrupted_containers:
            return "CBC decryption receives a fabricated container rather than the encryption result"
        argument_names = set(re.findall(r"\b[A-Za-z_$][\w$]*\b", argument))
        scope_ciphertexts = set(encrypted_values.get(scope, ()))
        envelope_values = {
            name for name, roles in copied_roles.items() if {"iv", "ciphertext"} <= roles
        }
        if compact in envelope_values and (
            len(call.arguments) == 1
            or len(call.arguments) >= 3
            and call.arguments[1].strip() == "0"
        ) and not _java_decrypted_envelope_prefix_is_discarded(
            analyzer,
            source,
            call,
            containing,
        ):
            return (
                "decryption consumes the complete serialized IV and ciphertext "
                "without selecting the ciphertext portion"
            )
        if scope_ciphertexts:
            changed = True
            while changed:
                changed = False
                for assignment in assignments:
                    if assignment.start >= call.start:
                        continue
                    name = re.sub(r"\s+", "", assignment.name)
                    references = set(
                        re.findall(
                            r"\b[A-Za-z_$][\w$]*\b", assignment.expression
                        )
                    )
                    if references & scope_ciphertexts and name not in scope_ciphertexts:
                        scope_ciphertexts.add(name)
                        changed = True
                for copy_call in getattr(analyzer, "calls")("arraycopy"):
                    if copy_call.start >= call.start or len(copy_call.arguments) < 3:
                        continue
                    source_names = set(
                        re.findall(
                            r"\b[A-Za-z_$][\w$]*\b", copy_call.arguments[0]
                        )
                    )
                    destination = re.sub(r"\s+", "", copy_call.arguments[2])
                    if (
                        source_names & scope_ciphertexts
                        and re.fullmatch(r"[A-Za-z_$][\w$]*", destination)
                        and destination not in scope_ciphertexts
                    ):
                        scope_ciphertexts.add(destination)
                        changed = True
                for append_call in getattr(analyzer, "calls")("append"):
                    if append_call.start >= call.start or not append_call.arguments:
                        continue
                    references = set(
                        re.findall(
                            r"\b[A-Za-z_$][\w$]*\b", append_call.arguments[0]
                        )
                    )
                    receiver = re.sub(r"\s+", "", append_call.callee).split(".", 1)[0]
                    if references & scope_ciphertexts and receiver not in scope_ciphertexts:
                        scope_ciphertexts.add(receiver)
                        changed = True
                for append_chain in re.finditer(
                    r"\b(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*append\s*\([^;]+;",
                    source[: call.start],
                    flags=re.DOTALL,
                ):
                    references = set(
                        re.findall(r"\b[A-Za-z_$][\w$]*\b", append_chain.group(0))
                    )
                    receiver = append_chain.group("receiver")
                    if references & scope_ciphertexts and receiver not in scope_ciphertexts:
                        scope_ciphertexts.add(receiver)
                        changed = True
                prefix = source[: call.start]
                for ciphertext_name in tuple(scope_ciphertexts):
                    for mutation in re.finditer(
                        rf"\b(?P<name>[A-Za-z_$][\w$]*)\s*\+=\s*"
                        rf"[^;]*\b{re.escape(ciphertext_name)}\b",
                        prefix,
                    ):
                        name = mutation.group("name")
                        if name not in scope_ciphertexts:
                            scope_ciphertexts.add(name)
                            changed = True
            if not argument_names & scope_ciphertexts:
                return "decryption does not consume the ciphertext produced by encryption"
            completed_scopes.add(scope)
    return None


def _java_recovered_envelope_prefix_is_removed(
    analyzer: object,
    source: str,
    call: object,
) -> bool:
    assignment = next(
        (
            value
            for value in extract_assignments(source)
            if value.start <= call.start <= value.end
        ),
        None,
    )
    if assignment is None:
        return False
    name = re.escape(assignment.name.rsplit(".", 1)[-1])
    suffix = source[call.end :]
    return bool(
        re.search(
            rf"new\s+byte\s*\[\s*{name}\.length\s*-\s*"
            r"(?:iv|nonce)[A-Za-z0-9_$]*\.length\s*\]"
            rf"[\s\S]*?System\.arraycopy\s*\(\s*{name}\s*,\s*"
            r"(?:iv|nonce)[A-Za-z0-9_$]*\.length",
            suffix,
            flags=re.IGNORECASE,
        )
    )


def _java_encrypt_helper_returns_unpopulated_buffer(
    analyzer: object,
    source: str,
) -> str | None:
    """Find an encrypt helper that computes ciphertext but returns fresh zeros."""

    assignments = extract_assignments(source)
    for method in getattr(getattr(analyzer, "flow", None), "methods", ()):
        if not _directional_function_name(method.name, "encrypt"):
            continue
        body = source[method.body_start : method.body_end]
        produced = {
            re.sub(r"\s+", "", assignment.name)
            for assignment in assignments
            if method.body_start <= assignment.start <= method.body_end
            and re.search(r"\.\s*doFinal\s*\(", assignment.expression)
        }
        if not produced:
            continue
        if re.search(r"\breturn\s+new\s+byte\s*\[", body):
            return "the encryption helper returns a fresh buffer instead of its ciphertext"
        allocated = {
            match.group("name")
            for match in re.finditer(
                r"\bbyte\s*\[\s*\]\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
                r"new\s+byte\s*\[",
                body,
            )
        }
        for name in allocated:
            if re.search(rf"\breturn\s+{re.escape(name)}\s*;", body) is None:
                continue
            if re.search(
                rf"(?:arraycopy\s*\(\s*(?:{'|'.join(map(re.escape, produced))})\s*,"
                rf"[^;]*,\s*{re.escape(name)}\s*,|"
                rf"\b{re.escape(name)}\s*\[[^\]]+\]\s*=)",
                body,
                flags=re.IGNORECASE,
            ) is None:
                return "the encryption helper returns a fresh buffer instead of its ciphertext"
    return None


def _java_envelope_input_is_sliced(call: object, argument: str) -> bool:
    """Recognize a ciphertext suffix selected before ``Cipher.doFinal``."""

    if re.search(
        r"Arrays\s*\.\s*copyOfRange\s*\([^,]+,\s*(?!0\s*[,)]).*?,",
        argument,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return True
    arguments = tuple(getattr(call, "arguments", ()))
    if len(arguments) < 3:
        return False
    offset = str(arguments[1]).strip()
    return bool(offset and not re.fullmatch(r"0+[lL]?", offset))


def _java_decrypted_envelope_prefix_is_discarded(
    analyzer: object,
    source: str,
    call: object,
    assignment: object | None,
) -> bool:
    """Recognize recovery that explicitly drops the decrypted prefix block."""

    if assignment is None:
        return False
    output = re.sub(r"\s+", "", str(getattr(assignment, "name", "")))
    if not re.fullmatch(r"[A-Za-z_$][\w$]*", output):
        return False
    for copy_call in getattr(analyzer, "calls")("arraycopy"):
        if copy_call.start <= getattr(call, "end", 0) or len(copy_call.arguments) < 2:
            continue
        if re.sub(r"\s+", "", copy_call.arguments[0]) != output:
            continue
        offset = copy_call.arguments[1].strip()
        if offset and not re.fullmatch(r"0+[lL]?", offset):
            return True
    return bool(
        re.search(
            rf"Arrays\s*\.\s*copyOfRange\s*\(\s*{re.escape(output)}\s*,\s*"
            r"(?!0\s*[,)]).*?,",
            source[getattr(call, "end", 0) :],
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


def _python_directional_helper_call_failure(
    tree: ast.Module,
    task_id: str,
) -> str | None:
    """Reject definite argument-count or semantic-role helper failures."""

    required_directions = (
        {"encrypt", "decrypt"}
        if task_id.endswith("T1")
        else {"encrypt"}
        if task_id.endswith("T2")
        else {"decrypt"}
        if task_id.endswith("T3")
        else {"encrypt", "decrypt"}
    )
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            _directional_function_name(node.name, direction)
            for direction in required_directions
        )
    }
    for call in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in functions
    ):
        function = functions[call.func.id]
        parameters = (*function.args.posonlyargs, *function.args.args)
        required_count = len(parameters) - len(function.args.defaults)
        supplied_names = {
            keyword.arg for keyword in call.keywords if keyword.arg is not None
        }
        supplied_count = len(call.args) + len(supplied_names)
        if supplied_count < required_count:
            return f"cipher helper {function.name!r} is called without a required argument"
        for position, parameter in enumerate(parameters[: len(call.args)]):
            expected = _python_cipher_value_role(parameter.arg)
            supplied = _python_cipher_value_role(ast.unparse(call.args[position]))
            directional_data_input = bool(
                expected in {"payload", "ciphertext"}
                and supplied in {"payload", "ciphertext"}
                and (
                    _directional_function_name(function.name, "encrypt")
                    or _directional_function_name(function.name, "decrypt")
                )
            )
            if (
                expected is not None
                and supplied is not None
                and expected != supplied
                and not directional_data_input
            ):
                return (
                    f"cipher helper {function.name!r} receives {supplied} material "
                    f"for its {expected} parameter {parameter.arg!r}"
                )
    return None


def _python_cipher_value_role(value: str) -> str | None:
    stripped = value.strip()
    if re.match(r"(?:encrypt|seal|wrap)[A-Za-z0-9_]*\s*\(", stripped, re.IGNORECASE):
        return "ciphertext"
    if re.search(
        r"\b(?:encryptor|cipher[A-Za-z0-9_]*)\s*\.\s*(?:update|encrypt)\s*\(",
        stripped,
        re.IGNORECASE,
    ):
        return "ciphertext"
    if re.match(r"(?:decrypt|open|unwrap)[A-Za-z0-9_]*\s*\(", stripped, re.IGNORECASE):
        return "payload"
    leaf = _python_value_role_leaf(stripped)
    leaf_compact = re.sub(r"[^a-z0-9]", "", leaf.lower())
    compact = re.sub(r"[^a-z0-9]", "", value.lower())
    if re.search(r"(?:secret)?key|keymaterial", leaf_compact):
        return "key"
    if re.search(r"cipher|encrypted|sealed", leaf_compact):
        return "ciphertext"
    if "nonce" in leaf_compact or leaf_compact == "iv":
        return "nonce"
    if re.search(
        r"plain|payload|message|record|data|note|profile|header|content",
        leaf_compact,
    ):
        return "payload"
    if re.search(r"(?:secret)?key|keymaterial", compact):
        return "key"
    if re.search(r"cipher|encrypted|sealed", compact):
        return "ciphertext"
    if "nonce" in compact or re.search(r"(?:^|_)iv(?:$|_)", value.lower()):
        return "nonce"
    if re.search(r"plain|payload|message|record|data|note|profile|header|content", compact):
        return "payload"
    return None


def _python_value_role_leaf(value: str) -> str:
    """Return the field or name that carries an expression's semantic role."""

    try:
        expression = ast.parse(value, mode="eval").body
    except SyntaxError:
        return value
    if isinstance(expression, ast.Attribute):
        return expression.attr
    if isinstance(expression, ast.Subscript):
        index = expression.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, str):
            return index.value
    if isinstance(expression, ast.Name):
        return expression.id
    return value


def _python_assigned_call_names(tree: ast.AST, call_names: set[str]) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(
            node.value, ast.Call
        ):
            continue
        if not isinstance(node.value.func, ast.Attribute):
            continue
        if _python_name(node.value.func).rsplit(".", 1)[-1] not in call_names:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        names.update(target.id for target in targets if isinstance(target, ast.Name))

    factories = {
        function.name
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(returned.value, ast.Name)
            and returned.value.id in names
            for returned in ast.walk(function)
            if isinstance(returned, ast.Return) and returned.value is not None
        )
    }
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(
            node.value, ast.Call
        ):
            continue
        if _python_name(node.value.func).rsplit(".", 1)[-1] not in factories:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _directional_function_name(name: str, direction: str) -> bool:
    """Return whether a helper represents exactly one cipher direction."""

    lowered = name.lower()
    if lowered.startswith(
        ("decrypt", "decipher", "recover", "open_", "opencache", "opensealed")
    ):
        return direction == "decrypt"
    encrypt = "encrypt" in lowered or "seal" in lowered or "protect" in lowered
    decrypt = "decrypt" in lowered or "open" in lowered or "recover" in lowered
    return (encrypt and not decrypt) if direction == "encrypt" else (decrypt and not encrypt)


def _has_distinct_directional_functions(names: set[str]) -> bool:
    return any(_directional_function_name(name, "encrypt") for name in names) and any(
        _directional_function_name(name, "decrypt") for name in names
    )


def _cstyle_named_parameter_arguments(
    analyzer: object,
    *,
    function_hint: str,
    parameter_hints: tuple[str, ...],
) -> tuple[str, ...]:
    """Return call arguments bound to a semantically named helper parameter."""

    values: list[str] = []
    functions = getattr(analyzer, "functions", ())
    calls = getattr(analyzer, "calls", lambda *names: ())
    for function in functions:
        if function_hint not in function.name.lower():
            continue
        position = next(
            (
                index
                for index, parameter in enumerate(function.parameters)
                if any(hint in parameter.lower() for hint in parameter_hints)
            ),
            None,
        )
        if position is None:
            continue
        for call in calls(function.name):
            if position < len(call.arguments):
                values.append(call.arguments[position])
    return tuple(values)


def _python_receiver_calls(tree: ast.AST, receiver: str, operation: str) -> tuple[ast.Call, ...]:
    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == operation
        and _python_name(node.func.value) == receiver
    )


def _python_attribute_calls(
    tree: ast.AST,
    operation: str,
    *,
    receiver_hint: str | None = None,
) -> tuple[ast.Call, ...]:
    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == operation
        and (
            receiver_hint is None
            or receiver_hint.lower() in _python_name(node.func.value).lower()
        )
    )


def _python_function_generates_cipher_key(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    fresh_helpers: set[str],
) -> bool:
    fresh_names: set[str] = set()
    for statement in ast.walk(function):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        if not _python_expression_is_fresh(
            statement.value,
            fresh_helpers,
            fresh_names,
        ):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else (statement.target,)
        )
        fresh_names.update(
            name
            for target in targets
            for name in _python_assignment_target_names(target)
        )

    return any(
        call.args
        and _python_name(call.func).endswith("algorithms.AES")
        and _python_expression_is_fresh(call.args[0], fresh_helpers, fresh_names)
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
    )


def _python_encrypts_with_untransported_fresh_key(
    tree: ast.Module,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    decrypt_functions: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
    fresh_helpers: set[str],
) -> bool:
    if any(
        isinstance(call, ast.Call)
        and call.args
        and _python_name(call.func).endswith("algorithms.AES")
        and not isinstance(call.args[0], ast.Name)
        and _python_expression_is_fresh(call.args[0], fresh_helpers, set())
        for call in ast.walk(function)
    ):
        return True
    fresh_names: set[str] = set()
    for statement in sorted(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ),
        key=lambda node: getattr(node, "lineno", 0),
    ):
        if not _python_expression_is_fresh(
            statement.value,
            fresh_helpers,
            fresh_names,
        ):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
        fresh_names.update(
            name
            for target in targets
            for name in _python_assignment_target_names(target)
        )
    cipher_keys = {
        node.id
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and call.args
        and _python_name(call.func).endswith("algorithms.AES")
        for node in (call.args[0],)
        if isinstance(node, ast.Name) and node.id in fresh_names
    }
    if not cipher_keys:
        return False
    if any(
        isinstance(node, ast.Return)
        and node.value is not None
        and any(
            isinstance(value, ast.Name) and value.id in cipher_keys
            for value in ast.walk(node.value)
        )
        for node in ast.walk(function)
    ):
        return False
    return not _python_derived_cipher_key_is_reconstructible(
        tree,
        function,
        decrypt_functions,
        cipher_keys,
        fresh_names,
    )


def _python_derived_cipher_key_is_reconstructible(
    tree: ast.Module,
    encrypt_function: ast.FunctionDef | ast.AsyncFunctionDef,
    decrypt_functions: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
    cipher_keys: set[str],
    fresh_names: set[str],
) -> bool:
    """Recognize a key re-derived from shared input and transported material.

    The key itself need not be returned when encryption transports a fresh salt
    and decryption invokes the same local derivation helper with that salt and
    the same external secret.  This is a source-level connectivity check; it
    does not evaluate KDF parameters or runtime values.
    """

    assignments = {
        target.id: node.value
        for node in ast.walk(encrypt_function)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name)
    }
    returns = tuple(
        node.value
        for node in ast.walk(encrypt_function)
        if isinstance(node, ast.Return) and node.value is not None
    )
    local_helpers = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for key_name in cipher_keys:
        origin = assignments.get(key_name)
        if not isinstance(origin, ast.Call):
            continue
        helper_name = _python_name(origin.func).rsplit(".", 1)[-1]
        if helper_name not in local_helpers or not re.search(
            r"(?:derive|kdf|password).*key|key.*(?:derive|kdf|password)",
            helper_name,
            re.IGNORECASE,
        ):
            continue

        fresh_positions = {
            index
            for index, argument in enumerate(origin.args)
            if {
                node.id
                for node in ast.walk(argument)
                if isinstance(node, ast.Name)
            }
            & fresh_names
        }
        if not fresh_positions:
            continue
        transported = {
            node.id
            for returned in returns
            for node in ast.walk(returned)
            if isinstance(node, ast.Name)
        }
        if any(
            not (
                {
                    node.id
                    for node in ast.walk(origin.args[position])
                    if isinstance(node, ast.Name)
                }
                & transported
            )
            for position in fresh_positions
        ):
            continue

        for decrypt_function in decrypt_functions:
            for candidate in (
                node
                for node in ast.walk(decrypt_function)
                if isinstance(node, ast.Call)
                and _python_name(node.func).rsplit(".", 1)[-1] == helper_name
                and len(node.args) == len(origin.args)
            ):
                if all(
                    position in fresh_positions
                    or ast.dump(candidate.args[position]) == ast.dump(argument)
                    for position, argument in enumerate(origin.args)
                ):
                    return True

    # Some generated programs wrap PBKDF2 in a plainly named helper such as
    # ``generate_key`` or in a small provider class.  Others return ``salt,
    # key`` from one helper and later re-derive the key through a different
    # wrapper.  In both forms, require the encryption key and decryption key to
    # be connected to the same transported salt-like value and to a local
    # PBKDF2 implementation.  This recognizes the task-level data flow without
    # assuming that helper names or call expressions must be identical.
    transported_fresh = {
        node.id
        for returned in returns
        for node in ast.walk(returned)
        if isinstance(node, ast.Name)
        and node.id in fresh_names
        and re.search(r"(?:salt|iv|nonce)", node.id, re.IGNORECASE)
    }
    if not transported_fresh:
        return False

    encrypt_key_origins = _python_key_origin_expressions(
        encrypt_function,
        cipher_keys,
    )
    if not any(
        _python_expression_reaches_pbkdf2(tree, expression)
        for expression in encrypt_key_origins
    ):
        return False

    for decrypt_function in decrypt_functions:
        decrypt_assignments = _python_simple_assignments(decrypt_function)
        decrypt_fresh: set[str] = set()
        for statement in sorted(
            (
                node
                for node in ast.walk(decrypt_function)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
            ),
            key=lambda node: getattr(node, "lineno", 0),
        ):
            if _python_expression_is_fresh(
                statement.value,
                _python_fresh_material_helpers(tree),
                decrypt_fresh,
            ):
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else (statement.target,)
                )
                decrypt_fresh.update(
                    name
                    for target in targets
                    for name in _python_assignment_target_names(target)
                )

        for call in (
            node
            for node in ast.walk(decrypt_function)
            if isinstance(node, ast.Call)
            and node.args
            and _python_name(node.func).endswith("algorithms.AES")
        ):
            key_expression = _python_resolve_simple_name(
                call.args[0],
                decrypt_assignments,
            )
            if not _python_expression_reaches_pbkdf2(tree, key_expression):
                continue
            dependencies = _python_expression_dependencies(
                key_expression,
                decrypt_assignments,
            )
            shared_material = dependencies & transported_fresh
            if shared_material and not (shared_material & decrypt_fresh):
                return True
    return False


def _python_simple_assignments(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for node in sorted(
        (
            candidate
            for candidate in ast.walk(function)
            if isinstance(candidate, (ast.Assign, ast.AnnAssign))
        ),
        key=lambda candidate: getattr(candidate, "lineno", 0),
    ):
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for target in targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = node.value
    return assignments


def _python_key_origin_expressions(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    key_names: set[str],
) -> tuple[ast.AST, ...]:
    origins: list[ast.AST] = []
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for target in targets:
            names = _python_assignment_target_names(target)
            if names & key_names:
                origins.append(node.value)
    return tuple(origins)


def _python_resolve_simple_name(
    expression: ast.AST,
    assignments: dict[str, ast.AST],
) -> ast.AST:
    seen: set[str] = set()
    current = expression
    while isinstance(current, ast.Name) and current.id in assignments and current.id not in seen:
        seen.add(current.id)
        current = assignments[current.id]
    return current


def _python_expression_dependencies(
    expression: ast.AST,
    assignments: dict[str, ast.AST],
) -> set[str]:
    dependencies: set[str] = set()
    pending = [expression]
    expanded: set[str] = set()
    while pending:
        current = pending.pop()
        for node in ast.walk(current):
            if not isinstance(node, ast.Name):
                continue
            dependencies.add(node.id)
            if node.id in assignments and node.id not in expanded:
                expanded.add(node.id)
                pending.append(assignments[node.id])
    return dependencies


def _python_expression_reaches_pbkdf2(
    tree: ast.Module,
    expression: ast.AST,
) -> bool:
    definitions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.setdefault(node.name, []).append(node)

    pending = [
        _python_name(call.func).rsplit(".", 1)[-1]
        for call in ast.walk(expression)
        if isinstance(call, ast.Call)
    ]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in {"PBKDF2HMAC", "pbkdf2_hmac"}:
            return True
        if name in seen:
            continue
        seen.add(name)
        for definition in definitions.get(name, ()):
            for call in (
                node for node in ast.walk(definition) if isinstance(node, ast.Call)
            ):
                pending.append(_python_name(call.func).rsplit(".", 1)[-1])
    return False


def _python_decodes_raw_returned_key_as_base64(
    tree: ast.Module,
    encrypt_functions: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
    decrypt_functions: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
    fresh_helpers: set[str],
) -> bool:
    """Find raw random key bytes that a decrypt helper treats as encoded text."""

    raw_key_positions: dict[str, set[int]] = {}
    for function in encrypt_functions:
        fresh_names: set[str] = set()
        for statement in ast.walk(function):
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            if not _python_expression_is_fresh(
                statement.value, fresh_helpers, fresh_names
            ):
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
            fresh_names.update(
                name
                for target in targets
                for name in _python_assignment_target_names(target)
            )
        positions: set[int] = set()
        for returned in (
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Return) and node.value is not None
        ):
            values = returned.elts if isinstance(returned, (ast.Tuple, ast.List)) else (returned,)
            positions.update(
                index
                for index, value in enumerate(values)
                if isinstance(value, ast.Name) and value.id in fresh_names
            )
        if positions:
            raw_key_positions[function.name] = positions
    if not raw_key_positions:
        return False

    raw_result_names: set[str] = set()
    for assignment in ast.walk(tree):
        if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
            continue
        value = assignment.value
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
            continue
        positions = raw_key_positions.get(value.func.id)
        if not positions:
            continue
        targets = assignment.targets if isinstance(assignment, ast.Assign) else (assignment.target,)
        for target in targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                raw_result_names.update(
                    element.id
                    for index, element in enumerate(target.elts)
                    if index in positions and isinstance(element, ast.Name)
                )

    for function in decrypt_functions:
        parameters = (*function.args.posonlyargs, *function.args.args)
        decoded_parameters = {
            target.id
            for statement in ast.walk(function)
            if isinstance(statement, (ast.Assign, ast.AnnAssign))
            for target in (
                statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
            )
            if isinstance(target, ast.Name)
            and isinstance(statement.value, ast.Call)
            and _python_name(statement.value.func).endswith("b64decode")
            and statement.value.args
            and isinstance(statement.value.args[0], ast.Name)
            and statement.value.args[0].id == target.id
        }
        positions = {
            index
            for index, parameter in enumerate(parameters)
            if parameter.arg in decoded_parameters
        }
        key_positions = {
            index
            for index, parameter in enumerate(parameters)
            if "key" in parameter.arg.lower()
        }
        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == function.name
            ):
                continue
            if any(
                index < len(call.args)
                and isinstance(call.args[index], ast.Name)
                and call.args[index].id in raw_result_names
                for index in positions
            ):
                return True
            if any(
                index < len(call.args)
                and isinstance(call.args[index], ast.Call)
                and _python_name(call.args[index].func).endswith(
                    ("b64decode", "urlsafe_b64decode")
                )
                and call.args[index].args
                and isinstance(call.args[index].args[0], ast.Name)
                and call.args[index].args[0].id in raw_result_names
                for index in key_positions
            ):
                return True
    return False


def _python_function_generates_cipher_iv(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    fresh_helpers: set[str],
) -> bool:
    fresh_names: set[str] = set()
    assignments = sorted(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ),
        key=lambda node: getattr(node, "lineno", 0),
    )
    for statement in assignments:
        if not _python_expression_is_fresh(
            statement.value,
            fresh_helpers,
            fresh_names,
        ):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
        fresh_names.update(
            name
            for target in targets
            for name in _python_assignment_target_names(target)
        )
    return any(
        call.args
        and _python_name(call.func).endswith(("modes.CBC", "modes.GCM"))
        and _python_expression_is_fresh(call.args[0], fresh_helpers, fresh_names)
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
    )


def _python_directional_cipher_key_mismatch(
    tree: ast.Module,
    encrypt_functions: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
    decrypt_functions: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
) -> bool:
    def keys(
        functions: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
    ) -> tuple[ast.AST, ...]:
        values: list[ast.AST] = []
        for function in functions:
            local_assignments = {
                target.id: node.value
                for node in ast.walk(function)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                for target in (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
                if isinstance(target, ast.Name)
            }
            for call in ast.walk(function):
                if not (
                    isinstance(call, ast.Call)
                    and call.args
                    and _python_name(call.func).endswith("algorithms.AES")
                ):
                    continue
                key = call.args[0]
                if isinstance(key, ast.Name):
                    key = local_assignments.get(key.id, key)
                values.append(key)
        return tuple(values)

    encrypt_keys = keys(encrypt_functions)
    decrypt_keys = keys(decrypt_functions)
    assignments = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    }

    def encryption_calls_with(expression: ast.AST) -> bool:
        expected = ast.dump(expression)
        for function in encrypt_functions:
            parameters = (*function.args.posonlyargs, *function.args.args)
            key_positions = {
                index
                for index, parameter in enumerate(parameters)
                if any(
                    isinstance(call, ast.Call)
                    and call.args
                    and _python_name(call.func).endswith("algorithms.AES")
                    and isinstance(call.args[0], ast.Name)
                    and call.args[0].id == parameter.arg
                    for call in ast.walk(function)
                )
            }
            for call in ast.walk(tree):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == function.name
                ):
                    continue
                for position in key_positions:
                    if position >= len(call.args):
                        continue
                    argument = call.args[position]
                    resolved = (
                        assignments.get(argument.id, argument)
                        if isinstance(argument, ast.Name)
                        else argument
                    )
                    if ast.dump(resolved) == expected:
                        return True
        return False

    for decrypt_key in decrypt_keys:
        if (
            isinstance(decrypt_key, ast.Attribute)
            and _python_name(decrypt_key).startswith("self.")
            and re.search(r"(?:password|passphrase|secret)", decrypt_key.attr, re.IGNORECASE)
            and encrypt_keys
        ):
            return True
        if any(isinstance(node, ast.Name) for node in ast.walk(decrypt_key)):
            continue
        if encryption_calls_with(decrypt_key):
            continue
        if encrypt_keys and all(
            ast.dump(encrypt_key) != ast.dump(decrypt_key)
            for encrypt_key in encrypt_keys
        ):
            return True
    return False


def _python_encrypt_result_is_discarded(
    tree: ast.Module,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    has_cipher_operation = any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"encrypt", "update", "finalize"}
        for call in ast.walk(function)
    )
    returns_value = any(
        isinstance(node, ast.Return) and node.value is not None
        for node in ast.walk(function)
    )
    result_is_used = any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == function.name
        and not isinstance(_python_parent(tree, call), ast.Expr)
        for call in ast.walk(tree)
    )
    return has_cipher_operation and result_is_used and not returns_value


def _python_parent(tree: ast.AST, target: ast.AST) -> ast.AST | None:
    for parent in ast.walk(tree):
        if target in ast.iter_child_nodes(parent):
            return parent
    return None


def _python_fresh_material_helpers(tree: ast.Module) -> set[str]:
    helpers: set[str] = set()
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        if any(
            isinstance(call, ast.Call)
            and _python_name(call.func).rsplit(".", 1)[-1]
            in {"urandom", "token_bytes", "randbytes", "get_random_bytes"}
            for call in ast.walk(function)
        ):
            helpers.add(function.name)
    return helpers


def _python_aes_key_failure(tree: ast.AST) -> str | None:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    class_attributes: dict[ast.ClassDef, set[str]] = {}
    serialized_attributes: dict[ast.ClassDef, set[str]] = {}
    for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        # Dataclasses, attrs-style records, and ordinary annotated records
        # declare usable instance fields in the class body.  Their generated
        # or external initializer need not be reconstructed to establish that
        # ``self.key`` is a possible key input.
        assigned = {
            target.id
            for statement in class_node.body
            if isinstance(statement, (ast.Assign, ast.AnnAssign))
            for target in (
                statement.targets
                if isinstance(statement, ast.Assign)
                else (statement.target,)
            )
            if isinstance(target, ast.Name)
        }
        serialized: set[str] = set()
        for statement in ast.walk(class_node):
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else (statement.target,)
            )
            attributes = {
                target.attr
                for target in targets
                if isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            }
            assigned.update(attributes)
            if isinstance(statement.value, ast.Call) and _python_name(
                statement.value.func
            ).endswith(("private_bytes", "public_bytes")):
                serialized.update(attributes)
        class_attributes[class_node] = assigned
        serialized_attributes[class_node] = serialized

    serialized_methods: dict[str, set[str]] = {}
    for class_node, attributes in serialized_attributes.items():
        for method in (
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            if any(
                isinstance(returned, ast.Return)
                and isinstance(returned.value, ast.Attribute)
                and isinstance(returned.value.value, ast.Name)
                and returned.value.value.id == "self"
                and returned.value.attr in attributes
                for returned in ast.walk(method)
            ):
                serialized_methods.setdefault(class_node.name, set()).add(method.name)

    instance_classes: dict[str, str] = {}
    for statement in ast.walk(tree):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        class_name = _python_name(statement.value.func).rsplit(".", 1)[-1]
        if class_name not in serialized_methods:
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
        for target in targets:
            if isinstance(target, ast.Name):
                instance_classes[target.id] = class_name

    module_values: dict[str, ast.AST] = {}
    for statement in ast.walk(tree):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        if statement.value is None:
            continue
        if _python_containing_function(statement, parents) is not None:
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
        for target in targets:
            if isinstance(target, ast.Name):
                module_values[target.id] = statement.value

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    asymmetric_helpers = {
        node.name
        for node in functions.values()
        if any(
            isinstance(call, ast.Call)
            and (
                _python_name(call.func).endswith(
                    (
                        "load_pem_private_key",
                        "load_pem_public_key",
                        "generate_private_key",
                    )
                )
                or _python_name(call.func).endswith(".public_key")
            )
            for returned in ast.walk(node)
            if isinstance(returned, ast.Return) and returned.value is not None
            for call in ast.walk(returned.value)
        )
    }

    def asymmetric_key_expression(expression: ast.AST, before: ast.AST) -> bool:
        if isinstance(expression, ast.Name):
            scope = _python_containing_function(before, parents)
            candidates = [
                node
                for node in ast.walk(scope if scope is not None else tree)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and node.value is not None
                and getattr(node, "lineno", 0) < getattr(before, "lineno", 10**9)
                and _python_containing_function(node, parents) is scope
                and expression.id
                in {
                    name
                    for target in (
                        node.targets if isinstance(node, ast.Assign) else (node.target,)
                    )
                    for name in _python_assignment_target_names(target)
                }
            ]
            if candidates:
                latest = max(candidates, key=lambda node: getattr(node, "lineno", 0))
                return asymmetric_key_expression(latest.value, latest)
            return False
        if isinstance(expression, ast.Attribute) and isinstance(
            expression.value, ast.Name
        ):
            return asymmetric_key_expression(expression.value, before)
        if isinstance(expression, ast.Call) and isinstance(
            expression.func, ast.Attribute
        ):
            receiver = expression.func.value
            if (
                isinstance(receiver, ast.Name)
                and expression.func.attr
                in serialized_methods.get(instance_classes.get(receiver.id, ""), set())
            ):
                return True
        return any(
            isinstance(call, ast.Call)
            and (
                _python_name(call.func).endswith(
                    (
                        "load_pem_private_key",
                        "load_pem_public_key",
                        "generate_private_key",
                    )
                )
                or _python_name(call.func).rsplit(".", 1)[-1] in asymmetric_helpers
                or _python_name(call.func).endswith(".public_key")
            )
            for call in ast.walk(expression)
        )

    key_parameters: dict[str, int] = {}
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        key_constructor = _python_name(call.func)
        if not call.args or not key_constructor.endswith(
            ("algorithms.AES", "AESGCM", "AESCCM", "ChaCha20Poly1305")
        ):
            continue
        key = call.args[0]
        if (
            isinstance(key, ast.Attribute)
            and isinstance(key.value, ast.Name)
            and key.value.id == "self"
        ):
            owner = _python_containing_class(call, parents)
            if owner is not None and key.attr not in class_attributes.get(owner, set()):
                return "AES key expression refers to an unavailable instance attribute"
        if asymmetric_key_expression(key, call):
            return "symmetric encryption receives an asymmetric key object instead of key bytes"
        function = _python_containing_function(call, parents)
        if isinstance(key, ast.Name):
            if function is not None:
                parameters = (
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                )
                parameter_names = [parameter.arg for parameter in parameters]
                if key.id in parameter_names:
                    preceding_assignments = [
                        node
                        for node in ast.walk(function)
                        if isinstance(node, (ast.Assign, ast.AnnAssign))
                        and _python_containing_function(node, parents) is function
                        and getattr(node, "lineno", 0) < getattr(call, "lineno", 0)
                        and key.id
                        in {
                            name
                            for target in (
                                node.targets
                                if isinstance(node, ast.Assign)
                                else (node.target,)
                            )
                            for name in _python_assignment_target_names(target)
                        }
                    ]
                    if preceding_assignments:
                        latest = max(
                            preceding_assignments,
                            key=lambda node: getattr(node, "lineno", 0),
                        )
                        if isinstance(latest.value, ast.Constant) and isinstance(
                            latest.value.value,
                            (str, int),
                        ):
                            return (
                                "algorithms.AES requires key bytes, not a text "
                                "or integer value"
                            )
                    else:
                        key_parameters[function.name] = parameter_names.index(key.id)
                else:
                    local_names = {
                        name
                        for node in ast.walk(function)
                        if isinstance(node, (ast.Assign, ast.AnnAssign))
                        and node.value is not None
                        for target in (
                            node.targets if isinstance(node, ast.Assign) else (node.target,)
                        )
                        for name in _python_assignment_target_names(target)
                    }
                    if key.id not in local_names and key.id not in module_values:
                        return "AES key expression refers to an unavailable value"

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Name) or call.func.id not in key_parameters:
            continue
        position = key_parameters[call.func.id]
        if position >= len(call.args):
            continue
        argument = call.args[position]
        if asymmetric_key_expression(argument, call):
            return "algorithms.AES receives serialized asymmetric key material"
        if isinstance(argument, ast.Name):
            argument = module_values.get(argument.id, argument)
        if isinstance(argument, ast.Constant) and isinstance(argument.value, (str, int)):
            return "AES key argument is text or an integer rather than key bytes"
    return None


def _python_containing_class(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.ClassDef | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.ClassDef):
            return current
        current = parents.get(current)
    return None


def _python_containing_function(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _python_assignment_target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {
            name
            for element in target.elts
            for name in _python_assignment_target_names(element)
        }
    return set()


def _python_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _python_name(node.value)
        return f"{head}.{node.attr}" if head else node.attr
    return ""


__all__ = ("symmetric_completion_failure",)
