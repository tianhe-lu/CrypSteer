"""Structural signature and native-original curation contracts."""

from __future__ import annotations

import ast
import re

from crypto_steering.validation.java.source import iter_methods, strip_java_comments
from crypto_steering.validation.java.lightweight_ast import (
    find_method_calls,
    method_containing_index,
)
from crypto_steering.validation.models import (
    Language,
    TaskValidationResult,
)
from crypto_steering.validation.source.index import (
    CallExpression,
    SourceAnalyzer,
    analyzer_for,
)
from crypto_steering.validation.source.helpers import _c_style_conditional_parts
from crypto_steering.validation.tasks.shared.typescript_api import node_crypto_calls
from crypto_steering.validation.tasks.shared.python_callables import callable_aliases
from crypto_steering.validation.tasks.python_pbkdf2 import (
    python_pbkdf2_route_failure,
)


def _native_scaffold_completion_contract(
    language: Language,
    source: str,
    task_id: str,
    result: TaskValidationResult,
) -> TaskValidationResult | None:
    """Apply native-original completion checks used only for curation."""

    if task_id.split("_", 1)[0].upper() == "F7" and not _kdf_output_is_consumed(
        language,
        source,
    ):
        return _native_scaffold_failure(
            result,
            "target_operation_sequence_invalid",
            "the completed PBKDF2 output is not consumed",
        )
    if task_id in {"F6_T2", "F6_T3"} and not _mac_integrity_decision_evidence(
        source
    ):
        return _native_scaffold_failure(
            result,
            "target_integrity_decision_missing",
            "MAC verification tasks require the completed MAC result to participate in an integrity decision",
        )
    return None


def _known_target_argument_contract_failure(
    language: Language,
    source: str,
    task_id: str,
) -> str | None:
    """Compatibility adapter for callers that still carry a T-style ID."""

    direction = (
        frozenset({"encrypt"})
        if task_id.endswith("T2")
        else frozenset({"decrypt"})
        if task_id.endswith("T3")
        else frozenset({"encrypt", "decrypt"})
    )
    return _family_argument_contract_failure(
        language,
        source,
        family=task_id.split("_", 1)[0],
        directions=direction,
    )


def _family_argument_contract_failure(
    language: Language,
    source: str,
    *,
    family: str,
    directions: frozenset[str] = frozenset(),
) -> str | None:
    """Return a definite structural failure in every usable family route.

    Python's frontend is intentionally parse-based, so the target validator
    must check required native-call signatures that compiled frontends reject
    themselves.  A statically proven buffer length is also part of structural
    usability when the selected API cannot initialize with that length.
    """

    family = family.upper()
    if language == "java" and family == "F4":
        return _java_signature_algorithm_failure(source)
    if language == "csharp" and family in {"F1", "F2"}:
        return _csharp_symmetric_transform_argument_failure(source, directions)
    if language == "csharp" and family == "F4":
        # The C# compiler accepts RSA calls after a key has been replaced with
        # public-only material.  That state is nevertheless structurally
        # incapable of completing SignData/SignHash, so Rule 5 must retain it.
        from crypto_steering.validation.tasks.signatures import (
            csharp_signature_argument_failure,
        )

        return csharp_signature_argument_failure(source)
    if language == "typescript" and family in {"F1", "F2"}:
        return _typescript_symmetric_transform_argument_failure(source, directions)
    if language == "typescript" and family == "F4":
        return _typescript_rsa_signature_argument_failure(source)
    if language == "typescript" and family == "F6":
        return _typescript_hmac_argument_failure(source)
    if language == "typescript" and family == "F7":
        return _typescript_callback_question_missing_query(source)
    if language != "python":
        return None
    if family == "F7":
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        unbound_failure = _python_unbound_instance_method_failure(
            tree,
            method_names={"derive_key", "generate_key", "pbkdf2"},
        )
        return unbound_failure or python_pbkdf2_route_failure(source)
    return _python_cryptography_argument_failure(
        source,
        families=frozenset({family}),
    )


def _java_signature_algorithm_failure(source: str) -> str | None:
    """Reject a source-proven non-signature selector on every JCA route."""

    try:
        analyzer = analyzer_for("java", source)
    except Exception:
        return None
    outcomes: list[str | None] = []
    for call in analyzer.calls("getInstance"):
        if "Signature" not in call.callee or not call.arguments:
            continue
        values = analyzer.resolve_strings(call.arguments[0], call.start).values
        invalid = tuple(value for value in values if "/" in value)
        outcomes.append(
            f"Signature.getInstance received non-signature selector {invalid[0]!r}"
            if values and len(invalid) == len(values)
            else None
        )
    if outcomes and all(reason is not None for reason in outcomes):
        return next(reason for reason in outcomes if reason is not None)
    return None


def _java_symmetric_argument_contract_failure(
    source: str,
    *,
    family: str,
) -> str | None:
    """Reject only when every completed JCA cipher route is unusable.

    The Java compiler checks argument types, but it cannot reject fixed AES
    key/IV sizes or a fixed non-block-aligned input passed to ``NoPadding``.
    These source-proven sizes are Rule 5 evidence; unresolved values remain
    acceptable under the benchmark evidence policy.
    """

    try:
        analyzer = analyzer_for("java", source)
    except Exception:
        return None

    initializations = tuple(
        call
        for call in analyzer.calls("init")
        if "." in call.callee and len(call.arguments) >= 2
    )
    terminals = tuple(
        call
        for call in analyzer.calls("doFinal")
        if "." in call.callee
    )
    outcomes: list[str | None] = []
    failed_in_function: dict[str | None, int] = {}
    for initialization in initializations:
        receiver = re.sub(r"\s+", "", initialization.callee).rsplit(".", 1)[0]
        transformation = _java_cipher_transformation(
            analyzer,
            receiver,
            initialization.start,
        )
        if transformation is None:
            continue
        compact_transformation = re.sub(
            r"[^A-Za-z0-9]", "", transformation
        ).upper()
        if family.upper() == "F2" and "CBC" not in compact_transformation:
            continue
        if family.upper() == "F1" and compact_transformation.startswith("RSA"):
            continue

        function = analyzer.containing_function_name(initialization.start)
        next_initialization = min(
            (
                candidate.start
                for candidate in initializations
                if candidate.start > initialization.start
                and analyzer.containing_function_name(candidate.start) == function
                and re.sub(r"\s+", "", candidate.callee).rsplit(".", 1)[0]
                == receiver
            ),
            default=len(source),
        )
        route_terminals = list(
            (terminal, terminal.arguments[0], terminal.start, terminal.end)
            for terminal in terminals
            if initialization.start < terminal.start < next_initialization
            and analyzer.containing_function_name(terminal.start) == function
            and re.sub(r"\s+", "", terminal.callee).rsplit(".", 1)[0]
            == receiver
            and terminal.arguments
        )
        route_terminals.extend(
            _java_cipher_helper_terminals(
                analyzer,
                initialization_start=initialization.start,
                next_initialization=next_initialization,
                receiver=receiver,
                terminals=terminals,
            )
        )
        if not route_terminals:
            continue

        failure = _java_cipher_initialization_size_failure(
            analyzer,
            initialization,
            transformation,
        )
        if failure is None and len(transformation.split("/")) > 3:
            failure = (
                "Cipher transformations accept algorithm, optional mode, and "
                f"optional padding components, not {transformation!r}"
            )
        if (
            failure is None
            and "NOPADDING" in compact_transformation
            and _java_requires_block_aligned_input(transformation)
        ):
            block_size = _java_cipher_block_size(transformation)
            fixed_lengths = {
                analyzer.static_byte_length(expression, expression_index)
                for _, expression, expression_index, _ in route_terminals
            }
            fixed_lengths.discard(None)
            if block_size is not None and fixed_lengths and all(
                length % block_size != 0 for length in fixed_lengths
            ):
                actual = min(fixed_lengths)
                failure = (
                    f"{transformation} requires block-aligned input, not a "
                    f"source-proven {actual}-byte value"
                )

        prior_failure = failed_in_function.get(function)
        if failure is None and prior_failure is not None:
            intervening = source[prior_failure:initialization.start]
            if not re.search(r"\bcatch\s*\(", intervening):
                failure = (
                    "the candidate cipher route follows a source-proven "
                    "failing cipher operation in the same execution path"
                )
        if failure is not None:
            failed_in_function[function] = max(
                failed_in_function.get(function, 0),
                max(completion_end for _, _, _, completion_end in route_terminals),
            )
        outcomes.append(failure)

    if outcomes and all(failure is not None for failure in outcomes):
        return next(failure for failure in outcomes if failure is not None)
    return None


def _java_cipher_helper_terminals(
    analyzer: SourceAnalyzer,
    *,
    initialization_start: int,
    next_initialization: int,
    receiver: str,
    terminals: tuple[CallExpression, ...],
) -> tuple[tuple[CallExpression, str, int, int], ...]:
    """Bind a Cipher passed to a one-level ``doFinal`` helper.

    This preserves caller evidence for Rule 5.  In particular, a fixed input
    passed through ``finish(cipher, data)`` remains source-proven at the
    terminal instead of becoming an unresolved helper parameter.
    """

    flow = getattr(analyzer, "flow", None)
    methods = tuple(getattr(flow, "methods", ()))
    caller = (
        method_containing_index(methods, initialization_start)
        if methods
        else None
    )
    if caller is None:
        return ()

    results: list[tuple[CallExpression, str, int, int]] = []
    compact_receiver = re.sub(r"\s+", "", receiver)
    for helper in methods:
        parameters = [parameter.name for parameter in helper.params]
        if not parameters:
            continue
        helper_terminals = tuple(
            terminal
            for terminal in terminals
            if helper.body_start <= terminal.start <= helper.body_end
            and terminal.arguments
        )
        if not helper_terminals:
            continue
        for invocation in find_method_calls(analyzer.source, helper.name):
            if not (
                initialization_start < invocation.start < next_initialization
                and caller.body_start <= invocation.start <= caller.body_end
                and len(invocation.arguments) == len(parameters)
            ):
                continue
            for terminal in helper_terminals:
                terminal_receiver = re.sub(r"\s+", "", terminal.callee).rsplit(
                    ".", 1
                )[0]
                if terminal_receiver not in parameters:
                    continue
                receiver_index = parameters.index(terminal_receiver)
                supplied_receiver = re.sub(
                    r"\s+", "", invocation.arguments[receiver_index].text
                )
                if supplied_receiver != compact_receiver:
                    continue
                expression = terminal.arguments[0]
                expression_index = terminal.start
                compact_expression = re.sub(r"\s+", "", expression)
                if compact_expression in parameters:
                    argument_index = parameters.index(compact_expression)
                    expression = invocation.arguments[argument_index].text
                    expression_index = invocation.start
                results.append(
                    (terminal, expression, expression_index, invocation.end)
                )
    return tuple(results)


def _java_cipher_transformation(
    analyzer: SourceAnalyzer,
    receiver: str,
    index: int,
) -> str | None:
    flow = getattr(analyzer, "flow", None)
    assignment = flow._latest_assignment(receiver, index) if flow is not None else None
    if assignment is None:
        return None
    match = re.search(
        r"\bCipher\s*\.\s*getInstance\s*\((?P<arguments>.*)\)",
        assignment.expression,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    arguments = _split_java_arguments(match.group("arguments"))
    if not arguments:
        return None
    resolved = analyzer.resolve_strings(arguments[0], assignment.start)
    return resolved.values[0] if len(resolved.values) == 1 else None


def _java_cipher_initialization_size_failure(
    analyzer: SourceAnalyzer,
    initialization: CallExpression,
    transformation: str,
) -> str | None:
    algorithm = transformation.split("/", 1)[0].strip().upper()
    key_algorithm = _java_secret_key_algorithm(
        analyzer,
        initialization.arguments[1],
        initialization.start,
        seen=frozenset(),
    )
    if (
        algorithm.startswith("PBE")
        and key_algorithm is not None
        and not key_algorithm.upper().startswith("PBE")
    ):
        return (
            f"{transformation} requires password-based key material, not a "
            f"{key_algorithm} SecretKeySpec"
        )
    if _java_key_material_uses_array_identity_string(
        analyzer,
        initialization.arguments[1],
        initialization.start,
        seen=frozenset(),
    ):
        return (
            "SecretKey.getEncoded().toString() returns an array identity "
            "string rather than the encoded key bytes"
        )
    key_length = _java_secret_key_length(
        analyzer,
        initialization.arguments[1],
        initialization.start,
        seen=frozenset(),
    )
    allowed_key_lengths = {
        "AES": {16, 24, 32},
        "DES": {8},
        "DESEDE": {16, 24},
        "TRIPLEDES": {16, 24},
    }.get(algorithm)
    if (
        allowed_key_lengths is not None
        and key_length is not None
        and key_length not in allowed_key_lengths
    ):
        return (
            f"{algorithm} requires a supported key length, not "
            f"{key_length} bytes"
        )

    if "CBC" not in transformation.upper() or len(initialization.arguments) < 3:
        return None
    if _java_iv_uses_uninitialized_cipher_parameters(
        analyzer,
        initialization.arguments[2],
        initialization.start,
        seen=frozenset(),
    ):
        return (
            "CBC IV material is read from an uninitialized Cipher's "
            "parameters"
        )
    iv_length = _java_iv_length(
        analyzer,
        initialization.arguments[2],
        initialization.start,
        seen=frozenset(),
    )
    required_iv = _java_cipher_block_size(transformation)
    if (
        required_iv is not None
        and iv_length is not None
        and iv_length != required_iv
    ):
        return (
            f"{transformation} requires a {required_iv}-byte IV, not "
            f"{iv_length} bytes"
        )
    return None


def _java_key_material_uses_array_identity_string(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
    *,
    seen: frozenset[str],
) -> bool:
    """Trace key bytes back to Java array-identity string conversion.

    Calling ``toString`` on the byte array returned by ``SecretKey.getEncoded``
    produces an object-identity string such as ``[B@...``. Encoding that text
    does not recover the key. The trace is deliberately bounded to local
    assignments, ``getBytes`` wrappers, and helper return expressions.
    """

    compact = re.sub(r"\s+", "", expression)
    marker = f"java-key-identity-string:{index}:{compact}"
    if marker in seen:
        return False
    next_seen = seen | frozenset({marker})

    if re.search(
        r"\.getEncoded\(\)\.toString\(\)",
        compact,
        flags=re.IGNORECASE,
    ):
        return True

    key_spec = re.search(
        r"\bnew\s+(?:[A-Za-z_$][\w$]*\.)*SecretKeySpec\s*"
        r"\((?P<arguments>.*)\)",
        expression,
        flags=re.DOTALL,
    )
    if key_spec is not None:
        arguments = _split_java_arguments(key_spec.group("arguments"))
        return bool(arguments) and _java_key_material_uses_array_identity_string(
            analyzer,
            arguments[0],
            index,
            seen=next_seen,
        )

    encoded_text = re.fullmatch(
        r"(?P<receiver>.+?)\.getBytes\s*\([^)]*\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if encoded_text is not None:
        return _java_key_material_uses_array_identity_string(
            analyzer,
            encoded_text.group("receiver"),
            index,
            seen=next_seen,
        )

    variable = re.fullmatch(r"[A-Za-z_$][\w$]*", compact)
    flow = getattr(analyzer, "flow", None)
    if variable is not None and flow is not None:
        assignment = flow._latest_assignment(variable.group(0), index)
        if assignment is not None:
            return _java_key_material_uses_array_identity_string(
                analyzer,
                assignment.expression,
                assignment.start,
                seen=next_seen,
            )

    helper_call = re.fullmatch(
        r"(?:this\.)?(?P<name>[A-Za-z_$][\w$]*)\s*\([^)]*\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if helper_call is None:
        return False
    outcomes: list[bool] = []
    for method in iter_methods(analyzer.source):
        if method.name != helper_call.group("name"):
            continue
        returns = tuple(
            match.group("expression").strip()
            for match in re.finditer(
                r"\breturn\s+(?P<expression>[^;]+);",
                strip_java_comments(method.text),
                flags=re.DOTALL,
            )
        )
        outcomes.extend(
            _java_key_material_uses_array_identity_string(
                analyzer,
                returned,
                index,
                seen=next_seen,
            )
            for returned in returns
        )
    return bool(outcomes) and all(outcomes)


def _java_iv_uses_uninitialized_cipher_parameters(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
    *,
    seen: frozenset[str],
) -> bool:
    """Recognize IV helpers that read parameters before cipher initialization."""

    compact = re.sub(r"\s+", "", expression)
    state = f"java-uninitialized-parameters:{index}:{compact}"
    if state in seen:
        return False
    next_seen = seen | frozenset({state})
    variable = re.fullmatch(r"[A-Za-z_$][\w$]*", compact)
    flow = getattr(analyzer, "flow", None)
    if variable is not None and flow is not None:
        assignment = flow._latest_assignment(variable.group(0), index)
        if assignment is not None:
            return _java_iv_uses_uninitialized_cipher_parameters(
                analyzer,
                assignment.expression,
                assignment.start,
                seen=next_seen,
            )

    call = re.fullmatch(
        r"(?P<name>[A-Za-z_$][\w$]*)\s*\(.*\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if call is None:
        return bool(
            re.search(
                r"\bCipher\s*\.\s*getInstance\s*\([^;]+\)\s*"
                r"\.\s*getParameters\s*\(",
                expression,
                flags=re.DOTALL,
            )
        )

    for method in iter_methods(analyzer.source):
        if method.name != call.group("name"):
            continue
        body = strip_java_comments(method.text)
        for assignment in re.finditer(
            r"\bCipher\s+(?P<receiver>[A-Za-z_$][\w$]*)\s*=\s*"
            r"Cipher\s*\.\s*getInstance\s*\([^;]+\)\s*;",
            body,
            flags=re.DOTALL,
        ):
            receiver = assignment.group("receiver")
            parameters = re.search(
                rf"\b{re.escape(receiver)}\s*\.\s*getParameters\s*\(",
                body[assignment.end() :],
            )
            if parameters is None:
                continue
            prefix = body[
                assignment.end() : assignment.end() + parameters.start()
            ]
            if re.search(
                rf"\b{re.escape(receiver)}\s*\.\s*init\s*\(",
                prefix,
            ) is None:
                return True
    return False


def _java_secret_key_algorithm(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
    *,
    seen: frozenset[str],
) -> str | None:
    """Resolve an explicitly constructed SecretKeySpec algorithm."""

    compact = re.sub(r"\s+", "", expression)
    marker = f"java-key-algorithm:{index}:{compact}"
    if marker in seen:
        return None
    match = re.search(
        r"\bnew\s+(?:[A-Za-z_$][\w$]*\.)*SecretKeySpec\s*"
        r"\((?P<arguments>.*)\)",
        expression,
        flags=re.DOTALL,
    )
    if match is not None:
        arguments = _split_java_arguments(match.group("arguments"))
        if len(arguments) < 2:
            return None
        resolved = analyzer.resolve_strings(arguments[1], index)
        return resolved.values[0] if len(resolved.values) == 1 else None
    variable = re.fullmatch(r"[A-Za-z_$][\w$]*", compact)
    flow = getattr(analyzer, "flow", None)
    if variable is not None and flow is not None:
        assignment = flow._latest_assignment(variable.group(0), index)
        if assignment is not None:
            return _java_secret_key_algorithm(
                analyzer,
                assignment.expression,
                assignment.start,
                seen=seen | frozenset({marker}),
            )
    return None


def _java_secret_key_length(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
    *,
    seen: frozenset[str],
) -> int | None:
    compact = re.sub(r"\s+", "", expression)
    marker = f"java-key-length:{index}:{compact}"
    if marker in seen:
        return None
    match = re.search(
        r"\bnew\s+(?:[A-Za-z_$][\w$]*\.)*SecretKeySpec\s*\((?P<arguments>.*)\)",
        expression,
        flags=re.DOTALL,
    )
    if match is not None:
        arguments = _split_java_arguments(match.group("arguments"))
        if not arguments:
            return None
        direct = analyzer.static_byte_length(arguments[0], index)
        if direct is not None:
            return direct
        return _java_secret_key_length(
            analyzer,
            arguments[0],
            index,
            seen=seen | frozenset({marker}),
        )
    variable = re.fullmatch(r"[A-Za-z_$][\w$]*", compact)
    flow = getattr(analyzer, "flow", None)
    if variable is not None and flow is not None:
        assignment = flow._latest_assignment(variable.group(0), index)
        if assignment is not None:
            return _java_secret_key_length(
                analyzer,
                assignment.expression,
                assignment.start,
                seen=seen | frozenset({marker}),
            )
    helper_call = re.fullmatch(
        r"(?:this\s*\.\s*)?(?P<name>[A-Za-z_$][\w$]*)\s*\(.*\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if helper_call is not None:
        lengths: set[int] = set()
        for method in iter_methods(analyzer.source):
            if method.name != helper_call.group("name"):
                continue
            body = strip_java_comments(method.text)
            for returned in re.finditer(
                r"\breturn\s+(?P<expression>[^;]+);",
                body,
                flags=re.DOTALL,
            ):
                return_index = method.body_start + 1 + returned.start("expression")
                length = analyzer.static_byte_length(
                    returned.group("expression").strip(),
                    return_index,
                )
                if length is not None:
                    lengths.add(length)
        if len(lengths) == 1:
            return next(iter(lengths))
    return None


def _java_iv_length(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
    *,
    seen: frozenset[str],
) -> int | None:
    compact = re.sub(r"\s+", "", expression)
    marker = f"java-iv-length:{index}:{compact}"
    if marker in seen:
        return None
    match = re.search(
        r"\bnew\s+(?:[A-Za-z_$][\w$]*\.)*IvParameterSpec\s*\((?P<arguments>.*)\)",
        expression,
        flags=re.DOTALL,
    )
    if match is not None:
        arguments = _split_java_arguments(match.group("arguments"))
        return (
            analyzer.static_byte_length(arguments[0], index)
            if arguments
            else None
        )
    variable = re.fullmatch(r"[A-Za-z_$][\w$]*", compact)
    flow = getattr(analyzer, "flow", None)
    if variable is not None and flow is not None:
        assignment = flow._latest_assignment(variable.group(0), index)
        if assignment is not None:
            return _java_iv_length(
                analyzer,
                assignment.expression,
                assignment.start,
                seen=seen | frozenset({marker}),
            )
    return None


def _java_cipher_block_size(transformation: str) -> int | None:
    algorithm = transformation.split("/", 1)[0].strip().upper()
    if algorithm in {"AES", "CAMELLIA", "SEED", "SM4"}:
        return 16
    if algorithm in {"DES", "DESEDE", "TRIPLEDES", "BLOWFISH", "RC2"}:
        return 8
    return None


def _java_requires_block_aligned_input(transformation: str) -> bool:
    """Return whether ``NoPadding`` leaves a block mode's alignment exposed."""

    parts = [part.strip().upper() for part in transformation.split("/")]
    return len(parts) >= 2 and parts[1] in {"ECB", "CBC", "PCBC"}


def _split_java_arguments(value: str) -> tuple[str, ...]:
    """Split the simple constructor calls used by Rule 5 size analysis."""

    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return tuple(parts)


def _python_cryptography_argument_failure(
    source: str,
    *,
    families: frozenset[str] | None,
) -> str | None:
    check_cipher = families is None or bool(families & {"F1", "F2", "F8"})
    # PBKDF2 may be the key-producing operation in symmetric-cipher tasks as
    # well as the target operation in F7/F8. A malformed call signature breaks
    # that required operation sequence even though Python can still parse it.
    check_kdf = families is None or bool(families & {"F1", "F2", "F8"})
    check_rsa = families is None or bool(families & {"F3", "F4"})
    check_hmac = families is None or bool(families & {"F6"})
    check_digest = families is None or bool(families & {"F5"})

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    analyzer = analyzer_for("python", source)

    cipher_construction_failure = _python_cipher_construction_failure(tree)
    if check_cipher and cipher_construction_failure is not None:
        return cipher_construction_failure

    inline_context_failure = _python_inline_cipher_context_failure(tree)
    if check_cipher and inline_context_failure is not None:
        return inline_context_failure

    if check_rsa:
        unavailable_rsa_context = re.search(
            r"\brsa\s*\.\s*(?P<name>Signer|Verifier)\s*\(",
            source,
        )
        if unavailable_rsa_context is not None:
            return (
                "cryptography RSA keys use direct sign and verify methods; "
                f"rsa.{unavailable_rsa_context.group('name')} is unavailable"
            )
        unavailable_mgf = re.search(
            r"\bpadding\s*\.\s*MGF(?!1\b)[A-Za-z0-9_]*\b",
            source,
        )
        if unavailable_mgf is not None:
            return "PyCA signature padding references an unavailable MGF member"
        mgf1_failure = _python_mgf1_construction_failure(tree)
        if mgf1_failure is not None:
            return mgf1_failure
        dsa_failure = _python_dsa_argument_failure(analyzer)
        if dsa_failure is not None:
            return dsa_failure
        serialization_failure = _python_key_serialization_argument_failure(
            analyzer
        )
        if serialization_failure is not None:
            return serialization_failure
        rsa_failure = _python_rsa_argument_failure(
            analyzer,
            family=next(iter(families & {"F3", "F4"}), None)
            if families is not None
            else None,
        )
        if rsa_failure is not None:
            return rsa_failure
    if check_hmac:
        hmac_failure = _python_hmac_argument_failure(analyzer)
        if hmac_failure is not None:
            return hmac_failure
    if check_kdf:
        unbound_failure = _python_unbound_instance_method_failure(
            tree,
            method_names={"derive_key", "generate_key", "pbkdf2"},
        )
        if unbound_failure is not None:
            return unbound_failure
        pbkdf2_argument_failure = _python_hashlib_pbkdf2_argument_failure(tree)
        if pbkdf2_argument_failure is not None:
            return pbkdf2_argument_failure

    context_names: set[str] = set()
    padding_context_names: set[str] = set()
    aead_names: set[str] = set()
    cipher_modes: dict[str, ast.Call] = {}
    kdf_names: set[str] = set()
    valid_digest_terminals = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"digest", "hexdigest"}
        and not node.args
        and not node.keywords
    )
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        names = {
            target.id
            for target in targets
            if isinstance(target, ast.Name)
        }
        if not isinstance(value, ast.Call):
            continue
        simple = _python_call_name(value.func).rsplit(".", 1)[-1]
        if simple == "Cipher" and len(value.args) >= 2:
            for name in names:
                if isinstance(value.args[1], ast.Call):
                    cipher_modes[name] = value.args[1]
        if simple in {"AESGCM", "AESCCM", "ChaCha20Poly1305"}:
            aead_names.update(names)
        if simple in {"encryptor", "decryptor"}:
            context_names.update(names)
        if simple in {"padder", "unpadder"}:
            padding_context_names.update(names)
        if simple == "PBKDF2HMAC":
            kdf_names.update(names)

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        call_name = _python_call_name(call.func)
        simple = call_name.rsplit(".", 1)[-1]
        if check_cipher:
            cipher_failure = _python_cipher_call_failure(
                call,
                call_name=call_name,
                tree=tree,
                analyzer=analyzer,
                aead_names=aead_names,
                cipher_modes=cipher_modes,
                cipher_context_names=context_names,
                padding_context_names=padding_context_names,
            )
            if cipher_failure is not None:
                return cipher_failure
        if (
            check_digest
            and simple in {"digest", "hexdigest"}
            and isinstance(call.func, ast.Attribute)
            and (call.args or call.keywords)
            and not any(
                terminal not in set(ast.walk(call))
                and call not in set(ast.walk(terminal))
                for terminal in valid_digest_terminals
            )
        ):
            return f"hash object {simple} does not accept message arguments"
        if check_kdf and simple == "PBKDF2HMAC" and any(
            keyword.arg == "password" for keyword in call.keywords
        ):
            return (
                "PBKDF2HMAC does not accept password in its constructor; "
                "key material must be passed to derive or verify"
            )
        if check_kdf and simple == "PBKDF2HMAC" and not any(
            isinstance(argument, ast.Starred) for argument in call.args
        ) and not any(keyword.arg is None for keyword in call.keywords):
            required = ("algorithm", "length", "salt", "iterations")
            supplied = set(required[: min(len(call.args), len(required))]) | {
                keyword.arg for keyword in call.keywords if keyword.arg is not None
            }
            missing = tuple(name for name in required if name not in supplied)
            if missing:
                return "PBKDF2HMAC omits required " + ", ".join(missing)
        if check_kdf and simple == "PBKDF2HMAC":
            salt = (
                call.args[2]
                if len(call.args) >= 3
                else next(
                    (
                        keyword.value
                        for keyword in call.keywords
                        if keyword.arg == "salt"
                    ),
                    None,
                )
            )
            if isinstance(salt, ast.Constant) and salt.value is None:
                return "PBKDF2HMAC requires bytes-like salt, not None"
        if (
            check_kdf
            and simple == "PBKDF2HMAC"
            and _python_pbkdf2_positional_order_is_invalid(call, tree)
        ):
            return (
                "PBKDF2HMAC positional arguments do not follow "
                "algorithm, length, salt, iterations order"
            )
        if (
            check_kdf
            and simple == "PBKDF2HMAC"
            and any(
                keyword.arg not in {
                    "algorithm",
                    "length",
                    "salt",
                    "iterations",
                    "backend",
                }
                for keyword in call.keywords
                if keyword.arg is not None
            )
        ):
            return "PBKDF2HMAC received an unsupported constructor keyword"
        if check_cipher and simple == "GCM" and not call.args and not any(
            keyword.arg == "initialization_vector" for keyword in call.keywords
        ):
            return "modes.GCM requires an initialization vector"
        if (
            check_kdf
            and simple == "derive"
            and not call.args
            and not call.keywords
            and isinstance(call.func, ast.Attribute)
            and (
                not isinstance(call.func.value, ast.Name)
                or call.func.value.id in kdf_names
            )
        ):
            return "PBKDF2HMAC.derive requires key material"
        if (
            check_kdf
            and simple == "derive"
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in kdf_names
        ):
            key_material = call.args[0] if call.args else next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == "key_material"
                ),
                None,
            )
            if key_material is not None and _python_expression_is_definitely_text(
                key_material,
                tree,
                call,
            ):
                return "PBKDF2HMAC.derive requires bytes-like key material, not text"
        if (
            not check_cipher
            or simple != "finalize"
            or not (call.args or call.keywords)
        ):
            continue
        if not isinstance(call.func, ast.Attribute):
            continue
        receiver = call.func.value
        is_cipher_context = bool(
            isinstance(receiver, ast.Name)
            and receiver.id in context_names
        ) or bool(
            isinstance(receiver, ast.Call)
            and _python_call_name(receiver.func).rsplit(".", 1)[-1]
            in {"encryptor", "decryptor"}
        )
        if is_cipher_context:
            return "PyCA cipher-context finalize does not accept arguments"
    return None


def _python_mgf1_construction_failure(tree: ast.AST) -> str | None:
    """Return a failure for source-proven unusable PyCA ``MGF1`` calls."""

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if _python_call_name(call.func).rsplit(".", 1)[-1] != "MGF1":
            continue
        named = [keyword.arg for keyword in call.keywords]
        if (
            len(call.args) != 1
            or named
        ) and not (
            not call.args
            and named == ["algorithm"]
        ):
            return "PyCA MGF1 accepts exactly one hash algorithm argument"
    return None


def _python_inline_cipher_context_failure(tree: ast.AST) -> str | None:
    """Reject update/finalize chains made from different transient contexts."""

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    by_statement: dict[ast.stmt, list[ast.Call]] = {}
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Attribute) or call.func.attr not in {
            "encryptor",
            "decryptor",
        }:
            continue
        current: ast.AST = call
        while current in parents and not isinstance(current, ast.stmt):
            current = parents[current]
        if isinstance(current, ast.stmt):
            by_statement.setdefault(current, []).append(call)

    for statement, factories in by_statement.items():
        operations = {
            node.func.attr
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"update", "finalize"}
        }
        if len(factories) > 1 and operations == {"update", "finalize"}:
            return "cipher update and finalize use different transient cipher contexts"
        for call in (
            node
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "finalize"
        ):
            receiver = call.func.value
            if (
                isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Attribute)
                and receiver.func.attr == "update"
            ):
                return "cipher finalize is invoked on update output, not the cipher context"
    return None


def _python_cipher_call_failure(
    call: ast.Call,
    *,
    call_name: str,
    tree: ast.AST,
    analyzer: SourceAnalyzer,
    aead_names: set[str],
    cipher_modes: dict[str, ast.Call],
    cipher_context_names: set[str],
    padding_context_names: set[str],
) -> str | None:
    """Return definite PyCA symmetric-cipher API-shape failures."""

    simple = call_name.rsplit(".", 1)[-1]
    compact = call_name.replace(" ", "")
    if compact in {
        "modes.EAX",
        "padding.AES_GCM",
        "padding.AESGCM",
        "padding.AESGCMHash",
        "padding.MGF1",
        "padding.MGFv1",
        "padding.OAEP",
        "keywrap.KeyWrap",
    }:
        return f"{compact} is not an available symmetric-cipher API"
    if simple == "AESGCM":
        if not _python_ast_call_supplies_parameters(call, ("key",)):
            return "AESGCM construction requires only the key argument"

    if simple in {"CBC", "CTR", "GCM"}:
        allowed_keywords = (
            {"initialization_vector", "tag", "min_tag_length"}
            if simple == "GCM"
            else {"initialization_vector"}
        )
        unsupported = {
            keyword.arg
            for keyword in call.keywords
            if keyword.arg is not None and keyword.arg not in allowed_keywords
        }
        if unsupported:
            return (
                f"modes.{simple} received unsupported keyword "
                f"{sorted(unsupported)[0]}"
            )
        iv = call.args[0] if call.args else next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg in {"initialization_vector", "nonce", "iv"}
            ),
            None,
        )
        if iv is not None and _python_expression_is_definitely_text(iv, tree, call):
            return f"modes.{simple} requires bytes-like nonce or IV material, not text"
        required_length = (
            _python_mode_required_length(call, tree)
            if simple in {"CBC", "CTR"}
            else None
        )
        if iv is not None and required_length is not None:
            node_span = getattr(analyzer, "_node_span", None)
            call_start = node_span(call)[0] if callable(node_span) else 0
            actual_length = analyzer.static_byte_length(
                ast.unparse(iv),
                call_start,
            )
            if actual_length is not None and actual_length != required_length:
                return (
                    f"modes.{simple} requires a {required_length}-byte value "
                    f"for the selected block cipher, not {actual_length} bytes"
                )

    if simple == "Cipher" and call.args and isinstance(call.args[0], ast.Call):
        algorithm = _python_call_name(call.args[0].func).rsplit(".", 1)[-1]
        if algorithm in {"AESGCM", "AESCCM", "ChaCha20Poly1305", "Fernet"}:
            return (
                f"{algorithm} is a complete one-shot cipher, not an algorithm "
                "object accepted by Cipher"
            )

    if simple in {"AES", "AES128", "AES256"} and (
        compact.endswith(f"algorithms.{simple}")
        or _python_imported_call_is(
            analyzer,
            call_name,
            f"cryptography.hazmat.primitives.ciphers.algorithms.{simple}",
        )
    ):
        key = call.args[0] if call.args else next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg == "key"
            ),
            None,
        )
        if key is None:
            return f"algorithms.{simple} requires key bytes"
        value = _python_resolve_constant(key, tree, call)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return f"algorithms.{simple} requires key bytes, not text or an integer"
        node_span = getattr(analyzer, "_node_span", None)
        call_start = node_span(call)[0] if callable(node_span) else 0
        actual_length = analyzer.static_byte_length(ast.unparse(key), call_start)
        allowed_lengths = {
            "AES": {16, 24, 32},
            "AES128": {16},
            "AES256": {32},
        }[simple]
        if actual_length is not None and actual_length not in allowed_lengths:
            expected = ", ".join(str(value) for value in sorted(allowed_lengths))
            return (
                f"algorithms.{simple} requires a {expected}-byte key, not "
                f"{actual_length} bytes"
            )

    if isinstance(call.func, ast.Attribute):
        receiver = _python_call_name(call.func.value)
        if receiver in aead_names and simple in {"encrypt", "decrypt"}:
            if not _python_ast_call_supplies_parameters(
                call,
                ("nonce", "data", "associated_data"),
            ):
                return (
                    f"AES AEAD {simple} requires nonce, data, and associated-data arguments"
                )
        if receiver in cipher_context_names and simple == "finalize" and (
            call.args or call.keywords
        ):
            return "PyCA cipher-context finalize does not accept arguments"
        if receiver in padding_context_names and simple == "finalize" and (
            call.args or call.keywords
        ):
            return "PyCA padding-context finalize does not accept arguments"
        if receiver in padding_context_names and simple in {"padder", "unpadder"}:
            return (
                "padding contexts do not create nested padder or unpadder "
                "contexts"
            )
        if receiver in cipher_modes and simple == "verify":
            return "Cipher objects do not provide a verify operation"
    return None


def _python_mode_required_length(call: ast.Call, tree: ast.AST) -> int | None:
    """Return the block-sized mode argument required by its Cipher algorithm."""

    for cipher in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _python_call_name(node.func).rsplit(".", 1)[-1] == "Cipher"
    ):
        parameters = ("algorithm", "mode", "backend")
        bound = {name: value for name, value in zip(parameters, cipher.args)}
        bound.update(
            {
                keyword.arg: keyword.value
                for keyword in cipher.keywords
                if keyword.arg is not None
            }
        )
        if bound.get("mode") is not call:
            continue
        algorithm = bound.get("algorithm")
        if not isinstance(algorithm, ast.Call):
            return None
        name = _python_call_name(algorithm.func).rsplit(".", 1)[-1]
        if name in {"AES", "AES128", "AES256", "Camellia", "SEED", "SM4"}:
            return 16
        if name in {"TripleDES", "Blowfish", "CAST5", "IDEA"}:
            return 8
    return None


def _python_ast_call_supplies_parameters(
    call: ast.Call,
    parameters: tuple[str, ...],
) -> bool:
    """Return whether an AST call binds each required parameter exactly once."""

    if len(call.args) > len(parameters):
        return False
    named = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
    if any(keyword.arg is None for keyword in call.keywords):
        return False
    allowed = set(parameters)
    if named - allowed:
        return False
    positional = set(parameters[: len(call.args)])
    return not positional & named and positional | named == allowed


def _python_imported_call_is(
    analyzer: SourceAnalyzer,
    call_name: str,
    expected: str,
) -> bool:
    """Resolve the imported head of an AST call without executing source."""

    head, separator, tail = call_name.partition(".")
    canonical = getattr(analyzer, "import_aliases", {}).get(head, head)
    if separator:
        canonical += "." + tail
    return canonical == expected


def _python_resolve_constant(
    expression: ast.AST,
    tree: ast.AST,
    call: ast.Call,
) -> object | None:
    if isinstance(expression, ast.Constant):
        return expression.value
    if not isinstance(expression, ast.Name):
        return None
    best: tuple[int, object] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or getattr(node, "lineno", 0) >= getattr(
            call, "lineno", 0
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if any(isinstance(target, ast.Name) and target.id == expression.id for target in targets):
            best = (getattr(node, "lineno", 0), value.value)
    return best[1] if best is not None else None


def _python_expression_is_definitely_text(
    expression: ast.AST,
    tree: ast.AST,
    call: ast.Call,
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Recognize only statically certain text passed to a bytes-only API."""

    if isinstance(expression, ast.Constant):
        return isinstance(expression.value, str)
    if isinstance(expression, ast.Call):
        name = _python_call_name(expression.func)
        simple = name.rsplit(".", 1)[-1]
        return name == "getpass.getpass" or simple in {"input", "str"}
    if isinstance(expression, ast.Subscript):
        return _python_expression_is_definitely_text(
            expression.value,
            tree,
            call,
            seen,
        )
    if not isinstance(expression, ast.Name):
        return False
    already_seen = expression.id in seen

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def scope(node: ast.AST) -> ast.AST:
        current: ast.AST | None = node
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                return current
            current = parents.get(current)
        return tree

    call_scope = scope(call)
    best: tuple[int, ast.AST] | None = None
    call_line = getattr(call, "lineno", 0)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if scope(node) is not call_scope:
            continue
        line = getattr(node, "lineno", 0)
        if line >= call_line:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not already_seen and any(
            isinstance(target, ast.Name) and target.id == expression.id
            for target in targets
        ) and (best is None or line > best[0]):
            best = (line, node.value)
    if best is not None:
        return _python_expression_is_definitely_text(
            best[1],
            tree,
            call,
            seen | frozenset({expression.id}),
        )

    if isinstance(call_scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parameters = (*call_scope.args.posonlyargs, *call_scope.args.args)
        parameter_names = [parameter.arg for parameter in parameters]
        if expression.id in parameter_names:
            position = parameter_names.index(expression.id)
            marker = f"parameter:{call_scope.name}:{expression.id}"
            if marker in seen:
                return False
            supplied_values: list[tuple[ast.AST, ast.Call]] = []
            for candidate in ast.walk(tree):
                if not (
                    isinstance(candidate, ast.Call)
                    and _python_call_name(candidate.func).rsplit(".", 1)[-1]
                    == call_scope.name
                ):
                    continue
                supplied: ast.AST | None = (
                    candidate.args[position]
                    if position < len(candidate.args)
                    else next(
                        (
                            keyword.value
                            for keyword in candidate.keywords
                            if keyword.arg == expression.id
                        ),
                        None,
                    )
                )
                if supplied is not None:
                    supplied_values.append((supplied, candidate))
            return bool(supplied_values) and all(
                _python_expression_is_definitely_text(
                    supplied,
                    tree,
                    candidate,
                    seen | frozenset({marker}),
                )
                for supplied, candidate in supplied_values
            )

    return False


def _python_unbound_instance_method_failure(
    tree: ast.AST,
    *,
    method_names: set[str],
) -> str | None:
    """Find a definite ``Class.method(...)`` call that omits ``self``."""

    classes = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Attribute) or not isinstance(
            call.func.value, ast.Name
        ):
            continue
        class_node = classes.get(call.func.value.id)
        if class_node is None or call.func.attr not in method_names:
            continue
        method = next(
            (
                node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == call.func.attr
            ),
            None,
        )
        if method is None or any(
            isinstance(decorator, ast.Name)
            and decorator.id in {"staticmethod", "classmethod"}
            for decorator in method.decorator_list
        ):
            continue
        positional_parameters = tuple(method.args.posonlyargs) + tuple(method.args.args)
        required = len(positional_parameters) - len(method.args.defaults)
        supplied = len(call.args) + len(call.keywords)
        if (
            positional_parameters
            and positional_parameters[0].arg in {"self", "cls"}
            and supplied < required
        ):
            return (
                f"{class_node.name}.{method.name} is called without the required "
                "instance argument"
            )
    return None


def _native_scaffold_failure(
    result: TaskValidationResult,
    reason_code: str,
    reason: str,
) -> TaskValidationResult:
    return TaskValidationResult(
        language=result.language,
        task_id=result.task_id,
        success=False,
        reason_codes=(reason_code,),
        reason=reason,
        matched_operations=result.matched_operations,
    )


def _mac_integrity_decision_evidence(source: str) -> bool:
    direct = re.search(
        r"(?:MessageDigest\s*\.\s*isEqual|Arrays\s*\.\s*equals|"
        r"CryptographicOperations\s*\.\s*FixedTimeEquals|timingSafeEqual|"
        r"(?:hmac|secrets)\s*\.\s*compare_digest|\.\s*verify\s*\(|"
        r"\.\s*SequenceEqual\s*\(|"
        r"\b(?:constant[_$]?time|fixed[_$]?time)[A-Za-z0-9_$]*\s*\(|"
        r"\b[A-Za-z0-9_$]*(?:constant[_$]?time|fixed[_$]?time)"
        r"[A-Za-z0-9_$]*\s*\.\s*(?:equals?|compare)\s*\(|"
        r"\breturn\s+[A-Za-z_$][\w$]*\s*\.\s*(?:equals?|compare)\s*\()",
        source,
        flags=re.IGNORECASE,
    )
    if direct is not None:
        return True
    if re.search(
        r"\b[A-Za-z_$][\w$]*\s*\[[^\]]+\]\s*(?:==|!=|===|!==)\s*"
        r"[A-Za-z_$][\w$]*\s*\[[^\]]+\]",
        source,
    ) and re.search(
        r"(?:\bMac\s*\.\s*getInstance|\bcreateHmac\s*\(|"
        r"\bnew\s+HMAC|\bhmac\s*\.\s*new\s*\()",
        source,
        flags=re.IGNORECASE,
    ):
        return True
    return bool(
        _security_comparison_helper_match(source)
        or
        re.search(
            r"\b(?:expected|actual|computed|provided|received)?\s*"
            r"(?:mac|tag|digest|signature|auth(?:entication)?[_$]?code)\w*\s*"
            r"(?:===|!==|==|!=|\.\s*equals\s*\()",
            source,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"(?:===|!==|==|!=)\s*\b(?:expected|actual|computed|provided|received)?\s*"
            r"(?:mac|tag|digest|signature|auth(?:entication)?[_$]?code)\w*\b",
            source,
            flags=re.IGNORECASE,
        )
    )


def _security_comparison_helper_match(source: str) -> re.Match[str] | None:
    """Find helper, operator, or method comparisons of integrity values."""

    value = r"(?:hash|digest|md5|sha\d*|mac|tag|signature|auth(?:entication)?[_$]?code)"
    helper = re.search(
        r"\b(?:[A-Za-z_][A-Za-z0-9_]*)?"
        r"(?:compare|equal|match|verify)[A-Za-z0-9_]*\s*\(\s*"
        rf"(?=[^\n;]*{value}[^\n;]*{value})[^\n;]*\)",
        source,
        flags=re.IGNORECASE,
    )
    if helper is not None:
        return helper
    operator = re.search(
        rf"(?=[^\n;]*{value}[^\n;]*{value})"
        r"[^\n;]*(?:===|!==|==|!=)[^\n;]*",
        source,
        flags=re.IGNORECASE,
    )
    if operator is not None:
        return operator
    method = re.search(
        rf"[^\n;]*{value}[^\n;]*\.\s*(?:Equals|SequenceEqual)\s*\("
        rf"[^\n;]*{value}[^\n;]*\)",
        source,
        flags=re.IGNORECASE,
    )
    if method is not None:
        return method
    return re.search(
        rf"\b(?:string\s*\.\s*)?Equals\s*\([^\n;]*{value}[^\n;]*\)",
        source,
        flags=re.IGNORECASE,
    )


def _kdf_output_is_consumed(language: Language, source: str) -> bool:
    """Require an observable use after assigning completed KDF output."""

    if language == "python":
        return _python_kdf_output_is_consumed(source)
    if language in {"typescript", "csharp"}:
        return _c_style_kdf_output_is_consumed(language, source)
    return _java_kdf_output_is_consumed(source)


def _c_style_kdf_output_is_consumed(language: Language, source: str) -> bool:
    analyzer = analyzer_for(language, source)
    call_names = (
        ("pbkdf2", "pbkdf2Sync")
        if language == "typescript"
        else ("Pbkdf2", "GetBytes")
    )
    calls = analyzer.calls(*call_names)
    for call in calls:
        if _c_style_result_reaches_consumer(
            analyzer,
            source,
            call.start,
            call.end,
            visited_functions=set(),
        ):
            return True
    if language == "typescript":
        direct_return = re.search(
            r"\breturn\s+(?:[A-Za-z_$][\w$]*\s*\.\s*)?pbkdf2(?:Sync)?\s*\(",
            source,
        )
        alias_return = re.search(
            r"\b(?:const|let|var)\s+(?P<alias>[A-Za-z_$][\w$]*)\s*=\s*"
            r"(?:[A-Za-z_$][\w$]*\s*\.\s*)?pbkdf2(?:Sync)?\s*;"
            r"(?:(?!\b(?:const|let|var)\s+(?P=alias)\b).)*?"
            r"\breturn\s+(?P=alias)\s*\(",
            source,
            flags=re.DOTALL,
        )
        if direct_return or alias_return:
            return True
    return False


def _c_style_result_reaches_consumer(
    analyzer: SourceAnalyzer,
    source: str,
    start: int,
    end: int,
    *,
    visited_functions: set[str],
) -> bool:
    """Trace a returned KDF result through small local wrapper chains."""

    assignments = tuple(getattr(analyzer, "assignments", ()))
    containing_assignments = tuple(
        item
        for item in assignments
        if item.start <= start and end <= item.end
    )
    assignment = (
        min(containing_assignments, key=lambda item: item.end - item.start)
        if containing_assignments
        else None
    )
    if assignment is not None:
        name = assignment.name.rsplit(".", 1)[-1]
        scope_end = _c_style_scope_end(analyzer, start, len(source))
        containing = analyzer.containing_function_name(start)
        function = next(
            (
                item
                for item in getattr(analyzer, "functions", ())
                if item.name == containing and item.start <= start <= item.end
            ),
            None,
        )
        returned = bool(
            function is not None
            and any(
                re.search(
                    rf"(?<![\w$]){re.escape(name)}(?![\w$])",
                    returned.expression,
                )
                for returned in function.returns
            )
        )
        if _identifier_is_used_after(source[:scope_end], name, assignment.end):
            if not returned:
                return True
            body_after = source[assignment.end:scope_end]
            without_returns = re.sub(
                rf"\breturn\s+[^;]*\b{re.escape(name)}\b[^;]*;",
                "",
                body_after,
                flags=re.IGNORECASE,
            )
            if re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", without_returns):
                return True
            return _c_style_returned_function_result_is_consumed(
                analyzer,
                source,
                containing,
                visited_functions,
            )
        # Storing an otherwise unused result is normally sufficient.  It is
        # not sufficient when a value-returning KDF wrapper explicitly returns
        # a different value in place of the derived bytes.
        if function is not None and function.returns:
            return returned
        return True

    statement_start = max(
        source.rfind(";", 0, start),
        source.rfind("{", 0, start),
        source.rfind("}", 0, start),
        source.rfind("\n", 0, start),
    ) + 1
    prefix = source[statement_start:start]
    # A call nested in another expression is consumed even when the lightweight
    # assignment index has no enclosing assignment record.  This covers return
    # object properties and calls inside formatting/printing expressions while
    # keeping a bare ``factory.generateSecret(spec);`` classified as discarded.
    if (
        re.search(r"\b(?:return|yield)\b", prefix)
        or re.search(r"(?<![=!<>])=(?!=)", prefix)
        or prefix.count("(") > prefix.count(")")
        or re.search(r"(?:^|[,\n])\s*[A-Za-z_$][\w$]*\s*:\s*$", prefix)
    ):
        return True
    if re.match(r"\s*\.\s*length\b", source[end:]):
        return True
    if re.search(r"\breturn\b", prefix):
        return _c_style_returned_function_result_is_consumed(
            analyzer,
            source,
            analyzer.containing_function_name(start),
            visited_functions,
        )
    if re.match(
        r"\s*\.\s*(?:length\b|then\s*\(|pipe\s*\(|map\s*\(|flatMap\s*\()",
        source[end:],
    ):
        return True
    return bool(
        re.search(
            r"(?:\+=|-=|\*=|/=)\s*$|"
            r"\b(?:yield|print|println|WriteLine|log|write)\b|"
            r"\b(?:SecretKeySpec|createHmac|HashData)\s*\([^)]*$",
            prefix,
            flags=re.IGNORECASE,
        )
    )


def _c_style_returned_function_result_is_consumed(
    analyzer: SourceAnalyzer,
    source: str,
    function_name: str | None,
    visited_functions: set[str],
) -> bool:
    if function_name is None:
        return True
    if function_name in visited_functions:
        return False
    visited = visited_functions | {function_name}
    callsites = tuple(
        call
        for call in analyzer.calls(function_name)
        if analyzer.containing_function_name(call.start) != function_name
        and not re.search(
            r"\b(?:crypto|nodeCrypto|subtle|Rfc2898DeriveBytes)\s*\.\s*$",
            call.callee.rsplit(function_name, 1)[0],
            flags=re.IGNORECASE,
        )
    )
    if not callsites:
        # A reusable function that returns the derived key completes the task
        # contract even when the source contains no demonstration caller.
        return True
    return any(
        _c_style_result_reaches_consumer(
            analyzer,
            source,
            call.start,
            call.end,
            visited_functions=visited,
        )
        for call in callsites
    )


def _c_style_scope_end(analyzer: SourceAnalyzer, position: int, default: int) -> int:
    scopes = tuple(
        function
        for function in getattr(analyzer, "functions", ())
        if function.start <= position <= function.end
    )
    return min(scopes, key=lambda item: item.end - item.start).end if scopes else default


def _python_kdf_output_is_consumed(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    pbkdf_aliases = {"pbkdf2_hmac"}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Attribute):
            continue
        if _python_call_name(value).endswith("hashlib.pbkdf2_hmac"):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            pbkdf_aliases.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        name = _python_call_name(call.func).rsplit(".", 1)[-1]
        if name not in {"derive", "derive_key", *pbkdf_aliases}:
            continue
        current: ast.AST = call
        parent = parents.get(current)
        while isinstance(
            parent,
            (
                ast.Await,
                ast.Attribute,
                ast.BinOp,
                ast.BoolOp,
                ast.Compare,
                ast.Dict,
                ast.FormattedValue,
                ast.IfExp,
                ast.JoinedStr,
                ast.List,
                ast.Set,
                ast.Slice,
                ast.Starred,
                ast.Subscript,
                ast.Tuple,
                ast.UnaryOp,
            ),
        ):
            current = parent
            parent = parents.get(current)
        if isinstance(parent, ast.Return):
            return True
        if isinstance(parent, ast.Call):
            return True
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            return True
    return False


def _python_hashlib_pbkdf2_argument_failure(tree: ast.AST) -> str | None:
    """Reject a definite non-name digest in a native ``pbkdf2_hmac`` call."""

    # A bare name is native only when it was imported from hashlib or assigned
    # from hashlib.pbkdf2_hmac.  Generated code may define a local wrapper with
    # the same name and its own keyword contract.
    aliases: set[str] = set()
    assignments: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "hashlib":
            aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "pbkdf2_hmac"
            )
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for target in targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = node.value
                if _python_call_name(node.value).endswith("hashlib.pbkdf2_hmac"):
                    aliases.add(target.id)

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        callee = _python_call_name(call.func)
        is_native = callee == "hashlib.pbkdf2_hmac" or (
            isinstance(call.func, ast.Name) and call.func.id in aliases
        )
        if not is_native:
            continue
        parameters = ("hash_name", "password", "salt", "iterations", "dklen")
        keyword_names = {keyword.arg for keyword in call.keywords if keyword.arg}
        unknown = keyword_names - set(parameters)
        if unknown:
            return (
                "hashlib.pbkdf2_hmac received unsupported keyword "
                + ", ".join(sorted(unknown))
            )
        if len(call.args) > len(parameters):
            return "hashlib.pbkdf2_hmac accepts at most five arguments"
        duplicate = set(parameters[: len(call.args)]) & keyword_names
        if duplicate:
            return (
                "hashlib.pbkdf2_hmac supplies "
                + ", ".join(sorted(duplicate))
                + " more than once"
            )
        digest = (
            call.args[0]
            if call.args
            else next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == "hash_name"
                ),
                None,
            )
        )
        if digest is None:
            continue
        if isinstance(digest, ast.Name):
            digest = assignments.get(digest.id, digest)
        if _python_expression_is_hash_constructor(digest):
            return (
                "hashlib.pbkdf2_hmac requires a digest name string as its "
                "first argument, not a hash constructor or object"
            )
        wrapper_failure = _python_forwarded_pbkdf2_digest_failure(
            tree,
            call,
            digest,
        )
        if wrapper_failure is not None:
            return wrapper_failure
    return None


def _python_forwarded_pbkdf2_digest_failure(
    tree: ast.AST,
    native_call: ast.Call,
    digest: ast.AST,
) -> str | None:
    """Check a local wrapper parameter forwarded to hashlib.pbkdf2_hmac."""

    if not isinstance(digest, ast.Name):
        return None
    wrapper = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and native_call in set(ast.walk(node))
        ),
        None,
    )
    if wrapper is None:
        return None
    parameters = tuple(argument.arg for argument in wrapper.args.args)
    if digest.id not in parameters:
        return None
    parameter_index = parameters.index(digest.id)
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if call is native_call or _python_call_name(call.func) != wrapper.name:
            continue
        supplied: ast.AST | None = None
        if parameter_index < len(call.args):
            supplied = call.args[parameter_index]
        else:
            supplied = next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == digest.id
                ),
                None,
            )
        if supplied is not None and _python_expression_is_hash_constructor(supplied):
            return (
                "hashlib.pbkdf2_hmac requires a digest name string as its "
                "first argument, not a hash constructor or object"
            )
    return None


def _python_cipher_construction_failure(tree: ast.AST) -> str | None:
    """Reject a PyCA Cipher setup only when every candidate is malformed."""

    calls = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _python_call_name(node.func).rsplit(".", 1)[-1] == "Cipher"
    )
    if not calls:
        return None
    failures = tuple(_python_cipher_setup_failure(call) for call in calls)
    if any(failure is None for failure in failures):
        return None
    return next(failure for failure in failures if failure is not None)


def _python_cipher_setup_failure(call: ast.Call) -> str | None:
    parameters = ("algorithm", "mode", "backend")
    bound: dict[str, ast.AST] = {
        name: value for name, value in zip(parameters, call.args)
    }
    bound.update(
        {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }
    )
    if "algorithm" not in bound:
        return "Cipher construction is missing its algorithm argument"
    if "mode" not in bound:
        return "Cipher construction is missing its mode argument"

    mode = bound["mode"]
    if isinstance(mode, ast.Call):
        name = _python_call_name(mode.func)
        available_modes = {
            "CBC",
            "CFB",
            "CFB8",
            "CTR",
            "ECB",
            "GCM",
            "OFB",
            "XTS",
        }
        if name.startswith("modes."):
            if name.rsplit(".", 1)[-1] in available_modes:
                return None
            return f"{name} is not an available PyCA cipher mode"
        if name.startswith(("algorithms.", "padding.")):
            return "Cipher requires an instantiated mode object"
        return None
    if isinstance(mode, ast.Attribute):
        name = _python_call_name(mode)
        if name.startswith(("algorithms.", "modes.", "padding.")):
            return "Cipher requires an instantiated mode object"
    if isinstance(mode, ast.Constant) and mode.value is None:
        algorithm = bound["algorithm"]
        if (
            isinstance(algorithm, ast.Call)
            and _python_call_name(algorithm.func).endswith("algorithms.AES")
        ):
            return "AES Cipher construction requires a mode object"
    return None


def _python_expression_is_hash_constructor(expression: ast.AST) -> bool:
    value = expression.func if isinstance(expression, ast.Call) else expression
    name = _python_call_name(value)
    return bool(
        re.fullmatch(
            r"(?:hashlib\.)?(?:md5|sha1|sha224|sha256|sha384|sha512|"
            r"sha3_224|sha3_256|sha3_384|sha3_512|blake2b|blake2s)",
            name,
            flags=re.IGNORECASE,
        )
    )


def _python_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _python_call_name(node.value)
        return f"{head}.{node.attr}" if head else node.attr
    return ""


def _python_name_is_loaded_after(tree: ast.AST, name: str, line: int) -> bool:
    return any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == name
        and getattr(node, "lineno", 0) > line
        for node in ast.walk(tree)
    )


def _java_kdf_output_is_consumed(source: str) -> bool:
    """Trace Java ``generateSecret`` results within their declaring method.

    The shared lightweight source index does not expose Java assignments.  A
    source-wide identifier search therefore mistakes a later caller variable
    for consumption of a disconnected local result.  Keeping this trace
    method-local also handles chained ``SecretKeyFactory`` calls.
    """

    methods = iter_methods(source)
    bodies = tuple(method.text for method in methods) or (source,)
    return any(_java_method_consumes_generate_secret(body) for body in bodies)


def _java_method_consumes_generate_secret(body: str) -> bool:
    searchable = strip_java_comments(body)
    for call in re.finditer(r"\.\s*generateSecret\s*\(", searchable):
        statement_start = max(
            searchable.rfind(";", 0, call.start()),
            searchable.rfind("{", 0, call.start()),
            searchable.rfind("}", 0, call.start()),
        ) + 1
        statement_end = searchable.find(";", call.end())
        if statement_end < 0:
            statement_end = len(searchable)
        statement = searchable[statement_start:statement_end]
        call_offset = call.start() - statement_start

        if re.match(r"\s*return\b", statement):
            return True
        if _java_call_is_nested_in_consumer(statement, call_offset):
            return True

        assignment = re.match(
            r"\s*(?:final\s+)?(?:[\w.$<>?,\[\]]+\s+)?"
            r"(?P<name>[A-Za-z_$][\w$]*)\s*(?:\[\s*\])?\s*=",
            statement,
        )
        if assignment is None:
            # Calling generateSecret is itself the completed derivation
            # operation.  A bare call in a void/demo route is usable even if
            # the example does not subsequently display the returned object.
            return True
        name = assignment.group("name")
        if _identifier_is_used_after(searchable, name, statement_end + 1):
            return True
    return False


def _java_call_is_nested_in_consumer(statement: str, call_offset: int) -> bool:
    """Return whether the target call is an argument to an enclosing call."""

    depth = 0
    for character in statement[:call_offset]:
        if character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
    return depth > 0


def _identifier_is_used_after(source: str, name: str, index: int) -> bool:
    if not re.fullmatch(r"[A-Za-z_$][\w$]*", name):
        return False
    return re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", source[index:]) is not None


def _csharp_symmetric_transform_argument_failure(
    source: str,
    directions: frozenset[str],
) -> str | None:
    """Reject only when every observed .NET symmetric route is unusable."""

    try:
        analyzer = analyzer_for("csharp", source)
    except Exception:
        return None
    required = tuple(
        operation
        for direction, operation in (
            ("encrypt", "CreateEncryptor"),
            ("decrypt", "CreateDecryptor"),
        )
        if not directions or direction in directions
    )
    outcomes: list[str | None] = []
    for operation in required:
        calls = tuple(analyzer.calls(operation))
        if not calls:
            continue
        for call in calls:
            failure: str | None = None
            receiver = (
                re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
                if "." in call.callee
                else ""
            )
            algorithm = _csharp_symmetric_algorithm(analyzer, receiver, call.start)
            allowed_key_lengths = {
                "aes": {16, 24, 32},
                "des": {8},
                "tripledes": {16, 24},
            }.get(algorithm)
            required_iv_length = {
                "aes": 16,
                "des": 8,
                "tripledes": 8,
            }.get(algorithm)
            key_expression = (
                call.arguments[0]
                if call.arguments
                else f"{receiver}.Key"
                if receiver
                else ""
            )
            if key_expression:
                key_length = analyzer.static_byte_length(
                    key_expression,
                    call.start,
                )
                if (
                    key_length is not None
                    and allowed_key_lengths is not None
                    and key_length not in allowed_key_lengths
                ):
                    failure = (
                        f"{algorithm.upper()} requires a supported key length, "
                        f"not {key_length} bytes"
                    )
            if failure is None and receiver and algorithm == "aes":
                block_sizes = {
                    int(value)
                    for value in analyzer.resolve_ints(
                        f"{receiver}.BlockSize", call.start
                    ).values
                    if re.fullmatch(r"\d+", value)
                }
                if len(block_sizes) == 1 and next(iter(block_sizes)) != 128:
                    failure = (
                        "AES requires a 128-bit block size, not "
                        f"{next(iter(block_sizes))} bits"
                    )
            if failure is None:
                iv_expression = (
                    call.arguments[1]
                    if len(call.arguments) >= 2
                    else f"{receiver}.IV"
                    if receiver
                    else ""
                )
                mode_values = (
                    {
                        value.rsplit(".", 1)[-1].upper()
                        for value in analyzer.resolve_strings(
                            f"{receiver}.Mode", call.start
                        ).values
                    }
                    if receiver and len(call.arguments) < 2
                    else set()
                )
                # A source-controlled mode choice may expose a complete ECB
                # route even when a sibling CBC branch assigns an unusable IV.
                # ECB does not consume the IV property.
                iv_is_required_on_every_route = "ECB" not in mode_values
                if iv_expression and iv_is_required_on_every_route:
                    iv_length = analyzer.static_byte_length(
                        iv_expression,
                        call.start,
                    )
                    if (
                        iv_length is not None
                        and required_iv_length is not None
                        and iv_length != required_iv_length
                    ):
                        failure = (
                            f"{algorithm.upper()} requires a "
                            f"{required_iv_length}-byte IV, not "
                            f"{iv_length} bytes"
                        )
            if failure is None and len(call.arguments) >= 2:
                first = re.sub(r"[^A-Za-z0-9]", "", call.arguments[0]).lower()
                second = re.sub(r"[^A-Za-z0-9]", "", call.arguments[1]).lower()
                first_is_iv = bool(
                    re.search(
                        r"(?:^|[a-z0-9])iv(?:src|source|bytes|buffer|value)?$",
                        first,
                    )
                    or "initializationvector" in first
                )
                second_is_key = bool(
                    re.search(
                        r"(?:secret)?key(?:bytes|buffer|material|value)?$",
                        second,
                    )
                )
                if first_is_iv and (second_is_key or second == "null"):
                    failure = (
                        f"{operation} requires key as its first argument and "
                        "IV as its second argument"
                    )
            outcomes.append(failure)
    if outcomes and all(reason is not None for reason in outcomes):
        return next(reason for reason in outcomes if reason is not None)
    return None


def _csharp_symmetric_algorithm(
    analyzer: SourceAnalyzer,
    receiver: str,
    index: int,
) -> str | None:
    """Resolve the concrete .NET symmetric provider used by a transform."""

    expressions = [receiver]
    assignment = analyzer.latest_assignment(receiver, index) if receiver else None
    if assignment is not None:
        expressions.append(assignment.expression)
    function_name = analyzer.containing_function_name(index)
    function = next(
        (
            item
            for item in getattr(analyzer, "functions", ())
            if item.name == function_name and item.start <= index <= item.end
        ),
        None,
    )
    if function is not None:
        expressions.append(analyzer.source[function.start:index])
    text = "\n".join(expressions)
    compact = re.sub(r"[^A-Za-z0-9]", "", text).lower()
    if "tripledes" in compact or "tripledescryptoserviceprovider" in compact:
        return "tripledes"
    if "descryptoserviceprovider" in compact or re.search(
        r"(?:^|[^a-z])des(?:[^a-z]|$)", text.lower()
    ):
        return "des"
    if "aes" in compact:
        return "aes"
    return None


def _typescript_symmetric_transform_argument_failure(
    source: str,
    directions: frozenset[str],
) -> str | None:
    """Reject only when every observed Node cipher direction is unusable."""

    try:
        analyzer = analyzer_for("typescript", source)
    except Exception:
        return None
    outcomes: list[str | None] = []
    for direction, operation in (
        ("encrypt", "createCipheriv"),
        ("decrypt", "createDecipheriv"),
    ):
        if directions and direction not in directions:
            continue
        for call in analyzer.calls(operation):
            failure = _typescript_conditional_cipher_iv_failure(
                analyzer,
                call,
                operation,
            )
            if len(call.arguments) >= 2:
                algorithms = {
                    value.lower()
                    for value in analyzer.resolve_strings(
                        call.arguments[0], call.start
                    ).values
                }
                required_key_lengths = {
                    length
                    for algorithm in algorithms
                    if (
                        length := _typescript_cipher_key_length(algorithm)
                    )
                    is not None
                }
                actual_key_length = analyzer.static_byte_length(
                    call.arguments[1], call.start
                )
                slice_capacity = _typescript_fixed_slice_capacity(
                    analyzer,
                    call.arguments[1],
                    call.start,
                )
                if len(required_key_lengths) == 1 and actual_key_length is not None:
                    required = next(iter(required_key_lengths))
                    if actual_key_length != required:
                        failure = (
                            f"{operation} requires a {required}-byte key for "
                            f"the selected cipher, not {actual_key_length} bytes"
                        )
                elif len(required_key_lengths) == 1 and slice_capacity is not None:
                    required = next(iter(required_key_lengths))
                    if slice_capacity < required:
                        failure = (
                            f"{operation} requires a {required}-byte key for "
                            "the selected cipher, but the fixed slice can "
                            f"provide at most {slice_capacity} bytes"
                        )
            if failure is None and len(call.arguments) >= 3:
                algorithms = {
                    value.lower()
                    for value in analyzer.resolve_strings(
                        call.arguments[0], call.start
                    ).values
                }
                required_iv_lengths = {
                    length
                    for algorithm in algorithms
                    if (
                        length := _typescript_cipher_iv_length(algorithm)
                    )
                    is not None
                }
                actual_iv_length = analyzer.static_byte_length(
                    call.arguments[2], call.start
                )
                compact_algorithms = {
                    re.sub(r"[^a-z0-9]", "", algorithm)
                    for algorithm in algorithms
                }
                if (
                    len(compact_algorithms) == 1
                    and next(iter(compact_algorithms)).endswith("ecb")
                    and re.sub(r"\s+", "", call.arguments[2]) != "null"
                ):
                    failure = f"{operation} requires a null IV for ECB mode"
                elif len(required_iv_lengths) == 1 and actual_iv_length is not None:
                    required = next(iter(required_iv_lengths))
                    if actual_iv_length != required:
                        failure = (
                            f"{operation} requires a {required}-byte IV for "
                            f"the selected cipher, not {actual_iv_length} bytes"
                        )
            if failure is None and direction == "decrypt" and len(call.arguments) >= 2:
                key_expression = re.sub(r"\s+", "", call.arguments[1])
                key_identifiers = {
                    value.lower()
                    for value in re.findall(
                        r"[A-Za-z_$][\w$]*", key_expression
                    )
                }
                if (
                    key_identifiers & {"iv", "nonce", "initializationvector"}
                    or re.search(
                        r"(?i)(?:encrypt(?:ed)?|cipher(?:text)?|sealed|token|payload)"
                        r"[A-Za-z0-9_$.[\]()]*\.(?:slice|subarray)\(",
                        key_expression,
                    )
                ):
                    failure = (
                        "createDecipheriv receives IV, nonce, or ciphertext-derived "
                        "material in the key position instead of the encryption key flow"
                    )
            outcomes.append(failure)
    if outcomes and all(failure is not None for failure in outcomes):
        return next(failure for failure in outcomes if failure is not None)
    return None


def _typescript_conditional_cipher_iv_failure(
    analyzer: SourceAnalyzer,
    call: CallExpression,
    operation: str,
) -> str | None:
    """Reject a same-condition selector whose every algorithm/IV arm is invalid.

    Resolving the algorithm and IV independently loses their correlation.  For
    example, a ternary can select CBC with an empty IV in one arm and ECB with
    a non-null IV in the other.  Each expression has a superficially plausible
    alternative, but no execution arm can initialize the cipher.
    """

    if len(call.arguments) < 3:
        return None
    algorithm_parts = _c_style_conditional_parts(call.arguments[0])
    iv_parts = _c_style_conditional_parts(call.arguments[2])
    if algorithm_parts is None or iv_parts is None:
        return None
    algorithm_condition, *algorithm_arms = algorithm_parts
    iv_condition, *iv_arms = iv_parts
    if re.sub(r"\s+", "", algorithm_condition) != re.sub(
        r"\s+", "", iv_condition
    ):
        return None

    failures: list[str] = []
    for algorithm_expression, iv_expression in zip(algorithm_arms, iv_arms):
        algorithms = {
            value.lower()
            for value in analyzer.resolve_strings(
                algorithm_expression,
                call.start,
            ).values
        }
        if len(algorithms) != 1:
            return None
        algorithm = next(iter(algorithms))
        compact = re.sub(r"[^a-z0-9]", "", algorithm)
        compact_iv = re.sub(r"\s+", "", iv_expression)
        if compact.endswith("ecb"):
            if compact_iv == "null":
                return None
            failures.append(f"{algorithm} requires a null IV")
            continue

        required = _typescript_cipher_iv_length(algorithm)
        if required is None:
            return None
        actual = analyzer.static_byte_length(iv_expression, call.start)
        if actual is None:
            return None
        if actual == required:
            return None
        failures.append(f"{algorithm} requires a {required}-byte IV, not {actual} bytes")

    if len(failures) == 2:
        return f"{operation} has no usable algorithm/IV conditional arm: " + "; ".join(
            failures
        )
    return None


def _typescript_cipher_iv_length_failure(
    analyzer: SourceAnalyzer,
    directions: frozenset[str],
) -> str | None:
    """Reject a fixed IV length only when every candidate route proves it."""

    outcomes: list[str | None] = []
    for direction, operation in (
        ("encrypt", "createCipheriv"),
        ("decrypt", "createDecipheriv"),
    ):
        if directions and direction not in directions:
            continue
        calls = tuple(analyzer.calls(operation))
        if not calls:
            continue
        failures: list[str | None] = []
        for call in calls:
            if len(call.arguments) < 3:
                failures.append(None)
                continue
            algorithms = {
                value.lower()
                for value in analyzer.resolve_strings(
                    call.arguments[0], call.start
                ).values
            }
            required_lengths = {
                length
                for algorithm in algorithms
                if (length := _typescript_cipher_iv_length(algorithm)) is not None
            }
            actual = analyzer.static_byte_length(call.arguments[2], call.start)
            if len(required_lengths) == 1 and actual is not None:
                required = next(iter(required_lengths))
                failures.append(
                    (
                        f"{operation} requires a {required}-byte IV for the "
                        f"selected cipher, not {actual} bytes"
                    )
                    if actual != required
                    else None
                )
            else:
                failures.append(None)
        outcomes.append(
            next(failure for failure in failures if failure is not None)
            if failures and all(failure is not None for failure in failures)
            else None
        )
    if outcomes and all(failure is not None for failure in outcomes):
        return next(failure for failure in outcomes if failure is not None)
    return None


def _typescript_cipher_iv_length(algorithm: str) -> int | None:
    compact = re.sub(r"[^a-z0-9]", "", algorithm.lower())
    if compact.startswith(("aes", "aria")) and compact.endswith("cbc"):
        return 16
    if compact.startswith(("des", "tripledes")) and compact.endswith("cbc"):
        return 8
    return None


def _typescript_cipher_key_length(algorithm: str) -> int | None:
    match = re.search(r"(?:^|[^a-z])aes[-_]?((?:128|192|256))(?:[^0-9]|$)", algorithm)
    if match is not None:
        return int(match.group(1)) // 8
    compact = re.sub(r"[^a-z0-9]", "", algorithm.lower())
    if compact.startswith(("desede3", "tripledes")):
        return 24
    if compact.startswith("desede"):
        return 16
    if compact.startswith("des"):
        return 8
    return None


def _typescript_fixed_slice_capacity(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
) -> int | None:
    """Return the maximum bytes available from a fixed slice or subarray."""

    match = re.fullmatch(
        r".+?\.(?:slice|subarray)\s*\((?P<arguments>.*)\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    arguments = _split_java_arguments(match.group("arguments"))
    if len(arguments) < 2:
        return None
    starts = analyzer.resolve_ints(arguments[0], index).values
    ends = analyzer.resolve_ints(arguments[1], index).values
    if len(starts) != 1 or len(ends) != 1:
        return None
    try:
        start = int(starts[0])
        end = int(ends[0])
    except (TypeError, ValueError):
        return None
    return end - start if 0 <= start <= end else None


def _typescript_rsa_signature_argument_failure(source: str) -> str | None:
    """Reject only when every native Node signature route is malformed."""

    try:
        analyzer = analyzer_for("typescript", source)
    except Exception:
        return None
    outcomes: list[str | None] = []
    for native_call in node_crypto_calls(analyzer, "sign", "verify"):
        call = native_call.call
        failure: str | None = None
        if call.arguments:
            algorithm = re.sub(r"\s+", "", call.arguments[0]).lower()
            if re.search(
                r"(?:private|public)?key|message|data|payload",
                algorithm,
            ):
                failure = (
                    f"crypto.{native_call.operation} requires the "
                    "signature algorithm first"
                )
            if failure is None:
                values = analyzer.resolve_strings(
                    call.arguments[0], call.start
                ).values
                invalid = tuple(
                    value
                    for value in values
                    if _node_signature_digest_is_invalid(value)
                )
                if values and len(invalid) == len(values):
                    failure = (
                        f"crypto.{native_call.operation} requires a digest "
                        f"algorithm, not {invalid[0]!r}"
                    )
        if (
            failure is None
            and len(call.arguments) >= 3
            and _typescript_signature_key_expression(call.arguments[1])
            and _typescript_signature_data_expression(call.arguments[2])
        ):
            failure = (
                f"crypto.{native_call.operation} requires data as its second "
                "argument and key material as its third argument; those "
                "values are reversed"
            )
        if failure is None and len(call.arguments) >= 3:
            failure = _typescript_exported_signature_key_failure(
                analyzer,
                call.arguments[2],
                call.start,
                operation=native_call.operation,
                seen=frozenset(),
            )
        if failure is None and native_call.operation == "verify" and len(call.arguments) >= 4:
            data_argument = call.arguments[1].strip()
            if re.fullmatch(
                r"[A-Za-z_$][A-Za-z0-9_$]*", data_argument
            ) is not None:
                assigned = analyzer.latest_assignment(data_argument, call.start)
                if assigned is not None and re.search(
                    r"(?:^|\.)sign\s*\(", assigned.expression.strip()
                ) is not None:
                    signature_argument = call.arguments[3].strip()
                    if signature_argument != data_argument:
                        failure = (
                            "crypto.verify requires signed data as its second "
                            "argument and the signature as its fourth argument; "
                            "those values are reversed"
                        )
        outcomes.append(failure)
    if outcomes and all(failure is not None for failure in outcomes):
        return next(failure for failure in outcomes if failure is not None)
    return None


def _node_signature_digest_is_invalid(value: str) -> bool:
    """Recognize algorithm-family labels that are not Node digest names."""

    compact = re.sub(r"[^a-z0-9]", "", value.lower())
    return compact in {"rsa", "rsapss", "ecdsa", "dsa", "ed25519", "ed448"}


def _typescript_signature_key_expression(expression: str) -> bool:
    compact = re.sub(r"\s+", "", expression)
    return bool(
        re.search(
            r"(?:private|public)?key|keyPair|\.export\(",
            compact,
            flags=re.IGNORECASE,
        )
    )


def _typescript_signature_data_expression(expression: str) -> bool:
    compact = re.sub(r"\s+", "", expression)
    if _typescript_signature_key_expression(compact):
        return False
    return bool(
        re.search(
            r"(?:^|[^A-Za-z0-9_$])(?:data|message|payload|buffer|record|digest)"
            r"(?:[^A-Za-z0-9_$]|$)",
            compact,
            flags=re.IGNORECASE,
        )
    )


def _typescript_exported_signature_key_failure(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
    *,
    operation: str,
    seen: frozenset[str],
) -> str | None:
    """Reject SPKI exported from a private key before a Node sign/verify call."""

    compact = re.sub(r"\s+", "", expression)
    state = f"typescript-signature-key:{index}:{compact}"
    if state in seen:
        return None
    reference = re.fullmatch(r"[A-Za-z_$][\w$]*", compact)
    if reference is not None:
        assignment = analyzer.latest_assignment(reference.group(0), index)
        if assignment is not None:
            return _typescript_exported_signature_key_failure(
                analyzer,
                assignment.expression,
                assignment.start,
                operation=operation,
                seen=seen | frozenset({state}),
            )
    exported = re.search(
        r"(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*export\s*\(\s*\{"
        r"(?P<options>[\s\S]*?)\}\s*\)",
        expression,
    )
    if exported is None or re.search(
        r"\btype\s*:\s*['\"]spki['\"]",
        exported.group("options"),
        flags=re.IGNORECASE,
    ) is None:
        return None
    if operation == "sign" or "private" in exported.group("receiver").lower():
        return "SPKI is a public-key export format and cannot supply this signature key"
    return None


def _typescript_hmac_argument_failure(source: str) -> str | None:
    """Reject only source-proven invalid Node createHmac digest selectors."""

    try:
        analyzer = analyzer_for("typescript", source)
    except Exception:
        return None
    outcomes: list[str | None] = []
    for native in node_crypto_calls(analyzer, "createHmac"):
        call = native.call
        if len(call.arguments) < 2:
            outcomes.append("createHmac requires a digest selector and key")
            continue
        values = analyzer.resolve_strings(call.arguments[0], call.start).values
        invalid = tuple(
            value for value in values if _node_hmac_digest_is_invalid(value)
        )
        outcomes.append(
            f"createHmac received invalid digest selector {invalid[0]!r}"
            if values and len(invalid) == len(values)
            else None
        )
    if outcomes and all(failure is not None for failure in outcomes):
        return next(failure for failure in outcomes if failure is not None)
    return None


def _node_hmac_digest_is_invalid(value: str) -> bool:
    """Recognize selectors that name an HMAC construction, not a digest."""

    compact = re.sub(r"[^a-z0-9]", "", value.lower())
    return not compact or compact.startswith("hmac")


def _typescript_callback_question_missing_query(source: str) -> str | None:
    try:
        analyzer = analyzer_for("typescript", source)
    except Exception:
        return None
    for call in analyzer.calls("question"):
        if len(call.arguments) != 1:
            continue
        callback = call.arguments[0].strip()
        if re.match(
            r"(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*=>|function\s*\(",
            callback,
            flags=re.DOTALL,
        ):
            return (
                "readline.question callback form requires a query argument "
                "before the callback"
            )
    return None


def _python_canonical_call_name(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> str:
    if call.callee.startswith("cryptography."):
        return call.callee
    aliases = getattr(analyzer, "import_aliases", {})
    head, separator, tail = call.callee.partition(".")
    return aliases.get(head, head) + (separator + tail if separator else "")


def _python_native_rsa_generation_calls(
    analyzer: SourceAnalyzer,
) -> tuple[CallExpression, ...]:
    return tuple(
        call
        for call in analyzer.calls("generate_private_key")
        if _python_canonical_call_name(analyzer, call)
        == "cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key"
        and analyzer.position_is_locally_reachable(call.start)
    )


def _python_dsa_argument_failure(analyzer: SourceAnalyzer) -> str | None:
    """Reject source-proven malformed PyCA DSA key construction."""

    outcomes: list[str | None] = []
    for call in analyzer.calls("generate_parameters", "generate_private_key"):
        canonical = _python_canonical_call_name(analyzer, call)
        if canonical not in {
            "cryptography.hazmat.primitives.asymmetric.dsa.generate_parameters",
            "cryptography.hazmat.primitives.asymmetric.dsa.generate_private_key",
        }:
            continue
        allowed = {"key_size", "backend"}
        unknown = set(call.named_arguments) - allowed
        failure: str | None = None
        if unknown:
            failure = (
                f"{call.simple_name} received unsupported keyword "
                + ", ".join(sorted(unknown))
            )
        elif len(call.arguments) > 2:
            failure = f"{call.simple_name} accepts at most key_size and backend"
        elif call.arguments and "key_size" in call.named_arguments:
            failure = f"{call.simple_name} supplies key_size more than once"
        elif len(call.arguments) >= 2 and "backend" in call.named_arguments:
            failure = f"{call.simple_name} supplies backend more than once"
        elif not call.arguments and "key_size" not in call.named_arguments:
            failure = f"{call.simple_name} requires key_size"
        outcomes.append(failure)
    if outcomes and all(failure is not None for failure in outcomes):
        return next(failure for failure in outcomes if failure is not None)
    return None


def _python_key_serialization_argument_failure(
    analyzer: SourceAnalyzer,
) -> str | None:
    """Reject key serialization arguments read from nonexistent key fields."""

    for call in analyzer.calls("private_bytes", "public_bytes"):
        if "." not in call.callee:
            continue
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        parameters = (
            ("encoding", "format", "encryption_algorithm")
            if call.simple_name == "private_bytes"
            else ("encoding", "format")
        )
        for parameter in parameters[:2]:
            expression = _python_bound_call_expression(call, parameters, parameter)
            if expression is None:
                continue
            compact = re.sub(r"\s+", "", expression)
            if compact == f"{receiver}.{parameter}":
                return (
                    f"PyCA key objects do not expose a {parameter} value; "
                    f"{call.simple_name} requires a serialization enum"
                )
    return None


def _python_rsa_argument_failure(
    analyzer: SourceAnalyzer,
    *,
    family: str | None = None,
) -> str | None:
    """Validate the required PyCA RSA call shapes without executing code."""

    generation_failures = tuple(
        _python_rsa_generation_call_failure(call)
        for call in _python_native_rsa_generation_calls(analyzer)
    )
    has_loaded_key = any(
        _python_canonical_call_name(analyzer, call)
        in {
            "cryptography.hazmat.primitives.serialization.load_pem_private_key",
            "cryptography.hazmat.primitives.serialization.load_der_private_key",
            "cryptography.hazmat.primitives.serialization.load_ssh_private_key",
        }
        for call in analyzer.calls(
            "load_pem_private_key", "load_der_private_key", "load_ssh_private_key"
        )
    )
    if (
        generation_failures
        and not has_loaded_key
        and all(failure is not None for failure in generation_failures)
    ):
        return next(
            failure for failure in generation_failures if failure is not None
        )

    sign_outcomes: list[str | None] = []
    for call in analyzer.calls("sign"):
        if "." not in call.callee:
            continue
        receiver = call.callee.rsplit(".", 1)[0]
        private_key_names, public_key_names = _python_rsa_key_variable_roles(
            analyzer.source,
            analyzer.containing_function_name(call.start),
        )
        if not _python_receiver_is_native_rsa_key(
            receiver,
            private_key_names=private_key_names,
            public_key_names=public_key_names,
        ):
            continue
        failure: str | None = None
        if receiver in public_key_names or re.search(
            r"(?i)(?:^|\.)public[_]?key$",
            receiver,
        ):
            failure = "PyCA RSA signing requires a private-key receiver"
        elif not _python_call_supplies_parameters(
            call,
            ("data", "padding", "algorithm"),
        ):
            failure = "PyCA RSA sign requires data, padding, and hash algorithm"
        else:
            failure = _python_rsa_padding_argument_failure(analyzer, call)
        sign_outcomes.append(failure)
    verify_outcomes: list[str | None] = []
    for call in analyzer.calls("verify"):
        if "." not in call.callee:
            continue
        receiver = call.callee.rsplit(".", 1)[0]
        private_key_names, public_key_names = _python_rsa_key_variable_roles(
            analyzer.source,
            analyzer.containing_function_name(call.start),
        )
        if not _python_receiver_is_native_rsa_key(
            receiver,
            private_key_names=private_key_names,
            public_key_names=public_key_names,
        ):
            continue
        failure = None
        if receiver in private_key_names or re.search(
            r"(?i)(?:^|\.)private[_]?key$",
            receiver,
        ):
            failure = "PyCA RSA verification requires a public-key receiver"
        elif not _python_call_supplies_parameters(
            call,
            ("signature", "data", "padding", "algorithm"),
        ):
            failure = (
                "PyCA RSA verify requires signature, data, padding, and hash algorithm"
            )
        else:
            failure = _python_rsa_padding_argument_failure(analyzer, call)
        verify_outcomes.append(failure)
    if _python_has_complete_public_projection_verify(analyzer.source):
        verify_outcomes.append(None)
    signature_outcomes = (*sign_outcomes, *verify_outcomes)
    if family == "F4":
        if not signature_outcomes and analyzer.calls("signer", "verifier"):
            return (
                "PyCA RSA keys use direct sign and verify operations; "
                "legacy signer and verifier contexts are unavailable"
            )
        if signature_outcomes and all(
            failure is not None for failure in signature_outcomes
        ):
            return next(
                failure for failure in signature_outcomes if failure is not None
            )
        return None

    cipher_outcomes: list[str | None] = []
    for simple_name, description in (
        ("encrypt", "PyCA RSA encrypt requires plaintext and padding"),
        ("decrypt", "PyCA RSA decrypt requires ciphertext and padding"),
    ):
        for call in analyzer.calls(simple_name):
            if "." not in call.callee:
                continue
            receiver = call.callee.rsplit(".", 1)[0]
            private_key_names, public_key_names = _python_rsa_key_variable_roles(
                analyzer.source,
                analyzer.containing_function_name(call.start),
            )
            if not _python_receiver_is_native_rsa_key(
                receiver,
                private_key_names=private_key_names,
                public_key_names=public_key_names,
            ):
                continue
            parameter_names = (
                ("plaintext", "padding")
                if simple_name == "encrypt"
                else ("ciphertext", "padding")
            )
            cipher_outcomes.append(
                None
                if _python_call_supplies_parameters(call, parameter_names)
                else description
            )
    if family == "F3":
        if cipher_outcomes and all(
            failure is not None for failure in cipher_outcomes
        ):
            return next(failure for failure in cipher_outcomes if failure is not None)
        return None

    outcomes = (*signature_outcomes, *cipher_outcomes)
    if outcomes and all(failure is not None for failure in outcomes):
        return next(failure for failure in outcomes if failure is not None)
    return None


def _python_rsa_padding_argument_failure(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> str | None:
    """Reject source-proven unusable PyCA signature arguments."""

    arguments = (*call.arguments, *call.named_arguments.values())
    if any(
        re.search(
            r"\b[A-Za-z_]\w*\s*\.\s*PSS\s*\.\s*(?:MGF1|SAFT)\b"
            r"|\b(?:[A-Za-z_]\w*\s*\.\s*)*MGF(?!1\b)[A-Za-z0-9_]*\b",
            argument,
        )
        for argument in arguments
    ):
        return "PyCA signature padding references an unavailable MGF member"

    parameters = (
        ("data", "padding", "algorithm")
        if call.simple_name == "sign"
        else ("signature", "data", "padding", "algorithm")
    )
    padding_expression = _python_bound_call_expression(
        call, parameters, "padding"
    )
    algorithm_expression = _python_bound_call_expression(
        call, parameters, "algorithm"
    )
    data_expression = _python_bound_call_expression(call, parameters, "data")
    if data_expression and _python_expression_is_definitely_integer(
        analyzer,
        data_expression,
        call.start,
    ):
        return "PyCA sign and verify require bytes-like message data, not an integer"
    if padding_expression and re.match(
        r"\s*(?:(?:utils\s*\.\s*)?Prehashed|hashes\s*\.\s*[A-Za-z0-9_]+)\s*\(",
        padding_expression,
    ):
        return "PyCA signature padding and hash arguments are reversed"
    if algorithm_expression and re.search(
        r"\bpadding\s*\.\s*(?:PSS|PKCS1v15|OAEP)\s*\(",
        algorithm_expression,
    ):
        return "PyCA signature padding and hash arguments are reversed"
    if algorithm_expression and _python_expression_is_hash_context(
        analyzer,
        algorithm_expression,
        call.start,
    ):
        return (
            "PyCA sign and verify require a hash algorithm descriptor, "
            "not a mutable Hash context"
        )
    return None


def _python_expression_is_definitely_integer(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Resolve only values whose source form proves an integer result."""

    try:
        parsed = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return False
    if isinstance(parsed, ast.Constant):
        return isinstance(parsed.value, int) and not isinstance(parsed.value, bool)
    if isinstance(parsed, ast.Call):
        name = _python_call_name(parsed.func)
        return name == "int" or name.endswith("int.from_bytes")
    if isinstance(parsed, ast.Name) and parsed.id not in seen:
        assigned = _python_latest_assignment_expression(
            analyzer,
            parsed.id,
            index,
        )
        return bool(
            assigned
            and _python_expression_is_definitely_integer(
                analyzer,
                assigned,
                index - 1,
                seen | frozenset({parsed.id}),
            )
        )
    return False


def _python_bound_call_expression(
    call: CallExpression,
    parameters: tuple[str, ...],
    name: str,
) -> str | None:
    if name in call.named_arguments:
        return call.named_arguments[name]
    index = parameters.index(name)
    return call.arguments[index] if index < len(call.arguments) else None


def _python_expression_is_hash_context(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
) -> bool:
    compact = re.sub(r"\s+", "", expression)
    if re.search(r"(?:^|\.)Hash\s*\(", compact):
        return True
    if re.fullmatch(r"[A-Za-z_]\w*", compact):
        assigned = _python_latest_assignment_expression(analyzer, compact, index)
        if assigned is not None:
            return bool(re.search(r"(?:^|\.)Hash\s*\(", assigned))
    return False


def _python_rsa_generation_call_failure(
    call: CallExpression,
) -> str | None:
    allowed = {"public_exponent", "key_size", "backend"}
    unknown = set(call.named_arguments) - allowed
    if unknown:
        return (
            "rsa.generate_private_key received unsupported keyword "
            + ", ".join(sorted(unknown))
        )
    if len(call.arguments) > 3:
        return "rsa.generate_private_key accepts at most three positional arguments"
    supplied = {
        "public_exponent": len(call.arguments) >= 1,
        "key_size": len(call.arguments) >= 2,
        "backend": len(call.arguments) >= 3,
    }
    for name in tuple(supplied):
        if name not in call.named_arguments:
            continue
        if supplied[name]:
            return f"rsa.generate_private_key supplies {name} more than once"
        supplied[name] = True
    missing = [
        name for name in ("public_exponent", "key_size") if not supplied[name]
    ]
    if missing:
        return (
            "rsa.generate_private_key is missing required "
            + " and ".join(missing)
        )
    if len(call.arguments) < 2:
        return None
    first = call.arguments[0]
    second = call.arguments[1]
    if _python_expression_looks_like_rsa_key_size(
        first
    ) and _python_expression_looks_like_rsa_public_exponent(second):
        return (
            "rsa.generate_private_key requires public_exponent first "
            "and key_size second"
        )
    if _python_expression_looks_like_rsa_key_size(first) and re.search(
        r"(?i)(?:^|\.)default_backend\s*\(", second
    ):
        return (
            "rsa.generate_private_key is missing public_exponent and "
            "places key_size in the public_exponent position"
        )
    return None


def _python_has_complete_public_projection_verify(source: str) -> bool:
    """Recognize ``private_key.public_key().verify(...)`` call shapes."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    parameters = ("signature", "data", "padding", "algorithm")
    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "verify"
            and isinstance(call.func.value, ast.Call)
            and isinstance(call.func.value.func, ast.Attribute)
            and call.func.value.func.attr == "public_key"
        ):
            continue
        named_values = {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg
        }
        named = set(named_values)
        if (
            len(call.args) <= len(parameters)
            and not (named - set(parameters))
            and not (set(parameters[: len(call.args)]) & named)
            and set(parameters[: len(call.args)]) | named == set(parameters)
        ):
            padding_value = (
                call.args[2]
                if len(call.args) >= 3
                else named_values.get("padding")
            )
            algorithm_value = (
                call.args[3]
                if len(call.args) >= 4
                else named_values.get("algorithm")
            )
            if _python_ast_is_hash_descriptor(padding_value):
                continue
            if _python_ast_is_rsa_padding(algorithm_value):
                continue
            return True
    return False


def _python_ast_is_hash_descriptor(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = _python_call_name(node.func)
    return name.endswith("Prehashed") or bool(
        re.search(r"(?:^|\.)hashes\.[A-Za-z0-9_]+$", name)
    )


def _python_ast_is_rsa_padding(node: ast.AST | None) -> bool:
    return bool(
        isinstance(node, ast.Call)
        and re.search(
            r"(?:^|\.)padding\.(?:PSS|PKCS1v15|OAEP)$",
            _python_call_name(node.func),
        )
    )


def _python_receiver_is_native_rsa_key(
    receiver: str,
    *,
    private_key_names: set[str],
    public_key_names: set[str],
) -> bool:
    """Return whether a method receiver is demonstrably a PyCA RSA key.

    Application code commonly wraps native key operations behind methods named
    ``sign``, ``verify``, ``encrypt``, or ``decrypt``.  Their public call shapes
    are not PyCA's call shapes, so argument validation applies only when source
    evidence identifies the receiver as a native private/public key.
    """

    compact = re.sub(r"\s+", "", receiver)
    if compact in private_key_names or compact in public_key_names:
        return True
    if re.search(r"(?i)(?:^|\.)(?:private|public)[_]?key$", compact):
        return True
    if re.search(r"(?i)\.public[_]?key\(\)$", compact):
        return True
    return False


def _python_call_supplies_parameters(
    call: CallExpression,
    parameters: tuple[str, ...],
) -> bool:
    """Validate Python positional/keyword binding for a native API call."""

    if len(call.arguments) > len(parameters):
        return False
    allowed = set(parameters)
    named = set(call.named_arguments)
    if named - allowed:
        return False
    positional = set(parameters[: len(call.arguments)])
    if positional & named:
        return False
    return positional | named == allowed


def _python_expression_looks_like_rsa_key_size(expression: str) -> bool:
    compact = re.sub(r"\s+", "", expression).lower()
    if re.search(r"(?:key_?size|modulus(?:_?length)?|rsa_?bits|bits?)", compact):
        return True
    integers = tuple(int(value.replace("_", "")) for value in re.findall(r"\b\d[\d_]*\b", compact))
    return bool(integers) and all(value >= 512 for value in integers)


def _python_expression_looks_like_rsa_public_exponent(expression: str) -> bool:
    compact = re.sub(r"\s+", "", expression).lower()
    if "exponent" in compact:
        return True
    integers = {
        int(value.replace("_", ""))
        for value in re.findall(r"\b\d[\d_]*\b", compact)
    }
    return bool(integers) and integers <= {3, 17, 65537}


def _python_rsa_key_variable_roles(
    source: str,
    function_name: str | None,
) -> tuple[set[str], set[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(), set()

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def containing_function(node: ast.AST) -> str | None:
        current = parents.get(node)
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current.name
            current = parents.get(current)
        return None

    private_names: set[str] = set()
    public_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            if containing_function(node) != function_name:
                continue
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            names = {
                target.id for target in targets if isinstance(target, ast.Name)
            }
            if not names or not isinstance(value, ast.Call):
                continue
            call_name = _python_call_name(value.func)
            if call_name.endswith("rsa.generate_private_key"):
                before = len(private_names)
                private_names.update(names)
                changed = changed or len(private_names) != before
                continue
            if call_name.endswith(
                (
                    "serialization.load_der_private_key",
                    "serialization.load_pem_private_key",
                    "serialization.load_ssh_private_key",
                )
            ):
                before = len(private_names)
                private_names.update(names)
                changed = changed or len(private_names) != before
                continue
            if call_name.endswith(
                (
                    "serialization.load_der_public_key",
                    "serialization.load_pem_public_key",
                    "serialization.load_ssh_public_key",
                )
            ):
                before = len(public_names)
                public_names.update(names)
                changed = changed or len(public_names) != before
                continue
            if (
                isinstance(value.func, ast.Attribute)
                and value.func.attr == "public_key"
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id in private_names
            ):
                before = len(public_names)
                public_names.update(names)
                changed = changed or len(public_names) != before
    return private_names, public_names


def _python_native_hmac_calls(
    analyzer: SourceAnalyzer,
) -> tuple[tuple[CallExpression, str], ...]:
    allowed = {
        "hmac.HMAC",
        "hmac.new",
        "hmac.digest",
        "cryptography.hazmat.primitives.hmac.HMAC",
    }
    aliases = callable_aliases(
        analyzer.source,
        getattr(analyzer, "import_aliases", {}),
        allowed,
    )
    calls: list[tuple[CallExpression, str]] = []
    for call in analyzer.calls():
        canonical = _python_canonical_call_name(analyzer, call)
        if canonical not in allowed:
            canonical = aliases.get(re.sub(r"\s+", "", call.callee), "")
        if canonical in allowed and analyzer.position_is_locally_reachable(call.start):
            calls.append((call, canonical))
    return tuple(calls)


def _python_hmac_argument_failure(analyzer: SourceAnalyzer) -> str | None:
    """Reject HMAC arguments only when every native route is malformed."""

    # The indexed call-shape checks below distinguish missing and misplaced
    # arguments.  AST provenance additionally distinguishes a digest
    # constructor (valid) from an already-created hash object (invalid).
    from crypto_steering.validation.tasks.macs import (
        _python_hmac_digestmod_failure,
    )

    digestmod_failure = _python_hmac_digestmod_failure(analyzer.source)
    if digestmod_failure is not None:
        return digestmod_failure

    outcomes = tuple(
        _python_hmac_call_failure(analyzer, call, canonical)
        for call, canonical in _python_native_hmac_calls(analyzer)
    )
    if not outcomes or any(failure is None for failure in outcomes):
        return None
    return next(failure for failure in outcomes if failure is not None)


def _python_hmac_call_failure(
    analyzer: SourceAnalyzer,
    call: CallExpression,
    canonical: str,
) -> str | None:
    if canonical == "hmac.digest":
        if len(call.arguments) != 3 or call.named_arguments:
            return "hmac.digest requires key, message, and digest in that order"
        if _python_expression_looks_like_algorithm_selector(call.arguments[0]):
            return "hmac.digest places an algorithm selector in the key position"
        return None

    if canonical in {"hmac.new", "hmac.HMAC"}:
        allowed = {"key", "msg", "digestmod"}
        if set(call.named_arguments) - allowed:
            return "hmac.new received an unsupported keyword argument"
        if len(call.arguments) > 3:
            return "hmac.new accepts at most key, message, and digestmod"
        supplied_key = bool(call.arguments) or "key" in call.named_arguments
        supplied_digest = (
            len(call.arguments) >= 3 or "digestmod" in call.named_arguments
        )
        if not supplied_key or not supplied_digest:
            return "hmac.new requires key and digestmod"
        if call.arguments and "key" in call.named_arguments:
            return "hmac.new supplies key more than once"
        if len(call.arguments) >= 2 and "msg" in call.named_arguments:
            return "hmac.new supplies message more than once"
        if len(call.arguments) >= 3 and "digestmod" in call.named_arguments:
            return "hmac.new supplies digestmod more than once"
        if call.arguments and _python_expression_looks_like_algorithm_selector(
            call.arguments[0]
        ):
            return "hmac.new places an algorithm selector in the key position"
        message_expression = call.named_arguments.get("msg") or (
            call.arguments[1] if len(call.arguments) >= 2 else ""
        )
        if message_expression and _python_expression_is_direct_text(
            message_expression
        ):
            return "hmac.new requires bytes-like message data, not text"
        if message_expression and _python_expression_is_hash_instance(
            analyzer,
            message_expression,
            call.start,
        ):
            return (
                "hmac.new requires bytes-like message data, not an "
                "unfinalized hash object"
            )
        digest_expression = call.named_arguments.get("digestmod") or (
            call.arguments[2] if len(call.arguments) >= 3 else ""
        )
        if re.search(
            r"\bhashlib\s*\.\s*algorithms_available\s*\[",
            digest_expression,
        ):
            return (
                "hashlib.algorithms_available is a set and cannot be "
                "indexed as an HMAC digest selector"
            )
        if _python_expression_is_hash_instance(
            analyzer,
            digest_expression,
            call.start,
        ):
            return (
                "hmac.new requires a digest name or constructor, not an "
                "already-created hash object"
            )
        return None

    allowed = {"key", "algorithm", "backend"}
    if set(call.named_arguments) - allowed:
        return "PyCA HMAC received an unsupported keyword argument"
    if len(call.arguments) > 3:
        return "PyCA HMAC accepts at most key, algorithm, and backend"
    supplied_key = bool(call.arguments) or "key" in call.named_arguments
    supplied_algorithm = (
        len(call.arguments) >= 2 or "algorithm" in call.named_arguments
    )
    if not supplied_key or not supplied_algorithm:
        return "PyCA HMAC requires key and algorithm"
    if call.arguments and "key" in call.named_arguments:
        return "PyCA HMAC supplies key more than once"
    if len(call.arguments) >= 2 and "algorithm" in call.named_arguments:
        return "PyCA HMAC supplies algorithm more than once"
    if len(call.arguments) >= 3 and "backend" in call.named_arguments:
        return "PyCA HMAC supplies backend more than once"
    algorithm_expression = call.named_arguments.get("algorithm") or (
        call.arguments[1] if len(call.arguments) >= 2 else ""
    )
    if _python_expression_looks_like_backend(algorithm_expression):
        return (
            "PyCA HMAC requires a hash algorithm in its second argument; "
            "a backend object was supplied instead"
        )
    return None


def _python_expression_is_direct_text(expression: str) -> bool:
    """Recognize an intrinsically text-valued Python call argument."""

    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return False
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.Call):
        return _python_call_name(node.func).rsplit(".", 1)[-1] in {
            "input",
            "str",
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _python_expression_is_direct_text(ast.unparse(node.left)) and (
            _python_expression_is_direct_text(ast.unparse(node.right))
        )
    return False


def _python_expression_is_hash_instance(
    analyzer: SourceAnalyzer,
    expression: str,
    index: int,
) -> bool:
    """Recognize a concrete hash object where hmac expects a selector."""

    compact = re.sub(r"\s+", "", expression)
    try:
        parsed = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        parsed = None
    if isinstance(parsed, ast.Call):
        if _python_expression_is_hash_constructor(parsed):
            return True
        if (
            isinstance(parsed.func, ast.Attribute)
            and parsed.func.attr == "get"
            and isinstance(parsed.func.value, ast.Dict)
            and parsed.func.value.values
            and all(
                _python_expression_is_hash_constructor(value)
                for value in parsed.func.value.values
            )
        ):
            return True
        if (
            isinstance(parsed.func, ast.Call)
            and isinstance(parsed.func.func, ast.Name)
            and parsed.func.func.id == "getattr"
            and parsed.func.args
            and _python_call_name(parsed.func.args[0]) == "hashlib"
        ):
            return True
    if re.fullmatch(
        r"(?:hashlib\.)?(?:md5|sha1|sha224|sha256|sha384|sha512|"
        r"sha3_224|sha3_256|sha3_384|sha3_512|blake2b|blake2s)\(\)",
        compact,
        flags=re.IGNORECASE,
    ):
        return True
    called_alias = re.fullmatch(r"(?P<name>[A-Za-z_]\w*)\(\)", compact)
    if called_alias is not None:
        assigned = _python_latest_assignment_expression(
            analyzer,
            called_alias.group("name"),
            index,
        )
        if assigned is not None:
            assigned = assigned.strip()
            return _python_expression_is_hash_instance(
                analyzer,
                assigned if assigned.endswith(")") else f"{assigned}()",
                index - 1,
            )
    if re.fullmatch(r"[A-Za-z_]\w*", compact):
        assigned = _python_latest_assignment_expression(analyzer, compact, index)
        if assigned is not None:
            return _python_expression_is_hash_instance(
                analyzer,
                assigned,
                index - 1,
            )
    return False


def _python_latest_assignment_expression(
    analyzer: SourceAnalyzer,
    name: str,
    index: int,
) -> str | None:
    """Read one Python analyzer assignment without exposing AST details."""

    latest = getattr(analyzer, "_latest_assignment", lambda *_: None)(name, index)
    if latest is None:
        return None
    node, _ = latest
    return ast.unparse(node)


def _python_expression_looks_like_algorithm_selector(expression: str) -> bool:
    compact = re.sub(r"\s+", "", expression).lower()
    if re.search(
        r"(?:hashlib\.|hashes\.|getattr\([^,]+,(?:[^)]*algorithm|[^)]*digest)|"
        r"(?:mac|hash|digest)_?(?:algorithm|name)|algorithm|digestmod|"
        r"hash_?func(?:tion)?)",
        compact,
    ):
        return True
    return compact.strip("\"'") in {
        "md5",
        "sha1",
        "sha224",
        "sha256",
        "sha384",
        "sha512",
    }


def _python_expression_looks_like_backend(expression: str) -> bool:
    compact = re.sub(r"\s+", "", expression).lower()
    return bool(
        re.search(r"(?:^|\.)backend$", compact)
        or "default_backend(" in compact
    )


def _python_pbkdf2_positional_order_is_invalid(
    call: ast.Call,
    tree: ast.AST,
) -> bool:
    """Recognize an unambiguous hashlib-style/PyCA positional mix-up."""

    if len(call.args) < 4:
        return False
    assignments: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments[node.target.id] = node.value

    def resembles_salt(node: ast.AST, seen: frozenset[str] = frozenset()) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (str, bytes))
        if isinstance(node, ast.Name):
            if "salt" in node.id.lower():
                return True
            if node.id in seen or node.id not in assignments:
                return False
            return resembles_salt(
                assignments[node.id],
                seen | frozenset({node.id}),
            )
        return bool(
            isinstance(node, ast.Call)
            and _python_call_name(node.func).rsplit(".", 1)[-1]
            in {"bytes", "bytearray", "encode", "fromhex", "b64decode"}
        )

    def resembles_iterations(
        node: ast.AST,
        seen: frozenset[str] = frozenset(),
    ) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, int) and not isinstance(node.value, bool)
        if isinstance(node, ast.Name):
            if re.search(r"(?:iter|round|cost)", node.id, flags=re.IGNORECASE):
                return True
            if node.id in seen or node.id not in assignments:
                return False
            return resembles_iterations(
                assignments[node.id],
                seen | frozenset({node.id}),
            )
        return False

    if resembles_salt(call.args[1]) and resembles_iterations(call.args[2]):
        return True

    second = call.args[1]
    if isinstance(second, ast.Name) and re.fullmatch(
        r"(?:password|passphrase|passwd|pwd|secret)(?:_bytes)?",
        second.id,
        flags=re.IGNORECASE,
    ):
        return True

    if isinstance(second, ast.Call):
        call_name = _python_call_name(second.func).rsplit(".", 1)[-1]
        if call_name in {"encode", "bytes", "bytearray", "fromhex"}:
            return True
    if isinstance(second, ast.Constant) and isinstance(second.value, (str, bytes)):
        return True
    return False
