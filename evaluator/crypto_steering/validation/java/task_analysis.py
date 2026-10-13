"""Reusable Java source-analysis helpers for structural task validation."""

from __future__ import annotations

import re

from crypto_steering.validation.detectors.engine import detect_misuse
from crypto_steering.validation.java.source import CallSite, find_calls, iter_methods
from crypto_steering.validation.java.static_analysis import (
    Assignment,
    extract_method_returns,
    extract_assignments,
    is_char_array_expression,
    latest_assignment,
    simple_call_name,
    simple_variable,
    strip_outer_expression,
    variable_type_in_scope,
)


def _receiver_method_calls(source: str, method_name: str) -> tuple[tuple[str, CallSite], ...]:
    calls: list[tuple[str, CallSite]] = []
    for call in find_calls(source, rf"\b[A-Za-z_]\w*\s*\.\s*{re.escape(method_name)}\s*\(", method_name):
        receiver = _call_receiver(source, call)
        if receiver:
            calls.append((receiver, call))
    return tuple(calls)


def _receivers_call_method(source: str, receivers: set[str], method_name: str) -> bool:
    return any(
        receiver in receivers
        for receiver, _ in _receiver_method_calls(source, method_name)
    )


def _factory_receivers_assigned_from(source: str, factory_class: str) -> set[str]:
    receivers: set[str] = set()
    for call in find_calls(
        source,
        rf"\b(?:[\w.]+\.)?{re.escape(factory_class)}\s*\.\s*getInstance(?:Strong)?\s*\(",
        f"{factory_class}.getInstance",
    ):
        receiver = _assignment_target_for_call(source, call.start)
        if receiver:
            receivers.add(receiver)
    return receivers


def _cipher_receivers_with_transformation(source: str, required_mode: str) -> set[str]:
    receivers: set[str] = set()
    string_constants = _string_constants_before(source, len(source))
    for call in find_calls(source, r"\b(?:[\w.]+\.)?Cipher\s*\.\s*getInstance\s*\(", "Cipher.getInstance"):
        if not call.arguments:
            continue
        transformation = _constant_string_expression(call.arguments[0].text, string_constants)
        if required_mode.upper() not in transformation.upper():
            continue
        receiver = _assignment_target_for_call(source, call.start)
        if receiver:
            receivers.add(receiver)
    return receivers


def _symmetric_cipher_receivers(source: str) -> set[str]:
    receivers: set[str] = set()
    string_constants = _string_constants_before(source, len(source))
    mode_detection = detect_misuse("java", "M02", source)
    resolved_mode_expressions = {
        re.sub(r"\s+", "", finding.expression)
        for finding in mode_detection.findings
        if finding.label != "unknown"
    }
    symmetric_algorithms = {
        "AES",
        "ARCFOUR",
        "BLOWFISH",
        "CHACHA20",
        "DES",
        "DESEDE",
        "RC2",
        "RC4",
    }
    for call in find_calls(
        source,
        r"\b(?:[\w.]+\.)?Cipher\s*\.\s*getInstance\s*\(",
        "Cipher.getInstance",
    ):
        if not call.arguments:
            continue
        transformation = _constant_string_expression(
            call.arguments[0].text,
            string_constants,
        )
        algorithm = transformation.split("/", 1)[0].strip().upper()
        resolved_by_mode_detector = (
            re.sub(r"\s+", "", call.arguments[0].text)
            in resolved_mode_expressions
        )
        if (
            algorithm not in symmetric_algorithms
            and not algorithm.startswith("PBEWITH")
            and not resolved_by_mode_detector
        ):
            continue
        receiver = _assignment_target_for_call(source, call.start)
        if receiver:
            receivers.add(receiver)
    return receivers


def _assignment_target_for_call(source: str, call_start: int) -> str | None:
    statement_start = max(source.rfind(";", 0, call_start), source.rfind("{", 0, call_start))
    prefix = source[statement_start + 1 : call_start]
    match = re.search(
        r"(?:\b[\w.$<>\[\]]+\s+)?(?P<name>[A-Za-z_]\w*)\s*=\s*$",
        prefix,
        flags=re.DOTALL,
    )
    return match.group("name") if match else None


def _is_cipher_receiver(source: str, receiver: str, index: int) -> bool:
    receiver_type = variable_type_in_scope(source, receiver, index) or ""
    if receiver_type.endswith("Cipher"):
        return True
    return _has_typed_variable_declaration_before(source, receiver, index, ("Cipher",))


def _is_cipher_mode_expression(source: str, expression: str, index: int) -> bool:
    expr = strip_outer_expression(expression)
    if re.search(r"\bCipher\.(?:ENCRYPT|DECRYPT)_MODE\b", expr):
        return True
    variable = simple_variable(expr)
    if not variable:
        return False
    variable_type = variable_type_in_scope(source, variable, index) or ""
    if variable_type != "int" and not _has_typed_variable_declaration_before(source, variable, index, ("int",)):
        return False
    return bool(re.search(r"\bCipher\.(?:ENCRYPT|DECRYPT)_MODE\b", source))


def _is_key_expression(source: str, expression: str, index: int) -> bool:
    expr = strip_outer_expression(expression)
    if re.search(r"\bnew\s+(?:[\w.]+\.)?SecretKeySpec\s*\(", expr):
        return True
    if re.search(r"\bgenerateKey\s*\(", expr):
        return True
    if _helper_call_returns_type(source, expr, ("SecretKey", "SecretKeySpec", "Key")):
        return True
    variable = simple_variable(expr)
    if not variable:
        return not _is_definitely_non_key_expression(source, expr, index)
    variable_type = variable_type_in_scope(source, variable, index) or ""
    if variable_type:
        return variable_type.endswith(("SecretKey", "SecretKeySpec", "Key"))
    if _has_typed_variable_declaration_before(source, variable, index, ("SecretKey", "SecretKeySpec", "Key")):
        return True
    assignment = latest_assignment(source, variable, index)
    if assignment is None:
        # Rule 5 rejects only source-proven argument-role violations. The
        # configured Java compiler establishes whether an otherwise unresolved
        # expression is assignable to Key.
        return True
    if re.search(r"\bnew\s+(?:[\w.]+\.)?SecretKeySpec\s*\(", assignment.expression):
        return True
    if re.search(r"\b(?:[A-Za-z_]\w*\s*\.\s*)?generateKey\s*\(\s*\)", assignment.expression):
        return True
    if _helper_call_returns_type(source, assignment.expression, ("SecretKey", "SecretKeySpec", "Key")):
        return True
    return not _is_definitely_non_key_expression(
        source,
        assignment.expression,
        assignment.start,
    )


def _is_definitely_non_key_expression(
    source: str,
    expression: str,
    index: int,
) -> bool:
    """Identify only expressions whose source-proven type cannot be a Key."""

    expr = strip_outer_expression(expression).strip()
    if expr == "null" or re.fullmatch(
        r"(?:true|false|[-+]?\d+(?:\.\d+)?[fFdDlL]?|\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])')",
        expr,
        flags=re.DOTALL,
    ):
        return True
    if re.search(
        r"\bnew\s+(?:[\w.]+\.)?(?:IvParameterSpec|GCMParameterSpec|"
        r"AlgorithmParameters|SecureRandom)\s*\(",
        expr,
    ) or re.search(r"\bnew\s+byte\s*\[", expr):
        return True
    variable = simple_variable(expr)
    if variable is None:
        return False
    variable_type = variable_type_in_scope(source, variable, index) or ""
    return bool(
        variable_type
        and not variable_type.endswith(("SecretKey", "SecretKeySpec", "Key"))
    )


def _is_asymmetric_key_expression(source: str, expression: str, index: int) -> bool:
    expr = expression.strip()
    if re.search(r"\b(?:getPublic|getPrivate)\s*\(\s*\)$", expr):
        return True
    if re.search(
        r"\b(?:[A-Za-z_$][\w$]*)?(?:PublicKey|PrivateKey)\b"
        r"|\b(?:KeyPair|Key)\b",
        expr,
    ):
        return True
    variable = simple_variable(expr)
    if not variable:
        return False
    if _has_typed_variable_declaration_before(
        source,
        variable,
        index,
        ("PublicKey", "PrivateKey", "Key"),
    ):
        return True
    variable_type = variable_type_in_scope(source, variable, index) or ""
    simple_type = variable_type.rsplit(".", 1)[-1]
    if (
        simple_type in {"PublicKey", "PrivateKey", "Key"}
        or simple_type.endswith(("PublicKey", "PrivateKey"))
    ):
        return True
    assignment = latest_assignment(source, variable, index)
    if assignment is None:
        return False
    if re.search(r"\b(?:getPublic|getPrivate)\s*\(\s*\)", assignment.expression):
        return True
    if re.search(
        r"\b(?:[A-Za-z_$][\w$]*)?(?:PublicKey|PrivateKey)\b"
        r"|\bKey\b",
        assignment.expression,
    ):
        return True
    return False


def _is_iv_parameter_spec_expression(source: str, expression: str, index: int) -> bool:
    expr = strip_outer_expression(expression)
    if re.search(r"\bnew\s+(?:[\w.]+\.)?IvParameterSpec\s*\(", expr):
        return True
    if _helper_call_returns_type(source, expr, ("IvParameterSpec",)):
        return True
    variable = simple_variable(expr)
    if not variable:
        return False
    variable_type = variable_type_in_scope(source, variable, index) or ""
    if variable_type.endswith("IvParameterSpec"):
        return True
    if _has_typed_variable_declaration_before(source, variable, index, ("IvParameterSpec",)):
        return True
    assignment = latest_assignment(source, variable, index)
    return bool(
        assignment
        and (
            re.search(r"\bnew\s+(?:[\w.]+\.)?IvParameterSpec\s*\(", assignment.expression)
            or _helper_call_returns_type(source, assignment.expression, ("IvParameterSpec",))
        )
    )


def _secure_random_receivers(source: str) -> set[str]:
    receivers: set[str] = set()
    for assignment in extract_assignments(source):
        if assignment.type_name.endswith("SecureRandom"):
            receivers.add(assignment.name)
            continue
        if re.search(
            r"\bnew\s+(?:java\.security\.)?SecureRandom\s*\(|\b(?:java\.security\.)?SecureRandom\s*\.\s*getInstance(?:Strong)?\s*\(",
            assignment.expression,
            flags=re.DOTALL,
        ):
            receivers.add(assignment.name)
        elif _is_secure_random_expression(source, assignment.expression, assignment.start):
            receivers.add(assignment.name)
    return receivers


def _digest_receivers_used_by_digest_input_stream(source: str, digest_receivers: set[str]) -> set[str]:
    receivers: set[str] = set()
    for call in find_calls(
        source,
        r"\bnew\s+(?:java\.security\.)?DigestInputStream\s*\(",
        "DigestInputStream",
    ):
        if len(call.arguments) < 2:
            continue
        digest_variable = simple_variable(call.arguments[1].text)
        if digest_variable and digest_variable in digest_receivers:
            receivers.add(digest_variable)
    return receivers


def _helper_call_returns_type(
    source: str,
    expression: str,
    accepted_suffixes: tuple[str, ...],
    seen: frozenset[str] = frozenset(),
) -> bool:
    helper = simple_call_name(strip_outer_expression(expression))
    if not helper:
        return False
    if helper in seen:
        return False
    method = extract_method_returns(source).get(helper)
    if not method:
        return False
    if method.return_type.endswith(accepted_suffixes):
        return True
    if not method.expressions:
        return False
    if "Key" in accepted_suffixes or "SecretKey" in accepted_suffixes or "SecretKeySpec" in accepted_suffixes:
        return all(_is_key_expression(source, returned, method.body_end) for returned in method.expressions)
    if "IvParameterSpec" in accepted_suffixes:
        return all(
            _is_iv_parameter_spec_expression(source, returned, method.body_end)
            for returned in method.expressions
        )
    return False


def _is_secure_random_expression(
    source: str,
    expression: str,
    index: int,
    seen: frozenset[str] = frozenset(),
) -> bool:
    expr = strip_outer_expression(expression)
    if re.search(
        r"\bnew\s+(?:java\.security\.)?SecureRandom\s*\(|\b(?:java\.security\.)?SecureRandom\s*\.\s*getInstance(?:Strong)?\s*\(",
        expr,
        flags=re.DOTALL,
    ):
        return True
    variable = simple_variable(expr)
    if variable:
        if variable in seen:
            return False
        variable_type = variable_type_in_scope(source, variable, index) or ""
        if variable_type.endswith("SecureRandom"):
            return True
        assignment = latest_assignment(source, variable, index)
        if assignment and _is_secure_random_expression(
            source,
            assignment.expression,
            assignment.start,
            seen | frozenset({variable}),
        ):
            return True
        return False
    helper = simple_call_name(expr)
    if not helper:
        return False
    if helper in seen:
        return False
    method = extract_method_returns(source).get(helper)
    if not method:
        return False
    if method.return_type.endswith("SecureRandom"):
        return True
    return bool(
        method.expressions
        and all(
            _is_secure_random_expression(
                source,
                returned,
                method.body_end,
                seen | frozenset({helper}),
            )
            for returned in method.expressions
        )
    )


def _has_typed_variable_declaration_before(
    source: str,
    variable: str,
    index: int,
    accepted_suffixes: tuple[str, ...],
) -> bool:
    prefix = source[:index]
    type_pattern = "|".join(re.escape(type_name) for type_name in accepted_suffixes)
    return bool(
        re.search(
            rf"\b(?:[\w.]+\.)?(?:{type_pattern})\s+{re.escape(variable)}\b",
            prefix,
        )
    )


def _string_constants_before(source: str, end: int) -> dict[str, str]:
    declarations = tuple(
        (match.group("name"), match.group("expression").strip())
        for match in re.finditer(
            r"\b(?:public|private|protected|static|final|\s)*\s*"
            r"(?:String|java\.lang\.String)\s+"
            r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<expression>[^;]+)\s*;",
            source[:end],
            flags=re.DOTALL,
        )
    )
    constants: dict[str, str] = {}
    unresolved = list(declarations)
    while unresolved:
        remaining: list[tuple[str, str]] = []
        changed = False
        for name, expression in unresolved:
            resolved = _resolved_constant_string_expression(expression, constants)
            if resolved is None:
                remaining.append((name, expression))
                continue
            constants[name] = resolved
            changed = True
        if not changed:
            break
        unresolved = remaining
    return constants


def _resolved_constant_string_expression(
    expression: str,
    constants: dict[str, str],
) -> str | None:
    pieces: list[str] = []
    for item in _split_string_concat(expression):
        item = item.strip()
        literal = re.fullmatch(r'"((?:\\.|[^"\\])*)"', item, flags=re.DOTALL)
        if literal:
            pieces.append(literal.group(1))
            continue
        variable = re.fullmatch(r"[A-Za-z_]\w*", item)
        if variable and variable.group(0) in constants:
            pieces.append(constants[variable.group(0)])
            continue
        return None
    return "".join(pieces)


def _constant_string_expression(expression: str, constants: dict[str, str]) -> str:
    resolved = _resolved_constant_string_expression(expression, constants)
    if resolved is not None:
        return resolved
    pieces: list[str] = []
    for item in _split_string_concat(expression):
        item = item.strip()
        literal = re.fullmatch(r"\"((?:\\.|[^\"\\])*)\"", item, flags=re.DOTALL)
        if literal:
            pieces.append(literal.group(1))
            continue
        variable = re.fullmatch(r"[A-Za-z_]\w*", item)
        if variable and variable.group(0) in constants:
            pieces.append(constants[variable.group(0)])
            continue
        pieces.append(item)
    return "".join(pieces)


def _split_string_concat(expression: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(expression):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "([{":
            depth += 1
            continue
        if char in ")]}":
            depth = max(0, depth - 1)
            continue
        if char == "+" and depth == 0:
            parts.append(expression[start:index])
            start = index + 1
    parts.append(expression[start:])
    return parts


def _has_secret_key_factory_pbkdf2(
    source: str,
    *,
    allow_general_pbe: bool = False,
) -> bool:
    assignments = extract_assignments(source)
    for call in find_calls(
        source,
        r"\bSecretKeyFactory\s*\.\s*getInstance\s*\(",
        "SecretKeyFactory.getInstance",
    ):
        if not call.arguments:
            continue
        if allow_general_pbe:
            # A connected PBEKeySpec -> SecretKeyFactory.generateSecret route
            # establishes password-based derivation even when the algorithm
            # name arrives through a value the bounded resolver cannot reduce.
            # The configured compiler has already checked the API signature.
            return True
        algorithm_expr = call.arguments[0].text
        algorithm_pattern = r"PBKDF2"
        if re.search(algorithm_pattern, algorithm_expr, flags=re.IGNORECASE):
            return True
        algorithm_var = simple_variable(algorithm_expr)
        if algorithm_var:
            for assignment in assignments:
                if (
                    assignment.name == algorithm_var
                    and assignment.start <= call.start
                    and re.search(
                        algorithm_pattern,
                        assignment.expression,
                        flags=re.IGNORECASE,
                    )
                ):
                    return True
            method = next(
                (
                    candidate
                    for candidate in iter_methods(source)
                    if candidate.body_start <= call.start <= candidate.body_end
                    and algorithm_var in candidate.params
                ),
                None,
            )
            if method is None:
                continue
            parameter_index = method.params.index(algorithm_var)
            for invocation in find_calls(
                source,
                rf"\b{re.escape(method.name)}\s*\(",
                method.name,
            ):
                if method.start <= invocation.start <= method.body_end:
                    continue
                if parameter_index >= len(invocation.arguments):
                    continue
                if re.search(
                    algorithm_pattern,
                    invocation.arguments[parameter_index].text,
                    flags=re.IGNORECASE,
                ):
                    return True
    return False


def _pbe_spec_variables(source: str) -> set[str]:
    variables: set[str] = set()
    assignments = extract_assignments(source)
    for assignment in assignments:
        calls = find_calls(
            assignment.expression,
            r"\bnew\s+(?:javax\.crypto\.spec\.)?PBEKeySpec\s*\(",
            "PBEKeySpec",
        )
        if calls and _is_valid_f7_pbe_key_spec_call(source, calls[0], index=assignment.start):
            variables.add(assignment.name)
        if (
            assignment.type_name.endswith(("PBEKeySpec", "KeySpec"))
            and calls
            and _is_valid_f7_pbe_key_spec_call(source, calls[0], index=assignment.start)
        ):
            variables.add(assignment.name)
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            expression = simple_variable(assignment.expression)
            if expression not in variables or assignment.name in variables:
                continue
            if assignment.type_name.endswith(("PBEKeySpec", "KeySpec")):
                variables.add(assignment.name)
                changed = True
    return variables


def _is_pbe_key_spec_expression(source: str, expression: str, index: int, pbe_variables: set[str]) -> bool:
    expr = expression.strip()
    if re.search(r"\bnew\s+(?:javax\.crypto\.spec\.)?PBEKeySpec\s*\(", expr):
        call = find_calls(expr, r"\bnew\s+(?:javax\.crypto\.spec\.)?PBEKeySpec\s*\(", "PBEKeySpec")
        return bool(call and _is_valid_f7_pbe_key_spec_call(source, call[0], index=index))
    variable = simple_variable(expr)
    if variable and variable in pbe_variables:
        return True
    return False


def _is_valid_f7_pbe_key_spec_call(source: str, call: CallSite, index: int | None = None) -> bool:
    """Return whether a PBEKeySpec constructor is a usable F7 KDF sink.

    The one-argument constructor stores only the password. For PBKDF2 tasks it
    omits the salt and iteration count, so it should not make a generated
    program target-valid for M06/M07/F7 steering analysis.
    """

    if len(call.arguments) < 3:
        return False
    salt_expr = call.arguments[1].text.strip()
    iteration_expr = call.arguments[2].text.strip()
    if not salt_expr or salt_expr == "null":
        return False
    if not iteration_expr:
        return False
    return True


def _keystore_operation_calls(source: str, assignments: tuple[Assignment, ...]) -> list[CallSite]:
    calls: list[CallSite] = []
    for name in (
        "getInstance",
        "load",
        "store",
        "setKeyEntry",
        "setCertificateEntry",
        "setEntry",
        "getEntry",
        "getKey",
    ):
        if name == "getInstance":
            calls.extend(find_calls(source, r"\bKeyStore\s*\.\s*getInstance\s*\(", "KeyStore.getInstance"))
        else:
            for call in find_calls(source, rf"\b[A-Za-z_]\w*\s*\.\s*{name}\s*\(", f"KeyStore.{name}"):
                receiver = _call_receiver(source, call)
                if receiver and _is_keystore_receiver(receiver, assignments, source, call.start):
                    calls.append(call)
    return calls


def _first_bad_keystore_password_call(source: str, assignments: tuple[Assignment, ...]) -> CallSite | None:
    password_arg_indexes = {
        "KeyStore.load": 1,
        "KeyStore.store": 1,
        "KeyStore.getKey": 1,
    }
    for name, index in password_arg_indexes.items():
        method = name.split(".")[-1]
        for call in find_calls(source, rf"\b[A-Za-z_]\w*\s*\.\s*{method}\s*\(", name):
            receiver = _call_receiver(source, call)
            if receiver and not _is_keystore_receiver(receiver, assignments, source, call.start):
                continue
            if len(call.arguments) > index and not _is_valid_keystore_password_expression(
                source,
                call.arguments[index].text,
                call.start,
            ):
                return call

    for call in find_calls(source, r"\b[A-Za-z_]\w*\s*\.\s*setKeyEntry\s*\(", "KeyStore.setKeyEntry"):
        receiver = _call_receiver(source, call)
        if receiver and not _is_keystore_receiver(receiver, assignments, source, call.start):
            continue
        if len(call.arguments) >= 4 and not _is_valid_keystore_password_expression(
            source,
            call.arguments[2].text,
            call.start,
        ):
            return call

    for call in find_calls(
        source,
        r"\bnew\s+(?:java\.security\.)?(?:KeyStore\.)?PasswordProtection\s*\(",
        "KeyStore.PasswordProtection",
    ):
        if call.arguments and not _is_valid_keystore_password_expression(
            source,
            call.arguments[0].text,
            call.start,
        ):
            return call
    return None


def _is_valid_keystore_password_expression(source: str, expression: str, index: int) -> bool:
    expr = expression.strip()
    if expr == "null":
        return True
    return is_char_array_expression(source, expr, index)


def _has_certificate_storage_call(source: str, assignments: tuple[Assignment, ...]) -> bool:
    for call in find_calls(source, r"\b[A-Za-z_]\w*\s*\.\s*setCertificateEntry\s*\(", "KeyStore.setCertificateEntry"):
        receiver = _call_receiver(source, call)
        if receiver and _is_keystore_receiver(receiver, assignments, source, call.start):
            return True
    trusted_certificate_entries = _trusted_certificate_entry_variables(assignments)
    for call in find_calls(source, r"\b[A-Za-z_]\w*\s*\.\s*setEntry\s*\(", "KeyStore.setEntry"):
        receiver = _call_receiver(source, call)
        if receiver and _is_keystore_receiver(receiver, assignments, source, call.start):
            if len(call.arguments) > 1 and _is_trusted_certificate_entry_expression(
                call.arguments[1].text,
                trusted_certificate_entries,
            ):
                return True
    return False


def _has_key_or_entry_storage_call(source: str, assignments: tuple[Assignment, ...]) -> bool:
    for method_name in ("setKeyEntry", "setEntry"):
        for call in find_calls(source, rf"\b[A-Za-z_]\w*\s*\.\s*{method_name}\s*\(", f"KeyStore.{method_name}"):
            receiver = _call_receiver(source, call)
            if receiver and _is_keystore_receiver(receiver, assignments, source, call.start):
                return True
    return False


def _trusted_certificate_entry_variables(assignments: tuple[Assignment, ...]) -> set[str]:
    variables: set[str] = set()
    for assignment in assignments:
        if assignment.type_name.endswith("TrustedCertificateEntry"):
            variables.add(assignment.name)
        elif re.search(
            r"\bnew\s+(?:java\.security\.)?(?:KeyStore\.)?TrustedCertificateEntry\s*\(",
            assignment.expression,
        ):
            variables.add(assignment.name)
    return variables


def _is_trusted_certificate_entry_expression(expression: str, known_variables: set[str]) -> bool:
    if re.search(r"\bnew\s+(?:java\.security\.)?(?:KeyStore\.)?TrustedCertificateEntry\s*\(", expression):
        return True
    variable = simple_variable(expression)
    return bool(variable and variable in known_variables)


def _call_receiver(source: str, call: CallSite) -> str | None:
    prefix = source[: call.open_paren]
    match = re.search(r"([A-Za-z_]\w*)\s*\.\s*[A-Za-z_]\w*\s*$", prefix)
    return match.group(1) if match else None


def _is_keystore_receiver(
    receiver: str,
    assignments: tuple[Assignment, ...],
    source: str,
    index: int,
) -> bool:
    if re.search(r"^ks$|keystore", receiver, flags=re.IGNORECASE):
        return True
    receiver_type = variable_type_in_scope(source, receiver, index)
    if receiver_type and receiver_type.endswith("KeyStore"):
        return True
    for assignment in assignments:
        if assignment.name != receiver:
            continue
        if assignment.type_name.endswith("KeyStore"):
            return True
        if "KeyStore.getInstance" in assignment.expression:
            return True
    return False
