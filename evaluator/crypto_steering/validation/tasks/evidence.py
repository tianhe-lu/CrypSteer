"""Reusable source evidence queries for task-family validation."""

from __future__ import annotations

import ast
import re

from crypto_steering.validation.models import (
    TaskEvidence,
    TaskValidationResult,
)
from crypto_steering.validation.source.index import (
    CallExpression,
    SourceAnalyzer,
)
from crypto_steering.validation.source.python import PythonAnalyzer

from crypto_steering.validation.tasks.contracts import (
    _python_call_name,
    _python_native_rsa_generation_calls,
    _security_comparison_helper_match,
)
from crypto_steering.validation.tasks.shared.typescript_api import node_crypto_calls
from crypto_steering.validation.tasks.shared.python_callables import callable_aliases


def _first_evidence(analyzer: SourceAnalyzer, names: tuple[str, ...]) -> TaskEvidence | None:
    calls = tuple(
        call
        for call in analyzer.calls(*names)
        if analyzer.position_is_locally_reachable(call.start)
    )
    if not calls:
        return None
    call = calls[0]
    return TaskEvidence(
        operation=call.simple_name,
        expression=analyzer.source[call.start : call.end],
        span=analyzer.source_span(call.start, call.end),
    )


def _cipher_terminal_evidence(
    analyzer: SourceAnalyzer,
    names: tuple[str, ...],
) -> TaskEvidence | None:
    if analyzer.language == "python":
        return _python_cipher_terminal_evidence(analyzer)
    evidence = _first_evidence(analyzer, names)
    if evidence is not None:
        return evidence
    if analyzer.language == "typescript":
        return _typescript_cipher_stream_terminal_evidence(analyzer)
    if analyzer.language != "csharp":
        return None
    if not re.search(r"\bCryptoStream\b", analyzer.source):
        return None

    # C# symmetric operations are often completed through CryptoStream rather
    # than TransformFinalBlock. A terminal stream read/write is sufficient only
    # when the program actually constructs a CryptoStream; the separate F1/F2
    # direction checks still require CreateEncryptor/CreateDecryptor evidence.
    stream_names = {
        match.group("name")
        for match in re.finditer(
            r"\b(?:CryptoStream|var)\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
            r"new\s*(?:(?:System\s*\.\s*IO\s*\.)?CryptoStream\s*)?\(",
            analyzer.source,
        )
    }
    stream_names.update(
        assignment.name.rsplit(".", 1)[-1]
        for assignment in getattr(analyzer, "assignments", ())
        if re.search(r"\bnew\s+CryptoStream\s*\(", assignment.expression)
    )
    # Readers, writers, and compression streams commonly wrap the CryptoStream
    # and own the terminal Read/Write call. Follow only assignments whose
    # constructor argument directly names a resource already known to wrap the
    # cipher.
    stream_wrappers = (
        "StreamReader",
        "StreamWriter",
        "BinaryReader",
        "BinaryWriter",
        "GZipStream",
        "DeflateStream",
        "BrotliStream",
    )
    wrapper_pattern = "|".join(stream_wrappers)
    assignments = tuple(getattr(analyzer, "assignments", ()))
    changed = True
    while changed:
        changed = False
        wrapper_declarations = tuple(
            re.finditer(
                rf"\b(?:{wrapper_pattern}|var)\s+"
                r"(?P<name>[A-Za-z_]\w*)\s*=\s*new\s*"
                r"(?:(?:(?:System\s*\.\s*IO\s*\.)?"
                rf"(?:{wrapper_pattern}))\s*)?"
                r"\((?P<arguments>[^;]*)\)",
                analyzer.source,
            )
        )
        for declaration in wrapper_declarations:
            name = declaration.group("name")
            if name in stream_names:
                continue
            arguments = declaration.group("arguments")
            if any(
                re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(resource)}(?![A-Za-z0-9_])",
                    arguments,
                )
                for resource in stream_names
            ):
                stream_names.add(name)
                changed = True
        for assignment in assignments:
            name = assignment.name.rsplit(".", 1)[-1]
            if name in stream_names or not re.search(
                r"\bnew\s*(?:(?:(?:System\s*\.\s*IO\s*\.)?"
                rf"(?:{wrapper_pattern}))\s*)?\(",
                assignment.expression,
            ):
                continue
            if any(
                re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(resource)}(?![A-Za-z0-9_])",
                    assignment.expression,
                )
                for resource in stream_names
            ):
                stream_names.add(name)
                changed = True
    terminal_names = (
        "FlushFinalBlock",
        "Write",
        "WriteLine",
        "WriteAsync",
        "Read",
        "ReadBytes",
        "ReadLine",
        "ReadAsync",
        "ReadToEnd",
        "ReadToEndAsync",
        "CopyTo",
        "CopyToAsync",
        "Close",
    )
    for call in analyzer.calls(*terminal_names):
        compact = re.sub(r"\s+", "", call.callee)
        receiver = compact.rsplit(".", 1)[0].rsplit(".", 1)[-1]
        if receiver in stream_names:
            return _call_evidence(analyzer, call)
    return _csharp_disposed_write_stream_evidence(analyzer, stream_names)


def _csharp_disposed_write_stream_evidence(
    analyzer: SourceAnalyzer,
    stream_names: set[str],
) -> TaskEvidence | None:
    """Recognize write completion at the end of a C# ``using`` block."""

    source = analyzer.source
    for match in re.finditer(
        r"using\s*\(\s*(?:var|CryptoStream)\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*=\s*new\s+CryptoStream\s*\("
        r"[^;]*CryptoStreamMode\s*\.\s*Write[^;]*\)\s*\)\s*\{",
        source,
        flags=re.IGNORECASE,
    ):
        name = match.group("name")
        if name not in stream_names:
            continue
        opening = source.find("{", match.start(), match.end())
        closing = _matching_brace(source, opening)
        if closing is None:
            continue
        body = source[opening + 1 : closing]
        if not re.search(
            rf"\b{re.escape(name)}\s*\.\s*(?:Write|WriteAsync|CopyTo)\s*\(|"
            rf"\b(?:Serialize|CopyTo)\s*\([^;]*\b{re.escape(name)}\b",
            body,
            flags=re.IGNORECASE,
        ):
            continue
        return TaskEvidence(
            operation="CryptoStream.Dispose",
            expression=source[match.start() : closing + 1],
            span=analyzer.source_span(match.start(), closing + 1),
        )
    return None


def _matching_brace(source: str, opening: int) -> int | None:
    if opening < 0:
        return None
    depth = 0
    for position in range(opening, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return position
    return None


def _typescript_cipher_stream_terminal_evidence(
    analyzer: SourceAnalyzer,
) -> TaskEvidence | None:
    """Recognize completion through Node's Transform-stream interface."""

    cipher_names = {
        re.sub(r"\s+", "", assignment.name).rsplit(".", 1)[-1]
        for assignment in getattr(analyzer, "assignments", ())
        if re.search(
            r"\bcreate(?:De)?Cipheriv\s*\(",
            assignment.expression,
        )
    }
    cipher_names.update(
        match.group("name")
        for match in re.finditer(
            r"\b(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)"
            r"(?:\s*:\s*[^=;]+)?\s*=\s*"
            r"(?:crypto\s*\.\s*)?create(?:De)?Cipheriv\s*\(",
            analyzer.source,
        )
    )
    if not cipher_names:
        return None

    writes: set[str] = set()
    ends: dict[str, CallExpression] = {}
    for call in analyzer.calls("write", "end"):
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        receiver = receiver.rsplit(".", 1)[-1]
        if receiver not in cipher_names or not analyzer.position_is_locally_reachable(call.start):
            continue
        if call.simple_name == "write" and call.arguments:
            writes.add(receiver)
        elif call.simple_name == "end":
            ends[receiver] = call
    completed = sorted(writes & set(ends))
    return _call_evidence(analyzer, ends[completed[0]]) if completed else None


def _rsa_setup_evidence(analyzer: SourceAnalyzer) -> TaskEvidence | None:
    if analyzer.language == "python":
        calls = _python_native_rsa_generation_calls(analyzer)
        if calls:
            return _call_evidence(analyzer, calls[0])
        # Loading an existing private key is an equally valid source of RSA
        # key material.  Whether the concrete bytes successfully deserialize
        # is a runtime-value question, not part of the structural contract.
        return _first_evidence(
            analyzer,
            (
                "load_pem_private_key",
                "load_der_private_key",
                "load_ssh_private_key",
                "load_pem_public_key",
                "load_der_public_key",
                "load_ssh_public_key",
            ),
        )
    if analyzer.language == "typescript":
        calls = node_crypto_calls(
            analyzer,
            "generateKeyPair",
            "generateKeyPairSync",
            "createPrivateKey",
            "createPublicKey",
        )
        if calls:
            return _call_evidence(analyzer, calls[0].call)
        return None
    evidence = _first_evidence(
        analyzer,
        ("generateKeyPair", "generateKeyPairSync", "generate_private_key"),
    )
    if evidence is not None:
        return evidence
    if analyzer.language == "csharp":
        provider = _first_evidence(analyzer, ("RSACryptoServiceProvider",))
        if provider is not None:
            return provider
        for call in analyzer.calls("Create"):
            if re.fullmatch(
                r"(?:global::)?(?:System\.Security\.Cryptography\.)?RSA\.Create",
                re.sub(r"\s+", "", call.callee),
            ):
                return _call_evidence(analyzer, call)
    return None


def _call_evidence(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> TaskEvidence:
    return TaskEvidence(
        operation=call.simple_name,
        expression=analyzer.source[call.start : call.end],
        span=analyzer.source_span(call.start, call.end),
    )


def _qualified_call_evidence(
    analyzer: SourceAnalyzer,
    simple_name: str,
    qualified_name: str,
) -> TaskEvidence | None:
    for call in analyzer.calls(simple_name):
        if re.sub(r"\s+", "", call.callee) == qualified_name:
            return _call_evidence(analyzer, call)
    return None


def _resolved_call_mentions(call: CallExpression, analyzer: SourceAnalyzer, name: str) -> bool:
    """Return whether a call or its bounded assignment trail names an API."""

    resolved = analyzer.resolve_call_target_strings(call)
    expressions = (
        call.callee,
        resolved.expression,
        *(step.expression for step in resolved.steps),
    )
    compact_name = re.sub(r"\s+", "", name)
    return any(compact_name in re.sub(r"\s+", "", value) for value in expressions)


def _csharp_legacy_algorithm_factory_evidence(
    analyzer: SourceAnalyzer,
) -> TaskEvidence | None:
    for call in analyzer.calls("Create", "CreateFromName"):
        compact = re.sub(r"\s+", "", call.callee)
        if (
            compact.endswith("HashAlgorithm.Create")
            or compact.endswith("CryptoConfig.CreateFromName")
        ) and call.arguments:
            resolved = analyzer.resolve_strings(call.arguments[0], call.start)
            if resolved.values and all(
                "HMAC" in re.sub(r"[^A-Za-z0-9]", "", value).upper()
                for value in resolved.values
            ):
                return _call_evidence(analyzer, call)
    return None


def _csharp_hmac_activator_evidence(
    analyzer: SourceAnalyzer,
) -> TaskEvidence | None:
    for call in analyzer.calls("CreateInstance"):
        compact = re.sub(r"\s+", "", call.callee)
        if not compact.endswith("Activator.CreateInstance") or not call.arguments:
            continue
        resolved = analyzer.resolve_strings(call.arguments[0], call.start)
        if resolved.values and all(
            re.sub(r"[^A-Za-z0-9]", "", value).upper().startswith("HMAC")
            for value in resolved.values
        ):
            return _call_evidence(analyzer, call)
    return None


def _csharp_hmac_factory_evidence(
    analyzer: SourceAnalyzer,
) -> TaskEvidence | None:
    for call in analyzer.calls("CreateHMAC"):
        if re.fullmatch(
            r"(?:System\.Security\.Cryptography\.)?IncrementalHash\.CreateHMAC",
            re.sub(r"\s+", "", call.callee),
        ):
            return _call_evidence(analyzer, call)
    for call in analyzer.calls("Create"):
        if re.fullmatch(
            r"(?:System\.Security\.Cryptography\.)?"
            r"HMAC(?:MD5|SHA1|SHA256|SHA384|SHA512|SHA3_256|SHA3_384|SHA3_512)?"
            r"\.Create",
            re.sub(r"\s+", "", call.callee),
        ):
            return _call_evidence(analyzer, call)
    return None


def _csharp_typed_hmac_evidence(
    analyzer: SourceAnalyzer,
) -> TaskEvidence | None:
    """Recognize a statically HMAC-typed receiver despite runtime factory state."""

    for match in re.finditer(
        r"\bHMAC(?:MD5|SHA1|SHA256|SHA384|SHA512|SHA3_256|SHA3_384|SHA3_512)?"
        r"\s+(?P<receiver>[A-Za-z_]\w*)\s*=",
        analyzer.source,
    ):
        receiver = match.group("receiver")
        if any(
            re.sub(r"\s+", "", call.callee).startswith(receiver + ".")
            for call in analyzer.calls("ComputeHash", "TransformBlock", "TransformFinalBlock")
        ):
            return TaskEvidence(
                operation="typed HMAC receiver",
                expression=match.group(0),
                span=analyzer.source_span(match.start(), match.end()),
            )
    return None


def _mac_one_shot_evidence(analyzer: SourceAnalyzer) -> TaskEvidence | None:
    for call in analyzer.calls("HashData"):
        if "HMAC" in call.callee.upper() and len(call.arguments) >= 2:
            return _call_evidence(analyzer, call)
    for call in analyzer.calls("digest"):
        if re.sub(r"\s+", "", call.callee) == "hmac.digest" and len(call.arguments) >= 3:
            return _call_evidence(analyzer, call)
    if analyzer.language == "python" and re.search(
        r"\bhmac\s*\.\s*new\s*\([^\n]*\)\s*\.\s*(?:digest|hexdigest)\s*\(",
        analyzer.source,
    ):
        calls = analyzer.calls("new")
        for call in calls:
            if re.sub(r"\s+", "", call.callee) == "hmac.new" and len(call.arguments) >= 2:
                return _call_evidence(analyzer, call)
    return None


def _hash_constructor_input_evidence(
    analyzer: SourceAnalyzer,
) -> TaskEvidence | None:
    if analyzer.language != "python":
        return None
    for call in analyzer.calls(
        "md5",
        "sha1",
        "sha224",
        "sha256",
        "sha384",
        "sha512",
        "sha3_224",
        "sha3_256",
        "sha3_384",
        "sha3_512",
    ):
        if call.arguments:
            return _call_evidence(analyzer, call)
    for call in analyzer.calls("new"):
        if call.callee == "hashlib.new" and len(call.arguments) >= 2:
            return _call_evidence(analyzer, call)
    for call in analyzer.calls():
        if call.arguments and (
            _is_python_hash_callable(analyzer, call)
            or (
                call.simple_name != "new"
                and _resolved_call_mentions(call, analyzer, "hashlib")
            )
        ):
            return _call_evidence(analyzer, call)
    return None


def _python_hash_callable_evidence(
    analyzer: SourceAnalyzer,
) -> TaskEvidence | None:
    if analyzer.language != "python":
        return None
    for call in analyzer.calls():
        if (
            _is_python_hash_callable(analyzer, call)
            or _is_python_hashlib_callable(analyzer, call)
            or _resolved_call_mentions(call, analyzer, "hashlib")
        ):
            return _call_evidence(analyzer, call)
    return None


def _is_python_hash_callable(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    resolved = analyzer.resolve_call_target_strings(call)
    return bool(resolved.values) and all(
        re.sub(r"[^A-Za-z0-9]", "", value).upper()
        in {
            "MD5",
            "SHA1",
            "SHA224",
            "SHA256",
            "SHA384",
            "SHA512",
            "SHA3224",
            "SHA3256",
            "SHA3384",
            "SHA3512",
            "BLAKE2B512",
            "BLAKE2S256",
        }
        for value in resolved.values
    )


def _is_python_hashlib_callable(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    resolved = analyzer.resolve_call_target_strings(call)
    expressions = (
        call.callee,
        resolved.expression,
        *(step.expression for step in resolved.steps),
    )
    return any(
        re.search(
            r"\bhashlib\s*\.\s*(?:md5|sha1|sha224|sha256|sha384|sha512|"
            r"sha3_224|sha3_256|sha3_384|sha3_512)\b"
            r"|\bgetattr\s*\(\s*hashlib\s*,",
            expression,
        )
        for expression in expressions
    )


def _python_hash_terminal_evidence(
    analyzer: SourceAnalyzer,
    names: tuple[str, ...],
) -> TaskEvidence | None:
    for call in analyzer.calls(*names):
        expression = analyzer.source[call.start : call.end]
        if _python_terminal_is_called_on_update_result(expression):
            continue
        return _call_evidence(analyzer, call)
    return None


def _python_terminal_is_called_on_update_result(expression: str) -> bool:
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return False
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    receiver = node.func.value
    return (
        isinstance(receiver, ast.Call)
        and isinstance(receiver.func, ast.Attribute)
        and receiver.func.attr == "update"
    )


def _hmac_constructor_input_evidence(
    analyzer: SourceAnalyzer,
) -> TaskEvidence | None:
    if analyzer.language != "python":
        return None
    for call in analyzer.calls("new"):
        if call.callee == "hmac.new" and len(call.arguments) >= 2:
            return _call_evidence(analyzer, call)
    return None


def _random_source_evidence(analyzer: SourceAnalyzer) -> TaskEvidence | None:
    candidates = _random_source_evidences(analyzer)
    return candidates[0] if candidates else None


def _random_source_evidences(
    analyzer: SourceAnalyzer,
) -> tuple[TaskEvidence, ...]:
    names = (
        "randomBytes", "randomFill", "randomFillSync", "randomInt", "getRandomValues",
        "randomUUID", "random", "choice", "choices", "uuid4",
        "GetBytes", "Fill", "GetInt32", "GetHexString", "GetItems", "GetString",
        "NextBytes", "Next", "NextInt64", "NextDouble", "NextSingle", "urandom",
        "token_bytes", "token_hex", "token_urlsafe", "getrandbits", "randbits",
        "randbelow", "randbytes", "randint", "randrange", "uniform",
    )
    evidence: list[TaskEvidence] = []
    for call in analyzer.calls(*names):
        if not analyzer.position_is_locally_reachable(call.start):
            continue
        compact = re.sub(r"\s+", "", call.callee)
        if analyzer.language == "csharp" and call.simple_name in {"GetBytes", "Fill"}:
            if re.search(r"(?:Encoding|Convert|BitConverter|Buffer)$", compact.rsplit(".", 1)[0]):
                continue
        if analyzer.language == "csharp" and not _csharp_random_call_is_native(
            analyzer, call
        ):
            continue
        if analyzer.language == "typescript" and call.simple_name == "random":
            if compact != "Math.random":
                continue
        if analyzer.language == "typescript" and not _typescript_random_call_is_native(
            analyzer,
            call,
        ):
            continue
        if analyzer.language == "python":
            if not _python_random_call_is_native(analyzer, call):
                continue
        evidence.append(_call_evidence(analyzer, call))
    if analyzer.language == "python":
        evidence.extend(_python_random_callable_alias_evidences(analyzer))
    return tuple(evidence)


_PYTHON_RANDOM_CALLABLES = frozenset(
    {
        "os.urandom",
        "uuid.uuid4",
        "secrets.choice",
        "secrets.randbelow",
        "secrets.randbits",
        "secrets.token_bytes",
        "secrets.token_hex",
        "secrets.token_urlsafe",
        "random.choice",
        "random.choices",
        "random.getrandbits",
        "random.randbytes",
        "random.randint",
        "random.randrange",
        "random.random",
        "random.uniform",
    }
)

_PYTHON_RANDOM_INSTANCE_METHODS = frozenset(
    {
        "choice", "choices", "getrandbits", "randbytes", "randint",
        "randrange", "random", "uniform",
    }
)
_PYTHON_RANDOM_SIMPLE_NAMES = frozenset(
    target.rsplit(".", 1)[-1] for target in _PYTHON_RANDOM_CALLABLES
)


def _python_random_target_is_native(canonical: str | None) -> bool:
    if canonical in _PYTHON_RANDOM_CALLABLES:
        return True
    if canonical is None:
        return False
    receiver, _, method = canonical.rpartition(".")
    return (
        receiver in {"random.Random", "random.SystemRandom", "secrets.SystemRandom"}
        and method in _PYTHON_RANDOM_INSTANCE_METHODS
    )


def _python_random_call_target(analyzer: PythonAnalyzer, expression: str, index: int) -> str | None:
    """Resolve a native RNG passed to a plain helper from its actual callers.

    A type annotation does not supply a generator. Reassigned parameters,
    unresolved callers and incompatible bindings remain unresolved.
    """
    direct = analyzer.native_random_call_name(expression, index)
    if direct is not None:
        return direct
    try:
        target = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return None
    receiver = target.value if isinstance(target, ast.Attribute) else target
    if not isinstance(receiver, ast.Name):
        return None
    owner = analyzer._containing_function(index)
    if (not isinstance(owner, ast.FunctionDef) or owner.decorator_list
            or analyzer._enclosing_class(owner) is not None
            or analyzer._enclosing_function(owner) is not None):
        return None
    parameters = analyzer._function_parameter_names(owner)
    if receiver.id not in parameters:
        return None
    if any(analyzer._enclosing_function(value) is owner
           for value, _ in analyzer._assignments.get(receiver.id, ())):
        return None
    if analyzer._assignments.get(owner.name):
        return None
    resolved: set[str] = set()
    for caller in analyzer._call_nodes:
        if not analyzer._external_call_targets_function(caller, owner):
            continue
        if (not analyzer._python_literal_boolean_path_is_reachable(caller)
                or not analyzer._python_call_has_entry_path(caller, seen=frozenset())):
            continue
        if any(isinstance(arg, ast.Starred) for arg in caller.args) or any(keyword.arg is None for keyword in caller.keywords):
            return None
        argument = analyzer._call_argument_for_parameter(
            caller, owner, parameters.index(receiver.id), receiver.id,
        )
        if argument is None:
            return None
        bound = ast.unparse(argument)
        if isinstance(target, ast.Attribute):
            bound = f"({bound}).{target.attr}"
        canonical = analyzer.native_random_call_name(bound, analyzer._node_span(caller)[0])
        if canonical is None:
            return None
        resolved.add(canonical)
    return next(iter(resolved)) if len(resolved) == 1 else None


def _python_random_callable_alias_evidences(
    analyzer: SourceAnalyzer,
) -> tuple[TaskEvidence, ...]:
    """Recognize native random callables stored in variables or object state."""

    aliases = callable_aliases(
        analyzer.source,
        getattr(analyzer, "import_aliases", {}),
        _PYTHON_RANDOM_CALLABLES,
    )
    return tuple(
        _call_evidence(analyzer, call)
        for call in analyzer.calls()
        if (re.sub(r"\s+", "", call.callee) in aliases
            or (isinstance(analyzer, PythonAnalyzer)
                and call.simple_name not in _PYTHON_RANDOM_SIMPLE_NAMES
                and _python_random_target_is_native(
                    _python_random_call_target(analyzer, _python_call_target_expression(analyzer, call), call.start)
                )))
        and analyzer.position_is_locally_reachable(call.start)
    )


def _python_call_target_expression(analyzer: SourceAnalyzer, call: CallExpression) -> str:
    """Retain immediate constructor receivers omitted by the simple call name."""
    try:
        expression = ast.parse(analyzer.source[call.start:call.end].strip(), mode="eval").body
    except SyntaxError:
        return call.callee
    return ast.unparse(expression.func) if isinstance(expression, ast.Call) else call.callee


def _typescript_random_call_is_native(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    """Exclude random APIs selected from a source-proven wrong Node namespace."""

    if call.simple_name == "getRandomValues":
        return True
    compact = re.sub(r"\s+", "", call.callee)
    if re.search(r"(?:^|\.)webcrypto\.", compact, flags=re.IGNORECASE):
        return False
    if "." in compact:
        return True

    name = re.escape(call.simple_name)
    source = analyzer.source
    # A bare Node random function must retain source-level provenance. This
    # also keeps independent-route projection from turning an unresolved name
    # into an apparent native API after a bad destructuring assignment is
    # omitted from the lightweight projection.
    return any(
        re.search(pattern, source, flags=re.IGNORECASE | re.DOTALL) is not None
        for pattern in (
            rf"\bimport\s*\{{[^}}]*\b{name}\b[^}}]*\}}\s*from\s*"
            r"['\"](?:node:)?crypto['\"]",
            rf"\{{[^}}]*\b{name}\b[^}}]*\}}\s*=\s*require\s*\(\s*"
            r"['\"](?:node:)?crypto['\"]\s*\)(?!\s*\.\s*webcrypto\b)",
            rf"\b(?:const|let|var)\s+{name}\s*=\s*require\s*\(\s*"
            r"['\"](?:node:)?crypto['\"]\s*\)\s*\.\s*"
            rf"{name}\b",
        )
    )


def _python_canonical_call(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> str:
    aliases = getattr(analyzer, "import_aliases", {})
    head, separator, tail = call.callee.partition(".")
    return aliases.get(head, head) + (separator + tail if separator else "")


def _python_random_call_is_native(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    if isinstance(analyzer, PythonAnalyzer):
        return _python_random_target_is_native(
            _python_random_call_target(analyzer, _python_call_target_expression(analyzer, call), call.start)
        )
    canonical = _python_canonical_call(analyzer, call)
    if canonical == "random.randbelow":
        # ``randbelow`` belongs to ``secrets``, not Python's random module.
        return False
    if canonical in _PYTHON_RANDOM_CALLABLES:
        return True
    if re.match(r"^(?:random|secrets)\.(?:Random|SystemRandom)\s*\(", canonical):
        return True

    compact_receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
    if re.fullmatch(r"(?:self\.)?[A-Za-z_]\w*", compact_receiver) is None:
        return False
    receiver_pattern = re.escape(compact_receiver).replace(r"self\.", r"self\s*\.\s*")
    return re.search(
        rf"\b{receiver_pattern}\s*=\s*(?:random|secrets)\s*\.\s*"
        r"(?:Random|SystemRandom)\s*\(",
        analyzer.source,
    ) is not None


def _csharp_random_call_is_native(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    source = analyzer.source
    method = call.simple_name
    compact = re.sub(r"\s+", "", call.callee)
    receiver = compact.rsplit(".", 1)[0].rstrip("?")
    rng_methods = {
        "GetBytes",
        "Fill",
        "GetInt32",
        "GetHexString",
        "GetItems",
        "GetString",
    }
    random_methods = {"NextBytes", "Next", "NextInt64", "NextDouble", "NextSingle"}

    if method in rng_methods:
        fully_qualified = "System.Security.Cryptography.RandomNumberGenerator" in receiver
        if re.search(r"\bclass\s+RandomNumberGenerator\b", source) and not fully_qualified:
            return False
        if fully_qualified or receiver.endswith("RandomNumberGenerator"):
            return True
        if "RandomNumberGenerator.Create()" in receiver:
            return True
        receiver_prefix = source[max(0, call.start - 200) : call.start]
        if re.search(
            r"new\s+(?:(?:System\s*\.)?Security\s*\.\s*Cryptography\s*\.)?"
            r"RNGCryptoServiceProvider\s*\(\s*\)\s*\.\s*$",
            receiver_prefix,
        ):
            return True
        name = receiver.rsplit(".", 1)[-1]
        direct_declaration = re.search(
            rf"\b(?:RandomNumberGenerator|RNGCryptoServiceProvider)(?:\s*\?)?\s+"
            rf"{re.escape(name)}\s*=\s*(?:RandomNumberGenerator\s*\.\s*Create\s*\(|"
            r"new\s+RNGCryptoServiceProvider\s*\()",
            source,
        ) is not None or re.search(
            rf"\bvar\s+"
            rf"{re.escape(name)}\s*=\s*(?:RandomNumberGenerator\s*\.\s*Create\s*\(|"
            r"new\s+RNGCryptoServiceProvider\s*\()",
            source,
        ) is not None or re.search(
            rf"\b(?:RandomNumberGenerator|RNGCryptoServiceProvider)(?:\s*\?)?\s+"
            rf"{re.escape(name)}\b",
            source,
        ) is not None
        if direct_declaration:
            return True
        latest_assignment = getattr(analyzer, "latest_assignment", None)
        if not callable(latest_assignment):
            return False
        assignment = latest_assignment(name, call.start)
        if assignment is None:
            return False
        assigned_name = re.sub(r"\s+", "", assignment.expression).removeprefix(
            "this."
        )
        if re.fullmatch(r"[A-Za-z_]\w*", assigned_name) is None:
            return False
        return re.search(
            rf"\b(?:RandomNumberGenerator|RNGCryptoServiceProvider)"
            rf"(?:\s*\?)?\s+{re.escape(assigned_name)}\s*\{{",
            source,
        ) is not None

    if method in random_methods:
        fully_qualified = "System.Random" in receiver
        if re.search(r"\bclass\s+Random\b", source) and not fully_qualified:
            return False
        if fully_qualified or receiver in {"Random", "Random.Shared"}:
            return True
        if _csharp_inline_random_receiver(source, call.start):
            return True
        if re.search(r"(?:^|\.)newRandom\s*\(", receiver):
            return True
        name = receiver.rsplit(".", 1)[-1]
        latest_assignment = getattr(analyzer, "latest_assignment", None)
        if callable(latest_assignment):
            assignment = latest_assignment(name, call.start)
            if assignment is not None and re.fullmatch(
                r"(?:new\s+)?(?:System\s*\.\s*)?Random\s*\([^;]*\)"
                r"|(?:System\s*\.\s*)?Random\s*\.\s*Shared",
                assignment.expression.strip(),
                flags=re.DOTALL,
            ):
                return True
        return re.search(
            rf"\b(?:System\s*\.\s*)?Random(?:\s*\?)?\s+{re.escape(name)}\b",
            source,
        ) is not None or re.search(
            rf"\bvar\s+{re.escape(name)}\s*=\s*new\s+"
            r"(?:System\s*\.\s*)?Random\s*\(",
            source,
        ) is not None

    return False


def _csharp_inline_random_receiver(source: str, call_start: int) -> bool:
    """Recognize native ``Random`` expressions immediately owning a call."""

    prefix = source[max(0, call_start - 240) : call_start]
    return re.search(
        r"(?:"
        r"new\s+(?:System\s*\.\s*)?Random\s*\([^;{}]*\)"
        r"|(?:System\s*\.\s*)?Random\s*\.\s*Shared"
        r"|[A-Za-z_]\w*\s*\?\?=\s*new\s+"
        r"(?:System\s*\.\s*)?Random\s*\([^;{}]*\)"
        r")\s*\)*\s*\.\s*$",
        prefix,
        flags=re.DOTALL,
    ) is not None


def _pbkdf2_setup_evidence(
    analyzer: SourceAnalyzer,
) -> tuple[TaskEvidence | None, TaskEvidence | None, tuple[str, ...]]:
    candidates: list[tuple[CallExpression, tuple[str, ...]]] = []
    if analyzer.language == "typescript":
        seen: set[tuple[int, int]] = set()
        for call in analyzer.calls("pbkdf2", "pbkdf2Sync"):
            seen.add((call.start, call.end))
            is_sync = call.simple_name == "pbkdf2Sync" or _resolved_call_mentions(
                call, analyzer, "crypto.pbkdf2Sync"
            )
            required = (
                ("password", "salt", "iterations", "output length", "digest", "callback")
                if not is_sync
                else ("password", "salt", "iterations", "output length", "digest")
            )
            candidates.append((call, required[len(call.arguments) :]))
        for call in analyzer.calls():
            if (call.start, call.end) in seen or not any(
                _resolved_call_mentions(call, analyzer, name)
                for name in ("crypto.pbkdf2Sync", "crypto.pbkdf2")
            ):
                continue
            is_sync = _resolved_call_mentions(call, analyzer, "crypto.pbkdf2Sync")
            required = (
                ("password", "salt", "iterations", "output length", "digest")
                if is_sync
                else (
                    "password",
                    "salt",
                    "iterations",
                    "output length",
                    "digest",
                    "callback",
                )
            )
            candidates.append((call, required[len(call.arguments) :]))
    elif analyzer.language == "csharp":
        for call in analyzer.calls("Pbkdf2"):
            required = (
                "password",
                "salt",
                "iterations",
                "digest",
                "output length",
            )
            candidates.append((call, required[len(call.arguments) :]))
        for call in analyzer.calls("Rfc2898DeriveBytes"):
            local_definitions = (
                analyzer.function("Rfc2898DeriveBytes")
                if hasattr(analyzer, "function")
                else ()
            )
            source_call = analyzer.source[call.start : call.end]
            constructed = bool(
                re.search(
                    r"\bnew\s*$",
                    analyzer.source[max(0, call.start - 32) : call.start],
                )
                or re.match(r"\s*new\b", source_call)
            )
            if local_definitions and not constructed:
                continue
            # All public constructor overloads provide a password and salt;
            # older valid overloads supply iteration/PRF defaults. Those
            # defaults affect security labels, not whether PBKDF2 completed.
            required = ("password", "salt")
            candidates.append((call, required[len(call.arguments) :]))
    elif analyzer.language == "python":
        for call in analyzer.calls("PBKDF2HMAC"):
            required = ("digest", "output length", "salt", "iterations")
            named = {
                "digest": "algorithm",
                "output length": "length",
                "salt": "salt",
                "iterations": "iterations",
            }
            missing = tuple(
                field
                for position, field in enumerate(required)
                if position >= len(call.arguments)
                and named[field] not in call.named_arguments
            )
            candidates.append((call, missing))
        for call in analyzer.calls("pbkdf2_hmac"):
            if not _is_python_hashlib_pbkdf2_call(analyzer, call):
                continue
            required = ("digest", "password", "salt", "iterations")
            named = {
                "digest": "hash_name",
                "password": "password",
                "salt": "salt",
                "iterations": "iterations",
            }
            missing = tuple(
                field
                for position, field in enumerate(required)
                if position >= len(call.arguments)
                and named[field] not in call.named_arguments
            )
            candidates.append((call, missing))
        for call in analyzer.calls():
            if call.simple_name == "pbkdf2_hmac" or not _resolved_call_mentions(
                call, analyzer, "hashlib.pbkdf2_hmac"
            ):
                continue
            required = ("digest", "password", "salt", "iterations")
            named = {
                "digest": "hash_name",
                "password": "password",
                "salt": "salt",
                "iterations": "iterations",
            }
            missing = tuple(
                field
                for position, field in enumerate(required)
                if position >= len(call.arguments)
                and named[field] not in call.named_arguments
            )
            candidates.append((call, missing))

    if candidates:
        # A translated program may wrap the native operation in a helper also
        # named pbkdf2. Prefer the call that carries the complete configuration
        # contract. If all are incomplete, report the closest native call so
        # diagnostics identify the actual missing fields.
        call, missing = min(
            candidates,
            key=lambda item: (len(item[1]), item[0].start),
        )
        setup = _call_evidence(analyzer, call)
        return setup, setup if not missing else None, missing
    for call in analyzer.calls("PBEKeySpec"):
        setup = _call_evidence(analyzer, call)
        missing = (
            ()
            if len(call.arguments) >= 4
            else ("password", "salt", "iterations", "output length")[
                len(call.arguments) :
            ]
        )
        return setup, setup if not missing else None, missing
    return None, None, ()


def _is_python_hashlib_pbkdf2_call(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    import_aliases = getattr(analyzer, "import_aliases", {})
    head, separator, tail = call.callee.partition(".")
    canonical_head = import_aliases.get(head, head)
    canonical = canonical_head + (separator + tail if separator else "")
    return canonical == "hashlib.pbkdf2_hmac"


def _directional_task_result(
    analyzer: SourceAnalyzer,
    task_id: str,
    setup: TaskEvidence | None,
    terminal: TaskEvidence | None,
    encrypt: TaskEvidence | None,
    decrypt: TaskEvidence | None,
    *,
    setup_name: str,
    terminal_name: str,
) -> TaskValidationResult:
    requested = _requested_subset(task_id, both="T1", first="T2", second="T3")
    required = [setup, terminal]
    names = [setup_name, terminal_name]
    if requested in {"both", "first"}:
        required.append(encrypt)
        names.append("encryption direction")
    if requested in {"both", "second"}:
        required.append(decrypt)
        names.append("decryption direction")
    return _require_many(analyzer, task_id, required, names)


def _cbc_evidence(analyzer: SourceAnalyzer) -> TaskEvidence | None:
    match = re.search(
        r"(?:\baes-(?:[0-9]+|\$\{[^}]+\})-cbc\b|"
        r"\bCipherMode\s*\.\s*CBC\b|\bmodes\s*\.\s*CBC\b|"
        r"\bEncryptCbc\b|\bDecryptCbc\b)",
        analyzer.source,
        flags=re.IGNORECASE,
    )
    if match:
        return TaskEvidence("CBC", match.group(0), analyzer.source_span(match.start(), match.end()))
    if analyzer.language == "typescript":
        for call in analyzer.calls("createCipheriv", "createDecipheriv"):
            if not call.arguments:
                continue
            algorithm = call.arguments[0]
            resolved = analyzer.resolve_strings(algorithm, call.start)
            values = tuple(value.lower().replace("_", "-") for value in resolved.values)
            resolved_expressions = (algorithm,) + tuple(
                step.expression for step in resolved.steps
            )
            if any(value.endswith("-cbc") for value in values) or any(
                re.search(
                    r"[\"'`](?:-)?cbc[\"'`]",
                    expression,
                    flags=re.IGNORECASE,
                )
                for expression in resolved_expressions
            ):
                return TaskEvidence(
                    "CBC",
                    algorithm,
                    analyzer.source_span(call.start, call.end),
                )
    if analyzer.language == "csharp" and re.search(
        r"\b(?:Aes\s*\.\s*Create\s*\(|Aes(?:\s*\?)?\s+[A-Za-z_]\w*\b|"
        r"DES(?:CryptoServiceProvider|\.Create)\s*\()",
        analyzer.source,
    ) and not re.search(r"CipherMode\s*\.\s*ECB", analyzer.source):
        match = re.search(
            r"\b(?:Aes\s*\.\s*Create\s*\(|Aes(?:\s*\?)?\s+[A-Za-z_]\w*\b|"
            r"DES(?:CryptoServiceProvider|\.Create)\s*\()",
            analyzer.source,
        )
        assert match is not None
        return TaskEvidence("CBC_default", match.group(0), analyzer.source_span(match.start(), match.end()))
    return None


def _comparison_evidence(analyzer: SourceAnalyzer) -> TaskEvidence | None:
    match = re.search(
        r"(?:CryptographicOperations\.FixedTimeEquals|timingSafeEqual|"
        r"compare_digest|hmac\.compare_digest|MessageDigest\.isEqual|"
        r"\.SequenceEqual\s*\(|===|!==|==|!=|\.Equals\s*\()",
        analyzer.source,
    )
    if match is None:
        match = _security_comparison_helper_match(analyzer.source)
    if not match:
        return None
    return TaskEvidence("comparison", match.group(0), analyzer.source_span(match.start(), match.end()))


def _digest_comparison_evidence(analyzer: SourceAnalyzer) -> TaskEvidence | None:
    """Return evidence that an expected value is compared with a digest."""

    identifier = re.compile(r"\b[A-Za-z_$][A-Za-z0-9_$]*\b")
    value_markers = ("hash", "digest", "sha", "checksum")
    expected_markers = (
        "expected",
        "provided",
        "received",
        "known",
        "actual",
        "computed",
        "initial",
        "previous",
        "current",
        "new",
        "compare",
    )
    comparison = re.compile(
        r"===|!==|==|!=|\.\s*(?:equals|equalsIgnoreCase|sequenceEqual)\s*\(|"
        r"\b(?:Arrays\s*\.\s*equals|CompareByteArrays|AreHashesEqual)\s*\(",
        flags=re.IGNORECASE,
    )
    for fragment in re.finditer(r"[^;\n]+", analyzer.source):
        text = fragment.group(0)
        identifiers = identifier.findall(text)
        values = {
            name.lower()
            for name in identifiers
            if any(marker in name.lower() for marker in value_markers)
        }
        expected = any(
            any(marker in name.lower() for marker in expected_markers)
            for name in identifiers
        )
        if comparison.search(text) and (
            len(values) >= 2 or (values and expected)
        ):
            return TaskEvidence(
                "digest comparison",
                text.strip(),
                analyzer.source_span(fragment.start(), fragment.end()),
            )
    for call in analyzer.calls(
        "FixedTimeEquals",
        "timingSafeEqual",
        "compare_digest",
        "isEqual",
        "SequenceEqual",
        "Equals",
    ):
        expression = analyzer.source[call.start : call.end]
        identifiers = identifier.findall(expression)
        if any(
            marker in name.lower()
            for name in identifiers
            for marker in (*value_markers, *expected_markers)
        ):
            return _call_evidence(analyzer, call)
    helper = _security_comparison_helper_match(analyzer.source)
    if helper is not None:
        return TaskEvidence(
            "digest comparison",
            helper.group(0),
            analyzer.source_span(helper.start(), helper.end()),
        )
    derived = _digest_derived_names(analyzer)
    for fragment in re.finditer(r"[^;\n]+", analyzer.source):
        text = fragment.group(0)
        if re.search(r"\.\s*(?:equals|equalsIgnoreCase)\s*\(", text, re.I) is None:
            continue
        names = set(identifier.findall(text))
        derived_names = names & derived
        if not derived_names:
            continue
        # The indexed call view does not retain a nested instance receiver in
        # expressions such as ``BitConverter.ToString(hash).Equals(value)``.
        # Recover that comparison from its complete statement.  Later checks
        # reject self-comparisons and locally recomputed expected digests.
        argument = re.search(
            r"\.\s*(?:equals|equalsIgnoreCase)\s*\(\s*(?P<value>[^,)]+)",
            text,
            re.I,
        )
        if argument is None:
            continue
        argument_names = set(identifier.findall(argument.group("value")))
        if argument_names & derived_names:
            continue
        return TaskEvidence(
            "digest comparison",
            text.strip(),
            analyzer.source_span(fragment.start(), fragment.end()),
        )
    for call in analyzer.calls():
        if re.search(r"(?:compare|equal|match)", call.simple_name, re.IGNORECASE) is None:
            continue
        operands = list(call.arguments)
        if "." in call.callee:
            operands.insert(0, call.callee.rsplit(".", 1)[0])
        if len(operands) < 2:
            continue
        derived_operands = [
            bool(set(identifier.findall(operand)) & derived) for operand in operands
        ]
        expected_operand = any(
            any(marker in name.lower() for marker in expected_markers)
            for operand in operands
            for name in identifier.findall(operand)
        )
        if any(derived_operands) and (sum(derived_operands) >= 2 or expected_operand):
            return _call_evidence(analyzer, call)
    for match in re.finditer(
        r"(?P<left>[^;\n]+?)\s*(?:===|!==|==|!=)\s*"
        r"(?P<right>[^;\n{]+)",
        analyzer.source,
    ):
        left_names = set(identifier.findall(match.group("left")))
        right_names = set(identifier.findall(match.group("right")))
        left_digest = bool(left_names & derived)
        right_digest = bool(right_names & derived)
        left_expected = any(
            marker in name.lower()
            for name in left_names
            for marker in ("expected", "provided", "received", "known", "previous")
        )
        right_expected = any(
            marker in name.lower()
            for name in right_names
            for marker in ("expected", "provided", "received", "known", "previous")
        )
        left_literal = bool(
            re.fullmatch(r"\s*(?:[rubfRUBF]*)?(?:'.*'|\".*\")\s*", match.group("left"))
        )
        right_literal = bool(
            re.search(
                r"(?:[rubfRUBF]*)?(?:'[^']*'|\"[^\"]*\")",
                match.group("right"),
            )
        )
        if (
            (left_digest and (right_expected or right_literal))
            or (right_digest and (left_expected or left_literal))
        ):
            return TaskEvidence(
                "digest comparison",
                match.group(0).strip(),
                analyzer.source_span(match.start(), match.end()),
            )
    return None


def _digest_derived_names(analyzer: SourceAnalyzer) -> set[str]:
    """Collect local names that carry a completed digest value."""

    assignments: list[tuple[set[str], str]] = []
    if analyzer.language == "python":
        try:
            tree = ast.parse(analyzer.source)
        except SyntaxError:
            return set()
        for node in ast.walk(tree):
            if (
                not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
                or node.value is None
            ):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            names = {
                target.id for target in targets if isinstance(target, ast.Name)
            }
            if names:
                assignments.append((names, ast.unparse(node.value)))
    else:
        for assignment in getattr(analyzer, "assignments", ()):
            names = set(
                re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", assignment.name)
            )
            if names:
                assignments.append((names, assignment.expression))
        if not assignments:
            for match in re.finditer(
                r"\b(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*"
                r"(?<![=!<>])=(?!=)\s*(?P<value>[^;]+);",
                analyzer.source,
                flags=re.DOTALL,
            ):
                assignments.append(({match.group("name")}, match.group("value")))

    terminal = re.compile(
        r"\.\s*(?:digest|hexdigest|ComputeHash|HashData|GetHashAndReset)\s*\("
        r"|\b(?:calculate|compute|create|get|generate)[A-Za-z0-9_$]*"
        r"(?:hash|digest|checksum)[A-Za-z0-9_$]*\s*\(",
        flags=re.IGNORECASE,
    )
    identifier = re.compile(r"\b[A-Za-z_$][A-Za-z0-9_$]*\b")
    derived: set[str] = set()
    changed = True
    while changed:
        changed = False
        for names, expression in assignments:
            if not terminal.search(expression) and not (
                set(identifier.findall(expression)) & derived
            ):
                continue
            before = len(derived)
            derived.update(names)
            changed |= len(derived) != before
    return derived


def _require(
    analyzer: SourceAnalyzer,
    task_id: str,
    first: TaskEvidence | None,
    second: TaskEvidence | None,
    first_name: str,
    second_name: str,
) -> TaskValidationResult:
    return _require_many(analyzer, task_id, [first, second], [first_name, second_name])


def _require_many(
    analyzer: SourceAnalyzer,
    task_id: str,
    evidence: list[TaskEvidence | None],
    names: list[str],
) -> TaskValidationResult:
    missing = [name for item, name in zip(evidence, names) if item is None]
    matched = tuple(item for item in evidence if item is not None)
    if missing:
        setup_missing = not matched
        code = "target_api_not_found" if setup_missing else "target_operation_incomplete"
        return TaskValidationResult(
            language=analyzer.language,
            task_id=task_id,
            success=False,
            reason_codes=(code,),
            reason="missing " + ", ".join(missing),
            matched_operations=matched,
        )
    return TaskValidationResult(
        language=analyzer.language,
        task_id=task_id,
        success=True,
        reason_codes=(),
        reason="completed target-task operation",
        matched_operations=matched,
    )


def _requested_subset(task_id: str, *, both: str, first: str, second: str) -> str:
    suffix = task_id.rsplit("_", 1)[-1]
    if suffix == both:
        return "both"
    if suffix == first:
        return "first"
    if suffix == second:
        return "second"
    return "both"


def _python_cipher_terminal_evidence(
    analyzer: SourceAnalyzer,
) -> TaskEvidence | None:
    """Find a terminal on a real PyCA cipher context or one-shot AEAD object."""

    try:
        tree = ast.parse(analyzer.source)
    except SyntaxError:
        return None

    context_names: set[str] = set()
    one_shot_names: set[str] = set()
    context_helpers: set[str] = set()
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and _python_call_name(node.value.func).rsplit(".", 1)[-1]
            in {"encryptor", "decryptor"}
            for node in ast.walk(function)
        ):
            context_helpers.add(function.name)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        target_names = {
            re.sub(r"\s+", "", ast.unparse(target))
            for target in targets
            if isinstance(target, (ast.Name, ast.Attribute))
        }
        simple = _python_call_name(value.func).rsplit(".", 1)[-1]
        if simple in {"encryptor", "decryptor"}:
            context_names.update(target_names)
        if simple in context_helpers:
            context_names.update(target_names)
        if simple in {"AESGCM", "AESCCM", "ChaCha20Poly1305", "Fernet"}:
            one_shot_names.update(target_names)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            if not isinstance(item.context_expr, ast.Call) or item.optional_vars is None:
                continue
            if _python_call_name(item.context_expr.func).rsplit(".", 1)[-1] not in {
                "encryptor",
                "decryptor",
            }:
                continue
            if isinstance(item.optional_vars, (ast.Name, ast.Attribute)):
                context_names.add(
                    re.sub(r"\s+", "", ast.unparse(item.optional_vars))
                )

    calls_by_start = {
        (call.start, call.simple_name): call
        for call in analyzer.calls("finalize", "encrypt", "decrypt")
        if analyzer.position_is_locally_reachable(call.start)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        operation = node.func.attr
        receiver_node = node.func.value
        receiver = re.sub(r"\s+", "", ast.unparse(receiver_node))
        valid = False
        if operation == "finalize":
            valid = receiver in context_names or bool(
                isinstance(receiver_node, ast.Call)
                and _python_call_name(receiver_node.func).rsplit(".", 1)[-1]
                in {"encryptor", "decryptor"}
            )
            if isinstance(receiver_node, ast.Subscript):
                key = receiver_node.slice
                valid = valid or (
                    isinstance(key, ast.Constant)
                    and key.value in {"encryptor", "decryptor"}
                )
        elif operation in {"encrypt", "decrypt"}:
            valid = receiver in one_shot_names or bool(
                isinstance(receiver_node, ast.Call)
                and _python_call_name(receiver_node.func).rsplit(".", 1)[-1]
                in {"AESGCM", "AESCCM", "ChaCha20Poly1305", "Fernet"}
            )
        if not valid:
            continue
        start, _ = analyzer._node_span(node)  # type: ignore[attr-defined]
        call = calls_by_start.get((start, operation))
        if call is not None:
            return _call_evidence(analyzer, call)
    return None


def _cipher_terminal_names(language: str) -> tuple[str, ...]:
    """Return completion operations implemented by each supported crypto API."""

    if language == "java":
        return ("doFinal",)
    if language == "typescript":
        return ("final",)
    if language == "csharp":
        return (
            "TransformFinalBlock",
            "TransformBlock",
            "Encrypt",
            "Decrypt",
            "EncryptCbc",
            "DecryptCbc",
        )
    if language == "python":
        return ("finalize", "encrypt", "decrypt")
    return ()
