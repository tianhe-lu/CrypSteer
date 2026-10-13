"""Language-specific crypto API recognition shared by misuse detectors."""

from __future__ import annotations

import ast
from dataclasses import replace
import re
from typing import Iterable

from crypto_steering.validation.models import (
    DetectorFinding,
    ProgramLabel,
    ResolvedValue,
    ValueOrigin,
)
from crypto_steering.validation.source.index import (
    CStyleAnalyzer,
    CallExpression,
    PythonAnalyzer,
    SourceAnalyzer,
)

from crypto_steering.validation.detectors.core import (
    _BROKEN_DIGESTS,
    _SECURE_DIGESTS,
    _finding_from_resolved,
    _algorithm_finding,
    _resolved_value_is_registered_kdf_output,
    _random_label,
    _dedupe_findings,
    _call_argument,
    _normalize_algorithm,
)


def _security_sensitive_randomness_findings(
    analyzer: SourceAnalyzer,
) -> list[DetectorFinding]:
    """Classify random values only after they reach a registered sink.

    Canonical paired F9 tasks are handled separately because their dataset
    contract designates the produced IV, salt, or integer as the security use.
    Free-form generation must instead show a downstream cryptographic, KDF, or
    MAC use. Resolving values at those sinks also follows helper returns and
    in-place buffer fills supported by the bounded source analyzer.
    """

    findings: list[DetectorFinding] = []
    random_origins = {
        "secure_random",
        "non_csprng",
        "predictable",
    }
    for sink_id, sink_name, expression, start, end in _randomness_sink_values(
        analyzer
    ):
        resolved = analyzer.resolve_origin(expression, start)
        if resolved.origin not in random_origins:
            continue
        if (
            resolved.origin == "predictable"
            and (
                _resolved_value_is_registered_kdf_output(resolved)
                or not _resolved_value_has_random_source_evidence(analyzer, resolved)
            )
        ):
            # Constants and deterministic derivations can be unsuitable for
            # their own policy (for example M01 or M08), but they are not M05
            # evidence unless an actual PRNG or seed operation is in the
            # resolved data-flow chain.
            continue
        findings.append(
            _finding_from_resolved(
                analyzer,
                "M05",
                sink_id,
                sink_name,
                expression,
                start,
                end,
                resolved,
                _random_label,
            )
        )
    findings.extend(_role_labeled_randomness_findings(analyzer))
    return _dedupe_findings(findings)


def _role_labeled_randomness_findings(
    analyzer: SourceAnalyzer,
) -> list[DetectorFinding]:
    """Recognize random nonces/challenges embedded in protected payloads."""

    candidates = {
        "typescript": ("random",),
        "csharp": ("Next", "NextBytes", "NextInt64"),
        "python": ("random", "randint", "randrange", "getrandbits"),
    }[analyzer.language]
    role_pattern = re.compile(
        r"(?i)\b(?:nonce|challenge|session[_$]?token|security[_$]?code|"
        r"temporary|tmp|temp[_$]?(?:file|path|name)|verifier)\b"
    )
    downstream_pattern = re.compile(
        r"(?i)\b(?:createSign|sign[A-Za-z0-9_$]*|SignData|publicEncrypt|Encrypt|"
        r"createCipheriv|writeFile|writeFileSync|rename|move|open)\s*\("
    )
    findings: list[DetectorFinding] = []
    for call in analyzer.calls(*candidates):
        if not _is_random_source_call(analyzer, call):
            continue
        expression = analyzer.source[call.start : call.end]
        resolved = analyzer.resolve_origin(expression, call.start)
        if resolved.origin not in {"non_csprng", "predictable"}:
            continue
        statement_start = max(
            analyzer.source.rfind(";", 0, call.start),
            analyzer.source.rfind("\n", 0, call.start),
        ) + 1
        statement_end = analyzer.source.find(";", call.end)
        if statement_end == -1:
            statement_end = min(len(analyzer.source), call.end + 300)
        context = (
            (analyzer.containing_function_name(call.start) or "")
            + " "
            + analyzer.source[statement_start : statement_end + 1]
        )
        if not _random_role_present(role_pattern, context):
            continue
        if downstream_pattern.search(analyzer.source[call.end :]) is None:
            continue
        findings.append(
            _finding_from_resolved(
                analyzer,
                "M05",
                "security_role_randomness",
                call.callee,
                expression,
                call.start,
                call.end,
                resolved,
                _random_label,
            )
        )
    if analyzer.language == "typescript":
        findings.extend(
            _typescript_template_randomness_findings(
                analyzer,
                role_pattern=role_pattern,
                downstream_pattern=downstream_pattern,
            )
        )
    return findings


def _typescript_template_randomness_findings(
    analyzer: SourceAnalyzer,
    *,
    role_pattern: re.Pattern[str],
    downstream_pattern: re.Pattern[str],
) -> list[DetectorFinding]:
    """Inspect executable substitutions hidden by the C-style string mask."""

    findings: list[DetectorFinding] = []
    for template in re.finditer(r"`(?:\\.|[^`])*`", analyzer.source, re.DOTALL):
        template_source = template.group(0)
        if not _random_role_present(role_pattern, template_source):
            continue
        for match in re.finditer(
            r"\bMath\s*\.\s*random\s*\(\s*\)",
            template_source,
        ):
            prefix = template_source[: match.start()]
            if prefix.rfind("${") <= prefix.rfind("}"):
                continue
            start = template.start() + match.start()
            end = template.start() + match.end()
            if downstream_pattern.search(analyzer.source[end:]) is None:
                continue
            expression = analyzer.source[start:end]
            findings.append(
                _finding_from_resolved(
                    analyzer,
                    "M05",
                    "security_role_randomness",
                    "Math.random",
                    expression,
                    start,
                    end,
                    ResolvedValue(expression=expression, origin="non_csprng"),
                    _random_label,
                )
            )
    return findings


def _random_role_present(pattern: re.Pattern[str], source: str) -> bool:
    # In source text an escaped newline is the two-character sequence ``\n``;
    # without normalization its trailing ``n`` masks the word boundary of a
    # following role such as ``nonce``.
    return pattern.search(re.sub(r"\\[nrt]", " ", source)) is not None


def _resolved_value_has_random_source_evidence(
    analyzer: SourceAnalyzer,
    resolved: ResolvedValue,
) -> bool:
    expressions = (resolved.expression,) + tuple(
        step.expression for step in resolved.steps
    )
    pattern = {
        "java": re.compile(
            r"(?:\bnew\s+(?:java\.util\.)?Random\s*\(|"
            r"\bnew\s+(?:java\.security\.)?SecureRandom\s*\([^)]|"
            r"\.\s*setSeed\s*\(|System\s*\.\s*(?:currentTimeMillis|nanoTime)\s*\()",
            flags=re.IGNORECASE,
        ),
        "typescript": re.compile(
            r"(?:\bMath\s*\.\s*random\s*\(|\bDate\s*\.\s*now\s*\(|"
            r"\b(?:lcg|xorshift|seededRandom)\b)",
            flags=re.IGNORECASE,
        ),
        "csharp": re.compile(
            r"(?:\bnew\s+(?:System\s*\.\s*)?Random\s*\(|"
            r"\bRandom\s*\.\s*Shared\b|\.\s*(?:Next|NextBytes|NextInt64)\s*\(|"
            r"\bDateTimeOffset\b|\bEnvironment\s*\.\s*TickCount\b)",
            flags=re.IGNORECASE,
        ),
        "python": re.compile(
            r"(?:\brandom\s*\.\s*(?:Random|random|randbytes|randint|randrange|getrandbits)\s*\(|"
            r"\btime\s*\.\s*time(?:_ns)?\s*\(|"
            r"\b(?:lcg|xorshift|seeded_random)\b)",
            flags=re.IGNORECASE,
        ),
    }[analyzer.language]
    return any(pattern.search(expression) for expression in expressions)


def _resolved_value_has_any_random_source_evidence(
    analyzer: SourceAnalyzer,
    resolved: ResolvedValue,
) -> bool:
    if _resolved_value_has_random_source_evidence(analyzer, resolved):
        return True
    expressions = (resolved.expression,) + tuple(
        step.expression for step in resolved.steps
    )
    secure_pattern = {
        "java": re.compile(
            r"(?:\bnew\s+(?:java\.security\.)?SecureRandom\s*\(|"
            r"\bSecureRandom\s*\.\s*getInstance)",
            flags=re.IGNORECASE,
        ),
        "typescript": re.compile(
            r"\b(?:randomBytes|randomFillSync|randomInt|getRandomValues)\s*\(",
            flags=re.IGNORECASE,
        ),
        "csharp": re.compile(
            r"\bRandomNumberGenerator\s*\.\s*(?:GetBytes|Fill|GetInt32)\s*\(",
            flags=re.IGNORECASE,
        ),
        "python": re.compile(
            r"\b(?:secrets\s*\.\s*(?:token_bytes|token_hex|token_urlsafe|"
            r"choice|randbits|randbelow)|"
            r"os\s*\.\s*urandom)\s*\(",
            flags=re.IGNORECASE,
        ),
    }[analyzer.language]
    return any(secure_pattern.search(expression) for expression in expressions)


def _randomness_sink_values(
    analyzer: SourceAnalyzer,
) -> Iterable[tuple[str, str, str, int, int]]:
    call_specs: dict[str, tuple[tuple[int, str, tuple[str, ...]], ...]]
    if analyzer.language == "typescript":
        call_specs = {
            "createCipheriv": (
                (1, "crypto_key_input", ("key",)),
                (2, "crypto_iv_or_nonce_input", ("iv", "nonce")),
            ),
            "createDecipheriv": (
                (1, "crypto_key_input", ("key",)),
                (2, "crypto_iv_or_nonce_input", ("iv", "nonce")),
            ),
            "createHmac": ((1, "mac_key_input", ("key",)),),
            "pbkdf2": (
                (0, "kdf_password_input", ("password",)),
                (1, "kdf_salt_input", ("salt",)),
            ),
            "pbkdf2Sync": (
                (0, "kdf_password_input", ("password",)),
                (1, "kdf_salt_input", ("salt",)),
            ),
            "scrypt": (
                (0, "kdf_password_input", ("password",)),
                (1, "kdf_salt_input", ("salt",)),
            ),
            "scryptSync": (
                (0, "kdf_password_input", ("password",)),
                (1, "kdf_salt_input", ("salt",)),
            ),
        }
    elif analyzer.language == "csharp":
        call_specs = {
            "CreateEncryptor": (
                (0, "crypto_key_input", ("rgbKey", "key")),
                (1, "crypto_iv_or_nonce_input", ("rgbIV", "iv")),
            ),
            "CreateDecryptor": (
                (0, "crypto_key_input", ("rgbKey", "key")),
                (1, "crypto_iv_or_nonce_input", ("rgbIV", "iv")),
            ),
            "EncryptCbc": ((1, "crypto_iv_or_nonce_input", ("iv",)),),
            "DecryptCbc": ((1, "crypto_iv_or_nonce_input", ("iv",)),),
            "AesGcm": ((0, "crypto_key_input", ("key",)),),
            "AesCcm": ((0, "crypto_key_input", ("key",)),),
            "HMACMD5": ((0, "mac_key_input", ("key",)),),
            "HMACSHA1": ((0, "mac_key_input", ("key",)),),
            "HMACSHA256": ((0, "mac_key_input", ("key",)),),
            "HMACSHA384": ((0, "mac_key_input", ("key",)),),
            "HMACSHA512": ((0, "mac_key_input", ("key",)),),
            "HMACSHA3_256": ((0, "mac_key_input", ("key",)),),
            "HMACSHA3_384": ((0, "mac_key_input", ("key",)),),
            "HMACSHA3_512": ((0, "mac_key_input", ("key",)),),
            "Pbkdf2": (
                (0, "kdf_password_input", ("password",)),
                (1, "kdf_salt_input", ("salt",)),
            ),
            "Rfc2898DeriveBytes": (
                (0, "kdf_password_input", ("password",)),
                (1, "kdf_salt_input", ("salt",)),
            ),
        }
    else:
        call_specs = {
            "AESGCM": ((0, "crypto_key_input", ("key",)),),
            "AESCCM": ((0, "crypto_key_input", ("key",)),),
            "Fernet": ((0, "crypto_key_input", ("key",)),),
            "AES": ((0, "crypto_key_input", ("key",)),),
            "CBC": ((0, "crypto_iv_or_nonce_input", ("iv",)),),
            "CTR": ((0, "crypto_iv_or_nonce_input", ("nonce",)),),
            "GCM": ((0, "crypto_iv_or_nonce_input", ("iv", "nonce")),),
            "HMAC": ((0, "mac_key_input", ("key",)),),
            "PBKDF2HMAC": (
                (2, "kdf_salt_input", ("salt",)),
            ),
            "Scrypt": (
                (0, "kdf_salt_input", ("salt",)),
            ),
            "pbkdf2_hmac": (
                (2, "kdf_salt_input", ("salt",)),
            ),
            "scrypt": (
                (1, "kdf_salt_input", ("salt",)),
            ),
        }

    for call_name, specs in call_specs.items():
        for call in analyzer.calls(call_name):
            if (
                analyzer.language == "csharp"
                and call_name in {"Pbkdf2", "Rfc2898DeriveBytes"}
                and not _is_csharp_framework_pbkdf2_call(analyzer, call)
            ):
                continue
            if (
                analyzer.language == "python"
                and call_name == "pbkdf2_hmac"
                and not _is_python_hashlib_pbkdf2_call(analyzer, call)
            ):
                continue
            if (
                analyzer.language == "python"
                and call_name == "scrypt"
                and not _is_python_hashlib_scrypt_call(analyzer, call)
            ):
                continue
            for argument_index, sink_id, names in specs:
                expression = (
                    _python_kdf_salt_argument(
                        analyzer,
                        call,
                        argument_index,
                    )
                    if isinstance(analyzer, PythonAnalyzer)
                    and sink_id == "kdf_salt_input"
                    and call_name in {"PBKDF2HMAC", "Scrypt"}
                    else _call_argument(call, argument_index, *names)
                )
                if expression is None:
                    continue
                yield sink_id, call.callee, expression, call.start, call.end

    if analyzer.language == "typescript":
        for call, _canonical in _typescript_kdf_calls(
            analyzer,
            ("pbkdf2", "pbkdf2Sync", "scrypt", "scryptSync"),
        ):
            expression = _call_argument(call, 1, "salt")
            if expression is not None:
                yield (
                    "kdf_salt_input",
                    call.callee,
                    expression,
                    call.start,
                    call.end,
                )

    if isinstance(analyzer, CStyleAnalyzer):
        for assignment in analyzer.assignments:
            compact_name = re.sub(r"\s+", "", assignment.name)
            if "." not in compact_name:
                continue
            if (
                analyzer.language == "csharp"
                and not _csharp_provider_property_reaches_terminal(
                    analyzer,
                    compact_name,
                    assignment.end,
                )
            ):
                continue
            property_name = compact_name.rsplit(".", 1)[-1]
            if property_name == "Key":
                sink_id = "crypto_key_input"
            elif property_name == "IV":
                sink_id = "crypto_iv_or_nonce_input"
            else:
                continue
            yield (
                sink_id,
                compact_name,
                assignment.expression,
                assignment.start,
                assignment.end,
            )

    if analyzer.language == "csharp":
        for call in analyzer.calls("Encrypt", "Decrypt"):
            if not _csharp_aead_call(analyzer, call):
                continue
            expression = _call_argument(call, 0, "nonce")
            if expression is not None:
                yield (
                    "crypto_iv_or_nonce_input",
                    call.callee,
                    expression,
                    call.start,
                    call.end,
                )
    elif analyzer.language == "python":
        for call in analyzer.calls("encrypt", "decrypt"):
            if not _python_aead_call(analyzer, call):
                continue
            expression = _call_argument(call, 0, "nonce")
            if expression is not None:
                yield (
                    "crypto_iv_or_nonce_input",
                    call.callee,
                    expression,
                    call.start,
                    call.end,
                )
        if "PBKDF2HMAC" in analyzer.source:
            for call in analyzer.calls("derive", "verify"):
                if not _python_kdf_instance_call(analyzer, call):
                    continue
                expression = _call_argument(call, 0, "key_material")
                if expression is not None:
                    yield (
                        "kdf_password_input",
                        call.callee,
                        expression,
                        call.start,
                        call.end,
                    )


def _csharp_provider_property_reaches_terminal(
    analyzer: CStyleAnalyzer,
    property_name: str,
    after: int,
) -> bool:
    return (
        _csharp_provider_property_terminal_direction(
            analyzer,
            property_name,
            after,
        )
        is not None
    )


def _csharp_aead_call(analyzer: SourceAnalyzer, call: CallExpression) -> bool:
    compact = re.sub(r"\s+", "", call.callee)
    receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
    if not receiver:
        return False
    name = receiver.rsplit(".", 1)[-1]
    if isinstance(analyzer, CStyleAnalyzer):
        assignment = analyzer.latest_assignment(name, call.start)
        if assignment is not None and re.search(
            r"\b(?:new\s+)?Aes(?:Gcm|Ccm)\b",
            assignment.expression,
        ):
            return True
    return bool(
        re.search(
            rf"\bAes(?:Gcm|Ccm)\s+{re.escape(name)}\b",
            analyzer.source[: call.start],
        )
    )


def _python_aead_call(analyzer: SourceAnalyzer, call: CallExpression) -> bool:
    call_source = analyzer.source[call.start : call.end]
    if re.match(
        r"\s*AES(?:GCM|CCM)\s*\([^)]*\)\s*\.\s*(?:encrypt|decrypt)\s*\(",
        call_source,
    ):
        return True
    compact = re.sub(r"\s+", "", call.callee)
    receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
    if not receiver:
        return False
    name = receiver.rsplit(".", 1)[-1]
    return bool(
        re.search(
            rf"\b{re.escape(name)}\s*=\s*AES(?:GCM|CCM)\s*\(",
            analyzer.source[: call.start],
        )
    )


def _python_kdf_instance_call(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    call_source = analyzer.source[call.start : call.end]
    if re.match(
        r"\s*(?:(?:[A-Za-z_]\w*)\s*\.\s*)*(?:PBKDF2HMAC|Scrypt)"
        r"\s*\(.*\)\s*\.\s*"
        r"(?:derive|verify)\s*\(",
        call_source,
        flags=re.DOTALL,
    ):
        return True
    compact = re.sub(r"\s+", "", call.callee)
    receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
    if not receiver:
        return False
    name = receiver.rsplit(".", 1)[-1]
    return bool(
        re.search(
            rf"\b{re.escape(name)}\s*=\s*"
            rf"(?:(?:[A-Za-z_]\w*)\.)*(?:PBKDF2HMAC|Scrypt)\s*\(",
            analyzer.source[: call.start],
        )
    )


def _python_hkdf_instance_call(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    call_source = analyzer.source[call.start : call.end]
    if re.match(
        r"\s*HKDF\s*\(.*\)\s*\.\s*derive\s*\(",
        call_source,
        flags=re.DOTALL,
    ):
        return True
    compact = re.sub(r"\s+", "", call.callee)
    receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
    if not receiver:
        return False
    name = receiver.rsplit(".", 1)[-1]
    return bool(
        re.search(
            rf"\b{re.escape(name)}\s*=\s*HKDF\s*\(",
            analyzer.source[: call.start],
        )
    )


def _is_random_source_call(analyzer: SourceAnalyzer, call: CallExpression) -> bool:
    """Reject same-named non-random APIs before applying the M05 policy."""

    compact = re.sub(r"\s+", "", call.callee)
    simple = call.simple_name
    if analyzer.language == "typescript":
        canonical = simple
        if isinstance(analyzer, CStyleAnalyzer):
            canonical = next(
                (
                    imported
                    for imported, alias in analyzer.import_aliases.items()
                    if alias == simple
                ),
                simple,
            )
        if simple in {"m05CreateBytes", "m05RandomSource"}:
            return True
        if simple == "random":
            return compact == "Math.random"
        if simple == "imul":
            return compact == "Math.imul"
        if canonical in {
            "randomBytes",
            "randomFill",
            "randomFillSync",
            "cryptoRandomFillSync",
            "randomInt",
        }:
            return _is_typescript_crypto_function_call(
                analyzer,
                call,
                {"randomBytes", "randomFill", "randomFillSync", "randomInt"},
            )
        if canonical == "getRandomValues":
            receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
            return receiver in {"crypto", "globalThis.crypto", "window.crypto"}
        return False
    if analyzer.language == "csharp":
        if simple in {"m05CreateBytes", "m05RandomSource"}:
            return True
        receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
        if simple in {"GetBytes", "Fill", "GetInt32"}:
            if re.search(r"(?:Encoding|Convert|BitConverter|Buffer)$", receiver):
                return False
            if re.search(
                r"(?:RandomNumberGenerator|RNGCryptoServiceProvider)",
                receiver,
            ):
                return True
            if receiver:
                resolved = analyzer.resolve_origin(receiver, call.start)
                if resolved.origin == "secure_random":
                    return True
            return bool(
                receiver
                and re.search(
                    rf"\b(?:RandomNumberGenerator|RNGCryptoServiceProvider)\s+"
                    rf"{re.escape(receiver.rsplit('.', 1)[-1])}\b",
                    analyzer.source,
                )
            )
        if simple in {"NextBytes", "Next", "NextInt64"}:
            if compact in {
                "Random.Shared.NextBytes",
                "System.Random.Shared.NextBytes",
                "Random.Shared.Next",
                "System.Random.Shared.Next",
                "Random.Shared.NextInt64",
                "System.Random.Shared.NextInt64",
            }:
                return True
            if receiver:
                resolved = analyzer.resolve_origin(receiver, call.start)
                if resolved.origin == "non_csprng":
                    return True
            return bool(
                receiver
                and re.search(
                    rf"\b(?:System\.)?Random\s+{re.escape(receiver.rsplit('.', 1)[-1])}\b",
                    analyzer.source,
                )
            )
        return False
    if analyzer.language == "python":
        if simple in {"_m05_random_source", "_m05_create_bytes"}:
            return True
        if compact in {
            "os.urandom",
            "secrets.token_bytes",
            "secrets.token_hex",
            "secrets.token_urlsafe",
            "secrets.randbits",
            "secrets.randbelow",
            "secrets.choice",
            "random.random",
            "random.randbytes",
            "random.randint",
            "random.randrange",
        }:
            return True
        return simple in {"randbits", "randbelow", "randbytes", "randint", "randrange"}
    return False


def _is_typescript_crypto_api_call(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    return _is_typescript_crypto_function_call(
        analyzer,
        call,
        {"generateKeyPair", "generateKeyPairSync"},
    )


def _typescript_kdf_calls(
    analyzer: SourceAnalyzer,
    canonical_names: tuple[str, ...],
) -> list[tuple[CallExpression, str]]:
    """Return direct and ``promisify(kdf)`` Node KDF callsites."""

    records: list[tuple[CallExpression, str]] = []
    seen: set[tuple[int, int]] = set()
    for canonical in canonical_names:
        for call in analyzer.calls(canonical):
            if (
                isinstance(analyzer, CStyleAnalyzer)
                and analyzer.language == "typescript"
                and not _is_typescript_crypto_function_call(
                    analyzer,
                    call,
                    set(canonical_names),
                )
            ):
                continue
            key = (call.start, call.end)
            if key in seen:
                continue
            seen.add(key)
            records.append((call, canonical))
    if not isinstance(analyzer, CStyleAnalyzer) or analyzer.language != "typescript":
        return records

    allowed = "|".join(re.escape(name) for name in canonical_names)
    alias_pattern = re.compile(
        rf"\b(?:const|let|var)\s+(?P<alias>[A-Za-z_$][A-Za-z0-9_$]*)"
        rf"(?:\s*:[^=;]+)?\s*=\s*(?:promisify|util\s*\.\s*promisify)"
        rf"\s*\(\s*(?P<canonical>{allowed})\s*\)"
    )
    for match in alias_pattern.finditer(analyzer.source):
        alias = match.group("alias")
        canonical = match.group("canonical")
        if analyzer.import_aliases.get(canonical) != canonical:
            continue
        for call in analyzer.calls(alias):
            if call.start <= match.end():
                continue
            key = (call.start, call.end)
            if key in seen:
                continue
            seen.add(key)
            records.append((call, canonical))
    for assignment in analyzer.assignments:
        assigned = re.sub(r"\s+", "", assignment.expression)
        if "." not in assigned:
            continue
        receiver, canonical = assigned.rsplit(".", 1)
        if (
            canonical not in canonical_names
            or receiver not in analyzer.crypto_namespace_aliases
        ):
            continue
        alias = assignment.name.rsplit(".", 1)[-1]
        for call in analyzer.calls(alias):
            if call.start <= assignment.end:
                continue
            key = (call.start, call.end)
            if key in seen:
                continue
            seen.add(key)
            records.append((call, canonical))
    return records


def _is_typescript_crypto_function_call(
    analyzer: SourceAnalyzer,
    call: CallExpression,
    canonical_names: set[str],
) -> bool:
    if not isinstance(analyzer, CStyleAnalyzer) or analyzer.language != "typescript":
        return False
    compact = re.sub(r"\s+", "", call.callee)
    simple = call.simple_name
    imported = next(
        (
            original
            for original, alias in analyzer.import_aliases.items()
            if alias == simple
        ),
        "",
    )
    if imported in canonical_names:
        return True
    receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
    if simple in canonical_names and receiver in analyzer.crypto_namespace_aliases:
        return True
    if receiver:
        return False
    assignment = analyzer.latest_assignment(simple, call.start)
    if assignment is None:
        # Small snippets and generated programs frequently omit the import
        # while still using Node's conventional bare API name. Accept that
        # form only when no project-local declaration shadows the API.
        return bool(
            simple in canonical_names
            and not analyzer.function(simple)
            and re.search(
                rf"\b(?:function|class)\s+{re.escape(simple)}\b",
                analyzer.source,
            )
            is None
        )
    assigned = re.sub(r"\s+", "", assignment.expression)
    if "." not in assigned:
        return False
    assigned_receiver, assigned_name = assigned.rsplit(".", 1)
    return bool(
        assigned_name in canonical_names
        and assigned_receiver in analyzer.crypto_namespace_aliases
    )


def _is_python_hashlib_pbkdf2_call(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    return _python_hashlib_kdf_call_target(analyzer, call) == (
        "hashlib.pbkdf2_hmac"
    )


def _is_python_rsa_generate_private_key_call(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    """Return whether a call is PyCA's RSA key generator.

    A bare or member call named ``generate_private_key`` can be an ordinary
    application wrapper. Canonical import resolution keeps those wrappers out
    while retaining module aliases and directly imported PyCA functions.
    """

    if not isinstance(analyzer, PythonAnalyzer):
        return False
    if call.callee == (
        "cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key"
    ):
        return True
    head, separator, tail = call.callee.partition(".")
    canonical_head = analyzer.import_aliases.get(head, head)
    canonical = canonical_head + (separator + tail if separator else "")
    return canonical == (
        "cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key"
    )


def _is_python_hashlib_scrypt_call(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    return _python_hashlib_kdf_call_target(analyzer, call) == "hashlib.scrypt"


def _python_hashlib_kdf_call_target(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> str | None:
    """Resolve direct and source-local aliases of native hashlib KDFs."""

    if not isinstance(analyzer, PythonAnalyzer):
        return None
    target = analyzer.resolve_call_target_strings(call)
    candidates = (
        call.callee,
        target.expression,
        *(step.expression for step in target.steps),
    )
    for candidate in candidates:
        compact = re.sub(r"\s+", "", candidate)
        head, separator, tail = compact.partition(".")
        canonical = analyzer.import_aliases.get(head, head) + (
            separator + tail if separator else ""
        )
        if canonical in {"hashlib.pbkdf2_hmac", "hashlib.scrypt"}:
            return canonical
    return None


def _python_kdf_salt_argument(
    analyzer: PythonAnalyzer,
    call: CallExpression,
    default_index: int,
) -> str | None:
    """Return the semantic salt argument for native or local KDF wrappers.

    Generated programs often define a source-local class named
    ``PBKDF2HMAC`` whose constructor order differs from PyCA's. Prefer an
    explicit keyword, then the local constructor signature, then a clearly
    salt-named positional value before falling back to the native API slot.
    """

    if (named := call.named_arguments.get("salt")) is not None:
        return named

    simple_name = call.simple_name
    parameter_index: int | None = None
    local_definition_found = False
    for node in ast.walk(analyzer.tree):
        definitions: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...] = ()
        if isinstance(node, ast.ClassDef) and node.name == simple_name:
            local_definition_found = True
            definitions = tuple(
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == "__init__"
            )
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == simple_name
        ):
            local_definition_found = True
            definitions = (node,)
        for definition in definitions:
            parameters = list(definition.args.posonlyargs) + list(
                definition.args.args
            )
            if parameters and parameters[0].arg in {"self", "cls"}:
                parameters = parameters[1:]
            for index, parameter in enumerate(parameters):
                if parameter.arg.lower() == "salt":
                    parameter_index = index
                    break
            if parameter_index is not None:
                break
        if parameter_index is not None:
            break
    if parameter_index is not None and parameter_index < len(call.arguments):
        return call.arguments[parameter_index]
    if local_definition_found:
        return None

    salt_named = tuple(
        argument
        for argument in call.arguments
        if re.fullmatch(
            r"(?:self\s*\.\s*)?[A-Za-z_]\w*"
            r"(?:\s*\.\s*[A-Za-z_]\w*)?",
            argument.strip(),
        )
        and "salt" in re.sub(r"\W", "", argument).lower()
    )
    if len(salt_named) == 1:
        return salt_named[0]
    return _call_argument(call, default_index, "salt")


def _random_source_api_origin(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> ValueOrigin | None:
    """Return an origin when the called API itself determines randomness quality."""

    compact = re.sub(r"\s+", "", call.callee)
    simple = call.simple_name
    if analyzer.language == "typescript":
        canonical = simple
        if isinstance(analyzer, CStyleAnalyzer):
            canonical = next(
                (
                    imported
                    for imported, alias in analyzer.import_aliases.items()
                    if alias == simple
                ),
                simple,
            )
        if canonical in {
            "randomBytes",
            "randomFill",
            "randomFillSync",
            "cryptoRandomFillSync",
            "randomInt",
            "getRandomValues",
        } and _is_random_source_call(analyzer, call):
            return "secure_random"
        if compact == "Math.random":
            return "non_csprng"
        if compact == "Math.imul":
            # The corpus uses Math.imul as the state transition of a custom
            # linear-congruential generator. A fresh seed cannot turn that
            # deterministic generator into a CSPRNG.
            return "non_csprng"
    elif analyzer.language == "csharp":
        if simple in {"GetBytes", "Fill", "GetInt32"} and _is_random_source_call(
            analyzer,
            call,
        ):
            return "secure_random"
        if simple in {"NextBytes", "Next", "NextInt64"}:
            return "non_csprng"
    elif analyzer.language == "python":
        if isinstance(analyzer, PythonAnalyzer):
            return analyzer.native_random_call_origin(call.callee, call.start)
        if compact in {
            "os.urandom",
            "secrets.token_bytes",
            "secrets.token_hex",
            "secrets.token_urlsafe",
            "secrets.randbits",
            "secrets.randbelow",
            "secrets.choice",
        }:
            return "secure_random"
        if compact in {
            "random.random",
            "random.randbytes",
            "random.getrandbits",
            "random.randint",
            "random.randrange",
        }:
            return "non_csprng"
    return None


def _has_csharp_symmetric_terminal(source: str) -> bool:
    return bool(re.search(r"\b(?:CreateEncryptor|CreateDecryptor|EncryptCbc|DecryptCbc|AesGcm|AesCcm)\b", source))


def _has_csharp_rsa_terminal(source: str) -> bool:
    return bool(re.search(r"\b(?:Encrypt|Decrypt|SignData|VerifyData|ExportParameters|ExportRSAPublicKey)\s*\(", source))


def _is_csharp_rsa_key_size_assignment(
    analyzer: SourceAnalyzer,
    assignment_name: str,
    task_id: str | None,
) -> bool:
    if assignment_name.rsplit(".", 1)[-1] != "KeySize":
        return False
    if "." not in assignment_name:
        return bool(
            task_id
            and task_id.startswith(("F3", "F4"))
            and re.search(r"\bRSA\s*\.\s*Create\s*\(", analyzer.source)
        )
    receiver = assignment_name.rsplit(".", 1)[0].rsplit(".", 1)[-1]
    return bool(
        re.search(rf"\bRSA\s+{re.escape(receiver)}\b", analyzer.source)
        or re.search(
            rf"\bvar\s+{re.escape(receiver)}\s*=\s*(?:global::)?(?:System\.Security\.Cryptography\.)?RSA\s*\.\s*Create\s*\(",
            analyzer.source,
        )
    )


def _has_hash_terminal(analyzer: SourceAnalyzer) -> bool:
    if analyzer.language == "csharp" and analyzer.calls("TransformFinalBlock"):
        return True
    if (
        analyzer.language == "csharp"
        and re.search(r"\.\s*Hash\b", analyzer.source)
        and re.search(
            r"\b(?:MD5|SHA1|SHA224|SHA256|SHA384|SHA512|"
            r"SHA3_256|SHA3_384|SHA3_512)\s*\.\s*Create\s*\(",
            analyzer.source,
        )
    ):
        return True
    if (
        analyzer.language == "typescript"
        and analyzer.calls("end")
        and re.search(r"\bcreateHash\s*\(", analyzer.source)
    ):
        return True
    return bool(
        analyzer.calls(
            "digest",
            "hexdigest",
            "finalize",
            "HashData",
            "ComputeHash",
            "GetHashAndReset",
            "sign",
            "verify",
            "SignData",
            "VerifyData",
            "SignHash",
            "VerifyHash",
        )
    )


def _has_mac_terminal(analyzer: SourceAnalyzer) -> bool:
    return bool(
        analyzer.calls(
            "digest",
            "hexdigest",
            "doFinal",
            "finalize",
            "verify",
            "ComputeHash",
            "GetHashAndReset",
            "HashData",
        )
    )


def _csharp_uses_cbc(analyzer: SourceAnalyzer) -> bool:
    if re.search(
        r"CipherMode\s*\.\s*CBC|\bEncryptCbc\s*\(|\bDecryptCbc\s*\(",
        analyzer.source,
    ):
        return True
    if re.search(
        r"CipherMode\s*\.\s*(?:ECB|CFB|CTS|OFB)",
        analyzer.source,
    ):
        return False
    default_cbc_provider = re.search(
        r"\b(?:Aes|DES|TripleDES|Rijndael)\s*\.\s*Create\s*\("
        r"|\bnew\s+(?:Aes|DES|TripleDES|Rijndael)"
        r"(?:Managed|CryptoServiceProvider)?\s*\(",
        analyzer.source,
    )
    return bool(
        default_cbc_provider
        and _has_csharp_symmetric_terminal(analyzer.source)
    )


def _algorithm_from_csharp_callee(callee: str) -> str:
    normalized = re.sub(r"\s+", "", callee)
    normalized = re.sub(r"^new", "", normalized)
    for name in (
        "MD5", "SHA1", "SHA224", "SHA256", "SHA384", "SHA512",
        "SHA3_256", "SHA3_384", "SHA3_512",
    ):
        if re.search(
            rf"(?:^|\.){name}(?:CryptoServiceProvider|Managed|Cng)?"
            rf"(?:\(\))?(?:\.|$)",
            normalized,
        ):
            return name
    return ""


def _mac_from_csharp_callee(callee: str) -> str:
    normalized = re.sub(r"\s+", "", callee)
    normalized = re.sub(r"^new", "", normalized)
    for name in (
        "HMACMD5", "HMACSHA1", "HMACSHA256", "HMACSHA384", "HMACSHA512",
        "HMACSHA3_256", "HMACSHA3_384", "HMACSHA3_512",
    ):
        if re.search(rf"(?:^|\.){name}(?:\.|$)", normalized):
            return name
    return ""


def _typescript_signature_digest_findings(analyzer: SourceAnalyzer) -> list[DetectorFinding]:
    findings: list[DetectorFinding] = []
    for call in analyzer.calls("sign", "verify", "createSign", "createVerify"):
        if not call.arguments:
            continue
        if not _is_typescript_crypto_function_call(
            analyzer,
            call,
            {"sign", "verify", "createSign", "createVerify"},
        ):
            # signer.sign(key) and verifier.verify(key, signature) consume key
            # material. Their first argument is not a digest policy value.
            continue
        resolved = analyzer.resolve_strings(call.arguments[0], call.start)
        finding = _algorithm_finding(analyzer, "M09", call, call.arguments[0], resolved, _classify_signature_digest)
        findings.append(replace(finding, metadata={"signature_context": True}))
    return findings


def _csharp_signature_digest_findings(analyzer: SourceAnalyzer) -> list[DetectorFinding]:
    findings: list[DetectorFinding] = []
    for call in analyzer.calls("SignData", "VerifyData", "SignHash", "VerifyHash"):
        minimum_arity = 3 if call.simple_name in {"SignData", "SignHash"} else 4
        if "." not in re.sub(r"\s+", "", call.callee):
            continue
        if len(call.arguments) < minimum_arity:
            continue
        position = 1 if call.simple_name in {"SignData", "SignHash"} else 2
        expression = _call_argument(
            call,
            position,
            "hashAlgorithm",
            "algorithm",
        )
        if expression is None:
            continue
        resolved = analyzer.resolve_strings(expression, call.start)
        finding = _algorithm_finding(
            analyzer,
            "M09",
            call,
            expression,
            resolved,
            _classify_signature_digest,
        )
        findings.append(replace(finding, metadata={"signature_context": True}))
    return findings


def _python_signature_digest_findings(analyzer: SourceAnalyzer) -> list[DetectorFinding]:
    findings: list[DetectorFinding] = []
    for call in analyzer.calls("sign", "verify"):
        candidates = [argument for argument in call.arguments if "hashes." in argument]
        for expression in candidates:
            match = re.search(r"hashes\.([A-Za-z0-9_]+)\s*\(", expression)
            if not match:
                continue
            resolved = ResolvedValue(expression, "constant", (match.group(1),))
            finding = _algorithm_finding(analyzer, "M09", call, expression, resolved, _classify_signature_digest)
            findings.append(replace(finding, metadata={"signature_context": True}))
    return findings


def _classify_signature_digest(value: str) -> tuple[ProgramLabel, str, str]:
    normalized = _normalize_algorithm(value)
    # The legacy bare "SHA" alias is a prefix of every SHA-2/SHA-3 name.
    # Recognize approved digests first so SHA256withRSA is not classified as
    # broken merely because it contains that prefix.
    for digest in sorted(_SECURE_DIGESTS, key=len, reverse=True):
        if digest in normalized:
            return "secure", "approved_signature_digest", f"signature algorithm {value!r} uses an approved digest"
    for digest in sorted(_BROKEN_DIGESTS, key=len, reverse=True):
        if digest in normalized:
            return "misuse", "broken_signature_digest", f"signature algorithm {value!r} uses a broken digest"
    return "unknown", "unclassified_signature_digest", f"signature digest {value!r} is not classified"


def _python_password_argument(call: CallExpression) -> str | None:
    if "password" in call.named_arguments:
        return call.named_arguments["password"]
    if call.simple_name == "BestAvailableEncryption" and call.arguments:
        return call.arguments[0]
    if call.simple_name.startswith("load_") and len(call.arguments) > 1:
        return call.arguments[1]
    return None


def _is_csharp_framework_pbkdf2_call(
    analyzer: SourceAnalyzer,
    call: CallExpression,
) -> bool:
    """Distinguish native PBKDF2 calls from same-named project wrappers."""

    compact_callee = re.sub(r"\s+", "", call.callee)
    if compact_callee.startswith("newRfc2898DeriveBytes"):
        return True
    if compact_callee.endswith("Rfc2898DeriveBytes.Pbkdf2"):
        return True
    prefix = analyzer.source[max(0, call.start - 32) : call.start]
    if call.simple_name == "Rfc2898DeriveBytes" and re.search(
        r"\bnew\s*$",
        prefix,
    ):
        return True
    if call.simple_name == "Pbkdf2" and re.search(
        r"\busing\s+static\s+(?:System\.)?Security\.Cryptography\."
        r"Rfc2898DeriveBytes\s*;",
        analyzer.source,
    ):
        return True
    if isinstance(analyzer, CStyleAnalyzer) and analyzer.function(call.simple_name):
        return False
    return call.simple_name == "Rfc2898DeriveBytes"


def _csharp_provider_property_terminal_direction(
    analyzer: CStyleAnalyzer,
    property_name: str,
    after: int,
) -> str | None:
    receiver = property_name.rsplit(".", 1)[0]
    property_member = property_name.rsplit(".", 1)[-1]
    next_overwrite = min(
        (
            assignment.start
            for assignment in analyzer.assignments
            if assignment.start >= after
            and re.sub(r"\s+", "", assignment.name) == property_name
        ),
        default=len(analyzer.source),
    )
    directions: set[str] = set()
    for call in analyzer.calls():
        if call.start < after or call.start >= next_overwrite:
            continue
        callee = re.sub(r"\s+", "", call.callee)
        if callee in {f"{receiver}.CreateEncryptor", f"{receiver}.CreateDecryptor"}:
            # The parameterless overload consumes the provider's Key and IV.
            # The two-argument overload consumes the explicit arguments instead.
            if not call.arguments:
                directions.add(
                    "decrypt" if callee.endswith("CreateDecryptor") else "encrypt"
                )
            continue
        if callee in {f"{receiver}.EncryptCbc", f"{receiver}.DecryptCbc"}:
            # Span-based CBC helpers consume the provider key but receive their
            # IV explicitly as an argument.
            if property_member == "Key":
                directions.add(
                    "decrypt" if callee.endswith("DecryptCbc") else "encrypt"
                )
    if "encrypt" in directions:
        return "encrypt"
    if "decrypt" in directions:
        return "decrypt"
    return None
