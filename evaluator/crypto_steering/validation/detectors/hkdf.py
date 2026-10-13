"""Shared Python HKDF role analysis for M01 and M08."""

from __future__ import annotations

import ast
import re
from typing import Callable

from crypto_steering.validation.models import (
    DetectorFinding,
    ProgramLabel,
    ResolvedValue,
)
from crypto_steering.validation.source.index import PythonAnalyzer, SourceAnalyzer
from crypto_steering.validation.detectors.apis import _python_hkdf_instance_call
from crypto_steering.validation.detectors.core import (
    _call_argument,
    _dedupe_findings,
    _key_label,
    _origin_finding,
    _password_label,
)


_HKDF_PASSWORD_ROLE = re.compile(
    r"(?i)^(?:password|passphrase|passwd|pwd|credential)$"
)
_HKDF_KEY_MATERIAL_ROLE = re.compile(
    r"(?i)^(?:(?:client|user|shared|master)?_?secret"
    r"(?:_?(?:key|material))?|key_?material|ikm)$"
)


def python_hkdf_password_findings(
    analyzer: SourceAnalyzer,
) -> list[DetectorFinding]:
    """Compatibility name for the M08-owned HKDF secret check."""

    return _python_hkdf_secret_findings(analyzer)


def python_hkdf_key_findings(
    analyzer: SourceAnalyzer,
) -> list[DetectorFinding]:
    """Compatibility name for the M01-owned HKDF key-material check."""

    return _python_hkdf_key_material_findings(analyzer)


def _python_hkdf_role_findings(
    analyzer: SourceAnalyzer,
    *,
    misuse_id: str,
    sink_id: str,
    role_pattern: re.Pattern[str],
    classifier: Callable[[ResolvedValue], tuple[ProgramLabel, str, str]],
    include_direct_constant: bool,
) -> list[DetectorFinding]:
    """Classify selected components rather than a mixed HKDF expression.

    HKDF input may combine a fresh challenge with a long-lived password or
    secret. Classifying the combined expression alone can hide hardcoded
    material behind the fresh component. Public challenge and context values
    remain outside these findings.
    """

    if not isinstance(analyzer, PythonAnalyzer) or "HKDF" not in analyzer.source:
        return []
    findings: list[DetectorFinding] = []
    for call in analyzer.calls("derive"):
        if not _python_hkdf_instance_call(analyzer, call):
            continue
        expression = _call_argument(call, 0, "key_material")
        if expression is None:
            continue
        candidates = _python_hkdf_role_components(expression, role_pattern)
        if (
            not candidates
            and include_direct_constant
            and _python_hkdf_direct_literal(expression)
        ):
            resolved = analyzer.resolve_origin(expression, call.start)
            if resolved.origin in {"constant", "predictable"}:
                candidates = (expression,)
        for candidate in candidates:
            findings.append(
                _origin_finding(
                    analyzer,
                    misuse_id,
                    call,
                    candidate,
                    sink_id,
                    classifier,
                )
            )
    return _dedupe_findings(findings)


def _python_hkdf_direct_literal(expression: str) -> bool:
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return False
    return isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes))


def _python_hkdf_role_components(
    expression: str,
    role_pattern: re.Pattern[str],
) -> tuple[str, ...]:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return ()
    candidates: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        candidate: str | None = None
        role = ""
        if isinstance(node, ast.Name):
            candidate = node.id
            role = node.id
        elif isinstance(node, ast.Attribute):
            candidate = ast.get_source_segment(expression, node) or ast.unparse(node)
            role = node.attr
        if candidate is None:
            continue
        normalized_role = re.sub(r"[^A-Za-z0-9_]", "", role)
        if role_pattern.fullmatch(normalized_role) is None:
            continue
        candidates.append((getattr(node, "col_offset", 0), candidate))
    return tuple(dict.fromkeys(candidate for _, candidate in sorted(candidates)))


def _python_hkdf_secret_findings(
    analyzer: SourceAnalyzer,
) -> list[DetectorFinding]:
    """Inspect password-role components of Python HKDF input key material."""

    return _python_hkdf_role_findings(
        analyzer,
        misuse_id="M08",
        sink_id="hkdf_password_input",
        role_pattern=_HKDF_PASSWORD_ROLE,
        classifier=_password_label,
        include_direct_constant=True,
    )


def _python_hkdf_key_material_findings(
    analyzer: SourceAnalyzer,
) -> list[DetectorFinding]:
    """Inspect explicit secret/key-role components entering Python HKDF."""

    return _python_hkdf_role_findings(
        analyzer,
        misuse_id="M01",
        sink_id="hkdf_key_material_input",
        role_pattern=_HKDF_KEY_MATERIAL_ROLE,
        classifier=_key_label,
        include_direct_constant=False,
    )
