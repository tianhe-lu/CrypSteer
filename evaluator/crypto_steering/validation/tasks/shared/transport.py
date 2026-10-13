"""Task-neutral representation and transport dataflow checks."""

from __future__ import annotations

import re

from crypto_steering.validation.java.static_analysis import extract_assignments
from crypto_steering.validation.source.index import analyzer_for


def java_base64_decode_failure(source: str) -> str | None:
    """Return a definite mismatch between Java bytes and Base64 text."""

    for match in re.finditer(
        r"Base64\s*\.\s*getDecoder\s*\(\s*\)\s*\.\s*decode\s*\(\s*"
        r"(?P<literal>[\"'][^\"']*[\"'])\s*\+",
        source,
        flags=re.IGNORECASE,
    ):
        literal = match.group("literal")[1:-1]
        if ":" in literal or re.search(r"[A-Za-z]+\s", literal):
            return "descriptive text is included in the Base64 ciphertext value"

    assignments = extract_assignments(source)
    analyzer = analyzer_for("java", source)
    for call in analyzer.calls("decode"):
        prefix = source[max(0, call.start - 96) : call.start]
        if not call.arguments or re.search(
            r"Base64\s*\.\s*getDecoder\s*\(\s*\)\s*\.\s*$",
            prefix,
        ) is None:
            continue
        name = call.arguments[0].strip()
        if re.fullmatch(r"[A-Za-z_$][\w$]*", name) is None:
            continue
        scope = analyzer.containing_function_name(call.start)
        candidates = [
            assignment
            for assignment in assignments
            if assignment.name.rsplit(".", 1)[-1] == name
            and assignment.start < call.start
            and analyzer.containing_function_name(assignment.start) == scope
        ]
        if not candidates:
            continue
        assignment = max(candidates, key=lambda item: item.start)
        expression = assignment.expression
        if (
            assignment.type_name.replace(" ", "") == "byte[]"
            and re.search(r"Base64\s*\.\s*getEncoder", expression) is None
            and re.search(
                r"(?:\.\s*doFinal\s*\(|new\s+byte\s*\[|\.\s*getBytes\s*\()",
                expression,
            )
        ):
            return "raw binary ciphertext is passed to the Base64 decoder"
    return None


__all__ = ("java_base64_decode_failure",)
