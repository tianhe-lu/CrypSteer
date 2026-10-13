"""Java call discovery used by task checks and misuse detectors.

This module recognizes source locations and arguments only.  It deliberately
does not know which calls are secure or insecure; M01-M10 own those policies.
"""

from __future__ import annotations

import re

from crypto_steering.validation.java.source import (
    CallSite,
    find_calls,
    mask_java_comments_and_literals,
)


def constructor_calls(source: str, class_name: str) -> tuple[CallSite, ...]:
    regex = rf"\bnew\s+(?:[\w.]+\.)?{re.escape(class_name)}\s*\("
    return find_calls(source, regex, class_name)


def static_method_calls(
    source: str,
    class_name: str,
    method_name: str,
) -> tuple[CallSite, ...]:
    regex = (
        rf"\b(?:[\w.]+\.)?{re.escape(class_name)}\s*\.\s*"
        rf"{re.escape(method_name)}\s*\("
    )
    return find_calls(source, regex, f"{class_name}.{method_name}")


def method_calls(source: str, method_name: str) -> tuple[CallSite, ...]:
    regex = (
        rf"(?<![\w$])(?:[A-Za-z_]\w*\s*\.\s*)?"
        rf"{re.escape(method_name)}\s*\("
    )
    return find_calls(source, regex, method_name)


def receiver_method_calls(
    source: str,
    method_name: str,
) -> tuple[tuple[str, CallSite], ...]:
    calls: list[tuple[str, CallSite]] = []
    masked = mask_java_comments_and_literals(source)
    pattern = re.compile(
        rf"\b(?P<receiver>[A-Za-z_]\w*)\s*\.\s*"
        rf"{re.escape(method_name)}\s*\("
    )
    for match in pattern.finditer(masked):
        candidates = find_calls(source, re.escape(match.group(0)), method_name)
        for candidate in candidates:
            if candidate.start == match.start():
                calls.append((match.group("receiver"), candidate))
                break
    return tuple(calls)


def receivers_assigned_from_get_instance(
    source: str,
    factory_class: str,
) -> dict[str, str]:
    """Map a factory receiver to its getInstance algorithm expression."""

    escaped = re.escape(factory_class)
    pattern = re.compile(
        rf"(?:\b(?:{escaped}|[\w.]+\.{escaped})\s+)?"
        rf"(?P<name>[A-Za-z_]\w*)\s*=\s*"
        rf"(?:[\w.]+\.)?{escaped}\s*\.\s*getInstance\s*\("
    )
    receivers: dict[str, str] = {}
    for match in pattern.finditer(source):
        calls = static_method_calls(
            source[match.start() :],
            factory_class,
            "getInstance",
        )
        if calls and calls[0].arguments:
            receivers[match.group("name")] = calls[0].arguments[0].text
    return receivers


def source_contains_any_token(source: str, tokens: set[str]) -> bool:
    """Conservative masked-source prefilter for Java detector sinks."""

    if not tokens:
        return True
    masked = mask_java_comments_and_literals(source)
    return any(
        re.search(rf"(?<![\w$]){re.escape(token)}(?![\w$])", masked)
        for token in tokens
        if token
    )


def statement_text(source: str, index: int) -> str:
    start = max(
        source.rfind(";", 0, index),
        source.rfind("{", 0, index),
        source.rfind("}", 0, index),
    ) + 1
    end = source.find(";", index)
    return source[start : end + 1 if end != -1 else len(source)]


__all__ = (
    "constructor_calls",
    "method_calls",
    "receiver_method_calls",
    "receivers_assigned_from_get_instance",
    "source_contains_any_token",
    "statement_text",
    "static_method_calls",
)
