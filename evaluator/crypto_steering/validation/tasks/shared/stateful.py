"""Task-neutral state-machine primitives for stateful API objects."""

from __future__ import annotations

import re

from crypto_steering.validation.models import Language
from crypto_steering.validation.source.index import analyzer_for


def repeated_terminal_receiver(
    language: Language,
    source: str,
    *,
    setup_names: tuple[str, ...],
    terminal_names: tuple[str, ...],
) -> str | None:
    """Return a receiver finalized twice without replacement in one scope.

    Stateful digest, MAC, and cipher objects cannot provide a second terminal
    result after their first finalization.  Branch alternatives are not
    treated as sequential reuse.
    """

    analyzer = analyzer_for(language, source)
    assignments = tuple(getattr(analyzer, "assignments", ()))
    receiver_assignments = tuple(
        assignment
        for assignment in assignments
        if any(
            re.search(rf"\b{re.escape(name)}\s*\(", assignment.expression)
            for name in setup_names
        )
    )
    receivers = {_compact(assignment.name) for assignment in receiver_assignments}
    if not receivers:
        return None

    for assignment in receiver_assignments:
        receiver = _compact(assignment.name)
        owner = analyzer.containing_function_name(assignment.start)
        scopes = tuple(
            function
            for function in getattr(analyzer, "functions", ())
            if function.name == owner
            and function.start <= assignment.start <= function.end
        )
        if scopes:
            scope = min(scopes, key=lambda item: item.end - item.start)
            scope_source = source[scope.start : scope.end]
        else:
            function_ranges = tuple(
                (function.start, function.end)
                for function in getattr(analyzer, "functions", ())
            )
            pieces: list[str] = []
            cursor = 0
            for start, end in sorted(function_ranges):
                pieces.append(source[cursor:start])
                cursor = max(cursor, end)
            pieces.append(source[cursor:])
            scope_source = "\n".join(pieces)
        terminal = "|".join(re.escape(name) for name in terminal_names)
        occurrences = tuple(
            re.finditer(
                rf"(?<![\w$]){re.escape(receiver)}\s*\.\s*"
                rf"[^;\n]*?\b(?:{terminal})\s*\(",
                scope_source,
            )
        )
        if len(occurrences) > 1:
            return receiver

    grouped: dict[tuple[str | None, str], list[object]] = {}
    for call in analyzer.calls(*terminal_names):
        compact = _compact(call.callee)
        if "." not in compact:
            continue
        matching = [
            receiver
            for receiver in receivers
            if compact.startswith(receiver + ".")
        ]
        if not matching:
            continue
        receiver = max(matching, key=len)
        key = (analyzer.containing_function_name(call.start), receiver)
        grouped.setdefault(key, []).append(call)

    for (_scope, receiver), calls in grouped.items():
        ordered = sorted(calls, key=lambda item: item.start)
        for first, second in zip(ordered, ordered[1:]):
            if any(
                first.end < assignment.start < second.start
                and _compact(assignment.name) == receiver
                for assignment in assignments
            ):
                continue
            between = source[first.end : second.start]
            if re.search(r"\b(?:else|case|catch)\b", between):
                continue
            return receiver
    return None


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


__all__ = ("repeated_terminal_receiver",)
