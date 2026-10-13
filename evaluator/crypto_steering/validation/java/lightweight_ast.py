"""Small Java AST-style helpers for targeted detector fallbacks.

This is not a full Java parser. It only builds the method and call nodes needed
for conservative interprocedural tracing in generated Java snippets.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from crypto_steering.validation.java.source import (
    CallSite,
    find_calls,
    find_matching_delimiter,
    mask_java_comments_and_literals,
    split_arguments_with_spans,
)
from crypto_steering.validation.java.static_analysis import Parameter, parse_parameter_declarations


@dataclass(frozen=True)
class JavaMethodNode:
    name: str
    params: tuple[Parameter, ...]
    start: int
    body_start: int
    body_end: int


_METHOD_CANDIDATE_RE = re.compile(
    r"\b(?P<name>[A-Za-z_]\w*)\s*"
    r"\((?P<params>[^()]*)\)\s*"
    r"(?:throws\s+[^{;]+)?\{",
    flags=re.MULTILINE,
)
_CONTROL_OR_BLOCK_NAMES = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "try",
    "do",
    "else",
    "synchronized",
}
_METHOD_MODIFIERS = {
    "public",
    "private",
    "protected",
    "static",
    "final",
    "synchronized",
    "native",
    "abstract",
    "strictfp",
    "default",
}


def parse_method_nodes(source: str) -> tuple[JavaMethodNode, ...]:
    masked_source = mask_java_comments_and_literals(source)
    methods: list[JavaMethodNode] = []
    for match in _METHOD_CANDIDATE_RE.finditer(masked_source):
        name = match.group("name")
        if name in _CONTROL_OR_BLOCK_NAMES:
            continue
        header_start = _method_header_start(masked_source, match.start())
        header = masked_source[header_start : match.start()]
        if not _looks_like_method_header(header):
            continue
        body_start = source.find("{", match.start(), match.end())
        if body_start == -1:
            continue
        body_end = find_matching_delimiter(source, body_start, "{", "}")
        if body_end == -1:
            continue
        methods.append(
            JavaMethodNode(
                name=name,
                params=parse_parameter_declarations(match.group("params")),
                start=header_start,
                body_start=body_start,
                body_end=body_end,
            )
        )
    return tuple(methods)


def _method_header_start(source: str, name_start: int) -> int:
    separator_positions = (
        source.rfind(";", 0, name_start),
        source.rfind("{", 0, name_start),
        source.rfind("}", 0, name_start),
    )
    return max(separator_positions) + 1


def _looks_like_method_header(header: str) -> bool:
    text = header.strip()
    if not text or text.endswith(".") or text.endswith("->"):
        return False
    if "=" in text or "new " in text:
        return False
    words = re.findall(r"[A-Za-z_]\w*", text)
    if not words:
        return False
    if any(word in {"class", "interface", "enum", "record", "return", "throw", "case"} for word in words):
        return False
    return bool([word for word in words if word not in _METHOD_MODIFIERS])


def method_containing_index(methods: tuple[JavaMethodNode, ...], index: int) -> JavaMethodNode | None:
    matches = [method for method in methods if method.body_start <= index <= method.body_end]
    if not matches:
        return None
    return min(matches, key=lambda method: method.body_end - method.body_start)


def find_method_calls(source: str, method_name: str) -> tuple[CallSite, ...]:
    regex = rf"(?<![\w$])(?:(?:this|[A-Za-z_]\w*)\s*\.\s*)?{re.escape(method_name)}\s*\("
    return _dedupe_calls(
        tuple(find_calls(source, regex, method_name)) + _find_unmasked_method_calls(source, method_name)
    )


def _find_unmasked_method_calls(source: str, method_name: str) -> tuple[CallSite, ...]:
    calls: list[CallSite] = []
    masked_source = mask_java_comments_and_literals(source)
    pattern = re.compile(
        rf"(?<![\w$])(?:(?:this|[A-Za-z_]\w*)\s*\.\s*)?{re.escape(method_name)}\s*\(",
        flags=re.MULTILINE,
    )
    for match in pattern.finditer(source):
        if not masked_source[match.start() : match.end()].strip():
            continue
        open_paren = source.find("(", match.start(), match.end() + 1)
        if open_paren == -1:
            continue
        close_paren = find_matching_delimiter(source, open_paren)
        if close_paren == -1:
            continue
        calls.append(
            CallSite(
                name=method_name,
                start=match.start(),
                end=close_paren + 1,
                open_paren=open_paren,
                close_paren=close_paren,
                arguments=split_arguments_with_spans(source, open_paren, close_paren),
            )
        )
    return tuple(calls)


def _dedupe_calls(calls: tuple[CallSite, ...]) -> tuple[CallSite, ...]:
    deduped: dict[tuple[int, int], CallSite] = {}
    for call in calls:
        deduped[(call.start, call.end)] = call
    return tuple(sorted(deduped.values(), key=lambda call: call.start))
