"""Reachability checks for Java source positions.

The base value-flow graph establishes entry paths.  This module additionally
filters branches selected by statically fixed ``if`` and ``switch`` values.
It does not contain misuse-category policy.
"""

from __future__ import annotations

import re

from crypto_steering.validation.java.lightweight_ast import (
    find_method_calls,
    method_containing_index,
)
from crypto_steering.validation.java.resolution import (
    java_string_literal_value,
    resolve_boolean_expression,
    resolve_int_expression,
    resolve_string_expression_values,
)
from crypto_steering.validation.java.security_value_flow import SecurityValueFlow
from crypto_steering.validation.java.source import (
    find_matching_delimiter,
    mask_java_comments_and_literals,
)


def position_is_reachable(
    source: str,
    flow: SecurityValueFlow,
    index: int,
    seen_methods: frozenset[int] = frozenset(),
) -> bool:
    """Suppress positions excluded by entry paths and fixed selectors."""

    if not flow.position_has_entry_path(index):
        return False

    containing = method_containing_index(flow.methods, index)
    if (
        containing is not None
        and containing.name != "main"
        and any(method.name == "main" for method in flow.methods)
    ):
        if containing.start in seen_methods:
            return False
        next_seen = seen_methods | frozenset({containing.start})
        reachable_caller = False
        for call in find_method_calls(source, containing.name):
            if any(
                candidate.start <= call.start <= candidate.body_start
                for candidate in flow.methods
            ):
                continue
            if containing.body_start <= call.start <= containing.body_end:
                continue
            if len(call.arguments) != len(containing.params):
                continue
            if position_is_reachable(
                source,
                flow,
                call.start,
                next_seen,
            ):
                reachable_caller = True
                break
        if not reachable_caller:
            return False

    masked = mask_java_comments_and_literals(source)
    for match in re.finditer(r"\bswitch\s*\(", masked):
        opening = masked.find("(", match.start(), match.end())
        closing = find_matching_delimiter(masked, opening)
        if closing == -1:
            continue
        body_open = closing + 1
        while body_open < len(masked) and masked[body_open].isspace():
            body_open += 1
        if body_open >= len(masked) or masked[body_open] != "{":
            continue
        body_close = find_matching_delimiter(masked, body_open, "{", "}")
        if body_close == -1 or not (body_open < index < body_close):
            continue
        selector_expression = source[opening + 1 : closing].strip()
        selector_expression = re.sub(
            r"\.\s*to(?:Upper|Lower)Case\s*\(\s*\)\s*$",
            "",
            selector_expression,
        ).strip()
        selected = resolve_string_expression_values(
            selector_expression,
            flow,
            match.start(),
        )
        if selected is None or len(selected) != 1:
            continue
        selected_value = _normalize_selector(next(iter(selected)))
        labels: list[tuple[int, int, str | None]] = []
        for label_match in re.finditer(
            r"\b(?P<kind>case|default)\b(?P<label>[^:]*)\:",
            masked[body_open + 1 : body_close],
        ):
            absolute_start = body_open + 1 + label_match.start()
            absolute_end = body_open + 1 + label_match.end()
            label = None
            if label_match.group("kind") == "case":
                label_source = source[
                    body_open + 1 + label_match.start("label") :
                    body_open + 1 + label_match.end("label")
                ].strip()
                literal = java_string_literal_value(label_source)
                if literal is not None:
                    label = _normalize_selector(literal)
            labels.append((absolute_start, absolute_end, label))
        for label_index, (_, label_end, label) in enumerate(labels):
            segment_end = (
                labels[label_index + 1][0]
                if label_index + 1 < len(labels)
                else body_close
            )
            if not (label_end <= index < segment_end):
                continue
            if label is None:
                explicit = {item[2] for item in labels if item[2] is not None}
                if selected_value in explicit:
                    return False
            elif label != selected_value:
                return False

    for match in re.finditer(r"\bif\s*\(", masked):
        opening = masked.find("(", match.start(), match.end())
        closing = find_matching_delimiter(masked, opening)
        if closing == -1:
            continue
        body_open, body_close = _statement_bounds(masked, closing + 1)
        if body_open == -1:
            continue
        in_then = body_open <= index < body_close
        in_else = False
        cursor = body_close
        while cursor < len(masked) and masked[cursor].isspace():
            cursor += 1
        if masked.startswith("else", cursor):
            else_open, else_close = _statement_bounds(
                masked,
                cursor + len("else"),
            )
            # javalang reports an unbraced else-assignment from the ``else``
            # keyword rather than from the assigned identifier. Include that
            # prefix in the branch bounds so fixed conditions suppress it.
            in_else = else_open != -1 and cursor <= index < else_close
        if not in_then and not in_else:
            continue
        condition = source[opening + 1 : closing]
        condition_value = _resolve_condition(condition, flow, match.start())
        if condition_value is None:
            condition_value = _resolve_condition_from_callers(
                condition,
                source,
                flow,
                match.start(),
                seen_methods,
            )
        if condition_value is None:
            continue
        if in_then and not condition_value:
            return False
        if in_else and condition_value:
            return False
    return True


def _statement_bounds(masked: str, start: int) -> tuple[int, int]:
    cursor = start
    while cursor < len(masked) and masked[cursor].isspace():
        cursor += 1
    if cursor >= len(masked):
        return -1, -1
    if masked[cursor] == "{":
        closing = find_matching_delimiter(masked, cursor, "{", "}")
        return (cursor, closing + 1) if closing != -1 else (-1, -1)
    if re.match(r"if\b", masked[cursor:]):
        opening = masked.find("(", cursor)
        if opening == -1:
            return -1, -1
        closing = find_matching_delimiter(masked, opening)
        if closing == -1:
            return -1, -1
        _, then_end = _statement_bounds(masked, closing + 1)
        if then_end == -1:
            return -1, -1
        after_then = then_end
        while after_then < len(masked) and masked[after_then].isspace():
            after_then += 1
        if masked.startswith("else", after_then):
            _, else_end = _statement_bounds(
                masked,
                after_then + len("else"),
            )
            if else_end != -1:
                return cursor, else_end
        return cursor, then_end

    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for position in range(cursor, len(masked)):
        char = masked[position]
        if char in "([{":
            stack.append(char)
        elif char in ")]}" and stack and stack[-1] == pairs[char]:
            stack.pop()
        elif char == ";" and not stack:
            return cursor, position + 1
    return -1, -1


def _resolve_condition(
    condition: str,
    flow: SecurityValueFlow,
    index: int,
) -> bool | None:
    value = condition.strip()
    lookup_result = _resolve_lookup_index_condition(value, flow, index)
    if lookup_result is not None:
        return lookup_result
    equals = re.fullmatch(
        r"(?P<left>.+?)\s*\.\s*equals(?:IgnoreCase)?\s*"
        r"\(\s*(?P<right>.+)\s*\)",
        value,
        flags=re.DOTALL,
    )
    if equals is not None:
        left = resolve_string_expression_values(
            equals.group("left"),
            flow,
            index,
        )
        right = resolve_string_expression_values(
            equals.group("right"),
            flow,
            index,
        )
        if left is not None and right is not None and len(left) == len(right) == 1:
            return _normalize_selector(next(iter(left))) == _normalize_selector(
                next(iter(right))
            )
        return None
    objects_equals = re.fullmatch(
        r"(?:Objects|java\s*\.\s*util\s*\.\s*Objects)\s*\.\s*equals\s*"
        r"\(\s*(?P<left>.+?)\s*,\s*(?P<right>.+?)\s*\)",
        value,
        flags=re.DOTALL,
    )
    if objects_equals is not None:
        left = resolve_string_expression_values(
            objects_equals.group("left"),
            flow,
            index,
        )
        right = resolve_string_expression_values(
            objects_equals.group("right"),
            flow,
            index,
        )
        if left is not None and right is not None and len(left) == len(right) == 1:
            return _normalize_selector(next(iter(left))) == _normalize_selector(
                next(iter(right))
            )
        return None
    comparison = re.fullmatch(
        r"(?P<left>.+?)\s*(?P<operator>==|!=)\s*(?P<right>.+)",
        value,
        flags=re.DOTALL,
    )
    if comparison is not None:
        left_int = resolve_int_expression(
            comparison.group("left"),
            flow,
            index,
        )
        right_int = resolve_int_expression(
            comparison.group("right"),
            flow,
            index,
        )
        if left_int is not None and right_int is not None:
            equal = left_int == right_int
            return (
                not equal
                if comparison.group("operator") == "!="
                else equal
            )
        left = resolve_string_expression_values(
            comparison.group("left"),
            flow,
            index,
        )
        right = resolve_string_expression_values(
            comparison.group("right"),
            flow,
            index,
        )
        if left is not None and right is not None and len(left) == len(right) == 1:
            equal = _normalize_selector(next(iter(left))) == _normalize_selector(
                next(iter(right))
            )
            return (
                not equal
                if comparison.group("operator") == "!="
                else equal
            )
    return resolve_boolean_expression(value, flow, index)


def _resolve_condition_from_callers(
    condition: str,
    source: str,
    flow: SecurityValueFlow,
    index: int,
    seen_methods: frozenset[int],
) -> bool | None:
    method = method_containing_index(flow.methods, index)
    if method is None or not method.params:
        return None
    parameter_names = tuple(parameter.name for parameter in method.params)
    if not any(
        re.search(rf"\b{re.escape(name)}\b", condition)
        for name in parameter_names
    ):
        return None

    values: set[bool] = set()
    caller_evidence = False
    next_seen = seen_methods | frozenset({method.start})
    for call in find_method_calls(source, method.name):
        if any(
            candidate.start <= call.start <= candidate.body_start
            for candidate in flow.methods
        ):
            continue
        if method.body_start <= call.start <= method.body_end:
            continue
        if len(call.arguments) != len(parameter_names):
            continue
        if not position_is_reachable(
            source,
            flow,
            call.start,
            next_seen,
        ):
            continue
        caller_evidence = True
        substituted = condition
        for name, argument in zip(
            parameter_names,
            call.arguments,
            strict=True,
        ):
            substituted = re.sub(
                rf"\b{re.escape(name)}\b",
                f"({argument.text})",
                substituted,
            )
        resolved = _resolve_condition(substituted, flow, call.start)
        if resolved is None:
            return None
        values.add(resolved)
    if not caller_evidence or len(values) != 1:
        return None
    return next(iter(values))


def _resolve_lookup_index_condition(
    condition: str,
    flow: SecurityValueFlow,
    index: int,
) -> bool | None:
    comparison = re.fullmatch(
        r"(?:(?P<variable>[A-Za-z_]\w*)\s*==\s*(?P<number>\d+)|"
        r"(?P<number_first>\d+)\s*==\s*(?P<variable_second>[A-Za-z_]\w*))",
        condition.strip(),
    )
    if comparison is None:
        return None
    index_variable = (
        comparison.group("variable")
        or comparison.group("variable_second")
    )
    expected_index = int(
        comparison.group("number") or comparison.group("number_first")
    )
    prefix = flow.source[:index]
    declarations = list(
        re.finditer(
            r"String\s*\[\s*\]\s*(?P<array>[A-Za-z_]\w*)\s*=\s*"
            r"\{(?P<values>[^{}]*)\}\s*;",
            prefix,
            flags=re.DOTALL,
        )
    )
    for declaration in reversed(declarations):
        array_name = declaration.group("array")
        literals = re.findall(
            r'"(?:\\.|[^"\\])*"',
            declaration.group("values"),
        )
        if not literals:
            continue
        table = [java_string_literal_value(literal) for literal in literals]
        if any(item is None for item in table):
            continue
        search_pattern = re.compile(
            rf"if\s*\(\s*{re.escape(array_name)}\s*\[\s*"
            rf"(?P<iterator>[A-Za-z_]\w*)\s*\]\s*\.\s*"
            rf"equals(?:IgnoreCase)?\s*\(\s*"
            rf"(?P<selector>[A-Za-z_]\w*)\s*\)\s*\)"
            rf"(?P<body>.{{0,300}}?\b{re.escape(index_variable)}\s*=\s*"
            rf"(?P=iterator)\s*;)",
            flags=re.DOTALL,
        )
        search = search_pattern.search(prefix, declaration.end())
        if search is None:
            continue
        selected = resolve_string_expression_values(
            search.group("selector"),
            flow,
            search.start(),
        )
        if selected is None or len(selected) != 1:
            continue
        normalized_selected = _normalize_selector(next(iter(selected)))
        normalized_table = [
            _normalize_selector(str(item))
            for item in table
        ]
        if normalized_selected not in normalized_table:
            continue
        return normalized_table.index(normalized_selected) == expected_index
    return None


def _normalize_selector(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


__all__ = ("position_is_reachable",)
