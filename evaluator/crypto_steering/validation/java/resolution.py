"""Bounded Java value resolution independent of misuse policy.

The resolvers preserve literal and finite caller-provided values through the
small Java source shapes used by the experiment.  They intentionally return
``None`` when runtime or ambiguous values cannot be proved.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from itertools import product
import re

from crypto_steering.validation.java.calls import receiver_method_calls
from crypto_steering.validation.java.lightweight_ast import (
    find_method_calls,
    method_containing_index,
)
from crypto_steering.validation.java.security_value_flow import SecurityValueFlow
from crypto_steering.validation.java.source import CallSite
from crypto_steering.validation.java.static_analysis import (
    Assignment,
    Parameter,
    simple_variable,
    strip_outer_expression,
)


def resolve_string_expression(
    expression: str,
    flow: SecurityValueFlow,
    index: int,
    seen: frozenset[str] = frozenset(),
) -> str | None:
    expr = strip_outer_expression(expression)
    literal = java_string_literal_value(expr)
    if literal is not None:
        return literal
    transformed = _resolve_string_transform(expr, flow, index, seen)
    if transformed is not None:
        return transformed
    conditional = _split_top_level_conditional(expr)
    if conditional is not None:
        condition, true_branch, false_branch = conditional
        selected = resolve_boolean_expression(condition, flow, index, seen)
        if selected is not None:
            return resolve_string_expression(
                true_branch if selected else false_branch,
                flow,
                index,
                seen,
            )
        resolved_branches = {
            resolved
            for branch in (true_branch, false_branch)
            if (
                resolved := resolve_string_expression(
                    branch,
                    flow,
                    index,
                    seen,
                )
            )
            is not None
        }
        return (
            next(iter(resolved_branches))
            if len(resolved_branches) == 1
            else None
        )
    concatenated = _split_top_level_plus(expr)
    if len(concatenated) > 1:
        resolved_parts = []
        for part in concatenated:
            resolved = resolve_string_expression(part, flow, index, seen)
            if resolved is None:
                integer = resolve_int_expression(part, flow, index, seen)
                resolved = str(integer) if integer is not None else None
            resolved_parts.append(resolved)
        if all(part is not None for part in resolved_parts):
            return "".join(
                part for part in resolved_parts if part is not None
            )
        return None
    properties_lookup = _properties_get_property_expression(expr)
    if properties_lookup is not None:
        receiver, key_expression = properties_lookup
        return _resolve_local_properties_value(
            flow=flow,
            receiver=receiver,
            key_expression=key_expression,
            index=index,
            seen=seen,
        )
    structured_sources = flow.structured_member_sources(expr, index)
    if structured_sources:
        resolved = [
            resolve_string_expression(source, flow, source_index, seen)
            for source, source_index in structured_sources
        ]
        if all(item is not None for item in resolved):
            values = {item for item in resolved if item is not None}
            if len(values) == 1:
                return next(iter(values))
    call = _simple_call_signature(expr)
    if call is not None:
        helper_name, helper_arity = call
        helper_key = f"string-helper:{helper_name}/{helper_arity}"
        if helper_key in seen:
            return None
        resolved = [
            resolve_string_expression(
                returned,
                flow,
                index,
                seen | frozenset({helper_key}),
            )
            for returned in _helper_return_expressions(
                flow,
                helper_name,
                helper_arity,
            )
        ]
        resolved = [item for item in resolved if item is not None]
        if resolved and len(set(resolved)) == 1:
            return resolved[0]
    variable = simple_variable(expr)
    if variable is None:
        return None
    variable_key = _variable_seen_key(flow, variable, index, "string")
    if variable_key in seen:
        return None
    assignments = _reachable_assignments(flow, variable, index, seen)
    if assignments:
        resolved = {
            value
            for assignment in assignments
            if (
                value := resolve_string_expression(
                    assignment.expression,
                    flow,
                    assignment.start,
                    seen | frozenset({variable_key}),
                )
            )
            is not None
        }
        if len(resolved) == 1:
            return next(iter(resolved))
        return None
    caller_arguments = _caller_parameter_arguments(flow, variable, index)
    if caller_arguments is None:
        return None
    parameter_key = _parameter_seen_key(flow, variable, index, "string")
    if parameter_key in seen:
        return None
    resolved = [
        resolve_string_expression(
            caller_expression,
            flow,
            caller_index,
            seen | frozenset({parameter_key}),
        )
        for caller_expression, caller_index in caller_arguments
    ]
    if any(item is None for item in resolved):
        return None
    values = {item for item in resolved if item is not None}
    return next(iter(values)) if len(values) == 1 else None


def resolve_string_expression_values(
    expression: str,
    flow: SecurityValueFlow,
    index: int,
    seen: frozenset[str] = frozenset(),
) -> frozenset[str] | None:
    """Resolve every statically known string value reaching an expression."""

    expr = strip_outer_expression(expression)
    literal = java_string_literal_value(expr)
    if literal is not None:
        return frozenset({literal})

    indexed = re.fullmatch(
        r"(?P<owner>[A-Za-z_]\w*)\s*\[\s*(?P<index>[^\]]+)\s*\]",
        expr,
        flags=re.DOTALL,
    )
    if indexed is not None:
        values = _resolve_string_array_values(
            indexed.group("owner"),
            flow,
            index,
        )
        if values is None:
            return None
        selected = resolve_int_expression(
            indexed.group("index"),
            flow,
            index,
            seen,
        )
        if selected is None:
            return frozenset(values)
        if -len(values) <= selected < len(values):
            return frozenset({values[selected]})
        return None

    transformed = _resolve_string_transform_values(expr, flow, index, seen)
    if transformed is not None:
        return transformed

    conditional = _split_top_level_conditional(expr)
    if conditional is not None:
        condition, true_branch, false_branch = conditional
        selected = resolve_boolean_expression(condition, flow, index, seen)
        branches = (
            (true_branch if selected else false_branch,)
            if selected is not None
            else (true_branch, false_branch)
        )
        values: set[str] = set()
        for branch in branches:
            resolved = resolve_string_expression_values(
                branch,
                flow,
                index,
                seen,
            )
            if resolved is None:
                return None
            values.update(resolved)
        return frozenset(values) if values else None

    key_algorithms = _secret_key_get_algorithm_values(
        expr,
        flow,
        index,
        seen,
    )
    if key_algorithms is not None:
        return key_algorithms

    concatenated = _split_top_level_plus(expr)
    if len(concatenated) > 1:
        combined = {""}
        for part in concatenated:
            part_values = resolve_string_expression_values(
                part,
                flow,
                index,
                seen,
            )
            if part_values is None:
                integer = resolve_int_expression(part, flow, index, seen)
                if integer is not None:
                    part_values = frozenset({str(integer)})
            if part_values is None:
                return None
            combined = {
                prefix + suffix
                for prefix in combined
                for suffix in part_values
            }
        return frozenset(combined)

    properties_lookup = _properties_get_property_expression(expr)
    if properties_lookup is not None:
        receiver, key_expression = properties_lookup
        resolved = _resolve_local_properties_value(
            flow=flow,
            receiver=receiver,
            key_expression=key_expression,
            index=index,
            seen=seen,
        )
        return frozenset({resolved}) if resolved is not None else None

    map_lookup = _map_get_expression(expr)
    if map_lookup is not None:
        receiver, key_expression = map_lookup
        return _resolve_local_string_map_values(
            flow=flow,
            receiver=receiver,
            key_expression=key_expression,
            index=index,
            seen=seen,
        )

    structured_sources = flow.structured_member_sources(expr, index)
    if structured_sources:
        values: set[str] = set()
        for source, source_index in structured_sources:
            resolved = resolve_string_expression_values(
                source,
                flow,
                source_index,
                seen,
            )
            if resolved is None:
                return None
            values.update(resolved)
        return frozenset(values)

    call = _simple_call_signature(expr)
    if call is not None:
        helper_name, helper_arity = call
        helper_key = f"string-values-helper:{helper_name}/{helper_arity}"
        if helper_key in seen:
            return None
        returned = _helper_return_expressions(flow, helper_name, helper_arity)
        if returned:
            values: set[str] = set()
            for returned_expression in returned:
                resolved = resolve_string_expression_values(
                    returned_expression,
                    flow,
                    index,
                    seen | frozenset({helper_key}),
                )
                if resolved is None:
                    return None
                values.update(resolved)
            return frozenset(values) if values else None

    variable = simple_variable(expr)
    if variable is None:
        return None
    variable_key = _variable_seen_key(
        flow,
        variable,
        index,
        "string-values",
    )
    if variable_key in seen:
        return None
    assignments = _reachable_assignments(flow, variable, index, seen)
    if assignments:
        values: set[str] = set()
        for assignment in assignments:
            resolved = resolve_string_expression_values(
                assignment.expression,
                flow,
                assignment.start,
                seen | frozenset({variable_key}),
            )
            if resolved is None:
                return None
            values.update(resolved)
        return frozenset(values) if values else None

    caller_arguments = _caller_parameter_arguments(flow, variable, index)
    if caller_arguments is None:
        return None
    parameter_key = _parameter_seen_key(
        flow,
        variable,
        index,
        "string-values",
    )
    if parameter_key in seen:
        return None
    values: set[str] = set()
    for caller_expression, caller_index in caller_arguments:
        resolved = resolve_string_expression_values(
            caller_expression,
            flow,
            caller_index,
            seen | frozenset({parameter_key}),
        )
        if resolved is None:
            return None
        values.update(resolved)
    return frozenset(values) if values else None


def resolve_int_expression(
    expression: str,
    flow: SecurityValueFlow,
    index: int,
    seen: frozenset[str] = frozenset(),
) -> int | None:
    expr = strip_outer_expression(expression)
    value = java_int_literal_value(expr)
    if value is not None:
        return value
    index_of = re.fullmatch(
        r"(?:java\.util\.)?Arrays\s*\.\s*asList\s*\(\s*"
        r"(?P<owner>[A-Za-z_]\w*)\s*\)\s*\.\s*indexOf\s*\(\s*"
        r"(?P<needle>.+)\s*\)",
        expr,
        flags=re.DOTALL,
    )
    if index_of is not None:
        values = _resolve_string_array_values(
            index_of.group("owner"),
            flow,
            index,
        )
        needle = resolve_string_expression(
            index_of.group("needle"),
            flow,
            index,
            seen,
        )
        if values is not None and needle is not None:
            try:
                return values.index(needle)
            except ValueError:
                return -1
    call = _simple_call_signature(expr)
    if call is not None:
        helper_name, helper_arity = call
        helper_key = f"int-helper:{helper_name}/{helper_arity}"
        if helper_key in seen:
            return None
        resolved = [
            resolve_int_expression(
                returned,
                flow,
                index,
                seen | frozenset({helper_key}),
            )
            for returned in _helper_return_expressions(
                flow,
                helper_name,
                helper_arity,
            )
        ]
        resolved = [item for item in resolved if item is not None]
        if resolved and len(set(resolved)) == 1:
            return resolved[0]
    variable = simple_variable(expr)
    if variable is None:
        return None
    variable_key = _variable_seen_key(flow, variable, index, "int")
    if variable_key in seen:
        return None
    assignment = flow.latest_assignment(variable, index)
    if assignment is not None:
        return resolve_int_expression(
            assignment.expression,
            flow,
            assignment.start,
            seen | frozenset({variable_key}),
        )
    caller_arguments = _caller_parameter_arguments(flow, variable, index)
    if caller_arguments is None:
        return None
    parameter_key = _parameter_seen_key(flow, variable, index, "int")
    if parameter_key in seen:
        return None
    resolved = [
        resolve_int_expression(
            caller_expression,
            flow,
            caller_index,
            seen | frozenset({parameter_key}),
        )
        for caller_expression, caller_index in caller_arguments
    ]
    if any(item is None for item in resolved):
        return None
    values = {item for item in resolved if item is not None}
    return next(iter(values)) if len(values) == 1 else None


def resolve_boolean_expression(
    expression: str,
    flow: SecurityValueFlow,
    index: int,
    seen: frozenset[str] = frozenset(),
) -> bool | None:
    """Resolve bounded Java boolean values used as source selectors."""

    expr = strip_outer_expression(expression)
    if expr == "true":
        return True
    if expr == "false":
        return False
    if expr.startswith("!"):
        resolved = resolve_boolean_expression(expr[1:], flow, index, seen)
        return None if resolved is None else not resolved
    string_predicate = re.fullmatch(
        r"(?P<owner>.+)\.\s*(?P<method>startsWith|endsWith|equals|equalsIgnoreCase)"
        r"\s*\(\s*(?P<argument>.+)\s*\)",
        expr,
        flags=re.DOTALL,
    )
    if string_predicate is not None:
        owner = resolve_string_expression(
            string_predicate.group("owner"),
            flow,
            index,
            seen,
        )
        argument = resolve_string_expression(
            string_predicate.group("argument"),
            flow,
            index,
            seen,
        )
        if owner is not None and argument is not None:
            method = string_predicate.group("method")
            if method == "startsWith":
                return owner.startswith(argument)
            if method == "endsWith":
                return owner.endswith(argument)
            if method == "equalsIgnoreCase":
                return owner.casefold() == argument.casefold()
            return owner == argument
    call = _simple_call_signature(expr)
    if call is not None:
        helper_name, helper_arity = call
        helper_key = f"boolean-helper:{helper_name}/{helper_arity}"
        if helper_key in seen:
            return None
        resolved = [
            resolve_boolean_expression(
                returned,
                flow,
                index,
                seen | frozenset({helper_key}),
            )
            for returned in _helper_return_expressions(
                flow,
                helper_name,
                helper_arity,
            )
        ]
        resolved = [item for item in resolved if item is not None]
        if resolved and len(set(resolved)) == 1:
            return resolved[0]
    variable = simple_variable(expr)
    if variable is None:
        return None
    variable_key = _variable_seen_key(flow, variable, index, "boolean")
    if variable_key in seen:
        return None
    assignment = flow.latest_assignment(variable, index)
    if assignment is not None:
        return resolve_boolean_expression(
            assignment.expression,
            flow,
            assignment.start,
            seen | frozenset({variable_key}),
        )
    caller_arguments = _caller_parameter_arguments(flow, variable, index)
    if caller_arguments is None:
        return None
    parameter_key = _parameter_seen_key(flow, variable, index, "boolean")
    if parameter_key in seen:
        return None
    resolved = [
        resolve_boolean_expression(
            caller_expression,
            flow,
            caller_index,
            seen | frozenset({parameter_key}),
        )
        for caller_expression, caller_index in caller_arguments
    ]
    if any(item is None for item in resolved):
        return None
    values = {item for item in resolved if item is not None}
    return next(iter(values)) if len(values) == 1 else None


def reachable_parameter_string_values(
    expression: str,
    flow: SecurityValueFlow,
    index: int,
    position_is_reachable: Callable[[int], bool],
) -> frozenset[str] | None:
    """Resolve a parameter from every executable compatible caller."""

    variable = simple_variable(expression.strip())
    method = method_containing_index(flow.methods, index)
    if variable is None or method is None:
        return None
    parameter_index = next(
        (
            position
            for position, parameter in enumerate(method.params)
            if parameter.name == variable
        ),
        None,
    )
    if parameter_index is None:
        return None

    values: set[str] = set()
    for call in find_method_calls(flow.source, method.name):
        if any(
            candidate.start <= call.start <= candidate.body_start
            for candidate in flow.methods
        ):
            continue
        if method.body_start <= call.start <= method.body_end:
            continue
        if len(call.arguments) != len(method.params):
            continue
        if not position_is_reachable(call.start):
            continue
        resolved = resolve_string_expression_values(
            call.arguments[parameter_index].text,
            flow,
            call.start,
        )
        if resolved is None:
            return None
        values.update(resolved)
    return frozenset(values) if values else None


def java_string_literal_value(expression: str) -> str | None:
    expr = expression.strip()
    if not re.fullmatch(r'"(?:\\.|[^"\\])*"', expr, flags=re.DOTALL):
        return None
    try:
        return ast.literal_eval(expr)
    except Exception:
        return expr[1:-1]


def java_int_literal_value(expression: str) -> int | None:
    expr = expression.strip().replace("_", "")
    if expr.endswith(("L", "l")):
        expr = expr[:-1]
    try:
        if re.fullmatch(r"[-+]?0[xX][0-9A-Fa-f]+", expr):
            return int(expr, 16)
        if re.fullmatch(r"[-+]?0[bB][01]+", expr):
            return int(expr, 2)
        if re.fullmatch(r"[-+]?\d+", expr):
            return int(expr, 10)
    except ValueError:
        return None
    return None


def normalize_algorithm(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value).upper()


def split_transformation(
    transformation: str,
) -> tuple[str, str | None, str | None]:
    parts = [part.strip() for part in transformation.split("/")]
    algorithm = parts[0] if parts else transformation.strip()
    mode = parts[1] if len(parts) > 1 else None
    padding = parts[2] if len(parts) > 2 else None
    return algorithm, mode, padding


def _resolve_string_transform(
    expression: str,
    flow: SecurityValueFlow,
    index: int,
    seen: frozenset[str],
) -> str | None:
    values = _resolve_string_transform_values(expression, flow, index, seen)
    if values is None or len(values) != 1:
        return None
    return next(iter(values))


def _resolve_string_transform_values(
    expression: str,
    flow: SecurityValueFlow,
    index: int,
    seen: frozenset[str],
) -> frozenset[str] | None:
    expr = strip_outer_expression(expression)
    format_call = re.fullmatch(
        r"(?:java\s*\.\s*lang\s*\.\s*)?String\s*\.\s*format\s*"
        r"\((?P<arguments>.*)\)",
        expr,
        flags=re.DOTALL,
    )
    if format_call is not None:
        arguments = _split_top_level_csv(format_call.group("arguments"))
        if not arguments:
            return None
        formats = resolve_string_expression_values(
            arguments[0],
            flow,
            index,
            seen,
        )
        if formats is None:
            return None
        argument_values: list[frozenset[str]] = []
        for argument in arguments[1:]:
            values = resolve_string_expression_values(
                argument,
                flow,
                index,
                seen,
            )
            if values is None:
                integer = resolve_int_expression(argument, flow, index, seen)
                values = (
                    frozenset({str(integer)})
                    if integer is not None
                    else None
                )
            if values is None:
                return None
            argument_values.append(values)
        rendered: set[str] = set()
        combinations = product(*argument_values) if argument_values else [()]
        for values in combinations:
            for template in formats:
                value = _render_simple_java_format(template, values)
                if value is None:
                    return None
                rendered.add(value)
        return frozenset(rendered)

    case_call = re.fullmatch(
        r"(?P<owner>.+)\.\s*(?P<method>toUpperCase|toLowerCase)\s*\(\s*\)",
        expr,
        flags=re.DOTALL,
    )
    if case_call is not None:
        values = resolve_string_expression_values(
            case_call.group("owner"),
            flow,
            index,
            seen,
        )
        if values is None:
            return None
        transform = (
            str.upper
            if case_call.group("method") == "toUpperCase"
            else str.lower
        )
        return frozenset(transform(value) for value in values)

    constructor = re.fullmatch(
        r"new\s+(?:java\.lang\.)?String\s*\((?P<arguments>.*)\)",
        expr,
        flags=re.DOTALL,
    )
    if constructor is not None:
        arguments = _split_top_level_csv(constructor.group("arguments"))
        if arguments:
            return resolve_string_expression_values(
                arguments[0],
                flow,
                index,
                seen,
            )

    bytes_call = re.fullmatch(
        r"(?P<owner>.+)\.\s*getBytes\s*\([^)]*\)",
        expr,
        flags=re.DOTALL,
    )
    if bytes_call is not None:
        return resolve_string_expression_values(
            bytes_call.group("owner"),
            flow,
            index,
            seen,
        )

    trim_call = re.fullmatch(
        r"(?P<owner>.+)\.\s*(?:trim|strip)\s*\(\s*\)",
        expr,
        flags=re.DOTALL,
    )
    if trim_call is not None:
        values = resolve_string_expression_values(
            trim_call.group("owner"),
            flow,
            index,
            seen,
        )
        return (
            None
            if values is None
            else frozenset(value.strip() for value in values)
        )
    return None


def _render_simple_java_format(
    template: str,
    arguments: tuple[str, ...],
) -> str | None:
    """Render fixed ``%s``/``%d`` Java formats used as API selectors."""

    tokens = re.compile(r"%(?:(?P<position>\d+)\$)?(?P<kind>[sSdD%])")
    pieces: list[str] = []
    cursor = 0
    next_argument = 0
    for match in tokens.finditer(template):
        literal = template[cursor : match.start()]
        if "%" in literal:
            return None
        pieces.append(literal)
        kind = match.group("kind")
        if kind == "%":
            pieces.append("%")
        else:
            position = match.group("position")
            argument_index = int(position) - 1 if position else next_argument
            if not position:
                next_argument += 1
            if argument_index < 0 or argument_index >= len(arguments):
                return None
            value = arguments[argument_index]
            if kind in {"d", "D"} and re.fullmatch(r"[-+]?\d+", value) is None:
                return None
            pieces.append(value.upper() if kind == "S" else value)
        cursor = match.end()
    trailing = template[cursor:]
    if "%" in trailing:
        return None
    pieces.append(trailing)
    return "".join(pieces)


def _reachable_assignments(
    flow: SecurityValueFlow,
    variable: str,
    index: int,
    seen: frozenset[str],
) -> tuple[Assignment, ...]:
    latest = flow.latest_assignment(variable, index)
    if latest is None:
        return tuple()
    condition = _enclosing_if_condition(flow, latest.start)
    if condition is None:
        return (latest,)
    selected = resolve_boolean_expression(condition, flow, latest.start, seen)
    if selected is True:
        return (latest,)
    previous = _reachable_assignments(flow, variable, latest.start, seen)
    if selected is False:
        return previous
    return (latest, *previous)


def _enclosing_if_condition(
    flow: SecurityValueFlow,
    index: int,
) -> str | None:
    source = flow.source
    masked = flow.masked_source
    candidate: tuple[int, str] | None = None
    for match in re.finditer(r"\bif\s*\(", masked[:index]):
        open_paren = masked.find("(", match.start(), index)
        close_paren = _matching_delimiter_index(
            masked,
            open_paren,
            "(",
            ")",
        )
        if close_paren == -1:
            continue
        body_start = close_paren + 1
        while body_start < len(masked) and masked[body_start].isspace():
            body_start += 1
        if body_start >= len(masked) or masked[body_start] != "{":
            continue
        body_end = _matching_delimiter_index(
            masked,
            body_start,
            "{",
            "}",
        )
        if body_end == -1 or not (body_start < index < body_end):
            continue
        condition = source[open_paren + 1 : close_paren].strip()
        if condition and (candidate is None or body_start > candidate[0]):
            candidate = (body_start, condition)
    return candidate[1] if candidate is not None else None


def _matching_delimiter_index(
    text: str,
    open_index: int,
    opening: str,
    closing: str,
) -> int:
    if (
        open_index < 0
        or open_index >= len(text)
        or text[open_index] != opening
    ):
        return -1
    depth = 0
    for position in range(open_index, len(text)):
        char = text[position]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return position
    return -1


def _secret_key_get_algorithm_values(
    expression: str,
    flow: SecurityValueFlow,
    index: int,
    seen: frozenset[str],
) -> frozenset[str] | None:
    match = re.fullmatch(
        r"(?P<owner>[A-Za-z_]\w*)\s*\.\s*getAlgorithm\s*\(\s*\)",
        expression,
    )
    if match is None:
        return None
    sources = _secret_key_algorithm_sources(
        match.group("owner"),
        flow,
        index,
        seen,
    )
    if not sources:
        return None
    values: set[str] = set()
    for source, source_index in sources:
        resolved = resolve_string_expression_values(
            source,
            flow,
            source_index,
            seen,
        )
        if resolved is None:
            return None
        values.update(resolved)
    return frozenset(values) if values else None


def _secret_key_algorithm_sources(
    expression: str,
    flow: SecurityValueFlow,
    index: int,
    seen: frozenset[str],
) -> tuple[tuple[str, int], ...]:
    expr = strip_outer_expression(expression)
    key = f"secret-key-algorithm:{expr}:{index}"
    if key in seen:
        return tuple()
    next_seen = seen | frozenset({key})
    constructor = re.fullmatch(
        r"new\s+(?:[A-Za-z_]\w*\s*\.\s*)*SecretKeySpec\s*"
        r"\((?P<arguments>.*)\)",
        expr,
        flags=re.DOTALL,
    )
    if constructor is not None:
        arguments = _split_top_level_csv(constructor.group("arguments"))
        return ((arguments[1], index),) if len(arguments) > 1 else tuple()

    variable = simple_variable(expr)
    if variable is not None:
        assignment = flow.latest_assignment(variable, index)
        if assignment is not None:
            return _secret_key_algorithm_sources(
                assignment.expression,
                flow,
                assignment.start,
                next_seen,
            )
        caller_arguments = _caller_parameter_arguments(flow, variable, index)
        if caller_arguments is not None:
            return tuple(
                source
                for caller_expression, caller_index in caller_arguments
                for source in _secret_key_algorithm_sources(
                    caller_expression,
                    flow,
                    caller_index,
                    next_seen,
                )
            )

    call = _simple_call_signature(expr)
    if call is None:
        return tuple()
    helper_name, helper_arity = call
    return tuple(
        source
        for returned in _helper_return_expressions(
            flow,
            helper_name,
            helper_arity,
        )
        for source in _secret_key_algorithm_sources(
            returned,
            flow,
            index,
            next_seen,
        )
    )


def _resolve_string_array_values(
    variable: str,
    flow: SecurityValueFlow,
    index: int,
) -> tuple[str, ...] | None:
    assignment = flow.latest_assignment(variable, index)
    if assignment is None:
        return None
    expression = assignment.expression.strip()
    match = re.fullmatch(
        r"(?:new\s+(?:java\.lang\.)?String\s*\[\s*\]\s*)?"
        r"\{(?P<body>.*)\}",
        expression,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    values: list[str] = []
    for item in _split_top_level_csv(match.group("body")):
        value = java_string_literal_value(item)
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _properties_get_property_expression(
    expression: str,
) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"(?P<receiver>[A-Za-z_]\w*)\s*\.\s*getProperty\s*"
        r"\((?P<arguments>.*)\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    arguments = _split_top_level_csv(match.group("arguments"))
    if len(arguments) != 1:
        return None
    return match.group("receiver"), arguments[0]


def _resolve_local_properties_value(
    *,
    flow: SecurityValueFlow,
    receiver: str,
    key_expression: str,
    index: int,
    seen: frozenset[str],
) -> str | None:
    if _normalize_type(flow.variable_type(receiver, index) or "") != "Properties":
        return None
    method = method_containing_index(flow.methods, index)
    if method is None:
        return None
    lookup_key = (
        f"properties-lookup:{method.start}:{receiver}:"
        f"{key_expression.strip()}"
    )
    if lookup_key in seen:
        return None
    nested_seen = seen | frozenset({lookup_key})
    key = resolve_string_expression(
        key_expression,
        flow,
        index,
        nested_seen,
    )
    if key is None:
        return None

    opaque_mutators = (
        "clear",
        "compute",
        "computeIfAbsent",
        "computeIfPresent",
        "load",
        "loadFromXML",
        "merge",
        "putAll",
        "remove",
        "replace",
        "replaceAll",
    )
    for method_name in opaque_mutators:
        if any(
            call_receiver == receiver
            and call.start < index
            and method_containing_index(flow.methods, call.start) == method
            for call_receiver, call in receiver_method_calls(
                flow.source,
                method_name,
            )
        ):
            return None

    values: set[str] = set()
    for method_name in ("put", "setProperty"):
        for call_receiver, call in receiver_method_calls(
            flow.source,
            method_name,
        ):
            if (
                call_receiver != receiver
                or call.start >= index
                or len(call.arguments) < 2
            ):
                continue
            if method_containing_index(flow.methods, call.start) != method:
                continue
            write_key = resolve_string_expression(
                call.arguments[0].text,
                flow,
                call.start,
                nested_seen,
            )
            if write_key != key:
                continue
            write_value = resolve_string_expression(
                call.arguments[1].text,
                flow,
                call.start,
                nested_seen,
            )
            if write_value is None:
                return None
            values.add(write_value)
    return next(iter(values)) if len(values) == 1 else None


def _map_get_expression(
    expression: str,
) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"(?P<receiver>[A-Za-z_]\w*)\s*\.\s*get\s*"
        r"\((?P<arguments>.*)\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    arguments = _split_top_level_csv(match.group("arguments"))
    if len(arguments) != 1:
        return None
    return match.group("receiver"), arguments[0]


def _resolve_local_string_map_values(
    *,
    flow: SecurityValueFlow,
    receiver: str,
    key_expression: str,
    index: int,
    seen: frozenset[str],
) -> frozenset[str] | None:
    """Resolve values written to a source-local ``Map<String, String>``."""

    declaration = re.search(
        rf"\b(?:Map|HashMap|LinkedHashMap|TreeMap)\s*"
        rf"<\s*String\s*,\s*String\s*>\s+{re.escape(receiver)}\b",
        flow.masked_source[:index],
    )
    if declaration is None:
        return None
    method = method_containing_index(flow.methods, index)
    if method is None:
        return None
    lookup_key = f"map-lookup:{method.start}:{receiver}:{key_expression.strip()}"
    if lookup_key in seen:
        return None
    nested_seen = seen | frozenset({lookup_key})
    selected_keys = resolve_string_expression_values(
        key_expression,
        flow,
        index,
        nested_seen,
    )

    for method_name in (
        "clear",
        "compute",
        "computeIfAbsent",
        "computeIfPresent",
        "merge",
        "putAll",
        "remove",
        "replace",
        "replaceAll",
    ):
        if any(
            call_receiver == receiver
            and call.start < index
            and method_containing_index(flow.methods, call.start) == method
            for call_receiver, call in receiver_method_calls(
                flow.source,
                method_name,
            )
        ):
            return None

    values: set[str] = set()
    observed_write = False
    for call_receiver, call in receiver_method_calls(flow.source, "put"):
        if (
            call_receiver != receiver
            or call.start >= index
            or len(call.arguments) < 2
            or method_containing_index(flow.methods, call.start) != method
        ):
            continue
        observed_write = True
        write_keys = resolve_string_expression_values(
            call.arguments[0].text,
            flow,
            call.start,
            nested_seen,
        )
        if selected_keys is not None:
            if write_keys is None:
                return None
            if selected_keys.isdisjoint(write_keys):
                continue
        write_values = resolve_string_expression_values(
            call.arguments[1].text,
            flow,
            call.start,
            nested_seen,
        )
        if write_values is None:
            return None
        values.update(write_values)
    if not observed_write:
        return None
    return frozenset(values) if values else None


def _simple_call_signature(expression: str) -> tuple[str, int] | None:
    match = re.fullmatch(
        r"([A-Za-z_]\w*)\s*\((.*)\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    arguments_text = match.group(2).strip()
    if not arguments_text:
        return match.group(1), 0
    return match.group(1), len(_split_top_level_csv(arguments_text))


def _helper_return_expressions(
    flow: SecurityValueFlow,
    helper_name: str,
    helper_arity: int,
) -> tuple[str, ...]:
    expressions: list[str] = []
    for method in flow.methods:
        if method.name != helper_name or len(method.params) != helper_arity:
            continue
        body = flow.source[method.body_start + 1 : method.body_end]
        expressions.extend(
            item.strip()
            for item in re.findall(
                r"\breturn\s+([^;]+);",
                body,
                flags=re.DOTALL,
            )
        )
    return tuple(expressions)


def _caller_parameter_arguments(
    flow: SecurityValueFlow,
    variable: str,
    index: int,
) -> tuple[tuple[str, int], ...] | None:
    method = method_containing_index(flow.methods, index)
    if method is None:
        return None
    parameter_index = next(
        (
            position
            for position, parameter in enumerate(method.params)
            if parameter.name == variable
        ),
        None,
    )
    if parameter_index is None:
        return None

    arguments: list[tuple[str, int]] = []
    for call in find_method_calls(flow.source, method.name):
        if any(
            candidate.start <= call.start <= candidate.body_start
            for candidate in flow.methods
        ):
            continue
        if method.body_start <= call.start <= method.body_end:
            continue
        if len(call.arguments) != len(method.params):
            continue
        if not _call_arguments_may_target_method(call, method.params, flow):
            continue
        if not flow.call_has_entry_path(call):
            continue
        arguments.append((call.arguments[parameter_index].text, call.start))
    return tuple(arguments) if arguments else None


def _call_arguments_may_target_method(
    call: CallSite,
    parameters: tuple[Parameter, ...],
    flow: SecurityValueFlow,
) -> bool:
    for argument, parameter in zip(call.arguments, parameters, strict=True):
        argument_type = _expression_type(argument.text, flow, call.start)
        if (
            argument_type is not None
            and not _types_may_match(argument_type, parameter.type_name)
        ):
            return False
    return True


def _expression_type(
    expression: str,
    flow: SecurityValueFlow,
    index: int,
) -> str | None:
    expr = strip_outer_expression(expression)
    if java_string_literal_value(expr) is not None:
        return "String"
    if java_int_literal_value(expr) is not None:
        return "long" if expr.rstrip().endswith(("L", "l")) else "int"
    if len(_split_top_level_plus(expr)) > 1 and '"' in expr:
        return "String"
    for element_type in ("byte", "char", "int", "long", "String"):
        if re.fullmatch(
            rf"new\s+{element_type}\s*\[[^\]]*\]",
            expr,
            flags=re.DOTALL,
        ):
            return f"{element_type}[]"
    if re.fullmatch(
        r".+\.\s*toCharArray\s*\(\s*\)",
        expr,
        flags=re.DOTALL,
    ):
        return "char[]"
    if re.fullmatch(
        r".+\.\s*getBytes\s*\([^)]*\)",
        expr,
        flags=re.DOTALL,
    ):
        return "byte[]"
    variable = simple_variable(expr)
    return flow.variable_type(variable, index) if variable is not None else None


def _types_may_match(argument_type: str, parameter_type: str) -> bool:
    argument = _normalize_type(argument_type)
    parameter = _normalize_type(parameter_type)
    if not argument or not parameter:
        return True
    if argument == parameter:
        return True
    if argument == "int" and parameter in {"long", "Integer", "Number"}:
        return True
    return parameter in {"Object", "Serializable", "CharSequence"}


def _normalize_type(type_name: str) -> str:
    normalized = re.sub(
        r"\s+",
        "",
        type_name.replace("...", "[]").strip(),
    )
    array_suffix = "[]" if normalized.endswith("[]") else ""
    base = normalized[:-2] if array_suffix else normalized
    return f"{base.rsplit('.', 1)[-1]}{array_suffix}"


def _parameter_seen_key(
    flow: SecurityValueFlow,
    variable: str,
    index: int,
    value_type: str,
) -> str:
    method = method_containing_index(flow.methods, index)
    method_start = method.start if method is not None else -1
    return f"{value_type}-parameter:{method_start}:{variable}"


def _variable_seen_key(
    flow: SecurityValueFlow,
    variable: str,
    index: int,
    value_type: str,
) -> str:
    method = method_containing_index(flow.methods, index)
    method_start = method.start if method is not None else -1
    return f"{value_type}-variable:{method_start}:{variable}"


def _split_top_level_plus(text: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    for index, char in enumerate(text):
        if in_string or in_char:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif in_string and char == '"':
                in_string = False
            elif in_char and char == "'":
                in_char = False
            continue
        if char == '"':
            in_string = True
        elif char == "'":
            in_char = True
        elif char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "+" and depth == 0:
            part = text[start:index].strip()
            if not part:
                return (text,)
            parts.append(part)
            start = index + 1
    if not parts:
        return (text,)
    final = text[start:].strip()
    if not final:
        return (text,)
    parts.append(final)
    return tuple(parts)


def _split_top_level_conditional(
    text: str,
) -> tuple[str, str, str] | None:
    question = _top_level_operator_index(text, "?")
    if question == -1:
        return None
    colon = _top_level_operator_index(text, ":", start=question + 1)
    if colon == -1:
        return None
    condition = text[:question].strip()
    true_branch = text[question + 1 : colon].strip()
    false_branch = text[colon + 1 :].strip()
    if not condition or not true_branch or not false_branch:
        return None
    return condition, true_branch, false_branch


def _top_level_operator_index(
    text: str,
    operator: str,
    *,
    start: int = 0,
) -> int:
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    nested_conditionals = 0
    for index, char in enumerate(text):
        if index < start:
            continue
        if in_string or in_char:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif in_string and char == '"':
                in_string = False
            elif in_char and char == "'":
                in_char = False
            continue
        if char == '"':
            in_string = True
        elif char == "'":
            in_char = True
        elif char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif depth == 0 and operator == "?" and char == "?":
            return index
        elif depth == 0 and operator == ":":
            if char == "?":
                nested_conditionals += 1
            elif char == ":":
                if nested_conditionals:
                    nested_conditionals -= 1
                else:
                    return index
    return -1


def _split_top_level_csv(text: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    for index, char in enumerate(text):
        if in_string or in_char:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif in_string and char == '"':
                in_string = False
            elif in_char and char == "'":
                in_char = False
            continue
        if char == '"':
            in_string = True
        elif char == "'":
            in_char = True
        elif char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            part = text[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    final = text[start:].strip()
    if final:
        parts.append(final)
    return tuple(parts)


__all__ = (
    "java_int_literal_value",
    "java_string_literal_value",
    "normalize_algorithm",
    "reachable_parameter_string_values",
    "resolve_boolean_expression",
    "resolve_int_expression",
    "resolve_string_expression",
    "resolve_string_expression_values",
    "split_transformation",
)
