"""Java random-origin evidence shared by M05 and M06."""

from __future__ import annotations

import re

from crypto_steering.validation.java.reachability import position_is_reachable
from crypto_steering.validation.java.security_value_flow import SecurityValueFlow
from crypto_steering.validation.java.static_analysis import (
    Assignment,
    strip_outer_expression,
)


def random_origin_evidence(
    flow: SecurityValueFlow,
    expression: str,
    index: int,
    seen: frozenset[tuple[str, int]] = frozenset(),
) -> str:
    """Collect bounded source evidence that controls a generated value."""

    normalized = expression.strip()
    key = (normalized, index)
    if key in seen:
        return normalized
    next_seen = seen | frozenset({key})
    evidence = normalized
    receiver_calls = tuple(
        re.finditer(
            r"\b(?P<receiver>[A-Za-z_]\w*)\s*\.\s*"
            r"(?:generateSeed|nextBytes|nextInt|nextLong)\s*\(",
            normalized,
        )
    ) + tuple(
        re.finditer(
            r"(?:\(\s*)?\(\s*[A-Za-z_][\w$.<>?]*\s*\)\s*"
            r"(?P<receiver>[A-Za-z_]\w*)\s*\)?\s*\.\s*"
            r"(?:generateSeed|nextBytes|nextInt|nextLong)\s*\(",
            normalized,
        )
    )
    for receiver_call in receiver_calls:
        receiver = receiver_call.group("receiver")
        assignment = _latest_reachable_assignment(flow, receiver, index)
        if assignment is not None:
            evidence += "\n" + random_origin_evidence(
                flow,
                flow._assignment_expression(assignment),
                assignment.start,
                next_seen,
            )
    conditional = conditional_value_branches(normalized)
    if conditional is not None:
        for branch in conditional:
            evidence += "\n" + random_origin_evidence(
                flow,
                branch,
                index,
                next_seen,
            )
    if re.fullmatch(
        r"[A-Za-z_]\w*\s*\.\s*[A-Za-z_]\w*(?:\s*\(\s*\))?",
        normalized,
    ):
        for value, value_index in flow.structured_member_sources(
            normalized,
            index,
        ):
            evidence += "\n" + random_origin_evidence(
                flow,
                value,
                value_index,
                next_seen,
            )
    variable = re.fullmatch(r"[A-Za-z_]\w*", expression.strip())
    if variable is not None:
        variable_name = variable.group(0)
        escaped_name = re.escape(variable_name)
        prefix = flow.source[:index]
        mutations = tuple(
            re.finditer(
                rf"(?:\bnew\s+(?:java\.(?:security|util)\.)?"
                rf"(?:SecureRandom|Random)\s*\([^;\n]*\)|"
                rf"\b[A-Za-z_]\w*)\s*\.\s*nextBytes\s*\(\s*"
                rf"{escaped_name}\s*\)",
                prefix,
            )
        )
        if mutations:
            evidence += "\n" + mutations[-1].group(0)
        assignment = _latest_reachable_assignment(flow, variable_name, index)
        if assignment is not None:
            evidence += "\n" + random_origin_evidence(
                flow,
                flow._assignment_expression(assignment),
                assignment.start,
                next_seen,
            )
        containing = next(
            (
                method
                for method in flow.methods
                if method.body_start < index < method.body_end
                and any(
                    parameter.name == variable_name
                    for parameter in method.params
                )
            ),
            None,
        )
        if containing is not None:
            parameter_names = [
                parameter.name for parameter in containing.params
            ]
            parameter_index = parameter_names.index(variable_name)
            for call in flow._method_calls(containing.name):
                if not flow._call_may_target_declared_method(call, containing):
                    continue
                if any(
                    method.start <= call.start <= method.body_start
                    for method in flow.methods
                ):
                    continue
                if containing.body_start <= call.start <= containing.body_end:
                    continue
                if len(call.arguments) != len(parameter_names):
                    continue
                argument = call.arguments[parameter_index]
                evidence += "\n" + random_origin_evidence(
                    flow,
                    argument.text,
                    argument.start,
                    next_seen,
                )
    evidence += _helper_body_evidence(flow, expression)
    helper = re.match(r"\s*(?P<name>[A-Za-z_]\w*)\s*\(", expression)
    if helper is not None:
        for call in flow._method_calls(helper.group("name")):
            call_text = flow.source[call.start : call.end].strip()
            if call_text != normalized:
                continue
            for argument in call.arguments:
                evidence += "\n" + random_origin_evidence(
                    flow,
                    argument.text,
                    argument.start,
                    next_seen,
                )
            break
    return evidence


def has_explicit_random_origin(
    flow: SecurityValueFlow,
    expression: str,
    index: int,
) -> bool:
    evidence = random_origin_evidence(flow, expression, index)
    return bool(
        re.search(
            r"(?:\bnew\s+(?:java\.(?:security|util)\.)?"
            r"(?:SecureRandom|Random)\s*\(|"
            r"\b(?:java\.security\.)?SecureRandom\s*\.\s*getInstance|"
            r"\bMath\s*\.\s*random\s*\(|"
            r"\bSystem\s*\.\s*(?:currentTimeMillis|nanoTime)\s*\()",
            evidence,
        )
    )


def _helper_body_evidence(
    flow: SecurityValueFlow,
    expression: str,
) -> str:
    helper = re.match(r"\s*(?P<name>[A-Za-z_]\w*)\s*\(", expression)
    if helper is None:
        return ""
    by_name = {method.name: method for method in flow.methods}
    pending = [helper.group("name")]
    seen: set[str] = set()
    evidence = ""
    while pending:
        name = pending.pop()
        if name in seen or name not in by_name:
            continue
        seen.add(name)
        method = by_name[name]
        body = flow.source[method.body_start + 1 : method.body_end]
        evidence += "\n" + body
        for call_name in re.findall(r"\b([A-Za-z_]\w*)\s*\(", body):
            if call_name in by_name and call_name not in seen:
                pending.append(call_name)
    return evidence


def conditional_value_branches(expression: str) -> tuple[str, str] | None:
    """Return the two value branches of a top-level Java conditional."""

    value = strip_outer_expression(expression)
    question = _top_level_marker(value, "?")
    if question == -1:
        return None
    colon = _top_level_marker(value, ":", start=question + 1)
    if colon == -1:
        return None
    true_branch = value[question + 1 : colon].strip()
    false_branch = value[colon + 1 :].strip()
    if not true_branch or not false_branch:
        return None
    return true_branch, false_branch


def _top_level_marker(text: str, marker: str, *, start: int = 0) -> int:
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    for index, character in enumerate(text):
        if index < start:
            continue
        if in_string or in_char:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif in_string and character == '"':
                in_string = False
            elif in_char and character == "'":
                in_char = False
            continue
        if character == '"':
            in_string = True
        elif character == "'":
            in_char = True
        elif character in "([{":
            depth += 1
        elif character in ")]}" and depth:
            depth -= 1
        elif character == marker and depth == 0:
            return index
    return -1


def _latest_reachable_assignment(
    flow: SecurityValueFlow,
    variable: str,
    index: int,
) -> Assignment | None:
    assignments = [
        assignment
        for assignment in flow.assignments
        if assignment.name == variable
        and flow._assignment_is_code(assignment)
        and flow._assignment_visible_at(assignment, index)
        and position_is_reachable(flow.source, flow, assignment.start)
    ]
    if assignments:
        return max(assignments, key=lambda assignment: assignment.start)
    assignment = flow._latest_assignment(variable, index)
    if assignment is None:
        return None
    if position_is_reachable(flow.source, flow, assignment.start):
        return assignment
    return None


__all__ = (
    "conditional_value_branches",
    "has_explicit_random_origin",
    "random_origin_evidence",
)
