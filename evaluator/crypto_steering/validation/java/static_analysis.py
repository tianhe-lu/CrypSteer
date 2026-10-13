"""Lightweight Java static analysis helpers for validation and detectors."""

from __future__ import annotations

from dataclasses import dataclass
import re

from crypto_steering.validation.java.source import (
    find_matching_delimiter,
    iter_methods,
    mask_java_comments_and_literals,
    parse_parameter_names,
    strip_java_comments,
)


@dataclass(frozen=True)
class Assignment:
    name: str
    expression: str
    start: int
    end: int
    type_name: str = ""
    modifiers: str = ""


@dataclass(frozen=True)
class Parameter:
    name: str
    type_name: str


@dataclass(frozen=True)
class MethodReturn:
    name: str
    return_type: str
    expressions: tuple[str, ...]
    start: int
    body_start: int
    body_end: int
    params: tuple[Parameter, ...] = ()


def strip_imports(source: str) -> str:
    source = strip_java_comments(source)
    return "\n".join(line for line in source.splitlines() if not line.strip().startswith("import "))


def actual_symbol_usage(source: str, symbol: str) -> bool:
    code = strip_imports(source)
    if "." in symbol:
        return symbol in code
    return bool(re.search(rf"\b{re.escape(symbol)}\b", code))


def extract_assignments(source: str) -> tuple[Assignment, ...]:
    source = strip_java_comments(source)
    masked = mask_java_comments_and_literals(source)
    assignments: list[Assignment] = []
    declaration = re.compile(
        r"(?<!\()\b"
        r"(?P<mods>(?:(?:public|private|protected|static|final|transient|volatile)\s+)*)"
        r"(?P<type>[A-Za-z_][\w$.<>?]*(?:\s*\[\])?)\s+"
        r"(?P<name>[A-Za-z_]\w*)"
        r"(?P<post_array>\s*\[\s*\])?\s*=(?!=)\s*?(?P<expr>[^;]+);",
        flags=re.DOTALL,
    )
    for match in declaration.finditer(masked):
        if _looks_like_try_resource_declaration(source, match.start()):
            continue
        raw_type = match.group("type")
        if match.group("post_array"):
            raw_type = f"{raw_type}[]"
        assignments.append(
            Assignment(
                name=match.group("name"),
                expression=source[match.start("expr") : match.end("expr")].strip(),
                start=match.start(),
                end=match.end(),
                type_name=_normalize_type(raw_type),
                modifiers=match.group("mods").strip(),
            )
        )

    # Lambda bodies can contain normal local declarations. The broad
    # declaration regex above may consume the beginning of a lambda assignment
    # through the first semicolon inside the lambda body, preventing overlapping
    # local declarations such as ``char[] password = helper();`` from being
    # emitted. Run a narrower non-overlap pass to recover those declarations.
    nested_declaration = re.compile(
        r"(?<!\()\b"
        r"(?P<type>[A-Za-z_][\w$.<>?]*(?:\s*\[\])?)\s+"
        r"(?P<name>[A-Za-z_]\w*)"
        r"(?P<post_array>\s*\[\s*\])?\s*=(?!=)\s*?(?P<expr>[^;{}]+);",
        flags=re.DOTALL,
    )
    existing_starts = {assignment.start for assignment in assignments}
    existing_spans = tuple((assignment.start, assignment.end) for assignment in assignments)
    for match in nested_declaration.finditer(masked):
        if match.start() in existing_starts:
            continue
        if any(
            start <= match.start() < end and "->" not in source[start : match.start()]
            for start, end in existing_spans
        ):
            continue
        if _looks_like_try_resource_declaration(source, match.start()):
            continue
        raw_type = match.group("type")
        if match.group("post_array"):
            raw_type = f"{raw_type}[]"
        assignments.append(
            Assignment(
                name=match.group("name"),
                expression=source[match.start("expr") : match.end("expr")].strip(),
                start=match.start(),
                end=match.end(),
                type_name=_normalize_type(raw_type),
            )
        )

    plain = re.compile(r"(?<![=!<>.])\b(?P<name>[A-Za-z_]\w*)\s*=(?!=)\s*?(?P<expr>[^;]+);", flags=re.DOTALL)
    declaration_spans = {(assignment.start, assignment.end) for assignment in assignments}
    for match in plain.finditer(masked):
        if _looks_like_try_resource_declaration(source, match.start()):
            continue
        expression = source[match.start("expr") : match.end("expr")].strip()
        if expression.lstrip().startswith("="):
            continue
        if any(start <= match.start() < end for start, end in declaration_spans):
            continue
        assignments.append(
            Assignment(
                name=match.group("name"),
                expression=expression,
                start=match.start(),
                end=match.end(),
            )
        )

    field_write = re.compile(
        r"\b(?:this|[A-Za-z_]\w*)\s*\.\s*(?P<name>[A-Za-z_]\w*)\s*=(?!=)\s*?(?P<expr>[^;]+);",
        flags=re.DOTALL,
    )
    for match in field_write.finditer(masked):
        assignments.append(
            Assignment(
                name=match.group("name"),
                expression=source[match.start("expr") : match.end("expr")].strip(),
                start=match.start(),
                end=match.end(),
            )
        )
    return tuple(sorted(assignments, key=lambda assignment: assignment.start))


def _looks_like_try_resource_declaration(source: str, start: int) -> bool:
    prefix = source[max(0, start - 120) : start]
    return bool(re.search(r"\btry\s*\([^;{}]*$", prefix, flags=re.DOTALL))


def latest_assignment(source: str, variable: str, before_index: int) -> Assignment | None:
    methods = tuple(extract_method_returns(source).values())
    matches = [
        assignment
        for assignment in extract_assignments(source)
        if assignment.name == variable
        and _assignment_visible_at(source, methods, assignment, before_index)
    ]
    if not matches:
        return None
    return max(matches, key=lambda assignment: assignment.start)


def extract_method_returns(source: str) -> dict[str, MethodReturn]:
    source = strip_java_comments(source)
    pattern = re.compile(
        r"(?:^|[;\{\}\n])\s*"
        r"(?P<mods>(?:(?:public|private|protected|static|final|synchronized|native)\s+)*)"
        r"(?P<return>[\w.$<>\[\], ?]+)\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*"
        r"\((?P<params>[^)]*)\)\s*(?:throws\s+[\w.$<>\[\], ?]+\s*)?\{",
        flags=re.MULTILINE,
    )
    methods: dict[str, MethodReturn] = {}
    for match in pattern.finditer(source):
        body_start = source.rfind("{", match.start(), match.end())
        if body_start == -1:
            continue
        body_end = find_matching_delimiter(source, body_start, "{", "}")
        if body_end == -1:
            continue
        body = source[body_start + 1 : body_end]
        returns = tuple(expr.strip() for expr in re.findall(r"\breturn\s+([^;]+);", body, flags=re.DOTALL))
        methods[match.group("name")] = MethodReturn(
            name=match.group("name"),
            return_type=_normalize_type(match.group("return")),
            expressions=returns,
            start=match.start(),
            body_start=body_start,
            body_end=body_end,
            params=parse_parameter_declarations(match.group("params")),
        )
    for match in _constructor_pattern(source).finditer(source):
        body_start = source.rfind("{", match.start(), match.end())
        if body_start == -1:
            continue
        body_end = find_matching_delimiter(source, body_start, "{", "}")
        if body_end == -1:
            continue
        methods[match.group("name")] = MethodReturn(
            name=match.group("name"),
            return_type="",
            expressions=tuple(),
            start=match.start(),
            body_start=body_start,
            body_end=body_end,
            params=parse_parameter_declarations(match.group("params")),
        )
    return methods


def parameters_in_scope(source: str, index: int) -> tuple[Parameter, ...]:
    source = strip_java_comments(source)
    method_pattern = re.compile(
        r"(?:^|[;\{\}\n])\s*"
        r"(?:(?:public|private|protected|static|final|synchronized|native)\s+)*"
        r"[\w.$<>\[\], ?]+\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*"
        r"\((?P<params>[^)]*)\)\s*(?:throws\s+[\w.$<>\[\], ?]+\s*)?\{",
        flags=re.MULTILINE,
    )
    for match in method_pattern.finditer(source):
        body_start = source.rfind("{", match.start(), match.end())
        if body_start == -1:
            continue
        body_end = find_matching_delimiter(source, body_start, "{", "}")
        if body_start <= index <= body_end:
            return parse_parameter_declarations(match.group("params"))
    for match in _constructor_pattern(source).finditer(source):
        body_start = source.rfind("{", match.start(), match.end())
        if body_start == -1:
            continue
        body_end = find_matching_delimiter(source, body_start, "{", "}")
        if body_start <= index <= body_end:
            return parse_parameter_declarations(match.group("params"))
    return tuple()


def parse_parameter_declarations(params_text: str) -> tuple[Parameter, ...]:
    names = parse_parameter_names(params_text)
    if not names:
        return tuple()
    params: list[Parameter] = []
    raw_params = _split_top_level_commas(params_text)
    for raw, name in zip(raw_params, names):
        cleaned = re.sub(r"@\w+(?:\([^)]*\))?\s*", "", raw)
        cleaned = cleaned.replace("final ", "").replace("...", "[]").strip()
        type_text = cleaned[: cleaned.rfind(name)].strip()
        after_name = cleaned[cleaned.rfind(name) + len(name) :].strip()
        if after_name.startswith("[]"):
            type_text = f"{type_text}[]"
        params.append(Parameter(name=name, type_name=_normalize_type(type_text)))
    return tuple(params)


def variable_type_in_scope(source: str, variable: str, index: int) -> str | None:
    assignment = latest_assignment(source, variable, index)
    if assignment and assignment.type_name:
        return assignment.type_name
    for param in parameters_in_scope(source, index):
        if param.name == variable:
            return param.type_name
    return None


def is_char_array_expression(source: str, expression: str, index: int, seen: frozenset[str] = frozenset()) -> bool:
    expr = strip_outer_expression(expression)
    if expr == "null":
        return True
    if re.fullmatch(r"\(char\s*\[\]\)\s*.+", expr, flags=re.DOTALL):
        return True
    if re.search(r"\.\s*toCharArray\s*\(\s*\)$", expr):
        return True
    if re.search(r"\.\s*readPassword\s*\(", expr):
        return True
    if re.search(r"new\s+char\s*(?:\[\s*\]|\[\s*\d+\s*\])", expr):
        return True

    variable = simple_variable(expr)
    if variable:
        if variable in seen:
            return False
        if variable_type_in_scope(source, variable, index) == "char[]":
            return True
        assignment = latest_assignment(source, variable, index)
        if assignment:
            return is_char_array_expression(source, assignment.expression, assignment.start, seen | frozenset({variable}))

    helper = simple_call_name(expr)
    if helper and helper not in seen:
        method = extract_method_returns(source).get(helper)
        if method and method.return_type == "char[]":
            return True
        if method and method.expressions:
            return all(
                is_char_array_expression(source, returned, method.body_end, seen | frozenset({helper}))
                for returned in method.expressions
            )

    return False


def simple_variable(expression: str) -> str | None:
    match = re.fullmatch(r"(?:this\s*\.\s*)?([A-Za-z_]\w*)", expression.strip())
    return match.group(1) if match else None


def simple_call_name(expression: str) -> str | None:
    match = re.fullmatch(r"([A-Za-z_]\w*)\s*\([^;]*\)", expression.strip(), flags=re.DOTALL)
    return match.group(1) if match else None


def strip_outer_expression(expression: str) -> str:
    expr = expression.strip()
    changed = True
    while changed:
        changed = False
        cast = re.fullmatch(r"\(\s*[\w.$<>\[\]]+\s*\)\s*(?!\[)(.+)", expr, flags=re.DOTALL)
        if cast:
            expr = cast.group(1).strip()
            changed = True
        if expr.startswith("(") and expr.endswith(")"):
            close = find_matching_delimiter(expr, 0, "(", ")")
            if close != len(expr) - 1:
                continue
            inner = expr[1:-1].strip()
            if inner:
                expr = inner
                changed = True
    return expr


def expression_mentions_string_literal(expression: str) -> bool:
    return bool(re.search(r'"(?:\\.|[^"\\])*"', expression, flags=re.DOTALL))


def _normalize_type(type_name: str) -> str:
    return re.sub(r"\s+", "", type_name.replace("...", "[]").strip())


def _constructor_pattern(source: str) -> re.Pattern[str]:
    class_names = tuple(dict.fromkeys(re.findall(r"\bclass\s+([A-Za-z_]\w*)\b", source)))
    if not class_names:
        return re.compile(r"a^")
    names = "|".join(re.escape(name) for name in class_names)
    return re.compile(
        rf"(?:^|[;\{{\}}\n])\s*"
        rf"(?:(?:public|private|protected)\s+)?"
        rf"(?P<name>{names})\s*"
        rf"\((?P<params>[^)]*)\)\s*(?:throws\s+[\w.$<>\[\], ?]+\s*)?\{{",
        flags=re.MULTILINE,
    )


def _assignment_visible_at(
    source: str,
    methods: tuple[MethodReturn, ...],
    assignment: Assignment,
    before_index: int,
) -> bool:
    if assignment.start >= before_index:
        return False

    current_method = _method_containing_index(methods, before_index)
    assignment_method = _method_containing_index(methods, assignment.start)
    if current_method is None:
        return assignment_method is None
    if assignment_method == current_method:
        return True
    if assignment_method is None:
        return True
    return _is_this_field_write(source, assignment)


def _method_containing_index(methods: tuple[MethodReturn, ...], index: int) -> MethodReturn | None:
    for method in methods:
        if method.body_start <= index <= method.body_end:
            return method
    return None


def _is_this_field_write(source: str, assignment: Assignment) -> bool:
    snippet = source[assignment.start : assignment.end]
    return bool(re.match(rf"\s*this\s*\.\s*{re.escape(assignment.name)}\s*=", snippet, flags=re.DOTALL))


def _split_top_level_commas(text: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "(<{[":
            depth += 1
        elif char in ")>}]":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        items.append(tail)
    return items
