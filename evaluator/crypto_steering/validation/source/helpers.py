"""Shared lexical and value-resolution helpers for source analyzers."""

from __future__ import annotations

import ast
import operator
import re
from typing import Iterable, TYPE_CHECKING

from crypto_steering.validation.models import (
    AnalysisProfile,
    Language,
    ResolvedValue,
    ResolutionStep,
    ValueOrigin,
)
from crypto_steering.validation.source.base import (
    Assignment,
    CallExpression,
    ClassDefinition,
    FunctionDefinition,
    ReturnExpression,
)

if TYPE_CHECKING:
    from crypto_steering.validation.source.cstyle import CStyleAnalyzer


_IDENTIFIER = r"[A-Za-z_$][A-Za-z0-9_$]*"
_MAX_RESOLUTION_STEPS = 48
_CONTROL_NAMES = {
    "async",
    "await",
    "catch",
    "for",
    "foreach",
    "if",
    "lock",
    "new",
    "return",
    "sizeof",
    "switch",
    "typeof",
    "using",
    "while",
    "with",
}


def _python_statements_terminate(statements: list[ast.stmt]) -> bool:
    """Return whether one simple guard body cannot fall through."""

    if not statements:
        return False
    terminal = statements[-1]
    if isinstance(terminal, (ast.Raise, ast.Return)):
        return True
    if not (
        isinstance(terminal, ast.Expr)
        and isinstance(terminal.value, ast.Call)
    ):
        return False
    return _python_name(terminal.value.func) in {
        "exit",
        "quit",
        "sys.exit",
        "parser.error",
    }


def _python_guard_rejects_values_below(
    condition: ast.AST,
    variable: str,
    minimum: int,
) -> bool:
    """Recognize a small, sound subset of lower-bound rejection guards."""

    if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.Or):
        # Reaching the sink means every OR operand was false. One explicit
        # lower-bound operand is therefore sufficient to establish the floor.
        return any(
            _python_guard_rejects_values_below(item, variable, minimum)
            for item in condition.values
        )
    if isinstance(condition, ast.UnaryOp) and isinstance(condition.op, ast.Not):
        return _python_condition_accepts_minimum(
            condition.operand,
            variable,
            minimum,
        )
    if not isinstance(condition, ast.Compare) or len(condition.ops) != 1:
        return False

    operator_node = condition.ops[0]
    comparator = condition.comparators[0]
    if _python_guard_target_matches(condition.left, variable):
        boundary = _python_integer_literal(comparator)
        if isinstance(operator_node, ast.Lt) and boundary is not None:
            return boundary >= minimum
        if isinstance(operator_node, ast.LtE) and boundary is not None:
            return boundary >= minimum - 1
        if isinstance(operator_node, ast.NotIn):
            allowed = _python_integer_container(comparator)
            return bool(allowed) and min(allowed) >= minimum

    if _python_guard_target_matches(comparator, variable):
        boundary = _python_integer_literal(condition.left)
        if isinstance(operator_node, ast.Gt) and boundary is not None:
            return boundary >= minimum
        if isinstance(operator_node, ast.GtE) and boundary is not None:
            return boundary >= minimum - 1
    return False


def _python_condition_accepts_minimum(
    condition: ast.AST,
    variable: str,
    minimum: int,
) -> bool:
    if not isinstance(condition, ast.Compare) or len(condition.ops) != 1:
        return False
    operator_node = condition.ops[0]
    comparator = condition.comparators[0]
    if _python_guard_target_matches(condition.left, variable):
        boundary = _python_integer_literal(comparator)
        if isinstance(operator_node, ast.GtE) and boundary is not None:
            return boundary >= minimum
        if isinstance(operator_node, ast.Gt) and boundary is not None:
            return boundary >= minimum - 1
        if isinstance(operator_node, ast.In):
            allowed = _python_integer_container(comparator)
            return bool(allowed) and min(allowed) >= minimum
    return False


def _python_guard_target_matches(node: ast.AST, expression: str) -> bool:
    if not isinstance(node, (ast.Name, ast.Attribute)):
        return False
    return ast.unparse(node) == expression


def _python_integer_container(node: ast.AST) -> tuple[int, ...]:
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return ()
    values = tuple(_python_integer_literal(item) for item in node.elts)
    return tuple(value for value in values if value is not None) if all(
        value is not None for value in values
    ) else ()


def _python_integer_literal(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(
        node.value,
        bool,
    ):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
    ):
        return (
            node.operand.value
            if isinstance(node.op, ast.UAdd)
            else -node.operand.value
        )
    return None


def mask_c_style_comments_and_literals(source: str) -> str:
    """Blank comments and literal contents while preserving delimiters/offsets."""

    result = list(source)
    index = 0
    state = "code"
    quote = ""
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char == "/" and next_char == "/":
                result[index] = result[index + 1] = " "
                state = "line_comment"
                index += 2
                continue
            if char == "/" and next_char == "*":
                result[index] = result[index + 1] = " "
                state = "block_comment"
                index += 2
                continue
            if char in {'"', "'", "`"}:
                state = "literal"
                quote = char
                index += 1
                continue
        elif state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                result[index] = " "
        elif state == "block_comment":
            if char == "*" and next_char == "/":
                result[index] = result[index + 1] = " "
                state = "code"
                index += 2
                continue
            if char != "\n":
                result[index] = " "
        elif state == "literal":
            if char == "\\":
                if index + 1 < len(source):
                    if result[index] != "\n":
                        result[index] = " "
                    if result[index + 1] != "\n":
                        result[index + 1] = " "
                    index += 2
                    continue
            if char == quote:
                state = "code"
            elif char != "\n":
                result[index] = " "
        index += 1
    return "".join(result)


def _csharp_interpolation_expression_mask(
    source: str,
    masked_source: str,
) -> str:
    """Expose C# interpolation expressions while masking surrounding text."""

    result = ["\n" if char == "\n" else " " for char in source]
    index = 0
    while index < len(source):
        verbatim = False
        if masked_source.startswith('$@"', index) or masked_source.startswith(
            '@$"', index
        ):
            quote_index = index + 2
            verbatim = True
        elif masked_source.startswith('$"', index):
            quote_index = index + 1
        else:
            index += 1
            continue

        cursor = quote_index + 1
        while cursor < len(source):
            if not verbatim and source[cursor] == "\\":
                cursor += 2
                continue
            if verbatim and source.startswith('""', cursor):
                cursor += 2
                continue
            if source[cursor] == '"':
                cursor += 1
                break
            if source.startswith("{{", cursor):
                cursor += 2
                continue
            if source[cursor] != "{":
                cursor += 1
                continue

            closing = _csharp_interpolation_expression_end(source, cursor)
            if closing == -1:
                cursor += 1
                continue
            expression = source[cursor + 1 : closing]
            masked_expression = mask_c_style_comments_and_literals(expression)
            result[cursor + 1 : closing] = masked_expression
            cursor = closing + 1
        index = cursor
    return "".join(result)


def _csharp_interpolation_expression_end(source: str, opening: int) -> int:
    depth = 1
    quote = ""
    verbatim = False
    index = opening + 1
    while index < len(source):
        char = source[index]
        if quote:
            if (
                verbatim
                and char == '"'
                and index + 1 < len(source)
                and source[index + 1] == '"'
            ):
                index += 2
                continue
            if not verbatim and char == "\\":
                index += 2
                continue
            if char == quote:
                quote = ""
                verbatim = False
            index += 1
            continue
        if source.startswith('@"', index):
            quote = '"'
            verbatim = True
            index += 2
            continue
        if char in {'"', "'"}:
            quote = char
            index += 1
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def find_matching(source: str, opening: int, open_char: str, close_char: str) -> int:
    return _find_matching_masked(
        mask_c_style_comments_and_literals(source),
        opening,
        open_char,
        close_char,
    )


def mask_constant_boolean_inactive_branches(source: str) -> str:
    """Blank dormant C-style branch bodies without changing source offsets."""

    masked = mask_c_style_comments_and_literals(source)
    result = list(source)
    for condition, body_open, body_close, else_open, else_close in (
        _c_style_constant_boolean_branches(masked)
    ):
        inactive: tuple[int, int] | None = None
        if not condition:
            inactive = (body_open, body_close)
        elif else_open is not None and else_close is not None:
            inactive = (else_open, else_close)
        if inactive is None:
            continue
        start, end = inactive
        for index in range(start + 1, end):
            if result[index] not in {"\n", "\r"}:
                result[index] = " "
    return "".join(result)


def _c_style_literal_boolean_path_is_reachable(masked: str, index: int) -> bool:
    """Recognize only literal-boolean dormant branches; retain all ambiguity."""

    for condition, body_open, body_close, else_open, else_close in (
        _c_style_constant_boolean_branches(masked)
    ):
        if body_open < index < body_close and not condition:
            return False
        if (
            else_open is not None
            and else_close is not None
            and else_open < index < else_close
            and condition
        ):
            return False
    return True


def _c_style_constant_boolean_branches(
    masked: str,
) -> Iterable[tuple[bool, int, int, int | None, int | None]]:
    for match in re.finditer(
        r"\bif\s*\(\s*(?P<value>true|false)\s*\)",
        masked,
        flags=re.IGNORECASE,
    ):
        opening = masked.find("(", match.start(), match.end())
        closing = _find_matching_masked(masked, opening, "(", ")")
        if closing == -1:
            continue
        body_open = closing + 1
        while body_open < len(masked) and masked[body_open].isspace():
            body_open += 1
        if body_open >= len(masked) or masked[body_open] != "{":
            continue
        body_close = _find_matching_masked(masked, body_open, "{", "}")
        if body_close == -1:
            continue

        else_open: int | None = None
        else_close: int | None = None
        cursor = body_close + 1
        while cursor < len(masked) and masked[cursor].isspace():
            cursor += 1
        if masked.startswith("else", cursor):
            cursor += len("else")
            while cursor < len(masked) and masked[cursor].isspace():
                cursor += 1
            if cursor < len(masked) and masked[cursor] == "{":
                candidate_close = _find_matching_masked(
                    masked,
                    cursor,
                    "{",
                    "}",
                )
                if candidate_close != -1:
                    else_open = cursor
                    else_close = candidate_close
        yield (
            match.group("value").lower() == "true",
            body_open,
            body_close,
            else_open,
            else_close,
        )


def _c_style_direct_path_is_reachable(
    analyzer: CStyleAnalyzer,
    index: int,
) -> bool:
    """Evaluate simple constant-selected ``if`` and ``switch`` branches."""

    if analyzer._reachability_depth >= _MAX_RESOLUTION_STEPS:
        return True
    if analyzer._reachability_depth == 0 and not analyzer._entrypoint_check_active:
        cached = analyzer._reachability_cache.get(index)
        if cached is not None:
            return cached
    analyzer._reachability_depth += 1
    try:
        reachable = _c_style_direct_path_is_reachable_impl(analyzer, index)
    finally:
        analyzer._reachability_depth -= 1
    if analyzer._reachability_depth == 0 and not analyzer._entrypoint_check_active:
        analyzer._reachability_cache[index] = reachable
    return reachable


def _evaluate_c_style_string_condition(
    analyzer: CStyleAnalyzer,
    condition: str,
    index: int,
) -> bool | None:
    value = _strip_expression(condition)
    negated = value.startswith("!") and not value.startswith("!=")
    boolean_expression = _strip_expression(value[1:]) if negated else value
    boolean = analyzer.resolve_strings(boolean_expression, index)
    if boolean.origin == "constant" and len(boolean.values) == 1:
        normalized = boolean.values[0].strip().lower()
        if normalized in {"true", "false"}:
            result = normalized == "true"
            return not result if negated else result
    boolean_reference = _simple_reference(boolean_expression)
    if boolean_reference is not None:
        assignment = analyzer.latest_assignment(boolean_reference, index)
        if assignment is not None:
            assigned = _strip_expression(assignment.expression).lower()
            if assigned in {"true", "false"}:
                result = assigned == "true"
                return not result if negated else result
        literal_assignments = tuple(
            re.finditer(
                rf"\b(?:boolean|bool|const|let|var)\s+"
                rf"{re.escape(boolean_reference)}\s*=\s*"
                r"(?P<value>true|false)\s*;",
                analyzer.source[:index],
                flags=re.IGNORECASE,
            )
        )
        if literal_assignments:
            result = literal_assignments[-1].group("value").lower() == "true"
            return not result if negated else result
        resolved_boolean = analyzer.resolve_origin(boolean_reference, index)
        if resolved_boolean.origin in {
            "secure_random",
            "derived_secure",
            "predictable",
            "non_csprng",
        }:
            return not negated
        if (
            resolved_boolean.origin == "constant"
            and resolved_boolean.values
            and all(
                item.strip().lower()
                not in {"", "0", "false", "null", "none", "undefined"}
                for item in resolved_boolean.values
            )
        ):
            return not negated
    numeric_match = re.fullmatch(
        r"(?P<left>.+?)\s*(?P<operator><=|>=|<|>)\s*(?P<right>.+)",
        value,
        flags=re.DOTALL,
    )
    if numeric_match is not None:
        left = analyzer.resolve_ints(numeric_match.group("left"), index)
        right = analyzer.resolve_ints(numeric_match.group("right"), index)
        if (
            left.origin == "constant"
            and right.origin == "constant"
            and len(left.values) == 1
            and len(right.values) == 1
        ):
            left_value = int(left.values[0])
            right_value = int(right.values[0])
            operator = numeric_match.group("operator")
            return {
                "<": left_value < right_value,
                "<=": left_value <= right_value,
                ">": left_value > right_value,
                ">=": left_value >= right_value,
            }[operator]
    if analyzer.language == "csharp":
        empty_check = re.fullmatch(
            r"(?:string|String)\s*\.\s*IsNullOr(?:Empty|WhiteSpace)\s*"
            r"\(\s*(?P<expression>.+?)\s*\)",
            boolean_expression,
            flags=re.DOTALL,
        )
        if empty_check is not None:
            resolved = analyzer.resolve_strings(
                empty_check.group("expression"),
                index,
            )
            if resolved.origin == "constant" and resolved.values:
                emptiness = {not value.strip() for value in resolved.values}
                if len(emptiness) == 1:
                    result = next(iter(emptiness))
                    return not result if negated else result
        null_pattern = re.fullmatch(
            r"(?P<expression>.+?)\s+is\s+(?P<negated>not\s+)?null",
            value,
            flags=re.DOTALL,
        )
        if null_pattern is not None:
            resolved = analyzer.resolve_origin(
                null_pattern.group("expression"),
                index,
            )
            if _resolved_has_only_nullish_leaves(resolved):
                is_null = True
            elif resolved.origin != "unknown" and not any(
                _strip_expression(step.expression) in {"null", "undefined"}
                for step in resolved.steps
                if step.action == "inspect"
            ):
                is_null = False
            else:
                return None
            return not is_null if null_pattern.group("negated") else is_null
    match = re.fullmatch(
        r"(?P<left>.+?)\s*(?P<operator>===|!==|==|!=)\s*(?P<right>.+)",
        value,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    # Prefer string equality. Cipher and digest names commonly contain digits
    # (for example, ``aes-256-cbc``); resolving those expressions as integers
    # first incorrectly reduces both sides to ``256``.
    left_string = analyzer.resolve_strings(match.group("left"), index)
    right_string = analyzer.resolve_strings(match.group("right"), index)
    if (
        left_string.origin == "constant"
        and right_string.origin == "constant"
        and len(left_string.values) == 1
        and len(right_string.values) == 1
    ):
        equal = _normalize_control_value(
            left_string.values[0]
        ) == _normalize_control_value(right_string.values[0])
        return not equal if "!" in match.group("operator") else equal

    if any(
        re.fullmatch(r"[-+]?\d+", item.strip()) is None
        for item in (*left_string.values, *right_string.values)
    ):
        # A runtime-selected set of algorithm names is still textual. Do not
        # compare only the numeric fragments embedded in those names.
        return None

    left_int = analyzer.resolve_ints(match.group("left"), index)
    right_int = analyzer.resolve_ints(match.group("right"), index)
    if (
        left_int.origin == "constant"
        and right_int.origin == "constant"
        and len(left_int.values) == 1
        and len(right_int.values) == 1
    ):
        equal = int(left_int.values[0]) == int(right_int.values[0])
        return not equal if "!" in match.group("operator") else equal
    return None


def _normalize_control_value(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def _find_matching_masked(
    masked: str,
    opening: int,
    open_char: str,
    close_char: str,
) -> int:
    depth = 0
    for index in range(opening, len(masked)):
        if masked[index] == open_char:
            depth += 1
        elif masked[index] == close_char:
            depth -= 1
            if depth == 0:
                return index
    return -1


def split_arguments(source: str, opening: int, closing: int) -> tuple[str, ...]:
    text = source[opening + 1 : closing]
    if not text.strip():
        return ()
    masked = mask_c_style_comments_and_literals(text)
    starts = [0]
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(masked):
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
        elif char == "," and not stack:
            starts.append(index + 1)
    boundaries = starts[1:] + [len(text) + 1]
    return tuple(
        text[start : boundary - 1].strip()
        for start, boundary in zip(starts, boundaries)
    )


def csharp_activator_constructor_arguments(
    expression: str,
) -> tuple[str, ...]:
    """Return constructor arguments wrapped by Activator.CreateInstance."""

    masked = mask_c_style_comments_and_literals(expression)
    match = re.search(r"\bActivator\s*\.\s*CreateInstance\s*\(", masked)
    if match is None:
        return ()
    opening = masked.find("(", match.start(), match.end())
    closing = find_matching(masked, opening, "(", ")")
    if opening < 0 or closing < 0:
        return ()
    activator_arguments = split_arguments(expression, opening, closing)
    if len(activator_arguments) < 2:
        return ()
    object_array = activator_arguments[1].strip()
    array_match = re.fullmatch(
        r"new\s+object\s*\[\s*\]\s*\{(?P<body>.*)\}",
        object_array,
        flags=re.DOTALL,
    )
    if array_match is None:
        return ()
    body = array_match.group("body")
    synthetic = f"({body})"
    return split_arguments(synthetic, 0, len(synthetic) - 1)


def _bind_c_style_arguments(
    parameters: tuple[str, ...],
    arguments: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Bind positional and C# named arguments to declared parameters."""

    named: dict[str, str] = {}
    positional: list[str] = []
    parameter_names = set(parameters)
    for argument in arguments:
        match = re.match(
            rf"^(?P<name>{_IDENTIFIER})\s*:\s*(?P<value>.+)$",
            argument,
            flags=re.DOTALL,
        )
        if match is not None and match.group("name") in parameter_names:
            named[match.group("name")] = match.group("value").strip()
        else:
            positional.append(argument)

    bound: list[tuple[str, str]] = []
    positional_index = 0
    for parameter in parameters:
        argument = named.get(parameter)
        if argument is None and positional_index < len(positional):
            argument = positional[positional_index]
            positional_index += 1
        if argument is not None:
            bound.append((parameter, argument))
    return tuple(bound)


def _extract_c_style_calls(source: str, masked: str) -> Iterable[CallExpression]:
    pattern = re.compile(
        rf"(?<![\w$])(?P<callee>(?:new\s+)?{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)\s*\("
    )
    for match in pattern.finditer(masked):
        callee = re.sub(r"\s+", "", source[match.start("callee") : match.end("callee")])
        if callee == "new":
            inferred = _target_typed_constructor_name(source, masked, match.start())
            if inferred is not None:
                callee = inferred
        simple = re.sub(r"^new", "", callee).rsplit(".", 1)[-1]
        if not simple or simple in _CONTROL_NAMES:
            continue
        opening = match.end() - 1
        closing = _find_matching_masked(masked, opening, "(", ")")
        if closing == -1:
            continue
        arguments = split_arguments(source, opening, closing)
        named: dict[str, str] = {}
        for argument in arguments:
            named_match = re.match(rf"^({_IDENTIFIER})\s*:\s*(.+)$", argument, flags=re.DOTALL)
            if named_match:
                named[named_match.group(1)] = named_match.group(2).strip()
        yield CallExpression(callee, arguments, named, match.start(), closing + 1)


def _extract_csharp_implicit_object_initializer_calls(
    source: str,
    masked: str,
) -> Iterable[CallExpression]:
    """Yield C# ``new Type { ... }`` constructor calls without ``()``."""

    pattern = re.compile(
        rf"\bnew\s+(?P<callee>{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)\s*"
        r"(?P<opening>\{)"
    )
    for match in pattern.finditer(masked):
        opening = match.start("opening")
        closing = _find_matching_masked(masked, opening, "{", "}")
        if closing == -1:
            continue
        callee = re.sub(
            r"\s+",
            "",
            source[match.start("callee") : match.end("callee")],
        )
        yield CallExpression(
            f"new{callee}",
            (),
            {},
            match.start(),
            closing + 1,
        )


def _extract_c_style_assignments(
    source: str,
    masked: str,
    language: Language,
    functions: tuple[FunctionDefinition, ...] = (),
) -> Iterable[Assignment]:
    records: list[Assignment] = []
    handled_equals: set[int] = set()
    if language == "csharp":
        for assignment, equals_index in _extract_csharp_parenthesized_using_assignments(
            source,
            masked,
        ):
            records.append(assignment)
            handled_equals.add(equals_index)
        tuple_destructuring = re.compile(
            r"(?:\bvar\s*)?\((?P<items>[^()\r\n]+,[^()\r\n]+)\)"
            r"\s*(?P<equals>=(?!=|>))",
            flags=re.DOTALL,
        )
        for match in tuple_destructuring.finditer(masked):
            expression_end = _next_c_style_statement_end_masked(
                masked,
                match.end("equals"),
                language,
            )
            if expression_end == -1:
                continue
            owner = _strip_expression(
                source[match.end("equals") : expression_end].strip()
            )
            if _simple_reference(owner) is None and _simple_call(owner) is None:
                continue
            for item_index, raw_item in enumerate(
                _split_inline_arguments(
                    source[match.start("items") : match.end("items")]
                )
            ):
                local_match = re.search(
                    rf"(?P<name>{_IDENTIFIER})\s*$",
                    raw_item.strip(),
                )
                if local_match is None:
                    continue
                local_name = local_match.group("name")
                if local_name == "_":
                    continue
                records.append(
                    Assignment(
                        name=local_name,
                        expression=f"{owner}[{item_index}]",
                        start=match.start("items"),
                        end=expression_end + 1,
                    )
                )
            handled_equals.add(match.start("equals"))

    if language == "typescript":
        array_destructuring = re.compile(
            r"\b(?:const|let|var)\s*\[(?P<items>[^\[\]]*)\]\s*"
            r"(?P<equals>=(?!=|>))",
            flags=re.DOTALL,
        )
        for match in array_destructuring.finditer(masked):
            expression_end = _next_c_style_statement_end_masked(
                masked,
                match.end("equals"),
                language,
            )
            if expression_end == -1:
                continue
            owner = _strip_expression(
                source[match.end("equals") : expression_end].strip()
            )
            if _simple_reference(owner) is None and _simple_call(owner) is None:
                continue
            for item_index, raw_item in enumerate(
                _split_inline_arguments(
                    source[match.start("items") : match.end("items")]
                )
            ):
                item = raw_item.strip()
                if not item:
                    continue
                if item.startswith("..."):
                    local_name = item[3:].strip()
                    expression = f"{owner}.slice({item_index})"
                else:
                    default_parts = re.split(r"\s*=\s*", item, maxsplit=1)
                    local_name = default_parts[0].strip()
                    expression = f"{owner}[{item_index}]"
                    if len(default_parts) == 2:
                        expression = f"{expression} ?? {default_parts[1].strip()}"
                if re.fullmatch(_IDENTIFIER, local_name) is None:
                    continue
                records.append(
                    Assignment(
                        name=local_name,
                        expression=expression,
                        start=match.start("items"),
                        end=expression_end + 1,
                    )
                )
            handled_equals.add(match.start("equals"))

        destructuring = re.compile(
            r"\b(?:const|let|var)\s*\{(?P<items>[^{}]+)\}\s*"
            r"(?P<equals>=(?!=|>))",
            flags=re.DOTALL,
        )
        for match in destructuring.finditer(masked):
            expression_end = _next_c_style_statement_end_masked(
                masked,
                match.end("equals"),
                language,
            )
            if expression_end == -1:
                continue
            owner = source[match.end("equals") : expression_end].strip()
            owner = _strip_expression(owner)
            if _simple_reference(owner) is None and _simple_call(owner) is None:
                continue
            for raw_item in _split_inline_arguments(
                source[match.start("items") : match.end("items")]
            ):
                item = raw_item.strip()
                item = re.sub(r"\s*=.*$", "", item, flags=re.DOTALL).strip()
                item_match = re.fullmatch(
                    rf"(?P<property>{_IDENTIFIER})"
                    rf"(?:\s*:\s*(?P<local>{_IDENTIFIER}))?",
                    item,
                )
                if item_match is None:
                    continue
                property_name = item_match.group("property")
                local_name = item_match.group("local") or property_name
                records.append(
                    Assignment(
                        name=local_name,
                        expression=f"{owner}.{property_name}",
                        start=match.start("items"),
                        end=expression_end + 1,
                    )
                )
            handled_equals.add(match.start("equals"))

    for match in re.finditer(r"(?<![=!<>])=(?!=|>)", masked):
        if match.start() in handled_equals:
            continue
        if _inside_function_parameter_list(masked, match.start(), functions):
            continue
        lhs_start = max(
            masked.rfind(";", 0, match.start()),
            masked.rfind("{", 0, match.start()),
            masked.rfind("}", 0, match.start()),
            masked.rfind("\n", 0, match.start()),
        ) + 1
        raw_lhs = source[lhs_start : match.start()]
        lhs_offset = len(raw_lhs) - len(raw_lhs.lstrip())
        lhs = raw_lhs.strip()
        name_match = None
        if language == "typescript":
            name_match = re.search(
                rf"(?P<name>{_IDENTIFIER})\s*[!?]?\s*:\s*[^:]+$",
                lhs,
                flags=re.DOTALL,
            )
        if name_match is None:
            name_match = re.search(
                rf"(?P<name>(?:this\s*\.\s*)?{_IDENTIFIER}"
                rf"(?:\s*\.\s*{_IDENTIFIER})*)\s*$",
                lhs,
            )
        if not name_match:
            continue
        expression_end = _next_c_style_statement_end_masked(
            masked,
            match.end(),
            language,
        )
        if expression_end == -1:
            continue
        name = re.sub(r"\s+", "", name_match.group("name"))
        expression = source[match.end() : expression_end].strip()
        if language == "csharp":
            # A property in a C# object initializer is terminated by a comma
            # or the initializer's closing brace, not by the semicolon after
            # the whole ``new Type { ... }`` expression.  The generic
            # statement scanner can therefore leave one unmatched ``}`` on
            # the final property value.
            while (
                expression.endswith("}")
                and expression.count("}") > expression.count("{")
            ):
                expression = expression[:-1].rstrip()
        if expression:
            records.append(
                Assignment(
                    name,
                    expression,
                    lhs_start + lhs_offset + name_match.start("name"),
                    expression_end + 1,
                )
            )

    yield from sorted(records, key=lambda item: (item.start, item.end, item.name))


def _extract_c_style_conditional_ranges(
    masked: str,
    language: Language,
) -> Iterable[tuple[int, int]]:
    """Yield control-flow regions whose bodies may not execute.

    Strings and comments are already masked. These ranges are used only to
    decide whether a previous assignment remains a reachable alternative, so
    both braced and single-statement bodies must be represented.
    """

    parenthesized = re.compile(r"\b(?:if|for|while|switch|catch)\s*\(")
    for match in parenthesized.finditer(masked):
        opening_paren = masked.find("(", match.start(), match.end())
        closing_paren = _find_matching_masked(masked, opening_paren, "(", ")")
        if closing_paren == -1:
            continue
        body_start, body_end = _c_style_statement_bounds(
            masked,
            closing_paren + 1,
            language,
        )
        if body_start == -1 or body_end == -1:
            continue
        yield body_start, body_end

        cursor = body_end
        while cursor < len(masked) and masked[cursor].isspace():
            cursor += 1
        if masked.startswith("else", cursor):
            else_start, else_end = _c_style_statement_bounds(
                masked,
                cursor + len("else"),
                language,
            )
            if else_start != -1 and else_end != -1:
                yield else_start, else_end

    direct = re.compile(r"\b(?:else|try|finally|do)\s*\{")
    for match in direct.finditer(masked):
        opening_brace = masked.rfind("{", match.start(), match.end())
        closing_brace = _find_matching_masked(
            masked,
            opening_brace,
            "{",
            "}",
        )
        if closing_brace != -1:
            yield opening_brace, closing_brace


def _inside_function_parameter_list(
    masked: str,
    index: int,
    functions: tuple[FunctionDefinition, ...],
) -> bool:
    for definition in functions:
        if not (definition.start <= index < definition.end):
            continue
        opening = masked.find("(", definition.start, definition.end)
        if opening == -1 or opening >= index:
            continue
        closing = _find_matching_masked(masked, opening, "(", ")")
        if opening < index < closing:
            return True
    return False


def _extract_csharp_parenthesized_using_assignments(
    source: str,
    masked: str,
) -> Iterable[tuple[Assignment, int]]:
    """Extract the resource declaration from ``using (var x = value)``.

    The generic statement parser terminates assignments at semicolons. A
    parenthesized C# using declaration has no semicolon after its initializer,
    so without this small special case its expression absorbs the body that
    follows and origin resolution loses the resource constructor.
    """

    pattern = re.compile(
        r"\busing\s*\(\s*(?P<declaration>[^=;()]+?)\s*(?P<equals>=)(?!=|>)",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(masked):
        opening = masked.find("(", match.start(), match.end())
        if opening == -1:
            continue
        closing = _find_matching_masked(masked, opening, "(", ")")
        if closing == -1:
            continue
        declaration = source[match.start("declaration") : match.end("declaration")]
        name_match = re.search(rf"(?P<name>{_IDENTIFIER})\s*$", declaration)
        if name_match is None:
            continue
        expression = source[match.end("equals") : closing].strip()
        if not expression:
            continue
        name_start = match.start("declaration") + name_match.start("name")
        yield (
            Assignment(
                name=name_match.group("name"),
                expression=expression,
                start=name_start,
                end=closing + 1,
            ),
            match.start("equals"),
        )


def _extract_c_style_classes(
    source: str,
    masked: str,
) -> Iterable[ClassDefinition]:
    pattern = re.compile(
        rf"\b(?:class|struct|record)\s+(?P<name>{_IDENTIFIER})[^{{;]*\{{"
    )
    for match in pattern.finditer(masked):
        opening = match.end() - 1
        closing = _find_matching_masked(masked, opening, "{", "}")
        if closing == -1:
            continue
        yield ClassDefinition(
            name=source[match.start("name") : match.end("name")],
            start=match.start(),
            end=closing + 1,
        )


def _target_typed_constructor_name(
    source: str,
    masked: str,
    new_start: int,
) -> str | None:
    statement_start = max(
        masked.rfind(";", 0, new_start),
        masked.rfind("{", 0, new_start),
        masked.rfind("}", 0, new_start),
        masked.rfind("\n", 0, new_start),
    ) + 1
    prefix = source[statement_start:new_start]
    match = re.search(
        rf"(?P<type>(?:global::)?{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*"
        rf"(?:\s*<[^;=]+>)?(?:\s*\[\s*\])?)\s+{_IDENTIFIER}\s*=\s*$",
        prefix,
        flags=re.DOTALL,
    )
    if not match:
        return None
    value = re.sub(r"\s+", "", match.group("type"))
    return value.rsplit(".", 1)[-1]


def _extract_c_style_functions(
    source: str,
    masked: str,
    language: Language,
) -> Iterable[FunctionDefinition]:
    explicit_function_name_starts: set[int] = set()

    if language == "csharp":
        # Named tuple parameter types contain parentheses, so the flat method
        # pattern below cannot parse them. Read C# method headers with balanced
        # delimiters before falling back to the lightweight general pattern.
        csharp_method = re.compile(
            rf"(?m)^[ \t]*"
            rf"(?:(?:public|private|protected|internal|static|virtual|override|"
            rf"abstract|sealed|async|unsafe|extern|new|partial)\s+)*"
            rf"(?:global::)?{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*"
            rf"(?:\s*<[^;={{}}\n]+>)?(?:\s*\[\s*\])?\??\s+"
            rf"(?P<name>{_IDENTIFIER})\s*\("
        )
        for match in csharp_method.finditer(masked):
            name = match.group("name")
            if name in _CONTROL_NAMES:
                continue
            opening = masked.find("(", match.start("name"), match.end())
            parameter_end = _find_matching_masked(masked, opening, "(", ")")
            if parameter_end == -1:
                continue
            body_start = parameter_end + 1
            while body_start < len(masked) and masked[body_start].isspace():
                body_start += 1
            if body_start >= len(masked) or masked[body_start] != "{":
                continue
            body_end = _find_matching_masked(masked, body_start, "{", "}")
            if body_end == -1:
                continue
            parameters = tuple(
                item
                for item in (
                    _parameter_name(raw)
                    for raw in _split_parameter_text(
                        source[opening + 1 : parameter_end]
                    )
                )
                if item
            )
            explicit_function_name_starts.add(match.start("name"))
            yield FunctionDefinition(
                name,
                parameters,
                _c_style_return_expressions(
                    source,
                    masked,
                    body_start,
                    body_end,
                    language,
                ),
                match.start("name"),
                body_end + 1,
            )

    # Named TypeScript functions can have nested parameter types and
    # object-shaped return types. The broad method regex below cannot tell the
    # return-type brace from the function-body brace, so parse these headers by
    # balanced delimiters first.
    named_function = re.compile(
        rf"\b(?:(?:export|default|declare|async)\s+)*"
        rf"function\s*\*?\s*(?P<name>{_IDENTIFIER})"
    )
    for match in named_function.finditer(masked):
        opening = masked.find("(", match.end())
        if opening == -1:
            continue
        header_gap = masked[match.end() : opening]
        if ";" in header_gap or "{" in header_gap:
            continue
        parameter_end = _find_matching_masked(masked, opening, "(", ")")
        if parameter_end == -1:
            continue
        body_start = _typescript_function_body_start(masked, parameter_end + 1)
        if body_start == -1:
            continue
        body_end = _find_matching_masked(masked, body_start, "{", "}")
        if body_end == -1:
            continue
        parameters = tuple(
            item
            for item in (
                _parameter_name(raw)
                for raw in _split_parameter_text(
                    source[opening + 1 : parameter_end]
                )
            )
            if item
        )
        explicit_function_name_starts.add(match.start("name"))
        yield FunctionDefinition(
            source[match.start("name") : match.end("name")],
            parameters,
            _c_style_return_expressions(
                source,
                masked,
                body_start,
                body_end,
                language,
            ),
            match.start("name"),
            body_end + 1,
        )

    # TypeScript constructor defaults commonly contain object literals. A
    # delimiter-aware pass is required because the general method regex below
    # deliberately excludes braces from its flat parameter capture.
    for match in re.finditer(r"\bconstructor\s*\(", masked):
        opening = masked.find("(", match.start(), match.end())
        if opening == -1:
            continue
        parameter_end = _find_matching_masked(masked, opening, "(", ")")
        if parameter_end == -1:
            continue
        body_start = parameter_end + 1
        while body_start < len(masked) and masked[body_start].isspace():
            body_start += 1
        if body_start >= len(masked) or masked[body_start] != "{":
            continue
        body_end = _find_matching_masked(masked, body_start, "{", "}")
        if body_end == -1:
            continue
        parameters = tuple(
            item
            for item in (
                _parameter_name(raw)
                for raw in _split_parameter_text(
                    source[opening + 1 : parameter_end]
                )
            )
            if item
        )
        yield FunctionDefinition(
            "constructor",
            parameters,
            (),
            match.start(),
            body_end + 1,
        )

    if language == "typescript":
        classes = tuple(_extract_c_style_classes(source, masked))
        method_header = re.compile(
            rf"\b(?:(?:public|private|protected|static|abstract|async|override)\s+)*"
            rf"(?P<name>{_IDENTIFIER})\s*\("
        )
        for match in method_header.finditer(masked):
            name = match.group("name")
            if name in _CONTROL_NAMES or name == "constructor":
                continue
            if not any(
                definition.start <= match.start("name") <= definition.end
                for definition in classes
            ):
                continue
            opening = masked.find("(", match.start("name"), match.end())
            if opening == -1:
                continue
            parameter_end = _find_matching_masked(masked, opening, "(", ")")
            if parameter_end == -1:
                continue
            body_start = _typescript_function_body_start(masked, parameter_end + 1)
            if body_start == -1 or masked[body_start] != "{":
                continue
            body_end = _find_matching_masked(masked, body_start, "{", "}")
            if body_end == -1:
                continue
            parameters = tuple(
                item
                for item in (
                    _parameter_name(raw)
                    for raw in _split_parameter_text(
                        source[opening + 1 : parameter_end]
                    )
                )
                if item
            )
            explicit_function_name_starts.add(match.start("name"))
            yield FunctionDefinition(
                name,
                parameters,
                _c_style_return_expressions(
                    source,
                    masked,
                    body_start,
                    body_end,
                    language,
                ),
                match.start("name"),
                body_end + 1,
            )

        assigned_function = re.compile(
            rf"\b(?:const|let|var)\s+(?P<name>{_IDENTIFIER})"
            r"(?:\s*:[^=;]+)?\s*=\s*(?:async\s+)?function"
            rf"(?:\s+{_IDENTIFIER})?\s*\("
        )
        for match in assigned_function.finditer(masked):
            opening = masked.find("(", match.start("name"), match.end())
            if opening == -1:
                continue
            parameter_end = _find_matching_masked(masked, opening, "(", ")")
            if parameter_end == -1:
                continue
            body_start = _typescript_function_body_start(masked, parameter_end + 1)
            if body_start == -1 or masked[body_start] != "{":
                continue
            body_end = _find_matching_masked(masked, body_start, "{", "}")
            if body_end == -1:
                continue
            parameters = tuple(
                item
                for item in (
                    _parameter_name(raw)
                    for raw in _split_parameter_text(
                        source[opening + 1 : parameter_end]
                    )
                )
                if item
            )
            explicit_function_name_starts.add(match.start("name"))
            yield FunctionDefinition(
                match.group("name"),
                parameters,
                _c_style_return_expressions(
                    source,
                    masked,
                    body_start,
                    body_end,
                    language,
                ),
                match.start("name"),
                body_end + 1,
            )

    assigned_arrow = re.compile(
        rf"\b(?:const|let|var)\s+(?P<name>{_IDENTIFIER})"
        r"(?:\s*:[^=;]+)?\s*=\s*(?:async\s+)?"
    )
    for match in assigned_arrow.finditer(masked):
        parameter_start = match.end()
        while parameter_start < len(masked) and masked[parameter_start].isspace():
            parameter_start += 1
        if parameter_start >= len(masked):
            continue
        if masked[parameter_start] == "(":
            parameter_end = _find_matching_masked(
                masked,
                parameter_start,
                "(",
                ")",
            )
            if parameter_end == -1:
                continue
            parameter_text = source[parameter_start + 1 : parameter_end]
            arrow_search_start = parameter_end + 1
        else:
            single = re.match(_IDENTIFIER, masked[parameter_start:])
            if single is None:
                continue
            parameter_text = source[
                parameter_start : parameter_start + single.end()
            ]
            arrow_search_start = parameter_start + single.end()
        if language == "typescript":
            body_start = _typescript_arrow_body_start(
                masked,
                arrow_search_start,
            )
            if body_start == -1:
                continue
        else:
            arrow = re.match(
                r"\s*(?:\:\s*[^={;]+)?\s*=>",
                masked[arrow_search_start:],
            )
            if arrow is None:
                continue
            body_start = arrow_search_start + arrow.end()
        parameters = tuple(
            item
            for item in (
                _parameter_name(raw)
                for raw in _split_parameter_text(parameter_text)
            )
            if item
        )
        while body_start < len(masked) and masked[body_start].isspace():
            body_start += 1
        if body_start < len(masked) and masked[body_start] == "{":
            closing = _find_matching_masked(masked, body_start, "{", "}")
            if closing == -1:
                continue
            body = source[body_start + 1 : closing]
            body_masked = mask_c_style_comments_and_literals(body)
            returns: list[ReturnExpression] = []
            for return_match in re.finditer(r"\breturn\b", body_masked):
                end = _next_c_style_statement_end_masked(
                    body_masked,
                    return_match.end(),
                    language,
                )
                if end != -1:
                    returns.append(
                        ReturnExpression(
                            body[return_match.end() : end].strip(),
                            body_start + 1 + return_match.end(),
                        )
                    )
            yield FunctionDefinition(
                match.group("name"),
                parameters,
                tuple(returns),
                match.start(),
                closing + 1,
            )
            continue
        end = _next_c_style_statement_end_masked(masked, body_start, language)
        if end == -1:
            continue
        returned = source[body_start:end].strip()
        yield FunctionDefinition(
            match.group("name"),
            parameters,
            (ReturnExpression(returned, body_start),),
            match.start(),
            end + 1,
        )

    pattern = re.compile(
        rf"(?P<name>{_IDENTIFIER})\s*\((?P<params>[^(){{}};]*)\)"
        r"(?:\s*:\s*[^={;]+)?\s*(?P<body>\{|=>)"
    )
    for match in pattern.finditer(masked):
        name = match.group("name")
        if language == "typescript" and name == "function":
            property_match = re.search(
                rf"(?P<name>{_IDENTIFIER})\s*:\s*(?:async\s+)?$",
                masked[max(0, match.start() - 96) : match.start()],
            )
            if property_match is not None:
                name = property_match.group("name")
            else:
                continue
        if (
            name in _CONTROL_NAMES
            or name == "constructor"
            or match.start("name") in explicit_function_name_starts
        ):
            continue
        prefix = masked[max(0, match.start() - 24) : match.start()]
        if re.search(r"\bnew\s*$", prefix):
            continue
        parameters = tuple(_parameter_name(item) for item in _split_parameter_text(match.group("params")))
        parameters = tuple(item for item in parameters if item)
        if match.group("body") == "=>":
            end = _next_c_style_statement_end_masked(
                masked,
                match.end(),
                language,
            )
            if end == -1:
                continue
            returned = source[match.end() : end].strip()
            yield FunctionDefinition(
                name,
                parameters,
                (ReturnExpression(returned, match.end()),),
                match.start(),
                end + 1,
            )
            continue
        opening = match.end() - 1
        closing = _find_matching_masked(masked, opening, "{", "}")
        if closing == -1:
            continue
        body = source[opening + 1 : closing]
        body_masked = mask_c_style_comments_and_literals(body)
        returns: list[ReturnExpression] = []
        for return_match in re.finditer(r"\breturn\b", body_masked):
            end = _next_c_style_statement_end_masked(
                body_masked,
                return_match.end(),
                language,
            )
            if end != -1:
                returns.append(
                    ReturnExpression(
                        body[return_match.end() : end].strip(),
                        opening + 1 + return_match.end(),
                    )
                )
        yield FunctionDefinition(name, parameters, tuple(returns), match.start(), closing + 1)


def _typescript_function_body_start(masked: str, start: int) -> int:
    cursor = start
    while cursor < len(masked) and masked[cursor].isspace():
        cursor += 1
    if cursor >= len(masked):
        return -1
    if masked[cursor] != ":":
        return cursor if masked[cursor] == "{" else -1

    cursor += 1
    stack: list[str] = []
    type_started = False
    previous_significant = ":"
    closing_for = {")": "(", "]": "[", "}": "{", ">": "<"}
    while cursor < len(masked):
        char = masked[cursor]
        if char.isspace():
            cursor += 1
            continue
        if char == ";" and not stack:
            return -1
        if char == "{" and not stack:
            if not type_started or previous_significant in {"|", "&", "="}:
                stack.append(char)
                type_started = True
                previous_significant = char
                cursor += 1
                continue
            return cursor
        if char in "([<" or (char == "{" and stack):
            stack.append(char)
        elif char in ")]}>" and stack and stack[-1] == closing_for[char]:
            stack.pop()
        type_started = True
        previous_significant = char
        cursor += 1
    return -1


def _typescript_arrow_body_start(masked: str, start: int) -> int:
    """Return the expression/body start after a TypeScript arrow header."""

    cursor = start
    while cursor < len(masked) and masked[cursor].isspace():
        cursor += 1
    if masked[cursor : cursor + 2] == "=>":
        return cursor + 2
    if cursor >= len(masked) or masked[cursor] != ":":
        return -1

    cursor += 1
    stack: list[str] = []
    closing_for = {")": "(", "]": "[", "}": "{", ">": "<"}
    while cursor < len(masked):
        if not stack and masked[cursor : cursor + 2] == "=>":
            return cursor + 2
        char = masked[cursor]
        if char == ";" and not stack:
            return -1
        if char in "([{<":
            stack.append(char)
        elif char in ")]}>" and stack and stack[-1] == closing_for[char]:
            stack.pop()
        cursor += 1
    return -1


def _c_style_return_expressions(
    source: str,
    masked: str,
    body_start: int,
    body_end: int,
    language: Language,
) -> tuple[ReturnExpression, ...]:
    body = source[body_start + 1 : body_end]
    body_masked = masked[body_start + 1 : body_end]
    returns: list[ReturnExpression] = []
    for return_match in re.finditer(r"\breturn\b", body_masked):
        end = _next_c_style_statement_end_masked(
            body_masked,
            return_match.end(),
            language,
        )
        if end == -1:
            continue
        returns.append(
            ReturnExpression(
                body[return_match.end() : end].strip(),
                body_start + 1 + return_match.end(),
            )
        )
    return tuple(returns)


def _next_top_level_semicolon(source: str, start: int) -> int:
    return _next_top_level_semicolon_masked(
        mask_c_style_comments_and_literals(source),
        start,
    )


def _next_c_style_statement_end_masked(
    masked: str,
    start: int,
    language: Language,
) -> int:
    if language != "typescript":
        return _next_top_level_semicolon_masked(masked, start)

    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    quote = ""
    for index in range(start, len(masked)):
        char = masked[index]
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in {'"', "'", "`"}:
            quote = char
            continue
        if char in "([{":
            stack.append(char)
            continue
        if char in ")]}" and stack and stack[-1] == pairs[char]:
            stack.pop()
            continue
        if char == ";" and not stack:
            return index
        if char != "\n" or stack:
            continue

        before = masked[start:index].rstrip()
        after = masked[index + 1 :].lstrip()
        if not before:
            continue
        if _typescript_expression_continues_after_newline(before, after):
            continue
        return index
    return len(masked)


def _typescript_expression_continues_after_newline(
    before: str,
    after: str,
) -> bool:
    if before.endswith(
        (
            ".",
            ",",
            "+",
            "-",
            "*",
            "/",
            "%",
            "&&",
            "||",
            "??",
            "?",
            ":",
            "=",
            "=>",
            "<",
            ">",
            "&",
            "|",
            "^",
            "!",
        )
    ):
        return True
    if re.search(
        r"\b(?:await|delete|instanceof|new|throw|typeof|void|yield)\s*$",
        before,
    ):
        return True
    return bool(
        re.match(
            r"^(?:\.|,|\?|:|\+|-|\*|/|%|&&|\|\||\?\?|&|\||\^|"
            r"instanceof\b|in\b|as\b|satisfies\b)",
            after,
        )
    )


def _next_top_level_semicolon_masked(masked: str, start: int) -> int:
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index in range(start, len(masked)):
        char = masked[index]
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
        elif char == ";" and not stack:
            return index
    return -1


def _typescript_import_aliases(source: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    import_items = list(
        re.finditer(
            r"import\s*\{(?P<items>[^{}]*)\}\s*from\s*"
            r"[\"'](?:node:)?crypto[\"']",
            source,
        )
    )
    import_items.extend(
        re.finditer(
            r"\b(?:const|let|var)\s*\{(?P<items>[^{}]*)\}\s*=\s*"
            r"require\s*\(\s*[\"'](?:node:)?crypto[\"']\s*\)",
            source,
        )
    )
    for match in import_items:
        for raw in match.group("items").split(","):
            item = re.sub(r"\btype\s+", "", raw).strip()
            if not item:
                continue
            parts = re.split(r"\s+as\s+|\s*:\s*", item)
            imported = parts[0].strip()
            local = parts[-1].strip()
            if re.fullmatch(_IDENTIFIER, imported) and re.fullmatch(
                _IDENTIFIER,
                local,
            ):
                aliases[imported] = local
    return aliases


def _typescript_process_runtime_aliases(source: str) -> frozenset[str]:
    aliases: set[str] = set()
    for match in re.finditer(
        r"import\s*\{(?P<items>[^{}]*)\}\s*from\s*"
        r"[\"'](?:node:)?process[\"']",
        source,
    ):
        for raw in match.group("items").split(","):
            item = re.sub(r"\btype\s+", "", raw).strip()
            if not item:
                continue
            parts = re.split(r"\s+as\s+", item)
            if parts[0].strip() in {"argv", "env", "stdin"}:
                aliases.add(parts[-1].strip())
    return frozenset(aliases)


def _typescript_filesystem_read_aliases(source: str) -> frozenset[str]:
    aliases: set[str] = set()
    for match in re.finditer(
        r"import\s*\{(?P<items>[^{}]*)\}\s*from\s*"
        r"[\"'](?:node:)?fs(?:/promises)?[\"']",
        source,
    ):
        for raw in match.group("items").split(","):
            item = re.sub(r"\btype\s+", "", raw).strip()
            if not item:
                continue
            parts = re.split(r"\s+as\s+", item)
            if parts[0].strip() in {"readFile", "readFileSync"}:
                aliases.add(parts[-1].strip())
    return frozenset(aliases)


def _typescript_filesystem_namespace_aliases(source: str) -> frozenset[str]:
    return frozenset(
        match.group("alias")
        for match in re.finditer(
            rf"import\s*\*\s*as\s*(?P<alias>{_IDENTIFIER})\s*"
            r"from\s*[\"'](?:node:)?fs(?:/promises)?[\"']",
            source,
        )
    )


def _typescript_crypto_namespace_aliases(source: str) -> frozenset[str]:
    aliases = {
        match.group("alias")
        for match in re.finditer(
            rf"import\s*\*\s*as\s*(?P<alias>{_IDENTIFIER})\s*"
            r"from\s*[\"'](?:node:)?crypto[\"']",
            source,
        )
    }
    aliases.update(
        match.group("alias")
        for match in re.finditer(
            rf"\b(?:const|let|var)\s+(?P<alias>{_IDENTIFIER})\s*=\s*"
            r"require\s*\(\s*[\"'](?:node:)?crypto[\"']\s*\)",
            source,
        )
    )
    return frozenset(aliases)


def _direct_c_style_value(
    expression: str,
    language: Language,
    mode: str,
) -> tuple[ValueOrigin, tuple[str, ...]] | None:
    expr = expression.strip()
    if mode == "string":
        if expr in {"true", "false"}:
            return "constant", (expr,)
        literal = _string_literal(expr)
        if literal is not None:
            return "constant", (literal,)
        if len(expr) >= 2 and expr.startswith("`") and expr.endswith("`"):
            return "constant", (expr[1:-1],)
        if language == "csharp":
            type_match = re.fullmatch(
                r"typeof\s*\(\s*(?:global::)?"
                r"(?:System\s*\.\s*Security\s*\.\s*Cryptography\s*\.\s*)?"
                r"(?P<type>(?:HMAC)?(?:MD5|SHA1|SHA224|SHA256|SHA384|SHA512|"
                r"SHA3_224|SHA3_256|SHA3_384|SHA3_512))\s*\)",
                expr,
                flags=re.IGNORECASE,
            )
            if type_match is not None:
                return "constant", (type_match.group("type"),)
        enum_match = re.fullmatch(
            rf"(?P<owner>(?:{_IDENTIFIER}\.)+)(?P<member>{_IDENTIFIER})",
            expr,
        )
        enum_owner = (
            re.sub(r"\s+", "", enum_match.group("owner"))
            if enum_match is not None
            else ""
        )
        if (
            enum_match
            and not (
                language == "typescript"
                and enum_owner.startswith("process.env.")
            )
            and enum_match.group("member").upper()
            == enum_match.group("member")
        ):
            return "constant", (enum_match.group("member"),)
    if mode == "int":
        evaluated = _evaluate_c_style_int(expr)
        if evaluated is not None:
            return "constant", (str(evaluated),)
    if mode == "origin":
        if re.fullmatch(r"(?:string|String)\s*\.\s*Empty", expr):
            return "constant", ("",)
        if language == "csharp" and re.fullmatch(
            r"new\s+string\s*\(\s*'(?:\\.|[^'\\])'\s*,\s*\d+\s*\)",
            expr,
            flags=re.IGNORECASE,
        ):
            return "constant", ()
        if language == "typescript" and re.fullmatch(
            r"(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
            r"\s*\.\s*repeat\s*\(\s*\d+\s*\)",
            expr,
        ):
            return "constant", ()
        evaluated = _evaluate_c_style_int(expr)
        if evaluated is not None:
            return "constant", (str(evaluated),)

    secure_patterns = (
        r"\bRandomNumberGenerator\s*\.\s*(?:Create|GetBytes|Fill|GetInt32)\s*\(",
        r"\bnew\s+(?:System\s*\.\s*Security\s*\.\s*Cryptography\s*\.\s*)?"
        r"RNGCryptoServiceProvider\s*\(",
        r"\b(?:Aes|AES)\s*\.\s*Create\s*\(\s*\)\s*\.\s*(?:Key|IV)\b",
        r"\bnew\s+HMAC(?:SHA256|SHA384|SHA512|SHA3_256|SHA3_384|SHA3_512)"
        r"\s*\(\s*\)",
        r"\b(?:System\s*\.\s*)?Guid\s*\.\s*NewGuid\s*\(\s*\)",
    )
    if language == "java":
        secure_patterns += (
            r"\b(?:java\s*\.\s*util\s*\.\s*)?UUID\s*\.\s*randomUUID\s*\(\s*\)",
        )
    direct_process_runtime = bool(
        re.match(r"^process\s*\.\s*(?:env|argv|stdin)\b", expr)
    )
    direct_csharp_args = bool(re.match(r"^args\s*\[", expr))
    runtime_patterns = (
        r"\b(?:readline|createInterface|question|readInputLine|readLine)\s*\(",
        r"\breadFileSync\s*\(\s*0\b",
        r"\b(?:Environment\.GetEnvironmentVariable|Console\.ReadLine|Console\.ReadKey)\s*\(",
        r"\bEnvironment\s*\.\s*GetCommandLineArgs\s*\(\s*\)\s*\[",
        r"^(?:System\s*\.\s*IO\s*\.\s*)?File\s*\.\s*"
        r"(?:ReadAllText|ReadAllBytes)\s*\(",
        r"\b(?:System\s*\.\s*)?Console\s*\.\s*In\s*\.\s*"
        r"(?:ReadLine|ReadToEnd)\s*\(",
    )
    non_csprng_patterns = (
        r"\bMath\s*\.\s*random\s*\(",
        r"\bnew\s+(?:System\.)?Random\s*\(",
        r"\b(?:System\.)?Random\s*\.\s*Shared\b",
        r"\b(?:Random|System\.Random)\s+[A-Za-z_]\w*\s*=",
    )
    predictable_patterns = (
        r"\b(?:Date\.now|new\s+Date\b|"
        r"DateTime(?:\.Now|\.UtcNow)|"
        r"DateTimeOffset(?:\.Now|\.UtcNow|\..*ToUnixTime)|"
        r"Environment\.TickCount)\b",
        r"\b(?:constant|fixed|timestamp).*seed\b",
        r"\bEnvironment\s*\.\s*(?:UserName|MachineName|UserDomainName)\b",
    )
    if language == "typescript":
        predictable_patterns += (
            r"\b(?:os\s*\.\s*)?(?:hostname|type|release|platform|arch)\s*\(",
            r"\bprocess\s*\.\s*(?:pid|ppid|title)\b",
        )
    if any(re.search(pattern, expr, flags=re.IGNORECASE | re.DOTALL) for pattern in secure_patterns):
        return "secure_random", ()
    if (
        direct_process_runtime
        or direct_csharp_args
        or any(
            re.search(pattern, expr, flags=re.IGNORECASE | re.DOTALL)
            for pattern in runtime_patterns
        )
    ):
        if _c_style_has_literal_fallback(expr):
            return "predictable", ()
        return "runtime_input", ()
    if any(re.search(pattern, expr, flags=re.IGNORECASE | re.DOTALL) for pattern in non_csprng_patterns):
        return "non_csprng", ()
    if any(re.search(pattern, expr, flags=re.IGNORECASE | re.DOTALL) for pattern in predictable_patterns):
        return "predictable", ()

    if mode == "origin":
        if _string_literal(expr) is not None:
            return "constant", (_string_literal(expr) or "",)
        if re.fullmatch(r"\[\s*(?:[-+]?\d+\s*,\s*)*[-+]?\d+\s*\]", expr):
            return "constant", ()
        if re.fullmatch(
            r"new\s+byte\s*\[\s*\]\s*\{\s*"
            r"(?:[-+]?\d+\s*,\s*)*[-+]?\d+\s*\}",
            expr,
            flags=re.IGNORECASE,
        ):
            return "constant", ()
        if re.fullmatch(
            r"new\s+byte\s*\[\s*\d+\s*\]\s*\{\s*"
            r"(?:0[xX][0-9A-Fa-f]+|[-+]?\d+)"
            r"(?:\s*,\s*(?:0[xX][0-9A-Fa-f]+|[-+]?\d+))*\s*\}",
            expr,
            flags=re.IGNORECASE,
        ):
            return "constant", ()
        if language == "csharp" and re.fullmatch(
            r"\{\s*(?:0[xX][0-9A-Fa-f]+|[-+]?\d+)"
            r"(?:\s*,\s*(?:0[xX][0-9A-Fa-f]+|[-+]?\d+))*\s*\}",
            expr,
        ):
            return "constant", ()
        if re.fullmatch(
            r"(?:Buffer\s*\.\s*alloc\s*\([^)]*\)"
            r"|new\s+byte\s*\[[^]]*\](?:\s*\{(?s:.*)\})?"
            r"|new\s+byte\s*\[\s*\]\s*\{(?s:.*)\})",
            expr,
        ):
            if not re.search(r"\brandom|RandomNumberGenerator", expr, flags=re.IGNORECASE):
                return "constant", ()
    return None


def _typescript_template_interpolations(
    expression: str,
) -> tuple[str, ...] | None:
    """Return simple template expressions; leave nested shapes unresolved."""

    value = expression.strip()
    if len(value) < 2 or not value.startswith("`") or not value.endswith("`"):
        return None
    body = value[1:-1]
    matches = tuple(re.finditer(r"\$\{(?P<expression>[^{}]+)\}", body, flags=re.DOTALL))
    if not matches or body.count("${") != len(matches):
        return None
    return tuple(match.group("expression").strip() for match in matches)


def _typescript_template_fields(
    expression: str,
    delimiter: str,
) -> tuple[str, ...] | None:
    value = expression.strip()
    if len(value) < 2 or not value.startswith("`") or not value.endswith("`"):
        return None
    body = value[1:-1]
    matches = tuple(
        re.finditer(r"\$\{(?P<expression>[^{}]+)\}", body, flags=re.DOTALL)
    )
    if not matches or body.count("${") != len(matches):
        return None
    marked: list[str] = []
    cursor = 0
    for position, match in enumerate(matches):
        marked.append(body[cursor : match.start()])
        marked.append(f"\0{position}\0")
        cursor = match.end()
    marked.append(body[cursor:])
    fields: list[str] = []
    for field in "".join(marked).split(delimiter):
        marker = re.fullmatch(r"\0(?P<position>\d+)\0", field)
        if marker is None:
            return None
        fields.append(matches[int(marker.group("position"))].group("expression").strip())
    return tuple(fields)


def _c_style_container_elements(
    expression: str,
    *,
    allow_shorthand_initializer: bool = False,
) -> tuple[str, ...] | None:
    raw_value = expression.strip()
    if raw_value.startswith("(") and raw_value.endswith(")"):
        tuple_elements = split_arguments(raw_value, 0, len(raw_value) - 1)
        if len(tuple_elements) > 1:
            return tuple_elements
    value = _strip_expression(expression)
    opening = -1
    closing = -1
    if value.startswith("[") and value.endswith("]"):
        opening, closing = 0, len(value) - 1
    elif (
        allow_shorthand_initializer
        and value.startswith("{")
        and value.endswith("}")
    ):
        opening, closing = 0, len(value) - 1
    else:
        match = re.fullmatch(
            r"new\s*(?:[A-Za-z_$][A-Za-z0-9_$<>?,.]*\s*)?"
            r"\[\s*(?:\d+)?\s*\]\s*\{(?P<body>.*)\}",
            value,
            flags=re.DOTALL,
        )
        if match:
            body_start = match.start("body")
            synthetic = "[" + match.group("body") + "]"
            return split_arguments(synthetic, 0, len(synthetic) - 1)
    if opening == -1:
        return None
    return split_arguments(value, opening, closing)


def _c_style_postfix_update_operand(expression: str) -> str | None:
    value = _strip_expression(expression)
    match = re.fullmatch(
        rf"(?P<operand>{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)\s*(?:\+\+|--)",
        value,
    )
    return match.group("operand") if match is not None else None


def _c_style_object_property_value(
    expression: str,
    property_name: str,
) -> str | None:
    value = _strip_expression(expression)
    if value.startswith("{") and value.endswith("}"):
        body = value[1:-1]
    else:
        csharp_initializer = re.fullmatch(
            r"new\s+[A-Za-z_$][A-Za-z0-9_$.<>?,\s]*"
            r"(?:\([^{}]*\))?\s*\{(?P<body>.*)\}",
            value,
            flags=re.DOTALL,
        )
        if csharp_initializer is None:
            return None
        body = csharp_initializer.group("body")
    spread_sources: list[str] = []
    for item in _split_inline_arguments(body):
        candidate = item.strip()
        if candidate.startswith("..."):
            spread = _strip_expression(candidate[3:])
            if _simple_reference(spread) is not None:
                spread_sources.append(spread)
            continue
        if candidate == property_name:
            return candidate
        match = re.match(
            rf"^(?:[\"']?{re.escape(property_name)}[\"']?)\s*[:=]\s*(?P<value>.+)$",
            candidate,
            flags=re.DOTALL,
        )
        if match is not None:
            return match.group("value").strip()
    if spread_sources:
        return f"{spread_sources[-1]}.{property_name}"
    return None


def _c_style_object_value_expressions(
    expression: str,
) -> tuple[str, ...] | None:
    """Return every value from a finite C-style object initializer."""

    value = _strip_expression(expression)
    if value.startswith("{") and value.endswith("}"):
        body = value[1:-1]
    else:
        csharp_initializer = re.fullmatch(
            r"new\s+[A-Za-z_$][A-Za-z0-9_$.<>?,\s]*"
            r"(?:\([^{}]*\))?\s*\{(?P<body>.*)\}",
            value,
            flags=re.DOTALL,
        )
        if csharp_initializer is None:
            return None
        body = csharp_initializer.group("body")
    values: list[str] = []
    for item in _split_inline_arguments(body):
        candidate = item.strip()
        if not candidate:
            continue
        if candidate.startswith("..."):
            return None
        match = re.match(
            r"^(?:[\"'][^\"']+[\"']|-?\d+|[A-Za-z_$][A-Za-z0-9_$]*)"
            r"\s*[:=]\s*(?P<value>.+)$",
            candidate,
            flags=re.DOTALL,
        )
        if match is not None:
            values.append(match.group("value").strip())
            continue
        if re.fullmatch(_IDENTIFIER, candidate):
            values.append(candidate)
            continue
        return None
    return tuple(values) if values else None


def _csharp_primary_record_member_index(
    source: str,
    class_name: str,
    property_name: str,
) -> int | None:
    declaration = re.search(
        rf"\brecord(?:\s+(?:class|struct))?\s+{re.escape(class_name)}\s*"
        r"\((?P<parameters>[^)]*)\)",
        source,
        flags=re.DOTALL,
    )
    if declaration is None:
        return None
    for index, parameter in enumerate(
        _split_inline_arguments(declaration.group("parameters"))
    ):
        name = re.search(rf"(?P<name>{_IDENTIFIER})\s*$", parameter.strip())
        if name is not None and name.group("name") == property_name:
            return index
    return None


def _python_subscript_selector(node: ast.AST) -> tuple[str, object] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, str)):
        return "index", node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return "index", -node.operand.value
    if isinstance(node, ast.Slice) and all(
        value is None for value in (node.lower, node.upper, node.step)
    ):
        return "slice", None
    if isinstance(node, ast.Slice):
        lower = _evaluate_int_ast(node.lower) if node.lower is not None else None
        upper = _evaluate_int_ast(node.upper) if node.upper is not None else None
        step = _evaluate_int_ast(node.step) if node.step is not None else None
        if all(
            raw is None or value is not None
            for raw, value in (
                (node.lower, lower),
                (node.upper, upper),
                (node.step, step),
            )
        ):
            return "bounded_slice", (lower, upper, step)
    return None


def _python_literal_container_element(
    container: ast.AST,
    selector: tuple[str, object],
) -> ast.AST | None:
    kind, value = selector
    if kind != "index":
        return None
    if isinstance(container, (ast.List, ast.Tuple)) and isinstance(value, int):
        position = value if value >= 0 else len(container.elts) + value
        return container.elts[position] if 0 <= position < len(container.elts) else None
    if isinstance(container, ast.Dict):
        for key, item in zip(container.keys, container.values):
            if isinstance(key, ast.Constant) and key.value == value:
                return item
    return None


def _direct_python_name(name: str, mode: str) -> tuple[ValueOrigin, tuple[str, ...]] | None:
    if re.search(r"(?:^|\.)argv(?:\[|$)|(?:^|\.)environ(?:\[|$)", name):
        return "runtime_input", ()
    if mode == "string" and re.fullmatch(
        r"(?:cryptography\.hazmat\.primitives\.)?"
        r"hashes\.(?:MD5|SHA1|SHA224|SHA256|SHA384|SHA512|SHA3_\d+)",
        name,
    ):
        return "constant", (name.rsplit(".", 1)[-1],)
    if mode == "string" and re.fullmatch(
        r"hashlib\.(?:md5|sha1|sha224|sha256|sha384|sha512|"
        r"sha3_224|sha3_256|sha3_384|sha3_512|blake2b|blake2s)",
        name,
    ):
        simple = name.rsplit(".", 1)[-1]
        canonical = {
            "blake2b": "BLAKE2B512",
            "blake2s": "BLAKE2S256",
        }.get(simple, simple)
        return "constant", (canonical,)
    if mode == "string" and re.fullmatch(
        r"(?:cryptography\.hazmat\.primitives\.ciphers\.)?"
        r"modes\.(?:ECB|CBC|GCM|CCM|CTR|CFB|CFB8|OFB|XTS)",
        name,
    ):
        return "constant", (name.rsplit(".", 1)[-1],)
    return None


def _direct_python_call(
    name: str,
    node: ast.Call,
    mode: str,
) -> tuple[ValueOrigin, tuple[str, ...]] | None:
    simple = name.rsplit(".", 1)[-1]
    if mode == "string" and re.fullmatch(
        r"(?:cryptography\.hazmat\.primitives\.ciphers\.)?"
        r"modes\.(?:ECB|CBC|GCM|CCM|CTR|CFB|CFB8|OFB|XTS)",
        name,
    ):
        # Mode helpers return constructed objects, but policy resolution needs
        # the constructor identity. Handling the call directly also lets a
        # class-qualified helper return ``modes.CBC(iv)`` without losing the
        # mode while resolving the helper body.
        return "constant", (simple,)
    if any(
        name == candidate or name.endswith("." + candidate)
        for candidate in (
            "AESGCM.generate_key",
            "AESCCM.generate_key",
            "Fernet.generate_key",
        )
    ):
        return "secure_random", ()
    if name in {
        "cryptography.hazmat.primitives.asymmetric.ec.generate_private_key",
        "cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key",
        "ec.generate_private_key",
        "rsa.generate_private_key",
    }:
        return "secure_random", ()
    if name in {
        "os.urandom",
        "secrets.choice",
        "secrets.token_bytes",
        "secrets.token_hex",
        "secrets.token_urlsafe",
        "secrets.randbits",
        "secrets.randbelow",
    }:
        return "secure_random", ()
    if name in {"secrets.SystemRandom", "random.SystemRandom"}:
        return "secure_random", ()
    if name in {"os.getenv", "os.environ.get"}:
        if (
            len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value is not None
        ):
            return "predictable", (repr(node.args[1].value),)
        return "runtime_input", ()
    if name in {"input", "getpass.getpass"}:
        return "runtime_input", ()
    if name == "sys.argv.pop":
        # Consuming an argument preserves the same caller-controlled origin as
        # indexing sys.argv. Programs commonly remove a password argument
        # before forwarding the remaining command-line values.
        return "runtime_input", ()
    if name in {
        "random.Random",
        "random.choice",
        "random.random",
        "random.randbytes",
        "random.randint",
        "random.randrange",
    }:
        return "non_csprng", ()
    if name in {"time.time", "time.time_ns", "datetime.now", "datetime.utcnow"}:
        return "predictable", ()
    cryptography_hashes = {
        "MD5", "SHA1", "SHA224", "SHA256", "SHA384", "SHA512",
        "SHA3_224", "SHA3_256", "SHA3_384", "SHA3_512",
    }
    hashlib_hashes = {
        "md5", "sha1", "sha224", "sha256", "sha384", "sha512",
        "sha3_224", "sha3_256", "sha3_384", "sha3_512", "blake2b", "blake2s",
    }
    if mode == "string" and (
        simple in cryptography_hashes
        or (name.startswith("hashlib.") and simple in hashlib_hashes)
    ):
        canonical = {
            "blake2b": "BLAKE2B512",
            "blake2s": "BLAKE2S256",
        }.get(simple, simple)
        return "constant", (canonical,)
    if mode == "int" and simple == "int" and node.args and isinstance(node.args[0], ast.Constant):
        try:
            return "constant", (str(int(node.args[0].value)),)
        except (TypeError, ValueError):
            return None
    if (
        mode == "origin"
        and simple == "ord"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and len(node.args[0].value) == 1
    ):
        return "constant", (str(ord(node.args[0].value)),)
    return None


def _single_argument_wrapper(expression: str) -> str | None:
    opening = expression.find("(")
    if (
        opening == -1
        or find_matching(expression, opening, "(", ")") != len(expression) - 1
    ):
        return None
    match = re.fullmatch(
        rf"(?:new\s+)?(?P<name>{_IDENTIFIER}(?:\.{_IDENTIFIER})*)"
        r"\s*\((?P<inner>.*)\)",
        expression,
        flags=re.DOTALL,
    )
    if not match:
        return None
    name = match.group("name").replace(" ", "")
    if name != "Promise.resolve" and name.rsplit(".", 1)[-1] not in {
        "from", "freeze", "FromBase64String", "FromHexString", "ToArray", "ToByte",
        "ToBase64String", "ToHexString", "AsSpan", "Create", "valueOf",
        "HashAlgorithmName", "String",
    }:
        return None
    args = _split_inline_arguments(match.group("inner"))
    return args[0] if args else None


def _origin_preserving_receiver_wrapper(expression: str) -> str | None:
    find_match = re.fullmatch(
        r"(?P<receiver>.+)\.\s*find\s*\((?s:.*)\)",
        expression,
    )
    if find_match is not None:
        return find_match.group("receiver").strip()
    match = re.fullmatch(
        r"(?P<receiver>.+)\??\.\s*(?P<method>"
        r"toString|ToString|getTime|toISOString|toUTCString|charCodeAt|getBytes|ToArray|ToCharArray|Clone|AsBytes|AsSpan|AsReadOnlySpan|AsMemory|"
        r"slice|subarray|Subarray|split|substring|Substring|padStart|padEnd|"
        r"replace|replaceAll|Replace|"
        r"trim|trimStart|trimEnd|Trim|TrimStart|TrimEnd|normalize|Normalize|"
        r"toUpperCase|toLowerCase|ToUpper|ToLower"
        r")\s*\([^)]*\)",
        expression,
        flags=re.DOTALL,
    )
    if not match:
        allowed = {
            "toString", "ToString", "getTime", "toISOString", "toUTCString", "charCodeAt", "getBytes", "ToArray", "ToCharArray", "Clone", "AsBytes",
            "AsSpan", "AsReadOnlySpan", "AsMemory", "slice", "subarray", "Subarray",
            "split", "substring", "Substring", "padStart", "padEnd", "replace",
            "replaceAll", "Replace", "trim", "trimStart", "trimEnd", "Trim",
            "TrimStart", "TrimEnd", "normalize", "Normalize", "toUpperCase",
            "toLowerCase", "ToUpper", "ToLower",
        }
        masked = mask_c_style_comments_and_literals(expression)
        candidates = list(
            re.finditer(
                rf"\??\.\s*(?P<method>{_IDENTIFIER})\s*\(",
                masked,
            )
        )
        for candidate in reversed(candidates):
            if candidate.group("method") not in allowed:
                continue
            opening = masked.find("(", candidate.start(), candidate.end())
            closing = _find_matching_masked(masked, opening, "(", ")")
            if closing == len(masked) - 1:
                return expression[: candidate.start()].strip()
        return None
    receiver = match.group("receiver").strip().removesuffix("?").rstrip()
    if (
        match.group("method") == "ToString"
        and re.fullmatch(r"(?:System\s*\.\s*)?BitConverter", receiver)
    ):
        return None
    if re.fullmatch(r"[\"'](?:\\.|[^\"'])*[\"']\s*u8", receiver, flags=re.DOTALL):
        return re.sub(r"\s*u8$", "", receiver)
    return receiver


def _typescript_hmac_digest_inputs(expression: str) -> tuple[str, ...] | None:
    """Return the controlling key input of a fluent Node HMAC digest."""

    expr = _strip_expression(expression)
    masked = mask_c_style_comments_and_literals(expr)
    constructor = re.match(
        rf"^(?:{_IDENTIFIER}\s*\.\s*)*"
        r"(?:createHmac|createNodeHmac|createIncrementalHmac)\s*\(",
        masked,
    )
    if constructor is None:
        return None
    opening = masked.find("(", constructor.start(), constructor.end())
    closing = _find_matching_masked(masked, opening, "(", ")")
    if closing == -1:
        return None
    arguments = split_arguments(expr, opening, closing)
    if len(arguments) < 2:
        return None

    inputs = [arguments[1]]
    cursor = closing + 1
    saw_digest = False
    while cursor < len(masked):
        while cursor < len(masked) and masked[cursor].isspace():
            cursor += 1
        method = re.match(r"\.\s*(?P<name>update|digest)\s*\(", masked[cursor:])
        if method is None:
            return None
        name = method.group("name")
        method_opening = cursor + masked[cursor:].find("(", 0, method.end())
        method_closing = _find_matching_masked(
            masked,
            method_opening,
            "(",
            ")",
        )
        if method_closing == -1:
            return None
        method_arguments = split_arguments(expr, method_opening, method_closing)
        if name == "update":
            if not method_arguments:
                return None
        else:
            saw_digest = True
        cursor = method_closing + 1
        if saw_digest:
            break
    if not saw_digest or masked[cursor:].strip():
        return None
    return tuple(inputs)


def _typescript_hash_digest_inputs(expression: str) -> tuple[str, ...] | None:
    """Return inputs transformed by a fluent Node hash expression."""

    expr = _strip_expression(expression)
    masked = mask_c_style_comments_and_literals(expr)
    constructor = re.match(
        rf"^(?:{_IDENTIFIER}\s*\.\s*)*createHash\s*\(",
        masked,
    )
    if constructor is None:
        return None
    opening = masked.find("(", constructor.start(), constructor.end())
    closing = _find_matching_masked(masked, opening, "(", ")")
    if closing == -1:
        return None
    cursor = closing + 1
    inputs: list[str] = []
    saw_digest = False
    while cursor < len(masked):
        while cursor < len(masked) and masked[cursor].isspace():
            cursor += 1
        method = re.match(r"\.\s*(?P<name>update|digest)\s*\(", masked[cursor:])
        if method is None:
            return None
        name = method.group("name")
        method_opening = cursor + masked[cursor:].find("(", 0, method.end())
        method_closing = _find_matching_masked(
            masked,
            method_opening,
            "(",
            ")",
        )
        if method_closing == -1:
            return None
        arguments = split_arguments(expr, method_opening, method_closing)
        if name == "update":
            if not arguments:
                return None
            inputs.append(arguments[0])
        else:
            saw_digest = True
        cursor = method_closing + 1
        if saw_digest:
            break
    if not saw_digest or not inputs or masked[cursor:].strip():
        return None
    return tuple(inputs)


def _origin_preserving_property_receiver(expression: str) -> str | None:
    match = re.fullmatch(
        r"(?P<receiver>.+)\.\s*(?:Span|Memory|Ticks)\s*",
        expression,
        flags=re.DOTALL,
    )
    return match.group("receiver").strip() if match is not None else None


def _csharp_hash_algorithm_name_property_receiver(
    expression: str,
    source: str,
    index: int,
) -> str | None:
    value = _strip_expression(expression)
    match = re.fullmatch(
        r"(?P<receiver>.+)\.\s*Name",
        value,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    receiver = _strip_expression(match.group("receiver"))
    hash_name_type = r"(?:System\.Security\.Cryptography\.)?HashAlgorithmName"
    if re.fullmatch(rf"{hash_name_type}\s*\.\s*{_IDENTIFIER}", receiver):
        return receiver

    reference = _simple_reference(receiver)
    if reference is not None:
        local_name = reference.rsplit(".", 1)[-1]
        if re.search(
            rf"\b{hash_name_type}\s+{re.escape(local_name)}\b",
            source[:index],
        ):
            return receiver
        return None

    call = _simple_call(receiver)
    if call is None or call[1]:
        return None
    helper_name = call[0]
    if re.search(
        rf"\b{hash_name_type}\s+{re.escape(helper_name)}\s*\(\s*\)",
        source,
    ):
        return receiver
    return None


def _csharp_hash_algorithm_name_constructor_argument(
    expression: str,
) -> str | None:
    match = re.fullmatch(
        r"new\s+(?:System\s*\.\s*Security\s*\.\s*Cryptography\s*\.\s*)?"
        r"HashAlgorithmName\s*\((?P<arguments>.*)\)",
        _strip_expression(expression),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    arguments = _split_inline_arguments(match.group("arguments"))
    return arguments[0] if len(arguments) == 1 else None


def _origin_preserving_call_argument(expression: str) -> str | None:
    masked = mask_c_style_comments_and_literals(expression)
    for digest_match in reversed(
        tuple(
            re.finditer(
                r"\.\s*(?:ComputeHash|HashData)\s*\(",
                masked,
            )
        )
    ):
        opening = masked.find("(", digest_match.start(), digest_match.end())
        closing = _find_matching_masked(masked, opening, "(", ")")
        if closing != len(masked) - 1 or not masked[: digest_match.start()].strip():
            continue
        arguments = split_arguments(expression, opening, closing)
        if len(arguments) == 1:
            return arguments[0]

    constructed = re.fullmatch(
        r"new\s+(?:Buffer|string|Uint8Array)\s*\((?P<arguments>.*)\)",
        expression,
        flags=re.DOTALL,
    )
    if constructed is not None:
        arguments = _split_inline_arguments(constructed.group("arguments"))
        if arguments:
            return arguments[0]

    text_codec = re.fullmatch(
        r"new\s+Text(?:Encoder|Decoder)\s*\(\s*\)\s*\.\s*"
        r"(?:encode|decode)\s*\((?P<arguments>.*)\)",
        expression,
        flags=re.DOTALL,
    )
    if text_codec is not None:
        arguments = _split_inline_arguments(text_codec.group("arguments"))
        if arguments:
            return arguments[0]

    constructed_encoding = re.fullmatch(
        r"new\s+(?:(?:System\s*\.\s*)?Text\s*\.\s*)?"
        r"(?:ASCII|UTF8|Unicode|UTF32)Encoding\s*\([^)]*\)\s*\.\s*"
        r"GetBytes\s*\((?P<arguments>.*)\)",
        expression,
        flags=re.DOTALL,
    )
    if constructed_encoding is not None:
        arguments = _split_inline_arguments(
            constructed_encoding.group("arguments")
        )
        if arguments:
            return arguments[0]

    opening = expression.find("(")
    if opening == -1 or find_matching(expression, opening, "(", ")") != len(expression) - 1:
        return None
    match = re.fullmatch(
        rf"(?P<callee>{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)"
        r"\s*\((?P<arguments>.*)\)",
        expression,
        flags=re.DOTALL,
    )
    if not match:
        return None
    callee = re.sub(r"\s+", "", match.group("callee"))
    if callee not in {
        "Buffer.from",
        "Buffer.concat",
        "BitConverter.GetBytes",
        "BitConverter.ToString",
        "byte.Parse",
        "Byte.Parse",
        "sbyte.Parse",
        "SByte.Parse",
        "short.Parse",
        "Int16.Parse",
        "ushort.Parse",
        "UInt16.Parse",
        "int.Parse",
        "Int32.Parse",
        "uint.Parse",
        "UInt32.Parse",
        "long.Parse",
        "Int64.Parse",
        "ulong.Parse",
        "UInt64.Parse",
        "Convert.ToByte",
        "Convert.ToSByte",
        "Convert.ToInt16",
        "Convert.ToUInt16",
        "Convert.ToInt32",
        "Convert.ToUInt32",
        "Convert.ToInt64",
        "Convert.ToUInt64",
        "Convert.FromBase64String",
        "Convert.FromHexString",
        "Convert.ToBase64String",
    } and not re.fullmatch(
        r"(?:[A-Za-z0-9_]+\.)*Encoding\.[A-Za-z0-9_]+\.Get(?:Bytes|String)",
        callee,
    ) and not re.fullmatch(
        r"(?:[A-Za-z0-9_]+\.)*[A-Za-z0-9_]*Encoding\."
        r"[A-Za-z0-9_]+\.Get(?:Bytes|String)",
        callee,
    ):
        return None
    arguments = _split_inline_arguments(match.group("arguments"))
    return arguments[0] if arguments else None


def _simple_call(expression: str) -> tuple[str, tuple[str, ...]] | None:
    constructed_member = re.fullmatch(
        rf"\(?\s*new\s+(?:{_IDENTIFIER}\s*\.\s*)*{_IDENTIFIER}"
        rf"\s*\([^()]*\)\s*\)?\s*\.\s*(?P<name>{_IDENTIFIER})"
        r"\s*\((?P<args>.*)\)",
        expression,
        flags=re.DOTALL,
    )
    if constructed_member is not None:
        return (
            constructed_member.group("name"),
            _split_inline_arguments(constructed_member.group("args")),
        )
    match = re.fullmatch(
        rf"(?P<name>{_IDENTIFIER}(?:\.{_IDENTIFIER})*)\s*\((?P<args>.*)\)",
        expression,
        flags=re.DOTALL,
    )
    if not match:
        return None
    return match.group("name").rsplit(".", 1)[-1], _split_inline_arguments(match.group("args"))


def _csharp_constructor_arguments(
    expression: str,
    type_name: str,
) -> tuple[str, ...] | None:
    match = re.fullmatch(
        rf"new\s+(?:{_IDENTIFIER}\s*\.\s*)*{re.escape(type_name)}\s*"
        r"\((?P<args>.*)\)",
        _strip_expression(expression),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    return _split_inline_arguments(match.group("args"))


def _c_style_call_result_property(expression: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        rf"(?P<owner>.+\))\s*\.\s*(?P<property>{_IDENTIFIER})",
        _strip_expression(expression),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    owner = _strip_expression(match.group("owner"))
    if _simple_call(owner) is None:
        return None
    return owner, match.group("property")


def _split_inline_arguments(text: str) -> tuple[str, ...]:
    synthetic = f"({text})"
    return split_arguments(synthetic, 0, len(synthetic) - 1)


def _c_style_argument_value(
    arguments: tuple[str, ...],
    position: int,
    parameter_name: str,
) -> str:
    for argument in arguments:
        match = re.fullmatch(
            rf"\s*{re.escape(parameter_name)}\s*:\s*(?P<value>.+)",
            argument,
            flags=re.DOTALL,
        )
        if match is not None:
            return match.group("value").strip()
    argument = arguments[position]
    named = re.fullmatch(
        rf"\s*{_IDENTIFIER}\s*:\s*(?P<value>.+)",
        argument,
        flags=re.DOTALL,
    )
    return (named.group("value") if named is not None else argument).strip()


def _literal_values_in_expression(expression: str, mode: str) -> set[str]:
    if mode == "int":
        stripped = expression.strip()
        quoted = re.fullmatch(
            r"[\"'](?P<value>[-+]?\d(?:[\d_]*\d)?)[\"']",
            stripped,
        )
        if quoted is not None:
            return {str(int(quoted.group("value").replace("_", "")))}
        # Literal fallback is useful for array/object initializers, but an
        # unresolved arithmetic expression such as ``choice - 1`` must not be
        # reduced to its lone literal operand. Doing so turns runtime indexes
        # into a fixed element and gives downstream policy detectors a false
        # definite algorithm or iteration count.
        identifiers = re.findall(r"[A-Za-z_$][\w$]*", stripped)
        looks_like_container = bool(
            re.search(r"[\[{].*[\]}]", stripped, flags=re.DOTALL)
        )
        if identifiers and not looks_like_container:
            return set()
        return {
            str(int(match.group(0).replace("_", "")))
            for match in re.finditer(r"(?<![\w])\d[\d_]*(?![\w])", expression)
        }
    return {
        match.group("body")
        for match in re.finditer(r"(?P<q>[\"'])(?P<body>(?:\\.|(?!\1).)*)\1", expression)
    }


_INT_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.floordiv,
    ast.Mod: operator.mod,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.BitXor: operator.xor,
}
_INT_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}


def _evaluate_c_style_int(expression: str) -> int | None:
    normalized = expression.strip().replace(">>>", ">>")
    normalized = re.sub(r"(?<=\d)[uUlL]+\b", "", normalized)
    if not re.fullmatch(r"[\dA-Fa-fxX_\s()+\-*/%<>&|^~]+", normalized):
        return None
    try:
        node = ast.parse(normalized, mode="eval").body
    except SyntaxError:
        return None
    return _evaluate_int_ast(node)


def _evaluate_int_ast(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        return None
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_int_ast(node.operand)
        function = _INT_UNARY_OPERATORS.get(type(node.op))
        if operand is None or function is None:
            return None
        value = function(operand)
        return value if abs(value) <= 2**63 else None
    if isinstance(node, ast.BinOp):
        left = _evaluate_int_ast(node.left)
        right = _evaluate_int_ast(node.right)
        function = _INT_BINARY_OPERATORS.get(type(node.op))
        if left is None or right is None or function is None:
            return None
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            return None
        value = function(left, right)
        return value if isinstance(value, int) and abs(value) <= 2**63 else None
    return None


def _derived_kdf_origin(
    expression: str,
    kdf_name: str,
    password: ResolvedValue,
    salt: ResolvedValue,
) -> ResolvedValue:
    steps = (*password.steps, *salt.steps)
    if password.origin in {"constant", "predictable", "non_csprng"}:
        return ResolvedValue(expression, "predictable", steps=steps)
    if password.origin in {
        "runtime_input",
        "secure_random",
        "derived_secure",
    }:
        return ResolvedValue(expression, "derived_secure", steps=steps)
    return ResolvedValue(
        expression,
        "unknown",
        steps=steps,
        unresolved_reason=f"{kdf_name} password or salt origin could not be resolved",
    )


def _merge_resolved(expression: str, values: Iterable[ResolvedValue]) -> ResolvedValue:
    resolved = tuple(values)
    origins = {item.origin for item in resolved}
    flattened_values = tuple(sorted({value for item in resolved for value in item.values}))
    steps = tuple(step for item in resolved for step in item.steps)
    if len(origins) == 1:
        origin = next(iter(origins))
    elif "non_csprng" in origins:
        origin = "non_csprng"
    elif origins & {"constant", "predictable"}:
        # This function combines alternative values (fallbacks, branches, or
        # assignment histories). If any reachable alternative is fixed or
        # predictable, the effective security value can be predictable even
        # when another branch comes from runtime input or a CSPRNG.
        origin = "predictable"
    elif origins <= {"secure_random", "derived_secure", "runtime_input"}:
        # Alternative branches retain the least-assured reachable origin.
        # Runtime-provided keys/passwords can be acceptable to their own
        # policies, but runtime-provided IVs and salts do not establish
        # freshness. Collapsing ``runtime_input | secure_random`` to
        # ``secure_random`` therefore hid caller-controlled salt/IV branches.
        origin = "runtime_input" if "runtime_input" in origins else "secure_random"
    else:
        origin = "unknown"
    unresolved = None if origin != "unknown" else "multiple incompatible origins"
    return ResolvedValue(expression, origin, flattened_values, steps, unresolved)


def _resolved_has_only_nullish_leaves(resolved: ResolvedValue) -> bool:
    if resolved.origin != "unknown" or not resolved.steps:
        return False
    leaves = tuple(
        _strip_expression(step.expression)
        for step in resolved.steps
        if step.action == "inspect"
    )
    return bool(leaves) and all(
        value in {"None", "null", "undefined", "''", '""'}
        for value in leaves
    )


def _resolved_without_empty_alternatives(
    resolved: ResolvedValue,
) -> ResolvedValue:
    """Drop empty/null sentinels after a caller proves the value nonempty."""

    if resolved.origin not in {"constant", "predictable", "unknown"}:
        return resolved
    nonempty_values = tuple(value for value in resolved.values if value)
    if nonempty_values and len(nonempty_values) != len(resolved.values):
        return ResolvedValue(
            expression=resolved.expression,
            origin=resolved.origin,
            values=nonempty_values,
            steps=resolved.steps,
            unresolved_reason=resolved.unresolved_reason,
        )
    nullish = {"None", "null", "undefined", "''", '""'}
    literal_leaves = tuple(
        step
        for step in resolved.steps
        if step.action == "inspect"
        and (
            _strip_expression(step.expression) in nullish
            or _string_literal(_strip_expression(step.expression)) is not None
        )
    )
    if any(
        _strip_expression(step.expression) not in nullish
        and _string_literal(_strip_expression(step.expression)) != ""
        for step in literal_leaves
    ):
        return resolved
    evidence = {step.result for step in resolved.steps}
    if "non_csprng" in evidence:
        origin: ValueOrigin = "non_csprng"
    elif "runtime_input" in evidence:
        origin = "runtime_input"
    elif evidence & {"secure_random", "derived_secure"}:
        origin = "derived_secure"
    else:
        return resolved
    return ResolvedValue(
        expression=resolved.expression,
        origin=origin,
        values=tuple(value for value in resolved.values if value),
        steps=resolved.steps,
    )


def _merge_transformed_origins(
    expression: str,
    values: Iterable[ResolvedValue],
) -> ResolvedValue:
    """Combine inputs to a deterministic, non-fallback transformation.

    Appending a fixed suffix to runtime input does not make that input
    hardcoded. This differs from a fallback expression, where a constant branch
    can become the complete effective value.
    """

    resolved = tuple(values)
    origins = {item.origin for item in resolved}
    flattened_values = tuple(
        sorted({value for item in resolved for value in item.values})
    )
    steps = tuple(step for item in resolved for step in item.steps)
    if "unknown" in origins:
        origin = "unknown"
    elif "non_csprng" in origins:
        origin = "non_csprng"
    elif "predictable" in origins:
        origin = "predictable"
    elif origins & {"secure_random", "derived_secure"}:
        origin = "derived_secure"
    elif "runtime_input" in origins:
        origin = "runtime_input"
    elif origins == {"constant"}:
        origin = "constant"
    elif len(origins) == 1:
        origin = next(iter(origins))
    else:
        origin = "unknown"
    unresolved = None if origin != "unknown" else "transformed input origin is unresolved"
    return ResolvedValue(expression, origin, flattened_values, steps, unresolved)


def _prepend_resolution(
    expression: str,
    action: str,
    resolved: ResolvedValue,
    step: ResolutionStep,
) -> ResolvedValue:
    return ResolvedValue(
        expression=expression,
        origin=resolved.origin,
        values=resolved.values,
        steps=(ResolutionStep(expression, action, resolved.origin, step.span), *resolved.steps),
        unresolved_reason=resolved.unresolved_reason,
    )


def _strip_expression(expression: str) -> str:
    value = expression.strip()
    value = re.sub(r"(?<=[\w)\]])!\s*$", "", value)
    while True:
        previous = value
        value = re.sub(r"^await\s+", "", value).strip()
        while value.startswith("(") and value.endswith(")"):
            closing = find_matching(value, 0, "(", ")")
            if closing != len(value) - 1:
                break
            value = value[1:-1].strip()
        value = re.sub(
            r"^\([A-Za-z_$][\w$<>,.?\[\]\s]*\)(?!\s*\.)\s*",
            "",
            value,
        ).strip()
        if value == previous:
            break
    value = re.sub(r"\s+as\s+\{.*\}\s*$", "", value, flags=re.DOTALL)
    value = re.sub(r"\s+as\s+[A-Za-z_$][\w$<>,.?\[\]\s]*$", "", value)
    return value.strip()


def _simple_reference(expression: str) -> str | None:
    if re.fullmatch(rf"(?:this\.)?{_IDENTIFIER}(?:\.{_IDENTIFIER})?", expression):
        return expression
    return None


def _mutation_target_reference(expression: str) -> str | None:
    value = _strip_expression(expression)
    value = re.sub(r"^(?:ref|out|in)\s+", "", value)
    receiver = _origin_preserving_receiver_wrapper(value)
    if receiver is not None:
        value = _strip_expression(receiver)
    return _simple_reference(value)


def _normalize_reference(value: str) -> str:
    normalized = _compact_reference(value)
    return normalized.rsplit(".", 1)[-1]


def _compact_reference(value: str) -> str:
    normalized = re.sub(r"\s+", "", value)
    return re.sub(r"^this\.", "", normalized)


def _string_literal(expression: str) -> str | None:
    match = re.fullmatch(r"(?P<q>[\"'])(?P<body>(?:\\.|(?!\1).)*)\1", expression, flags=re.DOTALL)
    return match.group("body") if match else None


def _c_style_has_literal_fallback(expression: str) -> bool:
    return bool(
        re.search(
            r"(?:\?\?|\|\|)\s*[\"'][^\"']+[\"']"
            r"|\?[^:]+:\s*[\"'][^\"']+[\"']",
            expression,
            flags=re.DOTALL,
        )
    )


def _parameter_name(text: str) -> str:
    cleaned = text.split("=", 1)[0].strip()
    if ":" in cleaned:
        declaration = cleaned.split(":", 1)[0].strip().lstrip("...")
        matches = re.findall(_IDENTIFIER, declaration.rstrip("?!"))
        return matches[-1] if matches else ""
    cleaned = re.sub(
        r"\b(?:ref|out|in|params|this|readonly|public|private|protected)\b",
        " ",
        cleaned,
    )
    matches = re.findall(_IDENTIFIER, cleaned)
    return matches[-1] if matches else ""


def _csharp_tuple_parameter_member_index(
    source: str,
    masked: str,
    definition: FunctionDefinition,
    parameter_name: str,
    property_name: str,
) -> int | None:
    """Map a named C# tuple member back to its callsite element."""

    opening = masked.find("(", definition.start, definition.end)
    if opening == -1:
        return None
    closing = _find_matching_masked(masked, opening, "(", ")")
    if closing == -1:
        return None
    for raw_parameter in _split_parameter_text(source[opening + 1 : closing]):
        if _parameter_name(raw_parameter) != parameter_name:
            continue
        tuple_opening = raw_parameter.find("(")
        if tuple_opening == -1:
            return None
        tuple_closing = find_matching(raw_parameter, tuple_opening, "(", ")")
        if tuple_closing == -1:
            return None
        for index, member in enumerate(
            _split_parameter_text(raw_parameter[tuple_opening + 1 : tuple_closing])
        ):
            if _parameter_name(member) == property_name:
                return index
        return None
    return None


def _csharp_tuple_expression_elements(expression: str) -> tuple[str, ...]:
    value = expression.strip()
    if not value.startswith("("):
        return ()
    masked = mask_c_style_comments_and_literals(value)
    closing = _find_matching_masked(masked, 0, "(", ")")
    if closing != len(value) - 1:
        return ()
    elements = _split_parameter_text(value[1:-1])
    if len(elements) < 2:
        return ()
    values: list[str] = []
    for element in elements:
        named = re.fullmatch(
            rf"{_IDENTIFIER}\s*:\s*(?P<value>.+)",
            element,
            flags=re.DOTALL,
        )
        values.append(
            named.group("value").strip() if named is not None else element
        )
    return tuple(values)


def _top_level_c_style_fallback(
    expression: str,
) -> tuple[str, str] | None:
    masked = mask_c_style_comments_and_literals(expression)
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(masked):
        if char in "([{":
            stack.append(char)
            continue
        if char in ")]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
            continue
        if stack:
            continue
        operator = masked[index : index + 2]
        if operator not in {"||", "??"}:
            continue
        left = expression[:index].strip()
        right = expression[index + 2 :].strip()
        return (left, right) if left and right else None
    return None


def _top_level_c_style_conditional(
    expression: str,
) -> tuple[str, str] | None:
    """Return the two value branches of a top-level ``condition ? a : b``.

    Only the value branches affect provenance. The scanner is deliberately
    lexical and bounded: strings/comments are masked, nested delimiters are
    ignored, and nested ternaries are paired before selecting the matching
    colon. TypeScript optional chaining and nullish coalescing are not treated
    as conditional operators.
    """

    masked = mask_c_style_comments_and_literals(expression)
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    question = -1
    for index, char in enumerate(masked):
        if char in "([{":
            stack.append(char)
            continue
        if char in ")]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
            continue
        if stack or char != "?":
            continue
        previous = masked[index - 1] if index else ""
        following = masked[index + 1] if index + 1 < len(masked) else ""
        if previous == "?" or following in {"?", "."}:
            continue
        question = index
        break
    if question == -1:
        return None

    stack.clear()
    nested_conditionals = 0
    for index in range(question + 1, len(masked)):
        char = masked[index]
        if char in "([{":
            stack.append(char)
            continue
        if char in ")]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
            continue
        if stack:
            continue
        if char == "?":
            previous = masked[index - 1] if index else ""
            following = masked[index + 1] if index + 1 < len(masked) else ""
            if previous != "?" and following not in {"?", "."}:
                nested_conditionals += 1
            continue
        if char != ":":
            continue
        if nested_conditionals:
            nested_conditionals -= 1
            continue
        true_expression = expression[question + 1 : index].strip()
        false_expression = expression[index + 1 :].strip()
        return (
            (true_expression, false_expression)
            if true_expression and false_expression
            else None
        )
    return None


def _c_style_nonempty_true_branch_reference(expression: str) -> str | None:
    """Return ``x`` for the bounded form ``x.length > 0 ? x : other``."""

    match = re.match(
        rf"^\s*(?P<reference>{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)"
        r"\s*\.\s*length\s*(?:>|!==|!=)\s*0\s*\?"
        rf"\s*(?P<true>{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)\s*:",
        expression,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    reference = _compact_reference(match.group("reference"))
    true_value = _compact_reference(match.group("true"))
    return reference if reference == true_value else None


def _c_style_expression_never_returns(expression: str, language: str) -> bool:
    """Recognize bounded expression forms that cannot supply a value."""

    stripped = _strip_expression(expression)
    if language == "csharp" and re.match(r"^throw\b", stripped):
        return True
    if language != "typescript":
        return False
    compact = re.sub(r"\s+", " ", stripped)
    return bool(
        re.fullmatch(
            r"\(\s*\(\s*\)\s*:\s*never\s*=>\s*\{(?s:.*\bthrow\b.*)\}"
            r"\s*\)\s*\(\s*\)",
            compact,
        )
    )


def _top_level_c_style_concatenation(
    expression: str,
) -> tuple[str, str] | None:
    masked = mask_c_style_comments_and_literals(expression)
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(masked):
        if char in "([{" :
            stack.append(char)
            continue
        if char in ")]}" :
            if stack and stack[-1] == pairs[char]:
                stack.pop()
            continue
        if stack or char != "+":
            continue
        if masked[index : index + 2] == "++" or masked[max(0, index - 1) : index + 1] == "++":
            continue
        left = expression[:index].strip()
        right = expression[index + 1 :].strip()
        if left and right:
            return left, right
    return None


def _c_style_parameter_default(
    source: str,
    masked: str,
    function: FunctionDefinition,
    parameter_name: str,
) -> str | None:
    opening = masked.find("(", function.start, function.end)
    if opening == -1:
        return None
    closing = _find_matching_masked(masked, opening, "(", ")")
    if closing == -1 or closing > function.end:
        return None
    for parameter in split_arguments(source, opening, closing):
        if _parameter_name(parameter) != parameter_name:
            continue
        masked_parameter = mask_c_style_comments_and_literals(parameter)
        stack: list[str] = []
        pairs = {")": "(", "]": "[", "}": "{"}
        for position, char in enumerate(masked_parameter):
            if char in "([{":
                stack.append(char)
            elif char in ")]}" and stack and stack[-1] == pairs[char]:
                stack.pop()
            elif (
                char == "="
                and not stack
                and not masked_parameter[position : position + 2] in {"=>", "=="}
                and (position == 0 or masked_parameter[position - 1] not in "=!<>")
            ):
                value = parameter[position + 1 :].strip()
                return value or None
    return None


def _typescript_constructor_parameter_property(
    source: str,
    masked: str,
    constructor: FunctionDefinition,
    parameter_name: str,
) -> bool:
    opening = masked.find("(", constructor.start, constructor.end)
    if opening == -1:
        return False
    closing = _find_matching_masked(masked, opening, "(", ")")
    if closing == -1 or closing > constructor.end:
        return False
    for parameter in _split_parameter_text(source[opening + 1 : closing]):
        if _parameter_name(parameter) != parameter_name:
            continue
        declaration = parameter.split("=", 1)[0].split(":", 1)[0]
        return bool(
            re.search(
                r"\b(?:public|private|protected|readonly)\b",
                declaration,
            )
        )
    return False


def _c_style_destructured_parameter(
    source: str,
    masked: str,
    function: FunctionDefinition,
    local_name: str,
) -> tuple[int, str, str | None] | None:
    """Map ``({ property: local = value })`` to its positional argument."""

    opening = masked.find("(", function.start, function.end)
    if opening == -1:
        return None
    closing = _find_matching_masked(masked, opening, "(", ")")
    if closing == -1 or closing > function.end:
        return None
    parameters = _split_parameter_text(source[opening + 1 : closing])
    for parameter_index, parameter in enumerate(parameters):
        stripped = parameter.strip()
        parameter_masked = mask_c_style_comments_and_literals(stripped)
        object_start = next(
            (
                index
                for index, char in enumerate(parameter_masked)
                if not char.isspace()
            ),
            -1,
        )
        if object_start == -1 or parameter_masked[object_start] != "{":
            continue
        object_end = _find_matching_masked(
            parameter_masked,
            object_start,
            "{",
            "}",
        )
        if object_end == -1:
            continue
        body = stripped[object_start + 1 : object_end]
        for raw_item in _split_parameter_text(body):
            item = raw_item.strip().removeprefix("...").strip()
            default: str | None = None
            masked_item = mask_c_style_comments_and_literals(item)
            equals = _top_level_assignment_equals(masked_item)
            if equals is not None:
                default = item[equals + 1 :].strip() or None
                item = item[:equals].strip()
            alias = re.fullmatch(
                rf"(?P<property>{_IDENTIFIER})"
                rf"(?:\s*:\s*(?P<local>{_IDENTIFIER}))?",
                item,
            )
            if alias is None:
                continue
            property_name = alias.group("property")
            candidate_local = alias.group("local") or property_name
            if candidate_local == local_name:
                return parameter_index, property_name, default
    return None


def _top_level_assignment_equals(masked: str) -> int | None:
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(masked):
        if char in "([{":
            stack.append(char)
            continue
        if char in ")]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
            continue
        if stack or char != "=":
            continue
        if masked[index : index + 2] in {"=>", "=="}:
            continue
        if index and masked[index - 1] in "=!<>":
            continue
        return index
    return None


def _split_parameter_text(text: str) -> tuple[str, ...]:
    if not text.strip():
        return ()

    masked = mask_c_style_comments_and_literals(text)
    starts = [0]
    stack: list[str] = []
    angle_depth = 0
    pairs = {")": "(", "]": "[", "}": "{"}

    for index, char in enumerate(masked):
        if char in "([{":
            stack.append(char)
            continue
        if char in ")]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
            continue
        if stack:
            continue
        if char == "<" and masked[index : index + 2] not in {"<=", "<<"}:
            previous = next(
                (item for item in reversed(masked[:index]) if not item.isspace()),
                "",
            )
            if previous and (previous.isalnum() or previous in "_$>]?"):
                angle_depth += 1
            continue
        if char == ">" and angle_depth and masked[index : index + 2] != ">=":
            angle_depth -= 1
            continue
        if char == "," and angle_depth == 0:
            starts.append(index + 1)

    boundaries = starts[1:] + [len(text) + 1]
    return tuple(
        text[start : boundary - 1].strip()
        for start, boundary in zip(starts, boundaries)
    )


def _python_import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                local = imported.asname or imported.name.split(".", 1)[0]
                aliases[local] = imported.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for imported in node.names:
                if imported.name == "*":
                    continue
                local = imported.asname or imported.name
                aliases[local] = f"{node.module}.{imported.name}"
    return aliases


def _canonical_python_name(name: str, aliases: dict[str, str]) -> str:
    if not name:
        return name
    head, separator, tail = name.partition(".")
    canonical = aliases.get(head)
    if canonical is None:
        return name
    return canonical + (separator + tail if separator else "")


def _python_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _python_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _python_name(node.value) + "[]"
    return ""


def _python_target_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(name for item in node.elts for name in _python_target_names(item))
    if isinstance(node, ast.Attribute):
        return (_python_name(node),)
    return ()


def _python_assignment_bindings(
    target: ast.AST,
    value: ast.AST,
) -> tuple[tuple[str, ast.AST], ...]:
    """Bind destructured names to their corresponding value component."""

    if isinstance(target, ast.Name):
        return ((target.id, value),)
    if isinstance(target, ast.Attribute):
        return ((_python_name(target), value),)
    if not isinstance(target, (ast.Tuple, ast.List)):
        return tuple()

    value_items = value.elts if isinstance(value, (ast.Tuple, ast.List)) else None
    bindings: list[tuple[str, ast.AST]] = []
    for index, item in enumerate(target.elts):
        if isinstance(item, ast.Starred):
            continue
        if value_items is not None and index < len(value_items):
            component = value_items[index]
        else:
            component = ast.Subscript(
                value=value,
                slice=ast.Constant(index),
                ctx=ast.Load(),
            )
            ast.copy_location(component, value)
        bindings.extend(_python_assignment_bindings(item, component))
    return tuple(bindings)


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", source):
        offsets.append(match.end())
    return offsets
def _python_struct_pack_is_constant(node: ast.Call) -> bool:
    if not node.args or not (
        isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, (str, bytes))
    ):
        return False

    def constant_integer(value: ast.AST) -> bool:
        if isinstance(value, ast.Constant):
            return isinstance(value.value, int) and not isinstance(value.value, bool)
        return bool(
            isinstance(value, ast.UnaryOp)
            and isinstance(value.op, (ast.UAdd, ast.USub))
            and isinstance(value.operand, ast.Constant)
            and isinstance(value.operand.value, int)
            and not isinstance(value.operand.value, bool)
        )

    for argument in node.args[1:]:
        if constant_integer(argument):
            continue
        if not isinstance(argument, ast.Starred) or not isinstance(
            argument.value,
            ast.Call,
        ):
            return False
        expanded = argument.value
        if _python_name(expanded.func).rsplit(".", 1)[-1] != "range":
            return False
        if not expanded.args or not all(constant_integer(item) for item in expanded.args):
            return False
    return not node.keywords

def _python_string_guard_target_matches(node: ast.AST, reference: str) -> bool:
    if ast.unparse(node) == reference:
        return True
    return bool(
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"casefold", "lower"}
        and ast.unparse(node.func.value) == reference
    )

def _python_condition_rejects_empty(condition: ast.AST, reference: str) -> bool:
    if isinstance(condition, ast.UnaryOp) and isinstance(condition.op, ast.Not):
        return ast.unparse(condition.operand) == reference
    if not isinstance(condition, ast.Compare) or len(condition.ops) != 1:
        return False
    left, right = condition.left, condition.comparators[0]
    if (
        isinstance(condition.ops[0], ast.Eq)
        and isinstance(left, ast.Call)
        and _python_name(left.func) == "len"
        and left.args
        and ast.unparse(left.args[0]) == reference
        and isinstance(right, ast.Constant)
        and right.value == 0
    ):
        return True
    if ast.unparse(left) != reference:
        return False
    if isinstance(condition.ops[0], ast.Is) and isinstance(right, ast.Constant):
        return right.value is None
    if isinstance(condition.ops[0], ast.Eq) and isinstance(right, ast.Constant):
        return right.value in {None, "", b""}
    return False

def _python_condition_accepts_nonempty(condition: ast.AST, reference: str) -> bool:
    if isinstance(condition, (ast.Name, ast.Attribute)):
        return ast.unparse(condition) == reference
    if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.And):
        return any(
            _python_condition_accepts_nonempty(value, reference)
            for value in condition.values
        )
    if not isinstance(condition, ast.Compare) or len(condition.ops) != 1:
        return False
    left, right = condition.left, condition.comparators[0]
    if ast.unparse(left) != reference:
        return False
    if isinstance(condition.ops[0], ast.IsNot) and isinstance(right, ast.Constant):
        return right.value is None
    if isinstance(condition.ops[0], ast.NotEq) and isinstance(right, ast.Constant):
        return right.value in {None, "", b""}
    return False

def _python_resolved_is_fixed_for_branch(resolved: ResolvedValue) -> bool:
    """Return whether one resolved value can safely select a branch."""

    # An unresolved expression can retain a literal operand in ``values``
    # (for example, ``len(data) % 16``). Only a constant origin proves a path.
    return bool(
        resolved.origin == "constant"
        and len(resolved.values) == 1
        and not any(
            step.result in {"runtime_input", "unknown"}
            for step in resolved.steps
        )
    )

def _python_resolved_has_only_fixed_values(resolved: ResolvedValue) -> bool:
    """Return whether a finite set contains no unresolved runtime branch."""

    return bool(
        resolved.values
        and not any(
            step.result in {"runtime_input", "unknown"}
            for step in resolved.steps
        )
    )

def _python_restrict_resolved_values(
    resolved: ResolvedValue,
    allowed: tuple[str, ...],
) -> ResolvedValue:
    accepted = tuple(
        dict.fromkeys(value for value in resolved.values if value in allowed)
    )
    if accepted:
        return ResolvedValue(
            resolved.expression,
            resolved.origin,
            accepted,
            resolved.steps,
        )
    return ResolvedValue(
        resolved.expression,
        "unknown",
        steps=resolved.steps,
        unresolved_reason=(
            "all concrete values are rejected by the fail-closed whitelist"
        ),
    )

def _python_guard_allowed_integer_values(
    condition: ast.AST,
    variable: str,
) -> tuple[int, ...]:
    """Extract the allowlist from ``if value not in {...}: stop``."""

    if not isinstance(condition, ast.Compare) or len(condition.ops) != 1:
        return ()
    if not isinstance(condition.ops[0], ast.NotIn):
        return ()
    if not _python_guard_target_matches(condition.left, variable):
        return ()
    return _python_integer_container(condition.comparators[0])

def _python_statement_assigns_reference(
    statement: ast.stmt,
    reference: str,
) -> bool:
    if isinstance(statement, ast.Assign):
        targets: tuple[ast.AST, ...] = tuple(statement.targets)
    elif isinstance(statement, ast.AnnAssign):
        targets = (statement.target,)
    elif isinstance(statement, ast.AugAssign):
        targets = (statement.target,)
    else:
        return False
    return any(ast.unparse(target) == reference for target in targets)

def _typescript_template_expression_mask(
    source: str,
    masked_source: str,
) -> str:
    """Expose executable ``${...}`` expressions inside template literals."""

    result = ["\n" if char == "\n" else " " for char in source]
    index = 0
    while index < len(source):
        if source[index] != "`" or masked_source[index] != "`":
            index += 1
            continue
        cursor = index + 1
        while cursor < len(source):
            if source[cursor] == "\\":
                cursor += 2
                continue
            if source[cursor] == "`":
                cursor += 1
                break
            if not source.startswith("${", cursor):
                cursor += 1
                continue
            closing = _csharp_interpolation_expression_end(source, cursor + 1)
            if closing == -1:
                cursor += 2
                continue
            expression = source[cursor + 2 : closing]
            masked_expression = mask_c_style_comments_and_literals(expression)
            result[cursor + 2 : closing] = masked_expression
            cursor = closing + 1
        index = cursor
    return "".join(result)

def _c_style_conditional_parts(
    expression: str,
) -> tuple[str, str, str] | None:
    """Split one top-level ``condition ? left : right`` expression."""

    masked = mask_c_style_comments_and_literals(expression)
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    question = -1
    nested_questions = 0
    for index, char in enumerate(masked):
        if char in "([{":
            stack.append(char)
            continue
        if char in ")]}" and stack and stack[-1] == pairs[char]:
            stack.pop()
            continue
        if stack:
            continue
        if char == "?":
            if (
                masked[index : index + 2] in {"??", "?."}
                or (index > 0 and masked[index - 1] == "?")
            ):
                continue
            if question == -1:
                question = index
            else:
                nested_questions += 1
            continue
        if char == ":" and question != -1:
            if nested_questions:
                nested_questions -= 1
                continue
            condition = expression[:question].strip()
            when_true = expression[question + 1 : index].strip()
            when_false = expression[index + 1 :].strip()
            if condition and when_true and when_false:
                return condition, when_true, when_false
            return None
    return None

def _typescript_identity_conditional_expression(
    condition: str,
    when_true: str,
    when_false: str,
) -> str | None:
    """Collapse a strict-equality ternary that always returns one operand."""

    comparison = re.fullmatch(
        r"(?P<left>.+?)\s*(?P<operator>===|!==)\s*(?P<right>.+)",
        _strip_expression(condition),
        flags=re.DOTALL,
    )
    if comparison is None:
        return None
    left = _strip_expression(comparison.group("left"))
    right = _strip_expression(comparison.group("right"))
    true_value = _strip_expression(when_true)
    false_value = _strip_expression(when_false)
    operator = comparison.group("operator")
    if operator == "===":
        if true_value == right and false_value == left:
            return left
        if true_value == left and false_value == right:
            return right
    else:
        if true_value == left and false_value == right:
            return left
        if true_value == right and false_value == left:
            return right
    return None

def _c_style_statement_bounds(
    masked: str,
    start: int,
    language: Language,
) -> tuple[int, int]:
    """Return the half-open range of one braced or single C-style statement."""

    cursor = start
    while cursor < len(masked) and masked[cursor].isspace():
        cursor += 1
    if cursor >= len(masked):
        return -1, -1
    if masked[cursor] == "{":
        closing = _find_matching_masked(masked, cursor, "{", "}")
        return (cursor, closing + 1) if closing != -1 else (-1, -1)
    if re.match(r"if\b", masked[cursor:]):
        opening = masked.find("(", cursor)
        if opening == -1:
            return -1, -1
        closing = _find_matching_masked(masked, opening, "(", ")")
        if closing == -1:
            return -1, -1
        _, then_end = _c_style_statement_bounds(masked, closing + 1, language)
        if then_end == -1:
            return -1, -1
        after_then = then_end
        while after_then < len(masked) and masked[after_then].isspace():
            after_then += 1
        if masked.startswith("else", after_then):
            _, else_end = _c_style_statement_bounds(
                masked,
                after_then + len("else"),
                language,
            )
            if else_end != -1:
                return cursor, else_end
        return cursor, then_end
    end = _next_c_style_statement_end_masked(masked, cursor, language)
    if end < cursor:
        return -1, -1
    return cursor, min(len(masked), end + 1)

def _c_style_if_chain_overwrites_or_terminates(
    analyzer: CStyleAnalyzer,
    reference: str,
    start: int,
    end: int,
) -> bool:
    """Prove a simple if/else chain replaces a baseline on every live path."""

    compact_reference = r"\s*\.\s*".join(
        re.escape(part)
        for part in re.sub(r"\s+", "", reference).split(".")
    )

    def first_statement_satisfies(body_start: int, body_end: int) -> bool:
        cursor = body_start
        if cursor < len(analyzer.masked) and analyzer.masked[cursor] == "{":
            cursor += 1
        while cursor < body_end and analyzer.masked[cursor].isspace():
            cursor += 1
        statement_end = _next_c_style_statement_end_masked(
            analyzer.masked,
            cursor,
            analyzer.language,
        )
        if statement_end == -1 or statement_end > body_end:
            statement_end = body_end
        statement = analyzer.source[cursor:statement_end]
        return bool(
            re.match(
                rf"\s*(?:{compact_reference}\s*=|throw\b|return\b)",
                statement,
                flags=re.DOTALL,
            )
        )

    masked = analyzer.masked
    for match in re.finditer(r"\bif\s*\(", masked[start:end]):
        opening = start + masked[start:end].find(
            "(",
            match.start(),
            match.end(),
        )
        closing = _find_matching_masked(masked, opening, "(", ")")
        if closing == -1 or closing >= end:
            continue
        branch_start, branch_end = _c_style_statement_bounds(
            masked,
            closing + 1,
            analyzer.language,
        )
        if branch_start == -1 or branch_end > end:
            continue
        branches = [(branch_start, branch_end)]
        cursor = branch_end
        final_else = False
        while cursor < end:
            while cursor < end and masked[cursor].isspace():
                cursor += 1
            if not masked.startswith("else", cursor):
                break
            cursor += len("else")
            while cursor < end and masked[cursor].isspace():
                cursor += 1
            if masked.startswith("if", cursor):
                conditional_open = masked.find("(", cursor, end)
                conditional_close = _find_matching_masked(
                    masked,
                    conditional_open,
                    "(",
                    ")",
                )
                if conditional_open == -1 or conditional_close == -1:
                    break
                branch_start, branch_end = _c_style_statement_bounds(
                    masked,
                    conditional_close + 1,
                    analyzer.language,
                )
            else:
                branch_start, branch_end = _c_style_statement_bounds(
                    masked,
                    cursor,
                    analyzer.language,
                )
                final_else = True
            if branch_start == -1 or branch_end == -1 or branch_end > end:
                break
            branches.append((branch_start, branch_end))
            cursor = branch_end
            if final_else:
                break
        if final_else and branches and all(
            first_statement_satisfies(branch_start, branch_end)
            for branch_start, branch_end in branches
        ):
            return True
    return False

def _c_style_direct_path_is_reachable_impl(
    analyzer: CStyleAnalyzer,
    index: int,
) -> bool:
    """Evaluate a direct path while caller reachability is suspended."""

    masked = analyzer.masked
    for match in re.finditer(r"\bswitch\s*\(", masked):
        opening = masked.find("(", match.start(), match.end())
        closing = _find_matching_masked(masked, opening, "(", ")")
        if closing == -1:
            continue
        body_open = closing + 1
        while body_open < len(masked) and masked[body_open].isspace():
            body_open += 1
        if body_open >= len(masked) or masked[body_open] != "{":
            continue
        body_close = _find_matching_masked(masked, body_open, "{", "}")
        if body_close == -1 or not (body_open < index < body_close):
            continue

        selector = analyzer.source[opening + 1 : closing]
        resolved = analyzer.resolve_strings(selector, match.start())
        if resolved.origin == "unknown":
            continue
        selected = {
            _normalize_control_value(value) for value in resolved.values if value
        }
        if len(selected) != 1:
            continue
        selected_value = next(iter(selected))

        labels: list[tuple[int, int, str | None]] = []
        for label_match in re.finditer(
            r"\b(?P<kind>case|default)\b(?P<label>[^:]*)\:",
            masked[body_open + 1 : body_close],
        ):
            absolute_start = body_open + 1 + label_match.start()
            absolute_end = body_open + 1 + label_match.end()
            label: str | None = None
            if label_match.group("kind") == "case":
                source_label = analyzer.source[
                    body_open + 1 + label_match.start("label") :
                    body_open + 1 + label_match.end("label")
                ].strip()
                direct = _direct_c_style_value(
                    source_label,
                    analyzer.language,
                    "string",
                )
                if direct is not None and direct[1]:
                    label = _normalize_control_value(direct[1][0])
                else:
                    label = _normalize_control_value(source_label)
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
        closing = _find_matching_masked(masked, opening, "(", ")")
        if closing == -1:
            continue
        body_open, body_close = _c_style_statement_bounds(
            masked,
            closing + 1,
            analyzer.language,
        )
        if body_open == -1:
            continue
        in_then = body_open <= index < body_close
        in_else = False
        cursor = body_close
        while cursor < len(masked) and masked[cursor].isspace():
            cursor += 1
        if masked.startswith("else", cursor):
            cursor += len("else")
            else_open, else_close = _c_style_statement_bounds(
                masked,
                cursor,
                analyzer.language,
            )
            in_else = else_open != -1 and else_open <= index < else_close
        if not in_then and not in_else:
            continue
        condition = analyzer.source[opening + 1 : closing]
        condition_value = _evaluate_c_style_string_condition(
            analyzer,
            condition,
            match.start(),
        )
        if condition_value is None:
            continue
        if (in_then and not condition_value) or (in_else and condition_value):
            return False
    return True

def _c_style_direct_path_is_definitely_reached(
    analyzer: CStyleAnalyzer,
    index: int,
) -> bool:
    """Prove that resolved conditionals enclosing ``index`` select its arm.

    This is intentionally narrower than reachability. It is used to decide
    whether a conditional assignment definitely overwrites an earlier value.
    An unresolved branch, loop body, or ambiguous switch keeps the earlier
    assignment as an alternative.
    """

    masked = analyzer.masked
    enclosed_by_resolved_branch = False
    for match in re.finditer(r"\b(?:for|while)\s*\(", masked):
        opening = masked.find("(", match.start(), match.end())
        closing = _find_matching_masked(masked, opening, "(", ")")
        if closing == -1:
            continue
        body_open, body_close = _c_style_statement_bounds(
            masked,
            closing + 1,
            analyzer.language,
        )
        if body_open != -1 and body_open <= index < body_close:
            return False

    for match in re.finditer(r"\bswitch\s*\(", masked):
        opening = masked.find("(", match.start(), match.end())
        closing = _find_matching_masked(masked, opening, "(", ")")
        if closing == -1:
            continue
        body_open = closing + 1
        while body_open < len(masked) and masked[body_open].isspace():
            body_open += 1
        if body_open >= len(masked) or masked[body_open] != "{":
            continue
        body_close = _find_matching_masked(masked, body_open, "{", "}")
        if body_close == -1 or not (body_open < index < body_close):
            continue
        selector = analyzer.source[opening + 1 : closing]
        resolved = analyzer.resolve_strings(selector, match.start())
        selected = {
            _normalize_control_value(value)
            for value in resolved.values
            if value
        }
        if resolved.origin == "unknown" or len(selected) != 1:
            return False
        if not _c_style_direct_path_is_reachable_impl(analyzer, index):
            return False
        enclosed_by_resolved_branch = True

    for match in re.finditer(r"\bif\s*\(", masked):
        opening = masked.find("(", match.start(), match.end())
        closing = _find_matching_masked(masked, opening, "(", ")")
        if closing == -1:
            continue
        body_open, body_close = _c_style_statement_bounds(
            masked,
            closing + 1,
            analyzer.language,
        )
        if body_open == -1:
            continue
        in_then = body_open <= index < body_close
        in_else = False
        cursor = body_close
        while cursor < len(masked) and masked[cursor].isspace():
            cursor += 1
        if masked.startswith("else", cursor):
            cursor += len("else")
            else_open, else_close = _c_style_statement_bounds(
                masked,
                cursor,
                analyzer.language,
            )
            in_else = else_open != -1 and else_open <= index < else_close
        if not in_then and not in_else:
            continue
        condition = analyzer.source[opening + 1 : closing]
        condition_value = _evaluate_c_style_string_condition(
            analyzer,
            condition,
            match.start(),
        )
        if in_then:
            if condition_value is not True:
                return False
            enclosed_by_resolved_branch = True
            continue
        if condition_value is not False:
            return False
        enclosed_by_resolved_branch = True

    return enclosed_by_resolved_branch

def _top_level_c_style_arithmetic_operands(
    expression: str,
) -> tuple[str, str] | None:
    """Split a deterministic top-level numeric/bitwise expression.

    This is provenance analysis rather than numeric evaluation: arithmetic
    preserves whether its inputs are fixed, runtime-provided, or random.
    Logical fallbacks and string concatenation are handled separately.
    """

    value = _strip_expression(expression)
    masked = mask_c_style_comments_and_literals(value)
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    operators = (">>>", "<<", ">>", "*", "/", "%", "-", "&", "^", "|")
    for index, char in enumerate(masked):
        if char in "([{":
            stack.append(char)
            continue
        if char in ")]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
            continue
        if stack:
            continue
        operator = next(
            (item for item in operators if masked.startswith(item, index)),
            None,
        )
        if operator is None:
            continue
        if operator in {"-", "*", "&"} and not value[:index].strip():
            continue
        if operator == "|" and masked.startswith("||", index):
            continue
        if operator == "&" and masked.startswith("&&", index):
            continue
        left = value[:index].strip()
        right = value[index + len(operator) :].strip()
        if left and right:
            return left, right
    return None

def _csharp_first_concatenated_conversion_input(expression: str) -> str | None:
    match = re.fullmatch(
        r"(?:System\s*\.\s*)?Convert\s*\.\s*"
        r"(?:FromBase64String|FromHexString)\s*\((?P<argument>.*)\)",
        _strip_expression(expression),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    concatenation = _top_level_c_style_concatenation(match.group("argument"))
    return concatenation[0] if concatenation is not None else None

def _csharp_conversion_input_reference(expression: str) -> str | None:
    match = re.fullmatch(
        r"(?:System\s*\.\s*)?Convert\s*\.\s*"
        r"(?:FromBase64String|FromHexString)\s*\((?P<argument>.*)\)",
        _strip_expression(expression),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    return _simple_reference(_strip_expression(match.group("argument")))

def _evaluate_python_string_ast(node: ast.AST) -> str | None:
    """Evaluate a bounded literal-only Python string-format expression."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _evaluate_python_string_ast(node.left)
        right = _evaluate_python_string_ast(node.right)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if isinstance(node.right, ast.Constant) and isinstance(
            node.right.value, int
        ):
            value = _evaluate_python_string_ast(node.left)
            return (
                value * node.right.value
                if value is not None and node.right.value >= 0
                else None
            )
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, int):
            value = _evaluate_python_string_ast(node.right)
            return (
                value * node.left.value
                if value is not None and node.left.value >= 0
                else None
            )
    if not (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mod)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.left.value, str)
    ):
        return None
    operands = node.right.elts if isinstance(node.right, (ast.Tuple, ast.List)) else (node.right,)
    values: list[object] = []
    for operand in operands:
        if not isinstance(operand, ast.Constant) or not isinstance(
            operand.value,
            (str, bytes, int, float),
        ):
            return None
        values.append(operand.value)
    argument: object = tuple(values) if isinstance(node.right, (ast.Tuple, ast.List)) else values[0]
    try:
        rendered = node.left.value % argument
    except (TypeError, ValueError):
        return None
    return rendered if isinstance(rendered, str) else None

def _resolved_single_int(value: ResolvedValue) -> int | None:
    parsed = {
        parsed
        for item in value.values
        if (parsed := _parse_python_int_value(item)) is not None
    }
    return next(iter(parsed)) if len(parsed) == 1 else None

def _parse_python_int_value(value: str) -> int | None:
    try:
        return int(value.replace("_", ""), 10)
    except ValueError:
        return None

def _combine_resolved_int_operands(
    operation: ast.operator,
    left: ResolvedValue,
    right: ResolvedValue,
) -> tuple[str, ...]:
    function = _INT_BINARY_OPERATORS.get(type(operation))
    if function is None:
        return ()
    left_values = tuple(
        parsed
        for value in left.values
        if (parsed := _parse_python_int_value(value)) is not None
    )
    right_values = tuple(
        parsed
        for value in right.values
        if (parsed := _parse_python_int_value(value)) is not None
    )
    combined: set[int] = set()
    for left_value in left_values:
        for right_value in right_values:
            if (
                isinstance(operation, (ast.Div, ast.FloorDiv, ast.Mod))
                and right_value == 0
            ):
                continue
            try:
                value = function(left_value, right_value)
            except (ArithmeticError, ValueError):
                continue
            if isinstance(value, int) and abs(value) <= 2**63:
                combined.add(value)
    return tuple(str(value) for value in sorted(combined))

def _merge_entropy_preserving_origins(
    expression: str,
    values: Iterable[ResolvedValue],
) -> ResolvedValue:
    """Combine inputs to a digest that incorporates every supplied value."""

    resolved = tuple(values)
    origins = {item.origin for item in resolved}
    flattened_values = tuple(
        sorted({value for item in resolved for value in item.values})
    )
    steps = tuple(step for item in resolved for step in item.steps)
    if origins & {"secure_random", "derived_secure"}:
        origin: ValueOrigin = "derived_secure"
    elif "unknown" in origins:
        origin = "unknown"
    elif "non_csprng" in origins:
        origin = "non_csprng"
    elif origins & {"constant", "predictable"}:
        origin = "predictable"
    elif "runtime_input" in origins:
        origin = "runtime_input"
    elif len(origins) == 1:
        origin = next(iter(origins))
    else:
        origin = "unknown"
    unresolved = None if origin != "unknown" else "digest input origin is unresolved"
    return ResolvedValue(expression, origin, flattened_values, steps, unresolved)

def _append_resolution_step(
    resolved: ResolvedValue,
    step: ResolutionStep,
) -> ResolvedValue:
    return ResolvedValue(
        expression=resolved.expression,
        origin=resolved.origin,
        values=resolved.values,
        steps=(*resolved.steps, step),
        unresolved_reason=resolved.unresolved_reason,
    )

def _csharp_method_call_receiver(callee: str) -> str:
    compact = re.sub(r"\s+", "", callee)
    return compact.rsplit(".", 1)[0] if "." in compact else ""
