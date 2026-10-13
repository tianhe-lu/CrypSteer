"""Task-neutral, source-provable Python call-site type checks.

The task validators do not predict runtime values.  They do reject a local
helper call when the source itself proves that bytes reach a parameter on
which the helper unconditionally calls ``str.encode``.  Keeping that bounded
interprocedural check here avoids reimplementing it in digest, MAC, and KDF
validators.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable


def bytes_passed_to_encoded_parameter(
    source: str,
    *,
    function_names: Iterable[str] | None = None,
) -> tuple[str, str] | None:
    """Return ``(function, parameter)`` for a definite bytes/encode conflict."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    parents = _parents(tree)
    allowed = set(function_names) if function_names is not None else None
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (allowed is None or node.name in allowed)
    }
    return_kinds = _function_return_kinds(functions, parents)

    encoded_parameters: dict[str, set[str]] = {}
    for name, function in functions.items():
        parameters = {
            argument.arg
            for argument in _parameters(function)
        }
        encoded_parameters[name] = set()
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "encode"
                and _containing_function(call, parents) is function
            ):
                continue
            for parameter in parameters & {
                node.id
                for node in ast.walk(call.func.value)
                if isinstance(node, ast.Name)
            }:
                if not _guarded_as_text(call, parameter, parents):
                    encoded_parameters[name].add(parameter)

    changed = True
    while changed:
        changed = False
        for caller_name, caller in functions.items():
            caller_parameters = {
                argument.arg
                for argument in _parameters(caller)
            }
            for call in ast.walk(caller):
                if not isinstance(call, ast.Call):
                    continue
                callee_name = _local_function_name(call.func)
                callee = functions.get(callee_name)
                if (
                    callee is None
                    or _containing_function(call, parents) is not caller
                ):
                    continue
                for parameter, position in _parameter_bindings(callee):
                    if parameter.arg not in encoded_parameters[callee_name]:
                        continue
                    argument = _bound_argument(call, parameter.arg, position)
                    if (
                        isinstance(argument, ast.Name)
                        and argument.id in caller_parameters
                        and argument.id not in encoded_parameters[caller_name]
                    ):
                        encoded_parameters[caller_name].add(argument.id)
                        changed = True

    for name, function in functions.items():
        parameter_bindings = _parameter_bindings(function)
        positions = {
            parameter.arg: position
            for parameter, position in parameter_bindings
            if parameter.arg in encoded_parameters[name]
        }
        if not positions:
            continue
        call_kinds: dict[str, list[str | None]] = {
            parameter: [] for parameter in positions
        }
        for call in ast.walk(tree):
            if (
                not isinstance(call, ast.Call)
                or _local_function_name(call.func) != name
                or _containing_function(call, parents) is function
            ):
                continue
            scope = _containing_function(call, parents)
            assignments = _assignments_before(tree, parents, scope, call)
            for parameter, position in positions.items():
                argument = _bound_argument(call, parameter, position)
                if (
                    isinstance(argument, ast.Name)
                    and _guarded_as_text(call, argument.id, parents)
                ):
                    call_kinds[parameter].append("text")
                    continue
                possible_values = (
                    _all_assignments_to_name_before(
                        tree,
                        parents,
                        scope,
                        call,
                        argument.id,
                    )
                    if isinstance(argument, ast.Name)
                    else ()
                )
                if possible_values:
                    call_kinds[parameter].extend(
                        _expression_kind(
                            value,
                            assignments,
                            functions,
                            return_kinds,
                        )
                        for value in possible_values
                    )
                else:
                    call_kinds[parameter].append(
                        _expression_kind(
                            argument,
                            assignments,
                            functions,
                            return_kinds,
                        )
                    )
        for parameter, kinds in call_kinds.items():
            # A task is valid when at least one complete route supplies a
            # usable argument.  Reject only when every observed call to this
            # local helper is source-provably bytes; an alternate text or
            # unresolved call can supply the task-valid route.
            if kinds and all(kind == "bytes" for kind in kinds):
                return name, parameter
    return None


def source_proven_bytes_encoded_as_text(source: str) -> str | None:
    """Return the receiver of a definite bytes ``.encode()`` call.

    Python accepts this source syntactically, but bytes objects do not provide
    ``str.encode``.  The check is deliberately evidence-bounded: parameters
    and unresolved values remain acceptable, while byte literals, byte-returning
    calls, and local assignments to those values are source-provable failures.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    parents = _parents(tree)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return_kinds = _function_return_kinds(functions, parents)
    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "encode"
        ):
            continue
        scope = _containing_function(call, parents)
        assignments = _assignments_before(tree, parents, scope, call)
        if (
            _expression_kind(
                call.func.value,
                assignments,
                functions,
                return_kinds,
            )
            == "bytes"
        ):
            return ast.unparse(call.func.value)
    return None


def text_passed_to_binary_parameter(
    source: str,
    *,
    function_names: Iterable[str] | None = None,
) -> tuple[str, str] | None:
    """Return a definite text-to-bytes-only helper-call conflict.

    This deliberately covers only ``update`` calls on cryptography cipher and
    padding contexts.  Those APIs require bytes-like input, while a source
    literal or a value returned by ``str.decode`` is definitely text.  The
    small wrapper propagation mirrors :func:`bytes_passed_to_encoded_parameter`
    without trying to infer general Python runtime types.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    parents = _parents(tree)
    allowed = set(function_names) if function_names is not None else None
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (allowed is None or node.name in allowed)
    }
    return_kinds = _function_return_kinds(functions, parents)

    binary_parameters: dict[str, set[str]] = {}
    for name, function in functions.items():
        parameters = {
            argument.arg
            for argument in _parameters(function)
        }
        context_names = {
            target.id
            for assignment in ast.walk(function)
            if isinstance(assignment, (ast.Assign, ast.AnnAssign))
            and _containing_function(assignment, parents) is function
            for target in (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else (assignment.target,)
            )
            if isinstance(target, ast.Name)
            and isinstance(assignment.value, ast.Call)
            and isinstance(assignment.value.func, ast.Attribute)
            and assignment.value.func.attr
            in {"padder", "unpadder", "encryptor", "decryptor"}
        }
        binary_parameters[name] = set()
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "update"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in context_names
                and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id in parameters
                and _containing_function(call, parents) is function
            ):
                continue
            parameter = call.args[0].id
            # A later local assignment shadows the parameter.  The update then
            # consumes the reassigned value, not the caller's original value.
            if parameter not in _assignments_before(
                tree, parents, function, call
            ):
                binary_parameters[name].add(parameter)

    changed = True
    while changed:
        changed = False
        for caller_name, caller in functions.items():
            caller_parameters = {
                argument.arg
                for argument in _parameters(caller)
            }
            for call in ast.walk(caller):
                if not isinstance(call, ast.Call):
                    continue
                callee_name = _local_function_name(call.func)
                callee = functions.get(callee_name)
                if callee is None or _containing_function(call, parents) is not caller:
                    continue
                for parameter, position in _parameter_bindings(callee):
                    if parameter.arg not in binary_parameters[callee_name]:
                        continue
                    argument = _bound_argument(call, parameter.arg, position)
                    if (
                        isinstance(argument, ast.Name)
                        and argument.id in caller_parameters
                        and argument.id not in binary_parameters[caller_name]
                    ):
                        binary_parameters[caller_name].add(argument.id)
                        changed = True

    for name, function in functions.items():
        parameter_bindings = _parameter_bindings(function)
        positions = {
            parameter.arg: position
            for parameter, position in parameter_bindings
            if parameter.arg in binary_parameters[name]
        }
        if not positions:
            continue
        call_kinds: dict[str, list[str | None]] = {
            parameter: [] for parameter in positions
        }
        for call in ast.walk(tree):
            if (
                not isinstance(call, ast.Call)
                or _local_function_name(call.func) != name
                or _containing_function(call, parents) is function
            ):
                continue
            scope = _containing_function(call, parents)
            assignments = _assignments_before(tree, parents, scope, call)
            for parameter, position in positions.items():
                argument = _bound_argument(call, parameter, position)
                call_kinds[parameter].append(
                    _expression_kind(
                        argument,
                        assignments,
                        functions,
                        return_kinds,
                    )
                )
        for parameter, kinds in call_kinds.items():
            if kinds and all(kind == "text" for kind in kinds):
                return name, parameter
    return None


def _function_return_kinds(
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    parents: dict[ast.AST, ast.AST],
) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for _ in range(2):
        for name, function in functions.items():
            assignments = _assignments_before(
                function,
                parents,
                function,
                None,
            )
            observed = {
                (
                    "bytes"
                    if _binary_file_read(node.value, function)
                    else _expression_kind(node.value, assignments, functions, kinds)
                )
                for node in ast.walk(function)
                if isinstance(node, ast.Return)
                and node.value is not None
                and _containing_function(node, parents) is function
            }
            observed.discard(None)
            if len(observed) == 1:
                kinds[name] = observed.pop()  # type: ignore[assignment]
    return kinds


def _binary_file_read(expression: ast.AST, function: ast.AST) -> bool:
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr == "read"
    ):
        return False
    return any(
        isinstance(call, ast.Call)
        and _qualified_name(call.func) == "open"
        and any(
            isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and "b" in argument.value
            for argument in call.args[1:]
        )
        for call in ast.walk(function)
    )


def _expression_kind(
    expression: ast.AST | None,
    assignments: dict[str, ast.AST],
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    return_kinds: dict[str, str],
    seen: frozenset[str] = frozenset(),
) -> str | None:
    if isinstance(expression, ast.Constant):
        if isinstance(expression.value, bytes):
            return "bytes"
        if isinstance(expression.value, str):
            return "text"
    if isinstance(expression, ast.Name) and expression.id not in seen:
        return _expression_kind(
            assignments.get(expression.id),
            assignments,
            functions,
            return_kinds,
            seen | {expression.id},
        )
    if isinstance(expression, ast.Subscript):
        return _expression_kind(
            expression.value,
            assignments,
            functions,
            return_kinds,
            seen,
        )
    if isinstance(expression, ast.JoinedStr):
        return "text"
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = _expression_kind(
            expression.left, assignments, functions, return_kinds, seen
        )
        right = _expression_kind(
            expression.right, assignments, functions, return_kinds, seen
        )
        return left if left == right else None
    if not isinstance(expression, ast.Call):
        return None
    name = _qualified_name(expression.func)
    simple = name.rsplit(".", 1)[-1]
    if isinstance(expression.func, ast.Attribute) and simple == "encode":
        return "bytes"
    if isinstance(expression.func, ast.Attribute) and simple == "decode":
        return "text"
    if simple in {
        "urandom",
        "token_bytes",
        "get_random_bytes",
        "b64decode",
        "urlsafe_b64decode",
        "unhexlify",
        "digest",
    }:
        return "bytes"
    if simple in {"input", "getpass"}:
        return "text"
    if simple in return_kinds:
        return return_kinds[simple]
    return None


def _bound_argument(
    call: ast.Call,
    parameter: str,
    position: int | None,
) -> ast.AST | None:
    keyword = next(
        (keyword.value for keyword in call.keywords if keyword.arg == parameter),
        None,
    )
    if keyword is not None:
        return keyword
    if position is not None and position < len(call.args):
        return call.args[position]
    return None


def _parameters(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.arg, ...]:
    """Return every explicitly named parameter, including keyword-only ones."""

    return tuple(
        parameter for parameter, _position in _parameter_bindings(function)
    )


def _parameter_bindings(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[tuple[ast.arg, int | None], ...]:
    """Map positional parameters to indices and keyword-only parameters to none."""

    positional = (*function.args.posonlyargs, *function.args.args)
    return (
        *((parameter, index) for index, parameter in enumerate(positional)),
        *((parameter, None) for parameter in function.args.kwonlyargs),
    )


def _assignments_before(
    root: ast.AST,
    parents: dict[ast.AST, ast.AST],
    scope: ast.FunctionDef | ast.AsyncFunctionDef | None,
    before: ast.AST | None,
) -> dict[str, ast.AST]:
    boundary = (
        (getattr(before, "lineno", 10**9), getattr(before, "col_offset", 10**9))
        if before is not None
        else (10**9, 10**9)
    )
    values: dict[str, ast.AST] = {}
    assignments = sorted(
        (
            node
            for node in ast.walk(root)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and _containing_function(node, parents) is scope
            and (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
            < boundary
        ),
        key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)),
    )
    for node in assignments:
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = node.value
    if scope is not None:
        module_values = _assignments_before(root, parents, None, before)
        module_values.update(values)
        return module_values
    return values


def _all_assignments_to_name_before(
    root: ast.AST,
    parents: dict[ast.AST, ast.AST],
    scope: ast.FunctionDef | ast.AsyncFunctionDef | None,
    before: ast.AST,
    name: str,
) -> tuple[ast.AST, ...]:
    boundary = (getattr(before, "lineno", 10**9), getattr(before, "col_offset", 10**9))
    return tuple(
        node.value
        for node in ast.walk(root)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _containing_function(node, parents) is scope
        and (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)) < boundary
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
        )
    )


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _containing_function(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _guarded_as_text(
    node: ast.AST,
    name: str,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    child = node
    current = parents.get(child)
    while current is not None and not isinstance(
        current, (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        if isinstance(current, ast.If) and child in current.body:
            for call in ast.walk(current.test):
                if not (
                    isinstance(call, ast.Call)
                    and _qualified_name(call.func) == "isinstance"
                    and len(call.args) >= 2
                ):
                    continue
                if (
                    isinstance(call.args[0], ast.Name)
                    and call.args[0].id == name
                    and _qualified_name(call.args[1]) == "str"
                ):
                    return True
        child = current
        current = parents.get(current)
    return False


def _simple_name(node: ast.AST) -> str:
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


def _local_function_name(node: ast.AST) -> str:
    """Return a bare local-call name, never an object's method name."""

    return node.id if isinstance(node, ast.Name) else ""


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


__all__ = (
    "bytes_passed_to_encoded_parameter",
    "source_proven_bytes_encoded_as_text",
    "text_passed_to_binary_parameter",
)
