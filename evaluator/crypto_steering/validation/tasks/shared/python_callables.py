"""Bounded provenance for Python callables stored in local or object state."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Set


def callable_aliases(
    source: str,
    import_aliases: Mapping[str, str],
    allowed_targets: Set[str],
) -> dict[str, str]:
    """Return aliases whose value is source-proven to be an allowed callable.

    The resolver covers direct assignments, alias chains, and constructor
    parameters stored on ``self``.  It intentionally does not guess about
    values supplied only at runtime.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    aliases: dict[str, str] = {}
    assignments = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    )

    history: list[dict[str, str]] = [dict(aliases)]
    for _ in range(len(assignments) + 1):
        before = dict(aliases)
        for assignment in assignments:
            value = assignment.value
            canonical = _canonical_callable(value, import_aliases, aliases)
            if canonical not in allowed_targets:
                continue
            targets = (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else (assignment.target,)
            )
            for target in targets:
                name = _expression_name(target)
                if name and aliases.get(name) != canonical:
                    aliases[name] = canonical
        # Repeated assignments may overwrite the same name several times in
        # one pass. Convergence concerns the final mapping, not those writes.
        if aliases == before:
            break
        if aliases in history:
            cycle = history[history.index(aliases):]
            aliases = {
                name: value for name, value in aliases.items()
                if all(previous.get(name) == value for previous in cycle)
            }
            break
        history.append(dict(aliases))
    else:
        # A dependency cycle must not stall validation or select one arbitrary
        # phase. Retain only bindings that stayed constant across full passes.
        aliases = {
            name: value for name, value in aliases.items()
            if all(previous.get(name) == value for previous in history[1:])
        }

    for class_node in (
        node for node in tree.body if isinstance(node, ast.ClassDef)
    ):
        initializers = tuple(
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__init__"
        )
        if not initializers:
            continue
        # These are direct class-body definitions; Python retains the last one.
        initializer = initializers[-1]
        parameters = tuple(
            argument.arg
            for argument in (
                *initializer.args.posonlyargs,
                *initializer.args.args,
            )
            if argument.arg not in {"self", "cls"}
        )
        stored_parameters = {
            assignment.targets[0].attr
            if isinstance(assignment, ast.Assign)
            else assignment.target.attr: assignment.value.id
            for assignment in ast.walk(initializer)
            if isinstance(assignment, (ast.Assign, ast.AnnAssign))
            and len(assignment.targets if isinstance(assignment, ast.Assign) else (assignment.target,)) == 1
            and isinstance(
                (assignment.targets[0] if isinstance(assignment, ast.Assign) else assignment.target),
                ast.Attribute,
            )
            and isinstance(
                (assignment.targets[0] if isinstance(assignment, ast.Assign) else assignment.target).value,
                ast.Name,
            )
            and (assignment.targets[0] if isinstance(assignment, ast.Assign) else assignment.target).value.id
            in {"self", "cls"}
            and isinstance(assignment.value, ast.Name)
            and assignment.value.id in parameters
        }
        if not stored_parameters:
            continue

        defaults: dict[str, ast.AST] = {}
        if initializer.args.defaults:
            positional = tuple(
                argument.arg
                for argument in (
                    *initializer.args.posonlyargs,
                    *initializer.args.args,
                )
            )
            for name, default in zip(
                positional[-len(initializer.args.defaults) :],
                initializer.args.defaults,
            ):
                defaults[name] = default
        defaults.update(
            {
                argument.arg: default
                for argument, default in zip(
                    initializer.args.kwonlyargs,
                    initializer.args.kw_defaults,
                )
                if default is not None
            }
        )

        constructor_calls = tuple(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _expression_name(node.func) == class_node.name
        )
        for attribute, parameter in stored_parameters.items():
            candidates = [defaults[parameter]] if parameter in defaults else []
            parameter_index = parameters.index(parameter)
            for call in constructor_calls:
                if parameter_index < len(call.args):
                    candidates.append(call.args[parameter_index])
                candidates.extend(
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == parameter
                )
            canonical_values = {
                canonical
                for candidate in candidates
                if (
                    canonical := _canonical_callable(
                        candidate, import_aliases, aliases
                    )
                )
                in allowed_targets
            }
            if len(canonical_values) == 1:
                aliases[f"self.{attribute}"] = next(iter(canonical_values))
                aliases[f"cls.{attribute}"] = next(iter(canonical_values))

    return aliases


def _canonical_callable(
    expression: ast.AST,
    import_aliases: Mapping[str, str],
    aliases: Mapping[str, str],
) -> str:
    name = _expression_name(expression)
    if not name:
        return ""
    if name in aliases:
        return aliases[name]
    head, separator, tail = name.partition(".")
    return import_aliases.get(head, head) + (separator + tail if separator else "")


def _expression_name(expression: ast.AST) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        prefix = _expression_name(expression.value)
        return f"{prefix}.{expression.attr}" if prefix else expression.attr
    return ""
