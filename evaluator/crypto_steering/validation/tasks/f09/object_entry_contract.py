"""Bounded object errors on the actual F9 entry path, without running code."""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crypto_steering.validation.source.python import PythonAnalyzer


def object_entry_failure(analyzer: PythonAnalyzer, entry_path: Callable[[ast.AST], bool]) -> str | None:
    tree = analyzer.tree
    # Dynamic lookup, inherited descriptors and patches need a fuller object
    # model. A missing result here leaves those cases to existing checks.
    dynamic = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id in {"setattr", "exec", "eval", "globals", "locals"}
                  for n in ast.walk(tree))
    if dynamic:
        return None
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        if cls.bases or cls.decorator_list or cls.keywords:
            continue
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
        if any(n.name in {"__new__", "__setattr__", "__getattr__", "__getattribute__"} for n in methods):
            continue
        if any(isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
               and isinstance(n.value, ast.Name) and n.value.id == cls.name for n in ast.walk(tree)):
            continue
        if any(
            isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in {cls.name, "property"}
            or isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == "property"
            or isinstance(n, (ast.Import, ast.ImportFrom)) and any(
                (alias.asname or (alias.name.split(".")[0] if isinstance(n, ast.Import) else alias.name)) == "property"
                for alias in n.names
            )
            for n in ast.walk(tree)
        ):
            continue
        properties: dict[str, ast.FunctionDef] = {}
        for method in methods:
            if (len(method.decorator_list) == 1 and isinstance(method.decorator_list[0], ast.Name)
                    and method.decorator_list[0].id == "property"
                    and sum(n.name == method.name for n in methods) == 1
                    and not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                                and n.id == method.name for statement in cls.body for n in ast.walk(statement))):
                properties[method.name] = method
        for node in ast.walk(cls):
            if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                    and isinstance(node.value, ast.Name) and node.value.id == "self"
                    and node.attr in properties and analyzer._enclosing_class(node) is cls
                    and entry_path(node)):
                return f"line {node.lineno}: {cls.name}.{node.attr} is a read-only property without a setter"
        initializers = [n for n in methods if n.name == "__init__" and not n.decorator_list
                        and n in analyzer._function_nodes]
        if len(initializers) != 1:
            continue
        initializer = initializers[0]
        constructions = [n for n in analyzer._call_nodes
                         if not analyzer._node_inside(n, initializer)
                         and analyzer._external_call_targets_function(n, initializer) and entry_path(n)]
        if not constructions:
            continue
        # A direct self-construction before any branch/return has no base case.
        # A parameter or local binding of the class name instead calls that
        # value. Decline the recursion claim even if its identity is unknown.
        class_name_shadowed = cls.name in analyzer._function_parameter_names(initializer) or any(
            isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == cls.name
            or isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == cls.name
            or isinstance(n, (ast.Import, ast.ImportFrom)) and any(
                (alias.asname or (alias.name.split(".")[0] if isinstance(n, ast.Import) else alias.name)) == cls.name
                for alias in n.names
            )
            for n in ast.walk(initializer)
        )
        for statement in initializer.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
                break
            node = statement.value
            if (not class_name_shadowed and isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == cls.name and not node.args and not node.keywords
                    and not initializer.args.kwonlyargs
                    and len(initializer.args.posonlyargs) + len(initializer.args.args)
                    - len(initializer.args.defaults) <= 1):
                return f"line {node.lineno}: {cls.name}.__init__ unconditionally constructs {cls.name} again"
        for construction in constructions:
            if any(isinstance(a, ast.Starred) for a in construction.args) or any(k.arg is None for k in construction.keywords):
                continue
            bound: dict[str, object] = {}
            for position, parameter in enumerate(analyzer._function_parameter_names(initializer)):
                value = analyzer._call_argument_for_parameter(construction, initializer, position, parameter)
                if isinstance(value, ast.Constant):
                    bound[parameter] = value.value
            initialized = {target.id for statement in cls.body if isinstance(statement, ast.Assign)
                           for target in statement.targets if isinstance(target, ast.Name)}

            def visit(statements: list[ast.stmt]) -> str | None:
                for statement in statements:
                    if isinstance(statement, ast.If):
                        test = statement.test
                        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                                and test.left.id in bound and len(test.ops) == len(test.comparators) == 1
                                and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None
                                and isinstance(test.ops[0], (ast.Is, ast.IsNot))):
                            truth = bound[test.left.id] is None
                            if isinstance(test.ops[0], ast.IsNot):
                                truth = not truth
                            reason = visit(statement.body if truth else statement.orelse)
                            if reason:
                                return reason
                            continue
                        return None
                    if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
                        return None
                    value = statement.value
                    if value is None:
                        continue
                    # A method or external function receiving self could
                    # initialize fields; do not infer its side effects.
                    if any(isinstance(n, ast.Call) and (
                        isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"
                        or any(isinstance(a, ast.Name) and a.id == "self" for a in n.args)
                    ) for n in ast.walk(value)):
                        return None
                    # A direct property load is evaluated now; compound
                    # expressions may short-circuit or have earlier effects.
                    if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                            and value.value.id == "self" and value.attr in properties):
                        getter = properties[value.attr]
                        if len(getter.body) == 1 and isinstance(getter.body[0], ast.Return):
                            result = getter.body[0].value
                            if (isinstance(result, ast.Attribute) and isinstance(result.value, ast.Name)
                                    and result.value.id == "self" and result.attr not in initialized):
                                return (f"line {value.lineno}: {cls.name}.{value.attr} reads uninitialized "
                                        f"self.{result.attr} during construction")
                    targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,) if isinstance(statement, ast.AnnAssign) else ()
                    initialized.update(n.attr for target in targets for n in ast.walk(target)
                                       if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self")
                return None

            reason = visit(initializer.body)
            if reason:
                return reason
    return _late_bound_alias_failure(analyzer, entry_path)


def _late_bound_alias_failure(analyzer: PythonAnalyzer, entry_path: Callable[[ast.AST], bool]) -> str | None:
    """Resolve direct module aliases at entry time, not at function definition."""
    tree = analyzer.tree
    if any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(tree)):
        return None
    for entry in analyzer._call_nodes:
        if (analyzer._enclosing_function(entry) is not None or not isinstance(entry.func, ast.Name)
                or not entry_path(entry)):
            continue
        bindings: dict[str, ast.FunctionDef | None] = {}
        for statement in tree.body:
            if statement.lineno >= entry.lineno:
                break
            if isinstance(statement, ast.FunctionDef):
                bindings[statement.name] = statement if not statement.decorator_list else None
            elif isinstance(statement, ast.Assign):
                target = bindings.get(statement.value.id) if isinstance(statement.value, ast.Name) else None
                for name in statement.targets:
                    if isinstance(name, ast.Name):
                        bindings[name.id] = target
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                bindings[statement.target.id] = bindings.get(statement.value.id) if isinstance(statement.value, ast.Name) else None
            elif isinstance(statement, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                for node in ast.walk(statement):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        bindings[node.id] = None
        target = bindings.get(entry.func.id)
        if target is None:
            continue
        for statement in target.body:
            if not isinstance(statement, (ast.Return, ast.Assign, ast.AnnAssign, ast.Expr)):
                break
            value = statement.value
            for call in ([value] if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) else []):
                # Only actual global references: a parameter/local name wins.
                if (call.func.id in analyzer._function_parameter_names(target)
                        or any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == call.func.id
                               for n in ast.walk(target))):
                    continue
                callee = bindings.get(call.func.id)
                if callee is None or any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
                    continue
                positional = [a.arg for a in (*callee.args.posonlyargs, *callee.args.args)]
                if len(call.args) > len(positional) and callee.args.vararg is None:
                    return (f"line {call.lineno}: late-bound {call.func.id} resolves to {callee.name}, "
                            "which receives too many positional arguments")
    return None
