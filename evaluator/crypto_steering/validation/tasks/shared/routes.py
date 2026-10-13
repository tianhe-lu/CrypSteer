"""Existential validation over independent callable task routes.

Rule 6 accepts a program when any one top-level, entry-point, or callable route
forms a usable setup-to-completion sequence. Whole-source analysis runs first.
If it fails, this module projects independent definitions and top-level routes
so an unrelated example or sibling implementation cannot override a complete
route.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Iterable
from dataclasses import replace
import re

from crypto_steering.validation.java.source import iter_methods
from crypto_steering.validation.models import TaskValidationResult
from crypto_steering.validation.source.index import analyzer_for
from crypto_steering.validation.tasks.shared.context import TaskContext


TaskRouteValidator = Callable[[TaskContext], TaskValidationResult]


def validate_any_route(
    context: TaskContext,
    validator: TaskRouteValidator,
) -> TaskValidationResult:
    """Return success when the whole source or one independent route succeeds."""

    whole_program = validator(context)
    if whole_program.success or not context.allow_independent_routes:
        return whole_program
    if (
        "target_argument_contract_invalid" in whole_program.reason_codes
        and _argument_failure_depends_on_observed_values(whole_program.reason)
    ):
        # Projecting a called helper without its caller would erase fixed
        # argument evidence and invent a usable execution route.  Unrelated
        # malformed APIs still use the ordinary independent-route fallback.
        return whole_program
    analysis_profile = getattr(context.analyzer, "analysis_profile", "whole_program")
    try:
        route_sources = independent_route_sources(context)
    except Exception:
        # Route projection is a fallback over an already-completed family
        # assessment. Failure to project cannot turn that result into a
        # provider error.
        return whole_program
    for source in route_sources:
        try:
            analyzer = analyzer_for(
                context.language,
                source,
                analysis_profile=analysis_profile,
            )
            candidate = validator(
                replace(
                    context,
                    source=source,
                    analyzer=analyzer,
                    target_detection=None,
                    allow_independent_routes=False,
                )
            )
        except Exception:
            # A projection is supporting evidence, not a new provider.  Ignore
            # one the lightweight source index or family validator cannot
            # represent.
            continue
        if candidate.success:
            return candidate
    return whole_program


def independent_route_sources(context: TaskContext) -> tuple[str, ...]:
    """Project independent routes while retaining their local implementation."""

    if context.language == "python":
        candidates = _python_route_sources(context.source)
    elif context.language in {"typescript", "csharp"}:
        candidates = _c_style_route_sources(context)
    else:
        candidates = _java_route_sources(context.source)
    normalized_whole = _normalized_source(context.source)
    return tuple(
        candidate
        for candidate in _unique_nonempty(candidates)
        if _normalized_source(candidate) != normalized_whole
    )


def _python_route_sources(source: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()

    roots: list[ast.AST] = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    roots.extend(
        prefix
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for prefix in _python_completed_prefixes(node)
    )
    roots.extend(
        node
        for node in tree.body
        if not isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        )
        and any(isinstance(child, ast.Call) for child in ast.walk(node))
    )
    for class_node in (
        node for node in tree.body if isinstance(node, ast.ClassDef)
    ):
        methods = tuple(
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name not in {"__init__", "__new__"}
        )
        for method in methods:
            roots.append(_project_python_class_route(class_node, method))
            # A method is also a callable route in its own right.  This second
            # projection prevents a malformed constructor or sibling method
            # from hiding a complete implementation.
            if not _python_method_depends_on_malformed_dsa_state(
                class_node,
                method,
            ):
                roots.append(method)

    return tuple(_python_module_source(tree, root) for root in roots)


def _python_method_depends_on_malformed_dsa_state(
    class_node: ast.ClassDef,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Keep a source-proven broken DSA key constructor on its method route."""

    used_attributes = {
        node.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"self", "cls"}
        and isinstance(node.ctx, ast.Load)
    }
    if not used_attributes:
        return False
    initializer = next(
        (
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"__init__", "__new__"}
        ),
        None,
    )
    if initializer is None:
        return False
    changed = True
    while changed:
        changed = False
        for assignment in ast.walk(initializer):
            if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else (assignment.target,)
            )
            if not any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in {"self", "cls"}
                and target.attr in used_attributes
                for target in targets
            ):
                continue
            dependencies = {
                node.attr
                for node in ast.walk(assignment.value)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in {"self", "cls"}
                and isinstance(node.ctx, ast.Load)
            }
            before = len(used_attributes)
            used_attributes.update(dependencies)
            changed = changed or len(used_attributes) != before
    for assignment in ast.walk(initializer):
        if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
        if not any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id in {"self", "cls"}
            and target.attr in used_attributes
            for target in targets
        ):
            continue
        for call in (
            node for node in ast.walk(assignment.value) if isinstance(node, ast.Call)
        ):
            name = _python_dotted_name(call.func)
            if not name.endswith(
                ("dsa.generate_parameters", "dsa.generate_private_key")
            ):
                continue
            allowed = {"key_size", "backend"}
            if (
                set(
                    keyword.arg
                    for keyword in call.keywords
                    if keyword.arg is not None
                )
                - allowed
                or len(call.args) > 2
                or (not call.args and not any(
                    keyword.arg == "key_size" for keyword in call.keywords
                ))
            ):
                return True
    return False


def _python_dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _python_dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


_PYTHON_TERMINAL_OPERATIONS = frozenset(
    {
        "decrypt",
        "digest",
        "encrypt",
        "finalize",
        "finalize_with_tag",
        "hexdigest",
        "sign",
        "verify",
    }
)


def _python_completed_prefixes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    """Project prefixes ending at a task terminal in one callable route.

    A task can already be complete before unrelated post-processing or a
    sibling direction fails.  Keeping every preceding statement preserves the
    setup and data flow while preventing later code from overriding that
    completed route.
    """

    prefixes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for index, statement in enumerate(function.body):
        if not any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in _PYTHON_TERMINAL_OPERATIONS
            for call in ast.walk(statement)
        ):
            continue
        projected = copy.deepcopy(function)
        projected.body = projected.body[: index + 1]
        prefixes.append(projected)
    return tuple(prefixes)


def _project_python_class_route(
    class_node: ast.ClassDef,
    selected: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.ClassDef:
    methods = {
        node.name: node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    selected_names = {selected.name}
    pending = [selected]
    while pending:
        method = pending.pop()
        for call in (
            node for node in ast.walk(method) if isinstance(node, ast.Call)
        ):
            if not (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in {"self", "cls"}
            ):
                continue
            dependency = methods.get(call.func.attr)
            if dependency is not None and dependency.name not in selected_names:
                selected_names.add(dependency.name)
                pending.append(dependency)

    projected = copy.deepcopy(class_node)
    projected.body = [
        node
        for node in projected.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        or node.name in selected_names
        or node.name in {"__init__", "__new__"}
    ]
    if not projected.body:
        projected.body = [ast.Pass()]
    return projected


def _python_module_source(tree: ast.Module, root: ast.AST) -> str:
    imports = tuple(
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    definitions: dict[str, list[ast.AST]] = {}
    for node in tree.body:
        for name in _python_bound_names(node):
            definitions.setdefault(name, []).append(node)

    selected: dict[int, ast.AST] = {id(root): root}
    seen_names: set[str] = set()

    def include_dependencies(names: Iterable[str]) -> None:
        pending = list(names)
        while pending:
            name = pending.pop()
            if name in seen_names:
                continue
            seen_names.add(name)
            for dependency in definitions.get(name, ()):
                if dependency is root or id(dependency) in selected:
                    continue
                selected[id(dependency)] = dependency
                pending.extend(_python_external_names(dependency))

    include_dependencies(_python_external_names(root))

    # A top-level route is a sequence, not an arbitrary single statement.
    # Retain earlier method calls that mutate an object used by the selected
    # route (for example ``encryptor.authenticate_additional_data(...)`` or
    # ``padder.update(...)``). Omitting those calls could make an unusable
    # route appear valid merely because its final statement looks complete.
    if root in tree.body and not isinstance(
        root,
        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
    ):
        root_index = tree.body.index(root)
        relevant = {
            name
            for node in selected.values()
            for name in (*_python_bound_names(node), *_python_external_names(node))
        }
        changed = True
        while changed:
            changed = False
            for node in tree.body[:root_index]:
                if id(node) in selected or not _python_mutates_any(node, relevant):
                    continue
                selected[id(node)] = node
                relevant.update(_python_bound_names(node))
                relevant.update(_python_external_names(node))
                include_dependencies(_python_external_names(node))
                for dependency in selected.values():
                    relevant.update(_python_bound_names(dependency))
                    relevant.update(_python_external_names(dependency))
                changed = True

    ordered = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom)) or id(node) in selected
    ]
    if id(root) not in {id(node) for node in tree.body}:
        ordered.append(root)
    # Imports are already present in source order above; this fallback handles
    # a synthetic projected class built from a tree with no body imports.
    if not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ordered):
        ordered[:0] = imports
    return "\n\n".join(ast.unparse(node) for node in ordered) + "\n"


def _python_bound_names(node: ast.AST) -> frozenset[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return frozenset({node.name})
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        return frozenset(
            target.id
            for target in targets
            if isinstance(target, ast.Name)
        )
    return frozenset()


def _python_external_names(node: ast.AST) -> frozenset[str]:
    loaded = {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }
    local = {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    }
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local.update(argument.arg for argument in child.args.args)
            local.update(argument.arg for argument in child.args.posonlyargs)
            local.update(argument.arg for argument in child.args.kwonlyargs)
            if child.args.vararg is not None:
                local.add(child.args.vararg.arg)
            if child.args.kwarg is not None:
                local.add(child.args.kwarg.arg)
    return frozenset(loaded - local)


def _python_mutates_any(node: ast.AST, names: set[str]) -> bool:
    """Return whether a statement calls a method on a route-owned value."""

    return any(
        _python_receiver_root(call.func.value) in names
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        and call.func.attr not in {
            "decryptor",
            "encryptor",
            "padder",
            "public_key",
            "unpadder",
        }
    )


def _python_receiver_root(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _c_style_route_sources(context: TaskContext) -> tuple[str, ...]:
    analyzer = context.analyzer
    if analyzer is None:
        return ()
    source = context.source
    prefix = _import_prefix(source, context.language)
    candidates: list[str] = []
    functions = {
        (definition.start, definition.end, definition.name): definition
        for definition in getattr(analyzer, "functions", ())
    }.values()
    classes = tuple(getattr(analyzer, "classes", ()))
    for definition in sorted(functions, key=lambda item: (item.start, item.end)):
        containing_class = next(
            (
                candidate
                for candidate in classes
                if candidate.start <= definition.start < definition.end <= candidate.end
            ),
            None,
        )
        if containing_class is not None:
            if not _c_style_method_is_static(source, definition.start):
                if not _c_style_instance_method_is_standalone(
                    analyzer,
                    source,
                    definition,
                    containing_class,
                ):
                    # A stateful instance method is not independent of its
                    # constructor and fields. Project the enclosing class below
                    # so fixed member state remains visible. A class with an
                    # implicit default constructor may still expose a reusable,
                    # parameter-only instance route.
                    continue
            if _c_style_static_method_uses_class_state(
                analyzer,
                source,
                definition,
                containing_class,
            ):
                # Static methods can also depend on fixed class fields. A bare
                # method projection would turn ``Type.key`` into an unresolved
                # value and could make a source-proven bad argument appear
                # usable. The enclosing-class projection below retains that
                # state while still allowing any valid sibling route.
                continue
        line_start = source.rfind("\n", 0, definition.start) + 1
        dependencies = _c_style_top_level_dependencies(
            analyzer,
            source,
            definition,
            classes,
        )
        candidates.append(
            prefix
            + dependencies
            + source[line_start : definition.end].strip()
            + "\n"
        )
    for definition in classes:
        candidates.append(prefix + source[definition.start : definition.end].strip() + "\n")
    return tuple(candidates)


def _c_style_method_is_static(source: str, start: int) -> bool:
    """Return whether a class-contained C#/TypeScript method is static."""

    line_start = source.rfind("\n", 0, start) + 1
    header = source[line_start:start]
    return re.search(r"\bstatic\b", header) is not None


def _c_style_instance_method_is_standalone(
    analyzer: object,
    source: str,
    definition: object,
    containing_class: object,
) -> bool:
    """Return whether an instance method is a parameter-only callable route.

    A method on a class with an implicit default constructor can be used as a
    complete route without inheriting a malformed example call elsewhere in
    the source. Methods that explicitly read instance or class fields remain
    attached to their enclosing class projection.
    """

    class_body = source[containing_class.start : containing_class.end]
    modifiers = (
        r"(?:(?:public|private|protected|internal|static|abstract|sealed|"
        r"partial|extern|unsafe)\s+)*"
    )
    constructor_pattern = (
        rf"(?m)^[ \t]*{modifiers}constructor\s*\("
        if getattr(analyzer, "language", "") == "typescript"
        else rf"(?m)^[ \t]*{modifiers}{re.escape(containing_class.name)}\s*\("
    )
    if re.search(constructor_pattern, class_body):
        return False

    method_body = source[definition.start : definition.end]
    if re.search(r"\b(?:this|base)\s*\.", method_body):
        return False
    return not _c_style_static_method_uses_class_state(
        analyzer,
        source,
        definition,
        containing_class,
    )


def _c_style_static_method_uses_class_state(
    analyzer: object,
    source: str,
    definition: object,
    containing_class: object,
) -> bool:
    """Return whether a static route reads a field owned by its class."""

    assignment_scopes = getattr(analyzer, "_assignment_scopes", {})
    field_names = {
        re.sub(r"^(?:this|base)\.", "", assignment.name).rsplit(".", 1)[-1]
        for assignment in getattr(analyzer, "assignments", ())
        if assignment_scopes.get(assignment) is None
        and containing_class.start <= assignment.start < assignment.end <= containing_class.end
    }
    if not field_names:
        return False
    body = source[definition.start : definition.end]
    return any(
        re.search(
            rf"(?<![\w$])(?:{re.escape(containing_class.name)}\s*\.\s*)?"
            rf"{re.escape(name)}(?![\w$])",
            body,
        )
        is not None
        for name in field_names
    )


def _c_style_top_level_dependencies(
    analyzer: object,
    source: str,
    definition: object,
    classes: tuple[object, ...],
) -> str:
    """Retain fixed globals directly used by a projected callable route.

    A top-level helper may be independent of its example callers while still
    depending on module state.  Keeping those declarations prevents route
    projection from turning a source-proven bad key, IV, or algorithm into an
    unresolved (and therefore acceptable) value.
    """

    assignment_scopes = getattr(analyzer, "_assignment_scopes", {})
    top_level = tuple(
        assignment
        for assignment in getattr(analyzer, "assignments", ())
        if assignment_scopes.get(assignment) is None
        and not any(
            candidate.start <= assignment.start < assignment.end <= candidate.end
            for candidate in classes
        )
    )
    if not top_level:
        return ""

    local_names = set(getattr(definition, "parameters", ()))
    local_names.update(
        assignment.name.rsplit(".", 1)[-1]
        for assignment in getattr(analyzer, "assignments", ())
        if assignment_scopes.get(assignment) == definition
    )
    required = set(
        re.findall(
            r"\b[A-Za-z_$][\w$]*\b",
            source[definition.start : definition.end],
        )
    ) - local_names
    selected: dict[int, object] = {}
    changed = True
    while changed:
        changed = False
        for assignment in top_level:
            name = re.sub(r"^(?:this|base)\.", "", assignment.name).rsplit(".", 1)[-1]
            if name not in required or assignment.start in selected:
                continue
            selected[assignment.start] = assignment
            required.update(
                re.findall(r"\b[A-Za-z_$][\w$]*\b", assignment.expression)
            )
            changed = True
    if not selected:
        return ""

    statements: list[str] = []
    seen_spans: set[tuple[int, int]] = set()
    for assignment in sorted(selected.values(), key=lambda item: item.start):
        line_start = source.rfind("\n", 0, assignment.start) + 1
        span = (line_start, assignment.end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        statements.append(source[line_start : assignment.end].strip())
    return "\n".join(statements) + ("\n\n" if statements else "")


def _java_route_sources(source: str) -> tuple[str, ...]:
    prefix = _import_prefix(source, "java")
    methods = tuple(iter_methods(source))
    by_name: dict[str, tuple[object, ...]] = {}
    for name in {method.name for method in methods}:
        by_name[name] = tuple(method for method in methods if method.name == name)

    candidates: list[str] = []
    for root in methods:
        selected: dict[tuple[int, int], object] = {
            (root.start, root.body_end): root
        }
        pending = [root]
        while pending:
            method = pending.pop()
            body = source[method.start : method.body_end + 1]
            for name, definitions in by_name.items():
                if re.search(rf"(?<![\w$]){re.escape(name)}\s*\(", body) is None:
                    continue
                for dependency in definitions:
                    key = (dependency.start, dependency.body_end)
                    if key in selected:
                        continue
                    selected[key] = dependency
                    pending.append(dependency)
        implementation = "\n\n".join(
            source[method.start : method.body_end + 1].strip()
            for method in sorted(selected.values(), key=lambda item: item.start)
        )
        candidates.append(
            prefix + "class IndependentTaskRoute {\n" + implementation + "\n}\n"
        )
    return tuple(candidates)


def _argument_failure_depends_on_observed_values(reason: str) -> bool:
    """Keep call-site evidence for values that make the target call unusable."""

    normalized = reason.lower()
    return any(
        marker in normalized
        for marker in (
            "requires a supported key length",
            "requires block-aligned input",
            "is already bytes and cannot be encoded",
            "receives bytes for",
        )
    ) or bool(
        re.search(
            r"requires (?:a|an) \d+-byte (?:key|iv|value).*not \d+ bytes",
            normalized,
        )
    )


def _import_prefix(source: str, language: str) -> str:
    if language == "typescript":
        pattern = r"(?m)^[ \t]*import\b[^\n;]*(?:;|$)"
    elif language == "csharp":
        pattern = r"(?m)^[ \t]*(?:global\s+)?using\s+[A-Za-z_][\w.]*\s*;"
    else:
        pattern = r"(?m)^[ \t]*import\s+(?:static\s+)?[\w.*]+\s*;"
    matches = tuple(match.group(0).strip() for match in re.finditer(pattern, source))
    return "\n".join(matches) + ("\n\n" if matches else "")


def _normalized_source(source: str) -> str:
    return re.sub(r"\s+", "", source)


def _unique_nonempty(candidates: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalized_source(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(candidate)
    return tuple(result)


__all__ = ("independent_route_sources", "validate_any_route")
