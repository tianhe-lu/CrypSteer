"""Independent Python implementations for existential task validation.

Generated responses sometimes contain several independent implementations of
the same task.  The validity contract accepts the response when at least one
implementation has a usable setup-to-completion sequence; malformed extra
examples do not invalidate that route.  This module extracts small candidate
fragments without weakening the checks applied to any selected route.
"""

from __future__ import annotations

import ast
_ENCRYPT_NAMES = {"encrypt", "encryptor", "seal", "wrap", "createencryptor"}
_DECRYPT_NAMES = {"decrypt", "decryptor", "open", "unwrap", "createdecryptor"}
_ISOLATABLE_FAILURE_MARKERS = (
    "AESGCM construction",
    "AES AEAD ",
    "Cipher requires an instantiated mode object",
    "modes.GCM requires an initialization vector",
    "not an available symmetric-cipher API",
    "padding input is discarded",
    "PKCS7 padder exposes output",
    "GCM encryption result",
    "GCM round trip",
    "shared GCM cipher",
    "independently generate different keys",
    "splits the IV envelope",
    "assigned encrypt context",
    "assigned decrypt context",
    "secrets module namespace",
    "unavailable cipher material",
)


def failure_can_belong_to_an_independent_route(reason: str) -> bool:
    """Return whether a whole-source failure can be isolated by class route.

    Key derivation and other setup failures are deliberately excluded because
    an otherwise well-formed cipher class may consume their unusable output.
    """

    return any(marker in reason for marker in _ISOLATABLE_FAILURE_MARKERS)


def independent_symmetric_class_sources(source: str, _task_id: str) -> tuple[str, ...]:
    """Return independent class routes that may complete the family.

    Class boundaries provide a conservative way to separate genuinely
    independent implementations in one generated response. A callable route
    does not need to be invoked by this source, and either encryption or
    decryption can satisfy the family-level contract.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()

    imports = tuple(
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    classes = tuple(node for node in tree.body if isinstance(node, ast.ClassDef))

    candidates: list[str] = []
    for class_node in classes:
        if _class_can_supply_family_direction(class_node):
            dependencies = _constructor_dependencies(tree, class_node, classes)
            candidates.append(_module_source((*imports, *dependencies, class_node)))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return tuple(unique)


def independent_symmetric_function_sources(
    source: str,
    _task_id: str,
) -> tuple[str, ...]:
    """Return independent top-level symmetric-function routes.

    Each directional function is a candidate because F1 and F2 require only
    one completed direction. Failures in shared key derivation and setup remain
    whole-source failures.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()

    imports = tuple(
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    functions = tuple(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    encrypt_functions = tuple(
        node for node in functions if node.name.lower() in _ENCRYPT_NAMES
    )
    decrypt_functions = tuple(
        node for node in functions if node.name.lower() in _DECRYPT_NAMES
    )

    routes = (
        (function,)
        for function in (*encrypt_functions, *decrypt_functions)
        if not _route_has_external_helper_dependency(tree, (function,), functions)
    )

    candidates = [_module_source((*imports, *route)) for route in routes]
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _constructor_dependencies(
    tree: ast.Module,
    class_node: ast.ClassDef,
    classes: tuple[ast.ClassDef, ...],
) -> tuple[ast.ClassDef, ...]:
    """Keep support classes passed into an independently validated route."""

    by_name = {node.name: node for node in classes}
    assignments = [
        (target.id, node.value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    ]
    origins: dict[str, list[ast.AST]] = {}
    for name, value in assignments:
        origins.setdefault(name, []).append(value)
    dependencies: list[ast.ClassDef] = []
    for _, value in assignments:
        if not (
            isinstance(value, ast.Call)
            and _qualified_name(value.func).rsplit(".", 1)[-1] == class_node.name
        ):
            continue
        for argument in value.args:
            if not isinstance(argument, ast.Name):
                continue
            for origin in origins.get(argument.id, ()):
                if not isinstance(origin, ast.Call):
                    continue
                dependency = by_name.get(
                    _qualified_name(origin.func).rsplit(".", 1)[-1]
                )
                if dependency is not None and dependency is not class_node:
                    dependencies.append(dependency)
    return tuple(dict.fromkeys(dependencies))


def _class_can_supply_family_direction(node: ast.ClassDef) -> bool:
    return _contains_direction(node, "encrypt") or _contains_direction(
        node, "decrypt"
    )


def _contains_direction(node: ast.AST, direction: str) -> bool:
    wanted = _ENCRYPT_NAMES if direction == "encrypt" else _DECRYPT_NAMES
    return any(
        _simple_call_name(call.func).lower() in wanted
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    )


def _route_has_external_helper_dependency(
    tree: ast.Module,
    route: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
    functions: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
) -> bool:
    """Return whether a route call consumes another local helper's result.

    Such a helper is part of the selected route's setup and cannot be dropped
    when isolating the directional functions from an unrelated implementation.
    """

    route_names = {function.name for function in route}
    local_helpers = {function.name for function in functions} - route_names
    definitions = {function.name: function for function in functions}
    assignments = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name)
    }
    route_nodes = {node for function in route for node in ast.walk(function)}
    for call in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _simple_call_name(node.func) in route_names
        and node not in route_nodes
    ):
        for argument in (*call.args, *(keyword.value for keyword in call.keywords)):
            if not isinstance(argument, ast.Name):
                continue
            origin = assignments.get(argument.id)
            if (
                isinstance(origin, ast.Call)
                and _simple_call_name(origin.func) in local_helpers
            ):
                helper = definitions.get(_simple_call_name(origin.func))
                if helper is None or not _safe_fresh_bytes_helper(helper):
                    return True
    return False


def _safe_fresh_bytes_helper(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Return whether a setup helper only returns fresh byte material."""

    fresh_calls = {
        "urandom",
        "token_bytes",
        "randbytes",
        "get_random_bytes",
    }
    returns = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    return bool(returns) and all(
        any(
            isinstance(call, ast.Call)
            and _simple_call_name(call.func) in fresh_calls
            for call in ast.walk(value)
        )
        for value in returns
    )


def _module_source(nodes: tuple[ast.AST, ...]) -> str:
    return "\n\n".join(ast.unparse(node) for node in nodes) + "\n" if nodes else ""


def _simple_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


__all__ = (
    "failure_can_belong_to_an_independent_route",
    "independent_symmetric_class_sources",
    "independent_symmetric_function_sources",
)
