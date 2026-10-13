"""Route-aware Python PBKDF2 argument validation.

Python's parser accepts calls whose argument binding or source-provable value
types make the PBKDF2 operation unusable.  Generated responses can also
contain several independent PBKDF2 implementations.  This module therefore
checks complete constructor/call-to-derivation routes and accepts the source
when any one route is usable.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property, lru_cache
import re

from crypto_steering.validation.tasks.shared.python_dataflow import (
    bytes_passed_to_encoded_parameter,
)


_Function = ast.FunctionDef | ast.AsyncFunctionDef
_UNRESOLVED = object()


def python_pbkdf2_route_failure(source: str) -> str | None:
    """Return a definite argument/route failure, if every PBKDF2 route fails."""

    try:
        index = _PythonPbkdf2Index(source, ast.parse(source))
    except SyntaxError:
        return None

    failures: list[str] = []
    routes = 0
    for call in index.hashlib_calls:
        routes += 1
        failure = index.hashlib_call_failure(call)
        if failure is None:
            return None
        failures.append(failure)

    for constructor, terminal in index.pyca_routes:
        routes += 1
        failure = index.pyca_route_failure(constructor, terminal)
        if failure is None:
            return None
        failures.append(failure)

    linked_constructors = {id(constructor) for constructor, _ in index.pyca_routes}
    linked_terminals = {id(terminal) for _, terminal in index.pyca_routes}
    unlinked_terminals = tuple(
        call
        for call in index._calls()
        if id(call) not in linked_terminals
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"derive", "verify"}
        and not index._inside_lambda(call)
    )
    for constructor in index.pyca_constructors:
        if id(constructor) in linked_constructors:
            continue
        routes += 1
        constructor_failure = index.pyca_constructor_failure(constructor)
        if constructor_failure is not None:
            failures.append(constructor_failure)
            continue
        terminal_failures = [
            index.pyca_terminal_failure(terminal) for terminal in unlinked_terminals
        ]
        if any(failure is None for failure in terminal_failures):
            return None
        failures.extend(
            failure for failure in terminal_failures if failure is not None
        )
        if not terminal_failures:
            failures.append("PBKDF2HMAC construction is not connected to derive or verify")

    if routes:
        return failures[0]
    return None


@dataclass(frozen=True)
class _PythonPbkdf2Index:
    source: str
    tree: ast.Module

    @cached_property
    def parents(self) -> dict[ast.AST, ast.AST]:
        return {
            child: parent
            for parent in ast.walk(self.tree)
            for child in ast.iter_child_nodes(parent)
        }

    @cached_property
    def functions(self) -> tuple[_Function, ...]:
        return tuple(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )

    @cached_property
    def functions_by_name(self) -> dict[str, _Function]:
        return {function.name: function for function in self.functions}

    @cached_property
    def argparse_namespaces(self) -> frozenset[str]:
        return frozenset(
            target.id
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Call)
            and _name(node.value.func).rsplit(".", 1)[-1] == "parse_args"
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            if isinstance(target, ast.Name)
        )

    @cached_property
    def argparse_attribute_kinds(self) -> dict[str, str]:
        kinds: dict[str, str] = {}
        for call in self._calls():
            if _name(call.func).rsplit(".", 1)[-1] != "add_argument":
                continue
            destination = _argparse_destination(call)
            if destination is None:
                continue
            conversion = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "type"),
                None,
            )
            kind = _argparse_conversion_kind(conversion)
            default = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "default"),
                None,
            )
            required = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "required"),
                None,
            )
            default_kind = _argparse_default_kind(default)
            if default_kind is None and default is not None:
                # Generated command-line tools often obtain a binary default
                # from a small local helper (for example ``get_salt()``).
                # Follow that helper's return value just as we do at the
                # eventual PBKDF2 call site.
                default_kind = self.expression_kind(default, call)
            if default_kind is not None and not (
                isinstance(required, ast.Constant) and required.value is True
            ):
                # argparse preserves a non-text default without applying ``type``.
                # That default is one usable task route even when a supplied
                # command-line override would have another type.
                kind = default_kind
            kinds[destination] = kind
        return kinds

    @cached_property
    def hashlib_modules(self) -> frozenset[str]:
        names = {"hashlib"}
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Import):
                continue
            names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "hashlib"
            )
        return frozenset(names)

    @cached_property
    def hashlib_functions(self) -> frozenset[str]:
        names: set[str] = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "hashlib":
                continue
            names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "pbkdf2_hmac"
            )
        for node in ast.walk(self.tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            if not self._is_hashlib_reference(node.value):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            names.update(target.id for target in targets if isinstance(target, ast.Name))
        return frozenset(names)

    @cached_property
    def hashlib_calls(self) -> tuple[ast.Call, ...]:
        return tuple(
            call
            for call in self._calls()
            if self._is_hashlib_call(call) and not self._inside_lambda(call)
        )

    @cached_property
    def pyca_constructors(self) -> tuple[ast.Call, ...]:
        return tuple(
            call
            for call in self._calls()
            if _name(call.func).rsplit(".", 1)[-1] == "PBKDF2HMAC"
            and not self._inside_lambda(call)
        )

    @cached_property
    def pyca_routes(self) -> tuple[tuple[ast.Call, ast.Call], ...]:
        routes: list[tuple[ast.Call, ast.Call]] = []
        assignments: dict[tuple[_Function | None, str], list[tuple[int, ast.Call]]] = {}
        for node in ast.walk(self.tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Call) or _name(value.func).rsplit(".", 1)[-1] != "PBKDF2HMAC":
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault((self.scope(node), target.id), []).append(
                        (getattr(node, "lineno", 0), value)
                    )

        for terminal in self._calls():
            if not isinstance(terminal.func, ast.Attribute) or terminal.func.attr not in {"derive", "verify"}:
                continue
            receiver = terminal.func.value
            if isinstance(receiver, ast.Call) and _name(receiver.func).rsplit(".", 1)[-1] == "PBKDF2HMAC":
                routes.append((receiver, terminal))
                continue
            if not isinstance(receiver, ast.Name):
                continue
            candidates = assignments.get((self.scope(terminal), receiver.id), ())
            prior = [item for item in candidates if item[0] <= getattr(terminal, "lineno", 0)]
            if prior:
                routes.append((max(prior, key=lambda item: item[0])[1], terminal))
        return tuple(routes)

    def hashlib_call_failure(self, call: ast.Call) -> str | None:
        binding = _bind_call(
            call,
            ("hash_name", "password", "salt", "iterations", "dklen"),
            required=frozenset({"hash_name", "password", "salt", "iterations"}),
        )
        if isinstance(binding, str):
            return "hashlib.pbkdf2_hmac " + binding

        if not self.function_route_is_usable(self.scope(call)):
            function = self.scope(call)
            return (
                f"PBKDF2 helper {function.name} has no compatible invocation"
                if function is not None
                else "PBKDF2 call has no compatible invocation"
            )

        conflict = self.encoded_parameter_conflict(self.scope(call))
        if conflict is not None:
            function, parameter = conflict
            return f"{function} receives bytes for {parameter}, then calls str.encode on it"
        conflict = self.bytes_constructor_conflict(self.scope(call))
        if conflict is not None:
            function, parameter = conflict
            return (
                f"{function} receives bytes for {parameter}, then passes it "
                "to bytes with a text encoding"
            )

        digest = binding["hash_name"]
        if _is_hash_constructor(digest) or self.expression_kind(
            digest, call
        ) == "hash_constructor":
            return (
                "hashlib.pbkdf2_hmac requires a digest-name string, "
                "not a hash constructor"
            )
        length = binding.get("dklen")
        if length is not None and (
            _is_hash_constructor(length)
            or self.expression_kind(length, call) == "hash_constructor"
        ):
            return (
                "hashlib.pbkdf2_hmac output length must be an integer, "
                "not a hash constructor"
            )
        for field in ("password", "salt"):
            kind = self.expression_kind(binding[field], call)
            if kind == "text":
                return f"hashlib.pbkdf2_hmac requires bytes-like {field} input, not text"
            if kind == "bytes_encode":
                return f"hashlib.pbkdf2_hmac {field} calls str.encode on a bytes value"
            if kind in {"invalid_bytes", "missing_attribute"}:
                return f"hashlib.pbkdf2_hmac has no source-provable bytes-like {field} input"
        return None

    def pyca_route_failure(self, constructor: ast.Call, terminal: ast.Call) -> str | None:
        constructor_failure = self.pyca_constructor_failure(constructor)
        if constructor_failure is not None:
            return constructor_failure

        return self.pyca_terminal_failure(terminal)

    def pyca_constructor_failure(self, constructor: ast.Call) -> str | None:
        binding = _bind_call(
            constructor,
            ("algorithm", "length", "salt", "iterations", "backend"),
            required=frozenset({"algorithm", "length", "salt", "iterations"}),
        )
        if isinstance(binding, str):
            return "PBKDF2HMAC " + binding
        if self.expression_kind(binding["length"], constructor) in {"bytes", "text"} or (
            _definitely_non_length_argument(binding["length"])
        ):
            return (
                "PBKDF2HMAC positional arguments do not follow algorithm, "
                "length, salt, iterations order"
            )
        salt_kind = self.expression_kind(binding["salt"], constructor)
        if salt_kind in {"text", "bytes_encode", "invalid_bytes", "missing_attribute"}:
            return "PBKDF2HMAC requires bytes-like salt material"
        if self.expression_kind(binding["iterations"], constructor) in {"bytes", "text"}:
            return "PBKDF2HMAC iterations must be an integer"
        if isinstance(binding["salt"], ast.Constant) and binding["salt"].value is None:
            return "PBKDF2HMAC requires bytes-like salt, not None"
        return None

    def pyca_terminal_failure(self, terminal: ast.Call) -> str | None:
        terminal_binding = _bind_call(
            terminal,
            ("key_material",),
            required=frozenset({"key_material"}),
        )
        if isinstance(terminal_binding, str):
            return f"PBKDF2HMAC.{terminal.func.attr} {terminal_binding}"
        if not self.function_route_is_usable(self.scope(terminal)):
            function = self.scope(terminal)
            return (
                f"PBKDF2 helper {function.name} has no compatible invocation"
                if function is not None
                else "PBKDF2 route has no compatible invocation"
            )
        conflict = self.encoded_parameter_conflict(self.scope(terminal))
        if conflict is not None:
            function, parameter = conflict
            return f"{function} receives bytes for {parameter}, then calls str.encode on it"
        conflict = self.bytes_constructor_conflict(self.scope(terminal))
        if conflict is not None:
            function, parameter = conflict
            return (
                f"{function} receives bytes for {parameter}, then passes it "
                "to bytes with a text encoding"
            )
        kind = self.expression_kind(terminal_binding["key_material"], terminal)
        if kind == "text":
            return "PBKDF2HMAC.derive requires bytes-like key material"
        if kind == "bytes_encode":
            return "PBKDF2HMAC.derive calls str.encode on a bytes value"
        if kind in {"invalid_bytes", "missing_attribute"}:
            return "PBKDF2HMAC.derive has no source-provable bytes-like key material"
        return None

    @lru_cache(maxsize=None)
    def function_route_is_usable(
        self,
        function: _Function | None,
        seen: frozenset[str] = frozenset(),
    ) -> bool:
        if function is None:
            return True
        marker = f"{function.name}:{getattr(function, 'lineno', 0)}"
        if marker in seen:
            return True
        callsites = tuple(
            call
            for call in self._calls()
            if _call_targets_function(call, function, self)
            and self.scope(call) is not function
        )
        if not callsites:
            # A requested utility may validly be delivered as a callable
            # function without a demonstration invocation.
            return True
        for call in callsites:
            if not _function_call_binds(call, function, self, call):
                continue
            if not self.call_path_is_possible(call):
                continue
            if self.function_route_is_usable(
                self.scope(call), seen | frozenset({marker})
            ):
                return True
        return False

    def expression_kind(
        self,
        expression: ast.AST | None,
        at: ast.AST,
        seen: frozenset[str] = frozenset(),
    ) -> str | None:
        if isinstance(expression, ast.Constant):
            if isinstance(expression.value, bytes):
                return "bytes"
            if isinstance(expression.value, str):
                return "text"
            if isinstance(expression.value, int) and not isinstance(expression.value, bool):
                return "integer"
            return None
        if isinstance(expression, ast.JoinedStr):
            return "text"
        if _is_hash_constructor(expression):
            return "hash_constructor"
        if isinstance(expression, ast.Name):
            marker = f"name:{expression.id}:{getattr(at, 'lineno', 0)}"
            if marker in seen:
                return None
            scope = self.scope(at)
            values = self.assignments_before(expression.id, at, scope)
            destructured = False
            if not values:
                values = self.destructured_values_before(expression.id, at, scope)
                destructured = bool(values)
            if values:
                kinds: list[str | None] = []
                for value in values:
                    if _encodes_name(value, expression.id):
                        input_kind = (
                            self.parameter_kind(scope, expression.id, seen)
                            if scope is not None
                            and expression.id in _parameter_names(scope)
                            else None
                        )
                        kinds.append(
                            "bytes_encode" if input_kind == "bytes" else "bytes"
                        )
                    else:
                        kinds.append(
                            self.expression_kind(
                                value,
                                value if destructured else at,
                                seen | frozenset({marker}),
                            )
                        )
                return _combine_binary_kinds(kinds)
            if scope is not None and expression.id in _parameter_names(scope):
                return self.parameter_kind(
                    scope,
                    expression.id,
                    seen | frozenset({marker}),
                )
            return None
        if isinstance(expression, ast.Attribute):
            return self.attribute_kind(expression, at, seen)
        if isinstance(expression, ast.Subscript):
            if _name(expression.value).startswith("os.environ"):
                return "text"
            return self.expression_kind(expression.value, at, seen)
        if not isinstance(expression, ast.Call):
            return None
        name = _name(expression.func)
        simple = name.rsplit(".", 1)[-1]
        if isinstance(expression.func, ast.Attribute) and simple == "encode":
            receiver = self.expression_kind(expression.func.value, at, seen)
            return (
                "bytes_encode"
                if receiver in {"bytes", "bytes_or_invalid"}
                else "bytes"
            )
        if isinstance(expression.func, ast.Attribute) and simple == "decode":
            return "text"
        if simple == "getattr" and expression.args and _name(expression.args[0]) in self.hashlib_modules:
            return "hash_constructor"
        if simple == "read" and self._binary_read(expression):
            return "bytes"
        if isinstance(expression.func, ast.Attribute) and simple == "strip":
            return self.expression_kind(expression.func.value, at, seen)
        if simple == "bytes" and len(expression.args) >= 2:
            source_kind = self.expression_kind(expression.args[0], at, seen)
            if source_kind == "bytes":
                return "bytes_encode"
        if simple == "fromhex" and expression.args:
            input_kind = self.expression_kind(expression.args[0], at, seen)
            return "invalid_bytes" if input_kind == "bytes" else "bytes"
        if simple in {
            "urandom", "token_bytes", "get_random_bytes", "b64decode",
            "urlsafe_b64decode", "unhexlify", "digest",
            "bytes", "bytearray",
        }:
            return "bytes"
        if simple in {"input", "getpass", "str"}:
            return "text"
        if simple == "token_urlsafe":
            return "text"
        if simple in {"hex", "token_hex"}:
            return "text"
        if isinstance(expression.func, ast.Attribute) and simple == "join":
            return "text"
        if simple in {"derive", "verify"} and isinstance(expression.func, ast.Attribute):
            binding = _bind_call(
                expression,
                ("key_material",),
                required=frozenset({"key_material"}),
            )
            if isinstance(binding, str):
                return "invalid_bytes"
            material_kind = self.expression_kind(binding["key_material"], expression, seen)
            if material_kind in {"text", "bytes_encode", "invalid_bytes", "missing_attribute"}:
                return "invalid_bytes"
            return "bytes"
        function = self.functions_by_name.get(simple)
        if function is None:
            return None
        marker = f"function-return:{function.name}:{getattr(function, 'lineno', 0)}"
        if marker in seen:
            return None
        kinds = {
            self.expression_kind(node.value, node, seen | frozenset({marker}))
            for node in ast.walk(function)
            if isinstance(node, ast.Return)
            and node.value is not None
            and self.scope(node) is function
        }
        kinds.discard(None)
        return kinds.pop() if len(kinds) == 1 else None

    @lru_cache(maxsize=None)
    def encoded_parameter_conflict(
        self,
        function: _Function | None,
    ) -> tuple[str, str] | None:
        if function is None:
            return None
        return bytes_passed_to_encoded_parameter(
            self.source,
            function_names={function.name},
        )

    @lru_cache(maxsize=None)
    def bytes_constructor_conflict(
        self,
        function: _Function | None,
    ) -> tuple[str, str] | None:
        """Find ``bytes(already_bytes, encoding)`` on a reachable route."""

        if function is None:
            return None
        parameters = set(_parameter_names(function))
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and _name(call.func).rsplit(".", 1)[-1] == "bytes"
                and len(call.args) >= 2
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id in parameters
                and self.scope(call) is function
                and self.call_path_is_possible(call)
            ):
                continue
            parameter = call.args[0].id
            if self.parameter_kind(function, parameter, frozenset()) in {
                "bytes",
                "bytes_or_invalid",
            }:
                return function.name, parameter
        return None

    def parameter_kind(
        self,
        function: _Function,
        parameter: str,
        seen: frozenset[str],
    ) -> str | None:
        position = _parameter_names(function).index(parameter)
        kinds: list[str | None] = []
        observed = False
        for call in self._calls():
            if not (
                _call_targets_function(call, function, self)
                and self.scope(call) is not function
                and _function_call_binds(call, function, self, call)
                and self.call_path_is_possible(call)
                and self.function_route_is_usable(self.scope(call))
            ):
                continue
            supplied = _bound_function_argument(call, function, parameter, position)
            if supplied is None:
                continue
            observed = True
            kind = self.expression_kind(supplied, call, seen)
            kinds.append(kind)
        if observed:
            return _combine_binary_kinds(kinds)
        return None

    def call_path_is_possible(self, call: ast.Call) -> bool:
        """Reject only branches disproved by literal arguments at call sites."""

        function = self.scope(call)
        if function is None:
            return True
        current: ast.AST = call
        parent = self.parents.get(current)
        while parent is not None and parent is not function:
            if isinstance(parent, ast.If):
                condition = _simple_parameter_comparison(parent.test, function)
                if condition is not None:
                    parameter, operator, expected = condition
                    values, unresolved = self.parameter_constants(function, parameter)
                    if values and not unresolved:
                        in_body = any(
                            _contains_node(node, current) for node in parent.body
                        )
                        selected = {
                            value == expected if operator == "eq" else value != expected
                            for value in values
                        }
                        if in_body and True not in selected:
                            return False
                        if not in_body and False not in selected:
                            return False
            current = parent
            parent = self.parents.get(current)
        return True

    @lru_cache(maxsize=None)
    def parameter_constants(
        self,
        function: _Function,
        parameter: str,
    ) -> tuple[frozenset[object], bool]:
        position = _parameter_names(function).index(parameter)
        values: set[object] = set()
        unresolved = False
        observed = False
        for call in self._calls():
            if not (
                _call_targets_function(call, function, self)
                and self.scope(call) is not function
                and _function_call_binds(call, function, self, call)
            ):
                continue
            supplied = _bound_function_argument(call, function, parameter, position)
            if supplied is None:
                unresolved = True
                continue
            observed = True
            value = self.expression_constant(supplied, call)
            if value is _UNRESOLVED:
                unresolved = True
            else:
                values.add(value)
        return frozenset(values), unresolved or not observed

    def expression_constant(self, expression: ast.AST, at: ast.AST) -> object:
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, ast.Name):
            values = self.assignments_before(expression.id, at, self.scope(at))
            resolved = {
                value.value
                for value in values
                if isinstance(value, ast.Constant)
            }
            if len(resolved) == 1 and len(resolved) == len(values):
                return next(iter(resolved))
        return _UNRESOLVED

    def assignment_before(
        self,
        name: str,
        at: ast.AST,
        scope: _Function | None,
    ) -> ast.AST | None:
        boundary = (getattr(at, "lineno", 10**9), getattr(at, "col_offset", 10**9))
        values = self.assignments_before(name, at, scope)
        return values[-1] if values else None

    def assignments_before(
        self,
        name: str,
        at: ast.AST,
        scope: _Function | None,
    ) -> tuple[ast.AST, ...]:
        """Return values that can reach ``at`` on alternative source paths."""

        body = self.tree.body if scope is None else scope.body
        values, found = _reaching_values(
            body,
            name,
            at,
            (),
            self.condition_truth,
        )
        if found and values:
            return values
        if scope is not None and name not in _parameter_names(scope):
            return self.assignments_before(name, at, None)
        return ()

    def destructured_values_before(
        self,
        name: str,
        at: ast.AST,
        scope: _Function | None,
    ) -> tuple[ast.AST, ...]:
        """Resolve values assigned through tuple/list unpacking before ``at``."""

        boundary = getattr(at, "lineno", 10**9)
        candidates: list[tuple[int, tuple[ast.AST, ...]]] = []
        for node in ast.walk(self.tree if scope is None else scope):
            if not (
                isinstance(node, ast.Assign)
                and self.scope(node) is scope
                and getattr(node, "lineno", 0) <= boundary
            ):
                continue
            for target in node.targets:
                if not isinstance(target, (ast.Tuple, ast.List)):
                    continue
                positions = [
                    index
                    for index, item in enumerate(target.elts)
                    if isinstance(item, ast.Name) and item.id == name
                ]
                for position in positions:
                    values = self.destructured_result_values(node.value, position)
                    if values:
                        candidates.append((getattr(node, "lineno", 0), values))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]
        if scope is not None and name not in _parameter_names(scope):
            return self.destructured_values_before(name, at, None)
        return ()

    def destructured_result_values(
        self,
        expression: ast.AST,
        position: int,
    ) -> tuple[ast.AST, ...]:
        if isinstance(expression, (ast.Tuple, ast.List)):
            return (
                (expression.elts[position],)
                if position < len(expression.elts)
                else ()
            )
        if not isinstance(expression, ast.Call):
            return ()
        function = self.functions_by_name.get(_name(expression.func).rsplit(".", 1)[-1])
        if function is None:
            return ()
        return tuple(
            node.value.elts[position]
            for node in ast.walk(function)
            if isinstance(node, ast.Return)
            and isinstance(node.value, (ast.Tuple, ast.List))
            and position < len(node.value.elts)
            and self.scope(node) is function
        )

    def attribute_kind(
        self,
        expression: ast.Attribute,
        at: ast.AST,
        seen: frozenset[str],
    ) -> str | None:
        if not isinstance(expression.value, ast.Name):
            return None
        base = expression.value.id
        if base in self.argparse_namespaces:
            declared = self.argparse_attribute_kinds.get(expression.attr, "text")
            dotted_name = f"{base}.{expression.attr}"
            marker = f"attribute:{dotted_name}:{getattr(at, 'lineno', 0)}"
            if marker in seen:
                return declared
            values = self.assignments_before(dotted_name, at, self.scope(at))
            if values:
                assigned = [
                    self.expression_kind(value, at, seen | frozenset({marker}))
                    for value in values
                ]
                # Include the parser-provided value because a conditional
                # replacement may leave it untouched on another usable route.
                return _combine_binary_kinds([declared, *assigned])
            return declared
        if base == "self":
            class_node = self.class_scope(at)
            if class_node is not None:
                values = [
                    node.value
                    for node in ast.walk(class_node)
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                    and any(
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == expression.attr
                        for target in (
                            node.targets if isinstance(node, ast.Assign) else (node.target,)
                        )
                    )
                ]
                if values:
                    kinds: list[str | None] = []
                    initializer = next(
                        (
                            node
                            for node in class_node.body
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name == "__init__"
                        ),
                        None,
                    )
                    constructor_calls = tuple(
                        call
                        for call in self._calls()
                        if _name(call.func).rsplit(".", 1)[-1] == class_node.name
                        and self.class_scope(call) is not class_node
                    )
                    for value in values:
                        if (
                            initializer is not None
                            and isinstance(value, ast.Name)
                            and value.id in _parameter_names(initializer)[1:]
                        ):
                            parameters = _parameter_names(initializer)[1:]
                            position = parameters.index(value.id)
                            for call in constructor_calls:
                                supplied = _bound_class_argument(
                                    call,
                                    initializer,
                                    value.id,
                                    position,
                                )
                                if supplied is not None:
                                    kinds.append(
                                        self.expression_kind(supplied, call, seen)
                                    )
                        else:
                            kinds.append(self.expression_kind(value, value, seen))
                    return _combine_binary_kinds(kinds)

        scope = self.scope(at)
        if scope is not None and base in _parameter_names(scope):
            normalized = base.replace("_", "").lower()
            matching = [
                node
                for node in ast.walk(self.tree)
                if isinstance(node, ast.ClassDef)
                and node.name.lower() == normalized
            ]
            if matching and all(
                not _class_defines_attribute(node, expression.attr) for node in matching
            ):
                return "missing_attribute"
        return None

    def condition_truth(self, expression: ast.AST) -> bool | None:
        function = self.scope(expression)
        if function is None or not isinstance(expression, ast.Compare):
            return None
        condition = _simple_parameter_comparison(expression, function)
        if condition is not None:
            parameter, operator, expected = condition
            values, unresolved = self.parameter_constants(function, parameter)
            return _comparison_truth(values, unresolved, operator, expected)
        if not (
            len(expression.ops) == 1
            and len(expression.comparators) == 1
            and isinstance(expression.left, ast.Attribute)
            and isinstance(expression.left.value, ast.Name)
            and expression.left.value.id == "self"
            and isinstance(expression.comparators[0], ast.Constant)
            and isinstance(expression.ops[0], (ast.Eq, ast.NotEq))
        ):
            return None
        values, unresolved = self.self_attribute_constants(
            function,
            expression.left.attr,
        )
        return _comparison_truth(
            values,
            unresolved,
            "eq" if isinstance(expression.ops[0], ast.Eq) else "ne",
            expression.comparators[0].value,
        )

    @lru_cache(maxsize=None)
    def self_attribute_constants(
        self,
        function: _Function,
        attribute: str,
    ) -> tuple[frozenset[object], bool]:
        class_node = self.class_scope(function)
        if class_node is None:
            return frozenset(), True
        initializer = next(
            (
                node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__init__"
            ),
            None,
        )
        if initializer is None:
            return frozenset(), True
        assigned = next(
            (
                node.value
                for node in ast.walk(initializer)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and any(
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == attribute
                    for target in (
                        node.targets if isinstance(node, ast.Assign) else (node.target,)
                    )
                )
            ),
            None,
        )
        if isinstance(assigned, ast.Constant):
            return frozenset({assigned.value}), False
        if not (
            isinstance(assigned, ast.Name)
            and assigned.id in _parameter_names(initializer)[1:]
        ):
            return frozenset(), True
        parameters = _parameter_names(initializer)[1:]
        position = parameters.index(assigned.id)
        values: set[object] = set()
        unresolved = False
        observed = False
        for call in self._calls():
            if (
                _name(call.func).rsplit(".", 1)[-1] != class_node.name
                or self.class_scope(call) is class_node
            ):
                continue
            supplied = _bound_class_argument(
                call,
                initializer,
                assigned.id,
                position,
            )
            if supplied is None:
                unresolved = True
                continue
            observed = True
            value = self.expression_constant(supplied, call)
            if value is _UNRESOLVED:
                unresolved = True
            else:
                values.add(value)
        return frozenset(values), unresolved or not observed

    def class_scope(self, node: ast.AST) -> ast.ClassDef | None:
        current = self.parents.get(node)
        while current is not None:
            if isinstance(current, ast.ClassDef):
                return current
            current = self.parents.get(current)
        return None

    def _binary_read(self, call: ast.Call) -> bool:
        receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
        if _name(receiver).startswith("sys.stdin.buffer"):
            return True
        if (
            isinstance(receiver, ast.Call)
            and _name(receiver.func) == "open"
            and any(
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and "b" in argument.value
                for argument in receiver.args[1:]
            )
        ):
            return True
        receiver_name = receiver.id if isinstance(receiver, ast.Name) else None
        current: ast.AST | None = call
        while current is not None:
            if isinstance(current, ast.With):
                for item in current.items:
                    if not (
                        isinstance(item.optional_vars, ast.Name)
                        and item.optional_vars.id == receiver_name
                        and isinstance(item.context_expr, ast.Call)
                        and _name(item.context_expr.func) == "open"
                    ):
                        continue
                    if any(
                        isinstance(argument, ast.Constant)
                        and isinstance(argument.value, str)
                        and "b" in argument.value
                        for argument in item.context_expr.args[1:]
                    ):
                        return True
            current = self.parents.get(current)
        return False

    def scope(self, node: ast.AST) -> _Function | None:
        current = self.parents.get(node)
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current
            current = self.parents.get(current)
        return None

    def _calls(self) -> tuple[ast.Call, ...]:
        return tuple(node for node in ast.walk(self.tree) if isinstance(node, ast.Call))

    def _inside_lambda(self, node: ast.AST) -> bool:
        current = self.parents.get(node)
        while current is not None:
            if isinstance(current, ast.Lambda):
                return True
            current = self.parents.get(current)
        return False

    def _is_hashlib_reference(self, expression: ast.AST) -> bool:
        return bool(
            isinstance(expression, ast.Attribute)
            and expression.attr == "pbkdf2_hmac"
            and isinstance(expression.value, ast.Name)
            and expression.value.id in self.hashlib_modules
        )

    def _is_hashlib_call(self, call: ast.Call) -> bool:
        if self._is_hashlib_reference(call.func):
            return True
        if not isinstance(call.func, ast.Name):
            return False
        if call.func.id in self.functions_by_name:
            return False
        return call.func.id in self.hashlib_functions


def _definitely_non_length_argument(expression: ast.AST) -> bool:
    """Recognize source-proven binary/text material in PyCA's length slot."""

    if isinstance(expression, ast.Constant):
        return isinstance(expression.value, (str, bytes))
    if isinstance(expression, ast.Name):
        return re.fullmatch(
            r"(?:password|passphrase|passwd|pwd|secret|salt)(?:_bytes)?",
            expression.id,
            flags=re.IGNORECASE,
        ) is not None
    return bool(
        isinstance(expression, ast.Call)
        and _name(expression.func).rsplit(".", 1)[-1]
        in {"encode", "bytes", "bytearray", "fromhex", "b64decode"}
    )


def _combine_binary_kinds(kinds: list[str | None]) -> str | None:
    """Combine alternative paths using the task's existential-route rule."""

    if not kinds:
        return None
    known = {kind for kind in kinds if kind is not None}
    if "bytes" in known and "text" in known:
        return "mixed"
    if "bytes" in known and known & {
        "bytes_encode", "missing_attribute", "invalid_bytes"
    }:
        return "bytes_or_invalid"
    if known == {"bytes"}:
        return "bytes"
    if any(kind is None for kind in kinds):
        return None
    if len(known) == 1:
        return next(iter(known))
    if known <= {"text", "bytes_encode", "missing_attribute", "invalid_bytes"}:
        return "invalid_bytes"
    return None


def _simple_parameter_comparison(
    expression: ast.AST,
    function: _Function,
) -> tuple[str, str, object] | None:
    if not (
        isinstance(expression, ast.Compare)
        and len(expression.ops) == 1
        and len(expression.comparators) == 1
        and isinstance(expression.left, ast.Name)
        and expression.left.id in _parameter_names(function)
        and isinstance(expression.comparators[0], ast.Constant)
        and isinstance(expression.ops[0], (ast.Eq, ast.NotEq))
    ):
        return None
    return (
        expression.left.id,
        "eq" if isinstance(expression.ops[0], ast.Eq) else "ne",
        expression.comparators[0].value,
    )


def _comparison_truth(
    values: frozenset[object],
    unresolved: bool,
    operator: str,
    expected: object,
) -> bool | None:
    """Resolve a comparison only when every reachable concrete value agrees."""

    if unresolved or not values:
        return None
    outcomes = {
        value == expected if operator == "eq" else value != expected
        for value in values
    }
    return next(iter(outcomes)) if len(outcomes) == 1 else None


def _reaching_values(
    statements: list[ast.stmt],
    name: str,
    target: ast.AST,
    incoming: tuple[ast.AST, ...],
    condition_truth: Callable[[ast.AST], bool | None],
) -> tuple[tuple[ast.AST, ...], bool]:
    state = incoming
    for statement in statements:
        if _contains_node(statement, target):
            for block in _child_blocks(statement):
                if any(_contains_node(child, target) for child in block):
                    return _reaching_values(
                        block,
                        name,
                        target,
                        state,
                        condition_truth,
                    )
            return state, True
        state = _flow_statement(statement, name, state, condition_truth)
    return state, False


def _flow_statement(
    statement: ast.stmt,
    name: str,
    incoming: tuple[ast.AST, ...],
    condition_truth: Callable[[ast.AST], bool | None],
) -> tuple[ast.AST, ...]:
    assigned = _assigned_value(statement, name)
    if assigned is not None:
        return (assigned,)
    if isinstance(statement, ast.If):
        truth = condition_truth(statement.test)
        if truth is True:
            return _flow_block(statement.body, name, incoming, condition_truth)
        if truth is False:
            return _flow_block(statement.orelse, name, incoming, condition_truth)
        body = _flow_block(statement.body, name, incoming, condition_truth)
        alternate = (
            _flow_block(statement.orelse, name, incoming, condition_truth)
            if statement.orelse
            else incoming
        )
        return _unique_nodes((*body, *alternate))
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
        body = _flow_block(statement.body, name, incoming, condition_truth)
        alternate = _flow_block(statement.orelse, name, incoming, condition_truth)
        return _unique_nodes((*incoming, *body, *alternate))
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return _flow_block(statement.body, name, incoming, condition_truth)
    if isinstance(statement, ast.Try):
        paths = [_flow_block(statement.body, name, incoming, condition_truth)]
        paths.extend(
            _flow_block(handler.body, name, incoming, condition_truth)
            for handler in statement.handlers
        )
        if statement.orelse:
            paths.append(
                _flow_block(statement.orelse, name, paths[0], condition_truth)
            )
        merged = _unique_nodes(node for path in paths for node in path)
        return _flow_block(statement.finalbody, name, merged, condition_truth)
    return incoming


def _flow_block(
    statements: list[ast.stmt],
    name: str,
    incoming: tuple[ast.AST, ...],
    condition_truth: Callable[[ast.AST], bool | None],
) -> tuple[ast.AST, ...]:
    state = incoming
    for statement in statements:
        state = _flow_statement(statement, name, state, condition_truth)
    return state


def _assigned_value(statement: ast.stmt, name: str) -> ast.AST | None:
    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return None
    targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
    if any(_name(target) == name for target in targets):
        return statement.value
    return None


def _child_blocks(statement: ast.stmt) -> tuple[list[ast.stmt], ...]:
    if isinstance(statement, ast.If):
        return statement.body, statement.orelse
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
        return statement.body, statement.orelse
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return (statement.body,)
    if isinstance(statement, ast.Try):
        return (
            statement.body,
            *(handler.body for handler in statement.handlers),
            statement.orelse,
            statement.finalbody,
        )
    return ()


def _contains_node(root: ast.AST, target: ast.AST) -> bool:
    return any(node is target for node in ast.walk(root))


def _unique_nodes(nodes: object) -> tuple[ast.AST, ...]:
    values: list[ast.AST] = []
    seen: set[int] = set()
    for node in nodes:  # type: ignore[union-attr]
        if id(node) not in seen:
            seen.add(id(node))
            values.append(node)
    return tuple(values)


def _class_defines_attribute(class_node: ast.ClassDef, attribute: str) -> bool:
    declared_field = any(
        isinstance(target, ast.Name) and target.id == attribute
        for node in class_node.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
    )
    assigned_instance_attribute = any(
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        and target.attr == attribute
        for node in ast.walk(class_node)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
    )
    return declared_field or assigned_instance_attribute


def _argparse_destination(call: ast.Call) -> str | None:
    explicit = next(
        (keyword.value for keyword in call.keywords if keyword.arg == "dest"),
        None,
    )
    if isinstance(explicit, ast.Constant) and isinstance(explicit.value, str):
        return explicit.value.replace("-", "_")
    options = [
        value.value
        for value in call.args
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    ]
    long_options = [value for value in options if value.startswith("--")]
    selected = long_options[-1] if long_options else (options[-1] if options else "")
    return selected.lstrip("-").replace("-", "_") or None


def _argparse_conversion_kind(conversion: ast.AST | None) -> str:
    if conversion is None:
        return "text"
    simple = _name(conversion).rsplit(".", 1)[-1] if conversion is not None else ""
    if isinstance(conversion, (ast.Name, ast.Attribute)):
        if simple in {
            "a2b_base64",
            "a2b_hex",
            "b64decode",
            "bytearray",
            "bytes",
            "fromhex",
            "hexlify",
            "unhexlify",
            "urlsafe_b64decode",
        }:
            return "bytes"
        if simple == "str":
            return "text"
        if simple == "int":
            return "integer"
    if isinstance(conversion, ast.Lambda) and isinstance(conversion.body, ast.Call):
        simple = _name(conversion.body.func).rsplit(".", 1)[-1]
        if simple in {
            "encode",
            "fromhex",
            "a2b_base64",
            "a2b_hex",
            "b64decode",
            "bytes",
            "bytearray",
            "hexlify",
            "unhexlify",
            "urlsafe_b64decode",
        }:
            return "bytes"
        if simple in {"decode", "str"}:
            return "text"
    return "text"


def _argparse_default_kind(default: ast.AST | None) -> str | None:
    """Return the source-provable kind of a useful argparse default."""

    if isinstance(default, ast.Constant):
        if isinstance(default.value, bytes):
            return "bytes"
        if isinstance(default.value, int) and not isinstance(default.value, bool):
            return "integer"
        return None
    if not isinstance(default, ast.Call):
        return None
    simple = _name(default.func).rsplit(".", 1)[-1]
    if simple in {
        "urandom",
        "token_bytes",
        "get_random_bytes",
        "b64decode",
        "urlsafe_b64decode",
        "unhexlify",
        "bytes",
        "bytearray",
        "fromhex",
    }:
        return "bytes"
    if simple == "int":
        return "integer"
    return None


def _bind_call(
    call: ast.Call,
    parameters: tuple[str, ...],
    *,
    required: frozenset[str],
) -> dict[str, ast.AST] | str:
    if any(isinstance(argument, ast.Starred) for argument in call.args) or any(
        keyword.arg is None for keyword in call.keywords
    ):
        return "uses argument unpacking that does not establish the required API positions"
    if len(call.args) > len(parameters):
        return f"accepts at most {len(parameters)} positional arguments"
    bound = dict(zip(parameters, call.args))
    for keyword in call.keywords:
        assert keyword.arg is not None
        if keyword.arg not in parameters:
            return f"received unsupported keyword {keyword.arg}"
        if keyword.arg in bound:
            return f"supplies {keyword.arg} more than once"
        bound[keyword.arg] = keyword.value
    missing = sorted(required - set(bound))
    if missing:
        return "omits required " + ", ".join(missing)
    return bound


def _function_call_binds(
    call: ast.Call,
    function: _Function,
    index: _PythonPbkdf2Index,
    at: ast.AST,
) -> bool:
    positional_parameters = _call_parameters(function, call)
    positional_count = sum(not isinstance(argument, ast.Starred) for argument in call.args)
    for argument in call.args:
        if not isinstance(argument, ast.Starred):
            continue
        kind = index.expression_kind(argument.value, at)
        if kind in {"bytes", "text", "bytes_encode", "integer"}:
            return False
        return True
    if function.args.vararg is None and positional_count > len(positional_parameters):
        return False
    parameter_names = {parameter.arg for parameter in positional_parameters}
    keyword_only = {parameter.arg for parameter in function.args.kwonlyargs}
    supplied_keywords = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
    if any(keyword.arg is None for keyword in call.keywords):
        return True
    if function.args.kwarg is None and supplied_keywords - parameter_names - keyword_only:
        return False
    if supplied_keywords & {
        parameter.arg for parameter in positional_parameters[:positional_count]
    }:
        return False
    required_positional = len(positional_parameters) - len(function.args.defaults)
    supplied_positional_names = {
        parameter.arg for parameter in positional_parameters[:positional_count]
    }
    if any(
        parameter.arg not in supplied_positional_names
        and parameter.arg not in supplied_keywords
        for parameter in positional_parameters[:required_positional]
    ):
        return False
    return all(
        default is not None or parameter.arg in supplied_keywords
        for parameter, default in zip(function.args.kwonlyargs, function.args.kw_defaults)
    )


def _bound_function_argument(
    call: ast.Call,
    function: _Function,
    parameter: str,
    position: int,
) -> ast.AST | None:
    parameters = _call_parameters(function, call)
    effective_position = next(
        (
            index
            for index, candidate in enumerate(parameters)
            if candidate.arg == parameter
        ),
        position,
    )
    if effective_position < len(call.args) and not isinstance(
        call.args[effective_position], ast.Starred
    ):
        return call.args[effective_position]
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg == parameter),
        None,
    )


def _bound_class_argument(
    call: ast.Call,
    initializer: _Function,
    parameter: str,
    position: int,
) -> ast.AST | None:
    """Return an argument supplied to ``__init__``, excluding ``self``."""

    if position < len(call.args) and not isinstance(call.args[position], ast.Starred):
        return call.args[position]
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg == parameter),
        None,
    )


def _parameter_names(function: _Function) -> tuple[str, ...]:
    return tuple(
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    )


def _call_parameters(function: _Function, call: ast.Call) -> tuple[ast.arg, ...]:
    parameters = (*function.args.posonlyargs, *function.args.args)
    if (
        isinstance(call.func, ast.Attribute)
        and parameters
        and parameters[0].arg in {"self", "cls"}
    ):
        return parameters[1:]
    return parameters


def _call_targets_function(
    call: ast.Call,
    function: _Function,
    index: _PythonPbkdf2Index,
) -> bool:
    if isinstance(call.func, ast.Name):
        return call.func.id == function.name
    if not isinstance(call.func, ast.Attribute) or call.func.attr != function.name:
        return False
    class_node = index.class_scope(function)
    if class_node is None:
        return False
    receiver = call.func.value
    if isinstance(receiver, ast.Call):
        return _name(receiver.func).rsplit(".", 1)[-1] == class_node.name
    if not isinstance(receiver, ast.Name):
        return True
    origins = index.assignments_before(receiver.id, call, index.scope(call))
    local_class_names = {
        _name(origin.func).rsplit(".", 1)[-1]
        for origin in origins
        if isinstance(origin, ast.Call)
    }
    return not local_class_names or class_node.name in local_class_names


def _is_hash_constructor(expression: ast.AST) -> bool:
    value = expression.func if isinstance(expression, ast.Call) else expression
    name = _name(value).lower()
    return name.rsplit(".", 1)[-1] in {
        "md5", "sha1", "sha224", "sha256", "sha384", "sha512",
        "sha3_224", "sha3_256", "sha3_384", "sha3_512", "blake2b", "blake2s",
    } and not isinstance(expression, ast.Constant)


def _encodes_name(expression: ast.AST, name: str) -> bool:
    return bool(
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr == "encode"
        and isinstance(expression.func.value, ast.Name)
        and expression.func.value.id == name
    )


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


__all__ = ("python_pbkdf2_route_failure",)
