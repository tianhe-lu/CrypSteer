"""Python AST-backed bounded source/value-flow analysis."""

from __future__ import annotations

import ast
import operator
import re
from typing import Iterable

from crypto_steering.validation.models import (
    AnalysisProfile,
    Language,
    ResolvedValue,
    ResolutionStep,
    SourceSpan,
    ValueOrigin,
)
from crypto_steering.validation.source.base import CallExpression, SourceAnalyzer
from crypto_steering.validation.source.helpers import (
    _MAX_RESOLUTION_STEPS,
    _append_resolution_step,
    _canonical_python_name,
    _combine_resolved_int_operands,
    _derived_kdf_origin,
    _direct_python_call,
    _direct_python_name,
    _evaluate_int_ast,
    _evaluate_python_string_ast,
    _line_offsets,
    _merge_entropy_preserving_origins,
    _merge_resolved,
    _merge_transformed_origins,
    _parameter_name,
    _prepend_resolution,
    _python_assignment_bindings,
    _python_condition_accepts_nonempty,
    _python_condition_rejects_empty,
    _python_guard_allowed_integer_values,
    _python_guard_rejects_values_below,
    _python_import_aliases,
    _python_literal_container_element,
    _python_name,
    _python_resolved_has_only_fixed_values,
    _python_resolved_is_fixed_for_branch,
    _python_restrict_resolved_values,
    _python_statement_assigns_reference,
    _python_statements_terminate,
    _python_string_guard_target_matches,
    _python_struct_pack_is_constant,
    _python_subscript_selector,
    _python_target_names,
    _resolved_has_only_nullish_leaves,
    _resolved_single_int,
    _resolved_without_empty_alternatives,
)


class PythonAnalyzer(SourceAnalyzer):
    """AST-backed bounded resolver for Python 3.11 source."""

    def __init__(
        self,
        source: str,
        *,
        analysis_profile: AnalysisProfile = "whole_program",
    ) -> None:
        self.language: Language = "python"
        self.source = source
        self.analysis_profile = analysis_profile
        self.tree = ast.parse(source)
        self.import_aliases = _python_import_aliases(self.tree)
        self._line_offsets = _line_offsets(source)
        self._parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                self._parents[child] = parent
        self._shadowed_methods = self._python_shadowed_plain_methods()
        self._inactive_method_body_nodes = frozenset(
            child for method in self._shadowed_methods
            for statement in method.body for child in ast.walk(statement)
        )
        self._call_nodes = tuple(
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and node not in self._inactive_method_body_nodes
        )
        self._assignments = self._collect_assignments()
        self._subscript_assignments = self._collect_subscript_assignments()
        self._append_arguments = self._collect_append_arguments()
        self._function_nodes = tuple(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node not in self._shadowed_methods
            and node not in self._inactive_method_body_nodes
        )
        self._class_nodes = tuple(
            node for node in ast.walk(self.tree) if isinstance(node, ast.ClassDef)
        )
        functions_by_name: dict[
            str,
            list[ast.FunctionDef | ast.AsyncFunctionDef],
        ] = {}
        for function in self._function_nodes:
            functions_by_name.setdefault(function.name, []).append(function)
        self._functions_by_name = {
            name: tuple(functions)
            for name, functions in functions_by_name.items()
        }
        self._resolution_depth = 0
        self._reachability_depth = 0
        self._condition_evaluation_nodes: set[int] = set()
        self._matching_function_calls: set[tuple[int, int | None]] = set()
        self._plain_field_resolutions: set[tuple[str, int]] = set()
        self._bound_helper_calls: list[
            tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Call, int]
        ] = []

    def _python_shadowed_plain_methods(self) -> frozenset[ast.FunctionDef]:
        """Identify only unambiguously overwritten ordinary class methods."""
        shadowed: set[ast.FunctionDef] = set()
        for owner in (n for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)):
            if owner.bases or owner.decorator_list or owner.keywords:
                continue
            grouped: dict[str, list[ast.FunctionDef]] = {}
            for statement in owner.body:
                if isinstance(statement, ast.FunctionDef):
                    grouped.setdefault(statement.name, []).append(statement)
            for name, methods in grouped.items():
                if len(methods) < 2 or any(m.decorator_list for m in methods):
                    continue
                # Captured old functions, conditional definitions, descriptors
                # and definition-time calls retain their previous treatment.
                if any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(owner)):
                    continue
                if any(
                    isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == name and n not in methods
                    and self._enclosing_class(n) is owner
                    for n in ast.walk(owner)
                ):
                    continue
                if any(
                    isinstance(n, ast.Call)
                    for method in methods
                    for expression in (method.args, *([method.returns] if method.returns else []))
                    for n in ast.walk(expression)
                ):
                    continue
                shadowed.update(methods[:-1])
        return frozenset(shadowed)

    def calls(self, *simple_names: str) -> tuple[CallExpression, ...]:
        wanted = set(simple_names)
        records: list[CallExpression] = []
        for node in self._call_nodes:
            callee = _python_name(node.func)
            canonical = _canonical_python_name(callee, self.import_aliases)
            simple = callee.rsplit(".", 1)[-1]
            canonical_simple = canonical.rsplit(".", 1)[-1]
            if (
                wanted
                and simple not in wanted
                and callee not in wanted
                and canonical_simple not in wanted
                and canonical not in wanted
            ):
                continue
            arguments = tuple(self._segment(argument) for argument in node.args)
            named = {
                keyword.arg: self._segment(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            start, end = self._node_span(node)
            records.append(CallExpression(callee, arguments, named, start, end))
        return tuple(sorted(records, key=lambda item: item.start))

    def resolve_origin(self, expression: str, index: int) -> ResolvedValue:
        return self._resolve_text(expression, index, mode="origin")

    def resolve_strings(self, expression: str, index: int) -> ResolvedValue:
        return self._resolve_text(expression, index, mode="string")

    def resolve_callable_strings(
        self,
        expression: str,
        index: int,
    ) -> ResolvedValue:
        """Resolve the identity of a Python callable used by an expression.

        Algorithm-selection APIs sometimes receive a constructed object through
        a local variable, for example ``mode = modes.CBC(iv)`` followed by
        ``Cipher(..., mode)``.  Ordinary string resolution follows the value of
        ``mode`` and therefore sees the constructor result, not the constructor
        identity.  This bounded resolver follows assignments and parameters but
        unwraps calls to the callable that produced the value.
        """

        try:
            node = ast.parse(expression.strip(), mode="eval").body
        except SyntaxError:
            return ResolvedValue(
                expression,
                "unknown",
                unresolved_reason="expression is not independently parseable",
            )
        return self._resolve_callable_node(
            node,
            index,
            seen=frozenset(),
        )

    def resolve_ints(self, expression: str, index: int) -> ResolvedValue:
        return self._resolve_text(expression, index, mode="int")

    def native_random_call_name(self, expression: str, index: int) -> str | None:
        """Resolve a call target through source bindings, not variable spelling.

        Instance methods are class-qualified (``random.SystemRandom.randbytes``).
        A bare instance is not callable. Unknown/custom receivers remain unknown;
        type annotations alone never establish a native generator identity.
        """
        try:
            node = ast.parse(expression.strip(), mode="eval").body
        except SyntaxError:
            return None
        identity = self._native_random_identity(node, index, frozenset())
        if identity is not None and identity[0] in {"callable", "constructor"}:
            return identity[1]
        return None

    def native_random_call_origin(self, expression: str, index: int) -> ValueOrigin | None:
        name = self.native_random_call_name(expression, index)
        if name is None:
            return None
        methods = {
            "randbytes", "getrandbits", "randint", "randrange", "random",
            "choice", "choices", "sample", "uniform",
        }
        if name in {"random.Random", "random.SystemRandom", "secrets.SystemRandom"}:
            return "non_csprng" if name == "random.Random" else "secure_random"
        owner, _, method = name.rpartition(".")
        if owner in {"random", "random.Random", "random.SystemRandom", "secrets.SystemRandom"}:
            if method in methods:
                return "secure_random" if owner.endswith("SystemRandom") else "non_csprng"
            return None
        if name in {
            "os.urandom", "os.getrandom", "uuid.uuid4", "secrets.choice",
            "secrets.token_bytes", "secrets.token_hex", "secrets.token_urlsafe",
            "secrets.randbits", "secrets.randbelow",
        }:
            return "secure_random"
        return None

    def _native_random_identity(
        self, node: ast.AST, index: int, seen: frozenset[tuple[str, int]],
        method: str | None = None,
    ) -> tuple[str, str] | None:
        expression = ast.unparse(node)
        key = (expression, index)
        if key in seen or len(seen) >= 16:
            return None
        seen = seen | {key}
        containing = self._containing_function(index)
        if isinstance(node, ast.BoolOp):
            # Native modules, functions and unmodified RNG instances are
            # truthy. Resolve only a source-proven selected operand: a name
            # such as ``random`` can also be a parameter bound to None.
            for position, value in enumerate(node.values):
                identity = self._native_random_identity(value, index, seen, method)
                if position == len(node.values) - 1:
                    return identity
                if isinstance(value, ast.Constant):
                    truth = bool(value.value)
                elif identity is not None and all(
                    self._native_random_identity(value, index, seen, protocol) is not None
                    for protocol in ("__bool__", "__len__")
                ):
                    truth = True
                else:
                    return None
                if truth == isinstance(node.op, ast.Or):
                    return identity
            return None
        if isinstance(node, (ast.Name, ast.Attribute)):
            assignments = self._reaching_assignments(expression, index)
            if containing is None:
                # Base classes and class fields are evaluated when the class is
                # defined, before later module-level rebinding takes effect.
                assignments = tuple(item for item in assignments if item[1] < index)
            if isinstance(node, ast.Name):
                # Class attributes are not bare-name globals in method bodies,
                # and an attribute in another class cannot shadow an import.
                assignments = tuple(
                    item for item in assignments
                    if self._enclosing_class(item[0]) is None
                    or (containing is not None and self._enclosing_function(item[0]) is containing)
                )
            if isinstance(node, ast.Attribute) and expression.startswith(("self.", "cls.")):
                assigned = self._python_instance_attribute_assignment(expression, index)
                if assigned is not None:
                    assignments = (assigned,)
            if assignments:
                identities = {
                    self._native_random_identity(value, position, seen, method)
                    for value, position in assignments
                }
                return next(iter(identities)) if len(identities) == 1 else None
        if isinstance(node, ast.Name):
            # A local parameter/assignment/definition shadows a module import,
            # including an assignment after the textual use (UnboundLocalError).
            if containing is not None:
                if node.id in self._function_parameter_names(containing):
                    owner = self._enclosing_class(containing)
                    if node.id == "self" and owner is not None and method is not None:
                        inherited = self._native_random_inherited_base(owner, seen, method)
                        if inherited is not None:
                            return "instance", inherited
                    return self._native_random_parameter_identity(
                        node.id, containing, seen, method,
                    )
                if any(
                    self._enclosing_function(value) is containing
                    for value, _ in self._assignments.get(node.id, ())
                ):
                    return None
            definitions = [
                item for item in (*self._function_nodes, *self._class_nodes)
                if item.name == node.id
                and self._enclosing_function(item) in {None, containing}
                and self._enclosing_class(item) is None
            ]
            if definitions:
                return None
            imports: list[tuple[int, str]] = []
            for item in ast.walk(self.tree):
                if not isinstance(item, (ast.Import, ast.ImportFrom)):
                    continue
                scope = self._enclosing_function(item)
                if scope not in {None, containing}:
                    continue
                if self._node_span(item)[0] >= index:
                    continue
                for alias in item.names:
                    local = alias.asname or (alias.name.split(".")[0] if isinstance(item, ast.Import) else alias.name)
                    if local == node.id:
                        canonical = alias.name if isinstance(item, ast.Import) else f"{item.module}.{alias.name}"
                        imports.append((self._node_span(item)[0], canonical))
            if not imports:
                return None
            canonical = max(imports)[1]
            if canonical in {"random", "secrets", "os", "uuid"}:
                return "module", canonical
            if canonical in {"random.Random", "random.SystemRandom", "secrets.SystemRandom"}:
                return "constructor", canonical
            if canonical.split(".")[0] in {"random", "secrets", "os", "uuid"}:
                return "callable", canonical
            return None
        if isinstance(node, ast.Attribute):
            field_value = self._python_plain_field_binding(node, index)
            if field_value is not None:
                return self._native_random_identity(*field_value, seen, method)
            class_value = self._native_random_class_attribute(node, index)
            if class_value is not None:
                return self._native_random_identity(*class_value, seen, method)
            receiver = self._native_random_identity(node.value, index, seen, node.attr)
            if receiver is None:
                return None
            kind, name = receiver
            if kind in {"module", "instance", "callable"}:
                # A known native function used as an object (urandom.randbytes)
                # has no RNG method. Preserve its qualified identity so origin
                # tracing rejects that call instead of inheriting the function's
                # entropy while inspecting an unresolved receiver.
                canonical = f"{name}.{node.attr}"
                if canonical in {"random.Random", "random.SystemRandom", "secrets.SystemRandom"}:
                    return "constructor", canonical
                return "callable", canonical
            return None
        if isinstance(node, ast.Call):
            called = self._native_random_identity(node.func, index, seen)
            if called is not None and called[0] == "constructor":
                return "instance", called[1]
            if isinstance(node.func, ast.Name) and not self._reaching_assignments(node.func.id, index):
                classes = [
                    c for c in self._class_nodes if c.name == node.func.id
                    and self._enclosing_function(c) in {None, containing}
                ]
                if (
                    self._functions_by_name.get(node.func.id)
                    or (containing is not None and node.func.id in self._function_parameter_names(containing))
                ):
                    classes = []
                if len(classes) == 1:
                    inherited = self._native_random_inherited_base(classes[0], seen, method)
                    if inherited is not None:
                        return "instance", inherited
            # Simple source-defined factories, including class/static methods.
            name = _python_name(node.func).rsplit(".", 1)[-1]
            functions = self._functions_by_name.get(name, ())
            if len(functions) != 1 or node.args or node.keywords:
                return None
            function = functions[0]
            if isinstance(node.func, ast.Name):
                if self._enclosing_class(function) is not None:
                    return None
                if self._enclosing_function(function) not in {None, containing}:
                    return None
                if self._reaching_assignments(node.func.id, index):
                    return None
            else:
                owner = self._enclosing_class(function)
                if owner is None or not isinstance(node.func, ast.Attribute):
                    return None
                if not isinstance(node.func.value, ast.Name):
                    return None
                if node.func.value.id == owner.name:
                    if self._reaching_assignments(owner.name, index):
                        return None
                    if not any(_python_name(d) in {"classmethod", "staticmethod"} for d in function.decorator_list):
                        return None
                elif node.func.value.id == "self":
                    # One plain instance factory in the same class. Its body
                    # must only return a value; inherited/dynamic dispatch and
                    # patched methods are outside this small identity rule.
                    parameters = (*function.args.posonlyargs, *function.args.args)
                    guarded = {name, "__new__", "__getattribute__", "__getattr__", "__setattr__", "__delattr__"}
                    if (
                        containing is None or self._parents.get(containing) is not owner
                        or self._parents.get(function) is not owner
                        or not isinstance(function, ast.FunctionDef)
                        or owner.bases or owner.decorator_list or owner.keywords
                        or function.decorator_list or containing.decorator_list
                        or len(parameters) != 1 or parameters[0].arg != "self"
                        or function.args.vararg or function.args.kwarg or function.args.kwonlyargs
                        or not (*containing.args.posonlyargs, *containing.args.args)
                        or (*containing.args.posonlyargs, *containing.args.args)[0].arg != "self"
                        or any(self._enclosing_function(value) is containing for value, _ in self._assignments.get("self", ()))
                        or any(isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name in guarded - {name} for s in owner.body)
                        or any(expression.endswith(f".{attribute}") for expression in self._assignments for attribute in guarded)
                        or any(self._enclosing_class(value) is owner for attribute in guarded for value, _ in self._assignments.get(attribute, ()))
                        or any(_python_name(base) == owner.name for c in self._class_nodes for base in c.bases)
                        or any(isinstance(c.func, ast.Name) and c.func.id in {"setattr", "delattr"} for c in self._call_nodes)
                        or any(not isinstance(s, ast.Return) and not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)) for s in function.body)
                    ):
                        return None
                else:
                    return None
            returns = [item for item in ast.walk(function) if isinstance(item, ast.Return) and self._enclosing_function(item) is function]
            if len(returns) != 1 or returns[0].value is None or returns[0] not in function.body:
                return None
            returned = returns[0].value
            return self._native_random_identity(returned, self._node_span(returned)[0], seen, method)
        return None

    def _native_random_parameter_identity(
        self,
        name: str,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        seen: frozenset[tuple[str, int]],
        method: str | None,
    ) -> tuple[str, str] | None:
        """Bind a plain helper's RNG parameter from its actual live callers.

        Printing versus returning a generated value must not change its RNG
        identity. An annotation is not evidence; all explicit caller arguments
        must resolve to the same native identity without rebinding the helper
        or its parameter. Plain constructor defaults are evaluated in their
        definition scope; dynamic argument expansion stays unknown.
        """
        owner = self._enclosing_class(function)
        plain_constructor = (
            owner is not None and function.name == "__init__"
            and self._python_plain_class(owner)
            and function not in self._shadowed_methods
        )
        if (
            not isinstance(function, ast.FunctionDef)
            or function.decorator_list
            or (owner is not None and not plain_constructor)
            or self._enclosing_function(function) is not None
            or self._assignments.get(function.name)
            or any(
                self._enclosing_function(value) is function
                for value, _ in self._assignments.get(name, ())
            )
        ):
            return None
        parameters = self._function_parameter_names(function)
        identities: set[tuple[str, str]] = set()
        for caller in self._call_nodes:
            if not self._external_call_targets_function(caller, function):
                continue
            if (
                not self._python_literal_boolean_path_is_reachable(caller)
                or not self._python_call_has_entry_path(caller, seen=frozenset())
            ):
                continue
            if any(isinstance(arg, ast.Starred) for arg in caller.args) or any(
                keyword.arg is None for keyword in caller.keywords
            ):
                return None
            argument = self._call_argument_for_parameter(
                caller, function, parameters.index(name), name,
            )
            if argument is None:
                return None
            explicit = argument in (*caller.args, *(keyword.value for keyword in caller.keywords))
            if not explicit and not plain_constructor:
                return None
            if not explicit and self._python_default_has_class_shadow(argument, function):
                return None
            argument_index = (
                self._node_span(caller)[0] if explicit
                else self._node_span(function)[0] - 1
            )
            identity = self._native_random_identity(
                argument, argument_index, seen, method,
            )
            if identity is None:
                return None
            identities.add(identity)
        return next(iter(identities)) if len(identities) == 1 else None

    def _python_default_has_class_shadow(
        self, default: ast.AST, function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        """Do not mistake class-local default bindings for module imports.

        The native identity resolver models function-body names. Defaults are
        instead evaluated in the surrounding class namespace; class-local
        bindings remain unresolved by this small constructor-default rule.
        """
        owner = self._enclosing_class(function)
        if owner is None:
            return False
        references = {n.id for n in ast.walk(default) if isinstance(n, ast.Name)}
        for node in ast.walk(owner):
            if (
                self._enclosing_function(node) is not None
                or self._enclosing_class(node) is not owner
                or self._node_span(node)[0] >= self._node_span(function)[0]
            ):
                continue
            names: set[str] = set()
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.asname or a.name.split(".")[0] for a in node.names)
            if references & names:
                return True
        return False

    def _python_plain_class(self, owner: ast.ClassDef) -> bool:
        """Small class subset with ordinary field lookup and construction."""
        dynamic = {"__new__", "__getattribute__", "__getattr__", "__setattr__", "__delattr__"}
        return not (
            owner.bases or owner.decorator_list or owner.keywords
            or self._assignments.get(owner.name)
            or sum(c.name == owner.name for c in self._class_nodes) != 1
            or any(isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name in dynamic for s in owner.body)
            or any(self._enclosing_class(value) is owner for name in dynamic for value, _ in self._assignments.get(name, ()))
            or any(name.endswith(f".{attribute}") for name in self._assignments for attribute in dynamic)
            or any(_python_name(base) == owner.name for c in self._class_nodes for base in c.bases)
            or any(_python_name(c.func).rsplit(".", 1)[-1] in {"setattr", "delattr"} for c in self._call_nodes)
        )

    def _python_plain_field_binding(
        self, node: ast.Attribute, index: int,
    ) -> tuple[ast.AST, int] | None:
        """Follow one unchanged field of one source-proven local instance.

        Reuses constructor provenance for owned child objects and native RNG
        fields. Conditional/descriptor/mutated fields remain unresolved.
        """
        key = (ast.unparse(node), index)
        if key in self._plain_field_resolutions or len(self._plain_field_resolutions) >= 8:
            return None
        # Own self-fields already have scope-sensitive assignment handling.
        if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
            return None
        self._plain_field_resolutions.add(key)
        try:
            constructions = self._python_constructions(node.value, index)
            if len(constructions) != 1:
                return None
            construction = constructions[0]
            if not isinstance(construction.func, ast.Name):
                return None
            owners = [c for c in self._class_nodes if c.name == construction.func.id]
            if len(owners) != 1 or not self._python_plain_class(owners[0]):
                return None
            owner = owners[0]
            containing = self._enclosing_function(construction)
            if containing is not None and owner.name in self._function_parameter_names(containing):
                return None
            if any(isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == node.attr for s in owner.body):
                return None
            constructors = [
                s for s in owner.body if isinstance(s, ast.FunctionDef)
                and s.name == "__init__" and s not in self._shadowed_methods
            ]
            if len(constructors) != 1 or constructors[0].decorator_list:
                return None
            constructor = constructors[0]
            assignments = [
                (value, position) for value, position in self._assignments.get(f"self.{node.attr}", ())
                if self._enclosing_class(value) is owner
            ]
            if len(assignments) != 1 or self._enclosing_function(assignments[0][0]) is not constructor:
                return None
            value, position = assignments[0]
            if not any(
                isinstance(s, (ast.Assign, ast.AnnAssign)) and s.value is value
                for s in constructor.body
            ):
                return None
            if any(
                name.endswith(f".{node.attr}") and not name.startswith(("self.", "cls."))
                for name in self._assignments
            ) or any(self._enclosing_class(v) is owner for v, _ in self._assignments.get(node.attr, ())):
                return None
            if any(
                isinstance(n, (ast.AugAssign, ast.Delete))
                and any(isinstance(t, ast.Attribute) and t.attr == node.attr for t in ast.walk(n))
                for n in ast.walk(self.tree) if n not in self._inactive_method_body_nodes
            ):
                return None
            parameters = self._function_parameter_names(constructor)
            if isinstance(value, ast.Name) and value.id in parameters:
                if any(isinstance(arg, ast.Starred) for arg in construction.args) or any(k.arg is None for k in construction.keywords):
                    return None
                argument = self._call_argument_for_parameter(construction, constructor, parameters.index(value.id), value.id)
                if argument is None:
                    return None
                explicit = argument in (*construction.args, *(k.value for k in construction.keywords))
                if not explicit and self._python_default_has_class_shadow(argument, constructor):
                    return None
                return argument, (
                    self._node_span(construction)[0] if explicit
                    else self._node_span(constructor)[0] - 1
                )
            return value, position
        finally:
            self._plain_field_resolutions.remove(key)

    def _native_random_class_attribute(
        self, node: ast.Attribute, index: int,
    ) -> tuple[ast.AST, int] | None:
        """Follow one unchanged class-held generator, without descriptor inference."""
        if not isinstance(node.value, ast.Name):
            return None
        containing = self._containing_function(index)
        if node.value.id in {"self", "cls"}:
            owner = self._enclosing_class(containing) if containing is not None else None
        else:
            definitions = [c for c in self._class_nodes if c.name == node.value.id]
            owner = definitions[0] if len(definitions) == 1 else None
            if self._reaching_assignments(node.value.id, index):
                return None
            if containing is not None and node.value.id in self._function_parameter_names(containing):
                return None
        if owner is None or owner.bases or owner.decorator_list or owner.keywords:
            return None
        guarded = {node.attr, "__getattribute__", "__getattr__", "__setattr__", "__delattr__"}
        if any(isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name in guarded for s in owner.body):
            return None
        if any(
            self._assignments.get(f"{receiver}.{attribute}")
            for receiver in (owner.name, "self", "cls") for attribute in guarded
        ):
            return None
        # Dynamic mutation or subclass dispatch cannot be established by this
        # small class-field rule. Direct instance assignments are guarded too.
        if any(
            isinstance(c.func, ast.Name) and c.func.id in {"setattr", "delattr"}
            for c in self._call_nodes
        ) or any(
            _python_name(base) == owner.name for c in self._class_nodes for base in c.bases
        ):
            return None
        for expression, assignments in self._assignments.items():
            if not expression.endswith(f".{node.attr}"):
                continue
            for _, position in assignments:
                receiver = ast.parse(expression.rsplit(".", 1)[0], mode="eval").body
                if any(_python_name(c.func) == owner.name for c in self._python_constructions(receiver, position)):
                    return None
        values = [
            s.value for s in owner.body
            if isinstance(s, (ast.Assign, ast.AnnAssign)) and s.value is not None
            and any(isinstance(t, ast.Name) and t.id == node.attr for t in (s.targets if isinstance(s, ast.Assign) else [s.target]))
        ]
        if len(values) != 1:
            return None
        return values[0], self._node_span(values[0])[0]

    def _native_random_inherited_base(
        self, definition: ast.ClassDef, seen: frozenset[tuple[str, int]],
        method: str | None = None,
    ) -> str | None:
        """Prove native dispatch for the requested method of one native parent.

        SystemRandom.randbytes calls the OS directly; Random.randbytes dispatches
        through self.getrandbits. Overrides of unrelated methods therefore have
        different consequences for these two byte-producing methods.
        """
        if len(definition.bases) != 1 or definition.decorator_list or definition.keywords:
            return None
        base = self._native_random_identity(
            definition.bases[0], self._node_span(definition)[0], seen,
        )
        if base is None or base[0] != "constructor":
            return None
        dispatch = {
            "randbytes", "getrandbits", "randint", "randrange", "random",
            "choice", "choices", "sample", "uniform", "_randbelow",
            "_randbelow_with_getrandbits", "_randbelow_without_getrandbits",
            "seed", "__init__", "__new__", "__getattribute__", "__getattr__",
            "__setattr__", "__delattr__",
        }
        if method is not None:
            dependencies = {method}
            if method == "randbytes" and base[1] == "random.Random":
                dependencies.add("getrandbits")
            elif method not in {"randbytes", "getrandbits", "random"}:
                dependencies.update({"getrandbits", "random", "_randbelow", "randrange"})
            dispatch = dependencies | {
                "seed", "__init__", "__new__", "__getattribute__", "__getattr__",
                "__setattr__", "__delattr__",
            }
        if any(self._assignments.get(f"{definition.name}.{name}") for name in dispatch):
            return None
        if any(
            self._enclosing_class(value) is definition
            for name in dispatch - {"seed"}
            for receiver in ("self", "cls")
            for value, _ in self._assignments.get(f"{receiver}.{name}", ())
        ):
            return None
        for statement in definition.body:
            if isinstance(statement, ast.Pass) or (
                isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            if not isinstance(statement, ast.FunctionDef) or statement.decorator_list:
                return None
            if statement.name not in dispatch:
                continue
            if statement.name == "__init__" and self._native_random_metadata_constructor(statement, dispatch, seen):
                if method == "seed" and any(
                    isinstance(s, ast.Assign) and any(ast.unparse(t) == "self.seed" for t in s.targets)
                    for s in statement.body
                ):
                    return None
                continue
            if statement.name != "randbytes" or len(statement.body) != 1:
                return None
            returned = statement.body[0]
            arguments = statement.args
            parameters = (*arguments.posonlyargs, *arguments.args)
            if (
                not isinstance(returned, ast.Return) or not isinstance(returned.value, ast.Call)
                or len(parameters) != 2 or arguments.vararg or arguments.kwarg
                or arguments.kwonlyargs or arguments.defaults
            ):
                return None
            delegate = returned.value
            if (
                not isinstance(delegate.func, ast.Attribute) or delegate.func.attr != "randbytes"
                or not isinstance(delegate.func.value, ast.Call)
                or not isinstance(delegate.func.value.func, ast.Name)
                or delegate.func.value.func.id != "super"
                or delegate.func.value.args or delegate.func.value.keywords
                or len(delegate.args) != 1 or delegate.keywords
                or not isinstance(delegate.args[0], ast.Name)
                or delegate.args[0].id != parameters[1].arg
                or self._assignments.get("super") or self._functions_by_name.get("super")
            ):
                return None
        return base[1]

    def _native_random_metadata_constructor(
        self, function: ast.FunctionDef, dispatch: set[str], seen: frozenset[tuple[str, int]],
    ) -> bool:
        """Recognize super initialization followed by simple unused metadata.

        This covers a generated wrapper that stores a seed and another native
        generator but still draws from its unchanged Random parent.
        """
        args = function.args
        if (
            [a.arg for a in (*args.posonlyargs, *args.args)] != ["self"]
            or args.vararg or args.kwarg or args.kwonlyargs or args.defaults
            or not function.body or self._assignments.get("super") or self._functions_by_name.get("super")
        ):
            return False
        first = function.body[0]
        if not (
            isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
            and ast.unparse(first.value) == "super().__init__()"
        ):
            return False
        native_members: set[str] = set()
        for statement in function.body[1:]:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if not (
                    isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == "self" and not target.attr.startswith("__")
                    and target.attr not in dispatch - {"seed"}
                ):
                    return False
                value = statement.value
                if isinstance(value, ast.Constant):
                    continue
                if not isinstance(value, ast.Call):
                    return False
                identity = self._native_random_identity(value.func, self._node_span(value)[0], seen)
                if identity is None:
                    return False
                if identity == ("constructor", "random.SystemRandom") or identity == ("constructor", "secrets.SystemRandom"):
                    if value.args or value.keywords:
                        return False
                    native_members.add(target.attr)
                elif identity[1] not in {"secrets.token_bytes", "os.urandom"} or any(
                    not isinstance(a, ast.Constant) for a in (*value.args, *(k.value for k in value.keywords))
                ):
                    return False
                continue
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                if (
                    isinstance(call.func, ast.Attribute) and call.func.attr == "seed"
                    and isinstance(call.func.value, ast.Attribute)
                    and isinstance(call.func.value.value, ast.Name) and call.func.value.value.id == "self"
                    and call.func.value.attr in native_members and not call.keywords
                    and all(isinstance(a, (ast.Constant, ast.Attribute)) for a in call.args)
                ):
                    continue
            return False
        return True

    def static_byte_length(self, expression: str, index: int) -> int | None:
        try:
            node = ast.parse(expression.strip(), mode="eval").body
        except SyntaxError:
            return None
        direct = self._python_known_byte_length(node, index, seen=frozenset())
        if direct is not None:
            return direct
        resolved = self.resolve_origin(expression, index)
        if resolved.origin != "constant" or not resolved.values:
            return None
        lengths: set[int] = set()
        for value in resolved.values:
            try:
                literal = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return None
            if not isinstance(literal, (bytes, str)):
                return None
            lengths.add(len(literal))
        return next(iter(lengths)) if len(lengths) == 1 else None

    def integer_expression_has_minimum_guard(
        self,
        expression: str,
        index: int,
        minimum: int,
    ) -> bool:
        """Prove that a Python integer reaches ``index`` only above a floor.

        This deliberately recognizes only simple fail-closed guards in the
        same function as the sink. It does not treat a type annotation, CLI
        default, or arbitrary validation helper as proof of a security bound.
        """

        try:
            parsed = ast.parse(expression.strip(), mode="eval").body
        except SyntaxError:
            return False
        if (
            isinstance(parsed, ast.Call)
            and isinstance(parsed.func, ast.Name)
            and parsed.func.id == "max"
            and any(
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, int)
                and not isinstance(argument.value, bool)
                and argument.value >= minimum
                for argument in parsed.args
            )
        ):
            return True
        if not isinstance(parsed, (ast.Name, ast.Attribute)):
            return False
        guarded_expression = ast.unparse(parsed)

        sink_node = next(
            (
                node
                for node in self._call_nodes
                if self._node_span(node)[0] == index
            ),
            None,
        )
        if sink_node is None:
            return False
        function = self._enclosing_function(sink_node)
        if function is None:
            return False

        for statement in function.body:
            statement_start, statement_end = self._node_span(statement)
            if statement_start >= index:
                break
            if (
                statement_end <= index
                and isinstance(statement, ast.If)
                and _python_statements_terminate(statement.body)
                and _python_guard_rejects_values_below(
                    statement.test,
                    guarded_expression,
                    minimum,
                )
            ):
                return True
        return False

    def resolve_call_target_strings(self, call: CallExpression) -> ResolvedValue:
        for node in self._call_nodes:
            if self._node_span(node) == (call.start, call.end):
                return self._resolve_node(
                    node.func,
                    call.start,
                    mode="string",
                    seen=frozenset(),
                )
        return super().resolve_call_target_strings(call)

    def _resolve_text(self, expression: str, index: int, *, mode: str) -> ResolvedValue:
        try:
            node = ast.parse(expression.strip(), mode="eval").body
        except SyntaxError:
            return ResolvedValue(
                expression,
                "unknown",
                unresolved_reason="expression is not independently parseable",
            )
        return self._resolve_node(node, index, mode=mode, seen=frozenset())

    def _resolve_callable_node(
        self,
        node: ast.AST,
        index: int,
        *,
        seen: frozenset[str],
    ) -> ResolvedValue:
        expression = ast.unparse(node)

        if isinstance(node, ast.Call):
            resolved_call = self._resolve_node(
                node,
                index,
                mode="string",
                seen=seen,
            )
            if resolved_call.values:
                return resolved_call
            return self._resolve_callable_node(
                node.func,
                index,
                seen=seen,
            )

        if isinstance(node, ast.Name):
            key = f"callable:{node.id}:{index}"
            if key not in seen:
                assignments = self._reaching_assignments(node.id, index)
                if assignments:
                    values = tuple(
                        self._resolve_callable_node(
                            assigned_node,
                            assigned_index,
                            seen=seen | frozenset({key}),
                        )
                        for assigned_node, assigned_index in assignments
                    )
                    return _prepend_resolution(
                        expression,
                        f"resolve callable assignment {node.id}",
                        _merge_resolved(expression, values),
                        ResolutionStep(
                            expression,
                            "inspect callable",
                            "string",
                            self.source_span(index, index),
                        ),
                    )

        if isinstance(node, ast.IfExp):
            branches = (
                self._resolve_callable_node(node.body, index, seen=seen),
                self._resolve_callable_node(node.orelse, index, seen=seen),
            )
            return _merge_resolved(expression, branches)

        return self._resolve_node(
            node,
            index,
            mode="string",
            seen=seen,
        )

    def _resolve_node(
        self,
        node: ast.AST,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue:
        if self._resolution_depth >= _MAX_RESOLUTION_STEPS:
            if node in self._parents:
                expression = (
                    ast.get_source_segment(self.source, node)
                    or ast.unparse(node)
                )
            else:
                expression = ast.unparse(node)
            step = ResolutionStep(
                expression,
                "inspect",
                mode,
                self.source_span(index, index),
            )
            return ResolvedValue(
                expression,
                "unknown",
                steps=(step,),
                unresolved_reason="source-resolution budget exhausted",
            )
        self._resolution_depth += 1
        try:
            return self._resolve_node_impl(node, index, mode=mode, seen=seen)
        finally:
            self._resolution_depth -= 1

    def _resolve_node_impl(
        self,
        node: ast.AST,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue:
        # Nodes parsed from a standalone argument expression have line/column
        # offsets relative to that snippet, not to the complete source file.
        # Using those offsets against self.source would produce a plausible but
        # unrelated evidence string from the beginning of the program.
        if node in self._parents:
            expression = ast.get_source_segment(self.source, node) or ast.unparse(node)
        else:
            expression = ast.unparse(node)
        step = ResolutionStep(expression, "inspect", mode, self.source_span(index, index))
        if len(seen) >= _MAX_RESOLUTION_STEPS:
            return ResolvedValue(
                expression,
                "unknown",
                steps=(step,),
                unresolved_reason="source-resolution budget exhausted",
            )
        if mode == "int":
            evaluated = _evaluate_int_ast(node)
            if evaluated is not None:
                return ResolvedValue(
                    expression,
                    "constant",
                    (str(evaluated),),
                    (step,),
                )
        if isinstance(node, ast.Constant):
            if mode == "string" and isinstance(node.value, str):
                return ResolvedValue(expression, "constant", (node.value,), (step,))
            if mode == "int" and isinstance(node.value, int) and not isinstance(node.value, bool):
                return ResolvedValue(expression, "constant", (str(node.value),), (step,))
            if mode == "origin" and isinstance(node.value, (str, bytes, int, float)):
                return ResolvedValue(expression, "constant", (repr(node.value),), (step,))

        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            if mode != "origin":
                return ResolvedValue(expression, "constant", steps=(step,))
            elements: tuple[ast.AST, ...]
            if isinstance(node, ast.Dict):
                elements = tuple(node.values)
            else:
                elements = tuple(node.elts)
            if not elements:
                return ResolvedValue(expression, "constant", steps=(step,))
            resolved_elements = tuple(
                self._resolve_node(
                    element,
                    index,
                    mode=mode,
                    seen=seen,
                )
                for element in elements
            )
            return _prepend_resolution(
                expression,
                "combine container element sources",
                _merge_transformed_origins(expression, resolved_elements),
                step,
            )

        if isinstance(node, ast.JoinedStr) and mode == "origin":
            components = tuple(
                self._resolve_node(
                    value.value if isinstance(value, ast.FormattedValue) else value,
                    index,
                    mode=mode,
                    seen=seen,
                )
                for value in node.values
            )
            return _prepend_resolution(
                expression,
                "combine formatted string sources",
                _merge_transformed_origins(expression, components),
                step,
            )

        if isinstance(node, ast.NamedExpr):
            resolved = self._resolve_node(
                node.value,
                index,
                mode=mode,
                seen=seen,
            )
            return _prepend_resolution(
                expression,
                "resolve assignment expression",
                resolved,
                step,
            )

        if isinstance(node, ast.Name):
            imported = self.import_aliases.get(node.id)
            if imported is not None:
                direct = _direct_python_name(imported, mode)
                if direct is not None:
                    return ResolvedValue(expression, direct[0], direct[1], (step,))
            containing = self._containing_function(index)
            scope = self._node_span(containing)[0] if containing is not None else "global"
            key = f"variable:{scope}:{node.id}"
            if mode == "string":
                branch_values = self._python_enclosing_string_equality_values(
                    node.id,
                    index,
                )
                if branch_values:
                    return ResolvedValue(
                        expression,
                        "constant",
                        branch_values,
                        (step,),
                    )
                guarded_values = self._python_fail_closed_string_guard_values(
                    node.id,
                    index,
                )
                if guarded_values:
                    reaching = self._reaching_assignments(node.id, index)
                    fixed_sources = tuple(
                        self._resolve_node(
                            assigned,
                            assigned_index,
                            mode=mode,
                            seen=seen | frozenset({key}),
                        )
                        for assigned, assigned_index in reaching
                    )
                    if fixed_sources and all(
                        _python_resolved_is_fixed_for_branch(value)
                        for value in fixed_sources
                    ):
                        fixed = _python_restrict_resolved_values(
                            _merge_resolved(expression, fixed_sources),
                            guarded_values,
                        )
                        return _prepend_resolution(
                            expression,
                            "resolve fixed value before fail-closed whitelist",
                            fixed,
                            step,
                        )
                    if not reaching:
                        parameter = self._resolve_parameter(
                            node.id,
                            index,
                            mode=mode,
                            seen=seen | frozenset({key}),
                        )
                        if parameter is not None and _python_resolved_has_only_fixed_values(
                            parameter
                        ):
                            return _prepend_resolution(
                                expression,
                                "resolve fixed caller before fail-closed whitelist",
                                _python_restrict_resolved_values(
                                    parameter,
                                    guarded_values,
                                ),
                                step,
                            )
                    return ResolvedValue(
                        expression,
                        "constant",
                        guarded_values,
                        (
                            step,
                            ResolutionStep(
                                expression,
                                "resolve fail-closed string whitelist",
                                "constant",
                                self.source_span(index, index),
                            ),
                        ),
                    )
            if key in seen:
                prior_assignment = self._latest_assignment(node.id, index)
                if prior_assignment is not None:
                    resolved_prior = self._resolve_node(
                        prior_assignment[0],
                        prior_assignment[1],
                        mode=mode,
                        seen=seen,
                    )
                    return _prepend_resolution(
                        expression,
                        "resolve prior assignment during self-transform",
                        resolved_prior,
                        step,
                    )
                resolved_parameter = self._resolve_parameter(
                    node.id,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if resolved_parameter is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve self-transform from caller argument",
                        resolved_parameter,
                        step,
                    )
                return ResolvedValue(expression, "unknown", steps=(step,), unresolved_reason="cyclic assignment")
            if mode == "origin":
                equivalent = self._resolve_python_guarded_equivalent_origin(
                    node.id,
                    index,
                    seen=seen | frozenset({key}),
                )
                if equivalent is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve value proven equal by a fail-closed guard",
                        equivalent,
                        step,
                    )
            free_global = self._resolve_python_free_global_at_callers(
                node.id,
                index,
                mode=mode,
                seen=seen | frozenset({key}),
            )
            if free_global is not None:
                return _prepend_resolution(
                    expression,
                    "resolve free global at function callsite",
                    free_global,
                    step,
                )
            reaching_assignments = self._reaching_assignments(node.id, index)
            exclude_empty_assignment_paths = (
                mode == "origin"
                and (
                    self._python_fail_closed_nonempty_guard(node.id, index)
                    or self._python_reference_is_truthy_on_path(node.id, index)
                    or self._python_empty_value_replaced_before(node.id, index)
                )
            )
            exclude_empty_string_paths = (
                mode == "string"
                and (
                    self._python_fail_closed_nonempty_guard(node.id, index)
                    or self._python_reference_is_truthy_on_path(node.id, index)
                )
            )

            def resolve_assignment_source(
                assigned_node: ast.AST,
                assigned_index: int,
            ) -> ResolvedValue | None:
                if exclude_empty_assignment_paths:
                    return self._resolve_python_origin_without_empty_paths(
                        assigned_node,
                        assigned_index,
                        seen=seen | frozenset({key}),
                    )
                resolved = self._resolve_node(
                    assigned_node,
                    assigned_index,
                    mode=mode,
                    seen=seen | frozenset({key}),
                )
                return (
                    _resolved_without_empty_alternatives(resolved)
                    if exclude_empty_string_paths
                    else resolved
                )

            assignment = (
                reaching_assignments[-1]
                if reaching_assignments
                else None
            )
            mutation = self._latest_subscript_assignment(
                node.id,
                ("slice", None),
                index,
            )
            if mutation is not None and (
                assignment is None or mutation[1] > assignment[1]
            ):
                resolved = self._resolve_node(
                    mutation[0],
                    mutation[1],
                    mode=mode,
                    seen=seen | frozenset({key}),
                )
                return _prepend_resolution(
                    expression,
                    f"resolve full-slice assignment {node.id}",
                    resolved,
                    step,
                )
            indexed_mutations = [
                item
                for item in self._subscript_assignments.get(
                    (node.id, ("dynamic", None)),
                    (),
                )
                if self._enclosing_function(item[0]) is containing
                and item[1] <= index
                and (assignment is None or item[1] > assignment[1])
            ]
            if mode == "origin" and indexed_mutations:
                components: list[ResolvedValue] = []
                if assignment is not None:
                    assigned_node, assigned_index = assignment
                    components.append(
                        self._resolve_node(
                            assigned_node,
                            assigned_index,
                            mode=mode,
                            seen=seen | frozenset({key}),
                        )
                    )
                components.extend(
                    self._resolve_node(
                        assigned_node,
                        assigned_index,
                        mode=mode,
                        seen=seen | frozenset({key}),
                    )
                    for assigned_node, assigned_index in indexed_mutations
                )
                return _prepend_resolution(
                    expression,
                    f"combine indexed writes to {node.id}",
                    _merge_transformed_origins(expression, components),
                    step,
                )
            if len(reaching_assignments) > 1:
                resolved_assignments = tuple(
                    resolved
                    for assigned_node, assigned_index in reaching_assignments
                    if (
                        resolved := resolve_assignment_source(
                            assigned_node,
                            assigned_index,
                        )
                    )
                    is not None
                )
                if not resolved_assignments:
                    return ResolvedValue(
                        expression,
                        "unknown",
                        steps=(step,),
                        unresolved_reason=(
                            "all reaching assignment paths are rejected by "
                            "a fail-closed nonempty guard"
                        ),
                    )
                return _prepend_resolution(
                    expression,
                    f"merge conditional assignment sources for {node.id}",
                    _merge_resolved(expression, resolved_assignments),
                    step,
                )
            if assignment is not None:
                assigned_node, assigned_index = assignment
                resolved = resolve_assignment_source(
                    assigned_node,
                    assigned_index,
                )
                if resolved is None:
                    return ResolvedValue(
                        expression,
                        "unknown",
                        steps=(step,),
                        unresolved_reason=(
                            "reaching assignment is rejected by a fail-closed "
                            "nonempty guard"
                        ),
                    )
                return _prepend_resolution(expression, f"resolve assignment {node.id}", resolved, step)
            loop_value = (
                self._resolve_python_for_binding(
                    node.id,
                    index,
                    mode=mode,
                    seen=seen | frozenset({key}),
                )
                if mode in {"origin", "int"}
                else None
            )
            if loop_value is not None:
                return _prepend_resolution(
                    expression,
                    "resolve for-loop iterable origin",
                    loop_value,
                    step,
                )
            resolved_parameter = self._resolve_parameter(node.id, index, mode=mode, seen=seen)
            if resolved_parameter is not None:
                return _prepend_resolution(expression, "resolve caller arguments", resolved_parameter, step)

        if isinstance(node, (ast.Attribute, ast.Subscript)):
            name = _python_name(node)
            canonical_name = _canonical_python_name(name, self.import_aliases)
            if mode == "string":
                guarded_values = self._python_fail_closed_string_guard_values(
                    name,
                    index,
                )
                if guarded_values:
                    return ResolvedValue(
                        expression,
                        "constant",
                        guarded_values,
                        (
                            step,
                            ResolutionStep(
                                expression,
                                "resolve fail-closed string whitelist",
                                "constant",
                                self.source_span(index, index),
                            ),
                        ),
                    )
            if isinstance(node, ast.Subscript):
                local_element = self._resolve_python_subscript(
                    node,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if local_element is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve indexed container element",
                        local_element,
                        step,
                    )
            direct = _direct_python_name(canonical_name, mode)
            if direct is not None:
                return ResolvedValue(expression, direct[0], direct[1], (step,))
            if isinstance(node, ast.Attribute):
                if mode == "origin" and node.attr == "key":
                    cipher_key = self._resolve_python_cipher_key_property(
                        node.value,
                        index,
                        seen=seen,
                    )
                    if cipher_key is not None:
                        return _prepend_resolution(
                            expression,
                            "resolve cipher key property",
                            cipher_key,
                            step,
                        )
                    if (
                        isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "private_bytes"
                    ):
                        serialized = self._resolve_node(
                            node.value,
                            index,
                            mode=mode,
                            seen=seen,
                        )
                        if serialized.origin != "unknown":
                            return _prepend_resolution(
                                expression,
                                "unwrap serialized key attribute",
                                serialized,
                                step,
                            )
                if mode == "string" and node.attr == "name":
                    receiver_value = self._resolve_node(
                        node.value,
                        index,
                        mode=mode,
                        seen=seen,
                    )
                    if receiver_value.values:
                        return _prepend_resolution(
                            expression,
                            "resolve algorithm name property",
                            receiver_value,
                            step,
                        )
                destructured_value = self._resolve_python_destructured_return_attribute(
                    node,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if destructured_value is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve destructured helper return attribute",
                        destructured_value,
                        step,
                    )
                argparse_value = self._resolve_argparse_attribute(
                    node,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if argparse_value is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve argparse input",
                        argparse_value,
                        step,
                    )
                mutated_value = self._resolve_python_prior_instance_mutations(
                    node,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if mutated_value is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve prior instance-state mutations",
                        mutated_value,
                        step,
                    )
                constructed_value = self._resolve_constructed_attribute(
                    node,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if constructed_value is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve constructed instance attribute",
                        constructed_value,
                        step,
                    )
            if isinstance(node, ast.Attribute) and name.startswith(("self.", "cls.")):
                assignment = self._python_instance_attribute_assignment(
                    name,
                    index,
                )
                if assignment is not None:
                    key = f"attribute:{name}"
                    if key not in seen:
                        resolved = self._resolve_node(
                            assignment[0],
                            assignment[1],
                            mode=mode,
                            seen=seen | frozenset({key}),
                        )
                        return _prepend_resolution(
                            expression,
                            f"resolve instance attribute {name}",
                            resolved,
                            step,
                        )
            if isinstance(node, ast.Subscript):
                element = self._resolve_python_subscript(
                    node,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if element is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve indexed container element",
                        element,
                        step,
                    )
                selector = _python_subscript_selector(node.slice)
                container_name = _python_name(node.value).rsplit(".", 1)[-1]
                if (
                    mode == "origin"
                    and selector is not None
                    and selector[0] == "index"
                    and str(selector[1]).lower()
                    in {"salt", "saltb64", "salt_b64", "salt_b64u", "salt_b64url"}
                    and re.search(
                        r"(?i)(?:blob|record|payload|envelope|parsed|decoded|loaded)",
                        container_name,
                    )
                ):
                    return ResolvedValue(
                        expression,
                        "runtime_input",
                        steps=(
                            step,
                            ResolutionStep(
                                expression,
                                "resolve persisted record salt field",
                                "runtime_input",
                                self.source_span(*self._node_span(node)),
                            ),
                        ),
                    )
                indexed = self._resolve_node(
                    node.value,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if indexed.origin != "unknown":
                    return _prepend_resolution(
                        expression,
                        "resolve indexed value source",
                        indexed,
                        step,
                    )

        if isinstance(node, ast.Call):
            name = _canonical_python_name(
                _python_name(node.func),
                self.import_aliases,
            )
            if mode == "int" and name.rsplit(".", 1)[-1] == "max" and node.args:
                arguments = tuple(
                    self._resolve_node(
                        argument,
                        index,
                        mode=mode,
                        seen=seen,
                    )
                    for argument in node.args
                )
                literal_bounds = tuple(
                    value
                    for argument in node.args
                    if (value := _evaluate_int_ast(argument)) is not None
                )
                merged = _merge_resolved(expression, arguments)
                if literal_bounds:
                    lower_bound = max(literal_bounds)
                    bound_step = ResolutionStep(
                        expression,
                        "apply numeric maximum lower bound",
                        str(lower_bound),
                        self.source_span(index, index),
                    )
                    return ResolvedValue(
                        expression,
                        merged.origin,
                        merged.values,
                        (*merged.steps, bound_step),
                        merged.unresolved_reason,
                    )
                return _prepend_resolution(
                    expression,
                    "resolve numeric maximum arguments",
                    merged,
                    step,
                )
            if mode == "origin":
                crypto_payload = self._resolve_python_crypto_payload_call(
                    node,
                    index,
                    seen=seen,
                )
                if crypto_payload is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve cryptographic payload transformation",
                        crypto_payload,
                        step,
                    )
                stateful_digest = self._resolve_python_stateful_digest(
                    node,
                    index,
                    seen=seen,
                )
                if stateful_digest is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve stateful digest inputs",
                        stateful_digest,
                        step,
                    )
                selected_callable = self._resolve_python_selected_callable(
                    node,
                    index,
                    seen=seen,
                )
                if selected_callable is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve callable selected from mapping",
                        selected_callable,
                        step,
                    )
                loop_callable = self._resolve_python_loop_callable(
                    node,
                    index,
                    seen=seen,
                )
                if loop_callable is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve callable selected from loop iterable",
                        loop_callable,
                        step,
                    )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
            ):
                selector = _python_subscript_selector(node.args[0])
                selectors = (selector,) if selector is not None else tuple(
                    ("index", value)
                    for value in self._resolve_node(
                        node.args[0],
                        index,
                        mode="string",
                        seen=seen,
                    ).values
                )
                candidates = [
                    selected
                    for candidate_selector in selectors
                    if (
                        selected := self._resolve_python_container_element(
                            node.func.value,
                            candidate_selector,
                            index,
                            mode=mode,
                            seen=seen,
                        )
                    )
                    is not None
                ]
                if not candidates and selector is None and mode == "string":
                    candidates.extend(
                        self._resolve_node(
                            candidate,
                            index,
                            mode=mode,
                            seen=seen,
                        )
                        for candidate in self._python_mapping_values(
                            node.func.value,
                            index,
                        )
                    )
                if len(node.args) >= 2:
                    candidates.append(
                        self._resolve_node(
                            node.args[1],
                            index,
                            mode=mode,
                            seen=seen,
                        )
                    )
                if candidates:
                    return _prepend_resolution(
                        expression,
                        "resolve mapping get",
                        _merge_resolved(expression, candidates),
                        step,
                    )
            if mode == "origin" and name in {"json.load", "pickle.load"} and node.args:
                opening = self._python_runtime_stream_open_call(node.args[0], index)
                if opening is not None:
                    return ResolvedValue(
                        expression,
                        "runtime_input",
                        steps=(step,),
                    )
            if (
                mode == "origin"
                and isinstance(node.func, ast.Attribute)
                and node.func.attr
                in {
                    "read",
                    "read1",
                    "readall",
                    "readline",
                    "readlines",
                    "read_bytes",
                    "read_text",
                }
            ):
                receiver = node.func.value
                opening = self._python_runtime_stream_open_call(receiver, index)
                if opening is not None:
                    persisted = self._resolve_python_file_round_trip(
                        opening,
                        index,
                        seen=seen,
                        read_call=node,
                    )
                    if persisted is not None:
                        return _prepend_resolution(
                            expression,
                            "resolve persisted file value",
                            persisted,
                            step,
                        )
                    return ResolvedValue(
                        expression,
                        "runtime_input",
                        steps=(step,),
                    )
                if _python_name(receiver) in {"sys.stdin", "sys.stdin.buffer"}:
                    return ResolvedValue(
                        expression,
                        "runtime_input",
                        steps=(step,),
                    )
            if mode == "origin":
                derived = self._resolve_python_derived_origin(
                    node,
                    index,
                    seen=seen,
                )
                if derived is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve deterministic derivation inputs",
                        derived,
                        step,
                    )
            if (
                mode == "string"
                and name == "getattr"
                and len(node.args) >= 2
                and _canonical_python_name(
                    _python_name(node.args[0]),
                    self.import_aliases,
                )
                in {
                    "hashlib",
                    "cryptography.hazmat.primitives.hashes",
                }
            ):
                resolved = self._resolve_node(
                    node.args[1],
                    index,
                    mode="string",
                    seen=seen,
                )
                return _prepend_resolution(
                    expression,
                    "resolve hash callable selected by getattr",
                    resolved,
                    step,
                )
            if (
                mode == "string"
                and name == "hashlib.new"
                and node.args
            ):
                resolved = self._resolve_node(
                    node.args[0],
                    index,
                    mode="string",
                    seen=seen,
                )
                return _prepend_resolution(
                    expression,
                    "resolve hashlib.new algorithm",
                    resolved,
                    step,
                )
            if mode == "string" and isinstance(node.func, ast.Subscript):
                selected_callable = self._resolve_node(
                    node.func,
                    index,
                    mode="string",
                    seen=seen,
                )
                if selected_callable.values:
                    return _prepend_resolution(
                        expression,
                        "resolve callable selected from mapping",
                        selected_callable,
                        step,
                    )
            if mode == "string" and not node.args and not node.keywords:
                selected_callable = self._resolve_node(
                    node.func,
                    index,
                    mode="string",
                    seen=seen,
                )
                if selected_callable.values:
                    return _prepend_resolution(
                        expression,
                        "resolve zero-argument callable",
                        selected_callable,
                        step,
                    )
            native_origin = self.native_random_call_origin(ast.unparse(node.func), index)
            if mode == "origin" and native_origin is not None:
                return ResolvedValue(expression, native_origin, steps=(step,))
            native_name = self.native_random_call_name(ast.unparse(node.func), index)
            if (
                mode == "origin"
                and native_name is not None
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "randbytes"
            ):
                receiver = self._native_random_identity(node.func.value, index, frozenset())
                if (
                    receiver is not None and receiver[0] == "callable"
                    and self.native_random_call_origin(ast.unparse(node.func.value), index) is not None
                ):
                    # Passing os.urandom (or a bound native RNG method) into
                    # a wrapper does not give that function a randbytes method.
                    # Keep this definite failure visible to secondary origin
                    # tracing, which otherwise sees the constructor argument.
                    return ResolvedValue(
                        expression, "unknown",
                        steps=(ResolutionStep(
                            expression, "reject native RNG callable used as generator",
                            "unknown", step.span,
                        ),),
                        unresolved_reason="native RNG function has no randbytes method",
                    )
            if mode == "origin" and native_name is not None and native_name.startswith((
                "random.Random.", "random.SystemRandom.", "secrets.SystemRandom.",
            )):
                return ResolvedValue(
                    expression, "unknown", steps=(step,),
                    unresolved_reason="native generator method has no registered random output",
                )
            direct = _direct_python_call(name, node, mode)
            if name.startswith(("random.", "secrets.")) or name in {"os.urandom", "os.getrandom", "uuid.uuid4"}:
                # A source binding may replace an imported module/class. The
                # lexical API spelling must not override that binding.
                direct = None
            if direct is None and isinstance(node.func, ast.Name):
                alias = self._latest_assignment(node.func.id, index)
                if alias is not None:
                    alias_name = _python_name(alias[0])
                    if alias_name and not alias_name.startswith(("random.", "secrets.", "os.urandom", "os.getrandom")):
                        direct = _direct_python_call(alias_name, node, mode)
            if direct is not None:
                return ResolvedValue(expression, direct[0], direct[1], (step,))
            if (
                mode == "int"
                and node.args
                and name.rsplit(".", 1)[-1]
                in {"int", "round", "floor", "ceil", "trunc"}
            ):
                resolved = self._resolve_node(
                    node.args[0],
                    index,
                    mode=mode,
                    seen=seen,
                )
                return _prepend_resolution(
                    expression,
                    "unwrap numeric conversion",
                    resolved,
                    step,
                )
            if (
                mode == "origin"
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"split", "rsplit", "partition", "rpartition"}
            ):
                resolved = self._resolve_node(
                    node.func.value,
                    index,
                    mode=mode,
                    seen=seen,
                )
                return _prepend_resolution(
                    expression,
                    "unwrap sequence split",
                    resolved,
                    step,
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr
                in {
                    "encode",
                    "decode",
                    "to_bytes",
                    "hex",
                    "lower",
                    "upper",
                    "strip",
                    "lstrip",
                    "rstrip",
                    "ljust",
                    "rjust",
                    "center",
                    "zfill",
                    "removeprefix",
                    "removesuffix",
                    "casefold",
                    "replace",
                }
            ):
                source_node = node.func.value
                if (
                    node.func.attr == "encode"
                    and _python_name(node.func.value) in {"str", "bytes"}
                    and node.args
                ):
                    source_node = node.args[0]
                resolved = self._resolve_node(
                    source_node,
                    index,
                    mode=mode,
                    seen=seen,
                )
                return _prepend_resolution(
                    expression,
                    "unwrap encoding conversion",
                    resolved,
                    step,
                )
            if (
                mode == "origin"
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"public_bytes", "private_bytes"}
            ):
                resolved = self._resolve_node(
                    node.func.value,
                    index,
                    mode=mode,
                    seen=seen,
                )
                return _prepend_resolution(
                    expression,
                    "unwrap serialization method",
                    resolved,
                    step,
                )
            if (
                mode == "origin"
                and isinstance(node.func, ast.Attribute)
                and node.func.attr
                in {
                    "private_numbers",
                    "public_numbers",
                    "public_key",
                    "specifier",
                }
            ):
                resolved = self._resolve_node(
                    node.func.value,
                    index,
                    mode=mode,
                    seen=seen,
                )
                return _prepend_resolution(
                    expression,
                    f"unwrap asymmetric key method {node.func.attr}",
                    resolved,
                    step,
                )
            if (
                mode == "origin"
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "exchange"
            ):
                inputs = (
                    self._resolve_node(
                        node.func.value,
                        index,
                        mode=mode,
                        seen=seen,
                    ),
                    *(
                        self._resolve_node(
                            argument,
                            index,
                            mode=mode,
                            seen=seen,
                        )
                        for argument in node.args[1:]
                    ),
                )
                return _prepend_resolution(
                    expression,
                    "derive shared secret from asymmetric key material",
                    _merge_entropy_preserving_origins(expression, inputs),
                    step,
                )
            if (
                mode == "origin"
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "join"
                and node.args
            ):
                resolved = self._resolve_node(
                    node.args[0],
                    index,
                    mode=mode,
                    seen=seen,
                )
                return _prepend_resolution(
                    expression,
                    "unwrap joined value sequence",
                    resolved,
                    step,
                )
            if (
                mode in {"origin", "string"}
                and name in {"str"}
                and node.args
            ):
                resolved = self._resolve_node(
                    node.args[0],
                    index,
                    mode=mode,
                    seen=seen,
                )
                return _prepend_resolution(
                    expression,
                    "unwrap scalar conversion",
                    resolved,
                    step,
                )
            if (
                mode == "origin"
                and name.rsplit(".", 1)[-1] in {"zip", "enumerate"}
                and node.args
            ):
                inputs = tuple(
                    self._resolve_node(
                        argument,
                        index,
                        mode=mode,
                        seen=seen,
                    )
                    for argument in node.args
                )
                return _prepend_resolution(
                    expression,
                    "combine iterable value sources",
                    _merge_transformed_origins(expression, inputs),
                    step,
                )
            if mode == "origin" and name == "len" and node.args:
                resolved = self._resolve_node(
                    node.args[0],
                    index,
                    mode=mode,
                    seen=seen,
                )
                return _prepend_resolution(
                    expression,
                    "resolve measured value source",
                    resolved,
                    step,
                )
            if name.rsplit(".", 1)[-1] in {
                "bytes",
                "bytearray",
                "b64encode",
                "b64decode",
                "urlsafe_b64encode",
                "urlsafe_b64decode",
                "hexlify",
                "unhexlify",
                "a2b_hex",
                "fromhex",
                "from_bytes",
                "memoryview",
            } and node.args and not (
                name.rsplit(".", 1)[-1] == "bytes"
                and self._matching_functions(node, caller_index=index)
            ):
                resolved = self._resolve_node(node.args[0], index, mode=mode, seen=seen)
                return _prepend_resolution(expression, "unwrap conversion", resolved, step)
            if (
                mode == "origin"
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pop"
            ):
                popped = self._resolve_python_appended_value(
                    node.func.value,
                    node,
                    index,
                    seen=seen,
                )
                if popped is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve value removed from populated sequence",
                        popped,
                        step,
                    )
            if (
                mode == "origin"
                and name.rsplit(".", 1)[-1]
                in {
                    "load_pem_private_key",
                    "load_der_private_key",
                    "load_pem_public_key",
                    "load_der_public_key",
                    "load_string",
                }
                and node.args
            ):
                resolved = self._resolve_node(
                    node.args[0],
                    index,
                    mode=mode,
                    seen=seen,
                )
                return _prepend_resolution(
                    expression,
                    "unwrap serialization input",
                    resolved,
                    step,
                )
            if mode == "origin":
                fresh_derivation = (
                    self._resolve_python_fresh_custom_derivation_return(
                        node,
                        index,
                        seen=seen,
                    )
                )
                if fresh_derivation is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve fresh custom key-derivation helper returns",
                        fresh_derivation,
                        step,
                    )
                custom_derivation = self._resolve_python_custom_key_derivation(
                    node,
                    index,
                    seen=seen,
                )
                if custom_derivation is not None:
                    return _prepend_resolution(
                        expression,
                        "resolve custom key-derivation secret inputs",
                        custom_derivation,
                        step,
                    )
            helpers = self._matching_functions(node, caller_index=index)
            helper_call, helper_index = node, index
            if not helpers and isinstance(node.func, (ast.Name, ast.Attribute)):
                assignments = self._reaching_assignments(ast.unparse(node.func), index)
                alias = assignments[0] if len(assignments) == 1 else None
                if alias is not None and isinstance(alias[0], ast.Attribute):
                    helper_call = ast.Call(func=alias[0], args=node.args, keywords=node.keywords)
                    helper_index = alias[1]
                    helpers = self._matching_functions(helper_call, caller_index=helper_index)
            helper_key = f"function:{name}"
            if helpers and helper_key not in seen:
                values = self._resolve_bound_helper_returns(
                    helper_call,
                    helpers,
                    caller_index=helper_index,
                    mode=mode,
                    seen=seen | frozenset({helper_key}),
                )
                if values:
                    return _prepend_resolution(
                        expression,
                        f"resolve helper {name}",
                        _merge_resolved(expression, values),
                        step,
                    )

        if isinstance(node, ast.UnaryOp):
            resolved = self._resolve_node(
                node.operand,
                index,
                mode=mode,
                seen=seen,
            )
            return _prepend_resolution(
                expression,
                "apply unary transformation",
                resolved,
                step,
            )

        if isinstance(node, ast.BinOp):
            if mode == "string":
                formatted = _evaluate_python_string_ast(node)
                if formatted is not None:
                    return ResolvedValue(
                        expression,
                        "constant",
                        (formatted,),
                        (step,),
                    )
            left = self._resolve_node(node.left, index, mode=mode, seen=seen)
            right = self._resolve_node(node.right, index, mode=mode, seen=seen)
            if mode == "int":
                combined_values = _combine_resolved_int_operands(
                    node.op,
                    left,
                    right,
                )
                if combined_values:
                    combined = ResolvedValue(
                        expression,
                        "constant",
                        combined_values,
                        (*left.steps, *right.steps),
                    )
                    return _prepend_resolution(
                        expression,
                        "evaluate integer expression",
                        combined,
                        step,
                    )
            if mode == "origin":
                merged = _merge_transformed_origins(expression, (left, right))
            else:
                merged = _merge_resolved(expression, (left, right))
                if left.origin == right.origin == "constant":
                    merged = ResolvedValue(
                        expression,
                        "constant",
                        merged.values,
                        merged.steps,
                    )
            return _prepend_resolution(expression, "combine expression", merged, step)

        if isinstance(node, ast.IfExp):
            condition_id = id(node.test)
            condition: bool | None = None
            if condition_id not in self._condition_evaluation_nodes:
                self._condition_evaluation_nodes.add(condition_id)
                try:
                    condition = self._evaluate_python_condition(
                        node.test,
                        index=index,
                    )
                finally:
                    self._condition_evaluation_nodes.remove(condition_id)
            if condition is not None:
                selected = node.body if condition else node.orelse
                resolved = self._resolve_node(
                    selected,
                    index,
                    mode=mode,
                    seen=seen,
                )
                return _prepend_resolution(
                    expression,
                    "resolve selected conditional value source",
                    resolved,
                    step,
                )
            branches = (
                self._resolve_node(node.body, index, mode=mode, seen=seen),
                self._resolve_node(node.orelse, index, mode=mode, seen=seen),
            )
            return _prepend_resolution(
                expression,
                "merge conditional value sources",
                _merge_resolved(expression, branches),
                step,
            )

        if isinstance(node, ast.BoolOp):
            values_list: list[ResolvedValue] = []
            for value in node.values:
                resolved = self._resolve_node(
                    value,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if (
                    mode == "origin"
                    and isinstance(node.op, ast.Or)
                    and _resolved_has_only_nullish_leaves(resolved)
                ):
                    continue
                values_list.append(resolved)
            values = tuple(values_list)
            return _prepend_resolution(
                expression,
                "merge fallback value sources",
                _merge_resolved(expression, values),
                step,
            )

        if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
            element = self._resolve_node(
                node.elt,
                index,
                mode=mode,
                seen=seen,
            )
            if element.origin != "unknown":
                return _prepend_resolution(
                    expression,
                    "resolve comprehension element source",
                    element,
                    step,
                )
            input_nodes: list[ast.AST] = []
            for generator in node.generators:
                iterator = generator.iter
                if (
                    isinstance(iterator, ast.Call)
                    and _python_name(iterator.func).rsplit(".", 1)[-1] == "zip"
                ):
                    input_nodes.extend(iterator.args)
                else:
                    input_nodes.append(iterator)
            inputs = tuple(
                self._resolve_node(
                    input_node,
                    index,
                    mode=mode,
                    seen=seen,
                )
                for input_node in input_nodes
            )
            if inputs:
                if (
                    mode == "origin"
                    and isinstance(node.elt, ast.BinOp)
                    and isinstance(node.elt.op, ast.BitXor)
                    and any(
                        item.origin in {"secure_random", "derived_secure"}
                        for item in inputs
                    )
                ):
                    merged = ResolvedValue(
                        expression,
                        "derived_secure",
                        steps=tuple(
                            item
                            for resolved in inputs
                            for item in resolved.steps
                        ),
                    )
                else:
                    merged = (
                        _merge_transformed_origins(expression, inputs)
                        if mode == "origin"
                        else _merge_resolved(expression, inputs)
                    )
                return _prepend_resolution(
                    expression,
                    "resolve comprehension input sources",
                    merged,
                    step,
                )

        return ResolvedValue(
            expression,
            "unknown",
            steps=(step,),
            unresolved_reason="expression origin could not be resolved",
        )

    def _python_runtime_stream_open_call(
        self,
        receiver: ast.AST,
        index: int,
    ) -> ast.Call | None:
        """Return the opening call behind a file-like receiver, when known."""

        if isinstance(receiver, ast.Call):
            receiver_name = _canonical_python_name(
                _python_name(receiver.func),
                self.import_aliases,
            )
            if receiver_name in {
                "open",
                "io.open",
                "Path",
                "pathlib.Path",
                "Path.open",
                "pathlib.Path.open",
            }:
                return receiver
            return None
        if not isinstance(receiver, ast.Name):
            return None

        assignment = self._latest_assignment(receiver.id, index)
        if assignment is not None and isinstance(assignment[0], ast.Call):
            assigned_name = _canonical_python_name(
                _python_name(assignment[0].func),
                self.import_aliases,
            )
            if assigned_name in {
                "open",
                "io.open",
                "Path.open",
                "pathlib.Path.open",
            }:
                return assignment[0]

        for with_node in (
            item
            for item in ast.walk(self.tree)
            if isinstance(item, (ast.With, ast.AsyncWith))
        ):
            with_start, with_end = self._node_span(with_node)
            if not (with_start <= index <= with_end):
                continue
            for item in with_node.items:
                if not isinstance(item.optional_vars, ast.Name):
                    continue
                if item.optional_vars.id != receiver.id:
                    continue
                if not isinstance(item.context_expr, ast.Call):
                    continue
                opened_name = _canonical_python_name(
                    _python_name(item.context_expr.func),
                    self.import_aliases,
                )
                if opened_name in {
                    "open",
                    "io.open",
                    "Path.open",
                    "pathlib.Path.open",
                }:
                    return item.context_expr
        return None

    def _resolve_python_file_round_trip(
        self,
        opening: ast.Call,
        index: int,
        *,
        seen: frozenset[str],
        read_call: ast.Call | None = None,
        prefix_length: int | None = None,
    ) -> ResolvedValue | None:
        """Trace a simple file read back to values written to the same path."""

        path_key = self._python_open_path_key(opening)
        if path_key is None:
            return None
        requested_length = prefix_length
        if requested_length is None and read_call is not None and read_call.args:
            requested_length = _evaluate_int_ast(read_call.args[0])
        if requested_length is not None and requested_length < 0:
            requested_length = None
        key = f"python-file-round-trip:{path_key}:{requested_length}"
        if key in seen:
            return None

        write_sequences: list[list[tuple[ast.AST, int]]] = []
        for with_node in (
            item
            for item in ast.walk(self.tree)
            if isinstance(item, (ast.With, ast.AsyncWith))
        ):
            for item in with_node.items:
                candidate = item.context_expr
                if not isinstance(candidate, ast.Call):
                    continue
                if self._python_open_path_key(candidate) != path_key:
                    continue
                if not self._python_open_call_may_write(candidate):
                    continue
                if not isinstance(item.optional_vars, ast.Name):
                    continue
                handle_name = item.optional_vars.id
                writes: list[tuple[ast.AST, int]] = []
                for call in sorted(
                    (
                        nested
                        for nested in ast.walk(with_node)
                        if isinstance(nested, ast.Call)
                    ),
                    key=lambda nested: self._node_span(nested)[0],
                ):
                    if not isinstance(call.func, ast.Attribute):
                        continue
                    if call.func.attr not in {"write", "writelines"}:
                        continue
                    if _python_name(call.func.value) != handle_name or not call.args:
                        continue
                    writes.append((call.args[0], self._node_span(call)[0]))
                if writes:
                    write_sequences.append(writes)

        direct_writes = [
            (call.args[0], self._node_span(call)[0])
            for call in self._call_nodes
            if isinstance(call.func, ast.Attribute)
            and call.func.attr in {"write", "writelines"}
            and call.args
            and isinstance(call.func.value, ast.Call)
            and self._python_open_path_key(call.func.value) == path_key
            and self._python_open_call_may_write(call.func.value)
        ]
        write_sequences.extend([[item] for item in direct_writes])
        if not write_sequences:
            return None

        values: list[ResolvedValue] = []
        for sequence in write_sequences:
            if requested_length is not None:
                prefix = self._resolve_python_written_prefix(
                    sequence,
                    requested_length,
                    seen=frozenset({key}),
                )
                if prefix is not None:
                    values.append(prefix)
                    continue
            values.extend(
                self._resolve_node(
                    value,
                    value_index,
                    mode="origin",
                    # The read path may already be resolving the same local
                    # variable that was persisted. Retain only the file key as
                    # the independent write-side cycle guard.
                    seen=frozenset({key}),
                )
                for value, value_index in sequence
            )
        merged = _merge_resolved(path_key, values)
        if merged.origin == "unknown":
            return None
        return merged

    def _python_open_path_key(self, call: ast.Call) -> str | None:
        call_name = _canonical_python_name(
            _python_name(call.func),
            self.import_aliases,
        )
        if call_name in {"open", "io.open"} and call.args:
            return ast.unparse(call.args[0])
        if call_name in {"Path", "pathlib.Path"} and call.args:
            return ast.unparse(call.args[0])
        if call_name in {"Path.open", "pathlib.Path.open"}:
            if isinstance(call.func, ast.Attribute):
                return ast.unparse(call.func.value)
        return None

    def _python_open_call_may_write(self, call: ast.Call) -> bool:
        mode_node: ast.AST | None = None
        if len(call.args) >= 2:
            mode_node = call.args[1]
        for keyword in call.keywords:
            if keyword.arg == "mode":
                mode_node = keyword.value
                break
        if mode_node is None:
            return False
        return bool(
            isinstance(mode_node, ast.Constant)
            and isinstance(mode_node.value, str)
            and re.search(r"[wax+]", mode_node.value)
        )

    def _resolve_python_derived_origin(
        self,
        node: ast.Call,
        index: int,
        *,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        call_name = _canonical_python_name(
            _python_name(node.func),
            self.import_aliases,
        )
        if call_name == "struct.pack" and _python_struct_pack_is_constant(node):
            return ResolvedValue(
                self._segment(node),
                "predictable",
                steps=(
                    ResolutionStep(
                        self._segment(node),
                        "resolve constant struct.pack arguments",
                        "predictable",
                        self.source_span(*self._node_span(node)),
                    ),
                ),
            )
        if call_name == "hashlib.pbkdf2_hmac" and len(node.args) >= 3:
            password = self._resolve_node(
                node.args[1],
                index,
                mode="origin",
                seen=seen,
            )
            salt = self._resolve_node(
                node.args[2],
                index,
                mode="origin",
                seen=seen,
            )
            return _derived_kdf_origin(
                self._segment(node),
                "hashlib.pbkdf2_hmac",
                password,
                salt,
            )
        if call_name == "hashlib.scrypt" and node.args:
            password = self._resolve_node(
                node.args[0],
                index,
                mode="origin",
                seen=seen,
            )
            salt_node = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "salt"
                ),
                None,
            )
            salt = (
                self._resolve_node(
                    salt_node,
                    index,
                    mode="origin",
                    seen=seen,
                )
                if salt_node is not None
                else ResolvedValue("salt", "unknown")
            )
            return _derived_kdf_origin(
                self._segment(node),
                "hashlib.scrypt",
                password,
                salt,
            )

        if not isinstance(node.func, ast.Attribute):
            return None

        if node.func.attr in {"digest", "hexdigest"} and isinstance(
            node.func.value,
            ast.Call,
        ):
            constructor = node.func.value
            constructor_name = _canonical_python_name(
                _python_name(constructor.func),
                self.import_aliases,
            )
            if constructor_name.startswith("hashlib.") and constructor.args:
                material = self._resolve_node(
                    constructor.args[0],
                    index,
                    mode="origin",
                    seen=seen,
                )
                if material.origin in {"constant", "predictable"}:
                    return ResolvedValue(
                        self._segment(node),
                        "predictable",
                        steps=material.steps,
                    )
                if material.origin == "runtime_input":
                    return ResolvedValue(
                        self._segment(node),
                        "runtime_input",
                        steps=material.steps,
                    )
                if material.origin in {"secure_random", "derived_secure"}:
                    return ResolvedValue(
                        self._segment(node),
                        "derived_secure",
                        steps=material.steps,
                    )

        if node.func.attr != "derive" or not node.args:
            return None
        receiver = node.func.value
        constructor: ast.Call | None = None
        if isinstance(receiver, ast.Call):
            constructor = receiver
        elif isinstance(receiver, ast.Name):
            assignment = self._latest_assignment(receiver.id, index)
            if assignment is not None and isinstance(assignment[0], ast.Call):
                constructor = assignment[0]
        elif isinstance(receiver, ast.Attribute):
            assignment = self._python_instance_attribute_assignment(
                _python_name(receiver),
                index,
            )
            if assignment is not None and isinstance(assignment[0], ast.Call):
                constructor = assignment[0]
        if constructor is None:
            return None
        constructor_name = _canonical_python_name(
            _python_name(constructor.func),
            self.import_aliases,
        )
        constructor_simple = constructor_name.rsplit(".", 1)[-1]
        if constructor_simple not in {"PBKDF2HMAC", "Scrypt", "HKDF"}:
            return None

        password = self._resolve_node(
            node.args[0],
            index,
            mode="origin",
            seen=seen,
        )
        salt_node = next(
            (
                keyword.value
                for keyword in constructor.keywords
                if keyword.arg == "salt"
            ),
            (
                constructor.args[2]
                if constructor_simple in {"PBKDF2HMAC", "HKDF"}
                and len(constructor.args) > 2
                else constructor.args[0]
                if constructor_simple == "Scrypt" and constructor.args
                else None
            ),
        )
        if constructor_simple == "HKDF" and salt_node is None:
            salt_node = ast.Constant(value=None)
        if (
            constructor_simple == "HKDF"
            and isinstance(salt_node, ast.Constant)
            and salt_node.value is None
        ):
            salt = ResolvedValue("None", "constant", ("None",))
        else:
            salt = (
                self._resolve_node(
                    salt_node,
                    self._node_span(constructor)[0],
                    mode="origin",
                    seen=seen,
                )
                if salt_node is not None
                else ResolvedValue("salt", "unknown")
            )
        return _derived_kdf_origin(
            self._segment(node),
            constructor_simple,
            password,
            salt,
        )

    def _resolve_python_custom_key_derivation(
        self,
        node: ast.Call,
        index: int,
        *,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        """Resolve project-local KDF wrappers from their secret parameters.

        Generic return-value tracing can mistake an HKDF accumulator initialized
        with ``b\"\"`` for the effective key origin. For helpers whose name and
        parameters explicitly describe key derivation, follow only the secret
        input parameters; salts, context strings, counters, and output lengths
        are not key material.
        """

        functions = self._matching_functions(node)
        if not functions:
            return None
        call_name = _python_name(node.func).rsplit(".", 1)[-1]
        if re.search(r"(?:^|_)salt(?:_|$)", call_name, re.IGNORECASE):
            return None
        if re.search(
            r"(?i)(?:derive|derivation|hkdf|kdf|expand).*(?:key|token|secret|material)"
            r"|(?:key|token|secret|material).*(?:derive|derivation|hkdf|kdf|expand)"
            r"|(?:^|_)(?:hkdf|kdf)(?:_|$)",
            call_name,
        ) is None:
            return None

        resolved_secrets: list[ResolvedValue] = []
        implementation_steps: list[ResolutionStep] = []
        functions_to_scan = list(functions)
        scanned_functions: set[int] = set()
        while functions_to_scan:
            function = functions_to_scan.pop()
            function_start = self._node_span(function)[0]
            if function_start in scanned_functions:
                continue
            scanned_functions.add(function_start)
            function_calls = tuple(
                candidate
                for candidate in ast.walk(function)
                if isinstance(candidate, ast.Call)
            )
            registered_kdf_calls = tuple(
                call
                for call in function_calls
                if _canonical_python_name(
                    _python_name(call.func),
                    self.import_aliases,
                ).rsplit(".", 1)[-1]
                in {"PBKDF2HMAC", "Scrypt", "HKDF", "pbkdf2_hmac", "scrypt"}
            )
            if registered_kdf_calls:
                implementation_steps.extend(
                    ResolutionStep(
                        self._segment(call),
                        "resolve registered KDF helper",
                        "derived_secure",
                        self.source_span(*self._node_span(call)),
                    )
                    for call in registered_kdf_calls
                )
            else:
                direct_hash_calls = tuple(
                    call
                    for call in function_calls
                    if _canonical_python_name(
                        _python_name(call.func),
                        self.import_aliases,
                    ).rsplit(".", 1)[-1]
                    in {
                        "Hash",
                        "sha1",
                        "sha224",
                        "sha256",
                        "sha384",
                        "sha512",
                        "md5",
                        "blake2b",
                        "blake2s",
                    }
                )
                implementation_steps.extend(
                    ResolutionStep(
                        self._segment(call),
                        "resolve unapproved password hash transform",
                        "derived_secure",
                        self.source_span(*self._node_span(call)),
                    )
                    for call in direct_hash_calls
                )
            for nested_call in function_calls:
                functions_to_scan.extend(self._matching_functions(nested_call))

        for function in functions:
            positional = tuple(function.args.posonlyargs) + tuple(function.args.args)
            parameters = tuple(argument.arg for argument in positional)
            keyword_values = {
                keyword.arg: keyword.value
                for keyword in node.keywords
                if keyword.arg is not None
            }
            for parameter_index, parameter in enumerate(parameters):
                normalized = re.sub(r"[^a-z0-9]", "", parameter.lower())
                if re.search(
                    r"(?:password|passphrase|passwd|pwd|credential|"
                    r"masterkey|secretkey|clientsecret|usersecret|"
                    r"sharedsecret|keymaterial|^ikm$)",
                    normalized,
                ) is None:
                    continue
                argument: ast.AST | None = None
                if parameter_index < len(node.args):
                    argument = node.args[parameter_index]
                elif parameter in keyword_values:
                    argument = keyword_values[parameter]
                if argument is None:
                    continue
                resolved_secrets.append(
                    self._resolve_node(
                        argument,
                        index,
                        mode="origin",
                        seen=seen,
                    )
                )
        if not resolved_secrets:
            return None

        steps = (
            *implementation_steps,
            *(
                step
                for resolved in resolved_secrets
                for step in resolved.steps
            ),
        )
        origins = {resolved.origin for resolved in resolved_secrets}
        expression = self._segment(node)
        if origins & {"constant", "predictable", "non_csprng"}:
            return ResolvedValue(expression, "predictable", steps=steps)
        if origins <= {"runtime_input", "secure_random", "derived_secure"}:
            return ResolvedValue(expression, "derived_secure", steps=steps)
        return ResolvedValue(
            expression,
            "unknown",
            steps=steps,
            unresolved_reason="custom key-derivation secret origin is unresolved",
        )

    def _resolve_argparse_attribute(
        self,
        node: ast.Attribute,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        exclude_empty: bool = False,
    ) -> ResolvedValue | None:
        if not isinstance(node.value, ast.Name):
            return None
        namespace = node.value.id
        exclude_empty = exclude_empty or (
            mode == "origin"
            and self._python_fail_closed_nonempty_guard(
                f"{namespace}.{node.attr}",
                index,
            )
        )
        parsed = self._latest_assignment(namespace, index)
        parse_call = (
            parsed[0]
            if parsed is not None and isinstance(parsed[0], ast.Call)
            else parsed[0].value
            if (
                parsed is not None
                and isinstance(parsed[0], ast.Subscript)
                and isinstance(parsed[0].value, ast.Call)
                and _python_subscript_selector(parsed[0].slice) == ("index", 0)
            )
            else None
        )
        if parse_call is None:
            containing = self._containing_function(index)
            if (
                containing is None
                or namespace not in self._function_parameter_names(containing)
            ):
                return None
            key = f"argparse-parameter:{self._node_span(containing)[0]}:{namespace}.{node.attr}"
            if key in seen:
                return None
            parameter_index = self._function_parameter_names(containing).index(
                namespace
            )
            call_arguments: list[tuple[ast.Call, ast.AST, int]] = []
            for bound_function, bound_call, caller_index in reversed(
                self._bound_helper_calls
            ):
                if bound_function is not containing:
                    continue
                argument = self._call_argument_for_parameter(
                    bound_call,
                    containing,
                    parameter_index,
                    namespace,
                )
                if argument is not None:
                    call_arguments.append((bound_call, argument, caller_index))
                break
            if not call_arguments:
                call_arguments = [
                    (call, argument, self._node_span(call)[0])
                    for call in self._call_nodes
                    if self._external_call_targets_function(call, containing)
                    if (
                        argument := self._call_argument_for_parameter(
                            call,
                            containing,
                            parameter_index,
                            namespace,
                        )
                    )
                    is not None
                ]
            values = [
                self._resolve_node(
                    ast.Attribute(
                        value=argument,
                        attr=node.attr,
                        ctx=ast.Load(),
                    ),
                    caller_index,
                    mode=mode,
                    seen=seen | frozenset({key}),
                )
                for call, argument, caller_index in call_arguments
            ]
            return _merge_resolved(self._segment(node), values) if values else None
        parser_names: set[str] = set()
        if (
            isinstance(parse_call.func, ast.Attribute)
            and parse_call.func.attr == "parse_args"
        ):
            parser_names.add(_python_name(parse_call.func.value))
        else:
            for helper in self._matching_functions(parse_call):
                for returned in self._direct_function_returns(helper):
                    if not (
                        isinstance(returned, ast.Call)
                        and isinstance(returned.func, ast.Attribute)
                        and returned.func.attr == "parse_args"
                    ):
                        continue
                    parser_names.add(_python_name(returned.func.value))
        parser_names.discard("")
        if not parser_names:
            return None
        changed = True
        while changed:
            changed = False
            for parser_name in tuple(parser_names):
                assigned = self._latest_assignment(
                    parser_name,
                    self._node_span(parse_call)[0],
                )
                if assigned is None or not isinstance(assigned[0], ast.Call):
                    continue
                for helper in self._matching_functions(assigned[0]):
                    for returned in self._direct_function_returns(helper):
                        if (
                            isinstance(returned, ast.Name)
                            and returned.id not in parser_names
                        ):
                            parser_names.add(returned.id)
                            changed = True
        argument_owner_names = set(parser_names)
        changed = True
        while changed:
            changed = False
            for candidate in ast.walk(self.tree):
                if not isinstance(candidate, (ast.Assign, ast.AnnAssign)):
                    continue
                value = candidate.value
                if not (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr
                    in {
                        "add_mutually_exclusive_group",
                        "add_argument_group",
                        "add_subparsers",
                        "add_parser",
                    }
                    and _python_name(value.func.value) in argument_owner_names
                ):
                    continue
                targets = (
                    candidate.targets
                    if isinstance(candidate, ast.Assign)
                    else (candidate.target,)
                )
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in argument_owner_names:
                        argument_owner_names.add(target.id)
                        changed = True

        destination = node.attr
        for call in self._call_nodes:
            if not (
                isinstance(call.func, ast.Attribute)
                and call.func.attr in {"add_argument", "add_option"}
                and _python_name(call.func.value) in argument_owner_names
            ):
                continue
            keyword_values = {
                keyword.arg: keyword.value
                for keyword in call.keywords
                if keyword.arg is not None
            }
            configured_destination = ""
            explicit_destination = keyword_values.get("dest")
            if (
                isinstance(explicit_destination, ast.Constant)
                and isinstance(explicit_destination.value, str)
            ):
                configured_destination = explicit_destination.value
            option_names = [
                argument.value
                for argument in call.args
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
            ]
            if not configured_destination and option_names:
                preferred_option = next(
                    (
                        option
                        for option in option_names
                        if option.startswith("--")
                    ),
                    option_names[0],
                )
                configured_destination = preferred_option.lstrip("-").replace("-", "_")
            if configured_destination != destination:
                continue

            choices = keyword_values.get("choices")
            if mode == "string" and isinstance(
                choices,
                (ast.List, ast.Tuple, ast.Set),
            ):
                resolved_choices = tuple(
                    value
                    for element in choices.elts
                    for value in self._resolve_node(
                        element,
                        self._node_span(call)[0],
                        mode="string",
                        seen=seen,
                    ).values
                )
                if resolved_choices:
                    return ResolvedValue(
                        f"{namespace}.{destination}",
                        "constant",
                        tuple(sorted(set(resolved_choices))),
                        steps=(
                            ResolutionStep(
                                f"{namespace}.{destination}",
                                "resolve argparse choices",
                                "constant",
                                self.source_span(*self._node_span(call)),
                            ),
                        ),
                    )

            required = keyword_values.get("required")
            is_required = (
                isinstance(required, ast.Constant) and required.value is True
            )
            if is_required:
                return ResolvedValue(
                    f"{namespace}.{destination}",
                    "runtime_input",
                    steps=(
                        ResolutionStep(
                            f"{namespace}.{destination}",
                            "resolve required argparse value",
                            "runtime_input",
                            self.source_span(*self._node_span(call)),
                        ),
                    ),
                )

            default = keyword_values.get("default")
            if default is not None and not (
                isinstance(default, ast.Constant) and default.value is None
            ):
                if exclude_empty:
                    if isinstance(default, ast.Constant) and default.value == "":
                        return ResolvedValue(
                            f"{namespace}.{destination}",
                            "runtime_input",
                            steps=(
                                ResolutionStep(
                                    f"{namespace}.{destination}",
                                    "discard rejected empty argparse default",
                                    "runtime_input",
                                    self.source_span(*self._node_span(call)),
                                ),
                            ),
                        )
                    if isinstance(default, ast.Call) and _canonical_python_name(
                        _python_name(default.func),
                        self.import_aliases,
                    ) in {"os.getenv", "os.environ.get"}:
                        return ResolvedValue(
                            f"{namespace}.{destination}",
                            "runtime_input",
                            steps=(
                                ResolutionStep(
                                    f"{namespace}.{destination}",
                                    "discard rejected empty environment fallback",
                                    "runtime_input",
                                    self.source_span(*self._node_span(call)),
                                ),
                            ),
                        )
                default_value = self._resolve_node(
                    default,
                    self._node_span(call)[0],
                    mode=mode,
                    seen=seen | frozenset({f"argparse:{namespace}.{destination}"}),
                )
                runtime_value = ResolvedValue(
                    f"{namespace}.{destination}",
                    "runtime_input",
                    steps=(
                        ResolutionStep(
                            f"{namespace}.{destination}",
                            "resolve optional argparse override",
                            "runtime_input",
                            self.source_span(*self._node_span(call)),
                        ),
                    ),
                )
                if mode == "string":
                    return ResolvedValue(
                        f"{namespace}.{destination}",
                        "runtime_input",
                        steps=(*default_value.steps, *runtime_value.steps),
                        unresolved_reason=(
                            "optional argparse value is unrestricted when supplied"
                        ),
                    )
                return _merge_resolved(
                    f"{namespace}.{destination}",
                    (default_value, runtime_value),
                )

            is_positional = bool(option_names and not option_names[0].startswith("-"))
            if is_positional:
                return ResolvedValue(
                    f"{namespace}.{destination}",
                    "runtime_input",
                    steps=(
                        ResolutionStep(
                            f"{namespace}.{destination}",
                            "resolve required argparse value",
                            "runtime_input",
                            self.source_span(*self._node_span(call)),
                        ),
                    ),
                )
            return ResolvedValue(
                f"{namespace}.{destination}",
                "runtime_input",
                steps=(
                    ResolutionStep(
                        f"{namespace}.{destination}",
                        "resolve optional argparse value when supplied",
                        "runtime_input",
                        self.source_span(*self._node_span(call)),
                    ),
                ),
            )
        return None

    def _resolve_python_destructured_return_attribute(
        self,
        node: ast.Attribute,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        if not isinstance(node.value, ast.Name):
            return None
        assignment = self._latest_assignment(node.value.id, index)
        if assignment is None or not isinstance(assignment[0], ast.Subscript):
            return None
        selected = assignment[0]
        if not isinstance(selected.value, ast.Call):
            return None
        selector = _python_subscript_selector(selected.slice)
        if selector is None or selector[0] != "index" or not isinstance(selector[1], int):
            return None
        component_index = selector[1]
        key = (
            f"destructured-return:{node.value.id}:{node.attr}:"
            f"{assignment[1]}:{component_index}"
        )
        if key in seen:
            return None
        values: list[ResolvedValue] = []
        for helper in self._matching_functions(selected.value):
            for returned in self._direct_function_returns(helper):
                if not isinstance(returned, (ast.Tuple, ast.List)):
                    continue
                if not 0 <= component_index < len(returned.elts):
                    continue
                component = returned.elts[component_index]
                attribute = ast.Attribute(
                    value=component,
                    attr=node.attr,
                    ctx=ast.Load(),
                )
                ast.copy_location(attribute, component)
                values.append(
                    self._resolve_node(
                        attribute,
                        self._node_span(returned)[0],
                        mode=mode,
                        seen=seen | frozenset({key}),
                    )
                )
        return _merge_resolved(self._segment(node), values) if values else None

    def _python_fail_closed_nonempty_guard(
        self,
        reference: str,
        index: int,
    ) -> bool:
        containing = self._containing_function(index)
        tree = containing if containing is not None else self.tree
        if any(
            isinstance(candidate, ast.If)
            and self._node_span(candidate)[0] < index
            and _python_statements_terminate(candidate.body)
            and _python_condition_rejects_empty(candidate.test, reference)
            for candidate in ast.walk(tree)
        ):
            return True
        start = self._node_span(containing)[0] if containing is not None else 0
        prefix = self.source[start:index]
        value = r"\s*\.\s*".join(
            re.escape(part) for part in reference.split(".")
        )
        conditions = (
            rf"not\s+{value}",
            rf"len\s*\(\s*{value}\s*\)\s*==\s*0",
            rf"{value}\s+is\s+None",
            rf"{value}\s*==\s*[\"'][\"']",
        )
        return any(
            re.search(
                rf"\bif\s+{condition}\s*:\s*(?:\n[ \t]+)?"
                rf"(?:raise\b|return\b|(?:sys\s*\.\s*)?exit\s*\()",
                prefix,
                flags=re.DOTALL,
            )
            is not None
            for condition in conditions
        )

    def _resolve_python_origin_without_empty_paths(
        self,
        node: ast.AST,
        index: int,
        *,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        if isinstance(node, ast.Constant) and node.value in {None, "", b""}:
            return None
        if isinstance(node, ast.IfExp):
            values = tuple(
                resolved
                for branch in (node.body, node.orelse)
                if (
                    resolved := self._resolve_python_origin_without_empty_paths(
                        branch,
                        index,
                        seen=seen,
                    )
                )
                is not None
            )
            return _merge_resolved(self._segment(node), values) if values else None
        if isinstance(node, ast.Name):
            key = f"nonempty-name:{node.id}:{index}"
            if key not in seen:
                values = tuple(
                    resolved
                    for assigned_node, assigned_index in self._reaching_assignments(
                        node.id,
                        index,
                    )
                    if (
                        resolved := self._resolve_python_origin_without_empty_paths(
                            assigned_node,
                            assigned_index,
                            seen=seen | frozenset({key}),
                        )
                    )
                    is not None
                )
                if values:
                    return _merge_resolved(self._segment(node), values)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call):
            selector = _python_subscript_selector(node.slice)
            if (
                selector is not None
                and selector[0] == "index"
                and isinstance(selector[1], int)
            ):
                values: list[ResolvedValue] = []
                for helper in self._matching_functions(
                    node.value,
                    caller_index=index,
                ):
                    self._bound_helper_calls.append((helper, node.value, index))
                    try:
                        for returned in self._direct_function_returns(helper):
                            selected = _python_literal_container_element(
                                returned,
                                selector,
                            )
                            if selected is None:
                                continue
                            resolved = self._resolve_python_origin_without_empty_paths(
                                selected,
                                self._node_span(returned)[0],
                                seen=seen,
                            )
                            if resolved is not None:
                                values.append(resolved)
                    finally:
                        self._bound_helper_calls.pop()
                if values:
                    return _merge_resolved(self._segment(node), values)
        if isinstance(node, ast.Attribute):
            argparse_value = self._resolve_argparse_attribute(
                node,
                index,
                mode="origin",
                seen=seen,
                exclude_empty=True,
            )
            if argparse_value is not None:
                return argparse_value
        if isinstance(node, ast.Call):
            name = _canonical_python_name(
                _python_name(node.func),
                self.import_aliases,
            )
            if name in {"os.getenv", "os.environ.get"}:
                return ResolvedValue(
                    self._segment(node),
                    "runtime_input",
                    steps=(
                        ResolutionStep(
                            self._segment(node),
                            "discard rejected empty environment fallback",
                            "runtime_input",
                            self.source_span(*self._node_span(node)),
                        ),
                    ),
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr
                in {"encode", "strip", "lstrip", "rstrip", "lower", "upper"}
            ):
                return self._resolve_python_origin_without_empty_paths(
                    node.func.value,
                    index,
                    seen=seen,
                )
            if name in {"bytes", "bytearray", "str"} and node.args:
                return self._resolve_python_origin_without_empty_paths(
                    node.args[0],
                    index,
                    seen=seen,
                )
            helpers = self._matching_functions(node, caller_index=index)
            helper_key = f"nonempty-function:{name}:{index}"
            if helpers and helper_key not in seen:
                values: list[ResolvedValue] = []
                for helper in helpers:
                    self._bound_helper_calls.append((helper, node, index))
                    try:
                        for returned in self._direct_function_returns(helper):
                            resolved = self._resolve_python_origin_without_empty_paths(
                                returned,
                                self._node_span(returned)[0],
                                seen=seen | frozenset({helper_key}),
                            )
                            if resolved is not None:
                                values.append(resolved)
                    finally:
                        self._bound_helper_calls.pop()
                if values:
                    return _merge_resolved(self._segment(node), values)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            values = [
                resolved
                for value in node.values
                if (
                    resolved := self._resolve_python_origin_without_empty_paths(
                        value,
                        index,
                        seen=seen,
                    )
                )
                is not None
            ]
            return _merge_resolved(self._segment(node), values) if values else None
        return _resolved_without_empty_alternatives(
            self._resolve_node(
                node,
                index,
                mode="origin",
                seen=seen,
            )
        )

    def _collect_assignments(self) -> dict[str, list[tuple[ast.AST, int]]]:
        records: dict[str, list[tuple[ast.AST, int]]] = {}
        for node in ast.walk(self.tree):
            if node in self._inactive_method_body_nodes:
                continue
            targets: list[ast.AST] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            elif isinstance(node, ast.NamedExpr):
                targets = [node.target]
                value = node.value
            if value is None:
                continue
            index = self._node_span(node)[0]
            for target in targets:
                for name, bound_value in _python_assignment_bindings(target, value):
                    # Tuple/list destructuring creates synthetic Subscript nodes
                    # such as ``encrypt(...)[0]``.  Give those nodes the same
                    # lexical parent as the assignment so scope-sensitive
                    # resolution does not mistake a local overwrite for a
                    # module-level binding.
                    if bound_value not in self._parents:
                        self._parents[bound_value] = node
                    records.setdefault(name, []).append((bound_value, index))
        for values in records.values():
            values.sort(key=lambda item: item[1])
        return records

    def _collect_subscript_assignments(
        self,
    ) -> dict[tuple[str, tuple[str, object]], list[tuple[ast.AST, int]]]:
        records: dict[
            tuple[str, tuple[str, object]],
            list[tuple[ast.AST, int]],
        ] = {}
        for node in ast.walk(self.tree):
            if node in self._inactive_method_body_nodes:
                continue
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Subscript):
                    continue
                container_name = _python_name(target.value)
                if not container_name:
                    continue
                selector = _python_subscript_selector(target.slice)
                if selector is None:
                    selector = ("dynamic", None)
                records.setdefault((container_name, selector), []).append(
                    (value, self._node_span(node)[0])
                )
        for values in records.values():
            values.sort(key=lambda item: item[1])
        return records

    def _collect_append_arguments(
        self,
    ) -> dict[str, list[tuple[ast.AST, int]]]:
        records: dict[str, list[tuple[ast.AST, int]]] = {}
        for call in self._call_nodes:
            if (
                not isinstance(call.func, ast.Attribute)
                or call.func.attr != "append"
                or len(call.args) != 1
            ):
                continue
            receiver = _python_name(call.func.value)
            if not receiver:
                continue
            records.setdefault(receiver, []).append(
                (call.args[0], self._node_span(call)[0])
            )
        for values in records.values():
            values.sort(key=lambda item: item[1])
        return records

    def _latest_subscript_assignment(
        self,
        name: str,
        selector: tuple[str, object],
        index: int,
    ) -> tuple[ast.AST, int] | None:
        containing = self._containing_function(index)
        matches = [
            item
            for item in self._subscript_assignments.get((name, selector), ())
            if self._enclosing_function(item[0]) is containing
            and self._node_span(item[0])[1] <= index
        ]
        dynamic_matches = [
            item
            for item in self._subscript_assignments.get(
                (name, ("dynamic", None)),
                (),
            )
            if self._enclosing_function(item[0]) is containing
            and self._node_span(item[0])[1] <= index
        ]
        candidates = (
            matches + dynamic_matches
            if selector[0] == "index"
            else matches
        )
        return max(candidates, key=lambda item: item[1]) if candidates else None

    def _resolve_python_subscript(
        self,
        node: ast.Subscript,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        selector = _python_subscript_selector(node.slice)
        if selector is None:
            if isinstance(node.slice, ast.Slice) and mode == "origin":
                return self._resolve_node(
                    node.value,
                    index,
                    mode=mode,
                    seen=seen,
                )
            resolved_selector = self._resolve_node(
                node.slice,
                index,
                mode="string",
                seen=seen,
            )
            values: list[ResolvedValue] = []
            for value in resolved_selector.values:
                resolved = self._resolve_python_container_element(
                    node.value,
                    ("index", value),
                    index,
                    mode=mode,
                    seen=seen,
                )
                if resolved is not None:
                    values.append(resolved)
            if mode == "string" and not values:
                values.extend(
                    self._resolve_python_all_container_values(
                        node.value,
                        index,
                        mode=mode,
                        seen=seen,
                    )
                )
            return _merge_resolved(self._segment(node), values) if values else None
        if selector[0] == "slice":
            if mode == "string":
                resolved = self._resolve_node(
                    node.value,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if resolved.values:
                    return resolved
            if mode == "origin":
                # Selecting any byte range is a deterministic view of the
                # same material; it does not change key provenance.
                return self._resolve_node(
                    node.value,
                    index,
                    mode=mode,
                    seen=seen,
                )
            return None
        if selector[0] == "bounded_slice":
            lower, upper, step = selector[1]
            if mode == "string":
                resolved = self._resolve_node(
                    node.value,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if _python_resolved_has_only_fixed_values(resolved):
                    return ResolvedValue(
                        self._segment(node),
                        "constant",
                        tuple(
                            dict.fromkeys(
                                value[slice(lower, upper, step)]
                                for value in resolved.values
                            )
                        ),
                        resolved.steps,
                    )
            if lower in {None, 0} and isinstance(upper, int) and step is None:
                prefix = self._resolve_python_prefix_slice(
                    node.value,
                    upper,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if prefix is not None:
                    return prefix
            if mode == "origin":
                return self._resolve_node(
                    node.value,
                    index,
                    mode=mode,
                    seen=seen,
                )
            return None
        return self._resolve_python_container_element(
            node.value,
            selector,
            index,
            mode=mode,
            seen=seen,
        )

    def _resolve_python_all_container_values(
        self,
        container: ast.AST,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> list[ResolvedValue]:
        """Resolve every value of a mapping selected by a runtime key."""

        if isinstance(container, ast.Dict):
            return [
                self._resolve_node(
                    value,
                    index,
                    mode=mode,
                    seen=seen,
                )
                for value in container.values
            ]
        if isinstance(container, ast.Name):
            key = f"all-container-values:{container.id}:{index}"
            if key in seen:
                return []
            assignment = self._latest_assignment(container.id, index)
            if assignment is not None:
                return self._resolve_python_all_container_values(
                    assignment[0],
                    assignment[1],
                    mode=mode,
                    seen=seen | frozenset({key}),
                )
        return []

    def _resolve_python_prefix_slice(
        self,
        container: ast.AST,
        upper: int,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        if upper < 0:
            return None
        if isinstance(container, ast.Name):
            # The same parameter/local name commonly appears on both sides of
            # an encrypt/decrypt helper boundary.  Source position is part of
            # the resolution identity so following the caller does not look
            # like recursion merely because both functions call it ``data``.
            key = f"prefix-slice:{container.id}:{upper}:{index}"
            if key in seen:
                return None
            assignment = self._latest_assignment(container.id, index)
            if assignment is not None:
                return self._resolve_python_prefix_slice(
                    assignment[0],
                    upper,
                    assignment[1],
                    mode=mode,
                    seen=seen | frozenset({key}),
                )
            containing = self._containing_function(index)
            if containing is not None:
                parameters = self._function_parameter_names(containing)
                if container.id in parameters:
                    parameter_index = parameters.index(container.id)
                    values = [
                        resolved
                        for call in self._call_nodes
                        if self._external_call_targets_function(call, containing)
                        if (
                            argument := self._call_argument_for_parameter(
                                call,
                                containing,
                                parameter_index,
                                container.id,
                            )
                        )
                        is not None
                        if (
                            resolved := self._resolve_python_prefix_slice(
                                argument,
                                upper,
                                self._node_span(call)[0],
                                mode=mode,
                                seen=seen | frozenset({key}),
                            )
                        )
                        is not None
                    ]
                    if values:
                        return _merge_resolved(ast.unparse(container), values)
        if isinstance(container, ast.Call):
            key = (
                f"prefix-slice-call:{_python_name(container.func)}:{upper}:"
                f"{self._node_span(container)[0]}"
            )
            if key in seen:
                return None
            call_name = _canonical_python_name(
                _python_name(container.func),
                self.import_aliases,
            )
            if (
                mode == "origin"
                and isinstance(container.func, ast.Attribute)
                and container.func.attr
                in {"read", "read1", "readall", "read_bytes"}
            ):
                opening = self._python_runtime_stream_open_call(
                    container.func.value,
                    index,
                )
                if opening is not None:
                    persisted = self._resolve_python_file_round_trip(
                        opening,
                        index,
                        seen=seen | frozenset({key}),
                        read_call=container,
                        prefix_length=upper,
                    )
                    if persisted is not None:
                        return persisted
            if call_name.rsplit(".", 1)[-1] in {
                "bytes",
                "bytearray",
                "b64encode",
                "b64decode",
                "urlsafe_b64encode",
                "urlsafe_b64decode",
                "hexlify",
                "unhexlify",
                "a2b_hex",
                "fromhex",
                "memoryview",
            } and container.args:
                return self._resolve_python_prefix_slice(
                    container.args[0],
                    upper,
                    index,
                    mode=mode,
                    seen=seen | frozenset({key}),
                )
            if (
                isinstance(container.func, ast.Attribute)
                and container.func.attr in {"encode", "decode"}
            ):
                return self._resolve_python_prefix_slice(
                    container.func.value,
                    upper,
                    index,
                    mode=mode,
                    seen=seen | frozenset({key}),
                )
            values = [
                resolved
                for helper in self._matching_functions(container)
                for returned in self._direct_function_returns(helper)
                if (
                    resolved := self._resolve_python_prefix_slice(
                        returned,
                        upper,
                        self._node_span(returned)[0],
                        mode=mode,
                        seen=seen | frozenset({key}),
                    )
                )
                is not None
            ]
            if values:
                return _merge_resolved(self._segment(container), values)
            if mode == "origin":
                # A byte-prefix view preserves the provenance of a producer
                # even when its exact output length is not statically known.
                # This is especially important for local wrappers around
                # os.urandom/secrets and serialized key records.
                resolved = self._resolve_node(
                    container,
                    index,
                    mode=mode,
                    seen=seen | frozenset({key}),
                )
                if resolved.origin != "unknown":
                    return resolved
            return None
        if isinstance(container, ast.BinOp) and isinstance(container.op, ast.Add):
            left_length = self._python_known_byte_length(
                container.left,
                index,
                seen=seen,
            )
            if left_length is not None and upper <= left_length:
                return self._resolve_node(
                    container.left,
                    index,
                    mode=mode,
                    seen=seen,
                )
            if left_length is not None:
                left = self._resolve_node(
                    container.left,
                    index,
                    mode=mode,
                    seen=seen,
                )
                right = self._resolve_python_prefix_slice(
                    container.right,
                    upper - left_length,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if right is None:
                    right = self._resolve_node(
                        container.right,
                        index,
                        mode=mode,
                        seen=seen,
                    )
                return _merge_transformed_origins(
                    ast.unparse(container),
                    (left, right),
                )
            nested_prefix = self._resolve_python_prefix_slice(
                container.left,
                upper,
                index,
                mode=mode,
                seen=seen,
            )
            if nested_prefix is not None:
                return nested_prefix
        return None

    def _python_known_byte_length(
        self,
        node: ast.AST,
        index: int,
        *,
        seen: frozenset[str],
    ) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, str)):
            return len(node.value)
        if isinstance(node, ast.Name):
            key = f"byte-length:{node.id}:{index}"
            if key in seen:
                return None
            assignment = self._latest_assignment(node.id, index)
            if assignment is not None:
                return self._python_known_byte_length(
                    assignment[0],
                    assignment[1],
                    seen=seen | frozenset({key}),
                )
            return None
        if isinstance(node, ast.Attribute):
            reference = ast.unparse(node)
            key = f"byte-length-attribute:{reference}:{index}"
            if key in seen:
                return None
            sink = next(
                (
                    candidate
                    for candidate in self._call_nodes
                    if self._node_span(candidate)[0] == index
                ),
                None,
            )
            owner = self._enclosing_class(sink) if sink is not None else None
            assignments: list[tuple[ast.AST, int]] = []
            scope: ast.AST = owner if owner is not None else self.tree
            for assignment in ast.walk(scope):
                if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                    continue
                if self._enclosing_class(assignment) is not owner:
                    continue
                targets = (
                    assignment.targets
                    if isinstance(assignment, ast.Assign)
                    else (assignment.target,)
                )
                if assignment.value is None:
                    continue
                if not any(
                    isinstance(target, ast.Attribute)
                    and ast.unparse(target) == reference
                    for target in targets
                ):
                    continue
                assignments.append(
                    (assignment.value, self._node_span(assignment)[0])
                )
            lengths = [
                self._python_known_byte_length(
                    value,
                    assignment_index,
                    seen=seen | frozenset({key}),
                )
                for value, assignment_index in assignments
            ]
            if lengths and all(length is not None for length in lengths):
                distinct = set(lengths)
                if len(distinct) == 1:
                    return next(iter(distinct))
            return None
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice):
                lower = (
                    _evaluate_int_ast(node.slice.lower)
                    if node.slice.lower is not None
                    else 0
                )
                upper = (
                    _evaluate_int_ast(node.slice.upper)
                    if node.slice.upper is not None
                    else None
                )
                step = (
                    _evaluate_int_ast(node.slice.step)
                    if node.slice.step is not None
                    else 1
                )
                if (
                    lower is not None
                    and upper is not None
                    and lower >= 0
                    and upper >= lower
                    and step == 1
                ):
                    return upper - lower
            selector = _python_subscript_selector(node.slice)
            if (
                selector is not None
                and selector[0] == "index"
                and isinstance(selector[1], int)
                and isinstance(node.value, ast.Call)
            ):
                lengths: set[int] = set()
                for helper in self._matching_functions(node.value, caller_index=index):
                    for returned in self._direct_function_returns(helper):
                        selected = _python_literal_container_element(
                            returned,
                            selector,
                        )
                        if selected is None:
                            continue
                        length = self._python_known_byte_length(
                            selected,
                            self._node_span(returned)[0],
                            seen=seen
                            | frozenset(
                                {
                                    "byte-length-component:"
                                    f"{self._node_span(node.value)[0]}:"
                                    f"{selector[1]}"
                                }
                            ),
                        )
                        if length is not None:
                            lengths.add(length)
                if len(lengths) == 1:
                    return next(iter(lengths))
            return None
        if isinstance(node, ast.Call):
            name = _canonical_python_name(
                _python_name(node.func),
                self.import_aliases,
            )
            if name in {"bytes.fromhex", "bytearray.fromhex"} and node.args:
                resolved = self._resolve_node(
                    node.args[0],
                    index,
                    mode="string",
                    seen=seen,
                )
                lengths: set[int] = set()
                for value in resolved.values:
                    try:
                        lengths.add(len(bytes.fromhex(value)))
                    except ValueError:
                        return None
                if len(lengths) == 1:
                    return next(iter(lengths))
            if name in {
                "os.urandom",
                "secrets.token_bytes",
                "random.randbytes",
            } and node.args:
                return _evaluate_int_ast(node.args[0])
            lengths = {
                length
                for helper in self._matching_functions(node, caller_index=index)
                for returned in self._direct_function_returns(helper)
                if (
                    length := self._python_known_byte_length(
                        returned,
                        self._node_span(returned)[0],
                        seen=seen
                        | frozenset(
                            {
                                f"byte-length-call:{name}:"
                                f"{self._node_span(node)[0]}"
                            }
                        ),
                    )
                )
                is not None
            }
            if len(lengths) == 1:
                return next(iter(lengths))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._python_known_byte_length(node.left, index, seen=seen)
            right = self._python_known_byte_length(node.right, index, seen=seen)
            return left + right if left is not None and right is not None else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if isinstance(node.left, ast.Constant) and isinstance(
                node.left.value, (bytes, str)
            ):
                multiplier = _evaluate_int_ast(node.right)
                return (
                    len(node.left.value) * multiplier
                    if multiplier is not None and multiplier >= 0
                    else None
                )
            if isinstance(node.right, ast.Constant) and isinstance(
                node.right.value, (bytes, str)
            ):
                multiplier = _evaluate_int_ast(node.left)
                return (
                    len(node.right.value) * multiplier
                    if multiplier is not None and multiplier >= 0
                    else None
                )
        return None

    def _resolve_python_container_element(
        self,
        container: ast.AST,
        selector: tuple[str, object],
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        return self._resolve_python_container_path(
            container,
            (selector,),
            index,
            mode=mode,
            seen=seen,
        )

    def _resolve_python_container_path(
        self,
        container: ast.AST,
        selectors: tuple[tuple[str, object], ...],
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        if not selectors:
            return self._resolve_node(container, index, mode=mode, seen=seen)

        selector = selectors[0]
        remaining = selectors[1:]
        selected = _python_literal_container_element(container, selector)
        if selected is not None:
            return self._resolve_python_container_path(
                selected,
                remaining,
                index,
                mode=mode,
                seen=seen,
            )

        if isinstance(container, ast.Attribute):
            container_name = _python_name(container)
            mutation = self._latest_subscript_assignment(
                container_name,
                selector,
                index,
            )
            if mutation is not None:
                return self._resolve_python_container_path(
                    mutation[0],
                    remaining,
                    mutation[1],
                    mode=mode,
                    seen=seen
                    | frozenset(
                        {f"indexed-attribute:{container_name}:{selectors!r}"}
                    ),
                )
            values = [
                resolved
                for candidate in self._python_constructed_attribute_nodes(
                    container,
                    index,
                )
                if (
                    resolved := self._resolve_python_container_path(
                        candidate,
                        selectors,
                        index,
                        mode=mode,
                        seen=seen,
                    )
                )
                is not None
            ]
            if values:
                return _merge_resolved(self._segment(container), values)

        if isinstance(container, ast.Subscript):
            nested_selector = _python_subscript_selector(container.slice)
            if nested_selector is not None:
                return self._resolve_python_container_path(
                    container.value,
                    (nested_selector, *selectors),
                    index,
                    mode=mode,
                    seen=seen,
                )

        if isinstance(container, ast.Subscript) and mode == "origin":
            base = self._resolve_node(
                container.value,
                index,
                mode=mode,
                seen=seen,
            )
            if base.origin != "unknown":
                return base

        if isinstance(container, (ast.BoolOp, ast.IfExp)):
            branches = (
                tuple(container.values)
                if isinstance(container, ast.BoolOp)
                else (container.body, container.orelse)
            )
            values = [
                resolved
                for branch in branches
                if (
                    resolved := self._resolve_python_container_path(
                        branch,
                        selectors,
                        index,
                        mode=mode,
                        seen=seen,
                    )
                )
                is not None
            ]
            return _merge_resolved(self._segment(container), values) if values else None

        if isinstance(container, ast.Name):
            containing = self._containing_function(index)
            scope = self._node_span(containing)[0] if containing is not None else "global"
            key = f"indexed:{scope}:{container.id}:{selectors!r}"
            if key in seen:
                return None
            mutation = self._latest_subscript_assignment(
                container.id,
                selector,
                index,
            )
            if mutation is not None:
                return self._resolve_python_container_path(
                    mutation[0],
                    remaining,
                    mutation[1],
                    mode=mode,
                    seen=seen | frozenset({key}),
                )
            if selector[0] == "index" and isinstance(selector[1], int):
                appends = [
                    item
                    for item in self._append_arguments.get(container.id, ())
                    if self._enclosing_function(item[0])
                    is self._containing_function(index)
                    and self._node_span(item[0])[1] <= index
                ]
                position = int(selector[1])
                if position < 0:
                    position += len(appends)
                if 0 <= position < len(appends):
                    argument, argument_index = appends[position]
                    return self._resolve_python_container_path(
                        argument,
                        remaining,
                        argument_index,
                        mode=mode,
                        seen=seen | frozenset({key}),
                    )
            assignment = self._latest_assignment(container.id, index)
            if assignment is not None:
                return self._resolve_python_container_path(
                    assignment[0],
                    selectors,
                    assignment[1],
                    mode=mode,
                    seen=seen | frozenset({key}),
                )
            containing = self._containing_function(index)
            if containing is not None:
                parameters = self._function_parameter_names(containing)
                if container.id in parameters:
                    parameter_index = parameters.index(container.id)
                    values: list[ResolvedValue] = []
                    for call in self._call_nodes:
                        if not self._external_call_targets_function(
                            call,
                            containing,
                        ):
                            continue
                        argument = self._call_argument_for_parameter(
                            call,
                            containing,
                            parameter_index,
                            container.id,
                        )
                        if argument is None:
                            continue
                        resolved = self._resolve_python_container_path(
                            argument,
                            selectors,
                            self._node_span(call)[0],
                            mode=mode,
                            seen=seen | frozenset({key}),
                        )
                        if resolved is not None:
                            values.append(resolved)
                    if values:
                        return _merge_resolved(container.id, values)

        if isinstance(container, ast.Call):
            if (
                mode in {"origin", "string"}
                and isinstance(container.func, ast.Attribute)
                and container.func.attr in {"split", "rsplit", "partition", "rpartition"}
            ):
                split_element = self._resolve_python_split_element(
                    container,
                    selectors,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if split_element is not None:
                    return split_element
                return self._resolve_node(
                    container.func.value,
                    index,
                    mode=mode,
                    seen=seen,
                )
            if (
                isinstance(container.func, ast.Attribute)
                and container.func.attr == "get"
                and container.args
                and (inner_selector := _python_subscript_selector(container.args[0]))
                is not None
            ):
                values: list[ResolvedValue] = []
                nested = self._resolve_python_container_path(
                    container.func.value,
                    (inner_selector, *selectors),
                    index,
                    mode=mode,
                    seen=seen,
                )
                if nested is not None:
                    values.append(nested)
                if len(container.args) >= 2:
                    fallback = self._resolve_python_container_path(
                        container.args[1],
                        selectors,
                        index,
                        mode=mode,
                        seen=seen,
                    )
                    if fallback is not None:
                        values.append(fallback)
                if values:
                    return _merge_resolved(self._segment(container), values)
            canonical_name = _canonical_python_name(
                _python_name(container.func),
                self.import_aliases,
            )
            if (
                canonical_name == "struct.unpack"
                and len(container.args) >= 2
                and selector[0] == "index"
                and isinstance(selector[1], int)
            ):
                unpacked = self._resolve_python_struct_unpack_value(
                    container,
                    int(selector[1]),
                    index,
                    mode=mode,
                    seen=seen,
                )
                if unpacked is not None:
                    return unpacked if not remaining else None
            if (
                mode == "origin"
                and isinstance(container.func, ast.Attribute)
                and container.func.attr in {"readline", "readlines"}
            ):
                opening = self._python_runtime_stream_open_call(
                    container.func.value,
                    index,
                )
                if opening is not None:
                    persisted = self._resolve_python_file_round_trip(
                        opening,
                        index,
                        seen=seen,
                    )
                    if persisted is not None:
                        return persisted
            if (
                canonical_name.rsplit(".", 1)[-1] == "map"
                and len(container.args) >= 2
            ):
                mapped = self._resolve_python_container_path(
                    container.args[1],
                    selectors,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if mapped is not None:
                    return mapped
            if canonical_name == "json.load" and container.args:
                structured = self._resolve_python_json_file_path(
                    container,
                    selectors,
                    index,
                    mode=mode,
                    seen=seen,
                )
                if structured is not None:
                    return structured
            if canonical_name in {"json.loads", "json.dumps"} and container.args:
                return self._resolve_python_container_path(
                    container.args[0],
                    selectors,
                    index,
                    mode=mode,
                    seen=seen,
                )
            key = f"indexed-function:{_python_name(container.func)}:{selectors!r}"
            if key in seen:
                return None
            values = [
                resolved
                for helper in self._matching_functions(container)
                for returned in self._direct_function_returns(helper)
                if (
                    resolved := self._resolve_python_container_path(
                        returned,
                        selectors,
                        self._node_span(returned)[0],
                        mode=mode,
                        seen=seen | frozenset({key}),
                    )
                )
                is not None
            ]
            if values:
                return _merge_resolved(self._segment(container), values)
        return None

    def _resolve_python_split_element(
        self,
        call: ast.Call,
        selectors: tuple[tuple[str, object], ...],
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        """Resolve one field selected from a statically delimited record."""

        if (
            mode not in {"origin", "string"}
            or len(selectors) != 1
            or selectors[0][0] != "index"
            or not isinstance(selectors[0][1], int)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "split"
            or not call.args
            or not isinstance(call.args[0], ast.Constant)
            or not isinstance(call.args[0].value, str)
            or not call.args[0].value
        ):
            return None
        return self._resolve_python_delimited_element(
            call.func.value,
            call.args[0].value,
            int(selectors[0][1]),
            index,
            mode=mode,
            seen=seen,
        )

    def _resolve_python_delimited_element(
        self,
        node: ast.AST,
        delimiter: str,
        element_index: int,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        if isinstance(node, ast.Name):
            key = f"delimited:{node.id}:{delimiter}:{element_index}"
            if key in seen:
                return None
            assignment = self._latest_assignment(node.id, index)
            if (
                assignment is not None
                and any(
                    isinstance(candidate, ast.Name)
                    and isinstance(candidate.ctx, ast.Load)
                    and candidate.id == node.id
                    for candidate in ast.walk(assignment[0])
                )
            ):
                assignment = self._latest_assignment(node.id, assignment[1] - 1)
            if assignment is None:
                containing = self._containing_function(index)
                if containing is not None:
                    parameters = self._function_parameter_names(containing)
                    if node.id in parameters:
                        parameter_index = parameters.index(node.id)
                        caller_values = [
                            resolved
                            for call in self._call_nodes
                            if self._external_call_targets_function(call, containing)
                            if (
                                argument := self._call_argument_for_parameter(
                                    call,
                                    containing,
                                    parameter_index,
                                    node.id,
                                )
                            )
                            is not None
                            if (
                                resolved := self._resolve_python_delimited_element(
                                    argument,
                                    delimiter,
                                    element_index,
                                    self._node_span(call)[0],
                                    mode=mode,
                                    seen=seen | frozenset({key}),
                                )
                            )
                            is not None
                        ]
                        if caller_values:
                            return _merge_resolved(node.id, caller_values)
                parameter = self._resolve_parameter(
                    node.id,
                    index,
                    mode="string",
                    seen=seen | frozenset({key}),
                )
                if (
                    parameter is None
                    or parameter.origin != "constant"
                    or not parameter.values
                ):
                    return None
                selected_values: set[str] = set()
                for value in parameter.values:
                    fields = value.split(delimiter)
                    position = (
                        element_index
                        if element_index >= 0
                        else len(fields) + element_index
                    )
                    if not 0 <= position < len(fields):
                        return None
                    selected_values.add(fields[position])
                return ResolvedValue(
                    node.id,
                    "constant",
                    tuple(sorted(selected_values)),
                    parameter.steps,
                )
            resolved = self._resolve_python_delimited_element(
                assignment[0],
                delimiter,
                element_index,
                assignment[1],
                mode=mode,
                seen=seen | frozenset({key}),
            )
            if (
                resolved is not None
                and mode == "string"
                and self._python_fail_closed_nonempty_guard(node.id, index)
            ):
                return _resolved_without_empty_alternatives(resolved)
            return resolved

        if isinstance(node, ast.Call):
            key = (
                f"delimited-call:{_python_name(node.func)}:{delimiter}:"
                f"{element_index}:{self._node_span(node)[0]}"
            )
            if key not in seen:
                helper_values = [
                    resolved
                    for helper in self._matching_functions(node, caller_index=index)
                    for returned in self._direct_function_returns(helper)
                    if (
                        resolved := self._resolve_python_delimited_element(
                            returned,
                            delimiter,
                            element_index,
                            self._node_span(returned)[0],
                            mode=mode,
                            seen=seen | frozenset({key}),
                        )
                    )
                    is not None
                ]
                if helper_values:
                    return _merge_resolved(self._segment(node), helper_values)

        if isinstance(node, ast.JoinedStr):
            markers: list[str] = []
            values: list[ast.AST] = []
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    markers.append(part.value)
                elif isinstance(part, ast.FormattedValue):
                    markers.append(f"\0{len(values)}\0")
                    values.append(part.value)
                else:
                    return None
            fields = "".join(markers).split(delimiter)
            position = element_index if element_index >= 0 else len(fields) + element_index
            if not 0 <= position < len(fields):
                return None
            selected = re.fullmatch(r"\0(?P<index>\d+)\0", fields[position])
            if selected is None:
                return None
            value = values[int(selected.group("index"))]
            return self._resolve_node(
                value,
                self._node_span(value)[0],
                mode=mode,
                seen=seen,
            )

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and isinstance(node.func.value, ast.Constant)
            and node.func.value.value == delimiter
            and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))
        ):
            elements = node.args[0].elts
            position = (
                element_index
                if element_index >= 0
                else len(elements) + element_index
            )
            if 0 <= position < len(elements):
                selected = elements[position]
                return self._resolve_node(
                    selected,
                    self._node_span(selected)[0],
                    mode=mode,
                    seen=seen,
                )

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            fields = node.value.split(delimiter)
            position = element_index if element_index >= 0 else len(fields) + element_index
            if 0 <= position < len(fields):
                return ResolvedValue(
                    repr(fields[position]),
                    "constant",
                    (fields[position],),
                )
        resolved = self._resolve_node(
            node,
            index,
            mode="string",
            seen=seen,
        )
        if resolved.origin == "constant" and resolved.values:
            selected_values: set[str] = set()
            for value in resolved.values:
                fields = value.split(delimiter)
                position = (
                    element_index
                    if element_index >= 0
                    else len(fields) + element_index
                )
                if not 0 <= position < len(fields):
                    return None
                selected_values.add(fields[position])
            return ResolvedValue(
                ast.unparse(node),
                "constant",
                tuple(sorted(selected_values)),
                resolved.steps,
            )
        return None

    def _resolve_python_json_file_element(
        self,
        load_call: ast.Call,
        selector: tuple[str, object],
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        return self._resolve_python_json_file_path(
            load_call,
            (selector,),
            index,
            mode=mode,
            seen=seen,
        )

    def _resolve_python_json_file_path(
        self,
        load_call: ast.Call,
        selectors: tuple[tuple[str, object], ...],
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        """Resolve a JSON field from values this program persists to the same path."""

        opening = self._python_runtime_stream_open_call(load_call.args[0], index)
        if opening is None:
            return None
        path_key = self._python_open_path_key(opening)
        if path_key is None:
            return None
        cycle_key = f"python-json-round-trip:{path_key}:{selectors!r}"
        if cycle_key in seen:
            return None

        values: list[ResolvedValue] = []
        for with_node in (
            item
            for item in ast.walk(self.tree)
            if isinstance(item, (ast.With, ast.AsyncWith))
        ):
            for item in with_node.items:
                candidate = item.context_expr
                if not isinstance(candidate, ast.Call):
                    continue
                if self._python_open_path_key(candidate) != path_key:
                    continue
                if not self._python_open_call_may_write(candidate):
                    continue
                if not isinstance(item.optional_vars, ast.Name):
                    continue
                handle_name = item.optional_vars.id
                for call in (
                    nested
                    for nested in ast.walk(with_node)
                    if isinstance(nested, ast.Call)
                ):
                    call_name = _canonical_python_name(
                        _python_name(call.func),
                        self.import_aliases,
                    )
                    if call_name != "json.dump" or len(call.args) < 2:
                        continue
                    if _python_name(call.args[1]) != handle_name:
                        continue
                    resolved = self._resolve_python_container_path(
                        call.args[0],
                        selectors,
                        self._node_span(call)[0],
                        mode=mode,
                        seen=seen | frozenset({cycle_key}),
                    )
                    if resolved is not None:
                        values.append(resolved)
        if values:
            return _merge_resolved(self._segment(load_call), values)
        if mode == "origin":
            return ResolvedValue(self._segment(load_call), "runtime_input")
        return None

    def _latest_assignment(self, name: str, index: int) -> tuple[ast.AST, int] | None:
        assignments = self._assignments.get(name, ())
        containing = self._containing_function(index)
        scoped = [
            item
            for item in assignments
            if self._enclosing_function(item[0]) is containing
        ]
        earlier = [
            item
            for item in scoped
            if self._node_span(item[0])[1] <= index
        ]
        if earlier:
            return earlier[-1]
        if containing is not None and name in self._function_parameter_names(containing):
            return None
        non_overlapping = [
            item
            for item in assignments
            if self._enclosing_function(item[0]) is None
            if not (item[1] <= index < self._node_span(item[0])[1])
        ]
        earlier_globals = [
            item
            for item in non_overlapping
            if self._node_span(item[0])[1] <= index
        ]
        if earlier_globals:
            return earlier_globals[-1]
        return non_overlapping[0] if non_overlapping else None

    def _profile_python_node_is_reachable(self, node: ast.AST) -> bool:
        if self.analysis_profile != "reachable_security_flow":
            return True
        return self._python_literal_boolean_path_is_reachable(node)

    def _python_literal_boolean_path_is_reachable(self, node: ast.AST) -> bool:
        current = node
        parent = self._parents.get(current)
        while parent is not None:
            if (
                isinstance(parent, ast.If)
                and isinstance(parent.test, ast.Constant)
                and isinstance(parent.test.value, bool)
            ):
                in_body = any(
                    current is statement or self._node_inside(current, statement)
                    for statement in parent.body
                )
                in_else = any(
                    current is statement or self._node_inside(current, statement)
                    for statement in parent.orelse
                )
                if (in_body and not parent.test.value) or (
                    in_else and parent.test.value
                ):
                    return False
            current = parent
            parent = self._parents.get(current)
        return True

    def _reaching_assignments(
        self,
        name: str,
        index: int,
    ) -> tuple[tuple[ast.AST, int], ...]:
        """Return mutually exclusive Python assignments reaching one use.

        A lexical "latest assignment" loses fixed defaults in shapes such as
        ``if args.key is None: key = FIXED; else: key = runtime``. Walk the
        enclosing ``if`` tree of the latest assignment and retain every branch
        assignment to the same binding. If only one branch assigns the value,
        include the preceding baseline assignment as the other reachable path.
        """

        latest = self._latest_assignment(name, index)
        if latest is None:
            return tuple()
        containing = self._containing_function(index)
        scoped = tuple(
            item
            for item in self._assignments.get(name, ())
            if self._enclosing_function(item[0]) is containing
            and self._node_span(item[0])[1] <= index
            and self._python_direct_path_is_reachable(item[0])
        )
        if not scoped:
            return (latest,)
        if latest not in scoped:
            # The lexically latest assignment may sit in an ``elif`` branch
            # that a fixed caller/default proves unreachable.  Continue from
            # the latest reachable definition instead of reintroducing the
            # very branch filtered above.
            latest = scoped[-1]

        conditionals: list[ast.If] = []
        current: ast.AST | None = latest[0]
        while current is not None and current is not containing:
            current = self._parents.get(current)
            if isinstance(current, ast.If):
                conditionals.append(current)

        for conditional in reversed(conditionals):
            members = tuple(
                item
                for item in scoped
                if self._python_node_is_within(item[0], conditional)
            )
            if len(members) >= 2:
                return tuple(sorted(members, key=lambda item: item[1]))
            if len(members) == 1:
                conditional_start = self._node_span(conditional)[0]
                baseline = tuple(
                    item for item in scoped if item[1] < conditional_start
                )
                if baseline:
                    return (baseline[-1], members[0])
        return (latest,)

    def _python_node_is_within(
        self,
        node: ast.AST,
        ancestor: ast.AST,
    ) -> bool:
        current: ast.AST | None = node
        while current is not None:
            if current is ancestor:
                return True
            current = self._parents.get(current)
        return False

    def _python_instance_attribute_assignment(
        self,
        name: str,
        index: int,
    ) -> tuple[ast.AST, int] | None:
        containing = self._containing_function(index)
        owner = self._enclosing_class(containing) if containing is not None else None
        candidates = [
            item
            for item in self._assignments.get(name, ())
            if self._enclosing_class(item[0]) is owner
        ]
        if (
            not candidates
            and owner is None
            and name.startswith(("self.", "cls."))
        ):
            all_candidates = self._assignments.get(name, ())
            candidate_owners = {
                class_node
                for item in all_candidates
                if (class_node := self._enclosing_class(item[0])) is not None
            }
            if len(candidate_owners) == 1:
                candidates = list(all_candidates)
        return candidates[-1] if candidates else None

    def _python_mapping_node(
        self,
        node: ast.AST,
        index: int,
    ) -> ast.Dict | None:
        if isinstance(node, ast.Dict):
            return node
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in {
                "self",
                "cls",
            }:
                assignment = self._python_instance_attribute_assignment(
                    _python_name(node),
                    index,
                )
                if assignment is not None and isinstance(
                    assignment[0],
                    ast.Dict,
                ):
                    return assignment[0]
            constructed = self._python_constructed_attribute_nodes(node, index)
            return next(
                (candidate for candidate in constructed if isinstance(candidate, ast.Dict)),
                None,
            )
        name = _python_name(node)
        assignment = None
        if name.startswith(("self.", "cls.")):
            assignment = self._python_instance_attribute_assignment(name, index)
        elif isinstance(node, ast.Name):
            assignment = self._latest_assignment(node.id, index)
        if assignment is None or not isinstance(assignment[0], ast.Dict):
            return None
        return assignment[0]

    def _python_mapping_values(
        self,
        node: ast.AST,
        index: int,
        selector: object | None = None,
    ) -> tuple[ast.AST, ...]:
        mapping = self._python_mapping_node(node, index)
        if mapping is None:
            return ()
        values: list[ast.AST] = []
        for key, value in zip(mapping.keys, mapping.values, strict=True):
            if key is None:
                continue
            if selector is not None:
                literal = key.value if isinstance(key, ast.Constant) else object()
                if literal != selector:
                    continue
            values.append(value)
        return tuple(values)

    def _resolve_python_selected_callable(
        self,
        node: ast.Call,
        index: int,
        *,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        if node.args or node.keywords:
            return None
        callable_nodes: tuple[ast.AST, ...]
        if isinstance(node.func, ast.Subscript):
            selector = _python_subscript_selector(node.func.slice)
            selected_keys: tuple[object, ...]
            if selector is not None and selector[0] == "index":
                selected_keys = (selector[1],)
            else:
                resolved_selector = self._resolve_node(
                    node.func.slice,
                    index,
                    mode="string",
                    seen=seen,
                )
                selected_keys = tuple(resolved_selector.values)
            if selected_keys:
                callable_nodes = tuple(
                    callable_node
                    for selected_key in selected_keys
                    for callable_node in self._python_mapping_values(
                        node.func.value,
                        index,
                        selected_key,
                    )
                )
            else:
                callable_nodes = self._python_mapping_values(
                    node.func.value,
                    index,
                )
        elif (
            isinstance(node.func, ast.Call)
            and _canonical_python_name(
                _python_name(node.func.func),
                self.import_aliases,
            )
            in {"random.choice", "secrets.choice"}
            and node.func.args
            and isinstance(node.func.args[0], (ast.List, ast.Tuple, ast.Set))
        ):
            callable_nodes = tuple(node.func.args[0].elts)
        else:
            return None
        if not callable_nodes:
            return None
        containing_class = self._enclosing_class(node)
        key = f"selected-callable:{ast.unparse(node.func)}:{index}"
        if key in seen:
            return None
        values: list[ResolvedValue] = []
        for callable_node in callable_nodes:
            if isinstance(callable_node, ast.Lambda):
                resolved = self._resolve_node(
                    callable_node.body,
                    self._node_span(callable_node)[0],
                    mode="origin",
                    seen=seen | frozenset({key}),
                )
                values.append(
                    _append_resolution_step(
                        resolved,
                        ResolutionStep(
                            ast.unparse(callable_node),
                            "resolve selected callable alternative",
                            resolved.origin,
                            self.source_span(*self._node_span(callable_node)),
                        ),
                    )
                )
                continue
            helper_name = _python_name(callable_node).rsplit(".", 1)[-1]
            helpers = tuple(
                helper
                for helper in self._functions_by_name.get(helper_name, ())
                if self._enclosing_class(helper) is containing_class
            )
            for helper in helpers:
                if self._python_helper_uses_secure_random(
                    helper,
                    seen=frozenset(),
                ):
                    values.append(
                        ResolvedValue(
                            ast.unparse(callable_node),
                            "secure_random",
                            steps=(
                                ResolutionStep(
                                    ast.unparse(callable_node),
                                    "resolve selected callable alternative",
                                    "secure_random",
                                    self.source_span(*self._node_span(callable_node)),
                                ),
                            ),
                        )
                    )
                    continue
                for returned in self._direct_function_returns(helper):
                    resolved = self._resolve_node(
                        returned,
                        self._node_span(returned)[0],
                        mode="origin",
                        seen=seen | frozenset({key}),
                    )
                    values.append(
                        _append_resolution_step(
                            resolved,
                            ResolutionStep(
                                ast.unparse(callable_node),
                                "resolve selected callable alternative",
                                resolved.origin,
                                self.source_span(*self._node_span(callable_node)),
                            ),
                        )
                    )
        return _merge_resolved(ast.unparse(node), values) if values else None

    def _python_helper_uses_secure_random(
        self,
        helper: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        seen: frozenset[int],
    ) -> bool:
        helper_start = self._node_span(helper)[0]
        if helper_start in seen:
            return False
        next_seen = seen | frozenset({helper_start})
        for call in ast.walk(helper):
            if not isinstance(call, ast.Call) or self._enclosing_function(call) is not helper:
                continue
            call_name = _canonical_python_name(
                _python_name(call.func),
                self.import_aliases,
            )
            if call_name in {
                "os.urandom",
                "secrets.choice",
                "secrets.randbelow",
                "secrets.token_bytes",
                "secrets.token_hex",
                "secrets.token_urlsafe",
            }:
                return True
            if any(
                self._python_helper_uses_secure_random(
                    nested,
                    seen=next_seen,
                )
                for nested in self._matching_functions(call)
            ):
                return True
        return False

    def _resolve_python_prior_instance_mutations(
        self,
        node: ast.Attribute,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        name = _python_name(node)
        if not name.startswith(("self.", "cls.")):
            return None
        containing = self._containing_function(index)
        owner = self._enclosing_class(containing) if containing is not None else None
        if containing is None or owner is None:
            return None

        sink_callers = tuple(
            call
            for call in self._call_nodes
            if self._external_call_targets_function(call, containing)
            and self._python_call_has_entry_path(call, seen=frozenset())
        )
        reachable_assignments: list[tuple[ast.AST, int]] = []
        for assignment in self._assignments.get(name, ()):
            assigned_node, assigned_index = assignment
            assignment_function = self._enclosing_function(assigned_node)
            if self._enclosing_class(assigned_node) is not owner:
                continue
            if assignment_function is containing and assigned_index < index:
                reachable_assignments.append(assignment)
                continue
            if assignment_function is None or assignment_function.name == "__init__":
                continue
            callers = tuple(
                call
                for call in self._call_nodes
                if self._external_call_targets_function(
                    call,
                    assignment_function,
                )
                and self._python_call_has_entry_path(call, seen=frozenset())
            )
            if any(
                self._enclosing_function(caller) is containing
                and self._node_span(caller)[0] < index
                for caller in callers
            ) or any(
                self._enclosing_function(caller)
                is self._enclosing_function(sink_caller)
                and self._node_span(caller)[0] < self._node_span(sink_caller)[0]
                for caller in callers
                for sink_caller in sink_callers
            ):
                reachable_assignments.append(assignment)
        if not reachable_assignments:
            return None

        key = f"instance-mutations:{name}:{index}"
        if key in seen:
            return None
        values = [
            self._resolve_node(
                assigned_node,
                assigned_index,
                mode=mode,
                seen=seen | frozenset({key}),
            )
            for assigned_node, assigned_index in reachable_assignments
            if not (
                isinstance(assigned_node, ast.Constant)
                and assigned_node.value is None
            )
        ]
        if not values or any(value.origin == "unknown" for value in values):
            return None
        return _merge_resolved(name, values)

    def _resolve_python_stateful_digest(
        self,
        node: ast.Call,
        index: int,
        *,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        if (
            not isinstance(node.func, ast.Attribute)
            or node.func.attr not in {"digest", "final", "finalize"}
        ):
            return None
        receiver_node = node.func.value
        if isinstance(receiver_node, ast.Call):
            constructor = receiver_node
            constructor_index = self._node_span(receiver_node)[0]
            receiver = ast.unparse(receiver_node)
        elif isinstance(receiver_node, ast.Name):
            receiver = receiver_node.id
            assignment = self._latest_assignment(receiver, index)
            if assignment is None or not isinstance(assignment[0], ast.Call):
                return None
            constructor, constructor_index = assignment
        else:
            return None
        constructor_name = _canonical_python_name(
            _python_name(constructor.func),
            self.import_aliases,
        )
        key = f"stateful-digest:{receiver}:{constructor_index}"
        if key in seen:
            return None
        if constructor_name in {
            "hmac.new",
            "cryptography.hazmat.primitives.hmac.HMAC",
        }:
            if not constructor.args:
                return None
            resolved = self._resolve_node(
                constructor.args[0],
                constructor_index,
                mode="origin",
                seen=seen | frozenset({key}),
            )
            return _merge_transformed_origins(self._segment(node), (resolved,))
        if constructor_name.endswith("hashes.HMAC"):
            if len(constructor.args) < 2:
                return None
            resolved = self._resolve_node(
                constructor.args[1],
                constructor_index,
                mode="origin",
                seen=seen | frozenset({key}),
            )
            return _merge_transformed_origins(self._segment(node), (resolved,))

        is_hash = (
            constructor_name.endswith(".hashes.Hash")
            or constructor_name.endswith(".Hash")
            or constructor_name.startswith("hashlib.")
        )
        if not is_hash:
            return None
        containing = self._containing_function(index)
        updates = tuple(
            call
            for call in self._call_nodes
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "update"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == receiver
            and call.args
            and constructor_index < self._node_span(call)[0] < index
            and self._enclosing_function(call) is containing
        )
        if not updates and constructor.args:
            values = tuple(
                self._resolve_node(
                    argument,
                    constructor_index,
                    mode="origin",
                    seen=seen | frozenset({key}),
                )
                for argument in constructor.args
            )
            return _merge_entropy_preserving_origins(self._segment(node), values)
        if not updates:
            return ResolvedValue(
                self._segment(node),
                "predictable",
                steps=(
                    ResolutionStep(
                        self._segment(node),
                        "resolve digest of empty input",
                        "predictable",
                        self.source_span(*self._node_span(node)),
                    ),
                ),
            )
        values = tuple(
            self._resolve_node(
                call.args[0],
                self._node_span(call)[0],
                mode="origin",
                seen=seen | frozenset({key}),
            )
            for call in updates
        )
        return _merge_entropy_preserving_origins(self._segment(node), values)

    def _resolve_constructed_attribute(
        self,
        node: ast.Attribute,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        """Resolve a field on a locally constructed source-defined class."""

        constructions = list(self._python_constructions(node.value, index))
        if isinstance(node.value, ast.Name) and not constructions:
            containing = self._containing_function(index)
            if containing is not None and node.value.id in {"self", "cls"}:
                for call in self._call_nodes:
                    if (
                        not self._external_call_targets_function(
                            call,
                            containing,
                        )
                        or not isinstance(call.func, ast.Attribute)
                    ):
                        continue
                    constructions.extend(
                        self._python_constructions(
                            call.func.value,
                            self._node_span(call)[0],
                        )
                    )
            if (
                containing is not None
                and node.value.id in self._function_parameter_names(containing)
            ):
                parameters = self._function_parameter_names(containing)
                parameter_index = parameters.index(node.value.id)
                callers = [
                    (call, argument)
                    for call in self._call_nodes
                    if self._external_call_targets_function(call, containing)
                    if (
                        argument := self._call_argument_for_parameter(
                            call,
                            containing,
                            parameter_index,
                            node.value.id,
                        )
                    )
                    is not None
                ]
                reachable = [
                    item
                    for item in callers
                    if self._python_call_has_entry_path(item[0], seen=frozenset())
                ]
                if reachable:
                    callers = reachable
                for call, argument in callers:
                    constructions.extend(
                        self._python_constructions(
                            argument,
                            self._node_span(call)[0],
                        )
                    )
        if not constructions:
            return None

        class_names = tuple(
            dict.fromkeys(
                _python_name(construction.func).rsplit(".", 1)[-1]
                for construction in constructions
            )
        )
        key = (
            f"constructed-attribute:{','.join(class_names)}.{node.attr}:"
            f"{ast.unparse(node.value)}"
        )
        if key in seen:
            return None

        values = [
            value
            for construction in constructions
            for class_node in self._class_nodes
            if class_node.name
            == _python_name(construction.func).rsplit(".", 1)[-1]
            if (
                value := self._resolve_class_attribute_value(
                    class_node,
                    construction,
                    node.attr,
                    index,
                    mode=mode,
                    seen=seen | frozenset({key}),
                )
            )
            is not None
        ]
        return _merge_resolved(ast.unparse(node), values) if values else None

    def _python_constructions(
        self,
        node: ast.AST,
        index: int,
        seen: frozenset[str] = frozenset(),
    ) -> tuple[ast.Call, ...]:
        if isinstance(node, ast.IfExp):
            return tuple(
                construction
                for branch in (node.body, node.orelse)
                for construction in self._python_constructions(
                    branch,
                    index,
                    seen,
                )
            )
        if isinstance(node, ast.BoolOp):
            return tuple(
                construction
                for branch in node.values
                for construction in self._python_constructions(
                    branch,
                    index,
                    seen,
                )
            )
        if isinstance(node, ast.Call):
            call_name = _python_name(node.func).rsplit(".", 1)[-1]
            if any(class_node.name == call_name for class_node in self._class_nodes):
                return (node,)
            key = f"construction-call:{call_name}:{self._node_span(node)[0]}"
            if key in seen:
                return ()
            values = [
                construction
                for helper in self._matching_functions(node, caller_index=index)
                for returned in self._direct_function_returns(helper)
                for construction in self._python_constructions(
                    returned,
                    self._node_span(returned)[0],
                    seen | frozenset({key}),
                )
            ]
            return tuple(values)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call):
            selector = _python_subscript_selector(node.slice)
            if (
                selector is None
                or selector[0] != "index"
                or not isinstance(selector[1], int)
            ):
                return ()
            position = selector[1]
            key = (
                f"construction-component:{_python_name(node.value.func)}:"
                f"{position}:{self._node_span(node.value)[0]}"
            )
            if key in seen:
                return ()
            values: list[ast.Call] = []
            for helper in self._matching_functions(
                node.value,
                caller_index=index,
            ):
                for returned in self._direct_function_returns(helper):
                    if not isinstance(returned, (ast.Tuple, ast.List)):
                        continue
                    component = position
                    if component < 0:
                        component += len(returned.elts)
                    if not 0 <= component < len(returned.elts):
                        continue
                    values.extend(
                        self._python_constructions(
                            returned.elts[component],
                            self._node_span(returned)[0],
                            seen | frozenset({key}),
                        )
                    )
            return tuple(values)
        if isinstance(node, ast.Attribute):
            attribute_name = _python_name(node)
            if attribute_name.startswith(("self.", "cls.")):
                key = f"construction-attribute:{attribute_name}:{index}"
                if key in seen:
                    return ()
                next_seen = seen | frozenset({key})
                assignment = self._python_instance_attribute_assignment(
                    attribute_name,
                    index,
                )
                if assignment is not None:
                    assigned_node, assigned_index = assignment
                    direct = self._python_constructions(
                        assigned_node,
                        assigned_index,
                        next_seen,
                    )
                    if direct:
                        return direct
                    constructor = self._enclosing_function(assigned_node)
                    owner = self._enclosing_class(assigned_node)
                    if (
                        constructor is not None
                        and constructor.name == "__init__"
                        and owner is not None
                        and isinstance(assigned_node, ast.Name)
                    ):
                        parameters = self._function_parameter_names(constructor)
                        if assigned_node.id in parameters:
                            parameter_index = parameters.index(assigned_node.id)
                            values: list[ast.Call] = []
                            for call in self._call_nodes:
                                call_name = _python_name(call.func).rsplit(".", 1)[-1]
                                if not self._python_class_is_or_extends(
                                    call_name,
                                    owner.name,
                                ):
                                    continue
                                argument = self._call_argument_for_parameter(
                                    call,
                                    constructor,
                                    parameter_index,
                                    assigned_node.id,
                                )
                                if argument is not None:
                                    values.extend(
                                        self._python_constructions(
                                            argument,
                                            self._node_span(call)[0],
                                            next_seen,
                                        )
                                    )
                            return tuple(values)
            field_value = self._python_plain_field_binding(node, index)
            if field_value is not None:
                key = f"plain-field-construction:{attribute_name}:{index}"
                if key in seen:
                    return ()
                return self._python_constructions(*field_value, seen | frozenset({key}))
        if isinstance(node, ast.Name):
            key = f"construction-name:{node.id}:{index}"
            if key in seen:
                return ()
            assignment = self._latest_assignment(node.id, index)
            if assignment is not None:
                return self._python_constructions(
                    assignment[0],
                    assignment[1],
                    seen | frozenset({key}),
                )
            context_constructions = tuple(
                construction
                for with_node in ast.walk(self.tree)
                if isinstance(with_node, (ast.With, ast.AsyncWith))
                and self._node_span(with_node)[0] <= index
                for item in with_node.items
                if item.optional_vars is not None
                and node.id in _python_target_names(item.optional_vars)
                for construction in self._python_constructions(
                    item.context_expr,
                    self._node_span(item.context_expr)[0],
                    seen | frozenset({key}),
                )
            )
            if context_constructions:
                return context_constructions
            containing = self._containing_function(index)
            if containing is not None:
                parameters = self._function_parameter_names(containing)
                if node.id in parameters:
                    parameter_index = parameters.index(node.id)
                    values: list[ast.Call] = []
                    for call in self._call_nodes:
                        if (
                            _python_name(call.func).rsplit(".", 1)[-1]
                            != containing.name
                        ):
                            continue
                        if (
                            not self._external_call_targets_function(
                                call,
                                containing,
                            )
                        ):
                            continue
                        argument = self._call_argument_for_parameter(
                            call,
                            containing,
                            parameter_index,
                            node.id,
                        )
                        if argument is not None:
                            values.extend(
                                self._python_constructions(
                                    argument,
                                    self._node_span(call)[0],
                                    seen | frozenset({key}),
                                )
                            )
                    if values:
                        return tuple(values)
        return ()

    def _resolve_class_attribute_value(
        self,
        class_node: ast.ClassDef,
        construction: ast.Call,
        attribute_name: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        fields: list[tuple[str, ast.AST | None]] = []
        for statement in class_node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target,
                ast.Name,
            ):
                fields.append((statement.target.id, statement.value))
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        fields.append((target.id, statement.value))

        if self._python_context_manager_call_precedes(construction, index):
            enter_methods = [
                statement
                for statement in class_node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == "__enter__"
                and any(
                    isinstance(returned, ast.Name) and returned.id == "self"
                    for returned in self._direct_function_returns(statement)
                )
            ]
            for enter_method in enter_methods:
                assignment = next(
                    (
                        item
                        for item in reversed(
                            self._assignments.get(f"self.{attribute_name}", ())
                        )
                        if self._enclosing_function(item[0]) is enter_method
                    ),
                    None,
                )
                if assignment is not None:
                    return self._resolve_node(
                        assignment[0],
                        assignment[1],
                        mode=mode,
                        seen=seen,
                    )

        for keyword in construction.keywords:
            if keyword.arg == attribute_name:
                return self._resolve_node(
                    keyword.value,
                    self._node_span(keyword.value)[0],
                    mode=mode,
                    seen=seen,
                )
        field_names = [name for name, _ in fields]
        if attribute_name in field_names:
            field_index = field_names.index(attribute_name)
            if field_index < len(construction.args):
                return self._resolve_node(
                    construction.args[field_index],
                    self._node_span(construction.args[field_index])[0],
                    mode=mode,
                    seen=seen,
                )

        constructors = [
            statement
            for statement in class_node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == "__init__"
            and statement not in self._shadowed_methods
        ]
        for constructor in constructors:
            assignment = next(
                (
                    item
                    for item in reversed(self._assignments.get(f"self.{attribute_name}", ()))
                    if self._enclosing_function(item[0]) is constructor
                ),
                None,
            )
            if assignment is None:
                continue
            assigned_node, assigned_index = assignment
            if isinstance(assigned_node, ast.Name):
                parameters = self._function_parameter_names(constructor)
                if assigned_node.id in parameters:
                    argument = self._call_argument_for_parameter(
                        construction,
                        constructor,
                        parameters.index(assigned_node.id),
                        assigned_node.id,
                    )
                    if argument is not None:
                        return self._resolve_node(
                            argument,
                            self._node_span(argument)[0],
                            mode=mode,
                            seen=seen,
                        )
            return self._resolve_node(
                assigned_node,
                assigned_index,
                mode=mode,
                seen=seen,
            )

        default = next(
            (value for name, value in fields if name == attribute_name),
            None,
        )
        if default is None:
            return None
        return self._resolve_node(
            default,
            self._node_span(default)[0],
            mode=mode,
            seen=seen,
        )

    def _python_context_manager_call_precedes(
        self,
        construction: ast.Call,
        index: int,
    ) -> bool:
        del index
        return any(
            item.context_expr is construction
            for with_node in ast.walk(self.tree)
            if isinstance(with_node, (ast.With, ast.AsyncWith))
            for item in with_node.items
        )

    def _resolve_python_for_binding(
        self,
        name: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        loops = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.For, ast.AsyncFor))
            and name in _python_target_names(node.target)
            and self._node_span(node)[0] <= index <= self._node_span(node)[1]
        ]
        if not loops:
            return None
        loop = max(loops, key=lambda node: self._node_span(node)[0])
        if (
            mode == "origin"
            and isinstance(loop.target, (ast.Tuple, ast.List))
            and isinstance(loop.iter, ast.Call)
            and isinstance(loop.iter.func, ast.Attribute)
            and loop.iter.func.attr == "items"
        ):
            positions = [
                position
                for position, target in enumerate(loop.target.elts)
                if name in _python_target_names(target)
            ]
            if positions == [1]:
                mapping_values = self._python_mapping_values(
                    loop.iter.func.value,
                    self._node_span(loop.iter)[0],
                )
                if mapping_values:
                    resolved = [
                        self._resolve_node(
                            value,
                            self._node_span(value)[0],
                            mode=mode,
                            seen=seen,
                        )
                        for value in mapping_values
                    ]
                    return _merge_resolved(name, resolved)
        if mode == "int" and isinstance(loop.iter, ast.Call):
            call_name = _python_name(loop.iter.func).rsplit(".", 1)[-1]
            resolved_arguments = tuple(
                self._resolve_node(
                    item,
                    self._node_span(loop.iter)[0],
                    mode="int",
                    seen=seen,
                )
                for item in loop.iter.args
            )
            arguments = tuple(
                _resolved_single_int(item) for item in resolved_arguments
            )
            if (
                call_name == "range"
                and 1 <= len(arguments) <= 3
                and all(value is not None for value in arguments)
            ):
                concrete = tuple(int(value) for value in arguments if value is not None)
                values = range(*concrete)
                if len(values) > 0:
                    endpoints = tuple(
                        dict.fromkeys((str(values[0]), str(values[-1])))
                    )
                    return ResolvedValue(
                        self._segment(loop.iter),
                        "constant",
                        endpoints,
                        steps=(
                            ResolutionStep(
                                self._segment(loop.iter),
                                "resolve bounded range loop values",
                                "constant",
                                self.source_span(*self._node_span(loop.iter)),
                            ),
                        ),
                    )
        if mode == "origin" and isinstance(loop.iter, ast.Call):
            call_name = _python_name(loop.iter.func).rsplit(".", 1)[-1]
            if call_name == "range" and loop.iter.args:
                arguments = tuple(
                    self._resolve_node(
                        item,
                        self._node_span(loop.iter)[0],
                        mode="int",
                        seen=seen,
                    )
                    for item in loop.iter.args
                )
                if all(_resolved_single_int(item) is not None for item in arguments):
                    return ResolvedValue(
                        self._segment(loop.iter),
                        "constant",
                        tuple(
                            value
                            for item in arguments
                            for value in item.values
                        ),
                        tuple(step for item in arguments for step in item.steps),
                    )
        return self._resolve_node(
            loop.iter,
            self._node_span(loop.iter)[0],
            mode=mode,
            seen=seen,
        )

    def _resolve_parameter(
        self,
        name: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        containing_candidates = [
            function
            for function in self._function_nodes
            if self._node_span(function)[0] <= index <= self._node_span(function)[1]
            and name in self._function_parameter_names(function)
        ]
        if not containing_candidates:
            return None
        containing = min(
            containing_candidates,
            key=lambda item: self._node_span(item)[1] - self._node_span(item)[0],
        )
        parameters = self._function_parameter_names(containing)
        parameter_index = parameters.index(name)
        for bound_function, bound_call, caller_index in reversed(
            self._bound_helper_calls
        ):
            if bound_function is not containing:
                continue
            argument = self._call_argument_for_parameter(
                bound_call,
                containing,
                parameter_index,
                name,
            )
            if argument is None:
                continue
            key = f"bound-parameter:{self._node_span(containing)[0]}:{name}"
            if key in seen:
                return None
            if mode == "origin" and self._python_fail_closed_nonempty_guard(
                name,
                index,
            ):
                return self._resolve_python_origin_without_empty_paths(
                    argument,
                    caller_index,
                    seen=seen | frozenset({key}),
                )
            return self._resolve_node(
                argument,
                caller_index,
                mode=mode,
                seen=seen | frozenset({key}),
            )
        caller_arguments = [
            (call, argument)
            for call in self._call_nodes
            if self._external_call_targets_function(call, containing)
            if (
                argument := self._call_argument_for_parameter(
                    call,
                    containing,
                    parameter_index,
                    name,
                )
            )
            is not None
        ]
        reachable_arguments = [
            item
            for item in caller_arguments
            if self._python_call_has_entry_path(item[0], seen=frozenset())
        ]
        if reachable_arguments:
            caller_arguments = reachable_arguments
        if not caller_arguments:
            positional = (*containing.args.posonlyargs, *containing.args.args)
            default_start = len(positional) - len(containing.args.defaults)
            default: ast.AST | None = None
            if default_start <= parameter_index < len(positional):
                default = containing.args.defaults[parameter_index - default_start]
            else:
                keyword_index = parameter_index - len(positional)
                if 0 <= keyword_index < len(containing.args.kw_defaults):
                    default = containing.args.kw_defaults[keyword_index]
            if default is not None:
                return self._resolve_node(
                    default,
                    self._node_span(containing)[0],
                    mode=mode,
                    seen=seen,
                )
            return None
        key = f"parameter:{containing.name}:{name}"
        if key in seen:
            return None
        exclude_empty = mode == "origin" and self._python_fail_closed_nonempty_guard(
            name,
            index,
        )
        values: list[ResolvedValue] = []
        for call, argument in caller_arguments:
            caller_index = self._node_span(call)[0]
            resolved = (
                self._resolve_python_origin_without_empty_paths(
                    argument,
                    caller_index,
                    seen=seen | frozenset({key}),
                )
                if exclude_empty
                else self._resolve_node(
                    argument,
                    caller_index,
                    mode=mode,
                    seen=seen | frozenset({key}),
                )
            )
            if resolved is not None:
                values.append(resolved)
        return _merge_resolved(name, values)

    def _resolve_bound_helper_returns(
        self,
        call: ast.Call,
        helpers: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
        *,
        caller_index: int,
        mode: str,
        seen: frozenset[str],
    ) -> list[ResolvedValue]:
        values: list[ResolvedValue] = []
        for helper in helpers:
            self._bound_helper_calls.append((helper, call, caller_index))
            try:
                values.extend(
                    self._resolve_node(
                        returned,
                        self._node_span(returned)[0],
                        mode=mode,
                        seen=seen,
                    )
                    for returned in self._direct_function_returns(helper)
                )
            finally:
                popped = self._bound_helper_calls.pop()
                if popped != (helper, call, caller_index):
                    raise RuntimeError("Python helper-call binding stack was corrupted")
        return values

    def _python_call_is_reachable(self, call: ast.Call) -> bool:
        return self._python_node_is_reachable(call, seen=frozenset())

    def _python_call_has_entry_path(
        self,
        call: ast.Call,
        *,
        seen: frozenset[str],
    ) -> bool:
        if self._reachability_depth >= _MAX_RESOLUTION_STEPS:
            return False
        self._reachability_depth += 1
        try:
            return self._python_call_has_entry_path_impl(call, seen=seen)
        finally:
            self._reachability_depth -= 1

    def _python_call_has_entry_path_impl(
        self,
        call: ast.Call,
        *,
        seen: frozenset[str],
    ) -> bool:
        """Return whether a call is reached from module scope or ``main``.

        Uncalled functions remain valid analysis roots when a source is a
        library-style snippet. Parameter resolution uses this stricter check
        only to prefer concrete entry paths when at least one exists, avoiding
        dead helper calls from polluting an otherwise resolved parameter.
        """
        if len(seen) >= _MAX_RESOLUTION_STEPS:
            return False
        if not self._python_direct_path_is_reachable(call):
            return False
        function = self._enclosing_function(call)
        if function is None:
            return True
        key = f"python-entry-path:{self._node_span(function)[0]}"
        if key in seen:
            return False
        callers = [
            candidate
            for candidate in self._call_nodes
            if self._external_call_targets_function(candidate, function)
        ]
        return any(
            self._python_call_has_entry_path(
                candidate,
                seen=seen | frozenset({key}),
            )
            for candidate in callers
        )

    def _call_argument_for_parameter(
        self,
        call: ast.Call,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        parameter_index: int,
        parameter_name: str,
    ) -> ast.AST | None:
        for keyword in call.keywords:
            if keyword.arg == parameter_name:
                return keyword.value

        parameters = self._function_parameter_names(function)
        bound_receiver = bool(
            parameters
            and parameters[0] in {"self", "cls"}
            and (
                isinstance(call.func, ast.Attribute)
                or function.name == "__init__"
            )
        )
        call_index = parameter_index - int(bound_receiver)
        if 0 <= call_index < len(call.args):
            return call.args[call_index]
        if call_index < 0:
            return None

        positional = (*function.args.posonlyargs, *function.args.args)
        default_start = len(positional) - len(function.args.defaults)
        if default_start <= parameter_index < len(positional):
            return function.args.defaults[parameter_index - default_start]
        keyword_index = parameter_index - len(positional)
        if 0 <= keyword_index < len(function.args.kw_defaults):
            return function.args.kw_defaults[keyword_index]
        return None

    def _matching_functions(
        self,
        call: ast.Call,
        *,
        caller_index: int | None = None,
    ) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
        key = (id(call), caller_index)
        if key in self._matching_function_calls:
            return ()
        self._matching_function_calls.add(key)
        try:
            return self._matching_functions_impl(
                call,
                caller_index=caller_index,
            )
        finally:
            self._matching_function_calls.remove(key)

    def _external_call_targets_function(
        self,
        call: ast.Call,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        """Return whether an outside call can invoke ``function``.

        Checking containment and the syntactic call name before receiver
        inference avoids recursive caller discovery for same-named API calls
        inside a helper, such as ``key.sign(...)`` inside ``sign(...)``.
        """

        if self._node_inside(call, function):
            return False
        expected_names = {function.name}
        if function.name == "__init__":
            owner = self._enclosing_class(function)
            if owner is not None:
                expected_names.add(owner.name)
        if _python_name(call.func).rsplit(".", 1)[-1] not in expected_names:
            return False
        return function in self._matching_functions(call)

    def _python_class_lineage(
        self,
        class_name: str,
        seen: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        if class_name in seen:
            return frozenset()
        class_node = next(
            (candidate for candidate in self._class_nodes if candidate.name == class_name),
            None,
        )
        if class_node is None:
            return frozenset()
        lineage = {class_name}
        for base in class_node.bases:
            base_name = _python_name(base).rsplit(".", 1)[-1]
            lineage.update(
                self._python_class_lineage(
                    base_name,
                    seen | frozenset({class_name}),
                )
            )
        return frozenset(lineage)

    def _receiver_class_name(
        self,
        call: ast.Call,
        *,
        caller_index: int | None = None,
    ) -> str | None:
        if not isinstance(call.func, ast.Attribute):
            return None
        receiver_node = call.func.value
        receiver_name = _python_name(receiver_node).rsplit(".", 1)[-1]
        inference_index = (
            caller_index
            if caller_index is not None
            else self._node_span(call)[0]
        )
        containing_class = self._enclosing_class(call)
        if receiver_name in {"self", "cls"} and containing_class is not None:
            return containing_class.name

        known_classes = {
            owner.name
            for function in self._function_nodes
            if (owner := self._enclosing_class(function)) is not None
        }
        if receiver_name in known_classes:
            return receiver_name

        constructed_name = ""
        if isinstance(receiver_node, ast.Call):
            constructed_name = _python_name(receiver_node.func).rsplit(".", 1)[-1]
        elif isinstance(receiver_node, ast.Name):
            assignment = self._latest_assignment(
                receiver_node.id,
                inference_index,
            )
            if assignment is not None and isinstance(assignment[0], ast.Call):
                constructed_name = _python_name(
                    assignment[0].func,
                ).rsplit(".", 1)[-1]
            if not constructed_name:
                containing_function = self._enclosing_function(call)
                if containing_function is not None:
                    parameters = self._function_parameter_names(
                        containing_function,
                    )
                    if receiver_node.id in parameters:
                        parameter_index = parameters.index(receiver_node.id)
                        caller_constructions = {
                            _python_name(construction.func).rsplit(".", 1)[-1]
                            for caller in self._call_nodes
                            if self._external_call_targets_function(
                                caller,
                                containing_function,
                            )
                            if (
                                argument := self._call_argument_for_parameter(
                                    caller,
                                    containing_function,
                                    parameter_index,
                                    receiver_node.id,
                                )
                            )
                            is not None
                            for construction in self._python_constructions(
                                argument,
                                self._node_span(caller)[0],
                            )
                        } & known_classes
                        if len(caller_constructions) == 1:
                            constructed_name = next(iter(caller_constructions))
        elif isinstance(receiver_node, ast.Attribute):
            attribute_name = _python_name(receiver_node)
            if attribute_name.startswith(("self.", "cls.")):
                assignment = self._python_instance_attribute_assignment(
                    attribute_name,
                    inference_index,
                )
                if assignment is not None and isinstance(assignment[0], ast.Call):
                    constructed_name = _python_name(
                        assignment[0].func,
                    ).rsplit(".", 1)[-1]
        if constructed_name in known_classes:
            return constructed_name

        inferred_classes = {
            _python_name(construction.func).rsplit(".", 1)[-1]
            for construction in self._python_constructions(
                receiver_node,
                inference_index,
            )
        } & known_classes
        return next(iter(inferred_classes)) if len(inferred_classes) == 1 else None

    def _direct_function_returns(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[ast.AST, ...]:
        return tuple(
            item.value
            for item in ast.walk(function)
            if isinstance(item, ast.Return)
            and item.value is not None
            and self._enclosing_function(item) is function
            and self._python_direct_path_is_reachable(item)
        )

    def _containing_function(self, index: int) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        matches = [
            function
            for function in self._function_nodes
            if self._node_span(function)[0] <= index <= self._node_span(function)[1]
        ]
        return min(matches, key=lambda item: self._node_span(item)[1] - self._node_span(item)[0]) if matches else None

    def containing_function_name(self, index: int) -> str | None:
        function = self._containing_function(index)
        return function.name if function is not None else None

    def containing_function_is_called(self, index: int) -> bool:
        function = self._containing_function(index)
        if function is None:
            return False
        return any(
            self._external_call_targets_function(call, function)
            for call in self._call_nodes
        )

    def position_is_locally_reachable(self, index: int) -> bool:
        candidates = [
            node
            for node in self._call_nodes
            if self._node_span(node)[0] == index
        ]
        if not candidates:
            candidates = [
                node
                for node in ast.walk(self.tree)
                if self._node_span(node)[0] <= index <= self._node_span(node)[1]
            ]
        if not candidates:
            return True
        node = min(
            candidates,
            key=lambda item: self._node_span(item)[1] - self._node_span(item)[0],
        )
        return self._python_direct_path_is_reachable(node)

    def position_is_reachable(self, index: int) -> bool:
        candidates = [
            node
            for node in self._call_nodes
            if self._node_span(node)[0] == index
        ]
        if not candidates:
            candidates = [
                node
                for node in ast.walk(self.tree)
                if self._node_span(node)[0] <= index <= self._node_span(node)[1]
            ]
        if not candidates:
            return True
        node = min(
            candidates,
            key=lambda item: self._node_span(item)[1] - self._node_span(item)[0],
        )
        return self._python_node_is_reachable(node, seen=frozenset())

    def _python_node_is_reachable(
        self,
        node: ast.AST,
        *,
        seen: frozenset[str],
    ) -> bool:
        if self._reachability_depth >= _MAX_RESOLUTION_STEPS:
            return True
        self._reachability_depth += 1
        try:
            return self._python_node_is_reachable_impl(node, seen=seen)
        finally:
            self._reachability_depth -= 1

    def _python_node_is_reachable_impl(
        self,
        node: ast.AST,
        *,
        seen: frozenset[str],
    ) -> bool:
        if len(seen) >= _MAX_RESOLUTION_STEPS:
            return True
        if not self._python_direct_path_is_reachable(node):
            return False
        function = self._enclosing_function(node)
        if function is None or function.name.lower() == "main":
            return True
        key = f"python-function-reachability:{self._node_span(function)[0]}"
        if key in seen:
            return True
        callers = [
            call
            for call in self._call_nodes
            if self._external_call_targets_function(call, function)
        ]
        if not callers and self._returned_nested_function_is_called(
            function,
            seen=seen | frozenset({key}),
        ):
            return True
        if not callers:
            if self._python_framework_callback_has_entry_path(
                function,
                seen=seen | frozenset({key}),
            ):
                return True
            # Treat function bodies as library roots only when the snippet has
            # no executable local entry path. Once module-level code calls a
            # project-local function/class, other uncalled helpers are dormant.
            return not self._has_explicit_local_entrypoint()
        return any(
            self._python_node_is_reachable(
                call,
                seen=seen | frozenset({key}),
            )
            for call in callers
        )

    def _python_direct_path_is_reachable(self, node: ast.AST) -> bool:
        if node in self._inactive_method_body_nodes:
            return False
        containing = self._enclosing_function(node)
        statements = containing.body if containing is not None else self.tree.body
        node_start = self._node_span(node)[0]
        for statement in statements:
            if self._node_span(statement)[1] >= node_start:
                break
            if not (
                isinstance(statement, ast.If)
                and _python_statements_terminate(statement.body)
            ):
                continue
            condition_id = id(statement.test)
            if condition_id in self._condition_evaluation_nodes:
                continue
            self._condition_evaluation_nodes.add(condition_id)
            try:
                blocked = self._evaluate_python_condition(statement.test)
            finally:
                self._condition_evaluation_nodes.remove(condition_id)
            if blocked is True:
                return False

        current = node
        parent = self._parents.get(current)
        while parent is not None:
            if isinstance(parent, ast.If):
                condition_id = id(parent.test)
                if condition_id in self._condition_evaluation_nodes:
                    condition = None
                else:
                    self._condition_evaluation_nodes.add(condition_id)
                    try:
                        condition = self._evaluate_python_condition(parent.test)
                    finally:
                        self._condition_evaluation_nodes.remove(condition_id)
                if condition is not None:
                    in_body = any(
                        current is statement or self._node_inside(current, statement)
                        for statement in parent.body
                    )
                    in_else = any(
                        current is statement or self._node_inside(current, statement)
                        for statement in parent.orelse
                    )
                    if (in_body and not condition) or (in_else and condition):
                        return False
            current = parent
            parent = self._parents.get(current)
        return True

    def _evaluate_python_condition(
        self,
        node: ast.AST,
        *,
        index: int | None = None,
    ) -> bool | None:
        if (
            self.analysis_profile == "reachable_security_flow"
            and isinstance(node, ast.Constant)
            and isinstance(node.value, bool)
        ):
            return node.value
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            return None
        resolution_index = (
            index
            if index is not None
            else self._node_span(node)[0]
        )
        implied = self._python_prior_terminating_guard_truth(
            node,
            resolution_index,
        )
        if implied is not None:
            return implied
        operator_node = node.ops[0]
        comparator = node.comparators[0]
        if isinstance(operator_node, (ast.Is, ast.IsNot)):
            none_state: bool | None = None
            if isinstance(comparator, ast.Constant) and comparator.value is None:
                none_state = self._python_parameter_is_always_none(
                    node.left,
                    resolution_index,
                )
            elif isinstance(node.left, ast.Constant) and node.left.value is None:
                none_state = self._python_parameter_is_always_none(
                    comparator,
                    resolution_index,
                )
            if none_state is not None:
                return (
                    none_state
                    if isinstance(operator_node, ast.Is)
                    else not none_state
                )
        left_numeric = self._resolve_node(
            node.left,
            resolution_index,
            mode="int",
            seen=frozenset(),
        )
        right_numeric = self._resolve_node(
            comparator,
            resolution_index,
            mode="int",
            seen=frozenset(),
        )
        if (
            _python_resolved_is_fixed_for_branch(left_numeric)
            and _python_resolved_is_fixed_for_branch(right_numeric)
        ):
            try:
                left_value = int(left_numeric.values[0].replace("_", ""))
                right_value = int(right_numeric.values[0].replace("_", ""))
            except ValueError:
                pass
            else:
                if isinstance(operator_node, ast.Eq):
                    return left_value == right_value
                if isinstance(operator_node, ast.NotEq):
                    return left_value != right_value

        left = self._resolve_node(
            node.left,
            resolution_index,
            mode="string",
            seen=frozenset(),
        )
        membership_container = comparator
        if isinstance(membership_container, ast.Name):
            assignment = self._latest_assignment(
                membership_container.id,
                resolution_index,
            )
            if assignment is not None:
                membership_container = assignment[0]
        if isinstance(operator_node, (ast.In, ast.NotIn)) and isinstance(
            membership_container,
            (ast.Dict, ast.List, ast.Tuple, ast.Set),
        ):
            if not _python_resolved_is_fixed_for_branch(left):
                return None
            elements = (
                tuple(
                    key
                    for key in membership_container.keys
                    if key is not None
                )
                if isinstance(membership_container, ast.Dict)
                else tuple(membership_container.elts)
            )
            choices = {
                value
                for element in elements
                for value in self._resolve_node(
                    element,
                    resolution_index,
                    mode="string",
                    seen=frozenset(),
                ).values
            }
            contained = left.values[0] in choices
            return not contained if isinstance(operator_node, ast.NotIn) else contained
        right = self._resolve_node(
            comparator,
            resolution_index,
            mode="string",
            seen=frozenset(),
        )
        if not (
            _python_resolved_is_fixed_for_branch(left)
            and _python_resolved_is_fixed_for_branch(right)
        ):
            return None
        equal = left.values[0] == right.values[0]
        if isinstance(operator_node, ast.Eq):
            return equal
        if isinstance(operator_node, ast.NotEq):
            return not equal
        return None

    def _python_parameter_is_always_none(
        self,
        node: ast.AST,
        index: int,
    ) -> bool | None:
        if (
            isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            containing = self._containing_function(index)
            owner = self._enclosing_class(containing) if containing is not None else None
            bindings = [
                item for item in self._assignments.get(ast.unparse(node), ())
                if self._enclosing_class(item[0]) is owner
            ]
            # Any later field mutation or dynamic setter makes this proof
            # inapplicable. Only a single constructor parameter copy is used.
            stores = [
                item for item in ast.walk(self.tree)
                if isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Store)
                and item.attr == node.attr
            ]
            if owner is None or len(bindings) != 1 or len(stores) != 1:
                return None
            if any(
                isinstance(call.func, ast.Name) and call.func.id == "setattr"
                for call in self._call_nodes
            ):
                return None
            assigned, assigned_index = bindings[0]
            constructor = self._enclosing_function(assigned)
            if constructor is None or constructor.name != "__init__":
                return None
            statement = self._parents.get(assigned)
            if statement not in constructor.body:
                return None
            if isinstance(assigned, ast.Constant):
                return assigned.value is None
            if isinstance(assigned, ast.Name):
                return self._python_parameter_is_always_none(assigned, assigned_index)
            return None
        if not isinstance(node, ast.Name):
            return None
        containing = self._containing_function(index)
        if containing is None:
            return None
        parameters = self._function_parameter_names(containing)
        if node.id not in parameters:
            return None
        parameter_index = parameters.index(node.id)
        caller_arguments = [
            (call, argument)
            for call in self._call_nodes
            if self._external_call_targets_function(call, containing)
            if (
                argument := self._call_argument_for_parameter(
                    call,
                    containing,
                    parameter_index,
                    node.id,
                )
            )
            is not None
        ]
        reachable = [
            item
            for item in caller_arguments
            if self._python_call_has_entry_path(item[0], seen=frozenset())
        ]
        if reachable:
            caller_arguments = reachable
        if not caller_arguments:
            return None
        states = [
            self._python_none_state(argument, self._node_span(call)[0])
            for call, argument in caller_arguments
        ]
        if all(state is True for state in states):
            return True
        if all(state is False for state in states):
            return False
        return None

    def _node_inside(self, node: ast.AST, parent: ast.AST) -> bool:
        current = self._parents.get(node)
        while current is not None:
            if current is parent:
                return True
            current = self._parents.get(current)
        return False

    def _enclosing_function(
        self,
        node: ast.AST,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        current = self._parents.get(node)
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current
            current = self._parents.get(current)
        return None

    def _enclosing_class(self, node: ast.AST) -> ast.ClassDef | None:
        current = self._parents.get(node)
        while current is not None:
            if isinstance(current, ast.ClassDef):
                return current
            current = self._parents.get(current)
        return None

    @staticmethod
    def _function_parameter_names(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        arguments = function.args
        return [
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        ]

    def _segment(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.source, node) or ast.unparse(node)

    def _node_span(self, node: ast.AST) -> tuple[int, int]:
        start = self._line_offsets[max(0, getattr(node, "lineno", 1) - 1)] + getattr(node, "col_offset", 0)
        end_line = max(0, getattr(node, "end_lineno", getattr(node, "lineno", 1)) - 1)
        end = self._line_offsets[end_line] + getattr(node, "end_col_offset", getattr(node, "col_offset", 0))
        return start, end
    def integer_expression_guard_values(
        self,
        expression: str,
        index: int,
    ) -> tuple[int, ...]:
        """Return an exact fail-closed integer allowlist before one sink.

        Only a simple parameter or attribute checked against a literal integer
        container in the same function is accepted. A local reassignment before
        the guard is deliberately excluded because it may change the value's
        shape instead of merely validating it.
        """

        try:
            parsed = ast.parse(expression.strip(), mode="eval").body
        except SyntaxError:
            return ()
        if not isinstance(parsed, (ast.Name, ast.Attribute)):
            return ()
        guarded_expression = ast.unparse(parsed)
        sink_node = next(
            (
                node
                for node in self._call_nodes
                if self._node_span(node)[0] == index
            ),
            None,
        )
        if sink_node is None:
            return ()
        function = self._enclosing_function(sink_node)
        if function is None:
            return ()

        for statement in function.body:
            statement_start, statement_end = self._node_span(statement)
            if statement_start >= index:
                break
            if _python_statement_assigns_reference(statement, guarded_expression):
                return ()
            if not (
                statement_end <= index
                and isinstance(statement, ast.If)
                and _python_statements_terminate(statement.body)
            ):
                continue
            values = _python_guard_allowed_integer_values(
                statement.test,
                guarded_expression,
            )
            if values:
                return values
        return ()

    def call_is_container_literal_value(self, call: CallExpression) -> bool:
        """Return whether a call only constructs a value inside a literal container."""

        node = next(
            (
                candidate
                for candidate in self._call_nodes
                if self._node_span(candidate) == (call.start, call.end)
            ),
            None,
        )
        if node is None:
            return False
        parent = self._parents.get(node)
        return isinstance(parent, (ast.Dict, ast.List, ast.Tuple, ast.Set))

    def associated_kdf_secret_expressions(
        self,
        call: CallExpression,
    ) -> tuple[str, ...]:
        """Return secrets supplied to ``derive``/``verify`` for a KDF object.

        PyCA separates KDF configuration from the secret input: the salt is
        passed to ``PBKDF2HMAC(...)`` while the password reaches a later
        ``kdf.derive(password)`` call.  Detectors that compare those two
        origins therefore need this small constructor-to-use link.
        """

        constructor = next(
            (
                node
                for node in self._call_nodes
                if self._node_span(node) == (call.start, call.end)
            ),
            None,
        )
        if constructor is None:
            return ()

        receivers: set[str] = set()
        parent = self._parents.get(constructor)
        if isinstance(parent, ast.Assign):
            receivers.update(
                target.id for target in parent.targets if isinstance(target, ast.Name)
            )
        elif isinstance(parent, ast.AnnAssign) and isinstance(parent.target, ast.Name):
            receivers.add(parent.target.id)

        candidates: list[ast.Call] = []
        if (
            isinstance(parent, ast.Attribute)
            and parent.attr in {"derive", "verify"}
            and isinstance(self._parents.get(parent), ast.Call)
        ):
            candidates.append(self._parents[parent])
        candidates.extend(
            node
            for node in self._call_nodes
            if isinstance(node.func, ast.Attribute)
            and node.func.attr in {"derive", "verify"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in receivers
            and self._node_span(node)[0] >= call.end
            and self._enclosing_function(node) is self._enclosing_function(constructor)
            and self._python_call_is_reachable(node)
        )

        expressions: list[str] = []
        for candidate in candidates:
            secret = candidate.args[0] if candidate.args else next(
                (
                    keyword.value
                    for keyword in candidate.keywords
                    if keyword.arg in {"key_material", "data"}
                ),
                None,
            )
            if secret is not None:
                value = self._segment(secret)
                if value not in expressions:
                    expressions.append(value)
        return tuple(expressions)

    def _resolve_python_written_prefix(
        self,
        writes: Iterable[tuple[ast.AST, int]],
        length: int,
        *,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        """Resolve the origins that contribute to a persisted byte prefix."""

        if length == 0:
            return ResolvedValue("file prefix", "constant")
        remaining = length
        values: list[ResolvedValue] = []
        for value, value_index in writes:
            value_length = self._python_known_byte_length(
                value,
                value_index,
                seen=seen,
            )
            if value_length is None or remaining <= value_length:
                selected = self._resolve_python_prefix_slice(
                    value,
                    remaining,
                    value_index,
                    mode="origin",
                    seen=seen,
                )
                if selected is None:
                    selected = self._resolve_node(
                        value,
                        value_index,
                        mode="origin",
                        seen=seen,
                    )
                values.append(selected)
                remaining = 0
                break
            values.append(
                self._resolve_node(
                    value,
                    value_index,
                    mode="origin",
                    seen=seen,
                )
            )
            remaining -= value_length
        if not values or remaining > 0:
            return None
        return _merge_transformed_origins("file prefix", values)

    def _resolve_python_fresh_custom_derivation_return(
        self,
        node: ast.Call,
        index: int,
        *,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        """Prefer proven-fresh helper returns over unused secret arguments.

        Only ``secure_random`` is accepted here. Derived outputs continue
        through the custom-KDF input policy below.
        """

        call_name = _python_name(node.func).rsplit(".", 1)[-1]
        if re.search(r"(?:^|_)salt(?:_|$)", call_name, re.IGNORECASE):
            return None
        if re.search(
            r"(?i)(?:derive|derivation|hkdf|kdf|expand).*(?:key|token|secret|material)"
            r"|(?:key|token|secret|material).*(?:derive|derivation|hkdf|kdf|expand)"
            r"|(?:^|_)(?:hkdf|kdf)(?:_|$)",
            call_name,
        ) is None:
            return None
        functions = self._matching_functions(node, caller_index=index)
        if not functions:
            return None

        direct_returns = tuple(
            self._direct_function_returns(function) for function in functions
        )
        if any(not returned for returned in direct_returns):
            return None
        canonical_name = _canonical_python_name(
            _python_name(node.func),
            self.import_aliases,
        )
        helper_key = f"function:{canonical_name}"
        if helper_key in seen:
            return None
        resolved = self._resolve_bound_helper_returns(
            node,
            functions,
            caller_index=index,
            mode="origin",
            seen=seen | frozenset({helper_key}),
        )
        if not resolved or any(
            value.origin != "secure_random" for value in resolved
        ):
            return None
        return _merge_resolved(self._segment(node), resolved)

    def _resolve_python_guarded_equivalent_origin(
        self,
        reference: str,
        index: int,
        *,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        """Resolve a value proven equal to another origin before a sink."""

        containing = self._containing_function(index)
        tree: ast.AST = containing if containing is not None else self.tree
        sink_nodes = tuple(
            node
            for node in self._call_nodes
            if self._node_span(node)[0] == index
        )

        def other_side(
            condition: ast.AST,
            operator_types: tuple[type[ast.cmpop], ...],
        ) -> ast.AST | None:
            if isinstance(condition, ast.BoolOp) and isinstance(
                condition.op,
                ast.And,
            ):
                return next(
                    (
                        candidate
                        for term in condition.values
                        if (
                            candidate := other_side(term, operator_types)
                        )
                        is not None
                    ),
                    None,
                )
            if (
                not isinstance(condition, ast.Compare)
                or len(condition.ops) != 1
                or not isinstance(condition.ops[0], operator_types)
            ):
                return None
            left, right = condition.left, condition.comparators[0]
            if ast.unparse(left) == reference and ast.unparse(right) != reference:
                return right
            if ast.unparse(right) == reference and ast.unparse(left) != reference:
                return left
            return None

        candidates: list[tuple[int, ast.AST]] = []
        for candidate in ast.walk(tree):
            start, end = self._node_span(candidate)
            guard_key = f"guarded-equivalence-condition:{start}"
            if guard_key in seen:
                continue
            if isinstance(candidate, ast.Assert) and end < index:
                equivalent = other_side(candidate.test, (ast.Eq, ast.Is))
                if equivalent is not None:
                    candidates.append((start, equivalent))
                continue
            if not isinstance(candidate, ast.If):
                continue
            if end < index and _python_statements_terminate(candidate.body):
                equivalent = other_side(candidate.test, (ast.NotEq, ast.IsNot))
                if equivalent is not None:
                    candidates.append((start, equivalent))
            if any(
                self._node_inside(sink, statement)
                for sink in sink_nodes
                for statement in candidate.body
            ):
                equivalent = other_side(candidate.test, (ast.Eq, ast.Is))
                if equivalent is not None:
                    candidates.append((start, equivalent))

        if not candidates:
            return None
        guard_start, equivalent_node = max(candidates, key=lambda item: item[0])
        key = f"guarded-equivalent:{reference}:{index}"
        if key in seen:
            return None
        return self._resolve_node(
            equivalent_node,
            self._node_span(equivalent_node)[0],
            mode="origin",
            seen=seen
            | frozenset(
                {key, f"guarded-equivalence-condition:{guard_start}"}
            ),
        )

    def _python_reference_is_truthy_on_path(
        self,
        reference: str,
        index: int,
    ) -> bool:
        """Recognize a sink enclosed by a positive nonempty branch."""

        containing = self._containing_function(index)
        tree = containing if containing is not None else self.tree
        for candidate in ast.walk(tree):
            if not isinstance(candidate, ast.If):
                continue
            if not any(
                self._node_span(statement)[0]
                <= index
                <= self._node_span(statement)[1]
                for statement in candidate.body
            ):
                continue
            if _python_condition_accepts_nonempty(candidate.test, reference):
                return True
        return False

    def _python_empty_value_replaced_before(
        self,
        reference: str,
        index: int,
    ) -> bool:
        """Recognize ``if value is None: value = replacement`` before use."""

        containing = self._containing_function(index)
        tree = containing if containing is not None else self.tree
        for candidate in ast.walk(tree):
            if not isinstance(candidate, ast.If):
                continue
            start, end = self._node_span(candidate)
            if start >= index or end > index:
                continue
            if not _python_condition_rejects_empty(candidate.test, reference):
                continue
            for statement in candidate.body:
                targets: tuple[ast.AST, ...]
                if isinstance(statement, ast.Assign):
                    targets = tuple(statement.targets)
                elif isinstance(statement, ast.AnnAssign):
                    targets = (statement.target,)
                else:
                    targets = tuple()
                if any(
                    reference in _python_target_names(target)
                    for target in targets
                ):
                    return True
                if isinstance(statement, (ast.Raise, ast.Return)):
                    break
        return False

    def _resolve_python_appended_value(
        self,
        receiver: ast.AST,
        pop_call: ast.Call,
        index: int,
        *,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        receiver_name = _python_name(receiver)
        if not receiver_name:
            return None
        appends = [
            item
            for item in self._append_arguments.get(receiver_name, ())
            if item[1] <= index
        ]
        if not appends:
            return None
        position = -1
        if pop_call.args:
            selected = _evaluate_int_ast(pop_call.args[0])
            if selected is None:
                return None
            position = selected
        if position < 0:
            position += len(appends)
        if not 0 <= position < len(appends):
            return None
        value, value_index = appends[position]
        return self._resolve_node(
            value,
            value_index,
            mode="origin",
            seen=seen | frozenset({f"pop:{receiver_name}:{index}"}),
        )

    def _resolve_python_cipher_key_property(
        self,
        receiver: ast.AST,
        index: int,
        *,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        constructor: ast.Call | None = None
        if isinstance(receiver, ast.Call):
            constructor = receiver
        elif isinstance(receiver, ast.Name):
            assignment = self._latest_assignment(receiver.id, index)
            if assignment is not None and isinstance(assignment[0], ast.Call):
                constructor = assignment[0]
        if constructor is None:
            return None
        constructor_name = _canonical_python_name(
            _python_name(constructor.func),
            self.import_aliases,
        ).rsplit(".", 1)[-1]
        if constructor_name != "Cipher" or not constructor.args:
            return None
        algorithm = constructor.args[0]
        if (
            not isinstance(algorithm, ast.Call)
            or _canonical_python_name(
                _python_name(algorithm.func),
                self.import_aliases,
            ).rsplit(".", 1)[-1]
            != "AES"
            or not algorithm.args
        ):
            return None
        return self._resolve_node(
            algorithm.args[0],
            self._node_span(algorithm)[0],
            mode="origin",
            seen=seen | frozenset({f"cipher-key:{self._node_span(constructor)[0]}"}),
        )

    def _resolve_python_free_global_at_callers(
        self,
        name: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        """Resolve a function's free variable at the call that executes it.

        A function body is lexically earlier than common module-level setup
        such as ``key = key.ljust(...)``.  Resolving the free name at its
        definition therefore sees a stale assignment.  Follow external calls
        to the function and use the latest module binding that precedes each
        call instead.
        """

        containing = self._containing_function(index)
        if containing is None or name in self._function_parameter_names(containing):
            return None
        assignments = tuple(self._assignments.get(name, ()))
        if not assignments or any(
            self._enclosing_function(node) is containing
            for node, _ in assignments
        ):
            return None
        globals_only = tuple(
            item
            for item in assignments
            if self._enclosing_function(item[0]) is None
        )
        if not globals_only:
            return None

        calls = tuple(
            call
            for call in self._call_nodes
            if self._external_call_targets_function(call, containing)
        )
        if not calls:
            return None
        values: list[ResolvedValue] = []
        for call in calls:
            call_index = self._node_span(call)[0]
            reaching = tuple(
                item for item in globals_only if item[1] <= call_index
            )
            if not reaching:
                continue
            assigned_node, assigned_index = reaching[-1]
            values.append(
                self._resolve_node(
                    assigned_node,
                    assigned_index,
                    mode=mode,
                    seen=seen,
                )
            )
        return _merge_resolved(name, values) if values else None

    def _python_constructed_attribute_nodes(
        self,
        node: ast.Attribute,
        index: int,
    ) -> tuple[ast.AST, ...]:
        """Return source nodes bound to a field of a local construction.

        Value resolution already follows these bindings.  Container-aware
        operations such as ``request.headers.get(...)`` additionally need the
        original ``dict`` node so they can select a concrete key.
        """

        constructions = list(self._python_constructions(node.value, index))
        if isinstance(node.value, ast.Name) and not constructions:
            containing = self._containing_function(index)
            if (
                containing is not None
                and node.value.id in self._function_parameter_names(containing)
            ):
                parameters = self._function_parameter_names(containing)
                parameter_index = parameters.index(node.value.id)
                callers = [
                    (call, argument)
                    for call in self._call_nodes
                    if self._external_call_targets_function(call, containing)
                    if (
                        argument := self._call_argument_for_parameter(
                            call,
                            containing,
                            parameter_index,
                            node.value.id,
                        )
                    )
                    is not None
                ]
                reachable = [
                    item
                    for item in callers
                    if self._python_call_has_entry_path(item[0], seen=frozenset())
                ]
                if reachable:
                    callers = reachable
                for call, argument in callers:
                    constructions.extend(
                        self._python_constructions(
                            argument,
                            self._node_span(call)[0],
                        )
                    )
        values: list[ast.AST] = []
        for construction in constructions:
            class_name = _python_name(construction.func).rsplit(".", 1)[-1]
            class_node = next(
                (candidate for candidate in self._class_nodes if candidate.name == class_name),
                None,
            )
            if class_node is None:
                continue
            constructor = next(
                (
                    statement
                    for statement in class_node.body
                    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and statement.name == "__init__"
                    and statement not in self._shadowed_methods
                ),
                None,
            )
            if constructor is None:
                continue
            assignment = next(
                (
                    item
                    for item in reversed(
                        self._assignments.get(f"self.{node.attr}", ())
                    )
                    if self._enclosing_function(item[0]) is constructor
                ),
                None,
            )
            if assignment is None:
                continue
            assigned_node = assignment[0]
            if isinstance(assigned_node, ast.Name):
                parameters = self._function_parameter_names(constructor)
                if assigned_node.id in parameters:
                    argument = self._call_argument_for_parameter(
                        construction,
                        constructor,
                        parameters.index(assigned_node.id),
                        assigned_node.id,
                    )
                    if argument is not None:
                        values.append(argument)
                        continue
            values.append(assigned_node)
        return tuple(values)

    def _resolve_python_struct_unpack_value(
        self,
        unpack_call: ast.Call,
        tuple_index: int,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        """Link a local ``struct.unpack`` result to matching packed values."""

        if tuple_index < 0 or not unpack_call.args:
            return None
        unpack_formats = set(
            self._resolve_node(
                unpack_call.args[0],
                index,
                mode="string",
                seen=seen,
            ).values
        )
        if not unpack_formats:
            return None

        candidates: list[tuple[ast.AST, int]] = []
        for call in self._call_nodes:
            call_name = _canonical_python_name(
                _python_name(call.func),
                self.import_aliases,
            )
            if call_name != "struct.pack" or len(call.args) <= tuple_index + 1:
                continue
            pack_formats = set(
                self._resolve_node(
                    call.args[0],
                    self._node_span(call)[0],
                    mode="string",
                    seen=seen,
                ).values
            )
            if unpack_formats.isdisjoint(pack_formats):
                continue
            candidates.append((call.args[tuple_index + 1], self._node_span(call)[0]))
        if len(candidates) != 1:
            return None
        candidate, candidate_index = candidates[0]
        return self._resolve_node(
            candidate,
            candidate_index,
            mode=mode,
            seen=seen,
        )

    def _resolve_python_loop_callable(
        self,
        node: ast.Call,
        index: int,
        *,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        if not isinstance(node.func, ast.Name) or node.args or node.keywords:
            return None
        callable_name = node.func.id
        loops = [
            loop
            for loop in ast.walk(self.tree)
            if isinstance(loop, (ast.For, ast.AsyncFor))
            and callable_name in _python_target_names(loop.target)
            and self._node_span(loop)[0] <= index <= self._node_span(loop)[1]
        ]
        if not loops:
            return None
        loop = max(loops, key=lambda item: self._node_span(item)[0])
        iterable: ast.AST = loop.iter
        if isinstance(iterable, ast.Call) and (
            _python_name(iterable.func).rsplit(".", 1)[-1] == "enumerate"
            and iterable.args
        ):
            iterable = iterable.args[0]
        if isinstance(iterable, ast.Name):
            assignment = self._latest_assignment(
                iterable.id,
                self._node_span(loop)[0],
            )
            if assignment is None:
                return None
            iterable = assignment[0]
        if not isinstance(iterable, (ast.List, ast.Tuple, ast.Set)):
            return None
        key = f"loop-callable:{callable_name}:{index}"
        if key in seen:
            return None
        values: list[ResolvedValue] = []
        for callable_node in iterable.elts:
            helper_name = _python_name(callable_node).rsplit(".", 1)[-1]
            for helper in self._functions_by_name.get(helper_name, ()):
                if self._python_helper_uses_secure_random(
                    helper,
                    seen=frozenset(),
                ):
                    values.append(
                        ResolvedValue(ast.unparse(callable_node), "secure_random")
                    )
                    continue
                values.extend(
                    self._resolve_node(
                        returned,
                        self._node_span(returned)[0],
                        mode="origin",
                        seen=seen | frozenset({key}),
                    )
                    for returned in self._direct_function_returns(helper)
                )
        return _merge_resolved(ast.unparse(node), values) if values else None

    def _resolve_python_crypto_payload_call(
        self,
        node: ast.Call,
        index: int,
        *,
        seen: frozenset[str],
    ) -> ResolvedValue | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        operation = node.func.attr
        if operation not in {"encrypt", "decrypt"} or not node.args:
            return None
        if len(node.args) >= 3:
            payload = node.args[1]
        else:
            receiver_name = ast.unparse(node.func.value).lower()
            if not any(
                marker in receiver_name
                for marker in ("key", "rsa", "public", "private")
            ):
                return None
            payload = node.args[0]
        return self._resolve_node(
            payload,
            index,
            mode="origin",
            seen=seen,
        )

    def _python_class_is_or_extends(
        self,
        candidate_name: str,
        base_name: str,
        seen: frozenset[str] = frozenset(),
    ) -> bool:
        if candidate_name == base_name:
            return True
        if candidate_name in seen:
            return False
        candidate = next(
            (node for node in self._class_nodes if node.name == candidate_name),
            None,
        )
        if candidate is None:
            return False
        return any(
            self._python_class_is_or_extends(
                _python_name(parent).rsplit(".", 1)[-1],
                base_name,
                seen | frozenset({candidate_name}),
            )
            for parent in candidate.bases
        )

    def _matching_functions_impl(
        self,
        call: ast.Call,
        *,
        caller_index: int | None = None,
    ) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
        simple_name = _python_name(call.func).rsplit(".", 1)[-1]
        candidates = self._functions_by_name.get(simple_name, ())
        if not candidates:
            candidates = tuple(
                function
                for function in self._functions_by_name.get("__init__", ())
                if (
                    owner := self._enclosing_class(function)
                ) is not None
                and owner.name == simple_name
            )
        if not candidates:
            return ()

        constructors = tuple(
            function
            for function in candidates
            if function.name == "__init__"
            and (
                owner := self._enclosing_class(function)
            ) is not None
            and owner.name == simple_name
        )
        if constructors:
            return constructors

        if isinstance(call.func, ast.Name):
            containing_function = self._enclosing_function(call)
            lexical = tuple(
                function
                for function in candidates
                if self._enclosing_function(function) is containing_function
                and self._enclosing_class(function) is None
            )
            if lexical:
                return lexical
            return tuple(
                function
                for function in candidates
                if self._enclosing_function(function) is None
                and self._enclosing_class(function) is None
            )

        if isinstance(call.func, ast.Attribute):
            receiver = call.func.value
            if (
                isinstance(receiver, ast.Call)
                and _python_name(receiver.func).rsplit(".", 1)[-1] == "super"
            ):
                containing_class = self._enclosing_class(call)
                if containing_class is None:
                    return ()
                ancestor_names = (
                    self._python_class_lineage(containing_class.name)
                    - {containing_class.name}
                )
                return tuple(
                    function
                    for function in candidates
                    if (
                        owner := self._enclosing_class(function)
                    ) is not None
                    and owner.name in ancestor_names
                )
            owner_name = self._receiver_class_name(
                call,
                caller_index=caller_index,
            )
            if owner_name is None:
                known_classes = {item.name for item in self._class_nodes}
                inference_index = (
                    caller_index
                    if caller_index is not None
                    else self._node_span(call)[0]
                )
                inferred_classes = {
                    _python_name(construction.func).rsplit(".", 1)[-1]
                    for construction in self._python_constructions(
                        call.func.value,
                        inference_index,
                    )
                } & known_classes
                if (
                    not inferred_classes
                    and isinstance(receiver, ast.Attribute)
                    and isinstance(receiver.value, ast.Name)
                    and receiver.value.id in {"self", "cls"}
                ):
                    containing = self._containing_function(inference_index)
                    owner = self._enclosing_class(containing) if containing is not None else None
                    class_fields = [
                        value for value, _ in self._assignments.get(receiver.attr, ())
                        if self._enclosing_class(value) is owner
                        and self._enclosing_function(value) is None
                    ]
                    if len(class_fields) == 1 and isinstance(class_fields[0], ast.Call):
                        field_class = _python_name(class_fields[0].func)
                        if field_class in known_classes:
                            inferred_classes = {field_class}
                if not inferred_classes:
                    return ()
                allowed_classes: set[str] = set()
                for class_name in inferred_classes:
                    allowed_classes.update(self._python_class_lineage(class_name))
                return tuple(
                    function
                    for function in candidates
                    if (
                        owner := self._enclosing_class(function)
                    ) is not None
                    and owner.name in allowed_classes
                )
            exact = tuple(
                function
                for function in candidates
                if (
                    owner := self._enclosing_class(function)
                ) is not None
                and owner.name == owner_name
            )
            if exact:
                return exact
            lineage = self._python_class_lineage(owner_name)
            return tuple(
                function
                for function in candidates
                if (
                    owner := self._enclosing_class(function)
                ) is not None
                and owner.name in lineage
            )
        return ()

    def position_is_locally_reachable(self, index: int) -> bool:
        candidates = [
            node
            for node in self._call_nodes
            if self._node_span(node)[0] == index
        ]
        if not candidates:
            candidates = [
                node
                for node in ast.walk(self.tree)
                if self._node_span(node)[0] <= index <= self._node_span(node)[1]
            ]
        if not candidates:
            return True
        node = min(
            candidates,
            key=lambda item: self._node_span(item)[1] - self._node_span(item)[0],
        )
        return self._python_direct_path_is_reachable(node)

    def _python_framework_callback_has_entry_path(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        seen: frozenset[str],
    ) -> bool:
        owner = self._enclosing_class(function)
        if (
            owner is None
            or re.fullmatch(r"do_[A-Z]+", function.name) is None
            or not any(
                _canonical_python_name(
                    _python_name(base),
                    self.import_aliases,
                ).endswith("BaseHTTPRequestHandler")
                for base in owner.bases
            )
        ):
            return False
        for call in self._call_nodes:
            canonical = _canonical_python_name(
                _python_name(call.func),
                self.import_aliases,
            )
            if not canonical.endswith(("HTTPServer", "ThreadingHTTPServer")):
                continue
            if not any(
                isinstance(argument, ast.Name) and argument.id == owner.name
                for argument in call.args
            ):
                continue
            if self._python_node_is_reachable(call, seen=seen):
                return True
        return False

    def _returned_nested_function_is_called(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        seen: frozenset[str],
    ) -> bool:
        factory = self._enclosing_function(function)
        if factory is None or not any(
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == function.name
            and self._enclosing_function(node) is factory
            for node in ast.walk(factory)
        ):
            return False

        for factory_call in self._call_nodes:
            if not self._external_call_targets_function(factory_call, factory):
                continue
            if not self._python_node_is_reachable(factory_call, seen=seen):
                continue
            parent = self._parents.get(factory_call)
            if isinstance(parent, ast.Call) and parent.func is factory_call:
                if self._python_node_is_reachable(parent, seen=seen):
                    return True
                continue
            targets: tuple[str, ...] = ()
            if isinstance(parent, ast.Assign) and parent.value is factory_call:
                targets = tuple(
                    name
                    for target in parent.targets
                    for name in _python_target_names(target)
                )
            elif isinstance(parent, ast.AnnAssign) and parent.value is factory_call:
                targets = tuple(_python_target_names(parent.target))
            if not targets:
                continue
            for call in self._call_nodes:
                if self._node_span(call)[0] <= self._node_span(parent)[1]:
                    continue
                if _python_name(call.func) not in targets:
                    continue
                if self._python_node_is_reachable(call, seen=seen):
                    return True
        return False

    def _has_explicit_local_entrypoint(self) -> bool:
        for call in self._call_nodes:
            if self._enclosing_function(call) is not None:
                continue
            if any(
                self._external_call_targets_function(call, function)
                for function in self._function_nodes
            ):
                return True
        return False

    def _python_enclosing_string_equality_values(
        self,
        reference: str,
        index: int,
    ) -> tuple[str, ...]:
        sink = next(
            (
                node
                for node in self._call_nodes
                if self._node_span(node)[0] == index
            ),
            None,
        )
        if sink is None:
            return ()
        current: ast.AST | None = sink
        while current is not None:
            parent = self._parents.get(current)
            if isinstance(parent, ast.If) and any(
                current is statement or self._node_inside(current, statement)
                for statement in parent.body
            ):
                values = self._python_equality_condition_values(
                    parent.test,
                    reference,
                )
                if values:
                    return values
            current = parent
        return ()

    def _python_equality_condition_values(
        self,
        condition: ast.AST,
        reference: str,
    ) -> tuple[str, ...]:
        if not isinstance(condition, ast.Compare) or len(condition.ops) != 1:
            return ()
        if not isinstance(condition.ops[0], (ast.Eq, ast.Is)):
            return ()
        left, right = condition.left, condition.comparators[0]
        if ast.unparse(left) == reference and isinstance(right, ast.Constant):
            return (right.value,) if isinstance(right.value, str) else ()
        if ast.unparse(right) == reference and isinstance(left, ast.Constant):
            return (left.value,) if isinstance(left.value, str) else ()
        return ()

    def _python_fail_closed_string_guard_values(
        self,
        reference: str,
        index: int,
    ) -> tuple[str, ...]:
        containing = self._containing_function(index)
        statements = containing.body if containing is not None else self.tree.body
        guards = [
            statement
            for statement in statements
            if isinstance(statement, ast.If)
            and self._node_span(statement)[1] < index
            and _python_statements_terminate(statement.body)
            and (
                values := self._python_rejected_condition_allowed_values(
                    statement.test,
                    reference,
                    index,
                )
            )
        ]
        if not guards:
            return ()
        latest = max(guards, key=lambda item: self._node_span(item)[0])
        return self._python_rejected_condition_allowed_values(
            latest.test,
            reference,
            index,
        )

    def _python_rejected_condition_allowed_values(
        self,
        condition: ast.AST,
        reference: str,
        index: int,
    ) -> tuple[str, ...]:
        if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.And):
            values = tuple(
                value
                for term in condition.values
                for value in self._python_rejected_condition_allowed_values(
                    term,
                    reference,
                    index,
                )
            )
            return tuple(dict.fromkeys(values)) if len(values) == len(condition.values) else ()
        if not isinstance(condition, ast.Compare) or len(condition.ops) != 1:
            return ()
        left, right = condition.left, condition.comparators[0]
        if not _python_string_guard_target_matches(left, reference):
            return ()
        if isinstance(condition.ops[0], (ast.NotEq, ast.IsNot)):
            return (right.value,) if isinstance(right, ast.Constant) and isinstance(right.value, str) else ()
        if not isinstance(condition.ops[0], ast.NotIn):
            return ()
        container = right
        if isinstance(container, ast.Name):
            assignment = self._latest_assignment(container.id, index)
            if assignment is not None:
                container = assignment[0]
        elements = (
            tuple(container.keys)
            if isinstance(container, ast.Dict)
            else tuple(container.elts)
            if isinstance(container, (ast.List, ast.Tuple, ast.Set))
            else ()
        )
        values = tuple(
            element.value
            for element in elements
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        )
        return values if len(values) == len(elements) and values else ()

    def _python_prior_terminating_guard_truth(
        self,
        condition: ast.Compare,
        index: int,
    ) -> bool | None:
        """Use a prior terminating equality guard as a local path fact."""

        if len(condition.ops) != 1 or not isinstance(
            condition.ops[0],
            (ast.Eq, ast.NotEq),
        ):
            return None
        containing = self._containing_function(index)
        statements = containing.body if containing is not None else self.tree.body
        current_left = ast.unparse(condition.left)
        current_right = ast.unparse(condition.comparators[0])
        for statement in reversed(statements):
            if self._node_span(statement)[1] >= index:
                continue
            if not (
                isinstance(statement, ast.If)
                and _python_statements_terminate(statement.body)
                and isinstance(statement.test, ast.Compare)
                and len(statement.test.ops) == 1
                and isinstance(statement.test.ops[0], (ast.Eq, ast.NotEq))
                and ast.unparse(statement.test.left) == current_left
                and ast.unparse(statement.test.comparators[0]) == current_right
            ):
                continue
            # Reaching this point proves the earlier guard false.
            return type(statement.test.ops[0]) is not type(condition.ops[0])
        return None

    def _python_none_state(
        self,
        node: ast.AST,
        index: int,
        *,
        seen: frozenset[str] = frozenset(),
    ) -> bool | None:
        if isinstance(node, ast.Constant):
            return node.value is None
        if isinstance(
            node,
            (ast.List, ast.Tuple, ast.Set, ast.Dict, ast.JoinedStr, ast.BinOp),
        ):
            return False
        if not isinstance(node, ast.Name):
            return None
        key = f"none-state:{node.id}:{index}"
        if key in seen:
            return None
        assignment = self._latest_assignment(node.id, index)
        if assignment is None:
            return None
        return self._python_none_state(
            assignment[0],
            assignment[1],
            seen=seen | frozenset({key}),
        )
