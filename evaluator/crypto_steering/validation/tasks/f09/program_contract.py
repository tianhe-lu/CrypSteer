"""Bounded checks for definite API errors on a Python program's entry path.

This is separate from F9's existential callable-route policy. It checks known
call contracts without executing generated code or enforcing prompt wording.
"""

from __future__ import annotations

import ast
import re
import struct

from crypto_steering.validation.source.python import PythonAnalyzer
from crypto_steering.validation.tasks.f09.arguments import _python_random_argument_failure
from crypto_steering.validation.tasks.f09.extra_program_contract import known_api_failure, source_call_failure
from crypto_steering.validation.tasks.f09.object_entry_contract import object_entry_failure
from crypto_steering.validation.tasks.evidence import _python_random_call_target


def python_random_program_failure(source: str) -> str | None:
    """Return a definite entry-path call error, or no established failure.

    Uncalled helpers and unresolved/shadowed APIs are not rejected. Absence of
    a finding is not proof of runtime success or exact output correctness.
    """
    try:
        analyzer = PythonAnalyzer(source)
    except SyntaxError:
        return None  # The parser owns syntax failures.
    tree = analyzer.tree
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def implicit_str_callers(owner: ast.AST) -> list[ast.FormattedValue]:
        """Only a plain local instance's unformatted f-string invokes __str__."""
        if not isinstance(owner, ast.FunctionDef) or owner.name != "__str__" or owner.decorator_list:
            return []
        cls = parents.get(owner)
        if (not isinstance(cls, ast.ClassDef) or parents.get(cls) is not tree
                or cls.bases or cls.decorator_list or cls.keywords):
            return []
        if any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
               and item.name in {"__format__", "__new__", "__getattribute__", "__getattr__"}
               for item in cls.body):
            return []
        if sum(isinstance(item, ast.FunctionDef) and item.name == "__str__" for item in cls.body) != 1:
            return []
        if any(isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Store)
               and item.attr in {"__str__", "__format__"} for item in ast.walk(tree)):
            return []
        if any(isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
               and item.id in {cls.name, "__str__", "__format__"} for item in ast.walk(tree)):
            return []
        callers = []
        for formatted in (item for item in ast.walk(tree) if isinstance(item, ast.FormattedValue)):
            if (formatted.conversion not in {-1, ord("s")} or formatted.format_spec is not None
                    or not isinstance(formatted.value, ast.Name)):
                continue
            values = analyzer._reaching_assignments(formatted.value.id, analyzer._node_span(formatted)[0])
            if len(values) != 1:
                continue
            value, at = values[0]
            if (not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name)
                    or value.func.id != cls.name or analyzer._node_span(cls)[0] >= at):
                continue
            containing = analyzer._containing_function(at)
            if containing is not None and cls.name in analyzer._function_parameter_names(containing):
                continue
            if any(isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item is not cls
                   and item.name == cls.name for item in tree.body):
                continue
            callers.append(formatted)
        return callers

    def entry_path(call: ast.AST, seen: frozenset[int] = frozenset()) -> bool:
        """Require a caller chain whose literal-boolean branches are live."""
        if (not analyzer._python_direct_path_is_reachable(call)
                or not analyzer._python_literal_boolean_path_is_reachable(call)):
            return False
        owner = analyzer._enclosing_function(call)
        if owner is None:
            return True
        if id(owner) in seen:
            return False
        return any(
            entry_path(caller, seen | {id(owner)})
            for caller in analyzer._call_nodes
            if analyzer._external_call_targets_function(caller, owner)
        ) or any(entry_path(caller, seen | {id(owner)}) for caller in implicit_str_callers(owner))

    def scope(node: ast.AST) -> ast.AST:
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return node
        return tree

    def enclosing_class(node: ast.AST) -> ast.ClassDef | None:
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.ClassDef):
                return node
        return None

    def parameter_values(owner: ast.AST, parameter: str) -> list[tuple[ast.AST, ast.AST]]:
        """Bind a plain module-level helper only from its reachable direct calls.

        Unknown expansions, rebinding, methods, and callbacks remain unresolved.
        All observed callers must bind the parameter for a type claim below.
        """
        if not isinstance(owner, ast.FunctionDef) or parents.get(owner) is not tree:
            return []
        if any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == owner.name
               for node in ast.walk(tree)):
            return []
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               and node is not owner and node.name == owner.name for node in ast.walk(tree)):
            return []
        positional = [arg.arg for arg in (*owner.args.posonlyargs, *owner.args.args)]
        defaults = dict(zip(positional[-len(owner.args.defaults):], owner.args.defaults)) if owner.args.defaults else {}
        defaults.update({arg.arg: default for arg, default in zip(owner.args.kwonlyargs, owner.args.kw_defaults)
                         if default is not None})
        values: list[tuple[ast.AST, ast.AST]] = []
        for caller in analyzer._call_nodes:
            if not isinstance(caller.func, ast.Name) or caller.func.id != owner.name:
                continue
            caller_scope = scope(caller)
            if isinstance(caller_scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if any(arg.arg == owner.name for arg in ast.walk(caller_scope.args) if isinstance(arg, ast.arg)):
                    continue
            if not analyzer._python_literal_boolean_path_is_reachable(caller) or not analyzer._python_call_has_entry_path(caller, seen=frozenset()):
                continue
            if any(isinstance(arg, ast.Starred) for arg in caller.args) or any(k.arg is None for k in caller.keywords):
                return []
            bound = dict(zip(positional, caller.args))
            bound.update({k.arg: k.value for k in caller.keywords})
            if parameter in bound:
                values.append((bound[parameter], caller))
            elif parameter in defaults:
                values.append((defaults[parameter], owner))
            else:
                return []
        return values

    def attribute_text(node: ast.AST) -> str:
        text = ast.unparse(node)
        head, separator, tail = text.partition(".")
        canonical = analyzer.import_aliases.get(head, head) + separator + tail
        return canonical.removeprefix("os.") if canonical.startswith("os.sys.") else canonical

    def name(node: ast.AST, at: ast.AST, seen: frozenset[tuple[str, int]] = frozenset()) -> str:
        key = (ast.unparse(node), at.lineno)
        if key in seen:
            return ""
        if isinstance(node, ast.Constant):
            return {bytes: "bytes()", str: "str()", int: "int()", type(None): "none()"}.get(type(node.value), "")
        if isinstance(node, (ast.List, ast.ListComp)):
            return "list()"
        if isinstance(node, ast.Tuple):
            return "tuple()"
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            base = name(node.value, at, seen | {key})
            return base if type_from_identity(base) in {"bytes", "bytearray", "str"} else ""
        if isinstance(node, ast.Attribute):
            # An explicit module/object patch invalidates the native identity.
            if not (isinstance(node.value, ast.Name) and node.value.id == "self") and any(isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Store)
                   and attribute_text(item) == attribute_text(node)
                   and scope(item) in (scope(at), tree) for item in ast.walk(tree)):
                return ""
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                assignments = [item for item in ast.walk(tree)
                               if isinstance(item, ast.Assign) and enclosing_class(item) is enclosing_class(at)
                               and any(ast.unparse(t) == ast.unparse(node) for t in item.targets)]
                if len(assignments) == 1:
                    return name(assignments[0].value, assignments[0], seen | {key})
            prefix = name(node.value, at, seen)
            identity = f"{prefix}.{node.attr}" if prefix else ""
            return "sys" if identity == "os.sys" else identity
        if isinstance(node, ast.Call):
            function = name(node.func, at, seen)
            return function + "()" if function else ""
        if not isinstance(node, ast.Name):
            return ""
        owner = scope(at)
        if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if any(isinstance(arg, ast.arg) and arg.arg == node.id for arg in ast.walk(owner.args)):
                local_writes = [item for item in ast.walk(owner) if isinstance(item, ast.Name)
                                and isinstance(item.ctx, ast.Store) and item.id == node.id
                                and not (isinstance(parents.get(item), ast.Assign)
                                         and isinstance(parents[item].value, ast.Name)
                                         and parents[item].value.id == node.id)]
                if local_writes:
                    return ""  # Reassigned parameters need flow-sensitive analysis.
                values = parameter_values(owner, node.id)
                resolved = {name(value, caller, seen | {key}) for value, caller in values}
                return next(iter(resolved)) if len(resolved) == 1 else ""
        assignments = [item for item in ast.walk(tree)
                       if isinstance(item, (ast.Assign, ast.AnnAssign))
                       and scope(item) in (owner, tree) and item.lineno < at.lineno
                       and any(isinstance(t, ast.Name) and t.id == node.id
                               for t in (item.targets if isinstance(item, ast.Assign) else (item.target,)))]
        if assignments and all(parents.get(item) in (owner, tree) for item in assignments):
            latest = max(assignments, key=lambda item: item.lineno)
            if latest.value is not None:
                return name(latest.value, latest, seen | {key})
        # Conservatively decline a built-in/import identity if the same name
        # is rebound in this scope or at module level.
        for item in ast.walk(tree):
            if scope(item) not in (owner, tree):
                continue
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store) and item.id == node.id:
                return ""
            if isinstance(item, ast.arg) and item.arg == node.id:
                return ""
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and item.name == node.id:
                return ""
        return analyzer.import_aliases.get(node.id, node.id if node.id in {"bytes", "bytearray", "str", "int", "hex", "range", "input"} else "").removeprefix("builtins.")

    def type_from_identity(identity: str) -> str:
        if identity in {f"{kind}()" for kind in ("bytes", "bytearray", "str", "int", "none", "list", "tuple")}:
            return identity[:-2]
        if identity in {"os.urandom()", "os.getrandom()", "secrets.token_bytes()", "random.randbytes()",
                        "hashlib.pbkdf2_hmac()", "struct.pack()", "binascii.hexlify()", "os.fsencode()"}:
            return "bytes"
        if identity in {"input()", "getpass.getpass()", "bytes().hex()", "bytearray().hex()"}:
            return "str"
        if identity.endswith(".encode()") and type_from_identity(identity[:-9]) == "str":
            return "bytes"
        if identity.endswith(".hex()") and type_from_identity(identity[:-6]) in {"bytes", "bytearray"}:
            return "str"
        if identity.endswith(".decode()") and type_from_identity(identity[:-9]) in {"bytes", "bytearray"}:
            return "str"
        if identity.startswith("hashlib.") and identity.endswith("().digest()"):
            return "bytes"
        if identity == "cryptography.hazmat.primitives.hashes.Hash().finalize()":
            return "bytes"
        return ""

    def value_type(node: ast.AST, at: ast.AST) -> str:
        return type_from_identity(name(node, at))

    dynamic_namespace = any(
        isinstance(item, ast.ImportFrom) and any(alias.name == "*" for alias in item.names)
        or isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
        and item.func.id in {"exec", "eval", "globals", "locals"}
        for item in ast.walk(tree)
    )

    def module_bound(identifier: str, before: int | None = None) -> bool:
        if identifier in analyzer.import_aliases:
            return True
        return any(
            scope(item) is tree and (before is None or getattr(item, "lineno", before) < before)
            and (isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store) and item.id == identifier
                 or isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and item.name == identifier)
            for item in ast.walk(tree)
        )

    def missing_binding(call: ast.Call, identity: str) -> str | None:
        # Two source-proven forms from the audit; this is not a general Python
        # undefined-name pass. Dynamic namespace creation stays unresolved.
        if dynamic_namespace:
            return None
        parent = parents.get(call)
        local_owner = scope(call)
        if (isinstance(call.func, ast.Name) and isinstance(local_owner, ast.FunctionDef)
                and isinstance(parent, ast.Assign) and parents.get(parent) is local_owner
                and any(isinstance(target, ast.Name) and target.id == call.func.id for target in parent.targets)
                and any(isinstance(item, ast.ClassDef) and item.name == call.func.id for item in tree.body)):
            identifier = call.func.id
            prior_binding = any(
                (isinstance(item, ast.arg) and item.arg == identifier)
                or (isinstance(item, (ast.Global, ast.Nonlocal)) and identifier in item.names)
                or (getattr(item, "lineno", call.lineno) < call.lineno
                    and ((isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store) and item.id == identifier)
                         or (isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name == identifier)
                         or (isinstance(item, (ast.Import, ast.ImportFrom)) and any(
                             (alias.asname or alias.name.split(".")[0]) == identifier for alias in item.names))))
                for item in ast.walk(local_owner)
            )
            if not prior_binding:
                return f"{identifier} is read before its first local assignment; the assignment shadows the module class"
        if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "hashlib" and not module_bound("hashlib")
                and scope(call) is tree):
            return "hashlib is used without an import or module binding"
        if (identity in {"bytes", "bytearray"} and call.args and isinstance(call.args[0], ast.Name)
                and scope(call) is tree and isinstance(parents.get(call), ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == call.args[0].id for target in parents[call].targets)
                and not module_bound(call.args[0].id, call.lineno)):
            return f"{call.args[0].id} has no prior module binding for the byte conversion"
        return None

    def length(node: ast.AST, at: ast.AST, seen: frozenset[str] = frozenset()) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, str)):
            return len(node.value)
        if isinstance(node, (ast.List, ast.Tuple)) and not any(isinstance(x, ast.Starred) for x in node.elts):
            return len(node.elts)
        if isinstance(node, ast.Call) and not node.args and not node.keywords and name(node.func, at) in {"bytes", "bytearray"}:
            return 0
        if isinstance(node, ast.Name) and node.id not in seen:
            assignments = [item for item in ast.walk(tree)
                           if isinstance(item, (ast.Assign, ast.AnnAssign))
                           and scope(item) is scope(at) and item.lineno < at.lineno
                           and any(isinstance(t, ast.Name) and t.id == node.id
                                   for t in (item.targets if isinstance(item, ast.Assign) else (item.target,)))]
            if len(assignments) == 1 and assignments[0].value is not None:
                if any(isinstance(item, (ast.Assign, ast.AnnAssign)) and isinstance(item.value, ast.Name)
                       and item.value.id == node.id and scope(item) is scope(at)
                       and assignments[0].lineno < item.lineno < at.lineno for item in ast.walk(tree)):
                    return None  # Another name may mutate the same buffer.
                mutations = [item for item in ast.walk(tree)
                             if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                             and isinstance(item.func.value, ast.Name) and item.func.value.id == node.id
                             and item.func.attr in {"append", "extend", "insert", "pop", "remove", "clear"}
                             and scope(item) is scope(at)
                             and assignments[0].lineno < item.lineno < at.lineno]
                added = 0
                for mutation in mutations:
                    if mutation.func.attr != "append" or len(mutation.args) != 1 or mutation.keywords:
                        return None
                    statement = parents.get(mutation)
                    if not isinstance(statement, ast.Expr):
                        return None
                    block = parents.get(statement)
                    if block is parents.get(assignments[0]):
                        added += 1
                        continue
                    if not (isinstance(block, ast.For) and parents.get(block) is parents.get(assignments[0])
                            and statement in block.body and not block.orelse
                            and not any(isinstance(child, (ast.If, ast.Try, ast.For, ast.While, ast.Break, ast.Continue, ast.Return))
                                        for stmt in block.body for child in ast.walk(stmt))):
                        return None
                    iteration = block.iter
                    if not (isinstance(iteration, ast.Call) and name(iteration.func, iteration) == "range"
                            and not iteration.keywords and 1 <= len(iteration.args) <= 3
                            and all(isinstance(arg, ast.Constant) and type(arg.value) is int for arg in iteration.args)):
                        return None
                    try:
                        added += len(range(*(arg.value for arg in iteration.args)))
                    except (ValueError, OverflowError):
                        return None
                if any(assignments[0].lineno < item.lineno < at.lineno
                       and ((isinstance(item, ast.AugAssign) and isinstance(item.target, ast.Name)
                             and item.target.id == node.id)
                            or (isinstance(item, ast.Subscript) and isinstance(item.value, ast.Name)
                                and item.value.id == node.id
                                and (isinstance(item.ctx, ast.Del)
                                     or (isinstance(item.slice, ast.Slice) and isinstance(item.ctx, ast.Store)))))
                       for item in ast.walk(tree) if hasattr(item, "lineno")):
                    return None
                initial = length(assignments[0].value, assignments[0], seen | {node.id})
                return initial + added if initial is not None else None
        if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
            identity = name(node.func, at)
            if identity in {"bytes", "bytearray", "os.urandom", "secrets.token_bytes", "random.randbytes"}:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and type(arg.value) is int and arg.value >= 0:
                    return arg.value
                if identity in {"bytes", "bytearray"}:
                    return length(arg, at, seen)
        return None

    # The observed annotation mistakes occur in module-level function
    # signatures, evaluated when the def statement executes. Local variable
    # annotations and postponed/string annotations do not execute this lookup.
    postponed = any(isinstance(node, ast.ImportFrom) and node.module == "__future__"
                    and any(alias.name == "annotations" for alias in node.names) for node in tree.body)
    if not postponed and not dynamic_namespace:
        for function in (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
            annotations = [arg.annotation for arg in ast.walk(function.args) if isinstance(arg, ast.arg) and arg.annotation is not None]
            if function.returns is not None:
                annotations.append(function.returns)
            for annotation in annotations:
                if isinstance(annotation, ast.Constant):
                    continue
                for node in ast.walk(annotation):
                    if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
                        continue
                    identifier = node.value.id
                    if identifier == "random" and node.attr == "Random" and not module_bound("random"):
                        return f"line {node.lineno}: eager function annotation uses random.Random without binding random"
                    canonical = name(node, function)
                    imported_before = any(
                        isinstance(item, (ast.Import, ast.ImportFrom)) and item.lineno < function.lineno
                        and any((alias.asname or alias.name.split(".")[0]) == identifier for alias in item.names)
                        for item in tree.body
                    )
                    if imported_before and canonical in {"random.Generator", "random._Random"}:
                        return f"line {node.lineno}: eager function annotation refers to nonexistent {canonical}"

    for assignment in (node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))):
        value = assignment.value
        if isinstance(value, ast.Subscript) and name(value.value, assignment) == "typing.TypeAlias":
            return f"line {value.lineno}: typing.TypeAlias is an annotation marker and cannot be subscripted"

    object_failure = object_entry_failure(analyzer, entry_path)
    if object_failure:
        return object_failure

    for call in sorted(analyzer._call_nodes, key=lambda n: (n.lineno, n.col_offset)):
        identity = name(call.func, call)
        if not identity:
            target = _python_random_call_target(analyzer, ast.unparse(call.func), analyzer._node_span(call)[0])
            if target is not None:
                receiver, _, method = target.rpartition(".")
                identity = (f"{receiver}().{method}" if receiver in {
                    "random.Random", "random.SystemRandom", "secrets.SystemRandom",
                } else target)
        reason = None
        if identity == "bytearray" and any(k.arg == "length" for k in call.keywords):
            reason = "bytearray does not accept a length keyword; use a positional size"
        elif identity in {"random.getrandint", "random.randbelow", "secrets.randbytes", "os.random.seed"}:
            # Do not reject an explicitly provided extension with this name.
            if not any(isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                       and ast.unparse(n) == ast.unparse(call.func) for n in ast.walk(tree)):
                reason = f"{identity} is not an available standard-library API"
        elif identity.startswith(("random.", "secrets.", "os.", "uuid.")):
            reason = _python_random_argument_failure(
                ast.unparse(call), callable_names={ast.unparse(call.func): identity},
            )
        elif identity == "struct.pack" and call.args and isinstance(call.args[0], ast.Constant):
            fmt = call.args[0].value
            if isinstance(fmt, str):
                reason = _pack_failure(fmt, call.args[1:], lambda n: length(n, call))
        elif identity == "cryptography.hazmat.primitives.ciphers.algorithms.AES":
            reason = _signature_failure(call, ("key",), required=1, api="AES")
        elif identity == "cryptography.hazmat.primitives.ciphers.Cipher":
            reason = _signature_failure(call, ("algorithm", "mode", "backend"), required=2, api="Cipher")
        elif identity == "cryptography.hazmat.primitives.kdf.hkdf.HKDF":
            reason = _signature_failure(call, ("algorithm", "length", "salt", "info", "backend"), required=4, api="HKDF")
        elif identity == "cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC":
            reason = _signature_failure(call, ("algorithm", "length", "salt", "iterations", "backend"), required=4, api="PBKDF2HMAC")
        elif identity == "cryptography.hazmat.primitives.hashes.Hash":
            reason = _signature_failure(call, ("algorithm", "backend"), required=1, api="Hash")
        elif identity in {"hmac.new", "hmac.HMAC"}:
            reason = _signature_failure(call, ("key", "msg", "digestmod"), required=1, api="HMAC")
            digest = next((k.value for k in call.keywords if k.arg == "digestmod"),
                          call.args[2] if len(call.args) > 2 else None)
            if digest is not None and name(digest, call).startswith("cryptography.hazmat.primitives.hashes."):
                reason = "stdlib hmac.new requires a hashlib-compatible digestmod, not a PyCA hash algorithm"
        elif (identity.endswith(".finalize")
              and identity.startswith(("cryptography.hazmat.primitives.ciphers.Cipher().",
                                       "cryptography.hazmat.primitives.hashes.Hash()."))):
            reason = _signature_failure(call, (), required=0, api="finalize")
        elif identity == "cryptography.hazmat.primitives.padding.PKCS7().padder().update().finalize":
            reason = "padder.update returns bytes, which have no finalize method"
        elif (isinstance(call.func, ast.Attribute) and call.func.attr == "pad"
              and isinstance(call.func.value, ast.Call)
              and name(call.func.value.func, call) == "cryptography.hazmat.primitives.padding.PKCS7"):
            reason = "PKCS7 exposes padder(), not pad()"
        reason = reason or known_api_failure(call, identity, lambda node: name(node, call), lambda node: value_type(node, call))
        reason = reason or source_call_failure(call, analyzer, identity)
        reason = reason or missing_binding(call, identity)
        if reason and entry_path(call):
            return f"line {call.lineno}: {reason}"
    # Consecutive finalization in a single straight-line block is a definite
    # AlreadyFinalized error. Branches/reassignments remain unresolved here.
    for owner in ast.walk(tree):
        statements = getattr(owner, "body", None)
        if not isinstance(statements, list):
            continue
        finalized: set[str] = set()
        for statement in statements:
            if isinstance(statement, (ast.If, ast.For, ast.While, ast.Try, ast.With,
                                      ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                finalized.clear()
                continue
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
                finalized.difference_update(ast.unparse(target) for target in targets)
            for call in (n for n in ast.walk(statement) if isinstance(n, ast.Call)):
                if not isinstance(call.func, ast.Attribute) or call.func.attr != "finalize":
                    continue
                if isinstance(call.func.value, ast.Call):
                    continue  # A fresh inline context is a different object.
                identity = name(call.func, call)
                if not identity.startswith(("cryptography.hazmat.primitives.hashes.Hash().",
                                            "cryptography.hazmat.primitives.ciphers.Cipher().",
                                            "cryptography.hazmat.primitives.padding.PKCS7().")):
                    continue
                receiver = ast.unparse(call.func.value)
                if receiver in finalized and entry_path(call):
                    return f"line {call.lineno}: {receiver}.finalize() is called twice without a new context"
                finalized.add(receiver)
    return None


def _signature_failure(call: ast.Call, parameters: tuple[str, ...], *, required: int, api: str) -> str | None:
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(k.arg is None for k in call.keywords):
        return None
    keywords = {k.arg for k in call.keywords}
    if len(call.args) > len(parameters) or keywords - set(parameters):
        return f"{api} received unsupported arguments"
    if keywords & set(parameters[:len(call.args)]):
        return f"{api} supplies an argument more than once"
    if set(parameters[:required]) - (set(parameters[:len(call.args)]) | keywords):
        return f"{api} is missing a required argument"
    return None


def _pack_failure(fmt: str, args: list[ast.AST], length) -> str | None:
    try:
        struct.calcsize(fmt)
    except (struct.error, OverflowError):
        return "struct.pack received an invalid format"
    body = fmt.lstrip("@=<>!")
    fields = re.findall(r"(\d*)([xcbB?hHiIlLqQnNefdspP])", body)
    if re.sub(r"\s+", "", body) != "".join(n + code for n, code in fields):
        return None
    expected = sum(0 if code == "x" else 1 if code in "sp" else int(n or 1) for n, code in fields)
    actual = 0
    for arg in args:
        if isinstance(arg, ast.Starred):
            count = length(arg.value)
            if count is None:
                return None
            actual += count
        else:
            actual += 1
    if expected != actual:
        return f"struct.pack format requires {expected} values but receives {actual}"
    return None
