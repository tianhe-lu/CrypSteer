"""Small source-proven API contracts used by the F9 entry-path checker.

The caller owns reachability and name resolution. These checks do not execute
code, infer unresolved values, or enforce a particular prompt's output format.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crypto_steering.validation.source.python import PythonAnalyzer


_HASHES = "cryptography.hazmat.primitives.hashes."
_HKDF = "cryptography.hazmat.primitives.kdf.hkdf.HKDF"
_PBKDF2 = "cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC"
_RNG_INSTANCES = (
    "random.Random()", "random.SystemRandom()", "secrets.SystemRandom()",
)


def known_api_failure(
    call: ast.Call,
    identity: str,
    resolved_name: Callable[[ast.AST], str],
    value_type: Callable[[ast.AST], str] | None = None,
) -> str | None:
    """Return an error only for a resolved API with a definite bad contract.

    ``identity`` names the called API, with constructors represented as ``()``.
    ``resolved_name`` resolves an argument in the call's source context. Empty
    or unfamiliar identities are deliberately left to the existing validator.
    """
    def kind(node: ast.AST) -> str:
        inferred = value_type(node) if value_type is not None else ""
        return inferred or ("bytes" if _definitely_bytes(node, resolved_name) else "")
    if identity in {f"{receiver}.bytes" for receiver in _RNG_INSTANCES}:
        return "Python Random/SystemRandom provides randbytes(n), not bytes(n)"
    if identity in {f"{receiver}.randbits" for receiver in _RNG_INSTANCES}:
        return "Python Random/SystemRandom provides getrandbits(k), not randbits(k)"
    if identity in {f"{receiver}._randbytes" for receiver in _RNG_INSTANCES}:
        return "Python Random/SystemRandom provides randbytes(n), not _randbytes(n)"
    if identity in {f"{receiver}.generate_reference" for receiver in _RNG_INSTANCES}:
        return "Python Random/SystemRandom has no generate_reference method"
    if identity == _HASHES + "HashAlgorithm":
        return "HashAlgorithm is an abstract algorithm interface, not a hashing context"
    if identity == "cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key().encrypt":
        return "RSA encryption requires the public key; the generated private key has no encrypt method"
    if identity == "cryptography.hazmat.primitives.asymmetric.padding.MGF1" and any(
        keyword.arg == "backend" for keyword in call.keywords
    ):
        return "MGF1 accepts a hash algorithm, not a backend keyword"
    if identity == "cryptography.hazmat.backends.default_backend().load_private_key":
        return "default_backend() has no load_private_key method; use a serialization loader"
    if identity in {"os.hexlify", "hmac.pbkdf2_hmac", "os.urandom.getrandbits",
                    "cryptography.hazmat.primitives.serialization.utils.encode_pem",
                    "cryptography.hazmat.primitives.serialization.Encoding.PEM.encode"}:
        return f"{identity} is not an available API"
    if identity == "sys.stdin.buffer.write":
        return "the default standard-input buffer is not writable"
    if isinstance(call.func, ast.Attribute):
        receiver = kind(call.func.value)
        if receiver in {"bytes", "bytearray"} and call.func.attr in {"encode", "tobytes", "to_bytes"}:
            return f"{receiver} has no {call.func.attr} method"
    if identity == _HASHES + "Hash().digest":
        return "PyCA Hash exposes finalize(), not digest()"
    if identity == _HASHES + "Hash().update().finalize":
        return "PyCA Hash.update returns None, which has no finalize method"

    # A *args/**kwargs expansion may bind algorithm; do not guess its type.
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(keyword.arg is None for keyword in call.keywords):
        return None

    def argument(index: int, key: str) -> ast.AST | None:
        return call.args[index] if len(call.args) > index else next(
            (keyword.value for keyword in call.keywords if keyword.arg == key), None,
        )

    rsa_private_key = "cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key()"
    rsa_export = identity == rsa_private_key + ".private_bytes"
    # The audited source serializes its generated RSA key, loads those same
    # bytes, then immediately re-exports it. Unknown PEM contents stay unknown.
    if (identity == "cryptography.hazmat.primitives.serialization.load_pem_private_key().private_bytes"
            and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Call)):
        loader = call.func.value
        data = loader.args[0] if loader.args else next(
            (keyword.value for keyword in loader.keywords if keyword.arg == "data"), None,
        )
        rsa_export = data is not None and resolved_name(data) == rsa_private_key + ".private_bytes()"
    if rsa_export:
        encoding, format_value = argument(0, "encoding"), argument(1, "format")
        if (encoding is not None and resolved_name(encoding) == "cryptography.hazmat.primitives.serialization.Encoding.Raw"
                or format_value is not None and resolved_name(format_value) == "cryptography.hazmat.primitives.serialization.PrivateFormat.Raw"):
            return "RSA private-key serialization does not support Raw encoding or Raw format"

    if identity == "cryptography.hazmat.primitives.asymmetric.ec.generate_private_key().private_bytes":
        encoding, format_value = argument(0, "encoding"), argument(1, "format")
        if (encoding is not None and format_value is not None
                and resolved_name(encoding) == "cryptography.hazmat.primitives.serialization.Encoding.Raw"
                and resolved_name(format_value) == "cryptography.hazmat.primitives.serialization.PrivateFormat.PKCS8"):
            return "EC private-key PKCS8 serialization requires DER or PEM encoding, not Raw"

    if identity == "cryptography.hazmat.primitives.asymmetric.ec.generate_private_key().public_key().public_bytes":
        encoding, format_value = argument(0, "encoding"), argument(1, "format")
        if (encoding is not None and format_value is not None
                and resolved_name(encoding) == "cryptography.hazmat.primitives.serialization.Encoding.X962"
                and resolved_name(format_value) == "cryptography.hazmat.primitives.serialization.PublicFormat.SubjectPublicKeyInfo"):
            return "EC public-key SubjectPublicKeyInfo serialization requires DER or PEM encoding, not X962"

    if identity in {_HKDF, _PBKDF2, _HASHES + "Hash"}:
        algorithm = argument(0, "algorithm")
        if algorithm is not None:
            if kind(algorithm) in {"bytes", "bytearray", "str", "int", "none"}:
                return f"{identity.rsplit('.', 1)[-1]} algorithm requires a hash-algorithm object, not bytes or scalar data"
            algorithm_identity = resolved_name(algorithm)
            if algorithm_identity == _HASHES + "Hash()" or (algorithm_identity.startswith("hashlib.") and algorithm_identity.endswith("()")):
                return "algorithm requires a hash-algorithm object, not a hashing context"
    if identity == _PBKDF2:
        salt = argument(2, "salt")
        if salt is not None and kind(salt) in {"bytearray", "str", "int", "none", "list", "tuple"}:
            return "PBKDF2HMAC salt must be bytes"
    if identity == "os.fsencode":
        path = argument(0, "filename")
        if path is not None and kind(path) in {"bytearray", "int", "none", "list", "tuple"}:
            return "os.fsencode requires a string, bytes, or a path-like object"
    if identity == "os.getrandom":
        size = argument(0, "size")
        if size is not None and kind(size) in {"bytes", "bytearray", "str", "none", "list", "tuple"}:
            return "os.getrandom size must be an integer"
    if identity in {
        "random.randbytes", *{f"{receiver}.randbytes" for receiver in _RNG_INSTANCES},
    }:
        size = argument(0, "n")
        if size is not None and kind(size) == "bytearray":
            return "randbytes length must be an integer, not a bytearray buffer"
    if identity == "hex" and call.args and kind(call.args[0]) in {"bytes", "bytearray", "str", "none", "list", "tuple"}:
        return "hex requires an integer, not encoded text or a byte buffer"
    if identity in {"hmac.new", "hmac.HMAC"}:
        key = argument(0, "key")
        digest = argument(2, "digestmod")
        if key is not None and (kind(key) in {"str", "int", "none"} or
                                resolved_name(key).startswith("hashlib.") and not resolved_name(key).endswith("()")):
            return "HMAC key must be bytes or bytearray, not a hash constructor or scalar"
        if digest is not None:
            digest_identity = resolved_name(digest)
            if digest_identity.startswith("hashlib.") and digest_identity.endswith("()"):
                return "HMAC digestmod requires a hash constructor, not a hash instance"
    if identity in {"bytes", "bytearray"} and call.args:
        items = call.args[0]
        if isinstance(items, (ast.List, ast.Tuple)):
            if any(kind(item) in {"bytes", "bytearray", "str", "none", "list", "tuple"} for item in items.elts):
                return f"{identity} iterable elements must be integers"
        if isinstance(items, (ast.ListComp, ast.GeneratorExp)) and _nonempty_comprehension(items, resolved_name):
            if kind(items.elt) in {"bytes", "bytearray", "str", "none", "list", "tuple"}:
                return f"{identity} iterable elements must be integers"
    return None


def source_call_failure(call: ast.Call, analyzer: PythonAnalyzer, identity: str = "") -> str | None:
    """Observed local-object defects; caller enforces entry reachability."""
    index = analyzer._node_span(call)[0]
    if (isinstance(call.func, ast.Attribute) and call.func.attr == "hex"
            and _integer_expression(call.func.value, index, analyzer, frozenset())):
        return "int has no hex method; convert the integer to bytes or use hex(value)"
    if (isinstance(call.func, ast.Attribute) and call.func.attr == "getrandbits"
            and _integer_expression(call.func.value, index, analyzer, frozenset())):
        return "int has no getrandbits method; call getrandbits on the generator"
    if (isinstance(call.func, ast.Attribute) and call.func.attr == "to_bytes"
            and _float_expression(call.func.value, index, analyzer, frozenset())):
        return "float has no to_bytes method"
    if identity in {"bytes", "bytearray"} and call.args and _float_expression(call.args[0], index, analyzer, frozenset()):
        return f"{identity} does not accept a float value"
    if (identity in {"bytes", "bytearray"} and call.args
            and (len(call.args) > 1 and not isinstance(call.args[1], ast.Starred)
                 or any(keyword.arg == "encoding" for keyword in call.keywords))
            and _integer_expression(call.args[0], index, analyzer, frozenset())):
        return f"{identity} with an encoding requires a string, not an integer"
    if identity == "cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key":
        size = call.args[1] if len(call.args) > 1 else next(
            (keyword.value for keyword in call.keywords if keyword.arg == "key_size"), None,
        )
        if size is not None and not any(isinstance(arg, ast.Starred) for arg in call.args):
            resolved = analyzer.resolve_ints(ast.unparse(size), index)
            if resolved.values and set(resolved.values) == {"32"}:
                return "RSA key_size=32 bits is too small for native key generation"

    def local_class(node: ast.AST, at: int) -> ast.ClassDef | None:
        if not isinstance(node, ast.Name) or analyzer._reaching_assignments(node.id, at):
            return None
        containing = analyzer._containing_function(at)
        if containing is not None and node.id in analyzer._function_parameter_names(containing):
            return None
        if any(item.name == node.id and analyzer._enclosing_function(item) in {None, containing}
               for item in analyzer._function_nodes):
            return None
        classes = [item for item in analyzer.tree.body
                   if isinstance(item, ast.ClassDef) and item.name == node.id
                   and analyzer._node_span(item)[0] < at]
        return classes[0] if len(classes) == 1 else None

    def members(owner: ast.ClassDef) -> set[str]:
        result = {item.name for item in owner.body
                  if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for item in owner.body:
            targets = item.targets if isinstance(item, ast.Assign) else (item.target,) if isinstance(item, ast.AnnAssign) and item.value is not None else ()
            result.update(target.id for target in targets if isinstance(target, ast.Name))
        return result

    def class_is_patched(owner: ast.ClassDef) -> bool:
        return any(
            isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Store)
            and isinstance(item.value, ast.Name) and item.value.id == owner.name
            for item in ast.walk(analyzer.tree)
        )

    def plain_instance_class(node: ast.AST) -> ast.ClassDef | None:
        at = index
        if isinstance(node, (ast.Name, ast.Attribute)):
            values = analyzer._reaching_assignments(ast.unparse(node), at)
            if len(values) != 1:
                return None
            # A class-body name is not a lexical binding for its methods or
            # another class. Attribute lookups (self.field/Class.field) keep
            # their existing resolution; only bare names need this guard.
            value = values[0][0]
            if (isinstance(node, ast.Name)
                    and analyzer._enclosing_function(value) is None
                    and (scope := analyzer._enclosing_class(value)) is not None
                    and (analyzer._enclosing_function(node) is not None
                         or analyzer._enclosing_class(node) is not scope)):
                return None
            node, at = values[0]
        if not isinstance(node, ast.Call):
            return None
        owner = local_class(node.func, at)
        if (owner is None or owner.bases or owner.decorator_list or owner.keywords
                or class_is_patched(owner)
                or members(owner) & {"__getattr__", "__getattribute__", "__new__"}):
            return None
        return owner

    if identity == "hex" and call.args:
        owner = plain_instance_class(call.args[0])
        if owner is not None and "__index__" not in members(owner):
            return f"hex requires __index__, which the {owner.name} instance does not provide"

    # A plain source-defined object must actually provide its called method.
    # Special methods may come from object itself; uncertain inheritance,
    # dynamic lookup and patches remain unresolved. Only randbytes signatures
    # are checked against their source definition below.
    if isinstance(call.func, ast.Attribute) and not call.func.attr.startswith("__"):
        method_name = call.func.attr
        owner = plain_instance_class(call.func.value)
        # One audited wrapper receives a plain object in __init__, stores it,
        # then calls an absent randbytes method. Resolve only this direct
        # constructor-argument form, with every actual constructor agreeing.
        receiver = call.func.value
        if method_name == "randbytes" and owner is None and isinstance(receiver, ast.Attribute) and isinstance(receiver.value, ast.Name) and receiver.value.id == "self":
            stored = analyzer._python_instance_attribute_assignment(ast.unparse(receiver), index)
            if stored is not None and isinstance(stored[0], ast.Name):
                initializer = analyzer._enclosing_function(stored[0])
                holder = analyzer._enclosing_class(stored[0])
                if (initializer is not None and initializer.name == "__init__"
                        and holder is not None and not holder.bases and not holder.decorator_list
                        and not initializer.decorator_list and not class_is_patched(holder)
                        and not members(holder) & {"__getattr__", "__getattribute__", "__new__"}):
                    parameter = stored[0].id
                    parameters = analyzer._function_parameter_names(initializer)
                    writes = [node for node in ast.walk(holder) if isinstance(node, ast.Attribute)
                              and isinstance(node.ctx, ast.Store) and ast.unparse(node) == ast.unparse(receiver)]
                    constructions = [node for node in analyzer._call_nodes
                                     if analyzer._external_call_targets_function(node, initializer)]
                    external_write = any(isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                                         and node.attr == receiver.attr and analyzer._enclosing_class(node) is None
                                         for node in ast.walk(analyzer.tree))
                    rebound_parameter = any(analyzer._enclosing_function(node) is initializer
                                            for node, _ in analyzer._assignments.get(parameter, ()))
                    if parameter in parameters and len(writes) == 1 and constructions and not external_write and not rebound_parameter:
                        classes = []
                        for construction in constructions:
                            argument = analyzer._call_argument_for_parameter(
                                construction, initializer, parameters.index(parameter), parameter,
                            )
                            classes.append(plain_instance_class(argument) if isinstance(argument, ast.Call) else None)
                        if classes[0] is not None and all(candidate is classes[0] for candidate in classes):
                            owner = classes[0]
        methods = [item for item in owner.body
                   if isinstance(item, ast.FunctionDef) and item.name == method_name] if owner else []
        patched = any(isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Store)
                      and item.attr == method_name for item in ast.walk(analyzer.tree))
        patched = patched or any(
            isinstance(item, ast.Call)
            and (isinstance(item.func, ast.Name) and item.func.id == "setattr"
                 or isinstance(item.func, ast.Attribute) and item.func.attr == "__setattr__")
            and (len(item.args) < 2 or not isinstance(item.args[1], ast.Constant) or item.args[1].value == method_name)
            for item in ast.walk(analyzer.tree)
        )
        patched = patched or any(
            isinstance(item, ast.Subscript) and isinstance(item.ctx, ast.Store)
            and isinstance(item.value, ast.Attribute) and item.value.attr == "__dict__"
            for item in ast.walk(analyzer.tree)
        )
        patched = patched or any(
            isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == "update"
            and (isinstance(item.func.value, ast.Attribute) and item.func.value.attr == "__dict__"
                 or isinstance(item.func.value, ast.Call) and isinstance(item.func.value.func, ast.Name)
                 and item.func.value.func.id == "vars")
            for item in ast.walk(analyzer.tree)
        )
        if owner is not None:
            patched = patched or any(
                isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
                and item.id == method_name and analyzer._enclosing_function(item) is None
                for statement in owner.body for item in ast.walk(statement)
            )
        if owner is not None and method_name not in members(owner) and not patched:
            return f"{owner.name} instance has no {method_name} method"
        if method_name == "randbytes" and len(methods) == 1 and not methods[0].decorator_list and not patched:
            reason = _bound_method_argument_failure(call, methods[0])
            if reason:
                return reason

    # A source-defined instance without a base or __call__ cannot replace a
    # callable such as os.urandom. Unknown inheritance/dynamic class patches
    # remain unresolved rather than being treated as noncallable.
    assignments = analyzer._reaching_assignments(ast.unparse(call.func), index)
    if len(assignments) == 1 and isinstance(assignments[0][0], ast.Call):
        construction, at = assignments[0]
        owner = local_class(construction.func, at)
        if owner is not None and not owner.bases and not owner.decorator_list and not owner.keywords:
            if "__call__" not in members(owner) and not class_is_patched(owner):
                return f"{ast.unparse(call.func)} is a {owner.name} instance without __call__"

    owner = local_class(call.func, index)
    if owner is not None and len(owner.bases) == 1 and not owner.decorator_list and not owner.keywords:
        base = owner.bases[0]
        if isinstance(base, ast.Subscript):
            base = base.value
        base_text = ast.unparse(base)
        head, separator, tail = base_text.partition(".")
        canonical = analyzer.import_aliases.get(head, head) + separator + tail
        # A local class with the same name as an imported Generator is custom.
        shadowed = analyzer._reaching_assignments(head, analyzer._node_span(base)[0]) or any(
            isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name == head
            for item in analyzer.tree.body
        )
        if canonical in {"typing.Generator", "collections.abc.Generator"} and not shadowed and not class_is_patched(owner):
            implemented = members(owner)
            if "__abstractmethods__" in implemented:
                return None
            missing = {"send", "throw"} - implemented
            if missing:
                return f"{owner.name} inherits abstract Generator without implementing {', '.join(sorted(missing))}"
    return None


def _bound_method_argument_failure(call: ast.Call, method: ast.FunctionDef) -> str | None:
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(keyword.arg is None for keyword in call.keywords):
        return None
    positional = [arg.arg for arg in (*method.args.posonlyargs, *method.args.args)]
    if not positional:
        return None
    required_count = max(0, len(positional) - len(method.args.defaults) - 1)
    positional = positional[1:]  # The instance binds the first parameter.
    positional_only = {arg.arg for arg in method.args.posonlyargs[1:]}
    keywords = {keyword.arg for keyword in call.keywords}
    label = f"local method {method.name}"
    if len(call.args) > len(positional) and method.args.vararg is None:
        return f"{label} received too many positional arguments"
    if (set(positional[:len(call.args)]) - positional_only) & keywords:
        return f"{label} supplies an argument more than once"
    accepted_keywords = (set(positional) - positional_only) | {arg.arg for arg in method.args.kwonlyargs}
    if keywords - accepted_keywords and method.args.kwarg is None:
        return f"{label} received an unsupported keyword argument"
    bound = set(positional[:len(call.args)]) | (keywords - positional_only)
    required = set(positional[:required_count]) | {
        arg.arg for arg, default in zip(method.args.kwonlyargs, method.args.kw_defaults)
        if default is None
    }
    if required - bound:
        return f"{label} is missing a required argument"
    return None


def _float_expression(node: ast.AST, index: int, analyzer: PythonAnalyzer,
                      seen: frozenset[tuple[str, int]]) -> bool:
    """Resolve the native random() float conversion errors observed in N028."""
    key = (ast.unparse(node), index)
    if key in seen or len(seen) >= 12:
        return False
    seen = seen | {key}
    if isinstance(node, ast.Constant):
        return type(node.value) is float
    if isinstance(node, (ast.Name, ast.Attribute)):
        assignments = analyzer._reaching_assignments(ast.unparse(node), index)
        return bool(assignments) and all(
            _float_expression(value, at, analyzer, seen) for value, at in assignments
        )
    if isinstance(node, ast.Call):
        native = analyzer.native_random_call_name(ast.unparse(node.func), index)
        return native in {
            "random.random", "random.Random.random", "random.SystemRandom.random",
            "secrets.SystemRandom.random",
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub)):
        def numeric(value: ast.AST) -> bool:
            if _float_expression(value, index, analyzer, seen) or _integer_expression(value, index, analyzer, frozenset()):
                return True
            resolved = analyzer.resolve_ints(ast.unparse(value), index)
            return bool(resolved.values) and all(item.lstrip("+-").isdigit() for item in resolved.values)
        return (
            _float_expression(node.left, index, analyzer, seen) and numeric(node.right)
        ) or (
            _float_expression(node.right, index, analyzer, seen) and numeric(node.left)
        )
    return False


def _integer_expression(node: ast.AST, index: int, analyzer: PythonAnalyzer,
                        seen: frozenset[tuple[str, int]]) -> bool:
    """Trace a single native integer/choice result through simple local returns."""
    key = (ast.unparse(node), index)
    if key in seen or len(seen) >= 12:
        return False
    seen = seen | {key}
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    if isinstance(node, (ast.Name, ast.Attribute)):
        assignments = analyzer._reaching_assignments(ast.unparse(node), index)
        if (not assignments and isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            owner = analyzer._enclosing_class(node)
            candidates = [item for item in analyzer._assignments.get(ast.unparse(node), ())
                          if analyzer._enclosing_class(item[0]) is owner]
            if len(candidates) == 1:
                assignments = candidates
        if assignments:
            return all(_integer_expression(value, at, analyzer, seen) for value, at in assignments)
        if isinstance(node, ast.Attribute):
            # A plain read-only property can expose an integer stored by the
            # constructor. Do not infer inherited or dynamically patched types.
            constructions = analyzer._python_constructions(node.value, index)
            owners = {construction.func.id for construction in constructions
                      if isinstance(construction.func, ast.Name)}
            classes = [item for item in analyzer.tree.body
                       if isinstance(item, ast.ClassDef) and item.name in owners]
            if len(classes) == 1 and len(owners) == 1:
                owner = classes[0]
                properties = [item for item in owner.body if isinstance(item, ast.FunctionDef)
                              and item.name == node.attr]
                patched = any(isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Store)
                              and item.attr == node.attr for item in ast.walk(analyzer.tree))
                if (not owner.bases and not owner.decorator_list and not owner.keywords and not patched
                        and len(properties) == 1 and len(properties[0].decorator_list) == 1
                        and isinstance(properties[0].decorator_list[0], ast.Name)
                        and properties[0].decorator_list[0].id == "property"):
                    returns = analyzer._direct_function_returns(properties[0])
                    if len(returns) == 1:
                        return _integer_expression(returns[0], analyzer._node_span(returns[0])[0], analyzer, seen)
        return False
    if not isinstance(node, ast.Call):
        return False
    if (isinstance(node.func, ast.Name) and node.func.id == "next" and 1 <= len(node.args) <= 2
            and not node.keywords and not analyzer._reaching_assignments("next", index)
            and not any(function.name == "next" for function in analyzer._function_nodes)
            and not any(arg.arg == "next" for arg in ast.walk(analyzer.tree) if isinstance(arg, ast.arg))):
        if len(node.args) == 2 and not _integer_expression(node.args[1], index, analyzer, seen):
            return False
        generator = node.args[0]
        if isinstance(generator, ast.GeneratorExp):
            return _integer_expression(generator.elt, analyzer._node_span(generator.elt)[0], analyzer, seen)
        if isinstance(generator, ast.Call):
            functions = analyzer._matching_functions(generator, caller_index=index)
            if len(functions) == 1 and not functions[0].decorator_list:
                function = functions[0]
                yields = [item for item in ast.walk(function) if isinstance(item, (ast.Yield, ast.YieldFrom))
                          and analyzer._enclosing_function(item) is function]
                return bool(yields) and all(
                    isinstance(item, ast.Yield) and item.value is not None
                    and _integer_expression(item.value, analyzer._node_span(item.value)[0], analyzer, seen)
                    for item in yields
                )
    native = analyzer.native_random_call_name(ast.unparse(node.func), index)
    if native is not None:
        receiver, _, method = native.rpartition(".")
        if receiver in {"random", "random.Random", "random.SystemRandom", "secrets.SystemRandom"}:
            if method in {"getrandbits", "randint", "randrange"}:
                return True
        if native in {"secrets.randbits", "secrets.randbelow"}:
            return True
        if method == "choice" and receiver in {"random", "random.Random", "random.SystemRandom", "secrets.SystemRandom", "secrets"} and node.args:
            sequence = node.args[0]
            if isinstance(sequence, (ast.List, ast.Tuple)):
                return bool(sequence.elts) and all(_integer_expression(item, index, analyzer, seen) for item in sequence.elts)
            if isinstance(sequence, ast.ListComp):
                return _integer_expression(sequence.elt, analyzer._node_span(sequence.elt)[0], analyzer, seen)
    functions = analyzer._matching_functions(node, caller_index=index)
    if len(functions) != 1:
        return False
    function = functions[0]
    if function.decorator_list or isinstance(function, ast.AsyncFunctionDef):
        return False
    if any(isinstance(item, (ast.Yield, ast.YieldFrom)) for item in ast.walk(function)):
        return False
    returns = analyzer._direct_function_returns(function)
    return len(returns) == 1 and _integer_expression(
        returns[0], analyzer._node_span(returns[0])[0], analyzer, seen,
    )


def _nonempty_comprehension(node: ast.ListComp | ast.GeneratorExp, resolved_name: Callable[[ast.AST], str]) -> bool:
    """Only infer element execution for simple, literal nonempty iterables."""
    for generator in node.generators:
        if generator.ifs or generator.is_async:
            return False
        iterable = generator.iter
        if isinstance(iterable, (ast.List, ast.Tuple)) and iterable.elts:
            continue
        if (isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Name)
                and resolved_name(iterable.func) == "range" and not iterable.keywords
                and 1 <= len(iterable.args) <= 3
                and all(isinstance(arg, ast.Constant) and type(arg.value) is int for arg in iterable.args)):
            try:
                if len(range(*(arg.value for arg in iterable.args))) > 0:
                    continue
            except (ValueError, OverflowError):
                pass
        return False
    return bool(node.generators)


def _definitely_bytes(node: ast.AST, resolved_name: Callable[[ast.AST], str]) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, bytes)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            return _definitely_bytes(node.left, resolved_name) and _definitely_bytes(node.right, resolved_name)
        if isinstance(node.op, ast.Mult):
            return (
                isinstance(node.right, ast.Constant) and type(node.right.value) is int
                and _definitely_bytes(node.left, resolved_name)
            ) or (
                isinstance(node.left, ast.Constant) and type(node.left.value) is int
                and _definitely_bytes(node.right, resolved_name)
            )
    identity = resolved_name(node)
    if identity in {
        "bytes()", "bytearray()", "os.urandom()", "secrets.token_bytes()",
        "random.randbytes()", *{f"{receiver}.randbytes()" for receiver in _RNG_INSTANCES},
        _HASHES + "Hash().finalize()",
    }:
        return True
    return (
        identity.startswith("hashlib.") and identity.endswith("().digest()")
    )
