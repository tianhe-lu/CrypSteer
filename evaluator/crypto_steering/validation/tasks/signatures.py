"""Connected RSA-signature operation contracts.

The language frontends establish syntax and type availability where possible.
These checks cover the remaining definite source-level failures: unavailable
native operations, unusable native call shapes, wrong key roles, and signing
and verification flows that cannot share a key pair.
"""

from __future__ import annotations

import ast
import re

from crypto_steering.validation.models import Language
from crypto_steering.validation.source.index import analyzer_for
from crypto_steering.validation.tasks.shared.typescript_api import node_crypto_calls


def signature_completion_failure(
    language: Language,
    source: str,
    task_id: str,
) -> str | None:
    if language == "python":
        return _python_signature_failure(source)
    if language == "typescript":
        return _typescript_signature_failure(source)
    if language == "csharp":
        return _csharp_signature_failure(source)
    if language == "java":
        return _java_signature_failure(source, task_id)
    return None


def csharp_signature_argument_failure(source: str) -> str | None:
    """Reject only when every C# signing route has public-only key state."""

    analyzer = analyzer_for("csharp", source)
    outcomes: list[str | None] = []
    for call in analyzer.calls("SignData", "SignHash"):
        function_name = analyzer.containing_function_name(call.start)
        definitions = analyzer.function(function_name) if function_name else ()
        if not definitions:
            continue
        body = source[definitions[0].start : definitions[0].end]
        relative_call = call.start - definitions[0].start
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        failure: str | None = None
        if receiver and re.search(
            rf"\b{re.escape(receiver)}\s*\.\s*ImportSubjectPublicKeyInfo\s*\(",
            body[:relative_call],
        ):
            failure = "RSA signing cannot use an imported public-only key"
        elif receiver and re.search(
            rf"\b{re.escape(receiver)}\s*\.\s*ImportParameters\s*\(\s*"
            rf"{re.escape(receiver)}\s*\.\s*ExportParameters\s*\(\s*false\s*\)",
            body[:relative_call],
            flags=re.IGNORECASE,
        ):
            failure = "RSA signing cannot use re-imported public-only parameters"
        outcomes.append(failure)
    if outcomes and all(failure is not None for failure in outcomes):
        return next(failure for failure in outcomes if failure is not None)
    return None


def _python_signature_failure(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    if "serialization" not in imported_names and any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "serialization"
        for node in ast.walk(tree)
    ):
        return "cryptography serialization is used without importing or defining it"

    parents = _python_parents(tree)
    backend_generation = _python_backend_key_generation_failure(tree)
    if backend_generation is not None:
        return backend_generation
    unavailable_rsa_imports = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "cryptography.hazmat.primitives.asymmetric.rsa"
        for alias in node.names
        if alias.name in {"RSASigner", "Signer", "Verifier"}
    }
    if unavailable_rsa_imports:
        imported = sorted(unavailable_rsa_imports)[0]
        return f"cryptography RSA has no {imported} constructor"
    unavailable = {
        "rsa.generate_key": (
            "cryptography RSA uses generate_private_key, not generate_key"
        ),
        "rsa.RSASigner": (
            "cryptography RSA keys use their sign method; RSASigner is unavailable"
        ),
        "rsa.Signer": (
            "cryptography RSA keys use their sign method; Signer is unavailable"
        ),
        "rsa.Verifier": (
            "cryptography RSA keys use their verify method; Verifier is unavailable"
        ),
        "rsa.RSAPublicKey.from_private_key": (
            "cryptography RSA public keys come from private_key.public_key(); "
            "RSAPublicKey has no from_private_key operation"
        ),
        "serialization.generate_private_key": (
            "cryptography serialization does not generate RSA keys; "
            "use rsa.generate_private_key"
        ),
        "rsa.generate_public_key": "cryptography RSA has no generate_public_key operation",
        "utils.encode_int_to_bytes": (
            "cryptography asymmetric utils has no encode_int_to_bytes operation"
        ),
        "serialization.load_der_private_bytes": (
            "cryptography serialization uses load_der_private_key, not "
            "load_der_private_bytes"
        ),
        "serialization.load_der_public_bytes": (
            "cryptography serialization uses load_der_public_key, not "
            "load_der_public_bytes"
        ),
        "hashes.HashAlgorithmIdentifier": "cryptography hashes has no HashAlgorithmIdentifier operation",
        "padding.MGFf": "cryptography RSA padding has no MGFf operation",
        "padding.MGFv1": "cryptography RSA padding has no MGFv1 operation",
        "padding.MGFHash": "cryptography RSA padding has no MGFHash operation",
        "padding.MGF256": "cryptography RSA padding has no MGF256 operation",
        "padding.MGF8": "cryptography RSA padding has no MGF8 operation",
        "padding.MGF31H": "cryptography RSA padding has no MGF31H operation",
        "padding.MGF_Sha256": "cryptography RSA padding has no MGF_Sha256 operation",
        "padding.PKS1v15": "cryptography RSA padding uses PKCS1v15, not PKS1v15",
        "padding.PSS.MAX_PRESET_SALT_LENGTH": (
            "cryptography RSA PSS has no MAX_PRESET_SALT_LENGTH constant"
        ),
        "serialization.Encoding.Der": "serialization.Encoding uses DER, not Der",
    }
    symmetric_padding_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "cryptography.hazmat.primitives"
        for alias in node.names
        if alias.name == "padding"
    }
    asymmetric_padding_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "cryptography.hazmat.primitives.asymmetric"
        for alias in node.names
        if alias.name == "padding"
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = _python_name(node)
            unavailable_name = next(
                (
                    operation
                    for operation in unavailable
                    if name == operation or name.endswith(f".{operation}")
                ),
                None,
            )
            if unavailable_name is not None:
                return unavailable[unavailable_name]
            if ".PSS.MGF1" in name:
                return "RSA MGF1 is provided by the padding module, not by PSS"
            if ".PSS.SAFT" in name:
                return "cryptography RSA PSS has no SAFT namespace"
            if node.attr in {"sign_with", "verify_with", "with_algorithm"}:
                return (
                    f"cryptography RSA keys do not provide a {node.attr} operation"
                )
            if node.attr in {"signer", "signing", "verifier", "verifyer"}:
                return (
                    "cryptography RSA keys use direct sign and verify methods, "
                    f"not {node.attr}"
                )
            if (
                name.endswith(".PSS")
                and name.rsplit(".", 1)[0] in symmetric_padding_aliases
                and name.rsplit(".", 1)[0] not in asymmetric_padding_aliases
            ):
                return "RSA PSS must come from cryptography's asymmetric padding module"
            if node.attr == "publickey":
                return "cryptography RSA private keys expose public_key(), not publickey()"
            if name == "rsa.RSAPrivateKey.load_from_pem":
                return (
                    "cryptography RSA interfaces do not provide load_from_pem; "
                    "use serialization.load_pem_private_key"
                )

    if _python_public_key_called_on_known_public_key(tree):
        return "public_key() is called on a value that is already a public key"

    if _python_rsa_receives_hash_object(tree):
        return "PyCA RSA sign and verify require bytes, not a hashlib hash object"

    if _python_called_hash_parameter_receives_instance(tree):
        return "a helper calls a constructed hash-algorithm object as though it were a constructor"

    if _python_rsa_receives_integer_data(tree):
        return "PyCA RSA sign and verify require bytes, not an integer digest"

    for call in _python_calls(tree):
        name = _python_name(call.func)
        simple = name.rsplit(".", 1)[-1]
        if simple == "PSS":
            mask_generation = next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == "mgf"
                ),
                call.args[0] if call.args else None,
            )
            if (
                isinstance(mask_generation, ast.Attribute)
                and mask_generation.attr.startswith("MGF")
            ):
                return "RSA PSS requires a constructed mask-generation function"
        if simple == "MGF1" and (
            len(call.args) > 1
            or any(keyword.arg != "algorithm" for keyword in call.keywords)
            or (call.args and any(keyword.arg == "algorithm" for keyword in call.keywords))
        ):
            return "MGF1 accepts exactly one hash-algorithm argument"
        if (
            simple == "exchange"
            and isinstance(call.func, ast.Attribute)
            and not call.args
            and not call.keywords
        ):
            receiver = _python_name(call.func.value)
            if re.search(
                rf"\b{re.escape(receiver)}\s*=\s*serialization\.load_(?:pem|der|ssh)_private_key\s*\(",
                source,
            ):
                return "loaded RSA private-key material does not provide a zero-argument exchange operation"
        if simple == "encode_dss_signature" and len(call.args) != 2:
            return "encode_dss_signature requires the two integer DSS components"
        if simple == "load_ssh_public_key" and (
            len(call.args) > 2
            or any(keyword.arg == "password" for keyword in call.keywords)
        ):
            return "load_ssh_public_key does not accept a password argument"
        if simple == "sign" and isinstance(call.func, ast.Attribute):
            if call.func.attr == "signer":
                return "cryptography RSA keys no longer provide a signer operation"
            if _python_likely_key_receiver(call.func.value) and any(
                keyword.arg not in {"algorithm", "data", "padding"}
                for keyword in call.keywords
            ):
                return "PyCA RSA sign received an unsupported keyword argument"
            if len(call.args) >= 3 and _is_hash_expression(call.args[1]) and _is_padding_expression(call.args[2]):
                return "RSA sign requires padding before the hash algorithm"
        if simple == "verify" and isinstance(call.func, ast.Attribute):
            if _python_likely_key_receiver(call.func.value) and any(
                keyword.arg not in {"algorithm", "data", "padding", "signature"}
                for keyword in call.keywords
            ):
                return "PyCA RSA verify received an unsupported keyword argument"
            if len(call.args) >= 4 and _is_hash_expression(call.args[2]) and _is_padding_expression(call.args[3]):
                return "RSA verify requires padding before the hash algorithm"
            if len(call.args) >= 2 and _looks_like_message(call.args[0]) and _looks_like_signature(call.args[1]):
                return "RSA verify requires the signature before the message"
            if (
                len(call.args) >= 4
                and _looks_like_signature(call.args[1])
                and _looks_like_message(call.args[3])
            ):
                return "RSA verify requires the signature first and message second"

    if any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr in {"signer", "signing"}
        for call in _python_calls(tree)
    ):
        return "cryptography RSA keys use a direct sign operation"

    legacy_verifiers = {
        target.id
        for assignment in ast.walk(tree)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        and isinstance(assignment.value, ast.Call)
        and isinstance(assignment.value.func, ast.Attribute)
        and assignment.value.func.attr in {"verifier", "verify"}
        and not assignment.value.args
        and not assignment.value.keywords
        for target in (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
        if isinstance(target, ast.Name)
    }
    if legacy_verifiers and any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "verify"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in legacy_verifiers
        for call in _python_calls(tree)
    ):
        return "cryptography RSA keys use a direct verify operation"

    complete_round_trip = _python_has_complete_signature_round_trip(tree, parents)
    connection_failure = _python_signature_connection_failure(tree, parents)
    if connection_failure is not None:
        if not complete_round_trip:
            return connection_failure

    if not complete_round_trip and _python_signs_with_an_unshared_generated_key(tree, parents):
        return "signing and verification use independently generated RSA key pairs"

    if not complete_round_trip and _python_signs_and_verifies_with_fresh_loaded_keys(tree, parents):
        return "signing and verification load independently generated RSA key pairs"

    if _python_decodes_raw_signature(tree):
        return "raw signature bytes are decoded as Base64 before verification"

    undefined = _python_undefined_module_receiver(tree, parents, "public_key")
    if undefined is not None:
        return f"RSA key expression {undefined!r} is unavailable in module scope"

    unavailable = _python_unavailable_local_key_receiver(tree, parents)
    if unavailable is not None:
        return f"RSA key expression {unavailable!r} is unavailable in its function scope"

    role_failure = _python_key_role_failure(tree, parents)
    if role_failure is not None:
        if not complete_round_trip:
            return role_failure

    loader_failure = _python_serialized_key_loader_failure(tree, parents)
    if loader_failure is not None:
        if not complete_round_trip:
            return loader_failure

    if not complete_round_trip and _python_same_material_loaded_as_private_and_public(tree):
        return (
            "the same serialized bytes are loaded as both an RSA private key "
            "and its public key"
        )

    if not complete_round_trip and _python_serialized_private_material_is_loaded_as_public_key(tree, parents):
        return (
            "serialized private-key material is supplied where verification "
            "requires the corresponding public key"
        )

    return None


def _python_has_complete_signature_round_trip(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Return whether one connected native sign/verify route is usable."""

    def scope(node: ast.AST) -> ast.AST:
        return _python_containing_function(node, parents) or tree

    def assignment_before(name: str, before: ast.AST) -> ast.AST | None:
        candidates = [
            node
            for node in ast.walk(scope(before))
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and getattr(node, "lineno", 0) < getattr(before, "lineno", 10**9)
            and any(
                name in _python_target_names((target,))
                for target in (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
            )
        ]
        return max(candidates, key=lambda item: getattr(item, "lineno", 0)).value if candidates else None

    def public_receiver(expression: ast.AST, before: ast.AST) -> tuple[bool, str | None]:
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "public_key"
        ):
            return True, _python_name(expression.func.value)
        if isinstance(expression, ast.Name):
            origin = assignment_before(expression.id, before)
            if isinstance(origin, ast.Call):
                callee = _python_name(origin.func)
                if callee.endswith(".public_key"):
                    return True, _python_name(origin.func.value) if isinstance(origin.func, ast.Attribute) else None
                if callee.endswith(
                    (
                        "load_pem_public_key",
                        "load_der_public_key",
                        "load_ssh_public_key",
                    )
                ):
                    return True, None
        return False, None

    for verify in _python_calls(tree):
        if not (
            isinstance(verify.func, ast.Attribute)
            and verify.func.attr == "verify"
            and len(verify.args) >= 4
        ):
            continue
        receiver_is_public, private_origin = public_receiver(verify.func.value, verify)
        if not receiver_is_public:
            continue
        signature = verify.args[0]
        if not isinstance(signature, ast.Name):
            continue
        origin = assignment_before(signature.id, verify)
        if not (
            isinstance(origin, ast.Call)
            and isinstance(origin.func, ast.Attribute)
            and origin.func.attr == "sign"
            and origin.args
        ):
            continue
        if ast.dump(origin.args[0]) != ast.dump(verify.args[1]):
            continue
        signing_receiver = _python_name(origin.func.value)
        if private_origin is not None and signing_receiver != private_origin:
            continue
        return True
    return False


def _python_likely_key_receiver(expression: ast.AST) -> bool:
    """Identify direct PyCA key receivers without classifying custom wrappers."""

    name = _python_name(expression).lower()
    return bool(
        re.search(r"(?:^|\.)(?:key|private_key|public_key)$", name)
        or name.endswith(
            (
                ".public_key()",
                "serialization.load_der_private_key()",
                "serialization.load_der_public_key()",
                "serialization.load_pem_private_key()",
                "serialization.load_pem_public_key()",
                "serialization.load_ssh_private_key()",
                "serialization.load_ssh_public_key()",
            )
        )
    )


def _python_backend_key_generation_failure(tree: ast.Module) -> str | None:
    """Reject RSA generation methods that the backend object does not expose."""

    backend_names = {
        target.id
        for assignment in ast.walk(tree)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        and isinstance(assignment.value, ast.Call)
        and _python_name(assignment.value.func).rsplit(".", 1)[-1]
        == "default_backend"
        for target in (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
        if isinstance(target, ast.Name)
    }
    for call in _python_calls(tree):
        if not isinstance(call.func, ast.Attribute) or call.func.attr not in {
            "generate_key",
            "generate_private_key",
        }:
            continue
        receiver = call.func.value
        if not (
            isinstance(receiver, ast.Call)
            and _python_name(receiver.func).rsplit(".", 1)[-1] == "default_backend"
        ) and not (
            isinstance(receiver, ast.Name) and receiver.id in backend_names
        ):
            continue
        return (
            "cryptography backends do not generate RSA keys; "
            "use rsa.generate_private_key"
        )
    return None


def _python_signs_and_verifies_with_fresh_loaded_keys(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Find sign/verify helpers that load unrelated fresh private-key bytes."""

    sign_functions: set[str] = set()
    verify_functions: set[str] = set()
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        local_calls = tuple(
            call
            for call in _python_calls(function)
            if _containing_function(call, parents) is function
        )
        loads_fresh_private = any(
            _python_name(call.func).endswith(
                ("serialization.load_pem_private_key", "serialization.load_der_private_key")
            )
            and call.args
            and any(
                isinstance(nested, ast.Call)
                and _python_name(nested.func).rsplit(".", 1)[-1]
                in {"urandom", "token_bytes", "get_random_bytes"}
                for nested in ast.walk(call.args[0])
            )
            for call in local_calls
        )
        if not loads_fresh_private:
            continue
        if any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "sign"
            for call in local_calls
        ):
            sign_functions.add(function.name)
        if any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "verify"
            for call in local_calls
        ):
            verify_functions.add(function.name)
    return any(
        sign_function != verify_function
        for sign_function in sign_functions
        for verify_function in verify_functions
    )


def _python_public_key_called_on_known_public_key(tree: ast.Module) -> bool:
    public_key_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "public_key"
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        public_key_names.update(
            target.id for target in targets if isinstance(target, ast.Name)
        )

    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "public_key"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in public_key_names
        for node in ast.walk(tree)
    )


def _python_serialized_key_loader_failure(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    """Reject source-provable non-byte inputs to PyCA key loaders."""

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return_roles = _python_function_return_roles(tree, parents)
    tuple_roles = _python_tuple_function_return_roles(tree, parents, return_roles)
    loader_parameters: dict[str, dict[int, str]] = {}

    for function_name, function in functions.items():
        parameters = tuple(function.args.posonlyargs) + tuple(function.args.args)
        positions = {parameter.arg: index for index, parameter in enumerate(parameters)}
        required: dict[int, str] = {}
        for call in _python_calls(function):
            if _containing_function(call, parents) is not function:
                continue
            required_role = _python_loader_required_role(call)
            argument = _python_loader_data_argument(call)
            if (
                required_role is not None
                and isinstance(argument, ast.Name)
                and argument.id in positions
            ):
                required[positions[argument.id]] = required_role
        if required:
            loader_parameters[function_name] = required

    for call in _python_calls(tree):
        required_role = _python_loader_required_role(call)
        argument = _python_loader_data_argument(call)
        if required_role is not None and argument is not None:
            scope = _containing_function(call, parents)
            if (
                scope is not None
                and isinstance(argument, ast.Name)
                and argument.id
                in {
                    parameter.arg
                    for parameter in (
                        *scope.args.posonlyargs,
                        *scope.args.args,
                    )
                }
            ):
                # The local parameter's concrete role is established from
                # calls to this helper below.  A same-named value in another
                # scope must not leak into this loader expression.
                continue
            roles = _python_roles_in_scope(tree, parents, scope, return_roles)
            roles.update(
                _python_tuple_roles_in_scope(
                    tree,
                    parents,
                    scope,
                    tuple_roles,
                )
            )
            observed = _python_expression_role(argument, roles, return_roles)
            failure = _python_loader_role_failure(required_role, observed)
            if failure is not None:
                return failure

        if not isinstance(call.func, ast.Name):
            continue
        requirements = loader_parameters.get(call.func.id)
        if not requirements:
            continue
        scope = _containing_function(call, parents)
        roles = _python_roles_in_scope(tree, parents, scope, return_roles)
        roles.update(
            _python_tuple_roles_in_scope(
                tree,
                parents,
                scope,
                tuple_roles,
            )
        )
        for position, required_role in requirements.items():
            if position >= len(call.args):
                continue
            observed = _python_expression_role(
                call.args[position],
                roles,
                return_roles,
            )
            failure = _python_loader_role_failure(required_role, observed)
            if failure is not None:
                return f"{call.func.id} {failure}"
    return None


def _python_loader_required_role(call: ast.Call) -> str | None:
    name = _python_name(call.func)
    for key_role in ("private", "public"):
        for encoding in ("der", "pem", "ssh"):
            if name.endswith(f"serialization.load_{encoding}_{key_role}_key"):
                return f"serialized_{key_role}_{encoding}"
    return None


def _python_loader_data_argument(call: ast.Call) -> ast.AST | None:
    if call.args:
        return call.args[0]
    return next(
        (
            keyword.value
            for keyword in call.keywords
            if keyword.arg in {"data", "key_data"}
        ),
        None,
    )


def _python_same_material_loaded_as_private_and_public(tree: ast.Module) -> bool:
    """Find one concrete expression passed to opposing serialized-key loaders."""

    inputs: dict[str, set[str]] = {"private": set(), "public": set()}
    for call in _python_calls(tree):
        role = _python_loader_required_role(call)
        argument = _python_loader_data_argument(call)
        if role is None or argument is None or isinstance(argument, ast.Name):
            continue
        key_role = "private" if "_private_" in role else "public"
        inputs[key_role].add(
            ast.dump(argument, annotate_fields=True, include_attributes=False)
        )
    return bool(inputs["private"] & inputs["public"])


def _python_loader_role_failure(
    required: str,
    observed: str | None,
) -> str | None:
    if observed is None:
        return None
    if observed == required:
        return None
    if observed in {"private", "public"}:
        return "receives a live RSA key object where serialized key bytes are required"
    if observed == "signature":
        return "receives signature bytes where serialized key bytes are required"
    if observed.startswith("text_serialized_"):
        return "receives decoded text where serialized key bytes are required"
    if not observed.startswith("serialized_"):
        return None
    required_parts = required.split("_")
    observed_parts = observed.split("_")
    required_key_role = required_parts[1]
    observed_key_role = observed_parts[1]
    if required_key_role != observed_key_role:
        return (
            f"receives serialized {observed_key_role} key material where a "
            f"serialized {required_key_role} key is required"
        )
    if len(observed_parts) == 2:
        return None
    if len(required_parts) >= 3 and observed_parts[2] != required_parts[2]:
        return (
            f"receives {observed_parts[2].upper()} key material through a "
            f"{required_parts[2].upper()} loader"
        )
    return None


def _python_encodes_definite_bytes(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Find a helper parameter whose invoked value is statically bytes."""

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    bytes_assignments = [
        (node, target.id)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, bytes)
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    ]
    for function in functions.values():
        parameters = [
            argument.arg
            for argument in (*function.args.posonlyargs, *function.args.args)
        ]
        encoded = {
            node.func.value.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "encode"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in parameters
        }
        for parameter in encoded:
            position = parameters.index(parameter)
            for call in ast.walk(tree):
                if not (
                    isinstance(call, ast.Call)
                    and _python_name(call.func).rsplit(".", 1)[-1] == function.name
                    and position < len(call.args)
                ):
                    continue
                argument = call.args[position]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, bytes):
                    return True
                if isinstance(argument, ast.Name):
                    call_scope = _python_containing_function(call, parents)
                    candidates = [
                        statement
                        for statement, name in bytes_assignments
                        if name == argument.id
                        and _python_containing_function(statement, parents) is call_scope
                        and getattr(statement, "lineno", 0)
                        < getattr(call, "lineno", 10**9)
                    ]
                    if candidates:
                        return True
    return False


def _python_containing_function(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _typescript_signature_failure(source: str) -> str | None:
    analyzer = analyzer_for("typescript", source)
    if re.search(r"\bgenerateKeyPairSync\s*\(", source) and re.search(
        r"\.\s*get(?:Private|Public)\s*\(",
        source,
    ):
        return "Node key-pair results expose privateKey and publicKey properties, not getter methods"
    if _typescript_encrypted_private_key_lacks_passphrase(source, analyzer):
        return "the encrypted private key is used for signing without its passphrase"
    hash_receivers = {
        re.sub(r"\s+", "", assignment.name)
        for assignment in getattr(analyzer, "assignments", ())
        if re.search(r"\bcreateHash\s*\(", assignment.expression)
    }
    digest_counts: dict[str, int] = {}
    for call in analyzer.calls("digest"):
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        if receiver in hash_receivers:
            digest_counts[receiver] = digest_counts.get(receiver, 0) + 1
    if any(count > 1 for count in digest_counts.values()):
        return "a Node hash object is finalized more than once in the signature round trip"

    verifier_receivers = {
        re.sub(r"\s+", "", assignment.name)
        for assignment in getattr(analyzer, "assignments", ())
        if re.search(r"\bcreateVerify\s*\(", assignment.expression)
    }
    for call in analyzer.calls("verify"):
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        if receiver not in verifier_receivers:
            continue
        if len(call.arguments) < 2 or (
            len(call.arguments) == 2
            and re.fullmatch(
                r"[\"'](?:base64|base64url|hex|binary|latin1|utf-?8)[\"']",
                call.arguments[1].strip(),
                flags=re.IGNORECASE,
            )
        ):
            return "Verify.verify requires signature data as its second argument"

    generated_pairs = node_crypto_calls(
        analyzer,
        "generateKeyPair",
        "generateKeyPairSync",
    )
    uses_eddsa_key = any(
        pair.call.arguments
        and pair.call.arguments[0].strip().strip("\"'").lower()
        in {"ed25519", "ed448"}
        for pair in generated_pairs
    )
    for native_call in node_crypto_calls(analyzer, "sign", "verify"):
        call = native_call.call
        if not call.arguments:
            continue
        algorithm = re.sub(r"\s+", "", call.arguments[0]).lower()
        if algorithm.strip("\"'") == "rsa-pkcs1-v1_5":
            return "Node signing requires a supported digest algorithm, not RSA-PKCS1-v1_5"
        if uses_eddsa_key and algorithm not in {"null", "undefined"}:
            return "Node Ed25519 and Ed448 signing require a null algorithm argument"
        if re.search(r"(?:private|public)?key|message|data|payload", algorithm):
            return f"crypto.{native_call.operation} requires the signature algorithm first"
        # Node accepts an asymmetric private key for verification and derives
        # its public component.  Do not reject that supported overload.
        if native_call.operation == "sign" and len(call.arguments) >= 3:
            second = re.sub(r"\s+", "", call.arguments[1]).lower()
            third = re.sub(r"\s+", "", call.arguments[2]).lower()
            if ("privatekey" in second or ".export(" in second) and re.search(
                r"(?:data|message|payload|buffer)", third
            ):
                return "crypto.sign requires data before the private key"
    if len(re.findall(r"\bgenerateKeyPairSync\s*\(", source)) > 1 and re.search(
        r"\.publicKey\s*;.*?\.privateKey\s*;|\.privateKey\s*;.*?\.publicKey\s*;",
        source,
        flags=re.DOTALL,
    ):
        return "signing and verification use keys from independently generated key pairs"
    return None


def _typescript_encrypted_private_key_lacks_passphrase(
    source: str,
    analyzer: object,
) -> bool:
    encrypted_encoding = re.search(
        r"\bprivateKeyEncoding\s*:\s*\{"
        r"(?=[^{}]*\bcipher\s*:)(?=[^{}]*\bpassphrase\s*:)[^{}]*\}",
        source,
        flags=re.DOTALL,
    )
    if encrypted_encoding is None:
        return False
    native_sign_calls = [
        native_call.call for native_call in node_crypto_calls(analyzer, "sign")
    ]
    if not native_sign_calls:
        return False
    for call in native_sign_calls:
        if len(call.arguments) < 3:
            continue
        key = call.arguments[2]
        if re.search(r"\bpassphrase\s*:", key) or re.search(
            r"\bcreatePrivateKey\s*\(", key
        ):
            return False
    if re.search(r"\bcreatePrivateKey\s*\([^;]*\bpassphrase\s*:", source, re.DOTALL):
        return False
    return True


def _csharp_signature_failure(source: str) -> str | None:
    analyzer = analyzer_for("csharp", source)
    if (
        re.search(r"\.\s*SignHash\s*\(", source)
        and re.search(r"\.\s*VerifyData\s*\(", source)
        and not re.search(r"\.\s*(?:SignData|VerifyHash)\s*\(", source)
    ):
        return "SignHash and VerifyData apply incompatible hashing semantics"
    if _csharp_signs_and_verifies_different_literal_data(analyzer, source):
        return "signing and verification use different source data"
    independent_helpers = _csharp_independent_keypair_helpers(analyzer, source)
    if independent_helpers:
        return "signing and verification helpers create independent RSA key pairs"
    if _csharp_signing_helper_is_disconnected_from_exported_public_key(
        analyzer,
        source,
    ):
        return "the signing helper and exported verification key use independent RSA key pairs"
    if re.search(r"\bRSAPKCS1Signature(?:Formatter|Deformatter)\b", source) and not re.search(
        r"\.\s*SetHashAlgorithm\s*\(", source
    ):
        return "RSAPKCS1 signature formatters require SetHashAlgorithm before use"

    public_only_failure = csharp_signature_argument_failure(source)
    if public_only_failure is not None:
        return public_only_failure

    if re.search(r"\.\s*VerifyData\s*\([^;]*\)\s*;\s*return\s+true\s*;", source, re.DOTALL):
        return "the Boolean returned by RSA VerifyData is ignored"
    if re.search(r"\.\s*VerifyHash\s*\([^;]*\)\s*;\s*return\s+true\s*;", source, re.DOTALL):
        return "the Boolean returned by RSA VerifyHash is ignored"

    for call in analyzer.calls("VerifyData", "VerifyHash"):
        statement_start = max(
            source.rfind(";", 0, call.start),
            source.rfind("{", 0, call.start),
            source.rfind("}", 0, call.start),
            source.rfind("\n", 0, call.start),
        ) + 1
        prefix = source[statement_start:call.start]
        if not prefix.strip():
            return f"the Boolean returned by RSA {call.simple_name} is ignored"

    created = {
        re.sub(r"\s+", "", assignment.name)
        for assignment in getattr(analyzer, "assignments", ())
        if re.search(r"\bRSA\s*\.\s*Create\s*\(", assignment.expression)
    }
    helper_sign_keys = _csharp_helper_key_arguments(
        analyzer,
        source,
        operation_names=("SignData", "SignHash"),
        created=created,
    )
    helper_verify_keys = _csharp_helper_key_arguments(
        analyzer,
        source,
        operation_names=("VerifyData", "VerifyHash"),
        created=created,
    )
    if (
        helper_sign_keys
        and helper_verify_keys
        and helper_sign_keys.isdisjoint(helper_verify_keys)
    ):
        return "signing and verification use independently created RSA key pairs"
    if len(created) > 1 and not re.search(
        r"\.\s*(?:Import|Export)(?:RSA|Parameters)", source
    ):
        sign_keys = _cstyle_call_receivers_named(
            analyzer, ("SignData", "SignHash"), created
        ) | _cstyle_call_arguments_named(
            analyzer, ("SignData", "SignHash"), created
        )
        verify_keys = _cstyle_call_receivers_named(
            analyzer, ("VerifyData", "VerifyHash"), created
        ) | _cstyle_call_arguments_named(
            analyzer, ("VerifyData", "VerifyHash"), created
        )
        if sign_keys and verify_keys and sign_keys.isdisjoint(verify_keys):
            return "signing and verification use independently created RSA key pairs"
    if (
        re.search(r"\bSign(?:Data|Hash)\b[^{]*\{[^}]*RSA\s*\.\s*Create\s*\(", source, re.DOTALL)
        and re.search(r"\bVerifySignature\b[^{]*\{[^}]*ImportSubjectPublicKeyInfo\s*\(", source, re.DOTALL)
        and re.search(r"ExportSubjectPublicKeyInfo\s*\(", source) is None
    ):
        return "the locally generated signing key is not connected to the imported verification key"
    return None


def _csharp_signs_and_verifies_different_literal_data(
    analyzer: object,
    source: str,
) -> bool:
    """Compare source-provable literal origins of SignData and VerifyData."""

    def literal_origin(expression: str, before: int) -> str | None:
        current = expression.strip()
        seen: set[str] = set()
        for _ in range(5):
            compact = re.sub(r"\s+", "", current)
            if re.fullmatch(r"[A-Za-z_]\w*", compact) and compact not in seen:
                seen.add(compact)
                assignment = getattr(analyzer, "latest_assignment")(compact, before)
                if assignment is None:
                    return None
                current = assignment.expression
                continue
            string_names = re.findall(
                r"Encoding\s*\.\s*UTF8\s*\.\s*GetBytes\s*\(\s*"
                r"([A-Za-z_]\w*)\s*\)",
                current,
                flags=re.IGNORECASE,
            )
            if len(string_names) == 1 and string_names[0] not in seen:
                current = string_names[0]
                continue
            literal = re.fullmatch(r"\s*[\"'](?P<value>.*)[\"']\s*", current, re.DOTALL)
            return literal.group("value") if literal else None
        return None

    signed = [
        (call, literal_origin(call.arguments[0], call.start))
        for call in getattr(analyzer, "calls")("SignData")
        if call.arguments
    ]
    verified = [
        (call, literal_origin(call.arguments[0], call.start))
        for call in getattr(analyzer, "calls")("VerifyData")
        if call.arguments
    ]
    return any(
        left is not None and right is not None and left != right
        for _sign_call, left in signed
        for _verify_call, right in verified
    )


def _csharp_signing_helper_is_disconnected_from_exported_public_key(
    analyzer: object,
    source: str,
) -> bool:
    """Find a helper-local signing key paired with another freshly exported key.

    A helper that creates an RSA provider, returns only signature bytes, and
    never exposes public-key material cannot be verified with public material
    exported from a separate ``RSA.Create`` expression at its call site.
    """

    signing_helpers: set[str] = set()
    for function in getattr(analyzer, "functions", ()):
        body = source[function.start : function.end]
        if not re.search(r"\bRSA\s*\.\s*Create\s*\(", body):
            continue
        if not any(
            call.simple_name in {"SignData", "SignHash"}
            and analyzer.containing_function_name(call.start) == function.name
            for call in analyzer.calls("SignData", "SignHash")
        ):
            continue
        if re.search(r"\bExport(?:SubjectPublicKeyInfo|RSAPublicKey|Parameters)\s*\(", body):
            continue
        signing_helpers.add(function.name)
    if not signing_helpers:
        return False

    called_helpers = {
        call.simple_name
        for helper in signing_helpers
        for call in analyzer.calls(helper)
        if analyzer.containing_function_name(call.start) not in signing_helpers
    }
    if not called_helpers:
        return False

    for match in re.finditer(
        r"RSA\s*\.\s*Create\s*\([^;]*?\)\s*\)?\s*\.\s*"
        r"Export(?:SubjectPublicKeyInfo|RSAPublicKey|Parameters)\s*\(",
        source,
        flags=re.DOTALL,
    ):
        if not any(
            function.start <= match.start() <= function.end
            for function in getattr(analyzer, "functions", ())
            if function.name in signing_helpers
        ):
            return True
    return False


def _csharp_independent_keypair_helpers(analyzer: object, source: str) -> bool:
    """Find separate helpers that each create their own unshared RSA object."""

    functions = tuple(getattr(analyzer, "functions", ()))
    signing: set[str] = set()
    verifying: set[str] = set()
    definitions = {function.name: function for function in functions}
    for function in functions:
        body = source[function.start : function.end]
        if not re.search(r"\bRSA\s*\.\s*Create\s*\(", body):
            continue
        local_calls = tuple(
            call
            for call in getattr(analyzer, "calls")(
                "SignData", "SignHash", "VerifyData", "VerifyHash"
            )
            if getattr(analyzer, "containing_function_name")(call.start)
            == function.name
        )
        if any(call.simple_name in {"SignData", "SignHash"} for call in local_calls):
            signing.add(function.name)
        if any(call.simple_name in {"VerifyData", "VerifyHash"} for call in local_calls):
            verifying.add(function.name)
    if not signing or not verifying or not signing.isdisjoint(verifying):
        return False

    # A key-bearing verification parameter can connect a key created by a
    # caller or signing helper.  Without one, separately created helper-local
    # RSA objects cannot form a corresponding key pair.
    for name in verifying:
        function = definitions[name]
        if any(
            re.search(r"(?:rsa|key|certificate)", parameter, re.IGNORECASE)
            and not re.search(
                r"(?:size|length|bits?)$",
                re.sub(r"[^A-Za-z0-9]", "", parameter),
                re.IGNORECASE,
            )
            for parameter in function.parameters
        ):
            return False
    return True


def _csharp_helper_key_arguments(
    analyzer: object,
    source: str,
    *,
    operation_names: tuple[str, ...],
    created: set[str],
) -> set[str]:
    """Resolve RSA receivers inside helpers back to their call-site arguments."""

    key_arguments: set[str] = set()
    for function in getattr(analyzer, "functions", ()):
        body = source[function.start : function.end]
        parameter_positions = {
            re.sub(r"\s+", "", parameter): index
            for index, parameter in enumerate(function.parameters)
        }
        key_positions: set[int] = set()
        for call in getattr(analyzer, "calls")(*operation_names):
            if getattr(analyzer, "containing_function_name")(call.start) != function.name:
                continue
            compact_callee = re.sub(r"\s+", "", call.callee)
            if "." not in compact_callee:
                continue
            receiver = compact_callee.rsplit(".", 1)[0]
            if receiver in parameter_positions:
                key_positions.add(parameter_positions[receiver])
        if not key_positions:
            continue
        for call in getattr(analyzer, "calls")(function.name):
            if "." in re.sub(r"\s+", "", call.callee):
                continue
            for position in key_positions:
                if position >= len(call.arguments):
                    continue
                argument = re.sub(r"\s+", "", call.arguments[position])
                if argument in created:
                    key_arguments.add(argument)
    return key_arguments


def _java_signature_failure(source: str, task_id: str) -> str | None:
    if not task_id.endswith(("T1", "T2")):
        return None
    for match in re.finditer(r"\.\s*verify\s*\([^;]*\)\s*;", source, re.DOTALL):
        statement_start = max(
            source.rfind(";", 0, match.start()),
            source.rfind("{", 0, match.start()),
            source.rfind("}", 0, match.start()),
            source.rfind("\n", 0, match.start()),
        ) + 1
        prefix = source[statement_start : match.start()]
        if re.fullmatch(
            r"\s*[A-Za-z_$][\w$]*(?:\s*\.\s*[A-Za-z_$][\w$]*)*\s*",
            prefix,
        ):
            return "the Boolean returned by Signature.verify is ignored"
    analyzer = analyzer_for("java", source)
    signed_receivers: dict[tuple[str | None, str], int] = {}
    post_sign_verify_routes: list[bool] = []
    verify_state: dict[tuple[str | None, str], bool] = {}
    for call in sorted(
        analyzer.calls("initVerify", "update", "sign", "verify"),
        key=lambda item: item.start,
    ):
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        key = (analyzer.containing_function_name(call.start), receiver)
        if call.simple_name == "sign":
            signed_receivers[key] = call.start
        elif call.simple_name == "initVerify" and key in signed_receivers:
            verify_state[key] = False
        elif call.simple_name == "update" and key in verify_state:
            verify_state[key] = True
        elif call.simple_name == "verify" and key in verify_state:
            post_sign_verify_routes.append(verify_state.pop(key))
    if post_sign_verify_routes and not any(post_sign_verify_routes):
        return "verification does not update the verifier with signed data after initVerify"
    return None


def _python_signs_with_an_unshared_generated_key(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    functions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    generator_helpers = {
        function.name
        for function in functions
        if any(
            _python_name(call.func).endswith("rsa.generate_private_key")
            for call in _python_calls(function)
            if _containing_function(call, parents) is function
        )
    }
    sign_helpers = {
        function.name
        for function in functions
        if any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "sign"
            for call in _python_calls(function)
            if _containing_function(call, parents) is function
        )
    }
    verify_helpers = {
        function.name
        for function in functions
        if any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "verify"
            for call in _python_calls(function)
            if _containing_function(call, parents) is function
        )
    }
    generation_sites = tuple(
        call
        for call in _python_calls(tree)
        if (
            _python_name(call.func).endswith("rsa.generate_private_key")
            and _containing_function(call, parents) is None
        )
        or (
            isinstance(call.func, ast.Name)
            and call.func.id in generator_helpers
        )
    )
    if len(generation_sites) < 2:
        return False
    for function in functions:
        local_calls = tuple(
            call
            for call in _python_calls(function)
            if _containing_function(call, parents) is function
        )
        locally_generates = any(
            _python_name(call.func).endswith("rsa.generate_private_key")
            or (
                isinstance(call.func, ast.Name)
                and call.func.id in generator_helpers
            )
            for call in local_calls
        )
        signs = any(
            (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "sign"
            )
            or (
                isinstance(call.func, ast.Name)
                and call.func.id in sign_helpers
            )
            for call in local_calls
        )
        verifies = any(
            (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "verify"
            )
            or (
                isinstance(call.func, ast.Name)
                and call.func.id in verify_helpers
            )
            for call in local_calls
        )
        returns_public_material = any(
            isinstance(node, ast.Return)
            and node.value is not None
            and not any(
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "sign"
                or isinstance(call.func, ast.Name)
                and call.func.id in sign_helpers
                for call in _python_calls(node.value)
            )
            and re.search(r"(?:public|key)", ast.unparse(node.value), re.IGNORECASE)
            for node in ast.walk(function)
        )
        if (
            locally_generates
            and signs
            and not verifies
            and not returns_public_material
        ):
            return True
    return False


def _python_rsa_receives_hash_object(tree: ast.Module) -> bool:
    """Find a constructed hashlib object passed as RSA data."""

    hash_objects = {
        name
        for assignment in ast.walk(tree)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        and isinstance(assignment.value, ast.Call)
        and (
            re.search(
                r"(?:^|\.)(?:md5|sha(?:1|224|256|384|512)|sha3_\d+)$",
                _python_name(assignment.value.func),
                flags=re.IGNORECASE,
            )
            or _python_name(assignment.value.func).endswith("hashes.Hash")
        )
        for target in (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
        if isinstance(target, ast.Name)
        for name in (target.id,)
    }
    if not hash_objects:
        return False
    for call in _python_calls(tree):
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr == "sign" and call.args:
            data = call.args[0]
        elif call.func.attr == "verify" and len(call.args) >= 2:
            data = call.args[1]
        else:
            continue
        if any(
            isinstance(node, ast.Name) and node.id in hash_objects
            for node in ast.walk(data)
        ):
            return True
    return False


def _python_called_hash_parameter_receives_instance(tree: ast.Module) -> bool:
    """Find ``hash_algorithm()`` when its call site supplies ``SHA256()``."""

    assignments = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name)
    }
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name, function in functions.items():
        parameters = tuple(argument.arg for argument in function.args.args)
        called = {
            call.func.id
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in parameters
            and re.search(r"(?:hash|digest|algorithm)", call.func.id, re.IGNORECASE)
        }
        if not called:
            continue
        for call in _python_calls(tree):
            if not isinstance(call.func, ast.Name) or call.func.id != name:
                continue
            for parameter in called:
                position = parameters.index(parameter)
                if position >= len(call.args):
                    continue
                argument = call.args[position]
                if isinstance(argument, ast.Name):
                    argument = assignments.get(argument.id, argument)
                if isinstance(argument, ast.Call) and _is_hash_expression(argument):
                    return True
    return False


def _python_rsa_receives_integer_data(tree: ast.Module) -> bool:
    """Find a source-provable integer passed as RSA message data."""

    integer_values = {
        name
        for assignment in ast.walk(tree)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        and (
            isinstance(assignment.value, ast.Constant)
            and isinstance(assignment.value.value, int)
            and not isinstance(assignment.value.value, bool)
            or isinstance(assignment.value, ast.Call)
            and _python_name(assignment.value.func) == "int"
        )
        for name in _python_target_names(
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
    }
    for call in _python_calls(tree):
        if not isinstance(call.func, ast.Attribute):
            continue
        data = (
            call.args[0]
            if call.func.attr == "sign" and call.args
            else call.args[1]
            if call.func.attr == "verify" and len(call.args) >= 2
            else None
        )
        if data is None:
            continue
        if (
            isinstance(data, ast.Constant)
            and isinstance(data.value, int)
            and not isinstance(data.value, bool)
        ) or (
            isinstance(data, ast.Call) and _python_name(data.func) == "int"
        ) or (
            isinstance(data, ast.Name) and data.id in integer_values
        ):
            return True
    return False


def _cstyle_call_arguments_named(
    analyzer: object,
    operations: tuple[str, ...],
    names: set[str],
) -> set[str]:
    values: set[str] = set()
    for call in getattr(analyzer, "calls")(*operations):
        for argument in call.arguments:
            compact = re.sub(r"\s+", "", argument)
            if compact in names:
                values.add(compact)
    return values


def _cstyle_call_receivers_named(
    analyzer: object,
    operations: tuple[str, ...],
    names: set[str],
) -> set[str]:
    values: set[str] = set()
    for call in getattr(analyzer, "calls")(*operations):
        compact = re.sub(r"\s+", "", call.callee)
        receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
        if receiver in names:
            values.add(receiver)
    return values


def _python_key_role_failure(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return_roles = _python_function_return_roles(tree, parents)
    tuple_return_roles = _python_tuple_function_return_roles(
        tree, parents, return_roles
    )
    required_roles: dict[str, dict[int, str]] = {}
    required_signature_positions: dict[str, set[int]] = {}
    for name, function in functions.items():
        parameters = tuple(function.args.posonlyargs) + tuple(function.args.args)
        positions = {parameter.arg: index for index, parameter in enumerate(parameters)}
        roles: dict[int, str] = {}
        role_sets: dict[int, set[str]] = {}
        signature_positions: set[int] = set()
        for call in _python_calls(function):
            if _containing_function(call, parents) is not function:
                continue
            if not isinstance(call.func, ast.Attribute):
                continue
            if (
                call.func.attr == "verify"
                and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id in positions
            ):
                signature_positions.add(positions[call.args[0].id])
            if not isinstance(call.func.value, ast.Name):
                continue
            parameter = call.func.value.id
            if parameter not in positions:
                continue
            reassigned_role = _python_parameter_role_before_call(
                function,
                call,
                parameter,
                return_roles,
            )
            if call.func.attr == "sign":
                if reassigned_role != "private":
                    roles[positions[parameter]] = "private"
                    role_sets.setdefault(positions[parameter], set()).add("private")
            elif call.func.attr == "verify":
                if reassigned_role != "public":
                    roles[positions[parameter]] = "public"
                    role_sets.setdefault(positions[parameter], set()).add("public")
        if any(values == {"private", "public"} for values in role_sets.values()):
            return (
                f"{name} uses one unchanged key parameter for both RSA signing "
                "and verification"
            )
        if roles:
            required_roles[name] = roles
        if signature_positions:
            required_signature_positions[name] = signature_positions

    helper_call_failures: dict[str, list[str | None]] = {}
    for call in _python_calls(tree):
        if not isinstance(call.func, ast.Name) or (
            call.func.id not in required_roles
            and call.func.id not in required_signature_positions
        ):
            continue
        scope = _containing_function(call, parents)
        roles = _python_roles_in_scope(tree, parents, scope, return_roles)
        roles.update(
            _python_tuple_roles_in_scope(
                tree,
                parents,
                scope,
                tuple_return_roles,
            )
        )
        call_failure: str | None = None
        for position, required in required_roles.get(call.func.id, {}).items():
            if position >= len(call.args):
                continue
            argument = call.args[position]
            if isinstance(argument, ast.Name) and not _python_name_is_available(
                tree,
                parents,
                scope,
                argument.id,
            ):
                call_failure = (
                    f"{call.func.id} receives unavailable RSA key expression "
                    f"{argument.id!r}"
                )
                break
            observed = _python_expression_role(argument, roles, return_roles)
            if required == "private" and observed is not None and observed.startswith(
                ("serialized_private", "serialized_public", "text_serialized")
            ):
                call_failure = f"{call.func.id} requires a loaded RSA private-key object"
            if required == "public" and observed is not None and observed.startswith(
                ("serialized_private", "serialized_public", "text_serialized")
            ):
                call_failure = f"{call.func.id} requires a loaded RSA public-key object"
            if required == "public" and observed == "signature":
                call_failure = (
                    f"{call.func.id} receives signature bytes where RSA "
                    "verification requires a public key"
                )
            if required == "public" and observed == "bytes":
                call_failure = (
                    f"{call.func.id} receives byte strings where RSA "
                    "verification requires a public-key object"
                )
            if required == "private" and observed == "bytes":
                call_failure = (
                    f"{call.func.id} receives byte strings where RSA signing "
                    "requires a private-key object"
                )
            if required == "private" and observed == "public":
                call_failure = f"{call.func.id} receives a public key for RSA signing"
            if required == "public" and observed == "private":
                call_failure = f"{call.func.id} receives a private key for RSA verification"
            if call_failure is not None:
                break

        for position in required_signature_positions.get(call.func.id, ()):
            if call_failure is not None:
                break
            if position >= len(call.args):
                continue
            observed = _python_expression_role(call.args[position], roles, return_roles)
            if observed == "encoded_signature":
                call_failure = (
                    f"{call.func.id} passes an encoded signature to RSA verify "
                    "without decoding it"
                )
        helper_call_failures.setdefault(call.func.id, []).append(call_failure)

    for failures in helper_call_failures.values():
        if failures and all(failure is not None for failure in failures):
            return next(failure for failure in failures if failure is not None)

    receiver_call_failures: dict[str, list[str | None]] = {}
    for call in _python_calls(tree):
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr not in {"sign", "verify", "public_key"}:
            continue
        scope = _containing_function(call, parents)
        if (
            isinstance(call.func.value, ast.Name)
            and scope is not None
            and call.func.value.id in {
                parameter.arg
                for parameter in tuple(scope.args.posonlyargs)
                + tuple(scope.args.args)
            }
        ):
            # Helper parameter roles are decided from actual helper call sites
            # above.  An uncalled helper cannot provide a successful route.
            continue
        roles = _python_roles_in_scope(tree, parents, scope, return_roles)
        roles.update(
            _python_tuple_roles_in_scope(
                tree,
                parents,
                scope,
                tuple_return_roles,
            )
        )
        role = _python_expression_role(call.func.value, roles, return_roles)
        failure: str | None = None
        if call.func.attr == "verify" and role is not None and role.startswith(
            ("serialized_private", "serialized_public", "text_serialized")
        ):
            failure = "serialized public-key bytes do not provide an RSA verify operation"
        if call.func.attr == "sign" and role is not None and role.startswith(
            ("serialized_private", "serialized_public", "text_serialized")
        ):
            failure = "serialized private-key bytes do not provide an RSA sign operation"
        if call.func.attr == "verify" and role == "signature":
            failure = "signature bytes do not provide an RSA verify operation"
        if call.func.attr == "verify" and role == "bytes":
            failure = "byte strings do not provide an RSA verify operation"
        if call.func.attr == "sign" and role == "bytes":
            failure = "byte strings do not provide an RSA sign operation"
        if call.func.attr == "sign" and role == "public":
            failure = "PyCA RSA signing requires a private-key receiver"
        if call.func.attr == "verify" and role == "private":
            failure = "PyCA RSA verification requires a public-key receiver"
        if call.func.attr == "public_key" and role == "signature":
            failure = "signature bytes do not provide a public_key operation"
        receiver_call_failures.setdefault(call.func.attr, []).append(failure)
    for failures in receiver_call_failures.values():
        if failures and all(failure is not None for failure in failures):
            return next(failure for failure in failures if failure is not None)
    return None


def _python_parameter_role_before_call(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    call: ast.Call,
    parameter: str,
    return_roles: dict[str, str],
) -> str | None:
    """Resolve a key parameter that a helper deserializes before using it."""

    roles: dict[str, str] = {}
    assignments = sorted(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
            < (getattr(call, "lineno", 0), getattr(call, "col_offset", 0))
        ),
        key=lambda node: (
            getattr(node, "lineno", 0),
            getattr(node, "col_offset", 0),
        ),
    )
    for assignment in assignments:
        role = _python_expression_role(assignment.value, roles, return_roles)
        if role is None:
            continue
        targets = assignment.targets if isinstance(assignment, ast.Assign) else (assignment.target,)
        for name in _python_target_names(targets):
            roles[name] = role
    return roles.get(parameter)


def _python_serialized_private_material_is_loaded_as_public_key(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Find tuple-returned private PEM later treated as a verification key."""

    return_roles = _python_function_return_roles(tree, parents)
    tuple_roles = _python_tuple_function_return_roles(tree, parents, return_roles)
    private_element_functions = {
        (name, index)
        for name, roles in tuple_roles.items()
        for index, role in enumerate(roles)
        if role is not None and role.startswith("serialized_private")
    }
    if not private_element_functions:
        return False

    serialized_private_names: set[str] = set()
    for assignment in ast.walk(tree):
        if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
            continue
        value = assignment.value
        if not (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Call)
            and isinstance(value.value.func, ast.Name)
            and isinstance(value.slice, ast.Constant)
            and isinstance(value.slice.value, int)
            and (value.value.func.id, value.slice.value) in private_element_functions
        ):
            continue
        targets = assignment.targets if isinstance(assignment, ast.Assign) else (assignment.target,)
        serialized_private_names.update(_python_target_names(targets))

    if any(
        isinstance(call, ast.Call)
        and _python_name(call.func).endswith(
            (
                "serialization.load_der_public_key",
                "serialization.load_pem_public_key",
                "serialization.load_ssh_public_key",
            )
        )
        and call.args
        and any(
            isinstance(node, ast.Name) and node.id in serialized_private_names
            for node in ast.walk(call.args[0])
        )
        for call in _python_calls(tree)
    ):
        return True

    public_loader_parameters: dict[str, set[int]] = {}
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        parameters = tuple(function.args.posonlyargs) + tuple(function.args.args)
        positions = {parameter.arg: index for index, parameter in enumerate(parameters)}
        for call in _python_calls(function):
            if not (
                _containing_function(call, parents) is function
                and _python_name(call.func).endswith(
                    (
                        "serialization.load_der_public_key",
                        "serialization.load_pem_public_key",
                        "serialization.load_ssh_public_key",
                    )
                )
                and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id in positions
            ):
                continue
            public_loader_parameters.setdefault(function.name, set()).add(
                positions[call.args[0].id]
            )

    for call in _python_calls(tree):
        if not isinstance(call.func, ast.Name):
            continue
        positions = public_loader_parameters.get(call.func.id)
        if not positions:
            continue
        scope = _containing_function(call, parents)
        roles = _python_roles_in_scope(tree, parents, scope, return_roles)
        roles.update(_python_tuple_roles_in_scope(tree, parents, scope, tuple_roles))
        if any(
            position < len(call.args)
            and (
                _python_expression_role(call.args[position], roles, return_roles)
                or ""
            ).startswith("serialized_private")
            for position in positions
        ):
            return True
    return False


def _python_name_is_available(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
    scope: ast.FunctionDef | ast.AsyncFunctionDef | None,
    name: str,
) -> bool:
    """Return whether a required key name is declared locally or globally."""

    if scope is not None:
        parameters = tuple(scope.args.posonlyargs) + tuple(scope.args.args)
        if name in {parameter.arg for parameter in parameters}:
            return True
    for node in ast.walk(tree):
        node_scope = _containing_function(node, parents)
        if node_scope not in {None, scope}:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(
                (alias.asname or alias.name.split(".", 1)[0]) == name
                for alias in node.names
            ):
                return True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if name in _python_target_names(targets):
                return True
    return False


def _python_tuple_function_return_roles(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
    return_roles: dict[str, str],
) -> dict[str, tuple[str | None, ...]]:
    results: dict[str, tuple[str | None, ...]] = {}
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        local = _python_roles_in_scope(tree, parents, function, return_roles)
        returned = tuple(
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Return)
            and isinstance(node.value, (ast.Tuple, ast.List))
            and _containing_function(node, parents) is function
        )
        if len(returned) != 1:
            continue
        results[function.name] = tuple(
            _python_expression_role(value, local, return_roles)
            for value in returned[0].elts
        )
    return results


def _python_tuple_roles_in_scope(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
    scope: ast.FunctionDef | ast.AsyncFunctionDef | None,
    return_roles: dict[str, tuple[str | None, ...]],
) -> dict[str, str]:
    roles: dict[str, str] = {}
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _containing_function(node, parents) is scope
    ]
    assignments.sort(
        key=lambda node: (
            getattr(node, "lineno", 0),
            getattr(node, "col_offset", 0),
        )
    )
    for assignment in assignments:
        targets = assignment.targets if isinstance(assignment, ast.Assign) else (assignment.target,)
        if len(targets) != 1:
            continue
        value = assignment.value
        target = targets[0]
        if isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
                continue
            returned = return_roles.get(value.func.id)
            if returned is None or len(returned) != len(target.elts):
                continue
            for item, role in zip(target.elts, returned):
                if isinstance(item, ast.Name) and role is not None:
                    roles[item.id] = role
            continue

        if not (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Call)
            and isinstance(value.value.func, ast.Name)
            and isinstance(value.slice, ast.Constant)
            and isinstance(value.slice.value, int)
        ):
            continue
        returned = return_roles.get(value.value.func.id)
        index = value.slice.value
        if returned is not None and 0 <= index < len(returned):
            role = returned[index]
            if role is not None:
                roles[target.id] = role

    for assignment in assignments:
        targets = assignment.targets if isinstance(assignment, ast.Assign) else (assignment.target,)
        if (
            len(targets) == 1
            and isinstance(targets[0], ast.Name)
            and isinstance(assignment.value, ast.Name)
            and assignment.value.id in roles
        ):
            roles[targets[0].id] = roles[assignment.value.id]
    return roles


def _python_roles_in_scope(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
    scope: ast.FunctionDef | ast.AsyncFunctionDef | None,
    return_roles: dict[str, str],
) -> dict[str, str]:
    roles: dict[str, str] = {}
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _containing_function(node, parents) is scope
    ]
    assignments.sort(key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))
    for assignment in assignments:
        targets = assignment.targets if isinstance(assignment, ast.Assign) else (assignment.target,)
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if not names:
            continue
        role = _python_expression_role(assignment.value, roles, return_roles)
        if role is not None:
            roles.update({name: role for name in names})
    return roles


def _python_expression_role(
    expression: ast.AST,
    roles: dict[str, str],
    return_roles: dict[str, str],
) -> str | None:
    if isinstance(expression, ast.Name):
        return roles.get(expression.id)
    if isinstance(expression, ast.Constant) and isinstance(expression.value, bytes):
        return "bytes"
    if not isinstance(expression, ast.Call):
        return None
    name = _python_name(expression.func)
    if name in return_roles:
        return return_roles[name]
    if isinstance(expression.func, ast.Attribute):
        if expression.func.attr in {"private_bytes", "public_bytes"}:
            key_role = (
                "private" if expression.func.attr == "private_bytes" else "public"
            )
            encoding = _python_serialization_encoding(expression)
            return f"serialized_{key_role}" + (f"_{encoding}" if encoding else "")
        if expression.func.attr == "decode":
            role = _python_expression_role(expression.func.value, roles, return_roles)
            return (
                f"text_{role}"
                if role is not None and role.startswith("serialized_")
                else role
            )
    if name.endswith(("base64.b64encode", "base64.urlsafe_b64encode")) and expression.args:
        argument_role = _python_expression_role(expression.args[0], roles, return_roles)
        if argument_role in {"signature", "encoded_signature"}:
            return "encoded_signature"
    if name.endswith(("base64.b64decode", "base64.urlsafe_b64decode")) and expression.args:
        if _python_expression_role(expression.args[0], roles, return_roles) == "encoded_signature":
            return "signature"
    if name.endswith(
        (
            "os.urandom",
            "secrets.token_bytes",
            "bytes.fromhex",
        )
    ):
        return "bytes"
    if name.endswith(
        (
            "rsa.generate_private_key",
            "serialization.load_der_private_key",
            "serialization.load_pem_private_key",
            "serialization.load_ssh_private_key",
        )
    ):
        return "private"
    if name.endswith(
        (
            "serialization.load_der_public_key",
            "serialization.load_pem_public_key",
            "serialization.load_ssh_public_key",
        )
    ):
        return "public"
    if isinstance(expression.func, ast.Attribute) and expression.func.attr == "public_key":
        return "public"
    if isinstance(expression.func, ast.Attribute) and expression.func.attr == "sign":
        return "signature"
    return None


def _python_serialization_encoding(call: ast.Call) -> str | None:
    value = next(
        (keyword.value for keyword in call.keywords if keyword.arg == "encoding"),
        call.args[0] if call.args else None,
    )
    name = _python_name(value) if value is not None else ""
    if name.endswith("Encoding.PEM"):
        return "pem"
    if name.endswith("Encoding.DER"):
        return "der"
    if name.endswith("Encoding.OpenSSH"):
        return "ssh"
    return None


def _python_function_return_roles(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> dict[str, str]:
    roles: dict[str, str] = {}
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        local_roles: dict[str, str] = {}
        assignments = [
            node
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and _containing_function(node, parents) is function
        ]
        assignments.sort(
            key=lambda node: (
                getattr(node, "lineno", 0),
                getattr(node, "col_offset", 0),
            )
        )
        for assignment in assignments:
            role = _python_expression_role(assignment.value, local_roles, roles)
            if role is None:
                continue
            targets = assignment.targets if isinstance(assignment, ast.Assign) else (assignment.target,)
            for name in _python_target_names(targets):
                local_roles[name] = role
        returned = {
            _python_expression_role(node.value, local_roles, roles)
            for node in ast.walk(function)
            if isinstance(node, ast.Return)
            and node.value is not None
            and _containing_function(node, parents) is function
        }
        returned.discard(None)
        if len(returned) == 1:
            roles[function.name] = returned.pop()  # type: ignore[assignment]
    return roles


def _python_signature_connection_failure(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    """Find an explicitly discarded signature followed by literal verification."""

    definitions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    directly_ignored = any(
        isinstance(parents.get(call), ast.Expr)
        and isinstance(call.func, ast.Name)
        and call.func.id in definitions
        and "sign" in call.func.id.lower()
        for call in _python_calls(tree)
    )
    untransported_helpers = {
        name
        for name, function in definitions.items()
        if _python_function_discards_signature(function, parents)
    }
    ignored_sign = directly_ignored or any(
        isinstance(call.func, ast.Name)
        and call.func.id in untransported_helpers
        and _containing_function(call, parents) is not definitions[call.func.id]
        for call in _python_calls(tree)
    )
    literal_verify = False
    for call in _python_calls(tree):
        if not isinstance(call.func, ast.Name) or "verif" not in call.func.id.lower():
            continue
        definition = definitions.get(call.func.id)
        if definition is None:
            continue
        parameters = tuple(argument.arg for argument in definition.args.args)
        signature_positions = tuple(
            index for index, name in enumerate(parameters) if "sig" in name.lower()
        )
        scope = _containing_function(call, parents)
        assignments = _python_assignments_in_scope(tree, parents, scope)
        if any(
            index < len(call.args)
            and _python_is_static_bytes(call.args[index], assignments)
            for index in signature_positions
        ):
            literal_verify = True
            break
    if ignored_sign and literal_verify:
        return "the generated signature is discarded before verification"
    generated_signatures = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and (
            _python_name(node.value.func).rsplit(".", 1)[-1] == "sign"
            or "sign" in _python_name(node.value.func).lower()
        )
        for name in _python_target_names(
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
    }
    for call in _python_calls(tree):
        if not isinstance(call.func, ast.Name) or "verif" not in call.func.id.lower():
            continue
        definition = definitions.get(call.func.id)
        if definition is None:
            continue
        parameters = tuple(argument.arg for argument in definition.args.args)
        signature_positions = tuple(
            index for index, name in enumerate(parameters) if "sig" in name.lower()
        )
        scope = _containing_function(call, parents)
        assignments = _python_assignments_in_scope(tree, parents, scope)
        for index in signature_positions:
            if index >= len(call.args):
                continue
            argument = call.args[index]
            resolved = assignments.get(argument.id, argument) if isinstance(argument, ast.Name) else argument
            if isinstance(resolved, ast.BinOp) and any(
                isinstance(node, ast.Name) and node.id in generated_signatures
                for node in ast.walk(resolved)
            ):
                return "verification receives a composite payload instead of the generated signature"
    return None


def _python_function_discards_signature(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    signature_names = {
        name
        for assignment in ast.walk(function)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        and isinstance(assignment.value, ast.Call)
        and isinstance(assignment.value.func, ast.Attribute)
        and assignment.value.func.attr == "sign"
        and _containing_function(assignment, parents) is function
        for name in _python_target_names(
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
    }
    direct_sign = any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "sign"
        and _containing_function(call, parents) is function
        for call in _python_calls(function)
    )
    if not direct_sign:
        return False
    for returned in (
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and node.value is not None
        and _containing_function(node, parents) is function
    ):
        if any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "sign"
            for call in _python_calls(returned)
        ) or any(
            isinstance(node, ast.Name) and node.id in signature_names
            for node in ast.walk(returned)
        ):
            return False
    return True


def _python_is_static_bytes(
    expression: ast.AST,
    assignments: dict[str, ast.AST],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(expression, ast.Constant):
        return isinstance(expression.value, bytes)
    if isinstance(expression, ast.Name) and expression.id not in seen:
        value = assignments.get(expression.id)
        return value is not None and _python_is_static_bytes(
            value, assignments, seen | {expression.id}
        )
    return False


def _python_decodes_raw_signature(tree: ast.AST) -> bool:
    """Return whether bytes returned by ``sign`` are decoded as Base64."""

    parents = _python_parents(tree)
    raw_signatures: set[tuple[ast.AST | None, str]] = set()
    assignments = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ),
        key=lambda node: (
            getattr(node, "lineno", 0),
            getattr(node, "col_offset", 0),
        ),
    )
    for assignment in assignments:
        scope = _containing_function(assignment, parents)
        targets = assignment.targets if isinstance(assignment, ast.Assign) else (assignment.target,)
        names = _python_target_names(targets)
        value = assignment.value
        if (
            isinstance(value, ast.Call)
            and _python_name(value.func).rsplit(".", 1)[-1] == "sign"
        ):
            raw_signatures.update((scope, name) for name in names)
        elif (
            isinstance(value, ast.Call)
            and _python_name(value.func).endswith("base64.b64encode")
        ):
            raw_signatures.difference_update((scope, name) for name in names)

    return any(
        _python_name(call.func).endswith("base64.b64decode")
        and bool(call.args)
        and isinstance(call.args[0], ast.Name)
        and (_containing_function(call, parents), call.args[0].id) in raw_signatures
        for call in _python_calls(tree)
    )


def _looks_like_message(node: ast.AST) -> bool:
    return bool(re.search(r"(?:data|message|payload|padder?\s*\.\s*update)", ast.unparse(node), re.IGNORECASE))


def _looks_like_signature(node: ast.AST) -> bool:
    return bool(re.search(r"(?:signature|sig)", ast.unparse(node), re.IGNORECASE))


def _python_independent_keypair_helpers(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    sign_generators: set[str] = set()
    verify_generators: set[str] = set()
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        local_calls = [
            call
            for call in _python_calls(function)
            if _containing_function(call, parents) is function
        ]
        if not any(_python_name(call.func).endswith("rsa.generate_private_key") for call in local_calls):
            continue
        if any(isinstance(call.func, ast.Attribute) and call.func.attr == "sign" for call in local_calls):
            sign_generators.add(function.name)
        if any(isinstance(call.func, ast.Attribute) and call.func.attr == "verify" for call in local_calls):
            verify_generators.add(function.name)
    return bool(sign_generators and verify_generators and sign_generators.isdisjoint(verify_generators))


def _python_callable_parameter_failure(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name, function in functions.items():
        parameters = tuple(function.args.posonlyargs) + tuple(function.args.args)
        positions = {parameter.arg: index for index, parameter in enumerate(parameters)}
        called_positions = {
            positions[call.func.id]
            for call in _python_calls(function)
            if _containing_function(call, parents) is function
            and isinstance(call.func, ast.Name)
            and call.func.id in positions
        }
        if not called_positions:
            continue
        for call in _python_calls(tree):
            if not isinstance(call.func, ast.Name) or call.func.id != name:
                continue
            scope = _containing_function(call, parents)
            assignments = _python_assignments_in_scope(tree, parents, scope)
            for position in called_positions:
                if position >= len(call.args):
                    continue
                argument = call.args[position]
                if isinstance(argument, ast.Name):
                    argument = assignments.get(argument.id, argument)
                if isinstance(argument, ast.Call) and _is_hash_expression(argument):
                    return f"{name} calls a hash-algorithm object as though it were a constructor"
    return None


def _python_undefined_module_receiver(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
    operation: str,
) -> str | None:
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if _containing_function(node, parents) is not None:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                symbols.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            symbols.update(_python_target_names(targets))
    for call in _python_calls(tree):
        if _containing_function(call, parents) is not None:
            continue
        if not isinstance(call.func, ast.Attribute) or call.func.attr != operation:
            continue
        if isinstance(call.func.value, ast.Name) and call.func.value.id not in symbols:
            return call.func.value.id
    return None


def _python_unavailable_local_key_receiver(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    """Find a bare key receiver not bound in its function or module scope."""

    module_symbols: set[str] = set()
    for node in ast.walk(tree):
        if _containing_function(node, parents) is not None:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_symbols.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            module_symbols.update(
                alias.asname or alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            module_symbols.update(_python_target_names(targets))

    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        local_symbols = {
            argument.arg
            for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
        }
        for node in ast.walk(function):
            if _containing_function(node, parents) is not function:
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                local_symbols.update(_python_target_names(targets))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                local_symbols.update(
                    alias.asname or alias.name.split(".", 1)[0] for alias in node.names
                )
        for call in _python_calls(function):
            if _containing_function(call, parents) is not function:
                continue
            if not isinstance(call.func, ast.Attribute) or call.func.attr not in {
                "public_key",
                "sign",
                "verify",
            }:
                continue
            receiver = call.func.value
            if (
                isinstance(receiver, ast.Name)
                and receiver.id not in local_symbols
                and receiver.id not in module_symbols
            ):
                return receiver.id
    return None


def _python_assignments_in_scope(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
    scope: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> dict[str, ast.AST]:
    values: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or _containing_function(node, parents) is not scope:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for name in _python_target_names(targets):
            values[name] = node.value
    return values


def _python_target_names(targets: tuple[ast.AST, ...] | list[ast.AST]) -> set[str]:
    names: set[str] = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.update(
                item.id for item in target.elts if isinstance(item, ast.Name)
            )
    return names


def _python_parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


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


def _python_calls(tree: ast.AST) -> tuple[ast.Call, ...]:
    return tuple(node for node in ast.walk(tree) if isinstance(node, ast.Call))


def _python_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _python_name(node.value)
        return f"{head}.{node.attr}" if head else node.attr
    return ""


def _is_hash_expression(node: ast.AST) -> bool:
    name = _python_name(node.func) if isinstance(node, ast.Call) else _python_name(node)
    return bool(
        re.search(r"(?:^|\.)hashes\.[A-Za-z0-9_]+$", name)
        or name.endswith("utils.Prehashed")
    )


def _is_padding_expression(node: ast.AST) -> bool:
    name = _python_name(node.func) if isinstance(node, ast.Call) else _python_name(node)
    return bool(re.search(r"(?:^|\.)padding\.(?:PSS|PKCS1v15)$", name))


__all__ = (
    "csharp_signature_argument_failure",
    "signature_completion_failure",
)
