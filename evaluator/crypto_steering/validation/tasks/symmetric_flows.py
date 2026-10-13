"""Static connection checks for symmetric encryption tasks.

The ordinary operation validator establishes that a supported cipher is
created and completed.  This module checks only definite source-level breaks
between those operations: unrelated keys or IVs, a missing authenticated-mode
tag, or a transformed representation that cannot be the produced ciphertext.
It does not predict runtime values such as padding length or key strength.
"""

from __future__ import annotations

import ast
import re

from crypto_steering.validation.java.static_analysis import (
    extract_assignments,
    simple_call_name,
)
from crypto_steering.validation.java.source import iter_methods
from crypto_steering.validation.models import Language
from crypto_steering.validation.source.index import analyzer_for


def symmetric_flow_failure(
    language: Language,
    source: str,
    task_id: str,
) -> str | None:
    """Return a definite break in a required symmetric-cipher dataflow."""

    if language == "python":
        return _python_failure(source, task_id)
    if language == "typescript":
        return _typescript_failure(source, task_id)
    if language == "csharp":
        return _csharp_failure(source, task_id)
    if language == "java":
        return _java_failure(source, task_id)
    return None


def _python_failure(source: str, task_id: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    assignments = _python_assignments(tree)
    parents = _python_parents(tree)

    static_failure = _python_static_contract_failure(tree, source, task_id, parents)
    if static_failure is not None:
        return static_failure

    if task_id.endswith("T1"):
        callsite_failure = _python_callsite_material_failure(tree, parents)
        if callsite_failure is not None:
            return callsite_failure

    if task_id.endswith("T1") and _python_directions_call_fresh_key_method(
        tree, parents
    ):
        return "encryption and decryption request unrelated fresh keys"

    for call in _python_calls(tree, "Cipher"):
        if len(call.args) >= 2 and isinstance(call.args[1], ast.Attribute):
            if _python_name(call.args[1]) in {"modes.CBC", "modes.GCM"}:
                return "Cipher requires a constructed mode object, not the mode class"

    if not task_id.endswith("T1"):
        return None
    gcm_calls = _python_calls(tree, "GCM")
    has_encrypt = any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"encrypt", "encryptor"}
        for call in ast.walk(tree)
    )
    has_decrypt = any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"decrypt", "decryptor"}
        for call in ast.walk(tree)
    )
    if gcm_calls and has_encrypt and has_decrypt:
        decrypt_gcm = [
            call
            for call in gcm_calls
            if _python_call_belongs_to_direction(call, parents, "decrypt")
        ]
        if decrypt_gcm and all(
            _python_gcm_tag(call) is None
            for call in decrypt_gcm
        ) and not _python_calls(tree, "finalize_with_tag"):
            return "GCM decryption does not receive the encryption authentication tag"

    if re.search(r"\bPBKDF2HMAC\s*\(", source):
        for call in _python_calls(tree, "PBKDF2HMAC"):
            if len(call.args) < 2 and _python_keyword(call, "length") is None:
                return "PBKDF2HMAC omits the output length required for cipher key material"

    if _python_one_shot_directions_use_fresh_keys(tree, parents):
        return "encryption and decryption instantiate one-shot ciphers with unrelated fresh keys"
    if (
        _python_padded_encryption_has_no_unpadding(tree)
        and not _python_has_complete_one_shot_round_trip(tree, parents)
    ):
        return "decryption returns padded bytes without completing the matching unpadding operation"
    return None


def _python_static_contract_failure(
    tree: ast.Module,
    source: str,
    task_id: str,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    """Check source-provable API and round-trip failures missed by parsing."""

    unpacked = _python_return_unpack_failure(tree, parents)
    if unpacked is not None:
        return unpacked

    if _python_kdf_reads_local_before_assignment(tree):
        return "PBKDF2 derivation reads local key material before it is assigned"

    if _python_uses_pycryptodome_key_methods_on_pyca_rsa(tree):
        return "PyCA RSA keys expose public_key(), not publickey() or exportKey()"

    if _python_uses_rsa_padding_as_key_exchange_algorithm(tree):
        return "asymmetric key exchange does not accept RSA OAEP padding"

    if _python_calls_encode_on_definite_bytes(tree, parents):
        return "bytes-like key material does not expose the text encode operation"

    if any(
        isinstance(node, (ast.With, ast.AsyncWith))
        and any(
            isinstance(item.context_expr, ast.Call)
            and _python_name(item.context_expr.func).rsplit(".", 1)[-1]
            in {"encryptor", "decryptor"}
            for item in node.items
        )
        for node in ast.walk(tree)
    ):
        return "PyCA cipher contexts do not implement the context-manager protocol"

    cipher_names = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and _python_name(node.value.func).rsplit(".", 1)[-1] == "Cipher"
        for name in _python_target_names(
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        name = _python_name(node)
        if name.endswith((".parameters.iv", ".context.iv")):
            return "the cipher result reads an IV attribute that PyCA does not expose"
        if name.endswith(".padding.ANSI_X923"):
            return "PyCA provides padding.ANSIX923, not cipher.padding.ANSI_X923"
        if (
            node.attr in {"nonce", "tag"}
            and isinstance(node.value, ast.Name)
            and node.value.id in cipher_names
            # A bad optional decryption route cannot invalidate an already
            # completed encryption route under the family-level contract.
            and task_id.endswith("T3")
        ):
            return f"PyCA Cipher objects do not expose a {node.attr} attribute"
        if name == "algorithms.ECDH":
            return "ECDH is an asymmetric operation and is not provided by ciphers.algorithms"
        if name == "utils.quote_private":
            return "cryptography asymmetric utils has no quote_private operation"
        if (
            node.attr == "key"
            and isinstance(node.value, ast.Call)
            and re.search(
                r"(?:^|\.)hashlib\.(?:md5|sha1|sha224|sha256|sha384|sha512)$",
                _python_name(node.value.func),
                re.IGNORECASE,
            )
        ):
            return "hashlib digest objects do not expose cipher key material"

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        simple_name = _python_name(call.func).rsplit(".", 1)[-1]
        if simple_name == "CBC" and (
            len(call.args) != 1
            or any(
                keyword.arg not in {"initialization_vector"}
                for keyword in call.keywords
            )
            or (
                call.args
                and any(
                    keyword.arg == "initialization_vector"
                    for keyword in call.keywords
                )
            )
        ) and (
            task_id.startswith("F2_")
            or not _python_has_non_cbc_cipher_route(tree)
        ):
            return "modes.CBC accepts exactly one initialization-vector argument"
        if simple_name == "AES" and any(
            keyword.arg == "backend" for keyword in call.keywords
        ):
            return "algorithms.AES does not accept a backend argument"
        if simple_name == "AES" and (
            len(call.args) > 1
            or any(keyword.arg not in {"key"} for keyword in call.keywords)
            or (call.args and any(keyword.arg == "key" for keyword in call.keywords))
        ):
            return "algorithms.AES accepts exactly one key argument"

    imported_or_bound = {
        alias.asname or alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    imported_or_bound.update(
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.ClassDef))
        for name in (
            _python_target_names(
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            else {node.name}
        )
    )
    if "hashlib" not in imported_or_bound and any(
        isinstance(node, ast.Attribute)
        and _python_name(node).startswith("hashlib.")
        for node in ast.walk(tree)
    ):
        return "the required hashlib key-derivation route is not imported or defined"

    wrong_material = _python_definite_cipher_material_type_failure(tree, parents)
    if wrong_material is not None:
        return wrong_material

    if _python_directional_aes_key_transform_failure(tree, parents):
        return "encryption and decryption transform the AES key differently"

    if _python_encrypts_padding_context_instead_of_bytes(tree):
        return "the cipher receives a padding context instead of padded bytes"

    # PyCA padding contexts are streaming transforms: ``update`` returns the
    # bytes produced for that input and ``finalize`` returns only the tail.
    # Discarding the update result therefore disconnects the payload from the
    # cipher even when a later finalize call is present.
    if _python_discards_padding_update_output(tree):
        return "the encryption route discards bytes returned by the padding update"

    invalid_padding_member = _python_invalid_padding_context_member(tree)
    if invalid_padding_member is not None:
        return (
            "PyCA padding contexts do not expose a completed-data attribute "
            f"named {invalid_padding_member}"
        )

    fixed_block_failure = _python_fixed_block_input_failure(source)
    if fixed_block_failure is not None:
        return fixed_block_failure

    has_one_shot_round_trip = _python_has_complete_one_shot_round_trip(tree, parents)
    if not has_one_shot_round_trip:
        if _python_uses_unavailable_aes_gcm_algorithm(tree):
            return "cryptography provides AESGCM as a one-shot cipher, not algorithms.AES_GCM"
        if _python_uses_aad_on_non_gcm_cipher_context(tree):
            return "authenticate_additional_data is used on a non-GCM cipher context"

    if task_id.endswith("T1"):
        if _python_shared_gcm_decrypts_without_tag(tree):
            return "GCM decryption does not receive the encryption authentication tag"
        if _python_decrypts_base64_literal_instead_of_encryption_result(tree, parents):
            return "decryption uses an unrelated literal instead of the encryption result"
        if not has_one_shot_round_trip:
            if _python_zero_padding_is_removed_as_one_byte(tree):
                return "zero-padded plaintext is truncated by one byte instead of removing the padding"
            if _python_unpads_a_slice_using_padding_context_block_size(tree):
                return "unpadding slices plaintext through an unavailable padding-context block_size"
        slice_failure = _python_envelope_slice_failure(tree)
        if slice_failure is not None:
            return slice_failure
        if _python_mutates_ciphertext_before_decryption(tree):
            return "the produced ciphertext is modified before decryption"
        if _python_discards_cipher_update_output(tree):
            return "the encryption route discards bytes returned by cipher update"
    return None


def _python_kdf_reads_local_before_assignment(tree: ast.Module) -> bool:
    """Find ``key = kdf.derive(key)``-style unbound local key material."""

    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        parameters = {
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        }
        prior: set[str] = set(parameters)
        statements = sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
            ),
            key=lambda node: getattr(node, "lineno", 0),
        )
        for statement in statements:
            if statement.value is None:
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
            names = _python_target_names(targets)
            derives = tuple(
                call
                for call in ast.walk(statement.value)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {"derive", "verify"}
            )
            if derives and any(
                name not in prior
                and any(
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id == name
                    for call in derives
                    for node in ast.walk(call)
                )
                for name in names
            ):
                return True
            prior.update(names)
    return False


def _python_uses_pycryptodome_key_methods_on_pyca_rsa(tree: ast.Module) -> bool:
    """Recognize incompatible key-object APIs on a PyCA RSA route."""

    imports_pyca_rsa = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "cryptography.hazmat.primitives.asymmetric"
        and any(alias.name == "rsa" for alias in node.names)
        for node in ast.walk(tree)
    )
    if not imports_pyca_rsa:
        return False
    return any(
        isinstance(node, ast.Attribute)
        and node.attr in {"publickey", "exportKey"}
        for node in ast.walk(tree)
    )


def _python_uses_rsa_padding_as_key_exchange_algorithm(tree: ast.Module) -> bool:
    """Return whether ``exchange`` receives RSA encryption padding.

    PyCA key-agreement APIs accept a key-agreement algorithm (for example,
    ``ec.ECDH``), while ``OAEP`` is padding for RSA encryption/decryption.
    This is an API-signature failure independent of runtime key values.
    """

    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "exchange"
        and call.args
        and isinstance(call.args[0], ast.Call)
        and _python_name(call.args[0].func).rsplit(".", 1)[-1] == "OAEP"
        for call in ast.walk(tree)
    )


def _python_calls_encode_on_definite_bytes(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Find ``bytes.encode(...)`` when the receiver type is source-provable."""

    assignments: dict[
        tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, str],
        list[tuple[ast.AST, ast.AST]],
    ] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        scope = _python_containing_function(node, parents)
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for name in _python_target_names(targets):
            assignments.setdefault((scope, name), []).append((node, node.value))

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    byte_producers = {
        "b64decode",
        "b64encode",
        "digest",
        "derive",
        "get_random_bytes",
        "private_bytes",
        "public_bytes",
        "token_bytes",
        "urandom",
    }

    def latest_assignment(
        name: str,
        before: ast.AST,
    ) -> tuple[ast.AST, ast.AST] | None:
        scope = _python_containing_function(before, parents)
        candidates = [
            item
            for item in assignments.get((scope, name), ())
            if getattr(item[0], "lineno", 0) < getattr(before, "lineno", 10**9)
        ]
        if not candidates and scope is not None:
            candidates = [
                item
                for item in assignments.get((None, name), ())
                if getattr(item[0], "lineno", 0) < getattr(before, "lineno", 10**9)
            ]
        return (
            max(candidates, key=lambda item: getattr(item[0], "lineno", 0))
            if candidates
            else None
        )

    def is_bytes(
        expression: ast.AST,
        before: ast.AST,
        seen: frozenset[tuple[str, str]],
    ) -> bool:
        if isinstance(expression, ast.Constant):
            return isinstance(expression.value, bytes)
        if isinstance(expression, ast.Name):
            marker = (
                getattr(_python_containing_function(before, parents), "name", "<module>"),
                expression.id,
            )
            if marker in seen:
                return False
            assigned = latest_assignment(expression.id, before)
            return bool(
                assigned
                and is_bytes(assigned[1], assigned[0], seen | {marker})
            )
        if isinstance(expression, ast.Call):
            simple = _python_name(expression.func).rsplit(".", 1)[-1]
            if simple in byte_producers:
                return True
            function = functions.get(simple)
            if function is None:
                return False
            returns = [
                node.value
                for node in ast.walk(function)
                if isinstance(node, ast.Return) and node.value is not None
                and _python_containing_function(node, parents) is function
            ]
            return bool(returns) and all(
                is_bytes(value, value, seen | {(function.name, "<return>")})
                for value in returns
            )
        if isinstance(expression, ast.Subscript):
            index = expression.slice
            if not (
                isinstance(index, ast.Constant)
                and isinstance(index.value, int)
                and isinstance(expression.value, ast.Call)
            ):
                return is_bytes(expression.value, before, seen)
            function = functions.get(
                _python_name(expression.value.func).rsplit(".", 1)[-1]
            )
            if function is None:
                return False
            elements: list[ast.AST] = []
            for returned in (
                node.value
                for node in ast.walk(function)
                if isinstance(node, ast.Return) and node.value is not None
                and _python_containing_function(node, parents) is function
            ):
                if not isinstance(returned, (ast.Tuple, ast.List)):
                    return False
                position = index.value
                if position < 0:
                    position += len(returned.elts)
                if position < 0 or position >= len(returned.elts):
                    return False
                elements.append(returned.elts[position])
            return bool(elements) and all(
                is_bytes(value, value, seen | {(function.name, str(index.value))})
                for value in elements
            )
        return False

    invalid_encodes = {
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "encode"
        and is_bytes(call.func.value, call, frozenset())
    }
    if not invalid_encodes:
        return False

    tainted: set[
        tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, str]
    ] = set()
    assignment_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and node.value is not None
    ]
    for node in assignment_nodes:
        if not any(call in invalid_encodes for call in ast.walk(node.value)):
            continue
        scope = _python_containing_function(node, parents)
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        tainted.update((scope, name) for name in _python_target_names(targets))

    changed = True
    while changed:
        changed = False
        for node in assignment_nodes:
            scope = _python_containing_function(node, parents)
            referenced = {
                (scope, item.id)
                for item in ast.walk(node.value)
                if isinstance(item, ast.Name)
            }
            if not referenced & tainted:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            additions = {
                (scope, name) for name in _python_target_names(targets)
            } - tainted
            if additions:
                tainted.update(additions)
                changed = True

    def receives_tainted(expression: ast.AST, before: ast.AST) -> bool:
        scope = _python_containing_function(before, parents)
        return any(
            (scope, node.id) in tainted or (None, node.id) in tainted
            for node in ast.walk(expression)
            if isinstance(node, ast.Name)
        )

    if any(
        isinstance(call, ast.Call)
        and _python_name(call.func).rsplit(".", 1)[-1] == "AES"
        and call.args
        and receives_tainted(call.args[0], call)
        for call in ast.walk(tree)
    ):
        return True

    class_definitions = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    for class_name, class_node in class_definitions.items():
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
            continue
        parameters = (*initializer.args.posonlyargs, *initializer.args.args)
        offset = int(bool(parameters) and parameters[0].arg in {"self", "cls"})
        positions = {
            parameter.arg: index - offset
            for index, parameter in enumerate(parameters)
            if index >= offset
        }
        aes_attributes = {
            call.args[0].attr
            for call in ast.walk(class_node)
            if isinstance(call, ast.Call)
            and _python_name(call.func).rsplit(".", 1)[-1] == "AES"
            and call.args
            and isinstance(call.args[0], ast.Attribute)
            and isinstance(call.args[0].value, ast.Name)
            and call.args[0].value.id == "self"
        }
        material_positions = {
            positions[value.id]
            for node in ast.walk(initializer)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            for value in (node.value,)
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr in aes_attributes
            and isinstance(value, ast.Name)
            and value.id in positions
        }
        if any(
            isinstance(call, ast.Call)
            and _python_name(call.func).rsplit(".", 1)[-1] == class_name
            and any(
                position < len(call.args)
                and receives_tainted(call.args[position], call)
                for position in material_positions
            )
            for call in ast.walk(tree)
        ):
            return True
    return False


def _python_callsite_material_failure(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    """Find explicit round trips whose callsite changes cipher material."""

    assignments = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ),
        key=lambda node: getattr(node, "lineno", 0),
    )
    object_material: dict[str, tuple[str, ...]] = {}
    encrypted_results: list[tuple[str, str, int]] = []
    for statement in assignments:
        targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
        names = tuple(_python_target_names(targets))
        value = statement.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            arguments = tuple(
                argument.id for argument in value.args if isinstance(argument, ast.Name)
            )
            if arguments:
                for name in names:
                    object_material[name] = arguments
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and "encrypt" in value.func.attr.lower()
            and isinstance(value.func.value, ast.Name)
        ):
            for name in names:
                encrypted_results.append(
                    (name, value.func.value.id, getattr(statement, "lineno", 0))
                )

    for call in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and "decrypt" in node.func.attr.lower()
        and isinstance(node.func.value, ast.Name)
    ):
        call_names = {
            node.id for argument in call.args for node in ast.walk(argument)
            if isinstance(node, ast.Name)
        }
        for result, encrypt_receiver, _ in encrypted_results:
            if result not in call_names:
                continue
            decrypt_receiver = call.func.value.id
            encrypt_material = set(object_material.get(encrypt_receiver, ()))
            decrypt_material = set(object_material.get(decrypt_receiver, ()))
            if (
                encrypt_receiver != decrypt_receiver
                and encrypt_material
                and decrypt_material
                and encrypt_material.isdisjoint(decrypt_material)
            ):
                return "encryption and decryption objects are initialized from different key material"

    directional_calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and "encrypt" in _python_name(node.func).rsplit(".", 1)[-1].lower()
        ),
        key=lambda node: getattr(node, "lineno", 0),
    )
    decrypt_calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and "decrypt" in _python_name(node.func).rsplit(".", 1)[-1].lower()
        ),
        key=lambda node: getattr(node, "lineno", 0),
    )
    for encrypt in directional_calls:
        owner = _python_containing_function(encrypt, parents)
        encrypt_names = {
            node.id for argument in encrypt.args for node in ast.walk(argument)
            if isinstance(node, ast.Name) and "key" in node.id.lower()
        }
        for decrypt in decrypt_calls:
            if getattr(decrypt, "lineno", 0) <= getattr(encrypt, "lineno", 0):
                continue
            if _python_containing_function(decrypt, parents) is not owner:
                continue
            shared = encrypt_names & {
                node.id for argument in decrypt.args for node in ast.walk(argument)
                if isinstance(node, ast.Name)
            }
            if any(
                name in _python_target_names(
                    statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
                )
                and getattr(encrypt, "lineno", 0)
                < getattr(statement, "lineno", 0)
                < getattr(decrypt, "lineno", 0)
                and _python_containing_function(statement, parents) is owner
                and not _python_restores_serialized_material(
                    name,
                    statement,
                    assignments,
                    parents,
                )
                for name in shared
                for statement in assignments
            ):
                return "the encryption key variable is replaced before decryption"
    return None


def _python_restores_serialized_material(
    name: str,
    statement: ast.Assign | ast.AnnAssign,
    assignments: list[ast.Assign | ast.AnnAssign],
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Recognize an encode/decode round trip of the same material.

    Serializing a key for display and then decoding that serialization does
    not replace the key used by the round trip.  This is deliberately narrow:
    the decoder must consume either the original name directly or an alias
    whose preceding definition encodes that same name.
    """

    value = statement.value
    if value is None:
        # A bare annotation declares a name without assigning a value. It is
        # not a serialization origin and must not be passed to ``ast.walk``.
        return False
    decoder = next(
        (
            call
            for call in ast.walk(value)
            if isinstance(call, ast.Call)
            and _python_name(call.func).rsplit(".", 1)[-1]
            in {"b64decode", "urlsafe_b64decode", "unhexlify", "fromhex"}
            and call.args
        ),
        None,
    )
    if decoder is None:
        return False
    if any(
        isinstance(node, ast.Name) and node.id == name
        for node in ast.walk(decoder.args[0])
    ):
        return True

    owner = _python_containing_function(statement, parents)
    boundary = getattr(statement, "lineno", 0)
    aliases = {
        node.id
        for node in ast.walk(decoder.args[0])
        if isinstance(node, ast.Name)
    }
    for alias in aliases:
        candidates = [
            candidate
            for candidate in assignments
            if candidate.value is not None
            and alias
            in _python_target_names(
                candidate.targets
                if isinstance(candidate, ast.Assign)
                else (candidate.target,)
            )
            and getattr(candidate, "lineno", 0) < boundary
            and _python_containing_function(candidate, parents) is owner
        ]
        if not candidates:
            continue
        origin = max(candidates, key=lambda item: getattr(item, "lineno", 0)).value
        encodes_same_name = any(
            isinstance(call, ast.Call)
            and _python_name(call.func).rsplit(".", 1)[-1]
            in {"b64encode", "urlsafe_b64encode", "hexlify", "hex"}
            and any(
                isinstance(node, ast.Name) and node.id == name
                for argument in call.args
                for node in ast.walk(argument)
            )
            for call in ast.walk(origin)
        )
        if encodes_same_name:
            return True
    return False


def _python_return_unpack_failure(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    """Find a definite fixed-tuple return unpacked with the wrong arity."""

    return_arities: dict[str, int] = {}
    for function in (
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        returns = [
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Return) and node.value is not None
        ]
        arities = {
            len(value.elts)
            for value in returns
            if isinstance(value, (ast.Tuple, ast.List))
        }
        if len(returns) == len(arities) == 1:
            return_arities[function.name] = next(iter(arities))

    assigned_arities: dict[tuple[ast.AST | None, str], int] = {}
    statements = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ),
        key=lambda node: node.lineno,
    )
    for statement in statements:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        scope = _python_containing_function(statement, parents)
        value = statement.value
        targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in return_arities
        ):
            for name in _python_target_names(targets):
                assigned_arities[(scope, name)] = return_arities[value.func.id]
            continue
        for target in targets:
            if not isinstance(target, (ast.Tuple, ast.List)):
                continue
            if not isinstance(value, ast.Name) or (
                scope,
                value.id,
            ) not in assigned_arities:
                continue
            arity = assigned_arities[(scope, value.id)]
            if len(target.elts) != arity:
                return (
                    f"the {value.id} result has {arity} values "
                    f"but is unpacked into {len(target.elts)} targets"
                )
    return None


def _python_definite_cipher_material_type_failure(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    assignments: dict[str, list[tuple[ast.AST, ast.AST]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for name in _python_target_names(targets):
            assignments.setdefault(name, []).append((node, node.value))

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    class_definitions = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    asymmetric_object_names = {
        (_python_containing_function(node, parents), node.func.value.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.attr
        in {
            "private_bytes",
            "private_numbers",
            "public_bytes",
            "public_key",
            "public_numbers",
        }
    }

    def definitely_text(expression: ast.AST, before: ast.AST, seen: set[str]) -> bool:
        if isinstance(expression, ast.Constant):
            return isinstance(expression.value, str)
        if isinstance(expression, ast.BinOp) and isinstance(
            expression.op, (ast.Add, ast.Mult)
        ):
            return definitely_text(expression.left, before, seen) or definitely_text(
                expression.right, before, seen
            )
        if isinstance(expression, ast.Call) and _python_name(expression.func).rsplit(
            ".", 1
        )[-1] in {"input", "getpass"}:
            return True
        if isinstance(expression, ast.Name) and expression.id not in seen:
            scope = _python_containing_function(before, parents)
            candidates = [
                (statement, value)
                for statement, value in assignments.get(expression.id, ())
                if getattr(statement, "lineno", 0) < getattr(before, "lineno", 10**9)
                and _python_containing_function(statement, parents) is scope
            ]
            if not candidates and scope is not None:
                candidates = [
                    (statement, value)
                    for statement, value in assignments.get(expression.id, ())
                    if _python_containing_function(statement, parents) is None
                ]
            if candidates:
                statement, value = max(
                    candidates, key=lambda item: getattr(item[0], "lineno", 0)
                )
                return definitely_text(value, statement, seen | {expression.id})
        return False

    def definitely_falsey(
        expression: ast.AST,
        before: ast.AST,
        seen: set[str],
    ) -> bool:
        if isinstance(expression, ast.Constant):
            return not expression.value
        if isinstance(expression, ast.Name) and expression.id not in seen:
            scope = _python_containing_function(before, parents)
            candidates = [
                (statement, value)
                for statement, value in assignments.get(expression.id, ())
                if getattr(statement, "lineno", 0)
                < getattr(before, "lineno", 10**9)
                and _python_containing_function(statement, parents) is scope
            ]
            if not candidates and scope is not None:
                candidates = [
                    (statement, value)
                    for statement, value in assignments.get(expression.id, ())
                    if _python_containing_function(statement, parents) is None
                ]
            if candidates:
                statement, value = max(
                    candidates, key=lambda item: getattr(item[0], "lineno", 0)
                )
                return definitely_falsey(
                    value, statement, seen | {expression.id}
                )
        return False

    def definitely_non_bytes(
        expression: ast.AST,
        before: ast.AST,
        seen: set[str],
    ) -> bool:
        if definitely_text(expression, before, seen):
            return True
        if isinstance(expression, ast.Constant):
            return expression.value is None or (
                isinstance(expression.value, (int, float, complex))
                and not isinstance(expression.value, bool)
            )
        if isinstance(expression, ast.BinOp):
            return definitely_non_bytes(
                expression.left, before, seen
            ) and definitely_non_bytes(expression.right, before, seen)
        if isinstance(expression, ast.Call) and _python_name(
            expression.func
        ).rsplit(".", 1)[-1] in {
            "generate_private_key",
            "load_pem_private_key",
            "load_ssh_public_key",
            "public_key",
        }:
            return True
        if isinstance(expression, ast.Name) and (
            _python_containing_function(before, parents), expression.id
        ) in asymmetric_object_names:
            return True
        if isinstance(expression, ast.Name) and expression.id not in seen:
            scope = _python_containing_function(before, parents)
            candidates = [
                (statement, value)
                for statement, value in assignments.get(expression.id, ())
                if getattr(statement, "lineno", 0)
                < getattr(before, "lineno", 10**9)
                and _python_containing_function(statement, parents) is scope
            ]
            if not candidates and scope is not None:
                candidates = [
                    (statement, value)
                    for statement, value in assignments.get(expression.id, ())
                    if _python_containing_function(statement, parents) is None
                ]
            if candidates:
                statement, value = max(
                    candidates, key=lambda item: getattr(item[0], "lineno", 0)
                )
                return definitely_non_bytes(
                    value, statement, seen | {expression.id}
                )
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id not in seen
        ):
            object_name = expression.value.id
            scope = _python_containing_function(before, parents)
            candidates = [
                (statement, value)
                for statement, value in assignments.get(object_name, ())
                if getattr(statement, "lineno", 0)
                < getattr(before, "lineno", 10**9)
                and _python_containing_function(statement, parents) is scope
            ]
            if candidates:
                statement, value = max(
                    candidates, key=lambda item: getattr(item[0], "lineno", 0)
                )
                if isinstance(value, ast.Call):
                    class_node = class_definitions.get(
                        _python_name(value.func).rsplit(".", 1)[-1]
                    )
                    initializer = next(
                        (
                            node
                            for node in class_node.body
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name == "__init__"
                        ),
                        None,
                    ) if class_node is not None else None
                    if initializer is not None:
                        parameters = (
                            *initializer.args.posonlyargs,
                            *initializer.args.args,
                        )
                        offset = int(
                            bool(parameters)
                            and parameters[0].arg in {"self", "cls"}
                        )
                        parameter = next(
                            (
                                assigned.id
                                for node in ast.walk(initializer)
                                if isinstance(node, (ast.Assign, ast.AnnAssign))
                                for target in (
                                    node.targets
                                    if isinstance(node, ast.Assign)
                                    else (node.target,)
                                )
                                for assigned in (node.value,)
                                if isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                                and target.attr == expression.attr
                                and isinstance(assigned, ast.Name)
                            ),
                            None,
                        )
                        names = [parameter.arg for parameter in parameters[offset:]]
                        if parameter in names:
                            position = names.index(parameter)
                            if position < len(value.args):
                                return definitely_non_bytes(
                                    value.args[position],
                                    statement,
                                    seen | {object_name},
                                )
        return False

    def definitely_bytes(expression: ast.AST, before: ast.AST, seen: set[str]) -> bool:
        if isinstance(expression, ast.Constant):
            return isinstance(expression.value, bytes)
        if isinstance(expression, ast.Call):
            simple = _python_name(expression.func).rsplit(".", 1)[-1]
            return simple in {
                "bytes",
                "bytearray",
                "digest",
                "encode",
                "pbkdf2_hmac",
                "urandom",
                "token_bytes",
                "get_random_bytes",
                "b64decode",
                "urlsafe_b64decode",
            }
        if isinstance(expression, ast.Name) and expression.id not in seen:
            scope = _python_containing_function(before, parents)
            candidates = [
                (statement, value)
                for statement, value in assignments.get(expression.id, ())
                if getattr(statement, "lineno", 0) < getattr(before, "lineno", 10**9)
                and _python_containing_function(statement, parents) is scope
            ]
            if candidates:
                statement, value = max(
                    candidates, key=lambda item: getattr(item[0], "lineno", 0)
                )
                return definitely_bytes(value, statement, seen | {expression.id})
        return False

    def material_failure(call: ast.Call, expression: ast.AST, role: str) -> str | None:
        if definitely_non_bytes(expression, call, set()):
            return f"{role} requires bytes-like material rather than text"
        if definitely_bytes(expression, call, set()):
            return None
        if not isinstance(expression, ast.Name):
            return None
        owner = _python_containing_function(call, parents)
        if owner is None:
            return None
        parameters = [
            argument.arg
            for argument in (*owner.args.posonlyargs, *owner.args.args)
        ]
        if expression.id not in parameters:
            return None
        position = parameters.index(expression.id)
        callsites = [
            candidate
            for candidate in ast.walk(tree)
            if isinstance(candidate, ast.Call)
            and _python_name(candidate.func).rsplit(".", 1)[-1] == owner.name
            and candidate is not call
        ]
        supplied = [
            candidate.args[position]
            for candidate in callsites
            if position < len(candidate.args)
        ]
        has_binary_fallback = any(
            getattr(statement, "lineno", 0) < getattr(call, "lineno", 10**9)
            and _python_containing_function(statement, parents) is owner
            and expression.id in _python_target_names(
                statement.targets
                if isinstance(statement, ast.Assign)
                else (statement.target,)
            )
            and isinstance(statement.value, ast.BoolOp)
            and isinstance(statement.value.op, ast.Or)
            and any(
                isinstance(value, ast.Name) and value.id == expression.id
                for value in statement.value.values
            )
            and any(
                definitely_bytes(value, statement, set())
                for value in statement.value.values
            )
            for statement in (
                node
                for node in ast.walk(owner)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
            )
        )
        if has_binary_fallback and supplied and all(
            definitely_falsey(value, candidate, set())
            for candidate, value in (
                (candidate, candidate.args[position])
                for candidate in callsites
                if position < len(candidate.args)
            )
        ):
            return None
        if supplied and all(definitely_non_bytes(value, candidate, set()) for candidate, value in (
            (candidate, candidate.args[position])
            for candidate in callsites
            if position < len(candidate.args)
        )):
            return f"{role} receives non-bytes material on every connected call route"
        return None

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        simple = _python_name(call.func).rsplit(".", 1)[-1]
        expression: ast.AST | None = None
        role = ""
        if simple == "AES" and call.args:
            expression, role = call.args[0], "AES key"
        elif simple == "CBC" and call.args:
            expression, role = call.args[0], "CBC IV"
        if expression is not None:
            failure = material_failure(call, expression, role)
            if failure is not None:
                return failure

    for class_name, class_node in class_definitions.items():
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
            continue
        parameters = tuple(initializer.args.posonlyargs) + tuple(initializer.args.args)
        receiver_offset = int(bool(parameters) and parameters[0].arg in {"self", "cls"})
        positions = {
            parameter.arg: index - receiver_offset
            for index, parameter in enumerate(parameters)
            if index >= receiver_offset
        }
        attribute_parameters = {
            target.attr: positions[value.id]
            for node in ast.walk(initializer)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
            for value in (node.value,)
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and isinstance(value, ast.Name)
            and value.id in positions
        }
        required_attributes = {
            node.args[0].attr: (
                "AES key"
                if _python_name(node.func).rsplit(".", 1)[-1] == "AES"
                else "CBC IV"
            )
            for node in ast.walk(class_node)
            if isinstance(node, ast.Call)
            and _python_name(node.func).rsplit(".", 1)[-1] in {"AES", "CBC"}
            and node.args
            and isinstance(node.args[0], ast.Attribute)
            and isinstance(node.args[0].value, ast.Name)
            and node.args[0].value.id == "self"
        }
        for attribute, role in required_attributes.items():
            position = attribute_parameters.get(attribute)
            if position is None:
                continue
            supplied = [
                (node, node.args[position])
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and _python_name(node.func).rsplit(".", 1)[-1] == class_name
                and len(node.args) > position
            ]
            if supplied and all(
                definitely_non_bytes(value, call, set())
                for call, value in supplied
            ):
                return f"{role} receives non-bytes material on every constructor route"
    return None


def _python_directional_aes_key_transform_failure(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Find helpers that apply opposite encodings to their AES key input."""

    def transforms(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        local_assignments = {
            name: node.value
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and _python_containing_function(node, parents) is function
            for name in _python_target_names(
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
        }

        def chain(expression: ast.AST, seen: set[str]) -> set[str]:
            if isinstance(expression, ast.Name) and expression.id not in seen:
                value = local_assignments.get(expression.id)
                return (
                    chain(value, seen | {expression.id})
                    if value is not None
                    else set()
                )
            if isinstance(expression, ast.Call):
                simple = _python_name(expression.func).rsplit(".", 1)[-1]
                nested = set().union(*(chain(arg, seen) for arg in expression.args))
                return {simple, *nested}
            return set()

        return set().union(
            *(
                chain(call.args[0], set())
                for call in ast.walk(function)
                if isinstance(call, ast.Call)
                and _python_name(call.func).rsplit(".", 1)[-1] == "AES"
                and call.args
            )
        )

    encrypt = set().union(
        *(
            transforms(function)
            for function in ast.walk(tree)
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _python_directional_helper_name(function.name, "encrypt")
        )
    )
    decrypt = set().union(
        *(
            transforms(function)
            for function in ast.walk(tree)
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _python_directional_helper_name(function.name, "decrypt")
        )
    )
    encoders = {"b64encode", "urlsafe_b64encode", "hexlify"}
    decoders = {"b64decode", "urlsafe_b64decode", "unhexlify", "fromhex"}
    return bool((encrypt & encoders and decrypt & decoders) or (
        encrypt & decoders and decrypt & encoders
    ))


def _python_directional_helper_name(name: str, direction: str) -> bool:
    lowered = name.lower()
    encrypt = "encrypt" in lowered or "seal" in lowered or "protect" in lowered
    decrypt = "decrypt" in lowered or "open" in lowered or "recover" in lowered
    return (encrypt and not decrypt) if direction == "encrypt" else (
        decrypt and not encrypt
    )


def _python_has_non_cbc_cipher_route(tree: ast.Module) -> bool:
    """Return whether a selected route constructs a usable non-CBC mode."""

    if any(
        isinstance(node, ast.Call)
        and _python_name(node.func).rsplit(".", 1)[-1] == "Cipher"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Call)
        and _python_name(node.args[1].func).rsplit(".", 1)[-1]
        in {"ECB", "CFB", "CFB8", "CTR", "GCM", "OFB", "XTS"}
        for node in ast.walk(tree)
    ):
        return True
    return bool(
        any(
            isinstance(node, ast.Call)
            and _python_name(node.func).rsplit(".", 1)[-1]
            in {"ECB", "CFB", "CFB8", "CTR", "GCM", "OFB", "XTS"}
            for node in ast.walk(tree)
        )
        and any(
            isinstance(node, ast.Call)
            and _python_name(node.func).rsplit(".", 1)[-1] == "Cipher"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Attribute)
            and isinstance(node.args[1].value, ast.Name)
            and node.args[1].value.id == "self"
            for node in ast.walk(tree)
        )
    )


def _python_encrypts_padding_context_instead_of_bytes(tree: ast.Module) -> bool:
    padder_names = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and _python_name(node.value.func).rsplit(".", 1)[-1] == "padder"
        for name in _python_target_names(
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
    }
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and "encrypt" in _python_name(node.func.value).lower()
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in padder_names
        for node in ast.walk(tree)
    )


def _python_decrypts_base64_literal_instead_of_encryption_result(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Find an explicit round trip whose decrypt call decodes a fixed literal.

    This is a source-level connection error, not a prediction about Base64 or
    cipher runtime behaviour: the program has produced an encryption result,
    but the invoked decryption route receives a separately embedded value.
    Decoding the produced ciphertext itself remains outside this check.
    """

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    decode_helpers = {
        name
        for name, function in functions.items()
        if any(
            isinstance(node, ast.Call)
            and _python_name(node.func).rsplit(".", 1)[-1]
            in {"b64decode", "urlsafe_b64decode"}
            for node in ast.walk(function)
        )
    }
    has_encryption_result = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and "encrypt" in _python_name(node.value.func).rsplit(".", 1)[-1].lower()
        for node in ast.walk(tree)
    )
    if not has_encryption_result:
        return False

    def invalid_decode(expression: ast.AST) -> bool:
        if not isinstance(expression, ast.Call):
            return False
        name = _python_name(expression.func).rsplit(".", 1)[-1]
        if name not in decode_helpers | {"b64decode", "urlsafe_b64decode"}:
            return False
        if not expression.args:
            return False
        value = expression.args[0]
        return isinstance(value, ast.Constant) and isinstance(
            value.value, (str, bytes)
        )

    return any(
        isinstance(call, ast.Call)
        and "decrypt" in _python_name(call.func).rsplit(".", 1)[-1].lower()
        and any(invalid_decode(argument) for argument in call.args)
        and _python_containing_function(call, parents) is not functions.get(
            _python_name(call.func).rsplit(".", 1)[-1]
        )
        for call in ast.walk(tree)
    )


def _python_shared_gcm_decrypts_without_tag(tree: ast.Module) -> bool:
    if any(
        isinstance(node, ast.Call)
        and _python_name(node.func).rsplit(".", 1)[-1] == "GCM"
        and (
            len(node.args) >= 2
            or any(keyword.arg == "tag" for keyword in node.keywords)
        )
        for node in ast.walk(tree)
    ):
        return False
    gcm_ciphers: set[str] = set()
    encryptors: set[str] = set()
    decryptors: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(
            node.value, ast.Call
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        names = _python_target_names(targets)
        simple = _python_name(node.value.func).rsplit(".", 1)[-1]
        if simple == "Cipher" and len(node.value.args) >= 2 and any(
            isinstance(item, ast.Call)
            and _python_name(item.func).rsplit(".", 1)[-1] == "GCM"
            and len(item.args) < 2
            and not any(keyword.arg == "tag" for keyword in item.keywords)
            for item in ast.walk(node.value.args[1])
        ):
            gcm_ciphers.update(names)
        if simple in {"encryptor", "decryptor"} and isinstance(
            node.value.func, ast.Attribute
        ) and isinstance(node.value.func.value, ast.Name):
            receiver = node.value.func.value.id
            if receiver in gcm_ciphers:
                (encryptors if simple == "encryptor" else decryptors).add(receiver)
    return bool(encryptors and decryptors) and not any(
        isinstance(node, ast.Call)
        and _python_name(node.func).rsplit(".", 1)[-1] == "finalize_with_tag"
        for node in ast.walk(tree)
    )


def _python_padded_encryption_has_no_unpadding(tree: ast.Module) -> bool:
    has_padder = any(
        isinstance(node, ast.Call)
        and _python_name(node.func).rsplit(".", 1)[-1] == "padder"
        for node in ast.walk(tree)
    )
    has_decrypt = any(
        isinstance(node, ast.Call)
        and _python_name(node.func).rsplit(".", 1)[-1] in {"decrypt", "decryptor"}
        for node in ast.walk(tree)
    )
    has_unpadder = any(
        isinstance(node, ast.Call)
        and _python_name(node.func).rsplit(".", 1)[-1] == "unpadder"
        for node in ast.walk(tree)
    )
    return has_padder and has_decrypt and not has_unpadder


def _python_zero_padding_is_removed_as_one_byte(tree: ast.Module) -> bool:
    """Find a full-block zero-padding route paired with ``plaintext[:-1]``."""

    adds_zero_block_padding = any(
        isinstance(node, ast.AugAssign)
        and isinstance(node.op, ast.Add)
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.Mult)
        and isinstance(node.value.left, ast.Constant)
        and node.value.left.value == b"\x00"
        for node in ast.walk(tree)
    )
    if not adds_zero_block_padding:
        return False
    return any(
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Slice)
        and node.slice.lower is None
        and isinstance(node.slice.upper, ast.UnaryOp)
        and isinstance(node.slice.upper.op, ast.USub)
        and isinstance(node.slice.upper.operand, ast.Constant)
        and node.slice.upper.operand.value == 1
        for node in ast.walk(tree)
    )


def _python_has_complete_one_shot_round_trip(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    """Recognize an independent AESGCM/AESCCM/Fernet encrypt/decrypt route."""

    directions: set[str] = set()
    one_shot_names = {"AESGCM", "AESCCM", "Fernet"}
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Attribute) or call.func.attr not in {
            "encrypt",
            "decrypt",
        }:
            continue
        function = _python_containing_function(call, parents)
        direction = (
            _python_direction(function.name)
            if function is not None
            else call.func.attr
        )
        if direction not in {"encrypt", "decrypt"}:
            continue
        receiver = call.func.value
        if isinstance(receiver, ast.Call) and (
            _python_name(receiver.func).rsplit(".", 1)[-1] in one_shot_names
        ):
            directions.add(direction)
            continue
        if not isinstance(receiver, ast.Name):
            continue
        for assignment in ast.walk(function if function is not None else tree):
            if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else (assignment.target,)
            )
            if receiver.id not in _python_target_names(targets):
                continue
            value = assignment.value
            if isinstance(value, ast.Call) and (
                _python_name(value.func).rsplit(".", 1)[-1] in one_shot_names
            ):
                directions.add(direction)
                break
    return directions == {"encrypt", "decrypt"}


def _python_uses_unavailable_aes_gcm_algorithm(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Attribute)
        and _python_name(node) == "algorithms.AES_GCM"
        for node in ast.walk(tree)
    )


def _python_uses_aad_on_non_gcm_cipher_context(tree: ast.Module) -> bool:
    """Find AAD calls on contexts created from a statically non-GCM Cipher."""

    non_gcm_ciphers: set[str] = set()
    for assignment in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and _python_name(node.value.func).rsplit(".", 1)[-1] == "Cipher"
    ):
        if len(assignment.value.args) < 2:
            continue
        mode = assignment.value.args[1]
        mode_name = (
            _python_name(mode.func).rsplit(".", 1)[-1]
            if isinstance(mode, ast.Call)
            else _python_name(mode).rsplit(".", 1)[-1]
        )
        if mode_name not in {"CBC", "CFB", "CFB8", "CTR", "ECB", "OFB", "XTS"}:
            continue
        targets = (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
        non_gcm_ciphers.update(_python_target_names(targets))

    non_gcm_contexts: set[str] = set()
    for assignment in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr in {"encryptor", "decryptor"}
    ):
        receiver = assignment.value.func.value
        receiver_is_non_gcm = (
            isinstance(receiver, ast.Name)
            and receiver.id in non_gcm_ciphers
        )
        if isinstance(receiver, ast.Call) and (
            _python_name(receiver.func).rsplit(".", 1)[-1] == "Cipher"
            and len(receiver.args) >= 2
        ):
            mode = receiver.args[1]
            mode_name = (
                _python_name(mode.func).rsplit(".", 1)[-1]
                if isinstance(mode, ast.Call)
                else _python_name(mode).rsplit(".", 1)[-1]
            )
            receiver_is_non_gcm |= mode_name in {
                "CBC",
                "CFB",
                "CFB8",
                "CTR",
                "ECB",
                "OFB",
                "XTS",
            }
        if not receiver_is_non_gcm:
            continue
        targets = (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
        non_gcm_contexts.update(_python_target_names(targets))

    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "authenticate_additional_data"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in non_gcm_contexts
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call))
    )


def _python_discards_padding_update_output(tree: ast.Module) -> bool:
    padding_contexts = {
        name
        for assignment in ast.walk(tree)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        and isinstance(assignment.value, ast.Call)
        and _python_name(assignment.value.func).rsplit(".", 1)[-1] == "padder"
        for name in _python_target_names(
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
    }
    return any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "update"
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id in padding_contexts
        for node in ast.walk(tree)
    )


def _python_invalid_padding_context_member(tree: ast.Module) -> str | None:
    """Return a source-proven unavailable PyCA padding-context member.

    A padding context exposes only ``update`` and ``finalize``.  This check is
    deliberately tied to variables constructed by ``padder()`` or
    ``unpadder()`` so ordinary user objects with attributes such as ``data``
    are unaffected.
    """

    contexts = {
        name
        for assignment in ast.walk(tree)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        and isinstance(assignment.value, ast.Call)
        and _python_name(assignment.value.func).rsplit(".", 1)[-1]
        in {"padder", "unpadder"}
        for name in _python_target_names(
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
    }
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in contexts
            and node.attr not in {"update", "finalize"}
        ):
            continue
        return node.attr
    return None


def _python_fixed_block_input_failure(source: str) -> str | None:
    """Reject a fixed unaligned value passed to a raw block-cipher context.

    Only ECB and CBC expose the underlying block-alignment requirement here.
    Unresolved parameters remain acceptable, as required by the evidence
    policy, and independent-route projection keeps a bad sibling or example
    call from invalidating a usable callable route.
    """

    try:
        analyzer = analyzer_for("python", source)
    except Exception:
        return None

    for call in analyzer.calls("update"):
        if not call.arguments or "." not in call.callee:
            continue
        receiver = call.callee.rsplit(".", 1)[0]
        block_size = _python_context_block_size(
            analyzer,
            receiver,
            call.start,
            seen=frozenset(),
        )
        if block_size is None:
            continue
        argument_name = re.fullmatch(r"[A-Za-z_]\w*", call.arguments[0].strip())
        if argument_name is not None and _python_name_is_augmented_before(
            analyzer,
            argument_name.group(0),
            call.start,
        ):
            # An augmented assignment changes the value but is intentionally
            # not treated as a new, statically fixed length by the source
            # resolver.  Preserve that uncertainty instead of reusing the
            # variable's earlier literal length.
            continue
        length = analyzer.static_byte_length(call.arguments[0], call.start)
        if length is not None and length % block_size != 0:
            return (
                "the raw block cipher requires block-aligned input, not a "
                f"source-proven {length}-byte value"
            )
    return None


def _python_name_is_augmented_before(
    analyzer: object,
    name: str,
    index: int,
) -> bool:
    tree = getattr(analyzer, "tree", None)
    node_span = getattr(analyzer, "_node_span", None)
    enclosing = getattr(analyzer, "_enclosing_function", None)
    containing = getattr(analyzer, "_containing_function", None)
    if (
        tree is None
        or not callable(node_span)
        or not callable(enclosing)
        or not callable(containing)
    ):
        return False
    scope = containing(index)
    return any(
        isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
        and enclosing(node) is scope
        and node_span(node)[1] <= index
        for node in ast.walk(tree)
    )


def _python_context_block_size(
    analyzer: object,
    expression: str,
    index: int,
    *,
    seen: frozenset[str],
) -> int | None:
    """Resolve the ECB/CBC block size behind an encryptor/decryptor value."""

    compact = re.sub(r"\s+", "", expression)
    state = f"{index}:{compact}"
    if state in seen:
        return None
    next_seen = seen | frozenset({state})

    if re.fullmatch(r"[A-Za-z_]\w*", compact):
        assignment = getattr(analyzer, "_latest_assignment", lambda *_: None)(
            compact, index
        )
        if assignment is None:
            return None
        assigned, assigned_index = assignment
        return _python_context_block_size(
            analyzer,
            ast.unparse(assigned),
            assigned_index,
            seen=next_seen,
        )

    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return None

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"encryptor", "decryptor"}
    ):
        return _python_cipher_block_size(
            analyzer,
            node.func.value,
            index,
            seen=next_seen,
        )
    return None


def _python_cipher_block_size(
    analyzer: object,
    expression: ast.AST,
    index: int,
    *,
    seen: frozenset[str],
) -> int | None:
    if isinstance(expression, ast.Name):
        state = f"cipher:{index}:{expression.id}"
        if state in seen:
            return None
        assignment = getattr(analyzer, "_latest_assignment", lambda *_: None)(
            expression.id, index
        )
        if assignment is None:
            return None
        assigned, assigned_index = assignment
        return _python_cipher_block_size(
            analyzer,
            assigned,
            assigned_index,
            seen=seen | frozenset({state}),
        )

    if not (
        isinstance(expression, ast.Call)
        and _python_name(expression.func).rsplit(".", 1)[-1] == "Cipher"
    ):
        return None
    arguments = {
        name: value
        for name, value in zip(("algorithm", "mode", "backend"), expression.args)
    }
    arguments.update(
        {
            keyword.arg: keyword.value
            for keyword in expression.keywords
            if keyword.arg is not None
        }
    )
    algorithm = arguments.get("algorithm")
    mode = arguments.get("mode")
    if not (isinstance(algorithm, ast.Call) and isinstance(mode, ast.Call)):
        return None
    mode_name = _python_name(mode.func).rsplit(".", 1)[-1]
    if mode_name not in {"CBC", "ECB"}:
        return None
    algorithm_name = _python_name(algorithm.func).rsplit(".", 1)[-1]
    if algorithm_name in {"AES", "AES128", "AES256", "Camellia", "SEED", "SM4"}:
        return 16
    if algorithm_name in {"TripleDES", "Blowfish", "CAST5", "IDEA"}:
        return 8
    return None


def _python_unpads_a_slice_using_padding_context_block_size(
    tree: ast.Module,
) -> bool:
    padding_contexts = {
        name
        for assignment in ast.walk(tree)
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        and isinstance(assignment.value, ast.Call)
        and _python_name(assignment.value.func).rsplit(".", 1)[-1] == "padder"
        for name in _python_target_names(
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else (assignment.target,)
        )
    }
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "update"
        and "unpad" in _python_name(call.func.value).lower()
        and call.args
        and isinstance(call.args[0], ast.Subscript)
        and isinstance(call.args[0].slice, ast.Slice)
        and any(
            isinstance(node, ast.Attribute)
            and node.attr == "block_size"
            and isinstance(node.value, ast.Name)
            and node.value.id in padding_contexts
            for node in ast.walk(call.args[0].slice)
        )
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call))
    )


def _python_envelope_slice_failure(tree: ast.Module) -> str | None:
    assignments = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    }
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "decrypt" in node.name.lower()
    ):
        slices = {
            target.id: node.value.slice
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.slice, ast.Slice)
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            if isinstance(target, ast.Name)
        }
        tag_slice = next(
            (value for name, value in slices.items() if "tag" in name.lower()),
            None,
        )
        cipher_slice = next(
            (
                value
                for name, value in slices.items()
                if "cipher" in name.lower() or "encrypt" in name.lower()
            ),
            None,
        )
        if (
            tag_slice is not None
            and cipher_slice is not None
            and tag_slice.upper is not None
            and cipher_slice.lower is not None
            and ast.dump(tag_slice.upper, include_attributes=False)
            == ast.dump(cipher_slice.lower, include_attributes=False)
            and not isinstance(tag_slice.upper, (ast.Constant, ast.UnaryOp))
        ):
            return "GCM tag and ciphertext are split at a data-dependent overlapping boundary"
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "update"
            and "decrypt" in _python_name(call.func.value).lower()
            and call.args
            and isinstance(call.args[0], ast.Subscript)
            and isinstance(call.args[0].slice, ast.Slice)
        ):
            continue
        section = call.args[0].slice
        lower = section.lower.value if isinstance(section.lower, ast.Constant) else None
        upper = section.upper.value if isinstance(section.upper, ast.Constant) else None
        if lower is None and isinstance(upper, int) and upper > 0:
            return "decryption consumes an IV prefix instead of the following ciphertext"

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    prefix_sizes: set[int] = set()
    for function in functions.values():
        if "encrypt" not in function.name.lower():
            continue
        assigned_sizes = {
            target.id: int(node.value.args[0].value)
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Call)
            and _python_name(node.value.func).rsplit(".", 1)[-1]
            in {"urandom", "token_bytes"}
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and isinstance(node.value.args[0].value, int)
            for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
            if isinstance(target, ast.Name)
        }
        for returned in (
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Return) and node.value is not None
        ):
            if isinstance(returned, ast.BinOp) and isinstance(returned.op, ast.Add):
                if isinstance(returned.left, ast.Name) and returned.left.id in assigned_sizes:
                    prefix_sizes.add(assigned_sizes[returned.left.id])
    if prefix_sizes:
        for function in functions.values():
            if "decrypt" not in function.name.lower():
                continue
            cuts = {
                int(node.value)
                for item in ast.walk(function)
                if isinstance(item, ast.Subscript)
                and isinstance(item.slice, ast.Slice)
                for node in (item.slice.lower, item.slice.upper)
                if isinstance(node, ast.Constant) and isinstance(node.value, int)
            }
            if cuts and cuts.isdisjoint(prefix_sizes):
                return "decryption splits the IV envelope at a different width than encryption"
    return None


def _python_mutates_ciphertext_before_decryption(tree: ast.Module) -> bool:
    decrypt_inputs = {
        argument.id
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"update", "decrypt"}
        and "decrypt" in _python_name(call.func.value).lower()
        for argument in call.args[:1]
        if isinstance(argument, ast.Name)
    }
    return any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(name in decrypt_inputs for name in _python_target_names(
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        ))
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.Add)
        and any(
            isinstance(item, ast.Constant) and isinstance(item.value, bytes)
            for item in ast.walk(node.value)
        )
        for node in ast.walk(tree)
    )


def _python_discards_cipher_update_output(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "update"
        and "encrypt" in _python_name(node.value.func.value).lower()
        for node in ast.walk(tree)
    )


def _typescript_failure(source: str, task_id: str) -> str | None:
    if _typescript_finalizes_one_context_more_than_once(source):
        return "the same cipher context is finalized more than once"
    if not task_id.endswith("T1"):
        return None
    if _typescript_gcm_roundtrip_has_no_tag(source):
        return "GCM decryption does not receive the encryption authentication tag"
    if _typescript_directional_cipher_keys_are_incompatible(source):
        return "encryption and decryption use incompatible key slices"
    if _typescript_decrypts_unrelated_literal(source):
        return "decryption uses an unrelated literal instead of the encryption result"
    if re.search(
        r"(?P<part>[A-Za-z_$][\w$]*)\s*=\s*"
        r"(?P<input>[A-Za-z_$][\w$]*)\.\s*(?:slice|substring)\s*"
        r"\(\s*0\s*,\s*-\s*\d+\s*\)",
        source,
    ):
        part = re.search(
            r"(?P<part>[A-Za-z_$][\w$]*)\s*=\s*"
            r"[A-Za-z_$][\w$]*\.\s*(?:slice|substring)\s*"
            r"\(\s*0\s*,\s*-\s*\d+\s*\)",
            source,
        )
        assert part is not None
        if re.search(
            rf"\.\s*update\s*\(\s*{re.escape(part.group('part'))}\b",
            source[part.end() :],
        ):
            return "decryption drops a suffix of the produced ciphertext"
    for match in re.finditer(
        r"Buffer\s*\.\s*concat\s*\(\s*\[\s*"
        r"(?P<cipher>(?:protected|encrypted|cipher)[A-Za-z0-9_$]*)\s*,\s*"
        r"(?P<iv>(?:iv|nonce)[A-Za-z0-9_$]*)\s*\]\s*\)",
        source,
        flags=re.IGNORECASE,
    ):
        suffix = source[match.end() :]
        if re.search(
            r"\.\s*(?:slice|subarray)\s*\(\s*0\s*,\s*(?:12|16)\s*\)",
            suffix,
        ) and re.search(
            r"\.\s*(?:slice|subarray)\s*\(\s*(?:12|16)\s*\)",
            suffix,
        ):
            return "decryption reads an IV prefix from an envelope where encryption appended the IV"
    return None


def _typescript_finalizes_one_context_more_than_once(source: str) -> bool:
    """Find a context finalized twice inside one unfinished expression.

    A completed cipher route remains usable even if later, independent source
    tries to reuse the same context.  Two terminal calls in one expression,
    however, cannot complete that expression and remain a definite structural
    failure.
    """

    try:
        analyzer = analyzer_for("typescript", source)
    except Exception:
        return False
    previous: dict[tuple[str | None, str, int | None], object] = {}
    for call in analyzer.calls("final"):
        if "." not in call.callee:
            continue
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        assignment = analyzer.latest_assignment(receiver, call.start)
        if assignment is not None and not re.search(
            r"\bcreate(?:De)?Cipheriv\s*\(",
            assignment.expression,
        ):
            continue
        key = (
            analyzer.containing_function_name(call.start),
            receiver,
            assignment.start if assignment is not None else None,
        )
        prior = previous.get(key)
        if prior is not None and ";" not in source[getattr(prior, "end"):call.start]:
            return True
        previous[key] = call
    return False


def _typescript_gcm_roundtrip_has_no_tag(source: str) -> bool:
    return bool(
        re.search(r"\bcreateCipheriv\s*\(\s*['\"][^'\"]*gcm", source, re.I)
        and re.search(r"\bcreateDecipheriv\s*\(\s*['\"][^'\"]*gcm", source, re.I)
        and not re.search(r"\.\s*getAuthTag\s*\(", source)
        and not re.search(r"\.\s*setAuthTag\s*\(", source)
    )


def _typescript_directional_cipher_keys_are_incompatible(source: str) -> bool:
    analyzer = analyzer_for("typescript", source)
    encrypt = tuple(analyzer.calls("createCipheriv"))
    decrypt = tuple(analyzer.calls("createDecipheriv"))
    for left in encrypt:
        if len(left.arguments) < 2:
            continue
        left_key = re.sub(r"\s+", "", left.arguments[1])
        for right in decrypt:
            if len(right.arguments) < 2:
                continue
            right_key = re.sub(r"\s+", "", right.arguments[1])
            for sliced, whole in ((left_key, right_key), (right_key, left_key)):
                match = re.fullmatch(
                    r"(?P<name>[A-Za-z_$][\w$]*)\.(?:slice|subarray)\([^)]*\)",
                    sliced,
                )
                if match is not None and whole == match.group("name"):
                    return True
    return False


def _typescript_decrypts_unrelated_literal(source: str) -> bool:
    """Find an application decrypt call disconnected from its encrypt call."""

    analyzer = analyzer_for("typescript", source)
    encrypt_calls = tuple(analyzer.calls("encrypt", "Encrypt"))
    decrypt_calls = tuple(analyzer.calls("decrypt", "Decrypt"))
    if not encrypt_calls or not decrypt_calls:
        return False

    encrypted_names = {
        re.sub(r"\s+", "", assignment.name)
        for assignment in getattr(analyzer, "assignments", ())
        if re.search(r"\b(?:encrypt|Encrypt)\s*\(", assignment.expression)
    }
    connected = False
    literal = False
    for call in decrypt_calls:
        if not call.arguments:
            continue
        argument = call.arguments[0].strip()
        compact = re.sub(r"\s+", "", argument)
        if re.match(r"^[\"'`]", argument):
            literal = True
        if re.search(r"\b(?:encrypt|Encrypt)\s*\(", argument) or any(
            re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", compact)
            for name in encrypted_names
        ):
            connected = True
    return literal and not connected


def _csharp_failure(source: str, task_id: str) -> str | None:
    if task_id.endswith("T2") and _csharp_writes_placeholder_as_plaintext(source):
        return "encryption writes an empty destination buffer instead of the task payload"
    if task_id.endswith("T2") and _csharp_transform_copies_from_write_only_stream(
        source
    ):
        return "the encryption helper reads from a write-only CryptoStream"
    if not task_id.endswith("T1"):
        return None
    if _csharp_decrypts_from_unpopulated_destination(source):
        return "decryption reads from an unpopulated destination buffer"
    if _csharp_decrypt_prepends_iv_to_result(source):
        return "decryption prepends clear IV bytes to the recovered plaintext"
    if _csharp_halves_a_serialized_iv_envelope(source):
        return "decryption splits the serialized IV and ciphertext at half its total length"
    if _csharp_decrypts_non_ciphertext_stream(source):
        return "decryption reads from a fresh constant stream instead of the produced ciphertext"
    if _csharp_transform_arguments_are_disconnected(source):
        return "encryption and decryption use incompatible key or IV arguments"
    if _csharp_encrypt_regenerates_supplied_iv(source):
        return "encryption replaces the IV that is later supplied to decryption"
    if _csharp_decrypt_regenerates_key(source):
        return "decryption replaces the encryption key before creating its decryptor"
    if _csharp_base64_envelope_delimiter_mismatch(source):
        return "encryption concatenates Base64 fields without the delimiter required by decryption"

    appended_cipher = re.search(
        r"Array\s*\.\s*Resize\s*\(\s*ref\s+"
        r"(?P<cipher>(?:encrypted|cipher)[A-Za-z0-9_]*)\s*,[^;]+\)\s*;"
        r"[\s\S]*?Array\s*\.\s*Copy\s*\([^;]*\b(?P=cipher)\b[^;]*\)\s*;",
        source,
        flags=re.IGNORECASE,
    )
    if appended_cipher is not None and re.search(
        rf"new\s+MemoryStream\s*\(\s*{re.escape(appended_cipher.group('cipher'))}\s*\)",
        source[appended_cipher.end() :],
        flags=re.IGNORECASE,
    ):
        return "decryption consumes ciphertext with an appended clear IV still attached"

    if re.search(
        r"\b(?P<value>(?:protected|encrypted|cipher)[A-Za-z0-9_]*)\s*=\s*"
        r"[\"'][^\"']*[\"']\s*;[\s\S]*?"
        r"\b(?:Recover|Decrypt)[A-Za-z0-9_]*\s*\(\s*(?P=value)\b",
        source,
        flags=re.IGNORECASE,
    ):
        return "the encryption result is overwritten before decryption"

    if re.search(
        r"(?:using\s*\(\s*)?Aes\s+(?P<name>[A-Za-z_]\w*)\s*=\s*Aes\.Create\(\)"
        r"[\s\S]*?CreateEncryptor[\s\S]*?"
        r"(?:using\s*\(\s*)?Aes\s+(?P=name)\s*=\s*Aes\.Create\(\)"
        r"[\s\S]*?\b(?P=name)\.Key\s*=\s*(?P=name)\.Key\s*;"
        r"[\s\S]*?CreateDecryptor",
        source,
        flags=re.IGNORECASE,
    ):
        return "the decrypting AES instance retains unrelated default key and IV state"

    directional = _csharp_directional_bodies(source)
    if directional["encrypt"] and directional["decrypt"]:
        fresh_helpers = _csharp_fresh_byte_helpers(source)
        encrypt_fresh = _csharp_body_uses_fresh_key(directional["encrypt"], fresh_helpers)
        decrypt_fresh = _csharp_body_uses_fresh_key(directional["decrypt"], fresh_helpers)
        if decrypt_fresh:
            return "encryption and decryption independently generate different keys"
        encrypt_iv = _csharp_body_uses_fresh_iv(directional["encrypt"], fresh_helpers)
        decrypt_iv = _csharp_body_uses_fresh_iv(directional["decrypt"], fresh_helpers)
        if encrypt_iv and decrypt_iv:
            return "encryption and decryption independently generate different IVs"
        if any(
            re.search(
                r"CryptoStream[\s\S]*?\.\s*Write\s*\(\s*(?:iv|nonce)\b",
                body,
                flags=re.IGNORECASE,
            )
            for body in directional["encrypt"]
        ) and any(
            re.search(
                r"Array\.Copy\s*\(\s*[A-Za-z_]\w*\s*,\s*(?:0\s*,\s*)?"
                r"(?:iv|nonce)[A-Za-z0-9_]*\s*,",
                body,
                flags=re.IGNORECASE,
            )
            for body in directional["decrypt"]
        ):
            return "encryption encrypts IV bytes while decryption expects a clear IV prefix"

    if re.search(
        r"CryptoStream\s*\([^;]*CryptoStreamMode\s*\.\s*Write[^;]*\)"
        r"[\s\S]*?\.\s*Write\s*\(\s*(?:iv|nonce)\b",
        source,
        flags=re.IGNORECASE,
    ) and re.search(
        r"(?:Array\s*\.\s*Copy|Buffer\s*\.\s*BlockCopy)\s*\(\s*"
        r"(?:data|encrypted|cipher)[A-Za-z0-9_]*\s*,[^;]*\b(?:iv|nonce)[A-Za-z0-9_]*\b",
        source,
        flags=re.IGNORECASE,
    ):
        return "encryption encrypts IV bytes as payload while decryption treats them as an IV prefix"

    if re.search(
        r"CryptoStream\s*\([^;]*CreateDecryptor[^;]*\)"
        r"[\s\S]*?\.\s*Write\s*\(\s*(?P<envelope>(?:encrypted|cipher)[A-Za-z0-9_]*)\s*,"
        r"\s*0\s*,\s*(?P=envelope)\.Length\s*\)",
        source,
        flags=re.IGNORECASE,
    ) and re.search(
        r"Array\.Copy\s*\(\s*(?:iv|nonce)[A-Za-z0-9_]*\s*,\s*0\s*,\s*"
        r"(?:encrypted|cipher)[A-Za-z0-9_]*\s*,\s*0\s*,",
        source,
        flags=re.IGNORECASE,
    ):
        return "decryption consumes a clear-IV prefix as ciphertext"

    if re.search(
        r"CryptoStream\s*\([^;]*CreateEncryptor\s*\(\s*\)[^;]*\)"
        r"[\s\S]*?\.\s*(?:Write|CopyTo)\s*\(",
        source,
        flags=re.IGNORECASE,
    ) and re.search(
        r"\b(?P<aes>[A-Za-z_]\w*)\.IV\s*=\s*(?:iv|nonce)[A-Za-z0-9_]*\s*;"
        r"[\s\S]*?\b(?P=aes)\.CreateDecryptor\s*\(",
        source,
        flags=re.IGNORECASE,
    ) and not re.search(
        r"\b[A-Za-z_]\w*\.IV\s*=\s*(?:iv|nonce)[A-Za-z0-9_]*\s*;"
        r"[\s\S]*?CreateEncryptor\s*\(",
        source,
        flags=re.IGNORECASE,
    ) and not re.search(r"\.\s*GenerateIV\s*\(", source, flags=re.IGNORECASE):
        return "the stored IV was not assigned to the encrypting AES instance"

    copied_iv = re.search(
        r"(?:Array\s*\.\s*Copy\s*\(\s*"
        r"(?P<array_cipher>(?:encrypted|cipher)[A-Za-z0-9_]*)\s*,\s*"
        r"(?:\d+\s*,\s*)?(?P<array_iv>(?:iv|nonce)[A-Za-z0-9_]*)\s*,|"
        r"Buffer\s*\.\s*BlockCopy\s*\(\s*"
        r"(?P<block_cipher>(?:encrypted|cipher)[A-Za-z0-9_]*)\s*,\s*[^,]+,\s*"
        r"(?P<block_iv>(?:iv|nonce)[A-Za-z0-9_]*)\s*,)",
        source,
        flags=re.IGNORECASE,
    )
    copied_cipher = (
        copied_iv.group("array_cipher") or copied_iv.group("block_cipher")
        if copied_iv
        else None
    )
    copied_iv_name = (
        copied_iv.group("array_iv") or copied_iv.group("block_iv")
        if copied_iv
        else None
    )
    copied_iv_used = bool(
        copied_iv
        and (
            re.search(
                rf"CreateDecryptor\s*\([^;]*\b{re.escape(copied_iv_name or '')}\b",
                source,
                flags=re.IGNORECASE,
            )
            or any(
                re.search(
                    rf"\b{re.escape(owner)}\s*\.\s*CreateDecryptor\s*\([^;]*"
                    rf"\b{re.escape(owner)}\s*\.\s*IV\b",
                    source,
                    flags=re.IGNORECASE,
                )
                for owner in re.findall(
                    rf"\b([A-Za-z_]\w*)\s*\.\s*IV\s*=\s*"
                    rf"{re.escape(copied_iv_name or '')}\s*;",
                    source,
                    flags=re.IGNORECASE,
                )
            )
        )
    )
    if copied_iv_used and not _csharp_serializes_iv_prefix(source) and not re.search(
        rf"(?:Concat|BlockCopy|Array\.Copy)[^;]*\b{re.escape(copied_iv_name or '')}\b"
        rf"[^;]*\b{re.escape(copied_cipher or '')}\b",
        source[: copied_iv.start()],
        flags=re.IGNORECASE,
    ):
        return "decryption reads an IV prefix that encryption never prepended"

    if re.search(
        r"ToBase64String\s*\(\s*[A-Za-z_]\w*\.IV\s*\)\s*\+\s*"
        r"Convert\.ToBase64String",
        source,
        flags=re.IGNORECASE,
    ):
        split = re.search(r"Substring\s*\(\s*0\s*,\s*(?P<size>\d+)\s*\)", source)
        if split and int(split.group("size")) != 24:
            return "the Base64 IV prefix is split at a width that cannot recover a 16-byte AES IV"

    if _csharp_ciphertext_text_round_trip(source):
        return "binary ciphertext is corrupted by a text-reader and UTF-8 round trip"

    if re.search(
        r"(?P<bytes>[A-Za-z_]\w*)\s*=\s*Convert\.FromBase64String\s*\(\s*"
        r"(?:key[A-Za-z0-9_]*\s*\+\s*(?:iv|nonce)[A-Za-z0-9_]*|"
        r"(?:iv|nonce)[A-Za-z0-9_]*\s*\+\s*key[A-Za-z0-9_]*)\s*\)"
        r"[\s\S]*?new\s+MemoryStream\s*\(\s*(?P=bytes)\b",
        source,
        flags=re.IGNORECASE,
    ):
        return "decryption consumes serialized key metadata instead of the encryption result"
    return None


def _csharp_serializes_iv_prefix(source: str) -> bool:
    """Recognize an IV and ciphertext copied into one returned envelope."""

    destinations: dict[str, set[str]] = {}
    for match in re.finditer(
        r"(?:Array\s*\.\s*Copy|Buffer\s*\.\s*BlockCopy)\s*\(\s*"
        r"(?P<input>[A-Za-z_]\w*(?:\s*\.\s*IV)?)\s*,[^,]*,\s*"
        r"(?P<output>[A-Za-z_]\w*)\s*,",
        source,
        flags=re.IGNORECASE,
    ):
        input_name = match.group("input")
        kind = (
            "iv"
            if re.search(r"(?:\biv\b|nonce|\.\s*IV)$", input_name, re.I)
            else "cipher"
            if re.search(r"encrypt|cipher", input_name, re.I)
            else "other"
        )
        destinations.setdefault(match.group("output").lower(), set()).add(kind)
    return any({"iv", "cipher"} <= kinds for kinds in destinations.values())


def _csharp_ciphertext_text_round_trip(source: str) -> bool:
    """Find direct ciphertext decoding followed by UTF-8 re-encoding.

    A ``StreamReader`` over a plaintext staging stream is ordinary input
    preparation.  Only a reader attached directly to a stream constructed
    from a ciphertext value destroys arbitrary binary ciphertext.
    """

    for stream_match in re.finditer(
        r"\b(?:var|MemoryStream)\s+(?P<stream>[A-Za-z_]\w*)\s*=\s*"
        r"new\s+MemoryStream\s*\(\s*"
        r"(?P<cipher>(?:encrypted|cipher)[A-Za-z0-9_]*)\s*\)",
        source,
        flags=re.IGNORECASE,
    ):
        suffix = source[stream_match.end() :]
        reader_match = re.search(
            r"\b(?:var|StreamReader)\s+(?P<reader>[A-Za-z_]\w*)\s*=\s*"
            r"new\s+StreamReader\s*\(\s*"
            + re.escape(stream_match.group("stream"))
            + r"\s*\)",
            suffix,
            flags=re.IGNORECASE,
        )
        if reader_match is None:
            continue
        text_match = re.search(
            r"\b(?:var|string)\s+(?P<text>[A-Za-z_]\w*)\s*=\s*"
            + re.escape(reader_match.group("reader"))
            + r"\s*\.\s*ReadToEnd\s*\(\s*\)",
            suffix[reader_match.end() :],
            flags=re.IGNORECASE,
        )
        if text_match is not None and re.search(
            r"Encoding\s*\.\s*UTF8\s*\.\s*GetBytes\s*\(\s*"
            + re.escape(text_match.group("text"))
            + r"\s*\)",
            suffix[reader_match.end() + text_match.end() :],
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _csharp_transform_copies_from_write_only_stream(source: str) -> bool:
    """Find a shared helper that reads a stream configured for encryption writes."""

    return bool(
        re.search(
            r"CryptoStreamMode\s*\.\s*Read\s*:\s*CryptoStreamMode\s*\.\s*Write"
            r"[\s\S]*?\b(?P<stream>[A-Za-z_]\w*)\s*\.\s*CopyTo\s*\(",
            source,
            flags=re.IGNORECASE,
        )
        and re.search(r"\bTransform\s*\([^;]*\bfalse\s*\)", source, re.I)
        and re.search(r"\bTransform\s*\([^;]*\btrue\s*\)", source, re.I)
    )


def _csharp_discards_transform_output(source: str) -> bool:
    return bool(
        re.search(
            r"\b[A-Za-z_]\w*\s*\.\s*TransformFinalBlock\s*\(\s*"
            r"(?P<input>[A-Za-z_]\w*)\s*,[^;]+\)\s*;"
            r"[\s\S]*?\.\s*Write\s*\(\s*(?P=input)\s*,[\s\S]*?\breturn\s+[^;]+",
            source,
            flags=re.IGNORECASE,
        )
    )


def _csharp_encrypt_regenerates_supplied_iv(source: str) -> bool:
    directional = _csharp_directional_bodies(source)
    return bool(
        any(re.search(r"\.\s*GenerateIV\s*\(", body, re.I) for body in directional["encrypt"])
        and re.search(
            r"(?P<iv>(?:iv|nonce)[A-Za-z0-9_]*)\s*=\s*"
            r"(?P<aes>[A-Za-z_]\w*)\s*\.\s*IV\s*;[\s\S]*?"
            r"\bEncrypt[A-Za-z0-9_]*\s*\(\s*(?P=aes)\s*,[^;]*\b(?P=iv)\b",
            source,
            flags=re.IGNORECASE,
        )
    )


def _csharp_decrypt_regenerates_key(source: str) -> bool:
    return any(
        re.search(r"\.\s*GenerateKey\s*\(\)[\s\S]*?CreateDecryptor", body, re.I)
        for body in _csharp_directional_bodies(source)["decrypt"]
    )


def _csharp_base64_envelope_delimiter_mismatch(source: str) -> bool:
    concatenated = re.search(
        r"ToBase64String\s*\([^)]*\)\s*\+\s*Convert\s*\.\s*ToBase64String\s*\(",
        source,
        flags=re.IGNORECASE,
    )
    if concatenated is None:
        return False
    return any(
        re.search(r"\.\s*Split\s*\(", body) is not None
        for body in _csharp_directional_bodies(source)["decrypt"]
    )


def _csharp_returns_unpopulated_encryption_buffer(source: str) -> bool:
    """Find an encryption route that replaces its output with a fresh zero buffer."""

    return any(
        re.search(r"\bCreateEncryptor\s*\(", body, flags=re.IGNORECASE)
        and re.search(
            r"\breturn\s+new\s+byte\s*\[[^;]+\]\s*;", body, re.IGNORECASE
        )
        for body in _csharp_directional_bodies(source)["encrypt"]
    )


def _csharp_decrypts_from_unpopulated_destination(source: str) -> bool:
    """Find a decryptor wired to a fresh destination rather than ciphertext."""

    for allocation in re.finditer(
        r"\bbyte\s*\[\s*\]\s+(?P<buffer>(?:decrypt|recover)[A-Za-z0-9_]*)\s*=\s*"
        r"new\s+byte\s*\[[^\]]+\]\s*;",
        source,
        flags=re.IGNORECASE,
    ):
        suffix = source[allocation.end() :]
        stream = re.search(
            rf"\b(?:MemoryStream|var)\s+(?P<stream>[A-Za-z_]\w*)\s*=\s*"
            rf"new\s+MemoryStream\s*\(\s*{re.escape(allocation.group('buffer'))}\s*\)",
            suffix,
            flags=re.IGNORECASE,
        )
        if stream is None:
            continue
        stream_name = stream.group("stream")
        crypto = re.search(
            rf"new\s+CryptoStream\s*\(\s*{re.escape(stream_name)}\s*,\s*"
            r"(?:[A-Za-z_]\w*\s*,|[^,;]*CreateDecryptor\s*\([^;]*,)\s*"
            r"CryptoStreamMode\s*\.\s*Write",
            suffix[stream.end() :],
            flags=re.IGNORECASE | re.DOTALL,
        )
        if crypto is not None and not re.search(
            r"\.\s*Write\s*\(\s*(?:encrypted|cipher)[A-Za-z0-9_]*\b",
            suffix[crypto.end() :],
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _csharp_decrypts_non_ciphertext_stream(source: str) -> bool:
    """Find a read-mode decryptor backed by newly created constant bytes."""

    return bool(
        re.search(
            r"new\s+MemoryStream\s*\(\s*(?:BitConverter\s*\.\s*GetBytes\s*\(|"
            r"new\s+byte\s*\[)",
            source,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"new\s+CryptoStream\s*\([^;]*CreateDecryptor\s*\([^;]*"
            r"CryptoStreamMode\s*\.\s*Read",
            source,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


def _csharp_transform_arguments_are_disconnected(source: str) -> bool:
    """Reject literal named-argument mismatches between paired AES transforms."""

    encrypt = re.search(
        r"CreateEncryptor\s*\(\s*(?P<key>[A-Za-z_]\w*)\s*,\s*"
        r"(?P<iv>[A-Za-z_]\w*)\s*\)",
        source,
        flags=re.IGNORECASE,
    )
    decrypt = re.search(
        r"CreateDecryptor\s*\(\s*(?P<key>[A-Za-z_]\w*)\s*,\s*"
        r"(?P<iv>[A-Za-z_]\w*)\s*\)",
        source,
        flags=re.IGNORECASE,
    )
    if encrypt is None or decrypt is None:
        return False
    encrypt_args = (encrypt.group("key").lower(), encrypt.group("iv").lower())
    decrypt_args = (decrypt.group("key").lower(), decrypt.group("iv").lower())
    if encrypt_args == decrypt_args:
        return False
    role = re.compile(r"(?:^|_)(?:key|iv|nonce)(?:$|_)", re.IGNORECASE)
    return any(role.search(value) for value in (*encrypt_args, *decrypt_args))


def _csharp_decrypt_prepends_iv_to_result(source: str) -> bool:
    """Find decrypt helpers that return an IV followed by recovered bytes."""

    for body in _csharp_directional_bodies(source)["decrypt"]:
        stream = re.search(
            r"\b(?:MemoryStream|var)\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
            r"new\s+(?:System\.IO\.)?MemoryStream\s*\(\s*\)",
            body,
            flags=re.IGNORECASE,
        )
        if stream is None:
            continue
        name = stream.group("name")
        if (
            re.search(
                rf"\b{re.escape(name)}\s*\.\s*Write\s*\(\s*[^,;]*(?:IV|iv|nonce)",
                body,
                flags=re.IGNORECASE,
            )
            and re.search(
                rf"new\s+CryptoStream\s*\(\s*{re.escape(name)}\s*,[^;]*decryptor",
                body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            and _csharp_returns_stream_bytes(body, name)
        ):
            return True
    return False


def _csharp_returns_stream_bytes(body: str, stream_name: str) -> bool:
    if re.search(
        rf"\breturn\s+{re.escape(stream_name)}\s*\.\s*ToArray\s*\(\s*\)",
        body,
        flags=re.IGNORECASE,
    ):
        return True
    assigned = {
        match.group("name")
        for match in re.finditer(
            rf"\b(?:byte\s*\[\s*\]|var)\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
            rf"{re.escape(stream_name)}\s*\.\s*ToArray\s*\(\s*\)\s*;",
            body,
            flags=re.IGNORECASE,
        )
    }
    return any(
        re.search(rf"\breturn\s+{re.escape(name)}\s*;", body, re.IGNORECASE)
        for name in assigned
    )


def _csharp_halves_a_serialized_iv_envelope(source: str) -> bool:
    return bool(
        re.search(r"BitConverter\s*\.\s*ToString\s*\(", source, re.I)
        and re.search(r"\.\s*Split\s*\(\s*['\"]-['\"]\s*\)", source, re.I)
        and re.search(
            r"byte\s*\[\s*\]\s+(?:iv|nonce)[A-Za-z0-9_]*\s*=\s*"
            r"new\s+byte\s*\[[^\]]*\.Length\s*/\s*2\s*\]",
            source,
            flags=re.IGNORECASE,
        )
    )


def _java_failure(source: str, task_id: str) -> str | None:
    if not task_id.endswith("T1"):
        return None
    if _java_direction_calls_are_mutually_exclusive(source):
        return "encryption and decryption are placed in mutually exclusive branches"
    if _java_calls_fresh_iv_helper_in_both_directions(source):
        return "encryption and decryption independently request fresh IV values"
    if _java_encrypt_ignores_supplied_key(source):
        return "encryption replaces the key that is supplied to decryption"
    if _java_round_trips_ciphertext_through_raw_string(source):
        return "binary ciphertext is corrupted by a platform-default string round trip"
    if _java_decrypt_helper_uses_unrelated_literal(source):
        return "decryption receives an unrelated literal instead of the produced ciphertext"
    if _java_decrypts_literal_instead_of_produced_envelope(source):
        return "decryption uses a literal placeholder instead of the produced envelope"
    if _java_gcm_derives_untransported_iv_from_ciphertext(source):
        return "GCM decryption derives an IV from ciphertext that encryption did not prefix"
    if _java_ciphertext_only_is_treated_as_iv_envelope(source):
        return "encryption returns ciphertext alone while decryption treats its prefix as an IV"
    if _java_cbc_envelope_split_is_malformed(source):
        return "decryption splits the IV-and-ciphertext envelope at incompatible offsets"
    if _java_single_init_selects_both_directions(source):
        return "one conditional Cipher.init call cannot complete both encryption and decryption"
    analyzer = analyzer_for("java", source)
    assignments = extract_assignments(source)
    literal_transformations: dict[tuple[str | None, str], str] = {}
    for call in analyzer.calls("getInstance"):
        if "Cipher" not in call.callee or not call.arguments:
            continue
        assignment = next(
            (value for value in assignments if value.start <= call.start <= value.end),
            None,
        )
        if assignment is None:
            continue
        receiver = re.sub(r"\s+", "", assignment.name)
        expression = call.arguments[0].strip()
        resolved = set(analyzer.resolve_strings(expression, call.start).values)
        literal = re.fullmatch(r"([\"'])(?P<value>.*?)\1", expression, re.DOTALL)
        if literal is not None:
            resolved.add(literal.group("value"))
        selected = next(
            (
                value
                for value in resolved
                if "/CBC/" in value.upper() or "/GCM/" in value.upper()
            ),
            None,
        )
        if selected is not None:
            literal_transformations[
                (analyzer.containing_function_name(call.start), receiver)
            ] = selected

    modes: dict[tuple[str | None, str], str] = {}
    decrypt_init: dict[tuple[str | None, str], object] = {}
    decrypt_terminal: set[tuple[str | None, str]] = set()
    for call in sorted(analyzer.calls("init", "doFinal"), key=lambda item: item.start):
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        key = (analyzer.containing_function_name(call.start), receiver)
        if call.simple_name == "init":
            direction = (
                "decrypt"
                if call.arguments and "DECRYPT_MODE" in call.arguments[0]
                else "encrypt"
                if call.arguments and "ENCRYPT_MODE" in call.arguments[0]
                else ""
            )
            modes[key] = direction
            if direction == "decrypt":
                decrypt_init[key] = call
                transformation = literal_transformations.get(key, "").upper()
                if (
                    ("/CBC/" in transformation or "/GCM/" in transformation)
                    and len(call.arguments) < 3
                ):
                    return (
                        "literal CBC/GCM decryption omits the associated "
                        "parameter argument"
                    )
        elif modes.get(key) == "decrypt":
            decrypt_terminal.add(key)
    if decrypt_init and any(key not in decrypt_terminal for key in decrypt_init):
        return "the initialized decryption operation never calls Cipher.doFinal"
    if (
        any(value == "encrypt" for value in modes.values())
        and not decrypt_init
        and len(analyzer.calls("doFinal")) >= 2
    ):
        return "the second cipher terminal runs without decryption initialization"

    if re.search(
        r"\bencrypt[A-Za-z0-9_]*\s*\([^;]*\bgenerateKey\s*\(\s*\)\s*\)"
        r"[\s\S]*?\bdecrypt[A-Za-z0-9_]*\s*\([^;]*\bgenerateKey\s*\(\s*\)\s*\)",
        source,
        flags=re.IGNORECASE,
    ):
        return "encryption and decryption receive independently generated keys"
    if re.search(
        r"\bencrypt[A-Za-z0-9_]*\s*\([^;]*\b(?P<helper>(?:get|create|new|generate)"
        r"[A-Za-z0-9_]*Key)\s*\(\s*\)\s*\)"
        r"[\s\S]*?\bdecrypt[A-Za-z0-9_]*\s*\([^;]*\b(?P=helper)\s*\(\s*\)\s*\)",
        source,
        flags=re.IGNORECASE,
    ):
        return "encryption and decryption receive independently generated keys"

    envelope_sources: dict[tuple[str | None, str], set[str]] = {}
    for match in re.finditer(
        r"System\.arraycopy\s*\(\s*(?P<source>[A-Za-z_$][\w$]*)\s*,[^,]*,\s*"
        r"(?P<target>[A-Za-z_$][\w$]*)\s*,",
        source,
    ):
        key = (analyzer.containing_function_name(match.start()), match.group("target"))
        envelope_sources.setdefault(key, set()).add(
            match.group("source")
        )
    envelopes = {
        key
        for key, inputs in envelope_sources.items()
        if len(inputs) >= 2
        and any(re.search(r"(?:iv|nonce)", value, re.I) for value in inputs)
        and any(re.search(r"(?:encrypt|cipher)", value, re.I) for value in inputs)
    }
    envelope_names = {name for _scope, name in envelopes}
    serialized_envelopes = set(envelope_names)
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            if assignment.name in serialized_envelopes:
                continue
            identifiers = set(re.findall(r"[A-Za-z_$][\w$]*", assignment.expression))
            if identifiers & serialized_envelopes and re.search(
                r"Base64\s*\.\s*getEncoder|new\s+String\s*\(",
                assignment.expression,
                flags=re.IGNORECASE,
            ):
                serialized_envelopes.add(assignment.name)
                changed = True
    for call in analyzer.calls("doFinal"):
        scope = analyzer.containing_function_name(call.start)
        if not call.arguments:
            continue
        envelope = call.arguments[0].strip()
        compact_envelope = re.sub(r"\s+", "", envelope)
        full_envelope_argument = any(
            compact_envelope == name
            or re.fullmatch(
                r"(?:Base64\.)?getDecoder\(\)\.decode\(" + re.escape(name) + r"\)",
                compact_envelope,
            )
            for name in serialized_envelopes
        )
        if (
            modes.get(
                (scope, re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0])
            )
            == "decrypt"
            and full_envelope_argument
            and (
                len(call.arguments) == 1
                or (
                    len(call.arguments) >= 3
                    and call.arguments[1].strip() == "0"
                )
            )
        ):
            result_name = next(
                (
                    assignment.name
                    for assignment in assignments
                    if assignment.start <= call.start <= assignment.end
                ),
                None,
            )
            if not (
                _java_envelope_has_iv_prefix(source, envelope_names)
                and result_name is not None
                and _java_decryption_drops_plaintext_prefix(source)
            ):
                return "decryption consumes the complete serialized IV-and-ciphertext envelope"
        if (scope, envelope) not in envelopes:
            continue
        if len(call.arguments) == 1 or (
            len(call.arguments) >= 3 and call.arguments[1].strip() == "0"
        ):
            result_name = next(
                (
                    assignment.name
                    for assignment in assignments
                    if assignment.start <= call.start <= assignment.end
                ),
                None,
            )
            if (
                _java_envelope_has_iv_prefix(source, envelope_names)
                and result_name is not None
                and _java_decryption_drops_plaintext_prefix(source)
            ):
                continue
            return "decryption consumes the complete IV-prefixed envelope"

    discarded_decrypt_results = []
    consumed_decrypt_results = []
    for call in analyzer.calls("doFinal"):
        scope = analyzer.containing_function_name(call.start)
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        if modes.get((scope, receiver)) != "decrypt" or len(call.arguments) not in {1, 3}:
            continue
        line_prefix = source[source.rfind("\n", 0, call.start) + 1 : call.start]
        if line_prefix.strip():
            consumed_decrypt_results.append(call)
        else:
            discarded_decrypt_results.append(call)
    if discarded_decrypt_results and not consumed_decrypt_results:
        return "decryption discards the bytes returned by Cipher.doFinal"

    raw_ciphertexts = {
        assignment.name
        for assignment in assignments
        if re.search(r"\.\s*doFinal\s*\(", assignment.expression)
    }
    if any(
        re.search(
            rf"System\.arraycopy\s*\(\s*(?:iv|nonce)[A-Za-z0-9_$]*\s*,\s*0\s*,\s*"
            rf"{re.escape(name)}\s*,\s*0\s*,",
            source,
            flags=re.IGNORECASE,
        )
        for name in raw_ciphertexts
    ):
        return "the IV overwrites the first block of the produced ciphertext"
    if re.search(
        r"return\s+(?:iv|nonce)[A-Za-z0-9_$]*\s*=\s*"
        r"[A-Za-z_$][\w$]*\.\s*doFinal\s*\(",
        source,
        flags=re.IGNORECASE,
    ) and re.search(
        r"new\s+IvParameterSpec\s*\(\s*(?:encrypted|cipher)[A-Za-z0-9_$]*\s*,\s*0\s*,",
        source,
        flags=re.IGNORECASE,
    ):
        return "encryption returns ciphertext alone but decryption treats it as IV-prefixed"

    if re.search(
        r"(?:ivEnc|encodedIv)\s*\+\s*(?:encrypted|cipher)[A-Za-z0-9_]*"
        r"[\s\S]*?\.getBytes\s*\(\s*\)[\s\S]*?System\.arraycopy",
        source,
        flags=re.IGNORECASE,
    ):
        return "concatenated Base64 IV and ciphertext text is sliced as raw binary data"

    return None


def _java_direction_calls_are_mutually_exclusive(source: str) -> bool:
    """Find a command branch that chooses encryption or decryption, but not both."""

    main = next((method.text for method in iter_methods(source) if method.name == "main"), "")
    if not main:
        return False
    directional_branch = re.compile(
        r"\bif\s*\([^{};]*\)\s*\{(?P<first>.*?)\}\s*"
        r"else(?:\s+if\s*\([^{};]*\))?\s*\{(?P<second>.*?)\}",
        flags=re.DOTALL,
    )
    for match in directional_branch.finditer(main):
        first = match.group("first")
        second = match.group("second")
        if (
            re.search(r"\bencrypt[A-Za-z0-9_$]*\s*\(", first, re.I)
            and re.search(r"\bdecrypt[A-Za-z0-9_$]*\s*\(", second, re.I)
        ) or (
            re.search(r"\bdecrypt[A-Za-z0-9_$]*\s*\(", first, re.I)
            and re.search(r"\bencrypt[A-Za-z0-9_$]*\s*\(", second, re.I)
        ):
            return True
    return False


def _java_overwrites_accumulated_cipher_output_with_final_only(source: str) -> bool:
    """Find an encryption result that drops preceding ``Cipher.update`` bytes."""

    pattern = re.compile(
        r"(?P<stream>[A-Za-z_$][\w$]*)\s*\.\s*write\s*\(\s*"
        r"(?P<cipher>[A-Za-z_$][\w$]*)\s*\.\s*update\s*\([^;]*?\)\s*\)\s*;"
        r"[\s\S]*?"
        r"(?P<output>[A-Za-z_$][\w$]*)\s*=\s*(?P=stream)\s*\.\s*toByteArray\s*\(\s*\)\s*;"
        r"[\s\S]*?"
        r"(?P=output)\s*=\s*(?P=cipher)\s*\.\s*doFinal\s*\([^;]*\)\s*;",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(source):
        suffix = source[match.end() :]
        stream = re.escape(match.group("stream"))
        cipher = re.escape(match.group("cipher"))
        next_initialization = re.search(
            rf"\b{cipher}\s*\.\s*init\s*\(",
            suffix,
            flags=re.IGNORECASE,
        )
        route_suffix = (
            suffix[: next_initialization.start()]
            if next_initialization is not None
            else suffix
        )
        if re.search(
            rf"\b{stream}\s*\.\s*write\s*\(\s*{cipher}\s*\.\s*doFinal\s*\(",
            route_suffix,
            flags=re.IGNORECASE,
        ) or re.search(
            rf"\b{stream}\s*\.\s*toByteArray\s*\(\s*\)",
            route_suffix,
            flags=re.IGNORECASE,
        ):
            continue
        return True
    return False


def _java_base64_decodes_raw_ciphertext(source: str) -> bool:
    analyzer = analyzer_for("java", source)
    raw_names = {
        assignment.name.rsplit(".", 1)[-1]
        for assignment in extract_assignments(source)
        if re.search(r"\.\s*doFinal\s*\(", assignment.expression)
    }
    for name in raw_names:
        decoded = rf"Base64\s*\.\s*getDecoder\s*\(\s*\)\s*\.\s*decode\s*\(\s*{re.escape(name)}\s*\)"
        if re.search(rf"\.\s*doFinal\s*\(\s*{decoded}", source):
            return True
        decoded_names = {
            assignment.name.rsplit(".", 1)[-1]
            for assignment in extract_assignments(source)
            if re.search(decoded, assignment.expression)
        }
        if decoded_names and any(
            call.arguments
            and any(
                re.fullmatch(rf"\s*{re.escape(value)}\s*", call.arguments[0])
                for value in decoded_names
            )
            for call in analyzer.calls("doFinal")
        ):
            return True
    return False


def _java_encrypt_ignores_supplied_key(source: str) -> bool:
    for method in iter_methods(source):
        if "encrypt" not in method.name.lower():
            continue
        key_parameters = {
            parameter
            for parameter in method.params
            if re.search(r"key", parameter, re.IGNORECASE)
        }
        if not key_parameters or not re.search(
            r"KeyGenerator[\s\S]*?\.\s*generateKey\s*\(",
            method.text,
            flags=re.IGNORECASE,
        ):
            continue
        init = re.search(
            r"\.\s*init\s*\(\s*Cipher\s*\.\s*ENCRYPT_MODE\s*,\s*"
            r"(?P<key>[A-Za-z_$][\w$]*)",
            method.text,
            flags=re.IGNORECASE,
        )
        if init is not None and init.group("key") not in key_parameters:
            return True
    return False


def _java_ciphertext_only_is_treated_as_iv_envelope(source: str) -> bool:
    encrypt_returns_raw = any(
        "encrypt" in method.name.lower()
        and re.search(r"\breturn\s+[^;]*\.\s*doFinal\s*\(", method.text)
        for method in iter_methods(source)
    )
    if not encrypt_returns_raw:
        return False
    return any(
        ("decrypt" in method.name.lower() or "recover" in method.name.lower())
        and (
            re.search(
                r"new\s+IvParameterSpec\s*\(\s*(?:data|ciphertext|encrypted)\b",
                method.text,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"(?:copyOfRange|arraycopy)\s*\(\s*(?:data|ciphertext|encrypted)\b[^;]*(?:8|12|16)",
                method.text,
                flags=re.IGNORECASE,
            )
        )
        for method in iter_methods(source)
    )


def _java_cbc_envelope_split_is_malformed(source: str) -> bool:
    if re.search(
        r"new\s+byte\s*\[\s*(?P<input>[A-Za-z_$][\w$]*)\.length\s*-\s*"
        r"(?P=input)\.length\s*/\s*\d+\s*\]",
        source,
    ):
        return True
    if re.search(
        r"new\s+byte\s*\[\s*(?P<input>[A-Za-z_$][\w$]*)\.length\s*-\s*"
        r"(?P=input)\.length\s*%\s*\d+\s*\]",
        source,
    ):
        return True
    if re.search(
        r"new\s+byte\s*\[\s*(?P<encoded>[A-Za-z_$][\w$]*)\.getBytes\s*\([^)]*\)\.length\s*-\s*"
        r"(?:iv|nonce)[A-Za-z0-9_$]*\.length\s*\][\s\S]*?"
        r"(?P<input>[A-Za-z_$][\w$]*)\s*=\s*Base64\s*\.\s*getDecoder",
        source,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def _java_decryption_drops_plaintext_prefix(source: str) -> bool:
    return bool(
        re.search(
            r"(?P<plain>[A-Za-z_$][\w$]*)\s*=\s*[^;]*\.\s*doFinal\s*\([^;]*\)\s*;"
            r"[\s\S]*?new\s+byte\s*\[\s*(?P=plain)\.length\s*-\s*"
            r"(?P<prefix>(?:iv|nonce)[A-Za-z0-9_$]*\.length|8|12|16)\s*\][\s\S]*?"
            r"System\s*\.\s*arraycopy\s*\(\s*(?P=plain)\s*,\s*"
            r"(?P=prefix)",
            source,
            flags=re.IGNORECASE,
        )
    )


def _java_single_init_selects_both_directions(source: str) -> bool:
    analyzer = analyzer_for("java", source)
    directional = [
        call
        for call in analyzer.calls("init")
        if call.arguments
        and "ENCRYPT_MODE" in call.arguments[0]
        and "DECRYPT_MODE" in call.arguments[0]
    ]
    if not directional or len(analyzer.calls("init")) != 1:
        return False
    for method in iter_methods(source):
        if not (method.body_start <= directional[0].start <= method.body_end):
            continue
        calls = tuple(
            match.group("argument").lower()
            for match in re.finditer(
                rf"\b{re.escape(method.name)}\s*\([^;]*?,\s*"
                r"(?P<argument>true|false)\s*\)",
                source,
                flags=re.IGNORECASE,
            )
        )
        if {"true", "false"} <= set(calls):
            return False
    return True


def _java_envelope_has_iv_prefix(
    source: str,
    envelope_names: set[str],
) -> bool:
    """Return whether an envelope stores IV bytes at offset zero."""

    return any(
        re.search(
            rf"System\.arraycopy\s*\(\s*(?:iv|nonce)[A-Za-z0-9_$]*\s*,\s*0\s*,\s*"
            rf"{re.escape(name)}\s*,\s*0\s*,",
            source,
            flags=re.IGNORECASE,
        )
        is not None
        for name in envelope_names
    )


def _java_decryption_result_reaches_output(
    source: str,
    result_name: str,
    call_end: int,
) -> bool:
    """Recognize a direct task-level use of completed decryption bytes."""

    suffix = source[call_end:]
    name = re.escape(result_name.rsplit(".", 1)[-1])
    if (
        re.search(rf"\breturn\s+{name}\b", suffix)
        or re.search(rf"new\s+String\s*\(\s*{name}\b", suffix)
        or re.search(
            rf"\b(?:print|println|write)\s*\([^;]*\b{name}\b",
            suffix,
            flags=re.IGNORECASE,
        )
    ):
        return True

    copied_targets = {
        match.group("target")
        for match in re.finditer(
            rf"System\.arraycopy\s*\(\s*{name}\s*,[^,]*,\s*"
            r"(?P<target>[A-Za-z_$][\w$]*)\s*,",
            suffix,
        )
    }
    return any(
        _java_decryption_result_reaches_output(source, target, call_end)
        for target in copied_targets
    )


def _java_calls_fresh_iv_helper_in_both_directions(source: str) -> bool:
    encrypt = re.search(
        r"init\s*\(\s*Cipher\s*\.\s*ENCRYPT_MODE[^;]*?"
        r"new\s+IvParameterSpec\s*\(\s*(?P<helper>[A-Za-z_$][\w$]*)\s*\(\s*\)\s*\)",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if encrypt is not None:
        suffix = source[encrypt.end() :]
        helper = re.escape(encrypt.group("helper"))
        if re.search(rf"\b{helper}\s*\(\s*\)", suffix) and re.search(
            r"init\s*\(\s*Cipher\s*\.\s*DECRYPT_MODE",
            suffix,
            re.I,
        ):
            return True

    methods = iter_methods(source)
    random_iv_helpers = {
        method.name
        for method in methods
        if re.search(
            r"(?:SecureRandom|Random)\s*\([^)]*\)|\.\s*nextBytes\s*\(",
            method.text,
            flags=re.IGNORECASE,
        )
        and re.search(r"\breturn\b", method.text)
    }
    for helper_name in random_iv_helpers:
        called_from = {
            direction
            for direction in ("encrypt", "decrypt")
            if any(
                direction in method.name.lower()
                and re.search(rf"\b{re.escape(helper_name)}\s*\(", method.text)
                for method in methods
            )
        }
        if called_from == {"encrypt", "decrypt"}:
            return True

    fresh_helpers = {
        method.name
        for method in methods
        if re.search(
            r"init\s*\(\s*Cipher\s*\.\s*ENCRYPT_MODE",
            method.text,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"\.\s*getIV\s*\(",
            method.text,
            flags=re.IGNORECASE,
        )
    }
    for helper_name in fresh_helpers:
        called_from = {
            direction
            for direction in ("encrypt", "decrypt")
            if any(
                direction in method.name.lower()
                and re.search(rf"\b{re.escape(helper_name)}\s*\(", method.text)
                for method in methods
            )
        }
        if called_from == {"encrypt", "decrypt"}:
            return True
    return False


def _java_round_trips_ciphertext_through_raw_string(source: str) -> bool:
    helper_round_trip = bool(
        re.search(
            r"return\s+[^;]*new\s+String\s*\(\s*"
            r"(?:encrypted|cipher)[A-Za-z0-9_$]*\s*\)",
            source,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"(?:substring|split)\s*\([^;]+\)\s*\.\s*getBytes\s*\(",
            source,
            flags=re.IGNORECASE,
        )
    )
    if helper_round_trip:
        return True
    if re.search(
        r"return\s+new\s+String\s*\(\s*"
        r"(?:encrypted|cipher)[A-Za-z0-9_$]*(?:\s*,\s*[^)]+)?\s*\)",
        source,
        flags=re.IGNORECASE,
    ) and re.search(
        r"\.\s*doFinal\s*\(\s*(?:encrypted|cipher)[A-Za-z0-9_$]*"
        r"\s*\.\s*getBytes\s*\(",
        source,
        flags=re.IGNORECASE,
    ):
        return True
    for assignment in re.finditer(
        r"\bString\s+(?P<text>[A-Za-z_$][\w$]*)\s*=\s*new\s+String\s*\(\s*"
        r"(?P<bytes>(?:encrypted|cipher)[A-Za-z0-9_$]*)"
        r"(?:\s*,\s*[^)]+)?\s*\)\s*;",
        source,
        flags=re.IGNORECASE,
    ):
        if re.search(
            rf"\.\s*doFinal\s*\(\s*{re.escape(assignment.group('text'))}"
            r"\s*\.\s*getBytes\s*\(",
            source[assignment.end() :],
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _java_uses_encoded_iv_text_as_raw_bytes(source: str) -> bool:
    assignments = extract_assignments(source)
    raw_text_names = {
        re.sub(r"\s+", "", assignment.name)
        for assignment in assignments
        if re.search(r"\.\s*split\s*\([^;]+\)\s*\[[^\]]+\]\s*\.\s*getBytes\s*\(", assignment.expression)
    }
    return any(
        re.search(
            rf"new\s+IvParameterSpec\s*\(\s*{re.escape(name)}\s*\)",
            source,
            flags=re.IGNORECASE,
        )
        for name in raw_text_names
    )


def _java_decrypts_literal_instead_of_produced_envelope(source: str) -> bool:
    return bool(
        re.search(
            r"byte\s*\[\s*\]\s+(?:combined|envelope)[A-Za-z0-9_$]*\s*=",
            source,
            flags=re.IGNORECASE,
        )
        and re.search(
            r"Base64\s*\.\s*getDecoder\s*\(\s*\)\s*\.\s*decode\s*\(\s*"
            r"[\"'][^\"']+[\"']\s*\)",
            source,
            flags=re.IGNORECASE,
        )
    )


def _java_decrypt_helper_uses_unrelated_literal(source: str) -> bool:
    """Find a decrypt helper whose ciphertext parameter receives a literal."""

    analyzer = analyzer_for("java", source)
    methods = {
        method.name: method
        for method in iter_methods(source)
        if "decrypt" in method.name.lower() or "recover" in method.name.lower()
    }
    if not methods:
        return False

    encrypted_names = {
        assignment.name
        for assignment in extract_assignments(source)
        if (simple_call_name(assignment.expression) or "").lower().startswith(
            ("encrypt", "protect", "seal")
        )
    }
    if not encrypted_names:
        return False

    saw_literal = False
    for name, method in methods.items():
        ciphertext_indexes = {
            index
            for index, parameter in enumerate(method.params)
            if re.search(
                rf"(?:doFinal|decode)\s*\([^)]*\b{re.escape(parameter)}\b",
                method.text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        }
        if not ciphertext_indexes:
            continue
        for call in analyzer.calls(name):
            for index in ciphertext_indexes:
                if len(call.arguments) <= index:
                    continue
                argument = call.arguments[index].strip()
                if any(
                    re.search(rf"\b{re.escape(value)}\b", argument)
                    for value in encrypted_names
                ):
                    return False
                if re.fullmatch(r'"(?:\\.|[^"\\])*"', argument, re.DOTALL):
                    saw_literal = True
    return saw_literal


def _java_gcm_derives_untransported_iv_from_ciphertext(source: str) -> bool:
    if re.search(r"Cipher\s*\.\s*getInstance\s*\(\s*[\"'][^\"']*GCM", source, re.I) is None:
        return False
    encryption_without_parameters = re.search(
        r"init\s*\(\s*Cipher\s*\.\s*ENCRYPT_MODE\s*,\s*[^,;)]+\s*\)",
        source,
        flags=re.IGNORECASE,
    )
    derived_from_ciphertext = re.search(
        r"new\s+GCMParameterSpec\s*\([^;]*?"
        r"(?:encrypted|cipher)[A-Za-z0-9_$]*\s*,\s*0\s*,",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return encryption_without_parameters is not None and derived_from_ciphertext is not None


def _python_assignments(tree: ast.AST) -> dict[str, ast.AST]:
    values: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for name in _python_target_names(targets):
            values[name] = node.value
    return values


def _python_parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _python_calls(tree: ast.AST, simple_name: str) -> tuple[ast.Call, ...]:
    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _python_name(node.func).rsplit(".", 1)[-1] == simple_name
    )


def _python_keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def _python_gcm_tag(call: ast.Call) -> ast.AST | None:
    tag = call.args[1] if len(call.args) >= 2 else _python_keyword(call, "tag")
    if isinstance(tag, ast.Constant) and tag.value is None:
        return None
    return tag


def _python_is_definite_text(
    expression: ast.AST | None,
    assignments: dict[str, ast.AST],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(expression, ast.Constant):
        return isinstance(expression.value, str)
    if isinstance(expression, ast.Name) and expression.id not in seen:
        return _python_is_definite_text(
            assignments.get(expression.id),
            assignments,
            seen | {expression.id},
        )
    if isinstance(expression, ast.BinOp) and isinstance(
        expression.op, (ast.Add, ast.Mult)
    ):
        return _python_is_definite_text(
            expression.left, assignments, seen
        ) or _python_is_definite_text(expression.right, assignments, seen)
    return False


def _python_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _python_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _python_target_names(targets: tuple[ast.AST, ...] | list[ast.AST]) -> set[str]:
    return {
        node.id
        for target in targets
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


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


def _python_call_belongs_to_direction(
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
    direction: str,
) -> bool:
    function = _python_containing_function(call, parents)
    if function is not None and _python_direction(function.name) == direction:
        return True
    current: ast.AST | None = call
    for _ in range(5):
        current = parents.get(current) if current is not None else None
        if current is None:
            break
        if isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
            if current.func.attr == f"{direction}or":
                return True
    return False


def _python_direction(name: str) -> str | None:
    lowered = name.lower()
    if any(marker in lowered for marker in ("decrypt", "recover", "open", "unprotect")):
        return "decrypt"
    if any(marker in lowered for marker in ("encrypt", "protect", "seal")):
        return "encrypt"
    return None


def _python_directional_scopes(tree: ast.AST, direction: str) -> tuple[ast.AST, ...]:
    functions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _python_direction(node.name) == direction
    )
    if functions:
        return functions
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == f"{direction}or"
        for node in ast.walk(tree)
    ):
        return (tree,)
    return ()


def _python_one_shot_directions_use_fresh_keys(
    tree: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    fresh_helpers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(call, ast.Call)
            and _python_name(call.func).endswith(("generate_key", "urandom", "token_bytes"))
            for call in ast.walk(node)
        )
    }
    directions: set[str] = set()
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or _python_name(call.func).rsplit(".", 1)[-1] not in {
            "AESGCM",
            "AESCCM",
            "Fernet",
        }:
            continue
        function = _python_containing_function(call, parents)
        if function is None:
            continue
        direction = _python_direction(function.name)
        if direction is None or not call.args:
            continue
        key = call.args[0]
        if isinstance(key, ast.Call) and _python_name(key.func).rsplit(".", 1)[-1] in fresh_helpers:
            directions.add(direction)
    return directions == {"encrypt", "decrypt"}


def _python_directions_call_fresh_key_method(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    fresh_calls = {"urandom", "token_bytes", "get_random_bytes", "generate_key"}
    fresh_methods = {
        function.name
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and re.search(r"key", node.name, re.IGNORECASE)
        )
        if any(
            isinstance(call, ast.Call)
            and _python_name(call.func).rsplit(".", 1)[-1] in fresh_calls
            and _python_containing_function(call, parents) is function
            for call in ast.walk(function)
        )
    }
    if not fresh_methods:
        return False

    directions: set[str] = set()
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        direction = _python_direction(function.name)
        if direction is None:
            continue
        fresh_names = {
            name
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Call)
            and _python_name(node.value.func).rsplit(".", 1)[-1] in fresh_methods
            for name in _python_target_names(
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
        }
        if fresh_names and any(
            isinstance(call, ast.Call)
            and _python_name(call.func).rsplit(".", 1)[-1]
            in {"AES", "AESGCM", "AESCCM", "Fernet", "Cipher"}
            and any(
                isinstance(child, ast.Name) and child.id in fresh_names
                for argument in call.args
                for child in ast.walk(argument)
            )
            for call in ast.walk(function)
        ):
            directions.add(direction)
    return directions == {"encrypt", "decrypt"}


def _csharp_directional_bodies(source: str) -> dict[str, tuple[str, ...]]:
    analyzer = analyzer_for("csharp", source)
    bodies: dict[str, list[str]] = {"encrypt": [], "decrypt": []}
    for function in getattr(analyzer, "functions", ()):
        lower = function.name.lower()
        for direction in bodies:
            if direction in lower or (direction == "decrypt" and "recover" in lower):
                bodies[direction].append(source[function.start : function.end])
    return {name: tuple(values) for name, values in bodies.items()}


def _csharp_fresh_byte_helpers(source: str) -> set[str]:
    return {
        match.group("name")
        for match in re.finditer(
            r"(?:public\s+|private\s+|protected\s+|internal\s+)?"
            r"(?:static\s+)?byte\s*\[\s*\]\s+(?P<name>[A-Za-z_]\w*)"
            r"\s*\([^)]*\)\s*\{(?P<body>.*?)\}",
            source,
            flags=re.DOTALL,
        )
        if re.search(
            r"RandomNumberGenerator|RNGCryptoServiceProvider|"
            r"\b(?:rng|random[A-Za-z0-9_]*)\s*\.\s*GetBytes\s*\(|"
            r"\b[A-Za-z_]\w*\s*\.\s*Generate(?:Key|IV)\s*\(",
            match.group("body"),
            flags=re.IGNORECASE,
        )
    }


def _csharp_body_uses_fresh_key(bodies: tuple[str, ...], helpers: set[str]) -> bool:
    helper_pattern = "|".join(re.escape(name) for name in sorted(helpers))
    for body in bodies:
        if re.search(
            r"\.\s*Key\s*=\s*(?:"
            + (rf"(?:{helper_pattern})\s*\(" if helper_pattern else r"(?!)")
            + r"|[A-Za-z_]\w*\s*\(\s*\)\s*\.\s*Key\b)",
            body,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _csharp_body_uses_fresh_iv(bodies: tuple[str, ...], helpers: set[str]) -> bool:
    helper_pattern = "|".join(re.escape(name) for name in sorted(helpers))
    if not helper_pattern:
        return False
    return any(
        re.search(
            rf"\b(?:iv|nonce)[A-Za-z0-9_]*\s*=\s*(?:{helper_pattern})\s*\(",
            body,
            flags=re.IGNORECASE,
        )
        is not None
        for body in bodies
    )


def _csharp_writes_placeholder_as_plaintext(source: str) -> bool:
    for allocation in re.finditer(
        r"(?:var|byte\s*\[\s*\])\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
        r"new\s+byte\s*\[[^\]]+\]\s*;",
        source,
        flags=re.IGNORECASE,
    ):
        name = allocation.group("name")
        suffix = source[allocation.end() :]
        if re.search(
            rf"\.\s*Write\s*\(\s*{re.escape(name)}\s*,[^;]+\)"
            rf"[\s\S]*?\breturn\s+{re.escape(name)}\s*;",
            suffix,
            flags=re.IGNORECASE,
        ):
            return True
    return False


__all__ = ("symmetric_flow_failure",)
