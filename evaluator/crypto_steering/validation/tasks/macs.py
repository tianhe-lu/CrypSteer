"""Connected MAC-verification operation contracts."""

from __future__ import annotations

import ast
import re

from crypto_steering.validation.java.source import iter_methods, strip_java_comments
from crypto_steering.validation.models import Language
from crypto_steering.validation.source.index import analyzer_for
from crypto_steering.validation.tasks.shared.python_dataflow import (
    bytes_passed_to_encoded_parameter,
)
_TAG_NAME = re.compile(
    r"(?:mac|hmac|tag|token|digest|sig(?:nature)?|"
    r"auth(?:entication)?[_$]?code|verification[_$]?code|header)",
    flags=re.IGNORECASE,
)
_MAC_TERMINAL = re.compile(
    r"(?:\.\s*(?:doFinal|digest|hexdigest|ComputeHash|GetHashAndReset)\s*\(|"
    r"\b(?:hmac\.)?digest\s*\()",
    flags=re.IGNORECASE,
)
_COMPUTED_HELPER = re.compile(
    r"\b(?:compute|calculate|create|generate|sign)[\w$]*"
    r"(?:mac|hmac|tag|signature|auth(?:entication)?[_$]?code)\s*\(",
    flags=re.IGNORECASE,
)


def mac_generation_failure(language: Language, source: str) -> str | None:
    """Return a definite failure of a MAC-generation terminal operation."""

    if language == "typescript" and re.search(
        r"\b(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*digest\s*\([^;]*\)\s*;"
        r"\s*return\s+(?P=receiver)\s*\.\s*digest\s*\(",
        source,
        flags=re.DOTALL,
    ):
        return "the returned MAC uses a hash object after it was already finalized"
    if language == "csharp" and _csharp_mac_generation_uses_empty_message(source):
        return "the returned MAC authenticates an empty buffer instead of the task message"
    if language == "csharp" and _csharp_consumes_key_instead_of_computed_mac(source):
        return "the computed MAC is discarded and the HMAC key is consumed as the tag"
    return None


def _csharp_consumes_key_instead_of_computed_mac(source: str) -> bool:
    """Find an invoked wrapper that exposes ``HMAC.Key`` as its generated tag.

    Merely declaring a key accessor is harmless: a program remains valid when
    another invoked route consumes the bytes returned by ``ComputeHash`` (or a
    wrapper around it).  The failure is the narrower case where the computed
    result is assigned and abandoned while the value returned by a key
    accessor is the value that is subsequently consumed.
    """

    analyzer = analyzer_for("csharp", source)
    functions = tuple(getattr(analyzer, "functions", ()))
    key_helpers = {
        function.name
        for function in functions
        if re.search(
            r"\breturn\s+[^;]*\.\s*Key\s*;",
            source[function.start : function.end],
            flags=re.IGNORECASE,
        )
    }
    if not key_helpers:
        return False

    assignments = tuple(getattr(analyzer, "assignments", ()))

    def containing_assignment(call: object) -> object | None:
        return next(
            (
                assignment
                for assignment in assignments
                if assignment.start <= getattr(call, "start") <= assignment.end
            ),
            None,
        )

    def assigned_value_is_used(assignment: object) -> bool:
        names = re.findall(
            r"[A-Za-z_$][A-Za-z0-9_$]*",
            str(getattr(assignment, "name")),
        )
        if not names:
            return False
        name = names[-1]
        return re.search(
            rf"(?<![\w$]){re.escape(name)}(?![\w$])",
            source[int(getattr(assignment, "end")) :],
        ) is not None

    consumed_key = False
    for call in analyzer.calls(*key_helpers):
        if any(function.start <= call.start <= function.end for function in functions):
            continue
        assignment = containing_assignment(call)
        if assignment is not None and assigned_value_is_used(assignment):
            consumed_key = True
            break
    if not consumed_key:
        return False

    producer_names = _c_style_mac_producer_names(
        source,
        _c_style_functions("csharp", analyzer),
    )
    for call in analyzer.calls(*producer_names):
        if any(function.start <= call.start <= function.end for function in functions):
            continue
        assignment = containing_assignment(call)
        if assignment is None:
            # A wrapper result used directly as an argument or returned from a
            # caller is already consumed.  A bare statement is not.
            statement_start = max(
                source.rfind(";", 0, call.start),
                source.rfind("{", 0, call.start),
            ) + 1
            prefix = source[statement_start : call.start]
            if re.search(r"\breturn\b|\([^;{}]*$", prefix):
                return False
            continue
        if assigned_value_is_used(assignment):
            return False

    # ``ComputeHash`` updates the HMAC object's Hash property.  An invoked
    # accessor that returns that property is therefore another valid way to
    # consume the computed result.
    hash_helpers = {
        function.name
        for function in functions
        if re.search(
            r"\breturn\s+[^;]*\.\s*Hash\s*;",
            source[function.start : function.end],
            flags=re.IGNORECASE,
        )
    }
    for call in analyzer.calls(*hash_helpers) if hash_helpers else ():
        if any(function.start <= call.start <= function.end for function in functions):
            continue
        assignment = containing_assignment(call)
        if assignment is None or assigned_value_is_used(assignment):
            return False
    return True


def _csharp_mac_generation_uses_empty_message(source: str) -> bool:
    """Find a returned MAC helper that substitutes a fresh empty buffer."""

    analyzer = analyzer_for("csharp", source)
    functions = {
        function.name: source[function.start : function.end]
        for function in getattr(analyzer, "functions", ())
    }
    mac_helpers = {
        name
        for name, body in functions.items()
        if re.search(r"\b(?:ComputeHash|GetHashAndReset)\s*\(", body)
    }
    changed = True
    while changed:
        changed = False
        for name, body in functions.items():
            if name in mac_helpers:
                continue
            if any(
                re.search(rf"\b{re.escape(helper)}\s*\(", body)
                for helper in mac_helpers
            ):
                mac_helpers.add(name)
                changed = True

    empty = r"(?:new\s+byte\s*\[\s*0\s*\]|Array\s*\.\s*Empty\s*<\s*byte\s*>\s*\(\s*\))"
    bad_helpers: set[str] = set()
    for name in mac_helpers:
        body = functions[name]
        if not re.search(rf"\breturn\b[^;]*{empty}", body, re.IGNORECASE):
            continue
        if any(
            re.search(
                rf"\b{re.escape(name)}\s*\(\s*\)",
                other,
                re.IGNORECASE,
            )
            for other_name, other in functions.items()
            if other_name != name
        ) or re.search(rf"\b{re.escape(name)}\s*\(\s*\)", source[: source.find(body)]):
            bad_helpers.add(name)
    if not bad_helpers:
        return False

    def empty_expression(value: str) -> bool:
        return re.fullmatch(rf"\s*{empty}\s*", value, re.IGNORECASE) is not None

    # One invoked, non-empty MAC route is sufficient.  Generated programs may
    # also contain an unrelated demonstration helper that hashes an empty
    # buffer; that extra route does not erase the completed task operation.
    for call in analyzer.calls():
        if call.simple_name not in mac_helpers or not call.arguments:
            continue
        definition = next(
            (
                function
                for function in getattr(analyzer, "functions", ())
                if function.name == call.simple_name
            ),
            None,
        )
        if definition is not None and definition.start <= call.start <= definition.end:
            continue
        if not empty_expression(call.arguments[0]):
            return False
    return True


def _python_verification_branch_is_unreachable(source: str) -> bool:
    """Find a command-line verification flag tested but never declared."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    tested_verify_flag = bool(
        re.search(r"['\"]--verify['\"]\s+in\s+(?:sys\.)?argv", source)
    )
    if not tested_verify_flag:
        return False
    declared_flags = {
        value.value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        for value in call.args
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }
    return "--verify" not in declared_flags


def _python_mac_bytes_passed_to_encoded_parameter(
    source: str,
) -> tuple[str, str] | None:
    """Apply the shared bytes-to-``str.encode`` check to an HMAC route."""

    if re.search(r"\b(?:hmac\s*\.\s*new|HMAC)\s*\(", source) is None:
        return None
    direct = bytes_passed_to_encoded_parameter(source)
    if direct is not None:
        return direct
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
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
        encoded = {
            call.func.value.id
            for call in ast.walk(initializer)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "encode"
            and isinstance(call.func.value, ast.Name)
        }
        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == class_node.name
            ):
                continue
            for index, parameter in enumerate(parameters[offset:]):
                if parameter.arg not in encoded or index >= len(call.args):
                    continue
                argument = call.args[index]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, bytes):
                    return class_node.name, parameter.arg
                if isinstance(argument, ast.Name):
                    assignments = [
                        node.value
                        for node in ast.walk(tree)
                        if isinstance(node, (ast.Assign, ast.AnnAssign))
                        and getattr(node, "lineno", 0) < getattr(call, "lineno", 0)
                        and any(
                            isinstance(target, ast.Name) and target.id == argument.id
                            for target in (
                                node.targets if isinstance(node, ast.Assign) else (node.target,)
                            )
                        )
                    ]
                    if assignments and isinstance(assignments[-1], ast.Constant) and isinstance(
                        assignments[-1].value, bytes
                    ):
                        return class_node.name, parameter.arg
    return None


def mac_verification_failure(language: Language, source: str) -> str | None:
    """Return a definite unusable or disconnected MAC-verification flow."""

    java_ignored_tag = False

    if language == "typescript" and re.search(
        r"\bcreateHmac\s*\(\s*(?:[A-Za-z_$][\w$]*|[\"'])",
        source,
    ):
        analyzer = analyzer_for("typescript", source)
        selected_algorithms = {
            value.lower()
            for call in analyzer.calls("createHmac")
            if call.arguments
            for value in analyzer.resolve_strings(call.arguments[0], call.start).values
        }
        selected_algorithms.update(
            value.lower()
            for value in re.findall(
                r"\balgorithm(?:\s*:\s*[A-Za-z_$][\w$]*)?\s*=\s*"
                r"[\"']([^\"']+)[\"']",
                source,
                flags=re.IGNORECASE,
            )
        )
        if any(re.match(r"^hmac(?:[-_]|sha)", value) for value in selected_algorithms):
            return (
                "Node createHmac expects a digest name such as sha256, "
                "not an hmac-prefixed algorithm name"
            )
        if _typescript_reuses_finalized_hmac(source):
            return "the verification route reuses an HMAC after digest finalized it"
    if language == "python":
        if _python_verification_branch_is_unreachable(source):
            return "the MAC verification branch cannot be selected by the declared command-line interface"
        encoded_parameter = _python_mac_bytes_passed_to_encoded_parameter(source)
        if encoded_parameter is not None:
            function, parameter = encoded_parameter
            return f"{function} receives bytes for {parameter}, then calls str.encode on it"
        unavailable_comparison = _python_unavailable_mac_comparison(source)
        if unavailable_comparison is not None:
            return unavailable_comparison
        unusable_api = _python_unusable_mac_api(source)
        if unusable_api is not None:
            return unusable_api
        argument_failure = _python_hmac_digestmod_failure(source)
        if argument_failure is not None:
            return argument_failure
        value_failure = _python_hmac_value_failure(source)
        if value_failure is not None:
            return value_failure
        representation_failure = _python_mac_representation_failure(source)
        if representation_failure is not None:
            return representation_failure
        message_flow_failure = _python_mac_message_flow_failure(source)
        if message_flow_failure is not None:
            return message_flow_failure
        if _python_compares_mac_to_task_message(source):
            return "the verifier compares its MAC with the task message instead of a tag"
        if _python_compares_mac_to_key_material(source):
            return "the verifier compares its MAC with key material instead of a tag"
        if _python_hmac_authenticates_a_computed_mac(source):
            return "the verifier authenticates a MAC value instead of the task message"
        if _python_uses_tag_as_hmac_message(source):
            return "the verifier authenticates a tag value instead of the task message"
        if _python_compared_hmacs_authenticate_different_messages(source):
            return "the compared HMAC values authenticate different messages"
        ignored_message = _python_hmac_ignores_message_parameter(source)
        if ignored_message is not None:
            return ignored_message
    if language == "csharp":
        if re.search(r"\b(?:true\s*==\s*true|true)\s*\|\|", source, re.IGNORECASE):
            return "a constant-true branch bypasses the MAC comparison"
        if _csharp_uses_reference_equality_for_mac_bytes(source):
            return "C# byte-array equality does not compare MAC tag contents"
        if _csharp_verifier_ignores_supplied_tag(source):
            return "the verifier ignores its supplied tag"
    if language == "java":
        if _java_uses_byte_array_object_equality(source):
            return "Java byte-array object equality does not compare MAC tag contents"
        if _java_discards_mac_result_into_input_buffer(source):
            return "the returned MAC is discarded and an unchanged input buffer is compared"
        if _java_uses_tag_as_mac_message(source):
            return "the verifier authenticates a tag value instead of the task message"
        representation_failure = _java_mac_representation_failure(source)
        if representation_failure is not None:
            return representation_failure
        java_ignored_tag = _java_verifier_ignores_supplied_tag(source)
        if java_ignored_tag:
            return "the verifier ignores its supplied tag"
        elementwise_failure = _java_elementwise_mac_comparison_failure(source)
        if elementwise_failure is not None:
            return elementwise_failure
        if _java_compares_mac_values_for_different_messages(source):
            return "the compared MAC values authenticate different messages"
        if _java_decoded_supplied_tag_is_unused(source):
            return "the verifier decodes its supplied tag but excludes it from the integrity comparison"
        if _java_compares_mac_values_for_different_keys(source):
            return "the compared MAC values use different keys"
    if language == "typescript":
        if _typescript_compares_hmac_to_key_material(source):
            return "the supplied MAC tag is key material rather than a tag"
        if _typescript_compares_mac_values_for_different_keys(source):
            return "the compared HMAC values use different keys"
        if _typescript_compares_mac_values_for_different_messages(source):
            return "the compared MAC values authenticate different messages"
        representation_failure = _typescript_mac_representation_failure(source)
        if representation_failure is not None:
            return representation_failure
        message_failure = _typescript_mac_message_flow_failure(source)
        if message_failure is not None:
            return message_failure

    if not _comparison_connects_received_tag(language, source):
        if java_ignored_tag:
            return "the verifier ignores its supplied tag and returns only the computed MAC"
        return "the received or expected tag is disconnected from the MAC comparison"
    return None


def _python_unavailable_mac_comparison(source: str) -> str | None:
    """Reject ``compare_digest`` on PyCA's HMAC module.

    Constant-time comparison is available from Python's standard ``hmac`` or
    ``secrets`` modules.  ``cryptography.hazmat.primitives.hmac`` instead
    exposes the stateful ``HMAC`` class and has no ``compare_digest`` member.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    pyca_hmac_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "cryptography.hazmat.primitives"
        for alias in node.names
        if alias.name == "hmac"
    }
    pyca_hmac_names.update(
        alias.asname or alias.name.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "cryptography.hazmat.primitives.hmac"
    )
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compare_digest"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in pyca_hmac_names
        for node in ast.walk(tree)
    ):
        return "cryptography.hazmat.primitives.hmac has no compare_digest operation"
    return None


def _python_unusable_mac_api(source: str) -> str | None:
    """Reject source-provable Python API failures on the selected MAC route."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    hmac_scopes = {
        _python_scope(call, parents, tree)
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and _python_name(call.func) in {"hmac.new", "hmac.HMAC", "HMAC"}
    }
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "hexdigest"
            and (call.args or call.keywords)
            and _python_scope(call, parents, tree) in hmac_scopes
        ):
            return "hash and HMAC hexdigest methods do not accept arguments"
    if any(
        isinstance(node, ast.Subscript)
        and _python_name(node.value) == "hashlib.algorithms_available"
        for node in ast.walk(tree)
    ):
        return "hashlib.algorithms_available is a set and cannot be indexed by algorithm name"

    resolver = _PythonValueKindResolver(tree, parents)
    for node in ast.walk(tree):
        operands: tuple[ast.AST, ast.AST] | None = None
        if isinstance(node, ast.Compare) and node.comparators:
            operands = node.left, node.comparators[0]
        elif (
            isinstance(node, ast.Call)
            and _python_name(node.func).endswith("compare_digest")
            and len(node.args) >= 2
        ):
            operands = node.args[0], node.args[1]
        if operands is None:
            continue
        scope = _python_scope(node, parents, tree)
        if any(
            (
                _python_conditioned_name_kind(
                    operand,
                    node,
                    tree,
                    parents,
                    resolver,
                )
                or resolver.kind(operand, scope)
            )
            == "mac_object"
            for operand in operands
        ):
            return "MAC verification compares an unfinished HMAC object instead of its digest"
    return None


def _python_hmac_value_failure(source: str) -> str | None:
    """Reject source-provable non-byte or self-derived HMAC operands."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    resolver = _PythonValueKindResolver(tree, parents)
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        name = _python_name(call.func)
        scope = _python_scope(call, parents, tree)
        if name in {"hmac.new", "new"}:
            key = call.args[0] if call.args else _keyword_value(call, "key")
            message = (
                call.args[1] if len(call.args) >= 2 else _keyword_value(call, "msg")
            )
            if message is not None and resolver.kind(message, scope) == "hash_object":
                return "hmac.new requires message bytes, not a live hash object"
            if key is not None:
                key_names = {
                    node.id for node in ast.walk(key) if isinstance(node, ast.Name)
                }
                if any(
                    re.search(r"(?:mac|tag|digest|signature)", value, re.I)
                    and not re.search(r"key", value, re.I)
                    for value in key_names
                ) and not _python_is_plain_hash_kind(resolver.kind(key, scope)):
                    return "the verifier uses a generated MAC value as the HMAC key"
        if name.startswith("hashlib.") and call.args:
            if _python_value_base_kind(resolver.kind(call.args[0], scope)) == "text":
                return f"{name} requires bytes-like message input, not text"

    if any(
        isinstance(call, ast.Call)
        and _python_name(call.func) in {"binascii.hexlify", "hexlify"}
        and call.args
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "encode"
            and isinstance(child.func.value, ast.Name)
            and re.search(r"(?:expected|received|provided|tag|mac)", child.func.value.id, re.I)
            for child in ast.walk(call.args[0])
        )
        and isinstance(parents.get(call), ast.Attribute)
        and parents[call].attr == "decode"
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
    ):
        return "the verifier hex-encodes the text representation of the supplied tag"
    return None


def _typescript_reuses_finalized_hmac(source: str) -> bool:
    """Find a second update/digest chain on an already-finalized HMAC."""

    analyzer = analyzer_for("typescript", source)
    setups: dict[str, list[int]] = {}
    for assignment in getattr(analyzer, "assignments", ()):
        if not re.search(r"\bcreateHmac\s*\(", assignment.expression):
            continue
        setups.setdefault(re.sub(r"\s+", "", assignment.name), []).append(
            assignment.start
        )
    digests_by_setup: dict[tuple[str | None, str, int], list[object]] = {}
    for call in analyzer.calls("digest"):
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        if "." not in call.callee:
            containing_assignment = next(
                (
                    assignment
                    for assignment in getattr(analyzer, "assignments", ())
                    if assignment.start <= call.start <= assignment.end
                ),
                None,
            )
            chained = (
                re.search(
                    r"\b(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*update\s*\(.*\)"
                    r"\s*\.\s*digest\s*\(",
                    containing_assignment.expression,
                    flags=re.DOTALL,
                )
                if containing_assignment is not None
                else None
            )
            if chained is not None:
                receiver = chained.group("receiver")
        setup = max(
            (position for position in setups.get(receiver, ()) if position < call.start),
            default=None,
        )
        if setup is None:
            continue
        key = (analyzer.containing_function_name(call.start), receiver, setup)
        digests_by_setup.setdefault(key, []).append(call)

    first_digest_results: set[str] = set()
    for calls in digests_by_setup.values():
        first = min(calls, key=lambda call: call.start)
        first_digest_results.update(
            assignment.name
            for assignment in getattr(analyzer, "assignments", ())
            if assignment.start <= first.start <= assignment.end
        )

    complete_first_generation = False
    for (function, receiver, setup), calls in digests_by_setup.items():
        first = min(calls, key=lambda call: call.start)
        prior_message_update = any(
            call.start < first.start
            and analyzer.containing_function_name(call.start) == function
            and re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0] == receiver
            for call in analyzer.calls("update")
        )
        if prior_message_update:
            complete_first_generation = True
            continue
        later_digests = tuple(
            call for call in calls if call.start > first.end
        )
        if later_digests and any(
            first.end < call.start < later_digest.start
            and analyzer.containing_function_name(call.start) == function
            and re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0] == receiver
            for later_digest in later_digests
            for call in analyzer.calls("update")
        ):
            return True
        if re.search(
            rf"\b{re.escape(receiver)}\s*\.\s*update\s*\([^;]*?\)"
            r"\s*\.\s*digest\s*\(",
            source[first.end :],
            flags=re.DOTALL,
        ):
            return True

    if complete_first_generation:
        return False

    # A later illustrative misuse does not invalidate an earlier complete
    # comparison route.  Keep this check existential, like the task contract.
    for comparison in analyzer.calls("timingSafeEqual", "equals"):
        operands = comparison.arguments
        if len(operands) == 1 and "." in comparison.callee:
            operands = (comparison.callee.rsplit(".", 1)[0], operands[0])
        if len(operands) < 2:
            continue
        referenced = [
            {
                name
                for name in first_digest_results
                if re.search(
                    rf"(?<![.$\w]){re.escape(name)}(?![\w$])",
                    operand,
                )
            }
            for operand in operands[:2]
        ]
        if all(referenced):
            return False
    for calls in digests_by_setup.values():
        ordered = sorted(calls, key=lambda call: call.start)
        for call in ordered[1:]:
            statement_start = max(
                source.rfind(";", 0, call.start),
                source.rfind("\n", 0, call.start),
            ) + 1
            statement_end = source.find(";", call.end)
            statement = source[
                statement_start : len(source) if statement_end < 0 else statement_end
            ]
            if re.search(
                r"timingSafeEqual|(?:===|!==|==|!=)",
                statement,
                flags=re.IGNORECASE,
            ):
                return True

    for match in re.finditer(
        r"\b(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*update\s*\([^;]*?\)"
        r"\s*\.\s*digest\s*\(",
        source,
        flags=re.DOTALL,
    ):
        receiver = re.escape(match.group("receiver"))
        second = re.search(
            rf"\b{receiver}\s*\.\s*update\s*\([^;]*?\)\s*\.\s*digest\s*\(",
            source[match.end() :],
            flags=re.DOTALL,
        )
        if second is None:
            continue
        second_start = match.end() + second.start()
        if analyzer.containing_function_name(match.start()) != analyzer.containing_function_name(
            second_start
        ):
            continue
        between = source[match.end() : second_start]
        if re.search(
            rf"\b(?:const|let|var)\s+{receiver}\s*=\s*(?:crypto\s*\.\s*)?createHmac\s*\(",
            between,
            flags=re.IGNORECASE,
        ):
            continue
        if second is not None:
            return True
    return False


def _java_uses_byte_array_object_equality(source: str) -> bool:
    """Return whether ``equals`` is invoked directly on ``doFinal`` bytes.

    A wrapper such as ``bytesToHex(mac.doFinal(...)).equals(tag)`` compares
    the wrapper result, not the Java byte-array object.  Matching from the
    terminal call itself is therefore too greedy for nested expressions.
    """

    analyzer = analyzer_for("java", source)
    raw_mac_values = set(_java_mac_assignment_messages(source))
    if raw_mac_values and any(
        match.group("left") in raw_mac_values
        and match.group("right") in raw_mac_values
        for match in re.finditer(
            r"\b(?P<left>[A-Za-z_$][\w$]*)\s*(?:==|!=)\s*"
            r"(?P<right>[A-Za-z_$][\w$]*)\b",
            source,
        )
    ):
        return True
    for call in analyzer.calls("equals"):
        receiver = (
            call.callee.rsplit(".", 1)[0]
            if "." in call.callee
            else _receiver_before_call(source, call.start)
        )
        compact = re.sub(r"\s+", "", receiver)
        if compact in raw_mac_values:
            return True
        if (
            re.search(r"\.getBytes\(", compact)
            and _identifier_names(call.arguments[0]) & raw_mac_values
        ):
            return True
        if re.fullmatch(
            r"[A-Za-z_$][\w$]*\.doFinal\(.*\)",
            compact,
            flags=re.DOTALL,
        ):
            return True
    return False


def _java_discards_mac_result_into_input_buffer(source: str) -> bool:
    """Find ``Mac.doFinal(input)`` used as though it filled ``input``."""

    for match in re.finditer(
        r"(?m)^\s*[A-Za-z_$][\w$]*\s*\.\s*doFinal\s*\(\s*"
        r"(?P<input>[A-Za-z_$][\w$]*(?:\s*\.\s*array\s*\(\s*\))?)"
        r"\s*\)\s*;",
        source,
    ):
        root = re.match(r"[A-Za-z_$][\w$]*", match.group("input"))
        if root is None:
            continue
        name = root.group(0)
        if re.search(
            rf"(?:equals|isEqual)\s*\([^;]*\b{re.escape(name)}\b|"
            rf"\b{re.escape(name)}\b[^;]*\.\s*equals\s*\(",
            source[match.end() :],
            flags=re.IGNORECASE | re.DOTALL,
        ):
            return True
    return False


def _java_uses_tag_as_mac_message(source: str) -> bool:
    """Find a MAC terminal whose only message is named as an expected tag."""

    for match in re.finditer(
        r"\.\s*doFinal\s*\(\s*(?P<message>[A-Za-z_$][\w$]*)\s*\)",
        source,
    ):
        message = match.group("message")
        if re.search(
            r"(?:message|data|payload|body|content)",
            message,
            re.IGNORECASE,
        ):
            continue
        if re.search(
            r"(?:tag|signature|mac|digest|expected[A-Za-z0-9_$]*bytes)",
            message,
            re.IGNORECASE,
        ) is None:
            continue
        if re.search(
            r"\.\s*doFinal\s*\(\s*(?:data|message|payload|body|request)"
            r"[A-Za-z0-9_$]*\s*\)",
            source,
            flags=re.IGNORECASE,
        ) is None:
            return True
    return False


def _csharp_uses_reference_equality_for_mac_bytes(source: str) -> bool:
    for match in re.finditer(
        r"\b(?:ComputeHash|GetHashAndReset)\s*\([^;]*?\)\s*==\s*"
        r"[A-Za-z_$][\w$]*",
        source,
        flags=re.DOTALL,
    ):
        prefix = source[max(0, match.start() - 64) : match.start()]
        if re.search(
            r"Convert\s*\.\s*ToBase64String\s*\(\s*"
            r"(?:[A-Za-z_$][\w$]*\s*\.\s*)?$",
            prefix,
        ):
            continue
        return True
    return False


def _csharp_verifier_ignores_supplied_tag(source: str) -> bool:
    analyzer = analyzer_for("csharp", source)
    global_assignments = [
        (
            set(re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", assignment.name)),
            assignment.expression,
        )
        for assignment in getattr(analyzer, "assignments", ())
    ]
    global_computed, global_external = _value_roles(global_assignments, set())
    global_computed = _expanded_c_style_computed_values(
        source,
        global_assignments,
        global_computed,
        _c_style_mac_producer_names(
            source,
            _c_style_functions("csharp", analyzer),
        ),
    )
    global_external = _expanded_c_style_external_values(
        source,
        global_assignments,
        global_external,
    )
    global_external.difference_update(global_computed)
    comparison_calls = {
        "Equals",
        "SequenceEqual",
        "FixedTimeEquals",
        "CompareMacTags",
        "CompareTags",
    }
    for function in getattr(analyzer, "functions", ()):
        if not _looks_like_verification_name(function.name):
            continue
        external_parameters = {
            parameter
            for parameter in function.parameters
            if _external_name_priority(parameter) == 1
        }
        parameter_indexes = {
            parameter: index
            for index, parameter in enumerate(function.parameters)
            if _external_name_priority(parameter) is not None
        }
        for call in analyzer.calls(function.name):
            if function.start <= call.start <= function.end:
                continue
            for parameter, index in parameter_indexes.items():
                if index >= len(call.arguments):
                    continue
                argument = call.arguments[index]
                if _expression_is_computed(argument, global_computed):
                    continue
                if _expression_is_external_tag(argument, global_external):
                    external_parameters.add(parameter)
        if not external_parameters:
            continue
        body = source[function.start : function.end]
        if re.search(r"\b(?:ComputeHash|GetHashAndReset)\s*\(", body) is None:
            continue
        if re.search(
            r"(?:ComputeHash|GetHashAndReset)\s*\([^;]*?\)\s*"
            r"\.\s*SequenceEqual\s*\([^;]*?\b"
            r"[A-Za-z_]\w*(?:tag|mac|signature)\w*\b",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            continue
        body_statements = body.split("{", 1)[1] if "{" in body else body
        explicit_tags = {
            parameter
            for parameter in function.parameters
            if _is_external_tag_name(parameter)
        }
        if explicit_tags and any(
            re.search(rf"\b{re.escape(parameter)}\b", body_statements) is None
            for parameter in explicit_tags
        ):
            return True
        scoped_assignments = [
            (
                set(re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", assignment.name)),
                assignment.expression,
            )
            for assignment in getattr(analyzer, "assignments", ())
            if function.start <= assignment.start <= function.end
        ]
        supplied_values = _parameter_value_flow(
            scoped_assignments,
            external_parameters,
        )
        computed_values, _ = _value_roles(scoped_assignments, set())
        connected = False
        for call in analyzer.calls():
            if not (function.start <= call.start <= function.end):
                continue
            if (
                call.simple_name not in comparison_calls
                and not _looks_like_verification_name(call.simple_name)
            ):
                continue
            expressions = tuple(call.arguments)
            if len(expressions) == 1 and "." in call.callee:
                expressions = (call.callee.rsplit(".", 1)[0], *expressions)
            names = [_identifier_names(value) for value in expressions]
            if (
                any(value & supplied_values for value in names)
                and any(value & computed_values for value in names)
            ):
                connected = True
                break
        for comparison in re.finditer(
            r"(?P<left>[^\n;=!]+?)\s*(?:==|!=)\s*(?P<right>[^\n;{]+)",
            body,
        ):
            left = _identifier_names(comparison.group("left"))
            right = _identifier_names(comparison.group("right"))
            if (left & supplied_values and right & computed_values) or (
                right & supplied_values and left & computed_values
            ):
                connected = True
                break
        if not connected:
            return True
    return False


def _java_elementwise_mac_comparison_failure(source: str) -> str | None:
    computed = _java_mac_assignment_messages(source)
    for left_name, right_name in _java_elementwise_comparisons(source):
        left = computed.get(left_name)
        right = computed.get(right_name)
        if left is None or right is None:
            continue
        if left != right:
            return "the compared MAC values authenticate different messages"
    return None


def _java_compares_mac_values_for_different_messages(source: str) -> bool:
    messages = _java_mac_assignment_messages(source)
    if _java_candidate_loop_includes_reference_message(source, set(messages.values())):
        return False
    assignments = tuple(
        (match.group("name"), match.group("expression"))
        for match in re.finditer(
            r"\b(?:byte\s*\[\s*\]|String|var)\s+"
            r"(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
            r"(?P<expression>[^;]+);",
            source,
            flags=re.DOTALL,
        )
    )
    changed = True
    while changed:
        changed = False
        for name, expression in assignments:
            origins = {
                messages[value]
                for value in _identifier_names(expression)
                if value in messages
            }
            if len(origins) == 1 and name not in messages:
                messages[name] = origins.pop()
                changed = True

    analyzer = analyzer_for("java", source)
    for call in analyzer.calls("equals", "isEqual"):
        if not call.arguments:
            continue
        receiver = (
            call.callee.rsplit(".", 1)[0]
            if "." in call.callee
            else _receiver_before_call(source, call.start)
        )
        left = next((name for name in _identifier_names(receiver) if name in messages), None)
        right = next(
            (name for name in _identifier_names(call.arguments[0]) if name in messages),
            None,
        )
        if left is not None and right is not None and messages[left] != messages[right]:
            return True
    return False


def _java_candidate_loop_includes_reference_message(
    source: str,
    reference_messages: set[str],
) -> bool:
    """Accept a verification loop when one candidate is the tagged message.

    A verifier may deliberately test several candidate messages against one
    tag.  Different candidates are expected to fail; the task is still
    completed when the candidate collection also contains the message from
    which the reference tag was calculated.
    """

    for array in re.finditer(
        r"\bString\s*\[\s*\]\s*(?P<array>[A-Za-z_$][\w$]*)\s*=\s*"
        r"\{(?P<values>[^}]*)\}\s*;",
        source,
        flags=re.DOTALL,
    ):
        loop = re.search(
            rf"\bfor\s*\(\s*String\s+(?P<item>[A-Za-z_$][\w$]*)\s*:\s*"
            rf"{re.escape(array.group('array'))}\s*\)",
            source[array.end() :],
        )
        if loop is None:
            continue
        item = loop.group("item")
        loop_source = source[array.end() + loop.start() :]
        if re.search(
            rf"\.\s*doFinal\s*\(\s*{re.escape(item)}\s*\.\s*getBytes\s*\(",
            loop_source,
        ) is None:
            continue
        candidates = {
            re.sub(r"\s+", "", literal) + ".getBytes()"
            for literal in re.findall(r'"(?:\\.|[^"\\])*"', array.group("values"))
        }
        if candidates & reference_messages:
            return True
    return False


def _java_compares_mac_values_for_different_keys(source: str) -> bool:
    """Trace compared ``doFinal`` values to their latest ``Mac.init`` key."""

    aliases = {
        match.group("name"): match.group("value").strip()
        for match in re.finditer(
            r"\b(?:byte\s*\[\s*\]|String|var)\s+"
            r"(?P<name>[A-Za-z_$][\w$]*)\s*=\s*(?P<value>[^;]+)\s*;",
            source,
            flags=re.DOTALL,
        )
    }

    def key_origins(expression: str, seen: frozenset[str] = frozenset()) -> set[str]:
        compact = re.sub(r"\s+", "", expression)
        if re.fullmatch(r"[A-Za-z_$][\w$]*", compact):
            if compact in seen:
                return {f"name:{compact}"}
            value = aliases.get(compact)
            if value is None:
                return {f"name:{compact}"}
            dependencies = {
                name for name in _identifier_names(value) if name in aliases
            }
            if len(dependencies) == 1:
                origins: set[str] = set()
                for dependency in dependencies:
                    origins.update(
                        key_origins(
                            dependency,
                            seen | frozenset({compact}),
                        )
                    )
                return origins
            if len(dependencies) > 1:
                return {f"origin:{compact}"}
            if re.fullmatch(r"(?:new)?String\(?.*", compact, re.IGNORECASE):
                return {f"value:{compact}"}
            return {f"origin:{compact}"}
        dependencies = {
            name for name in _identifier_names(expression) if name in aliases
        }
        if dependencies:
            origins: set[str] = set()
            for dependency in dependencies:
                origins.update(key_origins(dependency, seen))
            return origins
        return {f"value:{compact}"}

    key_specs: dict[str, list[tuple[int, str]]] = {}
    for match in re.finditer(
        r"\b(?P<name>[A-Za-z_$][\w$]*)\s*=\s*new\s+SecretKeySpec\s*\(\s*"
        r"(?P<key>[^,]+),",
        source,
        flags=re.DOTALL,
    ):
        key_specs.setdefault(match.group("name"), []).append(
            (match.start(), re.sub(r"\s+", "", match.group("key")))
        )

    initializations: dict[str, list[tuple[int, str]]] = {}
    analyzer = analyzer_for("java", source)
    for call in analyzer.calls("init"):
        if not call.arguments:
            continue
        receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
        expression = re.sub(r"\s+", "", call.arguments[0])
        if re.fullmatch(r"[A-Za-z_$][\w$]*", expression):
            candidates = [
                value
                for position, value in key_specs.get(expression, ())
                if position < call.start
            ]
            if candidates:
                expression = candidates[-1]
        elif expression.startswith("newSecretKeySpec("):
            expression = expression[len("newSecretKeySpec(") :].split(",", 1)[0]
        initializations.setdefault(receiver, []).append((call.start, expression))

    result_keys: dict[str, str] = {}
    for match in re.finditer(
        r"\bbyte\s*\[\s*\]\s*(?P<result>[A-Za-z_$][\w$]*)\s*=\s*"
        r"(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*doFinal\s*\(",
        source,
    ):
        candidates = [
            key
            for position, key in initializations.get(match.group("receiver"), ())
            if position < match.start()
        ]
        if candidates:
            result_keys[match.group("result")] = candidates[-1]

    for call in analyzer.calls("equals", "isEqual"):
        expressions = list(call.arguments)
        if len(expressions) == 1 and "." in call.callee:
            expressions.insert(0, call.callee.rsplit(".", 1)[0])
        if len(expressions) < 2:
            continue
        left = next(
            (name for name in _identifier_names(expressions[0]) if name in result_keys),
            None,
        )
        right = next(
            (name for name in _identifier_names(expressions[1]) if name in result_keys),
            None,
        )
        if (
            left is not None
            and right is not None
            and key_origins(result_keys[left]).isdisjoint(
                key_origins(result_keys[right])
            )
        ):
            return True
    return False


def _java_mac_assignment_messages(source: str) -> dict[str, str]:
    aliases = {
        match.group("name"): re.sub(r"\s+", "", match.group("value"))
        for match in re.finditer(
            r"\b(?:byte\s*\[\s*\]|String|var)\s+"
            r"(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
            r"(?P<value>[^;]+)\s*;",
            source,
            flags=re.DOTALL,
        )
    }

    def canonical(expression: str) -> str:
        compact = re.sub(r"\s+", "", expression)
        seen: set[str] = set()
        while re.fullmatch(r"[A-Za-z_$][\w$]*", compact) and compact in aliases:
            if compact in seen:
                break
            seen.add(compact)
            compact = aliases[compact]
        encoded = re.fullmatch(
            r"(?P<name>[A-Za-z_$][\w$]*)\.getBytes\([^)]*\)", compact
        )
        if encoded is not None and encoded.group("name") in aliases:
            compact = canonical(encoded.group("name")) + ".getBytes()"
        return compact

    updates: dict[str, list[tuple[int, str]]] = {}
    for match in re.finditer(
        r"\b(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*update\s*"
        r"\((?P<message>[^;]+)\)\s*;",
        source,
        flags=re.DOTALL,
    ):
        updates.setdefault(match.group("receiver"), []).append(
            (match.start(), canonical(match.group("message")))
        )

    messages: dict[str, str] = {}
    pattern = re.compile(
        r"\bbyte\s*\[\s*\]\s*(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
        r"(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*doFinal\s*\(",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(source):
        opening = source.find("(", match.start())
        closing = _matching_parenthesis(source, opening)
        if closing is None:
            continue
        argument = source[opening + 1 : closing].strip()
        if argument:
            message = canonical(argument)
        else:
            preceding = [
                value
                for position, value in updates.get(match.group("receiver"), ())
                if position < match.start()
            ]
            message = preceding[-1] if preceding else ""
        messages[match.group("name")] = message

    analyzer = analyzer_for("java", source)
    for call in analyzer.calls("doFinal"):
        if not call.arguments:
            continue
        statement_start = max(
            source.rfind(";", 0, call.start),
            source.rfind("{", 0, call.start),
        )
        prefix = source[statement_start + 1 : call.start]
        targets = re.findall(
            r"(?:\b[\w.$<>\[\]]+\s+)?(?P<name>[A-Za-z_$][\w$]*)\s*=",
            prefix,
        )
        if targets:
            messages[targets[-1]] = canonical(call.arguments[0])
    return messages


def _java_compares_recomputed_mac_for_same_message(source: str) -> bool:
    messages = _java_mac_assignment_messages(source)
    if len(messages) < 2:
        return False
    aliases = {
        match.group("name"): match.group("value")
        for match in re.finditer(
            r"\b(?:byte\s*\[\s*\]|String|var)\s+"
            r"(?P<name>[A-Za-z_$][\w$]*)\s*=\s*(?P<value>[^;]+);",
            source,
            flags=re.DOTALL,
        )
    }

    def digest_sources(expression: str, seen: set[str] | None = None) -> set[str]:
        seen = set() if seen is None else seen
        names = set(re.findall(r"[A-Za-z_$][\w$]*", expression))
        direct = names & set(messages)
        for name in names - seen:
            if name in aliases:
                direct.update(digest_sources(aliases[name], seen | {name}))
        return direct

    comparisons: list[tuple[str, str]] = []
    analyzer = analyzer_for("java", source)
    for call in analyzer.calls("equals", "isEqual", "compare"):
        if len(call.arguments) >= 2:
            comparisons.append((call.arguments[0], call.arguments[1]))
        elif len(call.arguments) == 1 and "." in call.callee:
            comparisons.append((call.callee.rsplit(".", 1)[0], call.arguments[0]))
    for left, right in comparisons:
        left_sources = digest_sources(left)
        right_sources = digest_sources(right)
        if not left_sources or not right_sources:
            continue
        if all(
            messages[left_name] == messages[right_name]
            for left_name in left_sources
            for right_name in right_sources
        ):
            return True
    return False


def _typescript_value_aliases(source: str, assignments: tuple[object, ...]) -> dict[str, str]:
    """Collect source-provable TypeScript aliases used by MAC data flow."""

    aliases = {
        assignment.name: assignment.expression
        for assignment in assignments
    }
    aliases.update(
        {
            f"this.{match.group('property')}": match.group("value")
            for match in re.finditer(
                r"\bthis\s*\.\s*(?P<property>[A-Za-z_$][\w$]*)\s*=\s*"
                r"(?P<value>[A-Za-z_$][\w$]*)\s*;",
                source,
            )
        }
    )

    # A full Buffer.copy establishes identity even though the destination was
    # initially allocated independently.
    for match in re.finditer(
        r"\b(?P<source>[A-Za-z_$][\w$]*)\s*\.\s*copy\s*\(\s*"
        r"(?P<target>[A-Za-z_$][\w$]*)\s*,\s*0\s*,\s*0\s*,\s*"
        r"(?P<end>\d+)\s*\)",
        source,
    ):
        source_name = match.group("source")
        target_name = match.group("target")
        copied = int(match.group("end"))
        if (
            _typescript_known_buffer_length(source_name, aliases) == copied
            and _typescript_known_buffer_length(target_name, aliases) == copied
        ):
            aliases[target_name] = source_name
    return aliases


def _typescript_known_buffer_length(
    expression: str,
    aliases: dict[str, str],
) -> int | None:
    compact = re.sub(r"\s+", "", expression)
    seen: set[str] = set()
    while compact in aliases and compact not in seen:
        seen.add(compact)
        compact = re.sub(r"\s+", "", aliases[compact])
    match = re.fullmatch(r"(?:[\w$.]+\.)?(?:randomBytes|alloc)\((\d+)\)", compact)
    return int(match.group(1)) if match is not None else None


def _typescript_first_call_argument(expression: str, callee: str) -> str | None:
    compact = re.sub(r"\s+", "", expression)
    prefix = f"{callee}("
    if not compact.startswith(prefix) or not compact.endswith(")"):
        return None
    body = compact[len(prefix) : -1]
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(body):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'", "`"}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            return body[:index]
    return body


def _typescript_resolve_value(
    expression: str,
    aliases: dict[str, str],
    *,
    unwrap_buffer: bool,
) -> str:
    compact = re.sub(r"\s+", "", expression)
    seen: set[str] = set()
    while compact in aliases and compact not in seen:
        seen.add(compact)
        compact = re.sub(r"\s+", "", aliases[compact])

    slice_match = re.fullmatch(
        r"(?P<source>[A-Za-z_$][\w$]*)\.slice\(0,(?P<end>\d+)\)",
        compact,
    )
    if slice_match is not None and _typescript_known_buffer_length(
        slice_match.group("source"), aliases
    ) == int(slice_match.group("end")):
        return _typescript_resolve_value(
            slice_match.group("source"), aliases, unwrap_buffer=unwrap_buffer
        )

    if unwrap_buffer:
        payload = _typescript_first_call_argument(compact, "Buffer.from")
        if payload is not None:
            return _typescript_resolve_value(
                payload, aliases, unwrap_buffer=unwrap_buffer
            )
    return compact


def _typescript_compares_mac_values_for_different_messages(source: str) -> bool:
    """Trace compared TypeScript HMAC digests to their update arguments."""

    analyzer = analyzer_for("typescript", source)
    assignments = tuple(getattr(analyzer, "assignments", ()))
    aliases = _typescript_value_aliases(source, assignments)
    hmac_receivers = {
        assignment.name
        for assignment in assignments
        if re.search(r"\bcreateHmac\s*\(", assignment.expression)
    }

    def canonical(expression: str) -> str:
        compact = _typescript_resolve_value(
            expression, aliases, unwrap_buffer=True
        )
        text = re.fullmatch(r"(?P<quote>['\"])(?P<value>.*)(?P=quote)", compact)
        if text is not None:
            return f"utf8:{text.group('value')}"
        return compact

    updates = tuple(analyzer.calls("update"))
    messages: dict[str, str] = {}
    helper_messages: dict[str, str] = {}
    for function in getattr(analyzer, "functions", ()):
        body = source[function.start : function.end]
        if not re.search(r"\.\s*digest\s*\(", body):
            continue
        helper_updates = tuple(
            call
            for call in updates
            if function.start <= call.start <= function.end and call.arguments
        )
        if helper_updates:
            helper_messages[function.name] = canonical(
                helper_updates[-1].arguments[0]
            )
        elif re.search(r"\bcreateHmac\s*\(", body):
            helper_messages[function.name] = "utf8:"
    for assignment in assignments:
        digest_calls = tuple(
            call
            for call in analyzer.calls("digest")
            if assignment.start <= call.start <= assignment.end
        )
        if not digest_calls:
            helper = next(
                (
                    message
                    for name, message in helper_messages.items()
                    if re.search(rf"(?<![\w$]){re.escape(name)}\s*\(", assignment.expression)
                ),
                None,
            )
            if helper is not None:
                messages[assignment.name] = helper
            continue
        digest = digest_calls[-1]
        local_updates = tuple(
            call
            for call in updates
            if assignment.start <= call.start <= digest.start and call.arguments
        )
        if local_updates:
            if (
                assignment.name in hmac_receivers
                or re.search(r"\bcreateHmac\s*\(", assignment.expression)
            ):
                messages[assignment.name] = canonical(local_updates[-1].arguments[0])
                continue
        if re.search(r"\bcreateHmac\s*\(", assignment.expression):
            messages[assignment.name] = "utf8:"
            continue
        receiver = (
            digest.callee.rsplit(".", 1)[0]
            if "." in digest.callee
            else ""
        )
        if receiver not in hmac_receivers:
            continue
        receiver_updates = tuple(
            call
            for call in updates
            if call.start < digest.start
            and call.arguments
            and "." in call.callee
            and call.callee.rsplit(".", 1)[0] == receiver
        )
        if receiver_updates:
            messages[assignment.name] = canonical(receiver_updates[-1].arguments[0])

    comparisons: list[tuple[str, str]] = []
    for call in analyzer.calls(
        "equals", "timingSafeEqual", "compare_digest", "FixedTimeEquals"
    ):
        if len(call.arguments) >= 2:
            comparisons.append((call.arguments[0], call.arguments[1]))
        elif len(call.arguments) == 1 and "." in call.callee:
            comparisons.append((call.callee.rsplit(".", 1)[0], call.arguments[0]))
    comparisons.extend(
        (match.group("left"), match.group("right"))
        for match in re.finditer(
            r"(?P<left>[A-Za-z_$][\w$]*(?:\s*\.\s*[A-Za-z_$][\w$]*)*)"
            r"\s*(?:===|!==|==|!=)\s*"
            r"(?P<right>[A-Za-z_$][\w$]*(?:\s*\.\s*[A-Za-z_$][\w$]*)*)",
            source,
        )
    )

    external_members = {
        parameter
        for function in getattr(analyzer, "functions", ())
        if _looks_like_verification_name(function.name)
        for parameter in function.parameters
        if re.search(
            rf"\b{re.escape(parameter)}\s*\.\s*(?:signature|tag|mac|digest)\b",
            source[function.start : function.end],
            re.IGNORECASE,
        )
    }

    def message_for_operand(expression: str) -> str | None:
        inline = analyzer_for(
            "typescript", f"const __validator_operand = {expression};"
        )
        inline_updates = tuple(
            call for call in inline.calls("update") if call.arguments
        )
        if inline.calls("digest") and inline_updates:
            return canonical(inline_updates[-1].arguments[0])

        referenced = {
            message
            for name, message in messages.items()
            if re.search(
                rf"(?<![.$\w]){re.escape(name)}(?![\w$])",
                expression,
            )
        }
        return next(iter(referenced)) if len(referenced) == 1 else None

    for left, right in comparisons:
        if any(
            re.search(
                rf"\b{re.escape(parameter)}\s*\.\s*(?:signature|tag|mac|digest)\b",
                expression,
                re.IGNORECASE,
            )
            for parameter in external_members
            for expression in (left, right)
        ):
            continue
        left_message = message_for_operand(left)
        right_message = message_for_operand(right)
        if (
            left_message is not None
            and right_message is not None
            and left_message != right_message
        ):
            return True
    return False


def _typescript_compares_mac_values_for_different_keys(source: str) -> bool:
    """Find compared Node HMAC values whose keys have definite distinct origins."""

    analyzer = analyzer_for("typescript", source)
    assignments = tuple(getattr(analyzer, "assignments", ()))
    aliases = _typescript_value_aliases(source, assignments)

    def resolve(expression: str) -> str:
        return _typescript_resolve_value(expression, aliases, unwrap_buffer=True)

    receivers: dict[str, str] = {}
    for assignment in assignments:
        calls = tuple(
            call
            for call in analyzer.calls("createHmac")
            if assignment.start <= call.start <= assignment.end
            and len(call.arguments) >= 2
        )
        if calls:
            receivers[assignment.name] = resolve(calls[-1].arguments[1])

    digest_keys: dict[str, str] = {}
    for assignment in assignments:
        digests = tuple(
            call
            for call in analyzer.calls("digest")
            if assignment.start <= call.start <= assignment.end
        )
        if not digests:
            continue
        inline_hmac = tuple(
            call
            for call in analyzer.calls("createHmac")
            if assignment.start <= call.start <= assignment.end
            and len(call.arguments) >= 2
        )
        if inline_hmac:
            digest_keys[assignment.name] = resolve(inline_hmac[-1].arguments[1])
            continue
        receiver = digests[-1].callee.rsplit(".", 1)[0]
        if receiver in receivers:
            digest_keys[assignment.name] = receivers[receiver]

    def key_for(expression: str) -> str | None:
        inline = analyzer_for("typescript", f"const __validator_operand = {expression};")
        calls = tuple(
            call for call in inline.calls("createHmac") if len(call.arguments) >= 2
        )
        if calls and inline.calls("digest"):
            return resolve(calls[-1].arguments[1])
        values = {
            key
            for name, key in digest_keys.items()
            if re.search(rf"(?<![.$\w]){re.escape(name)}(?![\w$])", expression)
        }
        return next(iter(values)) if len(values) == 1 else None

    comparisons = [
        (call.arguments[0], call.arguments[1])
        for call in analyzer.calls("timingSafeEqual", "equals", "FixedTimeEquals")
        if len(call.arguments) >= 2
    ]
    for left, right in comparisons:
        left_key = key_for(left)
        right_key = key_for(right)
        if left_key is None or right_key is None or left_key == right_key:
            continue
        one_is_derived = any(
            "createHash(" in value or "randomBytes(" in value
            for value in (left_key, right_key)
        )
        if one_is_derived:
            return True
    return False


def _matching_parenthesis(source: str, opening: int) -> int | None:
    if opening < 0:
        return None
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(opening, len(source)):
        char = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _java_elementwise_comparisons(source: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (match.group("left"), match.group("right"))
        for match in re.finditer(
        r"\b(?P<left>[A-Za-z_$][\w$]*)\s*\[[^\]]+\]\s*"
        r"(?:==|!=)\s*(?P<right>[A-Za-z_$][\w$]*)\s*\[[^\]]+\]",
        source,
        )
    )


def _java_has_connected_elementwise_mac_comparison(source: str) -> bool:
    computed = _java_mac_assignment_messages(source)
    computed_names = set(computed)
    changed = True
    while changed:
        changed = False
        for match in re.finditer(
            r"(?:System\s*\.\s*)?arraycopy\s*\(\s*"
            r"(?P<source>[A-Za-z_$][\w$]*)\s*,[^,]*,\s*"
            r"(?P<target>[A-Za-z_$][\w$]*)\s*,",
            source,
            flags=re.IGNORECASE,
        ):
            if (
                match.group("source") in computed_names
                and match.group("target") not in computed_names
            ):
                computed_names.add(match.group("target"))
                changed = True
    direct = any(
        (
            left in computed_names
            and right in computed_names
            and left in computed
            and right in computed
            and computed[left] == computed[right]
        )
        or (left in computed_names and _is_external_tag_name(right))
        or (right in computed_names and _is_external_tag_name(left))
        for left, right in _java_elementwise_comparisons(source)
    )
    if direct:
        return True
    if any(
        match.group("computed") in computed_names
        and _is_external_tag_name(match.group("tag"))
        for match in re.finditer(
            r"\b(?P<computed>[A-Za-z_$][\w$]*)\s*\[[^\]]+\]\s*"
            r"(?:==|!=)\s*\(*\s*(?:\(\s*byte\s*\)\s*)?"
            r"(?P<tag>[A-Za-z_$][\w$]*)\s*\.\s*charAt\s*\(",
            source,
            flags=re.IGNORECASE,
        )
    ):
        return True
    return any(
        match.group("left") in computed
        and match.group("right") in computed
        and computed[match.group("left")] == computed[match.group("right")]
        for match in re.finditer(
            r"\b[A-Za-z_$][\w$]*(?:equal|compare|match)[\w$]*\s*\(\s*"
            r"(?P<left>[A-Za-z_$][\w$]*)\s*,\s*"
            r"(?P<right>[A-Za-z_$][\w$]*)\s*\)",
            source,
            flags=re.IGNORECASE,
        )
    )


def _java_mac_representation_failure(source: str) -> str | None:
    """Reject definite Java MAC type and representation mismatches."""

    assignments = [
        (match.group("name"), match.group("expression"))
        for match in re.finditer(
            r"\b(?P<name>[A-Za-z_$][\w$]*)\s*(?<![=!<>])=(?!=)\s*"
            r"(?P<expression>[^;]+);",
            source,
            flags=re.DOTALL,
        )
    ]
    raw = {
        name
        for name, expression in assignments
        if re.search(r"\.\s*doFinal\s*\(", expression)
    }
    hex_values: set[str] = set()
    base64_values: set[str] = set()
    text_bytes: set[str] = set()
    raw_text: set[str] = set()
    java_functions = _c_style_functions("java", analyzer_for("java", source))
    parameter_tags = {
        parameter
        for _name, _start, _end, parameters in java_functions
        for parameter in parameters
        if _TAG_NAME.search(parameter)
    }
    changed = True
    while changed:
        changed = False
        for match in re.finditer(
            r"\bfor\s*\([^:;]*\b(?P<item>[A-Za-z_$][\w$]*)\s*:\s*"
            r"(?P<source>[A-Za-z_$][\w$]*)\s*\)",
            source,
        ):
            if match.group("source") in raw and match.group("item") not in raw:
                raw.add(match.group("item"))
                changed = True
        for match in re.finditer(
            r"\b(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*append\s*"
            r"\((?P<value>[^;]+)\)\s*;",
            source,
            flags=re.DOTALL,
        ):
            value = match.group("value")
            if not (_identifier_names(value) & raw):
                continue
            receiver = match.group("receiver")
            if re.search(r"(?:%0?2x|toHexString)", value, re.IGNORECASE):
                if receiver not in hex_values:
                    hex_values.add(receiver)
                    changed = True
        for name, expression in assignments:
            identifiers = _identifier_names(expression)
            encoding: set[str] | None = None
            if identifiers & raw and re.search(r"Base64", expression, re.IGNORECASE):
                encoding = base64_values
            elif identifiers & raw and re.search(r"\bnew\s+String\s*\(", expression):
                encoding = raw_text
            elif identifiers & hex_values:
                encoding = hex_values
            elif identifiers & base64_values:
                encoding = base64_values
            elif identifiers & raw_text:
                encoding = raw_text
            elif identifiers & raw or re.search(r"\.\s*doFinal\s*\(", expression):
                if re.search(r"Base64", expression, re.IGNORECASE):
                    encoding = base64_values
                elif re.search(
                    r"(?:%0?2x|toHexString|bytesToHex)",
                    expression,
                    re.IGNORECASE,
                ):
                    encoding = hex_values
                else:
                    encoding = raw
            elif name not in parameter_tags and re.search(r"\.\s*getBytes\s*\(", expression) and (
                any(_is_external_tag_name(value) for value in identifiers)
                or re.search(r"['\"][^'\"]+['\"]\s*\.\s*getBytes", expression)
            ):
                encoding = text_bytes
            if encoding is not None and name not in encoding:
                encoding.add(name)
                changed = True
        for copy in re.finditer(
            r"(?:System\s*\.\s*)?arraycopy\s*\(\s*"
            r"(?P<source>[A-Za-z_$][\w$]*)\s*,[^,]*,\s*"
            r"(?P<target>[A-Za-z_$][\w$]*)\s*,",
            source,
            flags=re.IGNORECASE,
        ):
            if copy.group("source") in raw and copy.group("target") not in raw:
                raw.add(copy.group("target"))
                changed = True

    analyzer = analyzer_for("java", source)
    comparisons: list[tuple[str, str]] = []
    for call in analyzer.calls("equals", "isEqual"):
        if not call.arguments:
            continue
        receiver = (
            call.callee.rsplit(".", 1)[0]
            if "." in call.callee
            else _receiver_before_call(source, call.start)
        )
        comparisons.append((receiver, call.arguments[0]))
    for call in analyzer.calls("equals"):
        if len(call.arguments) >= 2:
            comparisons.append((call.arguments[0], call.arguments[1]))
    for left, right in comparisons:
        left_names = _identifier_names(left)
        right_names = _identifier_names(right)
        if (left_names & hex_values and right_names & base64_values) or (
            left_names & base64_values and right_names & hex_values
        ):
            return "the MAC comparison mixes hexadecimal and Base64 representations"
    return None


def _java_verifier_has_no_candidate(source: str) -> bool:
    """Find a verifier that creates both operands instead of receiving a tag."""

    analyzer = analyzer_for("java", source)
    functions = _c_style_functions("java", analyzer)
    producer_names = _c_style_mac_producer_names(source, functions)
    for function_name, start, end, parameters in functions:
        if not _looks_like_verification_name(function_name):
            continue
        if any(_TAG_NAME.search(parameter) for parameter in parameters):
            continue
        body = source[start:end]
        if re.search(r"\.\s*doFinal\s*\(", body) is None:
            continue
        if not any(
            re.search(rf"\b{re.escape(producer)}\s*\(", body)
            for producer in producer_names
            if producer != function_name
        ):
            continue
        if re.search(
            r"(?:Arrays\s*\.\s*equals|MessageDigest\s*\.\s*isEqual|\.\s*equals)\s*\(",
            body,
        ):
            return True
    return False


def _typescript_mac_message_flow_failure(source: str) -> str | None:
    analyzer = analyzer_for("typescript", source)
    functions = {
        function.name: source[function.start : function.end]
        for function in getattr(analyzer, "functions", ())
        if function.name
    }
    if not functions:
        return None

    # An empty byte string is a legitimate task message.  A verifier that
    # explicitly restricts its data parameter to that value connects the
    # constant update to the selected task route.
    if re.search(
        r"\b(?:data|message|payload)\w*\s*===?\s*(['\"])\s*\1",
        source,
        flags=re.IGNORECASE,
    ):
        return None

    empty_expected_failure = _typescript_compares_task_mac_to_empty_message_mac(
        source,
        functions,
    )
    if empty_expected_failure is not None:
        return empty_expected_failure

    defective: set[str] = set()
    sound: set[str] = set()
    for name, body in functions.items():
        if not re.search(r"\bcreateHmac\s*\(", body) or not re.search(
            r"\.\s*digest\s*\(", body
        ):
            continue
        updates = re.findall(r"\.\s*update\s*\(([^)]*)\)", body, re.DOTALL)
        if not updates or all(re.fullmatch(r"\s*(['\"])\s*\1\s*", value) for value in updates):
            defective.add(name)
        else:
            sound.add(name)
    if not defective:
        return None

    call_graph = {
        name: {
            candidate
            for candidate in functions
            if candidate != name
            and re.search(rf"(?<![\w$]){re.escape(candidate)}\s*\(", body)
        }
        for name, body in functions.items()
    }
    roots = {
        name
        for name, body in functions.items()
        if re.search(r"\b(?:timingSafeEqual|compare|equal)\w*\s*\(", body, re.IGNORECASE)
        or re.search(r"(?:===|!==|==|!=)", body)
    }
    saw_defective_route = False
    for root in roots:
        reachable = {root}
        pending = [root]
        while pending:
            current = pending.pop()
            for target in call_graph.get(current, set()):
                if target not in reachable:
                    reachable.add(target)
                    pending.append(target)
        producers = reachable & (defective | sound)
        if producers and producers <= defective:
            saw_defective_route = True
        elif producers & sound:
            return None
    if saw_defective_route:
        return "the verification HMAC does not authenticate the task message"
    return None


def _typescript_compares_hmac_to_key_material(source: str) -> bool:
    """Find a tag-bearing object constructed with key material as its tag."""

    analyzer = analyzer_for("typescript", source)
    mac_helpers = {
        function.name
        for function in getattr(analyzer, "functions", ())
        if re.search(
            r"\bcreateHmac\s*\(",
            source[function.start:function.end],
        )
        and re.search(
            r"\.\s*digest\s*\(",
            source[function.start:function.end],
        )
    }
    for call in analyzer.calls("timingSafeEqual", "equals", "compare"):
        if len(call.arguments) < 2:
            continue
        for key_index, tag_index in ((0, 1), (1, 0)):
            key = call.arguments[key_index]
            candidate = call.arguments[tag_index]
            # A computed HMAC expression commonly contains ``key`` as an
            # argument.  Only a direct key-valued expression is evidence that
            # the key itself is being used as the supplied tag.
            compact_key = re.sub(r"\s+", "", key)
            if re.fullmatch(
                r"(?:this\.)?[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*",
                compact_key,
            ) is None or re.search(
                r"(?:key|secret)", compact_key, re.IGNORECASE
            ) is None:
                continue
            if re.search(r"(?:tag|mac|signature)", candidate, re.IGNORECASE) or any(
                re.search(rf"\b{re.escape(helper)}\s*\(", candidate)
                for helper in mac_helpers
            ):
                return True

    tag_positions: dict[str, int] = {}
    for match in re.finditer(
        r"\bclass\s+(?P<class>[A-Za-z_$][\w$]*)[^\{]*\{.*?"
        r"\bconstructor\s*\((?P<parameters>[^)]*)\)",
        source,
        flags=re.DOTALL,
    ):
        parameters = [value.strip() for value in match.group("parameters").split(",")]
        for index, parameter in enumerate(parameters):
            if re.search(r"\b(?:tag|mac|signature)\b", parameter, re.IGNORECASE):
                tag_positions[match.group("class")] = index
                break

    for assignment in getattr(analyzer, "assignments", ()):
        instance = re.sub(r"\s+", "", assignment.name)
        constructor = re.search(
            r"\bnew\s+(?P<class>[A-Za-z_$][\w$]*)\s*\((?P<arguments>[^)]*)\)",
            assignment.expression,
            flags=re.DOTALL,
        )
        if constructor is None or constructor.group("class") not in tag_positions:
            continue
        arguments = [value.strip() for value in constructor.group("arguments").split(",")]
        position = tag_positions[constructor.group("class")]
        if position >= len(arguments) or re.search(
            r"(?:key|secret)", arguments[position], re.IGNORECASE
        ) is None:
            continue
        if re.search(
            rf"(?:timingSafeEqual|equals|compare)\s*\([^;]*"
            rf"\b{re.escape(instance)}\s*\.\s*(?:tag|mac|signature)\b",
            source[assignment.end :],
            flags=re.IGNORECASE | re.DOTALL,
        ):
            return True
    return False


def _typescript_compares_task_mac_to_empty_message_mac(
    source: str,
    functions: dict[str, str],
) -> str | None:
    """Find expected tags produced by an HMAC helper with no message update."""

    empty_helpers = {
        name
        for name, body in functions.items()
        if re.search(r"\bcreateHmac\s*\(", body)
        and re.search(r"\.\s*digest\s*\(", body)
        and not re.search(r"\.\s*update\s*\(", body)
    }
    for helper in empty_helpers:
        expected = re.search(
            rf"\b(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
            rf"{re.escape(helper)}\s*\(",
            source,
        )
        if expected is None:
            continue
        expected_name = expected.group("name")
        if re.search(
            rf"\btimingSafeEqual\s*\([^;]*\b{re.escape(expected_name)}\b",
            source[expected.end() :],
            flags=re.DOTALL,
        ) and re.search(r"\.\s*update\s*\(", source):
            return "the expected MAC authenticates an empty message rather than the task message"
    return None


def _typescript_mac_representation_failure(source: str) -> str | None:
    hex_text_values = {
        match.group("name")
        for match in re.finditer(
            r"\b(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
            r"[^;]*\.\s*digest\s*\(\s*\)\s*\.\s*toString\s*\(\s*['\"]hex['\"]\s*\)",
            source,
            flags=re.IGNORECASE,
        )
    }
    for name in hex_text_values:
        if re.search(
            rf"timingSafeEqual\s*\(\s*Buffer\s*\.\s*from\s*\(\s*"
            rf"{re.escape(name)}\s*\)\s*,\s*Buffer\s*\.\s*from\s*\([^;]*['\"]hex['\"]",
            source,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            return "hexadecimal MAC text is compared as UTF-8 bytes with decoded tag bytes"
    return None


def _python_hmac_digestmod_failure(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    assignments: dict[str, ast.AST] = {}
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    kind_resolver = _PythonValueKindResolver(tree, parents)
    pyca_hmac_aliases: set[str] = set()
    hashlib_constructors: set[str] = set()
    stdlib_hmac_names: set[str] = set()
    imported_hmac_new_names: set[str] = set()
    callable_assignments: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            stdlib_hmac_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "hmac"
            )
        if isinstance(node, ast.ImportFrom) and node.module == "hashlib":
            hashlib_constructors.update(alias.asname or alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module == "hmac":
            imported_hmac_new_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "new"
            )
        if isinstance(node, ast.ImportFrom) and node.module == "cryptography.hazmat.primitives":
            pyca_hmac_aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "hmac"
            )
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for target in targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = node.value
            target_name = _python_name(target)
            value_name = _python_name(node.value)
            if target_name and value_name:
                callable_assignments[target_name] = value_name

    stdlib_constructors = {
        *(f"{name}.new" for name in stdlib_hmac_names),
        *imported_hmac_new_names,
    }
    # Resolve simple aliases such as ``new_mac = hmac.new`` and
    # ``self.mac = hmac.new``.  A short fixed-point pass also handles one alias
    # assigned from another without trying to predict runtime rebinding.
    changed = True
    while changed:
        changed = False
        for target, value in callable_assignments.items():
            if value in stdlib_constructors and target not in stdlib_constructors:
                stdlib_constructors.add(target)
                changed = True

    constructed_hashes = {
        name
        for name, value in assignments.items()
        if isinstance(value, ast.Call)
        and any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == name
            and call.func.attr in {"update", "digest", "hexdigest"}
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call))
        )
    }

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        call_name = _python_name(call.func)
        if call_name == "hmac.new" and "hmac" not in stdlib_hmac_names:
            return "hmac.new is used without importing the stdlib hmac module"
        if call_name in {f"{alias}.new" for alias in pyca_hmac_aliases}:
            return "the imported PyCA hmac namespace does not provide hmac.new"
        if call_name not in stdlib_constructors:
            continue
        message = call.args[1] if len(call.args) >= 2 else _keyword_value(call, "msg")
        if isinstance(message, ast.Constant) and message.value is None:
            return "hmac.new requires message bytes rather than None"
        digestmod = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "digestmod"),
            call.args[2] if len(call.args) >= 3 else None,
        )
        if digestmod is None:
            return "hmac.new requires a digestmod argument"
        digestmod_kind = (
            kind_resolver.kind(
                digestmod,
                _python_scope(call, parents, tree),
            )
            if digestmod is not None
            else None
        )
        if (
            isinstance(digestmod, ast.Call)
            and _python_name(digestmod.func) == "getattr"
            and digestmod.args
            and _python_name(digestmod.args[0]) in stdlib_hmac_names
        ):
            return "the stdlib hmac module does not provide digest constructors"
        if (
            isinstance(digestmod, ast.Call)
            and _python_name(digestmod.func) == "getattr"
            and len(digestmod.args) >= 2
            and _python_name(digestmod.args[0]) == "hashlib"
            and isinstance(digestmod.args[1], ast.Call)
            and isinstance(digestmod.args[1].func, ast.Attribute)
            and digestmod.args[1].func.attr == "upper"
        ):
            return "hashlib digest constructor names are lowercase, not uppercased"
        static_getattr_failure = _python_static_hashlib_getattr_failure(
            digestmod,
            tree=tree,
            parents=parents,
        )
        if static_getattr_failure is not None:
            return static_getattr_failure
        if digestmod_kind == "hash_object" or (
            isinstance(digestmod, ast.Name)
            and digestmod.id in constructed_hashes
        ) or _python_digestmod_is_constructed_hash(
            digestmod,
            assignments,
            hashlib_constructors,
        ):
            return (
                "hmac.new requires a digest name or constructor, not an "
                "already constructed hash object"
            )
    return None


def _python_static_hashlib_getattr_failure(
    expression: ast.AST | None,
    *,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> str | None:
    """Reject a statically uppercase ``getattr(hashlib, name)`` selector.

    Dynamic algorithm selection remains valid.  The check applies only when
    every source-visible invocation binds the selector parameter to a literal
    name that cannot exist in ``hashlib`` because its constructor names are
    lowercase.
    """

    if not (
        isinstance(expression, ast.Call)
        and _python_name(expression.func) == "getattr"
        and len(expression.args) >= 2
        and _python_name(expression.args[0]) == "hashlib"
    ):
        return None
    selector = expression.args[1]
    if isinstance(selector, ast.Constant) and isinstance(selector.value, str):
        return (
            "hashlib digest constructor names are lowercase, not uppercased"
            if _python_hashlib_name_is_statically_uppercase(selector.value)
            else None
        )
    if not isinstance(selector, ast.Name):
        return None

    scope = _python_scope(expression, parents, tree)
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    parameters = (*scope.args.posonlyargs, *scope.args.args)
    position = next(
        (index for index, parameter in enumerate(parameters) if parameter.arg == selector.id),
        None,
    )
    if position is None:
        return None
    defaults = {
        len(parameters) - len(scope.args.defaults) + index: value
        for index, value in enumerate(scope.args.defaults)
    }

    values: list[str] = []
    invocations = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node is not expression
        and _python_name(node.func).rsplit(".", 1)[-1] == scope.name
    )
    if not invocations and position in defaults:
        invocations = ()
        supplied_values: tuple[ast.AST, ...] = (defaults[position],)
    else:
        collected: list[ast.AST] = []
        for invocation in invocations:
            supplied = (
                invocation.args[position]
                if position < len(invocation.args)
                else next(
                    (
                        keyword.value
                        for keyword in invocation.keywords
                        if keyword.arg == selector.id
                    ),
                    defaults.get(position),
                )
            )
            if supplied is None:
                return None
            collected.append(supplied)
        supplied_values = tuple(collected)
    if not supplied_values:
        return None
    for supplied in supplied_values:
        if not isinstance(supplied, ast.Constant) or not isinstance(supplied.value, str):
            return None
        values.append(supplied.value)
    if values and all(_python_hashlib_name_is_statically_uppercase(value) for value in values):
        return "hashlib digest constructor names are lowercase, not uppercased"
    return None


def _python_hashlib_name_is_statically_uppercase(value: str) -> bool:
    return bool(value) and value.lower() in {
        "md5",
        "sha1",
        "sha224",
        "sha256",
        "sha384",
        "sha512",
        "sha3_224",
        "sha3_256",
        "sha3_384",
        "sha3_512",
        "blake2b",
        "blake2s",
    } and value != value.lower()


def _python_digestmod_is_constructed_hash(
    expression: ast.AST | None,
    assignments: dict[str, ast.AST],
    imported_constructors: set[str],
) -> bool:
    if expression is None:
        return False
    if isinstance(expression, ast.Name) and expression.id in assignments:
        return _python_digestmod_is_constructed_hash(
            assignments[expression.id],
            assignments,
            imported_constructors,
        )
    if not isinstance(expression, ast.Call):
        return False

    # ``hashlib.sha256().copy()`` and an alias of that expression both remain
    # constructed hash objects.  HMAC needs the constructor itself.
    if (
        isinstance(expression.func, ast.Attribute)
        and expression.func.attr == "copy"
        and _python_digestmod_is_constructed_hash(
            expression.func.value,
            assignments,
            imported_constructors,
        )
    ):
        return True

    # A dictionary lookup may select one of several digest constructors.  A
    # subsequent call constructs a hash object regardless of which key is
    # selected, so it is invalid as ``digestmod`` when every dictionary value
    # is a known hashlib constructor.
    if isinstance(expression.func, ast.Subscript):
        container = expression.func.value
        if isinstance(container, ast.Name):
            assigned = assignments.get(container.id)
            if isinstance(assigned, ast.Dict) and assigned.values:
                constructor_names = {
                    _python_name(value)
                    for value in assigned.values
                    if value is not None
                }
                if constructor_names and all(
                    name in imported_constructors or name.startswith("hashlib.")
                    for name in constructor_names
                ):
                    return True

    name = _python_name(expression.func)
    if (
        name in imported_constructors
        or name.startswith("hashlib.")
        or (
            isinstance(expression.func, ast.Call)
            and isinstance(expression.func.func, ast.Name)
            and expression.func.func.id == "getattr"
            and expression.func.args
            and _python_name(expression.func.args[0]) == "hashlib"
        )
    ):
        return True
    return False


def _python_compares_against_an_embedded_tag(tree: ast.Module) -> bool:
    """Recognize a verification route that extracts a tag from its envelope."""

    return any(
        isinstance(call, ast.Call)
        and _python_name(call.func).endswith("compare_digest")
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in {"split", "rsplit", "partition", "rpartition"}
            for argument in call.args
            for child in ast.walk(argument)
        )
        for call in ast.walk(tree)
    )


def _python_mac_representation_failure(source: str) -> str | None:
    """Reject definite Python HMAC type and representation mismatches."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    resolver = _PythonValueKindResolver(tree, parents)

    if re.search(
        r"(?:X-Hub-Signature|(?:header|signature|tag)[A-Za-z0-9_]*\s*[:=])[^\n]*"
        r"(?:\.\s*digest\s*\(\s*\)|\.\s*hexdigest\s*\(\s*\))"
        r"\s*\.\s*hex\s*\(",
        source,
        flags=re.IGNORECASE,
    ) and re.search(
        r"(?:expected|computed)[A-Za-z0-9_]*(?:signature|tag|mac)\s*=\s*"
        r"(?:base64\s*\.\s*)?b64encode\s*\(",
        source,
        flags=re.IGNORECASE,
    ):
        return "the received tag is hexadecimal while the computed MAC is Base64"

    def base64_direction(expression: ast.AST, before_line: int) -> str | None:
        if isinstance(expression, ast.Name):
            candidates = [
                node.value
                for node in ast.walk(tree)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and getattr(node, "lineno", 0) <= before_line
                and any(
                    isinstance(target, ast.Name) and target.id == expression.id
                    for target in (
                        node.targets if isinstance(node, ast.Assign) else (node.target,)
                    )
                )
            ]
            candidates.sort(key=lambda item: getattr(item, "lineno", 0))
            return (
                base64_direction(candidates[-1], before_line)
                if candidates
                else None
            )
        if isinstance(expression, ast.Call):
            simple = _python_name(expression.func).rsplit(".", 1)[-1]
            if simple == "b64encode":
                return "encode"
            if simple == "b64decode":
                return "decode"
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _python_name(node).startswith(
            "hmac.algorithms."
        ):
            return "the stdlib hmac module has no algorithms namespace"
        if isinstance(node, ast.Compare) and node.comparators:
            scope = _python_scope(node, parents, tree)
            operands = (node.left, node.comparators[0])
            left = _python_value_base_kind(
                _python_conditioned_name_kind(
                    operands[0], node, tree, parents, resolver
                )
                or resolver.kind(operands[0], scope)
            )
            right = _python_value_base_kind(
                _python_conditioned_name_kind(
                    operands[1], node, tree, parents, resolver
                )
                or resolver.kind(operands[1], scope)
            )
            if left is not None and right is not None and left != right:
                return "the MAC comparison mixes text and byte representations"
            continue
        if not isinstance(node, ast.Call):
            continue
        name = _python_name(node.func)
        scope = _python_scope(node, parents, tree)
        if name.rsplit(".", 1)[-1] in {"b64decode", "urlsafe_b64decode"} and node.args:
            decoded_kind = resolver.kind(node.args[0], scope)
            if isinstance(node.args[0], ast.Name):
                prior = [
                    assignment.value
                    for assignment in ast.walk(scope)
                    if isinstance(assignment, (ast.Assign, ast.AnnAssign))
                    and getattr(assignment, "lineno", 0) < getattr(node, "lineno", 0)
                    and any(
                        node.args[0].id in _python_target_names(target)
                        for target in (
                            assignment.targets
                            if isinstance(assignment, ast.Assign)
                            else (assignment.target,)
                        )
                    )
                ]
                if prior:
                    decoded_kind = resolver.kind(prior[-1], scope)
            if decoded_kind == "raw_mac_digest":
                return "Base64 decoding is applied to raw MAC bytes"
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"digest", "hexdigest"}
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Attribute)
            and node.func.value.func.attr == "update"
        ):
            return "hmac update returns no object on which digest can be called"
        if name in {"binascii.hexlify", "hexlify"} and node.args:
            if any(
                isinstance(child, ast.Call)
                and _python_name(child.func).endswith("compare_digest")
                for child in ast.walk(node.args[0])
            ):
                return "binascii.hexlify cannot encode a Boolean MAC decision"
            if resolver.kind(node.args[0], scope) == "text":
                return "binascii.hexlify requires bytes-like input, not text"
        if name in {"hmac.new", "new"}:
            key = node.args[0] if node.args else _keyword_value(node, "key")
            message = (
                node.args[1] if len(node.args) >= 2 else _keyword_value(node, "msg")
            )
            if key is not None and _python_value_base_kind(
                resolver.kind(key, scope)
            ) == "text":
                return "hmac.new requires bytes-like key input, not text"
            if message is not None and _python_value_base_kind(
                resolver.kind(message, scope)
            ) == "text":
                return "hmac.new requires bytes-like message input, not text"
        if name.endswith("compare_digest") and len(node.args) >= 2:
            if any(
                isinstance(child, ast.Call)
                and child is not node
                and _python_name(child.func).endswith("compare_digest")
                for argument in node.args[:2]
                for child in ast.walk(argument)
            ):
                return "compare_digest receives a Boolean comparison result"
            for argument in node.args[:2]:
                for child in ast.walk(argument):
                    if not (
                        isinstance(child, ast.Call)
                        and _python_name(child.func).rsplit(".", 1)[-1]
                        in {"b64decode", "urlsafe_b64decode"}
                        and child.args
                    ):
                        continue
                    if resolver.kind(child.args[0], scope) == "raw_mac_digest":
                        return "Base64 decoding is applied to raw MAC bytes"
            directions = tuple(
                base64_direction(argument, getattr(node, "lineno", 0))
                for argument in node.args[:2]
            )
            if set(directions) == {"encode", "decode"}:
                return "the MAC comparison mixes Base64 text bytes and decoded raw tag bytes"
            left = _python_value_base_kind(
                _python_conditioned_name_kind(
                    node.args[0], node, tree, parents, resolver
                )
                or resolver.kind(node.args[0], scope)
            )
            right = _python_value_base_kind(
                _python_conditioned_name_kind(
                    node.args[1], node, tree, parents, resolver
                )
                or resolver.kind(node.args[1], scope)
            )
            if left is not None and right is not None and left != right:
                return "the MAC comparison mixes text and byte representations"
            semantic_kinds = tuple(
                resolver.kind(argument, scope) for argument in node.args[:2]
            )
            if set(semantic_kinds) in (
                {"hex_mac_bytes", "raw_bytes_from_hex"},
                {"base64_mac_bytes", "raw_bytes_from_base64"},
            ):
                return "the MAC comparison mixes encoded digest bytes with decoded tag bytes"
            if (
                _python_is_mac_kind(semantic_kinds[0])
                and _python_is_plain_hash_kind(semantic_kinds[1])
            ) or (
                _python_is_plain_hash_kind(semantic_kinds[0])
                and _python_is_mac_kind(semantic_kinds[1])
            ):
                return "the verifier compares a MAC with an unkeyed hash"
            for argument in node.args[:2]:
                for hmac_call in (
                    child
                    for child in ast.walk(argument)
                    if isinstance(child, ast.Call)
                    and _python_name(child.func) in {"hmac.new", "new"}
                ):
                    message = (
                        hmac_call.args[1]
                        if len(hmac_call.args) >= 2
                        else _keyword_value(hmac_call, "msg")
                    )
                    if message is None:
                        return "the verification HMAC authenticates an empty message"
    return None


def _python_uses_tag_as_hmac_message(source: str) -> bool:
    """Find a verifier that ignores its explicit message and authenticates a tag."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and re.search(r"(?:verify|check|match)", node.name, re.IGNORECASE)
    ):
        parameters = {
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
            if argument.arg not in {"self", "cls"}
        }
        message_parameters = {
            name
            for name in parameters
            if re.search(
                r"(?:data|message|payload|record|body|request|header)",
                name,
                re.IGNORECASE,
            )
        }
        generic_tag_parameters = {
            name
            for name in parameters
            if _TAG_NAME.search(name)
        }
        preferred_tag_parameters = {
            name
            for name in generic_tag_parameters
            if _external_name_priority(name) == 1
        }
        tag_parameters = preferred_tag_parameters or generic_tag_parameters
        if not tag_parameters:
            continue
        assignments = {
            target.id: node.value
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and _python_scope(node, parents, tree) is function
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            if isinstance(target, ast.Name)
        }
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and _python_name(call.func) in {"hmac.new", "new"}
            ):
                continue
            message = call.args[1] if len(call.args) >= 2 else _keyword_value(call, "msg")
            names = _python_expression_origins(
                message,
                assignments,
            )
            if names & tag_parameters and not names & message_parameters:
                return True
        hmac_objects = {
            target.id
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Call)
            and _python_name(node.value.func) in {"hmac.new", "new"}
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            if isinstance(target, ast.Name)
        }
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "update"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in hmac_objects
                and call.args
            ):
                continue
            names = _python_expression_origins(call.args[0], assignments)
            text = ast.unparse(call.args[0])
            if (
                (names & tag_parameters or _TAG_NAME.search(text))
                and not (names & message_parameters)
                and re.search(r"(?:data|message|payload|record|body|request)", text, re.I)
                is None
            ):
                return True
    return False


def _python_compares_mac_to_task_message(source: str) -> bool:
    """Find a MAC comparison whose purported tag resolves to task input."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    resolver = _PythonValueKindResolver(tree, parents)

    def resolved(expression: ast.AST, scope: ast.AST) -> ast.AST:
        seen: set[str] = set()
        while isinstance(expression, ast.Name) and expression.id not in seen:
            seen.add(expression.id)
            replacement = resolver._assignment(expression.id, scope, before=expression)
            if replacement is None:
                break
            expression = replacement
        return expression

    for call in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _python_name(node.func).endswith("compare_digest")
        and len(node.args) >= 2
    ):
        scope = _python_scope(call, parents, tree)
        kinds = tuple(resolver.kind(argument, scope) for argument in call.args[:2])
        for index, kind in enumerate(kinds):
            if not _python_is_mac_kind(kind):
                continue
            other = resolved(call.args[1 - index], scope)
            if any(
                isinstance(child, ast.Call)
                and (
                    _python_name(child.func) in {"hmac.new", "hmac.HMAC"}
                    or re.search(
                        r"(?:compute|calculate|create|generate|sign)[a-z0-9_]*"
                        r"(?:mac|hmac|tag|signature|auth(?:entication)?_?code)",
                        _python_name(child.func).rsplit(".", 1)[-1],
                        flags=re.IGNORECASE,
                    )
                )
                for child in ast.walk(other)
            ):
                continue
            names = {
                node.id.lower()
                for node in ast.walk(other)
                if isinstance(node, ast.Name)
            }
            names.update(
                node.attr.lower()
                for node in ast.walk(other)
                if isinstance(node, ast.Attribute)
            )
            if names & {
                "data",
                "message",
                "payload",
                "record",
                "body",
                "request",
                "content",
            }:
                return True
    return False


def _python_compares_mac_to_key_material(source: str) -> bool:
    """Find a computed MAC compared with the secret/key input itself."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    resolver = _PythonValueKindResolver(tree, parents)

    comparisons: list[tuple[ast.AST, ast.AST, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and node.comparators:
            comparisons.append((node.left, node.comparators[0], node))
        elif (
            isinstance(node, ast.Call)
            and _python_name(node.func).endswith("compare_digest")
            and len(node.args) >= 2
        ):
            comparisons.append((node.args[0], node.args[1], node))

    for left, right, comparison in comparisons:
        scope = _python_scope(comparison, parents, tree)
        kinds = (resolver.kind(left, scope), resolver.kind(right, scope))
        for index, kind in enumerate(kinds):
            if not _python_is_mac_kind(kind):
                continue
            if _python_is_mac_kind(kinds[1 - index]):
                continue
            other = (right, left)[index]
            seen: set[str] = set()
            while isinstance(other, ast.Name) and other.id not in seen:
                seen.add(other.id)
                replacement = resolver._assignment(other.id, scope, before=comparison)
                if replacement is None:
                    break
                other = replacement
            names = {
                node.id.lower()
                for node in ast.walk(other)
                if isinstance(node, ast.Name)
            }
            names.update(
                node.attr.lower()
                for node in ast.walk(other)
                if isinstance(node, ast.Attribute)
            )
            if any(
                re.fullmatch(r"(?:secret_?)?key|secret|api_?key", name)
                for name in names
            ):
                return True
    return False


def _python_expression_origins(
    expression: ast.AST | None,
    assignments: dict[str, ast.AST],
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    if expression is None:
        return set()
    origins: set[str] = set()
    for node in ast.walk(expression):
        if not isinstance(node, ast.Name):
            continue
        origins.add(node.id)
        if node.id in assignments and node.id not in seen:
            origins.update(
                _python_expression_origins(
                    assignments[node.id], assignments, seen | {node.id}
                )
            )
    return origins


def _python_hmac_ignores_message_parameter(source: str) -> str | None:
    """Find a MAC helper whose authenticated value cannot reach task data."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        message_parameters = {
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
            if re.search(
                r"(?:data|message|payload|record|body|request|content|serialized)",
                argument.arg,
                re.IGNORECASE,
            )
        }
        if not message_parameters:
            continue
        messages = tuple(
            call.args[1] if len(call.args) >= 2 else _keyword_value(call, "msg")
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and _python_name(call.func) in {"hmac.new", "new"}
            and _python_scope(call, parents, tree) is function
        )
        messages = tuple(message for message in messages if message is not None)
        if messages and not any(
            _python_expression_reaches_parameters(
                message,
                function,
                message_parameters,
                parents,
                tree,
            )
            for message in messages
        ):
            return f"MAC helper {function.name!r} does not authenticate its task message"
    return None


def _python_expression_reaches_parameters(
    expression: ast.AST,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    parameters: set[str],
    parents: dict[ast.AST, ast.AST],
    tree: ast.Module,
    *,
    before_line: int | None = None,
    seen: frozenset[tuple[str, int]] = frozenset(),
) -> bool:
    """Trace local aliases backwards to any task-message parameter."""

    boundary = before_line or getattr(expression, "lineno", 10**9)
    digest_receivers = {
        _python_name(node.func.value)
        for node in ast.walk(expression)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"digest", "hexdigest"}
    }
    for call in (
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and _python_name(node.func.value) in digest_receivers
        and node.args
        and getattr(node, "lineno", 0) < boundary
    ):
        if _python_expression_reaches_parameters(
            call.args[0],
            function,
            parameters,
            parents,
            tree,
            before_line=getattr(call, "lineno", boundary),
            seen=seen,
        ):
            return True
    for name_node in (node for node in ast.walk(expression) if isinstance(node, ast.Name)):
        if name_node.id in parameters:
            return True
        assignments: list[tuple[int, ast.AST]] = []
        for node in ast.walk(function):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            if _python_scope(node, parents, tree) is not function:
                continue
            line = getattr(node, "lineno", 0)
            if line >= boundary:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if any(name_node.id in _python_target_names(target) for target in targets):
                assignments.append((line, node.value))
        for line, value in sorted(assignments, reverse=True):
            marker = (name_node.id, line)
            if marker in seen:
                continue
            if _python_expression_reaches_parameters(
                value,
                function,
                parameters,
                parents,
                tree,
                before_line=line,
                seen=seen | {marker},
            ):
                return True
    return False


def _python_hmac_authenticates_a_computed_mac(source: str) -> bool:
    """Find a verifier that feeds a previously computed MAC back as message data."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    resolver = _PythonValueKindResolver(tree, parents)
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if _python_name(call.func) not in {"hmac.new", "new"}:
            continue
        message = call.args[1] if len(call.args) >= 2 else _keyword_value(call, "msg")
        if message is None:
            continue
        scope = _python_scope(call, parents, tree)
        if _python_is_mac_kind(resolver.kind(message, scope)):
            return True
    return False


def _python_conditioned_name_kind(
    expression: ast.AST,
    comparison: ast.AST,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
    resolver: "_PythonValueKindResolver",
) -> str | None:
    """Resolve a name assigned under the condition guarding a comparison.

    Generated command-line programs commonly assign a received tag in one
    ``if args.verify`` statement and compare it in a later statement guarded
    by the same condition.  A scope-wide last-assignment lookup conflates that
    path with the generation-only ``else`` branch.
    """

    if not isinstance(expression, ast.Name):
        return None
    guarded = _python_guarded_branch(comparison, parents)
    if guarded is None:
        return None
    active_test, in_body = guarded
    scope = _python_scope(comparison, parents, tree)
    candidates: list[tuple[int, str]] = []
    for conditional in (
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.If)
        and ast.dump(node.test, include_attributes=False)
        == ast.dump(active_test, include_attributes=False)
    ):
        branch = conditional.body if in_body else conditional.orelse
        for node in branch:
            for assignment in ast.walk(node):
                if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = (
                    assignment.targets
                    if isinstance(assignment, ast.Assign)
                    else (assignment.target,)
                )
                if not any(
                    expression.id in _python_target_names(target)
                    for target in targets
                ):
                    continue
                kind = resolver.kind(assignment.value, scope)
                if kind is not None:
                    candidates.append((getattr(assignment, "lineno", 0), kind))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _python_guarded_branch(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> tuple[ast.AST, bool] | None:
    child = node
    parent = parents.get(child)
    while parent is not None:
        if isinstance(parent, ast.If):
            if child is not parent.test:
                return parent.test, child in parent.body
        child = parent
        parent = parents.get(child)
    return None


def _python_mac_message_flow_failure(source: str) -> str | None:
    """Reject a tag parsed from the exact value being authenticated.

    A request of the form ``message + separator + tag`` must authenticate the
    message portion, not the whole request including its received tag.  This
    check only reports the definite straight-line case where ``hmac.new``
    consumes the complete parameter and the comparison extracts its other
    operand from that same parameter with ``split`` or ``partition``.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        hmac_objects: set[str] = set()
        computed_mac_values: set[str] = set()
        authenticated_roots: set[str] = set()
        assignments = tuple(
            node
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.value is not None
            and _python_scope(node, parents, tree) is function
        )
        for node in assignments:
            hmac_calls = tuple(
                call
                for call in ast.walk(node.value)
                if isinstance(call, ast.Call)
                and _python_name(call.func) in {"hmac.new", "new"}
            )
            if not hmac_calls:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            target_names = {
                name for target in targets for name in _python_target_names(target)
            }
            for call in hmac_calls:
                message = (
                    call.args[1]
                    if len(call.args) >= 2
                    else _keyword_value(call, "msg")
                )
                root = _python_complete_value_root(message)
                if root is not None:
                    authenticated_roots.add(root)
            if isinstance(node.value, ast.Call) and _python_name(
                node.value.func
            ) in {"hmac.new", "new"}:
                hmac_objects.update(target_names)
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {"digest", "hexdigest"}
                for call in ast.walk(node.value)
            ):
                computed_mac_values.update(target_names)
        if not authenticated_roots:
            continue
        split_values = {
            name
            for node in assignments
            if _python_expression_splits_root(node.value, authenticated_roots)
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            for name in _python_target_names(target)
        }
        for node in ast.walk(function):
            if isinstance(node, ast.Compare) and node.comparators:
                operands = (node.left, node.comparators[0])
            elif (
                isinstance(node, ast.Call)
                and _python_name(node.func).rsplit(".", 1)[-1]
                == "compare_digest"
                and len(node.args) >= 2
            ):
                operands = (node.args[0], node.args[1])
            else:
                continue
            computed = tuple(
                _python_expression_finalizes_named_hmac(operand, hmac_objects)
                or bool(
                    {
                        name.id
                        for name in ast.walk(operand)
                        if isinstance(name, ast.Name)
                    }
                    & computed_mac_values
                )
                for operand in operands
            )
            split = tuple(
                _python_expression_splits_root(operand, authenticated_roots)
                or bool(
                    {
                        name.id
                        for name in ast.walk(operand)
                        if isinstance(name, ast.Name)
                    }
                    & split_values
                )
                for operand in operands
            )
            if not ((computed[0] and split[1]) or (computed[1] and split[0])):
                continue
            return "the received MAC tag is parsed from the complete value being authenticated"
    return None


def _python_compared_hmacs_authenticate_different_messages(source: str) -> bool:
    """Compare the message origins of two locally generated HMAC values."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    def targets(node: ast.Assign | ast.AnnAssign) -> set[str]:
        raw = node.targets if isinstance(node, ast.Assign) else (node.target,)
        return {
            name
            for target in raw
            for name in _python_target_names(target)
        }

    hash_inputs: dict[str, str] = {}
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "update"
            and isinstance(call.func.value, ast.Name)
            and call.args
        ):
            hash_inputs[call.func.value.id] = ast.dump(call.args[0], include_attributes=False)

    hashed_values: dict[str, str] = {}
    mac_objects: dict[str, list[tuple[int, str]]] = {}
    mac_values: dict[str, list[tuple[int, str]]] = {}
    ordinary_values: dict[str, list[tuple[int, ast.AST]]] = {}
    assignments = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ),
        key=lambda node: getattr(node, "lineno", 0),
    )

    def hmac_message(call: ast.Call, before_line: int) -> str | None:
        if _python_name(call.func) not in {"hmac.new", "new"}:
            return None
        message = call.args[1] if len(call.args) >= 2 else _keyword_value(call, "msg")
        if message is None:
            return None
        if isinstance(message, ast.Name) and message.id in hashed_values:
            return "hash(" + hashed_values[message.id] + ")"
        if (
            isinstance(message, ast.Call)
            and isinstance(message.func, ast.Attribute)
            and message.func.attr == "encode"
            and isinstance(message.func.value, ast.Name)
        ):
            values = [
                value
                for line, value in ordinary_values.get(message.func.value.id, ())
                if line < before_line
            ]
            if values:
                return "encode(" + ast.dump(values[-1], include_attributes=False) + ")"
        if isinstance(message, ast.Name):
            values = [
                value
                for line, value in ordinary_values.get(message.id, ())
                if line < before_line
            ]
            if values:
                return ast.dump(values[-1], include_attributes=False)
        return ast.dump(message, include_attributes=False)

    def latest(
        histories: dict[str, list[tuple[int, str]]],
        name: str,
        before_line: int,
    ) -> str | None:
        candidates = [
            value for line, value in histories.get(name, ()) if line < before_line
        ]
        return candidates[-1] if candidates else None

    for assignment in assignments:
        line = getattr(assignment, "lineno", 0)
        names = targets(assignment)
        value = assignment.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in {"digest", "hexdigest"}
            and isinstance(value.func.value, ast.Name)
        ):
            receiver = value.func.value.id
            if receiver in hash_inputs:
                for name in names:
                    hashed_values[name] = hash_inputs[receiver]
            mac_origin = latest(mac_objects, receiver, line)
            if mac_origin is not None:
                for name in names:
                    mac_values.setdefault(name, []).append((line, mac_origin))
        direct = hmac_message(value, line) if isinstance(value, ast.Call) else None
        if direct is not None:
            for name in names:
                mac_objects.setdefault(name, []).append((line, direct))
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in {"digest", "hexdigest"}
            and isinstance(value.func.value, ast.Call)
        ):
            chained = hmac_message(value.func.value, line)
            if chained is not None:
                for name in names:
                    mac_values.setdefault(name, []).append((line, chained))
        for name in names:
            ordinary_values.setdefault(name, []).append((line, value))

    comparison_pairs: list[tuple[str, str]] = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not _python_name(call.func).endswith("compare_digest") or len(call.args) < 2:
            continue
        call_line = getattr(call, "lineno", 10**9)
        origins = [
            latest(mac_values, argument.id, call_line)
            if isinstance(argument, ast.Name)
            else None
            for argument in call.args[:2]
        ]
        if all(origin is not None for origin in origins):
            comparison_pairs.append((origins[0], origins[1]))  # type: ignore[arg-type]
    return bool(comparison_pairs) and not any(
        left == right for left, right in comparison_pairs
    )


def _python_complete_value_root(expression: ast.AST | None) -> str | None:
    if isinstance(expression, ast.Name):
        return expression.id
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr == "encode"
        and isinstance(expression.func.value, ast.Name)
    ):
        return expression.func.value.id
    return None


def _python_expression_finalizes_named_hmac(
    expression: ast.AST,
    names: set[str],
) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"digest", "hexdigest"}
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in names
        for node in ast.walk(expression)
    )


def _python_expression_splits_root(
    expression: ast.AST,
    roots: set[str],
) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"split", "rsplit", "partition", "rpartition"}
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in roots
        for node in ast.walk(expression)
    )


def _keyword_value(call: ast.Call, name: str) -> ast.AST | None:
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg == name),
        None,
    )


def _python_expected_tag_literal(expression: ast.AST) -> bool:
    return bool(
        any(
            isinstance(node, ast.Constant) and isinstance(node.value, (bytes, str))
            for node in ast.walk(expression)
        )
        and not any(isinstance(node, (ast.Name, ast.Call)) for node in ast.walk(expression))
    )


def _python_value_base_kind(kind: str | None) -> str | None:
    if kind in {
        "bytes",
        "raw_digest",
        "raw_mac_digest",
        "raw_hash_digest",
        "base64_bytes",
        "raw_bytes_from_base64",
        "hex_bytes",
        "raw_bytes_from_hex",
        "encoded_text",
        "base64_mac_bytes",
        "base64_hash_bytes",
        "hex_mac_bytes",
    }:
        return "bytes"
    if kind in {
        "text",
        "hex_text",
        "hex_mac_text",
        "hex_hash_text",
        "base64_text",
    }:
        return "text"
    return None


def _python_is_mac_kind(kind: str | None) -> bool:
    return kind in {
        "raw_mac_digest",
        "hex_mac_text",
        "hex_mac_bytes",
        "base64_mac_bytes",
    }


def _python_is_plain_hash_kind(kind: str | None) -> bool:
    return kind in {"raw_hash_digest", "hex_hash_text", "base64_hash_bytes"}


class _PythonValueKindResolver:
    """Small, conservative type resolver for required HMAC operands."""

    def __init__(self, tree: ast.Module, parents: dict[ast.AST, ast.AST]) -> None:
        self.tree = tree
        self.parents = parents
        self.functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.classes = {
            node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }
        self.argparse_namespaces = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            if isinstance(target, ast.Name)
            and isinstance(node.value, ast.Call)
            and _python_name(node.value.func).endswith("parse_args")
        }
        self.argparse_attribute_kinds = _python_argparse_attribute_kinds(tree)
        self.argv_text_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target, value in _python_paired_assignment_elements(node)
            if isinstance(target, ast.Name)
            and isinstance(value, ast.Subscript)
            and _python_name(value.value) in {"sys.argv", "argv"}
        }
        self._active: set[tuple[int, int]] = set()

    def kind(self, node: ast.AST, scope: ast.AST) -> str | None:
        key = (id(node), id(scope))
        if key in self._active:
            return None
        self._active.add(key)
        try:
            return self._kind(node, scope)
        finally:
            self._active.remove(key)

    def _kind(self, node: ast.AST, scope: ast.AST) -> str | None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return "text"
            if isinstance(node.value, bytes):
                return "bytes"
            return None
        if isinstance(node, ast.Subscript):
            if _python_name(node.value) in {"sys.argv", "argv"}:
                return "text"
            if isinstance(node.value, ast.Name):
                value = self._assignment(node.value.id, scope)
                if isinstance(value, ast.Dict):
                    kinds = {
                        self.kind(item, scope)
                        for item in value.values
                        if item is not None
                    }
                    kinds.discard(None)
                    if len(kinds) == 1:
                        return kinds.pop()
            return self.kind(node.value, scope)
        if isinstance(node, ast.Name):
            if node.id in self.argv_text_names:
                return "text"
            value = self._assignment(node.id, scope, before=node)
            if value is not None:
                return self.kind(value, scope)
            if self._is_parameter(node.id, scope):
                callsite_kind = self._parameter_kind(node.id, scope)
                if callsite_kind is not None:
                    return callsite_kind
                return self._parameter_annotation(node.id, scope)
            return None
        if isinstance(node, ast.Attribute):
            name = _python_name(node)
            if name.startswith("hashlib."):
                return "hash_constructor"
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in self.argparse_namespaces
            ):
                return self.argparse_attribute_kinds.get(node.attr, "text")
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                value, owner = self._self_attribute(node.attr, scope)
                if value is not None and owner is not None:
                    return self.kind(value, owner)
            return None
        if isinstance(node, ast.Call):
            name = _python_name(node.func)
            simple = name.rsplit(".", 1)[-1]
            if simple == "encode":
                return "encoded_text"
            if simple == "decode":
                inner = self.kind(node.func.value, scope) if isinstance(node.func, ast.Attribute) else None
                return {
                    "base64_bytes": "base64_text",
                    "hex_bytes": "hex_text",
                }.get(inner, "text")
            if simple in {"digest", "hexdigest"} and isinstance(
                node.func, ast.Attribute
            ):
                receiver_kind = self.kind(node.func.value, scope)
                if receiver_kind == "mac_object":
                    return (
                        "raw_mac_digest"
                        if simple == "digest"
                        else "hex_mac_text"
                    )
                if receiver_kind == "hash_object":
                    return (
                        "raw_hash_digest"
                        if simple == "digest"
                        else "hex_hash_text"
                    )
                return "raw_digest" if simple == "digest" else "hex_text"
            if simple == "finalize" and isinstance(node.func, ast.Attribute):
                if self.kind(node.func.value, scope) == "mac_object":
                    return "raw_mac_digest"
            if name in {
                "base64.b64encode",
                "base64.urlsafe_b64encode",
                "b64encode",
                "urlsafe_b64encode",
            }:
                inner = self.kind(node.args[0], scope) if node.args else None
                if _python_is_mac_kind(inner):
                    return "base64_mac_bytes"
                if _python_is_plain_hash_kind(inner):
                    return "base64_hash_bytes"
                return "base64_bytes"
            if name in {
                "base64.b64decode",
                "base64.urlsafe_b64decode",
                "b64decode",
                "urlsafe_b64decode",
            }:
                return "raw_bytes_from_base64"
            if name in {"binascii.hexlify", "hexlify"}:
                inner = self.kind(node.args[0], scope) if node.args else None
                return "hex_mac_bytes" if _python_is_mac_kind(inner) else "hex_bytes"
            if name in {"binascii.unhexlify", "unhexlify", "bytes.fromhex"}:
                return "raw_bytes_from_hex"
            if name in {"bytes", "bytearray", "os.urandom", "secrets.token_bytes"}:
                return "bytes"
            if name in {
                "str",
                "input",
                "getpass.getpass",
                "os.getenv",
                "os.environ.get",
            }:
                return "text"
            if name in {"hmac.new", "hmac.HMAC"}:
                return "mac_object"
            if (
                name == "getattr"
                and node.args
                and _python_name(node.args[0]) == "hashlib"
            ):
                return "hash_constructor"
            if name.startswith("hashlib."):
                return "hash_object"
            callee_kind = self.kind(node.func, scope)
            if callee_kind == "hash_constructor":
                return "hash_object"
            candidates = self._return_kinds(simple)
            return next(iter(candidates)) if len(candidates) == 1 else None
        return None

    def _assignment(
        self,
        name: str,
        scope: ast.AST,
        *,
        before: ast.AST | None = None,
    ) -> ast.AST | None:
        boundary = (
            (getattr(before, "lineno", 0), getattr(before, "col_offset", 0))
            if before is not None
            else None
        )
        for candidate_scope in (scope, self.tree):
            values: list[tuple[tuple[int, int], ast.AST]] = []
            for node in ast.walk(candidate_scope):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                if _python_scope(node, self.parents, self.tree) is not candidate_scope:
                    continue
                position = (
                    getattr(node, "lineno", 0),
                    getattr(node, "col_offset", 0),
                )
                if boundary is not None and position >= boundary:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                if any(name in _python_target_names(target) for target in targets):
                    values.append((position, node.value))
            if values:
                return max(values, key=lambda item: item[0])[1]
        return None

    def _parameter_annotation(self, name: str, scope: ast.AST) -> str | None:
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        parameters = tuple(scope.args.posonlyargs) + tuple(scope.args.args)
        parameter = next((item for item in parameters if item.arg == name), None)
        if parameter is None or parameter.annotation is None:
            return None
        annotation = _python_name(parameter.annotation).lower()
        if annotation in {"str", "string"}:
            return "text"
        if annotation in {"bytes", "bytearray"}:
            return "bytes"
        return None

    @staticmethod
    def _is_parameter(name: str, scope: ast.AST) -> bool:
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        return any(
            parameter.arg == name
            for parameter in (
                *scope.args.posonlyargs,
                *scope.args.args,
                *scope.args.kwonlyargs,
            )
        )

    def _parameter_kind(self, name: str, scope: ast.AST) -> str | None:
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        parameters = tuple(scope.args.posonlyargs) + tuple(scope.args.args)
        position = next(
            (index for index, parameter in enumerate(parameters) if parameter.arg == name),
            None,
        )
        if position is None:
            return None
        is_method = bool(parameters and parameters[0].arg in {"self", "cls"})
        argument_position = position - int(is_method)
        if argument_position < 0:
            return None
        owner = self._containing_class(scope)
        call_name = owner.name if scope.name == "__init__" and owner is not None else scope.name
        kinds: set[str] = set()
        for call in (node for node in ast.walk(self.tree) if isinstance(node, ast.Call)):
            if call is scope or _python_name(call.func).rsplit(".", 1)[-1] != call_name:
                continue
            caller_scope = _python_scope(call, self.parents, self.tree)
            argument = (
                call.args[argument_position]
                if argument_position < len(call.args)
                else next(
                    (
                        keyword.value
                        for keyword in call.keywords
                        if keyword.arg == name
                    ),
                    None,
                )
            )
            if argument is None:
                continue
            kind = self.kind(argument, caller_scope)
            if kind is not None:
                kinds.add(kind)
        return next(iter(kinds)) if len(kinds) == 1 else None

    def _self_attribute(
        self,
        name: str,
        scope: ast.AST,
    ) -> tuple[ast.AST | None, ast.AST | None]:
        owner = self._containing_class(scope)
        if owner is None:
            return None, None
        initializer = next(
            (
                node
                for node in owner.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__init__"
            ),
            None,
        )
        if initializer is None:
            return None, None
        for node in ast.walk(initializer):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == name
                for target in targets
            ):
                return node.value, initializer
        return None, None

    def _return_kinds(self, name: str) -> set[str]:
        function = self.functions.get(name)
        if function is None:
            return set()
        return {
            kind
            for node in ast.walk(function)
            if isinstance(node, ast.Return)
            and node.value is not None
            and _python_scope(node, self.parents, self.tree) is function
            and (kind := self.kind(node.value, function)) is not None
        }

    def _containing_class(self, node: ast.AST) -> ast.ClassDef | None:
        current = self.parents.get(node)
        while current is not None:
            if isinstance(current, ast.ClassDef):
                return current
            current = self.parents.get(current)
        return None


def _python_argparse_attribute_kinds(tree: ast.Module) -> dict[str, str]:
    """Resolve explicit ``argparse`` conversions used by required HMAC calls."""

    kinds: dict[str, str] = {}
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if _python_name(call.func).rsplit(".", 1)[-1] != "add_argument":
            continue
        destination = _python_argparse_destination(call)
        if destination is None:
            continue
        type_value = _keyword_value(call, "type")
        if type_value is None:
            kinds.setdefault(destination, "text")
            continue
        kind = _python_argparse_conversion_kind(type_value)
        if kind is not None:
            kinds[destination] = kind
    return kinds


def _python_argparse_destination(call: ast.Call) -> str | None:
    explicit = _keyword_value(call, "dest")
    if isinstance(explicit, ast.Constant) and isinstance(explicit.value, str):
        return explicit.value.replace("-", "_")
    options = [
        value.value
        for value in call.args
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    ]
    long_options = [value for value in options if value.startswith("--")]
    selected = long_options[-1] if long_options else (options[-1] if options else "")
    destination = selected.lstrip("-").replace("-", "_")
    return destination or None


def _python_argparse_conversion_kind(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        if node.id in {"bytes", "bytearray"}:
            return "bytes"
        if node.id == "str":
            return "text"
    if isinstance(node, ast.Lambda) and isinstance(node.body, ast.Call):
        simple = _python_name(node.body.func).rsplit(".", 1)[-1]
        if simple in {"encode", "fromhex", "b64decode", "bytes", "bytearray"}:
            return "bytes"
        if simple in {"decode", "str"}:
            return "text"
    return None


def _java_verifier_ignores_supplied_tag(source: str) -> bool:
    """Find a Java verifier whose integrity parameter is never read."""

    for match in re.finditer(
        r"(?:public|private|protected|static|final|\s)+[\w<>\[\].?]+\s+"
        r"(?P<name>[A-Za-z_$][\w$]*)\s*"
        r"\((?P<params>[^)]*)\)\s*(?:throws[^\{]+)?\{",
        source,
        flags=re.IGNORECASE,
    ):
        if re.search(r"(?:verify|check|match)", match.group("name"), re.IGNORECASE) is None:
            continue
        start = match.end()
        depth = 1
        index = start
        while index < len(source) and depth:
            depth += (source[index] == "{") - (source[index] == "}")
            index += 1
        body = source[start : index - 1]
        parameters = re.findall(
            r"(?:^|,)\s*[\w<>\[\].?]+\s+([A-Za-z_$][\w$]*)",
            match.group("params"),
        )
        integrity = [
            name
            for name in parameters
            if re.search(r"(?:signature|tag|digest|mac)", name, re.IGNORECASE)
        ]
        supplied = [
            name
            for name in integrity
            if re.search(
                r"(?:provided|supplied|received|expected|incoming)",
                name,
                re.IGNORECASE,
            )
        ]
        required = supplied or integrity
        outside_body = source[: match.start()] + source[index:]
        returned_mac_is_compared_by_caller = bool(
            re.search(
                rf"\b{re.escape(match.group('name'))}\s*\([^;]*?\)\s*"
                r"\.\s*equals\s*\(",
                outside_body,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        if (
            required
            and all(
                re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", body) is None
                for name in required
            )
            and not returned_mac_is_compared_by_caller
            and re.search(
                r"(?:\.\s*equals\s*\(|Arrays\s*\.\s*equals\s*\(|"
                r"MessageDigest\s*\.\s*isEqual\s*\(|==|!=)",
                body,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"(?:\.\s*doFinal\s*\(|\b(?:compute|calculate|create|generate)\w*"
                r"(?:mac|hmac|tag|signature)\s*\()",
                body,
                flags=re.IGNORECASE,
            )
        ):
            return True
    return False


def _java_decoded_supplied_tag_is_unused(source: str) -> bool:
    """Find a decoded tag local that never reaches the method's comparison."""

    for method in iter_methods(source):
        parameters = {
            parameter
            for parameter in method.params
            if _TAG_NAME.search(parameter)
        }
        if not parameters:
            continue
        body = strip_java_comments(method.text)
        for parameter in parameters:
            for match in re.finditer(
                r"\b(?:byte\s*\[\s*\]|String|var)\s+"
                r"(?P<name>[A-Za-z_$][\w$]*)\s*=\s*[^;]*"
                r"(?:decode|parse|fromHex)\s*\([^;]*\b"
                + re.escape(parameter)
                + r"\b[^;]*\)\s*;",
                body,
                flags=re.IGNORECASE,
            ):
                name = match.group("name")
                uses = tuple(
                    re.finditer(
                        rf"(?<![\w$]){re.escape(name)}(?![\w$])",
                        body[match.end() :],
                    )
                )
                if not uses and re.search(
                    r"(?:\.\s*equals\s*\(|Arrays\s*\.\s*equals\s*\(|"
                    r"MessageDigest\s*\.\s*isEqual\s*\(|==|!=)",
                    body[match.end() :],
                    flags=re.IGNORECASE,
                ):
                    return True
    return False


def _comparison_connects_received_tag(language: Language, source: str) -> bool:
    if language == "python":
        return _python_comparison_connects_received_tag(source)
    if language == "java" and _java_has_connected_elementwise_mac_comparison(source):
        return True
    if language == "java" and re.search(
        r"\.\s*put\s*\(\s*(?:signature|tag|received)[A-Za-z0-9_$]*\s*\).*?"
        r"\.\s*put\s*\(\s*(?:hash|mac|digest|expected)[A-Za-z0-9_$]*\s*\).*?"
        r"\.\s*equals\s*\(",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return True
    return _c_style_comparison_connects_received_tag(language, source)


def _python_comparison_connects_received_tag(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    resolver = _PythonValueKindResolver(tree, parents)

    scopes: list[ast.Module | ast.FunctionDef | ast.AsyncFunctionDef] = [tree]
    scopes.extend(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    assignments_by_scope: dict[ast.AST, list[tuple[set[str], str]]] = {
        scope: [] for scope in scopes
    }
    for node in ast.walk(tree):
        if (
            not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
            or node.value is None
        ):
            continue
        value = node.value
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else (node.target,)
        )
        names = {
            name
            for target in targets
            for name in _python_target_names(target)
        }
        if names:
            assignments_by_scope[_python_scope(node, parents, tree)].append(
                (names, ast.unparse(value))
            )

    producer_names = _python_mac_producer_names(scopes, parents, tree)
    values_by_scope: dict[ast.AST, tuple[set[str], set[str]]] = {}
    preferred_by_scope: dict[ast.AST, set[str]] = {}
    module_assignments = assignments_by_scope[tree]
    for scope in scopes:
        parameters = _python_scope_parameters(scope)
        assignments = (
            module_assignments
            if scope is tree
            else [*module_assignments, *assignments_by_scope[scope]]
        )
        computed, external = _value_roles(assignments, parameters)
        for names, expression in assignments:
            if _expression_calls_any(expression, producer_names):
                computed.update(names)
        external.difference_update(computed)
        values_by_scope[scope] = computed, external
        preferred_by_scope[scope] = _preferred_external_values(
            assignments,
            parameters,
            computed,
        )

    saw_computed_comparison = False
    for node in ast.walk(tree):
        operands: tuple[ast.AST, ast.AST] | None = None
        if isinstance(node, ast.Call):
            simple = _python_name(node.func).rsplit(".", 1)[-1].lower()
            if simple in {
                "compare_digest",
                "timingsafeequal",
                "fixedtimeequals",
                "isequal",
                "equals",
                "sequenceequal",
            } and len(node.args) >= 2:
                operands = (node.args[0], node.args[1])
            elif simple == "verify" and node.args:
                receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
                if receiver is not None:
                    operands = (receiver, node.args[0])
        elif isinstance(node, ast.Compare):
            for operator, right in zip(node.ops, node.comparators):
                if isinstance(operator, (ast.Eq, ast.NotEq)):
                    operands = (node.left, right)
                    break
        if operands is None:
            continue

        left = ast.unparse(operands[0])
        right = ast.unparse(operands[1])
        if _normalized_expression(left) == _normalized_expression(right):
            if _expression_has_mac_computation(left):
                saw_computed_comparison = True
            continue
        scope = _python_scope(node, parents, tree)
        computed, external = values_by_scope[scope]
        preferred_external = preferred_by_scope[scope]
        comparison_external = external
        left_computed = _expression_is_computed(
            left, computed
        ) or _expression_calls_any(left, producer_names)
        right_computed = _expression_is_computed(
            right, computed
        ) or _expression_calls_any(right, producer_names)
        left_kind = resolver.kind(operands[0], scope)
        right_kind = resolver.kind(operands[1], scope)
        if _python_is_plain_hash_kind(left_kind):
            left_computed = False
        elif _python_is_mac_kind(left_kind):
            left_computed = True
        if _python_is_plain_hash_kind(right_kind):
            right_computed = False
        elif _python_is_mac_kind(right_kind):
            right_computed = True
        if (
            left_computed and _python_expected_tag_literal(operands[1])
        ) or (
            right_computed and _python_expected_tag_literal(operands[0])
        ):
            return True
        left_external = (
            not left_computed
            and _expression_is_external_tag(left, comparison_external)
        )
        right_external = (
            not right_computed
            and _expression_is_external_tag(right, comparison_external)
        )
        if (left_computed and right_external) or (
            right_computed and left_external
        ):
            return True
        if _python_comparison_has_connected_callsite(
            tree,
            parents,
            scope,
            operands,
            values_by_scope,
            producer_names,
        ):
            return True
        if left_computed and right_computed:
            return True
        saw_computed_comparison |= left_computed or right_computed
    return False


def _typescript_expression_is_mac_result(
    expression: str,
    assignments: list[tuple[set[str], str]],
) -> bool:
    """Recognize named or inline Node HMAC digest results."""

    if re.search(r"\bcreateHmac\s*\([^;]*?\.\s*digest\s*\(", expression, re.DOTALL):
        return True
    hmac_receivers = {
        name
        for names, value in assignments
        if re.search(r"\bcreateHmac\s*\(", value)
        for name in names
    }
    mac_values: set[str] = set()
    changed = True
    while changed:
        changed = False
        for names, value in assignments:
            identifiers = _identifier_names(value)
            direct_digest = (
                re.search(r"\.\s*digest\s*\(", value)
                and bool(identifiers & hmac_receivers)
            )
            normalized = _normalized_expression(value)
            direct_alias = bool(identifiers & mac_values) and bool(
                re.fullmatch(r"[A-Za-z_$][\w$]*", normalized)
                or re.fullmatch(
                    r"(?:Buffer\.from\()?"
                    r"[A-Za-z_$][\w$]*"
                    r"(?:\.toString\([^)]*\))?\)?",
                    normalized,
                )
            )
            if not (direct_digest or direct_alias):
                continue
            before = len(mac_values)
            mac_values.update(names)
            changed |= len(mac_values) != before
    return bool(_identifier_names(expression) & mac_values)


def _python_mac_producer_names(
    scopes: list[ast.Module | ast.FunctionDef | ast.AsyncFunctionDef],
    parents: dict[ast.AST, ast.AST],
    module: ast.Module,
) -> set[str]:
    """Return helpers whose own route computes and returns a MAC value."""

    producers: set[str] = set()
    changed = True
    while changed:
        changed = False
        for scope in scopes:
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            returned = [
                node.value
                for node in ast.walk(scope)
                if isinstance(node, ast.Return)
                and node.value is not None
                and _python_scope(node, parents, module) is scope
            ]
            if not returned:
                continue
            body = ast.unparse(scope)
            computes_mac = bool(
                re.search(r"\bhmac\s*\.\s*(?:new|HMAC)\s*\(", body)
                and re.search(r"\.\s*(?:digest|hexdigest|finalize)\s*\(", body)
            ) or any(
                _expression_calls_any(ast.unparse(value), producers)
                for value in returned
            )
            if computes_mac and scope.name not in producers:
                producers.add(scope.name)
                changed = True
    return producers


def _expression_calls_any(expression: str, names: set[str]) -> bool:
    return any(
        re.search(rf"(?<![\w$]){re.escape(name)}\s*\(", expression)
        for name in names
    )


def _python_comparison_has_connected_callsite(
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
    operands: tuple[ast.AST, ast.AST],
    values_by_scope: dict[ast.AST, tuple[set[str], set[str]]],
    producer_names: set[str],
) -> bool:
    """Resolve compared helper parameters from their concrete call sites."""

    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    parameters = (
        *scope.args.posonlyargs,
        *scope.args.args,
        *scope.args.kwonlyargs,
    )
    indexes = {parameter.arg: index for index, parameter in enumerate(parameters)}
    operand_names = [
        next(
            (name for name in _python_target_names(operand) if name in indexes),
            None,
        )
        for operand in operands
    ]
    if any(name is None for name in operand_names):
        return False

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if _python_name(call.func).rsplit(".", 1)[-1] != scope.name:
            continue
        caller = _python_scope(call, parents, tree)
        computed, external = values_by_scope.get(caller, (set(), set()))
        roles: list[tuple[bool, bool]] = []
        for name in operand_names:
            assert name is not None
            index = indexes[name]
            if index >= len(call.args):
                break
            expression = ast.unparse(call.args[index])
            is_computed = _expression_is_computed(expression, computed) or (
                _expression_calls_any(expression, producer_names)
            )
            is_external = not is_computed and _expression_is_external_tag(
                expression,
                external,
            )
            roles.append((is_computed, is_external))
        if len(roles) != 2:
            continue
        if (
            roles[0][0]
            and not roles[1][0]
            and roles[1][1]
        ) or (
            roles[1][0]
            and not roles[0][0]
            and roles[0][1]
        ):
            return True
    return False


def _c_style_comparison_connects_received_tag(
    language: Language,
    source: str,
) -> bool:
    analyzer = analyzer_for(language, source)
    functions = _c_style_functions(language, analyzer)
    producer_names = _c_style_mac_producer_names(source, functions)
    if _c_style_verifier_call_connects_tag(
        language,
        analyzer,
        source,
        functions,
        producer_names,
    ):
        return True
    if language == "typescript" and _typescript_verifier_receives_computed_tag(
        source,
        analyzer,
        functions,
        producer_names,
    ):
        return True
    member_computed, member_external = _c_style_member_mac_roles(
        source,
        analyzer,
        functions,
        producer_names,
    )
    comparison_names = (
        "equals",
        "equalsIgnoreCase",
        "Equals",
        "SequenceEqual",
        "timingSafeEqual",
        "compare_digest",
        "FixedTimeEquals",
        "isEqual",
    )
    comparisons: list[tuple[int, str, str]] = []
    for call in analyzer.calls(*comparison_names):
        receiver = (
            call.callee.rsplit(".", 1)[0]
            if "." in call.callee
            else _receiver_before_call(source, call.start)
        )
        instance_equals = (
            call.simple_name.lower() == "equals"
            and receiver
            and call.arguments
            and (
                len(call.arguments) == 1
                or (
                    len(call.arguments) >= 2
                    and re.search(
                        r"\b(?:StringComparison|StringComparer)\s*\.",
                        call.arguments[1],
                    )
                )
            )
        )
        if instance_equals:
            comparisons.append((call.start, receiver, call.arguments[0]))
        elif len(call.arguments) >= 2:
            comparisons.append((call.start, call.arguments[0], call.arguments[1]))
        elif len(call.arguments) == 1:
            receiver = (
                call.callee.rsplit(".", 1)[0]
                if "." in call.callee
                else _receiver_before_call(source, call.start)
            )
            if receiver:
                comparisons.append((call.start, receiver, call.arguments[0]))

    if language in {"java", "typescript", "csharp"}:
        for call in analyzer.calls():
            if not re.search(r"(?:compare|equal|match)", call.simple_name, re.IGNORECASE):
                continue
            if len(call.arguments) >= 2:
                comparisons.append((call.start, call.arguments[0], call.arguments[1]))

    operator_pattern = re.compile(
        r"(?P<left>[^;\n&|]+?)\s*(?:===|!==|==|!=)\s*"
        r"(?P<right>[^;\n&|]+)"
    )
    for match in operator_pattern.finditer(source):
        left = match.group("left")
        right = match.group("right")
        if re.search(r"\b(?:if|while)\s*\(", left) and ")" in right:
            right = right.split(")", 1)[0]
        comparisons.append(
            (match.start(), left, right)
        )

    saw_computed_comparison = False
    for position, left, right in comparisons:
        if _normalized_expression(left) == _normalized_expression(right):
            continue
        assignments, parameters = _scoped_c_style_values(
            language, source, analyzer, position
        )
        direct_mac_values = {
            name
            for names, expression in assignments
            if _expression_has_mac_computation(expression)
            for name in names
        }
        computed, external = _value_roles(assignments, parameters)
        computed = _expanded_c_style_computed_values(
            source,
            assignments,
            computed,
            producer_names,
        )
        external = _expanded_c_style_external_values(
            source,
            assignments,
            external,
        )
        plain_hash_values = _c_style_plain_hash_values(assignments)
        computed.update(member_computed)
        external.update(member_external)
        external.difference_update(computed)
        external.difference_update(plain_hash_values)
        preferred_external = _preferred_external_values(
            assignments, parameters, computed
        )
        required_external = _expanded_c_style_external_values(
            source,
            assignments,
            preferred_external
            or {name for name in parameters if _is_external_tag_name(name)},
        )
        required_external.difference_update(computed)
        required_external.difference_update(plain_hash_values)
        if language == "java":
            key_inputs = {
                match.group("key")
                for match in re.finditer(
                    r"new\s+SecretKeySpec\s*\(\s*"
                    r"(?P<key>[A-Za-z_$][\w$]*)\s*,",
                    source,
                )
            }
            required_external.difference_update(key_inputs)
        comparison_external = external
        left_computed = _expression_is_computed(
            left, computed
        ) or _expression_calls_any(left, producer_names)
        right_computed = _expression_is_computed(
            right, computed
        ) or _expression_calls_any(right, producer_names)
        left_external_wrapper = _c_style_external_conversion_call(
            left, functions, producer_names, comparison_external
        )
        right_external_wrapper = _c_style_external_conversion_call(
            right, functions, producer_names, comparison_external
        )
        if left_external_wrapper:
            left_computed = False
        if right_external_wrapper:
            right_computed = False
        left_typescript_mac = language == "typescript" and (
            _typescript_expression_is_mac_result(left, assignments)
        )
        right_typescript_mac = language == "typescript" and (
            _typescript_expression_is_mac_result(right, assignments)
        )
        left_computed |= left_typescript_mac
        right_computed |= right_typescript_mac
        if (
            _expression_uses_plain_hash(left, plain_hash_values)
            and not left_typescript_mac
        ):
            left_computed = False
        if (
            _expression_uses_plain_hash(right, plain_hash_values)
            and not right_typescript_mac
        ):
            right_computed = False
        if (left_computed and _c_style_expected_literal(right)) or (
            right_computed and _c_style_expected_literal(left)
        ):
            return True
        left_external = (
            not left_computed
            and not _expression_uses_plain_hash(left, plain_hash_values)
            and (
                left_external_wrapper
                or _expression_is_external_tag(left, comparison_external)
            )
        )
        right_external = (
            not right_computed
            and not _expression_uses_plain_hash(right, plain_hash_values)
            and (
                right_external_wrapper
                or _expression_is_external_tag(right, comparison_external)
            )
        )
        if _c_style_comparison_has_connected_callsite(
            language,
            analyzer,
            source,
            position,
            left,
            right,
            computed,
            producer_names,
        ):
            return True
        if left_computed and _c_style_parameter_has_external_callsite(
            language,
            analyzer,
            source,
            position,
            right,
            parameters,
            computed,
        ):
            return True
        if right_computed and _c_style_parameter_has_external_callsite(
            language,
            analyzer,
            source,
            position,
            left,
            parameters,
            computed,
        ):
            return True
        if (left_computed and right_external) or (
            right_computed and left_external
        ):
            return True
        if (
            left_computed
            and right_computed
            and not required_external
            and not _c_style_malformed_derived_mac_operand(
                left, assignments, direct_mac_values
            )
            and not _c_style_malformed_derived_mac_operand(
                right, assignments, direct_mac_values
            )
        ):
            return True
        if language == "typescript" and (left_computed or right_computed):
            other = right if left_computed else left
            if re.search(
                r"\bBuffer\s*\.\s*(?:alloc|from)\s*\(",
                other,
                flags=re.IGNORECASE,
            ):
                return True
        saw_computed_comparison |= left_computed or right_computed
    return False


def _c_style_external_conversion_call(
    expression: str,
    functions: tuple[tuple[str, int, int, tuple[str, ...]], ...],
    producer_names: set[str],
    external: set[str],
) -> bool:
    """Recognize a source-defined conversion around a supplied tag value."""

    match = re.fullmatch(
        r"\s*(?P<function>[A-Za-z_$][\w$]*)\s*\(\s*(?P<argument>.*)\s*\)\s*",
        expression,
        flags=re.DOTALL,
    )
    if match is None:
        return False
    function = match.group("function")
    declared = {name for name, _start, _end, _parameters in functions}
    return (
        function in declared
        and function not in producer_names
        and _expression_is_external_tag(match.group("argument"), external)
    )


def _c_style_verifier_call_connects_tag(
    language: Language,
    analyzer: object,
    source: str,
    functions: tuple[tuple[str, int, int, tuple[str, ...]], ...],
    producer_names: set[str],
) -> bool:
    """Recognize a comparison wrapper called with a computed and supplied tag."""

    verifier_names = {
        name
        for name, _start, _end, _parameters in functions
        if _looks_like_verification_name(name)
        or re.search(r"(?:signature|tag|digest|mac)", name, re.IGNORECASE)
    }
    if not verifier_names:
        return False
    assignments = [
        ({name}, expression)
        for name, expression in _assignments(language, source).items()
    ]
    computed, external = _value_roles(assignments, set())
    computed = _expanded_c_style_computed_values(
        source, assignments, computed, producer_names
    )
    external = _expanded_c_style_external_values(source, assignments, external)
    external.difference_update(computed)
    for call in analyzer.calls(*verifier_names):
        if len(call.arguments) < 2:
            continue
        roles = [
            (
                _expression_is_computed(argument, computed)
                or _expression_calls_any(argument, producer_names),
                _expression_is_external_tag(argument, external),
            )
            for argument in call.arguments
        ]
        if any(role[0] for role in roles) and any(
            external_role and not computed_role
            for computed_role, external_role in roles
        ):
            return True
    return False


def _c_style_expected_literal(expression: str) -> bool:
    return bool(
        re.fullmatch(r"\s*(['\"]).*\1\s*", expression, re.DOTALL)
        or re.fullmatch(
            r"\s*new\s+(?:byte|Buffer)\s*\[?[^\{]*\{.*\}\s*",
            expression,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


def _typescript_verifier_receives_computed_tag(
    source: str,
    analyzer: object,
    functions: tuple[tuple[str, int, int, tuple[str, ...]], ...],
    producer_names: set[str],
) -> bool:
    """Follow a computed tag into a comparison helper's tag parameter."""

    assignments = [
        ({name}, expression)
        for name, expression in _assignments("typescript", source).items()
    ]
    computed, _external = _value_roles(assignments, set())
    computed = _expanded_c_style_computed_values(
        source,
        assignments,
        computed,
        producer_names,
    )
    for name, start, end, parameters in functions:
        if not _looks_like_verification_name(name):
            continue
        body = source[start:end]
        executable_body = body.split("{", 1)[1] if "{" in body else body
        tag_positions = {
            index
            for index, parameter in enumerate(parameters)
            if _is_external_tag_name(parameter)
            and re.search(
                rf"(?<![\w$]){re.escape(parameter)}(?![\w$])",
                executable_body,
            )
        }
        if not tag_positions:
            continue
        if not re.search(r"(?:===|!==|==|!=|timingSafeEqual)", body):
            continue
        for call in getattr(analyzer, "calls")(name):
            if start <= call.start <= end:
                continue
            for position in tag_positions:
                if position >= len(call.arguments):
                    continue
                argument = call.arguments[position]
                if _expression_is_computed(argument, computed) or _expression_calls_any(
                    argument,
                    producer_names,
                ):
                    return True
    return False


def _c_style_member_mac_roles(
    source: str,
    analyzer: object,
    functions: tuple[tuple[str, int, int, tuple[str, ...]], ...],
    producer_names: set[str],
) -> tuple[set[str], set[str]]:
    """Recover MAC roles carried through object fields across methods."""

    assignments: list[tuple[set[str], str]] = []
    for assignment in getattr(analyzer, "assignments", ()):
        names = re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", assignment.name)
        if names:
            assignments.append(({names[-1]}, assignment.expression))
    parameters = {
        parameter
        for _name, _start, _end, function_parameters in functions
        for parameter in function_parameters
    }
    computed, external = _value_roles(assignments, parameters)
    computed = _expanded_c_style_computed_values(
        source,
        assignments,
        computed,
        producer_names,
    )
    external = _expanded_c_style_external_values(
        source,
        assignments,
        external,
    )
    external.difference_update(computed)
    preferred = _preferred_external_values(assignments, parameters, computed)
    return computed, preferred or external


def _c_style_comparison_has_connected_callsite(
    language: Language,
    analyzer: object,
    source: str,
    comparison_position: int,
    left: str,
    right: str,
    local_computed: set[str],
    producer_names: set[str],
) -> bool:
    """Resolve a comparison helper's operands from its concrete callers."""

    functions = _c_style_functions(language, analyzer)
    containing = [
        function
        for function in functions
        if function[1] <= comparison_position <= function[2]
    ]
    if not containing:
        return False
    function_name, _start, _end, parameters = max(
        containing,
        key=lambda function: function[1],
    )
    parameter_indexes = {name: index for index, name in enumerate(parameters)}
    local_assignments, _local_parameters = _scoped_c_style_values(
        language,
        source,
        analyzer,
        comparison_position,
    )
    operand_parameters = (
        _expression_parameter_dependencies(left, local_assignments, set(parameters)),
        _expression_parameter_dependencies(right, local_assignments, set(parameters)),
    )
    locally_computed = (
        _expression_is_computed(left, local_computed)
        or _expression_calls_any(left, producer_names),
        _expression_is_computed(right, local_computed)
        or _expression_calls_any(right, producer_names),
    )
    if not any(operand_parameters):
        return False

    global_assignments = [({name}, expression) for name, expression in _assignments(language, source).items()]
    global_computed, global_external = _value_roles(global_assignments, set())
    global_computed = _expanded_c_style_computed_values(
        source,
        global_assignments,
        global_computed,
        producer_names,
    )
    global_external.difference_update(global_computed)

    for call in getattr(analyzer, "calls")(function_name):
        if _start <= call.start <= _end:
            continue
        roles: list[tuple[bool, bool]] = []
        for expression, names in zip((left, right), operand_parameters):
            is_computed = _expression_is_computed(expression, local_computed)
            is_external = False
            for name in names:
                index = parameter_indexes[name]
                if index >= len(call.arguments):
                    continue
                argument = call.arguments[index]
                argument_computed = _expression_is_computed(
                    argument,
                    global_computed,
                ) or _expression_calls_any(argument, producer_names)
                is_computed |= argument_computed
                is_external |= not argument_computed and _expression_is_external_tag(
                    argument,
                    global_external,
                )
            roles.append((is_computed, is_external))
        if (
            roles[0][0]
            and not roles[1][0]
            and roles[1][1]
        ) or (
            roles[1][0]
            and not roles[0][0]
            and roles[0][1]
        ):
            return True
        if (
            roles[0][0]
            and roles[1][0]
            and not all(locally_computed)
            and _looks_like_verification_name(function_name)
        ):
            return True
    return False


def _looks_like_verification_name(name: str) -> bool:
    return re.search(r"(?:verify|check|match|equal|compare)", name, re.IGNORECASE) is not None


def _c_style_functions(
    language: Language,
    analyzer: object,
) -> tuple[tuple[str, int, int, tuple[str, ...]], ...]:
    """Return a common function view for the indexed C-style languages."""

    if language == "java":
        return tuple(
            (
                method.name,
                method.body_start,
                method.body_end,
                tuple(parameter.name for parameter in method.params),
            )
            for method in getattr(getattr(analyzer, "flow", None), "methods", ())
        )
    return tuple(
        (function.name, function.start, function.end, tuple(function.parameters))
        for function in getattr(analyzer, "functions", ())
    )


def _c_style_mac_producer_names(
    source: str,
    functions: tuple[tuple[str, int, int, tuple[str, ...]], ...],
) -> set[str]:
    """Find helpers that return a MAC, including ordinary wrapper helpers."""

    bodies: dict[str, list[str]] = {}
    for name, start, end, _parameters in functions:
        if name:
            bodies.setdefault(name, []).append(source[start:end])
    producers: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, overloads in bodies.items():
            if name in producers:
                continue
            if any(
                _c_style_body_returns_mac(body, producers)
                for body in overloads
            ):
                producers.add(name)
                changed = True
    return producers


def _c_style_body_returns_mac(body: str, producers: set[str]) -> bool:
    assignments = [
        (
            {match.group("name")},
            match.group("expression"),
        )
        for match in re.finditer(
            r"\b(?P<name>[A-Za-z_$][\w$]*)\s*"
            r"(?<![=!<>])=(?!=)\s*(?P<expression>[^;]+);",
            body,
            flags=re.DOTALL,
        )
    ]
    local_computed, _external = _value_roles(assignments, set())
    local_computed = _expanded_c_style_computed_values(
        body,
        assignments,
        local_computed,
        producers,
    )
    returned = re.findall(r"\breturn\s+([^;]+);", body, flags=re.DOTALL)
    return any(
        not re.search(
            r"(?:===|!==|==|!=|timingSafeEqual|compare_digest|"
            r"FixedTimeEquals|SequenceEqual|\.\s*equals\s*\()",
            expression,
            flags=re.IGNORECASE,
        )
        and (
            _expression_has_mac_computation(expression)
            or _expression_calls_any(expression, producers)
            or bool(_identifier_names(expression) & local_computed)
        )
        for expression in returned
    )


def _expression_parameter_dependencies(
    expression: str,
    assignments: list[tuple[set[str], str]],
    parameters: set[str],
) -> set[str]:
    """Trace a comparison operand back to parameters of its local helper."""

    names = _identifier_names(expression)
    changed = True
    while changed:
        changed = False
        for targets, value in assignments:
            if not (targets & names):
                continue
            before = len(names)
            names.update(_identifier_names(value))
            changed |= len(names) != before
    return names & parameters


def _expanded_c_style_computed_values(
    source: str,
    assignments: list[tuple[set[str], str]],
    initial: set[str],
    producer_names: set[str],
) -> set[str]:
    """Propagate computed MAC bytes through ordinary representation helpers."""

    computed = set(initial)
    changed = True
    while changed:
        changed = False
        for names, expression in assignments:
            if not (
                _identifier_names(expression) & computed
                or _expression_calls_any(expression, producer_names)
            ):
                continue
            before = len(computed)
            computed.update(names)
            changed |= len(computed) != before

        for match in re.finditer(
            r"\bfor\s*\([^:;]*\b(?P<item>[A-Za-z_$][\w$]*)\s*:\s*"
            r"(?P<source>[A-Za-z_$][\w$]*)\s*\)",
            source,
        ):
            if match.group("source") in computed and match.group("item") not in computed:
                computed.add(match.group("item"))
                changed = True

        for match in re.finditer(
            r"\b(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*"
            r"(?:append|put|putLong|set|write|update)\s*"
            r"\((?P<arguments>[^;)]*)\)",
            source,
            flags=re.IGNORECASE,
        ):
            if _identifier_names(match.group("arguments")) & computed:
                receiver = match.group("receiver")
                if receiver not in computed:
                    computed.add(receiver)
                    changed = True

        for match in re.finditer(
            r"\b(?P<target>[A-Za-z_$][\w$]*)\s*\+=\s*"
            r"(?P<value>[^;]+);",
            source,
            flags=re.DOTALL,
        ):
            if (
                _identifier_names(match.group("value")) & computed
                and match.group("target") not in computed
            ):
                computed.add(match.group("target"))
                changed = True

        for match in re.finditer(
            r"\b(?:System\s*\.\s*)?arraycopy\s*\(\s*"
            r"(?P<source>[A-Za-z_$][\w$]*)\s*,[^,]*,\s*"
            r"(?P<target>[A-Za-z_$][\w$]*)\s*,",
            source,
            flags=re.IGNORECASE,
        ):
            if (
                match.group("source") in computed
                and match.group("target") not in computed
            ):
                computed.add(match.group("target"))
                changed = True

        for match in re.finditer(
            r"\b(?P<source>[A-Za-z_$][\w$]*)\s*\.\s*copy\s*\(\s*"
            r"(?P<target>[A-Za-z_$][\w$]*)\b",
            source,
        ):
            if (
                match.group("source") in computed
                and match.group("target") not in computed
            ):
                computed.add(match.group("target"))
                changed = True

        for match in re.finditer(
            r"\b(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*get\s*\(\s*"
            r"(?P<target>[A-Za-z_$][\w$]*)\s*\)",
            source,
        ):
            if (
                match.group("receiver") in computed
                and match.group("target") not in computed
            ):
                computed.add(match.group("target"))
                changed = True

        for match in re.finditer(
            r"\bByteBuffer\s*\.\s*wrap\s*\(\s*"
            r"(?P<source>[A-Za-z_$][\w$]*)\s*\)\s*\.\s*get\s*\(\s*"
            r"(?P<target>[A-Za-z_$][\w$]*)\s*\)",
            source,
        ):
            if (
                match.group("source") in computed
                and match.group("target") not in computed
            ):
                computed.add(match.group("target"))
                changed = True

        for match in re.finditer(
            r"\b(?P<target>[A-Za-z_$][\w$]*)\s*\[[^\]]+\]\s*="
            r"\s*(?P<value>[^;]+);",
            source,
            flags=re.DOTALL,
        ):
            if (
                _identifier_names(match.group("value")) & computed
                and match.group("target") not in computed
            ):
                computed.add(match.group("target"))
                changed = True
    return computed


def _expanded_c_style_external_values(
    source: str,
    assignments: list[tuple[set[str], str]],
    initial: set[str],
) -> set[str]:
    """Propagate a supplied tag through aliases and mutable containers."""

    external = _parameter_value_flow(assignments, initial)
    changed = True
    while changed:
        changed = False
        for match in re.finditer(
            r"\b(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*"
            r"(?:append|put|putLong|write|update)\s*"
            r"\((?P<arguments>[^;]*)\)\s*;",
            source,
            flags=re.IGNORECASE,
        ):
            if (
                _identifier_names(match.group("arguments")) & external
                and match.group("receiver") not in external
            ):
                external.add(match.group("receiver"))
                changed = True
        for match in re.finditer(
            r"\bByteBuffer\s*\.\s*wrap\s*\(\s*"
            r"(?P<target>[A-Za-z_$][\w$]*)[^)]*\)\s*\.\s*putLong\s*"
            r"\((?P<arguments>[^;]*)\)\s*;",
            source,
            flags=re.DOTALL,
        ):
            if (
                _identifier_names(match.group("arguments")) & external
                and match.group("target") not in external
            ):
                external.add(match.group("target"))
                changed = True
        for match in re.finditer(
            r"\b(?P<target>[A-Za-z_$][\w$]*)\s*\[[^\]]+\]\s*="
            r"\s*(?P<value>[^;]+);",
            source,
            flags=re.DOTALL,
        ):
            if (
                _identifier_names(match.group("value")) & external
                and match.group("target") not in external
            ):
                external.add(match.group("target"))
                changed = True
        for match in re.finditer(
            r"\b(?:System\s*\.\s*)?(?:arraycopy|Array\s*\.\s*Copy)\s*\(\s*"
            r"(?P<source>[A-Za-z_$][\w$]*)\s*,[^,]*,\s*"
            r"(?P<target>[A-Za-z_$][\w$]*)\s*,",
            source,
            flags=re.IGNORECASE,
        ):
            if (
                match.group("source") in external
                and match.group("target") not in external
            ):
                external.add(match.group("target"))
                changed = True
        for match in re.finditer(
            r"\b(?P<source>[A-Za-z_$][\w$]*)\s*\.\s*copy\s*\(\s*"
            r"(?P<target>[A-Za-z_$][\w$]*)\b",
            source,
        ):
            if (
                match.group("source") in external
                and match.group("target") not in external
            ):
                external.add(match.group("target"))
                changed = True
    return external


def _c_style_plain_hash_values(
    assignments: list[tuple[set[str], str]],
) -> set[str]:
    """Track ordinary digest values so they cannot stand in for a MAC."""

    plain_objects: set[str] = set()
    plain_values: set[str] = set()
    for names, expression in assignments:
        if re.search(
            r"\b(?:createHash\s*\(|MessageDigest\s*\.\s*getInstance\s*\(|"
            r"new\s+(?:SHA\d*|MD5)(?:CryptoServiceProvider|Managed)?\b|"
            r"(?:SHA\d*|MD5)\s*\.\s*Create\s*\()",
            expression,
            flags=re.IGNORECASE,
        ):
            plain_objects.update(names)

    changed = True
    while changed:
        changed = False
        for names, expression in assignments:
            if re.search(
                r"\b(?:new\s+HMAC[A-Za-z0-9_]*|createHmac|Mac\s*\.\s*getInstance|"
                r"hmac\s*\.\s*new)\b",
                expression,
                flags=re.IGNORECASE,
            ):
                continue
            identifiers = _identifier_names(expression)
            direct_plain_terminal = bool(
                re.search(
                    r"\b(?:SHA\d*|MD5)\s*\.\s*(?:HashData|Create)\b|"
                    r"\bnew\s+(?:SHA\d*|MD5)(?:CryptoServiceProvider|Managed)?"
                    r"\s*\([^)]*\)\s*\.\s*ComputeHash\s*\(|"
                    r"\b(?:crypto\s*\.\s*)?createHash\s*\([^;]+"
                    r"\.\s*digest\s*\(",
                    expression,
                    flags=re.IGNORECASE,
                )
            )
            receiver_terminal = bool(
                identifiers & plain_objects
                and re.search(
                    r"\.\s*(?:digest|ComputeHash|GetHashAndReset)\s*\(",
                    expression,
                    flags=re.IGNORECASE,
                )
            )
            if direct_plain_terminal or receiver_terminal or identifiers & plain_values:
                before = len(plain_values)
                plain_values.update(names)
                changed |= len(plain_values) != before
    return plain_values


def _c_style_malformed_derived_mac_operand(
    expression: str,
    assignments: list[tuple[set[str], str]],
    direct_mac_values: set[str],
) -> bool:
    """Reject a sliced key-plus-MAC construction used as an expected tag.

    Truncating a MAC can be a legitimate protocol.  This check is deliberately
    narrower: it only rejects slicing a value whose construction combines a
    directly computed MAC with an identifier that is visibly key material.
    """

    sliced = re.search(
        r"\b(?P<name>[A-Za-z_$][\w$]*)\s*\.\s*"
        r"(?:slice|substring|substr)\s*\(",
        expression,
        flags=re.IGNORECASE,
    )
    if sliced is None:
        return False
    assignment_by_name = {
        name: value
        for names, value in assignments
        for name in names
    }
    value = assignment_by_name.get(sliced.group("name"), "")
    identifiers = _identifier_names(value)
    contains_mac = bool(identifiers & direct_mac_values) or _expression_has_mac_computation(
        value
    )
    contains_key = any(
        re.search(r"(?:key|secret|password|credential)", name, re.IGNORECASE)
        for name in identifiers
    )
    return contains_mac and contains_key


def _expression_uses_plain_hash(expression: str, values: set[str]) -> bool:
    return bool(_identifier_names(expression) & values) or bool(
        re.search(
            r"\b(?:SHA\d*|MD5)\s*\.\s*HashData\s*\(|"
            r"\bnew\s+(?:SHA\d*|MD5)(?:CryptoServiceProvider|Managed)?"
            r"\s*\([^)]*\)\s*\.\s*ComputeHash\s*\(",
            expression,
            flags=re.IGNORECASE,
        )
    )


def _value_roles(
    assignments: list[tuple[set[str], str]],
    parameters: set[str],
) -> tuple[set[str], set[str]]:
    computed: set[str] = set()
    external = set(parameters)

    changed = True
    while changed:
        changed = False
        for names, expression in assignments:
            identifiers = _identifier_names(expression)
            expression_computed = _expression_has_mac_computation(expression) or bool(
                identifiers & computed
            ) and re.search(r"\bnew\s+byte\s*\[", expression, re.IGNORECASE) is None
            if expression_computed:
                before = len(computed)
                computed.update(names)
                changed |= len(computed) != before
                continue
            expression_external = bool(identifiers & external) or (
                any(_is_external_value_name(name) for name in names)
                and _expression_supplies_external_value(expression)
            )
            if expression_external:
                before = len(external)
                external.update(names)
                changed |= len(external) != before
    external.difference_update(computed)
    return computed, external


def _parameter_value_flow(
    assignments: list[tuple[set[str], str]], seeds: set[str]
) -> set[str]:
    """Propagate caller-supplied integrity values through local aliases."""

    values = set(seeds)
    changed = True
    while changed:
        changed = False
        for names, expression in assignments:
            if not (_identifier_names(expression) & values):
                continue
            before = len(values)
            values.update(names)
            changed |= len(values) != before
    return values


def _preferred_external_values(
    assignments: list[tuple[set[str], str]],
    parameters: set[str],
    computed: set[str],
) -> set[str]:
    priorities = {
        name: priority
        for name in parameters
        if (priority := _external_name_priority(name)) is not None
    }
    changed = True
    while changed:
        changed = False
        for names, expression in assignments:
            if names & computed or _expression_has_mac_computation(expression):
                continue
            identifiers = _identifier_names(expression)
            inherited = [priorities[name] for name in identifiers if name in priorities]
            seeded = [
                priority
                for name in names
                if (priority := _external_name_priority(name)) is not None
                and _expression_supplies_external_value(expression)
            ]
            candidates = inherited + seeded
            if not candidates:
                continue
            priority = min(candidates)
            for name in names:
                previous = priorities.get(name)
                if previous is None or priority < previous:
                    priorities[name] = priority
                    changed = True
    for name in computed:
        priorities.pop(name, None)
    if not priorities:
        return set()
    best = min(priorities.values())
    return {name for name, priority in priorities.items() if priority == best}


def _external_name_priority(name: str) -> int | None:
    compact = re.sub(r"[^a-z0-9]", "", name.lower())
    if (
        re.search(r"(?:data|message|payload|record|body|request)", compact)
        and not re.search(
            r"(?:mac|hmac|tag|digest|signature|authcode|authenticationcode|header)",
            compact,
        )
    ):
        return None
    if any(
        marker in compact
        for marker in (
            "expected",
            "provided",
            "supplied",
            "received",
            "incoming",
            "header",
            "verification",
            "actual",
            "callback",
        )
    ):
        return 1
    if compact in {
        "mac",
        "hmac",
        "tag",
        "digest",
        "signature",
        "authcode",
        "authenticationcode",
    }:
        return 2
    if re.search(r"(?:tag|digest|signature)(?:str|string|hex|b64|value|bytes)$", compact):
        return 2
    return 3 if _is_external_tag_name(name) else None


def _is_external_value_name(name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", name.lower())
    return _is_external_tag_name(name) or any(
        marker in compact
        for marker in (
            "expected",
            "provided",
            "supplied",
            "received",
            "incoming",
            "header",
            "userinput",
        )
    )


def _expression_has_mac_computation(expression: str) -> bool:
    return bool(
        _MAC_TERMINAL.search(expression)
        or _COMPUTED_HELPER.search(expression)
        or re.search(r"\b(?:self\.)?sign\s*\(", expression, re.IGNORECASE)
        or re.search(
            r"\b(?:mac|hmac)[\w$]*(?:generator|compute|calculate|create|generate)"
            r"[\w$]*\s*\(",
            expression,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:Mac\s*\.\s*getInstance|createHmac|"
            r"new\s+HMAC[A-Za-z0-9_]*|hmac\s*\.\s*new)\s*\(",
            expression,
            re.IGNORECASE,
        )
        or re.search(
            r"\b[A-Za-z_$][\w$]*(?:mac|hmac)[\w$]*\s*\.\s*"
            r"(?:doFinal|digest|hexdigest|finalize|ComputeHash|GetHashAndReset)\s*\(",
            expression,
            re.IGNORECASE,
        )
        or re.search(
            r"\bmac\s*\.\s*(?:doFinal|digest|hexdigest|finalize|ComputeHash)\s*\(",
            expression,
            re.IGNORECASE,
        )
        or re.search(
            r"\b[A-Za-z_$][A-Za-z0-9_$]*(?:mac|hmac)[A-Za-z0-9_$]*"
            r"\s*\.\s*Hash\b",
            expression,
            re.IGNORECASE,
        )
    )


def _expression_supplies_external_value(expression: str) -> bool:
    stripped = expression.strip()
    return bool(
        re.fullmatch(
            r"(?:[rubfRUBF]*)?(?:'.*'|\".*\")\s*\)*(?:\s*\{)?\s*",
            stripped,
            re.DOTALL,
        )
        or bool(
            re.search(
                r"\bnew\s+byte\s*\[[^\]]*\](?:\s*\{)?",
                expression,
                re.IGNORECASE,
            )
        )
        or bool(re.fullmatch(r"\s*\{[^{}]*\}\s*", stripped, re.DOTALL))
        or bool(
            re.fullmatch(
                r"(?:[rubfRUBF]*)?(?:'.*'|\".*\")\s*\.\s*"
                r"(?:getBytes|encode)\s*\([^)]*\)",
                stripped,
                re.DOTALL,
            )
        )
        or bool(
            re.search(
                r"\b(?:Buffer\s*\.\s*from|GetBytes|FromHexString|"
                r"FromBase64String)\s*\(\s*['\"]",
                expression,
                re.IGNORECASE,
            )
        )
        or re.search(
            r"(?:\binput\s*\(|\bgetpass\b|\b(?:request|headers?|args?|argv)\b|"
            r"\b(?:read|nextLine|receive|provided|incoming)\w*\s*\(|"
            r"\b(?:decode|fromHexString|fromhex)\s*\()",
            expression,
            re.IGNORECASE,
        )
    )


def _expression_is_computed(expression: str, computed: set[str]) -> bool:
    content_expression = expression
    for name in computed:
        content_expression = re.sub(
            rf"\b{re.escape(name)}\s*\.\s*(?:length|Length)\b",
            "",
            content_expression,
        )
    if _expression_has_mac_computation(content_expression) or bool(
        any(
            re.search(
                rf"(?<![\w$]){re.escape(name)}(?![\w$]|\s*\()",
                content_expression,
            )
            for name in computed
        )
    ):
        return True
    return any(
        re.search(rf"\{{[^}}]*\b{re.escape(name)}\b[^}}]*\}}", expression)
        for name in computed
    )


def _expression_is_external_tag(expression: str, external: set[str]) -> bool:
    identifiers = _identifier_names(expression)
    named_as_tag = bool(_TAG_NAME.search(expression)) or any(
        marker in expression.lower()
        for marker in (
            "expected",
            "provided",
            "supplied",
            "received",
            "incoming",
            "legacy",
            "actual",
        )
    )
    explicit_external_name = any(
        marker in expression.lower()
        for marker in (
            "expected",
            "provided",
            "supplied",
            "received",
            "incoming",
            "legacy",
            "actual",
        )
    )
    external_transport = any(
        re.search(
            r"(?:request|header|provided|supplied|received|incoming|tag|digest|signature)",
            name,
            re.IGNORECASE,
        )
        for name in identifiers & external
    )
    object_property = bool(
        named_as_tag
        and re.search(r"\b(?:this|self|request|headers?)\s*(?:\.|\[)", expression)
    )
    parameter_property = any(
        re.search(rf"\b{re.escape(name)}\s*(?:\.|\[)", expression)
        for name in identifiers & external
    )
    literal_tag = bool(
        _expression_supplies_external_value(expression)
        and named_as_tag
        and not _expression_is_mac_algorithm_literal(expression)
    )
    return (
        explicit_external_name
        or external_transport
        or object_property
        or parameter_property
        or bool(identifiers & external and named_as_tag)
        or literal_tag
    )


def _expression_is_mac_algorithm_literal(expression: str) -> bool:
    """Distinguish an algorithm selector from a literal received tag."""

    match = re.fullmatch(
        r"\s*(['\"])(?P<value>[^'\"]+)\1\s*\)*(?:\s*\{)?\s*",
        expression,
    )
    if match is None:
        return False
    compact = re.sub(r"[-_]", "", match.group("value").lower())
    return re.fullmatch(r"(?:hmac)?(?:md5|sha\d+)", compact) is not None


def _c_style_parameter_has_external_callsite(
    language: Language,
    analyzer: object,
    source: str,
    comparison_position: int,
    expression: str,
    parameters: set[str],
    computed: set[str],
) -> bool:
    """Recognize a compared parameter whose callers supply the tag value.

    Generated code often gives the tag parameter an unhelpful name such as
    ``data`` or ``secret``.  The call site is authoritative in that case.  A
    parameter fed by a locally computed MAC is not an external tag; this keeps
    ignored-tag cases distinct from merely misnamed parameters.
    """

    compared = _identifier_names(expression) & parameters
    if not compared:
        return False
    functions = tuple(getattr(analyzer, "functions", ()))
    containing = [
        function
        for function in functions
        if function.start <= comparison_position <= function.end
    ]
    if not containing:
        return False
    function = max(containing, key=lambda item: item.start)
    parameter_order = tuple(function.parameters)
    parameter_indexes = {
        index for index, name in enumerate(parameter_order) if name in compared
    }
    if not parameter_indexes:
        return False

    assignments = [
        (
            set(re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", assignment.name)),
            assignment.expression,
        )
        for assignment in getattr(analyzer, "assignments", ())
    ]
    globally_computed, _ = _value_roles(assignments, set())
    for call in getattr(analyzer, "calls", lambda *names: ())(function.name):
        if function.start <= call.start <= function.end:
            continue
        for index in parameter_indexes:
            if index >= len(call.arguments):
                continue
            argument = call.arguments[index]
            if _expression_is_computed(argument, globally_computed):
                continue
            resolved = getattr(analyzer, "resolve_strings")(argument, call.start)
            if resolved.values or _expression_supplies_external_value(argument):
                return True
    return False


def _normalized_expression(expression: str) -> str:
    return re.sub(r"\s+", "", expression).strip("()")


def _python_scope(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    module: ast.Module,
) -> ast.Module | ast.FunctionDef | ast.AsyncFunctionDef:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return module


def _python_scope_parameters(
    scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    if isinstance(scope, ast.Module):
        return set()
    return {
        parameter.arg
        for parameter in (
            *scope.args.posonlyargs,
            *scope.args.args,
            *scope.args.kwonlyargs,
        )
    }


def _python_target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {
            name
            for element in target.elts
            for name in _python_target_names(element)
        }
    return set()


def _python_paired_assignment_elements(
    node: ast.Assign | ast.AnnAssign,
) -> tuple[tuple[ast.AST, ast.AST], ...]:
    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
    pairs: list[tuple[ast.AST, ast.AST]] = []
    for target in targets:
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
            node.value, (ast.Tuple, ast.List)
        ):
            pairs.extend(zip(target.elts, node.value.elts))
        else:
            pairs.append((target, node.value))
    return tuple(pairs)


def _scoped_c_style_values(
    language: Language,
    source: str,
    analyzer: object,
    position: int,
) -> tuple[list[tuple[set[str], str]], set[str]]:
    if language in {"typescript", "csharp"}:
        functions = tuple(getattr(analyzer, "functions", ()))
        containing = [
            function
            for function in functions
            if function.start <= position <= function.end
        ]
        scope = max(containing, key=lambda item: item.start) if containing else None
        assignments = []
        for assignment in getattr(analyzer, "assignments", ()):
            if assignment.start > position:
                continue
            owner = next(
                (
                    function
                    for function in sorted(
                        functions, key=lambda item: item.start, reverse=True
                    )
                    if function.start <= assignment.start <= function.end
                ),
                None,
            )
            if owner != scope:
                continue
            identifiers = re.findall(
                r"[A-Za-z_$][A-Za-z0-9_$]*", assignment.name
            )
            names = {identifiers[-1]} if identifiers else set()
            if names:
                assignments.append((names, assignment.expression))
        parameters = set(scope.parameters) if scope is not None else set()
        return assignments, parameters

    methods = tuple(getattr(getattr(analyzer, "flow", None), "methods", ()))
    containing = [
        method
        for method in methods
        if method.body_start <= position <= method.body_end
    ]
    scope = max(containing, key=lambda item: item.body_start) if containing else None
    assignments = []
    for match in re.finditer(
        r"\b(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*"
        r"(?<![=!<>])=(?!=)\s*(?P<expression>[^;]+);",
        source,
        flags=re.DOTALL,
    ):
        if match.start() > position:
            continue
        owner = next(
            (
                method
                for method in sorted(
                    methods, key=lambda item: item.body_start, reverse=True
                )
                if method.body_start <= match.start() <= method.body_end
            ),
            None,
        )
        if owner == scope:
            assignments.append(
                ({match.group("name")}, match.group("expression"))
            )
    parameters = (
        {parameter.name for parameter in scope.params}
        if scope is not None
        else set()
    )
    return assignments, parameters


def _receiver_before_call(source: str, call_start: int) -> str:
    index = call_start - 1
    while index >= 0 and source[index].isspace():
        index -= 1
    if index < 0 or source[index] != ".":
        return ""
    index -= 1
    end = index + 1
    round_depth = 0
    square_depth = 0
    while index >= 0:
        char = source[index]
        if char == ")":
            round_depth += 1
        elif char == "(":
            if round_depth:
                round_depth -= 1
            else:
                break
        elif char == "]":
            square_depth += 1
        elif char == "[":
            if square_depth:
                square_depth -= 1
        elif round_depth == square_depth == 0 and char in "=;{},\n":
            break
        index -= 1
    return source[index + 1 : end].strip()


def _assignments(language: Language, source: str) -> dict[str, str]:
    if language == "python":
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}
        values: dict[str, str] = {}
        for node in ast.walk(tree):
            if (
                not isinstance(node, (ast.Assign, ast.AnnAssign))
                or node.value is None
            ):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    values[target.id] = ast.unparse(node.value)
        return values
    if language in {"typescript", "csharp"}:
        analyzer = analyzer_for(language, source)
        return {
            re.sub(r"\s+", "", assignment.name): assignment.expression
            for assignment in getattr(analyzer, "assignments", ())
        }

    values: dict[str, str] = {}
    for match in re.finditer(
        r"\b(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*(?<![=!<>])=(?!=)\s*"
        r"(?P<expression>[^;]+);",
        source,
        flags=re.DOTALL,
    ):
        values[match.group("name")] = match.group("expression")
    return values


def _is_external_tag_name(name: str) -> bool:
    lowered = name.lower()
    return bool(_TAG_NAME.search(name)) and not any(
        marker in lowered
        for marker in (
            "algorithm",
            "digestmod",
            "hashname",
            "macname",
            "key",
            "secret",
        )
    )


def _source_line(source: str, start: int, end: int) -> str:
    line_start = source.rfind("\n", 0, start) + 1
    line_end = source.find("\n", end)
    return source[line_start : len(source) if line_end < 0 else line_end]


def _identifier_names(expression: str) -> set[str]:
    without_literals = re.sub(
        r"(?:'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")",
        "",
        expression,
    )
    return set(re.findall(r"\b[A-Za-z_$][A-Za-z0-9_$]*\b", without_literals))


def _function_parameter_names(
    language: Language,
    source: str,
    analyzer: object,
) -> set[str]:
    if language in {"typescript", "csharp"}:
        return {
            parameter
            for function in getattr(analyzer, "functions", ())
            for parameter in function.parameters
        }
    if language == "python":
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()
        return {
            parameter.arg
            for function in ast.walk(tree)
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
            for parameter in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        }

    names: set[str] = set()
    patterns = (
        r"\b[A-Za-z_$][\w$<>\[\],.? ]+\s+[A-Za-z_$][\w$]*\s*"
        r"\((?P<parameters>[^)]*)\)\s*(?:throws\b[^\{]*)?\{",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, source, flags=re.DOTALL):
            for parameter in match.group("parameters").split(","):
                identifiers = re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", parameter)
                if identifiers:
                    names.add(identifiers[-1])
    return names


def _python_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _python_name(node.value)
        return f"{head}.{node.attr}" if head else node.attr
    return ""


__all__ = ("mac_generation_failure", "mac_verification_failure")
