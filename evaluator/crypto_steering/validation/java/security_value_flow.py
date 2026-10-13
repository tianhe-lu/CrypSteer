"""Shared javalang-backed value-flow analysis for security-sensitive Java values."""

from __future__ import annotations

from dataclasses import dataclass
import re

from crypto_steering.validation.java.models import ProgramLabel
from crypto_steering.validation.java.ast_adapter import JavaAstContext, build_java_ast_context
from crypto_steering.validation.java.lightweight_ast import find_method_calls, method_containing_index
from crypto_steering.validation.java.source import find_matching_delimiter, mask_java_comments_and_literals
from crypto_steering.validation.java.static_analysis import (
    Assignment,
    simple_variable,
    strip_outer_expression,
)


@dataclass(frozen=True)
class SourceClassification:
    label: ProgramLabel
    reason: str


_STRING_LITERAL = r'"(?:\\.|[^"\\])*"'
_SECURITY_VALUE_NAMES = re.compile(
    r"(?:key|secret|token|iv|nonce|salt|seed|password|pwd|passphrase|credential)",
    flags=re.IGNORECASE,
)
_PASSWORD_VALUE_NAMES = re.compile(
    r"(?:password|passphrase|pwd|credential)",
    flags=re.IGNORECASE,
)


class SecurityValueFlow:
    """Resolve common generated-Java source flows for security-sensitive values.

    This is intentionally conservative. It uses a required javalang parse for
    method/parameter scope, detects obvious constants and obvious secure
    runtime/generated sources, and leaves unclear source shapes as ``unknown``.
    """

    def __init__(self, source: str) -> None:
        self.ast_context: JavaAstContext = build_java_ast_context(source)
        self.source = self.ast_context.source
        self.masked_source = mask_java_comments_and_literals(self.source)
        self.assignments = self.ast_context.assignments
        self.methods = self.ast_context.methods
        self.class_names = frozenset(
            re.findall(r"\b(?:class|record|enum)\s+([A-Za-z_]\w*)\b", self.masked_source)
        )
        self._method_calls_cache = {}
        self._caller_parameter_cache = {}
        self._caller_parameter_in_progress: set[tuple[int, int, str]] = set()
        self._method_entry_reachability_cache: dict[int, bool] = {}
        self._method_entry_reachability_in_progress: set[int] = set()

    def classify(self, expression: str, index: int, role: str, seen: frozenset[str] = frozenset()) -> SourceClassification:
        expr = strip_outer_expression(expression)
        if not expr:
            return SourceClassification("unknown", f"empty {role} expression")
        if expr == "null":
            return SourceClassification("unknown", f"null {role} expression")

        if role == "salt" and self._is_predictable_salt_source(expr):
            return SourceClassification(
                "misuse",
                "salt is deterministically derived from a password or stable host identity",
            )

        conditional = self._classify_conditional(expr, index, role, seen)
        if conditional is not None:
            return conditional

        password_concatenation = self._classify_password_concatenation(
            expr,
            index,
            role,
            seen,
        )
        if password_concatenation is not None:
            return password_concatenation

        security_concatenation = self._classify_security_concatenation(
            expr,
            index,
            role,
            seen,
        )
        if security_concatenation is not None:
            return security_concatenation

        base64_source = self._base64_codec_source(expr, index)
        if base64_source is not None:
            return self.classify(base64_source, index, role, seen)

        unwrapped = self._unwrap(expr)
        if unwrapped != expr:
            return self.classify(unwrapped, index, role, seen)

        runtime_array_element = self._classify_runtime_array_element(
            expr,
            index,
            role,
            seen,
        )
        if runtime_array_element is not None:
            return runtime_array_element

        assigned_value = _assignment_expression_value(expr)
        if assigned_value is not None:
            return self.classify(assigned_value, index, role, seen)

        constructor_arg = self._constructor_sensitive_argument(expr, role)
        if constructor_arg is not None:
            return self.classify(constructor_arg, index, role, seen)

        derived_component = self._derived_component_source(expr)
        if derived_component is not None:
            return self.classify(derived_component, index, role, seen)

        byte_buffer_source = self._byte_buffer_array_source(expr, index)
        if byte_buffer_source is not None:
            return self.classify(byte_buffer_source, index, role, seen)

        digest_output_source = self._message_digest_output_source(expr, index)
        if digest_output_source is not None:
            digest_source = strip_outer_expression(digest_output_source)
            digest_key = f"{role}:message_digest:{expr}:{index}"
            if digest_source == expr or digest_key in seen:
                return SourceClassification(
                    "unknown",
                    f"recursive {role} MessageDigest flow could not be resolved",
                )
            return self.classify(
                digest_source,
                index,
                role,
                seen | frozenset({digest_key}),
            )

        if role == "prng_seed" and self._prng_expression_uses_predictable_variable(expr, index):
            return SourceClassification("misuse", f"{role} expression uses predictable variable source")

        structured_member = self._classify_structured_member(
            expr,
            index,
            role,
            seen,
        )
        if structured_member is not None:
            return structured_member

        indexed_array = self._classify_indexed_array_source(
            expr,
            index,
            role,
            seen,
        )
        if indexed_array is not None:
            return indexed_array

        if self._is_misuse_constant(expr, role):
            return SourceClassification("misuse", f"{role} source is hardcoded, fixed, or predictable")
        if self._is_secure_source(expr, index, role):
            return SourceClassification("secure", f"{role} comes from a generated or runtime source")

        transparent_arg = _transparent_conversion_call_argument(expr)
        if transparent_arg is not None:
            return self.classify(transparent_arg, index, role, seen)

        persisted_source = self._file_round_trip_source(expr, index)
        if persisted_source is not None:
            source_expression, source_index = persisted_source
            key = f"{role}:file-round-trip:{expr}:{index}"
            if key not in seen:
                return self.classify(
                    source_expression,
                    source_index,
                    role,
                    seen | frozenset({key}),
                )

        call = _simple_call_signature(expr)
        if call is not None:
            helper_name, helper_arity = call
            helper = self._classify_helper(helper_name, helper_arity, index, role, seen)
            if helper is not None:
                return helper

        variable = simple_variable(expr)
        if variable:
            key = f"{role}:{variable}:{index}"
            if key in seen:
                return SourceClassification("unknown", f"recursive {role} flow could not be resolved")
            if role == "prng_seed" and self._variable_filled_from_secure_random_seed(variable, index):
                return SourceClassification("secure", f"{role} variable is derived from SecureRandom.generateSeed")
            for_each_source = self._enhanced_for_each_source(variable, index)
            if for_each_source is not None:
                return self.classify(for_each_source, index, role, seen | frozenset({key}))
            copied_sources = self._variable_copied_sources(variable, index)
            if copied_sources:
                labels = [
                    self.classify(source, index, role, seen | frozenset({key}))
                    for source in copied_sources
                ]
                if role == "key_material":
                    base_assignment = self._latest_assignment(variable, index)
                    base = (
                        self.classify(
                            self._assignment_expression(base_assignment),
                            base_assignment.start,
                            role,
                            seen | frozenset({key}),
                        )
                        if base_assignment is not None
                        else None
                    )
                    if base is not None and base.label == "secure" and any(
                        item.label == "misuse" for item in labels
                    ):
                        return SourceClassification(
                            "misuse",
                            "generated key material is overwritten from a predictable source",
                        )
                    if any(item.label == "secure" for item in labels):
                        return SourceClassification(
                            "secure",
                            "key material retains bytes from a generated or runtime copied source",
                        )
                    if labels and all(item.label == "misuse" for item in labels):
                        return SourceClassification(
                            "misuse",
                            "key material is copied only from predictable sources",
                        )
                copied = _combine(
                    labels,
                    f"{role} variable is composed from copied sources",
                )
                if copied.label != "unknown":
                    return copied
            assignment = self._latest_assignment(variable, index)
            if assignment is not None:
                cipher_output = self._variable_filled_by_cipher_output(
                    variable,
                    assignment.end,
                    index,
                    role,
                    seen | frozenset({key}),
                )
                if cipher_output is not None:
                    return cipher_output
                predictable_overwrite = self._variable_predictably_overwritten(
                    variable,
                    assignment.end,
                    index,
                )
                if predictable_overwrite is not None:
                    return SourceClassification(
                        "misuse",
                        f"{role} variable is predictably overwritten after initialization: "
                        f"{predictable_overwrite}",
                    )
                if self._variable_filled_by_secure_random(variable, index, after=assignment.end):
                    return SourceClassification("secure", f"{role} variable is filled by SecureRandom")
                if role == "salt" and self._variable_filled_by_varying_salt_source(
                    variable,
                    index,
                    after=assignment.end,
                ):
                    return SourceClassification(
                        "secure",
                        "salt variable is filled by a varying source; randomness quality is evaluated by M05",
                    )
                assignment_expr = self._assignment_expression(assignment)
                if (
                    role == "password"
                    and self._password_variable_rejects_empty(
                        variable,
                        assignment.end,
                        index,
                    )
                ):
                    guarded = self._classify_password_helper_without_empty_returns(
                        assignment_expr,
                        assignment.start,
                        seen | frozenset({key}),
                    )
                    if guarded is not None:
                        return guarded
                if _is_bare_array_initializer(assignment_expr) and assignment.type_name in {"byte[]", "char[]"}:
                    return SourceClassification("misuse", f"{role} array source is a hardcoded initializer")
                nested = self.classify(
                    assignment_expr,
                    assignment.start,
                    role,
                    seen | frozenset({key}),
                )
                if self._is_static_final(assignment) and nested.label == "misuse":
                    return SourceClassification("misuse", f"static final {role} source resolves to a constant")
                return nested
            if self._variable_filled_by_secure_random(variable, index):
                return SourceClassification("secure", f"{role} variable is filled by SecureRandom")
            if role == "salt" and self._variable_filled_by_varying_salt_source(
                variable,
                index,
            ):
                return SourceClassification(
                    "secure",
                    "salt variable is filled by a varying source; randomness quality is evaluated by M05",
                )
            if any(param.name == variable for param in self._parameters_in_scope(index)):
                caller = self._classify_parameter_from_callers(variable, index, role, seen)
                if caller is not None:
                    return caller
                if role == "salt":
                    return SourceClassification(
                        "unknown",
                        "caller-provided salt is not proven freshly generated",
                    )
                return SourceClassification("secure", f"{role} is supplied by method parameter")

        return SourceClassification("unknown", f"could not resolve {role} source")

    def _classify_password_concatenation(
        self,
        expression: str,
        index: int,
        role: str,
        seen: frozenset[str],
    ) -> SourceClassification | None:
        """Preserve password provenance through deterministic concatenation."""

        if role != "password":
            return None
        parts = _split_top_level_operator(expression, "+")
        if len(parts) < 2:
            return None
        labels = [self.classify(part, index, role, seen) for part in parts]
        if any(label.label == "unknown" for label in labels):
            return SourceClassification(
                "unknown",
                "password concatenation contains an unresolved source",
            )
        if any(label.label == "secure" for label in labels):
            return SourceClassification(
                "secure",
                "password concatenation retains a runtime source",
            )
        return SourceClassification(
            "misuse",
            "password concatenation is composed only from predictable values",
        )

    def _classify_security_concatenation(
        self,
        expression: str,
        index: int,
        role: str,
        seen: frozenset[str],
    ) -> SourceClassification | None:
        """Preserve provenance through deterministic security-value composition."""

        if role not in {"key_material", "iv", "salt", "prng_seed"}:
            return None
        parts = _split_top_level_operator(expression, "+")
        if len(parts) < 2:
            return None
        labels = [self.classify(part, index, role, seen) for part in parts]
        if any(label.label == "unknown" for label in labels):
            return SourceClassification(
                "unknown",
                f"{role} concatenation contains an unresolved source",
            )
        if any(label.label == "secure" for label in labels):
            return SourceClassification(
                "secure",
                f"{role} concatenation retains a generated or runtime source",
            )
        return SourceClassification(
            "misuse",
            f"{role} concatenation is composed only from predictable values",
        )

    def _variable_predictably_overwritten(
        self,
        variable: str,
        start: int,
        end: int,
    ) -> str | None:
        """Detect deterministic writes that replace generated byte material."""

        if start >= end:
            return None
        segment = self.source[start:end]
        escaped = re.escape(variable)
        fill = re.search(
            rf"\b(?:java\.util\.)?Arrays\s*\.\s*fill\s*\(\s*{escaped}\s*,"
            r"\s*(?P<value>[^,)]+)",
            segment,
            flags=re.DOTALL,
        )
        if fill is not None and not re.search(
            r"SecureRandom|generateSeed|nextBytes|runtime|System\s*\.\s*getenv",
            fill.group("value"),
            flags=re.IGNORECASE,
        ):
            return "Arrays.fill uses deterministic material"

        for write in re.finditer(
            rf"\b{escaped}\s*\[[^\]]+\]\s*=\s*(?P<value>[^;]+);",
            segment,
            flags=re.DOTALL,
        ):
            value = write.group("value").strip()
            if re.search(
                r"SecureRandom|generateSeed|nextBytes|System\s*\.\s*getenv"
                r"|System\s*\.\s*console\s*\(|Scanner\s*\(",
                value,
                flags=re.IGNORECASE,
            ):
                continue
            if re.search(r"\b[A-Za-z_$][\w$]*\s*\[", value):
                # Copying another array element needs ordinary source-flow
                # analysis; do not guess that the source is predictable.
                continue
            if re.fullmatch(
                r"(?:\(\s*byte\s*\)\s*)?"
                r"(?:[-+]?\d+|[A-Za-z_$][\w$]*(?:\s*[-+*/%]\s*\d+)*)",
                value,
            ):
                return f"array element assignment uses {value!r}"
        return None

    def _classify_runtime_array_element(
        self,
        expression: str,
        index: int,
        role: str,
        seen: frozenset[str],
    ) -> SourceClassification | None:
        if role != "password":
            return None
        match = re.fullmatch(
            r"(?P<owner>[A-Za-z_]\w*)\s*\[[^\]]+\]",
            expression,
            flags=re.DOTALL,
        )
        if match is None:
            return None
        owner = match.group("owner")
        assignment = self._latest_assignment(owner, index)
        if assignment is not None:
            key = f"password-array:{owner}:{assignment.start}:{index}"
            if key not in seen:
                resolved = self.classify(
                    self._assignment_expression(assignment),
                    assignment.start,
                    role,
                    seen | frozenset({key}),
                )
                if resolved.label != "unknown":
                    return resolved
        parameter = next(
            (
                item
                for item in self._parameters_in_scope(index)
                if item.name == owner
                and item.type_name.replace(" ", "").endswith("String[]")
            ),
            None,
        )
        if parameter is None:
            return None
        caller = self._classify_parameter_from_callers(
            owner,
            index,
            role,
            seen,
        )
        if caller is not None:
            return caller
        return SourceClassification(
            "secure",
            "password is supplied by a runtime String[] method parameter",
        )

    def variable_filled_by_secure_random(self, variable: str, index: int) -> bool:
        return self._variable_filled_by_secure_random(variable, index)

    def latest_assignment(self, variable: str, index: int) -> Assignment | None:
        return self._latest_assignment(variable, index)

    def variable_type(self, variable: str, index: int) -> str | None:
        assignment = self._latest_assignment(variable, index)
        if assignment and assignment.type_name:
            return assignment.type_name
        for parameter in self._parameters_in_scope(index):
            if parameter.name == variable:
                return parameter.type_name
        return None

    def structured_member_sources(
        self,
        expression: str,
        index: int,
    ) -> tuple[tuple[str, int], ...]:
        """Return constructor arguments reaching a simple field/accessor."""

        match = re.fullmatch(
            r"(?P<owner>[A-Za-z_]\w*)\s*\.\s*(?P<member>[A-Za-z_]\w*)"
            r"(?:\s*\(\s*\))?",
            strip_outer_expression(expression),
        )
        if match is None:
            return tuple()
        owner = match.group("owner")
        member = match.group("member")
        seen = frozenset({f"structured:{owner}.{member}:{index}"})
        return tuple(
            value
            for object_expression, object_index in self._object_sources(owner, index)
            for value in self._structured_member_values(
                object_expression,
                member,
                object_index,
                seen,
            )
        )

    def _message_digest_output_source(self, expression: str, index: int) -> str | None:
        expr = strip_outer_expression(expression)
        match_with_arg = re.fullmatch(r"([A-Za-z_]\w*)\s*\.\s*digest\s*\(\s*(.+)\s*\)", expr, flags=re.DOTALL)
        if match_with_arg:
            return _first_argument_text(match_with_arg.group(2))
        match = re.fullmatch(r"([A-Za-z_]\w*)\s*\.\s*digest\s*\(\s*\)", expr)
        if not match:
            return None
        receiver = match.group(1)
        method = method_containing_index(self.methods, index)
        prefix_start = method.body_start if method is not None else 0
        prefix = self.source[prefix_start:index]
        updates = list(
            re.finditer(
                rf"\b{re.escape(receiver)}\s*\.\s*update\s*\(\s*([^;]+?)\s*\)\s*;",
                prefix,
                flags=re.DOTALL,
            )
        )
        if not updates:
            return None
        return _first_argument_text(updates[-1].group(1))

    def _enhanced_for_each_source(self, variable: str, index: int) -> str | None:
        method = method_containing_index(self.methods, index)
        prefix_start = method.body_start if method is not None else 0
        prefix = self.source[prefix_start:index]
        matches = list(
            re.finditer(
                rf"\bfor\s*\(\s*(?:final\s+)?(?:byte|short|int|long|char|String)\s+{re.escape(variable)}\s*:\s*([^)]+)\)",
                prefix,
                flags=re.DOTALL,
            )
        )
        if not matches:
            return None
        return matches[-1].group(1).strip()

    def _variable_copied_sources(self, variable: str, index: int) -> tuple[str, ...]:
        """Resolve common array-copy flows such as salt/hash -> combined salt buffer."""

        method = method_containing_index(self.methods, index)
        prefix_start = method.body_start if method is not None else 0
        prefix = self.source[prefix_start:index]
        escaped = re.escape(variable)
        patterns = (
            rf"\b(?:System|java\.lang\.System)\s*\.\s*arraycopy\s*\(\s*(?P<source>[^,;]+?)\s*,\s*[^,;]+,\s*{escaped}\s*,",
            rf"\b{escaped}\s*=\s*(?:java\.util\.)?Arrays\s*\.\s*copyOf(?:Range)?\s*\(\s*(?P<source>[^,;]+?)\s*,",
            rf"\b{escaped}\s*=\s*(?P<source>[A-Za-z_]\w*)\s*\.\s*clone\s*\(\s*\)",
        )
        matches: list[re.Match[str]] = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, prefix, flags=re.DOTALL))
        if not matches:
            return tuple()
        ordered_sources: list[str] = []
        for match in sorted(matches, key=lambda item: item.start()):
            source = match.group("source").strip()
            if source not in ordered_sources:
                ordered_sources.append(source)
        return tuple(ordered_sources)

    def _variable_filled_from_secure_random_seed(self, variable: str, index: int) -> bool:
        method = method_containing_index(self.methods, index)
        prefix_start = method.body_start if method is not None else 0
        prefix = self.source[prefix_start:index]
        return bool(
            re.search(
                rf"\bfor\s*\(\s*byte\s+[A-Za-z_]\w*\s*:\s*[A-Za-z_]\w*\.generateSeed\s*\([^)]*\)\s*\)"
                rf"\s*\{{[^}}]*\b{re.escape(variable)}\s*=",
                prefix,
                flags=re.DOTALL,
            )
        )

    def _variable_filled_by_cipher_output(
        self,
        variable: str,
        after: int,
        index: int,
        role: str,
        seen: frozenset[str],
    ) -> SourceClassification | None:
        """Trace a byte buffer populated by the four-argument Cipher.update API."""

        if role not in {"key_material", "iv", "salt"}:
            return None
        writes = tuple(
            call
            for call in find_method_calls(self.source, "update")
            if after <= call.start < index
            and len(call.arguments) >= 4
            and simple_variable(call.arguments[3].text) == variable
        )
        if not writes:
            return None
        update = writes[-1]
        receiver = _java_call_receiver(self.source, update.start)
        if receiver is None:
            return None
        initializations = tuple(
            call
            for call in find_method_calls(self.source, "init")
            if call.start < update.start
            and _java_call_receiver(self.source, call.start) == receiver
            and len(call.arguments) >= 3
            and "Cipher.ENCRYPT_MODE" in call.arguments[0].text
        )
        if not initializations:
            return None
        initialization = initializations[-1]
        inputs = (
            self.classify(
                initialization.arguments[1].text,
                initialization.start,
                "key_material",
                seen,
            ),
            self.classify(
                initialization.arguments[2].text,
                initialization.start,
                "iv",
                seen,
            ),
        )
        if any(item.label == "secure" for item in inputs):
            return SourceClassification(
                "secure",
                f"{role} buffer is populated by Cipher.update from a generated or runtime cipher state",
            )
        if all(item.label == "misuse" for item in inputs):
            return SourceClassification(
                "misuse",
                f"{role} buffer is populated by Cipher.update from predictable cipher inputs",
            )
        return SourceClassification(
            "unknown",
            f"{role} buffer is populated by Cipher.update with unresolved cipher inputs",
        )

    def _byte_buffer_array_source(self, expression: str, index: int) -> str | None:
        expr = strip_outer_expression(expression)
        chained_wrap = re.fullmatch(
            r"(?:java\.nio\.)?ByteBuffer\s*\.\s*wrap\s*\(\s*(?P<source>.+?)\s*\)"
            r"(?:\s*\.\s*[A-Za-z_]\w*\s*\([^;]*?\))*"
            r"\s*\.\s*array\s*\(\s*\)",
            expr,
            flags=re.DOTALL,
        )
        if chained_wrap is not None:
            return chained_wrap.group("source").strip()
        match = re.fullmatch(r"([A-Za-z_]\w*)\s*\.\s*array\s*\(\s*\)", expr)
        if not match:
            return None
        receiver = match.group(1)
        assignment = self._latest_assignment(receiver, index)
        if assignment is None:
            return None
        assignment_expr = self._assignment_expression(assignment)
        wrap_match = re.fullmatch(
            r"(?:java\.nio\.)?ByteBuffer\s*\.\s*wrap\s*\(\s*(.+)\s*\)",
            assignment_expr,
            flags=re.DOTALL,
        )
        if wrap_match:
            return _first_argument_text(wrap_match.group(1))
        if re.search(r"(?:java\.nio\.)?ByteBuffer\s*\.\s*allocate\s*\(", assignment_expr):
            method = method_containing_index(self.methods, index)
            prefix_start = method.body_start if method is not None else assignment.end
            prefix = self.source[prefix_start:index]
            put = re.search(
                rf"\b{re.escape(receiver)}\s*\.\s*put(?:Int|Long|Short|Byte|Char|Float|Double)?\s*\(",
                prefix,
            )
            external_array_write = re.search(
                rf"\b(?:nextBytes|read|readFully)\s*\(\s*"
                rf"{re.escape(receiver)}\s*\.\s*array\s*\(\s*\)",
                prefix,
                flags=re.DOTALL,
            )
            other_array_write = re.search(
                rf"\b(?:arraycopy|copyInto|read)\s*\([^;]*"
                rf"\b{re.escape(receiver)}\s*\.\s*array\s*\(\s*\)",
                prefix,
                flags=re.DOTALL,
            )
            predictable_put = put is not None and not re.search(
                rf"\b{re.escape(receiver)}\s*\.\s*put(?:Int|Long|Short|Byte|Char|Float|Double)?\s*\([^;]*(?:SecureRandom|generateSeed|nextBytes)",
                prefix,
                flags=re.DOTALL,
            )
            if external_array_write is None and other_array_write is None and (
                put is None or predictable_put
            ):
                return '"constant-byte-buffer"'
        return None

    def _prng_expression_uses_predictable_variable(self, expression: str, index: int) -> bool:
        expr = strip_outer_expression(expression)
        if not re.search(r"[-+*/%]", expr):
            return False
        for name in set(re.findall(r"\b[A-Za-z_]\w*\b", expr)):
            if name in {"System", "Math", "Long", "Integer"}:
                continue
            assignment = self._latest_assignment(name, index)
            if assignment is None:
                continue
            value = self._assignment_expression(assignment)
            if re.search(r"System\s*\.\s*(?:currentTimeMillis|nanoTime)\s*\(\s*\)", value):
                return True
            if re.fullmatch(r"[-+]?(?:0[xX][0-9a-fA-F]+|\d+)(?:[lL])?", value.strip()):
                return True
        return False

    def _derived_component_source(self, expression: str) -> str | None:
        """Return the source container for common decoded/indexed components."""

        expr = strip_outer_expression(expression)
        match = re.fullmatch(
            r"([A-Za-z_]\w*)\s*\.\s*split\s*\([^)]*\)\s*\[\s*[-+]?\d+\s*\]",
            expr,
            flags=re.DOTALL,
        )
        if match:
            return match.group(1)
        return None

    def expression_is_security_context(self, expression: str) -> bool:
        return bool(_SECURITY_VALUE_NAMES.search(expression))

    def _classify_conditional(
        self,
        expression: str,
        index: int,
        role: str,
        seen: frozenset[str],
    ) -> SourceClassification | None:
        question = _top_level_char(expression, "?")
        if question == -1:
            return None
        colon = _top_level_char(expression, ":", start=question + 1)
        if colon == -1:
            return None
        condition = strip_outer_expression(expression[:question])
        true_branch = expression[question + 1 : colon].strip()
        false_branch = expression[colon + 1 :].strip()
        guarded = re.fullmatch(
            r"(?P<variable>[A-Za-z_]\w*)\s*(?P<operator>!=|==)\s*null"
            r"|null\s*(?P<reverse_operator>!=|==)\s*(?P<reverse_variable>[A-Za-z_]\w*)",
            condition,
        )
        if guarded is not None:
            variable = guarded.group("variable") or guarded.group("reverse_variable")
            operator = guarded.group("operator") or guarded.group("reverse_operator")
            non_null_branch = true_branch if operator == "!=" else false_branch
            null_branch = false_branch if operator == "!=" else true_branch
            if simple_variable(non_null_branch) == variable:
                caller_evidence, non_null = (
                    self._classify_non_null_parameter_from_callers(
                        variable,
                        index,
                        role,
                        seen,
                    )
                )
                labels = [self.classify(null_branch, index, role, seen)]
                if non_null is not None:
                    labels.append(non_null)
                elif not caller_evidence:
                    labels.append(self.classify(non_null_branch, index, role, seen))
                return _combine(labels, f"conditional {role} expression")

        branches = (true_branch, false_branch)
        labels = [self.classify(branch, index, role, seen) for branch in branches if branch != "null"]
        if not labels:
            return SourceClassification("unknown", f"conditional {role} expression has no non-null branch")
        return _combine(labels, f"conditional {role} expression")

    def _classify_non_null_parameter_from_callers(
        self,
        variable: str,
        index: int,
        role: str,
        seen: frozenset[str],
    ) -> tuple[bool, SourceClassification | None]:
        method = method_containing_index(self.methods, index)
        if method is None:
            return False, None
        parameters = [parameter.name for parameter in method.params]
        if variable not in parameters:
            return False, None
        parameter_index = parameters.index(variable)
        caller_evidence = False
        labels: list[SourceClassification] = []
        for call in self._method_calls(method.name):
            if not self._call_may_target_declared_method(call, method):
                continue
            if any(
                candidate.start <= call.start <= candidate.body_start
                for candidate in self.methods
            ):
                continue
            if method.body_start <= call.start <= method.body_end:
                continue
            if len(call.arguments) != len(parameters):
                continue
            caller_evidence = True
            argument = strip_outer_expression(call.arguments[parameter_index].text)
            if argument == "null":
                continue
            labels.append(self.classify(argument, call.start, role, seen))
        return (
            caller_evidence,
            _combine(labels, f"non-null caller-provided {role} argument")
            if labels
            else None,
        )

    def _classify_helper(
        self,
        helper_name: str,
        helper_arity: int,
        index: int,
        role: str,
        seen: frozenset[str],
    ) -> SourceClassification | None:
        methods = [
            method
            for method in self.methods
            if method.name == helper_name and len(method.params) == helper_arity
        ]
        if not methods:
            return None
        labels: list[SourceClassification] = []
        for method in methods:
            helper_key = f"helper:{helper_name}/{len(method.params)}@{method.start}"
            if helper_key in seen:
                labels.append(SourceClassification("unknown", f"recursive helper {helper_name} could not be resolved"))
                continue
            body = self.source[method.body_start + 1 : method.body_end]
            body_heuristic = self._classify_helper_body(
                body,
                helper_name,
                role,
            )
            if role == "password" and body_heuristic is not None:
                labels.append(body_heuristic)
                continue
            returns = tuple(
                (
                    item.group(1).strip(),
                    method.body_start + 1 + item.start(1),
                )
                for item in re.finditer(
                    r"\breturn\s+([^;]+);",
                    body,
                    flags=re.DOTALL,
                )
            )
            non_null_returns = tuple(
                item for item in returns if item[0] != "null"
            )
            for returned, return_index in non_null_returns or returns:
                if _return_follows_terminal_exit(body, return_index - method.body_start - 1):
                    continue
                # Evaluate a return at its lexical execution point. Using the
                # method end makes cleanup in a following ``finally`` block
                # look like it overwrote the value before construction.
                labels.append(
                    self.classify(
                        returned,
                        return_index,
                        role,
                        seen | frozenset({helper_key}),
                    )
                )
            if not labels or all(label.label == "unknown" for label in labels):
                body_heuristic = self._classify_helper_body(body, helper_name, role)
                if body_heuristic is not None:
                    labels = [body_heuristic]
        if not labels:
            return SourceClassification("unknown", f"helper {helper_name} has no resolvable returns")
        return _combine(labels, f"helper {helper_name} {role} source")

    def _classify_indexed_array_source(
        self,
        expression: str,
        index: int,
        role: str,
        seen: frozenset[str],
    ) -> SourceClassification | None:
        match = re.fullmatch(
            r"(?P<owner>[A-Za-z_]\w*)\s*\[\s*(?P<index>[^\]]+)\s*\]",
            expression.strip(),
            flags=re.DOTALL,
        )
        if match is None:
            return None
        owner = match.group("owner")
        assignment = self._latest_assignment(owner, index)
        if assignment is None:
            return None
        assigned = self._assignment_expression(assignment).strip()
        split = re.fullmatch(
            r"(?P<record>[A-Za-z_]\w*)\s*\.\s*split\s*\(\s*"
            r'"(?P<delimiter>(?:\\.|[^"\\])*)"\s*\)',
            assigned,
            flags=re.DOTALL,
        )
        literal_index = re.fullmatch(r"[-+]?\d+", match.group("index").strip())
        if split is not None and literal_index is not None:
            selected = self._classify_delimited_field(
                split.group("record"),
                _java_split_literal_delimiter(split.group("delimiter")),
                int(literal_index.group(0)),
                assignment.start,
                role,
                seen,
            )
            if selected is not None:
                return selected
        if re.fullmatch(
            r"(?:new\s+(?:byte|char|short|int|long)\s*\[\s*\]\s*)+"
            r"\{.*\}",
            assigned,
            flags=re.DOTALL,
        ) or re.fullmatch(r"\{.*\}", assigned, flags=re.DOTALL):
            return SourceClassification(
                "misuse",
                f"{role} is selected from a source-defined fixed array",
            )
        key = f"{role}:indexed-array:{owner}:{assignment.start}"
        if key in seen:
            return SourceClassification(
                "unknown",
                f"recursive indexed {role} source could not be resolved",
            )
        return self.classify(
            assigned,
            assignment.start,
            role,
            seen | frozenset({key}),
        )

    def _classify_delimited_field(
        self,
        expression: str,
        delimiter: str,
        field_index: int,
        index: int,
        role: str,
        seen: frozenset[str],
    ) -> SourceClassification | None:
        """Trace one field of a simple concatenated Java record."""

        expr = strip_outer_expression(expression)
        key = f"{role}:delimited:{field_index}:{expr}:{index}"
        if key in seen:
            return None
        next_seen = seen | frozenset({key})

        variable = simple_variable(expr)
        if variable is not None:
            assignment = self._latest_assignment(variable, index)
            if assignment is not None:
                return self._classify_delimited_field(
                    self._assignment_expression(assignment),
                    delimiter,
                    field_index,
                    assignment.start,
                    role,
                    next_seen,
                )
            method = method_containing_index(self.methods, index)
            if method is not None:
                parameters = [parameter.name for parameter in method.params]
                if variable in parameters:
                    parameter_index = parameters.index(variable)
                    labels: list[SourceClassification] = []
                    for call in self._method_calls(method.name):
                        if not self._call_may_target_declared_method(call, method):
                            continue
                        if method.body_start <= call.start <= method.body_end:
                            continue
                        if len(call.arguments) != len(parameters):
                            continue
                        if not self.call_has_entry_path(call):
                            continue
                        selected = self._classify_delimited_field(
                            call.arguments[parameter_index].text,
                            delimiter,
                            field_index,
                            call.start,
                            role,
                            next_seen,
                        )
                        if selected is not None:
                            labels.append(selected)
                    if labels:
                        return _combine(labels, f"persisted {role} field")

        called = _java_call_details(expr)
        if called is not None:
            helper_name, arguments = called
            labels: list[SourceClassification] = []
            for method in self.methods:
                if method.name != helper_name or len(method.params) != len(arguments):
                    continue
                body = self.source[method.body_start + 1 : method.body_end]
                for returned in re.finditer(
                    r"\breturn\s+([^;]+);",
                    body,
                    flags=re.DOTALL,
                ):
                    selected = self._classify_delimited_field(
                        returned.group(1).strip(),
                        delimiter,
                        field_index,
                        method.body_start + 1 + returned.start(1),
                        role,
                        next_seen,
                    )
                    if selected is not None:
                        labels.append(selected)
            if labels:
                return _combine(labels, f"serialized helper {role} field")

        parts = _split_top_level_operator(expr, "+")
        if len(parts) <= 1:
            return None
        fields: list[list[str]] = [[]]
        for part in parts:
            value = part.strip()
            if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
                chunks = value[1:-1].split(delimiter)
                if len(chunks) > 1:
                    for position, chunk in enumerate(chunks):
                        if chunk:
                            fields[-1].append(f'"{chunk}"')
                        if position < len(chunks) - 1:
                            fields.append([])
                    continue
            fields[-1].append(part)
        selected_index = field_index if field_index >= 0 else len(fields) + field_index
        if not 0 <= selected_index < len(fields) or not fields[selected_index]:
            return None
        labels = [
            self.classify(part, index, role, next_seen)
            for part in fields[selected_index]
        ]
        return _combine(labels, f"serialized {role} field")

    def _file_round_trip_source(
        self,
        expression: str,
        index: int,
    ) -> tuple[str, int] | None:
        """Resolve a local byte-array file loader to its prior matching writer."""

        called = _java_call_details(strip_outer_expression(expression))
        if called is None:
            return None
        reader_name, reader_arguments = called
        if len(reader_arguments) != 1:
            return None
        reader_methods = tuple(
            method
            for method in self.methods
            if method.name == reader_name and len(method.params) == 1
        )
        if not any(
            re.search(r"\bFileInputStream\b", self.source[method.body_start : method.body_end])
            and re.search(r"\b(?:read|readAllBytes)\s*\(", self.source[method.body_start : method.body_end])
            for method in reader_methods
        ):
            return None

        persisted_name = strip_outer_expression(reader_arguments[0])
        matches: list[tuple[str, int]] = []
        for writer in self.methods:
            body = self.source[writer.body_start : writer.body_end]
            if not (
                re.search(r"\b(?:FileOutputStream|DataOutputStream)\b", body)
                and re.search(r"\bwrite\s*\(", body)
            ):
                continue
            for call in self._method_calls(writer.name):
                if call.start >= index or len(call.arguments) != len(writer.params):
                    continue
                arguments = tuple(
                    strip_outer_expression(argument.text)
                    for argument in call.arguments
                )
                path_positions = tuple(
                    position
                    for position, argument in enumerate(arguments)
                    if argument == persisted_name
                )
                if len(path_positions) != 1:
                    continue
                source_positions = tuple(
                    position
                    for position in range(len(arguments))
                    if position != path_positions[0]
                )
                if len(source_positions) == 1:
                    matches.append((arguments[source_positions[0]], call.start))
        return matches[-1] if matches else None

    def _classify_password_helper_without_empty_returns(
        self,
        expression: str,
        index: int,
        seen: frozenset[str],
    ) -> SourceClassification | None:
        call = _simple_call_signature(strip_outer_expression(expression))
        if call is None:
            return None
        helper_name, helper_arity = call
        methods = [
            method
            for method in self.methods
            if method.name == helper_name and len(method.params) == helper_arity
        ]
        labels: list[SourceClassification] = []
        for method in methods:
            helper_key = (
                f"guarded-password-helper:{helper_name}/"
                f"{len(method.params)}@{method.start}"
            )
            if helper_key in seen:
                continue
            body = self.source[method.body_start + 1 : method.body_end]
            body_heuristic = self._classify_helper_body(
                body,
                helper_name,
                "password",
            )
            if body_heuristic is not None:
                labels.append(body_heuristic)
                continue
            for returned in re.findall(r"\breturn\s+([^;]+);", body, flags=re.DOTALL):
                labels.extend(
                    self._classify_nonempty_password_return(
                        returned.strip(),
                        method.body_end,
                        seen | frozenset({helper_key}),
                    )
                )
        if not labels:
            return None
        return _combine(
            labels,
            f"guarded helper {helper_name} password source",
        )

    def _classify_nonempty_password_return(
        self,
        expression: str,
        index: int,
        seen: frozenset[str],
    ) -> list[SourceClassification]:
        expr = strip_outer_expression(expression)
        if _is_empty_password_value(expr) or expr == "null":
            return []
        question = _top_level_char(expr, "?")
        colon = _top_level_char(expr, ":", start=question + 1) if question != -1 else -1
        if question != -1 and colon != -1:
            labels: list[SourceClassification] = []
            for branch in (
                expr[question + 1 : colon].strip(),
                expr[colon + 1 :].strip(),
            ):
                labels.extend(
                    self._classify_nonempty_password_return(
                        branch,
                        index,
                        seen,
                    )
                )
            return labels
        classification = self.classify(expr, index, "password", seen)
        if classification.label != "unknown":
            return [classification]
        variable = simple_variable(self._unwrap(expr))
        if variable is None:
            return [classification]
        reassigned = [
            self.classify(value, value_index, "password", seen)
            for value, value_index in self._textual_reassignments(
                variable,
                index,
            )
            if value != "null" and not _is_empty_password_value(value)
        ]
        return reassigned or [classification]

    def _textual_reassignments(
        self,
        variable: str,
        index: int,
    ) -> tuple[tuple[str, int], ...]:
        method = method_containing_index(self.methods, index)
        start = method.body_start + 1 if method is not None else 0
        end = min(index, method.body_end if method is not None else len(self.source))
        masked = self.masked_source[start:end]
        pattern = re.compile(
            rf"(?<![=!<>])\b{re.escape(variable)}\s*=\s*(?!=)"
            rf"(?P<value>[^;]+);",
            flags=re.DOTALL,
        )
        values: list[tuple[str, int]] = []
        for match in pattern.finditer(masked):
            value_start = start + match.start("value")
            value_end = start + match.end("value")
            values.append(
                (
                    self.source[value_start:value_end].strip(),
                    value_start,
                )
            )
        return tuple(values)

    def _password_variable_rejects_empty(
        self,
        variable: str,
        after_index: int,
        before_index: int,
    ) -> bool:
        if before_index <= after_index:
            return False
        segment = self.masked_source[after_index:before_index]
        escaped = re.escape(variable)
        condition = (
            rf"(?:\b{escaped}\s*==\s*null\b|"
            rf"\b{escaped}\s*\.\s*length\s*(?:==|<=)\s*0\b|"
            rf"\b{escaped}\s*\.\s*length\s*<\s*1\b|"
            rf"\b{escaped}\s*\.\s*isEmpty\s*\(\s*\))"
        )
        return bool(
            re.search(
                rf"\bif\s*\([^)]*{condition}[^)]*\)\s*(?:"
                rf"\{{[^{{}}]{{0,1200}}?(?:throw\b|return\b|"
                rf"System\s*\.\s*exit\s*\()[^{{}}]*\}}|"
                rf"(?:throw\b|return\b|System\s*\.\s*exit\s*\())",
                segment,
                flags=re.DOTALL,
            )
        )

    def _classify_helper_body(self, body: str, helper_name: str, role: str) -> SourceClassification | None:
        scanner_input = bool(
            re.search(
                r"\bnew\s+(?:java\.util\.)?Scanner\s*\(\s*"
                r"(?:System|java\.lang\.System)\s*\.\s*in\s*\)",
                body,
            )
            and re.search(r"\b[A-Za-z_]\w*\s*\.\s*(?:next|nextLine)\s*\(", body)
        )
        console_input = bool(
            re.search(
                r"\b(?:System|java\.lang\.System)\s*\.\s*console\s*\(\s*\)"
                r"\s*\.\s*(?:readLine|readPassword)\s*\(",
                body,
            )
        )
        buffered_input = bool(
            re.search(r"\b(?:BufferedReader|InputStreamReader)\b", body)
            and re.search(r"\b(?:System|java\.lang\.System)\s*\.\s*in\b", body)
            and re.search(r"\b[A-Za-z_]\w*\s*\.\s*readLine\s*\(", body)
        )
        if role == "password" and (
            re.search(
                r"\b(?:System|java\.lang\.System)\s*\.\s*in\s*\.\s*read\s*\(",
                body,
            )
            or scanner_input
            or console_input
            or buffered_input
        ):
            # A byte-reader helper that directly consumes System.in is a
            # runtime source. Exclude helpers with an explicit nonempty
            # literal return so a decoy read cannot bless a fixed password.
            returned_literals = re.findall(
                r"\breturn\s+(?P<literal>\"(?:\\.|[^\"\\])*\")",
                body,
                flags=re.DOTALL,
            )
            if not any(value.strip() not in {'""'} for value in returned_literals):
                return SourceClassification(
                    "secure",
                    f"helper {helper_name} reads password material from runtime input",
                )
        if role in {"key_material", "iv", "salt", "random_value"}:
            if re.search(r"\bMath\s*\.\s*random\s*\(", body):
                if role == "salt":
                    return SourceClassification(
                        "secure",
                        "helper produces a varying salt; randomness quality is evaluated by M05",
                    )
                return SourceClassification(
                    "misuse",
                    f"helper {helper_name} derives {role} from Math.random",
                )
            if re.search(r"\bnew\s+(?:java\.util\.)?Random\s*\(", body) and not re.search(
                r"\bSecureRandom\b",
                body,
            ):
                if role == "salt" and re.search(
                    r"\bnew\s+(?:java\.util\.)?Random\s*\(\s*\)",
                    body,
                ):
                    return SourceClassification(
                        "secure",
                        "helper produces a varying salt; randomness quality is evaluated by M05",
                    )
                return SourceClassification(
                    "misuse",
                    f"helper {helper_name} derives {role} from java.util.Random",
                )
            if role == "salt" and re.search(
                r"\bSystem\s*\.\s*(?:currentTimeMillis|nanoTime)\s*\(",
                body,
            ):
                return SourceClassification(
                    "secure",
                    "helper produces a varying salt; source quality is evaluated by M05",
                )
            if re.search(r"\bSecureRandom\b", body) and re.search(
                r"\b(?:nextBytes|generateSeed|nextInt|nextLong)\s*\(",
                body,
            ):
                return SourceClassification(
                    "secure",
                    f"helper {helper_name} derives {role} from SecureRandom",
                )
            if role == "key_material" and re.search(r"\bKeyGenerator\b", body) and re.search(
                r"\bgenerateKey\s*\(",
                body,
            ):
                return SourceClassification(
                    "secure",
                    f"helper {helper_name} derives key_material from KeyGenerator",
                )
        return None

    def _classify_parameter_from_callers(
        self,
        variable: str,
        index: int,
        role: str,
        seen: frozenset[str],
    ) -> SourceClassification | None:
        method = method_containing_index(self.methods, index)
        if method is None:
            return None
        param_names = [param.name for param in method.params]
        if variable not in param_names:
            return None
        param_index = param_names.index(variable)
        cache_key = (method.start, param_index, role)
        if cache_key in self._caller_parameter_cache:
            return self._caller_parameter_cache[cache_key]
        if cache_key in self._caller_parameter_in_progress:
            return SourceClassification("unknown", f"recursive caller-provided {role} argument could not be resolved")
        self._caller_parameter_in_progress.add(cache_key)
        try:
            labels: list[SourceClassification] = []
            for call in self._method_calls(method.name):
                if not self._call_may_target_declared_method(call, method):
                    continue
                if any(candidate.start <= call.start <= candidate.body_start for candidate in self.methods):
                    continue
                if method.body_start <= call.start <= method.body_end:
                    continue
                if len(call.arguments) != len(param_names):
                    continue
                if not self.call_has_entry_path(call):
                    continue
                if role == "password" and _is_explicit_negative_password_test_call(
                    self.source,
                    method.name,
                    call.start,
                    call.arguments[param_index].text,
                ):
                    continue
                labels.append(self.classify(call.arguments[param_index].text, call.start, role, seen))
            if not labels:
                self._caller_parameter_cache[cache_key] = None
                return None
            result = _combine(labels, f"caller-provided {role} argument")
            self._caller_parameter_cache[cache_key] = result
            return result
        finally:
            self._caller_parameter_in_progress.discard(cache_key)

    def call_has_entry_path(self, call) -> bool:
        """Return whether a Java call is reachable from the program entry.

        Generated programs sometimes retain an explicitly unused self-check
        that calls an active helper with an unresolved parameter.  Such a
        dormant caller must not poison the value supplied by the real ``main``
        path.  Definite findings inside the dormant helper remain visible to
        the detectors; this method only bounds caller-argument propagation.
        """

        if not any(method.name == "main" for method in self.methods):
            return True
        containing = method_containing_index(self.methods, call.start)
        if containing is None or containing.name == "main":
            return True
        return self._method_has_entry_path(containing)

    def position_has_entry_path(self, index: int) -> bool:
        """Return whether a source position belongs to an executable method."""

        if not any(method.name == "main" for method in self.methods):
            return True
        containing = method_containing_index(self.methods, index)
        if containing is None or containing.name == "main":
            return True
        return self._method_has_entry_path(containing)

    def _method_has_entry_path(self, method) -> bool:
        cached = self._method_entry_reachability_cache.get(method.start)
        if cached is not None:
            return cached
        if method.start in self._method_entry_reachability_in_progress:
            return False
        self._method_entry_reachability_in_progress.add(method.start)
        try:
            for call in self._method_calls(method.name):
                if not self._call_may_target_declared_method(call, method):
                    continue
                if any(
                    candidate.start <= call.start <= candidate.body_start
                    for candidate in self.methods
                ):
                    continue
                if method.body_start <= call.start <= method.body_end:
                    continue
                if len(call.arguments) != len(method.params):
                    continue
                caller = method_containing_index(self.methods, call.start)
                if caller is None or caller.name == "main":
                    self._method_entry_reachability_cache[method.start] = True
                    return True
                if self._method_has_entry_path(caller):
                    self._method_entry_reachability_cache[method.start] = True
                    return True
            self._method_entry_reachability_cache[method.start] = False
            return False
        finally:
            self._method_entry_reachability_in_progress.discard(method.start)

    def _classify_structured_member(
        self,
        expression: str,
        index: int,
        role: str,
        seen: frozenset[str],
    ) -> SourceClassification | None:
        match = re.fullmatch(
            r"(?P<owner>[A-Za-z_]\w*)\s*\.\s*(?P<member>[A-Za-z_]\w*)"
            r"(?:\s*\(\s*\))?",
            expression,
        )
        if match is None:
            return None
        owner = match.group("owner")
        member = match.group("member")
        key = f"structured:{owner}.{member}:{index}"
        if key in seen:
            return SourceClassification(
                "unknown",
                f"recursive structured {role} flow could not be resolved",
            )
        next_seen = seen | frozenset({key})
        labels = [
            self.classify(value, value_index, role, next_seen)
            for value, value_index in self.structured_member_sources(
                expression,
                index,
            )
        ]
        if not labels:
            return None
        return _combine(labels, f"structured {role} member {member}")

    def _object_sources(
        self,
        owner: str,
        index: int,
    ) -> tuple[tuple[str, int], ...]:
        assignment = self._latest_assignment(owner, index)
        if assignment is not None:
            return ((self._assignment_expression(assignment), assignment.start),)
        method = method_containing_index(self.methods, index)
        if method is None:
            return tuple()
        parameters = [parameter.name for parameter in method.params]
        if owner not in parameters:
            return tuple()
        parameter_index = parameters.index(owner)
        values: list[tuple[str, int]] = []
        for call in self._method_calls(method.name):
            if not self._call_may_target_declared_method(call, method):
                continue
            if method.body_start <= call.start <= method.body_end:
                continue
            if len(call.arguments) != len(parameters):
                continue
            values.append((call.arguments[parameter_index].text, call.start))
        return tuple(values)

    def _structured_member_values(
        self,
        expression: str,
        member: str,
        index: int,
        seen: frozenset[str],
    ) -> tuple[tuple[str, int], ...]:
        expr = strip_outer_expression(expression)
        key = f"structured-value:{member}:{expr}:{index}"
        if key in seen:
            return tuple()
        next_seen = seen | frozenset({key})

        variable = simple_variable(expr)
        if variable is not None:
            return tuple(
                value
                for source, source_index in self._object_sources(variable, index)
                for value in self._structured_member_values(
                    source,
                    member,
                    source_index,
                    next_seen,
                )
            )

        constructor = _java_constructor_call(expr)
        if constructor is not None:
            class_name, arguments = constructor
            member_index = _java_constructor_member_index(
                self.source,
                class_name,
                member,
            )
            if member_index is not None and member_index < len(arguments):
                return ((arguments[member_index], index),)
            return tuple()

        called = _java_call_details(expr)
        if called is None:
            return tuple()
        helper_name, arguments = called
        values: list[tuple[str, int]] = []
        for method in self.methods:
            if method.name != helper_name or len(method.params) != len(arguments):
                continue
            body = self.source[method.body_start + 1 : method.body_end]
            for returned in re.findall(r"\breturn\s+([^;]+);", body, flags=re.DOTALL):
                values.extend(
                    self._structured_member_values(
                        returned.strip(),
                        member,
                        method.body_end,
                        next_seen,
                    )
                )
        return tuple(values)

    def _method_calls(self, method_name: str):
        if method_name not in self._method_calls_cache:
            self._method_calls_cache[method_name] = find_method_calls(self.source, method_name)
        return self._method_calls_cache[method_name]

    def _call_may_target_declared_method(self, call, method) -> bool:
        prefix = self.source[call.start : call.open_paren]
        match = re.fullmatch(
            rf"\s*(?:(?P<receiver>[A-Za-z_]\w*)\s*\.\s*)?"
            rf"{re.escape(method.name)}\s*",
            prefix,
            flags=re.DOTALL,
        )
        if match is None:
            return True
        receiver = match.group("receiver")
        if receiver is None or receiver in {"this", "super"}:
            return True
        if receiver in self.class_names:
            return True
        receiver_type = self.variable_type(receiver, call.start)
        if receiver_type is None:
            return False
        simple_type = re.sub(r"<.*>", "", receiver_type).replace("[]", "").rsplit(".", 1)[-1]
        return simple_type in self.class_names

    def _latest_assignment(self, variable: str, index: int) -> Assignment | None:
        matches = [
            assignment
            for assignment in self.assignments
            if assignment.name == variable
            and self._assignment_is_code(assignment)
            and self._assignment_visible_at(assignment, index)
        ]
        if not matches:
            return self._latest_regex_assignment(variable, index)
        return max(matches, key=lambda assignment: assignment.start)

    def _assignment_is_code(self, assignment: Assignment) -> bool:
        snippet = self.masked_source[assignment.start : assignment.end]
        return bool(
            re.search(
                rf"\b{re.escape(assignment.name)}(?:\s*\[\s*\])?\s*=",
                snippet,
            )
        )

    def _latest_regex_assignment(self, variable: str, index: int) -> Assignment | None:
        pattern = re.compile(
            rf"(?:(?P<type>[A-Za-z_][\w$.<>?]*(?:\s*\[\])?)\s+)?"
            rf"{re.escape(variable)}(?:\s*\[\s*\])?\s*=\s*(?P<expr>[^;]+);",
            flags=re.DOTALL,
        )
        assignments: list[Assignment] = []
        for match in pattern.finditer(self.masked_source, 0, index):
            if match.group("expr").lstrip().startswith("="):
                continue
            expression_start, expression_end = match.span("expr")
            assignment = Assignment(
                name=variable,
                expression=self.source[expression_start:expression_end].strip(),
                start=match.start(),
                end=match.end(),
                type_name=(match.group("type") or "").strip(),
            )
            if self._assignment_visible_at(assignment, index):
                assignments.append(assignment)
        return max(assignments, key=lambda assignment: assignment.start) if assignments else None

    def _assignment_visible_at(self, assignment: Assignment, index: int) -> bool:
        if assignment.start >= index or assignment.end > index:
            return False
        current_method = method_containing_index(self.methods, index)
        assignment_method = method_containing_index(self.methods, assignment.start)
        if current_method is None:
            return assignment_method is None
        if assignment_method == current_method:
            return True
        if assignment_method is None:
            return True
        if any(
            parameter.name == assignment.name
            for parameter in current_method.params
        ):
            # A bare parameter reference is lexically closer than a
            # same-named ``this.field`` assignment in another method. The AST
            # adapter records both under the field name, so allowing the field
            # write here can turn a caller-provided key into a false cycle.
            return False
        return _is_this_field_write(self.source, assignment)

    def _assignment_expression(self, assignment: Assignment) -> str:
        """Return the full RHS, including lambda bodies with internal semicolons."""

        equals_index = self.source.find("=", assignment.start)
        if equals_index == -1:
            return assignment.expression
        statement_end = _find_top_level_statement_semicolon(self.source, equals_index + 1)
        if statement_end == -1:
            return assignment.expression
        return self.source[equals_index + 1 : statement_end].strip()

    def _parameters_in_scope(self, index: int):
        method = method_containing_index(self.methods, index)
        if method is not None:
            return method.params
        return tuple()

    def _variable_filled_by_secure_random(self, variable: str, index: int, after: int | None = None) -> bool:
        escaped = re.escape(variable)
        method = method_containing_index(self.methods, index)
        prefix_start = after if after is not None else (method.body_start if method is not None else 0)
        prefix = self.source[prefix_start:index]
        direct_patterns = (
            rf"\bnew\s+(?:java\.security\.)?SecureRandom\s*\(\s*\)\s*\.\s*nextBytes\s*\(\s*{escaped}\s*\)",
            rf"\b(?:java\.security\.)?SecureRandom\s*\.\s*getInstance(?:Strong)?\s*\([^)]*\)\s*\.\s*nextBytes\s*\(\s*{escaped}\s*\)",
        )
        if any(re.search(pattern, prefix, flags=re.DOTALL) for pattern in direct_patterns):
            return True
        helper_pattern = re.compile(
            rf"\b(?P<helper>[A-Za-z_]\w*)\s*\(\s*\)\s*\.\s*"
            rf"nextBytes\s*\(\s*{escaped}\s*\)",
            flags=re.DOTALL,
        )
        if any(
            self._helper_returns_secure_random(match.group("helper"))
            for match in helper_pattern.finditer(prefix)
        ):
            return True
        receiver_pattern = re.compile(
            rf"\b(?P<receiver>[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*)"
            rf"\s*\.\s*nextBytes\s*\(\s*{escaped}\s*\)",
            flags=re.DOTALL,
        )
        return any(
            self._receiver_is_secure_random(match.group("receiver"), prefix_start + match.start())
            for match in receiver_pattern.finditer(prefix)
        )

    def _variable_filled_by_varying_salt_source(
        self,
        variable: str,
        index: int,
        *,
        after: int | None = None,
    ) -> bool:
        escaped = re.escape(variable)
        method = method_containing_index(self.methods, index)
        prefix_start = after if after is not None else (
            method.body_start if method is not None else 0
        )
        prefix = self.source[prefix_start:index]
        if re.search(
            rf"\bnew\s+(?:java\.util\.)?Random\s*\(\s*\)\s*\.\s*"
            rf"nextBytes\s*\(\s*{escaped}\s*\)",
            prefix,
            flags=re.DOTALL,
        ):
            return True
        for match in re.finditer(
            rf"\b(?P<receiver>[A-Za-z_]\w*)\s*\.\s*nextBytes\s*\(\s*"
            rf"{escaped}\s*\)",
            prefix,
            flags=re.DOTALL,
        ):
            receiver = re.escape(match.group("receiver"))
            before_call = prefix[: match.start()]
            if re.search(
                rf"\b(?:java\.util\.)?Random\s+{receiver}\b[^;]*="
                rf"\s*new\s+(?:java\.util\.)?Random\s*\(\s*\)",
                before_call,
                flags=re.DOTALL,
            ):
                return True
        return False

    def _helper_returns_secure_random(self, helper_name: str) -> bool:
        for method in self.methods:
            if method.name != helper_name or method.params:
                continue
            body = self.source[method.body_start + 1 : method.body_end]
            if re.search(
                r"\breturn\s+(?:new\s+(?:java\.security\.)?SecureRandom"
                r"\s*\(\s*\)|(?:java\.security\.)?SecureRandom\s*\.\s*"
                r"getInstance(?:Strong)?\s*\([^;]*\))\s*;",
                body,
                flags=re.DOTALL,
            ):
                return True
        return False

    def _receiver_is_secure_random(self, receiver: str, index: int) -> bool:
        receiver = re.sub(r"\s+", "", receiver)
        if "." in receiver:
            member = re.escape(receiver.rsplit(".", 1)[-1])
            secure_initializer = (
                r"(?:new\s+(?:java\.security\.)?SecureRandom\s*\(\s*\)"
                r"|(?:java\.security\.)?SecureRandom\s*\.\s*"
                r"getInstance(?:Strong)?\s*\([^;]*\))"
            )
            if re.search(
                rf"\b(?:java\.security\.)?SecureRandom\s+{member}\s*=\s*"
                rf"{secure_initializer}\s*;",
                self.source,
                flags=re.DOTALL,
            ):
                return True
        assignment = self._latest_assignment(receiver, index)
        if assignment is not None:
            return bool(
                re.search(
                    r"\bnew\s+(?:java\.security\.)?SecureRandom\s*\(\s*\)"
                    r"|\b(?:java\.security\.)?SecureRandom\s*\.\s*getInstance(?:Strong)?\s*\(",
                    self._assignment_expression(assignment),
                )
            )
        return any(
            parameter.name == receiver
            and parameter.type_name.rsplit(".", 1)[-1] == "SecureRandom"
            for parameter in self._parameters_in_scope(index)
        )

    def _unwrap(self, expression: str) -> str:
        expr = expression.strip()
        constructor = _java_constructor_call(expr)
        if (
            constructor is not None
            and constructor[0] == "String"
            and constructor[1]
        ):
            return constructor[1][0]
        wrappers = (
            rf"(.+?)\s*\.\s*(?:trim|strip)\s*\(\s*\)",
            rf"(.+?)\s*\.\s*(?:toLowerCase|toUpperCase)\s*\(\s*\)",
            rf"(.+?)\s*\.\s*(?:replace|replaceAll|replaceFirst)\s*\(.*\)",
            rf"(.+?)\s*\.\s*(?:substring|subSequence)\s*\(.*\)",
            rf"[A-Za-z_][\w.]*\s*\.\s*encode\s*\(\s*(.+?)\s*\)\s*\.\s*array\s*\(\s*\)",
            rf"[A-Za-z_][\w.]*\s*\.\s*encode\s*\(\s*(.+?)\s*\)",
            rf"({_STRING_LITERAL}|[A-Za-z_]\w*|.+?)\s*\.\s*getBytes\s*\([^)]*\)",
            rf"({_STRING_LITERAL}|[A-Za-z_]\w*|.+?)\s*\.\s*toCharArray\s*\(\s*\)",
            rf"([A-Za-z_]\w*|.+?)\s*\.\s*clone\s*\(\s*\)",
            rf"String\s*\.\s*valueOf\s*\(\s*(.+)\s*\)",
            rf"java\.util\.Base64\s*\.\s*getDecoder\s*\(\s*\)\s*\.\s*decode\s*\(\s*(.+)\s*\)",
            rf"Base64\s*\.\s*getDecoder\s*\(\s*\)\s*\.\s*decode\s*\(\s*(.+)\s*\)",
            rf"java\.util\.Base64\s*\.\s*getEncoder\s*\(\s*\)\s*\.\s*encode\s*\(\s*(.+)\s*\)",
            rf"Base64\s*\.\s*getEncoder\s*\(\s*\)\s*\.\s*encode\s*\(\s*(.+)\s*\)",
            rf"java\.util\.Base64\s*\.\s*getEncoder\s*\(\s*\)\s*\.\s*encodeToString\s*\(\s*(.+)\s*\)",
            rf"Base64\s*\.\s*getEncoder\s*\(\s*\)\s*\.\s*encodeToString\s*\(\s*(.+)\s*\)",
            rf"java\.util\.Base64\s*\.\s*getUrlEncoder\s*\(\s*\)\s*"
            rf"\.\s*withoutPadding\s*\(\s*\)\s*\.\s*encodeToString\s*\(\s*(.+)\s*\)",
            rf"Base64\s*\.\s*getUrlEncoder\s*\(\s*\)\s*"
            rf"\.\s*withoutPadding\s*\(\s*\)\s*\.\s*encodeToString\s*\(\s*(.+)\s*\)",
            rf"java\.util\.Base64\s*\.\s*getUrlEncoder\s*\(\s*\)\s*\.\s*encodeToString\s*\(\s*(.+)\s*\)",
            rf"Base64\s*\.\s*getUrlEncoder\s*\(\s*\)\s*\.\s*encodeToString\s*\(\s*(.+)\s*\)",
            rf"HexFormat\s*\.\s*of\s*\(\s*\)\s*\.\s*parseHex\s*\(\s*(.+)\s*\)",
            rf"DatatypeConverter\s*\.\s*parseHexBinary\s*\(\s*(.+)\s*\)",
            rf"(?:java\.util\.)?Arrays\s*\.\s*hashCode\s*\(\s*(.+?)\s*\)",
            rf"(?:java\.util\.)?Arrays\s*\.\s*copyOfRange\s*\(\s*(.+?)\s*,\s*[^,]+,\s*[^,]+\s*\)",
            rf"(?:java\.util\.)?Arrays\s*\.\s*copyOf\s*\(\s*(.+?)\s*,\s*[^,]+\s*\)",
            rf"([A-Za-z_]\w*|.+?)\s*\.\s*getEncoded\s*\(\s*\)",
        )
        for pattern in wrappers:
            match = re.fullmatch(pattern, expr, flags=re.DOTALL)
            if match:
                return match.group(1).strip()
        return expr

    def _is_predictable_salt_source(self, expression: str) -> bool:
        """Recognize deterministic values that cannot provide salt freshness."""

        expr = strip_outer_expression(expression)
        if re.search(
            r"\b(?:java\.lang\.)?System\s*\.\s*getProperty\s*\(\s*"
            r'"(?:user\.name|java\.home)"\s*\)',
            expr,
        ):
            return True

        base = re.match(
            r"(?:this\s*\.\s*)?(?P<name>[A-Za-z_$][\w$]*)"
            r"(?:\s*\.\s*(?:getBytes|toCharArray|substring|subSequence)\s*\(.*)?$",
            expr,
            flags=re.DOTALL,
        )
        return bool(base and _PASSWORD_VALUE_NAMES.search(base.group("name")))

    def _base64_codec_source(self, expression: str, index: int) -> str | None:
        match = re.fullmatch(
            r"(?P<receiver>[A-Za-z_]\w*)\s*\.\s*"
            r"(?:decode|encode|encodeToString)\s*\((?P<arguments>.*)\)",
            expression.strip(),
            flags=re.DOTALL,
        )
        if match is None:
            return None
        arguments = _split_top_level_arguments(match.group("arguments"))
        if len(arguments) != 1:
            return None
        receiver = match.group("receiver")
        receiver_type = self.variable_type(receiver, index) or ""
        assignment = self._latest_assignment(receiver, index)
        assignment_expression = (
            self._assignment_expression(assignment) if assignment is not None else ""
        )
        if not (
            re.search(r"(?:^|\.)Base64\s*\.\s*(?:Encoder|Decoder)$", receiver_type)
            or re.search(
                r"(?:^|\.)Base64\s*\.\s*get(?:Url)?(?:Encoder|Decoder)\s*\(",
                assignment_expression,
            )
        ):
            return None
        return arguments[0]

    def _constructor_sensitive_argument(self, expression: str, role: str) -> str | None:
        expr = expression.strip()
        if role == "key_material":
            match = re.fullmatch(
                r"new\s+(?:[\w.]+\.)?(?:SecretKeySpec|DESKeySpec|DESedeKeySpec|"
                r"PKCS8EncodedKeySpec|X509EncodedKeySpec)\s*\((.+)\)",
                expr,
                flags=re.DOTALL,
            )
            if match:
                return _first_argument_text(match.group(1))
        if role == "iv":
            match = re.fullmatch(r"new\s+(?:[\w.]+\.)?IvParameterSpec\s*\((.+)\)", expr, flags=re.DOTALL)
            if match:
                return _first_argument_text(match.group(1))
        if role == "password":
            for class_pattern, argument_index in (
                (r"(?:[\w.]+\.)?(?:KeyStore\.)?PasswordProtection", 0),
                (r"(?:[\w.]+\.)?PasswordAuthentication", 1),
            ):
                match = re.fullmatch(rf"new\s+{class_pattern}\s*\((.+)\)", expr, flags=re.DOTALL)
                if match:
                    arguments = _split_top_level_arguments(match.group(1))
                    if len(arguments) > argument_index:
                        return arguments[argument_index]
        return None

    def _is_misuse_constant(self, expression: str, role: str) -> bool:
        expr = strip_outer_expression(expression)
        if re.fullmatch(_STRING_LITERAL, expr, flags=re.DOTALL):
            return True
        if role == "key_material" and re.search(r"\bKeyGenerator\s*\.\s*getInstance\s*\(", expr) and re.search(
            r"\.\s*generateKey\s*\(", expr
        ):
            return False
        if role in {"key_material", "iv", "salt"} and re.search(
            r"\b(?:new\s+(?:java\.security\.)?SecureRandom\s*\(\s*\)"
            r"|(?:java\.security\.)?SecureRandom\s*\.\s*getInstance(?:Strong)?\s*\([^)]*\))\s*\.\s*nextBytes\s*\(",
            expr,
        ):
            return False
        if role in {"key_material", "iv", "salt"} and re.search(
            r"\bnew\s+(?:java\.security\.)?SecureRandom\s*\(\s*[^)]*"
            r"(?:\"(?:\\.|[^\"\\])*\"|new\s+(?:byte|int|long)\s*\[\s*\]\s*\{|System\s*\.\s*(?:currentTimeMillis|nanoTime)\s*\()",
            expr,
            flags=re.DOTALL,
        ):
            return True
        if role == "prng_seed" and re.fullmatch(r"[-+]?\d+(?:[lL])?", expr):
            return True
        if role == "prng_seed" and re.fullmatch(r"[-+]?0[xX][0-9a-fA-F]+(?:[lL])?", expr):
            return True
        if role == "prng_seed" and re.fullmatch(r"[-+*/%()0-9lL\s]+", expr):
            return True
        if role == "prng_seed" and re.search(_STRING_LITERAL, expr, flags=re.DOTALL):
            return True
        if role == "prng_seed" and re.search(
            r"\b(?:getAndIncrement|incrementAndGet|getAndDecrement|decrementAndGet)\s*\(",
            expr,
        ):
            return True
        if role == "prng_seed" and re.search(
            r"(?:\b[A-Za-z_]\w*\s*(?:\+\+|--)|(?:\+\+|--)\s*[A-Za-z_]\w*\b)",
            expr,
        ):
            return True
        if role == "prng_seed" and re.search(
            r"\b(?:counter|count|index|idx|sequence|seq|iteration|iter)\b",
            expr,
            flags=re.IGNORECASE,
        ) and re.search(r"[-+*/%]", expr):
            return True
        if role == "prng_seed" and re.search(r"\b[A-Za-z_]\w*\s*\.\s*(?:nextInt|nextLong)\s*\(", expr):
            return True
        if re.search(r"new\s+(?:byte|char)\s*\[\s*\]\s*\{[^}]*\}", expr, flags=re.DOTALL):
            return True
        if role == "prng_seed" and re.search(
            r"new\s+(?:byte|short|int|long|char)\s*\[\s*\]\s*\{[^}]*\}",
            expr,
            flags=re.DOTALL,
        ):
            return True
        if re.search(r"\{[^}]*\}", expr, flags=re.DOTALL) and re.search(r"\b(?:byte|char)\b", expr):
            return True
        if role == "prng_seed" and re.search(r"\{[^}]*\}", expr, flags=re.DOTALL):
            return True
        if re.search(r"new\s+(?:byte|char)\s*\[\s*[^]]+\s*\]", expr):
            return True
        if role == "prng_seed" and re.search(r"new\s+(?:byte|short|int|long|char)\s*\[\s*[^]]+\s*\]", expr):
            return True
        if role == "prng_seed" and re.search(r"System\s*\.\s*(?:currentTimeMillis|nanoTime)\s*\(\s*\)", expr):
            return True
        if role in {"key_material", "iv", "random_value"} and re.search(
            r"(?:\bnew\s+(?:java\.util\.)?Random\s*\(|"
            r"\bMath\s*\.\s*random\s*\(|"
            r"\bSystem\s*\.\s*(?:currentTimeMillis|nanoTime)\s*\()",
            expr,
        ):
            return True
        if role == "random_value" and re.fullmatch(r"[-+]?\d+(?:[lL])?", expr):
            return True
        if re.search(r"\b(?:Base64|DatatypeConverter|HexFormat)\b", expr) and re.search(_STRING_LITERAL, expr):
            return True
        if re.search(_STRING_LITERAL, expr) and re.search(r"\b(?:getBytes|toCharArray|copyOf|String|valueOf)\b", expr):
            return True
        if self._contains_hardcoded_default(expr):
            return True
        return False

    def _is_secure_source(self, expression: str, index: int, role: str) -> bool:
        expr = strip_outer_expression(expression)
        if role in {"key_material", "iv", "prng_seed"} and (
            re.search(r"\b[A-Za-z_]\w*\s*\[\s*[-+]?\d+\s*\]", expr)
            or re.search(r"\b(?:System|java\.lang\.System)\s*\.\s*getenv\s*\(", expr)
            or re.search(r"\b(?:System|java\.lang\.System)\s*\.\s*getProperty\s*\(\s*[^,)]*\)", expr)
            or re.search(r"\b[A-Za-z_]\w*\s*\.\s*(?:read|readLine|next|nextLine|get)\s*\(", expr)
        ):
            return True
        if role == "key_material":
            key_store_get_key = re.fullmatch(
                r"(?P<receiver>[A-Za-z_]\w*)\s*\.\s*getKey\s*\(.*\)",
                expr,
                flags=re.DOTALL,
            )
            key_store_source = bool(
                key_store_get_key
                and (self.variable_type(key_store_get_key.group("receiver"), index) or "")
                .replace(" ", "")
                .rsplit(".", 1)[-1]
                == "KeyStore"
            )
            generated_key_call = re.fullmatch(
                r"(?P<receiver>[A-Za-z_]\w*)\s*\.\s*generateKey\s*\(\s*\)",
                expr,
            )
            generated_by_key_generator = bool(
                generated_key_call
                and (
                    (self.variable_type(generated_key_call.group("receiver"), index) or "")
                    .replace(" ", "")
                    .rsplit(".", 1)[-1]
                    == "KeyGenerator"
                    or bool(
                        (
                            receiver_assignment := self._latest_assignment(
                                generated_key_call.group("receiver"),
                                index,
                            )
                        )
                        and re.search(
                            r"\bKeyGenerator\s*\.\s*getInstance\s*\(",
                            self._assignment_expression(receiver_assignment),
                        )
                    )
                )
            )
            key_pair_component = re.fullmatch(
                r"(?P<receiver>[A-Za-z_]\w*)\s*\.\s*get(?:Private|Public)\s*\(\s*\)",
                expr,
            )
            generated_by_key_pair = bool(
                key_pair_component
                and (
                    (self.variable_type(key_pair_component.group("receiver"), index) or "")
                    .replace(" ", "")
                    .rsplit(".", 1)[-1]
                    == "KeyPair"
                )
            )
            return bool(
                re.search(r"\bKeyGenerator\s*\.\s*getInstance\s*\([^)]*\)\s*\.\s*generateKey\s*\(\s*\)", expr)
                or generated_by_key_generator
                or generated_by_key_pair
                or key_store_source
                or re.search(r"\b[A-Za-z_]\w*\s*\.\s*generateSecret\s*\(", expr)
                or re.search(r"\b(?:new\s+(?:java\.security\.)?SecureRandom\s*\(\s*\)|[A-Za-z_]\w*)\s*\.\s*generateSeed\s*\(", expr)
                or re.search(
                    r"\b(?:new\s+(?:java\.security\.)?SecureRandom\s*\(\s*\)"
                    r"|(?:java\.security\.)?SecureRandom\s*\.\s*getInstance(?:Strong)?\s*\([^)]*\))\s*\.\s*nextBytes\s*\(",
                    expr,
                )
            )
        if role in {"iv", "salt"}:
            if role == "salt" and (
                re.search(
                    r"\bnew\s+(?:java\.util\.)?Random\s*\(\s*\)",
                    expr,
                )
                or re.search(
                    r"\bSystem\s*\.\s*(?:currentTimeMillis|nanoTime)\s*\(",
                    expr,
                )
            ):
                return True
            return bool(
                re.search(r"\b[A-Za-z_]\w*\s*\.\s*getIV\s*\(\s*\)", expr)
                or (
                    role in {"iv", "salt"}
                    and re.search(
                        r"\b(?:java\.util\.)?UUID\s*\.\s*randomUUID\s*\(\s*\)",
                        expr,
                    )
                )
                or re.search(r"\bgetParameterSpec\s*\(\s*IvParameterSpec\s*\.\s*class\s*\)\s*\.\s*getIV\s*\(\s*\)", expr)
                or re.search(r"\b[A-Za-z_]\w*\s*\.\s*generateKey\s*\(\s*\)", expr)
                or re.search(r"\b[A-Za-z_]\w*\s*\.\s*generateKey\s*\(\s*\)\s*\.\s*getEncoded\s*\(\s*\)", expr)
                or re.search(r"\b(?:new\s+(?:java\.security\.)?SecureRandom\s*\(\s*\)|[A-Za-z_]\w*)\s*\.\s*generateSeed\s*\(", expr)
                or re.search(r"\b(?:java\.security\.)?SecureRandom\s*\.\s*getInstanceStrong\s*\(\s*\)\s*\.\s*generateSeed\s*\(", expr)
                or re.search(
                    r"\b(?:new\s+(?:java\.security\.)?SecureRandom\s*\(\s*\)"
                    r"|(?:java\.security\.)?SecureRandom\s*\.\s*getInstance(?:Strong)?\s*\([^)]*\))\s*\.\s*nextBytes\s*\(",
                    expr,
                )
            )
        if role == "random_value":
            if re.search(
                r"\b(?:new\s+(?:java\.security\.)?SecureRandom\s*\(\s*\)|"
                r"(?:java\.security\.)?SecureRandom\s*\.\s*"
                r"getInstance(?:Strong)?\s*\([^)]*\))"
                r"\s*\.\s*(?:generateSeed|nextInt|nextLong)\s*\(",
                expr,
            ):
                return True
            receiver = re.match(
                r"\s*(?P<name>[A-Za-z_]\w*)\s*\.\s*"
                r"(?:generateSeed|nextInt|nextLong)\s*\(",
                expr,
            )
            if receiver is None:
                return False
            name = re.escape(receiver.group("name"))
            return bool(
                re.search(
                    rf"\b(?:java\.security\.)?SecureRandom\s+{name}\b|"
                    rf"\b{name}\s*=\s*(?:new\s+)?(?:java\.security\.)?"
                    rf"SecureRandom\b",
                    self.source,
                )
            )
        if role == "prng_seed":
            return bool(
                re.fullmatch(r"new\s+(?:java\.security\.)?SecureRandom\s*\(\s*\)", expr)
                or re.search(r"SecureRandom\s*\.\s*getInstanceStrong\s*\(\s*\)", expr)
                or re.search(r"\bnew\s+(?:java\.security\.)?SecureRandom\s*\(\s*\)\s*\.\s*generateSeed\s*\(", expr)
                or re.search(r"\bnew\s+(?:java\.security\.)?SecureRandom\s*\(\s*\)\s*\.\s*nextInt\s*\(", expr)
                or re.search(r"\bnew\s+(?:java\.security\.)?SecureRandom\s*\(\s*\)\s*\.\s*nextLong\s*\(", expr)
                or re.search(r"\b(?:java\.security\.)?SecureRandom\s*\.\s*getInstanceStrong\s*\(\s*\)\s*\.\s*nextInt\s*\(", expr)
                or re.search(r"\b(?:java\.security\.)?SecureRandom\s*\.\s*getInstanceStrong\s*\(\s*\)\s*\.\s*nextLong\s*\(", expr)
                or re.search(r"\b[A-Za-z_]\w*\s*\.\s*generateSeed\s*\(", expr)
            )
        if role == "password":
            return bool(
                re.search(r"\b(?:System|java\.lang\.System)\s*\.\s*getenv\s*\(", expr)
                or re.search(r"\b(?:System|java\.lang\.System)\s*\.\s*getProperty\s*\(\s*[^,)]*\)", expr)
                or re.search(r"\b[A-Za-z_]\w*\s*\[\s*[-+]?\d+\s*\]", expr)
                or (
                    re.search(r"\b(?:System|java\.lang\.System)\s*\.\s*in\b", expr)
                    and re.search(
                        r"\b(?:read|readAllBytes|readNBytes|readLine)\s*\(",
                        expr,
                    )
                )
                or re.search(r"\b(?:System\s*\.\s*)?console\s*\(\s*\)\s*\.\s*readPassword\s*\(", expr)
                or re.search(r"\b[A-Za-z_]\w*\s*\.\s*readPassword\s*\(", expr)
                or re.search(r"\b[A-Za-z_]\w*\s*\.\s*(?:next|nextLine|getPassword)\s*\(", expr)
                or re.search(r"\bnew\s+(?:java\.util\.)?Scanner\s*\([^)]*\)\s*\.\s*(?:next|nextLine)\s*\(", expr)
                or re.search(r"\bFiles\s*\.\s*readString\s*\(", expr)
                or re.search(r"\b(?:System\s*\.\s*)?console\s*\(\s*\)\s*\.\s*readLine\s*\(", expr)
                or re.search(r"\b[A-Za-z_]\w*\s*\.\s*readLine\s*\(", expr)
            )
        return False

    def _contains_hardcoded_default(self, expression: str) -> bool:
        return bool(
            re.search(r"System\s*\.\s*getProperty\s*\([^,]+,\s*" + _STRING_LITERAL, expression, flags=re.DOTALL)
            or re.search(r"\.getOrDefault\s*\([^,]+,\s*" + _STRING_LITERAL, expression, flags=re.DOTALL)
            or re.search(r"\.orElse\s*\(\s*" + _STRING_LITERAL, expression, flags=re.DOTALL)
            or re.search(r"requireNonNullElse\s*\([^,]+,\s*" + _STRING_LITERAL, expression, flags=re.DOTALL)
        )

    def _is_static_final(self, assignment: Assignment) -> bool:
        return "static" in assignment.modifiers.split() and "final" in assignment.modifiers.split()


def _is_this_field_write(source: str, assignment: Assignment) -> bool:
    snippet = source[assignment.start : assignment.end]
    return bool(re.match(rf"\s*this\s*\.\s*{re.escape(assignment.name)}\s*=", snippet, flags=re.DOTALL))


def _find_top_level_statement_semicolon(source: str, start: int) -> int:
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    for index in range(start, len(source)):
        current = source[index]
        if in_string or in_char:
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif in_string and current == '"':
                in_string = False
            elif in_char and current == "'":
                in_char = False
            continue
        if current == '"':
            in_string = True
            continue
        if current == "'":
            in_char = True
            continue
        if current in "([{":
            depth += 1
            continue
        if current in ")]}":
            depth = max(0, depth - 1)
            continue
        if current == ";" and depth == 0:
            return index
    return -1


def combine_program_label(misuse_id: str, findings: tuple, no_sink_reason: str):
    from crypto_steering.validation.java.models import DetectionResult

    if not findings:
        return DetectionResult(misuse_id, "unknown", tuple(), no_sink_reason)
    labels = {finding.label for finding in findings}
    if "mixed" in labels or ("secure" in labels and "misuse" in labels):
        return DetectionResult(misuse_id, "mixed", findings, "both secure and misuse sinks were found")
    if "misuse" in labels:
        return DetectionResult(misuse_id, "misuse", findings, "at least one sink uses a hardcoded or predictable source")
    if labels == {"secure"}:
        return DetectionResult(misuse_id, "secure", findings, "all sinks use non-constant sources")
    return DetectionResult(misuse_id, "unknown", findings, "sinks found, but sources were unresolved")


def security_context_present(source: str) -> bool:
    return bool(
        re.search(
            r"\b(?:Cipher|Mac|MessageDigest|Signature|SecretKeySpec|IvParameterSpec|PBEKeySpec|"
            r"PBEParameterSpec|KeyGenerator|KeyPairGenerator|SecureRandom|KeyStore|PasswordProtection)\b",
            source,
        )
    )


def _combine(classifications: list[SourceClassification], reason: str) -> SourceClassification:
    labels = {item.label for item in classifications}
    if "mixed" in labels or ("secure" in labels and "misuse" in labels):
        return SourceClassification("mixed", f"{reason} contains both secure and misuse sources")
    if "misuse" in labels:
        return SourceClassification("misuse", f"{reason} resolves to a hardcoded or predictable source")
    if labels == {"secure"}:
        return SourceClassification("secure", f"{reason} resolves to secure sources")
    return SourceClassification("unknown", f"{reason} could not be fully resolved")


def _is_explicit_negative_password_test_call(
    source: str,
    method_name: str,
    call_start: int,
    argument: str,
) -> bool:
    """Recognize a literal wrong-password check that is not credential use."""

    if re.search(r"verify|check|authenticate|match", method_name, re.IGNORECASE) is None:
        return False
    if re.fullmatch(
        _STRING_LITERAL + r"(?:\s*\.\s*toCharArray\s*\(\s*\))?",
        argument.strip(),
    ) is None:
        return False
    prefix = source[max(0, call_start - 160) : call_start]
    boundary = max(
        prefix.rfind(";"),
        prefix.rfind("{"),
        prefix.rfind("}"),
        prefix.rfind("\n"),
    )
    statement_prefix = prefix[boundary + 1 :]
    return bool(
        re.search(
            r"\b\w*(?:wrong|incorrect|invalid|negative|bad|failure|tamper)\w*\b",
            statement_prefix,
            flags=re.IGNORECASE,
        )
    )


def _is_empty_password_value(expression: str) -> bool:
    expr = strip_outer_expression(expression)
    return bool(
        re.fullmatch(r'""(?:\s*\.\s*toCharArray\s*\(\s*\))?', expr)
        or re.fullmatch(r"new\s+char\s*\[\s*0\s*\]", expr)
        or re.fullmatch(r"new\s+char\s*\[\s*\]\s*\{\s*\}", expr)
    )


def _top_level_char(expression: str, char: str, start: int = 0) -> int:
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    for index in range(start, len(expression)):
        current = expression[index]
        if in_string or in_char:
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif in_string and current == '"':
                in_string = False
            elif in_char and current == "'":
                in_char = False
            continue
        if current == '"':
            in_string = True
        elif current == "'":
            in_char = True
        elif current in "([{":
            depth += 1
        elif current in ")]}":
            depth -= 1
        elif current == char and depth == 0:
            return index
    return -1


def _first_argument_text(arguments_text: str) -> str:
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(arguments_text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            return arguments_text[:index].strip()
    return arguments_text.strip()


def _assignment_expression_value(expression: str) -> str | None:
    match = re.fullmatch(r"(?:[A-Za-z_]\w*|this\s*\.\s*[A-Za-z_]\w*)\s*=\s*(?!=)(.+)", expression.strip(), flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _is_bare_array_initializer(expression: str) -> bool:
    return bool(re.fullmatch(r"\{[^}]*\}", expression.strip(), flags=re.DOTALL))


def _simple_call_signature(expression: str) -> tuple[str, int] | None:
    match = re.fullmatch(
        r"(?:(?:[A-Za-z_]\w*)\s*\.\s*)?([A-Za-z_]\w*)\s*\((.*)\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if not match:
        match = re.fullmatch(
            r"new\s+(?:[A-Za-z_]\w*\s*\.\s*)*[A-Za-z_]\w*\s*\([^()]*\)"
            r"\s*\.\s*([A-Za-z_]\w*)\s*\((.*)\)",
            expression.strip(),
            flags=re.DOTALL,
        )
    if not match:
        return None
    arguments_text = match.group(2).strip()
    if not arguments_text:
        return (match.group(1), 0)
    return (match.group(1), len(_split_top_level_arguments(arguments_text)))


def _java_call_details(expression: str) -> tuple[str, tuple[str, ...]] | None:
    match = re.fullmatch(
        r"(?:(?:[A-Za-z_]\w*)\s*\.\s*)?([A-Za-z_]\w*)\s*\((.*)\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    arguments_text = match.group(2).strip()
    arguments = (
        _split_top_level_arguments(arguments_text)
        if arguments_text
        else tuple()
    )
    return match.group(1), arguments


def _java_constructor_call(
    expression: str,
) -> tuple[str, tuple[str, ...]] | None:
    expr = expression.strip()
    match = re.match(
        r"new\s+(?:[A-Za-z_]\w*\s*\.\s*)*(?P<name>[A-Za-z_]\w*)\s*\(",
        expr,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    opening = expr.find("(", match.start())
    closing = find_matching_delimiter(expr, opening)
    if closing != len(expr) - 1:
        return None
    arguments_text = expr[opening + 1 : closing].strip()
    arguments = (
        _split_top_level_arguments(arguments_text)
        if arguments_text
        else tuple()
    )
    return match.group("name"), arguments


def _return_follows_terminal_exit(body: str, return_offset: int) -> bool:
    """Recognize a return used only to satisfy Java after System.exit."""

    prefix = body[:return_offset]
    return bool(
        re.search(
            r"(?:java\.lang\.)?System\s*\.\s*exit\s*\([^;]+\)\s*;"
            r"\s*return\s*$",
            prefix,
            flags=re.DOTALL,
        )
    )


def _java_constructor_member_index(
    source: str,
    class_name: str,
    member: str,
) -> int | None:
    record = re.search(
        rf"\brecord\s+{re.escape(class_name)}\s*\((?P<parameters>[^)]*)\)",
        source,
        flags=re.DOTALL,
    )
    parameter_texts: tuple[str, ...] = tuple()
    if record is not None:
        parameter_texts = _split_top_level_arguments(record.group("parameters"))
    else:
        constructor = re.search(
            rf"\b{re.escape(class_name)}\s*\((?P<parameters>[^)]*)\)\s*"
            r"(?:throws\s+[^\{]+)?\{",
            source,
            flags=re.DOTALL,
        )
        if constructor is not None:
            parameter_texts = _split_top_level_arguments(
                constructor.group("parameters")
            )
    names = [
        identifiers[-1]
        for parameter in parameter_texts
        if (identifiers := re.findall(r"[A-Za-z_]\w*", parameter))
    ]
    return names.index(member) if member in names else None


def _transparent_conversion_call_argument(expression: str) -> str | None:
    cipher_transform = re.fullmatch(
        r"[A-Za-z_]\w*\s*\.\s*doFinal\s*\((.*)\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if cipher_transform:
        arguments = _split_top_level_arguments(cipher_transform.group(1).strip())
        if len(arguments) == 1:
            # A completed cipher transform cannot make already-random input
            # predictable.  Propagating the input is conservative for fixed
            # plaintext, while preserving generated-key provenance through
            # serialization/wrapping helpers.
            return arguments[0]

    chained_encode = re.fullmatch(
        r"[A-Za-z_][\w.]*\s*\.\s*encode\s*\((.*)\)\s*\.\s*array\s*\(\s*\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if chained_encode:
        return _first_argument_text(chained_encode.group(1))

    key_factory = re.fullmatch(
        r".+\s*\.\s*(?:generatePrivate|generatePublic)\s*\((.*)\)",
        expression.strip(),
        flags=re.DOTALL,
    )
    if key_factory:
        arguments = _split_top_level_arguments(key_factory.group(1).strip())
        if len(arguments) == 1:
            return arguments[0]

    match = re.fullmatch(r"([A-Za-z_]\w*)\s*\((.*)\)", expression.strip(), flags=re.DOTALL)
    if not match:
        return None
    helper_name = match.group(1)
    arguments = _split_top_level_arguments(match.group(2).strip())
    if len(arguments) != 1:
        return None
    lowered = helper_name.lower()
    transparent_name = (
        "base64" in lowered
        or ("hex" in lowered and ("byte" in lowered or "binary" in lowered or lowered.startswith(("to", "from", "parse"))))
        or re.fullmatch(r"(?:get|to|from|create|make)?(?:aes|secret)?keybytes?", lowered) is not None
        or re.fullmatch(r"(?:get|to|from|decode|parse)(?:aes|secret)?key", lowered) is not None
        or re.fullmatch(r"(?:to|from|stringto)?bytes?", lowered) is not None
    )
    if transparent_name:
        return arguments[0]
    return None


def _java_call_receiver(source: str, call_start: int) -> str | None:
    suffix = source[call_start : call_start + 160]
    direct = re.match(
        r"\s*([A-Za-z_]\w*)\s*\.\s*[A-Za-z_]\w*\s*\(",
        suffix,
    )
    if direct:
        return direct.group(1)
    prefix = source[max(0, call_start - 160) : call_start]
    match = re.search(r"([A-Za-z_]\w*)\s*\.\s*$", prefix)
    return match.group(1) if match else None


def _java_split_literal_delimiter(raw: str) -> str:
    """Return the literal separator represented by a simple split regex."""

    java_value = raw.replace("\\\\", "\\")
    escaped_literal = re.fullmatch(r"\\([.$|?*+(){}\[\]^\\-])", java_value)
    return escaped_literal.group(1) if escaped_literal is not None else java_value


def _split_top_level_arguments(arguments_text: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    for index, char in enumerate(arguments_text):
        if in_string or in_char:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif in_string and char == '"':
                in_string = False
            elif in_char and char == "'":
                in_char = False
            continue
        if char == '"':
            in_string = True
        elif char == "'":
            in_char = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(arguments_text[start:index].strip())
            start = index + 1
    parts.append(arguments_text[start:].strip())
    return tuple(part for part in parts if part)


def _split_top_level_operator(expression: str, operator: str) -> tuple[str, ...]:
    """Split on a one-character operator outside strings and delimiters."""

    if len(operator) != 1:
        raise ValueError("top-level operator must be one character")
    parts: list[str] = []
    start = 0
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    for index, current in enumerate(expression):
        if in_string or in_char:
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif in_string and current == '"':
                in_string = False
            elif in_char and current == "'":
                in_char = False
            continue
        if current == '"':
            in_string = True
        elif current == "'":
            in_char = True
        elif current in "([{":
            depth += 1
        elif current in ")]}":
            depth -= 1
        elif current == operator and depth == 0:
            parts.append(expression[start:index].strip())
            start = index + 1
    if not parts:
        return (expression.strip(),)
    parts.append(expression[start:].strip())
    return tuple(part for part in parts if part)
