"""TypeScript and C# bounded source/value-flow analysis."""

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
from crypto_steering.validation.source.base import (
    Assignment,
    CallExpression,
    ClassDefinition,
    FunctionDefinition,
    ReturnExpression,
    SourceAnalyzer,
)
from crypto_steering.validation.source.helpers import (
    _IDENTIFIER,
    _MAX_RESOLUTION_STEPS,
    _bind_c_style_arguments,
    _c_style_argument_value,
    _c_style_call_result_property,
    _c_style_conditional_parts,
    _c_style_container_elements,
    _c_style_destructured_parameter,
    _c_style_direct_path_is_definitely_reached,
    _c_style_direct_path_is_reachable,
    _c_style_expression_never_returns,
    _c_style_has_literal_fallback,
    _c_style_if_chain_overwrites_or_terminates,
    _c_style_literal_boolean_path_is_reachable,
    _c_style_nonempty_true_branch_reference,
    _c_style_object_property_value,
    _c_style_object_value_expressions,
    _c_style_parameter_default,
    _c_style_postfix_update_operand,
    _compact_reference,
    _csharp_constructor_arguments,
    _csharp_conversion_input_reference,
    _csharp_first_concatenated_conversion_input,
    _csharp_hash_algorithm_name_constructor_argument,
    _csharp_hash_algorithm_name_property_receiver,
    _csharp_interpolation_expression_mask,
    _csharp_method_call_receiver,
    _csharp_primary_record_member_index,
    _csharp_tuple_expression_elements,
    _csharp_tuple_parameter_member_index,
    _direct_c_style_value,
    _evaluate_c_style_int,
    _evaluate_c_style_string_condition,
    _extract_c_style_assignments,
    _extract_c_style_calls,
    _extract_c_style_classes,
    _extract_c_style_conditional_ranges,
    _extract_c_style_functions,
    _extract_csharp_implicit_object_initializer_calls,
    _find_matching_masked,
    _literal_values_in_expression,
    _merge_resolved,
    _merge_transformed_origins,
    _mutation_target_reference,
    _next_c_style_statement_end_masked,
    _normalize_reference,
    _origin_preserving_call_argument,
    _origin_preserving_property_receiver,
    _origin_preserving_receiver_wrapper,
    _parameter_name,
    _prepend_resolution,
    _resolved_without_empty_alternatives,
    _simple_call,
    _simple_reference,
    _single_argument_wrapper,
    _split_inline_arguments,
    _split_parameter_text,
    _string_literal,
    _strip_expression,
    _top_level_c_style_arithmetic_operands,
    _top_level_c_style_concatenation,
    _top_level_c_style_conditional,
    _top_level_c_style_fallback,
    _typescript_arrow_body_start,
    _typescript_constructor_parameter_property,
    _typescript_crypto_namespace_aliases,
    _typescript_filesystem_namespace_aliases,
    _typescript_filesystem_read_aliases,
    _typescript_hash_digest_inputs,
    _typescript_hmac_digest_inputs,
    _typescript_identity_conditional_expression,
    _typescript_import_aliases,
    _typescript_process_runtime_aliases,
    _typescript_template_expression_mask,
    _typescript_template_fields,
    _typescript_template_interpolations,
    csharp_activator_constructor_arguments,
    find_matching,
    mask_c_style_comments_and_literals,
)


_TYPESCRIPT_DIGEST_LENGTHS = {
    "md5": 16,
    "sha1": 20,
    "sha224": 28,
    "sha256": 32,
    "sha384": 48,
    "sha512": 64,
}


def _typescript_encoded_string_length(value: str, encoding: str) -> int | None:
    normalized = encoding.lower().replace("-", "")
    if normalized == "hex":
        prefix = re.match(r"[0-9a-fA-F]*", value).group(0)
        return len(prefix) // 2
    codecs = {
        "utf8": "utf-8",
        "utf16le": "utf-16-le",
        "ucs2": "utf-16-le",
        "ascii": "ascii",
        "latin1": "latin-1",
        "binary": "latin-1",
    }
    codec = codecs.get(normalized)
    if codec is None:
        return None
    try:
        return len(value.encode(codec))
    except UnicodeEncodeError:
        return None


def _chained_call_receiver(
    source: str,
    masked: str,
    call_start: int,
) -> str | None:
    """Recover ``producer(...)`` from ``producer(...).consumer(...)``.

    The lightweight call index starts the consumer call at ``consumer`` and
    therefore cannot include a receiver that is itself a call expression.
    Mutation analysis needs that receiver for idioms such as
    ``randomBytes(16).copy(iv)`` and
    ``BitConverter.GetBytes(ticks).CopyTo(salt, 0)``.
    """

    cursor = call_start
    while cursor > 0 and masked[cursor - 1].isspace():
        cursor -= 1
    if cursor == 0 or masked[cursor - 1] != ".":
        return None
    cursor -= 1
    while cursor > 0 and masked[cursor - 1].isspace():
        cursor -= 1
    receiver_end = cursor
    if cursor == 0 or masked[cursor - 1] != ")":
        return None

    depth = 0
    opening = -1
    for position in range(cursor - 1, -1, -1):
        character = masked[position]
        if character == ")":
            depth += 1
        elif character == "(":
            depth -= 1
            if depth == 0:
                opening = position
                break
    if opening == -1:
        return None

    callee = re.search(
        rf"(?P<callee>(?:new\s+)?(?:{_IDENTIFIER}\s*\.\s*)*"
        rf"{_IDENTIFIER})\s*$",
        masked[:opening],
    )
    if callee is None:
        return None
    receiver_start = callee.start("callee")
    receiver = source[receiver_start:receiver_end].strip()
    return receiver or None


class CStyleAnalyzer(SourceAnalyzer):
    """Bounded resolver for TypeScript and C# source."""

    def __init__(
        self,
        source: str,
        language: Language,
        *,
        analysis_profile: AnalysisProfile = "whole_program",
    ) -> None:
        if language not in {"typescript", "csharp"}:
            raise ValueError(f"CStyleAnalyzer does not support {language!r}")
        self.language = language
        self.source = source
        self.analysis_profile = analysis_profile
        self.masked = mask_c_style_comments_and_literals(source)
        self.classes = tuple(_extract_c_style_classes(source, self.masked))
        self.functions = tuple(
            _extract_c_style_functions(source, self.masked, language)
        )
        self.assignments = tuple(
            _extract_c_style_assignments(
                source,
                self.masked,
                language,
                self.functions,
            )
        )
        self._assignment_scopes = {
            assignment: self._containing_function(assignment.start)
            for assignment in self.assignments
        }
        self._conditional_ranges = tuple(
            _extract_c_style_conditional_ranges(self.masked, self.language)
        )
        definition_starts = {item.start for item in self.functions}
        calls = list(_extract_c_style_calls(source, self.masked))
        if language == "typescript":
            calls.extend(
                _extract_c_style_calls(
                    source,
                    _typescript_template_expression_mask(source, self.masked),
                )
            )
        if language == "csharp":
            calls.extend(
                _extract_csharp_implicit_object_initializer_calls(
                    source,
                    self.masked,
                )
            )
            calls.extend(
                _extract_c_style_calls(
                    source,
                    _csharp_interpolation_expression_mask(source, self.masked),
                )
            )
        self._calls = tuple(
            sorted(
                {
                    (call.start, call.end, call.callee): call
                    for call in calls
                    if call.start not in definition_starts
                }.values(),
                key=lambda item: item.start,
            )
        )
        self.import_aliases = _typescript_import_aliases(source) if language == "typescript" else {}
        self.process_runtime_aliases = (
            _typescript_process_runtime_aliases(source)
            if language == "typescript"
            else frozenset()
        )
        self.filesystem_read_aliases = (
            _typescript_filesystem_read_aliases(source)
            if language == "typescript"
            else frozenset()
        )
        self.filesystem_namespace_aliases = (
            _typescript_filesystem_namespace_aliases(source)
            if language == "typescript"
            else frozenset()
        )
        self.crypto_namespace_aliases = (
            _typescript_crypto_namespace_aliases(source)
            if language == "typescript"
            else frozenset()
        )
        self._resolution_depth = 0
        self._reachability_depth = 0
        self._entrypoint_check_active = False
        self._resolution_cache: dict[tuple[object, ...], ResolvedValue] = {}
        self._reachability_cache: dict[int, bool] = {}

    def calls(self, *simple_names: str) -> tuple[CallExpression, ...]:
        if not simple_names:
            return self._calls
        wanted = set(simple_names)
        aliases = {
            alias
            for imported, alias in self.import_aliases.items()
            if imported in wanted
        }
        return tuple(
            call
            for call in self._calls
            if call.simple_name in wanted or call.simple_name in aliases
        )

    def resolve_origin(self, expression: str, index: int) -> ResolvedValue:
        return self._resolve(
            expression,
            index,
            mode="origin",
            seen=frozenset(),
            bindings=None,
        )

    def resolve_strings(self, expression: str, index: int) -> ResolvedValue:
        return self._resolve(
            expression,
            index,
            mode="string",
            seen=frozenset(),
            bindings=None,
        )

    def resolve_ints(self, expression: str, index: int) -> ResolvedValue:
        return self._resolve(
            expression,
            index,
            mode="int",
            seen=frozenset(),
            bindings=None,
        )

    def static_byte_length(self, expression: str, index: int) -> int | None:
        if self.language == "typescript":
            return self._typescript_static_buffer_length(
                expression,
                index,
                seen=frozenset(),
                bindings=None,
            )
        if self.language == "csharp":
            return self._csharp_static_buffer_length(
                expression,
                index,
                seen=frozenset(),
                bindings=None,
            )
        return None

    def latest_assignment(self, name: str, index: int) -> Assignment | None:
        normalized = _normalize_reference(name)
        compact_name = re.sub(r"\s+", "", name)
        explicit_member = "." in compact_name

        def matches_reference(assignment: Assignment) -> bool:
            if explicit_member:
                compact_assignment = re.sub(r"\s+", "", assignment.name)
                if compact_assignment == compact_name:
                    return True
                owner, member = compact_name.rsplit(".", 1)
                return bool(
                    owner not in {"this", "base"}
                    and compact_assignment == member
                    and self._containing_class_name(assignment.start)
                    == owner.rsplit(".", 1)[-1]
                )
            return _normalize_reference(assignment.name) == normalized

        containing = self._containing_function(index)
        parameter_lookup = bool(
            containing is not None
            and name in containing.parameters
            and "." not in name
        )
        scoped = [
            assignment
            for assignment in self.assignments
            if matches_reference(assignment)
            and self._assignment_scopes[assignment] == containing
            and self._profile_assignment_is_reachable(assignment)
            and (
                not parameter_lookup
                # A parameter named ``algorithm`` and an instance field named
                # ``this.algorithm`` intentionally normalize to the same
                # reference for field lookups.  They are different bindings
                # while resolving the parameter itself, though; otherwise a
                # constructor assignment such as ``this.algorithm =
                # algorithm`` becomes a false self-cycle.
                or re.sub(r"\s+", "", assignment.name) == normalized
            )
        ]
        matches = [
            assignment
            for assignment in scoped
            if assignment.end <= index
        ]
        if matches:
            return max(matches, key=lambda item: item.start)
        if containing is not None and name in containing.parameters:
            return None
        if containing is not None:
            # TypeScript closures and C# local functions may capture locals
            # from a lexically enclosing function. Search only true enclosing
            # scopes, nearest first, so same-named locals in sibling methods
            # can never satisfy this reference.
            enclosing = sorted(
                (
                    candidate
                    for candidate in self.functions
                    if candidate is not containing
                    and candidate.start < containing.start
                    and containing.end <= candidate.end
                ),
                key=lambda candidate: candidate.end - candidate.start,
            )
            for parent in enclosing:
                parent_matches = [
                    assignment
                    for assignment in self.assignments
                    if matches_reference(assignment)
                    and self._assignment_scopes[assignment] == parent
                    and self._profile_assignment_is_reachable(assignment)
                    and assignment.end <= index
                ]
                if parent_matches:
                    return max(parent_matches, key=lambda item: item.start)
        # Constants and static fields can be declared after a method body, but
        # locals from another method must never satisfy this reference.
        matches = [
            assignment
            for assignment in self.assignments
            if matches_reference(assignment)
            and self._assignment_scopes[assignment] is None
            and self._profile_assignment_is_reachable(assignment)
            and not (assignment.start <= index < assignment.end)
        ]
        earlier = [assignment for assignment in matches if assignment.end <= index]
        if earlier:
            return max(earlier, key=lambda item: item.start)
        return min(matches, key=lambda item: item.start) if matches else None

    def reaching_assignment_alternatives(
        self,
        name: str,
        index: int,
    ) -> tuple[Assignment, ...]:
        """Return value assignments that may reach a C-style use site.

        The ordinary resolver intentionally follows the latest assignment.
        That is correct for an unconditional overwrite, but not for patterns
        such as ``let password = "demo"; if (arg) password = arg``.  In that
        case both the fixed default and the runtime branch can reach the sink.
        This helper keeps the analysis bounded: it considers only the same
        lexical binding and walks back to the nearest unconditional baseline.
        """

        latest = self.latest_assignment(name, index)
        if latest is None:
            return tuple()
        containing = self._assignment_scopes.get(latest)
        normalized = _normalize_reference(name)
        compact_name = re.sub(r"\s+", "", name)
        explicit_member = "." in compact_name

        def matches_reference(assignment: Assignment) -> bool:
            if explicit_member:
                compact_assignment = re.sub(r"\s+", "", assignment.name)
                if compact_assignment == compact_name:
                    return True
                owner, member = compact_name.rsplit(".", 1)
                return bool(
                    owner not in {"this", "base"}
                    and compact_assignment == member
                    and self._containing_class_name(assignment.start)
                    == owner.rsplit(".", 1)[-1]
                )
            return _normalize_reference(assignment.name) == normalized

        candidates = tuple(
            sorted(
                (
                    assignment
                    for assignment in self.assignments
                    if assignment.end <= index
                    and self._assignment_scopes[assignment] == containing
                    and matches_reference(assignment)
                    and self._profile_assignment_is_reachable(assignment)
                    and _c_style_direct_path_is_reachable(
                        self,
                        assignment.start,
                    )
                ),
                key=lambda item: item.start,
            )
        )
        if not candidates:
            return tuple()
        if (
            len(candidates) > 1
            and not self._assignment_is_conditional(candidates[0])
            and _c_style_if_chain_overwrites_or_terminates(
                self,
                name,
                candidates[0].end,
                index,
            )
        ):
            candidates = candidates[1:]
        latest = candidates[-1]
        if not self._assignment_is_conditional(latest):
            return (latest,)
        if self.position_is_definitely_reached(latest.start):
            return (latest,)

        start = len(candidates) - 1
        while start > 0 and self._assignment_is_conditional(candidates[start]):
            start -= 1
        return candidates[start:]

    def _profile_assignment_is_reachable(self, assignment: Assignment) -> bool:
        if self.analysis_profile != "reachable_security_flow":
            return True
        return _c_style_literal_boolean_path_is_reachable(
            self.masked,
            assignment.start,
        )

    def _assignment_is_conditional(self, assignment: Assignment) -> bool:
        if any(
            start < assignment.start < end
            for start, end in self._conditional_ranges
        ):
            return True
        statement_start = max(
            self.masked.rfind(";", 0, assignment.start),
            self.masked.rfind("{", 0, assignment.start),
            self.masked.rfind("}", 0, assignment.start),
            self.masked.rfind("\n", 0, assignment.start),
        ) + 1
        prefix = self.masked[statement_start:assignment.start]
        return bool(
            re.search(
                r"\b(?:if|for|while|switch|catch)\s*\([^;{}]*\)\s*$"
                r"|\b(?:else|do)\s*$",
                prefix,
            )
        )

    def latest_mutation_origin(
        self,
        name: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> tuple[CallExpression, ResolvedValue] | None:
        """Resolve a bounded in-place write to a local buffer.

        Initializer-only provenance is unsound for mutable byte arrays. A
        later ``Buffer.copy`` or ``Buffer.BlockCopy`` can replace random bytes
        with predictable material before the cryptographic sink.
        """

        normalized = _normalize_reference(name)
        candidates: list[tuple[CallExpression, ResolvedValue]] = []
        for call in self._calls:
            if (
                call.end > index
                or not self.position_is_in_same_or_enclosing_function(
                    call.start,
                    index,
                )
            ):
                continue
            mutation = self._mutation_call_value(
                call,
                normalized,
                seen=seen,
                bindings=bindings,
            )
            if mutation is not None:
                candidates.append((call, mutation))
        if not candidates:
            return None
        ordered = sorted(candidates, key=lambda item: item[0].start)
        if len(ordered) > 1 and all(
            self.language == "csharp"
            and call.simple_name == "BlockCopy"
            and re.search(
                r"(?:^|\.)Buffer\.BlockCopy$",
                re.sub(r"\s+", "", call.callee),
            )
            for call, _resolved in ordered
        ):
            # Multiple BlockCopy calls into one newly allocated destination
            # normally concatenate independent components. The origin of the
            # completed buffer therefore depends on every copied component,
            # not only the lexically last write.
            return (
                ordered[-1][0],
                _merge_transformed_origins(
                    name,
                    (resolved for _call, resolved in ordered),
                ),
            )
        return ordered[-1]

    def latest_indexed_assignment_origin(
        self,
        name: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> tuple[int, ResolvedValue] | None:
        """Resolve the value copied into a byte container by indexed writes."""

        if self.language not in {"typescript", "csharp"}:
            return None
        normalized_target = _normalize_reference(name)
        candidates: list[tuple[int, ResolvedValue]] = []
        pattern = re.compile(
            rf"(?P<target>(?:this\s*\.\s*)?{_IDENTIFIER}"
            rf"(?:\s*\.\s*{_IDENTIFIER})?)\s*\[[^\]\r\n]+\]\s*=(?!=|>)"
        )
        for match in pattern.finditer(self.masked, 0, index):
            if _normalize_reference(match.group("target")) != normalized_target:
                continue
            expression_end = _next_c_style_statement_end_masked(
                self.masked,
                match.end(),
                self.language,
            )
            if expression_end == -1 or expression_end >= index:
                continue
            if not self.position_is_in_same_or_enclosing_function(
                match.start(),
                index,
            ):
                continue
            state_key = f"indexed-write:{match.start()}:{normalized_target}"
            if state_key in seen:
                continue
            expression = self.source[match.end() : expression_end].strip()
            if not expression:
                continue
            candidates.append(
                (
                    match.start(),
                    self._resolve(
                        expression,
                        match.start(),
                        mode="origin",
                        seen=seen | frozenset({state_key}),
                        bindings=bindings,
                    ),
                )
            )
        return max(candidates, key=lambda item: item[0]) if candidates else None

    def _mutation_call_value(
        self,
        call: CallExpression,
        normalized_target: str,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        compact = re.sub(r"\s+", "", call.callee)
        simple = call.simple_name
        mutation_key = f"mutation:{call.start}:{normalized_target}"
        if mutation_key in seen:
            return None
        next_seen = seen | frozenset({mutation_key})

        if self.language == "typescript" and simple == "push" and call.arguments:
            receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
            if _normalize_reference(receiver) == normalized_target:
                return self._resolve(
                    call.arguments[0],
                    call.start,
                    mode="origin",
                    seen=next_seen,
                    bindings=bindings,
                )

        if self.language == "typescript" and simple == "copy" and call.arguments:
            target = _mutation_target_reference(call.arguments[0])
            receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
            if not receiver:
                receiver = _chained_call_receiver(
                    self.source,
                    self.masked,
                    call.start,
                ) or ""
            if (
                target is not None
                and _normalize_reference(target) == normalized_target
                and receiver
            ):
                return self._resolve(
                    receiver,
                    call.start,
                    mode="origin",
                    seen=next_seen,
                    bindings=bindings,
                )

        if self.language == "typescript" and simple == "set" and call.arguments:
            receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
            if _normalize_reference(receiver) == normalized_target:
                return self._resolve(
                    call.arguments[0],
                    call.start,
                    mode="origin",
                    seen=next_seen,
                    bindings=bindings,
                )

        if self.language == "csharp" and simple == "CopyTo" and call.arguments:
            target = _mutation_target_reference(call.arguments[0])
            receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
            if not receiver:
                receiver = _chained_call_receiver(
                    self.source,
                    self.masked,
                    call.start,
                ) or ""
            if (
                target is not None
                and _normalize_reference(target) == normalized_target
                and receiver
            ):
                return self._resolve(
                    receiver,
                    call.start,
                    mode="origin",
                    seen=next_seen,
                    bindings=bindings,
                )

        if (
            self.language == "csharp"
            and simple == "BlockCopy"
            and re.search(r"(?:^|\.)Buffer\.BlockCopy$", compact)
            and len(call.arguments) >= 3
        ):
            target = _mutation_target_reference(call.arguments[2])
            if target is not None and _normalize_reference(target) == normalized_target:
                role = normalized_target.rsplit(".", 1)[-1].lower()
                serialized_role = self._resolve_csharp_serialized_role(
                    call.arguments[0],
                    call.start,
                    role=role,
                    seen=next_seen,
                    bindings=bindings,
                )
                if serialized_role is not None:
                    return serialized_role
                return self._resolve(
                    call.arguments[0],
                    call.start,
                    mode="origin",
                    seen=next_seen,
                    bindings=bindings,
                )

        if (
            self.language == "csharp"
            and simple == "Copy"
            and re.search(r"(?:^|\.)Array\.Copy$", compact)
            and len(call.arguments) >= 2
        ):
            target_index = 2 if len(call.arguments) >= 5 else 1
            target = _mutation_target_reference(call.arguments[target_index])
            if target is not None and _normalize_reference(target) == normalized_target:
                copied_range = self._resolve_csharp_array_copy_source(
                    call,
                    seen=next_seen,
                    bindings=bindings,
                )
                if copied_range is not None:
                    return copied_range
                return self._resolve(
                    call.arguments[0],
                    call.start,
                    mode="origin",
                    seen=next_seen,
                    bindings=bindings,
                )

        if self.language == "csharp" and simple == "Pbkdf2":
            destination_expression = call.named_arguments.get("destination")
            if destination_expression is None and len(call.arguments) >= 4:
                candidate = call.arguments[2]
                if _simple_reference(candidate) is not None:
                    destination_expression = candidate
            destination = (
                _mutation_target_reference(destination_expression)
                if destination_expression is not None
                else None
            )
            if (
                destination is not None
                and _normalize_reference(destination) == normalized_target
            ):
                call_expression = self.source[call.start : call.end]
                derived = self._resolve_csharp_derived_origin(
                    call_expression,
                    call.start,
                    seen=next_seen,
                    bindings=bindings,
                )
                if derived is not None:
                    return derived

        if (
            self.language == "csharp"
            and simple in {"Read", "ReadAsync"}
            and call.arguments
        ):
            target = _mutation_target_reference(call.arguments[0])
            receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
            if (
                target is not None
                and _normalize_reference(target) == normalized_target
                and self._csharp_receiver_is_file_stream(receiver, call.start)
            ):
                round_trip = self._resolve_csharp_file_stream_round_trip(
                    receiver,
                    call.start,
                    seen=next_seen,
                    bindings=bindings,
                )
                if round_trip is not None:
                    return round_trip
                return ResolvedValue(
                    target,
                    "runtime_input",
                    steps=(
                        ResolutionStep(
                            target,
                            f"resolve file-stream buffer read {call.callee}",
                            "runtime_input",
                            self.source_span(call.start, call.end),
                        ),
                    ),
                )

        if not call.arguments:
            return None
        target = _mutation_target_reference(call.arguments[0])
        if target is None or _normalize_reference(target) != normalized_target:
            return None
        origin = self._mutation_call_origin(call)
        if origin is None:
            return None
        mutation_expression = self.source[call.start : call.end]
        origin_steps: tuple[ResolutionStep, ...] = ()
        if self.language == "csharp" and simple == "NextBytes":
            receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
            if not receiver:
                receiver = _chained_call_receiver(
                    self.source,
                    self.masked,
                    call.start,
                ) or ""
                if receiver:
                    mutation_expression = f"{receiver}.{mutation_expression}"
            if receiver:
                receiver_origin = self.resolve_origin(receiver, call.start)
                origin_steps = receiver_origin.steps
                receiver_reference = _simple_reference(receiver)
                receiver_assignment = (
                    self.latest_assignment(receiver_reference, call.start)
                    if receiver_reference is not None
                    else None
                )
                if receiver_assignment is not None:
                    origin_steps = (
                        ResolutionStep(
                            receiver_assignment.expression,
                            "resolve mutation receiver assignment",
                            receiver_origin.origin,
                            self.source_span(
                                receiver_assignment.start,
                                receiver_assignment.end,
                            ),
                        ),
                        *origin_steps,
                    )
        return ResolvedValue(
            target,
            origin,
            steps=origin_steps + (
                ResolutionStep(
                    mutation_expression,
                    f"resolve in-place fill {call.callee}",
                    origin,
                    self.source_span(call.start, call.end),
                ),
            ),
        )

    def _csharp_receiver_is_file_stream(self, receiver: str, index: int) -> bool:
        reference = _simple_reference(receiver)
        if reference is None:
            return False
        assignment = self.latest_assignment(reference, index)
        if assignment is None:
            return False
        expression = _strip_expression(assignment.expression)
        return bool(
            re.match(
                r"(?:(?:System\s*\.\s*IO\s*\.)?File\s*\.\s*"
                r"(?:Open|OpenRead)\s*\(|new\s+(?:System\s*\.\s*IO\s*\.)?"
                r"FileStream\s*\()",
                expression,
            )
        )

    def _mutation_call_origin(self, call: CallExpression) -> ValueOrigin | None:
        compact = re.sub(r"\s+", "", call.callee)
        simple = call.simple_name
        if self.language == "typescript":
            canonical = next(
                (
                    imported
                    for imported, alias in self.import_aliases.items()
                    if alias == simple
                ),
                simple,
            )
            if canonical in {"randomFill", "randomFillSync", "getRandomValues"}:
                return "secure_random"
            return None

        if simple == "Fill" and "RandomNumberGenerator" in compact:
            return "secure_random"
        receiver_prefix = self.source[max(0, call.start - 160) : call.start]
        if simple == "GetBytes" and re.search(
            r"(?:new\s+(?:(?:System\s*\.)?Security\s*\.\s*Cryptography\s*\.)?"
            r"RNGCryptoServiceProvider\s*\(\s*\)"
            r"|(?:System\s*\.\s*Security\s*\.\s*Cryptography\s*\.)?"
            r"RandomNumberGenerator\s*\.\s*Create\s*\(\s*\))"
            r"\s*\)?\s*\.\s*$",
            receiver_prefix,
        ):
            return "secure_random"
        if simple == "NextBytes" and re.search(
            r"new\s+(?:System\s*\.\s*)?Random\s*\([^)]*\)"
            r"\s*\)?\s*\.\s*$",
            receiver_prefix,
        ):
            return "non_csprng"
        receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
        receiver_origin = (
            self.resolve_origin(receiver, call.start) if receiver else None
        )
        if simple == "GetBytes" and receiver_origin is not None:
            if receiver_origin.origin == "secure_random":
                return "secure_random"
        if simple == "NextBytes" and receiver_origin is not None:
            if receiver_origin.origin in {"non_csprng", "predictable", "constant"}:
                return "non_csprng"
        return None

    def _containing_function(self, index: int) -> FunctionDefinition | None:
        matches = [
            function
            for function in self.functions
            if function.start <= index <= function.end
        ]
        return min(
            matches,
            key=lambda item: item.end - item.start,
        ) if matches else None

    def position_is_in_same_or_enclosing_function(
        self,
        candidate_index: int,
        target_index: int,
    ) -> bool:
        """Return whether a prior expression can belong to the target scope."""

        candidate = self._containing_function(candidate_index)
        target = self._containing_function(target_index)
        return candidate == target or bool(
            candidate is not None
            and target is not None
            and candidate.start <= target.start
            and target.end <= candidate.end
        )

    def function(self, name: str) -> tuple[FunctionDefinition, ...]:
        simple = _normalize_reference(name).rsplit(".", 1)[-1]
        return tuple(item for item in self.functions if item.name == simple)

    def containing_function_name(self, index: int) -> str | None:
        matches = [
            function
            for function in self.functions
            if function.start <= index <= function.end
        ]
        if not matches:
            return None
        return min(
            matches,
            key=lambda item: item.end - item.start,
        ).name

    def containing_function_is_called(self, index: int) -> bool:
        function = self._containing_function(index)
        if function is None:
            return False
        return any(
            not (function.start <= call.start <= function.end)
            and self._call_can_target_function(call, function)
            for name in self._call_names_for_function(function)
            for call in self.calls(name)
        )

    def position_is_locally_reachable(self, index: int) -> bool:
        return _c_style_direct_path_is_reachable(self, index)

    def position_is_reachable(self, index: int) -> bool:
        if self._reachability_depth >= _MAX_RESOLUTION_STEPS:
            # Reachability is used only to suppress proven-dead findings.  An
            # exhausted call-graph proof must therefore retain the finding.
            return True
        self._reachability_depth += 1
        try:
            if not _c_style_direct_path_is_reachable(self, index):
                return False
            containing = self._containing_function(index)
            if containing is None or containing.name.lower() == "main":
                return True
            callers = [
                call
                for name in self._call_names_for_function(containing)
                for call in self.calls(name)
                if not (containing.start <= call.start <= containing.end)
                and self._call_can_target_function(call, containing)
            ]
            callers.extend(self._callback_registrations_for_function(containing))
            if not callers and self._csharp_static_constructor_is_reached(
                containing
            ):
                return True
            if not callers:
                # Library-style functions are valid roots even when this source
                # does not contain an executable entrypoint for them. Once an
                # explicit application entrypoint exists, however, an uncalled
                # helper is proven dead and must not contaminate its results.
                return not self._has_explicit_entrypoint()
            return any(self._call_is_reachable(call) for call in callers)
        finally:
            self._reachability_depth -= 1

    def _has_explicit_entrypoint(self) -> bool:
        main_functions = tuple(
            function
            for function in self.functions
            if function.name.lower() == "main"
        )
        if not main_functions:
            if self.language in {"typescript", "csharp"}:
                # A script with executable top-level calls has a concrete
                # entry path. Source-defined helpers that are never reached
                # from that path must not contribute security findings.
                return any(
                    self._containing_function(call.start) is None
                    and (
                        self.language != "csharp"
                        or self._containing_class_name(call.start) is None
                    )
                    for call in self.calls()
                )
            return False
        if self.language == "csharp":
            return True
        if self._entrypoint_check_active:
            return False
        self._entrypoint_check_active = True
        try:
            return any(
                self._call_has_entry_path(call)
                for call in self.calls("main")
            )
        finally:
            self._entrypoint_check_active = False

    def _resolve(
        self,
        expression: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue:
        expr = _strip_expression(expression)
        cache_key = (
            expr,
            index,
            mode,
            seen,
            tuple(sorted((bindings or {}).items())),
            self._reachability_depth > 0,
            self._entrypoint_check_active,
        )
        cached = self._resolution_cache.get(cache_key)
        if cached is not None:
            return cached
        if self._resolution_depth >= _MAX_RESOLUTION_STEPS:
            step = ResolutionStep(
                expr,
                "inspect",
                mode,
                self.source_span(index, index),
            )
            return ResolvedValue(
                expr,
                "unknown",
                steps=(step,),
                unresolved_reason="source-resolution budget exhausted",
            )
        self._resolution_depth += 1
        try:
            resolved = self._resolve_impl(
                expression,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            if resolved.unresolved_reason != "source-resolution budget exhausted":
                self._resolution_cache[cache_key] = resolved
            return resolved
        finally:
            self._resolution_depth -= 1

    def _resolve_impl(
        self,
        expression: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue:
        expr = _strip_expression(expression)
        step = ResolutionStep(expr, "inspect", mode, self.source_span(index, index))
        if len(seen) >= _MAX_RESOLUTION_STEPS:
            return ResolvedValue(
                expr,
                "unknown",
                steps=(step,),
                unresolved_reason="source-resolution budget exhausted",
            )
        if not expr:
            return ResolvedValue(expression, "unknown", steps=(step,), unresolved_reason="empty expression")

        conditional = _c_style_conditional_parts(expr)
        conditional_key = f"conditional:{index}:{expr}"
        if conditional is not None and conditional_key not in seen:
            condition, when_true, when_false = conditional
            identity_expression = _typescript_identity_conditional_expression(
                condition,
                when_true,
                when_false,
            ) if self.language == "typescript" else None
            if identity_expression is not None:
                resolved = self._resolve(
                    identity_expression,
                    index,
                    mode=mode,
                    seen=seen | frozenset({conditional_key}),
                    bindings=bindings,
                )
                return _prepend_resolution(
                    expr,
                    "collapse identity conditional expression",
                    resolved,
                    step,
                )
            selected = _evaluate_c_style_string_condition(
                self,
                condition,
                index,
            )
            next_seen = seen | frozenset({conditional_key})
            if selected is not None:
                resolved = self._resolve(
                    when_true if selected else when_false,
                    index,
                    mode=mode,
                    seen=next_seen,
                    bindings=bindings,
                )
                return _prepend_resolution(
                    expr,
                    "resolve selected conditional expression",
                    resolved,
                    step,
                )
            resolved = _merge_resolved(
                expr,
                (
                    self._resolve(
                        when_true,
                        index,
                        mode=mode,
                        seen=next_seen,
                        bindings=bindings,
                    ),
                    self._resolve(
                        when_false,
                        index,
                        mode=mode,
                        seen=next_seen,
                        bindings=bindings,
                    ),
                ),
            )
            return _prepend_resolution(
                expr,
                "merge conditional expression alternatives",
                resolved,
                step,
            )

        if mode == "int":
            evaluated_expression = _evaluate_c_style_int(expr)
            if evaluated_expression is not None:
                return ResolvedValue(
                    expr,
                    "constant",
                    (str(evaluated_expression),),
                    (step,),
                )
            length_match = re.fullmatch(
                r"(?P<value>.+?)\s*\.\s*(?:Length|length)",
                expr,
                flags=re.DOTALL,
            )
            if length_match is not None:
                if self.language == "typescript":
                    buffer_length = self._typescript_static_buffer_length(
                        length_match.group("value"),
                        index,
                        seen=seen,
                        bindings=bindings,
                    )
                    if buffer_length is not None:
                        return ResolvedValue(
                            expr,
                            "constant",
                            (str(buffer_length),),
                            (
                                step,
                                ResolutionStep(
                                    expr,
                                    "resolve static Buffer length",
                                    "constant",
                                    self.source_span(index, index + len(expr)),
                                ),
                            ),
                        )
                resolved_value = self._resolve(
                    length_match.group("value"),
                    index,
                    mode="string",
                    seen=seen,
                    bindings=bindings,
                )
                if resolved_value.origin == "constant" and resolved_value.values:
                    lengths = tuple(
                        sorted(
                            {
                                str(len(value.encode("utf-8")))
                                for value in resolved_value.values
                            },
                            key=int,
                        )
                    )
                    return ResolvedValue(
                        expr,
                        "constant",
                        lengths,
                        (
                            step,
                            *resolved_value.steps,
                            ResolutionStep(
                                expr,
                                "resolve constant value length",
                                "constant",
                                self.source_span(index, index + len(expr)),
                            ),
                        ),
                    )
            substituted = expr
            substituted_steps: list[ResolutionStep] = []
            for length_reference in reversed(
                tuple(
                    re.finditer(
                        rf"(?:{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)"
                        r"\s*\.\s*(?:Length|length)\b",
                        substituted,
                    )
                )
            ):
                if _strip_expression(length_reference.group(0)) == expr:
                    continue
                resolved_length = self._resolve(
                    length_reference.group(0),
                    index,
                    mode="int",
                    seen=seen,
                    bindings=bindings,
                )
                if (
                    resolved_length.origin != "constant"
                    or len(resolved_length.values) != 1
                ):
                    continue
                substituted = (
                    substituted[: length_reference.start()]
                    + resolved_length.values[0]
                    + substituted[length_reference.end() :]
                )
                substituted_steps.extend(resolved_length.steps)
            if substituted != expr:
                evaluated = _evaluate_c_style_int(substituted)
                if evaluated is not None:
                    return ResolvedValue(
                        expr,
                        "constant",
                        (str(evaluated),),
                        (
                            step,
                            *substituted_steps,
                            ResolutionStep(
                                substituted,
                                "evaluate resolved numeric expression",
                                "constant",
                                self.source_span(index, index + len(expr)),
                            ),
                        ),
                    )
            symbolic_expression = re.fullmatch(
                r"[\s\w$.()+\-*/%<>&|^~]+",
                expr,
            )
            if symbolic_expression is not None and re.search(
                r"[+\-*/%<>&|^]",
                expr,
            ):
                substitutions = [expr]
                substituted_steps = []
                references = tuple(
                    dict.fromkeys(
                        match.group(0)
                        for match in re.finditer(
                            rf"\b{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*\b",
                            expr,
                        )
                        if not re.fullmatch(r"[A-Fa-f]+", match.group(0))
                    )
                )
                for reference in references:
                    resolution_key = f"numeric-expression:{reference}:{index}"
                    if resolution_key in seen:
                        continue
                    resolved_reference = self._resolve(
                        reference,
                        index,
                        mode="int",
                        seen=seen | frozenset({resolution_key}),
                        bindings=bindings,
                    )
                    numeric_values = tuple(
                        value
                        for value in resolved_reference.values
                        if re.fullmatch(r"[-+]?\d+(?:_\d+)*", value)
                    )
                    if (
                        resolved_reference.origin != "constant"
                        or not numeric_values
                        or len(numeric_values) != len(resolved_reference.values)
                    ):
                        continue
                    substitutions = [
                        re.sub(
                            rf"\b{re.escape(reference)}\b",
                            value,
                            candidate,
                        )
                        for candidate in substitutions
                        for value in numeric_values
                    ][:64]
                    substituted_steps.extend(resolved_reference.steps)
                if substitutions != [expr]:
                    evaluated_values = tuple(
                        sorted(
                            {
                                value
                                for candidate in substitutions
                                if (value := _evaluate_c_style_int(candidate))
                                is not None
                            }
                        )
                    )
                    if evaluated_values and len(evaluated_values) == len(
                        set(substitutions)
                    ):
                        return ResolvedValue(
                            expr,
                            "constant",
                            tuple(str(value) for value in evaluated_values),
                            (
                                step,
                                *substituted_steps,
                                ResolutionStep(
                                    " | ".join(substitutions),
                                    "evaluate resolved symbolic numeric expression",
                                    "constant",
                                    self.source_span(index, index + len(expr)),
                                ),
                            ),
                        )
            parsed = _simple_call(expr)
            numeric_wrappers = {
                "Number",
                "floor",
                "parseInt",
                "round",
                "trunc",
            }
            csharp_numeric_parse = bool(
                self.language == "csharp"
                and parsed is not None
                and re.sub(r"\s+", "", expr.split("(", 1)[0])
                in {
                    "int.Parse",
                    "Int32.Parse",
                    "System.Int32.Parse",
                    "Convert.ToInt32",
                    "System.Convert.ToInt32",
                }
            )
            if (
                parsed is not None
                and (
                    parsed[0].rsplit(".", 1)[-1] in numeric_wrappers
                    or csharp_numeric_parse
                )
            ):
                _, arguments = parsed
                if arguments:
                    resolved = self._resolve(
                        arguments[0],
                        index,
                        mode=mode,
                        seen=seen,
                        bindings=bindings,
                    )
                    if resolved.values or resolved.origin in {
                        "runtime_input",
                        "unknown",
                    }:
                        return _prepend_resolution(
                            expr,
                            "resolve numeric conversion input",
                            resolved,
                            step,
                        )
            if (
                parsed is not None
                and parsed[0].rsplit(".", 1)[-1] == "max"
                and parsed[1]
            ):
                _, arguments = parsed
                resolved_arguments = tuple(
                    self._resolve(
                        argument,
                        index,
                        mode=mode,
                        seen=seen,
                        bindings=bindings,
                    )
                    for argument in arguments
                )
                literal_bounds = tuple(
                    value
                    for argument in arguments
                    if (value := _evaluate_c_style_int(argument)) is not None
                )
                merged = _merge_resolved(expr, resolved_arguments)
                if literal_bounds:
                    lower_bound = max(literal_bounds)
                    bound_step = ResolutionStep(
                        expr,
                        "apply numeric maximum lower bound",
                        str(lower_bound),
                        self.source_span(index, index),
                    )
                    return ResolvedValue(
                        expr,
                        merged.origin,
                        merged.values,
                        (*merged.steps, bound_step),
                        merged.unresolved_reason,
                    )
                return _prepend_resolution(
                    expr,
                    "resolve numeric maximum arguments",
                    merged,
                    step,
                )
            template_inputs = (
                _typescript_template_interpolations(expr)
                if self.language == "typescript"
                else None
            )
            if template_inputs:
                resolved = _merge_resolved(
                    expr,
                    [
                        self._resolve(
                            interpolation,
                            index,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        )
                        for interpolation in template_inputs
                    ],
                )
                if resolved.values:
                    return _prepend_resolution(
                        expr,
                        "resolve numeric template interpolation",
                        resolved,
                        step,
                    )
            binary_reader = self._resolve_csharp_binary_integer_reader(
                expr,
                seen=seen,
                bindings=bindings,
            )
            if binary_reader is not None:
                return _prepend_resolution(
                    expr,
                    "resolve matching binary integer writer",
                    binary_reader,
                    step,
                )

        indexed = self._resolve_indexed_value(
            expr,
            index,
            mode=mode,
            seen=seen,
            bindings=bindings,
        )
        if indexed is not None:
            return _prepend_resolution(
                expr,
                "resolve indexed container element",
                indexed,
                step,
            )

        map_value = self._resolve_typescript_map_get(
            expr,
            index,
            mode=mode,
            seen=seen,
            bindings=bindings,
        )
        if map_value is not None:
            return _prepend_resolution(
                expr,
                "resolve Map.get values from Map.set",
                map_value,
                step,
            )

        iife_result = (
            self._resolve_typescript_iife_result_parameter(
                expr,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            if self.language == "typescript"
            else None
        )
        if iife_result is not None:
            return _prepend_resolution(
                expr,
                "resolve immediately-invoked arrow return parameter",
                iife_result,
                step,
            )

        file_round_trip = self._resolve_c_style_file_round_trip(
            expr,
            index,
            mode=mode,
            seen=seen,
            bindings=bindings,
        )
        if file_round_trip is not None:
            return _prepend_resolution(
                expr,
                "resolve prior file write",
                file_round_trip,
                step,
            )

        direct = self._direct_api_value(expr, mode)
        if direct is not None:
            return ResolvedValue(expr, direct[0], direct[1], (step,))

        if mode == "origin":
            template_inputs = (
                _typescript_template_interpolations(expr)
                if self.language == "typescript"
                else None
            )
            if template_inputs:
                resolved = _merge_transformed_origins(
                    expr,
                    (
                        self._resolve(
                            interpolation,
                            index,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        )
                        for interpolation in template_inputs
                    ),
                )
                return _prepend_resolution(
                    expr,
                    "resolve template interpolation origins",
                    resolved,
                    step,
                )
            postfix_operand = _c_style_postfix_update_operand(expr)
            if postfix_operand is not None:
                resolved = self._resolve(
                    postfix_operand,
                    index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
                return _prepend_resolution(
                    expr,
                    "resolve deterministic postfix update operand",
                    resolved,
                    step,
                )
            container_elements = _c_style_container_elements(
                expr,
                allow_shorthand_initializer=self.language == "csharp",
            )
            if container_elements is not None:
                if not container_elements:
                    return ResolvedValue(expr, "constant", steps=(step,))
                resolved = _merge_transformed_origins(
                    expr,
                    (
                        self._resolve(
                            element,
                            index,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        )
                        for element in container_elements
                    ),
                )
                return _prepend_resolution(
                    expr,
                    "resolve deterministic container element origins",
                    resolved,
                    step,
                )
            memory_stream_bytes = self._resolve_csharp_memory_stream_bytes(
                expr,
                index,
                seen=seen,
                bindings=bindings,
            )
            if memory_stream_bytes is not None:
                return _prepend_resolution(
                    expr,
                    "resolve bytes serialized to MemoryStream",
                    memory_stream_bytes,
                    step,
                )
            hash_property = self._resolve_csharp_hash_property(
                expr,
                index,
                seen=seen,
                bindings=bindings,
            )
            if hash_property is not None:
                return _prepend_resolution(
                    expr,
                    "resolve digest property input",
                    hash_property,
                    step,
                )
            derived = self._resolve_typescript_derived_origin(
                expr,
                index,
                seen=seen,
                bindings=bindings,
            )
            if derived is None:
                derived = self._resolve_csharp_derived_origin(
                    expr,
                    index,
                    seen=seen,
                    bindings=bindings,
                )
            if derived is not None:
                return _prepend_resolution(
                    expr,
                    "resolve deterministic derivation inputs",
                    derived,
                    step,
                )
            wrapper_argument = self._csharp_span_wrapper_argument(expr)
            if wrapper_argument is not None:
                resolved = self._resolve(
                    wrapper_argument,
                    index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
                return _prepend_resolution(
                    expr,
                    "unwrap custom span buffer",
                    resolved,
                    step,
                )

        conditional = _top_level_c_style_conditional(expr)
        fallback = _top_level_c_style_fallback(expr)
        direct = _direct_c_style_value(expr, self.language, mode)
        local_call = _simple_call(expr)
        if (
            mode == "origin"
            and self.language == "csharp"
            and local_call is not None
            and self.function(local_call[0])
        ):
            # Runtime-looking arguments do not determine a local helper's
            # return value. Trace the helper body below; it may deliberately
            # ignore an argument or pass it through a KDF before returning.
            direct = None
        if conditional is not None or fallback is not None:
            # A lexical runtime source in one branch must not hide a fixed or
            # otherwise unsafe alternative branch.
            direct = None
        if direct is not None and direct[0] == "runtime_input":
            guarded_values = self._enclosing_string_equality_guard_values(
                expr,
                index,
            )
            if guarded_values:
                return ResolvedValue(
                    expr,
                    "constant",
                    guarded_values,
                    (
                        step,
                        ResolutionStep(
                            expr,
                            "resolve enclosing string equality branch",
                            "constant",
                            self.source_span(index, index + len(expr)),
                        ),
                    ),
                )
        if (
            direct is not None
            and mode == "string"
            and _simple_reference(expr) is not None
            and self.latest_assignment(expr, index) is not None
        ):
            # A qualified class constant (Example.ALGORITHM) has the same
            # lexical shape as an enum member (HashAlgorithmName.SHA256).
            # Prefer a real declaration when one exists in this source.
            direct = None
        if (
            direct is not None
            and self.language == "typescript"
            and mode == "origin"
            and direct[0] == "runtime_input"
            and re.fullmatch(
                rf"process\s*\.\s*env\s*\.\s*{_IDENTIFIER}",
                expr,
            )
        ):
            assignment = self.latest_assignment(expr, index)
            if assignment is not None:
                compact_expr = re.sub(r"\s+", "", expr)
                assignment_key = (
                    f"runtime-member-assignment:{assignment.start}:{compact_expr}"
                )
                if assignment_key not in seen:
                    assigned = self._resolve(
                        assignment.expression,
                        assignment.start,
                        mode=mode,
                        seen=seen | frozenset({assignment_key}),
                        bindings=bindings,
                    )
                    if self._assignment_is_conditional(assignment):
                        runtime = ResolvedValue(
                            expr,
                            direct[0],
                            direct[1],
                            (
                                step,
                                ResolutionStep(
                                    expr,
                                    "retain pre-assignment environment source",
                                    "runtime_input",
                                    self.source_span(index, index + len(expr)),
                                ),
                            ),
                        )
                        return _prepend_resolution(
                            expr,
                            "merge environment value and conditional assignment",
                            _merge_resolved(expr, (runtime, assigned)),
                            step,
                        )
                    return _prepend_resolution(
                        expr,
                        "resolve environment member assignment",
                        assigned,
                        step,
                    )
        if direct is not None:
            return ResolvedValue(expr, direct[0], direct[1], (step,))

        if conditional is not None:
            true_expression, false_expression = conditional
            nonempty_reference = (
                _c_style_nonempty_true_branch_reference(expr)
                if mode == "origin"
                else None
            )
            if nonempty_reference is not None:
                true_value = self._resolve_origin_without_empty_paths(
                    true_expression,
                    index,
                    seen=seen,
                    bindings=bindings,
                )
                false_value = self._resolve(
                    false_expression,
                    index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
                values = tuple(
                    item for item in (true_value, false_value) if item is not None
                )
                resolved = (
                    values[0]
                    if len(values) == 1
                    else _merge_resolved(expr, values)
                )
                return _prepend_resolution(
                    expr,
                    f"resolve nonempty conditional source {nonempty_reference}",
                    resolved,
                    step,
                )
            reachable_values = tuple(
                branch
                for branch in (true_expression, false_expression)
                if not _c_style_expression_never_returns(branch, self.language)
            )
            resolved_values = tuple(
                self._resolve(
                    branch,
                    index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
                for branch in reachable_values
            )
            resolved = (
                resolved_values[0]
                if len(resolved_values) == 1
                else _merge_resolved(expr, resolved_values)
            )
            return _prepend_resolution(
                expr,
                "merge conditional value sources",
                resolved,
                step,
            )

        if fallback is not None:
            left_expression, right_expression = fallback
            stripped_left = _strip_expression(left_expression)
            left_reference = _simple_reference(stripped_left)
            if (
                left_reference is None
                and self.language == "typescript"
                and re.fullmatch(
                    rf"process\s*\.\s*env\s*\.\s*{_IDENTIFIER}",
                    stripped_left,
                )
            ):
                left_reference = stripped_left
            if (
                left_reference is not None
                and (
                    self._fail_closed_nonempty_guard(left_reference, index)
                    or self._reference_is_truthy_on_path(left_reference, index)
                )
            ):
                resolved = self._resolve(
                    left_expression,
                    index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
            elif self._parameter_is_always_omitted(left_expression, index):
                resolved = self._resolve(
                    right_expression,
                    index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
            else:
                operands = tuple(
                    item
                    for item in (
                        self._resolve_non_nullish_fallback_operand(
                            left_expression,
                            index,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        ),
                        self._resolve_non_nullish_fallback_operand(
                            right_expression,
                            index,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        ),
                    )
                    if item is not None
                )
                resolved = (
                    operands[0]
                    if len(operands) == 1
                    else _merge_resolved(expr, operands)
                    if operands
                    else ResolvedValue(
                        expr,
                        "unknown",
                        unresolved_reason="fallback has no effective value source",
                    )
                )
            return _prepend_resolution(
                expr,
                "resolve logical fallback",
                resolved,
                step,
            )

        if mode == "origin":
            concatenation = _top_level_c_style_concatenation(expr)
            if concatenation is not None:
                left_expression, right_expression = concatenation
                resolved = _merge_transformed_origins(
                    expr,
                    (
                        self._resolve(
                            left_expression,
                            index,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        ),
                        self._resolve(
                            right_expression,
                            index,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        ),
                    ),
                )
                return _prepend_resolution(
                    expr,
                    "resolve deterministic concatenation inputs",
                    resolved,
                    step,
                )

            arithmetic_operands = _top_level_c_style_arithmetic_operands(expr)
            if arithmetic_operands is not None:
                resolved = _merge_transformed_origins(
                    expr,
                    (
                        self._resolve(
                            operand,
                            index,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        )
                        for operand in arithmetic_operands
                    ),
                )
                return _prepend_resolution(
                    expr,
                    "resolve deterministic arithmetic inputs",
                    resolved,
                    step,
                )

            hmac_inputs = (
                _typescript_hmac_digest_inputs(expr)
                if self.language == "typescript"
                else None
            )
            if hmac_inputs is not None:
                resolved = _merge_transformed_origins(
                    expr,
                    (
                        self._resolve(
                            value,
                            index,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        )
                        for value in hmac_inputs
                    ),
                )
                return _prepend_resolution(
                    expr,
                    "resolve HMAC derivation inputs",
                    resolved,
                    step,
                )

            hash_inputs = (
                _typescript_hash_digest_inputs(expr)
                if self.language == "typescript"
                else None
            )
            if hash_inputs is not None:
                resolved = _merge_transformed_origins(
                    expr,
                    (
                        self._resolve(
                            value,
                            index,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        )
                        for value in hash_inputs
                    ),
                )
                return _prepend_resolution(
                    expr,
                    "resolve hash derivation inputs",
                    resolved,
                    step,
                )

            stateful_digest_inputs = (
                self._typescript_stateful_digest_inputs(expr, index)
                if self.language == "typescript"
                else None
            )
            if stateful_digest_inputs is not None:
                resolved = _merge_transformed_origins(
                    expr,
                    (
                        self._resolve(
                            value,
                            index,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        )
                        for value in stateful_digest_inputs
                    ),
                )
                return _prepend_resolution(
                    expr,
                    "resolve stateful digest inputs",
                    resolved,
                    step,
                )

            serialized_slice = (
                self._resolve_typescript_buffer_slice(
                    expr,
                    index,
                    seen=seen,
                    bindings=bindings,
                )
                if self.language == "typescript"
                else None
            )
            if serialized_slice is not None:
                return _prepend_resolution(
                    expr,
                    "resolve selected serialized buffer range",
                    serialized_slice,
                    step,
                )

        receiver_wrapper = _origin_preserving_receiver_wrapper(expr)
        if receiver_wrapper is not None:
            resolved = self._resolve(
                receiver_wrapper,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            return _prepend_resolution(
                expr,
                "unwrap receiver conversion",
                resolved,
                step,
            )

        wrapper = _origin_preserving_call_argument(expr)
        if wrapper is None:
            wrapper = _single_argument_wrapper(expr)
        if wrapper is not None:
            resolved = self._resolve(
                wrapper,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            return _prepend_resolution(expr, "unwrap conversion", resolved, step)

        property_wrapper = _origin_preserving_property_receiver(expr)
        if property_wrapper is not None:
            resolved = self._resolve(
                property_wrapper,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            return _prepend_resolution(
                expr,
                "unwrap receiver property view",
                resolved,
                step,
            )

        hash_algorithm_name_receiver = (
            _csharp_hash_algorithm_name_property_receiver(
                expr,
                self.source,
                index,
            )
            if self.language == "csharp" and mode == "string"
            else None
        )
        if hash_algorithm_name_receiver is not None:
            resolved = self._resolve(
                hash_algorithm_name_receiver,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            return _prepend_resolution(
                expr,
                "unwrap HashAlgorithmName.Name",
                resolved,
                step,
            )

        call_result_property = _c_style_call_result_property(expr)
        if call_result_property is not None:
            owner_expression, property_name = call_result_property
            resolved = self._resolve_object_property_expression(
                owner_expression,
                property_name,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            if resolved is not None:
                return _prepend_resolution(
                    expr,
                    "resolve call-result property",
                    resolved,
                    step,
                )

        object_property = self._resolve_object_property(
            expr,
            index,
            mode=mode,
            seen=seen,
            bindings=bindings,
        )
        if object_property is not None:
            return _prepend_resolution(
                expr,
                "resolve object property",
                object_property,
                step,
            )

        provider_property = self._resolve_csharp_provider_property(
            expr,
            index,
            mode=mode,
            seen=seen,
            bindings=bindings,
        )
        if provider_property is not None:
            return provider_property

        reference = _simple_reference(expr)
        if reference is not None:
            containing = self._containing_function(index)
            scope = containing.start if containing is not None else "global"
            reference_identity = "".join(reference.split())
            key = f"variable:{scope}:{reference_identity}"
            if key in seen:
                prior_assignment = self.latest_assignment(reference, index)
                if prior_assignment is not None and prior_assignment.start < index:
                    resolved_prior = self._resolve(
                        prior_assignment.expression,
                        prior_assignment.start,
                        mode=mode,
                        seen=seen,
                        bindings=bindings,
                    )
                    return _prepend_resolution(
                        expr,
                        "resolve prior assignment during serialization round trip",
                        resolved_prior,
                        step,
                    )
                return ResolvedValue(
                    expr,
                    "unknown",
                    steps=(step,),
                    unresolved_reason="cyclic assignment",
                )
            if mode == "string":
                branch_values = self._enclosing_string_equality_guard_values(
                    reference,
                    index,
                )
                if branch_values:
                    return ResolvedValue(
                        expr,
                        "constant",
                        branch_values,
                        (
                            step,
                            ResolutionStep(
                                reference,
                                "resolve enclosing string equality branch",
                                "constant",
                                self.source_span(index, index + len(reference)),
                            ),
                        ),
                    )
                guarded_values = self._fail_closed_string_guard_values(
                    reference,
                    index,
                )
                if guarded_values:
                    caller_value = self._resolve_parameter(
                        reference,
                        index,
                        mode=mode,
                        seen=seen,
                        bindings=bindings,
                    )
                    if (
                        caller_value is not None
                        and caller_value.origin != "unknown"
                    ):
                        return _prepend_resolution(
                            expr,
                            "resolve fixed caller before fail-closed whitelist",
                            caller_value,
                            step,
                        )
                    return ResolvedValue(
                        expr,
                        "constant",
                        guarded_values,
                        (
                            step,
                            ResolutionStep(
                                reference,
                                "resolve fail-closed string whitelist",
                                "constant",
                                self.source_span(index, index + len(reference)),
                            ),
                        ),
                    )
            callback_value = self._resolve_typescript_callback_parameter(
                reference,
                index,
            )
            if callback_value is not None:
                return _prepend_resolution(
                    expr,
                    "resolve crypto callback output",
                    callback_value,
                    step,
                )
            promise_value = self._resolve_typescript_promise_callback_parameter(
                reference,
                index,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if promise_value is not None:
                return _prepend_resolution(
                    expr,
                    "resolve Promise callback value",
                    promise_value,
                    step,
                )
            iife_value = self._resolve_typescript_iife_parameter(
                reference,
                index,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if iife_value is not None:
                return _prepend_resolution(
                    expr,
                    "resolve immediately-invoked arrow argument",
                    iife_value,
                    step,
                )
            closure_value = self._resolve_typescript_returned_closure_parameter(
                reference,
                index,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if closure_value is not None:
                return _prepend_resolution(
                    expr,
                    "resolve returned-arrow closure parameter",
                    closure_value,
                    step,
                )
            foreach_value = self._resolve_csharp_foreach_binding(
                reference,
                index,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if foreach_value is not None:
                return _prepend_resolution(
                    expr,
                    "resolve C# foreach binding",
                    foreach_value,
                    step,
                )
            assignment = self.latest_assignment(reference, index)
            loop_value = (
                self._resolve_typescript_for_of_binding(
                    reference,
                    index,
                    seen=seen | frozenset({key}),
                    bindings=bindings,
                )
                if mode == "origin" and assignment is None
                else None
            )
            if loop_value is not None:
                return _prepend_resolution(
                    expr,
                    "resolve for-of iterable origin",
                    loop_value,
                    step,
                )
            mutation = (
                self.latest_mutation_origin(
                    reference,
                    index,
                    seen=seen | frozenset({key}),
                    bindings=bindings,
                )
                if mode == "origin"
                else None
            )
            indexed_mutation = (
                self.latest_indexed_assignment_origin(
                    reference,
                    index,
                    seen=seen | frozenset({key}),
                    bindings=bindings,
                )
                if mode == "origin"
                else None
            )
            persisted_field = (
                self._resolve_csharp_persisted_envelope_field(
                    reference,
                    assignment,
                    index,
                    mode=mode,
                    seen=seen | frozenset({key}),
                    bindings=bindings,
                )
                if assignment is not None
                else None
            )
            if persisted_field is not None:
                return _prepend_resolution(
                    expr,
                    f"resolve persisted envelope field {reference}",
                    persisted_field,
                    step,
                )
            if (
                indexed_mutation is not None
                and (
                    mutation is None
                    or indexed_mutation[0] > mutation[0].start
                )
                and (
                    assignment is None
                    or indexed_mutation[0] > assignment.start
                )
            ):
                return _prepend_resolution(
                    expr,
                    "resolve later indexed buffer write",
                    indexed_mutation[1],
                    step,
                )
            if mutation is not None and (
                assignment is None or mutation[0].start > assignment.start
            ):
                call, resolved = mutation
                return _prepend_resolution(
                    expr,
                    f"resolve later buffer write {call.callee}",
                    resolved,
                    step,
                )
            if assignment is not None:
                if mode == "string":
                    allowlist = self._guarded_allowlist_values(reference, index)
                    if allowlist:
                        return ResolvedValue(
                            expr,
                            "constant",
                            allowlist,
                            (
                                step,
                                ResolutionStep(
                                    expr,
                                    "restrict value to terminating allowlist guard",
                                    "constant",
                                    self.source_span(index, index),
                                ),
                            ),
                        )
                alternatives = self.reaching_assignment_alternatives(
                    reference,
                    index,
                )
                if alternatives:
                    assignment = alternatives[-1]
                if len(alternatives) > 1:
                    exclude_empty = mode == "origin" and (
                        self._fail_closed_nonempty_guard(reference, index)
                        or self._reference_is_truthy_on_path(reference, index)
                    )
                    resolved_alternatives: list[ResolvedValue] = []
                    for candidate in alternatives:
                        candidate_value = None
                        if exclude_empty:
                            candidate_value = self._resolve_origin_without_empty_paths(
                                candidate.expression,
                                candidate.start,
                                seen=seen | frozenset({key}),
                                bindings=bindings,
                            )
                        else:
                            candidate_value = self._resolve(
                                candidate.expression,
                                candidate.start,
                                mode=mode,
                                seen=seen | frozenset({key}),
                                bindings=bindings,
                            )
                        if candidate_value is not None:
                            resolved_alternatives.append(candidate_value)
                    if not resolved_alternatives:
                        resolved_alternatives = [
                            self._resolve(
                                candidate.expression,
                                candidate.start,
                                mode=mode,
                                seen=seen | frozenset({key}),
                                bindings=bindings,
                            )
                            for candidate in alternatives
                        ]
                    resolved = _merge_resolved(expr, resolved_alternatives)
                    return _prepend_resolution(
                        expr,
                        "merge reachable conditional assignments",
                        resolved,
                        step,
                    )
                resolved = None
                if mode == "origin" and (
                    self._fail_closed_nonempty_guard(reference, index)
                    or self._reference_is_truthy_on_path(reference, index)
                ):
                    resolved = self._resolve_origin_without_empty_paths(
                        assignment.expression,
                        assignment.start,
                        seen=seen | frozenset({key}),
                        bindings=bindings,
                    )
                if resolved is None:
                    resolved = self._resolve(
                        assignment.expression,
                        assignment.start,
                        mode=mode,
                        seen=seen | frozenset({key}),
                        bindings=bindings,
                    )
                return _prepend_resolution(
                    expr,
                    f"resolve assignment {assignment.name}",
                    resolved,
                    step,
                )
            out_parameter = self._resolve_csharp_out_argument(
                reference,
                index,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if out_parameter is not None:
                return _prepend_resolution(
                    expr,
                    "resolve C# out argument",
                    out_parameter,
                    step,
                )
            destructured_parameter = self._resolve_destructured_parameter(
                reference,
                index,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if destructured_parameter is not None:
                return _prepend_resolution(
                    expr,
                    "resolve destructured caller argument",
                    destructured_parameter,
                    step,
                )
            parameter_values = self._resolve_parameter(
                reference,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            if parameter_values is not None:
                return _prepend_resolution(expr, "resolve caller arguments", parameter_values, step)
            instance_field = self._resolve_instance_field(
                reference,
                index,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if instance_field is not None:
                return _prepend_resolution(
                    expr,
                    f"resolve instance field {reference}",
                    instance_field,
                    step,
                )

        called = _simple_call(expr)
        inline_identity_argument = self._typescript_inline_identity_argument(
            expr,
            index,
        )
        if inline_identity_argument is not None:
            resolved = self._resolve(
                inline_identity_argument,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            return _prepend_resolution(
                expr,
                "unwrap immediately invoked identity factory",
                resolved,
                step,
            )
        if called is not None:
            call_name, call_arguments = called
            if (
                self.language == "csharp"
                and mode == "string"
                and call_name == "Parse"
                and re.match(
                    r"^(?:\([^()]+\)\s*)?(?:System\s*\.\s*)?Enum\s*\.\s*Parse\s*\(",
                    expr,
                )
                and len(call_arguments) >= 2
            ):
                resolved = self._resolve(
                    call_arguments[1],
                    index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
                return _prepend_resolution(
                    expr,
                    "resolve Enum.Parse value",
                    resolved,
                    step,
                )
            selected_method_group = self._resolve_csharp_selected_method_group_call(
                call_name,
                call_arguments,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            if selected_method_group is not None:
                return _prepend_resolution(
                    expr,
                    f"resolve selected method group {call_name}",
                    selected_method_group,
                    step,
                )
            returned_delegate = self._resolve_csharp_returned_delegate_call(
                call_name,
                call_arguments,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            if returned_delegate is not None:
                return _prepend_resolution(
                    expr,
                    f"resolve returned delegate {call_name}",
                    returned_delegate,
                    step,
                )
            callable_parameter = self._resolve_callable_parameter_call(
                call_name,
                call_arguments,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            if callable_parameter is not None:
                return _prepend_resolution(
                    expr,
                    f"resolve callable parameter {call_name}",
                    callable_parameter,
                    step,
                )
            identity_argument = self._typescript_returned_identity_argument(
                call_name,
                call_arguments,
                index,
            )
            if identity_argument is not None:
                resolved = self._resolve(
                    identity_argument,
                    index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
                return _prepend_resolution(
                    expr,
                    f"unwrap returned identity callable {call_name}",
                    resolved,
                    step,
                )
            preserving_argument = self._csharp_byte_serialization_argument(
                call_name,
                call_arguments,
            )
            if preserving_argument is None:
                preserving_argument = self._csharp_byte_decoder_argument(
                    call_name,
                    call_arguments,
                )
            if preserving_argument is not None and mode == "origin":
                resolved = self._resolve(
                    preserving_argument,
                    index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
                return _prepend_resolution(
                    expr,
                    f"unwrap byte serialization helper {call_name}",
                    resolved,
                    step,
                )
            call_identity = re.sub(r"\s+", "", expr)
            # The same helper expression can occur in multiple functions with
            # different parameter bindings (for example, Copy(keyBytes) in
            # both a generator and a key-rebuild helper).  Keep the lexical
            # callsite in the recursion key so one path does not suppress the
            # other while still terminating true recursive calls.
            key = f"function:{call_name}:{index}:{call_identity}"
            if key not in seen:
                definitions = self.function(call_name)
                resolved_returns: list[ResolvedValue] = []
                for definition in definitions:
                    if (
                        mode == "origin"
                        and self._typescript_runtime_input_helper(definition)
                        and not self._typescript_runtime_input_helper_has_predictable_fallback(
                            definition
                        )
                    ):
                        resolved_returns.append(
                            ResolvedValue(
                                expr,
                                "runtime_input",
                                steps=(
                                    ResolutionStep(
                                        expr,
                                        f"resolve runtime-input helper {call_name}",
                                        "runtime_input",
                                        self.source_span(
                                            definition.start,
                                            definition.end,
                                        ),
                                    ),
                                ),
                            )
                        )
                        continue
                    if (
                        mode == "origin"
                        and self._csharp_console_input_helper(definition)
                    ):
                        resolved_returns.append(
                            ResolvedValue(
                                expr,
                                "runtime_input",
                                steps=(
                                    ResolutionStep(
                                        expr,
                                        f"resolve console-input helper {call_name}",
                                        "runtime_input",
                                        self.source_span(
                                            definition.start,
                                            definition.end,
                                        ),
                                    ),
                                ),
                            )
                        )
                        continue
                    if (
                        mode == "origin"
                        and self._csharp_secure_random_string_helper(definition)
                    ):
                        resolved_returns.append(
                            ResolvedValue(
                                expr,
                                "secure_random",
                                steps=(
                                    ResolutionStep(
                                        expr,
                                        f"resolve secure-random output helper {call_name}",
                                        "secure_random",
                                        self.source_span(
                                            definition.start,
                                            definition.end,
                                        ),
                                    ),
                                ),
                            )
                        )
                        continue
                    call_bindings = dict(bindings or {})
                    call_bindings.update(
                        {
                            (definition.start, parameter): (argument, index)
                            for parameter, argument in _bind_c_style_arguments(
                                definition.parameters,
                                call_arguments,
                            )
                        }
                    )
                    for returned in definition.returns:
                        if self._containing_function(returned.index) is not definition:
                            continue
                        # The helper is already reached through the concrete
                        # call being resolved. Only its local branch needs a
                        # reachability proof here; requiring another global
                        # caller proof loses selected method groups and
                        # higher-order callbacks.
                        if not _c_style_direct_path_is_reachable(
                            self,
                            returned.index,
                        ):
                            continue
                        if mode == "origin" and _c_style_expression_never_returns(
                            returned.expression,
                            self.language,
                        ):
                            continue
                        returned_reference = _simple_reference(returned.expression)
                        resolved_return = None
                        if (
                            mode == "origin"
                            and returned_reference is not None
                            and (
                                self._fail_closed_nonempty_guard(
                                    returned_reference,
                                    returned.index,
                                )
                                or self._reference_is_truthy_on_path(
                                    returned_reference,
                                    returned.index,
                                )
                            )
                        ):
                            resolved_return = self._resolve_origin_without_empty_paths(
                                returned.expression,
                                returned.index,
                                seen=seen | frozenset({key}),
                                bindings=call_bindings,
                            )
                        if resolved_return is None:
                            resolved_return = self._resolve(
                                returned.expression,
                                returned.index,
                                mode=mode,
                                seen=seen | frozenset({key}),
                                bindings=call_bindings,
                            )
                        resolved_returns.append(resolved_return)
                if resolved_returns:
                    merged = _merge_resolved(expr, resolved_returns)
                    return _prepend_resolution(expr, f"resolve helper {call_name}", merged, step)

        if mode == "int" and _simple_call(expr) is not None:
            return ResolvedValue(
                expr,
                "unknown",
                steps=(step,),
                unresolved_reason="unrecognized numeric call result",
            )

        if mode in {"string", "int"}:
            literals = _literal_values_in_expression(expr, mode)
            if literals:
                origin: ValueOrigin = "constant" if len(literals) == 1 else "unknown"
                return ResolvedValue(expr, origin, tuple(sorted(literals)), (step,))

        return ResolvedValue(
            expr,
            "unknown",
            steps=(step,),
            unresolved_reason="expression origin could not be resolved",
        )

    def _resolve_csharp_out_argument(
        self,
        reference: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        if self.language != "csharp":
            return None
        containing = self._containing_function(index)
        values: list[ResolvedValue] = []
        for call in self._calls:
            if call.end > index or self._containing_function(call.start) != containing:
                continue
            for argument_index, argument in enumerate(call.arguments):
                out_match = re.fullmatch(
                    rf"\s*(?:out|ref)\s+(?:[^,()]+?\s+)?"
                    rf"{re.escape(reference)}\s*",
                    argument,
                )
                if out_match is None:
                    continue
                if mode == "int" and call.simple_name in {
                    "TryParse",
                    "tryParse",
                } and call.arguments:
                    input_expression = call.arguments[0]
                    input_origin = self._resolve(
                        input_expression,
                        call.start,
                        mode="origin",
                        seen=seen,
                        bindings=bindings,
                    )
                    if input_origin.origin == "runtime_input" or re.search(
                        r"(?:\bargs\s*\[|\bprocess\s*\.\s*argv\s*\[)",
                        input_expression,
                    ):
                        values.append(
                            ResolvedValue(
                                reference,
                                "runtime_input",
                                steps=(
                                    *input_origin.steps,
                                    ResolutionStep(
                                        input_expression,
                                        "resolve parsed runtime integer",
                                        "runtime_input",
                                        self.source_span(call.start, call.end),
                                    ),
                                ),
                            )
                        )
                        continue
                for definition in self.function(call.simple_name):
                    if argument_index >= len(definition.parameters):
                        continue
                    parameter = definition.parameters[argument_index]
                    call_bindings = dict(bindings or {})
                    call_bindings.update(
                        {
                            (definition.start, name): (
                                re.sub(r"^\s*(?:out|ref|in)\s+", "", value),
                                call.start,
                            )
                            for name, value in _bind_c_style_arguments(
                                definition.parameters,
                                call.arguments,
                            )
                        }
                    )
                    assignments = [
                        candidate
                        for candidate in self.assignments
                        if _normalize_reference(candidate.name)
                        == _normalize_reference(parameter)
                        and self._assignment_scopes[candidate] == definition
                    ]
                    if not assignments:
                        continue
                    exclude_empty = mode == "origin" and (
                        self._fail_closed_nonempty_guard(reference, index)
                        or self._reference_is_truthy_on_path(reference, index)
                    )
                    resolved_assignments: list[ResolvedValue] = []
                    for assignment in assignments:
                        persisted = self._resolve_csharp_persisted_envelope_field(
                            parameter,
                            assignment,
                            assignment.end,
                            mode=mode,
                            seen=seen,
                            bindings=call_bindings,
                        )
                        resolved = persisted
                        if resolved is None and exclude_empty:
                            resolved = self._resolve_origin_without_empty_paths(
                                assignment.expression,
                                assignment.start,
                                seen=seen,
                                bindings=call_bindings,
                            )
                            if resolved is None:
                                continue
                        elif resolved is None:
                            resolved = self._resolve(
                                assignment.expression,
                                assignment.start,
                                mode=mode,
                                seen=seen,
                                bindings=call_bindings,
                            )
                        if resolved is not None:
                            resolved_assignments.append(resolved)
                    if resolved_assignments:
                        values.append(
                            resolved_assignments[0]
                            if len(resolved_assignments) == 1
                            else _merge_resolved(reference, resolved_assignments)
                        )
        return _merge_resolved(reference, values) if values else None

    def _resolve_csharp_persisted_envelope_field(
        self,
        reference: str,
        assignment: Assignment,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        if self.language != "csharp":
            return None
        role = _normalize_reference(reference).rsplit(".", 1)[-1].lower()
        if role not in {"salt", "iterations", "iterationcount"}:
            return None
        if role == "salt" and mode != "origin":
            return None
        if role != "salt" and mode != "int":
            return None

        read_expression = re.sub(r"\s+", "", assignment.expression)
        buffer_match = re.search(
            rf"(?P<buffer>{_IDENTIFIER})\.AsSpan\(",
            read_expression,
        )
        buffer_name = (
            buffer_match.group("buffer")
            if buffer_match is not None
            else self._csharp_binary_reader_input(assignment)
        )
        if buffer_name is None:
            return None
        containing = self._containing_function(assignment.start)
        if containing is None:
            return None
        if buffer_name not in containing.parameters:
            buffer_assignment = self.latest_assignment(
                buffer_name,
                assignment.start,
            )
            if buffer_assignment is None:
                return None
            aliased_buffer = _simple_reference(
                _strip_expression(buffer_assignment.expression)
            )
            if aliased_buffer not in containing.parameters:
                return None
            buffer_name = aliased_buffer
        buffer_parameter_index = containing.parameters.index(buffer_name)

        consumer_arguments: list[tuple[str, int]] = []
        if bindings is not None:
            bound = bindings.get((containing.start, buffer_name))
            if bound is not None:
                consumer_arguments.append(bound)
        values: list[ResolvedValue] = []
        for consumer_call in self.calls(containing.name):
            if containing.start <= consumer_call.start <= containing.end:
                continue
            if not self._call_can_target_function(consumer_call, containing):
                continue
            buffer_argument = self._c_style_argument_for_parameter(
                consumer_call,
                buffer_parameter_index,
                buffer_name,
            )
            if buffer_argument is None:
                continue
            consumer_arguments.append((buffer_argument, consumer_call.start))

        for buffer_argument, consumer_index in dict.fromkeys(consumer_arguments):
            producer_expression = _strip_expression(buffer_argument)
            producer_reference = _simple_reference(producer_expression)
            if producer_reference is not None:
                producer_assignment = self.latest_assignment(
                    producer_reference,
                    consumer_index,
                )
                if producer_assignment is None:
                    continue
                producer_expression = _strip_expression(
                    producer_assignment.expression
                )
                producer_index = producer_assignment.start
            else:
                producer_index = consumer_index
            producer_call = _simple_call(producer_expression)
            if producer_call is None:
                continue
            producer_name, producer_arguments = producer_call
            for producer in self.function(producer_name):
                role_parameter = next(
                    (
                        parameter
                        for parameter in producer.parameters
                        if _normalize_reference(parameter).lower() == role
                        or (
                            role == "iterationcount"
                            and _normalize_reference(parameter).lower()
                            == "iterations"
                        )
                    ),
                    None,
                )
                if role_parameter is None:
                    continue
                if not self._csharp_envelope_producer_writes_role(
                    producer,
                    role_parameter,
                    role,
                ):
                    continue
                parameter_index = producer.parameters.index(role_parameter)
                if parameter_index >= len(producer_arguments):
                    continue
                argument = _c_style_argument_value(
                    producer_arguments,
                    parameter_index,
                    role_parameter,
                )
                values.append(
                    self._resolve(
                        argument,
                        producer_index,
                        mode=mode,
                        seen=seen,
                        bindings=bindings,
                    )
                )
        return _merge_resolved(reference, values) if values else None

    def _resolve_csharp_binary_integer_reader(
        self,
        expression: str,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        if self.language != "csharp":
            return None
        match = re.fullmatch(
            r"(?:System\s*\.\s*Buffers\s*\.\s*Binary\s*\.\s*)?"
            r"BinaryPrimitives\s*\.\s*ReadInt32(?P<endian>Big|Little)Endian\s*"
            r"\(.*\)",
            expression,
            flags=re.DOTALL,
        )
        if match is None:
            return None
        writer_name = f"WriteInt32{match.group('endian')}Endian"
        values: list[ResolvedValue] = []
        for call in self.calls(writer_name):
            if len(call.arguments) < 2:
                continue
            values.append(
                self._resolve(
                    call.arguments[1],
                    call.start,
                    mode="int",
                    seen=seen,
                    bindings=bindings,
                )
            )
        return _merge_resolved(expression, values) if values else None

    def _csharp_binary_reader_input(
        self,
        assignment: Assignment,
    ) -> str | None:
        expression = _strip_expression(assignment.expression)
        read_match = re.fullmatch(
            rf"(?P<reader>{_IDENTIFIER})\s*\.\s*"
            r"Read(?:Bytes|Int32|Int64|UInt32|UInt64)\s*\(.*\)",
            expression,
            flags=re.DOTALL,
        )
        if read_match is None:
            return None
        reader_assignment = self.latest_assignment(
            read_match.group("reader"),
            assignment.start,
        )
        if reader_assignment is None:
            return None
        reader_arguments = _csharp_constructor_arguments(
            reader_assignment.expression,
            "BinaryReader",
        )
        if not reader_arguments:
            return None
        stream_expression = _strip_expression(reader_arguments[0])
        stream_reference = _simple_reference(stream_expression)
        if stream_reference is not None:
            stream_assignment = self.latest_assignment(
                stream_reference,
                reader_assignment.start,
            )
            if stream_assignment is None:
                return None
            stream_expression = _strip_expression(stream_assignment.expression)
        stream_arguments = _csharp_constructor_arguments(
            stream_expression,
            "MemoryStream",
        )
        if not stream_arguments:
            return None
        return _simple_reference(stream_arguments[0])

    def _csharp_envelope_producer_writes_role(
        self,
        producer: FunctionDefinition,
        parameter: str,
        role: str,
    ) -> bool:
        body = self.source[producer.start : producer.end]
        returned_buffers: set[str] = set()
        for returned in producer.returns:
            returned_reference = _simple_reference(returned.expression)
            if returned_reference is None:
                returned_reference = _origin_preserving_receiver_wrapper(
                    returned.expression
                )
            if returned_reference is not None:
                returned_buffers.add(_normalize_reference(returned_reference))
        if not returned_buffers:
            return False
        parameter_pattern = re.escape(parameter)
        if role == "salt":
            writes = re.finditer(
                rf"\b{parameter_pattern}\s*\.\s*CopyTo\s*\(\s*"
                rf"(?P<buffer>{_IDENTIFIER})\b",
                body,
            )
        else:
            writes = re.finditer(
                rf"\bWriteInt(?:32|64)(?:Big|Little)Endian\s*\(\s*"
                rf"(?P<buffer>{_IDENTIFIER})\s*\.\s*AsSpan\s*\([^)]*\)\s*,\s*"
                rf"{parameter_pattern}\b",
                body,
            )
        if any(
            _normalize_reference(match.group("buffer")) in returned_buffers
            for match in writes
        ):
            return True
        if role == "salt" and any(
            _normalize_reference(match.group("buffer")) in returned_buffers
            for match in re.finditer(
                rf"\bBuffer\s*\.\s*BlockCopy\s*\(\s*"
                rf"{parameter_pattern}\s*,[^,]+,\s*"
                rf"(?P<buffer>{_IDENTIFIER})\b",
                body,
                flags=re.DOTALL,
            )
        ):
            return True

        writer_names: set[str] = set()
        for assignment in self.assignments:
            if self._assignment_scopes[assignment] != producer:
                continue
            arguments = _csharp_constructor_arguments(
                assignment.expression,
                "BinaryWriter",
            )
            if not arguments:
                continue
            stream = _simple_reference(arguments[0])
            if (
                stream is not None
                and _normalize_reference(stream) in returned_buffers
            ):
                writer_names.add(_normalize_reference(assignment.name))
        return any(
            re.search(
                rf"\b{re.escape(writer)}\s*\.\s*Write\s*\(\s*"
                rf"{parameter_pattern}\s*\)",
                body,
            )
            is not None
            for writer in writer_names
        )

    def _fail_closed_string_guard_values(
        self,
        reference: str,
        index: int,
    ) -> tuple[str, ...]:
        containing = self._containing_function(index)
        start = containing.start if containing is not None else 0
        prefix = self._normalize_guard_terminators(self.source[start:index])
        reference_pattern = r"\s*\.\s*".join(
            re.escape(part) for part in re.sub(r"\s+", "", reference).split(".")
        )
        literal = r"(?P<literal>(?P<quote>[\"'])(?:\\.|(?!\2).)*(?P=quote))"
        patterns = (
            rf"\bif\s*\(\s*{reference_pattern}\s*!==?\s*{literal}\s*\)"
            r"\s*(?:\{\s*)?throw\b",
            rf"\bif\s*\(\s*{literal}\s*!==?\s*{reference_pattern}\s*\)"
            r"\s*(?:\{\s*)?throw\b",
        )
        matches = [
            match
            for pattern in patterns
            for match in re.finditer(pattern, prefix, flags=re.DOTALL)
        ]
        guarded: list[tuple[int, tuple[str, ...]]] = []
        for match in matches:
            value = _string_literal(match.group("literal"))
            if value is not None:
                guarded.append((match.start(), (value,)))

        # A conjunction of negative comparisons followed by an immediate
        # throw is a finite allow-list: execution can continue only when the
        # value equals one of the listed literals.
        compound = re.compile(
            r"\bif\s*\(\s*(?P<condition>[^()]*)\s*\)"
            r"\s*(?:\{\s*)?throw\b",
            flags=re.DOTALL,
        )
        comparison_patterns = (
            rf"^\s*{reference_pattern}\s*!==?\s*{literal}\s*$",
            rf"^\s*{literal}\s*!==?\s*{reference_pattern}\s*$",
        )
        for match in compound.finditer(prefix):
            terms = re.split(r"\s*&&\s*", match.group("condition"))
            if len(terms) < 2:
                continue
            values: list[str] = []
            for term in terms:
                comparison = next(
                    (
                        candidate
                        for pattern in comparison_patterns
                        if (candidate := re.fullmatch(pattern, term, flags=re.DOTALL))
                        is not None
                    ),
                    None,
                )
                if comparison is None:
                    values = []
                    break
                value = _string_literal(comparison.group("literal"))
                if value is None:
                    values = []
                    break
                values.append(value)
            if values:
                guarded.append((match.start(), tuple(dict.fromkeys(values))))

        if not guarded:
            return ()
        return max(guarded, key=lambda item: item[0])[1]

    def _enclosing_string_equality_guard_values(
        self,
        reference: str,
        index: int,
    ) -> tuple[str, ...]:
        """Return a literal proven by an enclosing equality branch."""

        containing = self._containing_function(index)
        start = containing.start if containing is not None else 0
        end = containing.end if containing is not None else len(self.source)
        reference_pattern = r"\s*\.\s*".join(
            re.escape(part)
            for part in re.sub(r"\s+", "", reference).split(".")
        )
        literal = r"(?P<literal>(?P<quote>[\"'])(?:\\.|(?!\2).)*(?P=quote))"
        patterns = (
            rf"\bif\s*\(\s*{reference_pattern}\s*===?\s*{literal}\s*\)",
            rf"\bif\s*\(\s*{literal}\s*===?\s*{reference_pattern}\s*\)",
        )
        matches: list[tuple[int, str]] = []
        for pattern in patterns:
            for match in re.finditer(
                pattern,
                self.source[start:end],
                flags=re.DOTALL,
            ):
                body_start = start + match.end()
                while (
                    body_start < end
                    and self.masked[body_start].isspace()
                ):
                    body_start += 1
                if body_start >= index:
                    continue
                if body_start < end and self.masked[body_start] == "{":
                    body_end = _find_matching_masked(
                        self.masked,
                        body_start,
                        "{",
                        "}",
                    )
                else:
                    body_end = _next_c_style_statement_end_masked(
                        self.masked,
                        body_start,
                        self.language,
                    )
                if body_end == -1 or not (body_start < index < body_end):
                    continue
                value = _string_literal(match.group("literal"))
                if value is not None:
                    matches.append((body_start, value))
        if not matches:
            return ()
        return (max(matches, key=lambda item: item[0])[1],)

    def _fail_closed_nonempty_guard(self, reference: str, index: int) -> bool:
        containing = self._containing_function(index)
        start = containing.start if containing is not None else 0
        prefix = self._normalize_guard_terminators(self.source[start:index])
        value = r"\s*\.\s*".join(
            re.escape(part)
            for part in re.sub(r"\s+", "", reference).split(".")
        )
        conditions = (
            rf"{value}\s*\.\s*length\s*===?\s*0",
            rf"{value}\s*\.\s*length\s*<=?\s*[1-9][0-9]*",
            rf"!\s*{value}",
            rf"{value}\s*===?\s*[\"'][\"']",
            rf"[\"'][\"']\s*===?\s*{value}",
        )
        return any(
            re.search(
                rf"\bif\s*\(\s*(?:{condition})\s*\)\s*"
                rf"(?:throw\b|\{{[^{{}}]*\bthrow\b[^{{}}]*\}})",
                prefix,
                flags=re.DOTALL,
            )
            is not None
            for condition in conditions
        )

    def _reference_is_truthy_on_path(self, reference: str, index: int) -> bool:
        containing = self._containing_function(index)
        start = containing.start if containing is not None else 0
        end = containing.end if containing is not None else len(self.source)
        compact_reference = r"\s*\.\s*".join(
            re.escape(part)
            for part in re.sub(r"\s+", "", reference).split(".")
        )
        proof = re.compile(
            rf"^\s*(?:!!\s*)?{compact_reference}\s*(?:$|&&|"
            r"!==?\s*(?:undefined|null)\b|is\s+not\s+null\b|"
            r"\.\s*length\s*>\s*0\b)",
            flags=re.DOTALL,
        )
        csharp_nonempty = re.compile(
            rf"!\s*(?:string|String)\s*\.\s*"
            rf"IsNullOr(?:Empty|WhiteSpace)\s*\(\s*{compact_reference}\s*\)",
            flags=re.DOTALL,
        )

        def proves_truthy(condition: str) -> bool:
            return (
                proof.search(condition) is not None
                or csharp_nonempty.search(condition) is not None
            )

        for match in re.finditer(r"\bif\s*\(", self.masked[start:end]):
            opening = start + self.masked[start:end].find(
                "(",
                match.start(),
                match.end(),
            )
            closing = _find_matching_masked(self.masked, opening, "(", ")")
            if closing == -1 or closing >= index:
                continue
            cursor = closing + 1
            while cursor < len(self.masked) and self.masked[cursor].isspace():
                cursor += 1
            if cursor >= len(self.masked):
                continue
            condition = self.source[opening + 1 : closing]
            if self.masked[cursor] == "{":
                body_end = _find_matching_masked(self.masked, cursor, "{", "}")
                if body_end == -1 or not cursor < index < body_end:
                    continue
            else:
                statement_end = _next_c_style_statement_end_masked(
                    self.masked,
                    cursor,
                    self.language,
                )
                if statement_end == -1 or not cursor <= index <= statement_end:
                    continue
            if proves_truthy(condition):
                return True
        return False

    def _resolve_origin_without_empty_paths(
        self,
        expression: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        expr = _strip_expression(expression)
        if expr in {"null", "undefined"} or _string_literal(expr) == "":
            return None
        if self.language == "typescript" and (
            re.fullmatch(r"Buffer\s*\.\s*alloc\s*\(\s*0\s*\)", expr)
            or re.fullmatch(
                r"Buffer\s*\.\s*from\s*\(\s*([\"'])\1\s*(?:,[^)]*)?\)",
                expr,
                flags=re.DOTALL,
            )
        ):
            return None
        if _c_style_expression_never_returns(expr, self.language):
            return None
        wrapper = _origin_preserving_call_argument(expr)
        if wrapper is None:
            wrapper = _single_argument_wrapper(expr)
        if wrapper is not None:
            return self._resolve_origin_without_empty_paths(
                wrapper,
                index,
                seen=seen,
                bindings=bindings,
            )
        receiver = _origin_preserving_receiver_wrapper(expr)
        if receiver is not None:
            return self._resolve_origin_without_empty_paths(
                receiver,
                index,
                seen=seen,
                bindings=bindings,
            )
        branches = _top_level_c_style_conditional(expr)
        fallback = _top_level_c_style_fallback(expr) if branches is None else None
        if fallback is not None:
            left_expression, _ = fallback
            left_reference = _strip_expression(left_expression)
            if (
                (
                    _simple_reference(left_reference) is not None
                    or (
                        self.language == "typescript"
                        and re.fullmatch(
                            rf"process\s*\.\s*env\s*\.\s*{_IDENTIFIER}",
                            left_reference,
                        )
                    )
                )
                and self._fail_closed_nonempty_guard(left_reference, index)
            ):
                return self._resolve_origin_without_empty_paths(
                    left_expression,
                    index,
                    seen=seen,
                    bindings=bindings,
                )
            branches = fallback
        if branches is not None:
            resolved = tuple(
                candidate
                for branch in branches
                if (
                    candidate := self._resolve_origin_without_empty_paths(
                        branch,
                        index,
                        seen=seen,
                        bindings=bindings,
                    )
                )
                is not None
            )
            if not resolved:
                return None
            return resolved[0] if len(resolved) == 1 else _merge_resolved(expr, resolved)
        reference = _simple_reference(expr)
        if reference is not None:
            key = f"nonempty:{index}:{_compact_reference(reference)}"
            if key in seen:
                return None
            containing = self._containing_function(index)
            bound = (
                bindings.get((containing.start, reference))
                if bindings is not None and containing is not None
                else None
            )
            if bound is not None:
                bound_expression, bound_index = bound
                return self._resolve_origin_without_empty_paths(
                    bound_expression,
                    bound_index,
                    seen=seen | frozenset({key}),
                    bindings=bindings,
                )
            assignment = self.latest_assignment(reference, index)
            mutation = self.latest_mutation_origin(
                reference,
                index,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            indexed_mutation = self.latest_indexed_assignment_origin(
                reference,
                index,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if (
                indexed_mutation is not None
                and (
                    mutation is None
                    or indexed_mutation[0] > mutation[0].start
                )
                and (
                    assignment is None
                    or indexed_mutation[0] > assignment.start
                )
            ):
                return _resolved_without_empty_alternatives(indexed_mutation[1])
            if mutation is not None and (
                assignment is None or mutation[0].start > assignment.start
            ):
                return _resolved_without_empty_alternatives(mutation[1])
            if assignment is not None:
                return self._resolve_origin_without_empty_paths(
                    assignment.expression,
                    assignment.start,
                    seen=seen | frozenset({key}),
                    bindings=bindings,
                )
            if containing is not None and reference in containing.parameters:
                parameter_index = containing.parameters.index(reference)
                call_name = containing.name
                if call_name == "constructor":
                    call_name = self._containing_class_name(containing.start) or call_name
                default_expression = _c_style_parameter_default(
                    self.source,
                    self.masked,
                    containing,
                    reference,
                )
                callers = [
                    call
                    for call in self.calls(call_name)
                    if not (containing.start <= call.start <= containing.end)
                    and self._call_can_target_function(call, containing)
                ]
                if self._reachability_depth == 0:
                    reachable = [
                        call for call in callers if self._call_is_reachable(call)
                    ]
                    if reachable:
                        callers = reachable
                resolved_parameters: list[ResolvedValue] = []
                for call in callers:
                    argument = self._c_style_argument_for_parameter(
                        call,
                        parameter_index,
                        reference,
                    ) or default_expression
                    if argument is None:
                        continue
                    if (
                        re.search(r"password|passphrase|pwd", reference, re.IGNORECASE)
                        and self._is_explicit_negative_password_test_call(call, argument)
                    ):
                        continue
                    candidate = self._resolve_origin_without_empty_paths(
                        argument,
                        call.start,
                        seen=seen | frozenset({key}),
                        bindings=bindings,
                    )
                    if candidate is not None:
                        resolved_parameters.append(candidate)
                if resolved_parameters:
                    return (
                        resolved_parameters[0]
                        if len(resolved_parameters) == 1
                        else _merge_resolved(expr, tuple(resolved_parameters))
                    )
                if not callers and default_expression is not None:
                    return self._resolve_origin_without_empty_paths(
                        default_expression,
                        containing.start,
                        seen=seen | frozenset({key}),
                        bindings=bindings,
                    )
        called = _simple_call(expr)
        if called is not None:
            call_name, call_arguments = called
            key = f"nonempty-helper:{call_name}:{index}"
            if key not in seen:
                resolved: list[ResolvedValue] = []
                for definition in self.function(call_name):
                    if (
                        self._typescript_runtime_input_helper(definition)
                        and not self._typescript_runtime_input_helper_has_predictable_fallback(
                            definition
                        )
                    ):
                        resolved.append(
                            ResolvedValue(
                                expr,
                                "runtime_input",
                                steps=(
                                    ResolutionStep(
                                        expr,
                                        f"resolve runtime-input helper {call_name}",
                                        "runtime_input",
                                        self.source_span(
                                            definition.start,
                                            definition.end,
                                        ),
                                    ),
                                ),
                            )
                        )
                        continue
                    if self._csharp_console_input_helper(definition):
                        resolved.append(
                            ResolvedValue(
                                expr,
                                "runtime_input",
                                steps=(
                                    ResolutionStep(
                                        expr,
                                        f"resolve console-input helper {call_name}",
                                        "runtime_input",
                                        self.source_span(
                                            definition.start,
                                            definition.end,
                                        ),
                                    ),
                                ),
                            )
                        )
                        continue
                    if self._csharp_secure_random_string_helper(definition):
                        resolved.append(
                            ResolvedValue(
                                expr,
                                "secure_random",
                                steps=(
                                    ResolutionStep(
                                        expr,
                                        f"resolve secure-random output helper {call_name}",
                                        "secure_random",
                                        self.source_span(
                                            definition.start,
                                            definition.end,
                                        ),
                                    ),
                                ),
                            )
                        )
                        continue
                    call_bindings = dict(bindings or {})
                    call_bindings.update(
                        {
                            (definition.start, parameter): (argument, index)
                            for parameter, argument in _bind_c_style_arguments(
                                definition.parameters,
                                call_arguments,
                            )
                        }
                    )
                    for returned in definition.returns:
                        if self._containing_function(returned.index) is not definition:
                            continue
                        candidate = self._resolve_origin_without_empty_paths(
                            returned.expression,
                            returned.index,
                            seen=seen | frozenset({key}),
                            bindings=call_bindings,
                        )
                        if candidate is not None:
                            resolved.append(candidate)
                if resolved:
                    return (
                        resolved[0]
                        if len(resolved) == 1
                        else _merge_resolved(expr, tuple(resolved))
                    )
        return _resolved_without_empty_alternatives(self._resolve(
            expr,
            index,
            mode="origin",
            seen=seen,
            bindings=bindings,
        ))

    def _resolve_typescript_for_of_binding(
        self,
        reference: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        if self.language != "typescript" or "." in reference:
            return None
        containing = self._containing_function(index)
        start = containing.start if containing is not None else 0
        end = containing.end if containing is not None else len(self.source)
        segment = self.masked[start:end]
        pattern = re.compile(
            rf"\bfor\s*(?:await\s*)?\(\s*(?:const|let|var)\s+"
            rf"{re.escape(reference)}\s+"
            r"of\s+(?P<iterable>.+?)\)\s*\{",
            flags=re.DOTALL,
        )
        candidates: list[tuple[int, str]] = []
        for match in pattern.finditer(segment):
            body_start = start + match.end() - 1
            body_end = find_matching(self.masked, body_start, "{", "}")
            if body_end == -1 or not body_start < index < body_end:
                continue
            iterable_start = start + match.start("iterable")
            iterable_end = start + match.end("iterable")
            candidates.append(
                (body_start, self.source[iterable_start:iterable_end].strip())
            )
        if not candidates:
            return None
        body_start, iterable = max(candidates, key=lambda item: item[0])
        return self._resolve(
            iterable,
            body_start,
            mode="origin",
            seen=seen,
            bindings=bindings,
        )

    def _normalize_guard_terminators(self, prefix: str) -> str:
        """Represent proven no-return calls as ``throw`` for guard parsing."""

        terminating_names: set[str] = set()
        if self.language == "typescript":
            for definition in self.functions:
                segment = self.source[definition.start : definition.end]
                opening = segment.find("{")
                if opening == -1:
                    continue
                header = segment[:opening]
                if re.search(r"\)\s*:\s*never\b", header, flags=re.DOTALL):
                    terminating_names.add(definition.name)

        call_names = [r"process\s*\.\s*exit"]
        if self.language == "typescript":
            for match in re.finditer(
                r"import\s*\{(?P<items>[^{}]*)\}\s*from\s*"
                r"[\"'](?:node:)?process[\"']",
                self.source,
            ):
                for raw in match.group("items").split(","):
                    parts = re.split(r"\s+as\s+", raw.strip())
                    if parts and parts[0].strip() == "exit":
                        call_names.append(re.escape(parts[-1].strip()))
        call_names.extend(re.escape(name) for name in sorted(terminating_names))
        if not call_names:
            return prefix
        return re.sub(
            rf"\b(?:{'|'.join(call_names)})\s*\([^;{{}}]*\)\s*;?",
            "throw",
            prefix,
            flags=re.DOTALL,
        )

    def _typescript_stateful_digest_inputs(
        self,
        expression: str,
        index: int,
    ) -> tuple[str, ...] | None:
        match = re.fullmatch(
            rf"(?P<receiver>{_IDENTIFIER})\s*\.\s*digest\s*\([^)]*\)",
            _strip_expression(expression),
            flags=re.DOTALL,
        )
        if match is None:
            return None
        receiver = match.group("receiver")
        assignment = self.latest_assignment(receiver, index)
        if assignment is None:
            return None
        called = _simple_call(_strip_expression(assignment.expression))
        if called is None:
            return None
        call_name, arguments = called
        simple_name = _normalize_reference(call_name).rsplit(".", 1)[-1]
        if simple_name in {
            "createHmac",
            "createNodeHmac",
            "createIncrementalHmac",
        }:
            return (arguments[1],) if len(arguments) >= 2 else None
        if simple_name != "createHash":
            return None
        receiver_update = _compact_reference(f"{receiver}.update")
        updates = tuple(
            call.arguments[0]
            for call in self.calls("update")
            if assignment.start < call.start < index
            and _compact_reference(call.callee) == receiver_update
            and call.arguments
        )
        return updates or None

    def _typescript_runtime_input_helper(
        self,
        definition: FunctionDefinition,
    ) -> bool:
        if self.language != "typescript":
            return False
        body = self.masked[definition.start : definition.end]
        has_stdin = bool(
            re.search(
                r"\b(?:stdin|process\s*\.\s*stdin|input)\b",
                body,
                flags=re.IGNORECASE,
            )
            or re.search(r"\breadFileSync\s*\(\s*0\b", body)
            or re.search(r"\breadSync\s*\(\s*0\b", body)
            or re.search(
                r"\bread[A-Za-z0-9_$]*stdin[A-Za-z0-9_$]*\s*\(",
                body,
                flags=re.IGNORECASE,
            )
        )
        if not has_stdin:
            return False
        return bool(
            re.search(
                r"\b(?:createInterface|readline)\s*\(|\.\s*question\s*\(",
                body,
            )
            or re.search(
                r"\breadFileSync\s*\(\s*(?:0|stdin\b|process\s*\.\s*stdin)",
                body,
            )
            or re.search(r"\breadSync\s*\(", body)
            or re.search(
                r"\bread[A-Za-z0-9_$]*stdin[A-Za-z0-9_$]*\s*\(",
                body,
                flags=re.IGNORECASE,
            )
            or re.search(r"(?:process\s*\.\s*)?stdin\s*\.\s*read\s*\(", body)
            or re.search(
                r"(?:process\s*\.\s*)?stdin\s*\.\s*(?:on|once)\s*\(",
                body,
            )
            or re.search(
                r"\bfor\s+await\s*\([^)]*\bof\s+"
                r"(?:stdin|process\s*\.\s*stdin|input)\b",
                body,
            )
        )

    def _typescript_runtime_input_helper_has_predictable_fallback(
        self,
        definition: FunctionDefinition,
    ) -> bool:
        """Do not let an input API hide a fixed fallback return path."""

        if self.language != "typescript":
            return False
        for returned in definition.returns:
            expression = returned.expression.strip()
            literal = re.fullmatch(
                r"(?P<quote>[\"'])(?P<value>(?:\\.|[^\"'])*)(?P=quote)",
                expression,
                flags=re.DOTALL,
            )
            if literal is not None and literal.group("value"):
                return True
            if (
                expression.startswith("`")
                and re.search(
                    r"(?i)(?:fallback|default|demo|local|password|passphrase)",
                    expression,
                )
            ):
                return True
        return False

    def _typescript_runtime_record_helper(
        self,
        definition: FunctionDefinition,
        property_name: str,
    ) -> bool:
        """Recognize an empty record populated from a command-line array."""

        if self.language != "typescript" or not definition.parameters:
            return False
        body = self.source[definition.start : definition.end]
        masked = self.masked[definition.start : definition.end]
        returned = re.search(rf"\breturn\s+(?P<record>{_IDENTIFIER})\s*;", masked)
        if returned is None:
            return False
        record = returned.group("record")
        if re.search(
            rf"\b(?:const|let|var)\s+{re.escape(record)}\b[^=;]*=\s*\{{\s*\}}\s*;",
            masked,
            flags=re.DOTALL,
        ) is None:
            return False
        if re.search(
            rf"\b{re.escape(record)}\s*\.\s*{re.escape(property_name)}\s*=",
            masked,
        ):
            return False
        dynamic_writes = re.findall(
            rf"\b{re.escape(record)}\s*\[[^\]]+\]\s*=\s*(?P<value>[^;]+);",
            body,
            flags=re.DOTALL,
        )
        if not dynamic_writes:
            return False
        return any(
            re.search(
                rf"\b{re.escape(parameter)}\s*\[",
                masked,
            )
            for parameter in definition.parameters
        )

    def _csharp_byte_serialization_argument(
        self,
        call_name: str,
        call_arguments: tuple[str, ...],
    ) -> str | None:
        """Recognize a local byte-to-text encoder without trusting its name."""

        if self.language != "csharp" or len(call_arguments) != 1:
            return None
        for definition in self.function(call_name):
            if len(definition.parameters) != 1 or len(definition.returns) != 1:
                continue
            parameter = re.escape(definition.parameters[0])
            body = self.masked[definition.start : definition.end]
            loop = re.search(
                rf"\bforeach\s*\(\s*(?:var|byte)\s+"
                rf"(?P<item>{_IDENTIFIER})\s+in\s+{parameter}\s*\)",
                body,
            )
            if loop is not None:
                item = re.escape(loop.group("item"))
                append = re.search(
                    rf"(?P<builder>{_IDENTIFIER})\s*\.\s*Append\s*\("
                    rf"[^;]*\b{item}\b[^;]*\)\s*;",
                    body,
                    flags=re.DOTALL,
                )
                if append is not None:
                    builder = re.escape(append.group("builder"))
                    if re.search(
                        rf"\breturn\s+{builder}\s*\.\s*ToString\s*\(",
                        body,
                    ):
                        return call_arguments[0]

            indexed_digest = re.search(
                rf"\b(?:byte\s*\[\s*\]|var)\s+(?P<bytes>{_IDENTIFIER})\s*=\s*"
                rf"[^;]*\bComputeHash\s*\([^;]*\b{parameter}\b[^;]*\)\s*;",
                body,
                flags=re.DOTALL,
            )
            if indexed_digest is None:
                continue
            byte_name = re.escape(indexed_digest.group("bytes"))
            indexed_append = re.search(
                rf"(?P<builder>{_IDENTIFIER})\s*\.\s*Append\s*\("
                rf"[^;]*\b{byte_name}\s*\[[^\]]+\][^;]*\)\s*;",
                body,
                flags=re.DOTALL,
            )
            if indexed_append is None:
                continue
            indexed_builder = re.escape(indexed_append.group("builder"))
            if re.search(
                rf"\breturn\s+{indexed_builder}\s*\.\s*ToString\s*\(",
                body,
            ):
                return call_arguments[0]
        return None

    def _csharp_byte_decoder_argument(
        self,
        call_name: str,
        call_arguments: tuple[str, ...],
    ) -> str | None:
        """Recognize a local text-to-byte decoder as origin preserving."""

        if self.language != "csharp" or len(call_arguments) != 1:
            return None
        for definition in self.function(call_name):
            if len(definition.parameters) != 1 or len(definition.returns) != 1:
                continue
            parameter = re.escape(definition.parameters[0])
            body = self.source[definition.start : definition.end]
            if re.search(rf"\b{parameter}\b", body) is None:
                continue
            if not re.search(
                r"\b(?:Convert\s*\.\s*From(?:Hex|Base64)String|"
                r"byte\s*\.\s*(?:Parse|TryParse))\s*\(",
                body,
            ):
                continue
            if any(
                re.search(r"\b(?:byte\s*\[\s*\]|ReadOnlySpan\s*<\s*byte\s*>)\b", returned.expression)
                or re.search(r"\b[A-Za-z_$][\w$]*\b", returned.expression)
                for returned in definition.returns
            ):
                return call_arguments[0]
        return None

    def _csharp_console_input_helper(
        self,
        definition: FunctionDefinition,
    ) -> bool:
        """Recognize a local password reader backed by ``Console.ReadKey``.

        The returned string is runtime input even when it is accumulated in a
        ``StringBuilder`` rather than returned directly from ``ReadLine``.
        Keep the rule structural so an unrelated helper that merely logs a key
        press is not promoted to runtime input.
        """

        if self.language != "csharp":
            return False
        body = self.source[definition.start : definition.end]
        masked_body = self.masked[definition.start : definition.end]
        has_read_key = re.search(
            r"\bConsole\s*\.\s*ReadKey\s*\(",
            masked_body,
        ) is not None
        has_console_read = re.search(
            r"\bConsole\s*\.\s*(?:Read|ReadLine)\s*\(",
            masked_body,
        ) is not None
        if not has_read_key and not has_console_read:
            return False
        if (has_console_read or has_read_key) and re.search(
            r"password|passphrase|credential",
            definition.name,
            flags=re.IGNORECASE,
        ):
            if any(
                _c_style_has_literal_fallback(returned.expression)
                or re.search(
                    r"\bstring\s*\.\s*Empty\b",
                    returned.expression,
                )
                for returned in definition.returns
            ):
                return False
            returned_literals = tuple(
                _string_literal(returned.expression.strip())
                for returned in definition.returns
            )
            if definition.returns and not any(
                literal for literal in returned_literals if literal is not None
            ):
                return True
        appended_builders = tuple(
            dict.fromkeys(
                match.group("builder")
                for match in re.finditer(
                    rf"\b(?P<builder>{_IDENTIFIER})\s*\.\s*Append\s*\("
                    r"[^;]*\bKeyChar\b[^;]*\)",
                    body,
                    flags=re.DOTALL,
                )
            )
        )
        for builder_name in appended_builders:
            builder = re.escape(builder_name)
            encoded = re.search(
                rf"\b(?P<name>{_IDENTIFIER})\s*=\s*"
                r"(?:Encoding\s*\.\s*UTF8\s*\.\s*)?GetBytes\s*\(\s*"
                rf"{builder}\s*\.\s*ToString\s*\(\s*\)\s*\)",
                body,
                flags=re.DOTALL,
            )
            if encoded is not None:
                encoded_name = re.escape(encoded.group("name"))
                if any(
                    re.search(rf"\b{encoded_name}\b", returned.expression)
                    for returned in definition.returns
                ):
                    return True
        for returned in definition.returns:
            returned_expression = returned.expression.strip()
            match = re.fullmatch(
                rf"\s*(?P<builder>{_IDENTIFIER})\s*\.\s*ToString\s*\(\s*\)\s*",
                returned_expression,
            )
            if match is None:
                match = re.fullmatch(
                    rf"\s*.+?\?\s*(?:"
                    rf"null\s*:\s*(?P<builder>{_IDENTIFIER})\s*\.\s*ToString\s*\(\s*\)"
                    rf"|(?P<builder_first>{_IDENTIFIER})\s*\.\s*ToString\s*\(\s*\)\s*:\s*null"
                    rf")\s*",
                    returned_expression,
                    flags=re.DOTALL,
                )
            if match is None and re.fullmatch(_IDENTIFIER, returned_expression):
                assignment = self.latest_assignment(
                    returned_expression,
                    returned.index,
                )
                if (
                    assignment is not None
                    and self._assignment_scopes[assignment] is definition
                ):
                    match = re.fullmatch(
                        rf"\s*(?P<builder>{_IDENTIFIER})\s*\.\s*ToString\s*\(\s*\)\s*",
                        assignment.expression,
                    )
            if match is None:
                continue
            builder_name = match.groupdict().get("builder") or match.groupdict().get(
                "builder_first"
            )
            if not builder_name:
                continue
            builder = re.escape(builder_name)
            if re.search(
                rf"\b{builder}\s*\.\s*Append\s*\([^;]*\bKeyChar\b[^;]*\)",
                body,
                flags=re.DOTALL,
            ):
                return True
        key_char_buffers = tuple(
            dict.fromkeys(
                match.group("buffer")
                for match in re.finditer(
                    rf"\b(?P<buffer>{_IDENTIFIER})\s*\[[^\]]+\]\s*=\s*"
                    r"[^;]*\bKeyChar\b[^;]*;",
                    body,
                    flags=re.DOTALL,
                )
            )
        )
        for returned in definition.returns:
            if any(
                re.fullmatch(
                    rf"\s*new\s+string\s*\(\s*{re.escape(buffer)}\b.*\)\s*",
                    returned.expression,
                    flags=re.DOTALL,
                )
                for buffer in key_char_buffers
            ):
                return True
        return False

    def _csharp_secure_random_string_helper(
        self,
        definition: FunctionDefinition,
    ) -> bool:
        if self.language != "csharp":
            return False
        body = self.masked[definition.start : definition.end]
        random_buffers = {
            match.group("name")
            for match in re.finditer(
                rf"\bRandomNumberGenerator\s*\.\s*Fill\s*\(\s*"
                rf"(?P<name>{_IDENTIFIER})\b",
                body,
            )
        }
        random_buffers.update(
            match.group("name")
            for match in re.finditer(
                rf"\b(?P<name>{_IDENTIFIER})\s*=\s*"
                r"RandomNumberGenerator\s*\.\s*GetBytes\s*\(",
                body,
            )
        )
        random_receivers = {
            match.group("name")
            for match in re.finditer(
                rf"\b(?:var|RandomNumberGenerator)\s+(?P<name>{_IDENTIFIER})\s*=\s*"
                r"RandomNumberGenerator\s*\.\s*Create\s*\(\s*\)",
                body,
            )
        }
        for receiver in random_receivers:
            random_buffers.update(
                match.group("name")
                for match in re.finditer(
                    rf"\b{re.escape(receiver)}\s*\.\s*GetBytes\s*\(\s*"
                    rf"(?P<name>{_IDENTIFIER})\b",
                    body,
                )
            )
        if not random_buffers:
            return False
        for returned in definition.returns:
            expression = returned.expression.strip()
            if any(re.search(rf"\b{re.escape(name)}\b", expression) for name in random_buffers):
                return True
            string_result = re.fullmatch(
                rf"new\s+string\s*\(\s*(?P<name>{_IDENTIFIER})\s*\)",
                expression,
            )
            if string_result is None:
                continue
            output = re.escape(string_result.group("name"))
            if any(
                re.search(
                    rf"\b{output}\s*\[[^\]]+\]\s*=\s*[^;]*"
                    rf"\b{re.escape(random_buffer)}\s*\[",
                    body,
                    flags=re.DOTALL,
                )
                for random_buffer in random_buffers
            ):
                return True
        return False

    def _typescript_returned_identity_argument(
        self,
        call_name: str,
        call_arguments: tuple[str, ...],
        index: int,
    ) -> str | None:
        """Resolve a local factory that returns a one-argument identity arrow."""

        if self.language != "typescript" or len(call_arguments) != 1:
            return None
        assignment = self.latest_assignment(call_name, index)
        if assignment is None:
            return None
        factory_call = _simple_call(_strip_expression(assignment.expression))
        if factory_call is None or factory_call[1]:
            return None
        for definition in self.function(factory_call[0]):
            body = self.source[definition.start : definition.end]
            identity = re.search(
                rf"\breturn\s+\(\s*(?P<parameter>{_IDENTIFIER})"
                r"(?:\s*:\s*[^,)]+)?\s*\)\s*(?:\:\s*[^=]+)?=>\s*"
                rf"(?:\{{\s*return\s+(?P=parameter)\s*;\s*\}}|(?P=parameter))\s*;",
                body,
                flags=re.DOTALL,
            )
            if identity is not None:
                return call_arguments[0]
        return None

    def _resolve_typescript_returned_closure_parameter(
        self,
        reference: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        if self.language != "typescript":
            return None

        candidates: list[tuple[int, FunctionDefinition, int]] = []
        for returned in re.finditer(r"\breturn\s*\(", self.masked[:index]):
            opening = self.masked.find("(", returned.start(), returned.end())
            closing = _find_matching_masked(self.masked, opening, "(", ")")
            if closing == -1:
                continue
            parameters = tuple(
                item
                for item in (
                    _parameter_name(raw)
                    for raw in _split_parameter_text(
                        self.source[opening + 1 : closing]
                    )
                )
                if item
            )
            if reference not in parameters:
                continue
            body_start = _typescript_arrow_body_start(self.masked, closing + 1)
            if body_start == -1:
                continue
            while body_start < len(self.masked) and self.masked[body_start].isspace():
                body_start += 1
            if body_start >= len(self.masked) or self.masked[body_start] != "{":
                continue
            body_end = _find_matching_masked(self.masked, body_start, "{", "}")
            if body_end == -1 or not body_start <= index <= body_end:
                continue
            factory = self._containing_function(returned.start())
            if factory is None:
                continue
            candidates.append((body_end - body_start, factory, parameters.index(reference)))

        if not candidates:
            return None
        _span, factory, parameter_index = min(candidates, key=lambda item: item[0])
        state_key = (
            f"returned-closure:{factory.start}:{reference}:{parameter_index}:{index}"
        )
        if state_key in seen:
            return None

        values: list[ResolvedValue] = []
        factory_pattern = re.compile(rf"\b{re.escape(factory.name)}\s*\(")
        for call in factory_pattern.finditer(self.masked):
            if factory.start <= call.start() <= factory.end:
                continue
            first_opening = self.masked.find("(", call.start(), call.end())
            first_closing = _find_matching_masked(
                self.masked,
                first_opening,
                "(",
                ")",
            )
            if first_closing == -1:
                continue
            second_opening = first_closing + 1
            while (
                second_opening < len(self.masked)
                and self.masked[second_opening].isspace()
            ):
                second_opening += 1
            if second_opening >= len(self.masked) or self.masked[second_opening] != "(":
                continue
            second_closing = _find_matching_masked(
                self.masked,
                second_opening,
                "(",
                ")",
            )
            if second_closing == -1:
                continue
            arguments = _split_inline_arguments(
                self.source[second_opening + 1 : second_closing]
            )
            if parameter_index >= len(arguments):
                continue
            values.append(
                self._resolve(
                    arguments[parameter_index],
                    call.start(),
                    mode=mode,
                    seen=seen | frozenset({state_key}),
                    bindings=bindings,
                )
            )
        return _merge_resolved(reference, values) if values else None

    def _resolve_csharp_selected_method_group_call(
        self,
        call_name: str,
        call_arguments: tuple[str, ...],
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        if self.language != "csharp":
            return None
        assignment = self.latest_assignment(call_name, index)
        if assignment is None:
            return None
        direct_target = _simple_reference(_strip_expression(assignment.expression))
        target_names: list[str] = []
        if direct_target is not None and self.function(direct_target):
            target_names.append(direct_target)
        else:
            factory_call = _simple_call(_strip_expression(assignment.expression))
            if factory_call is None:
                return None
            factory_name, factory_arguments = factory_call
            for definition in self.function(factory_name):
                body = self.source[definition.start : definition.end]
                switch_match = re.search(
                    rf"\bswitch\s*\(\s*(?P<selector>{_IDENTIFIER})"
                    r"(?P<lower>\s*\.\s*ToLower(?:Invariant)?\s*\(\s*\))?\s*\)",
                    body,
                )
                if switch_match is None:
                    continue
                selector = switch_match.group("selector")
                if selector not in definition.parameters:
                    continue
                selector_index = definition.parameters.index(selector)
                if selector_index >= len(factory_arguments):
                    continue
                resolved_selector = self._resolve(
                    factory_arguments[selector_index],
                    assignment.start,
                    mode="string",
                    seen=seen,
                    bindings=bindings,
                )
                selector_values = set(resolved_selector.values)
                if switch_match.group("lower"):
                    selector_values = {value.lower() for value in selector_values}
                for case in re.finditer(
                    rf"\bcase\s+(?P<label>[\"'](?:\\.|[^\"'])*[\"'])\s*:"
                    rf"(?:(?!\bcase\b|\bdefault\b)[\s\S])*?"
                    rf"\breturn\s+(?P<target>{_IDENTIFIER})\s*;",
                    body,
                ):
                    label = _string_literal(case.group("label"))
                    if label is not None and label in selector_values:
                        target_names.append(case.group("target"))

        state_key = f"selected-method-group:{call_name}:{assignment.start}:{index}"
        if state_key in seen:
            return None
        values = [
            self._resolve(
                f"{target}({', '.join(call_arguments)})",
                index,
                mode=mode,
                seen=seen | frozenset({state_key}),
                bindings=bindings,
            )
            for target in dict.fromkeys(target_names)
        ]
        return _merge_resolved(call_name, values) if values else None

    def _typescript_inline_identity_argument(
        self,
        expression: str,
        index: int,
    ) -> str | None:
        if self.language != "typescript":
            return None
        match = re.fullmatch(
            rf"(?P<factory>{_IDENTIFIER})\s*\(\s*\)\s*"
            r"\(\s*(?P<argument>.*)\s*\)",
            _strip_expression(expression),
            flags=re.DOTALL,
        )
        if match is None:
            return None
        argument = match.group("argument").strip()
        if len(_split_inline_arguments(argument)) != 1:
            return None
        for definition in self.function(match.group("factory")):
            if definition.start > index:
                continue
            body = self.source[definition.start : definition.end]
            identity = re.search(
                rf"\breturn\s+\(\s*(?P<parameter>{_IDENTIFIER})"
                r"(?:\s*:\s*[^,)]+)?\s*\)\s*(?:\:\s*[^=]+)?=>\s*"
                rf"(?:\{{\s*return\s+(?P=parameter)\s*;\s*\}}|(?P=parameter))\s*;",
                body,
                flags=re.DOTALL,
            )
            if identity is not None:
                return argument
        return None

    def _resolve_instance_field(
        self,
        reference: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        owner = self._containing_class_name(index)
        if owner is None:
            return None
        normalized = _normalize_reference(reference)
        compact_reference = re.sub(r"\s+", "", reference)
        class_qualified = compact_reference.startswith(f"{owner}.")
        assignments = [
            assignment
            for assignment in self.assignments
            if _normalize_reference(assignment.name) == normalized
            and self._containing_class_name(assignment.start) == owner
            and (
                (scope := self._assignment_scopes[assignment]) is not None
                and (
                    scope.name in {owner, "constructor"}
                    or (
                        class_qualified
                        and self._containing_class_name(scope.start) == owner
                    )
                )
            )
        ]
        if not assignments:
            if self.language == "typescript" and compact_reference.startswith(
                "this."
            ):
                field_name = normalized
                for constructor in self.functions:
                    if (
                        constructor.name == "constructor"
                        and self._containing_class_name(constructor.start) == owner
                        and field_name in constructor.parameters
                        and _typescript_constructor_parameter_property(
                            self.source,
                            self.masked,
                            constructor,
                            field_name,
                        )
                    ):
                        resolved = self._resolve_parameter(
                            field_name,
                            constructor.start,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        )
                        if resolved is not None:
                            return resolved
                getter_values: list[ResolvedValue] = []
                for getter in self.function(field_name):
                    prefix = self.masked[max(0, getter.start - 48) : getter.start]
                    if not re.search(r"\bget\s*$", prefix):
                        continue
                    for returned in getter.returns:
                        getter_values.append(
                            self._resolve(
                                returned.expression,
                                returned.index,
                                mode=mode,
                                seen=seen,
                                bindings=bindings,
                            )
                        )
                if getter_values:
                    return _merge_resolved(reference, getter_values)
            return None
        field_key = f"field:{owner}:{normalized}"
        if field_key in seen:
            return None
        values = [
            self._resolve(
                assignment.expression,
                assignment.start,
                mode=mode,
                seen=seen | frozenset({field_key}),
                bindings=bindings,
            )
            for assignment in assignments
        ]
        return _merge_resolved(reference, values)

    def _resolve_indexed_value(
        self,
        expression: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        match = re.fullmatch(
            rf"(?P<container>{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)"
            r"\s*\[\s*(?P<index>[^\[\]]+)\s*\]",
            expression,
        )
        indexed_call = None
        if match is None:
            # Keep arbitrary-expression indexing tightly bounded to call
            # results such as ``input.split(...)[0]``.  Treating every
            # expression ending in brackets as an index operation also
            # captures array literals and TypeScript type syntax.
            indexed_call = re.fullmatch(
                r"(?P<container>.+\))\s*\[\s*(?P<index>-?\d+)\s*\]",
                expression,
                flags=re.DOTALL,
            )
            match = indexed_call
        if match is None:
            return None
        container = match.group("container").strip()
        raw_index = match.group("index").strip()
        if not re.fullmatch(r"-?\d+", raw_index):
            resolved_index = self._resolve(
                raw_index,
                index,
                mode="int",
                seen=seen | frozenset({f"index-expression:{raw_index}"}),
                bindings=bindings,
            )
            if len(resolved_index.values) == 1 and re.fullmatch(
                r"-?\d+",
                resolved_index.values[0].replace("_", ""),
            ):
                raw_index = resolved_index.values[0].replace("_", "")
        if not re.fullmatch(r"-?\d+", raw_index):
            assignment = self.latest_assignment(container, index)
            if assignment is not None:
                value_expressions = _c_style_container_elements(
                    assignment.expression,
                    allow_shorthand_initializer=self.language == "csharp",
                )
                if value_expressions is None:
                    value_expressions = _c_style_object_value_expressions(
                        assignment.expression
                    )
                if value_expressions:
                    values = [
                        self._resolve(
                            value,
                            assignment.start,
                            mode=mode,
                            seen=seen
                            | frozenset(
                                {f"dynamic-object-index:{container}:{raw_index}"}
                            ),
                            bindings=bindings,
                        )
                        for value in value_expressions
                    ]
                    return _merge_resolved(expression, values)
            if mode != "origin":
                return None
            property_name = _string_literal(raw_index)
            if property_name is not None:
                owner_expression = (
                    assignment.expression if assignment is not None else container
                )
                owner_index = assignment.start if assignment is not None else index
                property_value = self._resolve_object_property_expression(
                    owner_expression,
                    property_name,
                    owner_index,
                    mode=mode,
                    seen=seen
                    | frozenset({f"dynamic-property:{container}:{property_name}"}),
                    bindings=bindings,
                )
                if property_value is not None:
                    return property_value
            return self._resolve(
                container,
                index,
                mode=mode,
                seen=seen | frozenset({f"dynamic-index:{container}"}),
                bindings=bindings,
            )
        element_index = int(raw_index)
        compact_container = re.sub(r"\s+", "", container)
        key = f"indexed:{compact_container}:{element_index}"
        if key in seen:
            return None

        if indexed_call is not None:
            return self._resolve_c_style_container_element(
                container,
                element_index,
                index,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )

        assignment = self.latest_assignment(container, index)
        if assignment is not None:
            resolved_element = self._resolve_c_style_container_element(
                assignment.expression,
                element_index,
                assignment.start,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if resolved_element is not None:
                return resolved_element
        if mode == "origin":
            mutation = self.latest_mutation_origin(
                container,
                index,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if mutation is not None:
                return mutation[1]
        if mode == "origin":
            if self.language == "csharp" and compact_container == "args":
                return ResolvedValue(
                    expression,
                    "runtime_input",
                    steps=(
                        ResolutionStep(
                            expression,
                            "resolve command-line argument",
                            "runtime_input",
                            self.source_span(index, index),
                        ),
                    ),
                )
            # An indexed function parameter has the provenance of its caller
            # argument even when no local container assignment exists.
            return self._resolve(
                container,
                index,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
        return None

    def _resolve_non_nullish_fallback_operand(
        self,
        expression: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        """Resolve only effective values of one nullish-fallback operand."""

        if mode != "origin":
            return self._resolve(
                expression,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
        if _strip_expression(expression) in {"null", "undefined"}:
            return None
        if _c_style_expression_never_returns(expression, self.language):
            return None
        values = self._resolve_non_nullish_sources(
            expression,
            index,
            seen=seen,
            bindings=bindings,
        )
        if values:
            return _merge_resolved(expression, values)
        return self._resolve(
            expression,
            index,
            mode=mode,
            seen=seen,
            bindings=bindings,
        )

    def _resolve_non_nullish_sources(
        self,
        expression: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> tuple[ResolvedValue, ...]:
        expr = _strip_expression(expression)
        if expr in {"null", "undefined"}:
            return ()
        if _c_style_expression_never_returns(expr, self.language):
            return ()
        conditional = _top_level_c_style_conditional(expr)
        if conditional is not None:
            values: list[ResolvedValue] = []
            for branch in conditional:
                values.extend(
                    self._resolve_non_nullish_sources(
                        branch,
                        index,
                        seen=seen,
                        bindings=bindings,
                    )
                )
            return tuple(values)

        fallback = _top_level_c_style_fallback(expr)
        if fallback is not None:
            values: list[ResolvedValue] = []
            for branch in fallback:
                values.extend(
                    self._resolve_non_nullish_sources(
                        branch,
                        index,
                        seen=seen,
                        bindings=bindings,
                    )
                )
            return tuple(values)

        reference = _simple_reference(expr)
        if reference is not None:
            assignment = self.latest_assignment(reference, index)
            reference_key = f"nonnull-reference:{reference}:{index}"
            if assignment is not None and reference_key not in seen:
                return self._resolve_non_nullish_sources(
                    assignment.expression,
                    assignment.start,
                    seen=seen | frozenset({reference_key}),
                    bindings=bindings,
                )

        iife = re.fullmatch(
            r"\(\s*\(\s*\)\s*=>\s*\{(?P<body>.*)\}\s*\)\s*\(\s*\)",
            expr,
            flags=re.DOTALL,
        )
        if iife is not None:
            values: list[ResolvedValue] = []
            body = iife.group("body")
            expression_start = self.source.find(expr, max(0, index))
            if expression_start == -1:
                expression_start = index
            for returned in re.finditer(
                r"\breturn\s+(?P<expression>[^;]+);",
                body,
                flags=re.DOTALL,
            ):
                returned_index = expression_start + iife.start("body") + returned.start(
                    "expression"
                )
                values.extend(
                    self._resolve_non_nullish_sources(
                        returned.group("expression"),
                        returned_index,
                        seen=seen,
                        bindings=bindings,
                    )
                )
            return tuple(values)

        called = _simple_call(expr)
        if called is not None:
            call_name, call_arguments = called
            helper_key = f"nonnull-helper:{call_name}:{index}"
            if helper_key not in seen:
                values: list[ResolvedValue] = []
                for definition in self.function(call_name):
                    call_bindings = dict(bindings or {})
                    call_bindings.update(
                        {
                            (definition.start, parameter): (argument, index)
                            for parameter, argument in _bind_c_style_arguments(
                                definition.parameters,
                                call_arguments,
                            )
                        }
                    )
                    for returned in definition.returns:
                        if self._containing_function(returned.index) is not definition:
                            continue
                        values.extend(
                            self._resolve_non_nullish_sources(
                                returned.expression,
                                returned.index,
                                seen=seen | frozenset({helper_key}),
                                bindings=call_bindings,
                            )
                        )
                if values:
                    return tuple(values)

        return (
            self._resolve(
                expr,
                index,
                mode="origin",
                seen=seen,
                bindings=bindings,
            ),
        )

    def _resolve_c_style_container_element(
        self,
        expression: str,
        element_index: int,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        identity = re.sub(r"\s+", "", _strip_expression(expression))
        state_key = f"container-element:{element_index}:{index}:{identity}"
        if state_key in seen:
            return None
        seen = seen | frozenset({state_key})

        elements = _c_style_container_elements(
            expression,
            allow_shorthand_initializer=self.language == "csharp",
        )
        if elements is not None:
            normalized_index = (
                element_index if element_index >= 0 else len(elements) + element_index
            )
            if 0 <= normalized_index < len(elements):
                return self._resolve(
                    elements[normalized_index],
                    index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
            return None

        split_match = re.fullmatch(
            r"(?P<receiver>.+)\.\s*split\s*\(\s*"
            r"(?P<delimiter>[\"'](?:\\.|[^\"'])*[\"'])\s*\)",
            _strip_expression(expression),
            flags=re.DOTALL,
        )
        if split_match is not None:
            delimiter = _string_literal(split_match.group("delimiter"))
            if delimiter:
                resolved_split = self._resolve_c_style_delimited_element(
                    split_match.group("receiver"),
                    delimiter,
                    element_index,
                    index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
                if resolved_split is not None:
                    return resolved_split

        receiver = _origin_preserving_receiver_wrapper(
            _strip_expression(expression)
        )
        if receiver is not None:
            return self._resolve(
                receiver,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )

        called = _simple_call(_strip_expression(expression))
        if called is None:
            return None
        if mode == "origin":
            resolved_container = self._resolve(
                expression,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            if resolved_container.origin != "unknown":
                return resolved_container
        call_name, call_arguments = called
        values: list[ResolvedValue] = []
        for definition in self.function(call_name):
            call_bindings = dict(bindings or {})
            call_bindings.update(
                {
                    (definition.start, parameter): (argument, index)
                    for parameter, argument in _bind_c_style_arguments(
                        definition.parameters,
                        call_arguments,
                    )
                }
            )
            for returned in definition.returns:
                if self._containing_function(returned.index) is not definition:
                    continue
                resolved = self._resolve_c_style_container_element(
                    returned.expression,
                    element_index,
                    returned.index,
                    mode=mode,
                    seen=seen,
                    bindings=call_bindings,
                )
                if resolved is not None:
                    values.append(resolved)
        return _merge_resolved(expression, values) if values else None

    def _resolve_c_style_delimited_element(
        self,
        expression: str,
        delimiter: str,
        element_index: int,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        expr = _strip_expression(expression)
        literal = _string_literal(expr)
        if literal is not None:
            fields = literal.split(delimiter)
            normalized_index = (
                element_index if element_index >= 0 else len(fields) + element_index
            )
            if 0 <= normalized_index < len(fields):
                return ResolvedValue(
                    expr,
                    "constant",
                    (fields[normalized_index],),
                    (
                        ResolutionStep(
                            expr,
                            "resolve delimited string literal element",
                            "constant",
                            self.source_span(index, index + len(expr)),
                        ),
                    ),
                )
            return None
        template_fields = _typescript_template_fields(expr, delimiter)
        if template_fields is not None:
            normalized_index = (
                element_index
                if element_index >= 0
                else len(template_fields) + element_index
            )
            if 0 <= normalized_index < len(template_fields):
                return self._resolve(
                    template_fields[normalized_index],
                    index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
            return None

        iife_element = self._resolve_typescript_iife_delimited_element(
            expr,
            delimiter,
            element_index,
            mode=mode,
        )
        if iife_element is not None:
            return iife_element

        reference = _simple_reference(expr)
        if reference is not None:
            assignment = self.latest_assignment(reference, index)
            if assignment is not None:
                return self._resolve_c_style_delimited_element(
                    assignment.expression,
                    delimiter,
                    element_index,
                    assignment.start,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
            containing = self._containing_function(index)
            if containing is not None and reference in containing.parameters:
                parameter_key = (
                    f"delimited-parameter:{containing.start}:{reference}:"
                    f"{element_index}"
                )
                if parameter_key in seen:
                    return None
                parameter_index = containing.parameters.index(reference)
                arguments: list[tuple[str, int]] = []
                if bindings is not None:
                    bound = bindings.get((containing.start, reference))
                    if bound is not None:
                        arguments.append(bound)
                for call in self.calls(containing.name):
                    if containing.start <= call.start <= containing.end:
                        continue
                    if not self._call_can_target_function(call, containing):
                        continue
                    argument = self._c_style_argument_for_parameter(
                        call,
                        parameter_index,
                        reference,
                    )
                    if argument is not None:
                        arguments.append((argument, call.start))
                values = [
                    resolved
                    for argument, caller_index in dict.fromkeys(arguments)
                    if (
                        resolved := self._resolve_c_style_delimited_element(
                            argument,
                            delimiter,
                            element_index,
                            caller_index,
                            mode=mode,
                            seen=seen | frozenset({parameter_key}),
                            bindings=bindings,
                        )
                    )
                    is not None
                ]
                if values:
                    return _merge_resolved(expression, values)

        called = _simple_call(expr)
        if called is not None:
            call_name, call_arguments = called
            values: list[ResolvedValue] = []
            for definition in self.function(call_name):
                call_bindings = dict(bindings or {})
                call_bindings.update(
                    {
                        (definition.start, parameter): (argument, index)
                        for parameter, argument in _bind_c_style_arguments(
                            definition.parameters,
                            call_arguments,
                        )
                    }
                )
                for returned in definition.returns:
                    if self._containing_function(returned.index) is not definition:
                        continue
                    resolved = self._resolve_c_style_delimited_element(
                        returned.expression,
                        delimiter,
                        element_index,
                        returned.index,
                        mode=mode,
                        seen=seen,
                        bindings=call_bindings,
                    )
                    if resolved is not None:
                        values.append(resolved)
            if values:
                return _merge_resolved(expression, values)

        resolved_expression = self._resolve(
            expr,
            index,
            mode=mode,
            seen=seen,
            bindings=bindings,
        )
        if resolved_expression.values:
            selected: list[str] = []
            for value in resolved_expression.values:
                fields = value.split(delimiter)
                normalized_index = (
                    element_index
                    if element_index >= 0
                    else len(fields) + element_index
                )
                if 0 <= normalized_index < len(fields):
                    selected.append(fields[normalized_index])
            if selected:
                return ResolvedValue(
                    expr,
                    resolved_expression.origin,
                    tuple(dict.fromkeys(selected)),
                    resolved_expression.steps,
                    resolved_expression.unresolved_reason,
                )

        return None

    def _resolve_object_property(
        self,
        expression: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        match = re.fullmatch(
            rf"(?P<owner>{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)"
            rf"(?:\s*\.\s*(?P<dot_property>{_IDENTIFIER})"
            rf"|\s*\[\s*(?P<quote>[\"'])(?P<bracket_property>[^\"']+)"
            rf"(?P=quote)\s*\])",
            expression,
        )
        if not match:
            return None
        owner = re.sub(r"\s+", "", match.group("owner"))
        property_name = match.group("dot_property") or match.group("bracket_property")
        containing = self._containing_function(index)
        scope = containing.start if containing is not None else "global"
        key = f"object-property:{scope}:{owner}.{property_name}"
        if key in seen:
            return None
        assignment = self.latest_assignment(owner, index)
        if assignment is None and owner in {"this", "base"}:
            class_name = self._containing_class_name(index)
            member_reference = _compact_reference(property_name)
            candidates = [
                candidate
                for candidate in self.assignments
                if _compact_reference(candidate.name) == member_reference
                and self._containing_class_name(candidate.start) == class_name
                and (
                    self._assignment_scopes[candidate] is None
                    or self._assignment_scopes[candidate].name == "constructor"
                )
            ]
            if candidates:
                member_assignment = max(candidates, key=lambda item: item.start)
                return self._resolve(
                    member_assignment.expression,
                    member_assignment.start,
                    mode=mode,
                    seen=seen | frozenset({key}),
                    bindings=bindings,
                )
        if assignment is None and owner.startswith("this."):
            class_name = self._containing_class_name(index)
            candidates = [
                candidate
                for candidate in self.assignments
                if _compact_reference(candidate.name) == _compact_reference(owner)
                and self._containing_class_name(candidate.start) == class_name
                and (
                    (scope := self._assignment_scopes[candidate]) is not None
                    and scope.name in {class_name, "constructor"}
                )
            ]
            assignment = max(candidates, key=lambda item: item.start) if candidates else None
        if assignment is None and "." in owner and not owner.startswith("this."):
            root, *nested_properties = owner.split(".")
            root_assignment = self.latest_assignment(root, index)
            if root_assignment is not None:
                nested = self._resolve_nested_object_property_expression(
                    root_assignment.expression,
                    tuple(nested_properties) + (property_name,),
                    root_assignment.start,
                    mode=mode,
                    seen=seen | frozenset({key}),
                    bindings=bindings,
                )
                if nested is not None:
                    return nested
            nested = self._resolve_nested_object_property_expression(
                root,
                tuple(nested_properties) + (property_name,),
                index,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if nested is not None:
                return nested
        if assignment is None:
            resolved = self._resolve_object_property_expression(
                owner,
                property_name,
                index,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
        else:
            resolved = self._resolve_object_property_expression(
                assignment.expression,
                property_name,
                assignment.start,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
        if (
            resolved is not None
            and mode == "origin"
            and (
                self._fail_closed_nonempty_guard(expression, index)
                or self._reference_is_truthy_on_path(expression, index)
            )
        ):
            return _resolved_without_empty_alternatives(resolved)
        return resolved

    def _resolve_nested_object_property_expression(
        self,
        expression: str,
        property_path: tuple[str, ...],
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        if not property_path:
            return self._resolve(
                expression,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
        identity = re.sub(r"\s+", "", _strip_expression(expression))
        key = f"nested-object-property:{'.'.join(property_path)}:{index}:{identity}"
        if key in seen:
            return None
        seen = seen | frozenset({key})

        property_expression = _c_style_object_property_value(
            expression,
            property_path[0],
        )
        if property_expression is not None:
            return self._resolve_nested_object_property_expression(
                property_expression,
                property_path[1:],
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )

        reference = _simple_reference(_strip_expression(expression))
        if reference is not None:
            containing = self._containing_function(index)
            bound = (
                bindings.get((containing.start, reference))
                if bindings is not None and containing is not None
                else None
            )
            if bound is not None:
                bound_expression, caller_index = bound
                return self._resolve_nested_object_property_expression(
                    bound_expression,
                    property_path,
                    caller_index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
            property_reference = _compact_reference(
                f"{reference}.{property_path[0]}"
            )
            property_assignments = list(
                self.reaching_assignment_alternatives(
                    property_reference,
                    index,
                )
            )
            if property_assignments:
                resolved_properties = [
                    resolved
                    for candidate in property_assignments
                    if (
                        resolved := self._resolve_nested_object_property_expression(
                            candidate.expression,
                            property_path[1:],
                            candidate.start,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        )
                    )
                    is not None
                ]
                owner_assignment = self.latest_assignment(reference, index)
                if (
                    owner_assignment is not None
                    and not any(
                        self.position_is_definitely_reached(candidate.start)
                        for candidate in property_assignments
                    )
                    and _strip_expression(owner_assignment.expression)
                    != _strip_expression(expression)
                ):
                    initial_property = self._resolve_nested_object_property_expression(
                        owner_assignment.expression,
                        property_path,
                        owner_assignment.start,
                        mode=mode,
                        seen=seen,
                        bindings=bindings,
                    )
                    if initial_property is not None:
                        resolved_properties.append(initial_property)
                if resolved_properties:
                    return _merge_resolved(expression, resolved_properties)
            assignment = self.latest_assignment(reference, index)
            if assignment is not None and _strip_expression(
                assignment.expression
            ) != _strip_expression(expression):
                return self._resolve_nested_object_property_expression(
                    assignment.expression,
                    property_path,
                    assignment.start,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
            if containing is not None and reference in containing.parameters:
                parameter_index = containing.parameters.index(reference)
                tuple_member_index = (
                    _csharp_tuple_parameter_member_index(
                        self.source,
                        self.masked,
                        containing,
                        reference,
                        property_path[0],
                    )
                    if self.language == "csharp"
                    else None
                )
                call_name = containing.name
                if call_name == "constructor":
                    call_name = self._containing_class_name(containing.start) or call_name
                default_expression = _c_style_parameter_default(
                    self.source,
                    self.masked,
                    containing,
                    reference,
                )
                values: list[ResolvedValue] = []
                for call in self.calls(call_name):
                    if containing.start <= call.start <= containing.end:
                        continue
                    if not self._call_can_target_function(call, containing):
                        continue
                    argument = self._c_style_argument_for_parameter(
                        call,
                        parameter_index,
                        reference,
                    ) or default_expression
                    if argument is None:
                        continue
                    argument_path = property_path
                    if tuple_member_index is not None:
                        tuple_values = _c_style_container_elements(argument)
                        if (
                            tuple_values is None
                            or tuple_member_index >= len(tuple_values)
                        ):
                            continue
                        argument = tuple_values[tuple_member_index]
                        argument_path = property_path[1:]
                    resolved = self._resolve_nested_object_property_expression(
                        argument,
                        argument_path,
                        call.start,
                        mode=mode,
                        seen=seen,
                        bindings=bindings,
                    )
                    if resolved is not None:
                        values.append(resolved)
                if values:
                    return _merge_resolved(expression, values)

        called = _simple_call(_strip_expression(expression))
        if called is None:
            return None
        call_name, call_arguments = called
        values: list[ResolvedValue] = []
        for definition in self.function(call_name):
            call_bindings = dict(bindings or {})
            call_bindings.update(
                {
                    (definition.start, parameter): (argument, index)
                    for parameter, argument in _bind_c_style_arguments(
                        definition.parameters,
                        call_arguments,
                    )
                }
            )
            for returned in definition.returns:
                if self._containing_function(returned.index) is not definition:
                    continue
                resolved = self._resolve_nested_object_property_expression(
                    returned.expression,
                    property_path,
                    returned.index,
                    mode=mode,
                    seen=seen,
                    bindings=call_bindings,
                )
                if resolved is not None:
                    values.append(resolved)
        return _merge_resolved(expression, values) if values else None

    def _resolve_object_property_expression(
        self,
        expression: str,
        property_name: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        identity = re.sub(r"\s+", "", _strip_expression(expression))
        state_key = (
            f"object-property-expression:{property_name}:{index}:{identity}"
        )
        if state_key in seen:
            return None
        seen = seen | frozenset({state_key})

        wrapper = _single_argument_wrapper(_strip_expression(expression))
        if wrapper is not None:
            return self._resolve_object_property_expression(
                wrapper,
                property_name,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )

        call_result_property = _c_style_call_result_property(
            _strip_expression(expression)
        )
        if call_result_property is not None:
            owner_expression, nested_property = call_result_property
            resolved = self._resolve_nested_object_property_expression(
                owner_expression,
                (nested_property, property_name),
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            if resolved is not None:
                return resolved

        nested_reference = _simple_reference(_strip_expression(expression))
        if nested_reference is not None and "." in nested_reference:
            root, *nested_properties = _compact_reference(nested_reference).split(".")
            root_assignment = self.latest_assignment(root, index)
            if root_assignment is not None:
                nested = self._resolve_nested_object_property_expression(
                    root_assignment.expression,
                    tuple(nested_properties) + (property_name,),
                    root_assignment.start,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
                if nested is not None:
                    return nested

        called = _simple_call(_strip_expression(expression))
        if called is not None:
            call_name, call_arguments = called
            for definition in self.function(call_name):
                if (
                    mode == "origin"
                    and self._typescript_runtime_record_helper(
                        definition,
                        property_name,
                    )
                ):
                    argument_origins = tuple(
                        self._resolve(
                            argument,
                            index,
                            mode="origin",
                            seen=seen,
                            bindings=bindings,
                        )
                        for argument in call_arguments
                    )
                    if any(
                        item.origin == "runtime_input" for item in argument_origins
                    ):
                        return ResolvedValue(
                            expression,
                            "runtime_input",
                            steps=(
                                ResolutionStep(
                                    expression,
                                    f"resolve command-line record helper {call_name}",
                                    "runtime_input",
                                    self.source_span(definition.start, definition.end),
                                ),
                            ),
                        )
            if (
                call_name in {"parse", "stringify"}
                and re.match(
                    rf"JSON\s*\.\s*{re.escape(call_name)}\s*\(",
                    _strip_expression(expression),
                )
                and call_arguments
            ):
                return self._resolve_object_property_expression(
                    call_arguments[0],
                    property_name,
                    index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )

        indexed_property = self._resolve_indexed_object_property(
            expression,
            property_name,
            index,
            mode=mode,
            seen=seen,
            bindings=bindings,
        )
        if indexed_property is not None:
            return indexed_property

        property_expression = _c_style_object_property_value(
            expression,
            property_name,
        )
        if property_expression is not None:
            return self._resolve(
                property_expression,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )

        stripped_expression = _strip_expression(expression)
        expression_iife = re.fullmatch(
            r"\(\s*\(\s*\)\s*(?::\s*[^=]+)?=>\s*"
            r"(?P<value>(?!\{).+)\)\s*\(\s*\)",
            stripped_expression,
            flags=re.DOTALL,
        )
        if expression_iife is not None:
            expression_start = self.source.find(
                stripped_expression,
                max(0, index),
            )
            if expression_start == -1:
                expression_start = index
            returned_index = expression_start + expression_iife.start("value")
            resolved = self._resolve_object_property_expression(
                expression_iife.group("value"),
                property_name,
                returned_index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            if resolved is not None:
                return resolved

        iife = re.fullmatch(
            r"\(\s*\(\s*\)\s*(?::\s*[^=]+)?=>\s*\{(?P<body>.*)\}"
            r"\s*\)\s*\(\s*\)",
            stripped_expression,
            flags=re.DOTALL,
        )
        if iife is not None:
            expression_start = self.source.find(
                stripped_expression,
                max(0, index),
            )
            if expression_start == -1:
                expression_start = index
            values: list[ResolvedValue] = []
            for returned in re.finditer(
                r"\breturn\s+(?P<expression>[^;]+);",
                iife.group("body"),
                flags=re.DOTALL,
            ):
                returned_expression = returned.group("expression")
                if _c_style_expression_never_returns(
                    returned_expression,
                    self.language,
                ):
                    continue
                returned_index = (
                    expression_start
                    + iife.start("body")
                    + returned.start("expression")
                )
                resolved = self._resolve_object_property_expression(
                    returned_expression,
                    property_name,
                    returned_index,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
                if resolved is not None:
                    values.append(resolved)
            if values:
                return _merge_resolved(expression, values)

        constructor = re.fullmatch(
            rf"new\s+(?P<class>{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)"
            r"\s*\((?P<arguments>.*)\)",
            _strip_expression(expression),
            flags=re.DOTALL,
        )
        class_name = None
        constructor_arguments: tuple[str, ...] = ()
        if constructor is not None:
            class_name = re.sub(
                r"\s+", "", constructor.group("class")
            ).rsplit(".", 1)[-1]
            constructor_arguments = _split_inline_arguments(
                constructor.group("arguments")
            )
            constructed_member = self._resolve_constructed_member(
                class_name,
                constructor_arguments,
                property_name,
                index,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
            if constructed_member is not None:
                return constructed_member

        if self.language == "csharp":
            target_typed_constructor = re.fullmatch(
                r"new\s*\((?P<arguments>.*)\)",
                _strip_expression(expression),
                flags=re.DOTALL,
            )
            if target_typed_constructor is not None:
                for argument in _split_inline_arguments(
                    target_typed_constructor.group("arguments")
                ):
                    named = re.fullmatch(
                        rf"\s*{re.escape(property_name)}\s*:\s*(?P<value>.+)",
                        argument,
                        flags=re.DOTALL,
                    )
                    if named is not None:
                        return self._resolve(
                            named.group("value"),
                            index,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        )

            if constructor is not None and class_name is not None:
                member_index = _csharp_primary_record_member_index(
                    self.source,
                    class_name,
                    property_name,
                )
                if member_index is None:
                    member_index = self._csharp_constructor_member_index(
                        class_name,
                        property_name,
                    )
                if (
                    member_index is not None
                    and member_index < len(constructor_arguments)
                ):
                    argument = _c_style_argument_value(
                        constructor_arguments,
                        member_index,
                        property_name,
                    )
                    return self._resolve(
                        argument,
                        index,
                        mode=mode,
                        seen=seen,
                        bindings=bindings,
                    )

        reference = _simple_reference(_strip_expression(expression))
        if reference is not None:
            property_reference = _compact_reference(
                f"{reference}.{property_name}"
            )
            containing = self._containing_function(index)
            property_assignments = list(
                self.reaching_assignment_alternatives(
                    property_reference,
                    index,
                )
            )
            if property_assignments:
                exclude_empty = mode == "origin" and (
                    self._fail_closed_nonempty_guard(property_reference, index)
                    or self._reference_is_truthy_on_path(property_reference, index)
                )
                values = [
                    (
                        self._resolve_origin_without_empty_paths(
                            candidate.expression,
                            candidate.start,
                            seen=seen,
                            bindings=bindings,
                        )
                        if exclude_empty
                        else self._resolve(
                            candidate.expression,
                            candidate.start,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        )
                    )
                    for candidate in property_assignments
                ]
                values = [value for value in values if value is not None]
                owner_assignment = self.latest_assignment(reference, index)
                if (
                    owner_assignment is not None
                    and not any(
                        self.position_is_definitely_reached(candidate.start)
                        for candidate in property_assignments
                    )
                    and _strip_expression(owner_assignment.expression)
                    != _strip_expression(expression)
                ):
                    initial_property = self._resolve_object_property_expression(
                        owner_assignment.expression,
                        property_name,
                        owner_assignment.start,
                        mode=mode,
                        seen=seen,
                        bindings=bindings,
                    )
                    if exclude_empty and initial_property is not None:
                        initial_property = _resolved_without_empty_alternatives(
                            initial_property
                        )
                    if initial_property is not None:
                        values.append(initial_property)
                return _merge_resolved(expression, values)
            assignment = self.latest_assignment(reference, index)
            if assignment is not None and _strip_expression(
                assignment.expression
            ) != _strip_expression(expression):
                return self._resolve_object_property_expression(
                    assignment.expression,
                    property_name,
                    assignment.start,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
            containing = self._containing_function(index)
            if containing is not None and reference in containing.parameters:
                parameter_index = containing.parameters.index(reference)
                tuple_member_index = (
                    _csharp_tuple_parameter_member_index(
                        self.source,
                        self.masked,
                        containing,
                        reference,
                        property_name,
                    )
                    if self.language == "csharp"
                    else None
                )
                call_name = containing.name
                if call_name == "constructor":
                    call_name = self._containing_class_name(containing.start) or call_name
                default_expression = _c_style_parameter_default(
                    self.source,
                    self.masked,
                    containing,
                    reference,
                )
                values: list[ResolvedValue] = []
                for call in self.calls(call_name):
                    if containing.start <= call.start <= containing.end:
                        continue
                    if not self._call_can_target_function(call, containing):
                        continue
                    argument = self._c_style_argument_for_parameter(
                        call,
                        parameter_index,
                        reference,
                    ) or default_expression
                    if argument is None:
                        continue
                    if tuple_member_index is not None:
                        if self.analysis_profile == "reachable_security_flow":
                            resolved = self._resolve_csharp_tuple_member_expression(
                                argument,
                                tuple_member_index,
                                call.start,
                                mode=mode,
                                seen=seen,
                                bindings=bindings,
                            )
                        else:
                            tuple_values = _c_style_container_elements(argument)
                            if (
                                tuple_values is None
                                or tuple_member_index >= len(tuple_values)
                            ):
                                continue
                            resolved = self._resolve(
                                tuple_values[tuple_member_index],
                                call.start,
                                mode=mode,
                                seen=seen,
                                bindings=bindings,
                            )
                    else:
                        resolved = self._resolve_object_property_expression(
                            argument,
                            property_name,
                            call.start,
                            mode=mode,
                            seen=seen,
                            bindings=bindings,
                        )
                    if resolved is not None:
                        values.append(resolved)
                if values:
                    return _merge_resolved(expression, values)

        called = _simple_call(_strip_expression(expression))
        if called is not None:
            call_name, call_arguments = called
            values: list[ResolvedValue] = []
            for definition in self.function(call_name):
                call_bindings = dict(bindings or {})
                call_bindings.update(
                    {
                        (definition.start, parameter): (argument, index)
                        for parameter, argument in _bind_c_style_arguments(
                            definition.parameters,
                            call_arguments,
                        )
                    }
                )
                for returned in definition.returns:
                    resolved = self._resolve_object_property_expression(
                        returned.expression,
                        property_name,
                        returned.index,
                        mode=mode,
                        seen=seen,
                        bindings=call_bindings,
                    )
                    if resolved is not None:
                        values.append(resolved)
            if values:
                return _merge_resolved(expression, values)
        if mode == "origin":
            owner_origin = self._resolve(
                _strip_expression(expression),
                index,
                mode="origin",
                seen=seen,
                bindings=bindings,
            )
            if owner_origin.origin == "runtime_input":
                return ResolvedValue(
                    f"{_strip_expression(expression)}.{property_name}",
                    "runtime_input",
                    steps=owner_origin.steps,
                )
        return None

    def _resolve_constructed_member(
        self,
        class_name: str,
        arguments: tuple[str, ...],
        property_name: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        constructor_name = (
            "constructor" if self.language == "typescript" else class_name
        )
        constructors = [
            definition
            for definition in self.function(constructor_name)
            if self._containing_class_name(definition.start) == class_name
        ]
        values: list[ResolvedValue] = []
        normalized_property = _normalize_reference(property_name)
        state_key = f"constructed-member:{class_name}:{property_name}:{index}"
        if state_key in seen:
            return None
        for constructor in constructors:
            call_bindings = dict(bindings or {})
            call_bindings.update(
                {
                    (constructor.start, parameter): (argument, index)
                    for parameter, argument in _bind_c_style_arguments(
                        constructor.parameters,
                        arguments,
                    )
                }
            )
            for assignment in self.assignments:
                if self._assignment_scopes[assignment] != constructor:
                    continue
                if _normalize_reference(assignment.name) != normalized_property:
                    continue
                values.append(
                    self._resolve(
                        assignment.expression,
                        assignment.start,
                        mode=mode,
                        seen=seen | frozenset({state_key}),
                        bindings=call_bindings,
                    )
                )
            if (
                self.language == "typescript"
                and property_name in constructor.parameters
                and _typescript_constructor_parameter_property(
                    self.source,
                    self.masked,
                    constructor,
                    property_name,
                )
            ):
                parameter_index = constructor.parameters.index(property_name)
                if parameter_index < len(arguments):
                    values.append(
                        self._resolve(
                            arguments[parameter_index],
                            index,
                            mode=mode,
                            seen=seen | frozenset({state_key}),
                            bindings=bindings,
                        )
                    )
        return _merge_resolved(property_name, values) if values else None

    def _resolve_csharp_tuple_member_expression(
        self,
        expression: str,
        member_index: int,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        identity = re.sub(r"\s+", "", expression)
        key = f"csharp-tuple-member:{member_index}:{index}:{identity}"
        if key in seen:
            return None
        next_seen = seen | frozenset({key})

        elements = _csharp_tuple_expression_elements(expression)
        if member_index < len(elements):
            return self._resolve(
                elements[member_index],
                index,
                mode=mode,
                seen=next_seen,
                bindings=bindings,
            )

        reference = _simple_reference(_strip_expression(expression))
        if reference is not None:
            values = [
                resolved
                for assignment in self.reaching_assignment_alternatives(
                    reference,
                    index,
                )
                if (
                    resolved := self._resolve_csharp_tuple_member_expression(
                        assignment.expression,
                        member_index,
                        assignment.start,
                        mode=mode,
                        seen=next_seen,
                        bindings=bindings,
                    )
                )
                is not None
            ]
            if values:
                return _merge_resolved(expression, values)

        called = _simple_call(_strip_expression(expression))
        if called is None:
            return None
        call_name, call_arguments = called
        values = []
        for definition in self.function(call_name):
            call_bindings = dict(bindings or {})
            call_bindings.update(
                {
                    (definition.start, parameter): (argument, index)
                    for parameter, argument in _bind_c_style_arguments(
                        definition.parameters,
                        call_arguments,
                    )
                }
            )
            for returned in definition.returns:
                resolved = self._resolve_csharp_tuple_member_expression(
                    returned.expression,
                    member_index,
                    returned.index,
                    mode=mode,
                    seen=next_seen,
                    bindings=call_bindings,
                )
                if resolved is not None:
                    values.append(resolved)
        return _merge_resolved(expression, values) if values else None

    def _csharp_constructor_member_index(
        self,
        class_name: str,
        property_name: str,
    ) -> int | None:
        """Map an ordinary C# constructor property to its source argument."""

        normalized_property = _normalize_reference(property_name)
        for constructor in self.function(class_name):
            for assignment in self.assignments:
                if self._assignment_scopes[assignment] != constructor:
                    continue
                if _normalize_reference(assignment.name) != normalized_property:
                    continue
                expression = _strip_expression(assignment.expression)
                parameter = _simple_reference(expression)
                if parameter is None:
                    fallback = _top_level_c_style_fallback(expression)
                    effective = tuple(
                        branch
                        for branch in (fallback or ())
                        if not _c_style_expression_never_returns(
                            branch,
                            self.language,
                        )
                    )
                    if len(effective) == 1:
                        parameter = _simple_reference(
                            _strip_expression(effective[0])
                        )
                if parameter in constructor.parameters:
                    return constructor.parameters.index(parameter)
        return None

    def _resolve_indexed_object_property(
        self,
        expression: str,
        property_name: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        match = re.fullmatch(
            rf"(?P<container>{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)"
            r"\s*\[.+\]",
            _strip_expression(expression),
            flags=re.DOTALL,
        )
        if match is None:
            return None
        container = re.sub(r"\s+", "", match.group("container"))
        key = f"indexed-object-property:{container}.{property_name}"
        if key in seen:
            return None
        container_pattern = r"\s*\.\s*".join(
            re.escape(part) for part in container.split(".")
        )
        assignment_pattern = re.compile(
            rf"\b{container_pattern}\s*\[[^\]]+\]\s*=(?!=|>)",
            flags=re.DOTALL,
        )
        values: list[ResolvedValue] = []
        for assignment_match in assignment_pattern.finditer(self.masked, 0, index):
            expression_end = _next_c_style_statement_end_masked(
                self.masked,
                assignment_match.end(),
                self.language,
            )
            if expression_end == -1 or expression_end > index:
                continue
            assigned_object = self.source[
                assignment_match.end() : expression_end
            ].strip()
            property_expression = _c_style_object_property_value(
                assigned_object,
                property_name,
            )
            if property_expression is None:
                continue
            values.append(
                self._resolve(
                    property_expression,
                    assignment_match.start(),
                    mode=mode,
                    seen=seen | frozenset({key}),
                    bindings=bindings,
                )
            )
        return _merge_resolved(expression, values) if values else None

    def _resolve_typescript_callback_parameter(
        self,
        reference: str,
        index: int,
    ) -> ResolvedValue | None:
        if self.language != "typescript":
            return None
        normalized = _normalize_reference(reference)
        for call in self.calls("randomBytes"):
            if not (call.start <= index <= call.end) or len(call.arguments) < 2:
                continue
            callback = call.arguments[-1].strip()
            match = re.match(
                rf"(?:\((?P<parameters>[^)]*)\)|(?P<single>{_IDENTIFIER}))"
                r"\s*=>",
                callback,
                flags=re.DOTALL,
            )
            if match is None:
                match = re.match(
                    r"function\s*\((?P<parameters>[^)]*)\)",
                    callback,
                    flags=re.DOTALL,
                )
            if match is None:
                continue
            parameter_text = match.groupdict().get("parameters") or match.groupdict().get("single") or ""
            parameters = tuple(
                item
                for item in (
                    _parameter_name(raw)
                    for raw in _split_parameter_text(parameter_text)
                )
                if item
            )
            if parameters and normalized == _normalize_reference(parameters[-1]):
                return ResolvedValue(
                    reference,
                    "secure_random",
                    steps=(
                        ResolutionStep(
                            reference,
                            "resolve randomBytes callback output",
                            "secure_random",
                            self.source_span(call.start, call.end),
                        ),
                    ),
                )
        for call in self.calls("question"):
            if not (call.start <= index <= call.end) or not call.arguments:
                continue
            callback = call.arguments[-1].strip()
            match = re.match(
                rf"(?:\((?P<parameters>[^)]*)\)|(?P<single>{_IDENTIFIER}))"
                r"\s*=>",
                callback,
                flags=re.DOTALL,
            )
            if match is None:
                match = re.match(
                    r"function\s*\((?P<parameters>[^)]*)\)",
                    callback,
                    flags=re.DOTALL,
                )
            if match is None:
                continue
            parameter_text = (
                match.groupdict().get("parameters")
                or match.groupdict().get("single")
                or ""
            )
            parameters = tuple(
                item
                for item in (
                    _parameter_name(raw)
                    for raw in _split_parameter_text(parameter_text)
                )
                if item
            )
            if parameters and normalized == _normalize_reference(parameters[0]):
                return ResolvedValue(
                    reference,
                    "runtime_input",
                    steps=(
                        ResolutionStep(
                            reference,
                            "resolve readline question callback input",
                            "runtime_input",
                            self.source_span(call.start, call.end),
                        ),
                    ),
                )
        for call in (*self.calls("on"), *self.calls("once")):
            if not (call.start <= index <= call.end) or len(call.arguments) < 2:
                continue
            if _compact_reference(call.callee) not in {
                "process.stdin.on",
                "process.stdin.once",
            }:
                continue
            event_name = _string_literal(call.arguments[0].strip())
            if event_name != "data":
                continue
            callback = call.arguments[-1].strip()
            match = re.match(
                rf"(?:\((?P<parameters>[^)]*)\)|(?P<single>{_IDENTIFIER}))"
                r"\s*=>",
                callback,
                flags=re.DOTALL,
            )
            if match is None:
                match = re.match(
                    r"function\s*\((?P<parameters>[^)]*)\)",
                    callback,
                    flags=re.DOTALL,
                )
            if match is None:
                continue
            parameter_text = (
                match.groupdict().get("parameters")
                or match.groupdict().get("single")
                or ""
            )
            parameters = tuple(
                item
                for item in (
                    _parameter_name(raw)
                    for raw in _split_parameter_text(parameter_text)
                )
                if item
            )
            if parameters and normalized == _normalize_reference(parameters[0]):
                return ResolvedValue(
                    reference,
                    "runtime_input",
                    steps=(
                        ResolutionStep(
                            reference,
                            "resolve process stdin data callback input",
                            "runtime_input",
                            self.source_span(call.start, call.end),
                        ),
                    ),
                )
        return None

    def _resolve_typescript_promise_callback_parameter(
        self,
        reference: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        if self.language != "typescript":
            return None
        normalized = _normalize_reference(reference)
        for callback_call in self.calls("then"):
            if not (
                callback_call.start <= index <= callback_call.end
                and callback_call.arguments
            ):
                continue
            callback = callback_call.arguments[0].strip()
            match = re.match(
                rf"(?:\((?P<parameters>[^)]*)\)|(?P<single>{_IDENTIFIER}))"
                r"\s*=>",
                callback,
                flags=re.DOTALL,
            )
            if match is None:
                continue
            parameter_text = (
                match.groupdict().get("parameters")
                or match.groupdict().get("single")
                or ""
            )
            parameters = tuple(
                item
                for item in (
                    _parameter_name(raw)
                    for raw in _split_parameter_text(parameter_text)
                )
                if item
            )
            if not parameters or normalized != _normalize_reference(parameters[0]):
                continue
            producers = tuple(
                call
                for call in self._calls
                if call.end < callback_call.start
                and self.source[call.end : callback_call.start].strip() == "."
            )
            if not producers:
                continue
            producer = max(producers, key=lambda item: item.end)
            return self._resolve(
                self.source[producer.start : producer.end],
                producer.start,
                mode=mode,
                seen=seen,
                bindings=bindings,
            )
        return None

    def _resolve_c_style_file_round_trip(
        self,
        expression: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        if mode != "origin" or self.language not in {"csharp", "typescript"}:
            return None
        if self.language == "csharp":
            match = re.fullmatch(
                r"(?:File|System\.IO\.File)\s*\.\s*"
                r"(?P<method>ReadAllBytes|ReadAllText)\s*\((?P<path>.*)\)",
                expression,
                flags=re.DOTALL,
            )
            write_name = (
                "WriteAllBytes"
                if match is not None and match.group("method") == "ReadAllBytes"
                else "WriteAllText"
            )
        else:
            match = re.fullmatch(
                r"(?:(?s:.+)\s*\.\s*)?"
                r"(?P<method>readFileSync|readFile)\s*\(\s*(?P<path>[^,]+)"
                r"(?:,.*)?\)",
                expression,
                flags=re.DOTALL,
            )
            write_name = (
                "writeFileSync"
                if match is not None and match.group("method") == "readFileSync"
                else "writeFile"
            )
        if match is None:
            return None
        read_path = match.group("path").strip()
        resolved_read_path = self._resolve(
            read_path,
            index,
            mode="string",
            seen=seen,
            bindings=bindings,
        )
        values: list[ResolvedValue] = []
        for call in self.calls(write_name):
            if call.start >= index or len(call.arguments) < 2:
                continue
            write_path = call.arguments[0]
            resolved_write_path = self._resolve(
                write_path,
                call.start,
                mode="string",
                seen=seen,
                bindings=bindings,
            )
            same_path = bool(
                set(resolved_read_path.values) & set(resolved_write_path.values)
            ) or _strip_expression(read_path) == _strip_expression(write_path)
            if not same_path:
                continue
            values.append(
                self._resolve(
                    call.arguments[1],
                    call.start,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
            )
        if self.language == "csharp" and write_name == "WriteAllBytes":
            for writer_assignment in self.assignments:
                if writer_assignment.start >= index:
                    continue
                writer_match = re.match(
                    r"(?:System\s*\.\s*IO\s*\.)?File\s*\.\s*"
                    r"(?:Create|OpenWrite)\s*\(\s*(?P<path>[^,)]+)",
                    _strip_expression(writer_assignment.expression),
                    flags=re.DOTALL,
                )
                if writer_match is None:
                    continue
                resolved_write_path = self._resolve(
                    writer_match.group("path"),
                    writer_assignment.start,
                    mode="string",
                    seen=seen,
                    bindings=bindings,
                )
                same_path = bool(
                    set(resolved_read_path.values)
                    & set(resolved_write_path.values)
                ) or _strip_expression(read_path) == _strip_expression(
                    writer_match.group("path")
                )
                if not same_path:
                    continue
                writer_reference = _compact_reference(writer_assignment.name)
                for call in self.calls("Write", "WriteAsync"):
                    compact = re.sub(r"\s+", "", call.callee)
                    call_receiver = (
                        compact.rsplit(".", 1)[0] if "." in compact else ""
                    )
                    if (
                        writer_assignment.end <= call.start < index
                        and call.arguments
                        and call_receiver == writer_reference
                    ):
                        values.append(
                            self._resolve(
                                call.arguments[0],
                                call.start,
                                mode=mode,
                                seen=seen,
                                bindings=bindings,
                            )
                        )
        return _merge_resolved(expression, values) if values else None

    def _resolve_csharp_provider_property(
        self,
        expression: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        """Resolve provider-generated C# key/IV properties without losing the receiver."""

        if self.language != "csharp" or mode != "origin":
            return None
        match = re.fullmatch(
            rf"(?P<receiver>{_IDENTIFIER})\s*\.\s*(?P<property>Key|IV)",
            expression,
        )
        if not match:
            return None

        receiver = match.group("receiver")
        property_name = match.group("property")
        qualified = f"{receiver}.{property_name}"
        containing = self._containing_function(index)
        scope = containing.start if containing is not None else "global"
        # A later property assignment may resolve through an earlier value of
        # the same property. Keep the use position in the recursion identity
        # so that historical values are not mistaken for a self-cycle.
        key = f"property:{scope}:{qualified}:{index}"
        step = ResolutionStep(
            expression,
            "inspect provider property",
            mode,
            self.source_span(index, index),
        )
        if key in seen:
            return ResolvedValue(
                expression,
                "unknown",
                steps=(step,),
                unresolved_reason="cyclic property assignment",
            )

        explicit_assignments = [
            assignment
            for assignment in self.assignments
            if _compact_reference(assignment.name) == qualified
            and self._assignment_scopes[assignment] == containing
            and assignment.end <= index
            and _compact_reference(_strip_expression(assignment.expression))
            != qualified
        ]
        if explicit_assignments:
            assignment = max(explicit_assignments, key=lambda item: item.start)
            # SymmetricAlgorithm Key/IV getters return copies. Filling the
            # getter result cannot update the provider property itself.
            resolved = self._resolve(
                assignment.expression,
                assignment.start,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            return _prepend_resolution(
                expression,
                f"resolve assignment {assignment.name}",
                resolved,
                step,
            )

        receiver_assignment = self.latest_assignment(receiver, index)
        if receiver_assignment is None:
            prior_method_value = self._resolve_csharp_prior_called_property_assignment(
                qualified,
                containing,
                index,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if prior_method_value is not None:
                return _prepend_resolution(
                    expression,
                    f"resolve prior called-method assignment of {qualified}",
                    prior_method_value,
                    step,
                )
        if receiver_assignment is None and containing is not None:
            if receiver in containing.parameters:
                parameter_index = containing.parameters.index(receiver)
                parameter_key = f"provider-parameter:{containing.start}:{receiver}"
                if parameter_key not in seen:
                    bound = (
                        bindings.get((containing.start, receiver))
                        if bindings is not None
                        else None
                    )
                    arguments: list[tuple[str, int]] = []
                    if bound is not None:
                        arguments.append(bound)
                    else:
                        call_name = containing.name
                        if call_name == "constructor":
                            call_name = (
                                self._containing_class_name(containing.start)
                                or call_name
                            )
                        for call in self.calls(call_name):
                            if containing.start <= call.start <= containing.end:
                                continue
                            if not self._call_can_target_function(call, containing):
                                continue
                            argument = self._c_style_argument_for_parameter(
                                call,
                                parameter_index,
                                receiver,
                            )
                            if argument is not None:
                                arguments.append((argument, call.start))
                    values: list[ResolvedValue] = []
                    for argument, caller_index in arguments:
                        argument_reference = _simple_reference(
                            _strip_expression(argument)
                        )
                        if argument_reference is None:
                            continue
                        resolved = self._resolve_csharp_provider_property(
                            f"{argument_reference}.{property_name}",
                            caller_index,
                            mode=mode,
                            seen=seen | frozenset({parameter_key}),
                            bindings=bindings,
                        )
                        if resolved is not None:
                            values.append(resolved)
                    if values:
                        return _prepend_resolution(
                            expression,
                            "resolve provider property through caller argument",
                            _merge_resolved(expression, values),
                            step,
                        )
        if receiver_assignment is None:
            return None
        initializer = _strip_expression(receiver_assignment.expression)
        initializer_reference = _simple_reference(initializer)
        if initializer_reference is not None and initializer_reference != receiver:
            resolved_alias = self._resolve_csharp_provider_property(
                f"{initializer_reference}.{property_name}",
                receiver_assignment.start,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if resolved_alias is not None:
                return _prepend_resolution(
                    expression,
                    f"resolve provider alias {receiver}",
                    resolved_alias,
                    step,
                )
        explicit_hmac = re.fullmatch(
            r"new\s+(?:global::)?(?:System\.Security\.Cryptography\.)?"
            r"HMAC(?:MD5|SHA1|SHA256|SHA384|SHA512|SHA3_256|SHA3_384|SHA3_512)"
            r"\s*\((?P<arguments>.+)\)",
            initializer,
            flags=re.DOTALL,
        )
        if property_name == "Key" and explicit_hmac is not None:
            arguments = _split_inline_arguments(explicit_hmac.group("arguments"))
            if arguments:
                resolved = self._resolve(
                    arguments[0],
                    receiver_assignment.start,
                    mode=mode,
                    seen=seen | frozenset({key}),
                    bindings=bindings,
                )
                return _prepend_resolution(
                    expression,
                    f"resolve explicit HMAC key from {receiver}",
                    resolved,
                    step,
                )
        activator_arguments = csharp_activator_constructor_arguments(initializer)
        if property_name == "Key" and activator_arguments:
            resolved = self._resolve(
                activator_arguments[0],
                receiver_assignment.start,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            return _prepend_resolution(
                expression,
                f"resolve Activator-created HMAC key from {receiver}",
                resolved,
                step,
            )
        provider_generated = bool(
            re.fullmatch(
                r"(?:global::)?(?:System\.Security\.Cryptography\.)?"
                r"(?:Aes|AES|DES|TripleDES|DESCryptoServiceProvider|"
                r"TripleDESCryptoServiceProvider|AesManaged|"
                r"AesCryptoServiceProvider|Rijndael|RijndaelManaged)"
                r"\s*\.\s*Create\s*\(\s*\)",
                initializer,
            )
            or re.fullmatch(
                r"new\s+(?:global::)?"
                r"(?:System\.Security\.Cryptography\.)?"
                r"(?:DESCryptoServiceProvider|TripleDESCryptoServiceProvider|"
                r"AesManaged|AesCryptoServiceProvider|RijndaelManaged)"
                r"\s*\(\s*\)",
                initializer,
            )
            or (
                property_name == "Key"
                and re.fullmatch(
                    r"new\s+(?:global::)?"
                    r"(?:System\.Security\.Cryptography\.)?"
                    r"HMAC(?:SHA256|SHA384|SHA512|SHA3_256|SHA3_384|SHA3_512)"
                    r"\s*\(\s*\)",
                    initializer,
                )
            )
        )
        generated_method = "GenerateKey" if property_name == "Key" else "GenerateIV"
        if not provider_generated:
            provider_generated = any(
                call.start < index
                and re.sub(r"\s+", "", call.callee)
                == f"{receiver}.{generated_method}"
                and self.position_is_in_same_or_enclosing_function(
                    call.start,
                    index,
                )
                for call in self.calls(generated_method)
            )
        if not provider_generated:
            resolved_initializer = self._resolve(
                initializer,
                receiver_assignment.start,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            provider_generated = resolved_initializer.origin == "secure_random"
        if not provider_generated:
            return None
        return ResolvedValue(
            expression,
            "secure_random",
            steps=(
                ResolutionStep(
                    expression,
                    f"resolve provider-generated {property_name} from {receiver}",
                    "secure_random",
                    self.source_span(receiver_assignment.start, receiver_assignment.end),
                ),
            ),
        )

    def _direct_api_value(
        self,
        expression: str,
        mode: str,
    ) -> tuple[ValueOrigin, tuple[str, ...]] | None:
        if self.language != "typescript" or mode != "origin":
            return None
        compact_expression = re.sub(r"\s+", "", expression)
        if any(
            compact_expression == alias
            or compact_expression.startswith(f"{alias}.")
            for alias in self.process_runtime_aliases
        ):
            return "runtime_input", ()
        commonjs_crypto = re.match(
            r"require\s*\(\s*[\"'](?:node:)?crypto[\"']\s*\)\s*\.\s*"
            rf"(?P<name>{_IDENTIFIER})\s*\(",
            expression,
        )
        if commonjs_crypto is not None and commonjs_crypto.group("name") in {
            "generateKey",
            "generateKeySync",
            "randomBytes",
            "randomFillSync",
            "randomInt",
        }:
            return "secure_random", ()
        match = re.match(
            rf"(?P<callee>{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)\s*\(",
            expression,
        )
        if not match:
            return None
        callee = re.sub(r"\s+", "", match.group("callee"))
        simple = callee.rsplit(".", 1)[-1]
        canonical = next(
            (
                imported
                for imported, alias in self.import_aliases.items()
                if alias == simple
            ),
            "",
        )
        receiver = callee.rsplit(".", 1)[0] if "." in callee else ""
        if simple in self.filesystem_read_aliases or (
            receiver in self.filesystem_namespace_aliases
            and simple in {"readFile", "readFileSync"}
        ):
            return "runtime_input", ()
        if not canonical and receiver in self.crypto_namespace_aliases:
            canonical = simple
        if not canonical and callee in {
            "crypto.getRandomValues",
            "globalThis.crypto.getRandomValues",
        }:
            canonical = "getRandomValues"
        if (
            not canonical
            and not receiver
            and simple
            in {
                "generateKey",
                "generateKeySync",
                "randomBytes",
                "randomFillSync",
                "randomInt",
            }
            and not self.function(simple)
            and re.search(
                rf"\b(?:function|class)\s+{re.escape(simple)}\b",
                self.source,
            )
            is None
        ):
            canonical = simple
        if canonical in {
            "generateKey",
            "generateKeySync",
            "getRandomValues",
            "randomBytes",
            "randomFillSync",
            "randomInt",
        }:
            return "secure_random", ()
        return None

    def _resolve_typescript_map_get(
        self,
        expression: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        if self.language != "typescript":
            return None
        match = re.fullmatch(
            rf"(?P<receiver>{_IDENTIFIER})\s*\.\s*get\s*\([^)]*\)",
            expression,
            flags=re.DOTALL,
        )
        if match is None:
            return None
        receiver = match.group("receiver")
        containing = self._containing_function(index)
        values: list[ResolvedValue] = []
        for call in self.calls("set"):
            if re.sub(r"\s+", "", call.callee) != f"{receiver}.set":
                continue
            if call.start >= index or len(call.arguments) < 2:
                continue
            if containing is not None and not (
                containing.start <= call.start <= containing.end
            ):
                continue
            values.append(
                self._resolve(
                    call.arguments[1],
                    call.start,
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
            )
        if not values:
            return None
        return _merge_resolved(expression, values)

    def _resolve_typescript_derived_origin(
        self,
        expression: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        if self.language != "typescript":
            return None
        called = _simple_call(expression)
        if called is None:
            return None
        simple_name, arguments = called
        canonical = next(
            (
                imported
                for imported, alias in self.import_aliases.items()
                if alias == simple_name
            ),
            "",
        )
        callee_match = re.match(
            rf"(?P<callee>{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)\s*\(",
            expression,
        )
        callee = (
            re.sub(r"\s+", "", callee_match.group("callee"))
            if callee_match is not None
            else ""
        )
        receiver = callee.rsplit(".", 1)[0] if "." in callee else ""
        if not canonical and receiver in self.crypto_namespace_aliases:
            canonical = simple_name
        if canonical not in {"pbkdf2Sync", "scryptSync"} or len(arguments) < 2:
            return None

        password = self._resolve(
            arguments[0],
            index,
            mode="origin",
            seen=seen,
            bindings=bindings,
        )
        salt = self._resolve(
            arguments[1],
            index,
            mode="origin",
            seen=seen,
            bindings=bindings,
        )
        steps = (*password.steps, *salt.steps)
        if password.origin in {"constant", "predictable", "non_csprng"}:
            return ResolvedValue(expression, "predictable", steps=steps)
        if password.origin in {
            "runtime_input",
            "secure_random",
            "derived_secure",
        }:
            return ResolvedValue(expression, "derived_secure", steps=steps)
        return ResolvedValue(
            expression,
            "unknown",
            steps=steps,
            unresolved_reason=(
                f"{canonical} password or salt origin could not be resolved"
            ),
        )

    def _resolve_csharp_derived_origin(
        self,
        expression: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        if self.language != "csharp":
            return None
        direct_constructor = re.fullmatch(
            r"new\s+(?:System\s*\.\s*Security\s*\.\s*Cryptography\s*\.\s*)?"
            r"Rfc2898DeriveBytes\s*\((?P<arguments>.*)\)\s*\.\s*"
            r"GetBytes\s*\([^)]*\)",
            expression,
            flags=re.DOTALL,
        )
        if direct_constructor is not None:
            constructor_arguments = _split_inline_arguments(
                direct_constructor.group("arguments")
            )
            if len(constructor_arguments) < 2:
                return None
            password_expression = _c_style_argument_value(
                constructor_arguments,
                0,
                "password",
            )
            salt_expression = _c_style_argument_value(
                constructor_arguments,
                1,
                "salt",
            )
            password = self._resolve(
                password_expression,
                index,
                mode="origin",
                seen=seen,
                bindings=bindings,
            )
            salt_size = self._resolve(
                salt_expression,
                index,
                mode="int",
                seen=seen,
                bindings=bindings,
            )
            if salt_size.origin == "constant" and salt_size.values:
                salt = ResolvedValue(
                    salt_expression,
                    "secure_random",
                    steps=(
                        *salt_size.steps,
                        ResolutionStep(
                            salt_expression,
                            "resolve provider-generated Rfc2898 salt",
                            "secure_random",
                            self.source_span(index, index + len(expression)),
                        ),
                    ),
                )
            else:
                salt = self._resolve(
                    salt_expression,
                    index,
                    mode="origin",
                    seen=seen,
                    bindings=bindings,
                )
            steps = (*password.steps, *salt.steps)
            if password.origin in {"constant", "predictable", "non_csprng"}:
                return ResolvedValue(expression, "predictable", steps=steps)
            if password.origin in {
                "runtime_input",
                "secure_random",
                "derived_secure",
            }:
                return ResolvedValue(expression, "derived_secure", steps=steps)
            return ResolvedValue(
                expression,
                "unknown",
                steps=steps,
                unresolved_reason=(
                    "Rfc2898DeriveBytes password or salt origin could not be resolved"
                ),
            )
        match = re.fullmatch(
            rf"(?P<callee>{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)"
            r"\s*\((?P<arguments>.*)\)",
            expression,
            flags=re.DOTALL,
        )
        if match is None:
            return None
        callee = re.sub(r"\s+", "", match.group("callee"))
        arguments = _split_inline_arguments(match.group("arguments"))
        simple_name = callee.rsplit(".", 1)[-1]
        if simple_name in {"Next", "NextInt64", "NextDouble", "NextSingle"}:
            receiver = callee.rsplit(".", 1)[0] if "." in callee else ""
            if receiver == "Random.Shared":
                return ResolvedValue(expression, "non_csprng")
            receiver_origin = (
                self._resolve(
                    receiver,
                    index,
                    mode="origin",
                    seen=seen,
                    bindings=bindings,
                )
                if receiver
                else None
            )
            if receiver_origin is not None and receiver_origin.origin in {
                "non_csprng",
                "predictable",
                "constant",
            }:
                return ResolvedValue(
                    expression,
                    "non_csprng",
                    steps=receiver_origin.steps,
                )
        password_expression: str | None = None
        salt_expression: str | None = None

        if callee.endswith("Rfc2898DeriveBytes.Pbkdf2") and len(arguments) >= 2:
            password_expression = _c_style_argument_value(
                arguments,
                0,
                "password",
            )
            salt_expression = _c_style_argument_value(arguments, 1, "salt")
        elif callee.endswith(".GetBytes") and arguments:
            receiver = callee.rsplit(".", 1)[0]
            assignment = self.latest_assignment(receiver, index)
            if assignment is None:
                return None
            constructor = re.fullmatch(
                r"new\s+(?:System\s*\.\s*Security\s*\.\s*Cryptography\s*\.)?"
                r"Rfc2898DeriveBytes\s*\((?P<arguments>.*)\)",
                _strip_expression(assignment.expression),
                flags=re.DOTALL,
            )
            if constructor is None:
                return None
            constructor_arguments = _split_inline_arguments(
                constructor.group("arguments")
            )
            if len(constructor_arguments) < 2:
                return None
            password_expression = _c_style_argument_value(
                constructor_arguments,
                0,
                "password",
            )
            salt_expression = _c_style_argument_value(
                constructor_arguments,
                1,
                "salt",
            )
            index = assignment.start
        else:
            return None

        password = self._resolve(
            password_expression,
            index,
            mode="origin",
            seen=seen,
            bindings=bindings,
        )
        salt = self._resolve(
            salt_expression,
            index,
            mode="origin",
            seen=seen,
            bindings=bindings,
        )
        steps = (*password.steps, *salt.steps)
        if password.origin in {"constant", "predictable", "non_csprng"}:
            return ResolvedValue(expression, "predictable", steps=steps)
        if password.origin in {
            "runtime_input",
            "secure_random",
            "derived_secure",
        }:
            return ResolvedValue(expression, "derived_secure", steps=steps)
        return ResolvedValue(
            expression,
            "unknown",
            steps=steps,
            unresolved_reason=(
                "Rfc2898DeriveBytes password or salt origin could not be resolved"
            ),
        )

    def _csharp_span_wrapper_argument(self, expression: str) -> str | None:
        if self.language != "csharp":
            return None
        match = re.fullmatch(
            rf"new\s+(?P<name>{_IDENTIFIER})\s*\((?P<arguments>.*)\)",
            expression,
            flags=re.DOTALL,
        )
        if match is None:
            return None
        arguments = _split_inline_arguments(match.group("arguments"))
        if len(arguments) != 1:
            return None
        class_definition = next(
            (
                item
                for item in self.classes
                if item.name == match.group("name")
            ),
            None,
        )
        if class_definition is None:
            return None
        body = self.source[class_definition.start : class_definition.end]
        if re.search(
            r"\b(?:ReadOnly)?Span\s*<[^>]+>\s+Span\s*=>",
            body,
        ) is None:
            return None
        return arguments[0]

    def _resolve_parameter(
        self,
        reference: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        # Nested TypeScript helpers may close over parameters from an outer
        # function, such as ``next = () => argv[++i]`` inside
        # ``parseArgs(argv)``. Select the narrowest enclosing declaration that
        # actually owns the reference.
        declaring_functions = [
            function
            for function in self.functions
            if function.start <= index <= function.end
            and reference in function.parameters
        ]
        if not declaring_functions:
            return None
        containing = min(
            declaring_functions,
            key=lambda item: item.end - item.start,
        )
        bound = (
            bindings.get((containing.start, reference))
            if bindings is not None
            else None
        )
        if bound is None:
            parameter_key = f"parameter:{containing.start}:{reference}:global"
        else:
            bound_expression, caller_index = bound
            bound_identity = re.sub(r"\s+", "", bound_expression)
            parameter_key = (
                f"parameter:{containing.start}:{reference}:"
                f"{caller_index}:{bound_identity}"
            )
        if parameter_key in seen:
            return None
        exclude_empty_paths = mode == "origin" and (
            self._fail_closed_nonempty_guard(reference, index)
            or self._reference_is_truthy_on_path(reference, index)
        )

        def resolve_argument(
            expression: str,
            caller_index: int,
        ) -> ResolvedValue:
            if exclude_empty_paths:
                resolved = self._resolve_origin_without_empty_paths(
                    expression,
                    caller_index,
                    seen=seen | frozenset({parameter_key}),
                    bindings=bindings,
                )
                if resolved is not None:
                    return resolved
            return self._resolve(
                expression,
                caller_index,
                mode=mode,
                seen=seen | frozenset({parameter_key}),
                bindings=bindings,
            )

        if bound is not None:
            expression, caller_index = bound
            return resolve_argument(expression, caller_index)
        if (
            mode == "origin"
            and self.language == "csharp"
            and containing.name.lower() == "main"
        ):
            return ResolvedValue(
                reference,
                "runtime_input",
                steps=(
                    ResolutionStep(
                        reference,
                        "resolve C# entrypoint argument",
                        "runtime_input",
                        self.source_span(containing.start, containing.end),
                    ),
                ),
            )
        parameter_index = containing.parameters.index(reference)
        function_call_name = containing.name
        if containing.name == "constructor":
            function_call_name = (
                self._containing_class_name(containing.start) or containing.name
            )
        default_expression = _c_style_parameter_default(
            self.source,
            self.masked,
            containing,
            reference,
        )
        callers = [
            call
            for call in self.calls(function_call_name)
            if not (containing.start <= call.start <= containing.end)
            and self._call_can_target_function(call, containing)
        ]
        if self.language == "typescript" and containing.name == "constructor":
            owner = self._containing_class_name(containing.start)
            if owner is not None:
                subclasses = frozenset(
                    re.findall(
                        rf"\bclass\s+([A-Za-z_$][A-Za-z0-9_$]*)\s+extends\s+"
                        rf"{re.escape(owner)}\b",
                        self.masked,
                    )
                )
                callers.extend(
                    call
                    for call in self.calls("super")
                    if (
                        (subclass_constructor := self._containing_function(call.start))
                        is not None
                        and subclass_constructor.name == "constructor"
                        and self._containing_class_name(subclass_constructor.start)
                        in subclasses
                    )
                )
        # Reachability conditions may themselves reference this parameter.
        # Re-entering caller reachability from that condition creates an
        # unbounded parameter -> condition -> caller cycle.  In that narrow
        # context retain every candidate caller; the outer reachability proof
        # will still decide whether the path is active.
        if self._reachability_depth == 0:
            reachable_callers = [
                call for call in callers if self._call_is_reachable(call)
            ]
            if reachable_callers:
                callers = reachable_callers
        if not callers:
            if default_expression is None:
                return None
            return resolve_argument(default_expression, containing.start)
        values: list[ResolvedValue] = []
        for call in callers:
            argument = self._c_style_argument_for_parameter(
                call,
                parameter_index,
                reference,
            ) or default_expression
            if argument is None:
                continue
            if (
                mode == "origin"
                and re.search(r"password|passphrase|pwd", reference, re.IGNORECASE)
                and self._is_explicit_negative_password_test_call(call, argument)
            ):
                continue
            values.append(
                resolve_argument(argument, call.start)
            )
        return _merge_resolved(reference, values) if values else None

    def _resolve_callable_parameter_call(
        self,
        call_name: str,
        call_arguments: tuple[str, ...],
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        declaring_functions = [
            function
            for function in self.functions
            if function.start <= index <= function.end
            and call_name in function.parameters
        ]
        if not declaring_functions:
            return None
        containing = min(
            declaring_functions,
            key=lambda item: item.end - item.start,
        )
        parameter_index = containing.parameters.index(call_name)
        parameter_key = f"callable-parameter:{containing.start}:{call_name}:{index}"
        if parameter_key in seen:
            return None

        sources: list[tuple[str, int]] = []
        bound = (
            bindings.get((containing.start, call_name))
            if bindings is not None
            else None
        )
        if bound is not None:
            sources.append(bound)
        else:
            function_call_name = containing.name
            if containing.name == "constructor":
                function_call_name = (
                    self._containing_class_name(containing.start)
                    or containing.name
                )
            default_expression = _c_style_parameter_default(
                self.source,
                self.masked,
                containing,
                call_name,
            )
            for call in self.calls(function_call_name):
                if containing.start <= call.start <= containing.end:
                    continue
                if not self._call_can_target_function(call, containing):
                    continue
                argument = self._c_style_argument_for_parameter(
                    call,
                    parameter_index,
                    call_name,
                ) or default_expression
                if argument is not None:
                    sources.append((argument, call.start))

        values: list[ResolvedValue] = []
        for target_expression, caller_index in dict.fromkeys(sources):
            target = _simple_reference(_strip_expression(target_expression))
            if target is None or target.rsplit(".", 1)[-1] == call_name:
                continue
            invocation = f"{target}({', '.join(call_arguments)})"
            values.append(
                self._resolve(
                    invocation,
                    caller_index,
                    mode=mode,
                    seen=seen | frozenset({parameter_key}),
                    bindings=bindings,
                )
            )
        return _merge_resolved(call_name, values) if values else None

    def _resolve_destructured_parameter(
        self,
        reference: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        """Resolve a TypeScript object-pattern parameter at its callsites."""

        if self.language != "typescript" or "." in reference:
            return None
        candidates = [
            (function, binding)
            for function in self.functions
            if function.start <= index <= function.end
            if (
                binding := _c_style_destructured_parameter(
                    self.source,
                    self.masked,
                    function,
                    reference,
                )
            )
            is not None
        ]
        if not candidates:
            return None
        containing, (parameter_index, property_name, property_default) = min(
            candidates,
            key=lambda item: item[0].end - item[0].start,
        )
        key = (
            f"destructured-parameter:{containing.start}:"
            f"{reference}:{property_name}"
        )
        if key in seen:
            return None
        call_name = containing.name
        if call_name == "constructor":
            call_name = self._containing_class_name(containing.start) or call_name
        callers = [
            call
            for call in self.calls(call_name)
            if not (containing.start <= call.start <= containing.end)
            and self._call_can_target_function(call, containing)
        ]
        if self._reachability_depth == 0:
            reachable = [call for call in callers if self._call_is_reachable(call)]
            if reachable:
                callers = reachable

        values: list[ResolvedValue] = []
        for call in callers:
            argument = (
                call.arguments[parameter_index]
                if parameter_index < len(call.arguments)
                else None
            )
            if argument is None:
                if property_default is not None:
                    values.append(
                        self._resolve(
                            property_default,
                            call.start,
                            mode=mode,
                            seen=seen | frozenset({key}),
                            bindings=bindings,
                        )
                    )
                continue
            resolved = self._resolve_object_property_expression(
                argument,
                property_name,
                call.start,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
            if resolved is None and property_default is not None:
                resolved = self._resolve(
                    property_default,
                    call.start,
                    mode=mode,
                    seen=seen | frozenset({key}),
                    bindings=bindings,
                )
            if resolved is not None:
                values.append(resolved)
        if values:
            return _merge_resolved(reference, values)
        if property_default is not None:
            return self._resolve(
                property_default,
                containing.start,
                mode=mode,
                seen=seen | frozenset({key}),
                bindings=bindings,
            )
        return None

    def _resolve_typescript_iife_parameter(
        self,
        reference: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        """Resolve a parameter of ``((value) => { ... })(argument)``."""

        if self.language != "typescript":
            return None
        candidates: list[tuple[int, str, int]] = []
        for match in re.finditer(
            r"\(\s*(?:async\s+)?\(\s*(?P<parameters>[^()]*)\)\s*=>\s*\{",
            self.masked,
        ):
            body_open = match.end() - 1
            body_close = _find_matching_masked(
                self.masked,
                body_open,
                "{",
                "}",
            )
            if body_close == -1 or not body_open < index < body_close:
                continue
            cursor = body_close + 1
            while cursor < len(self.masked) and self.masked[cursor].isspace():
                cursor += 1
            if cursor >= len(self.masked) or self.masked[cursor] != ")":
                continue
            cursor += 1
            while cursor < len(self.masked) and self.masked[cursor].isspace():
                cursor += 1
            if cursor >= len(self.masked) or self.masked[cursor] != "(":
                continue
            call_close = _find_matching_masked(
                self.masked,
                cursor,
                "(",
                ")",
            )
            if call_close == -1:
                continue
            parameter_names = tuple(
                _parameter_name(parameter)
                for parameter in _split_inline_arguments(
                    match.group("parameters")
                )
            )
            if reference not in parameter_names:
                continue
            arguments = _split_inline_arguments(
                self.source[cursor + 1 : call_close]
            )
            parameter_index = parameter_names.index(reference)
            if parameter_index >= len(arguments):
                continue
            candidates.append(
                (body_open, arguments[parameter_index], cursor + 1)
            )
        if not candidates:
            return None
        body_open, argument, argument_index = max(
            candidates,
            key=lambda item: item[0],
        )
        binding_key = f"typescript-iife-parameter:{body_open}:{reference}"
        if binding_key in seen:
            return None

        # ``((key) => use(key))(key)`` contains two distinct lexical bindings
        # even though both identifiers have the same spelling. Arrow IIFEs are
        # not part of the regular function inventory, so the generic variable
        # cycle key otherwise conflates the inner parameter with the outer
        # invocation argument.
        outer_function = self._containing_function(argument_index)
        outer_scope = (
            outer_function.start if outer_function is not None else "global"
        )
        outer_reference_key = (
            f"variable:{outer_scope}:{''.join(reference.split())}"
        )
        outer_seen = frozenset(
            item for item in seen if item != outer_reference_key
        ) | frozenset({binding_key})
        return self._resolve(
            argument,
            argument_index,
            mode=mode,
            seen=outer_seen,
            bindings=bindings,
        )

    def _resolve_typescript_iife_result_parameter(
        self,
        expression: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        """Resolve an IIFE whose direct returns forward invocation arguments.

        Bookkeeping inside an IIFE can contain unrelated values such as an
        expiry timestamp.  Those values must not determine the origin of an
        argument that the IIFE stores and returns unchanged.
        """

        if self.language != "typescript":
            return None
        masked = mask_c_style_comments_and_literals(expression)
        match = re.match(
            r"^\s*\(\s*\(\s*(?:async\s+)?(?P<parameters>[^()]*)\)"
            r"\s*(?::\s*[^=]+)?=>\s*\{",
            masked,
        )
        if match is None:
            return None
        body_open = match.end() - 1
        body_close = _find_matching_masked(masked, body_open, "{", "}")
        if body_close == -1:
            return None
        cursor = body_close + 1
        while cursor < len(masked) and masked[cursor].isspace():
            cursor += 1
        if cursor >= len(masked) or masked[cursor] != ")":
            return None
        cursor += 1
        while cursor < len(masked) and masked[cursor].isspace():
            cursor += 1
        if cursor >= len(masked) or masked[cursor] != "(":
            return None
        call_close = _find_matching_masked(masked, cursor, "(", ")")
        if call_close == -1 or masked[call_close + 1 :].strip():
            return None

        parameters = tuple(
            _parameter_name(parameter)
            for parameter in _split_inline_arguments(match.group("parameters"))
        )
        arguments = _split_inline_arguments(expression[cursor + 1 : call_close])
        if len(arguments) < len(parameters):
            return None

        body_masked = masked[body_open + 1 : body_close]
        body_source = expression[body_open + 1 : body_close]
        returned_expressions: list[tuple[str, int]] = []
        brace_depth = 0
        for token in re.finditer(r"[{}]|\breturn\b", body_masked):
            value = token.group(0)
            if value == "{":
                brace_depth += 1
                continue
            if value == "}":
                brace_depth = max(0, brace_depth - 1)
                continue
            if brace_depth:
                continue
            end = _next_c_style_statement_end_masked(
                body_masked,
                token.end(),
                "typescript",
            )
            if end == -1:
                return None
            returned = _strip_expression(body_source[token.end() : end])
            returned_expressions.append((returned, token.end()))
        if not returned_expressions:
            return None

        local = CStyleAnalyzer(body_source, "typescript")
        resolved: list[ResolvedValue] = []
        for returned, return_index in returned_expressions:
            if returned in parameters:
                resolved.append(
                    self._resolve(
                        arguments[parameters.index(returned)],
                        index + cursor + 1,
                        mode=mode,
                        seen=seen,
                        bindings=bindings,
                    )
                )
            else:
                resolved.append(
                    local._resolve(
                        returned,
                        return_index,
                        mode=mode,
                        seen=frozenset(),
                        bindings=None,
                    )
                )
        return resolved[0] if len(resolved) == 1 else _merge_resolved(expression, resolved)

    def _is_explicit_negative_password_test_call(
        self,
        call: CallExpression,
        argument: str,
    ) -> bool:
        literal = _string_literal(_strip_expression(argument))
        if literal is None or re.search(
            r"wrong|invalid|incorrect|bad|mismatch",
            literal,
            re.IGNORECASE,
        ) is None:
            return False
        statement_start = max(
            self.masked.rfind(";", 0, call.start),
            self.masked.rfind("{", 0, call.start),
            self.masked.rfind("}", 0, call.start),
            self.masked.rfind("\n", 0, call.start),
        ) + 1
        prefix = self.source[statement_start:call.start]
        receiving_name = re.search(
            rf"\b(?P<name>{_IDENTIFIER})\s*(?::[^=]+)?=\s*$",
            prefix,
            flags=re.DOTALL,
        )
        return receiving_name is not None and re.search(
            r"wrong|invalid|incorrect|bad|mismatch",
            receiving_name.group("name"),
            re.IGNORECASE,
        ) is not None

    def _call_is_reachable(
        self,
        call: CallExpression,
    ) -> bool:
        if self._reachability_depth >= _MAX_RESOLUTION_STEPS:
            return True
        self._reachability_depth += 1
        try:
            return self._call_has_entry_path(call)
        finally:
            self._reachability_depth -= 1

    def _call_has_entry_path(
        self,
        call: CallExpression,
    ) -> bool:
        pending = [call]
        visited_calls: set[int] = set()
        visited_functions: set[tuple[int, str]] = set()

        while pending:
            current = pending.pop()
            if current.start in visited_calls:
                # Re-entering the same call is a cycle, not proof of dead code.
                return True
            visited_calls.add(current.start)
            if not _c_style_direct_path_is_reachable(self, current.start):
                continue

            caller = self._containing_function(current.start)
            if caller is None or caller.name.lower() == "main":
                return True
            call_name = caller.name
            if call_name == "constructor":
                call_name = self._containing_class_name(caller.start) or call_name
            key = (caller.start, call_name)
            if key in visited_functions:
                return True
            visited_functions.add(key)
            if len(visited_functions) >= _MAX_RESOLUTION_STEPS:
                # A very large or cyclic generated call graph is not proof that
                # the sink is dead, so retain it conservatively.
                return True
            callers = [
                candidate
                for candidate in self.calls(call_name)
                if not (caller.start <= candidate.start <= caller.end)
                and self._call_can_target_function(candidate, caller)
            ]
            callers.extend(self._callback_registrations_for_function(caller))
            if not callers:
                if self._entrypoint_check_active:
                    # JavaScript/TypeScript entrypoints are commonly invoked
                    # from a top-level anonymous event callback rather than at
                    # module scope. The callback registration is the root.
                    return caller.name == "function"
                # A source without an executable entrypoint can be a library
                # snippet whose public method is the root. Preserve that rule
                # across helper-call hops, not only at the original sink.
                return not self._has_explicit_entrypoint()
            pending.extend(callers)
        return False

    def _parameter_is_always_omitted(
        self,
        expression: str,
        index: int,
    ) -> bool:
        reference = _simple_reference(_strip_expression(expression))
        containing = self._containing_function(index)
        if (
            reference is None
            or "." in reference
            or containing is None
            or reference not in containing.parameters
        ):
            return False
        parameter_index = containing.parameters.index(reference)
        call_name = containing.name
        if call_name == "constructor":
            call_name = self._containing_class_name(containing.start) or call_name
        callers = [
            call
            for call in self.calls(call_name)
            if not (containing.start <= call.start <= containing.end)
            and self._call_can_target_function(call, containing)
        ]
        return bool(callers) and all(
            self._c_style_argument_for_parameter(
                call,
                parameter_index,
                reference,
            )
            is None
            for call in callers
        )

    def _call_can_target_function(
        self,
        call: CallExpression,
        function: FunctionDefinition,
    ) -> bool:
        compact = re.sub(r"\s+", "", call.callee)
        compact = re.sub(r"^new", "", compact)
        if "." not in compact:
            return True

        owner = self._containing_class_name(function.start)
        if owner is None:
            if self.language != "typescript":
                return False
            object_property = re.search(
                rf"\b{re.escape(function.name)}\s*:\s*(?:async\s+)?$",
                self.masked[max(0, function.start - 96) : function.start],
            )
            return object_property is not None
        receiver = compact.rsplit(".", 1)[0]
        receiver_name = receiver.rsplit(".", 1)[-1]
        if receiver_name in {"this", "base"}:
            return self._containing_class_name(call.start) == owner
        if receiver_name == owner:
            return True
        receiver_assignment = self.latest_assignment(receiver_name, call.start)
        if receiver_assignment is not None:
            factory = _simple_call(_strip_expression(receiver_assignment.expression))
            if factory is not None:
                factory_name, _ = factory
                for definition in self.function(factory_name):
                    if any(
                        re.match(
                            rf"new\s+{re.escape(owner)}\s*\(",
                            _strip_expression(returned.expression),
                        )
                        for returned in definition.returns
                    ):
                        return True
        return bool(
            re.search(
                rf"\b{re.escape(owner)}\s+{re.escape(receiver_name)}\b"
                rf"|\b(?:const|let|var)\s+{re.escape(receiver_name)}"
                rf"(?:\s*:[^=;]+)?\s*=\s*(?:new\s+)?{re.escape(owner)}"
                rf"(?:\s*\(|\s*\.)"
                rf"|\b(?:(?:public|private|protected|readonly|static)\s+)*"
                rf"{re.escape(receiver_name)}\s*=\s*new\s+"
                rf"{re.escape(owner)}\s*\("
                rf"|\b(?:(?:public|private|protected|readonly|static)\s+)*"
                rf"{re.escape(receiver_name)}\s*[!?]?\s*:\s*"
                rf"{re.escape(owner)}\b",
                self.source,
            )
        )

    def _containing_class_name(self, index: int) -> str | None:
        matches = [
            definition
            for definition in self.classes
            if definition.start <= index <= definition.end
        ]
        if not matches:
            return None
        return min(matches, key=lambda item: item.end - item.start).name

    @staticmethod
    def _c_style_argument_for_parameter(
        call: CallExpression,
        parameter_index: int,
        parameter_name: str,
    ) -> str | None:
        named = call.named_arguments.get(parameter_name)
        if named is not None:
            return named
        if parameter_index < len(call.arguments):
            ordinal = call.arguments[parameter_index]
            ordinal_named = re.match(
                rf"^(?P<name>{_IDENTIFIER})\s*:\s*(?P<value>.+)$",
                ordinal,
                flags=re.DOTALL,
            )
            if ordinal_named is None:
                return ordinal
            if ordinal_named.group("name") == parameter_name:
                return ordinal_named.group("value").strip()
        positional = tuple(
            argument
            for argument in call.arguments
            if not re.match(rf"^{_IDENTIFIER}\s*:", argument)
        )
        return (
            positional[parameter_index]
            if parameter_index < len(positional)
            else None
        )
    def _resolve_csharp_array_copy_source(
        self,
        call: CallExpression,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        if len(call.arguments) >= 5:
            source, source_offset, _, _, count = call.arguments[:5]
        elif len(call.arguments) >= 3:
            source, _, count = call.arguments[:3]
            source_offset = "0"
        else:
            return None
        if _evaluate_c_style_int(source_offset) not in {0, None}:
            return None

        source_reference = _simple_reference(_strip_expression(source))
        if source_reference is not None:
            requested_count = re.sub(r"\s+", "", count)
            for prior in reversed(self.calls("Copy")):
                if prior.start >= call.start or len(prior.arguments) < 3:
                    continue
                compact = re.sub(r"\s+", "", prior.callee)
                if re.search(r"(?:^|\.)Array\.Copy$", compact) is None:
                    continue
                if len(prior.arguments) >= 5:
                    prior_source = prior.arguments[0]
                    prior_destination = prior.arguments[2]
                    prior_destination_offset = prior.arguments[3]
                    prior_count = prior.arguments[4]
                else:
                    prior_source = prior.arguments[0]
                    prior_destination = prior.arguments[1]
                    prior_destination_offset = "0"
                    prior_count = prior.arguments[2]
                if (
                    _normalize_reference(prior_destination)
                    != _normalize_reference(source_reference)
                    or _evaluate_c_style_int(prior_destination_offset) != 0
                    or re.sub(r"\s+", "", prior_count) != requested_count
                ):
                    continue
                return self._resolve(
                    prior_source,
                    prior.start,
                    mode="origin",
                    seen=seen,
                    bindings=bindings,
                )

            assignment = self.latest_assignment(source_reference, call.start)
            if assignment is not None:
                first_serialized_field = _csharp_first_concatenated_conversion_input(
                    assignment.expression
                )
                if first_serialized_field is None:
                    converted_reference = _csharp_conversion_input_reference(
                        assignment.expression
                    )
                    if converted_reference is not None:
                        serialized_assignment = self.latest_assignment(
                            converted_reference,
                            assignment.start,
                        )
                        if serialized_assignment is not None:
                            concatenation = _top_level_c_style_concatenation(
                                serialized_assignment.expression
                            )
                            if concatenation is not None:
                                first_serialized_field = concatenation[0]
                if first_serialized_field is not None:
                    return self._resolve(
                        first_serialized_field,
                        assignment.start,
                        mode="origin",
                        seen=seen,
                        bindings=bindings,
                    )
        return None

    def _resolve_csharp_file_stream_round_trip(
        self,
        receiver: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        """Trace a constant-path FileStream read back to a prior file write."""

        reference = _simple_reference(receiver)
        if reference is None:
            return None
        assignment = self.latest_assignment(reference, index)
        if assignment is None:
            return None
        stream = re.match(
            r"(?:System\s*\.\s*IO\s*\.)?File\s*\.\s*"
            r"(?:OpenRead|Open)\s*\(\s*(?P<path>[^,)]+)",
            _strip_expression(assignment.expression),
            flags=re.DOTALL,
        )
        if stream is None:
            return None
        read_path = self._resolve(
            stream.group("path"),
            assignment.start,
            mode="string",
            seen=seen,
            bindings=bindings,
        )
        values: list[ResolvedValue] = []
        for call in self.calls("WriteAllBytes"):
            if call.start >= assignment.start or len(call.arguments) < 2:
                continue
            write_path = self._resolve(
                call.arguments[0],
                call.start,
                mode="string",
                seen=seen,
                bindings=bindings,
            )
            if not set(read_path.values) & set(write_path.values):
                continue
            values.append(
                self._resolve(
                    call.arguments[1],
                    call.start,
                    mode="origin",
                    seen=seen,
                    bindings=bindings,
                )
            )
        if values:
            return _merge_resolved(receiver, values)

        read_calls = tuple(
            call
            for call in self.calls("Read", "ReadAsync")
            if call.start <= index
            and re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
            == re.sub(r"\s+", "", receiver)
        )
        read_ordinal = max(0, len(read_calls) - 1)
        for writer_assignment in self.assignments:
            if writer_assignment.start >= assignment.start:
                continue
            writer_match = re.match(
                r"(?:System\s*\.\s*IO\s*\.)?File\s*\.\s*"
                r"(?:Create|OpenWrite)\s*\(\s*(?P<path>[^,)]+)",
                _strip_expression(writer_assignment.expression),
                flags=re.DOTALL,
            )
            if writer_match is None:
                continue
            write_path = self._resolve(
                writer_match.group("path"),
                writer_assignment.start,
                mode="string",
                seen=seen,
                bindings=bindings,
            )
            if not set(read_path.values) & set(write_path.values):
                continue
            writer_reference = _compact_reference(writer_assignment.name)
            writes = tuple(
                call
                for call in self.calls("Write", "WriteAsync")
                if writer_assignment.end <= call.start < assignment.start
                and re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
                == writer_reference
                and call.arguments
            )
            if read_ordinal >= len(writes):
                continue
            return self._resolve(
                writes[read_ordinal].arguments[0],
                writes[read_ordinal].start,
                mode="origin",
                seen=seen,
                bindings=bindings,
            )
        return _merge_resolved(receiver, values) if values else None

    def _resolve_csharp_memory_stream_bytes(
        self,
        expression: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        """Trace a local MemoryStream.ToArray() to bytes written into it."""

        if self.language != "csharp":
            return None
        match = re.fullmatch(
            rf"(?P<stream>{_IDENTIFIER})\s*\.\s*ToArray\s*\(\s*\)",
            expression,
        )
        if match is None:
            return None
        stream = _normalize_reference(match.group("stream"))
        writer_names = {
            _normalize_reference(assignment.name)
            for assignment in self.assignments
            if assignment.start < index
            and (
                arguments := _csharp_constructor_arguments(
                    assignment.expression,
                    "BinaryWriter",
                )
            )
            and _simple_reference(arguments[0]) is not None
            and _normalize_reference(arguments[0]) == stream
        }
        values: list[ResolvedValue] = []
        for call in self.calls("Write", "WriteAsync"):
            if call.start >= index or not call.arguments:
                continue
            compact = re.sub(r"\s+", "", call.callee)
            receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
            if _normalize_reference(receiver) not in writer_names | {stream}:
                continue
            values.append(
                self._resolve(
                    call.arguments[0],
                    call.start,
                    mode="origin",
                    seen=seen,
                    bindings=bindings,
                )
            )
        return _merge_resolved(expression, values) if values else None

    def _resolve_csharp_serialized_role(
        self,
        expression: str,
        index: int,
        *,
        role: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        """Find a named field copied through a locally serialized envelope."""

        if self.language != "csharp" or role not in {"salt"}:
            return None
        expr = _strip_expression(expression)
        state_key = f"serialized-role:{index}:{role}:{expr}"
        if state_key in seen:
            return None
        next_seen = seen | frozenset({state_key})
        reference = _simple_reference(expr)
        values: list[ResolvedValue] = []
        if reference is not None:
            normalized = _normalize_reference(reference)
            if normalized.rsplit(".", 1)[-1].lower() == role:
                return self._resolve(
                    expr,
                    index,
                    mode="origin",
                    seen=next_seen,
                    bindings=bindings,
                )

            containing = self._containing_function(index)
            if containing is not None and reference in containing.parameters:
                parameter_index = containing.parameters.index(reference)
                caller_arguments: list[tuple[str, int]] = []
                if bindings is not None:
                    bound = bindings.get((containing.start, reference))
                    if bound is not None:
                        caller_arguments.append(bound)
                for caller in self.calls(containing.name):
                    if containing.start <= caller.start <= containing.end:
                        continue
                    if not self._call_can_target_function(caller, containing):
                        continue
                    argument = self._c_style_argument_for_parameter(
                        caller,
                        parameter_index,
                        reference,
                    )
                    if argument is not None:
                        caller_arguments.append((argument, caller.start))
                for argument, caller_index in dict.fromkeys(caller_arguments):
                    resolved = self._resolve_csharp_serialized_role(
                        argument,
                        caller_index,
                        role=role,
                        seen=next_seen,
                        bindings=bindings,
                    )
                    if resolved is not None:
                        values.append(resolved)

            assignment = self.latest_assignment(reference, index)
            if assignment is not None:
                resolved = self._resolve_csharp_serialized_role(
                    assignment.expression,
                    assignment.start,
                    role=role,
                    seen=next_seen,
                    bindings=bindings,
                )
                if resolved is not None:
                    values.append(resolved)

            for copy_call in self.calls("BlockCopy"):
                compact = re.sub(r"\s+", "", copy_call.callee)
                if (
                    copy_call.start >= index
                    or len(copy_call.arguments) < 3
                    or re.search(r"(?:^|\.)Buffer\.BlockCopy$", compact) is None
                    or not self.position_is_in_same_or_enclosing_function(
                        copy_call.start,
                        index,
                    )
                ):
                    continue
                target = _mutation_target_reference(copy_call.arguments[2])
                if target is None or _normalize_reference(target) != normalized:
                    continue
                resolved = self._resolve_csharp_serialized_role(
                    copy_call.arguments[0],
                    copy_call.start,
                    role=role,
                    seen=next_seen,
                    bindings=bindings,
                )
                if resolved is not None:
                    values.append(resolved)
            if values:
                return _merge_resolved(expr, values)

        stream_match = re.fullmatch(
            rf"(?P<stream>{_IDENTIFIER})\s*\.\s*ToArray\s*\(\s*\)",
            expr,
        )
        if stream_match is None:
            return None
        stream = _normalize_reference(stream_match.group("stream"))
        writer_names = {
            _normalize_reference(assignment.name)
            for assignment in self.assignments
            if assignment.start < index
            and (
                arguments := _csharp_constructor_arguments(
                    assignment.expression,
                    "BinaryWriter",
                )
            )
            and _simple_reference(arguments[0]) is not None
            and _normalize_reference(arguments[0]) == stream
        }
        for write_call in self.calls("Write", "WriteAsync"):
            if write_call.start >= index or not write_call.arguments:
                continue
            compact = re.sub(r"\s+", "", write_call.callee)
            receiver = compact.rsplit(".", 1)[0] if "." in compact else ""
            if _normalize_reference(receiver) not in writer_names | {stream}:
                continue
            written = _simple_reference(write_call.arguments[0])
            if (
                written is None
                or _normalize_reference(written).rsplit(".", 1)[-1].lower()
                != role
            ):
                continue
            return self._resolve(
                write_call.arguments[0],
                write_call.start,
                mode="origin",
                seen=next_seen,
                bindings=bindings,
            )
        return None

    def _resolve_csharp_hash_property(
        self,
        expression: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        """Trace HashAlgorithm.Hash to the preceding ComputeHash input."""

        if self.language != "csharp":
            return None
        match = re.fullmatch(
            rf"(?P<receiver>{_IDENTIFIER})\s*\.\s*Hash",
            expression,
        )
        if match is None:
            return None
        receiver = _normalize_reference(match.group("receiver"))
        values = [
            self._resolve(
                call.arguments[0],
                call.start,
                mode="origin",
                seen=seen,
                bindings=bindings,
            )
            for call in self.calls("ComputeHash", "HashData")
            if call.start < index
            and call.arguments
            and "." in re.sub(r"\s+", "", call.callee)
            and _normalize_reference(
                re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
            )
            == receiver
        ]
        return _merge_transformed_origins(expression, values) if values else None

    def _csharp_static_constructor_is_reached(
        self,
        function: FunctionDefinition,
    ) -> bool:
        """Recognize the implicit call to a C# static type initializer."""

        if self.language != "csharp" or function.parameters:
            return False
        owner = self._containing_class_name(function.start)
        if owner is None or function.name != owner:
            return False
        if any(
            candidate.name.lower() == "main"
            and self._containing_class_name(candidate.start) == owner
            for candidate in self.functions
        ):
            return True
        return any(
            call.start < function.start or call.start > function.end
            for call in self.calls()
            if re.sub(r"\s+", "", call.callee).startswith(f"{owner}.")
            and self._call_is_reachable(call)
        )

    def position_is_definitely_reached(self, index: int) -> bool:
        """Return whether every resolved entry path executes this position."""

        return _c_style_direct_path_is_definitely_reached(self, index)

    def _resolve_csharp_returned_delegate_call(
        self,
        call_name: str,
        call_arguments: tuple[str, ...],
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        """Resolve ``var f = Factory(); f()`` for an expression lambda.

        Generated C# commonly wraps a byte source in a returned ``Func<byte[]>``.
        This bounded case follows only a no-argument local delegate and leaves
        block-bodied or parameterized delegates unresolved.
        """

        if self.language != "csharp" or call_arguments:
            return None
        assignment = self.latest_assignment(call_name, index)
        if assignment is None:
            return None
        factory = _simple_call(assignment.expression)
        if factory is None or factory[1]:
            return None
        factory_name = factory[0]
        key = f"returned-delegate:{call_name}:{assignment.start}:{factory_name}"
        if key in seen:
            return None

        resolved_values: list[ResolvedValue] = []
        for definition in self.function(factory_name):
            for returned in definition.returns:
                if self._containing_function(returned.index) is not definition:
                    continue
                match = re.fullmatch(
                    r"\(\s*\)\s*=>\s*(?P<body>.+)",
                    returned.expression.strip(),
                    flags=re.DOTALL,
                )
                if match is None:
                    continue
                body = _strip_expression(match.group("body"))
                if body.startswith("{"):
                    continue
                resolved_values.append(
                    self._resolve(
                        body,
                        returned.index,
                        mode=mode,
                        seen=seen | frozenset({key}),
                        bindings=bindings,
                    )
                )
        if not resolved_values:
            return None
        return _merge_resolved(f"{call_name}()", resolved_values)

    def _resolve_csharp_foreach_binding(
        self,
        reference: str,
        index: int,
        *,
        mode: str,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        """Resolve a C# loop variable drawn from a source-defined array."""

        if self.language != "csharp":
            return None
        candidates: list[tuple[int, str]] = []
        pattern = re.compile(
            rf"\bforeach\s*\(\s*[^;()]*?\b{re.escape(reference)}\s+in\s+"
            rf"(?P<iterable>{_IDENTIFIER})\s*\)\s*\{{"
        )
        for match in pattern.finditer(self.masked[:index]):
            body_open = match.end() - 1
            body_close = _find_matching_masked(self.masked, body_open, "{", "}")
            if body_close != -1 and body_open < index < body_close:
                candidates.append((match.start(), match.group("iterable")))
        if not candidates:
            return None
        loop_start, iterable = max(candidates, key=lambda item: item[0])

        declaration = None
        for match in re.finditer(
            rf"\b(?:var|{_IDENTIFIER}(?:\s*\[\s*\])?)\s+"
            rf"{re.escape(iterable)}\s*=\s*\{{(?P<items>[^{{}};]*)\}}\s*;",
            self.masked[:loop_start],
            flags=re.DOTALL,
        ):
            declaration = match
        if declaration is not None:
            item_source = self.source[
                declaration.start("items") : declaration.end("items")
            ]
            items = tuple(
                item.strip()
                for item in _split_parameter_text(item_source)
                if item.strip()
            )
            resolved = [
                self._resolve(
                    item,
                    declaration.start("items"),
                    mode=mode,
                    seen=seen,
                    bindings=bindings,
                )
                for item in items
            ]
            if resolved:
                return _merge_resolved(reference, resolved)

        return self._resolve(
            iterable,
            loop_start,
            mode=mode,
            seen=seen,
            bindings=bindings,
        )

    def _guarded_allowlist_values(
        self,
        reference: str,
        index: int,
    ) -> tuple[str, ...]:
        """Resolve values admitted by a prior fail-closed Set/array guard."""

        if self.language not in {"typescript", "csharp"}:
            return ()
        containing = self._containing_function(index)
        start = containing.start if containing is not None else 0
        prefix = self.source[start:index]
        compact_reference = r"\s*\.\s*".join(
            re.escape(part)
            for part in re.sub(r"\s+", "", reference).split(".")
        )
        guards = tuple(
            re.finditer(
                rf"\bif\s*\(\s*!\s*(?P<allow>{_IDENTIFIER})"
                rf"\s*\.\s*(?:has|includes|Contains)\s*\(\s*"
                rf"{compact_reference}\s*\)\s*\)\s*"
                rf"(?:throw\b|\{{[^{{}}]*\bthrow\b[^{{}}]*\}})",
                self._normalize_guard_terminators(prefix),
                flags=re.DOTALL,
            )
        )
        if not guards:
            return ()
        allow_name = guards[-1].group("allow")
        declarations = tuple(
            re.finditer(
                rf"\b(?:const|let|var|[A-Za-z_$][\w$<>., ]*)\s+"
                rf"{re.escape(allow_name)}\s*=\s*"
                r"(?:new\s+(?:Set|HashSet)\s*(?:<[^>]+>)?\s*\()?"
                r"(?P<values>\[[^\]]*\]|\{[^}]*\})\s*\)?\s*;",
                prefix,
                flags=re.DOTALL,
            )
        )
        if not declarations:
            return ()
        return tuple(
            dict.fromkeys(
                value
                for literal in re.finditer(
                    r"(?P<quote>[\"'])(?P<value>(?:\\.|(?!\1).)*)(?P=quote)",
                    declarations[-1].group("values"),
                    flags=re.DOTALL,
                )
                if (value := _string_literal(literal.group(0))) is not None
            )
        )

    def _resolve_typescript_iife_delimited_element(
        self,
        expression: str,
        delimiter: str,
        element_index: int,
        *,
        mode: str,
    ) -> ResolvedValue | None:
        """Select one serialized field returned by a block-bodied arrow IIFE."""

        if self.language != "typescript":
            return None
        masked = mask_c_style_comments_and_literals(expression)
        match = re.match(
            r"^\s*\(\s*\(\s*(?:async\s+)?[^()]*\)"
            r"\s*(?::\s*[^=]+)?=>\s*\{",
            masked,
        )
        if match is None:
            return None
        body_open = match.end() - 1
        body_close = _find_matching_masked(masked, body_open, "{", "}")
        if body_close == -1:
            return None
        body_source = expression[body_open + 1 : body_close]
        body_masked = masked[body_open + 1 : body_close]
        returns: list[tuple[str, int]] = []
        brace_depth = 0
        for token in re.finditer(r"[{}]|\breturn\b", body_masked):
            if token.group(0) == "{":
                brace_depth += 1
                continue
            if token.group(0) == "}":
                brace_depth = max(0, brace_depth - 1)
                continue
            if brace_depth:
                continue
            end = _next_c_style_statement_end_masked(
                body_masked,
                token.end(),
                "typescript",
            )
            if end == -1:
                return None
            returns.append(
                (_strip_expression(body_source[token.end() : end]), token.end())
            )
        if len(returns) != 1:
            return None
        returned, return_index = returns[0]
        local = CStyleAnalyzer(body_source, "typescript")
        return local._resolve_c_style_delimited_element(
            returned,
            delimiter,
            element_index,
            return_index,
            mode=mode,
            seen=frozenset(),
            bindings=None,
        )

    def _resolve_typescript_buffer_slice(
        self,
        expression: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        match = re.fullmatch(
            r"(?P<receiver>.+)\.\s*(?:slice|subarray)\s*\((?P<arguments>.*)\)",
            _strip_expression(expression),
            flags=re.DOTALL,
        )
        if match is None:
            return None
        arguments = _split_inline_arguments(match.group("arguments"))
        if not arguments or len(arguments) > 2:
            return None
        start = _evaluate_c_style_int(arguments[0])
        end = (
            _evaluate_c_style_int(arguments[1])
            if len(arguments) == 2
            else None
        )
        if start is None or (len(arguments) == 2 and end is None):
            return None
        return self._resolve_typescript_buffer_range(
            match.group("receiver"),
            start,
            end,
            index,
            seen=seen,
            bindings=bindings,
        )

    def _resolve_typescript_buffer_range(
        self,
        expression: str,
        start: int,
        end: int | None,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        expr = _strip_expression(expression)
        identity = re.sub(r"\s+", "", expr)
        state_key = f"typescript-buffer-range:{start}:{end}:{index}:{identity}"
        if state_key in seen:
            return None
        next_seen = seen | frozenset({state_key})

        reference = _simple_reference(expr)
        if reference is not None:
            assignment = self.latest_assignment(reference, index)
            copied_range = self._resolve_typescript_copied_buffer_range(
                reference,
                start,
                end,
                index,
                after=assignment.start if assignment is not None else 0,
                seen=next_seen,
                bindings=bindings,
            )
            if copied_range is not None:
                return copied_range
            if assignment is not None:
                return self._resolve_typescript_buffer_range(
                    assignment.expression,
                    start,
                    end,
                    assignment.start,
                    seen=next_seen,
                    bindings=bindings,
                )
            containing = self._containing_function(index)
            if containing is not None and reference in containing.parameters:
                parameter_index = containing.parameters.index(reference)
                arguments: list[tuple[str, int]] = []
                if bindings is not None:
                    bound = bindings.get((containing.start, reference))
                    if bound is not None:
                        arguments.append(bound)
                for call in self.calls(containing.name):
                    if containing.start <= call.start <= containing.end:
                        continue
                    if not self._call_can_target_function(call, containing):
                        continue
                    argument = self._c_style_argument_for_parameter(
                        call,
                        parameter_index,
                        reference,
                    )
                    if argument is not None:
                        arguments.append((argument, call.start))
                values = [
                    resolved
                    for argument, caller_index in dict.fromkeys(arguments)
                    if (
                        resolved := self._resolve_typescript_buffer_range(
                            argument,
                            start,
                            end,
                            caller_index,
                            seen=next_seen,
                            bindings=bindings,
                        )
                    )
                    is not None
                ]
                if values:
                    return _merge_resolved(expression, values)

        receiver_wrapper = _origin_preserving_receiver_wrapper(expr)
        if receiver_wrapper is not None:
            return self._resolve_typescript_buffer_range(
                receiver_wrapper,
                start,
                end,
                index,
                seen=next_seen,
                bindings=bindings,
            )
        call_wrapper = _origin_preserving_call_argument(expr)
        if call_wrapper is not None:
            return self._resolve_typescript_buffer_range(
                call_wrapper,
                start,
                end,
                index,
                seen=next_seen,
                bindings=bindings,
            )

        called = _simple_call(expr)
        if called is not None:
            call_name, call_arguments = called
            if call_name == "concat" and call_arguments:
                elements = _c_style_container_elements(call_arguments[0])
                if elements is not None:
                    return self._resolve_typescript_concat_range(
                        expr,
                        elements,
                        start,
                        end,
                        index,
                        seen=next_seen,
                        bindings=bindings,
                    )
            values: list[ResolvedValue] = []
            for definition in self.function(call_name):
                call_bindings = dict(bindings or {})
                call_bindings.update(
                    {
                        (definition.start, parameter): (argument, index)
                        for parameter, argument in _bind_c_style_arguments(
                            definition.parameters,
                            call_arguments,
                        )
                    }
                )
                for returned in definition.returns:
                    if self._containing_function(returned.index) is not definition:
                        continue
                    resolved = self._resolve_typescript_buffer_range(
                        returned.expression,
                        start,
                        end,
                        returned.index,
                        seen=next_seen,
                        bindings=call_bindings,
                    )
                    if resolved is not None:
                        values.append(resolved)
            if values:
                return _merge_resolved(expression, values)
        return None

    def _resolve_typescript_copied_buffer_range(
        self,
        target: str,
        start: int,
        end: int | None,
        index: int,
        *,
        after: int,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        """Resolve a requested range fully overwritten by one Buffer.copy call."""

        if end is None:
            return None
        normalized_target = _normalize_reference(target)
        candidates: list[tuple[int, str]] = []
        for call in self.calls("copy"):
            if (
                call.start <= after
                or call.end > index
                or not self.position_is_in_same_or_enclosing_function(
                    call.start,
                    index,
                )
                or not call.arguments
            ):
                continue
            destination = _mutation_target_reference(call.arguments[0])
            if (
                destination is None
                or _normalize_reference(destination) != normalized_target
            ):
                continue
            destination_start = (
                _evaluate_c_style_int(call.arguments[1])
                if len(call.arguments) >= 2
                else 0
            )
            source_start = (
                _evaluate_c_style_int(call.arguments[2])
                if len(call.arguments) >= 3
                else 0
            )
            source_end = (
                _evaluate_c_style_int(call.arguments[3])
                if len(call.arguments) >= 4
                else None
            )
            if destination_start is None or source_start is None:
                continue
            receiver = re.sub(r"\s+", "", call.callee).rsplit(".", 1)[0]
            if not receiver:
                continue
            if source_end is None:
                source_length = self._typescript_static_buffer_length(
                    receiver,
                    call.start,
                    seen=seen,
                    bindings=bindings,
                )
                if source_length is None:
                    continue
                copied_length = source_length - source_start
            else:
                copied_length = source_end - source_start
            if copied_length <= 0:
                continue
            destination_end = destination_start + copied_length
            if destination_start <= start and destination_end >= end:
                candidates.append((call.start, receiver))
        if not candidates:
            return None
        call_index, receiver = max(candidates)
        resolved = self._resolve(
            receiver,
            call_index,
            mode="origin",
            seen=seen,
            bindings=bindings,
        )
        return _prepend_resolution(
            target,
            "resolve Buffer.copy range source",
            resolved,
            ResolutionStep(
                target,
                "inspect serialized Buffer.copy range",
                resolved.origin,
                self.source_span(call_index, call_index + len(receiver)),
            ),
        )

    def _resolve_typescript_concat_range(
        self,
        expression: str,
        elements: tuple[str, ...],
        start: int,
        end: int | None,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        selected: list[str] = []
        if start < 0 and end is None:
            required = -start
            consumed = 0
            for element in reversed(elements):
                length = self._typescript_static_buffer_length(
                    element,
                    index,
                    seen=seen,
                    bindings=bindings,
                )
                if length is None or consumed + length > required:
                    return None
                selected.append(element)
                consumed += length
                if consumed == required:
                    selected.reverse()
                    break
            else:
                return None
        elif start >= 0:
            cursor = 0
            for element in elements:
                length = self._typescript_static_buffer_length(
                    element,
                    index,
                    seen=seen,
                    bindings=bindings,
                )
                if length is None:
                    if end is None or end > cursor:
                        selected.append(element)
                    break
                element_end = cursor + length
                requested_end = end if end is not None else element_end
                if start < element_end and requested_end > cursor:
                    selected.append(element)
                cursor = element_end
                if end is not None and cursor >= end:
                    break
            if not selected:
                return None
        else:
            return None

        values = [
            self._resolve(
                element,
                index,
                mode="origin",
                seen=seen,
                bindings=bindings,
            )
            for element in selected
        ]
        merged = _merge_transformed_origins(expression, values)
        return ResolvedValue(
            expression,
            merged.origin,
            merged.values,
            (
                ResolutionStep(
                    expression,
                    "select Buffer.concat range",
                    merged.origin,
                    self.source_span(index, index + len(expression)),
                ),
                *merged.steps,
            ),
            merged.unresolved_reason,
        )

    def _typescript_static_buffer_length(
        self,
        expression: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> int | None:
        expr = _strip_expression(expression)
        identity = re.sub(r"\s+", "", expr)
        state_key = f"typescript-buffer-length:{index}:{identity}"
        if state_key in seen:
            return None
        next_seen = seen | frozenset({state_key})

        conditional = _top_level_c_style_conditional(expr)
        if conditional is not None:
            branch_lengths = tuple(
                self._typescript_static_buffer_length(
                    branch,
                    index,
                    seen=next_seen,
                    bindings=bindings,
                )
                for branch in conditional
            )
            known = {length for length in branch_lengths if length is not None}
            return (
                next(iter(known))
                if len(known) == 1 and all(
                    length is not None for length in branch_lengths
                )
                else None
            )

        buffer_from = re.fullmatch(
            r"(?:Buffer\s*\.\s*from|new\s+Buffer)\s*\((?P<arguments>.*)\)",
            expr,
            flags=re.DOTALL,
        )
        if buffer_from is not None:
            arguments = _split_inline_arguments(buffer_from.group("arguments"))
            if arguments:
                encoding = "utf8"
                if len(arguments) > 1:
                    encodings = {
                        value.lower().replace("-", "")
                        for value in self.resolve_strings(
                            arguments[1], index
                        ).values
                    }
                    if len(encodings) != 1:
                        return None
                    encoding = next(iter(encodings))
                repeated_literal = re.fullmatch(
                    r"(?P<literal>(?P<q>[\"'])(?:\\.|(?!\2).)*\2)"
                    r"\s*\.\s*repeat\s*\(\s*(?P<count>\d[\d_]*)\s*\)",
                    arguments[0],
                    flags=re.DOTALL,
                )
                if repeated_literal is not None:
                    literal = _string_literal(repeated_literal.group("literal"))
                    count = _evaluate_c_style_int(repeated_literal.group("count"))
                    if literal is not None and count is not None and count >= 0:
                        repeated_length = _typescript_encoded_string_length(
                            literal * count,
                            encoding,
                        )
                        if repeated_length is not None:
                            return repeated_length
                values = self.resolve_strings(arguments[0], index).values
                lengths = {
                    length
                    for value in values
                    if (
                        length := _typescript_encoded_string_length(
                            value, encoding
                        )
                    )
                    is not None
                }
                if values and len(lengths) == 1:
                    return next(iter(lengths))
                if encoding == "hex":
                    round_trip = self._typescript_hex_round_trip_length(
                        arguments[0],
                        index,
                        seen=next_seen,
                        bindings=bindings,
                    )
                    if round_trip is not None:
                        return round_trip

        encoded_text = re.fullmatch(
            r"(?P<receiver>.+)\.\s*toString\s*\(\s*['\"]hex['\"]\s*\)",
            expr,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if encoded_text is not None:
            receiver_length = self._typescript_static_buffer_length(
                encoded_text.group("receiver"),
                index,
                seen=next_seen,
                bindings=bindings,
            )
            if receiver_length is not None:
                # Passing a hex string directly to a Node crypto API uses the
                # string's UTF-8 bytes. Each input byte therefore occupies two
                # bytes in the API argument unless Buffer.from(..., "hex")
                # decodes it first.
                return receiver_length * 2

        if re.fullmatch(
            r".+\.\s*(?:digest|final)\s*\(\s*\)",
            expr,
            flags=re.DOTALL,
        ):
            # Node crypto returns a Buffer when no output encoding is given.
            # A resolved algorithm selector is not the returned byte length.
            return None

        derived_key_call = _simple_call(expr)
        if derived_key_call is not None:
            simple_name, arguments = derived_key_call
            canonical = next(
                (
                    imported
                    for imported, alias in self.import_aliases.items()
                    if alias == simple_name
                ),
                "",
            )
            callee_match = re.match(
                rf"(?P<callee>{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*)\s*\(",
                expr,
            )
            callee = (
                re.sub(r"\s+", "", callee_match.group("callee"))
                if callee_match is not None
                else ""
            )
            receiver = callee.rsplit(".", 1)[0] if "." in callee else ""
            if not canonical and receiver in self.crypto_namespace_aliases:
                canonical = simple_name
            length_index = {
                "pbkdf2Sync": 3,
                "scryptSync": 2,
            }.get(canonical)
            if length_index is not None and len(arguments) > length_index:
                resolved_length = self._resolve(
                    arguments[length_index],
                    index,
                    mode="int",
                    seen=next_seen,
                    bindings=bindings,
                )
                lengths = {
                    int(value)
                    for value in resolved_length.values
                    if re.fullmatch(r"-?\d+", value)
                }
                if len(lengths) == 1:
                    length = next(iter(lengths))
                    if length >= 0:
                        return length

        digest = re.fullmatch(
            r"(?s:.*)createHash\s*\(\s*(?P<algorithm>[^,)]+)[^)]*\)"
            r"(?s:.*)\.\s*digest\s*\([^)]*\)",
            expr,
        )
        if digest is not None:
            algorithms = {
                value.lower().replace("-", "")
                for value in self.resolve_strings(
                    digest.group("algorithm"), index
                ).values
            }
            lengths = {
                _TYPESCRIPT_DIGEST_LENGTHS[value]
                for value in algorithms
                if value in _TYPESCRIPT_DIGEST_LENGTHS
            }
            if len(lengths) == 1:
                return next(iter(lengths))

        length_call = re.fullmatch(
            r"(?s:.*)\.\s*(?:randomBytes|alloc|allocUnsafe)\s*\((?P<size>[^,)]*)[^)]*\)",
            expr,
        )
        if length_call is not None:
            return _evaluate_c_style_int(length_call.group("size"))

        slice_match = re.fullmatch(
            r"(?P<receiver>.+)\.\s*(?:slice|subarray)\s*\((?P<arguments>.*)\)",
            expr,
            flags=re.DOTALL,
        )
        if slice_match is not None:
            arguments = _split_inline_arguments(slice_match.group("arguments"))
            if arguments:
                start = _evaluate_c_style_int(arguments[0])
                end = (
                    _evaluate_c_style_int(arguments[1])
                    if len(arguments) == 2
                    else None
                )
                receiver_length = self._typescript_static_buffer_length(
                    slice_match.group("receiver"),
                    index,
                    seen=next_seen,
                    bindings=bindings,
                )
                if start is not None and receiver_length is not None:
                    normalized_start = start if start >= 0 else receiver_length + start
                    normalized_end = (
                        end if end is not None and end >= 0
                        else receiver_length + end
                        if end is not None
                        else receiver_length
                    )
                    return max(0, normalized_end - normalized_start)

        reference = _simple_reference(expr)
        if reference is not None:
            member = re.fullmatch(
                rf"(?P<owner>this|{_IDENTIFIER})\s*\.\s*"
                rf"(?P<name>{_IDENTIFIER})",
                reference,
            )
            if member is not None:
                owner = member.group("owner")
                containing_class = self._containing_class_name(index)
                target_class = containing_class if owner == "this" else owner
                candidates = tuple(
                    assignment
                    for assignment in self.assignments
                    if _normalize_reference(assignment.name)
                    == _normalize_reference(member.group("name"))
                    and self._containing_class_name(assignment.start)
                    == target_class
                    and not (
                        assignment.start <= index < assignment.end
                    )
                )
                if candidates:
                    lengths = tuple(
                        self._typescript_static_buffer_length(
                            assignment.expression,
                            assignment.start,
                            seen=next_seen,
                            bindings=bindings,
                        )
                        for assignment in candidates
                    )
                    known = {length for length in lengths if length is not None}
                    if len(known) == 1 and all(
                        length is not None for length in lengths
                    ):
                        return next(iter(known))
                if target_class is not None:
                    constructor_lengths: list[int | None] = []
                    for constructor in self.functions:
                        if (
                            constructor.name != "constructor"
                            or self._containing_class_name(constructor.start)
                            != target_class
                            or member.group("name") not in constructor.parameters
                        ):
                            continue
                        parameter_index = constructor.parameters.index(
                            member.group("name")
                        )
                        for call_name in self._call_names_for_function(constructor):
                            for call in self.calls(call_name):
                                if constructor.start <= call.start <= constructor.end:
                                    continue
                                argument = self._c_style_argument_for_parameter(
                                    call,
                                    parameter_index,
                                    member.group("name"),
                                )
                                if argument is not None:
                                    constructor_lengths.append(
                                        self._typescript_static_buffer_length(
                                            argument,
                                            call.start,
                                            seen=next_seen,
                                            bindings=bindings,
                                        )
                                    )
                    known_constructor_lengths = {
                        length
                        for length in constructor_lengths
                        if length is not None
                    }
                    if (
                        len(known_constructor_lengths) == 1
                        and all(
                            length is not None
                            for length in constructor_lengths
                        )
                    ):
                        return next(iter(known_constructor_lengths))
            assignment = self.latest_assignment(reference, index)
            if assignment is not None:
                return self._typescript_static_buffer_length(
                    assignment.expression,
                    assignment.start,
                    seen=next_seen,
                    bindings=bindings,
                )
            containing = self._containing_function(index)
            if containing is not None and reference in containing.parameters:
                parameter_index = containing.parameters.index(reference)
                arguments: list[tuple[str, int]] = []
                if bindings is not None:
                    bound = bindings.get((containing.start, reference))
                    if bound is not None:
                        arguments.append(bound)
                for call_name in self._call_names_for_function(containing):
                    for call in self.calls(call_name):
                        if containing.start <= call.start <= containing.end:
                            continue
                        argument = self._c_style_argument_for_parameter(
                            call,
                            parameter_index,
                            reference,
                        )
                        if argument is not None:
                            arguments.append((argument, call.start))
                lengths = {
                    length
                    for argument, caller_index in dict.fromkeys(arguments)
                    if (
                        length := self._typescript_static_buffer_length(
                            argument,
                            caller_index,
                            seen=next_seen,
                            bindings=bindings,
                        )
                    )
                    is not None
                }
                if len(lengths) == 1:
                    return next(iter(lengths))

        called = _simple_call(expr)
        if called is not None:
            call_name, call_arguments = called
            if call_name == "concat" and call_arguments:
                elements = _c_style_container_elements(call_arguments[0])
                if elements is not None:
                    lengths = [
                        self._typescript_static_buffer_length(
                            element,
                            index,
                            seen=next_seen,
                            bindings=bindings,
                        )
                        for element in elements
                    ]
                    if all(length is not None for length in lengths):
                        return sum(length for length in lengths if length is not None)
            definitions = tuple(self.function(call_name))
            returned_lengths: list[int | None] = []
            for definition in definitions:
                call_bindings = dict(bindings or {})
                call_bindings.update(
                    {
                        (definition.start, parameter): (argument, index)
                        for parameter, argument in _bind_c_style_arguments(
                            definition.parameters,
                            call_arguments,
                        )
                    }
                )
                for returned in definition.returns:
                    length = self._typescript_static_buffer_length(
                        returned.expression,
                        returned.index,
                        seen=next_seen,
                        bindings=call_bindings,
                    )
                    returned_lengths.append(length)
            if returned_lengths and all(
                length is not None for length in returned_lengths
            ):
                lengths = {length for length in returned_lengths if length is not None}
                if len(lengths) == 1:
                    return next(iter(lengths))
            if definitions:
                # A local helper with an unresolved Buffer-producing return
                # must stay unresolved.  Generic string resolution may expose
                # an internal algorithm selector (for example ``sha256``),
                # which is not the helper's output byte length.
                return None

        resolved_strings = self.resolve_strings(expr, index)
        if resolved_strings.origin == "constant" and resolved_strings.values:
            lengths = {
                len(value.encode("utf-8")) for value in resolved_strings.values
            }
            if len(lengths) == 1:
                return next(iter(lengths))
        return None

    def _typescript_hex_round_trip_length(
        self,
        expression: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> int | None:
        """Resolve bytes encoded as hex and immediately decoded again."""

        expr = _strip_expression(expression)
        match = re.fullmatch(
            r"(?P<receiver>.+)\.\s*toString\s*\(\s*['\"]hex['\"]\s*\)",
            expr,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match is not None:
            return self._typescript_static_buffer_length(
                match.group("receiver"),
                index,
                seen=seen,
                bindings=bindings,
            )

        reference = _simple_reference(expr)
        if reference is not None:
            assignment = self.latest_assignment(reference, index)
            if assignment is not None:
                return self._typescript_hex_round_trip_length(
                    assignment.expression,
                    assignment.start,
                    seen=seen,
                    bindings=bindings,
                )
            containing = self._containing_function(index)
            if containing is not None and reference in containing.parameters:
                parameter_index = containing.parameters.index(reference)
                lengths = {
                    length
                    for call in self.calls(containing.name)
                    if not (containing.start <= call.start <= containing.end)
                    if (
                        argument := self._c_style_argument_for_parameter(
                            call,
                            parameter_index,
                            reference,
                        )
                    )
                    is not None
                    if (
                        length := self._typescript_hex_round_trip_length(
                            argument,
                            call.start,
                            seen=seen,
                            bindings=bindings,
                        )
                    )
                    is not None
                }
                if len(lengths) == 1:
                    return next(iter(lengths))

        called = _simple_call(expr)
        if called is not None:
            call_name, call_arguments = called
            lengths: set[int] = set()
            for definition in self.function(call_name):
                call_bindings = dict(bindings or {})
                call_bindings.update(
                    {
                        (definition.start, parameter): (argument, index)
                        for parameter, argument in _bind_c_style_arguments(
                            definition.parameters,
                            call_arguments,
                        )
                    }
                )
                for returned in definition.returns:
                    length = self._typescript_hex_round_trip_length(
                        returned.expression,
                        returned.index,
                        seen=seen,
                        bindings=call_bindings,
                    )
                    if length is not None:
                        lengths.add(length)
            if len(lengths) == 1:
                return next(iter(lengths))
        return None

    def _csharp_static_buffer_length(
        self,
        expression: str,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> int | None:
        """Resolve source-proven byte-array lengths used by .NET crypto APIs."""

        expr = _strip_expression(expression)
        identity = re.sub(r"\s+", "", expr)
        state_key = f"csharp-buffer-length:{index}:{identity}"
        if state_key in seen:
            return None
        next_seen = seen | frozenset({state_key})

        allocated = re.fullmatch(
            r"new\s+(?:System\s*\.\s*)?(?:Byte|byte)\s*\[\s*(?P<size>[^]]+)\s*\]",
            expr,
            flags=re.IGNORECASE,
        )
        if allocated is not None:
            size = _evaluate_c_style_int(allocated.group("size"))
            if size is not None:
                return size

        initialized = re.fullmatch(
            r"new\s+(?:(?:System\s*\.\s*)?(?:Byte|byte)\s*\[\s*\]|\[\])"
            r"\s*\{(?P<items>.*)\}",
            expr,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if initialized is not None:
            items = _split_inline_arguments(initialized.group("items"))
            return len(items)

        implicit_initializer = re.fullmatch(
            r"\{(?P<items>.*)\}",
            expr,
            flags=re.DOTALL,
        )
        if implicit_initializer is not None:
            return len(_split_inline_arguments(implicit_initializer.group("items")))

        get_bytes = re.fullmatch(
            r"(?:(?:System\s*\.\s*Text\s*\.\s*)?Encoding\s*\.\s*)?"
            r"(?P<encoding>UTF8|ASCII|Unicode|UTF32)\s*\.\s*"
            r"GetBytes\s*\((?P<value>.*)\)",
            expr,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if get_bytes is not None:
            values = self.resolve_strings(get_bytes.group("value"), index).values
            codec = {
                "utf8": "utf-8",
                "ascii": "ascii",
                "unicode": "utf-16-le",
                "utf32": "utf-32-le",
            }[get_bytes.group("encoding").lower()]
            lengths = {len(value.encode(codec)) for value in values}
            if values and len(lengths) == 1:
                return next(iter(lengths))

        reference = _simple_reference(expr)
        if reference is not None:
            if bindings is not None:
                containing = self._containing_function(index)
                if containing is not None:
                    bound = bindings.get((containing.start, reference))
                    if bound is not None:
                        return self._csharp_static_buffer_length(
                            bound[0],
                            bound[1],
                            seen=next_seen,
                            bindings=bindings,
                        )
            assignment = self.latest_assignment(reference, index)
            if assignment is not None:
                return self._csharp_static_buffer_length(
                    assignment.expression,
                    assignment.start,
                    seen=next_seen,
                    bindings=bindings,
                )
            containing = self._containing_function(index)
            if containing is not None and reference in containing.parameters:
                parameter_index = containing.parameters.index(reference)
                lengths = {
                    length
                    for call in self.calls(containing.name)
                    if not (containing.start <= call.start <= containing.end)
                    if (
                        argument := self._c_style_argument_for_parameter(
                            call,
                            parameter_index,
                            reference,
                        )
                    )
                    is not None
                    if (
                        length := self._csharp_static_buffer_length(
                            argument,
                            call.start,
                            seen=next_seen,
                            bindings=bindings,
                        )
                    )
                    is not None
                }
                if len(lengths) == 1:
                    return next(iter(lengths))

        called = _simple_call(expr)
        if called is not None:
            call_name, call_arguments = called
            lengths: set[int] = set()
            for definition in self.function(call_name):
                call_bindings = dict(bindings or {})
                call_bindings.update(
                    {
                        (definition.start, parameter): (argument, index)
                        for parameter, argument in _bind_c_style_arguments(
                            definition.parameters,
                            call_arguments,
                        )
                    }
                )
                for returned in definition.returns:
                    length = self._csharp_static_buffer_length(
                        returned.expression,
                        returned.index,
                        seen=next_seen,
                        bindings=call_bindings,
                    )
                    if length is not None:
                        lengths.add(length)
            if len(lengths) == 1:
                return next(iter(lengths))
        return None

    def _resolve_csharp_prior_called_property_assignment(
        self,
        qualified: str,
        containing: FunctionDefinition | None,
        index: int,
        *,
        seen: frozenset[str],
        bindings: dict[tuple[int, str], tuple[str, int]] | None,
    ) -> ResolvedValue | None:
        """Resolve class state initialized by a method called before the sink method."""

        if containing is None:
            return None
        owner = self._containing_class_name(containing.start)
        if owner is None:
            return None
        candidate_assignments = [
            assignment
            for assignment in self.assignments
            if _compact_reference(assignment.name) == qualified
            and self._containing_class_name(assignment.start) == owner
            and (scope := self._assignment_scopes[assignment]) is not None
            and scope is not containing
        ]
        if not candidate_assignments:
            return None

        target_calls = [
            call
            for call in self.calls(containing.name)
            if not (containing.start <= call.start <= containing.end)
            and self._call_can_target_function(call, containing)
        ]
        values: list[ResolvedValue] = []
        for target_call in target_calls:
            caller = self._containing_function(target_call.start)
            if caller is None:
                continue
            target_receiver = _csharp_method_call_receiver(target_call.callee)
            for assignment in candidate_assignments:
                initializer = self._assignment_scopes[assignment]
                if initializer is None:
                    continue
                called_before = any(
                    call.start < target_call.start
                    and self._containing_function(call.start) == caller
                    and self._call_can_target_function(call, initializer)
                    and _csharp_method_call_receiver(call.callee) == target_receiver
                    for call in self.calls(initializer.name)
                )
                if not called_before:
                    continue
                resolved = self._resolve(
                    assignment.expression,
                    assignment.start,
                    mode="origin",
                    seen=seen,
                    bindings=bindings,
                )
                if resolved.origin != "unknown":
                    values.append(resolved)
        return _merge_resolved(qualified, values) if values else None

    def _callback_registrations_for_function(
        self,
        function: FunctionDefinition,
    ) -> list[CallExpression]:
        """Return standard callback registrations that invoke a named helper."""

        callback_apis = {
            "Promise",
            "Transform",
            "Writable",
            "addEventListener",
            "on",
            "once",
            "setTimeout",
            "setInterval",
            "setImmediate",
            "queueMicrotask",
            "then",
            "catch",
            "finally",
            "forEach",
            "map",
            "filter",
            "some",
            "every",
            "reduce",
            "sort",
        }
        records: list[CallExpression] = []
        for call in self.calls():
            if call.simple_name not in callback_apis:
                continue
            if function.start <= call.start <= function.end:
                continue
            if call.start <= function.start and function.end <= call.end:
                records.append(call)
                continue
            if any(
                _compact_reference(_strip_expression(argument)) == function.name
                for argument in call.arguments
            ):
                records.append(call)
        return records

    def _call_names_for_function(
        self,
        function: FunctionDefinition,
    ) -> tuple[str, ...]:
        if function.name != "constructor":
            return (function.name,)
        owner = self._containing_class_name(function.start)
        if owner is None:
            return (function.name,)
        subclasses = re.findall(
            rf"\bclass\s+([A-Za-z_$][A-Za-z0-9_$]*)\s+extends\s+"
            rf"{re.escape(owner)}\b",
            self.masked,
        )
        return tuple(dict.fromkeys((owner, *subclasses)))
