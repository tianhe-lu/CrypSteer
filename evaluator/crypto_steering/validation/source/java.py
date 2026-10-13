"""Java source-analysis context used by category-owned detectors."""

from __future__ import annotations

import re

from crypto_steering.validation.java.lightweight_ast import (
    find_method_calls,
    method_containing_index,
)
from crypto_steering.validation.java.reachability import position_is_reachable
from crypto_steering.validation.java.resolution import (
    reachable_parameter_string_values,
    resolve_int_expression,
    resolve_string_expression,
    resolve_string_expression_values,
)
from crypto_steering.validation.java.security_value_flow import SecurityValueFlow
from crypto_steering.validation.models import (
    AnalysisProfile,
    ResolvedValue,
    ValueOrigin,
)
from crypto_steering.validation.source.base import CallExpression, SourceAnalyzer
from crypto_steering.validation.source.helpers import _split_inline_arguments
from crypto_steering.validation.java.static_analysis import (
    simple_variable,
    strip_outer_expression,
)


class JavaAnalyzer(SourceAnalyzer):
    """Own one parsed Java value-flow graph for a detector invocation.

    Java keeps its javalang-backed semantic implementation, while participating
    in the same analyzer lifecycle and category dispatch as the other three
    languages.  The category modules may use ``flow`` for Java-specific API
    semantics; the small language-neutral methods support shared callers.
    """

    language = "java"

    def __init__(
        self,
        source: str,
        *,
        analysis_profile: AnalysisProfile = "whole_program",
    ) -> None:
        self.source = source
        self.analysis_profile = analysis_profile
        self.include_inactive_findings = False
        self.flow = SecurityValueFlow(source)

    def calls(self, *simple_names: str) -> tuple[CallExpression, ...]:
        calls: dict[tuple[int, int], CallExpression] = {}
        for name in simple_names:
            for call in find_method_calls(self.source, name):
                calls[(call.start, call.end)] = CallExpression(
                    callee=self.source[call.start : call.open_paren].strip(),
                    arguments=tuple(argument.text for argument in call.arguments),
                    named_arguments={},
                    start=call.start,
                    end=call.end,
                )
        return tuple(calls[key] for key in sorted(calls))

    def resolve_origin(self, expression: str, index: int) -> ResolvedValue:
        classification = self.flow.classify(expression, index, "security value")
        origin: ValueOrigin
        if classification.label == "secure":
            origin = "derived_secure"
        elif classification.label == "misuse":
            origin = "predictable"
        else:
            origin = "unknown"
        return ResolvedValue(
            expression=expression,
            origin=origin,
            unresolved_reason=(
                classification.reason if origin == "unknown" else None
            ),
        )

    def resolve_strings(self, expression: str, index: int) -> ResolvedValue:
        values = resolve_string_expression_values(expression, self.flow, index)
        return ResolvedValue(
            expression=expression,
            origin="constant" if values is not None else "unknown",
            values=tuple(sorted(values or ())),
            unresolved_reason=(
                None if values is not None else "Java string value is unresolved"
            ),
        )

    def resolve_ints(self, expression: str, index: int) -> ResolvedValue:
        value = resolve_int_expression(expression, self.flow, index)
        return ResolvedValue(
            expression=expression,
            origin="constant" if value is not None else "unknown",
            values=() if value is None else (str(value),),
            unresolved_reason=(
                None if value is not None else "Java integer value is unresolved"
            ),
        )

    def static_byte_length(self, expression: str, index: int) -> int | None:
        return self._static_byte_length(
            expression,
            index,
            seen=frozenset(),
        )

    def _static_byte_length(
        self,
        expression: str,
        index: int,
        *,
        seen: frozenset[str],
    ) -> int | None:
        expr = strip_outer_expression(expression).strip()
        identity = re.sub(r"\s+", "", expr)
        key = f"java-byte-length:{index}:{identity}"
        if key in seen:
            return None
        next_seen = seen | frozenset({key})

        allocated = re.fullmatch(
            r"new\s+(?:java\s*\.\s*lang\s*\.\s*)?byte\s*"
            r"\[\s*(?P<size>[^]]+)\s*\]",
            expr,
            flags=re.IGNORECASE,
        )
        if allocated is not None:
            resolved = self.resolve_ints(allocated.group("size"), index)
            values = {
                int(value)
                for value in resolved.values
                if re.fullmatch(r"[-+]?\d+", value)
            }
            if len(values) == 1:
                return next(iter(values))

        initialized = re.fullmatch(
            r"new\s+(?:java\s*\.\s*lang\s*\.\s*)?byte\s*"
            r"(?:\[\s*\])?\s*\{(?P<items>.*)\}",
            expr,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if initialized is not None:
            return len(_split_inline_arguments(initialized.group("items")))

        shorthand_initializer = re.fullmatch(
            r"\{(?P<items>.*)\}",
            expr,
            flags=re.DOTALL,
        )
        if shorthand_initializer is not None:
            return len(
                _split_inline_arguments(shorthand_initializer.group("items"))
            )

        get_bytes = re.fullmatch(
            r"(?P<value>.+)\.\s*getBytes\s*\((?P<arguments>.*)\)",
            expr,
            flags=re.DOTALL,
        )
        if get_bytes is not None:
            compact_value = re.sub(r"\s+", "", get_bytes.group("value"))
            if re.fullmatch(
                r"(?:java\.util\.)?UUID\.randomUUID\(\)\.toString\(\)",
                compact_value,
            ):
                # UUID.toString() is always 36 ASCII characters. The benchmark
                # Java environment uses UTF-8 as its default charset.
                return 36
            values = self.resolve_strings(get_bytes.group("value"), index).values
            arguments = _split_inline_arguments(get_bytes.group("arguments"))
            if arguments and re.fullmatch(
                r"(?:java\s*\.\s*nio\s*\.\s*charset\s*\.\s*)?"
                r"StandardCharsets\s*\.\s*UTF_?8",
                arguments[0],
                flags=re.IGNORECASE,
            ):
                encodings = ("UTF-8",)
            else:
                encodings = (
                    self.resolve_strings(arguments[0], index).values
                    if arguments
                    else ("UTF-8",)
                )
            normalized = {
                re.sub(r"[^A-Za-z0-9]", "", value).upper()
                for value in encodings
            }
            if normalized and normalized <= {"UTF8", "STANDARDCHARSETSUTF8"}:
                lengths = {len(value.encode("utf-8")) for value in values}
                if values and len(lengths) == 1:
                    return next(iter(lengths))

        variable = simple_variable(expr)
        if variable is not None:
            assignment = self.flow._latest_assignment(variable, index)
            if assignment is not None:
                return self._static_byte_length(
                    assignment.expression,
                    assignment.start,
                    seen=next_seen,
                )
            method = method_containing_index(self.flow.methods, index)
            if method is not None:
                parameters = [parameter.name for parameter in method.params]
                if variable in parameters:
                    parameter_index = parameters.index(variable)
                    caller_lengths: list[int] = []
                    observed_caller = False
                    for call in find_method_calls(self.source, method.name):
                        if method.start <= call.start <= method.body_end:
                            continue
                        if len(call.arguments) != len(parameters):
                            continue
                        if not self.position_is_reachable(call.start):
                            continue
                        observed_caller = True
                        argument = call.arguments[parameter_index]
                        length = self._static_byte_length(
                            argument.text,
                            call.start,
                            seen=next_seen,
                        )
                        if length is None:
                            return None
                        caller_lengths.append(length)
                    if observed_caller and len(set(caller_lengths)) == 1:
                        return caller_lengths[0]
        return None

    def position_is_reachable(self, index: int) -> bool:
        return position_is_reachable(self.source, self.flow, index)

    def position_is_locally_reachable(self, index: int) -> bool:
        return position_is_reachable(self.source, self.flow, index)

    def string_value(self, expression: str, index: int) -> str | None:
        """Resolve one statically fixed Java string."""

        return resolve_string_expression(expression, self.flow, index)

    def string_values(
        self,
        expression: str,
        index: int,
    ) -> frozenset[str] | None:
        """Resolve every finite Java string value reaching an expression."""

        return resolve_string_expression_values(expression, self.flow, index)

    def int_value(self, expression: str, index: int) -> int | None:
        """Resolve one statically fixed Java integer."""

        return resolve_int_expression(expression, self.flow, index)

    def reachable_parameter_values(
        self,
        expression: str,
        index: int,
    ) -> frozenset[str] | None:
        """Resolve finite values supplied by executable helper callers."""

        return reachable_parameter_string_values(
            expression,
            self.flow,
            index,
            self.position_is_reachable,
        )

    def containing_function_name(self, index: int) -> str | None:
        method = method_containing_index(self.flow.methods, index)
        return method.name if method is not None else None


__all__ = ("JavaAnalyzer",)
