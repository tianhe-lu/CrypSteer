"""Shared records and interface for language-specific source analyzers."""

from __future__ import annotations

from dataclasses import dataclass
import re

from crypto_steering.validation.models import (
    AnalysisProfile,
    Language,
    ResolvedValue,
    SourceSpan,
)


@dataclass(frozen=True)
class CallExpression:
    callee: str
    arguments: tuple[str, ...]
    named_arguments: dict[str, str]
    start: int
    end: int

    @property
    def simple_name(self) -> str:
        value = re.sub(r"\s+", "", self.callee)
        value = re.sub(r"^new", "", value)
        return value.rsplit(".", 1)[-1]

    @property
    def span(self) -> SourceSpan:
        return SourceSpan(start=self.start, end=self.end)


@dataclass(frozen=True)
class Assignment:
    name: str
    expression: str
    start: int
    end: int


@dataclass(frozen=True)
class ReturnExpression:
    expression: str
    index: int


@dataclass(frozen=True)
class FunctionDefinition:
    name: str
    parameters: tuple[str, ...]
    returns: tuple[ReturnExpression, ...]
    start: int
    end: int


@dataclass(frozen=True)
class ClassDefinition:
    name: str
    start: int
    end: int


class SourceAnalyzer:
    """Language-neutral interface used by detectors and smoke checks."""

    language: Language
    source: str
    analysis_profile: AnalysisProfile = "whole_program"

    def calls(self, *simple_names: str) -> tuple[CallExpression, ...]:
        raise NotImplementedError

    def resolve_origin(self, expression: str, index: int) -> ResolvedValue:
        raise NotImplementedError

    def resolve_strings(self, expression: str, index: int) -> ResolvedValue:
        raise NotImplementedError

    def resolve_ints(self, expression: str, index: int) -> ResolvedValue:
        raise NotImplementedError

    def static_byte_length(self, expression: str, index: int) -> int | None:
        """Return a byte-buffer length only when source structure proves it."""

        del expression, index
        return None

    def resolve_call_target_strings(self, call: CallExpression) -> ResolvedValue:
        """Resolve a callable used at a callsite as a policy string value."""

        return self.resolve_strings(call.callee, call.start)

    def contains_call(self, *simple_names: str) -> bool:
        return bool(self.calls(*simple_names))

    def position_is_reachable(self, index: int) -> bool:
        """Return false only when local constant control flow proves a path dead."""

        del index
        return True

    def containing_function_name(self, index: int) -> str | None:
        raise NotImplementedError

    def containing_function_is_called(self, index: int) -> bool:
        name = self.containing_function_name(index)
        return bool(name and self.calls(name))

    def text_after(self, index: int, pattern: str) -> bool:
        return bool(re.search(pattern, self.source[index:], flags=re.DOTALL | re.IGNORECASE))

    def source_span(self, start: int, end: int) -> SourceSpan:
        line = self.source.count("\n", 0, max(0, start)) + 1
        line_start = self.source.rfind("\n", 0, max(0, start)) + 1
        return SourceSpan(start=start, end=end, line=line, column=start - line_start + 1)

    def position_is_locally_reachable(self, index: int) -> bool:
        """Return whether local control flow can reach the position.

        Unlike ``position_is_reachable``, this query deliberately does not
        require the containing function to be called by an application entry
        point. Task validation uses it because a complete library-style route
        is valid even when the source does not invoke that route itself.
        """

        return self.position_is_reachable(index)
