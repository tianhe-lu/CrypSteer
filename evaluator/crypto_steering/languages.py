"""Canonical source-language metadata shared by every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSpec:
    id: str
    display_name: str
    source_extension: str
    fence_labels: tuple[str, ...]


_LANGUAGES = {
    "java": LanguageSpec("java", "Java", ".java", ("java",)),
    "typescript": LanguageSpec(
        "typescript", "TypeScript", ".ts", ("typescript", "ts")
    ),
    "csharp": LanguageSpec(
        "csharp", "C#", ".cs", ("csharp", "cs", "c#", "dotnet")
    ),
    "python": LanguageSpec("python", "Python", ".py", ("python", "py")),
}

SUPPORTED_LANGUAGES = tuple(_LANGUAGES)

_ALIASES = {
    alias: language_id
    for language_id, spec in _LANGUAGES.items()
    for alias in (language_id, *spec.fence_labels)
}


def canonical_language(value: object) -> str:
    """Return the canonical language ID for a supported name or alias."""

    normalized = str(value).strip().lower()
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        expected = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(
            f"Unsupported source language {value!r}; expected one of: {expected}"
        ) from exc


def language_spec(value: object) -> LanguageSpec:
    """Return metadata for a supported source language."""

    return _LANGUAGES[canonical_language(value)]
