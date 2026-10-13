"""The one coordinate-selection model shared by every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from crypto_steering.languages import SUPPORTED_LANGUAGES, canonical_language
from crypto_steering.records import MISUSE_IDS, PROMPT_FAMILIES, READOUTS


LayerSelection = str | tuple[int, ...]


def _unique(values: Iterable[str], field: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"selection.{field} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"selection.{field} contains duplicate values")
    return result


def _known(values: Iterable[object], supported: tuple[str, ...], field: str) -> tuple[str, ...]:
    result = _unique((str(value).strip() for value in values), field)
    unknown = tuple(value for value in result if value not in supported)
    if unknown:
        raise ValueError(
            f"selection.{field} contains unsupported values {unknown!r}; "
            f"supported values are {supported!r}"
        )
    return result


def _layers(value: object) -> LayerSelection:
    if value == "all":
        return "all"
    if not isinstance(value, (list, tuple)):
        raise ValueError("selection.layers must be 'all' or a non-empty list of layers")
    layers = tuple(value)
    if not layers:
        raise ValueError("selection.layers must not be empty")
    if any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layers):
        raise ValueError("selection.layers must contain non-negative integers")
    if len(set(layers)) != len(layers):
        raise ValueError("selection.layers contains duplicate values")
    return layers


def _items(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"selection.{field} must be a list")
    return tuple(value)


@dataclass(frozen=True)
class Selection:
    languages: tuple[str, ...] = SUPPORTED_LANGUAGES
    misuses: tuple[str, ...] = MISUSE_IDS
    prompt_families: tuple[str, ...] = ("code_only",)
    readouts: tuple[str, ...] = ("code_body_mean",)
    layers: LayerSelection = "all"

    def __post_init__(self) -> None:
        languages = _unique(
            (canonical_language(value) for value in self.languages), "languages"
        )
        object.__setattr__(self, "languages", languages)
        object.__setattr__(
            self,
            "misuses",
            _known((str(value).upper() for value in self.misuses), MISUSE_IDS, "misuses"),
        )
        object.__setattr__(
            self,
            "prompt_families",
            _known(self.prompt_families, PROMPT_FAMILIES, "prompt_families"),
        )
        object.__setattr__(
            self,
            "readouts",
            _known(self.readouts, READOUTS, "readouts"),
        )
        object.__setattr__(self, "layers", _layers(self.layers))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Selection:
        expected = {"languages", "misuses", "prompt_families", "readouts", "layers"}
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                f"selection fields are invalid; missing={sorted(missing)!r}, "
                f"unknown={sorted(unknown)!r}"
            )
        return cls(
            languages=_items(value["languages"], "languages"),  # type: ignore[arg-type]
            misuses=_items(value["misuses"], "misuses"),  # type: ignore[arg-type]
            prompt_families=_items(
                value["prompt_families"], "prompt_families"
            ),  # type: ignore[arg-type]
            readouts=_items(value["readouts"], "readouts"),  # type: ignore[arg-type]
            layers=value["layers"],  # type: ignore[arg-type]
        )

    def narrow(
        self,
        *,
        languages: Iterable[str] | None = None,
        misuses: Iterable[str] | None = None,
        prompt_families: Iterable[str] | None = None,
        readouts: Iterable[str] | None = None,
        layers: str | Iterable[int] | None = None,
    ) -> Selection:
        narrowed_languages = (
            tuple(canonical_language(value) for value in languages)
            if languages is not None
            else self.languages
        )
        narrowed_misuses = (
            tuple(str(value).upper() for value in misuses)
            if misuses is not None
            else self.misuses
        )
        narrowed_families = (
            tuple(prompt_families)
            if prompt_families is not None
            else self.prompt_families
        )
        narrowed_readouts = tuple(readouts) if readouts is not None else self.readouts
        narrowed_layers: LayerSelection
        if layers is None:
            narrowed_layers = self.layers
        elif layers == "all":
            narrowed_layers = "all"
        else:
            narrowed_layers = tuple(layers)

        _require_subset("languages", narrowed_languages, self.languages)
        _require_subset("misuses", narrowed_misuses, self.misuses)
        _require_subset("prompt_families", narrowed_families, self.prompt_families)
        _require_subset("readouts", narrowed_readouts, self.readouts)
        if self.layers != "all":
            if narrowed_layers == "all":
                raise ValueError("A layer override cannot broaden a configured layer subset")
            _require_subset("layers", narrowed_layers, self.layers)

        return Selection(
            languages=narrowed_languages,
            misuses=narrowed_misuses,
            prompt_families=narrowed_families,
            readouts=narrowed_readouts,
            layers=narrowed_layers,
        )


def _require_subset(field: str, requested: Iterable[object], configured: Iterable[object]) -> None:
    outside = set(requested) - set(configured)
    if outside:
        raise ValueError(
            f"Selection override for {field} broadens the configured selection: "
            f"{sorted(outside)!r}"
        )
