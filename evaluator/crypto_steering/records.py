"""Immutable identities shared across the research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path

from crypto_steering.languages import canonical_language


MISUSE_IDS = tuple(f"M{index:02d}" for index in range(1, 11))
PROMPT_FAMILIES = ("code_only", "generation_replay")
READOUTS = ("code_body_mean", "end_of_code")
LAYER_PROBE_PROTOCOLS = (
    "leave_one_misuse_out",
    "leave_one_language_out",
)
VECTOR_SOURCE_LEVELS = (
    "pairwise_structure_specific",
    "pairwise_global",
    "misuse_structure_specific_global",
    "misuse_specific_global",
)
GENERAL_MEAN_METHODS = (
    "raw_mean",
    "normalized_pair_mean",
    "cell_balanced_raw_mean",
    "cell_balanced_normalized_mean",
)
PCA_INPUT_MODES = ("raw", "normalized")
PCA_POINT_POPULATIONS = (
    "misuse_to_secure_difference",
    "structure_specific_pair_centered_state",
    "structure_specific_uncentered_state",
    "structure_pooled_pair_centered_state",
    "structure_pooled_uncentered_state",
)
HEATMAP_DIRECTION_MODES = (
    "pairwise_normalized_security_pc",
    "leave_one_language_out_security_pc",
    "leave_one_misuse_out_security_pc",
    "leave_one_language_and_misuse_out_security_pc",
)
SUPPORTED_HEATMAP_DIRECTION_MODES = (
    *HEATMAP_DIRECTION_MODES,
    "leave_one_language_out_same_misuse_security_pc",
)
POOLED_STRUCTURE = "pooled"


class ProgramLabel(str, Enum):
    SECURE = "secure"
    MISUSE = "misuse"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ProviderStatus(str, Enum):
    COMPLETED = "completed"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class AnalysisStatus(str, Enum):
    COMPLETED = "completed"
    NOT_RUN_INSUFFICIENT_ITEMS = "not_run_insufficient_items"
    NOT_RUN_ZERO_NORM = "not_run_zero_norm"
    NOT_RUN_MISSING_INPUT = "not_run_missing_input"
    NOT_RUN_RANK_DEFICIENT = "not_run_rank_deficient"
    NOT_RUN_DISABLED = "not_run_disabled"


def _text(value: object, field: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _misuse_id(value: object) -> str:
    misuse_id = _text(value, "misuse_id").upper()
    if misuse_id not in MISUSE_IDS:
        expected = ", ".join(MISUSE_IDS)
        raise ValueError(f"Unsupported misuse ID {value!r}; expected one of: {expected}")
    return misuse_id


def _prompt_family(value: object) -> str:
    family = _text(value, "prompt_family")
    if family not in PROMPT_FAMILIES:
        expected = ", ".join(PROMPT_FAMILIES)
        raise ValueError(
            f"Unsupported prompt family {value!r}; expected one of: {expected}"
        )
    return family


def _readout(value: object) -> str:
    readout = _text(value, "readout")
    if readout not in READOUTS:
        expected = ", ".join(READOUTS)
        raise ValueError(f"Unsupported readout {value!r}; expected one of: {expected}")
    return readout


def _layer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"layer must be a non-negative integer, got {value!r}")
    return value


def _source_level(value: object) -> str:
    source_level = _text(value, "source_level")
    if source_level not in VECTOR_SOURCE_LEVELS:
        raise ValueError(
            f"Unsupported source level {source_level!r}; expected one of "
            f"{VECTOR_SOURCE_LEVELS!r}"
        )
    return source_level


@dataclass(frozen=True)
class SampleKey:
    language: str
    misuse_id: str
    task_id: str
    sample_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "language", canonical_language(self.language))
        object.__setattr__(self, "misuse_id", _misuse_id(self.misuse_id))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "sample_id", _text(self.sample_id, "sample_id"))


@dataclass(frozen=True)
class VariantKey:
    sample_key: SampleKey
    label: ProgramLabel
    rule_id: str
    structure: str
    rewrite_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", ProgramLabel(self.label))
        object.__setattr__(self, "rule_id", _text(self.rule_id, "rule_id"))
        object.__setattr__(self, "structure", _text(self.structure, "structure"))
        object.__setattr__(self, "rewrite_id", _text(self.rewrite_id, "rewrite_id"))


@dataclass(frozen=True)
class ProgramVariant:
    key: VariantKey
    source_path: Path
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        if not isinstance(self.source, str):
            raise TypeError("source must be text")


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    variant_key: VariantKey
    prompt_family: str
    text: str
    source_span: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_id", _text(self.prompt_id, "prompt_id"))
        object.__setattr__(self, "prompt_family", _prompt_family(self.prompt_family))
        if not isinstance(self.text, str):
            raise TypeError("text must be text")
        start, end = self.source_span
        if start < 0 or end < start or end > len(self.text):
            raise ValueError(
                f"source_span {self.source_span!r} is outside prompt text of "
                f"length {len(self.text)}"
            )


@dataclass(frozen=True)
class StandaloneSourceRecord:
    prompt_id: str
    language: str
    source_path: Path
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_id", _text(self.prompt_id, "prompt_id"))
        object.__setattr__(self, "language", canonical_language(self.language))
        object.__setattr__(self, "source_path", Path(self.source_path))
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Standalone source must contain source text")


@dataclass(frozen=True)
class HiddenPartitionKey:
    model_run: str
    misuse_id: str
    prompt_family: str
    language: str
    readout: str
    layer: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_run", _text(self.model_run, "model_run"))
        object.__setattr__(self, "misuse_id", _misuse_id(self.misuse_id))
        object.__setattr__(self, "prompt_family", _prompt_family(self.prompt_family))
        object.__setattr__(self, "language", canonical_language(self.language))
        object.__setattr__(self, "readout", _readout(self.readout))
        object.__setattr__(self, "layer", _layer(self.layer))


@dataclass(frozen=True)
class LayerProbeFold:
    protocol: str
    fold_id: str
    train_languages: tuple[str, ...]
    test_languages: tuple[str, ...]
    train_misuses: tuple[str, ...]
    test_misuses: tuple[str, ...]

    def __post_init__(self) -> None:
        protocol = _text(self.protocol, "protocol")
        if protocol not in LAYER_PROBE_PROTOCOLS:
            raise ValueError(
                f"Unsupported layer-probe protocol {protocol!r}; expected "
                f"{LAYER_PROBE_PROTOCOLS!r}"
            )
        object.__setattr__(self, "protocol", protocol)
        languages = tuple(
            tuple(canonical_language(value) for value in values)
            for values in (self.train_languages, self.test_languages)
        )
        misuses = tuple(
            tuple(_misuse_id(value) for value in values)
            for values in (self.train_misuses, self.test_misuses)
        )
        for name, values in (
            ("train_languages", languages[0]),
            ("test_languages", languages[1]),
            ("train_misuses", misuses[0]),
            ("test_misuses", misuses[1]),
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
            object.__setattr__(self, name, values)
        if protocol == "leave_one_misuse_out":
            if (
                languages[0] != languages[1]
                or len(misuses[1]) != 1
                or set(misuses[0]) & set(misuses[1])
            ):
                raise ValueError("Invalid leave-one-misuse-out fold coordinates")
            expected_id = f"holdout_{misuses[1][0]}"
        else:
            if (
                misuses[0] != misuses[1]
                or len(languages[1]) != 1
                or set(languages[0]) & set(languages[1])
            ):
                raise ValueError("Invalid leave-one-language-out fold coordinates")
            expected_id = f"holdout_{languages[1][0]}"
        fold_id = _text(self.fold_id, "fold_id")
        if fold_id != expected_id:
            raise ValueError(f"fold_id must be {expected_id!r}")
        object.__setattr__(self, "fold_id", fold_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "fold_id": self.fold_id,
            "train_languages": list(self.train_languages),
            "test_languages": list(self.test_languages),
            "train_misuses": list(self.train_misuses),
            "test_misuses": list(self.test_misuses),
        }


@dataclass(frozen=True)
class LayerProbeFoldKey:
    model_run: str
    prompt_family: str
    readout: str
    layer: int
    protocol: str
    fold_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_run", _text(self.model_run, "model_run"))
        object.__setattr__(self, "prompt_family", _prompt_family(self.prompt_family))
        object.__setattr__(self, "readout", _readout(self.readout))
        object.__setattr__(self, "layer", _layer(self.layer))
        protocol = _text(self.protocol, "protocol")
        if protocol not in LAYER_PROBE_PROTOCOLS:
            raise ValueError(f"Unsupported layer-probe protocol {protocol!r}")
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "fold_id", _text(self.fold_id, "fold_id"))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_run": self.model_run,
            "prompt_family": self.prompt_family,
            "readout": self.readout,
            "layer": self.layer,
            "protocol": self.protocol,
            "fold_id": self.fold_id,
        }


@dataclass(frozen=True)
class LayerSelectionRecord:
    model_run: str
    analysis_id: str
    prompt_family: str
    readout: str
    layer: int
    transformer_layer_count: int
    lomo_macro_auroc: float
    lolo_macro_auroc: float
    joint_auroc: float
    min_protocol_auroc: float
    maximin_sensitivity_layer: int
    normalized_depth: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_run", _text(self.model_run, "model_run"))
        object.__setattr__(self, "analysis_id", _text(self.analysis_id, "analysis_id"))
        object.__setattr__(self, "prompt_family", _prompt_family(self.prompt_family))
        object.__setattr__(self, "readout", _readout(self.readout))
        layer = _layer(self.layer)
        if (
            isinstance(self.transformer_layer_count, bool)
            or not isinstance(self.transformer_layer_count, int)
            or self.transformer_layer_count < 2
        ):
            raise ValueError("transformer_layer_count must be an integer >= 2")
        if layer >= self.transformer_layer_count:
            raise ValueError("selected layer exceeds transformer_layer_count")
        maximin = _layer(self.maximin_sensitivity_layer)
        if maximin >= self.transformer_layer_count:
            raise ValueError("maximin_sensitivity_layer exceeds transformer_layer_count")
        for name in (
            "lomo_macro_auroc",
            "lolo_macro_auroc",
            "joint_auroc",
            "min_protocol_auroc",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "normalized_depth",
            layer / (self.transformer_layer_count - 1),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_run": self.model_run,
            "analysis_id": self.analysis_id,
            "prompt_family": self.prompt_family,
            "readout": self.readout,
            "layer": self.layer,
            "transformer_layer_count": self.transformer_layer_count,
            "normalized_depth": self.normalized_depth,
            "lomo_macro_auroc": self.lomo_macro_auroc,
            "lolo_macro_auroc": self.lolo_macro_auroc,
            "joint_auroc": self.joint_auroc,
            "min_protocol_auroc": self.min_protocol_auroc,
            "maximin_sensitivity_layer": self.maximin_sensitivity_layer,
        }


@dataclass(frozen=True)
class HiddenStateRow:
    prompt_id: str
    variant_key: VariantKey
    tensor_shard: Path
    tensor_row: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_id", _text(self.prompt_id, "prompt_id"))
        object.__setattr__(self, "tensor_shard", Path(self.tensor_shard))
        if (
            isinstance(self.tensor_row, bool)
            or not isinstance(self.tensor_row, int)
            or self.tensor_row < 0
        ):
            raise ValueError("tensor_row must be a non-negative integer")


@dataclass(frozen=True)
class VectorPartitionKey:
    model_run: str
    misuse_id: str
    prompt_family: str
    language: str
    readout: str
    layer: int
    source_level: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_run", _text(self.model_run, "model_run"))
        object.__setattr__(self, "misuse_id", _misuse_id(self.misuse_id))
        object.__setattr__(self, "prompt_family", _prompt_family(self.prompt_family))
        object.__setattr__(self, "language", canonical_language(self.language))
        object.__setattr__(self, "readout", _readout(self.readout))
        object.__setattr__(self, "layer", _layer(self.layer))
        object.__setattr__(self, "source_level", _source_level(self.source_level))


@dataclass(frozen=True)
class PairVectorKey:
    sample_key: SampleKey
    prompt_family: str
    readout: str
    layer: int
    structure: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_family", _prompt_family(self.prompt_family))
        object.__setattr__(self, "readout", _readout(self.readout))
        object.__setattr__(self, "layer", _layer(self.layer))
        object.__setattr__(self, "structure", _text(self.structure, "structure"))


@dataclass(frozen=True)
class CellVectorKey:
    language: str
    misuse_id: str
    prompt_family: str
    readout: str
    layer: int
    structure: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "language", canonical_language(self.language))
        object.__setattr__(self, "misuse_id", _misuse_id(self.misuse_id))
        object.__setattr__(self, "prompt_family", _prompt_family(self.prompt_family))
        object.__setattr__(self, "readout", _readout(self.readout))
        object.__setattr__(self, "layer", _layer(self.layer))
        object.__setattr__(self, "structure", _text(self.structure, "structure"))


@dataclass(frozen=True)
class GeneralVectorKey:
    scope: str
    source_level: str
    method: str
    prompt_family: str
    readout: str
    layer: int
    structure: str

    def __post_init__(self) -> None:
        for field in ("scope", "method", "structure"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "source_level", _source_level(self.source_level))
        if self.method not in GENERAL_MEAN_METHODS:
            raise ValueError(
                f"Unsupported general representation method {self.method!r}; expected one of "
                f"{GENERAL_MEAN_METHODS!r}"
            )
        object.__setattr__(self, "prompt_family", _prompt_family(self.prompt_family))
        object.__setattr__(self, "readout", _readout(self.readout))
        object.__setattr__(self, "layer", _layer(self.layer))


@dataclass(frozen=True)
class PcaBasisKey:
    model_run: str
    prompt_family: str
    readout: str
    layer: int
    source_level: str
    scope_id: str
    input_mode: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_run", _text(self.model_run, "model_run"))
        object.__setattr__(self, "prompt_family", _prompt_family(self.prompt_family))
        object.__setattr__(self, "readout", _readout(self.readout))
        object.__setattr__(self, "layer", _layer(self.layer))
        object.__setattr__(self, "source_level", _source_level(self.source_level))
        object.__setattr__(self, "scope_id", _text(self.scope_id, "scope_id"))
        input_mode = _text(self.input_mode, "input_mode")
        if input_mode not in PCA_INPUT_MODES:
            raise ValueError(
                f"Unsupported PCA input mode {input_mode!r}; expected {PCA_INPUT_MODES!r}"
            )
        object.__setattr__(self, "input_mode", input_mode)


@dataclass(frozen=True)
class PcaComponentRecord:
    basis_key: PcaBasisKey
    native_component_index: int
    tensor_row: int
    singular_value: float
    explained_energy_ratio: float
    cumulative_explained_energy_ratio: float

    def __post_init__(self) -> None:
        for field in ("native_component_index", "tensor_row"):
            value = getattr(self, field)
            minimum = 1 if field == "native_component_index" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
        for field in (
            "singular_value",
            "explained_energy_ratio",
            "cumulative_explained_energy_ratio",
        ):
            value = float(getattr(self, field))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")
            object.__setattr__(self, field, value)


@dataclass(frozen=True)
class PcaProjectionRecord:
    basis_key: PcaBasisKey
    point_population: str
    point_id: str
    pair_id: str
    language: str
    misuse_id: str
    task_id: str
    sample_id: str
    rule_id: str
    structure: str
    security_label: str
    source_path: Path | None
    native_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        population = _text(self.point_population, "point_population")
        if population not in PCA_POINT_POPULATIONS:
            raise ValueError(
                f"Unsupported PCA point population {population!r}; expected "
                f"{PCA_POINT_POPULATIONS!r}"
            )
        object.__setattr__(self, "point_population", population)
        for field in ("point_id", "pair_id"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "language", canonical_language(self.language))
        object.__setattr__(self, "misuse_id", _misuse_id(self.misuse_id))
        for field in ("task_id", "sample_id", "rule_id", "structure", "security_label"):
            if not isinstance(getattr(self, field), str):
                raise TypeError(f"{field} must be text")
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))
        scores = tuple(float(value) for value in self.native_scores)
        if not scores or not all(isfinite(value) for value in scores):
            raise ValueError("native_scores must contain finite component scores")
        object.__setattr__(self, "native_scores", scores)


@dataclass(frozen=True)
class ProbeVariantRecord:
    probe_id: str
    pair_id: str
    language: str
    misuse_id: str
    condition: str
    security_label: ProgramLabel
    source_path: Path

    def __post_init__(self) -> None:
        for field in ("probe_id", "pair_id", "condition"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "language", canonical_language(self.language))
        object.__setattr__(self, "misuse_id", _misuse_id(self.misuse_id))
        label = ProgramLabel(self.security_label)
        if label not in {ProgramLabel.SECURE, ProgramLabel.MISUSE}:
            raise ValueError("A probe variant must be secure or misuse")
        object.__setattr__(self, "security_label", label)
        object.__setattr__(self, "source_path", Path(self.source_path))


@dataclass(frozen=True)
class ProbePairRecord:
    pair_id: str
    language: str
    misuse_id: str
    condition: str
    secure: ProbeVariantRecord
    misuse: ProbeVariantRecord

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_id", _text(self.pair_id, "pair_id"))
        object.__setattr__(self, "language", canonical_language(self.language))
        object.__setattr__(self, "misuse_id", _misuse_id(self.misuse_id))
        object.__setattr__(self, "condition", _text(self.condition, "condition"))
        for variant, label in (
            (self.secure, ProgramLabel.SECURE),
            (self.misuse, ProgramLabel.MISUSE),
        ):
            if variant.pair_id != self.pair_id or variant.security_label is not label:
                raise ValueError("Probe variants do not match the pair identity and labels")
            if (
                variant.language,
                variant.misuse_id,
                variant.condition,
            ) != (self.language, self.misuse_id, self.condition):
                raise ValueError("Probe variant coordinates do not match the pair")


@dataclass(frozen=True)
class CompositionVariantRecord:
    """One program in the ten-decision full-factorial PCA probe corpus."""

    variant_id: str
    language: str
    composition_bits: str
    misuse_count: int
    active_misuse_ids: tuple[str, ...]
    source_path: Path

    def __post_init__(self) -> None:
        variant_id = _text(self.variant_id, "variant_id")
        language = canonical_language(self.language)
        object.__setattr__(self, "variant_id", variant_id)
        object.__setattr__(self, "language", language)
        bits = _text(self.composition_bits, "composition_bits")
        if len(bits) != len(MISUSE_IDS) or set(bits) - {"0", "1"}:
            raise ValueError(
                "composition_bits must contain ten binary choices in M01-M10 order"
            )
        object.__setattr__(self, "composition_bits", bits)
        expected_id = f"pca_composition__{language}__{bits}"
        if variant_id != expected_id:
            raise ValueError(
                f"variant_id must be the canonical composition identity {expected_id!r}"
            )
        if (
            isinstance(self.misuse_count, bool)
            or not isinstance(self.misuse_count, int)
            or self.misuse_count != bits.count("1")
        ):
            raise ValueError("misuse_count must equal the number of one bits")
        active = tuple(_misuse_id(value) for value in self.active_misuse_ids)
        expected = tuple(
            misuse_id
            for misuse_id, bit in zip(MISUSE_IDS, bits, strict=True)
            if bit == "1"
        )
        if active != expected:
            raise ValueError(
                "active_misuse_ids must exactly follow the one bits in M01-M10 order"
            )
        object.__setattr__(self, "active_misuse_ids", active)
        object.__setattr__(self, "source_path", Path(self.source_path))


@dataclass(frozen=True)
class HeatmapDirectionKey:
    model_run: str
    prompt_family: str
    readout: str
    layer: int
    vector_mode: str
    target_misuse_id: str
    source_languages: tuple[str, ...]
    source_misuse_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_run", _text(self.model_run, "model_run"))
        object.__setattr__(self, "prompt_family", _prompt_family(self.prompt_family))
        object.__setattr__(self, "readout", _readout(self.readout))
        object.__setattr__(self, "layer", _layer(self.layer))
        mode = _text(self.vector_mode, "vector_mode")
        if mode not in SUPPORTED_HEATMAP_DIRECTION_MODES:
            raise ValueError(
                f"Unsupported heatmap direction mode {mode!r}; expected "
                f"{SUPPORTED_HEATMAP_DIRECTION_MODES!r}"
            )
        object.__setattr__(self, "vector_mode", mode)
        object.__setattr__(self, "target_misuse_id", _misuse_id(self.target_misuse_id))
        languages = tuple(canonical_language(value) for value in self.source_languages)
        if not languages or len(languages) != len(set(languages)):
            raise ValueError("source_languages must contain unique languages")
        object.__setattr__(self, "source_languages", languages)
        misuses = tuple(_misuse_id(value) for value in self.source_misuse_ids)
        if not misuses or len(misuses) != len(set(misuses)):
            raise ValueError("source_misuse_ids must contain unique misuse IDs")
        object.__setattr__(self, "source_misuse_ids", misuses)


@dataclass(frozen=True)
class SteeringPromptKey:
    language: str
    misuse_id: str
    prompt_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "language", canonical_language(self.language))
        object.__setattr__(self, "misuse_id", _misuse_id(self.misuse_id))
        object.__setattr__(self, "prompt_id", _text(self.prompt_id, "prompt_id"))


@dataclass(frozen=True)
class SteeringVectorKey:
    method: str
    scope: str
    prompt_family: str
    readout: str
    layer: int
    structure: str
    source_level: str | None = None
    component: int | None = None
    input_mode: str | None = None

    def __post_init__(self) -> None:
        for field in ("method", "scope", "structure"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "prompt_family", _prompt_family(self.prompt_family))
        object.__setattr__(self, "readout", _readout(self.readout))
        object.__setattr__(self, "layer", _layer(self.layer))
        if self.source_level is not None:
            object.__setattr__(self, "source_level", _source_level(self.source_level))
        if self.component is not None:
            if (
                isinstance(self.component, bool)
                or not isinstance(self.component, int)
                or self.component < 1
            ):
                raise ValueError("component must be a positive integer when present")
        if self.input_mode is not None:
            input_mode = _text(self.input_mode, "input_mode")
            if input_mode not in PCA_INPUT_MODES:
                raise ValueError(
                    f"Unsupported PCA input mode {input_mode!r}; expected "
                    f"{PCA_INPUT_MODES!r}"
                )
            object.__setattr__(self, "input_mode", input_mode)


@dataclass(frozen=True)
class SteeringConditionKey:
    vector_key: SteeringVectorKey
    target_language: str
    target_misuse: str
    alpha: float
    application_site: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_language", canonical_language(self.target_language))
        object.__setattr__(self, "target_misuse", _misuse_id(self.target_misuse))
        alpha = float(self.alpha)
        if not isfinite(alpha):
            raise ValueError("alpha must be finite")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(
            self,
            "application_site",
            _text(self.application_site, "application_site"),
        )


@dataclass(frozen=True)
class GenerationKey:
    run_id: str
    prompt_id: str
    condition_id: str
    sample_index: int

    def __post_init__(self) -> None:
        for field in ("run_id", "prompt_id", "condition_id"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if (
            isinstance(self.sample_index, bool)
            or not isinstance(self.sample_index, int)
            or self.sample_index < 0
        ):
            raise ValueError("sample_index must be a non-negative integer")


@dataclass(frozen=True)
class ProviderOutcome:
    """Keep provider execution state separate from scientific labels."""

    status: ProviderStatus
    label: ProgramLabel | None = None
    message: str = ""

    def __post_init__(self) -> None:
        status = ProviderStatus(self.status)
        label = ProgramLabel(self.label) if self.label is not None else None
        if status is ProviderStatus.COMPLETED and label is None:
            raise ValueError("A completed provider outcome requires a program label")
        if status is not ProviderStatus.COMPLETED and label is not None:
            raise ValueError(
                f"Provider status {status.value!r} must not produce a scientific label"
            )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "label", label)

    @classmethod
    def unsupported(cls, message: str = "Provider is unsupported") -> ProviderOutcome:
        return cls(status=ProviderStatus.UNSUPPORTED, message=message)
