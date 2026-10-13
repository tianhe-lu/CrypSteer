"""Typed records shared by multilingual validation and detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from crypto_steering.validation.rules.models import RuleAssessment


Language = Literal["java", "typescript", "csharp", "python"]
ProgramLabel = Literal["secure", "misuse", "mixed", "unknown"]
AnalysisProfile = Literal["whole_program", "reachable_security_flow"]
ValueOrigin = Literal[
    "constant",
    "predictable",
    "non_csprng",
    "secure_random",
    "derived_secure",
    "runtime_input",
    "unknown",
]
MISUSE_IDS = tuple(f"M{index:02d}" for index in range(1, 11))


@dataclass(frozen=True)
class SourceSpan:
    start: int = -1
    end: int = -1
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None


@dataclass(frozen=True)
class ResolutionStep:
    expression: str
    action: str
    result: str
    span: SourceSpan = field(default_factory=SourceSpan)


@dataclass(frozen=True)
class ResolvedValue:
    expression: str
    origin: ValueOrigin
    values: tuple[str, ...] = field(default_factory=tuple)
    steps: tuple[ResolutionStep, ...] = field(default_factory=tuple)
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str
    message: str
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None


@dataclass(frozen=True)
class CompileOutcome:
    attempted: bool
    success: bool
    parse_success: bool
    frontend: str
    command: tuple[str, ...] = field(default_factory=tuple)
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    reason_code: str = "compile_not_attempted"


@dataclass(frozen=True)
class ImportFinding:
    module: str
    imported_name: str | None
    approved: bool
    resolved: bool
    reason_code: str


@dataclass(frozen=True)
class ImportOutcome:
    attempted: bool
    success: bool
    findings: tuple[ImportFinding, ...] = field(default_factory=tuple)
    reason_code: str = "not_applicable"


@dataclass(frozen=True)
class DetectorFinding:
    misuse_id: str
    language: Language
    sink_id: str
    sink_name: str
    expression: str
    label: ProgramLabel
    reason_code: str
    reason: str
    policy_id: str = ""
    span: SourceSpan = field(default_factory=SourceSpan)
    resolved_values: tuple[str, ...] = field(default_factory=tuple)
    value_origin: ValueOrigin = "unknown"
    resolution_steps: tuple[ResolutionStep, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionResult:
    misuse_id: str
    language: Language
    label: ProgramLabel
    reason: str
    findings: tuple[DetectorFinding, ...] = field(default_factory=tuple)
    analysis_error: str | None = None


def combine_detector_findings(
    misuse_id: str,
    language: Language,
    findings: list[DetectorFinding] | tuple[DetectorFinding, ...],
    *,
    no_sink_reason: str,
) -> DetectionResult:
    """Apply the documented secure/misuse/mixed/unknown aggregation policy."""

    items = tuple(findings)
    labels = {item.label for item in items}
    if "mixed" in labels or {"secure", "misuse"} <= labels:
        label: ProgramLabel = "mixed"
        reason = "secure and misuse evidence coexist at registered sinks"
    elif "misuse" in labels:
        # A definite misuse remains actionable even if an additional sink is
        # unresolved. This is the asymmetric rule documented for the audit.
        label = "misuse"
        reason = "at least one registered sink has resolved misuse evidence"
    elif "secure" in labels and "unknown" in labels:
        label = "unknown"
        reason = "secure evidence coexists with an unresolved registered sink"
    elif labels == {"secure"}:
        label = "secure"
        reason = "all registered sink evidence is resolved secure"
    elif items:
        label = "unknown"
        reason = "registered sink evidence could not be resolved"
    else:
        label = "unknown"
        reason = no_sink_reason
    return DetectionResult(
        misuse_id=misuse_id,
        language=language,
        label=label,
        reason=reason,
        findings=items,
    )


@dataclass(frozen=True)
class TaskEvidence:
    operation: str
    expression: str
    span: SourceSpan = field(default_factory=SourceSpan)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskValidationResult:
    language: Language
    task_id: str
    success: bool
    reason_codes: tuple[str, ...]
    reason: str
    matched_operations: tuple[TaskEvidence, ...] = field(default_factory=tuple)
    rule_assessments: tuple[RuleAssessment, ...] = field(default_factory=tuple)


def jsonable(value: Any) -> Any:
    """Convert dataclasses and paths to JSON-compatible values."""

    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


T = TypeVar("T")


ProviderStatus = Literal["completed", "skipped", "unsupported", "error"]
ValidationMode = Literal["strict", "original_scaffold", "multi_operation"]
CorpusKind = Literal[
    "dataset_original",
    "dataset_rewrite",
    "probe_multilingual",
    "probe_pca_interpretation",
    "probe_security_composition",
    "generated",
]


@dataclass(frozen=True)
class ProviderExecution(Generic[T]):
    """Execution state that cannot be confused with a scientific label."""

    status: ProviderStatus
    value: T | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"completed", "skipped", "unsupported", "error"}:
            raise ValueError(f"Unsupported provider status {self.status!r}")
        if self.status == "completed" and self.value is None:
            raise ValueError("A completed provider execution requires a value")
        if self.status != "completed" and self.value is not None:
            raise ValueError(
                f"Provider status {self.status!r} must not carry a scientific value"
            )

    @classmethod
    def completed(cls, value: T) -> "ProviderExecution[T]":
        return cls(status="completed", value=value)

    @classmethod
    def skipped(cls, message: str) -> "ProviderExecution[T]":
        return cls(status="skipped", message=message)

    @classmethod
    def unsupported(cls, message: str) -> "ProviderExecution[T]":
        return cls(status="unsupported", message=message)

    @classmethod
    def error(cls, message: str) -> "ProviderExecution[T]":
        return cls(status="error", message=message)


@dataclass(frozen=True)
class ProgramInput:
    """One source plus independently supplied validation expectations."""

    program_id: str
    relative_path: str
    source_path: Path
    language: Language
    source: str
    corpus_kind: CorpusKind
    validation_mode: ValidationMode
    analysis_profile: AnalysisProfile = "whole_program"
    target_misuse_id: str | None = None
    task_id: str | None = None
    intended_labels: dict[str, ProgramLabel] = field(default_factory=dict)
    operation_functions: dict[str, str] = field(default_factory=dict)
    expected_rule: str | None = None
    expected_structure: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.program_id.strip():
            raise ValueError("program_id must not be empty")
        if not self.relative_path.strip():
            raise ValueError("relative_path must not be empty")
        if self.language not in {"java", "typescript", "csharp", "python"}:
            raise ValueError(f"Unsupported language {self.language!r}")
        if self.analysis_profile not in {
            "whole_program",
            "reachable_security_flow",
        }:
            raise ValueError(
                f"Unsupported analysis profile {self.analysis_profile!r}"
            )
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must contain text")
        invalid = set(self.intended_labels) - set(MISUSE_IDS)
        if invalid:
            raise ValueError(f"Unsupported intended-label keys: {sorted(invalid)!r}")
        invalid_operations = set(self.operation_functions) - set(MISUSE_IDS)
        if invalid_operations:
            raise ValueError(
                "Unsupported operation-function keys: "
                f"{sorted(invalid_operations)!r}"
            )
        if self.validation_mode == "multi_operation":
            if set(self.operation_functions) != set(MISUSE_IDS):
                raise ValueError(
                    "multi_operation programs require one operation function "
                    "for every M01-M10 detector"
                )
            names = tuple(self.operation_functions.values())
            if any(not name.strip() for name in names) or len(set(names)) != len(names):
                raise ValueError(
                    "multi_operation operation functions must be non-empty and unique"
                )
        elif self.operation_functions:
            raise ValueError(
                "operation_functions are valid only for multi_operation programs"
            )
        if self.target_misuse_id is not None and self.target_misuse_id not in MISUSE_IDS:
            raise ValueError(f"Unsupported target misuse {self.target_misuse_id!r}")
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "intended_labels", dict(self.intended_labels))
        object.__setattr__(self, "operation_functions", dict(self.operation_functions))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ValidationResult:
    program: ProgramInput
    frontend: ProviderExecution[CompileOutcome]
    imports: ProviderExecution[ImportOutcome]
    tasks: dict[str, ProviderExecution[TaskValidationResult]]
    detectors: dict[str, ProviderExecution[DetectionResult]]
    program_valid: bool
    intended_label_matches: dict[str, bool]
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    validity_rules: tuple[RuleAssessment, ...] = field(default_factory=tuple)

    @property
    def provider_errors(self) -> tuple[str, ...]:
        failures = []
        for name, execution in (
            ("frontend", self.frontend),
            ("imports", self.imports),
            *((f"task:{key}", value) for key, value in self.tasks.items()),
            *((f"detector:{key}", value) for key, value in self.detectors.items()),
        ):
            if execution.status in {"unsupported", "error"}:
                failures.append(name)
        return tuple(failures)
