"""Legacy result records still used by Java value-flow compatibility code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ProgramLabel = Literal["secure", "misuse", "mixed", "unknown"]


@dataclass(frozen=True)
class DetectorSinkFinding:
    sink_name: str
    expression: str
    label: ProgramLabel
    reason: str
    start: int = -1
    end: int = -1


@dataclass(frozen=True)
class DetectionResult:
    misuse_id: str
    label: ProgramLabel
    findings: tuple[DetectorSinkFinding, ...]
    reason: str
