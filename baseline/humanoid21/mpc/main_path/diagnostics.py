"""Structured diagnostics for main-path planning and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ValidationIssue:
    """One machine-readable planning or validation failure."""

    code: str
    message: str
    time: Optional[float] = None
    interval: Optional[Tuple[float, float]] = None
    measured: Optional[float] = None
    limit: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    """Result of independently checking a main path."""

    passed: bool
    issues: Tuple[ValidationIssue, ...] = ()
    max_com_residual: float = 0.0
    max_lin_mom_residual: float = 0.0
    max_ang_mom_residual: float = 0.0
    method: str = "piecewise_rk4_dense_constraints"


@dataclass(frozen=True)
class PlanningResult:
    """Result returned by the second-stage trajectory planner."""

    success: bool
    path: Optional[Any]
    report: ValidationReport
    diagnostics: Dict[str, Any] = field(default_factory=dict)
