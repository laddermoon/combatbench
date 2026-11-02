"""Centroidal dynamics contracts and reduced-order model for Humanoid21."""

from .contracts import (
    AFF_ENV,
    AFF_ROBOT_A,
    AFF_ROBOT_B,
    ROBOT_IDS,
    AmbiguousContactDirection,
    BalanceStep,
    CentroidalDynamicsError,
    CentroidalState,
    DomainIssue,
    DomainReport,
    ExternalLoad,
    GeneralizedLoad,
    LoadSnapshot,
    MissingAppliedLoad,
    OutOfDomainContact,
    PredictionDomainViolation,
    StateNotConsistent,
    UnknownExternalSource,
    UnsupportedGeneralizedLoad,
    skew,
)
from .model_spec import CentroidalModelSpec, RobotModelSpec, build_model_spec
from .external_loads import ExternalLoadNormalizer
from .balance import momentum_balance
from .probe import PhysicsStepProbe, ProbeStepResult
from .domain import CentroidalDomainChecker, ContactAssessment, DomainLimits
from .predict import (
    DiscreteCentroidalPredictor,
    PredictionResult,
    PredictionStep,
)
from .planning import (
    CentroidalPlan,
    PlanContext,
    build_centroidal_plan,
    direct_reclip_solver,
    make_plan_context,
    run_same_budget_comparison,
    summarize_solver_result,
)
from .state import (
    CentroidalStateExtractor,
    ConsistencyTolerances,
    extract_centroidal_state,
)

__all__ = [
    "AFF_ENV",
    "AFF_ROBOT_A",
    "AFF_ROBOT_B",
    "ROBOT_IDS",
    "AmbiguousContactDirection",
    "BalanceStep",
    "CentroidalDynamicsError",
    "CentroidalModelSpec",
    "CentroidalState",
    "CentroidalStateExtractor",
    "CentroidalDomainChecker",
    "CentroidalPlan",
    "ConsistencyTolerances",
    "ContactAssessment",
    "DomainIssue",
    "DomainLimits",
    "DomainReport",
    "DiscreteCentroidalPredictor",
    "ExternalLoad",
    "ExternalLoadNormalizer",
    "GeneralizedLoad",
    "LoadSnapshot",
    "MissingAppliedLoad",
    "PhysicsStepProbe",
    "OutOfDomainContact",
    "PlanContext",
    "ProbeStepResult",
    "PredictionDomainViolation",
    "PredictionResult",
    "PredictionStep",
    "RobotModelSpec",
    "StateNotConsistent",
    "UnknownExternalSource",
    "UnsupportedGeneralizedLoad",
    "build_centroidal_plan",
    "build_model_spec",
    "direct_reclip_solver",
    "extract_centroidal_state",
    "make_plan_context",
    "momentum_balance",
    "run_same_budget_comparison",
    "skew",
    "summarize_solver_result",
]
