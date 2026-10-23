"""Reduced-order main-path planning primitives for Humanoid21."""

from .contracts import (
    MainPathRequest,
    MainPathState,
    SupportInterval,
    SupportSchedule,
)
from .constraints import MainPathConstraints, MainPathModel
from .diagnostics import PlanningResult, ValidationIssue, ValidationReport
from .gait_model import GaitModel, build_gait_model, default_gait_model
from .support_sequence import SupportSequencePlanner
from .trajectory import (
    FootPointTrack,
    MainPath,
    MainPathTrajectoryPlanner,
    MainPathTrajectoryRequest,
    PathPiece,
)
from .validation import MainPathValidator
from .visualize import (
    animate_main_path,
    animate_support_schedule,
    plot_support_schedule,
)

__all__ = [
    "FootPointTrack",
    "GaitModel",
    "MainPath",
    "MainPathConstraints",
    "MainPathModel",
    "MainPathRequest",
    "MainPathState",
    "MainPathTrajectoryPlanner",
    "MainPathTrajectoryRequest",
    "MainPathValidator",
    "PathPiece",
    "PlanningResult",
    "SupportInterval",
    "SupportSchedule",
    "SupportSequencePlanner",
    "ValidationIssue",
    "ValidationReport",
    "animate_main_path",
    "animate_support_schedule",
    "build_gait_model",
    "default_gait_model",
    "plot_support_schedule",
]
