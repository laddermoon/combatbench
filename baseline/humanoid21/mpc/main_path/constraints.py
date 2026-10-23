"""Model and capability constraints for continuous main-path planning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .contracts import _vec3
from .gait_model import GaitModel, default_gait_model


@dataclass(frozen=True)
class MainPathModel:
    """Fixed reduced-order model parameters."""

    mass: float
    gravity_world: np.ndarray
    control_dt: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("mass must be positive and finite")
        object.__setattr__(
            self, "gravity_world",
            _vec3(self.gravity_world, "gravity_world"))
        if not np.isfinite(self.control_dt) or self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")

    @classmethod
    def from_gait_model(cls, gait: GaitModel) -> "MainPathModel":
        return cls(
            mass=gait.mass,
            gravity_world=np.array([0.0, 0.0, -gait.gravity]),
            control_dt=gait.control_dt,
        )


@dataclass(frozen=True)
class MainPathConstraints:
    """Capability envelope used by the reduced-order planner."""

    friction: float
    max_normal_force: float
    max_force: float
    max_force_rate: float
    max_com_height_deviation: float
    max_com_speed: float
    max_ang_mom: float
    max_com_to_support_xy: float
    max_cop_x: float
    max_cop_y: float
    max_swing_speed: float
    swing_lift_height: float
    nominal_com_height: float
    tracking_time_constant: float
    angular_tracking_time_constant: float
    terminal_extension: float
    integration_dt: float
    constraint_dt: float
    com_residual_tol: float
    lin_mom_residual_tol: float
    ang_mom_residual_tol: float
    terminal_position_tol: float
    terminal_lin_mom_tol: float
    terminal_ang_mom_tol: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "friction", "max_normal_force", "max_force", "max_force_rate",
            "max_com_height_deviation", "max_com_speed", "max_ang_mom",
            "max_com_to_support_xy", "max_cop_x", "max_cop_y",
            "max_swing_speed", "swing_lift_height", "nominal_com_height",
            "tracking_time_constant", "angular_tracking_time_constant",
            "terminal_extension", "integration_dt", "constraint_dt",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")
        for name in (
            "com_residual_tol", "lin_mom_residual_tol",
            "ang_mom_residual_tol", "terminal_position_tol",
            "terminal_lin_mom_tol", "terminal_ang_mom_tol",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")

    @classmethod
    def from_gait_model(
        cls,
        gait: GaitModel,
        *,
        friction: Optional[float] = None,
    ) -> "MainPathConstraints":
        weight = gait.mass * gait.gravity
        max_com_speed = 1.5
        return cls(
            friction=0.7 if friction is None else float(friction),
            max_normal_force=1.8 * weight,
            max_force=2.0 * weight,
            max_force_rate=30.0 * weight,
            max_com_height_deviation=0.15,
            max_com_speed=max_com_speed,
            max_ang_mom=(
                2.0 * gait.mass * gait.leg_length * max_com_speed),
            max_com_to_support_xy=0.75 * gait.leg_length,
            max_cop_x=0.45 * gait.foot_length,
            max_cop_y=0.45 * gait.foot_width,
            max_swing_speed=3.0,
            swing_lift_height=0.05,
            nominal_com_height=gait.nominal_com_height,
            tracking_time_constant=0.25,
            angular_tracking_time_constant=0.30,
            terminal_extension=2.00,
            integration_dt=0.005,
            constraint_dt=0.005,
            com_residual_tol=2e-3,
            lin_mom_residual_tol=8e-2,
            ang_mom_residual_tol=8e-2,
            terminal_position_tol=gait.target_tolerance,
            terminal_lin_mom_tol=1.0,
            terminal_ang_mom_tol=1.0,
            metadata={
                "envelope_version": "heuristic_v1",
                "friction_source": "flat_ground_heuristic",
                "force_scale_N": weight,
                "ang_mom_scale": (
                    gait.mass * gait.leg_length * max_com_speed),
                "cop_bounds_source": "0.45 * foot_length / foot_width",
                "reach_source": "0.75 * leg_length",
            },
        )


def default_main_path_setup(
    gait: Optional[GaitModel] = None,
) -> tuple[MainPathModel, GaitModel, MainPathConstraints]:
    """Return model, gait envelope, and reduced capability constraints."""
    resolved = gait or default_gait_model()
    return (
        MainPathModel.from_gait_model(resolved),
        resolved,
        MainPathConstraints.from_gait_model(resolved),
    )
