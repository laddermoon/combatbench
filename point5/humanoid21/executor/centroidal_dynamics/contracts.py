"""Core contracts for Humanoid21 centroidal dynamics.

All vectors are world-frame SI quantities unless a field explicitly says
otherwise. ``ExternalLoad.torque_world`` is a free couple expressed at
``point_world``; point contacts therefore carry ``torque_world == 0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

ROBOT_IDS = ("robot_a", "robot_b")
AFF_ENV = 0
AFF_ROBOT_A = 1
AFF_ROBOT_B = 2
ROBOT_AFF = {"robot_a": AFF_ROBOT_A, "robot_b": AFF_ROBOT_B}
AFF_TO_ROBOT = {AFF_ROBOT_A: "robot_a", AFF_ROBOT_B: "robot_b"}

SOURCE_CONTACT_GROUND = "contact_ground"
SOURCE_CONTACT_WALL = "contact_wall"
SOURCE_CONTACT_OPPONENT = "contact_opponent"
SOURCE_CONTACT_SELF = "contact_self"
SOURCE_CONTACT_UNKNOWN = "contact_unknown"
SOURCE_XFRC_APPLIED = "xfrc_applied"
SOURCE_OTHER = "other"


class CentroidalDynamicsError(RuntimeError):
    """Base class for centroidal-dynamics contract violations."""


class UnknownExternalSource(CentroidalDynamicsError):
    """A contact or force cannot be assigned to a physical source."""


class AmbiguousContactDirection(CentroidalDynamicsError):
    """A contact does not identify which body receives the force."""


class OutOfDomainContact(CentroidalDynamicsError):
    """Contact mode or geometry is outside the model's declared domain."""


class StateNotConsistent(CentroidalDynamicsError):
    """Independent state extraction paths disagree beyond tolerance."""


class MissingAppliedLoad(CentroidalDynamicsError):
    """A transient applied load was not sampled before the physics step."""


class PredictionDomainViolation(CentroidalDynamicsError):
    """Prediction input violates the reduced-order model's domain."""


class UnsupportedGeneralizedLoad(CentroidalDynamicsError):
    """A qfrc-level load cannot yet be converted to an external wrench."""


def _vec3(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite (3,) vector, got {arr.shape}")
    return arr


@dataclass(frozen=True)
class CentroidalState:
    """Whole-robot centroidal state in world coordinates.

    ``ang_mom`` is always expressed about ``com_pos``. ``source`` records the
    extraction path (for example ``"manual"`` or ``"mujoco_subtree"``).
    """

    robot_id: str
    time: float
    mass: float
    com_pos: np.ndarray
    lin_mom: np.ndarray
    ang_mom: np.ndarray
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.robot_id not in ROBOT_IDS:
            raise ValueError(f"unknown robot_id {self.robot_id!r}")
        if not np.isfinite(self.time):
            raise ValueError("time must be finite")
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("mass must be finite and positive")
        object.__setattr__(self, "com_pos", _vec3(self.com_pos, "com_pos"))
        object.__setattr__(self, "lin_mom", _vec3(self.lin_mom, "lin_mom"))
        object.__setattr__(self, "ang_mom", _vec3(self.ang_mom, "ang_mom"))

    @property
    def com_vel(self) -> np.ndarray:
        return self.lin_mom / self.mass


@dataclass(frozen=True)
class ExternalLoad:
    """Spatial external load acting on the target robot.

    ``force_world`` acts at ``point_world``. ``torque_world`` is a free couple
    at that point. The equivalent moment about the whole-body COM is
    ``(point_world - com_pos) x force_world + torque_world``.
    """

    source_type: str
    robot_id: str
    point_world: np.ndarray
    force_world: np.ndarray
    torque_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))
    body_id: Optional[int] = None
    body_name: Optional[str] = None
    geom_pair: Optional[Tuple[int, int]] = None
    counterpart_aff: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.robot_id not in ROBOT_IDS:
            raise ValueError(f"unknown robot_id {self.robot_id!r}")
        object.__setattr__(
            self, "point_world", _vec3(self.point_world, "point_world"))
        object.__setattr__(
            self, "force_world", _vec3(self.force_world, "force_world"))
        object.__setattr__(
            self, "torque_world", _vec3(self.torque_world, "torque_world"))

    def moment_about(self, point_world: np.ndarray) -> np.ndarray:
        """Equivalent external moment about ``point_world``."""
        point = _vec3(point_world, "point_world")
        return np.cross(self.point_world - point, self.force_world) \
            + self.torque_world


@dataclass(frozen=True)
class GeneralizedLoad:
    """qfrc-level applied load that is not yet a spatial external wrench."""

    robot_id: str
    dof_indices: np.ndarray
    generalized_force: np.ndarray
    source_type: str = "qfrc_applied"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        idx = np.asarray(self.dof_indices, dtype=np.int32)
        val = np.asarray(self.generalized_force, dtype=np.float64)
        if idx.ndim != 1 or val.ndim != 1 or idx.shape != val.shape:
            raise ValueError("dof_indices and generalized_force must be equal 1D")
        if not np.all(np.isfinite(val)):
            raise ValueError("generalized_force must be finite")
        object.__setattr__(self, "dof_indices", idx)
        object.__setattr__(self, "generalized_force", val)


@dataclass(frozen=True)
class LoadSnapshot:
    """All known non-gravity external loads for one physics-step boundary."""

    robot_id: str
    time: float
    spatial_loads: Tuple[ExternalLoad, ...]
    generalized_loads: Tuple[GeneralizedLoad, ...] = ()
    gravity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.robot_id not in ROBOT_IDS:
            raise ValueError(f"unknown robot_id {self.robot_id!r}")
        if not np.isfinite(self.time):
            raise ValueError("time must be finite")
        object.__setattr__(self, "gravity", _vec3(self.gravity, "gravity"))
        object.__setattr__(self, "spatial_loads", tuple(self.spatial_loads))
        object.__setattr__(self, "generalized_loads",
                          tuple(self.generalized_loads))


@dataclass(frozen=True)
class BalanceStep:
    """Momentum balance for one recorded physics-step window."""

    robot_id: str
    t0: float
    t1: float
    state0: CentroidalState
    state1: CentroidalState
    lin_impulse: np.ndarray
    ang_impulse: np.ndarray
    lin_residual: np.ndarray
    ang_residual: np.ndarray
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.t0) and np.isfinite(self.t1)
                and self.t1 > self.t0):
            raise ValueError("require finite t0 < t1")
        for name in ("lin_impulse", "ang_impulse",
                     "lin_residual", "ang_residual"):
            object.__setattr__(self, name, _vec3(getattr(self, name), name))

    @property
    def dt(self) -> float:
        return self.t1 - self.t0


@dataclass(frozen=True)
class DomainIssue:
    severity: str                 # warning | invalid
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in ("warning", "invalid"):
            raise ValueError("severity must be 'warning' or 'invalid'")


@dataclass(frozen=True)
class DomainReport:
    valid: bool
    issues: Tuple[DomainIssue, ...] = ()
    margins: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        if self.valid and any(i.severity == "invalid" for i in self.issues):
            raise ValueError("valid report cannot contain invalid issues")


def skew(v: np.ndarray) -> np.ndarray:
    """Return the cross-product matrix of a 3D vector."""
    x, y, z = _vec3(v, "v")
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]], dtype=np.float64)


def quat_wxyz_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Convert a MuJoCo wxyz quaternion to a world rotation matrix."""
    from scipy.spatial.transform import Rotation as _R

    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {q.shape}")
    n = np.linalg.norm(q)
    if n <= 0.0 or not np.isfinite(n):
        raise ValueError("quaternion must be finite and non-zero")
    q = q / n
    return _R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
