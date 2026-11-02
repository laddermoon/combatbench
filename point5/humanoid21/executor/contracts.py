"""Executor contracts — the stateless GenAction interface.

Design (per bootstrip.md):

    Action = Executor.GenAction(Context)
    Context  = Observation (real current state) + P5Clip (remaining
               instruction, frame 0 anchored to current P5)

The Executor holds no execution state: every call is a fresh plan
from the current observation.  The caller owns progress by slicing
the clip to what remains.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


class ExecutorError(RuntimeError):
    pass


class AnchorError(ExecutorError):
    """Clip frame 0 does not match the observation's P5."""


class PlanError(ExecutorError):
    """Centroidal/reference planner found no acceptable plan."""


@dataclass
class Observation:
    """Real measured state at one control instant (caller-provided).

    All world-frame quantities are real simulator measurements, never
    P5-derived proxies.  ``p5`` is the local-frame state; ``frame_pose``
    its world placement (origin_xy, yaw_deg).
    """

    time: float
    robot_id: str
    core: Dict[str, np.ndarray]          # simulator CoreState (6 fields)
    p5: Any                              # Point5State (local + metadata)
    frame_pose: tuple                    # (origin_xy(2,), yaw_deg)
    p5_world: np.ndarray                 # (5,3)
    com_pos: np.ndarray                  # (3,) true mass-weighted COM
    com_vel: np.ndarray                  # (3,)
    ang_mom: np.ndarray                  # (3,)
    mass: float
    torso_tilt: float                    # rad from world up
    contacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """One GenAction call's input."""

    observation: Observation
    clip: Any                            # P5Clip — remaining instruction
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CentroidalPlan:
    """Output of algorithm 1 — the centroidal motion plan.

    ``frames`` are timed references; ``c_next`` is the next control
    step's required centroidal state.  Concrete fields TBD when
    algorithm 1 is implemented.
    """

    c_next: Optional[Dict[str, np.ndarray]] = None
    frames: list = field(default_factory=list)
    events: list = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """GenAction output — one control-step action plus diagnostics."""

    pd_target: np.ndarray                # (21,) normalized PD target
    diag: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoseResult:
    """A solved Humanoid21 reference pose for one local P5."""

    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    q_rad: np.ndarray                    # (21,) joint radians
    action: np.ndarray                   # (21,) normalized PD target
    satisfied: bool
    failed_stage: str = ""
    energy: float = 0.0
