"""Executor contracts — the stateless GenAction interface.

    Action  = Executor.GenAction(Context)
    Context = Observation (the simulator's own observation) +
              P5Clip (remaining instruction, frame 0 anchored)

The Executor holds no execution state; the caller owns progress.
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
    """The Humanoid21 simulator's own observation, verbatim.

    ``derived`` is exactly ``sim.get_derived_state()[robot_id]``
    (the structured per-robot view incl. the 96-dim flat
    ``observation``); ``core`` is ``get_core_state()[robot_id]``
    when the caller also supplies it.
    """

    robot_id: str
    time: float
    derived: Dict[str, Any]
    core: Optional[Dict[str, np.ndarray]] = None


# simulator input: the (21,) normalized PD target for one control step
Action = np.ndarray


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
    step's required centroidal state.  Concrete fields TBD.
    """

    c_next: Optional[Dict[str, np.ndarray]] = None
    frames: list = field(default_factory=list)
    events: list = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


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
