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


# The policy's observation — the flat 96-dim vector that a policy
# receives (get_derived_state()[robot_id]['observation']).  The
# Executor is a policy backend: this is ALL it may see.  Everything
# it needs (P5, centroidal state estimate, support cues) must be
# reconstructed from this vector — no core state, no sim handles.
Observation = np.ndarray


# simulator input: the (21,) normalized PD target for one control step
Action = np.ndarray


@dataclass
class Context:
    """One GenAction call's input."""

    observation: Observation
    clip: Any                            # P5Clip — remaining instruction
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FootSupport:
    """One foot's support state inside a LandMark.

    ``force_ratio`` is normal load divided by total body weight
    (matches obs[52:54]); 0 means no contact.  ``cop`` is the
    equivalent pressure point in the local frame (inside the foot's
    CoP patch, tool6); meaningless when force_ratio is 0.
    """

    force_ratio: float = 0.0
    cop: Optional[np.ndarray] = None   # (2,) local; contact only
    foot_pos: Optional[np.ndarray] = None  # (3,) foot position
                                        # (mainly for airborne)


@dataclass
class LandMark:
    """Algorithm 1 output unit — a self-consistent target state.

    No time index: describes WHAT state to reach (support + COM +
    torso facing), not when.  ``vel=None`` means free arrival speed;
    a given vector (incl. zeros) is the desired COM velocity.
    """

    com: np.ndarray                    # (3,) target COM, local frame
    feet: Dict[str, FootSupport]       # 'right'/'left'
    vel: Optional[np.ndarray] = None   # (3,) desired arrival COM vel
    torso_yaw: Optional[float] = None  # rad; None = not handled (v1)
